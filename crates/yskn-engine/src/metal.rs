use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{CpuShare, Engine, Error, Matrix, MetalOptions, Result};
use objc2::{rc::Retained, rc::autoreleasepool, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use yskn_format::Layout;

// MTLCreateSystemDefaultDevice lives in CoreGraphics (objc2-metal crate documentation).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

const SOURCE: &str = include_str!("linear.metal");

/// Upload encodings of every i8 value, indexed by its byte: i16 bits, and binary16
/// bits for the float kernel (|value| <= 128 is exact in binary16).
static WIDE: [u16; 256] = {
    let mut codes = [0; 256];
    let mut byte = 0;
    while byte < 256 {
        codes[byte] = byte as u8 as i8 as i16 as u16;
        byte += 1;
    }
    codes
};
static HALF: std::sync::LazyLock<[u16; 256]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|byte| yskn_format::f32_to_f16(f32::from(byte as u8 as i8)))
});

/// FORMAT.md §4.2 for one 32-step block of a `bits`-bit code. 32 steps use exactly
/// `bits` words, so all four blocks of a sequence share these constant shifts.
/// `wb[0]` is the previous block's last word, `wb[1..=bits]` this block's words:
/// the state of step j is the 12 bits ending at wb-bit 32 + bits * (j + 1).
fn decode_block(bits: usize) -> String {
    let mut text = format!("#define DECODE_BLOCK{bits}");
    for step in 0..32 {
        let start = 20 + bits * (step + 1);
        let (word, shift) = (start / 32, start % 32);
        let window = match shift {
            0 => format!("wb[{word}]"),
            1..=20 => format!("(wb[{word}] >> {shift})"),
            _ => format!(
                "((wb[{word}] >> {shift}) | (wb[{}] << {}))",
                word + 1,
                32 - shift
            ),
        };
        text += &format!(" \\\n    e[{step}] = lut[{window} & 4095];");
    }
    text + "\n"
}

fn shader_source() -> String {
    (4..=8).map(decode_block).collect::<String>() + SOURCE
}
/// Activation rows per dispatch; the shader's register arrays hold 16.
const MAX_ROWS_PER_DISPATCH: usize = 16;
/// Output rows per SIMD group (one row per lane).
const ROW_TILE: usize = 32;
/// The shader's threadgroup array: the 4,096-entry widened table, then the reduction.
const THREADGROUP_BYTES: usize = 16_384;
/// With |dot| <= 512 * 127 * 128 per group and |scale| <= 2^31, 516 groups cannot
/// overflow i64. Bounding the group count makes the reference's checked overflow
/// error unreachable, so the unchecked GPU sum is identical.
const MAX_GROUPS: usize = 512;

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

/// CPU fraction of each matrix's rows at M = 1, 2, 4, 8 and 16 (linear in between,
/// constant above 16), from `bench_metal --shares` on the real model on the
/// 4P+6E M5 with 10 CPU workers (D22). Each value is somewhat below the measured
/// optimum: a CPU that is slower than calibrated (other load) costs far more
/// than one that is faster.
const CPU_SHARE: [(usize, f64); 5] = [(1, 0.08), (2, 0.08), (4, 0.09), (8, 0.12), (16, 0.15)];
/// The Metal device and CPU worker count `CPU_SHARE` was measured with. On any
/// other machine `CpuShare::Auto` is off: a CPU slower than calibrated makes the
/// GPU wait (D22), and no other chip was measured (D23).
const CALIBRATED_DEVICE: &str = "Apple M5";
const CALIBRATED_THREADS: usize = 10;

fn share_table_applies(device: &str, cpu_threads: usize) -> bool {
    device == CALIBRATED_DEVICE && cpu_threads == CALIBRATED_THREADS
}

fn calibrated_cpu_share(m: usize) -> f64 {
    let m = m.clamp(1, 16);
    let upper = CPU_SHARE
        .iter()
        .position(|&(rows, _)| rows >= m)
        .unwrap_or(CPU_SHARE.len() - 1);
    let (m1, s1) = CPU_SHARE[upper];
    if m1 == m || upper == 0 {
        return s1;
    }
    let (m0, s0) = CPU_SHARE[upper - 1];
    s0 + (s1 - s0) * (m - m0) as f64 / (m1 - m0) as f64
}

/// CPU rows of an `n`-row matrix at `share`: a multiple of the 32-row tile,
/// leaving the GPU at least one tile (so 0 when n < 64).
fn cpu_rows(share: f64, n: usize) -> usize {
    let rows = (share * n as f64) as usize / ROW_TILE * ROW_TILE;
    rows.min(n.saturating_sub(ROW_TILE) / ROW_TILE * ROW_TILE)
}

/// The CPU part of a split call: row-major M×len sums of each (matrix, row
/// range) for the call's input, as `Engine::weighted_sums_rows`.
type CpuRows<'a> = dyn FnMut(&[(&Matrix, std::ops::Range<usize>)]) -> Result<Vec<Vec<i64>>> + 'a;

/// `YSKN_METAL_CPU_SHARE`: unset or `auto` = calibrated table, `off` = GPU only,
/// or a fixed fraction in 0..=0.9 for every M.
fn cpu_share_from_env() -> Result<CpuShare> {
    match std::env::var("YSKN_METAL_CPU_SHARE") {
        Err(std::env::VarError::NotPresent) => Some(CpuShare::Auto),
        Ok(value) => value.parse().ok(),
        Err(_) => None,
    }
    .ok_or_else(|| invalid("YSKN_METAL_CPU_SHARE must be auto, off or 0..=0.9"))
}

fn metal_error(context: &str, detail: impl std::fmt::Display) -> Error {
    Error::Io(std::io::Error::other(format!("Metal {context}: {detail}")))
}

fn ns_error(context: &str, error: &NSError) -> Error {
    metal_error(context, error.localizedDescription())
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

/// A no-copy GPU view of the page-aligned bytes `start..start + length` of a
/// read-only file mapping (the whole mapping when it fits `maxBufferLength`).
struct CodeBuffer {
    // Declared first so the Metal buffer is released before the mapping it wraps.
    buffer: Retained<Buffer>,
    start: usize,
    length: usize,
    _mapping_owner: Matrix,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    physical_rows: u32,
    sequences: u32,
    cols: u32,
    shift: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    bits: u32,
    rows_per_dispatch: u32,
    warp: bool,
    aligned: bool,
    float_mac: bool,
}

struct Dispatch {
    pipeline: Retained<Pipeline>,
    first_input: usize,
}

/// One validated matrix call without its input and output buffers.
struct Plan {
    codes: Rc<CodeBuffer>,
    code_offset: usize,
    table: Retained<Buffer>,
    params: Params,
    inputs: usize,
    groups: usize,
    code_bytes: u64,
    splits: usize,
    float_mac: bool,
    dispatches: Vec<Dispatch>,
}

/// Widen the activation rows to i16 (binary16 for the float kernel) and copy the
/// integer scales into shared buffers.
///
/// # Safety
/// No GPU work and no other CPU reference may access either buffer's contents
/// during the call.
unsafe fn upload(
    activation_buffer: &Buffer,
    scales_buffer: &Buffer,
    activation: &[i8],
    scales: &[i32],
    float_mac: bool,
) -> Result<()> {
    if activation_buffer.length() < activation.len() * 2
        || scales_buffer.length() < scales.len() * 4
    {
        return Err(metal_error("upload", "input buffer too small"));
    }
    // SAFETY: the lengths were just checked against the buffers, whose shared
    // contents are page-aligned and CPU-visible; the caller guarantees exclusive
    // access, so the mutable slice is not aliased.
    unsafe {
        let wide = std::slice::from_raw_parts_mut(
            activation_buffer.contents().as_ptr().cast::<u16>(),
            activation.len(),
        );
        let codes: &[u16; 256] = if float_mac { &HALF } else { &WIDE };
        for (wide, &value) in wide.iter_mut().zip(activation) {
            *wide = codes[value as u8 as usize];
        }
        std::ptr::copy_nonoverlapping(
            scales.as_ptr(),
            scales_buffer.contents().as_ptr().cast::<i32>(),
            scales.len(),
        );
    }
    Ok(())
}

/// Row-major M×(g + c) sums: each activation row's `g` GPU sums from the shared
/// output buffer (M×g), followed by its `c` CPU sums from `cpu` (M×c, may be empty).
///
/// # Safety
/// No GPU work may be writing the buffer during the call.
unsafe fn download(output: &Buffer, inputs: usize, g: usize, cpu: &[i64]) -> Result<Vec<i64>> {
    if inputs == 0 || output.length() < inputs * g * 8 || !cpu.len().is_multiple_of(inputs) {
        return Err(metal_error("readback", "output buffer too small"));
    }
    let c = cpu.len() / inputs;
    // SAFETY: the buffer holds at least M×g i64 values (checked above; shared
    // contents are page-aligned), and the caller guarantees no concurrent GPU write.
    let gpu =
        unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<i64>(), inputs * g) };
    let mut values = Vec::with_capacity(inputs * (g + c));
    for row in 0..inputs {
        values.extend_from_slice(&gpu[row * g..(row + 1) * g]);
        values.extend_from_slice(&cpu[row * c..(row + 1) * c]);
    }
    Ok(values)
}

/// The slot's shared buffer if it holds at least `bytes`, else a new one stored there.
fn reserve(
    device: &Device,
    slot: &mut Option<Retained<Buffer>>,
    bytes: usize,
) -> Result<Retained<Buffer>> {
    if let Some(buffer) = slot.as_ref().filter(|b| b.length() >= bytes) {
        return Ok(buffer.clone());
    }
    let buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| metal_error("buffer", "allocation failed"))?;
    *slot = Some(buffer.clone());
    Ok(buffer)
}

/// A committed command buffer that is waited for when dropped, so no exit path
/// (normal return, error or unwind) leaves GPU work reading or writing buffers
/// the engine may reuse.
struct InFlight(Retained<CommandBuffer>);

impl InFlight {
    /// Wait for completion; the GPU execution time reported by the command buffer.
    fn wait(self) -> Result<Duration> {
        let command = &self.0;
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            let detail = command.error().map_or_else(
                || "unknown".into(),
                |e| e.localizedDescription().to_string(),
            );
            return Err(metal_error("execution", detail));
        }
        Ok(Duration::from_secs_f64(
            (command.GPUEndTime() - command.GPUStartTime()).max(0.0),
        ))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        // Returns at once when `wait` has already run.
        self.0.waitUntilCompleted();
    }
}

/// One matrix call with its own input and output buffers, ready to encode.
pub struct Job {
    plan: Plan,
    activation: Retained<Buffer>,
    scales: Retained<Buffer>,
    output: Retained<Buffer>,
}

impl Job {
    /// Row-major M×N sums of the last completed [`MetalEngine::run`] containing this job.
    pub fn output(&self) -> Vec<i64> {
        // SAFETY: `run` waits for completion and takes jobs mutably, so no GPU work
        // writes this job's buffer while it is borrowed here.
        unsafe {
            download(
                &self.output,
                self.plan.inputs,
                self.plan.params.rows as usize,
                &[],
            )
        }
        // The buffer was allocated for exactly this length in `MetalEngine::job`.
        .expect("job output buffer")
    }

    /// Replace this job's activation rows and integer scales (same M) before a rerun.
    pub fn write_inputs(&mut self, activation: &[i8], scales: &[i32]) -> Result<()> {
        if activation.len() != self.plan.inputs * self.plan.params.cols as usize
            || scales.len() != self.plan.inputs * self.plan.groups
        {
            return Err(invalid("inputs must keep the job's M, K and K/512"));
        }
        // SAFETY: the job's private buffers are borrowed exclusively here, and `run`
        // waits for completion, so no GPU work reads them now.
        unsafe {
            upload(
                &self.activation,
                &self.scales,
                activation,
                scales,
                self.plan.float_mac,
            )
        }
    }

    /// Bytes of compressed code this job reads (once per pass, independent of M).
    pub fn code_bytes(&self) -> u64 {
        self.plan.code_bytes
    }

    fn encode(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        self.plan
            .encode(encoder, &self.activation, &self.scales, &self.output);
    }
}

impl Plan {
    fn output_len(&self) -> usize {
        self.inputs * self.params.rows as usize
    }

    /// The buffers must hold at least M×K i16, M×G i32 and M×N i64 values and stay
    /// alive until the command buffer completes.
    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        activation: &Buffer,
        scales: &Buffer,
        output: &Buffer,
    ) {
        let n = self.params.rows as usize;
        let cols = self.params.cols as usize;
        for dispatch in &self.dispatches {
            let first = dispatch.first_input;
            encoder.setComputePipelineState(&dispatch.pipeline);
            // SAFETY: the code and table buffers are retained by this plan and the
            // others by the caller, all until the command buffer completes (every
            // submission waits). Offsets were validated in `MetalEngine::plan`: the
            // code offset plus the stream length lies inside the mapping buffer, and
            // `first` < M addresses whole rows of the activation (M×K i16), scale
            // (M×G i32) and output (M×N i64) buffers, which are at least that long.
            // Params is a plain repr(C) value copied by setBytes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.codes.buffer), self.code_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.table), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(activation), first * cols * 2, 2);
                encoder.setBuffer_offset_atIndex(Some(scales), first * self.groups * 4, 3);
                encoder.setBuffer_offset_atIndex(Some(output), first * n * 8, 4);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&self.params).cast(),
                    size_of::<Params>(),
                    5,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(ROW_TILE),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: ROW_TILE,
                    height: self.splits,
                    depth: 1,
                },
            );
        }
    }
}

/// `YSKN_METAL_MAC`: `float` (default) or `integer` multiply-accumulate.
fn float_mac_from_env() -> Result<bool> {
    match std::env::var("YSKN_METAL_MAC").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("float") => Ok(true),
        Ok("integer") => Ok(false),
        _ => Err(invalid("YSKN_METAL_MAC must be float or integer")),
    }
}

/// Exact compressed-linear kernel on the system default Metal device.
///
/// Every file mapping wrapped for the GPU stays mapped until the engine is dropped.
pub struct MetalEngine {
    device: Retained<Device>,
    name: String,
    /// The device's `maxBufferLength`, or a smaller test limit.
    max_buffer_length: usize,
    /// A test limit on threads per threadgroup below each pipeline's own.
    max_threads: Option<usize>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<Pipeline>>,
    empty: Retained<Pipeline>,
    // Keyed by mapping address. The engine keeps every mapping it has wrapped alive
    // until it is dropped, so an address (and a table key) cannot be reused meanwhile.
    // One buffer per mapping, or several when it exceeds `max_buffer_length`.
    mappings: HashMap<usize, Vec<Rc<CodeBuffer>>>,
    /// Signed-byte tables keyed by (mapping address, lookup-table offset): the
    /// quantized table depends only on the f16 table at that offset, while tensors
    /// may share a code stream but not a table.
    tables: HashMap<(usize, u64), Retained<Buffer>>,
    splits: usize,
    float_mac: bool,
    cpu_share: CpuShare,
    /// Reusable input and per-matrix output buffers of `weighted_sums_shared` and
    /// `weighted_sums_split`, grown on demand and never shrunk.
    scratch_activation: Option<Retained<Buffer>>,
    scratch_scales: Option<Retained<Buffer>>,
    scratch_outputs: Vec<Option<Retained<Buffer>>>,
}

impl MetalEngine {
    /// `Ok(None)` when this process has no Metal device. Binary32 arithmetic and
    /// the calibrated CPU share; reads no environment variable (D14). An error
    /// when the shader does not compile or the device lacks what the kernel
    /// needs (SIMD width 32, 16 KB of threadgroup memory; D23).
    pub fn new() -> Result<Option<Self>> {
        Self::with_limits(None, None)
    }

    /// As [`Self::new`], but with limits below the device's, so tests can force a
    /// small device: `max_buffer_length` for the no-copy code buffers (a larger
    /// mapping is wrapped in several), `max_threads_per_threadgroup` for the
    /// kernel's threadgroups (fewer SIMD groups split each tile).
    pub fn with_limits(
        max_buffer_length: Option<usize>,
        max_threads_per_threadgroup: Option<usize>,
    ) -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let name = device.name().to_string();
        let memory = device.maxThreadgroupMemoryLength();
        if memory < THREADGROUP_BYTES {
            return Err(metal_error(
                "device",
                format!(
                    "{name} offers {memory} bytes of threadgroup memory; the kernel needs {THREADGROUP_BYTES}"
                ),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("queue", "newCommandQueue returned nil"))?;
        let options = MTLCompileOptions::new();
        // Safe math, so no float relaxation can creep in (the binary32 mode is exact
        // under any evaluation order or fusion anyway, METAL.md).
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(&shader_source()),
                Some(&options),
            )
            .map_err(|e| ns_error("shader compilation", &e))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("empty_kernel"))
            .ok_or_else(|| metal_error("shader", "missing empty_kernel"))?;
        let empty = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| ns_error("pipeline", &e))?;
        // The shader maps one SIMD lane to one output row of a 32-row tile.
        let width = empty.threadExecutionWidth();
        if width != ROW_TILE {
            return Err(metal_error(
                "device",
                format!("{name} has SIMD width {width}; the kernel needs {ROW_TILE}"),
            ));
        }
        let device_limit = device.maxBufferLength();
        Ok(Some(Self {
            max_buffer_length: max_buffer_length.map_or(device_limit, |n| n.min(device_limit)),
            max_threads: max_threads_per_threadgroup,
            name,
            device,
            queue,
            library,
            pipelines: HashMap::new(),
            empty,
            mappings: HashMap::new(),
            tables: HashMap::new(),
            // Measured on the M5 (DESIGN.md): 4 beats 8 by 3-4% for M <= 8 and ties
            // at 16; 1-2 and 16-32 are slower.
            splits: 4,
            float_mac: true,
            cpu_share: CpuShare::Auto,
            scratch_activation: None,
            scratch_scales: None,
            scratch_outputs: Vec::new(),
        }))
    }

    /// Whether this process has a system default Metal device (no shader compilation).
    pub fn device_exists() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }

    pub fn device_name(&self) -> String {
        self.name.clone()
    }

    /// The largest no-copy code buffer (the device's `maxBufferLength` unless a
    /// test limit is smaller).
    pub fn max_buffer_length(&self) -> usize {
        self.max_buffer_length
    }

    /// The device's recommended GPU working-set size in bytes (reported, not enforced).
    pub fn recommended_working_set(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// No-copy code buffers created so far (one per mapping that fits
    /// `maxBufferLength`, more for larger ones).
    pub fn code_buffers(&self) -> usize {
        self.mappings.values().map(Vec::len).sum()
    }

    /// Whether `CpuShare::Auto` uses the calibrated table for a CPU pool of
    /// `cpu_threads` workers: only on the machine it was measured on (D22, D23).
    pub fn share_table_applies(&self, cpu_threads: usize) -> bool {
        share_table_applies(&self.name, cpu_threads)
    }

    pub fn options(&self) -> MetalOptions {
        MetalOptions {
            float_mac: self.float_mac,
            cpu_share: self.cpu_share,
        }
    }

    /// Arithmetic and CPU share for calls made afterwards; neither changes a bit.
    pub fn set_options(&mut self, options: MetalOptions) {
        self.float_mac = options.float_mac;
        self.cpu_share = options.cpu_share;
    }

    /// SIMD groups per threadgroup that split one 32-row tile's sequences
    /// (applies to jobs created afterwards).
    pub fn splits(&self) -> usize {
        self.splits
    }

    pub fn set_splits(&mut self, splits: usize) -> Result<()> {
        // 32 * 32 threads is the Apple GPU threadgroup limit; the shader's 16 KB
        // reduction area holds 2048 partial sums.
        if !(1..=32).contains(&splits) {
            return Err(invalid("splits must be 1..=32"));
        }
        self.splits = splits;
        Ok(())
    }

    /// Apply `YSKN_METAL_MAC` and `YSKN_METAL_CPU_SHARE` (unset = the defaults of
    /// [`Self::new`]); unknown values are errors. Only the loading paths
    /// (`Model::load`, `FastDraft::load`) and benchmarks call this.
    pub fn configure_from_env(&mut self) -> Result<()> {
        let (float_mac, cpu_share) = (float_mac_from_env()?, cpu_share_from_env()?);
        self.float_mac = float_mac;
        self.cpu_share = cpu_share;
        Ok(())
    }

    /// Whether jobs created afterwards multiply in binary32 (exact, METAL.md) rather
    /// than in 16/32-bit integers (default binary32).
    pub fn float_mac(&self) -> bool {
        self.float_mac
    }

    pub fn set_float_mac(&mut self, float_mac: bool) {
        self.float_mac = float_mac;
    }

    /// Requested threads per threadgroup; a call uses fewer when the device or
    /// pipeline allows fewer, or the matrix has fewer sequences than splits.
    pub fn threads_per_threadgroup(&self) -> usize {
        ROW_TILE * self.splits
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<Retained<Pipeline>> {
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let values = MTLFunctionConstantValues::new();
        let constants: [(NonNull<c_void>, MTLDataType); 5] = [
            (NonNull::from(&key.bits).cast(), MTLDataType::UInt),
            (
                NonNull::from(&key.rows_per_dispatch).cast(),
                MTLDataType::UInt,
            ),
            (NonNull::from(&key.warp).cast(), MTLDataType::Bool),
            (NonNull::from(&key.aligned).cast(), MTLDataType::Bool),
            (NonNull::from(&key.float_mac).cast(), MTLDataType::Bool),
        ];
        for (index, (value, kind)) in constants.into_iter().enumerate() {
            // SAFETY: each pointer refers to a live u32 (UInt) or bool (Bool, one
            // byte) in `key`, matching the declared type; indices 0..4 are the
            // shader's function_constant indices.
            unsafe { values.setConstantValue_type_atIndex(value, kind, index) };
        }
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&NSString::from_str("weighted_sums"), &values)
            .map_err(|e| ns_error("function specialization", &e))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| ns_error("pipeline", &e))?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// A no-copy buffer holding `matrix`'s code stream from its 4-byte-aligned
    /// start: the whole mapping when it fits `max_buffer_length`, otherwise a
    /// window of the mapping starting at the stream's page. Nothing is copied.
    fn code_buffer(&mut self, matrix: &Matrix) -> Result<Rc<CodeBuffer>> {
        let mapping = matrix.mapping();
        let address = mapping.as_ptr() as usize;
        let range = matrix.code_range();
        let (first, end) = (range.start & !3, range.end);
        if let Some(buffer) = self.mappings.get(&address).and_then(|buffers| {
            buffers
                .iter()
                .find(|b| b.start <= first && end <= b.start + b.length)
        }) {
            return Ok(Rc::clone(buffer));
        }
        // SAFETY: getpagesize has no preconditions.
        let page = usize::try_from(unsafe { getpagesize() })
            .map_err(|_| metal_error("page size", "negative"))?;
        if page == 0 || !address.is_multiple_of(page) {
            return Err(metal_error("no-copy buffer", "mapping is not page-aligned"));
        }
        let mapped = mapping
            .len()
            .checked_next_multiple_of(page)
            .filter(|&length| length > 0 && end <= mapping.len())
            .ok_or_else(|| metal_error("no-copy buffer", "invalid mapping length"))?;
        let limit = self.max_buffer_length / page * page;
        let start = if mapped <= limit {
            0
        } else {
            first / page * page
        };
        let length = (mapped - start).min(limit);
        if end > start + length {
            return Err(metal_error(
                "no-copy buffer",
                format!(
                    "a code stream of {} bytes does not fit maxBufferLength {}",
                    range.len(),
                    self.max_buffer_length
                ),
            ));
        }
        let pointer = NonNull::new(mapping[start..].as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| metal_error("no-copy buffer", "null mapping"))?;
        // SAFETY: `pointer` is page-aligned (a page-aligned mapping plus a multiple
        // of the page size) inside a live read-only mapping, and `length` ends at
        // or before the end of the mapping's last page, which the VM maps as a
        // whole (bytes past EOF read as zero). The returned CodeBuffer owns a
        // Matrix clone, which keeps the mapping alive until after the Metal buffer
        // is released, so no deallocator is needed. Shaders only read through it.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| {
            metal_error(
                "no-copy buffer",
                "newBufferWithBytesNoCopy rejected the read-only mapping",
            )
        })?;
        let buffer = Rc::new(CodeBuffer {
            buffer,
            start,
            length,
            _mapping_owner: matrix.clone(),
        });
        self.mappings
            .entry(address)
            .or_default()
            .push(Rc::clone(&buffer));
        Ok(buffer)
    }

    fn empty_buffer(&self, length: usize) -> Result<Retained<Buffer>> {
        self.device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("buffer", "allocation failed"))
    }

    fn shared_buffer(&self, bytes: &[u8]) -> Result<Retained<Buffer>> {
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| metal_error("buffer", "null input"))?;
        // SAFETY: `pointer` addresses `bytes.len()` readable bytes; Metal copies them.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("buffer", "allocation failed"))
    }

    /// Validate one call against the matrix and prepare its buffers. `n` is the
    /// number of leading physical rows computed (all of them for a full call).
    pub fn job(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Result<Job> {
        let plan = self.plan(matrix, activation.len(), scales.len(), n)?;
        let mut job = Job {
            activation: self.empty_buffer(activation.len() * 2)?,
            scales: self.empty_buffer(scales.len() * 4)?,
            output: self.empty_buffer(plan.output_len() * 8)?,
            plan,
        };
        job.write_inputs(activation, scales)?;
        Ok(job)
    }

    /// Validate one call of `activation_len` i8 inputs and `scales_len` integer scales.
    fn plan(
        &mut self,
        matrix: &Matrix,
        activation_len: usize,
        scales_len: usize,
        n: usize,
    ) -> Result<Plan> {
        let d = matrix.descriptor();
        let (cols, physical_rows, bits) = (d.cols, d.physical_rows, d.bits);
        if activation_len == 0
            || cols == 0
            || !cols.is_multiple_of(512)
            || !activation_len.is_multiple_of(cols)
            || n == 0
            || n > physical_rows
        {
            return Err(invalid("invalid activation width or output row count"));
        }
        let inputs = activation_len / cols;
        let groups = cols / 512;
        let sequences = cols / 256;
        if scales_len != inputs * groups {
            return Err(invalid("integer scales must have M times K/512 elements"));
        }
        if groups > MAX_GROUPS {
            return Err(invalid("more than 512 groups could overflow i64"));
        }
        // Re-establish here every bound the shader's 32-bit indexing relies on,
        // independently of the format reader.
        let range = matrix.code_range();
        let stream_bytes = physical_rows
            .checked_mul(sequences)
            .and_then(|v| v.checked_mul(bits))
            .and_then(|v| v.checked_mul(16));
        if !(4..=8).contains(&bits)
            || stream_bytes != Some(range.len())
            || range.end > matrix.mapping().len()
            || range.len() > u32::MAX as usize - 4
            || (d.layout == Layout::Warp && !(physical_rows * sequences).is_multiple_of(32))
            || inputs
                .checked_mul(cols.max(n))
                .is_none_or(|v| v > u32::MAX as usize)
        {
            return Err(invalid("matrix shape exceeds the Metal kernel's bounds"));
        }
        // Bind at a 4-byte boundary; an unaligned stream uses byte loads from `shift`.
        let aligned = range.start.is_multiple_of(4);
        let codes = self.code_buffer(matrix)?;
        // The buffer starts at or before this boundary and contains the stream.
        let code_offset = (range.start & !3) - codes.start;
        let params = Params {
            rows: n as u32,
            physical_rows: physical_rows as u32,
            sequences: sequences as u32,
            cols: cols as u32,
            shift: (range.start & 3) as u32,
        };
        let mut dispatches = Vec::new();
        // Idle SIMD groups would only add barrier work for short rows.
        let mut splits = self.splits.min(sequences);
        for first_input in (0..inputs).step_by(MAX_ROWS_PER_DISPATCH) {
            let rows = (inputs - first_input).min(MAX_ROWS_PER_DISPATCH);
            let pipeline = self.pipeline(PipelineKey {
                bits: bits as u32,
                rows_per_dispatch: rows as u32,
                warp: d.layout == Layout::Warp,
                aligned,
                float_mac: self.float_mac,
            })?;
            // A device or pipeline allowing fewer threads than 32 x splits gets
            // fewer SIMD groups per tile: the sums are exact for any split count.
            let limit = self
                .max_threads
                .map_or(pipeline.maxTotalThreadsPerThreadgroup(), |n| {
                    n.min(pipeline.maxTotalThreadsPerThreadgroup())
                });
            if limit < ROW_TILE {
                return Err(metal_error(
                    "pipeline",
                    format!("supports {limit} threads per threadgroup, at least {ROW_TILE} needed"),
                ));
            }
            splits = splits.min(limit / ROW_TILE);
            dispatches.push(Dispatch {
                pipeline,
                first_input,
            });
        }
        let table_key = (matrix.mapping().as_ptr() as usize, d.table_offset);
        let table = match self.tables.get(&table_key) {
            Some(table) => table.clone(),
            None => {
                let bytes: Vec<u8> = matrix.weight_table8().iter().map(|&v| v as u8).collect();
                let table = self.shared_buffer(&bytes)?;
                self.tables.insert(table_key, table.clone());
                table
            }
        };
        Ok(Plan {
            codes,
            code_offset,
            table,
            params,
            inputs,
            groups,
            code_bytes: (n * sequences * bits * 16) as u64,
            splits,
            float_mac: self.float_mac,
            dispatches,
        })
    }

    /// Encode all jobs into one command buffer, commit, and wait once.
    /// Returns the GPU execution time reported by the command buffer.
    pub fn run(&self, jobs: &mut [Job]) -> Result<Duration> {
        self.submit(|encoder| {
            for job in jobs.iter() {
                job.encode(encoder);
            }
        })
    }

    /// Row-major M×N sums (padded physical rows included) of several matrices
    /// that read the same input, encoded into one command buffer. Input and
    /// output buffers are reused across calls.
    pub fn weighted_sums_shared(
        &mut self,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        self.shared(None, matrices, activation, scales)
    }

    /// As [`Self::weighted_sums_shared`], but the CPU kernel on `cpu` computes
    /// the last rows of each matrix while the GPU computes the others. The row
    /// split comes only from the matrix size, M and [`Self::cpu_share`], so a run
    /// is repeatable; the sums are exact whatever the split. A matrix with fewer
    /// than 64 physical rows runs on the GPU alone. Every call covers all physical
    /// rows: there is no prefix (head slice) variant of the split, and no
    /// projection call site computes a prefix on Metal.
    pub fn weighted_sums_split(
        &mut self,
        cpu: &mut Engine,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        let threads = cpu.threads();
        self.shared(
            Some((
                &mut |parts| cpu.weighted_sums_rows(parts, activation, scales),
                threads,
            )),
            matrices,
            activation,
            scales,
        )
    }

    /// The CPU's fraction of each matrix's rows for M activation rows and a CPU
    /// pool of `cpu_threads` workers (0 = GPU only). `CpuShare::Auto` is the
    /// calibrated table where [`Self::share_table_applies`], else 0.
    pub fn cpu_share(&self, m: usize, cpu_threads: usize) -> f64 {
        match self.cpu_share {
            CpuShare::Off => 0.0,
            CpuShare::Fixed(share) => share,
            CpuShare::Auto if self.share_table_applies(cpu_threads) => calibrated_cpu_share(m),
            CpuShare::Auto => 0.0,
        }
    }

    /// Override the CPU share for all M (`None` restores `CpuShare::Auto`).
    pub fn set_cpu_share(&mut self, share: Option<f64>) -> Result<()> {
        self.cpu_share = match share {
            None => CpuShare::Auto,
            Some(share) => CpuShare::fixed(share)?,
        };
        Ok(())
    }

    /// `cpu` is the CPU part and the CPU pool's worker count.
    fn shared(
        &mut self,
        mut cpu: Option<(&mut CpuRows, usize)>,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        let split: Vec<(usize, usize)> = matrices
            .iter()
            .map(|matrix| {
                let d = matrix.descriptor();
                let n = d.physical_rows;
                let m = activation.len() / d.cols.max(1);
                let cpu_rows = match &cpu {
                    Some((_, threads)) => cpu_rows(self.cpu_share(m, *threads), n),
                    None => 0,
                };
                (n - cpu_rows, n)
            })
            .collect();
        let plans = matrices
            .iter()
            .zip(&split)
            .map(|(matrix, &(gpu_rows, _))| {
                self.plan(matrix, activation.len(), scales.len(), gpu_rows)
            })
            .collect::<Result<Vec<_>>>()?;
        let activation_buffer = reserve(
            &self.device,
            &mut self.scratch_activation,
            activation.len() * 2,
        )?;
        let scales_buffer = reserve(&self.device, &mut self.scratch_scales, scales.len() * 4)?;
        if self.scratch_outputs.len() < plans.len() {
            self.scratch_outputs.resize(plans.len(), None);
        }
        let outputs = plans
            .iter()
            .zip(&mut self.scratch_outputs)
            .map(|(plan, slot)| reserve(&self.device, slot, plan.output_len() * 8))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: the scratch buffers are distinct objects owned by this engine,
        // which is borrowed mutably here. Every earlier command buffer of this
        // engine was waited for by its `InFlight` guard before its call returned
        // or unwound, and this call has not committed yet, so no GPU work or
        // other reference touches them.
        unsafe {
            upload(
                &activation_buffer,
                &scales_buffer,
                activation,
                scales,
                self.float_mac,
            )
        }?;
        let in_flight = self.commit(|encoder| {
            for (plan, output) in plans.iter().zip(&outputs) {
                plan.encode(encoder, &activation_buffer, &scales_buffer, output);
            }
        })?;
        // The CPU rows run while the GPU works; wait for the GPU even if they fail.
        let cpu_sums = match cpu.as_mut() {
            Some((cpu, _)) if split.iter().any(|&(gpu_rows, n)| gpu_rows < n) => {
                let parts: Vec<_> = matrices
                    .iter()
                    .zip(&split)
                    .map(|(matrix, &(gpu_rows, n))| (*matrix, gpu_rows..n))
                    .collect();
                cpu(&parts)
            }
            _ => Ok(vec![Vec::new(); matrices.len()]),
        };
        in_flight.wait()?;
        let cpu_sums = cpu_sums?;
        plans
            .iter()
            .zip(&outputs)
            .zip(&cpu_sums)
            .map(|((plan, output), cpu)| {
                // SAFETY: `wait` returned Ok, so the command buffer has completed and
                // nothing writes these engine-owned buffers any more.
                unsafe { download(output, plan.inputs, plan.params.rows as usize, cpu) }
            })
            .collect()
    }

    /// Encode with `encode` into one command buffer, commit, and wait for completion.
    /// Returns the GPU execution time reported by the command buffer.
    fn submit(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<Duration> {
        self.commit(encode)?.wait()
    }

    /// Encode with `encode` into one command buffer and commit it; the returned
    /// guard waits for completion when it is dropped.
    fn commit(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<InFlight> {
        autoreleasepool(|_| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| metal_error("command buffer", "nil"))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| metal_error("encoder", "nil"))?;
            encode(&encoder);
            encoder.endEncoding();
            command.commit();
            Ok(InFlight(command))
        })
    }

    /// Wall time of committing and waiting for one trivial one-thread dispatch.
    pub fn round_trip(&self) -> Result<Duration> {
        let start = Instant::now();
        self.submit(|encoder| {
            encoder.setComputePipelineState(&self.empty);
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        })?;
        Ok(start.elapsed())
    }

    /// Row-major M×N sums including padded physical rows, as `yskn_engine::Engine`.
    pub fn weighted_sums(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<i64>> {
        let n = matrix.descriptor().physical_rows;
        self.weighted_sums_prefix(matrix, activation, scales, n)
    }

    /// The first `n` physical rows, e.g. a vocabulary-head slice.
    pub fn weighted_sums_prefix(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Result<Vec<i64>> {
        let mut job = [self.job(matrix, activation, scales, n)?];
        self.run(&mut job)?;
        Ok(job[0].output())
    }
}

#[cfg(test)]
#[path = "../tests/fixture/mod.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::fixture::{TempFile, assert_sums, random, synthetic};
    use super::*;
    use crate::{Kernel, MappedModel};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// The calibrated share table is for the machine it was measured on only;
    /// another chip, or the M5 with another CPU worker count, runs GPU-only.
    #[test]
    fn share_table_applies_only_to_the_calibrated_machine() {
        assert!(share_table_applies("Apple M5", 10));
        for (device, threads) in [
            ("Apple M5", 1),
            ("Apple M5", 4),
            ("Apple M5", 16),
            ("Apple M4", 10),
            ("Apple M5 Max", 10),
            ("Apple M1", 8),
            ("Apple Paravirtual device", 10),
        ] {
            assert!(!share_table_applies(device, threads), "{device} {threads}");
        }
    }

    #[test]
    fn cpu_rows_are_whole_tiles_and_leave_the_gpu_a_tile() {
        // Below 64 rows the GPU's minimum tile leaves no whole tile for the CPU.
        for n in [1, 11, 31, 32, 33, 40, 63] {
            assert_eq!(cpu_rows(CpuShare::MAX, n), 0, "n={n}");
        }
        assert_eq!(cpu_rows(CpuShare::MAX, 64), 32);
        assert_eq!(cpu_rows(CpuShare::MAX, 75), 32);
        assert_eq!(cpu_rows(CpuShare::MAX, 104), 64);
        assert_eq!(cpu_rows(0.3, 75), 0);
        assert_eq!(cpu_rows(0.3, 320), 96);
        assert_eq!(cpu_rows(0.0, 5120), 0);
        assert_eq!(cpu_rows(0.08, 5120), 384);
    }

    /// A CPU part that panics or fails after the GPU part was committed: the call
    /// returns only once the GPU work has completed (its rows are in the scratch
    /// outputs), and the same engine then serves a normal call exactly.
    #[test]
    fn gpu_work_completes_before_a_failing_cpu_part_returns_and_the_engine_is_reused() {
        let Some(mut engine) = MetalEngine::new().unwrap() else {
            println!("skipped: no Metal device");
            return;
        };
        let mut cpu = Engine::new(Kernel::fastest(), 4).unwrap();
        // Enough GPU work (eight 4096-row matrices at M=16) to still be running
        // when the panic has unwound, unless something waits for it.
        let (n, k, m) = (4096, 5120, 16);
        let file = TempFile::new(&synthetic(8, "warp", n, n, k, false));
        // SAFETY: the private file is never modified and outlives all mapping owners.
        let model = unsafe { MappedModel::open(&file.0) }.unwrap();
        let matrix = model.matrix("linear").unwrap();
        let small_file = TempFile::new(&synthetic(4, "row", 9, 11, k, false));
        // SAFETY: as above.
        let small_model = unsafe { MappedModel::open(&small_file.0) }.unwrap();
        let small = small_model.matrix("linear").unwrap();
        let refs = [&matrix; 8];
        let mut seed = 0x0dd5_eed0_0000_0007;
        let mut input = || {
            let activation: Vec<i8> = (0..m * k).map(|_| random(&mut seed) as i8).collect();
            let scales: Vec<i32> = (0..m * k / 512).map(|_| random(&mut seed) as i32).collect();
            (activation, scales)
        };
        engine.set_cpu_share(Some(0.5)).unwrap();
        let gpu_rows = n - cpu_rows(0.5, n);
        for panics in [true, false] {
            engine.scratch_outputs.clear();
            let (activation, scales) = input();
            let expected = cpu.weighted_sums(&matrix, &activation, &scales).unwrap();
            let mut failing = |_: &[(&Matrix, std::ops::Range<usize>)]| -> Result<Vec<Vec<i64>>> {
                if panics {
                    panic!("expected CPU part panic");
                }
                Err(invalid("expected CPU part error"))
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                engine.shared(Some((&mut failing, 4)), &refs, &activation, &scales)
            }));
            match result {
                Err(_) => assert!(panics),
                Ok(result) => assert!(!panics && result.is_err()),
            }
            let gpu_part: Vec<i64> = (0..m)
                .flat_map(|row| expected[row * n..row * n + gpu_rows].iter().copied())
                .collect();
            for (index, output) in engine.scratch_outputs.iter().enumerate() {
                // SAFETY: `shared` has returned or unwound, and its `InFlight`
                // guard waited for the command buffer, so no GPU work writes it.
                let sums = unsafe { download(output.as_ref().unwrap(), m, gpu_rows, &[]) };
                assert_sums(
                    &sums.unwrap(),
                    &gpu_part,
                    &format!("GPU rows of matrix {index} after panics={panics}"),
                );
            }
            // The engine is reusable: new inputs overwrite the scratch buffers.
            let (activation, scales) = input();
            let expected = cpu.weighted_sums(&matrix, &activation, &scales).unwrap();
            let sums = engine
                .weighted_sums_split(&mut cpu, &refs, &activation, &scales)
                .unwrap();
            for (index, sums) in sums.iter().enumerate() {
                assert_sums(sums, &expected, &format!("reuse {index} panics={panics}"));
            }
            // Fewer than 64 rows leave the CPU nothing, so a CPU part is never called.
            let expected = cpu.weighted_sums(&small, &activation, &scales).unwrap();
            let sums = engine
                .shared(Some((&mut failing, 4)), &[&small], &activation, &scales)
                .unwrap();
            assert_sums(&sums[0], &expected, "small matrix without CPU rows");
        }
    }
}
