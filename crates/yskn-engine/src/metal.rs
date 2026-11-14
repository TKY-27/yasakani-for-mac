use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{Engine, Error, Matrix, Result};
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
/// With |dot| <= 512 * 127 * 128 per group and |scale| <= 2^31, 516 groups cannot
/// overflow i64. Bounding the group count makes the reference's checked overflow
/// error unreachable, so the unchecked GPU sum is identical.
const MAX_GROUPS: usize = 512;

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

/// Largest accepted CPU share of a matrix's rows.
const MAX_CPU_SHARE: f64 = 0.9;

#[derive(Clone, Copy)]
enum CpuShare {
    Off,
    Fixed(f64),
    Calibrated,
}

fn parse_share(share: f64) -> Result<CpuShare> {
    if share == 0.0 {
        Ok(CpuShare::Off)
    } else if (0.0..=MAX_CPU_SHARE).contains(&share) {
        Ok(CpuShare::Fixed(share))
    } else {
        Err(invalid("the CPU share must be within 0..=0.9"))
    }
}

/// CPU fraction of each matrix's rows at M = 1, 2, 4, 8 and 16 (linear in between,
/// constant above 16), from `bench_metal --shares` on the real model on the
/// 4P+6E M5 with 10 CPU workers (D21). Each value is somewhat below the measured
/// optimum: a CPU that is slower than calibrated (other load) costs far more
/// than one that is faster.
const CPU_SHARE: [(usize, f64); 5] = [(1, 0.08), (2, 0.10), (4, 0.12), (8, 0.14), (16, 0.15)];

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

/// `YSKN_METAL_CPU_SHARE`: unset or `auto` = calibrated table, `off` = GPU only,
/// or a fixed fraction in 0..=0.9 for every M.
fn cpu_share_from_env() -> Result<CpuShare> {
    match std::env::var("YSKN_METAL_CPU_SHARE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("auto") => Ok(CpuShare::Calibrated),
        Ok("off") => Ok(CpuShare::Off),
        Ok(value) => value
            .parse::<f64>()
            .map_err(|_| invalid("YSKN_METAL_CPU_SHARE must be auto, off or 0..=0.9"))
            .and_then(parse_share),
        Err(_) => Err(invalid("YSKN_METAL_CPU_SHARE must be auto, off or 0..=0.9")),
    }
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

/// A no-copy GPU view of one complete read-only file mapping.
struct CodeBuffer {
    // Declared first so the Metal buffer is released before the mapping it wraps.
    buffer: Retained<Buffer>,
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
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<Pipeline>>,
    empty: Retained<Pipeline>,
    // Keyed by mapping address. The engine keeps every mapping it has wrapped alive
    // until it is dropped, so an address (and a table key) cannot be reused meanwhile.
    mappings: HashMap<usize, Rc<CodeBuffer>>,
    /// Signed-byte tables keyed by (mapping address, lookup-table offset): the
    /// quantized table depends only on the f16 table at that offset, while tensors
    /// may share a code stream but not a table.
    tables: HashMap<(usize, u64), Retained<Buffer>>,
    splits: usize,
    float_mac: bool,
    cpu_share: CpuShare,
    /// Reusable input and per-matrix output buffers of `weighted_sums_shared`,
    /// grown on demand and never shrunk.
    scratch_activation: Option<Retained<Buffer>>,
    scratch_scales: Option<Retained<Buffer>>,
    scratch_outputs: Vec<Option<Retained<Buffer>>>,
}

impl MetalEngine {
    /// `Ok(None)` when this process has no Metal device.
    pub fn new() -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("queue", "newCommandQueue returned nil"))?;
        let options = MTLCompileOptions::new();
        // Integer-only kernel; safe math anyway, so no float relaxation can creep in.
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
        Ok(Some(Self {
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
            float_mac: float_mac_from_env()?,
            cpu_share: cpu_share_from_env()?,
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
        self.device.name().to_string()
    }

    pub fn max_buffer_length(&self) -> usize {
        self.device.maxBufferLength()
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

    /// Whether jobs created afterwards multiply in binary32 (exact, METAL.md) rather
    /// than in 16/32-bit integers. Default from `YSKN_METAL_MAC` (`float` | `integer`).
    pub fn float_mac(&self) -> bool {
        self.float_mac
    }

    pub fn set_float_mac(&mut self, float_mac: bool) {
        self.float_mac = float_mac;
    }

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

    fn code_buffer(&mut self, matrix: &Matrix) -> Result<Rc<CodeBuffer>> {
        let mapping = matrix.mapping();
        let address = mapping.as_ptr() as usize;
        if let Some(buffer) = self.mappings.get(&address) {
            return Ok(Rc::clone(buffer));
        }
        // SAFETY: getpagesize has no preconditions.
        let page = usize::try_from(unsafe { getpagesize() })
            .map_err(|_| metal_error("page size", "negative"))?;
        if page == 0 || !address.is_multiple_of(page) {
            return Err(metal_error("no-copy buffer", "mapping is not page-aligned"));
        }
        let length = mapping
            .len()
            .checked_next_multiple_of(page)
            .filter(|&length| length > 0)
            .ok_or_else(|| metal_error("no-copy buffer", "invalid mapping length"))?;
        if length > self.max_buffer_length() {
            return Err(metal_error(
                "no-copy buffer",
                format!(
                    "mapping of {length} bytes exceeds maxBufferLength {}",
                    self.max_buffer_length()
                ),
            ));
        }
        let pointer = NonNull::new(mapping.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| metal_error("no-copy buffer", "null mapping"))?;
        // SAFETY: `pointer` is the page-aligned start of a live read-only mapping;
        // `length` only rounds its size up to the end of the last mapped page, which
        // the VM maps as a whole (bytes past EOF read as zero). The returned
        // CodeBuffer owns a Matrix clone, which keeps the mapping alive until after
        // the Metal buffer is released, so no deallocator is needed. Shaders only
        // read through this buffer.
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
            _mapping_owner: matrix.clone(),
        });
        self.mappings.insert(address, Rc::clone(&buffer));
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
        let code_offset = range.start & !3;
        let params = Params {
            rows: n as u32,
            physical_rows: physical_rows as u32,
            sequences: sequences as u32,
            cols: cols as u32,
            shift: (range.start - code_offset) as u32,
        };
        let codes = self.code_buffer(matrix)?;
        let mut dispatches = Vec::new();
        for first_input in (0..inputs).step_by(MAX_ROWS_PER_DISPATCH) {
            let rows = (inputs - first_input).min(MAX_ROWS_PER_DISPATCH);
            let pipeline = self.pipeline(PipelineKey {
                bits: bits as u32,
                rows_per_dispatch: rows as u32,
                warp: d.layout == Layout::Warp,
                aligned,
                float_mac: self.float_mac,
            })?;
            if pipeline.maxTotalThreadsPerThreadgroup() < self.threads_per_threadgroup() {
                return Err(metal_error(
                    "pipeline",
                    format!(
                        "supports {} threads per threadgroup, {} requested",
                        pipeline.maxTotalThreadsPerThreadgroup(),
                        self.threads_per_threadgroup()
                    ),
                ));
            }
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
            // Idle SIMD groups would only add barrier work for short rows.
            splits: self.splits.min(sequences),
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
    /// is repeatable; the sums are exact whatever the split.
    pub fn weighted_sums_split(
        &mut self,
        cpu: &mut Engine,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        self.shared(Some(cpu), matrices, activation, scales)
    }

    /// The CPU's fraction of each matrix's rows for M activation rows (0 = GPU only).
    pub fn cpu_share(&self, m: usize) -> f64 {
        match self.cpu_share {
            CpuShare::Off => 0.0,
            CpuShare::Fixed(share) => share,
            CpuShare::Calibrated => calibrated_cpu_share(m),
        }
    }

    /// Override the CPU share for all M (`None` restores the calibrated table).
    pub fn set_cpu_share(&mut self, share: Option<f64>) -> Result<()> {
        self.cpu_share = match share {
            None => CpuShare::Calibrated,
            Some(share) => parse_share(share)?,
        };
        Ok(())
    }

    /// CPU rows of an `n`-row matrix: a multiple of the 32-row tile, leaving the
    /// GPU at least one tile.
    fn cpu_rows(&self, n: usize, m: usize) -> usize {
        let rows = (self.cpu_share(m) * n as f64) as usize / ROW_TILE * ROW_TILE;
        rows.min(n.saturating_sub(ROW_TILE) / ROW_TILE * ROW_TILE)
    }

    fn shared(
        &mut self,
        cpu: Option<&mut Engine>,
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
                let cpu_rows = if cpu.is_some() {
                    self.cpu_rows(n, m)
                } else {
                    0
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
        // which is borrowed mutably here, and every submission is waited for
        // before this function returns, so no GPU work or other reference
        // touches them meanwhile.
        unsafe {
            upload(
                &activation_buffer,
                &scales_buffer,
                activation,
                scales,
                self.float_mac,
            )
        }?;
        let command = self.commit(|encoder| {
            for (plan, output) in plans.iter().zip(&outputs) {
                plan.encode(encoder, &activation_buffer, &scales_buffer, output);
            }
        })?;
        // The CPU rows run while the GPU works; wait for the GPU even if they fail.
        let cpu_sums = match cpu {
            Some(cpu) if split.iter().any(|&(gpu_rows, n)| gpu_rows < n) => {
                let parts: Vec<_> = matrices
                    .iter()
                    .zip(&split)
                    .map(|(matrix, &(gpu_rows, n))| (*matrix, gpu_rows..n))
                    .collect();
                cpu.weighted_sums_rows(&parts, activation, scales)
            }
            _ => Ok(vec![Vec::new(); matrices.len()]),
        };
        Self::wait(&command)?;
        let cpu_sums = cpu_sums?;
        plans
            .iter()
            .zip(&outputs)
            .zip(&cpu_sums)
            .map(|((plan, output), cpu)| {
                // SAFETY: `wait` returned, so the command buffer has completed and
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
        Self::wait(&*self.commit(encode)?)
    }

    /// Encode with `encode` into one command buffer and commit it without waiting.
    fn commit(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<Retained<CommandBuffer>> {
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
            Ok(command)
        })
    }

    /// Wait for a committed command buffer; its GPU execution time.
    fn wait(command: &CommandBuffer) -> Result<Duration> {
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

    /// Wall time of committing and waiting for one trivial one-thread dispatch.
    pub fn round_trip(&self) -> Result<Duration> {
        autoreleasepool(|_| {
            let start = Instant::now();
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| metal_error("command buffer", "nil"))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| metal_error("encoder", "nil"))?;
            encoder.setComputePipelineState(&self.empty);
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(metal_error("execution", "empty dispatch failed"));
            }
            Ok(start.elapsed())
        })
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
