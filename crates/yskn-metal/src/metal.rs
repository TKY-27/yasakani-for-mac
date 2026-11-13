use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

use objc2::{rc::Retained, rc::autoreleasepool, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use yskn_engine::{Error, Matrix, Result};
use yskn_format::Layout;

// MTLCreateSystemDefaultDevice lives in CoreGraphics (objc2-metal crate documentation).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

const SOURCE: &str = include_str!("linear.metal");

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
}

struct Dispatch {
    pipeline: Retained<Pipeline>,
    first_input: usize,
}

/// One matrix call with its own input and output buffers, ready to encode.
pub struct Job {
    codes: Rc<CodeBuffer>,
    code_offset: usize,
    table: Retained<Buffer>,
    activation: Retained<Buffer>,
    scales: Retained<Buffer>,
    output: Retained<Buffer>,
    params: Params,
    inputs: usize,
    groups: usize,
    code_bytes: u64,
    splits: usize,
    dispatches: Vec<Dispatch>,
}

impl Job {
    /// Row-major M×N sums of the last completed [`MetalEngine::run`] containing this job.
    pub fn output(&self) -> Vec<i64> {
        let len = self.inputs * self.params.rows as usize;
        let mut values = vec![0_i64; len];
        // SAFETY: the shared buffer was allocated with exactly `len` i64 values in
        // `MetalEngine::job`. GPU writes finished before `run` returned (it waits for
        // completion), and `run` takes jobs mutably, so none is executing now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output.contents().as_ptr().cast::<i64>(),
                values.as_mut_ptr(),
                len,
            );
        }
        values
    }

    /// Replace this job's activation rows and integer scales (same M) before a rerun.
    pub fn write_inputs(&mut self, activation: &[i8], scales: &[i32]) -> Result<()> {
        if activation.len() != self.inputs * self.params.cols as usize
            || scales.len() != self.inputs * self.groups
        {
            return Err(invalid("inputs must keep the job's M, K and K/512"));
        }
        // SAFETY: the shared buffers hold exactly M×K i16 and M×G i32 values
        // (allocated in `MetalEngine::job` from the same M, K, G; Metal buffers are
        // page-aligned). No GPU work reads them now: `run` waits for completion and
        // this method borrows the job exclusively, so the slice is not aliased.
        unsafe {
            let wide = std::slice::from_raw_parts_mut(
                self.activation.contents().as_ptr().cast::<i16>(),
                activation.len(),
            );
            for (wide, &value) in wide.iter_mut().zip(activation) {
                *wide = i16::from(value);
            }
            std::ptr::copy_nonoverlapping(
                scales.as_ptr(),
                self.scales.contents().as_ptr().cast::<i32>(),
                scales.len(),
            );
        }
        Ok(())
    }

    /// Bytes of compressed code this job reads (once per pass, independent of M).
    pub fn code_bytes(&self) -> u64 {
        self.code_bytes
    }

    fn encode(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let n = self.params.rows as usize;
        let cols = self.params.cols as usize;
        for dispatch in &self.dispatches {
            let first = dispatch.first_input;
            encoder.setComputePipelineState(&dispatch.pipeline);
            // SAFETY: every buffer is retained by this job, which outlives the
            // command buffer (run waits for completion). Offsets were validated in
            // `MetalEngine::job`: the code offset plus the stream length lies inside
            // the mapping buffer, and `first` < M addresses whole rows of the
            // activation (M×K i16), scale (M×G i32) and output (M×N i64) buffers.
            // Params is a plain repr(C) value copied by setBytes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.codes.buffer), self.code_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.table), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.activation), first * cols * 2, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.scales), first * self.groups * 4, 3);
                encoder.setBuffer_offset_atIndex(Some(&self.output), first * n * 8, 4);
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

/// Integer compressed-linear kernel on the system default Metal device.
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
        }))
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

    pub fn threads_per_threadgroup(&self) -> usize {
        ROW_TILE * self.splits
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<Retained<Pipeline>> {
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let values = MTLFunctionConstantValues::new();
        let constants: [(NonNull<c_void>, MTLDataType); 4] = [
            (NonNull::from(&key.bits).cast(), MTLDataType::UInt),
            (
                NonNull::from(&key.rows_per_dispatch).cast(),
                MTLDataType::UInt,
            ),
            (NonNull::from(&key.warp).cast(), MTLDataType::Bool),
            (NonNull::from(&key.aligned).cast(), MTLDataType::Bool),
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
        let d = matrix.descriptor();
        let (cols, physical_rows, bits) = (d.cols, d.physical_rows, d.bits);
        if activation.is_empty()
            || cols == 0
            || !cols.is_multiple_of(512)
            || !activation.len().is_multiple_of(cols)
            || n == 0
            || n > physical_rows
        {
            return Err(invalid("invalid activation width or output row count"));
        }
        let inputs = activation.len() / cols;
        let groups = cols / 512;
        let sequences = cols / 256;
        if scales.len() != inputs * groups {
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
        let mut job = Job {
            codes,
            code_offset,
            table,
            activation: self.empty_buffer(activation.len() * 2)?,
            scales: self.empty_buffer(scales.len() * 4)?,
            output: self.empty_buffer(inputs * n * 8)?,
            params,
            inputs,
            groups,
            code_bytes: (n * sequences * bits * 16) as u64,
            // Idle SIMD groups would only add barrier work for short rows.
            splits: self.splits.min(sequences),
            dispatches,
        };
        job.write_inputs(activation, scales)?;
        Ok(job)
    }

    /// Encode all jobs into one command buffer, commit, and wait once.
    /// Returns the GPU execution time reported by the command buffer.
    pub fn run(&self, jobs: &mut [Job]) -> Result<Duration> {
        autoreleasepool(|_| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| metal_error("command buffer", "nil"))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| metal_error("encoder", "nil"))?;
            for job in jobs.iter() {
                job.encode(&encoder);
            }
            encoder.endEncoding();
            command.commit();
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
        })
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
