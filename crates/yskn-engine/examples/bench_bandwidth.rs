//! `bench_bandwidth [--gb 4] [--reps 5]`
//!
//! Read-only memory streaming rates on this machine, to compare the compressed
//! kernel's code read rate with (no model file needed). One shared Metal buffer
//! of `--gb` GiB is allocated and written once, then read by:
//!
//! - (a) the CPU, 1, 4 and 10 threads, each summing its own contiguous slice
//!   with eight independent 64-bit accumulators (vectorized adds);
//! - (b) the GPU, a grid-stride kernel in which the 32 lanes of a SIMD group load
//!   32 neighbouring 16-byte pieces (512 contiguous bytes per load, as the
//!   compressed kernel's `warp` layout does), threadgroups of 32 x 4 threads;
//! - (c) both at once on disjoint halves (GPU first half, 10 CPU threads second
//!   half, each read three times), each side timed separately, their sum, and
//!   all bytes over the longer side's time (a lower bound of the combined rate).
//!
//! Every rate is bytes read / the minimum time over `--reps` repetitions after a
//! warm-up. GB/s are 10^9 bytes per second.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_bandwidth requires macOS and a Metal device");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::main() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };
    use std::{hint::black_box, ptr::NonNull, time::Instant};
    use yskn_engine::{Error, Result};

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void read_stream(device const uint4 *data [[buffer(0)]],
                        device uint *sink [[buffer(1)]],
                        constant uint2 &p [[buffer(2)]], // (uint4 count, repetitions)
                        uint id [[thread_position_in_grid]],
                        uint threads [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint r = 0; r < p.y; ++r) {
        for (uint i = id; i < p.x; i += threads) {
            acc ^= data[i];
        }
    }
    sink[id] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"#;
    /// 640 threadgroups of 32 x 4: 64 per GPU core on a 10-core GPU.
    const THREADGROUPS: usize = 640;
    const THREADGROUP: usize = 128;

    fn usage() -> Error {
        Error::InvalidInput("usage: bench_bandwidth [--gb 4] [--reps 5]".into())
    }

    fn fail(context: &str) -> Error {
        Error::Io(std::io::Error::other(format!("Metal {context} failed")))
    }

    fn cpu_sum(words: &[u64]) -> u64 {
        let mut acc = [0_u64; 8];
        let (chunks, rest) = words.as_chunks::<8>();
        for chunk in chunks {
            for (acc, &word) in acc.iter_mut().zip(chunk) {
                *acc = acc.wrapping_add(word);
            }
        }
        rest.iter()
            .chain(&acc)
            .fold(0, |sum, &word| sum.wrapping_add(word))
    }

    /// Seconds for `threads` threads to sum `words` (split into equal slices) `reps` times.
    fn cpu_read(words: &[u64], threads: usize, reps: usize) -> f64 {
        let start = Instant::now();
        std::thread::scope(|scope| {
            for slice in words.chunks(words.len().div_ceil(threads)) {
                scope.spawn(move || {
                    for _ in 0..reps {
                        black_box(cpu_sum(black_box(slice)));
                    }
                });
            }
        });
        start.elapsed().as_secs_f64()
    }

    struct Gpu {
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        sink: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    impl Gpu {
        /// Commit one read of `bytes` bytes at `offset` of `buffer`, `reps` times over.
        fn commit(
            &self,
            buffer: &ProtocolObject<dyn MTLBuffer>,
            offset: usize,
            bytes: usize,
            reps: u32,
        ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
            let command = self.queue.commandBuffer().ok_or_else(|| fail("command"))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| fail("encoder"))?;
            encoder.setComputePipelineState(&self.pipeline);
            let params = [(bytes / 16) as u32, reps];
            // SAFETY: offset + bytes lies inside `buffer` (checked by the caller), the
            // sink holds one u32 per grid thread, and both buffers outlive the wait
            // that follows every commit. `params` is plain data copied by setBytes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buffer), offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.sink), 0, 1);
                encoder.setBytes_length_atIndex(NonNull::from(&params).cast(), 8, 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: THREADGROUPS,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADGROUP,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            Ok(command)
        }

        fn wait(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<f64> {
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(fail("execution"));
            }
            Ok(command.GPUEndTime() - command.GPUStartTime())
        }
    }

    fn gb_per_s(bytes: usize, seconds: f64) -> f64 {
        bytes as f64 / seconds / 1e9
    }

    fn min(mut times: impl Iterator<Item = Result<f64>>) -> Result<f64> {
        times.try_fold(f64::INFINITY, |best, t| Ok(best.min(t?)))
    }

    pub fn main() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let (mut gib, mut reps) = (4_usize, 5_usize);
        while let Some(arg) = args.next() {
            let value = args
                .next()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .ok_or_else(usage)?;
            match arg.as_str() {
                "--gb" => gib = value,
                "--reps" => reps = value,
                _ => return Err(usage()),
            }
        }
        let bytes = gib << 30;
        // The kernel indexes 16-byte pieces with 32-bit integers.
        if bytes / 16 > u32::MAX as usize {
            return Err(usage());
        }
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| fail("device"))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|_| fail("shader compilation"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("read_stream"))
            .ok_or_else(|| fail("function"))?;
        let gpu = Gpu {
            queue: device.newCommandQueue().ok_or_else(|| fail("queue"))?,
            pipeline: device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| fail("pipeline"))?,
            sink: device
                .newBufferWithLength_options(
                    THREADGROUPS * THREADGROUP * 4,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| fail("sink buffer"))?,
        };
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| fail("buffer"))?;
        // SAFETY: the shared buffer holds `bytes` page-aligned bytes, no GPU work
        // uses it yet, and the GPU only reads it while this slice is alive.
        let words = unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u64>(), bytes / 8)
        };
        for (i, word) in words.iter_mut().enumerate() {
            *word = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
        let words: &[u64] = words;
        let load = std::process::Command::new("uptime")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        println!(
            "device={} buffer_bytes={bytes} reps={reps} uptime: {load}",
            device.name()
        );
        println!("mode threads GB/s");
        for threads in [1, 4, 10] {
            cpu_read(words, threads, 1);
            let seconds = min((0..reps).map(|_| Ok(cpu_read(words, threads, 1))))?;
            println!("a_cpu {threads} {:.1}", gb_per_s(bytes, seconds));
        }
        Gpu::wait(&*gpu.commit(&buffer, 0, bytes, 1)?)?;
        let (mut wall, mut busy) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let start = Instant::now();
            busy = busy.min(Gpu::wait(&*gpu.commit(&buffer, 0, bytes, 1)?)?);
            wall = wall.min(start.elapsed().as_secs_f64());
        }
        println!(
            "b_gpu - {:.1} (GPU time {:.1})",
            gb_per_s(bytes, wall),
            gb_per_s(bytes, busy)
        );
        // Both halves are read three times so that the two sides overlap for most
        // of their run; each side is timed separately (GPU: command buffer time).
        let half = bytes / 2;
        let (mut gpu_best, mut cpu_best) = (f64::INFINITY, f64::INFINITY);
        for rep in 0..=reps {
            let command = gpu.commit(&buffer, 0, half, 3)?;
            let cpu_seconds = cpu_read(&words[half / 8..], 10, 3);
            Gpu::wait(&command)?;
            let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
            if rep > 0 {
                gpu_best = gpu_best.min(gpu_seconds);
                cpu_best = cpu_best.min(cpu_seconds);
            }
        }
        let (gpu_rate, cpu_rate) = (gb_per_s(3 * half, gpu_best), gb_per_s(3 * half, cpu_best));
        println!("c_gpu_concurrent - {gpu_rate:.1}");
        println!("c_cpu_concurrent 10 {cpu_rate:.1}");
        println!("c_sum - {:.1}", gpu_rate + cpu_rate);
        // Lower bound of the combined rate: all bytes over the longer side's time.
        println!(
            "c_total_over_longer_side - {:.1}",
            gb_per_s(6 * half, gpu_best.max(cpu_best))
        );
        Ok(())
    }
}
