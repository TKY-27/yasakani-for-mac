use crate::{Error, Matrix, Result};
use std::{ops::Range, str::FromStr};
use yskn_format::Layout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Portable,
    Neon,
    Dotprod,
}

impl Kernel {
    pub fn available(self) -> bool {
        match self {
            Self::Portable => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "aarch64")]
            Self::Dotprod => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(not(target_arch = "aarch64"))]
            _ => false,
        }
    }
    pub fn available_variants() -> Vec<Self> {
        [Self::Portable, Self::Neon, Self::Dotprod]
            .into_iter()
            .filter(|k| k.available())
            .collect()
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Neon => "neon",
            Self::Dotprod => "dotprod",
        }
    }
    pub fn fastest() -> Self {
        if Self::Dotprod.available() {
            Self::Dotprod
        } else if Self::Neon.available() {
            Self::Neon
        } else {
            Self::Portable
        }
    }
}

impl FromStr for Kernel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "portable" => Ok(Self::Portable),
            "neon" => Ok(Self::Neon),
            "dotprod" => Ok(Self::Dotprod),
            _ => Err(Error::InvalidInput(format!(
                "unknown kernel {s}; expected portable, neon, dotprod"
            ))),
        }
    }
}

pub(crate) fn compute(
    matrix: &Matrix,
    activation: &[i8],
    scales: &[i32],
    rows: Range<usize>,
    kernel: Kernel,
) -> Result<Vec<i64>> {
    match matrix.descriptor().layout {
        Layout::Row => dispatch_bits::<false>(matrix, activation, scales, rows, kernel),
        Layout::Warp => dispatch_bits::<true>(matrix, activation, scales, rows, kernel),
    }
}

fn dispatch_bits<const WARP: bool>(
    matrix: &Matrix,
    activation: &[i8],
    scales: &[i32],
    rows: Range<usize>,
    kernel: Kernel,
) -> Result<Vec<i64>> {
    match matrix.descriptor().bits {
        4 => run::<4, WARP>(matrix, activation, scales, rows, kernel),
        5 => run::<5, WARP>(matrix, activation, scales, rows, kernel),
        6 => run::<6, WARP>(matrix, activation, scales, rows, kernel),
        7 => run::<7, WARP>(matrix, activation, scales, rows, kernel),
        8 => run::<8, WARP>(matrix, activation, scales, rows, kernel),
        _ => unreachable!("validated bit width"),
    }
}

fn run<const B: usize, const WARP: bool>(
    matrix: &Matrix,
    activation: &[i8],
    scales: &[i32],
    rows: Range<usize>,
    kernel: Kernel,
) -> Result<Vec<i64>> {
    let d = matrix.descriptor();
    let groups = d.cols / 512;
    let m = activation.len() / d.cols;
    let n = rows.len();
    let mut result = vec![0_i64; m * n];
    let mut weights = [0_i8; 512];
    for batch in (0..m).step_by(16) {
        let batch_end = (batch + 16).min(m);
        for tile in (rows.start..rows.end).step_by(32) {
            for group in 0..groups {
                for row in tile..(tile + 32).min(rows.end) {
                    decode::<B, WARP>(matrix, row, group * 2, &mut weights[..256]);
                    decode::<B, WARP>(matrix, row, group * 2 + 1, &mut weights[256..]);
                    for input in batch..batch_end {
                        let start = input * d.cols + group * 512;
                        let a: &[i8; 512] = activation[start..start + 512].try_into().unwrap();
                        let dot = match kernel {
                            Kernel::Portable => portable(&weights, a),
                            #[cfg(target_arch = "aarch64")]
                            Kernel::Neon => {
                                // SAFETY: Engine::new checks host NEON support; both arrays have 512 bytes.
                                unsafe { neon(&weights, a) }
                            }
                            #[cfg(target_arch = "aarch64")]
                            Kernel::Dotprod => {
                                // SAFETY: Engine::new checks dotprod support; both arrays have 512 bytes.
                                unsafe { dotprod(&weights, a) }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            _ => unreachable!("unavailable kernel rejected by Engine"),
                        };
                        let sum = &mut result[input * n + row - rows.start];
                        *sum = sum
                            .checked_add(i64::from(dot) * i64::from(scales[input * groups + group]))
                            .ok_or_else(|| {
                                Error::InvalidInput(
                                    "compressed integer accumulation exceeds i64".into(),
                                )
                            })?;
                    }
                }
            }
        }
    }
    Ok(result)
}

#[inline]
fn decode<const B: usize, const WARP: bool>(
    matrix: &Matrix,
    row: usize,
    sequence: usize,
    weights: &mut [i8],
) {
    let d = matrix.descriptor();
    let bytes = matrix.code_words();
    // One 256-weight sequence is at most 128 bytes. The extra word is zero
    // extension, so the final 12-bit window never reads beyond the code stream.
    let mut words = [0_u32; 33];
    for (word, value) in words[..4 * B].iter_mut().enumerate() {
        let index = if WARP {
            let q = sequence * d.physical_rows + row;
            ((q / 32 * B + word / 4) * 32 + q % 32) * 4 + word % 4
        } else {
            (row * (d.cols / 256) + sequence) * (4 * B) + word
        };
        let offset = index * 4;
        *value = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    }
    // Constant shifts remove the per-step window construction for released formats.
    // The first state carries zeros from before the sequence, not the previous sequence.
    let mut carry = 0_u64;
    if B == 6 {
        for (chunk, out) in weights.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let a = u64::from(words[chunk * 3]) | (u64::from(words[chunk * 3 + 1]) << 32);
            let c = u64::from(words[chunk * 3 + 2]);
            macro_rules! pair {
                ($j:expr, $state:expr) => {{
                    let index = (($state & 4095) as usize) * 2;
                    out[2 * $j..2 * $j + 2].copy_from_slice(&matrix.0.table[index..index + 2]);
                }};
            }
            pair!(0, (a << 6) | carry);
            pair!(1, a);
            pair!(2, a >> 6);
            pair!(3, a >> 12);
            pair!(4, a >> 18);
            pair!(5, a >> 24);
            pair!(6, a >> 30);
            pair!(7, a >> 36);
            pair!(8, a >> 42);
            pair!(9, a >> 48);
            pair!(10, (a >> 54) | (c << 10));
            pair!(11, (a >> 60) | (c << 4));
            pair!(12, c >> 2);
            pair!(13, c >> 8);
            pair!(14, c >> 14);
            pair!(15, c >> 20);
            carry = c >> 26;
        }
        return;
    }
    if B == 8 {
        for (chunk, out) in weights.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let a = u64::from(words[chunk * 2]) | (u64::from(words[chunk * 2 + 1]) << 32);
            macro_rules! pair {
                ($j:expr, $state:expr) => {{
                    let index = (($state & 4095) as usize) * 2;
                    out[2 * $j..2 * $j + 2].copy_from_slice(&matrix.0.table[index..index + 2]);
                }};
            }
            pair!(0, (a << 4) | carry);
            pair!(1, a >> 4);
            pair!(2, a >> 12);
            pair!(3, a >> 20);
            pair!(4, a >> 28);
            pair!(5, a >> 36);
            pair!(6, a >> 44);
            pair!(7, a >> 52);
            carry = a >> 60;
        }
        return;
    }
    for step in 0..128 {
        let end = B * (step + 1);
        let state = if end < 12 {
            (words[0] << (12 - end)) & 4095
        } else {
            let start = end - 12;
            let window = u64::from(words[start / 32]) | (u64::from(words[start / 32 + 1]) << 32);
            ((window >> (start % 32)) & 4095) as u32
        } as usize;
        weights[step * 2] = matrix.0.table[state * 2];
        weights[step * 2 + 1] = matrix.0.table[state * 2 + 1];
    }
}

#[inline]
fn portable(w: &[i8; 512], a: &[i8; 512]) -> i32 {
    w.iter()
        .zip(a)
        .map(|(&w, &a)| i32::from(w) * i32::from(a))
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(w: &[i8; 512], a: &[i8; 512]) -> i32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_s32(0);
    for i in (0..512).step_by(16) {
        // SAFETY: i <= 496, so each 16-byte unaligned load is within the arrays.
        let (w, a) = unsafe { (vld1q_s8(w.as_ptr().add(i)), vld1q_s8(a.as_ptr().add(i))) };
        let low = vmull_s8(vget_low_s8(w), vget_low_s8(a));
        let high = vmull_high_s8(w, a);
        // Widen before adding products: an i16 sum of two arbitrary i8 products can overflow.
        sum = vpadalq_s16(sum, low);
        sum = vpadalq_s16(sum, high);
    }
    vaddvq_s32(sum)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dotprod(w: &[i8; 512], a: &[i8; 512]) -> i32 {
    use std::arch::aarch64::*;
    let mut sums = [vdupq_n_s32(0); 4];
    for i in (0..512).step_by(64) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let offset = i + lane * 16;
            // SAFETY: offset <= 496, so each 16-byte load is within the arrays.
            let (w, a) = unsafe {
                (
                    vld1q_s8(w.as_ptr().add(offset)),
                    vld1q_s8(a.as_ptr().add(offset)),
                )
            };
            *sum = vdotq_s32(*sum, w, a);
        }
    }
    vaddvq_s32(vaddq_s32(
        vaddq_s32(sums[0], sums[1]),
        vaddq_s32(sums[2], sums[3]),
    ))
}
