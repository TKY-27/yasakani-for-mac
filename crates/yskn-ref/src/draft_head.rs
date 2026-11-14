//! Reduced draft head, histogram shortlist and coupled choice (chapter 07 §§6–8).
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::gated_deltanet::xor_sum32;
use crate::output_head_sampler::gumbel;
use crate::portable_math::exp_d;
use crate::{Error, Result, require_finite};

fn invalid(message: &str) -> Error {
    Error::InvalidModel(message.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadLayout {
    pub vocabulary: usize,
    pub hidden: usize,
    pub rank: usize,
    /// Uninterpreted fourth header word (engine35.cpp:1705–1708).
    pub reserved: i32,
    /// Projection, approximate weights/scales, rescore weights/scales, in bytes.
    pub arrays: [std::ops::Range<u64>; 5],
    pub file_bytes: u64,
}
impl HeadLayout {
    pub fn read(reader: &mut (impl Read + Seek)) -> Result<Self> {
        let actual = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut header = [0; 24];
        reader.read_exact(&mut header)?;
        if &header[..8] != b"SHEAD001" {
            return Err(invalid("SHEAD magic"));
        }
        let word = |i| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        let (v, h, r) = (word(8), word(12), word(16));
        // The 32 lanes each consume 16 contiguous rank components (q35_bdraft.cu:808–839).
        if v < 16 || h <= 0 || h % 512 != 0 || r != 512 {
            return Err(invalid(
                "SHEAD dimensions: V>=16, H multiple of 512, rank=512 required",
            ));
        }
        let (v, h, r) = (v as u64, h as u64, r as u64);
        let lengths = [
            r.checked_mul(h).and_then(|n| n.checked_mul(4)),
            v.checked_mul(r),
            v.checked_mul(4),
            v.checked_mul(h),
            v.checked_mul(4),
        ];
        let mut offset = 24_u64;
        let mut arrays = std::array::from_fn(|_| 0..0);
        for (array, length) in arrays.iter_mut().zip(lengths) {
            let end = offset
                .checked_add(length.ok_or_else(|| invalid("SHEAD size overflow"))?)
                .ok_or_else(|| invalid("SHEAD offset overflow"))?;
            if end > actual || usize::try_from(end).is_err() {
                return Err(invalid("SHEAD truncated array or address overflow"));
            }
            *array = offset..end;
            offset = end;
        }
        if offset != actual {
            return Err(invalid("SHEAD trailing bytes"));
        }
        Ok(Self {
            vocabulary: v as usize,
            hidden: h as usize,
            rank: r as usize,
            reserved: word(20),
            arrays,
            file_bytes: actual,
        })
    }
}

pub struct ReducedHead {
    layout: HeadLayout,
    projection: Vec<f32>,
    approximate: Vec<u8>,
    approximate_scale: Vec<f32>,
    rescore: Vec<u8>,
    rescore_scale: Vec<f32>,
}
fn bytes(reader: &mut impl Read, count: usize) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| invalid("SHEAD allocation failed"))?;
    result.resize(count, 0);
    reader.read_exact(&mut result)?;
    Ok(result)
}
fn floats(reader: &mut impl Read, count: usize) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| invalid("SHEAD allocation failed"))?;
    let mut buffer = [0; 16384];
    while result.len() < count {
        let n = (count - result.len()).min(buffer.len() / 4);
        reader.read_exact(&mut buffer[..n * 4])?;
        for b in buffer[..n * 4].chunks_exact(4) {
            result.push(f32::from_le_bytes(b.try_into().unwrap()));
        }
    }
    require_finite(&result, "SHEAD weights/scales")?;
    Ok(result)
}
impl ReducedHead {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::read(&mut File::open(path)?)
    }
    pub fn read(reader: &mut (impl Read + Seek)) -> Result<Self> {
        // Validate every derived offset/extent and EOF before any payload allocation.
        let layout = HeadLayout::read(reader)?;
        reader.seek(SeekFrom::Start(24))?;
        let projection = floats(reader, layout.rank * layout.hidden)?;
        let approximate = bytes(reader, layout.vocabulary * layout.rank)?;
        let approximate_scale = floats(reader, layout.vocabulary)?;
        let rescore = bytes(reader, layout.vocabulary * layout.hidden)?;
        let rescore_scale = floats(reader, layout.vocabulary)?;
        Ok(Self {
            layout,
            projection,
            approximate,
            approximate_scale,
            rescore,
            rescore_scale,
        })
    }
    pub fn layout(&self) -> &HeadLayout {
        &self.layout
    }
    /// One position's top 16, before predecessor correction, sorted score descending/id ascending.
    pub fn candidates(&self, hidden: &[f32], count: usize) -> Result<Vec<Candidate>> {
        if hidden.len() != self.layout.hidden {
            return Err(Error::InvalidInput("draft head hidden width".into()));
        }
        require_finite(hidden, "draft head input")?;
        let mut low = vec![0.0; self.layout.rank];
        for (rank, value) in low.iter_mut().enumerate() {
            let mut partial = [0.0_f32; 32];
            for block in (0..hidden.len()).step_by(512) {
                for u in 0..4 {
                    for (lane, acc) in partial.iter_mut().enumerate() {
                        for component in 0..4 {
                            let i = block + 4 * (lane + 32 * u) + component;
                            // Four explicit fmaf sites: q35_bdraft.cu:786.
                            *acc =
                                hidden[i].mul_add(self.projection[rank * hidden.len() + i], *acc);
                        }
                    }
                }
            }
            *value = xor_sum32(&partial);
        }
        require_finite(&low, "draft low-rank projection")?;
        let mut scores = vec![0.0; self.layout.vocabulary];
        for (v, score) in scores.iter_mut().enumerate() {
            let mut partial = [0.0_f32; 32];
            for (lane, acc) in partial.iter_mut().enumerate() {
                for c in 0..16 {
                    let i = 16 * lane + c;
                    // q35_bdraft.cu:839 (contiguous 16 per lane).
                    *acc = low[i].mul_add(
                        self.approximate[v * self.layout.rank + i] as i8 as f32,
                        *acc,
                    );
                }
            }
            *score = xor_sum32(&partial) * self.approximate_scale[v];
        }
        let ids = shortlist(&scores, count)?;
        let mut candidates = Vec::with_capacity(count);
        for v in ids {
            let mut partial = [0.0_f32; 32];
            for t in 0..hidden.len() / 128 {
                for (lane, acc) in partial.iter_mut().enumerate() {
                    for component in 0..4 {
                        let i = 4 * (lane + 32 * t) + component;
                        // Four explicit fmaf sites: q35_bdraft.cu:920–921.
                        *acc = hidden[i].mul_add(
                            self.rescore[v as usize * hidden.len() + i] as i8 as f32,
                            *acc,
                        );
                    }
                }
            }
            candidates.push(Candidate {
                token: v,
                score: xor_sum32(&partial) * self.rescore_scale[v as usize],
            });
        }
        top16(candidates)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub token: u32,
    pub score: f32,
}

fn ordered_key(x: f32) -> u32 {
    let u = x.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}
fn cutoff(histogram: &[usize], need: usize) -> (usize, usize) {
    let mut above = 0;
    for (bin, &n) in histogram.iter().enumerate().rev() {
        if above + n >= need {
            return (bin, above);
        }
        above += n;
    }
    unreachable!("nonempty histogram and validated need")
}
/// Two-stage source histogram cutoff, then D15 ascending token IDs within the bucket.
/// This intentionally differs from an exact floating top-k (q35_bdraft.cu:861–896).
pub fn shortlist(scores: &[f32], count: usize) -> Result<Vec<u32>> {
    require_finite(scores, "draft approximate scores")?;
    if !(16..=2048).contains(&count)
        || count % 8 != 0
        || count > scores.len()
        || scores.len() > i32::MAX as usize
    {
        return Err(Error::InvalidInput(
            "shortlist must be 16..2048, multiple of 8 and <= vocabulary".into(),
        ));
    }
    let mut h1 = [0; 1024];
    for &score in scores {
        h1[(ordered_key(score) >> 22) as usize] += 1;
    }
    let (first, above1) = cutoff(&h1, count);
    let mut h2 = [0; 2048];
    for &score in scores {
        let key = ordered_key(score);
        if (key >> 22) as usize == first {
            h2[((key >> 11) & 2047) as usize] += 1;
        }
    }
    let (second, above2) = cutoff(&h2, count - above1);
    let prefix = ((first as u32) << 11) | second as u32;
    let above = above1 + above2;
    let mut result = Vec::with_capacity(count);
    for (id, &score) in scores.iter().enumerate() {
        if ordered_key(score) >> 11 > prefix {
            result.push(id as u32);
        }
    }
    for (id, &score) in scores.iter().enumerate() {
        if result.len() == count {
            break;
        }
        if ordered_key(score) >> 11 == prefix {
            result.push(id as u32);
        }
    }
    debug_assert_eq!(result.len(), count);
    debug_assert!(above <= count);
    Ok(result)
}

pub fn top16(mut candidates: Vec<Candidate>) -> Result<Vec<Candidate>> {
    if candidates.len() < 16 || candidates.iter().any(|c| !c.score.is_finite()) {
        return Err(invalid("top16 requires >=16 finite candidates"));
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap()
            .then(a.token.cmp(&b.token))
    });
    if candidates
        .iter()
        .map(|c| c.token)
        .collect::<std::collections::HashSet<_>>()
        .len()
        != candidates.len()
    {
        return Err(invalid("duplicate draft candidates"));
    }
    candidates.truncate(16);
    Ok(candidates)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftChoice {
    Greedy,
    /// Main sampler version 3 enabled; independent draft temperature in thousandths.
    Sampled {
        seed: u64,
        temperature_thousandths: u32,
    },
}
impl DraftChoice {
    pub fn sampled(seed: u64) -> Self {
        Self::Sampled {
            seed,
            temperature_thousandths: 850,
        }
    }
}
/// Greedy ties use the input candidate index; sampled ties use corrected-score rank.
pub fn choose(candidates: &[Candidate], choice: DraftChoice, position: u64) -> Result<u32> {
    if candidates.len() != 16 || candidates.iter().any(|c| !c.score.is_finite()) {
        return Err(Error::InvalidInput(
            "choice needs 16 finite candidates".into(),
        ));
    }
    let mut best = 0;
    for i in 1..16 {
        if candidates[i].score > candidates[best].score {
            best = i;
        }
    }
    if let DraftChoice::Sampled {
        seed,
        temperature_thousandths,
    } = choice
    {
        if temperature_thousandths == 0 {
            return Err(Error::InvalidInput(
                "draft temperature must be positive".into(),
            ));
        }
        let mut order: Vec<_> = (0..16).collect();
        order.sort_by(|&a, &b| {
            candidates[b]
                .score
                .partial_cmp(&candidates[a].score)
                .unwrap()
                .then(candidates[a].token.cmp(&candidates[b].token))
        });
        // Preserve binary64 multiply, not division: q35_bdraft.cu:404,426–434.
        let temperature = 0.001_f64 * f64::from(temperature_thousandths);
        let z: Vec<_> = candidates
            .iter()
            .map(|c| (f64::from(c.score) - f64::from(candidates[order[0]].score)) / temperature)
            .collect();
        let weights: Vec<_> = z.iter().map(|&x| exp_d(x)).collect();
        let mut total = 0.0;
        for &i in &order {
            total += weights[i];
        }
        let mut prefix = 0.0;
        let mut maximum = f64::NEG_INFINITY;
        for i in order {
            if prefix < 0.95 * total {
                let score = z[i] + gumbel(seed, position, candidates[i].token);
                if score > maximum {
                    maximum = score;
                    best = i;
                }
            }
            prefix += weights[i];
        }
    }
    Ok(candidates[best].token)
}
