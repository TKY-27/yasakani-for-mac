//! Checked KVQP v1 reader. Unknown header/record bytes are deliberately ignored.
use crate::{Error, Result};
use std::ops::Range;
use yskn_format::f16_to_f32;

#[derive(Clone, Debug)]
pub struct CodeTable {
    pub gain: f32,
    pub codes: [i16; 256],
}

impl CodeTable {
    /// src/kvq_gpu.cuh:53: two separately rounded binary32 products.
    pub fn reconstruction(&self, state: u8) -> f32 {
        (f32::from(self.codes[usize::from(state)]) * 0.00390625) * self.gain
    }
}

#[derive(Clone, Debug)]
pub struct Group {
    pub mode: u8,
    pub bits: u8,
    pub table: usize,
    pub guard: u8,
    pub basis_count: u8,
    pub mean: Option<[f32; 256]>,
    /// Only the first two vectors are consumed (src/kvq_fast.cuh:90-91).
    pub basis: Vec<[f32; 256]>,
}

#[derive(Clone, Debug)]
pub struct KvProfile {
    attention_layers: usize,
    kv_heads: usize,
    multipliers: Vec<f32>,
    tables: Vec<CodeTable>,
    groups: Vec<Group>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(format!("KV codec: {}", message.into()))
}

fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("profile allocation failed"))?;
    Ok(values)
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn range(&self, offset: usize, count: usize, width: usize) -> Result<Range<usize>> {
        let length = count
            .checked_mul(width)
            .ok_or_else(|| invalid("length overflow"))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("offset overflow"))?;
        if end > self.0.len() {
            return Err(invalid("truncated or out-of-range region"));
        }
        Ok(offset..end)
    }
    fn region(&self, offset: usize, count: usize, width: usize) -> Result<Range<usize>> {
        if offset < 96 {
            return Err(invalid("payload overlaps header"));
        }
        self.range(offset, count, width)
    }
    fn u16(&self, offset: usize) -> Result<u16> {
        let r = self.range(offset, 1, 2)?;
        Ok(u16::from_le_bytes([self.0[r.start], self.0[r.start + 1]]))
    }
    fn u32(&self, offset: usize) -> Result<u32> {
        let r = self.range(offset, 1, 4)?;
        Ok(u32::from_le_bytes(
            self.0[r].try_into().map_err(|_| invalid("u32"))?,
        ))
    }
    fn float(&self, offset: usize) -> Result<f32> {
        let value = f32::from_bits(self.u32(offset)?);
        if !value.is_finite() {
            return Err(invalid("non-finite profile float"));
        }
        Ok(value)
    }
    fn vector(&self, offset: usize) -> Result<[f32; 256]> {
        self.region(offset, 256, 2)?;
        let mut values = [0.0; 256];
        for (i, v) in values.iter_mut().enumerate() {
            *v = f16_to_f32(self.u16(offset + 2 * i)?);
            if !v.is_finite() {
                return Err(invalid("non-finite profile vector"));
            }
        }
        Ok(values)
    }
}

impl KvProfile {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let r = Reader(bytes);
        // src/engine35.cpp:40-43; dimensions are header-derived for tiny models.
        if bytes.len() < 100
            || bytes.get(..4) != Some(b"KVQP")
            || r.u16(4)? != 1
            || r.u16(6)? != 96
            || r.u32(8)? as usize != bytes.len()
            || r.u16(16)? != 256
        {
            return Err(invalid("invalid KVQP v1 header"));
        }
        let attention_layers = usize::from(r.u16(18)?);
        let kv_heads = usize::from(r.u16(20)?);
        let nt = usize::from(r.u16(22)?);
        let nm = usize::from(r.u16(24)?);
        let ng = r.u32(28)? as usize;
        let expected = attention_layers
            .checked_mul(kv_heads)
            .and_then(|n| n.checked_mul(2));
        if attention_layers == 0 || kv_heads == 0 || Some(ng) != expected || !(1..=16).contains(&nm)
        {
            return Err(invalid(
                "invalid dimensions, group count or multiplier count",
            ));
        }
        let om = r.u32(32)? as usize;
        let ot = r.u32(36)? as usize;
        let og = r.u32(40)? as usize;
        // Bound counts by actual bytes before allocation or any directory indexing.
        let directories = [
            r.region(om, nm, 4)?,
            r.region(ot, nt, 4)?,
            r.region(og, ng, 4)?,
        ];
        let region_count = ng
            .checked_add(nt)
            .and_then(|n| n.checked_add(3))
            .ok_or_else(|| invalid("region count overflow"))?;
        let mut regions = allocate(region_count)?;
        regions.extend(directories);
        let mut multipliers = allocate(nm)?;
        for i in 0..nm {
            let m = r.float(om + 4 * i)?;
            multipliers.push(m);
        }
        let mut tables = allocate(nt)?;
        for i in 0..nt {
            let o = r.u32(ot + 4 * i)? as usize;
            regions.push(r.region(o, 524, 1)?);
            if bytes[o] != 8 || bytes[o + 2] != 64 || bytes[o + 3] != 8 {
                return Err(invalid(
                    "table requires state bits 8, chain 64, fractional bits 8",
                ));
            }
            let gain = r.float(o + 4)?;
            let mut codes = [0; 256];
            for (s, c) in codes.iter_mut().enumerate() {
                *c = r.u16(o + 12 + 2 * s)? as i16;
            }
            tables.push(CodeTable { gain, codes });
        }
        let mut groups = allocate(ng)?;
        for i in 0..ng {
            let o = r.u32(og + 4 * i)? as usize;
            r.region(o, 8, 1)?;
            let mode = bytes[o];
            let bits = bytes[o + 1];
            let table = usize::from(bytes[o + 2]);
            let guard = bytes[o + 3];
            let basis_count = bytes[o + 4];
            let has_mean = bytes[o + 5] != 0;
            if mode == 0 && (!(3..=5).contains(&bits) || table >= nt) {
                return Err(invalid(
                    "inconsistent trellis bits or table directory index",
                ));
            }
            let consumed = usize::from(basis_count.min(2)) + usize::from(has_mean);
            regions.push(r.region(o, 8 + 512 * consumed, 1)?);
            let mut cursor = o + 8;
            let mean = if has_mean {
                let v = r.vector(cursor)?;
                cursor += 512;
                Some(v)
            } else {
                None
            };
            let mut basis = allocate(usize::from(basis_count.min(2)))?;
            for _ in 0..basis_count.min(2) {
                basis.push(r.vector(cursor)?);
                cursor += 512;
            }
            groups.push(Group {
                mode,
                bits,
                table,
                guard,
                basis_count,
                mean,
                basis,
            });
        }
        // Exact record aliases are safe; partial overlap is a directory inconsistency.
        regions.retain(|r| !r.is_empty());
        regions.sort_by_key(|r| (r.start, r.end));
        regions.dedup();
        if regions.windows(2).any(|p| p[0].end > p[1].start) {
            return Err(invalid("overlapping profile regions"));
        }
        Ok(Self {
            attention_layers,
            kv_heads,
            multipliers,
            tables,
            groups,
        })
    }
    pub fn attention_layers(&self) -> usize {
        self.attention_layers
    }
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }
    pub fn tables(&self) -> &[CodeTable] {
        &self.tables
    }
    pub fn multipliers(&self) -> &[f32] {
        &self.multipliers
    }
    pub fn group(&self, group: usize) -> Result<&Group> {
        self.groups
            .get(group)
            .ok_or_else(|| invalid("group outside profile"))
    }
    pub fn check_dimensions(
        &self,
        attention_layers: usize,
        kv_heads: usize,
        head_width: usize,
    ) -> Result<()> {
        if (self.attention_layers, self.kv_heads, 256) != (attention_layers, kv_heads, head_width) {
            return Err(invalid("profile/model dimensions differ"));
        }
        Ok(())
    }
    /// src/kvq_atxpk.hpp:73-83: ties choose four bits.
    pub fn packed_bits(&self) -> u8 {
        let count = |bits| {
            self.groups
                .iter()
                .filter(|g| g.mode == 0 && g.bits == bits)
                .count()
        };
        if count(5) > count(4) { 5 } else { 4 }
    }
    /// Generalize src/kvq_atxpk.hpp:60-70's all-head layer/side predicate.
    pub fn packable(&self, layer: usize, side: usize) -> bool {
        layer < self.attention_layers
            && side < 2
            && (0..self.kv_heads).all(|h| {
                let g = &self.groups[(layer * self.kv_heads + h) * 2 + side];
                g.mode == 0 && g.bits == self.packed_bits() && g.guard == 0 && g.basis_count == 0
            })
    }
    /// Global dispatch gate, not just the current group's table (src/kvq_fast.hpp:51-66).
    pub fn vit2_eligible(&self) -> bool {
        let mut found = false;
        for g in &self.groups {
            if g.mode == 0 && matches!(g.bits, 4 | 5) {
                found = true;
                if self.tables[g.table]
                    .codes
                    .iter()
                    .any(|&c| !(-1100..=1100).contains(&c))
                {
                    return false;
                }
            }
        }
        found
    }
}
