//! Matrix addressing and trellis states from [FORMAT.md §4](../../../docs/FORMAT.md#4-decoding-a-matrix).

use crate::{
    Error, Result, SignRule, allocate, as_usize, invalid, number, number_or, object, text, text_or,
    validate_region,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Row,
    Warp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Body,
    Head,
    Embedding,
    Draft,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rotation {
    pub blocks: usize,
    pub sign_seed: u64,
    pub sign_offset: u64,
    pub rule: SignRule,
}

impl Rotation {
    fn parse(value: &Value, blocks: usize, allow_none: bool) -> Result<Option<Self>> {
        if allow_none && value.is_null() {
            return Ok(None);
        }
        let rotation = object(value, "rotation")?;
        let kind = text(rotation, "kind")?;
        if matches!(kind, "H512" | "none") && number_or(rotation, "qm_off", 0)? != 0 {
            return Err(invalid(format!("rotation {kind} must have qm_off=0")));
        }
        match kind {
            "none" if allow_none => return Ok(None),
            "H512" => {}
            _ => return Err(Error::Unsupported(format!("rotation {kind}"))),
        }
        if as_usize(number(rotation, "m")?)? != blocks {
            return Err(invalid(
                "rotation block count differs from tensor dimensions",
            ));
        }
        let rule = match text_or(rotation, "rule", "splitmix64")? {
            "splitmix64" => SignRule::SplitMix64,
            "mulberry32_7919_13" => SignRule::Mulberry32,
            rule => return Err(Error::Unsupported(format!("sign rule {rule}"))),
        };
        Ok(Some(Self {
            blocks,
            sign_seed: number(rotation, "sign_seed")?,
            sign_offset: number(rotation, "sign_offset")?,
            rule,
        }))
    }
}

#[derive(Clone, Debug)]
pub struct TensorDescriptor {
    pub name: String,
    pub role: Role,
    pub rows: usize,
    pub cols: usize,
    pub physical_rows: usize,
    pub bits: usize,
    pub layout: Layout,
    pub stream_offset: u64,
    pub stream_bytes: u64,
    pub table_offset: u64,
    pub scale_offset: u64,
    pub input_rotation: Rotation,
    pub output_rotation: Option<Rotation>,
}

impl TensorDescriptor {
    pub(crate) fn parse(value: &Value, first: u64, end: u64) -> Result<Self> {
        let tensor = object(value, "tensor")?;
        let name = text(tensor, "name")?.to_owned();
        if name.is_empty() {
            return Err(invalid("empty tensor name"));
        }
        let role = match text(tensor, "role")? {
            "body" => Role::Body,
            "head" => Role::Head,
            "embedding" => Role::Embedding,
            "draft" => Role::Draft,
            role => return Err(Error::Unsupported(format!("tensor role {role}"))),
        };
        let rows = as_usize(number(tensor, "rows")?)?;
        let cols = as_usize(number(tensor, "cols")?)?;
        let physical_rows = as_usize(number_or(tensor, "physical_rows", rows as u64)?)?;
        if rows == 0 || cols == 0 || cols % 512 != 0 || physical_rows < rows {
            return Err(invalid(format!("invalid matrix dimensions for {name}")));
        }
        for (key, expected) in [("l", 12), ("d", 2), ("t", 256)] {
            if number_or(tensor, key, expected)? != expected {
                return Err(Error::Unsupported(format!(
                    "{name}: {key} must be {expected}"
                )));
            }
        }
        let bits = as_usize(number(tensor, "bits")?)?;
        if !(4..=8).contains(&bits) {
            return Err(Error::Unsupported(format!("{name}: bits must be 4..8")));
        }
        let start = text_or(tensor, "start", "zero")?;
        if start != "zero" {
            return Err(Error::Unsupported(format!("{name}: start={start}")));
        }
        if tensor.get("rar").is_some_and(|value| !value.is_null()) {
            return Err(Error::Unsupported(format!("{name}: rar is unspecified")));
        }
        if number_or(tensor, "bias_off", 0)? != 0 {
            return Err(Error::Unsupported(format!("{name}: bias_off must be zero")));
        }
        let layout = match text_or(tensor, "layout", "warp")? {
            "row" => Layout::Row,
            "warp" => Layout::Warp,
            layout => return Err(Error::Unsupported(format!("layout {layout}"))),
        };
        let sequences = physical_rows
            .checked_mul(cols / 256)
            .ok_or_else(|| invalid("sequence count overflow"))?;
        if layout == Layout::Warp && sequences % 32 != 0 {
            return Err(invalid("warp layout requires a multiple of 32 sequences"));
        }
        let expected_stream_bytes = (sequences as u64)
            .checked_mul(bits as u64)
            .and_then(|value| value.checked_mul(16))
            .ok_or_else(|| invalid("code stream length overflow"))?;
        let stream_offset = number(tensor, "stream_off")?;
        let stream_bytes = number(tensor, "stream_bytes")?;
        if stream_bytes != expected_stream_bytes {
            return Err(invalid(format!(
                "{name}: stream_bytes differs from matrix shape"
            )));
        }
        // Word addressing below uses usize; check its entire range at load time.
        as_usize(stream_bytes)?;
        let table_offset = number(tensor, "lut_off")?;
        let scale_offset = number(tensor, "scale_off")?;
        let scale_bytes = (physical_rows as u64)
            .checked_mul(2)
            .ok_or_else(|| invalid("scale length overflow"))?;
        validate_region(stream_offset, stream_bytes, first, end)?;
        validate_region(table_offset, 16_384, first, end)?;
        validate_region(scale_offset, scale_bytes, first, end)?;
        let input_rotation = if let Some(rotation) = tensor.get("rot_in") {
            Rotation::parse(rotation, cols / 512, false)?
                .ok_or_else(|| invalid("input rotation is required"))?
        } else {
            Rotation {
                blocks: cols / 512,
                sign_seed: 7,
                sign_offset: 0,
                rule: SignRule::Mulberry32,
            }
        };
        let output_rotation = if let Some(rotation) = tensor.get("rot_out") {
            Rotation::parse(rotation, rows / 512, true)?
        } else if matches!(role, Role::Head | Role::Embedding) {
            None
        } else {
            Some(Rotation {
                blocks: rows / 512,
                sign_seed: 7,
                sign_offset: cols as u64,
                rule: SignRule::Mulberry32,
            })
        };
        if output_rotation.is_some() && (rows % 512 != 0 || physical_rows % 512 != 0) {
            return Err(invalid("output rotation requires whole 512-row blocks"));
        }
        Ok(Self {
            name,
            role,
            rows,
            cols,
            physical_rows,
            bits,
            layout,
            stream_offset,
            stream_bytes,
            table_offset,
            scale_offset,
            input_rotation,
            output_rotation,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorType {
    F16,
    Bf16,
}

#[derive(Clone, Debug)]
pub struct VectorDescriptor {
    pub name: String,
    pub offset: u64,
    pub len: usize,
    pub dtype: VectorType,
}

impl VectorDescriptor {
    pub(crate) fn parse(value: &Value, first: u64, end: u64) -> Result<Self> {
        let vector = object(value, "vector")?;
        let name = text(vector, "name")?.to_owned();
        if name.is_empty() {
            return Err(invalid("empty vector name"));
        }
        let offset = number(vector, "off")?;
        let count = number(vector, "n")?;
        let length = count
            .checked_mul(2)
            .ok_or_else(|| invalid("vector length overflow"))?;
        validate_region(offset, length, first, end)?;
        let dtype = match text_or(vector, "dtype", "f16")? {
            "f16" => VectorType::F16,
            "bf16" => VectorType::Bf16,
            dtype => return Err(Error::Unsupported(format!("vector dtype {dtype}"))),
        };
        Ok(Self {
            name,
            offset,
            len: as_usize(count)?,
            dtype,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Vector {
    pub descriptor: VectorDescriptor,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    descriptor: TensorDescriptor,
    stream: Vec<u8>,
    table: Vec<f32>,
    scales: Vec<f32>,
    input_signs: Vec<i8>,
    output_signs: Option<Vec<i8>>,
}

impl Tensor {
    pub(crate) fn new(
        descriptor: TensorDescriptor,
        stream: Vec<u8>,
        table: Vec<f32>,
        scales: Vec<f32>,
        input_signs: Vec<i8>,
        output_signs: Option<Vec<i8>>,
    ) -> Self {
        Self {
            descriptor,
            stream,
            table,
            scales,
            input_signs,
            output_signs,
        }
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn rows(&self) -> usize {
        self.descriptor.rows
    }

    pub fn cols(&self) -> usize {
        self.descriptor.cols
    }

    pub fn physical_rows(&self) -> usize {
        self.descriptor.physical_rows
    }

    pub fn bits(&self) -> usize {
        self.descriptor.bits
    }

    pub fn table(&self) -> &[f32] {
        &self.table
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn input_signs(&self) -> &[i8] {
        &self.input_signs
    }

    pub fn output_signs(&self) -> Option<&[i8]> {
        self.output_signs.as_deref()
    }

    /// Read one state, including zero extension before the sequence begins.
    pub fn state(&self, row: usize, sequence: usize, step: usize) -> Result<u16> {
        if row >= self.physical_rows() || sequence >= self.cols() / 256 || step >= 128 {
            return Err(invalid("trellis coordinate out of range"));
        }
        let end = self.bits() * (step + 1);
        let mut state = 0_u16;
        for state_bit in 0..12 {
            let position = end as isize - 12 + state_bit as isize;
            if position >= 0 {
                let position = position as usize;
                let word = self.sequence_word(row, sequence, position / 32);
                state |= (((word >> (position % 32)) & 1) as u16) << state_bit;
            }
        }
        Ok(state)
    }

    fn sequence_word(&self, row: usize, sequence: usize, word: usize) -> u32 {
        let descriptor = &self.descriptor;
        let word_index = match descriptor.layout {
            Layout::Row => {
                (row * (descriptor.cols / 256) + sequence) * (4 * descriptor.bits) + word
            }
            Layout::Warp => {
                let sequence_index = sequence * descriptor.physical_rows + row;
                let group = sequence_index / 32;
                let lane = sequence_index % 32;
                ((group * descriptor.bits + word / 4) * 32 + lane) * 4 + word % 4
            }
        };
        let start = word_index * 4;
        // The immutable descriptor and stream were range-checked together on load.
        u32::from_le_bytes(self.stream[start..start + 4].try_into().unwrap())
    }

    pub fn row_states(&self, row: usize) -> Result<Vec<u16>> {
        if row >= self.physical_rows() {
            return Err(invalid("tensor row out of range"));
        }
        let mut states = allocate(self.cols() / 2)?;
        for sequence in 0..self.cols() / 256 {
            for step in 0..128 {
                states.push(self.state(row, sequence, step)?);
            }
        }
        Ok(states)
    }

    pub fn rotated_row(&self, row: usize) -> Result<Vec<f32>> {
        let states = self.row_states(row)?;
        let scale = self.scales[row];
        let mut values = allocate(self.cols())?;
        for state in states {
            for component in 0..2 {
                // FORMAT §8 requires widening both factors before this binary32 product.
                values.push(scale * self.table[usize::from(state) * 2 + component]);
            }
        }
        Ok(values)
    }
}
