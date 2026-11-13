use std::{fs::File, io::Cursor, ops::Range, path::Path, sync::Arc};

use memmap2::Mmap;
use yskn_format::{Container, Header, TensorDescriptor, f16_to_f32, signs};
use yskn_ref::compressed_linear::quantize_weight_table;

use crate::{Error, Result};

/// One read-only mapping, shared by all matrix handles (including after this is dropped).
pub struct MappedModel {
    mapping: Arc<Mmap>,
    header: Header,
}

impl MappedModel {
    /// Validate the complete container header and every payload range using the reader.
    ///
    /// # Safety
    /// The file must not be modified or truncated until this model and every cloned
    /// matrix have been dropped. This is the same external-file invariant as `Mmap`.
    /// A malformed but stable file is safe and returns a validation error.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the caller guarantees file stability for every mapping owner.
        let mapping = unsafe { Mmap::map(&file)? };
        let header = Container::new(Cursor::new(&mapping[..]))?.header().clone();
        Ok(Self {
            mapping: Arc::new(mapping),
            header,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn matrix(&self, name: &str) -> Result<Matrix> {
        let descriptor = self
            .header
            .tensors
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| Error::InvalidModel(format!("missing tensor {name}")))?
            .clone();
        let code_range = region(
            &self.mapping,
            descriptor.stream_offset,
            descriptor.stream_bytes,
        )?;
        let table = floats(&self.mapping, descriptor.table_offset, 8192)?;
        let scales = floats(
            &self.mapping,
            descriptor.scale_offset,
            descriptor.physical_rows,
        )?;
        let (table, effective_row_scale) = quantize_weight_table(&table, &scales)?;
        let input = &descriptor.input_rotation;
        let input_signs = signs(
            input.rule,
            input.sign_seed,
            input.sign_offset,
            descriptor.cols,
        )?;
        let output_signs = descriptor
            .output_rotation
            .as_ref()
            .map(|r| signs(r.rule, r.sign_seed, r.sign_offset, descriptor.physical_rows))
            .transpose()?;
        Ok(Matrix(Arc::new(MatrixData {
            mapping: Arc::clone(&self.mapping),
            descriptor,
            code_range,
            table: table.try_into().expect("quantized 8192-entry table"),
            effective_row_scale,
            input_signs,
            output_signs,
        })))
    }
}

fn region(bytes: &[u8], offset: u64, length: u64) -> Result<Range<usize>> {
    let invalid = || Error::InvalidModel("mapped region exceeds address space or file".into());
    let start = usize::try_from(offset).map_err(|_| invalid())?;
    let length = usize::try_from(length).map_err(|_| invalid())?;
    let end = start
        .checked_add(length)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(invalid)?;
    Ok(start..end)
}

fn floats(bytes: &[u8], offset: u64, count: usize) -> Result<Vec<f32>> {
    let length = (count as u64)
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidModel("float length overflow".into()))?;
    Ok(bytes[region(bytes, offset, length)?]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

#[derive(Clone)]
pub struct Matrix(pub(crate) Arc<MatrixData>);

pub(crate) struct MatrixData {
    mapping: Arc<Mmap>,
    pub descriptor: TensorDescriptor,
    code_range: Range<usize>,
    pub table: [i8; 8192],
    effective_row_scale: Vec<f32>,
    input_signs: Vec<i8>,
    output_signs: Option<Vec<i8>>,
}

impl Matrix {
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.0.descriptor
    }
    /// Little-endian code words, viewed as bytes to support unaligned file offsets.
    pub fn code_words(&self) -> &[u8] {
        &self.0.mapping[self.0.code_range.clone()]
    }
    pub fn weight_table8(&self) -> &[i8; 8192] {
        &self.0.table
    }
    pub fn effective_row_scale(&self) -> &[f32] {
        &self.0.effective_row_scale
    }
    pub fn input_signs(&self) -> &[i8] {
        &self.0.input_signs
    }
    pub fn output_signs(&self) -> Option<&[i8]> {
        self.0.output_signs.as_deref()
    }
}
