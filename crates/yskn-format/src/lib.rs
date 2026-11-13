//! Validated container reading from [FORMAT.md](../../../docs/FORMAT.md).

#![forbid(unsafe_code)]

mod floats;
mod signs;
mod tensor;

pub use floats::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use signs::{SignRule, signs};
pub use tensor::{
    Layout, Role, Rotation, Tensor, TensorDescriptor, Vector, VectorDescriptor, VectorType,
};

use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
    Unsupported(String),
    Allocation,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "container I/O: {error}"),
            Self::Json(error) => write!(formatter, "container JSON: {error}"),
            Self::Invalid(message) => write!(formatter, "invalid container: {message}"),
            Self::Unsupported(message) => write!(formatter, "unsupported container: {message}"),
            Self::Allocation => formatter.write_str("container allocation failed"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub struct Header {
    pub version: u32,
    pub header_bytes: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
    pub model: Value,
    pub tensors: Vec<TensorDescriptor>,
    pub vectors: Vec<VectorDescriptor>,
}

/// A reader retaining only the header until a tensor or vector is requested.
pub struct Container<R: Read + Seek> {
    reader: R,
    header: Header,
}

impl Container<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(File::open(path)?)
    }
}

impl Container<Cursor<Vec<u8>>> {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::new(Cursor::new(bytes))
    }
}

impl<R: Read + Seek> Container<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        let actual_length = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut prefix = [0_u8; 32];
        reader.read_exact(&mut prefix)?;
        if prefix[..4] != *b"YSKN" {
            return Err(invalid("magic must be YSKN"));
        }
        let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
        if version != 1 {
            return Err(Error::Unsupported(format!("version {version}")));
        }
        let header_bytes = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
        let data_offset = u64::from_le_bytes(prefix[16..24].try_into().unwrap());
        let file_bytes = u64::from_le_bytes(prefix[24..32].try_into().unwrap());
        if actual_length != file_bytes {
            return Err(invalid("declared file length differs from actual length"));
        }
        let header_end = 32_u64
            .checked_add(header_bytes)
            .ok_or_else(|| invalid("header length overflow"))?;
        if header_end > data_offset || data_offset > file_bytes {
            return Err(invalid("header or first data offset lies outside the file"));
        }
        let mut json = allocate(as_usize(header_bytes)?)?;
        json.resize(as_usize(header_bytes)?, 0);
        reader.read_exact(&mut json)?;
        let root: Value = serde_json::from_slice(&json)?;
        let object = object(&root, "header")?;
        let model = required(object, "model")?.clone();
        if !model.is_object() {
            return Err(invalid("model must be an object"));
        }
        let mut names = HashSet::new();
        let mut tensors = Vec::new();
        for entry in array(required(object, "tensors")?, "tensors")? {
            let descriptor = TensorDescriptor::parse(entry, data_offset, file_bytes)?;
            if !names.insert(descriptor.name.clone()) {
                return Err(invalid(format!("duplicate parameter {}", descriptor.name)));
            }
            tensors.push(descriptor);
        }
        let mut vectors = Vec::new();
        for entry in array(required(object, "vectors")?, "vectors")? {
            let descriptor = VectorDescriptor::parse(entry, data_offset, file_bytes)?;
            if !names.insert(descriptor.name.clone()) {
                return Err(invalid(format!("duplicate parameter {}", descriptor.name)));
            }
            vectors.push(descriptor);
        }

        // The format requires zeros between the JSON object and first data block.
        let mut padding_remaining = data_offset - header_end;
        let mut buffer = [0_u8; 4096];
        while padding_remaining != 0 {
            let count = padding_remaining.min(buffer.len() as u64) as usize;
            reader.read_exact(&mut buffer[..count])?;
            if buffer[..count].iter().any(|&byte| byte != 0) {
                return Err(invalid("nonzero bytes between header and first data block"));
            }
            padding_remaining -= count as u64;
        }
        Ok(Self {
            reader,
            header: Header {
                version,
                header_bytes,
                data_offset,
                file_bytes,
                model,
                tensors,
                vectors,
            },
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let descriptor = self
            .header
            .tensors
            .iter()
            .find(|descriptor| descriptor.name == name)
            .cloned()
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let stream = self.read_block(descriptor.stream_offset, descriptor.stream_bytes)?;
        let table = self.read_floats(descriptor.table_offset, 8192, VectorType::F16)?;
        let scales = self.read_floats(
            descriptor.scale_offset,
            descriptor.physical_rows,
            VectorType::F16,
        )?;
        let input_signs = signs(
            descriptor.input_rotation.rule,
            descriptor.input_rotation.sign_seed,
            descriptor.input_rotation.sign_offset,
            descriptor.cols,
        )?;
        let output_signs = descriptor
            .output_rotation
            .as_ref()
            .map(|rotation| {
                signs(
                    rotation.rule,
                    rotation.sign_seed,
                    rotation.sign_offset,
                    descriptor.physical_rows,
                )
            })
            .transpose()?;
        Ok(Tensor::new(
            descriptor,
            stream,
            table,
            scales,
            input_signs,
            output_signs,
        ))
    }

    pub fn load_vector(&mut self, name: &str) -> Result<Vector> {
        let descriptor = self
            .header
            .vectors
            .iter()
            .find(|descriptor| descriptor.name == name)
            .cloned()
            .ok_or_else(|| invalid(format!("missing vector {name}")))?;
        let values = self.read_floats(descriptor.offset, descriptor.len, descriptor.dtype)?;
        Ok(Vector { descriptor, values })
    }

    fn read_block(&mut self, offset: u64, length: u64) -> Result<Vec<u8>> {
        validate_region(
            offset,
            length,
            self.header.data_offset,
            self.header.file_bytes,
        )?;
        let length = as_usize(length)?;
        let mut bytes = allocate(length)?;
        bytes.resize(length, 0);
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn read_floats(&mut self, offset: u64, count: usize, dtype: VectorType) -> Result<Vec<f32>> {
        let length = (count as u64)
            .checked_mul(2)
            .ok_or_else(|| invalid("vector byte length overflow"))?;
        let bytes = self.read_block(offset, length)?;
        let mut values = allocate(count)?;
        for pair in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            values.push(match dtype {
                VectorType::F16 => f16_to_f32(bits),
                VectorType::Bf16 => bf16_to_f32(bits),
            });
        }
        Ok(values)
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub(crate) fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}

pub(crate) fn as_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("length does not fit address space"))
}

pub(crate) fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{context} must be an object")))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{context} must be an array")))
}

pub(crate) fn required<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))
}

pub(crate) fn number(object: &Map<String, Value>, key: &str) -> Result<u64> {
    required(object, key)?
        .as_u64()
        .ok_or_else(|| invalid(format!("{key} must be an unsigned integer")))
}

pub(crate) fn number_or(object: &Map<String, Value>, key: &str, default: u64) -> Result<u64> {
    if object.contains_key(key) {
        number(object, key)
    } else {
        Ok(default)
    }
}

pub(crate) fn text<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    required(object, key)?
        .as_str()
        .ok_or_else(|| invalid(format!("{key} must be a string")))
}

pub(crate) fn text_or<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str> {
    if object.contains_key(key) {
        text(object, key)
    } else {
        Ok(default)
    }
}

pub(crate) fn validate_region(offset: u64, length: u64, first: u64, end: u64) -> Result<()> {
    let region_end = offset
        .checked_add(length)
        .ok_or_else(|| invalid("data offset plus length overflow"))?;
    if offset < first || region_end > end {
        return Err(invalid(format!(
            "data region {offset}+{length} lies outside payload"
        )));
    }
    Ok(())
}
