//! Single-matrix synthetic containers shared by the CPU and Metal kernel tests.
#![allow(dead_code)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use yskn_format::f32_to_f16;

pub const DATA_OFFSET: usize = 4096;

pub struct TempFile(pub PathBuf);

impl TempFile {
    pub fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-linear-{}-{}-{}.yskn",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

pub fn random(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) as u32
}

/// The format permits arbitrary stream bits. The scalar reader independently
/// interprets their layout and trellis windows, including sequence boundaries.
pub fn synthetic(
    bits: usize,
    layout: &str,
    rows: usize,
    physical_rows: usize,
    cols: usize,
    output_rotation: bool,
) -> Vec<u8> {
    let stream_bytes = physical_rows * (cols / 256) * bits * 16;
    let table_offset = DATA_OFFSET + stream_bytes;
    let scale_offset = table_offset + 16_384;
    let file_bytes = scale_offset + physical_rows * 2;
    let header = json!({
        "model": {"source": "synthetic integration test; no model weights"},
        "tensors": [{
            "name": "linear", "role": if output_rotation { "body" } else { "head" },
            "rows": rows, "physical_rows": physical_rows, "cols": cols,
            "bits": bits, "layout": layout,
            "stream_off": DATA_OFFSET, "stream_bytes": stream_bytes,
            "lut_off": table_offset, "scale_off": scale_offset,
            "rot_in": {"kind": "H512", "m": cols / 512,
                "sign_seed": 91, "sign_offset": 13, "rule": "splitmix64"},
            "rot_out": if output_rotation {
                json!({"kind": "H512", "m": rows / 512,
                    "sign_seed": 7, "sign_offset": cols, "rule": "mulberry32_7919_13"})
            } else { Value::Null }
        }],
        "vectors": []
    });
    let mut bytes = vec![0; file_bytes];
    bytes[..4].copy_from_slice(b"YSKN");
    bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
    bytes[16..24].copy_from_slice(&(DATA_OFFSET as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&(file_bytes as u64).to_le_bytes());
    replace_header(&mut bytes, header);
    let mut seed = 0x3187_5bde_0219_faff;
    for byte in &mut bytes[DATA_OFFSET..table_offset] {
        *byte = random(&mut seed) as u8;
    }
    for pair in bytes[table_offset..scale_offset]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
    {
        let value = (random(&mut seed) % 2049) as f32 / 1024.0 - 1.0;
        pair.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    for (row, pair) in bytes[scale_offset..]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .enumerate()
    {
        let value = (row as f32 - physical_rows as f32 / 2.0) / 128.0;
        pair.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    bytes
}

pub fn replace_header(bytes: &mut [u8], header: Value) {
    let header = serde_json::to_vec(&header).unwrap();
    assert!(32 + header.len() <= DATA_OFFSET);
    bytes[8..16].copy_from_slice(&(header.len() as u64).to_le_bytes());
    bytes[32..DATA_OFFSET].fill(0);
    bytes[32..32 + header.len()].copy_from_slice(&header);
}

pub fn change_header(bytes: &[u8], change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut header = serde_json::from_slice(&bytes[32..32 + length]).unwrap();
    change(&mut header);
    replace_header(&mut bytes, header);
    bytes
}

pub fn assert_sums(actual: &[i64], expected: &[i64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: output shape");
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        panic!("{context}: sum {index}: {actual} != scalar {expected}");
    }
}
