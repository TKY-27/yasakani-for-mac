//! Small seeded containers written directly from docs/FORMAT.md and spec/01.

#[allow(dead_code)]
pub mod kv_profile;

use serde_json::{Value, json};
use yskn_format::{f32_to_bf16, f32_to_f16};

pub fn synthetic_model(seed: u64, layer_types: &[&str]) -> Vec<u8> {
    synthetic_model_with_dimensions(seed, layer_types, 512, 512, 128)
}

pub fn synthetic_model_with_dimensions(
    seed: u64,
    layer_types: &[&str],
    hidden: usize,
    feed_forward: usize,
    vocabulary: usize,
) -> Vec<u8> {
    build_model(
        seed,
        layer_types,
        hidden,
        feed_forward,
        vocabulary,
        false,
        false,
    )
}

/// Like the released files (FORMAT.md §6): `warp` matrices, 6-bit body, 8-bit head.
pub fn synthetic_model_released_layout(
    seed: u64,
    layer_types: &[&str],
    hidden: usize,
    feed_forward: usize,
    vocabulary: usize,
) -> Vec<u8> {
    build_model(
        seed,
        layer_types,
        hidden,
        feed_forward,
        vocabulary,
        false,
        true,
    )
}

pub fn synthetic_model_with_dense_embeddings(seed: u64, layer_types: &[&str]) -> Vec<u8> {
    build_model(seed, layer_types, 512, 512, 128, true, false)
}

fn build_model(
    seed: u64,
    layer_types: &[&str],
    hidden: usize,
    feed_forward: usize,
    vocabulary: usize,
    dense_embeddings: bool,
    released_layout: bool,
) -> Vec<u8> {
    assert!(hidden > 0 && hidden % 512 == 0);
    assert!(feed_forward > 0 && feed_forward % 512 == 0);
    assert!(vocabulary > 0);
    let mut writer = Writer::new(seed, released_layout);
    if dense_embeddings {
        writer.vector("embed_tokens", vocabulary * hidden, 0.0, 0.25);
    } else {
        writer.tensor("embed_tokens", "embedding", vocabulary, hidden, 0.25);
    }

    for (layer, kind) in layer_types.iter().enumerate() {
        let prefix = format!("layers.{layer}");
        writer.vector(&format!("{prefix}.input_layernorm"), hidden, 0.0, 0.0625);
        writer.vector(
            &format!("{prefix}.post_attention_layernorm"),
            hidden,
            0.0,
            0.0625,
        );
        writer.tensor(
            &format!("{prefix}.mlp.gate_proj"),
            "body",
            feed_forward,
            hidden,
            0.015625,
        );
        writer.tensor(
            &format!("{prefix}.mlp.up_proj"),
            "body",
            feed_forward,
            hidden,
            0.015625,
        );
        writer.tensor(
            &format!("{prefix}.mlp.down_proj"),
            "body",
            hidden,
            feed_forward,
            0.015625,
        );
        if *kind == "full_attention" {
            let prefix = format!("{prefix}.self_attn");
            writer.tensor(&format!("{prefix}.q_proj"), "body", 1024, hidden, 0.015625);
            writer.tensor(&format!("{prefix}.k_proj"), "body", 512, hidden, 0.015625);
            writer.tensor(&format!("{prefix}.v_proj"), "body", 512, hidden, 0.015625);
            writer.tensor(&format!("{prefix}.o_proj"), "body", hidden, 512, 0.015625);
            writer.vector(&format!("{prefix}.q_norm"), 256, 0.0, 0.0625);
            writer.vector(&format!("{prefix}.k_norm"), 256, 0.0, 0.0625);
        } else {
            let prefix = format!("{prefix}.linear_attn");
            writer.tensor(
                &format!("{prefix}.in_proj_qkv"),
                "body",
                1024,
                hidden,
                0.015625,
            );
            writer.tensor(
                &format!("{prefix}.in_proj_z"),
                "body",
                512,
                hidden,
                0.015625,
            );
            writer.tensor(&format!("{prefix}.out_proj"), "body", hidden, 512, 0.015625);
            writer.vector(&format!("{prefix}.in_proj_a"), 4 * hidden, 0.0, 0.03125);
            writer.vector(&format!("{prefix}.in_proj_b"), 4 * hidden, 0.0, 0.03125);
            writer.vector(&format!("{prefix}.conv1d"), 1024 * 4, 0.0, 0.25);
            writer.vector(&format!("{prefix}.A_log"), 4, -1.0, 0.125);
            writer.vector(&format!("{prefix}.dt_bias"), 4, -1.0, 0.125);
            writer.vector(&format!("{prefix}.norm"), 128, 1.0, 0.0625);
        }
    }
    writer.vector("norm", hidden, 0.0, 0.0625);
    writer.tensor("lm_head", "head", vocabulary, hidden, 0.03125);
    writer.finish(json!({
        "arch": "qwen3_5",
        "layers": layer_types.len(),
        "hidden": hidden,
        "ffn": feed_forward,
        "heads": 2,
        "kv_heads": 2,
        "head_dim": 256,
        "vocab": vocabulary,
        "eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": false,
        "layer_types": layer_types,
        "source": "seeded synthetic data; no trained model weights"
    }))
}

struct Writer {
    random_state: u64,
    released_layout: bool,
    payload: Vec<u8>,
    tensors: Vec<Value>,
    vectors: Vec<Value>,
}

impl Writer {
    fn new(seed: u64, released_layout: bool) -> Self {
        Self {
            random_state: seed,
            released_layout,
            payload: Vec::new(),
            tensors: Vec::new(),
            vectors: Vec::new(),
        }
    }

    fn random(&mut self) -> u32 {
        self.random_state = self.random_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.random_state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (value ^ (value >> 31)) as u32
    }

    fn signed_unit(&mut self) -> f32 {
        (self.random() % 2049) as f32 * (1.0 / 1024.0) - 1.0
    }

    fn block(&mut self, bytes: &[u8]) -> usize {
        let offset = (self.payload.len() + 255) & !255;
        self.payload.resize(offset, 0);
        self.payload.extend_from_slice(bytes);
        offset
    }

    fn tensor(&mut self, name: &str, role: &str, rows: usize, cols: usize, scale: f32) {
        let ordinal = self.tensors.len();
        let bits = match (self.released_layout, role) {
            (true, "head") => 8,
            (true, _) => 6,
            _ if ordinal % 2 == 0 => 6,
            _ => 8,
        };
        let layout = if role == "embedding" || (!self.released_layout && ordinal % 3 == 0) {
            "row"
        } else {
            "warp"
        };
        let physical_rows = if role == "head" {
            rows.div_ceil(64) * 64
        } else {
            rows
        };
        let stream_length = physical_rows * (cols / 256) * bits * 16;
        // Every stream is valid; random bytes exercise both state-window boundaries.
        let stream: Vec<_> = (0..stream_length).map(|_| self.random() as u8).collect();
        let stream_offset = self.block(&stream);
        let mut table = Vec::with_capacity(16_384);
        for _ in 0..8192 {
            let value = self.signed_unit();
            // Weight tables are binary16 even when parameter vectors use BF16.
            table.extend_from_slice(&f32_to_f16(value).to_le_bytes());
        }
        let table_offset = self.block(&table);
        let mut scales = Vec::with_capacity(physical_rows * 2);
        for _ in 0..physical_rows {
            let value = scale * (1.0 + 0.25 * self.signed_unit());
            scales.extend_from_slice(&f32_to_f16(value).to_le_bytes());
        }
        let scale_offset = self.block(&scales);
        let input_rotation = json!({
            "kind": "H512", "m": cols / 512, "sign_seed": 7,
            "sign_offset": 0, "rule": "mulberry32_7919_13", "qm_off": 0
        });
        let output_rotation = if role == "body" {
            json!({
                "kind": "H512", "m": rows / 512, "sign_seed": 7,
                "sign_offset": cols, "rule": "mulberry32_7919_13", "qm_off": 0
            })
        } else {
            Value::Null
        };
        self.tensors.push(json!({
            "name": name, "role": role, "rows": rows, "cols": cols,
            "physical_rows": physical_rows, "bits": bits, "l": 12, "d": 2, "t": 256,
            "layout": layout, "start": "zero", "rar": null, "bias_off": 0,
            "stream_off": stream_offset, "stream_bytes": stream_length,
            "lut_off": table_offset, "scale_off": scale_offset,
            "rot_in": input_rotation, "rot_out": output_rotation
        }));
    }

    fn vector(&mut self, name: &str, count: usize, center: f32, amplitude: f32) {
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            let value = center + amplitude * self.signed_unit();
            // Main-model vectors use the explicitly specified BF16 storage boundary.
            bytes.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
        }
        let offset = self.block(&bytes);
        self.vectors
            .push(json!({"name": name, "off": offset, "n": count, "dtype": "bf16"}));
    }

    fn finish(self, model: Value) -> Vec<u8> {
        let mut data_offset = 256;
        loop {
            let mut tensors = self.tensors.clone();
            for tensor in &mut tensors {
                for field in ["stream_off", "lut_off", "scale_off"] {
                    tensor[field] = json!(tensor[field].as_u64().unwrap() + data_offset as u64);
                }
            }
            let mut vectors = self.vectors.clone();
            for vector in &mut vectors {
                vector["off"] = json!(vector["off"].as_u64().unwrap() + data_offset as u64);
            }
            let header = serde_json::to_vec(&json!({
                "model": model, "tensors": tensors, "vectors": vectors
            }))
            .unwrap();
            let required_offset = (32 + header.len() + 255) & !255;
            if required_offset > data_offset {
                data_offset = required_offset;
                continue;
            }
            let file_bytes = data_offset + self.payload.len();
            let mut bytes = vec![0; data_offset];
            bytes[0..4].copy_from_slice(b"YSKN");
            bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
            bytes[8..16].copy_from_slice(&(header.len() as u64).to_le_bytes());
            bytes[16..24].copy_from_slice(&(data_offset as u64).to_le_bytes());
            bytes[24..32].copy_from_slice(&(file_bytes as u64).to_le_bytes());
            bytes[32..32 + header.len()].copy_from_slice(&header);
            bytes.extend_from_slice(&self.payload);
            return bytes;
        }
    }
}

#[test]
fn synthetic_writer_builds_reproducible_mixed_model() {
    let layer_types = [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ];
    let bytes = synthetic_model(19, &layer_types);
    assert_eq!(bytes, synthetic_model(19, &layer_types));
    assert_ne!(bytes, synthetic_model(20, &layer_types));
    let mut container = yskn_format::Container::from_bytes(bytes).unwrap();
    assert_eq!(container.header().model["layers"], 4);
    let tensor_names: Vec<_> = container
        .header()
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    let vector_names: Vec<_> = container
        .header()
        .vectors
        .iter()
        .map(|vector| vector.name.clone())
        .collect();
    for name in tensor_names {
        let tensor = container.load_tensor(&name).unwrap();
        assert!(
            tensor
                .table()
                .iter()
                .chain(tensor.scales())
                .all(|value| value.is_finite())
        );
        assert_eq!(tensor.row_states(0).unwrap().len(), tensor.cols() / 2);
    }
    for name in vector_names {
        assert!(
            container
                .load_vector(&name)
                .unwrap()
                .values
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn synthetic_writer_supports_other_dimensions_and_plain_embeddings() {
    let layer_types = ["linear_attention"];
    let bytes = synthetic_model_with_dimensions(21, &layer_types, 1024, 512, 65);
    let mut container = yskn_format::Container::from_bytes(bytes).unwrap();
    assert_eq!(container.header().model["hidden"], 1024);
    assert_eq!(container.header().model["ffn"], 512);
    assert_eq!(container.header().model["vocab"], 65);
    let head = container.load_tensor("lm_head").unwrap();
    assert_eq!(
        (head.rows(), head.cols(), head.physical_rows()),
        (65, 1024, 128)
    );
    let bytes = synthetic_model_with_dense_embeddings(21, &layer_types);
    let mut container = yskn_format::Container::from_bytes(bytes).unwrap();
    let embedding = container.load_vector("embed_tokens").unwrap();
    assert_eq!(embedding.values.len(), 128 * 512);
    assert_eq!(embedding.descriptor.dtype, yskn_format::VectorType::Bf16);
    assert!(container.load_tensor("embed_tokens").is_err());
}

#[test]
fn synthetic_writer_released_layout_uses_warp_and_released_bit_widths() {
    let bytes =
        synthetic_model_released_layout(23, &["linear_attention", "full_attention"], 512, 512, 65);
    let container = yskn_format::Container::from_bytes(bytes).unwrap();
    for tensor in &container.header().tensors {
        let (layout, bits) = match tensor.role {
            yskn_format::Role::Embedding => (yskn_format::Layout::Row, tensor.bits),
            yskn_format::Role::Head => (yskn_format::Layout::Warp, 8),
            _ => (yskn_format::Layout::Warp, 6),
        };
        assert_eq!(
            (tensor.layout, tensor.bits),
            (layout, bits),
            "{}",
            tensor.name
        );
    }
}
