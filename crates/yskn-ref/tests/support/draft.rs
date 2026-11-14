//! Seeded, tiny draft artifacts. All numbers are synthetic; no published weights.
use super::Writer;
use serde_json::json;

pub fn container(seed: u64) -> Vec<u8> {
    container_with_heads(seed, 4, 4)
}
pub fn container_with_heads(seed: u64, query_heads: usize, kv_heads: usize) -> Vec<u8> {
    let mut w = Writer::new(seed, false);
    w.tensor("bdraft2.fc", "draft", 512, 2560, 0.015625);
    for i in 0..2 {
        let p = format!("bdraft2.layers.{i}.");
        for (name, rows, cols) in [
            ("q", query_heads * 128, 512),
            ("k", kv_heads * 128, 512),
            ("v", kv_heads * 128, 512),
            ("o", 512, query_heads * 128),
            ("gate", 512, 512),
            ("up", 512, 512),
            ("down", 512, 512),
            ("akp", 128, 512),
            ("mkp", 128, 512),
        ] {
            w.tensor(&format!("{p}{name}"), "draft", rows, cols, 0.015625);
        }
        for name in ["ln1", "ln2"] {
            w.vector(&format!("{p}{name}"), 512, 1.0, 0.0625);
        }
        for name in ["q_norm", "k_norm"] {
            w.vector(&format!("{p}{name}"), 128, 1.0, 0.0625);
        }
        for name in ["abase", "mbase"] {
            w.vector(&format!("{p}{name}"), 4 * 512, 0.25, 0.125);
        }
    }
    w.vector("bdraft2.hidden_norm", 512, 1.0, 0.0625);
    w.vector("bdraft2.norm", 512, 1.0, 0.0625);
    w.vector("bdraft2.sel.hp", 32 * 512, 0.0, 0.0625);
    w.vector("bdraft2.sel.pred", 32 * 32, 0.0, 0.25);
    w.vector("bdraft2.sel.succ", 32 * 32, 0.0, 0.25);
    // Source assumes all projections use the fc sign prefix; ensure that remains
    // observable even when a matrix's own descriptor has different signs.
    w.tensors[1]["rot_in"]["sign_seed"] = json!(99);
    w.finish(json!({ "arch":"bdraft2","hidden":512,"ffn":512,"vocab":32,"layers":2,"heads":query_heads,"kv_heads":kv_heads,"head_dim":128,"window":8,"rank":32,"eps":1e-6,"block":8,"top_k":16 }))
}
pub fn shead(seed: u64) -> Vec<u8> {
    shead_with_hidden(seed, 512)
}
pub fn shead_with_hidden(seed: u64, hidden: usize) -> Vec<u8> {
    let mut w = Writer::new(seed, false);
    let mut b = b"SHEAD001".to_vec();
    for n in [32_i32, hidden as i32, 512, 12345] {
        b.extend_from_slice(&n.to_le_bytes());
    }
    for _ in 0..512 * hidden {
        b.extend_from_slice(&(w.signed_unit() * 0.03125).to_le_bytes());
    }
    for _ in 0..32 * 512 {
        b.push((w.random() % 255) as u8);
    }
    for _ in 0..32 {
        b.extend_from_slice(&0.03125_f32.to_le_bytes());
    }
    for _ in 0..32 * hidden {
        b.push((w.random() % 255) as u8);
    }
    for _ in 0..32 {
        b.extend_from_slice(&(1.0_f32 / 4096.0).to_le_bytes());
    }
    b
}
pub fn options() -> yskn_ref::draft::DraftOptions {
    yskn_ref::draft::DraftOptions {
        mask_token: 31,
        shortlist: 24,
        ..Default::default()
    }
}
pub fn embedding(token: u32) -> yskn_ref::Result<Vec<f32>> {
    Ok((0..512)
        .map(|i| {
            yskn_ref::portable_math::bfloat(
                ((i * 17 + token as usize * 11) % 101) as f32 / 64.0 - 0.75,
            )
        })
        .collect())
}
pub fn features(row: usize) -> Vec<f32> {
    (0..2560)
        .map(|i| yskn_ref::portable_math::bfloat(((i * 13 + row * 7) % 79) as f32 / 32.0 - 1.0))
        .collect()
}
