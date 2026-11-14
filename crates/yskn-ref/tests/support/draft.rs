//! Seeded, tiny draft artifacts. All numbers are synthetic; no published weights.
use super::Writer;
use serde_json::json;

pub fn container(seed: u64) -> Vec<u8> {
    container_with_heads(seed, 4, 4)
}
pub fn container_with_heads(seed: u64, query_heads: usize, kv_heads: usize) -> Vec<u8> {
    container_shape(
        seed,
        &Shape {
            query_heads,
            kv_heads,
            ..Shape::default()
        },
    )
}
/// Draft dimensions of a synthetic container (defaults: the original small fixture).
#[derive(Clone, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub ffn: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub window: usize,
    pub layers: usize,
    pub vocab: usize,
    pub rank: usize,
    pub slots: usize,
}
impl Default for Shape {
    fn default() -> Self {
        Self {
            hidden: 512,
            ffn: 512,
            query_heads: 4,
            kv_heads: 4,
            window: 8,
            layers: 2,
            vocab: 32,
            rank: 32,
            slots: 5,
        }
    }
}
pub fn container_shape(seed: u64, s: &Shape) -> Vec<u8> {
    let (h, qh, kvh) = (s.hidden, s.query_heads, s.kv_heads);
    let mut w = Writer::new(seed, false);
    w.tensor("bdraft2.fc", "draft", h, s.slots * h, 0.015625);
    for i in 0..s.layers {
        let p = format!("bdraft2.layers.{i}.");
        for (name, rows, cols) in [
            ("q", qh * 128, h),
            ("k", kvh * 128, h),
            ("v", kvh * 128, h),
            ("o", h, qh * 128),
            ("gate", s.ffn, h),
            ("up", s.ffn, h),
            ("down", h, s.ffn),
            ("akp", h / 4, h),
            ("mkp", h / 4, h),
        ] {
            w.tensor(&format!("{p}{name}"), "draft", rows, cols, 0.015625);
        }
        for name in ["ln1", "ln2"] {
            w.vector(&format!("{p}{name}"), h, 1.0, 0.0625);
        }
        for name in ["q_norm", "k_norm"] {
            w.vector(&format!("{p}{name}"), 128, 1.0, 0.0625);
        }
        for name in ["abase", "mbase"] {
            w.vector(&format!("{p}{name}"), 4 * h, 0.25, 0.125);
        }
    }
    w.vector("bdraft2.hidden_norm", h, 1.0, 0.0625);
    w.vector("bdraft2.norm", h, 1.0, 0.0625);
    w.vector("bdraft2.sel.hp", s.rank * h, 0.0, 0.0625);
    w.vector("bdraft2.sel.pred", s.vocab * s.rank, 0.0, 0.25);
    w.vector("bdraft2.sel.succ", s.vocab * s.rank, 0.0, 0.25);
    // Source assumes all projections use the fc sign prefix; ensure that remains
    // observable even when a matrix's own descriptor has different signs.
    w.tensors[1]["rot_in"]["sign_seed"] = json!(99);
    w.finish(json!({ "arch":"bdraft2","hidden":h,"ffn":s.ffn,"vocab":s.vocab,"layers":s.layers,"heads":qh,"kv_heads":kvh,"head_dim":128,"window":s.window,"rank":s.rank,"eps":1e-6,"block":8,"top_k":16 }))
}
pub fn shead(seed: u64) -> Vec<u8> {
    shead_with_hidden(seed, 512)
}
pub fn shead_with_hidden(seed: u64, hidden: usize) -> Vec<u8> {
    shead_shape(seed, 32, hidden)
}
pub fn shead_shape(seed: u64, vocab: usize, hidden: usize) -> Vec<u8> {
    let mut w = Writer::new(seed, false);
    let mut b = b"SHEAD001".to_vec();
    for n in [vocab as i32, hidden as i32, 512, 12345] {
        b.extend_from_slice(&n.to_le_bytes());
    }
    for _ in 0..512 * hidden {
        b.extend_from_slice(&(w.signed_unit() * 0.03125).to_le_bytes());
    }
    for _ in 0..vocab * 512 {
        b.push((w.random() % 255) as u8);
    }
    for _ in 0..vocab {
        b.extend_from_slice(&0.03125_f32.to_le_bytes());
    }
    for _ in 0..vocab * hidden {
        b.push((w.random() % 255) as u8);
    }
    for _ in 0..vocab {
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
    embedding_width(token, 512)
}
pub fn embedding_width(token: u32, hidden: usize) -> yskn_ref::Result<Vec<f32>> {
    Ok((0..hidden)
        .map(|i| {
            yskn_ref::portable_math::bfloat(
                ((i * 17 + token as usize * 11) % 101) as f32 / 64.0 - 0.75,
            )
        })
        .collect())
}
pub fn features(row: usize) -> Vec<f32> {
    features_width(row, 2560)
}
pub fn features_width(row: usize, width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| yskn_ref::portable_math::bfloat(((i * 13 + row * 7) % 79) as f32 / 32.0 - 1.0))
        .collect()
}
