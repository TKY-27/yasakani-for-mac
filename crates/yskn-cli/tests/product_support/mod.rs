//! Generated model artifacts and a separate scalar oracle; never published weights.
pub mod allocation;

#[allow(dead_code)]
#[path = "../../../yskn-ref/tests/support/mod.rs"]
pub mod model;

use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use yskn::{
    engine::Sampling,
    real_engine::{DraftMode, RealEngine, RealEngineConfig},
};
use yskn_engine::{Backend, KvMode};
use yskn_ref::{
    Sampler, SamplingParameters, SamplingVersion,
    speculation::{GenerationLimits, NoProposer, generate},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
pub const CONTEXT: usize = 512;

pub struct Folder(pub PathBuf);
impl Folder {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "yskn-product-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let folder = Self(path);
        folder.write(
            "weights.yskn",
            model::synthetic_model_with_dimensions(
                19,
                &["full_attention", "linear_attention"],
                512,
                512,
                32,
            ),
        );
        folder.write("profile.bin", model::kv_profile::Profile::new(1, 2).bytes());
        // One residual tap rather than the released model's five. Keep the
        // seeded payload and absolute offsets; only the FC input width changes.
        let mut draft = model::draft::container(23);
        let len = u64::from_le_bytes(draft[8..16].try_into().unwrap()) as usize;
        let mut header: Value = serde_json::from_slice(&draft[32..32 + len]).unwrap();
        let fc = header["tensors"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|t| t["name"] == "bdraft2.fc")
            .unwrap();
        fc["cols"] = json!(512);
        fc["stream_bytes"] = json!(fc["stream_bytes"].as_u64().unwrap() / 5);
        fc["rot_in"]["m"] = json!(1);
        let encoded = serde_json::to_vec(&header).unwrap();
        assert!(encoded.len() <= len);
        draft[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        draft[32..32 + len].fill(0);
        draft[32..32 + encoded.len()].copy_from_slice(&encoded);
        folder.write("draft.yskn", draft);
        folder.write("draft.shead", model::draft::shead(37));
        folder.json("yasakani.json", json!({"model_id":"synthetic-product", "max_ctx":CONTEXT,"files":{"weights":"weights.yskn","kv_profile":"profile.bin","draft":"draft.yskn","draft_head":"draft.shead"}}));
        // WordLevel permits the real fixed special IDs while keeping every
        // ordinary prompt and generated ID inside the 32-entry model head.
        // IDs 0-9 are the digits the formatter requires. One word per ID:
        // WordLevel inverts the vocab through a randomly seeded map, so an ID
        // with two words would decode differently on each load.
        let mut vocab = serde_json::Map::new();
        for id in 0..32 {
            vocab.insert(
                match id {
                    0..10 => id.to_string(),
                    31 => "[UNK]".into(),
                    _ => format!("t{id}"),
                },
                json!(id),
            );
        }
        let specials = [
            (248044, "<|endoftext|>"),
            (248045, "<|im_start|>"),
            (248046, "<|im_end|>"),
            (248047, "<think>"),
            (248048, "</think>"),
        ];
        for (id, token) in specials {
            vocab.insert(token.into(), json!(id));
        }
        let added: Vec<_> = specials.iter().map(|(id, token)|json!({"id":id,"content":token,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
        folder.json("tokenizer.json", json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}}));
        folder.json("tokenizer_config.json", json!({}));
        folder.write(
            "chat_template.jinja",
            "{% for m in messages %}{{ m.content }} {% endfor %}t10",
        );
        folder.json(
            "generation_config.json",
            json!({"temperature":0.75,"top_p":0.9,"top_k":11,"min_p":0.05}),
        );
        folder
    }
    /// CLI defaults require five capture taps and the published mask ID.
    /// Sparse zero head/selector data avoids manufacturing unused draft weights.
    pub fn neural_bench_fixture(&self) {
        self.write(
            "weights.yskn",
            model::synthetic_model_with_dimensions(19, &["full_attention"; 62], 512, 512, 248_071),
        );
        self.write(
            "profile.bin",
            model::kv_profile::Profile::new(62, 2).bytes(),
        );
        let mut draft = model::draft::container(23);
        let header_len = u64::from_le_bytes(draft[8..16].try_into().unwrap()) as usize;
        let data = u64::from_le_bytes(draft[16..24].try_into().unwrap()) as usize;
        let mut header: Value = serde_json::from_slice(&draft[32..32 + header_len]).unwrap();
        const VOCAB: u64 = 248_071;
        header["model"]["vocab"] = json!(VOCAB);
        // Enlarge only mapped selector extents, with zero-filled sparse data.
        let mut end = (draft.len() as u64 + 255) & !255;
        for vector in header["vectors"].as_array_mut().unwrap() {
            if matches!(
                vector["name"].as_str(),
                Some("bdraft2.sel.pred" | "bdraft2.sel.succ")
            ) {
                vector["off"] = json!(end);
                vector["n"] = json!(VOCAB * 32);
                end = (end + VOCAB * 32 * 2 + 255) & !255;
            }
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        assert!(32 + encoded.len() <= data);
        draft[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        draft[24..32].copy_from_slice(&end.to_le_bytes());
        draft[32..data].fill(0);
        draft[32..32 + encoded.len()].copy_from_slice(&encoded);
        self.write("draft.yskn", draft);
        fs::OpenOptions::new()
            .write(true)
            .open(self.0.join("draft.yskn"))
            .unwrap()
            .set_len(end)
            .unwrap();
        let mut head = b"SHEAD001".to_vec();
        for n in [VOCAB as i32, 512, 512, 12345] {
            head.extend_from_slice(&n.to_le_bytes());
        }
        self.write("draft.shead", head);
        let bytes = 24 + 4 * 512 * 512 + VOCAB * (512 + 4 + 512 + 4);
        fs::OpenOptions::new()
            .write(true)
            .open(self.0.join("draft.shead"))
            .unwrap()
            .set_len(bytes)
            .unwrap();
    }
    pub fn write(&self, name: &str, data: impl AsRef<[u8]>) {
        fs::write(self.0.join(name), data).unwrap();
    }
    pub fn json(&self, name: &str, value: Value) {
        self.write(name, serde_json::to_vec(&value).unwrap());
    }
    pub fn config(&self, draft: DraftMode) -> RealEngineConfig {
        RealEngineConfig {
            model_dir: self.0.clone(),
            kv: KvMode::Kv5,
            backend: Some(Backend::Cpu),
            kernel: None,
            cpu_stages: yskn_engine::CpuStages::Auto,
            metal: yskn_engine::MetalOptions::default(),
            draft,
            k: 3,
            depth: RealEngineConfig::default().depth,
            context: CONTEXT,
            threads: Some(1),
            prefill_width: 2,
        }
    }
    pub fn engine(&self, draft: DraftMode) -> RealEngine {
        self.engine_with_context(draft, CONTEXT)
    }
    pub fn engine_with_context(&self, draft: DraftMode, context: usize) -> RealEngine {
        let mut config = self.config(draft);
        config.context = context;
        let mut options = model::draft::options();
        options.capture_layers = vec![0];
        RealEngine::load_with_draft_options(config, options).unwrap()
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn greedy() -> Sampling {
    Sampling {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.0,
        seed: 0,
    }
}
pub fn sampled() -> Sampling {
    Sampling {
        temperature: 0.7314,
        top_p: 0.9134,
        top_k: 11,
        min_p: 0.0524,
        seed: -7,
    }
}

pub fn scalar(folder: &Folder, prompt: &[u32], max_new: usize, sampling: &Sampling) -> Vec<u32> {
    let mut model = yskn_ref::Model::load_folder_with(&folder.0, yskn_ref::KvMode::Kv5).unwrap();
    model.prefill(&prompt[..prompt.len() - 1]).unwrap();
    // Fixed upstream integer-thousandth boundary, independently specified here.
    let mut selector = if sampling.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::Sampled(SamplingParameters {
            temperature: (sampling.temperature * 1000.0).round_ties_even() / 1000.0,
            top_p: (sampling.top_p * 1000.0).round_ties_even() / 1000.0,
            min_p: (sampling.min_p * 1000.0).round_ties_even() / 1000.0,
            top_k: sampling.top_k as usize,
            seed: sampling.seed as u32 as u64,
            version: SamplingVersion::Gumbel,
        })
    };
    generate(
        &mut model,
        &mut NoProposer,
        &mut selector,
        prompt,
        GenerationLimits::new(max_new, CONTEXT),
    )
    .unwrap()
    .tokens
}
