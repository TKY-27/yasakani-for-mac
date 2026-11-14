use sha2::{Digest, Sha256};
mod support;

use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use support::{
    kv_profile::{Profile, mixed_profile},
    synthetic_model,
};
use yskn_format::Container;
use yskn_ref::{Error, KvMode, Model, kv_profile::KvProfile, model::model_location_with_kv};

const LAYERS: &[&str] = &["linear_attention", "full_attention", "full_attention"];
fn model(bytes: &[u8], mode: KvMode, profile: Option<Arc<KvProfile>>) -> yskn_ref::Result<Model> {
    Model::from_container_with_kv(
        Container::from_bytes(bytes.to_vec()).unwrap(),
        mode,
        profile,
    )
}
fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|r| r.iter().map(|v| v.to_bits()).collect())
        .collect()
}
fn hash(rows: &[Vec<u32>]) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            rows.iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    )
}

#[test]
fn all_modes_freeze_logits_and_profile_modes_preserve_long_pass_reset_and_continuation() {
    let bytes = synthetic_model(107, LAYERS);
    let profile = Arc::new(KvProfile::parse(&mixed_profile()).unwrap());
    assert!(profile.packable(0, 0) && profile.packable(0, 1));
    assert!(profile.packable(1, 0) && !profile.packable(1, 1));
    let tokens: Vec<_> = (0..73).map(|row| (row * 17 + 4) % 128).collect();
    let mut digests = Vec::new();
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        let mut whole = model(&bytes, mode, Some(Arc::clone(&profile))).unwrap();
        let mut individual = whole.new_session().unwrap();
        let mut chunks = whole.new_session().unwrap();
        for model in [&mut whole, &mut individual, &mut chunks] {
            model.prefill(&[8, 21, 5]).unwrap();
        }
        whole.prefill(&tokens).unwrap();
        let expected = bits(whole.row_logits());
        assert_eq!(expected.len(), 73);
        assert!(expected.windows(2).any(|p| p[0] != p[1]));
        let value = hash(&expected);
        digests.push(value);
        if mode == KvMode::Int8 {
            continue;
        } // existing Int8 tests already cover these paths
        let mut actual = Vec::new();
        for &token in &tokens {
            individual.prefill(&[token]).unwrap();
            actual.extend(bits(individual.row_logits()));
        }
        assert_eq!(actual, expected, "{mode:?} single rows");
        actual.clear();
        let mut start = 0;
        for count in [1, 31, 2, 33, 6] {
            chunks.prefill(&tokens[start..start + count]).unwrap();
            start += count;
            actual.extend(bits(chunks.row_logits()));
        }
        assert_eq!(start, 73);
        assert_eq!(actual, expected, "{mode:?} uneven chunks");
        let position = whole.position();
        assert!(whole.prefill(&[3, 128]).is_err());
        assert!(whole.run_provisional(&[3, 128]).is_err());
        assert!(!whole.has_provisional_pass());
        assert_eq!(whole.position(), position);
        assert_eq!(bits(whole.row_logits()), expected);
        for model in [&mut whole, &mut individual, &mut chunks] {
            model.prefill(&[19, 11]).unwrap();
        }
        assert_eq!(bits(whole.row_logits()), bits(individual.row_logits()));
        assert_eq!(bits(whole.row_logits()), bits(chunks.row_logits()));
        // Reset with different inputs checks both removal of old cache and retained mode/profile.
        whole.reset().unwrap();
        individual.reset().unwrap();
        assert_eq!(whole.position(), 0);
        assert!(whole.logits().is_none());
        whole.prefill(&[7, 3, 1, 9]).unwrap();
        let reset = bits(whole.row_logits());
        actual.clear();
        for token in [7, 3, 1, 9] {
            individual.prefill(&[token]).unwrap();
            actual.extend(bits(individual.row_logits()));
        }
        assert_eq!(reset, actual);
        // Partial provisional acceptance must preserve encoded state through replay.
        whole.run_provisional(&[17, 6, 21]).unwrap();
        assert!(whole.commit(4).is_err());
        assert_eq!(whole.position(), 4);
        whole.commit(2).unwrap();
        individual.prefill(&[17, 6]).unwrap();
        whole.prefill(&[13]).unwrap();
        individual.prefill(&[13]).unwrap();
        assert_eq!(bits(whole.row_logits()), bits(individual.row_logits()));
    }
    assert_ne!(digests[0], digests[1]);
    assert_ne!(digests[0], digests[2]);
    assert_ne!(digests[1], digests[2]);
    // Frozen source-port results, not CUDA golden captures. Int8 also retains D11's old digests.
    assert_eq!(
        digests,
        [
            "1e9a723c945f8b0b36a5204b16cf237500fdc2183727e5beb07b841b148114fd",
            "3b952134c81b86e93601de75379c10e80ac898cae92e30275f22265d781710dd",
            "5d474ca9559bcc6d8eb367a2634e578842c0813fc66a55a862f054f4521422e4"
        ]
    );
}

#[test]
fn profile_modes_reject_absent_profiles_and_wrong_attention_counts_or_heads() {
    let bytes = synthetic_model(109, LAYERS);
    for mode in [KvMode::Int8Profile, KvMode::Kv5] {
        assert!(
            matches!(model(&bytes, mode, None), Err(Error::InvalidModel(message))
            if message.contains("requires a KVQP profile"))
        );
        for (layers, heads) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
            let profile = Arc::new(KvProfile::parse(&Profile::new(layers, heads).bytes()).unwrap());
            assert!(
                matches!(model(&bytes, mode, Some(profile)), Err(Error::InvalidInput(message))
                if message.contains("profile/model dimensions differ"))
            );
        }
    }
}

#[test]
fn codec_error_in_later_model_row_is_atomic_and_guard_uses_absolute_position() {
    let bytes = synthetic_model(113, &["linear_attention", "full_attention"]);
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(i16::MAX);
    for group in &mut profile.groups {
        group.guard = 2;
    }
    let profile = Arc::new(KvProfile::parse(&profile.bytes()).unwrap());
    for mode in [KvMode::Int8Profile, KvMode::Kv5] {
        let mut whole = model(&bytes, mode, Some(Arc::clone(&profile))).unwrap();
        let mut control = whole.new_session().unwrap();
        whole.prefill(&[3]).unwrap();
        control.prefill(&[3]).unwrap();
        let before = bits(whole.row_logits());
        // Position 1 can execute, position 2 fails inside the profile codec.
        assert!(
            matches!(whole.prefill(&[5, 7]), Err(Error::InvalidInput(message)) if message.contains("overflow"))
        );
        assert_eq!(whole.position(), 1);
        assert_eq!(bits(whole.row_logits()), before);
        whole.prefill(&[11]).unwrap();
        control.prefill(&[11]).unwrap();
        assert_eq!(whole.position(), 2);
        assert_eq!(bits(whole.row_logits()), bits(control.row_logits()));
        // A second call at absolute position 2 must fail even though its local row index is 0.
        assert!(whole.prefill(&[9]).is_err());
        assert_eq!(whole.position(), 2);
    }
}

struct Folder(PathBuf);
impl Folder {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-05b-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn manifest(&self, value: serde_json::Value) {
        fs::write(
            self.0.join("yasakani.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn folder_modes_select_profile_manifest_default_and_preserve_plain_int8_default() {
    let folder = Folder::new();
    fs::write(folder.0.join("weights.yskn"), synthetic_model(127, LAYERS)).unwrap();
    fs::create_dir(folder.0.join("profiles")).unwrap();
    fs::write(folder.0.join("profiles/synthetic.bin"), mixed_profile()).unwrap();
    folder.manifest(serde_json::json!({"files": {"weights":"weights.yskn", "kv_profile":"profiles/synthetic.bin"}, "max_ctx":65}));
    let mut baseline = Model::load_folder(&folder.0).unwrap();
    let mut explicit = Model::load_folder_with(&folder.0, KvMode::Int8).unwrap();
    baseline.prefill(&[3, 7]).unwrap();
    explicit.prefill(&[3, 7]).unwrap();
    assert_eq!(bits(baseline.row_logits()), bits(explicit.row_logits()));
    let mut outputs = vec![bits(baseline.row_logits())];
    for mode in [KvMode::Int8Profile, KvMode::Kv5] {
        let mut model = Model::load_folder_with(&folder.0, mode).unwrap();
        assert_eq!(model.context_capacity(), 128);
        assert!(model.prefill(&[0; 129]).is_err());
        assert_eq!(model.position(), 0);
        model.prefill(&[3, 7]).unwrap();
        outputs.push(bits(model.row_logits()));
    }
    assert_ne!(outputs[0], outputs[1]);
    assert_ne!(outputs[0], outputs[2]);
    assert_ne!(outputs[1], outputs[2]);
    folder.manifest(serde_json::json!({"files":{"weights":"weights.yskn"}}));
    assert!(Model::load_folder_with(&folder.0, KvMode::Kv5).is_err());
    fs::write(folder.0.join("kvq_u5.bin"), mixed_profile()).unwrap();
    Model::load_folder_with(&folder.0, KvMode::Kv5).unwrap();
    fs::write(folder.0.join("kvq_u5.bin"), b"bad profile").unwrap();
    assert!(Model::load_folder_with(&folder.0, KvMode::Kv5).is_err());
    // Plain mode never requires or parses a profile, even if its manifest member is invalid.
    folder.manifest(serde_json::json!({"files":{"weights":"weights.yskn", "kv_profile":42}}));
    Model::load_folder(&folder.0).unwrap();
    assert!(Model::load_folder_with(&folder.0, KvMode::Kv5).is_err());
}

#[test]
fn manifest_profile_and_weights_paths_cannot_escape_folder() {
    let folder = Folder::new();
    for key in ["weights", "kv_profile"] {
        for path in [
            "../outside.bin",
            "/tmp/outside.bin",
            "a/../../outside.bin",
            "",
        ] {
            let mut files =
                serde_json::json!({"weights":"missing.yskn", "kv_profile":"missing.bin"});
            files[key] = serde_json::json!(path);
            folder.manifest(serde_json::json!({"files": files}));
            assert!(
                matches!(model_location_with_kv(&folder.0, KvMode::Kv5), Err(Error::InvalidModel(message))
                if message.contains(key) && message.contains("relative"))
            );
        }
    }
    #[cfg(unix)]
    {
        let outside = Folder::new();
        fs::write(outside.0.join("external.bin"), mixed_profile()).unwrap();
        std::os::unix::fs::symlink(outside.0.join("external.bin"), folder.0.join("link.bin"))
            .unwrap();
        for key in ["weights", "kv_profile"] {
            let mut files =
                serde_json::json!({"weights":"missing.yskn", "kv_profile":"missing.bin"});
            files[key] = serde_json::json!("link.bin");
            folder.manifest(serde_json::json!({"files":files}));
            assert!(
                matches!(model_location_with_kv(&folder.0, KvMode::Kv5), Err(Error::InvalidModel(message))
                if message.contains(key) && message.contains("inside"))
            );
        }
    }
    assert!(matches!(
        model_location_with_kv(&folder.0.join("direct.yskn"), KvMode::Kv5),
        Err(Error::InvalidModel(_))
    ));
}
