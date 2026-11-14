#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;
use std::sync::Arc;
use yskn_engine::{Engine, Kernel, KvMode};
use yskn_ref::attention::{Attention, AttentionDimensions};
use yskn_ref::kv_profile::KvProfile;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn independent_cache_sides_match_scalar_rows_and_errors() {
    let profile = Arc::new(KvProfile::parse(&support::kv_profile::mixed_profile()).unwrap());
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let mut seed = 7919u32;
    let mut sample = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        match seed % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            3 => f32::from_bits(0x80000001),
            4 => f32::from_bits(0x3f808000),
            5 => -127.0,
            _ => (seed as i32 as f32) / (i32::MAX as f32) * 4.0,
        }
    };
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        for ordinal in 0..2 {
            let layer = Attention::new(
                AttentionDimensions {
                    query_heads: 2,
                    kv_heads: 2,
                    head_dim: 256,
                    rotary_dim: 64,
                },
                vec![0.0; 256],
                vec![0.0; 256],
                1e-6,
            )
            .unwrap()
            .with_kv(mode, Some(profile.clone()), ordinal)
            .unwrap();
            let mut scalar = layer.new_state().unwrap();
            let mut parallel = layer.new_state().unwrap();
            for position in 0..34 {
                let q: Vec<_> = (0..1024).map(|_| sample()).collect();
                let k: Vec<_> = (0..512).map(|_| sample()).collect();
                let v: Vec<_> = (0..512).map(|_| sample()).collect();
                let expected = layer.prepare_row(position, &q, &k, &v).unwrap();
                let sides = engine
                    .map_range(0..4, |i| {
                        layer.prepare_cache_side(
                            position,
                            i / 2,
                            i % 2,
                            if i % 2 == 0 { &k } else { &v },
                        )
                    })
                    .unwrap();
                let actual = layer
                    .finish_prepared(layer.prepare_queries(position, &q, &k, &v).unwrap(), sides)
                    .unwrap();
                assert_eq!(
                    bits(&layer.output_row(&parallel, position, &actual).unwrap()),
                    bits(&layer.output_row(&scalar, position, &expected).unwrap())
                );
                layer
                    .append_prepared(&mut scalar, position, &expected)
                    .unwrap();
                layer
                    .append_prepared(&mut parallel, position, &actual)
                    .unwrap();
                let (mut a, mut b) = (0, 0);
                scalar.update_digest(&mut a);
                parallel.update_digest(&mut b);
                assert_eq!(a, b);
            }
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let q = vec![0.0; 1024];
                let mut k = vec![0.0; 512];
                let v = vec![0.0; 512];
                k[7] = invalid;
                assert_eq!(
                    layer
                        .prepare_queries(0, &q, &k, &v)
                        .err()
                        .unwrap()
                        .to_string(),
                    layer.prepare_row(0, &q, &k, &v).err().unwrap().to_string()
                );
            }
        }
    }
}

#[test]
fn cpu_stage_override_rejects_unknown_values_before_opening_weights() {
    const CHILD: &str = "YSKN_STAGE_CONFIG_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // SAFETY: this path is never opened: invalid CPU configuration must fail first.
        let result = unsafe {
            yskn_engine::Model::with_engine(
                "/nonexistent-yskn-stage-config-fixture/weights.yskn",
                Engine::new(Kernel::Portable, 1).unwrap(),
                1,
            )
        };
        assert!(
            matches!(result, Err(yskn_engine::Error::InvalidInput(message)) if message == "YSKN_CPU_STAGES must be auto or reference")
        );
        return;
    }
    for value in ["", "invalid"] {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cpu_stage_override_rejects_unknown_values_before_opening_weights",
            ])
            .env(CHILD, "1")
            .env("YSKN_CPU_STAGES", value)
            .status()
            .unwrap();
        assert!(status.success());
    }
}

fn check_public_stages(variant: &str, mode: KvMode, test: &str) {
    const CHILD: &str = "YSKN_CPU_STAGE_VARIANT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Stage selection is process-wide; isolate each selector in a child.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(CHILD, variant)
            .env("YSKN_CPU_STAGES", variant)
            .status()
            .unwrap();
        assert!(status.success(), "CPU stages {variant} {mode:?} failed");
        return;
    }
    let neon = {
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    };
    println!(
        "CPU stages: requested={variant}, ran={}",
        if variant == "auto" && neon {
            "NEON"
        } else {
            "scalar"
        }
    );
    let folder = std::env::temp_dir().join(format!("yskn-cpu-stages-{}", std::process::id()));
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(
        folder.join("weights.yskn"),
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
    )
    .unwrap();
    std::fs::write(
        folder.join("kvq_u5.bin"),
        support::kv_profile::mixed_profile(),
    )
    .unwrap();
    std::fs::write(
        folder.join("yasakani.json"),
        r#"{"files":{"weights":"weights.yskn"},"max_ctx":64}"#,
    )
    .unwrap();
    let mut reference = yskn_ref::Model::load_folder_with(&folder, mode).unwrap();
    // SAFETY: these synthetic weights remain immutable until both models are dropped.
    let mut model = unsafe {
        yskn_engine::Model::with_options(
            &folder,
            Engine::new(Kernel::Portable, 4).unwrap(),
            yskn_engine::Backend::Cpu,
            16,
            mode,
        )
    }
    .unwrap();
    let tokens: Vec<_> = (0..36).map(|i| (i * 7 + 11) % 65).collect();
    for chunk in [&[1, 2, 3][..], &tokens, &[17, 21]] {
        reference.prefill(chunk).unwrap();
        model.prefill(chunk).unwrap();
        assert_eq!(model.row_logits().len(), reference.row_logits().len());
        for (actual, expected) in model.row_logits().iter().zip(reference.row_logits()) {
            assert_eq!(bits(actual), bits(expected), "{variant} {mode:?}");
        }
        assert_eq!(
            model.state_digest(),
            reference.state_digest(),
            "{variant} {mode:?}"
        );
    }
    std::fs::remove_dir_all(folder).unwrap();
}

macro_rules! stage_case {
    ($name:ident, $variant:expr, $mode:expr) => {
        #[test]
        fn $name() {
            check_public_stages($variant, $mode, stringify!($name));
        }
    };
}
stage_case!(
    auto_int8_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Int8
);
stage_case!(
    auto_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Int8Profile
);
stage_case!(
    auto_kv5_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Kv5
);
stage_case!(
    forced_reference_int8_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Int8
);
stage_case!(
    forced_reference_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Int8Profile
);
stage_case!(
    forced_reference_kv5_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Kv5
);
