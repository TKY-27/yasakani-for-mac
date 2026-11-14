//! Write a small, fully packed synthetic benchmark folder (no trained payload):
//! `synthetic_kv_model OUT_DIR` (one attention layer, two KV/query heads).
#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

fn main() -> std::io::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .expect("usage: synthetic_kv_model OUT_DIR");
    let path = std::path::Path::new(&path);
    std::fs::create_dir(path)?;
    std::fs::write(
        path.join("weights.yskn"),
        support::synthetic_model(0x05_0a, &["full_attention"]),
    )?;
    let mut profile = support::kv_profile::Profile::new(1, 2);
    for group in &mut profile.groups {
        group.mean = Some(std::array::from_fn(|i| (i as f32 - 128.0) / 1024.0));
    }
    std::fs::write(path.join("kvq_u5.bin"), profile.bytes())?;
    std::fs::write(
        path.join("yasakani.json"),
        r#"{"files":{"weights":"weights.yskn","kv_profile":"kvq_u5.bin"},"max_ctx":2048}"#,
    )
}
