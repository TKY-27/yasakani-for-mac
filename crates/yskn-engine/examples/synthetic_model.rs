//! Write a seeded synthetic container with released-file layout for benchmarks:
//! `synthetic_model OUT.yskn [LAYERS]` (hidden 5120, ffn 17408, 4096-row head).
#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

use std::{fs::OpenOptions, io::Write};

fn main() -> std::io::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .expect("usage: synthetic_model OUT.yskn [LAYERS]");
    let layers: usize = args
        .next()
        .map(|v| v.to_str().and_then(|v| v.parse().ok()).expect("LAYERS"))
        .unwrap_or(8);
    let kinds: Vec<_> = (0..layers)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let bytes = support::synthetic_model_released_layout(0x04a, &kinds, 5120, 17408, 4096);
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&bytes)
}
