//! Read-only header inspection; never opens the main weights or loads draft payloads.
use std::fs::File;
use yskn_format::{Container, signs};
use yskn_ref::draft::{DraftDimensions, model_paths};
use yskn_ref::draft_head::HeadLayout;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = std::env::args_os()
        .nth(1)
        .ok_or("usage: draft_info MODEL_FOLDER")?;
    let (draft, head) = model_paths(folder)?;
    let c = Container::open(&draft)?;
    let h = c.header();
    let d = DraftDimensions::read(h)?;
    println!("draft: {}", draft.display());
    println!(
        "bytes={} header_bytes={} data_offset={}",
        h.file_bytes, h.header_bytes, h.data_offset
    );
    println!("model: {}", h.model);
    let fc = h
        .tensors
        .iter()
        .find(|t| t.name == "bdraft2.fc")
        .ok_or("missing fc")?;
    let common = signs(
        fc.input_rotation.rule,
        fc.input_rotation.sign_seed,
        fc.input_rotation.sign_offset,
        fc.cols,
    )?;
    for t in &h.tensors {
        println!(
            "tensor {} {}x{} physical={} bits={} layout={:?} rot_in={:?} rot_out={:?}",
            t.name,
            t.rows,
            t.cols,
            t.physical_rows,
            t.bits,
            t.layout,
            t.input_rotation,
            t.output_rotation
        );
        if t.cols > common.len() {
            return Err("input sign prefix too short".into());
        }
        let local = signs(
            t.input_rotation.rule,
            t.input_rotation.sign_seed,
            t.input_rotation.sign_offset,
            t.cols,
        )?;
        if local != common[..t.cols] || t.output_rotation.is_some() {
            return Err("shipping draft sign/rotation invariant failed".into());
        }
    }
    for v in &h.vectors {
        println!("vector {} n={} dtype={:?}", v.name, v.len, v.dtype);
    }
    println!(
        "fc input signs={} all matrix signs equal its prefix; all output rotations absent",
        common.len()
    );
    let sh = HeadLayout::read(&mut File::open(&head)?)?;
    if (sh.vocabulary, sh.hidden) != (d.vocabulary, d.hidden) {
        return Err("draft/head dimension mismatch".into());
    }
    println!("shead: {}", head.display());
    println!(
        "V={} H={} rank={} fourth_word={} bytes={}",
        sh.vocabulary, sh.hidden, sh.rank, sh.reserved, sh.file_bytes
    );
    for (name, range) in [
        "projection_f32",
        "approx_i8",
        "approx_scale_f32",
        "rescore_i8",
        "rescore_scale_f32",
    ]
    .into_iter()
    .zip(&sh.arrays)
    {
        println!(
            "{name}: offset={} bytes={}",
            range.start,
            range.end - range.start
        );
    }
    if h.tensors.len() != 1 + 9 * d.layers || h.vectors.len() != 5 + 6 * d.layers {
        return Err("unexpected shipping parameter count".into());
    }
    for i in 0..d.layers {
        let p = format!("bdraft2.layers.{i}.");
        for (name, rows, cols) in [
            ("q", d.query_heads * 128, d.hidden),
            ("k", d.kv_heads * 128, d.hidden),
            ("v", d.kv_heads * 128, d.hidden),
            ("o", d.hidden, d.query_heads * 128),
            ("gate", d.feed_forward, d.hidden),
            ("up", d.feed_forward, d.hidden),
            ("down", d.hidden, d.feed_forward),
            ("akp", d.hidden / 4, d.hidden),
            ("mkp", d.hidden / 4, d.hidden),
        ] {
            if !h
                .tensors
                .iter()
                .any(|t| t.name == format!("{p}{name}") && t.rows == rows && t.cols == cols)
            {
                return Err(format!("bad {p}{name} shape").into());
            }
        }
    }
    let mut vectors = vec![
        ("bdraft2.hidden_norm".to_string(), d.hidden),
        ("bdraft2.norm".to_string(), d.hidden),
        ("bdraft2.sel.hp".to_string(), d.correction_rank * d.hidden),
        (
            "bdraft2.sel.pred".to_string(),
            d.vocabulary * d.correction_rank,
        ),
        (
            "bdraft2.sel.succ".to_string(),
            d.vocabulary * d.correction_rank,
        ),
    ];
    for i in 0..d.layers {
        for (name, n) in [
            ("ln1", d.hidden),
            ("ln2", d.hidden),
            ("q_norm", 128),
            ("k_norm", 128),
            ("abase", 4 * d.hidden),
            ("mbase", 4 * d.hidden),
        ] {
            vectors.push((format!("bdraft2.layers.{i}.{name}"), n));
        }
    }
    if fc.rows != d.hidden || fc.cols != 5 * d.hidden {
        return Err("bad context projection shape".into());
    }
    for (name, n) in vectors {
        if !h
            .vectors
            .iter()
            .any(|v| v.name == name && v.len == n && v.dtype == yskn_format::VectorType::Bf16)
        {
            return Err(format!("bad vector {name}").into());
        }
    }
    println!(
        "correction BF16 vectors: hp=[{},{}], pred/succ=[{},{}]",
        d.correction_rank, d.hidden, d.vocabulary, d.correction_rank
    );
    println!("PASS: header dimensions, shapes, sign prefixes, rotations and SHEAD array extents");
    Ok(())
}
