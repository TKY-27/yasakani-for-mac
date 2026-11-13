mod common;
use std::path::PathBuf;
use yskn_engine::{Engine, Error, Kernel, MappedModel, Result};
use yskn_format::{Container, Role};
use yskn_ref::compressed_linear::CompressedLinear;

fn main() -> Result<()> {
    let path = PathBuf::from(common::argument_value(&mut std::env::args().skip(1))?);
    // SAFETY: this read-only verification requires the input file to remain unchanged until exit.
    let model = unsafe { MappedModel::open(&path)? };
    let tensors = &model.header().tensors;
    let mut selected = Vec::new();
    for needle in ["in_proj_qkv", "down_proj", "self_attn.q_proj"] {
        let d = tensors
            .iter()
            .find(|d| d.role == Role::Body && d.name.contains(needle))
            .ok_or_else(|| Error::InvalidModel(format!("missing verification tensor {needle}")))?;
        selected.push((d.name.clone(), d.physical_rows));
    }
    let head = tensors
        .iter()
        .find(|d| d.role == Role::Head)
        .ok_or_else(|| Error::InvalidModel("missing vocabulary head".into()))?;
    if head.physical_rows < 4096 {
        return Err(Error::InvalidModel(
            "verification head needs 4096 rows".into(),
        ));
    }
    selected.push((head.name.clone(), 4096));
    // The scalar reader owns only the tensor currently being verified, never a second model.
    let mut reader = Container::open(&path)?;
    let mut engines = Kernel::available_variants()
        .into_iter()
        .map(|kernel| Engine::new(kernel, Engine::default_threads()))
        .collect::<Result<Vec<_>>>()?;
    for (name, n) in selected {
        let matrix = model.matrix(&name)?;
        let k = matrix.descriptor().cols;
        let reference = CompressedLinear::new(reader.load_tensor(&name)?, false)?;
        let (a, s) = common::activations(k, 3);
        let mut expected = Vec::new();
        for m in 0..3 {
            expected.extend(reference.weighted_sums(
                &a[m * k..(m + 1) * k],
                &s[m * (k / 512)..(m + 1) * (k / 512)],
                n,
            )?);
        }
        for engine in &mut engines {
            let actual = engine.weighted_sums_prefix(&matrix, &a, &s, n)?;
            if let Some((index, (&actual, &expected))) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (a, b))| a != b)
            {
                eprintln!(
                    "FAIL tensor={name} kernel={} activation_row={} output_row={} actual={actual} reference={expected}",
                    engine.kernel().name(),
                    index / n,
                    index % n
                );
                return Err(Error::InvalidModel("kernel/reference mismatch".into()));
            }
            println!(
                "PASS tensor={name} N={n} K={k} M=3 kernel={}",
                engine.kernel().name()
            );
        }
    }
    Ok(())
}
