use std::{collections::BTreeMap, env, fs};
use yskn_ref::kv_profile::KvProfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os()
        .nth(1)
        .ok_or("usage: kv_profile_info PROFILE")?;
    let bytes = fs::read(path)?;
    let p = KvProfile::parse(&bytes)?;
    println!("KVQP v1 bytes={}", bytes.len());
    println!(
        "dimensions: width=256 attention_layers={} kv_heads={} groups={}",
        p.attention_layers(),
        p.kv_heads(),
        p.groups().len()
    );
    let mut modes = BTreeMap::new();
    for g in p.groups() {
        *modes.entry((g.mode, g.bits)).or_insert(0) += 1;
    }
    for ((mode, bits), count) in modes {
        println!("groups: mode={mode} bits={bits} count={count}");
    }
    println!(
        "multipliers: count={} values={:?}",
        p.multipliers().len(),
        p.multipliers()
    );
    println!(
        "packed_bits={} vit2_eligible={}",
        p.packed_bits(),
        p.vit2_eligible()
    );
    for side in 0..2 {
        let layers: Vec<_> = (0..p.attention_layers())
            .filter(|&l| p.packable(l, side))
            .collect();
        println!(
            "packable_{}: count={} ordinals={layers:?}",
            if side == 0 { "key" } else { "value" },
            layers.len()
        );
    }
    for (i, t) in p.tables().iter().enumerate() {
        println!(
            "table[{i}]: states=256 chain=64 fraction=8 gain={:?} min={} max={}",
            t.gain,
            t.codes.iter().min().unwrap(),
            t.codes.iter().max().unwrap()
        );
    }
    for layer in 0..p.attention_layers() {
        let start = layer * p.kv_heads() * 2;
        let groups = &p.groups()[start..start + p.kv_heads() * 2];
        let facts: Vec<_> = groups
            .iter()
            .map(|g| {
                format!(
                    "{}/{}/{}/{}/{}/{}",
                    g.mode,
                    g.bits,
                    g.table,
                    g.guard,
                    g.basis_count,
                    u8::from(g.mean.is_some())
                )
            })
            .collect();
        println!(
            "layer[{layer:02}] head-major K,V (mode/bits/table/guard/basis/mean): {}",
            facts.join(" ")
        );
    }
    Ok(())
}
