//! Explicit stage-only benchmark: real header/norm/profile, synthetic projected rows.
//! The caller MUST acquire the model.lock protocol before running this ignored test.
use super::*;
use yskn_ref::attention::{AttentionDimensions, CacheToken};

struct Inputs {
    prefix: Vec<CacheToken>,
    queries: Vec<Vec<Vec<f32>>>,
    means: Vec<Option<[f32; 256]>>,
}

#[test]
#[ignore = "real-model stage benchmark; requires model.lock and YSKN_MODEL_DIR"]
fn bench_real_attention_shapes() {
    assert!(super::super::neon_available());
    let folder = std::env::var("YSKN_MODEL_DIR").expect("YSKN_MODEL_DIR required");
    let context: usize = std::env::var("YSKN_BENCH_CONTEXT")
        .unwrap_or("2000".into())
        .parse()
        .unwrap();
    let rows: usize = std::env::var("YSKN_BENCH_ROWS")
        .unwrap_or("16".into())
        .parse()
        .unwrap();
    let passes: usize = std::env::var("YSKN_BENCH_PASSES")
        .unwrap_or("5".into())
        .parse()
        .unwrap();
    assert!(rows > 0 && passes > 0);
    let (path, _, profile) = yskn_ref::model::model_location_with_kv(
        std::path::Path::new(&folder),
        yskn_ref::KvMode::Kv5,
    )
    .unwrap();
    let mut container = yskn_format::Container::open(path).unwrap();
    let header = container.header().model.clone();
    let dims = AttentionDimensions {
        query_heads: header["heads"].as_u64().unwrap() as usize,
        kv_heads: header["kv_heads"].as_u64().unwrap() as usize,
        head_dim: header["head_dim"].as_u64().unwrap() as usize,
        rotary_dim: 64,
    };
    assert_eq!(header["arch"], "qwen3_5");
    let mut inputs = Vec::new();
    let mut seed = 751u32;
    let mut sample = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as i32 as f32 / i32::MAX as f32
    };
    for (index, kind) in header["layer_types"].as_array().unwrap().iter().enumerate() {
        if kind != "full_attention" {
            continue;
        }
        let layer = Attention::new(
            dims,
            container
                .load_vector(&format!("layers.{index}.self_attn.q_norm"))
                .unwrap()
                .values,
            container
                .load_vector(&format!("layers.{index}.self_attn.k_norm"))
                .unwrap()
                .values,
            header["eps"].as_f64().unwrap() as f32,
        )
        .unwrap()
        .with_kv(yskn_ref::KvMode::Kv5, profile.clone(), inputs.len())
        .unwrap();
        let mut state = layer.new_state().unwrap();
        let q: Vec<_> = (0..dims.query_heads * 512).map(|_| sample()).collect();
        let k: Vec<_> = (0..dims.kv_heads * 256).map(|_| sample()).collect();
        let v: Vec<_> = (0..dims.kv_heads * 256).map(|_| sample()).collect();
        layer
            .append_prepared(&mut state, 0, &layer.prepare_row(0, &q, &k, &v).unwrap())
            .unwrap();
        let mut bank = Vec::new();
        let mut queries = Vec::new();
        let mut means = Vec::new();
        for _ in 0..rows.max(16) {
            let q: Vec<_> = (0..dims.query_heads * 512).map(|_| sample()).collect();
            let k: Vec<_> = (0..dims.kv_heads * 256).map(|_| sample()).collect();
            let v: Vec<_> = (0..dims.kv_heads * 256).map(|_| sample()).collect();
            let prepared = layer.prepare_row(1, &q, &k, &v).unwrap();
            let mut row_queries = Vec::new();
            for head in 0..dims.query_heads {
                layer
                    .output_head_with(&state, 1, &prepared, head, |operands| {
                        row_queries.push(operands.query.to_vec());
                        if head == 0 {
                            bank.push(operands.current.clone());
                        }
                        if queries.is_empty() {
                            means.push(operands.value_mean.copied());
                        }
                        Ok(vec![0.0; 256])
                    })
                    .unwrap();
            }
            queries.push(row_queries);
        }
        let prefix = (0..context + rows)
            .map(|i| bank[i % bank.len()].clone())
            .collect();
        inputs.push(Inputs {
            prefix,
            queries,
            means,
        });
    }
    let threads = std::env::var("YSKN_BENCH_THREADS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(crate::Engine::default_threads());
    let mut engine = crate::Engine::new(crate::Kernel::fastest(), threads).unwrap();
    println!(
        "stage-only real header/norm/profile; synthetic projected bank; layers={} heads={} kv_heads={} context={context} rows={rows} threads={}",
        inputs.len(),
        dims.query_heads,
        dims.kv_heads,
        engine.threads()
    );
    let mut times = Vec::new();
    for pass in 0..=passes {
        let start = std::time::Instant::now();
        let mut setup = 0.0;
        let mut work = 0.0;
        let mut combine = 0.0;
        for input in &inputs {
            let phase = std::time::Instant::now();
            let operands = (0..rows * dims.query_heads)
                .map(|i| {
                    let row = i / dims.query_heads;
                    let head = i % dims.query_heads;
                    AttentionOperands {
                        query: &input.queries[row][head],
                        kv_head: head / (dims.query_heads / dims.kv_heads),
                        prefix: &input.prefix[..context + row],
                        current: &input.prefix[context + row],
                        value_mean: input.means[head].as_ref(),
                    }
                })
                .collect();
            let batch = grouped::Batch::new(operands, dims.kv_heads);
            setup += phase.elapsed().as_secs_f64();
            let phase = std::time::Instant::now();
            let partials = engine
                .map_range(0..batch.tasks(), |task| batch.run(task))
                .unwrap();
            work += phase.elapsed().as_secs_f64();
            let phase = std::time::Instant::now();
            let splits = batch.collect(partials);
            let outputs = engine
                .map_range(0..splits.len(), |i| combine_splits(&splits[i]))
                .unwrap();
            combine += phase.elapsed().as_secs_f64();
            std::hint::black_box(outputs);
        }
        let elapsed = start.elapsed().as_secs_f64();
        if pass > 0 {
            times.push(elapsed);
        }
        println!(
            "pass={pass} attention_s={elapsed:.9} setup_s={setup:.9} work_s={work:.9} combine_s={combine:.9}"
        );
    }
    times.sort_by(f64::total_cmp);
    println!(
        "attention min_s={:.9} median_s={:.9} per_row_min_s={:.9} per_row_median_s={:.9}",
        times[0],
        times[passes / 2],
        times[0] / rows as f64,
        times[passes / 2] / rows as f64
    );
}
