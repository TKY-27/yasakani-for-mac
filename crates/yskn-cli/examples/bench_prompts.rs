//! `bench_prompts MODEL_FOLDER`: the `yskn bench` prompts as token IDs, one JSON
//! object per line (`{"task":..,"tokens":[..]}`), for `draft_trace` (D21).
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = std::env::args()
        .nth(1)
        .ok_or("usage: bench_prompts MODEL_FOLDER")?;
    let formatter = yskn_text::ChatFormatter::load(folder)?;
    for (task, text, thinking) in yskn::cli::BENCH_TASKS {
        let (tokens, _) = formatter.prompt(
            &json!([{"role":"user","content":text}]),
            None,
            &json!({"enable_thinking":thinking}),
        )?;
        println!("{}", json!({"task":task,"tokens":tokens}));
    }
    Ok(())
}
