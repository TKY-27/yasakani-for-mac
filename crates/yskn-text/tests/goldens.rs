use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use yskn_text::{
    ChatFormatter, OutputStream, StreamDecoder, ToolCallFilter, parse_tool_call, tool_schemas,
};

fn ids(case: &Value) -> Vec<u32> {
    serde_json::from_value(case["ids"].clone()).unwrap()
}

#[test]
fn upstream_python_goldens() {
    let Some(folder) = std::env::var_os("YSKN_MODEL_DIR") else {
        // libtest captures stdout on success; stderr through a direct write keeps
        // the required CI skip reason visible even without --nocapture.
        use std::io::Write;
        writeln!(std::io::stderr(), "skipped: YSKN_MODEL_DIR is not set").unwrap();
        return;
    };
    let fmt = ChatFormatter::load(folder).expect("YSKN_MODEL_DIR must be a usable model folder");
    let data: Value = serde_json::from_str(include_str!("data/goldens.json")).unwrap();
    let mut total = 0;
    for category in ["selfcheck", "chat", "tools", "errors", "raw"] {
        let cases = data[category].as_array().unwrap();
        for case in cases {
            let name = format!("{category}/{}", case["name"]);
            let tools = (!case["tools"].is_null()).then_some(&case["tools"]);
            if let Some(expected) = case.get("error") {
                let actual = fmt
                    .render(&case["messages"], tools, &case["options"])
                    .unwrap_err();
                assert_eq!(actual.to_string(), expected.as_str().unwrap(), "{name}");
                continue;
            }
            let rendered = if let Some(text) = case.get("text") {
                text.as_str().unwrap().to_owned()
            } else {
                fmt.render(&case["messages"], tools, &case["options"])
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            assert_eq!(
                rendered,
                case["rendered"].as_str().unwrap(),
                "render {name}"
            );
            let encoded = fmt.encode(&rendered).unwrap();
            assert_eq!(encoded, ids(case), "encode {name}");
            assert_eq!(
                fmt.decode(&encoded).unwrap(),
                case["decoded"].as_str().unwrap(),
                "decode {name}"
            );
            if let Some(expected) = case.get("starts_inside_thinking") {
                let (prompt, inside) = fmt
                    .prompt(&case["messages"], tools, &case["options"])
                    .unwrap();
                assert_eq!(prompt, encoded, "prompt {name}");
                assert_eq!(inside, expected.as_bool().unwrap(), "thinking {name}");
            }
            if let Some(hash) = case.get("prompt_sha256") {
                let mut sha = Sha256::new();
                for id in encoded {
                    sha.update(id.to_le_bytes());
                }
                assert_eq!(
                    format!("{:x}", sha.finalize()),
                    hash.as_str().unwrap(),
                    "hash {name}"
                );
            }
        }
        eprintln!("goldens {category}: {}/{} passed", cases.len(), cases.len());
        total += cases.len();
    }
    for case in data["decode"].as_array().unwrap() {
        assert_eq!(
            fmt.decode(&ids(case)).unwrap(),
            case["decoded"].as_str().unwrap(),
            "decode {}",
            case["name"]
        );
    }
    for case in data["stream"].as_array().unwrap() {
        let all = ids(case);
        let mut stream = StreamDecoder::new(fmt.tokenizer());
        let mut combined = String::new();
        for (&id, expected) in all.iter().zip(case["pieces"].as_array().unwrap()) {
            let piece = stream.push(&[id]).unwrap();
            assert_eq!(piece, expected.as_str().unwrap(), "stream {}", case["name"]);
            combined.push_str(&piece);
        }
        let tail = stream.flush().unwrap();
        assert_eq!(tail, case["tail"].as_str().unwrap());
        combined.push_str(&tail);
        assert_eq!(combined, case["decoded"].as_str().unwrap());
        for chunk in [2, 7, 1000] {
            let mut stream = StreamDecoder::new(fmt.tokenizer());
            let mut text = String::new();
            for ids in all.chunks(chunk) {
                text.push_str(&stream.push(ids).unwrap());
            }
            text.push_str(&stream.flush().unwrap());
            assert_eq!(text, combined, "chunk {chunk}, {}", case["name"]);
        }
    }
    for case in data["output"].as_array().unwrap() {
        let all = ids(case);
        let hidden: Option<Vec<u32>> = case
            .get("hidden_ids")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let make_output = || {
            let inside = case["in_reasoning"].as_bool().unwrap();
            match &hidden {
                Some(ids) => OutputStream::with_hidden_ids(&fmt, inside, ids),
                None => OutputStream::new(&fmt, inside),
            }
        };
        let mut output = make_output();
        let mut expected_events = Vec::new();
        for (i, expected) in case["pieces"].as_array().unwrap().iter().enumerate() {
            let events = output.feed(&all[..i + 1], false).unwrap();
            assert_eq!(
                serde_json::to_value(&events).unwrap(),
                *expected,
                "output {} id {i}",
                case["name"]
            );
            expected_events.extend(events);
        }
        let tail = output.feed(&all, true).unwrap();
        assert_eq!(serde_json::to_value(&tail).unwrap(), case["tail"]);
        expected_events.extend(tail);
        assert_eq!(json!(output.think_end_at), case["think_end_at"]);
        for chunk in [3, 1000] {
            let mut output = make_output();
            let mut events = Vec::new();
            for end in (chunk..all.len()).step_by(chunk) {
                events.extend(output.feed(&all[..end], false).unwrap());
            }
            events.extend(output.feed(&all, true).unwrap());
            assert_eq!(events, expected_events, "output chunk {chunk}");
            assert_eq!(json!(output.think_end_at), case["think_end_at"]);
        }
    }
    for category in ["decode", "stream", "output"] {
        let count = data[category].as_array().unwrap().len();
        eprintln!("goldens {category}: {count}/{count} passed");
        total += count;
    }
    eprintln!("model goldens: {total}/{total} passed");
}

#[test]
fn upstream_tool_filter_goldens_without_model() {
    let data: Value = serde_json::from_str(include_str!("data/goldens.json")).unwrap();
    let cases = data["tool_filter"].as_array().unwrap();
    for case in cases {
        let mut filter = ToolCallFilter::new(Some(&case["tools"]));
        for (chunk, expected) in case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["pieces"].as_array().unwrap())
        {
            assert_eq!(
                json!(filter.push(chunk.as_str().unwrap())),
                *expected,
                "{}",
                case["name"]
            );
        }
        assert_eq!(json!(filter.flush()), case["tail"], "{}", case["name"]);
        assert_eq!(json!(filter.calls), case["calls"]);
    }
    eprintln!(
        "goldens tool_filter: {}/{} passed",
        cases.len(),
        cases.len()
    );
}

#[test]
fn upstream_parse_goldens_without_model() {
    let data: Value = serde_json::from_str(include_str!("data/goldens.json")).unwrap();
    let cases = data["parse"].as_array().unwrap();
    for case in cases {
        let schemas = tool_schemas(Some(&case["tools"]));
        assert_eq!(
            json!(parse_tool_call(case["block"].as_str().unwrap(), &schemas)),
            case["result"],
            "{}",
            case["name"]
        );
    }
    eprintln!("goldens parse: {}/{} passed", cases.len(), cases.len());
}
