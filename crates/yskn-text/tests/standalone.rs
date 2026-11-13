use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use tokenizers::{Tokenizer, decoders::byte_fallback::ByteFallback, models::wordlevel::WordLevel};
use yskn_text::{
    ChatFormatter, OutputStream, StreamDecoder, convert_parameter, parse_tool_call, tool_schemas,
};

fn tiny_tokenizer() -> Tokenizer {
    let vocab = [
        ("[UNK]", 0),
        ("<0xE3>", 1),
        ("<0x81>", 2),
        ("<0x82>", 3),
        ("<0xF0>", 4),
        ("<0x9F>", 5),
        ("<0x91>", 6),
        ("<0x8D>", 7),
        ("A", 8),
        ("<|im_start|>", 248045),
        ("<|im_end|>", 248046),
        ("<think>", 248047),
        ("</think>", 248048),
    ];
    let mut map: Vec<(String, u32)> = vocab.into_iter().map(|(s, i)| (s.into(), i)).collect();
    map.extend((0..10).map(|i| (i.to_string(), 20 + i)));
    let model = WordLevel::builder()
        .vocab(map.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteFallback::default()));
    tokenizer
}

struct Folder(PathBuf);
impl Folder {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("text-fixtures")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn tokenizer(&self) {
        tiny_tokenizer()
            .save(self.0.join("tokenizer.json"), false)
            .unwrap();
    }
    fn config(&self, cfg: Value) {
        fs::write(self.0.join("tokenizer_config.json"), cfg.to_string()).unwrap();
    }
    fn load_error(&self) -> String {
        ChatFormatter::load(&self.0)
            .err()
            .expect("must fail")
            .to_string()
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn incomplete_utf8_is_held_until_complete_and_flush_matches_decode() {
    let tokenizer = tiny_tokenizer();
    let mut stream = StreamDecoder::new(&tokenizer);
    assert_eq!(stream.push(&[1]).unwrap(), "");
    assert_eq!(stream.push(&[2]).unwrap(), "");
    assert_eq!(stream.push(&[3]).unwrap(), "あ");
    assert_eq!(stream.push(&[4, 5]).unwrap(), "");
    assert_eq!(stream.push(&[6, 7]).unwrap(), "👍");
    assert_eq!(stream.push(&[8]).unwrap(), "A");
    assert_eq!(stream.flush().unwrap(), "");
    for chunks in [1, 2, 3, 7] {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 1];
        let mut stream = StreamDecoder::new(&tokenizer);
        let mut result = String::new();
        for chunk in ids.chunks(chunks) {
            let piece = stream.push(chunk).unwrap();
            assert!(!piece.contains('\u{fffd}'));
            result.push_str(&piece);
        }
        result.push_str(&stream.flush().unwrap());
        assert_eq!(result, "あ👍A�");
        assert_eq!(result, tokenizer.decode(&ids, false).unwrap());
        assert_eq!(stream.flush().unwrap(), "");
    }
}

#[test]
fn folder_errors_are_reported_at_their_source() {
    let folder = Folder::new();
    assert!(folder.load_error().contains("tokenizer.json"));
    fs::write(folder.0.join("tokenizer.json"), "not json").unwrap();
    assert!(folder.load_error().contains("tokenizer.json"));
    folder.tokenizer();
    assert!(folder.load_error().contains("tokenizer_config.json"));
    fs::write(folder.0.join("tokenizer_config.json"), "{").unwrap();
    assert!(folder.load_error().contains("tokenizer_config.json"));
    folder.config(json!([]));
    assert_eq!(
        folder.load_error(),
        "tokenizer_config.json must be an object"
    );
    for cfg in [
        json!({}),
        json!({"chat_template": "  "}),
        json!({"chat_template": [{"name": "other", "template": "hello"}]}),
    ] {
        folder.config(cfg);
        assert!(
            folder
                .load_error()
                .starts_with("no chat template found in ")
        );
    }
    folder.config(json!({"chat_template": "{% broken"}));
    assert!(!folder.load_error().is_empty());
    folder.config(json!({"chat_template": "{{ raise_exception('invalid request') }}"}));
    let fmt = ChatFormatter::load(&folder.0).unwrap();
    assert_eq!(
        fmt.render(&json!([]), None, &json!({}))
            .unwrap_err()
            .to_string(),
        "invalid request"
    );
}

#[test]
fn template_fallback_file_precedence_tokens_and_unknown_ids() {
    let folder = Folder::new();
    folder.tokenizer();
    for cfg in [
        json!({"chat_template": "config"}),
        json!({"chat_template": [{"name": "other", "template": "wrong"}, {"name": "default", "template": "config"}]}),
    ] {
        folder.config(cfg);
        let fmt = ChatFormatter::load(&folder.0).unwrap();
        assert_eq!(fmt.render(&json!([]), None, &json!({})).unwrap(), "config");
    }
    fs::write(
        folder.0.join("chat_template.jinja"),
        "{{ custom }}{{ bos_token }}{{ eos_token }}{{ add_generation_prompt }}",
    )
    .unwrap();
    folder.config(json!({"chat_template": "wrong", "bos_token": "B", "eos_token": "E"}));
    let fmt = ChatFormatter::load(&folder.0).unwrap();
    assert_eq!(
        fmt.render(
            &json!([]),
            None,
            &json!({"custom": "X", "add_generation_prompt": false})
        )
        .unwrap(),
        "XBEFalse"
    );
    assert_eq!(fmt.digit_ids, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
    assert!(
        fmt.token_id("missing")
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    assert_eq!(fmt.decode(&[999]).unwrap(), "");
    let mut stream = StreamDecoder::new(fmt.tokenizer());
    assert_eq!(stream.push(&[999]).unwrap(), "");
    assert_eq!(stream.push(&[8]).unwrap(), "A");
    assert!(fmt.render(&json!({}), None, &json!({})).is_err());
    assert!(fmt.render(&json!([]), None, &json!(null)).is_err());
    assert!(fmt.prompt(&json!([]), None, &json!(null)).is_err());
    let mut output = OutputStream::new(&fmt, false);
    output.feed(&[8], false).unwrap();
    assert!(output.feed(&[], false).is_err());
    assert!(output.feed(&[8, 999], false).unwrap().is_empty());
    let mut tokenizer: Value =
        serde_json::from_str(&fs::read_to_string(folder.0.join("tokenizer.json")).unwrap())
            .unwrap();
    tokenizer["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("<think>");
    fs::write(folder.0.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    assert_eq!(folder.load_error(), "the tokenizer has no token <think>");
    tokenizer["model"]["vocab"]["<think>"] = json!(248047);
    tokenizer["model"]["vocab"]["<|im_start|>"] = json!(100);
    fs::write(folder.0.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    assert!(folder.load_error().contains("does not match the engine"));
    tokenizer["model"]["vocab"]["<|im_start|>"] = json!(248045);
    tokenizer["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("7");
    fs::write(folder.0.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    assert_eq!(folder.load_error(), "the tokenizer has no token 7");
}

#[test]
fn schema_directs_parameter_conversion_and_json_fallback() {
    assert_eq!(
        convert_parameter("3", Some(&json!({"type": "string"}))),
        json!("3")
    );
    assert_eq!(
        convert_parameter("null", Some(&json!({"type": ["null", "string"]}))),
        json!("null")
    );
    assert_eq!(convert_parameter("3", None), json!(3));
    assert_eq!(convert_parameter("\"x\"", None), json!("\"x\""));
    assert_eq!(convert_parameter("\"x\"", Some(&json!({}))), json!("x"));
    assert_eq!(
        convert_parameter("not json", Some(&json!({"type": "object"}))),
        json!("not json")
    );
    let schemas = tool_schemas(Some(
        &json!([{"name":"f", "parameters":{"properties":{"a":{"type":"string"}}}}]),
    ));
    assert_eq!(
        parse_tool_call(
            "<function=f><parameter=a>\n3\n</parameter><parameter=b>true</parameter>",
            &schemas
        ),
        Some(json!({"name":"f", "arguments":{"a":"3", "b":true}}))
    );
    assert_eq!(
        parse_tool_call("{\"name\":\"f\",\"parameters\":null}", &schemas),
        Some(json!({"name":"f", "arguments":{}}))
    );
    for bad in [
        "no function",
        "{",
        "{\"name\":\"\"}",
        "{\"name\":\"f\",\"arguments\":\"[1]\"}",
        "<function= >",
    ] {
        assert_eq!(parse_tool_call(bad, &schemas), None);
    }
}
