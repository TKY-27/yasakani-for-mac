use serde_json::{Value, json};
use std::sync::atomic::AtomicBool;
use yskn::{
    chat::{ChatParams, ChatRunner, Event, StopFilter},
    engine::{Engine, Sampling},
    loopguard::{self, LoopWatch},
    request::{RequestDefaults, clean_messages, clean_tools, parse_chat_request},
    stub::{self, Control, Script, StubEngine},
};
use yskn_text::{EOS_IDS, ToolCallFilter, convert_parameter, parse_tool_call, tool_schemas};
const EOS: u32 = EOS_IDS[0];
fn user() -> Value {
    json!([{"role":"user","content":"hi"}])
}
fn tools() -> Value {
    json!([
 {"type":"function","function":{"name":"get_weather","description":"Weather of a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}},
 {"type":"function","function":{"name":"write_file","parameters":{"properties":{"path":{"type":"string"},"content":{"type":"string"},"options":{"type":"object"},"overwrite":{"type":"boolean"}}}}}])
}
const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const CALL2: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
fn runner() -> (ChatRunner, Control) {
    let c = Control::default();
    (
        ChatRunner::new(
            Box::new(StubEngine::with_control(20000, c.clone())),
            stub::formatter().unwrap(),
            100,
        )
        .unwrap(),
        c,
    )
}
fn ids(s: &str) -> Vec<u32> {
    stub::formatter().unwrap().encode(s).unwrap()
}
fn script(c: &Control, s: &str) {
    let mut v = ids(s);
    v.push(EOS);
    c.push(Script::once(v));
}
fn run(r: &mut ChatRunner, p: ChatParams) -> (String, String, Vec<Value>, yskn::chat::Usage) {
    let (mut reason, mut answer, mut calls) = (String::new(), String::new(), vec![]);
    let mut done = 0;
    let u = r
        .run(&p, &mut |e| {
            match e {
                Event::Reasoning(t) => reason.push_str(&t),
                Event::Content(t) => answer.push_str(&t),
                Event::ToolCall(c) => calls.push(c),
                Event::Done(_) => done += 1,
            };
            Ok(())
        })
        .unwrap();
    assert_eq!(done, 1);
    (reason, answer, calls, u)
}
fn params() -> ChatParams {
    ChatParams::new(user())
}
// Upstream test_chat.StopFilterTests.test_cut_and_holdback
#[test]
fn cut_and_holdback() {
    let mut f = StopFilter::new(&["END".into()]);
    assert_eq!(
        f.push("hello E") + &f.push("N") + &f.push("D tail"),
        "hello "
    );
    assert!(f.hit);
    assert_eq!(f.push("more"), "");
}
// Upstream test_chat.StopFilterTests.test_partial_stop_is_released
#[test]
fn partial_stop_is_released() {
    let mut f = StopFilter::new(&["END".into()]);
    assert_eq!(
        f.push("abc EN") + &f.push("x yz") + &f.flush(),
        "abc ENx yz"
    );
    assert!(!f.hit);
}
// Upstream test_chat.StopFilterTests.test_no_stops_passthrough
#[test]
fn no_stops_passthrough() {
    assert_eq!(StopFilter::new(&[]).push("abc"), "abc");
}
// Upstream test_chat.StopFilterTests.test_two_stops_earliest_wins
#[test]
fn two_stops_earliest_wins() {
    assert_eq!(
        StopFilter::new(&["bb".into(), "a".into()]).push("xxbba"),
        "xx"
    );
}
#[test]
fn unicode_stop_is_held_at_character_boundaries() {
    let mut f = StopFilter::new(&["停止🛑".into()]);
    assert_eq!(f.push("a停"), "a");
    assert_eq!(f.push("止"), "");
    assert_eq!(f.push("🛑tail"), "");
    assert!(f.hit);
}
// Upstream test_chat.CleanMessageTests.test_ok_and_errors
#[test]
fn ok_and_errors() {
    let m=clean_messages(&json!([{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":null}])).unwrap();
    assert_eq!(m[0]["content"], json!([{"type":"text","text":"hi"}]));
    assert_eq!(m[1]["content"], "");
    for bad in [
        json!([]),
        Value::Null,
        json!([{"role":"function","content":"x"}]),
        json!([{"role":"user","content":5}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{}}]}]),
    ] {
        assert!(clean_messages(&bad).is_err());
    }
}
// Upstream test_chat.ChatTests.test_reasoning_then_answer
#[test]
fn reasoning_then_answer() {
    let (mut r, c) = runner();
    script(&c, "Let me think.</think>\n\nHello!");
    let (a, b, _, u) = run(&mut r, params());
    assert_eq!((a.as_str(), b.as_str()), ("Let me think.", "Hello!"));
    assert_eq!(
        u.completion_tokens,
        ids("Let me think.</think>\n\nHello!").len() + 1
    );
    assert_eq!(u.reasoning_tokens, ids("Let me think.").len());
    assert_eq!(u.finish, "stop");
    assert!(u.prompt_tokens > 20);
}
// Upstream test_chat.ChatTests.test_thinking_disabled
#[test]
fn thinking_disabled() {
    let (mut r, c) = runner();
    script(&c, "Hello there.");
    let mut p = params();
    p.template_kwargs = json!({"enable_thinking":false});
    let (a, b, _, u) = run(&mut r, p);
    assert_eq!((a.as_str(), b.as_str()), ("", "Hello there."));
    assert_eq!(u.reasoning_tokens, 0);
}
// Upstream test_chat.ChatTests.test_stop_string
#[test]
fn stop_string() {
    let (mut r, c) = runner();
    script(&c, "Hello STOP world");
    let mut p = params();
    p.template_kwargs = json!({"enable_thinking":false});
    p.stop = vec!["STOP".into()];
    let (_, b, _, u) = run(&mut r, p);
    assert_eq!(b, "Hello ");
    assert_eq!(u.finish, "stop");
    assert!(c.observation().stops > 0);
}
// Upstream test_chat.ChatTests.test_length_finish
#[test]
fn length_finish() {
    let (mut r, c) = runner();
    script(&c, &"word ".repeat(50));
    let mut p = params();
    p.template_kwargs = json!({"enable_thinking":false});
    p.max_tokens = 5;
    let (_, _, _, u) = run(&mut r, p);
    assert_eq!(u.finish, "length");
    assert_eq!(u.completion_tokens, 5);
}
// Upstream test_chat.ChatTests.test_loop_guard_bridges_to_an_answer
#[test]
fn loop_guard_bridges_to_an_answer() {
    let (mut r, c) = runner();
    let prefix = ids("I will think about this problem carefully. ");
    let mut v = prefix.clone();
    v.extend(ids("again"));
    c.push(Script {
        ids: v,
        loop_from: Some(prefix.len()),
    });
    script(&c, "Final answer.");
    let mut p = params();
    p.max_tokens = 100000;
    let (a, b, _, u) = run(&mut r, p);
    assert!(a.contains(loopguard::NOTICE.trim()));
    assert_eq!(b, "Final answer.");
    let info = u.loop_guard.unwrap();
    assert_eq!(info["where"], "reasoning");
    assert_eq!(info["kind"], "repeat");
    assert_eq!(u.finish, "stop");
    let first = info["tokens_before"].as_u64().unwrap() as usize;
    assert_eq!(
        u.completion_tokens,
        first + r.bridge.len() + ids("Final answer.").len() + 1
    );
    assert_eq!(
        c.observation().prompt.len(),
        u.prompt_tokens + first + r.bridge.len()
    );
    assert_eq!(info["answer_tokens"], ids("Final answer.").len() + 1);
}
// Upstream test_chat.ChatTests.test_loop_guard_off
#[test]
fn loop_guard_off() {
    let (mut r, c) = runner();
    c.push(Script {
        ids: ids("Thinking. ab"),
        loop_from: Some(10),
    });
    let mut p = params();
    p.max_tokens = 700;
    p.loop_guard = false;
    let (_, _, _, u) = run(&mut r, p);
    assert!(u.loop_guard.is_none());
    assert_eq!(u.finish, "length");
    assert_eq!(u.completion_tokens, 700);
}
// Upstream test_chat.ChatTests.test_loop_in_the_answer_stops
#[test]
fn loop_in_the_answer_stops() {
    let (mut r, c) = runner();
    let v = ids("ok</think>\n\nstart xy");
    let n = v.len();
    c.push(Script {
        ids: v,
        loop_from: Some(n - 2),
    });
    let mut p = params();
    p.max_tokens = 100000;
    let (_, _, _, u) = run(&mut r, p);
    assert_eq!(u.loop_guard.unwrap()["where"], "answer");
    assert_eq!(u.finish, "length");
}
// Upstream test_chat.ChatTests.test_prompt_too_long
#[test]
fn prompt_too_long() {
    let (mut r, _) = runner();
    let p = ChatParams::new(json!([{"role":"user","content":"word ".repeat(30000)}]));
    let e = r.run(&p, &mut |_| Ok(())).unwrap_err();
    assert_eq!(e.code, Some("context_length_exceeded"));
}
// Upstream test_chat.ChatTests.test_seed_counter_and_explicit_seed
#[test]
fn seed_counter_and_explicit_seed() {
    let (mut r, c) = runner();
    for (seed, want) in [(None, 100), (None, 101), (Some(7), 7), (None, 102)] {
        script(&c, "</think>x");
        let mut p = params();
        p.seed = seed;
        run(&mut r, p);
        assert_eq!(c.observation().sampling.unwrap().seed, want);
    }
}
// Upstream test_chat.ChatTests.test_emit_failure_stops_the_engine
#[test]
fn emit_failure_stops_the_engine() {
    let (mut r, c) = runner();
    c.push(Script {
        ids: ids("long answer"),
        loop_from: Some(0),
    });
    let mut p = params();
    p.max_tokens = 100000;
    p.template_kwargs = json!({"enable_thinking":false});
    let e = r
        .run(&p, &mut |_| Err(yskn::Error::engine("client went away")))
        .unwrap_err();
    assert_eq!(e.message, "client went away");
    assert_eq!(c.observation().stops, 1);
    script(&c, "</think>okay");
    assert_eq!(run(&mut r, params()).3.finish, "stop");
}
#[test]
fn thinking_budget_bridges_without_loop_detection() {
    let (mut r, c) = runner();
    c.push(Script {
        ids: ids("think"),
        loop_from: Some(0),
    });
    script(&c, "Answer");
    let mut p = params();
    p.thinking_budget = 12;
    p.loop_guard = false;
    p.max_tokens = 1000;
    let (a, b, _, u) = run(&mut r, p);
    assert!(a.contains(loopguard::BUDGET_NOTICE));
    assert_eq!(b, "Answer");
    assert_eq!(u.loop_guard.unwrap()["kind"], "budget");
}
#[test]
fn no_budget_for_bridge_and_ignore_eos() {
    let (mut r, c) = runner();
    c.push(Script {
        ids: ids("x"),
        loop_from: Some(0),
    });
    let mut p = params();
    p.thinking_budget = 12;
    p.max_tokens = 30;
    let (_, _, _, u) = run(&mut r, p.clone());
    assert_eq!(u.loop_guard.unwrap()["answer_tokens"], 0);
    assert_eq!(u.finish, "length");
    script(&c, "a");
    p.ignore_eos = true;
    let (_, _, _, u) = run(&mut r, p);
    assert!(u.loop_guard.is_none());
    assert_eq!(u.completion_tokens, 30);
    assert_eq!(u.finish, "length");
}
#[test]
fn digit_guard_and_watch_are_one_shot() {
    let fmt = stub::formatter().unwrap();
    let mut watch = LoopWatch::new(fmt.think_end, fmt.digit_ids, true, 0, 0, true);
    let mut seed = 1_u32;
    let digits = (0..1000)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            fmt.digit_ids[(seed >> 16) as usize % 10]
        })
        .collect::<Vec<_>>();
    let h = watch.check(&digits).unwrap();
    assert_eq!((h.kind, h.run), ("digits", 1000));
    assert!(watch.check(&digits).is_none());
    assert_eq!(loopguard::loop_period(&vec![1; 271], 0, 256), 0);
    assert_eq!(loopguard::loop_period(&vec![1; 272], 0, 256), 1);
}
#[test]
fn stub_reuse_reset_sampling_and_prompt_dependence() {
    let fmt = stub::formatter().unwrap();
    let mut e = StubEngine::new(4096);
    let s = Sampling {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 999,
        min_p: 0.1,
        seed: 8,
    };
    assert_eq!(s.normalized().top_k, 20);
    let p = fmt.encode("<|im_start|>user hello").unwrap();
    let first = e
        .generate(&p, 100, &s, Some(&EOS_IDS), &mut |_| Ok(true))
        .unwrap();
    let mut continued = p.clone();
    continued.extend_from_slice(&first.ids[..first.ids.len() - 1]);
    let second = e
        .generate(&continued, 3, &s, Some(&EOS_IDS), &mut |_| Ok(true))
        .unwrap();
    assert_eq!(second.cached_tokens, continued.len());
    assert_eq!(second.snap_mode, 1);
    e.reset().unwrap();
    let again = e
        .generate(&p, 100, &s, Some(&EOS_IDS), &mut |_| Ok(true))
        .unwrap();
    assert_eq!(again.ids, first.ids);
    assert_eq!(again.cached_tokens, 0);
    let other = e
        .generate(&ids("different"), 100, &s, Some(&EOS_IDS), &mut |_| {
            Ok(true)
        })
        .unwrap();
    assert_ne!(other.ids, first.ids);
}
// Upstream test_tools.ToolRequestTests.test_clean_tools
#[test]
fn clean_tools_cases() {
    assert_eq!(clean_tools(&Value::Null).unwrap(), None);
    assert_eq!(clean_tools(&json!([])).unwrap(), None);
    let t=clean_tools(&json!([{"type":"function","function":{"name":"f","strict":true}},{"function":{"name":"g","description":"d","parameters":{"type":"object"}}}])).unwrap().unwrap();
    assert_eq!(
        t[0],
        json!({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}})
    );
    assert_eq!(
        t[1]["function"],
        json!({"name":"g","description":"d","parameters":{"type":"object"}})
    );
    for bad in [
        json!("x"),
        json!([5]),
        json!([{"type":"retrieval"}]),
        json!([{"type":"function","function":{}}]),
        json!([{"type":"function","function":{"name":"a<b"}}]),
        json!([{"type":"function","function":{"name":"f","parameters":[]}}]),
    ] {
        assert!(clean_tools(&bad).is_err());
    }
}
// Upstream test_tools.ToolRequestTests.test_clean_messages_with_tool_calls
#[test]
fn clean_messages_with_tool_calls() {
    let m=clean_messages(&json!([{"role":"developer","content":"Be brief."},{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Osaka\"}"}},{"function":{"name":"get_weather","arguments":""}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny"}])).unwrap();
    assert_eq!(m[0]["role"], "system");
    assert_eq!(
        m[2]["tool_calls"],
        json!([{"type":"function","function":{"name":"get_weather","arguments":{"city":"Osaka"}}},{"type":"function","function":{"name":"get_weather","arguments":{}}}])
    );
    assert!(m[3].get("tool_call_id").is_none());
    for calls in [
        json!("x"),
        json!([{"function":{"name":""}}]),
        json!([{"function":{"name":"f","arguments":"{not json"}}]),
        json!([{"function":{"name":"f","arguments":"[1]"}}]),
    ] {
        assert!(
            clean_messages(&json!([{"role":"assistant","content":"","tool_calls":calls}])).is_err()
        );
    }
}
// Upstream test_tools.ToolRequestTests.test_parse_chat_request
#[test]
fn parse_chat_request_cases() {
    let cfg = RequestDefaults {
        thinking_budget: 0,
        ..RequestDefaults::default()
    };
    let mut b = json!({"messages":user(),"tools":tools(),"parallel_tool_calls":true});
    assert_eq!(
        parse_chat_request(&b, &cfg)
            .unwrap()
            .tools
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        2
    );
    b["tool_choice"] = json!("none");
    assert!(parse_chat_request(&b, &cfg).unwrap().tools.is_none());
    b["tool_choice"] = json!("auto");
    assert!(parse_chat_request(&b, &cfg).unwrap().tools.is_some());
    for extra in [
        json!({"tool_choice":"required"}),
        json!({"tool_choice":{"type":"function","function":{"name":"get_weather"}}}),
        json!({"functions":[{"name":"f"}]}),
        json!({"function_call":"auto"}),
        json!({"thinking_budget":-1}),
    ] {
        let mut v = json!({"messages":user(),"tools":tools()});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(parse_chat_request(&v, &cfg).is_err());
    }
    let p = parse_chat_request(
        &json!({"messages":user(),"thinking":{"type":"disabled"},"thinking_budget":500}),
        &cfg,
    )
    .unwrap();
    assert_eq!(p.template_kwargs["enable_thinking"], false);
    assert_eq!(p.thinking_budget, 500);
    assert_eq!(
        parse_chat_request(
            &json!({"messages":user()}),
            &RequestDefaults {
                thinking_budget: 4096,
                ..cfg
            }
        )
        .unwrap()
        .thinking_budget,
        4096
    );
}
#[test]
fn aliases_precedence_numbers_and_caps() {
    let cfg = RequestDefaults {
        max_tokens_cap: Some(8),
        thinking_budget: 0,
        ..RequestDefaults::default()
    };
    let p=parse_chat_request(&json!({"messages":user(),"max_tokens":99,"max_completion_tokens":"12","temperature":"0.5","top_k":3.9,"seed":true,"thinking":{"type":"enabled"},"enable_thinking":true,"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low","thinking_budget":17}}),&cfg).unwrap();
    assert_eq!(p.max_tokens, 8);
    assert_eq!(p.top_k, 3);
    assert_eq!(p.seed, Some(1));
    assert_eq!(p.temperature, 0.5);
    assert_eq!(
        p.template_kwargs,
        json!({"enable_thinking":false,"reasoning_effort":"low"})
    );
    assert_eq!(p.thinking_budget, 17);
    for e in ["minimal", "low", "medium", "high", "xhigh"] {
        assert!(parse_chat_request(&json!({"messages":user(),"reasoning_effort":e}), &cfg).is_ok());
    }
}
fn filtered(text: &str, step: usize) -> (String, Vec<Value>, usize) {
    let t = tools();
    let mut f = ToolCallFilter::new(Some(&t));
    let mut ev = vec![];
    for c in text.as_bytes().chunks(step) {
        ev.extend(f.push(std::str::from_utf8(c).unwrap()));
    }
    ev.extend(f.flush());
    let content = ev
        .iter()
        .filter(|(k, _)| k == "content")
        .filter_map(|(_, v)| v.as_str())
        .collect();
    let calls = ev
        .into_iter()
        .filter(|(k, _)| k == "tool_call")
        .map(|(_, v)| v)
        .collect();
    (content, calls, f.calls)
}
// Upstream test_tools.ToolCallFilterTests.test_plain_text_passes_unchanged
#[test]
fn plain_text_passes_unchanged() {
    for step in [1, 3, 50] {
        let s = "Hello,  world.\n\nA < B and <tool, done. ";
        assert_eq!(filtered(s, step), (s.into(), vec![], 0));
    }
}
// Upstream test_tools.ToolCallFilterTests.test_one_call_after_some_text
#[test]
fn one_call_after_some_text() {
    for step in [1, 7, 500] {
        assert_eq!(
            filtered(&format!("I will check.\n\n{CALL}"), step),
            (
                "I will check.".into(),
                vec![json!({"name":"get_weather","arguments":{"city":"Osaka","days":3}})],
                1
            )
        );
    }
}
// Upstream test_tools.ToolCallFilterTests.test_two_calls
#[test]
fn two_calls() {
    let (a, c, n) = filtered(&format!("{CALL}\n{CALL2}\n"), 7);
    assert_eq!(a, "");
    assert_eq!(n, 2);
    assert_eq!(c[0]["arguments"]["city"], "Osaka");
    assert_eq!(c[1]["arguments"]["city"], "Tokyo");
}
// Upstream test_tools.ToolCallFilterTests.test_an_unfinished_call_comes_back_as_text
#[test]
fn an_unfinished_call_comes_back_as_text() {
    let s = "Text\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOsa";
    assert_eq!(filtered(s, 7), (s.into(), vec![], 0));
}
// Upstream test_tools.ToolCallFilterTests.test_an_unreadable_block_comes_back_as_text
#[test]
fn an_unreadable_block_comes_back_as_text() {
    let s = "<tool_call>\nthis is not a call\n</tool_call>";
    assert_eq!(filtered(s, 7), (s.into(), vec![], 0));
}
// Upstream test_tools.ToolCallFilterTests.test_the_json_form_is_read_too
#[test]
fn the_json_form_is_read_too() {
    assert_eq!(filtered("<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Osaka\"}}\n</tool_call>",7).1,vec![json!({"name":"get_weather","arguments":{"city":"Osaka"}})]);
}
// Upstream test_tools.ToolCallFilterTests.test_parameter_values
#[test]
fn parameter_values() {
    let schemas = tool_schemas(Some(&tools()));
    let text = "line 1\n  \"quoted\" <b>tag</b>\n\n123\n";
    let block = format!(
        "\n<function=write_file>\n<parameter=path>\nC:\\tmp\\a.txt\n</parameter>\n<parameter=content>\n{text}\n</parameter>\n<parameter=options>\n{{\"mode\":\"w\",\"n\":[1,2]}}\n</parameter>\n<parameter=overwrite>\ntrue\n</parameter>\n</function>\n"
    );
    assert_eq!(
        parse_tool_call(&block, &schemas).unwrap(),
        json!({"name":"write_file","arguments":{"path":"C:\\tmp\\a.txt","content":text,"options":{"mode":"w","n":[1,2]},"overwrite":true}})
    );
    assert_eq!(
        convert_parameter("3", Some(&schemas["write_file"]["content"])),
        "3"
    );
    assert_eq!(convert_parameter("3", Some(&json!({"type":"integer"}))), 3);
    assert_eq!(convert_parameter("3", None), 3);
    assert_eq!(convert_parameter("\"x\"", None), "\"x\"");
    assert_eq!(
        convert_parameter("not json", Some(&json!({"type":"object"}))),
        "not json"
    );
    assert!(parse_tool_call("no function here", &serde_json::Map::new()).is_none());
}
// Upstream test_tools.ToolRunTests.test_the_template_shows_the_tools_and_reads_a_call_back
#[test]
fn the_template_shows_the_tools_and_reads_a_call_back() {
    let fmt = stub::formatter().unwrap();
    let msgs=clean_messages(&json!([{"role":"user","content":"weather"},{"role":"assistant","content":"I will check.","tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Osaka\",\"days\":3}"}}]},{"role":"tool","content":"sunny, 22 C"}])).unwrap();
    let rendered = fmt.render(&msgs, Some(&tools()), &json!({})).unwrap();
    assert!(rendered.contains("\"name\": \"get_weather\""));
    assert!(rendered.contains(CALL), "{rendered}");
    assert!(rendered.contains("<tool_response>\nsunny, 22 C\n</tool_response>"));
    assert_eq!(
        filtered(CALL, 7).1[0],
        json!({"name":"get_weather","arguments":{"city":"Osaka","days":3}})
    );
}
// Upstream test_tools.ToolRunTests.test_runner_returns_the_call
#[test]
fn runner_returns_the_call() {
    let (mut r, c) = runner();
    script(
        &c,
        &format!("The user wants the weather.</think>\n\nI will check.\n\n{CALL}"),
    );
    let mut p = params();
    p.tools = clean_tools(&tools()).unwrap();
    let (_, a, calls, u) = run(&mut r, p);
    assert_eq!(a, "I will check.");
    assert_eq!(
        calls,
        vec![json!({"name":"get_weather","arguments":{"city":"Osaka","days":3}})]
    );
    assert_eq!((u.finish, u.tool_calls), ("tool_calls", 1));
}
#[test]
fn terminal_thinking_commands_and_history() {
    let (mut r, c) = runner();
    script(&c, "Reason</think>\nAnswer");
    script(&c, "Second");
    let mut p = ChatParams::new(json!([{"role":"system","content":"base"}]));
    p.max_tokens = 1000;
    let mut output = vec![];
    yskn::cli::chat_loop(
        &mut r,
        p,
        false,
        "hi\n/think off\n/effort low\n/reset\nhi again\n/bogus\n/exit\n".as_bytes(),
        &mut output,
        &AtomicBool::new(false),
        &AtomicBool::new(false),
    )
    .unwrap();
    let out = String::from_utf8(output).unwrap();
    for needle in [
        "[thinking]",
        "[/thinking]",
        "Answer",
        "(thinking off)",
        "(reasoning effort low)",
        "(conversation cleared)",
        "commands:",
    ] {
        assert!(out.contains(needle), "{out}");
    }
    let prompt = r.formatter.decode(&c.observation().prompt).unwrap();
    assert!(prompt.contains("base"));
    assert!(!prompt.contains("Reason"));
    assert!(!prompt.contains("Answer"));
}
#[test]
fn integer_sampling_keeps_i64_precision_and_text_parts_match_python_conversion() {
    let cfg = RequestDefaults::default();
    for seed in [i64::MIN, 9007199254740993, i64::MAX] {
        let p = parse_chat_request(&json!({"messages":user(),"seed":seed}), &cfg).unwrap();
        assert_eq!(p.seed, Some(seed));
    }
    let m = clean_messages(&json!([{"role":"user","content":[{"text":true},{"text":null},{"text":{"x":["a",false]}}]}])).unwrap();
    assert_eq!(
        m[0]["content"],
        json!([{"type":"text","text":"True"},{"type":"text","text":"None"},{"type":"text","text":"{'x': ['a', False]}"}])
    );
}
// Review finding: every prefix of every stop string was compared on every
// push, so one huge stop string, or a long partial match, stalled the single
// engine thread. The work per push is now bounded by the held-back text.
fn within_patience(work: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        work();
        let _ = done.send(());
    });
    finished
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the stop filter did not finish within 30 s");
}
#[test]
fn stop_filter_work_is_bounded_by_the_held_back_text() {
    within_patience(|| {
        let mut f = StopFilter::new(&["x".repeat(16 << 20)]);
        let reply = "a short reply. ".repeat(200);
        let out: String = reply.chars().map(|c| f.push(&c.to_string())).collect();
        assert_eq!(out + &f.flush(), reply);
    });
    within_patience(|| {
        let n = 1 << 20;
        let mut f = StopFilter::new(&["a".repeat(n) + "b"]);
        let mut out = String::new();
        for _ in 0..20000 {
            out += &f.push("a");
        }
        assert_eq!(out, "", "a possible stop start is held back");
        out += &f.push("c");
        assert_eq!(out, "a".repeat(20000) + "c");
        // First match wins, after a long partial one.
        let mut f = StopFilter::new(&["a".repeat(n) + "b", "ac".into()]);
        assert_eq!(f.push(&"a".repeat(5000)) + &f.push("c"), "a".repeat(4999));
        assert!(f.hit);
    });
}
