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
// The terminal's interrupt during generation, driven through the injected
// reader, writer and cancel flag: the answer stops, is not kept in the
// history, and the loop returns to the prompt (here: the end of input).
#[test]
fn terminal_interrupt_stops_the_answer_and_keeps_the_prompt() {
    use std::sync::{Arc, atomic::Ordering};
    let c = Control::default();
    c.delay(std::time::Duration::from_millis(1));
    c.push(Script {
        ids: ids("on and on "),
        loop_from: Some(0),
    });
    let writing = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(AtomicBool::new(false));
    let (w, k, engine) = (writing.clone(), cancel.clone(), c.clone());
    // The runner is not Send: it is built on the thread that uses it.
    let chat = std::thread::spawn(move || {
        let engine = Box::new(StubEngine::with_control(20000, engine));
        let mut r = ChatRunner::new(engine, stub::formatter().unwrap(), 100).unwrap();
        let mut p = ChatParams::new(json!([]));
        p.max_tokens = 1_000_000;
        p.loop_guard = false;
        let mut output = vec![];
        let result = yskn::cli::chat_loop(
            &mut r,
            p,
            false,
            false,
            "hi\n".as_bytes(),
            &mut output,
            &w,
            &k,
        );
        (result, String::from_utf8(output).unwrap())
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    // Polls another thread's progress; the endless answer cannot end by itself
    // before the context room, which the cancel must beat.
    while !(writing.load(Ordering::Acquire) && c.observation().generated > 0) {
        assert!(
            std::time::Instant::now() < deadline,
            "generation never started"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    cancel.store(true, Ordering::Release);
    let (result, out) = chat.join().unwrap();
    result.unwrap();
    assert!(out.contains("[stopped]"), "{out}");
    assert!(!out.contains("tokens/s"), "{out}");
    assert_eq!(c.observation().stops, 1);
}
// Review finding: model output reached the terminal unfiltered, so escape
// sequences in it were interpreted. C0 controls other than newline and tab,
// DEL and C1 controls are shown in a visible escape form instead.
#[test]
fn terminal_shows_control_characters_visibly() {
    let hostile = "\u{1b}[2J\u{1b}]0;title\u{7}\r\u{7f}\u{9b}31m\tok\n";
    let chat = |raw: bool| {
        let (mut r, c) = runner();
        script(&c, &format!("{hostile}</think>{hostile}"));
        let mut p = ChatParams::new(json!([]));
        p.max_tokens = 1000;
        let mut output = vec![];
        yskn::cli::chat_loop(
            &mut r,
            p,
            false,
            raw,
            "hi\n".as_bytes(),
            &mut output,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
        )
        .unwrap();
        String::from_utf8(output).unwrap()
    };
    let out = chat(false);
    let shown = "\\x1b[2J\\x1b]0;title\\x07\\x0d\\x7f\\x9b31m\tok\n";
    assert_eq!(out.matches(shown).count(), 2, "{out}");
    assert!(
        !out.chars()
            .any(|ch| ch.is_control() && ch != '\n' && ch != '\t'),
        "{out:?}"
    );
    // --raw turns the filter off.
    assert_eq!(chat(true).matches(hostile).count(), 2);
    assert_eq!(yskn::cli::visible("plain\ttext\n"), "plain\ttext\n");
    // Error messages too.
    let (mut r, c) = runner();
    c.fail_next("engine says \u{1b}[2J");
    let mut output = vec![];
    yskn::cli::chat_loop(
        &mut r,
        ChatParams::new(json!([])),
        false,
        false,
        "hi\n".as_bytes(),
        &mut output,
        &AtomicBool::new(false),
        &AtomicBool::new(false),
    )
    .unwrap();
    let out = String::from_utf8(output).unwrap();
    assert!(
        out.contains("engine error: engine says \\x1b[2J"),
        "{out:?}"
    );
}
// Review finding: validation errors quoted request values in full, so one
// megabyte value became a multi-megabyte error response, built after the
// request's budget charge was released. Messages quote a short excerpt.
#[test]
fn error_messages_quote_a_short_excerpt_of_request_values() {
    let cfg = RequestDefaults::default();
    let message = |extra: Value| {
        let mut b = json!({"messages":user()});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        parse_chat_request(&b, &cfg).unwrap_err().message
    };
    // Short values keep upstream's wording exactly.
    assert_eq!(
        message(json!({"response_format":{"type":"json_object"}})),
        "response_format 'json_object' is not supported"
    );
    assert_eq!(
        message(json!({"tool_choice":"required"})),
        "tool_choice \"required\" is not supported; use \"auto\" or \"none\""
    );
    let long = "\u{7f}".repeat(7 << 20);
    for (extra, prefix, room) in [
        (
            json!({"response_format":{"type":long}}),
            "response_format '\\x7f\\x7f",
            64 + 1,
        ),
        (
            json!({"response_format":{"type":{"k":[long, long]}}}),
            "response_format {'k': ['\\x7f",
            64 + 1,
        ),
        // As upstream: the first 80 characters of the JSON text.
        (json!({"tool_choice":long}), "tool_choice \"\\u007f", 80),
    ] {
        let m = message(extra);
        assert!(m.starts_with(prefix), "{m}");
        let fixed = if prefix.starts_with("tool") {
            "tool_choice  is not supported; use \"auto\" or \"none\""
        } else {
            "response_format  is not supported"
        };
        assert!(m.chars().count() <= fixed.len() + room, "{}", m.len());
        assert!(
            !m.contains('\u{7f}'),
            "control characters are shown visibly"
        );
    }
    assert!(message(json!({"response_format":{"type":long}})).contains("…"));
}
// Review finding (third round): an excerpt was made by shortening every list to
// 80 items and serializing the whole shortened copy, so a nested value within
// every limit (the review's 8.3 MB `tool_choice` of 25,600 strings of 80 emoji)
// made the server hold about 150 MiB to write an 80-character message. The
// excerpt is now written into a sink that stops after it, so quoting the
// value costs a tiny fraction of merely building it. (Work is compared by
// time: counting allocations would need an `unsafe` allocator.)
#[test]
fn quoting_a_nested_value_costs_only_the_excerpt() {
    let cfg = RequestDefaults::default();
    let started = std::time::Instant::now();
    let nested = json!(vec![vec![vec!["\u{1f600}".repeat(80); 80]; 80]; 4]);
    let built = started.elapsed();
    for key in ["tool_choice", "response_format"] {
        let mut b = json!({"messages":user()});
        b[key] = if key == "tool_choice" {
            nested.clone()
        } else {
            json!({"type":nested.clone()})
        };
        // The best of a few tries, so a pause of this thread cannot fail it.
        let quoted = (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                let e = parse_chat_request(&b, &cfg).unwrap_err();
                assert!(
                    e.message.starts_with(&format!("{key} [[[")),
                    "{}",
                    e.message
                );
                assert!(e.message.chars().count() < 150, "{}", e.message);
                started.elapsed()
            })
            .min()
            .unwrap();
        assert!(
            quoted * 10 < built,
            "{key}: {quoted:?} to quote, {built:?} to build"
        );
    }
}
// Review finding: an early context check assumed normalization shrinks text at
// most four times, so a tokenizer whose normalizer deletes characters had a
// prompt refused that fits. Only the exact check after tokenization remains.
#[test]
fn a_prompt_that_normalizes_to_fit_is_not_refused() {
    let mut tok: Value = serde_json::from_str(include_str!("../data/stub-tokenizer.json")).unwrap();
    tok["normalizer"] =
        json!({"type":"Sequence","normalizers":[{"type":"NFD"},{"type":"StripAccents"}]});
    let fmt = yskn_text::ChatFormatter::from_json(
        &tok.to_string(),
        "{}",
        include_str!("../data/stub-template.jinja"),
    )
    .unwrap();
    let engine = StubEngine::new(128).using_formatter(fmt.clone());
    let r = ChatRunner::new(Box::new(engine), fmt, 1).unwrap();
    let text = "\u{301}".repeat(10_000) + "hi";
    let (ids, _) = r
        .prepare_checked(&ChatParams::new(json!([{"role":"user","content":text}])))
        .unwrap();
    assert_eq!(ids.len(), 23);
    // The exact check still refuses what does not fit.
    let e = r
        .prepare_checked(&ChatParams::new(
            json!([{"role":"user","content":"a".repeat(64)}]),
        ))
        .unwrap_err();
    assert_eq!(e.code, Some("context_length_exceeded"));
}
// Review finding: once preparation had started on the engine thread it ran to
// the end even after the deadline, a disconnect or shutdown. The server's check
// runs before rendering, after rendering and after tokenization, and its
// error ends the preparation there.
#[test]
fn preparation_stops_between_its_steps() {
    use std::cell::Cell;
    let (r, c) = runner();
    let prepare = |p: &ChatParams, stop_at: usize| {
        let calls = Cell::new(0);
        let result = r.prepare_interruptible(p, &|| {
            calls.set(calls.get() + 1);
            if calls.get() == stop_at {
                Err(yskn::Error::engine("stopped"))
            } else {
                Ok(())
            }
        });
        (result.map(|_| ()).map_err(|e| e.message), calls.get())
    };
    assert_eq!(prepare(&ChatParams::new(user()), 0), (Ok(()), 3));
    // A template that fails shows where rendering happens: after call 1.
    let bad = ChatParams::new(json!([{"role":"assistant","content":"",
        "tool_calls":[{"function":{"name":"f","arguments":"x"}}]}]));
    assert_eq!(prepare(&bad, 1), (Err("stopped".into()), 1));
    let (rendered, calls) = prepare(&bad, 2);
    assert!(rendered.unwrap_err().contains("template"), "rendering ran");
    assert_eq!(calls, 1);
    // A prompt longer than the context shows where tokenization happens:
    // after call 2, and the token check after call 3.
    let long = ChatParams::new(json!([{"role":"user","content":"a".repeat(30000)}]));
    assert_eq!(prepare(&long, 2), (Err("stopped".into()), 2));
    assert_eq!(prepare(&long, 3), (Err("stopped".into()), 3));
    let (counted, calls) = prepare(&long, 0);
    assert!(counted.unwrap_err().contains("tokens exceeds"));
    assert_eq!(calls, 3);
    assert_eq!(c.observation().calls, 0);
}
// Review finding: `tools` in chat_template_kwargs reached the template instead
// of the validated tools, bypassing the tool limit and tool_choice "none".
// Names the renderer sets itself are refused, naming the key.
#[test]
fn template_kwargs_cannot_set_what_the_server_sets() {
    let cfg = RequestDefaults::default();
    for key in [
        "tools",
        "messages",
        "add_generation_prompt",
        "bos_token",
        "eos_token",
        "raise_exception",
    ] {
        let b = json!({"messages":user(),"tool_choice":"none","chat_template_kwargs":{"enable_thinking":false,(key):tools()}});
        let e = parse_chat_request(&b, &cfg).unwrap_err();
        assert_eq!(e.status, 400);
        assert_eq!(
            e.message,
            format!("chat_template_kwargs: {key} is set by the server and cannot be given")
        );
    }
    // The renderer refuses them too, for callers that build ChatParams.
    let (r, c) = runner();
    let mut p = ChatParams::new(user());
    p.template_kwargs = json!({"tools":tools()});
    let e = r.prepare_checked(&p).unwrap_err();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("tools"), "{}", e.message);
    // Other kwargs still reach the template.
    p.template_kwargs = json!({"enable_thinking":false,"custom":1});
    let (ids, thinking) = r.prepare_checked(&p).unwrap();
    assert!(!thinking && !ids.is_empty());
    assert_eq!(c.observation().calls, 0);
}
// Review finding: bidirectional overrides and isolates could reorder what the
// terminal shows. They are shown visibly too; the zero-width joiner and
// variation selectors, which emoji need, are left alone.
#[test]
fn terminal_shows_bidirectional_controls_visibly() {
    let text = "a\u{202a}\u{202b}\u{202c}\u{202d}\u{202e}\u{2066}\u{2067}\u{2068}\u{2069}b";
    assert_eq!(
        yskn::cli::visible(text),
        "a\\u202a\\u202b\\u202c\\u202d\\u202e\\u2066\\u2067\\u2068\\u2069b"
    );
    let emoji = "\u{1f469}\u{200d}\u{1f4bb} \u{2764}\u{fe0f}";
    assert_eq!(yskn::cli::visible(emoji), emoji);
}
