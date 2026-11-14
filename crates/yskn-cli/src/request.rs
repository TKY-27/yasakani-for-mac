//! Validation at the OpenAI request boundary, matching chat.py/server.py.
use crate::{Error, Result, chat::ChatParams, server::Limits};
use serde_json::{Map, Value, json};
use std::fmt;

/// Upper bound on the number of values a JSON text can hold, counted without
/// parsing it: every value except the root follows `[`, `,` or `:` outside a
/// string. Lets an expanding document be refused before it is built.
pub fn json_values(bytes: &[u8]) -> usize {
    let (mut n, mut in_string, mut escaped) = (1, false, false);
    for &b in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if matches!(b, b'[' | b',' | b':') {
            n += 1;
        }
    }
    n
}
/// Per-request shape limits, checked on the parsed body before it is cleaned
/// or copied. `values` is the body's [`json_values`] count; the values that
/// tool-call argument strings will expand to are added to it and returned.
pub fn check_size(body: &Value, limits: &Limits, mut values: usize) -> Result<usize> {
    let too_many =
        |what: String, n: usize| Error::request(format!("{what}: at most {n} are accepted"));
    if let Some(messages) = body["messages"].as_array() {
        if messages.len() > limits.max_messages {
            return Err(too_many("messages".into(), limits.max_messages));
        }
        for (i, m) in messages.iter().enumerate() {
            if m["content"].as_array().map_or(0, Vec::len) > limits.max_content_parts {
                return Err(too_many(
                    format!("messages[{i}]: content parts"),
                    limits.max_content_parts,
                ));
            }
            let calls = m["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
            if calls.len() > limits.max_tool_calls {
                return Err(too_many(
                    format!("messages[{i}]: tool calls"),
                    limits.max_tool_calls,
                ));
            }
            for c in calls {
                if let Some(s) = c["function"]["arguments"].as_str() {
                    values = values.saturating_add(json_values(s.as_bytes()));
                }
            }
        }
    }
    if body["tools"].as_array().map_or(0, Vec::len) > limits.max_tools {
        return Err(too_many("tools".into(), limits.max_tools));
    }
    let stops = match &body["stop"] {
        Value::Array(a) => a.as_slice(),
        s => std::slice::from_ref(s),
    };
    if stops
        .iter()
        .any(|s| s.as_str().is_some_and(|s| s.len() > limits.max_stop_bytes))
    {
        return Err(Error::request(format!(
            "stop strings must be at most {} bytes long",
            limits.max_stop_bytes
        )));
    }
    check_values(values, limits)?;
    Ok(values)
}
pub fn check_values(values: usize, limits: &Limits) -> Result<()> {
    if values > limits.max_json_values {
        return Err(Error::request(format!(
            "the request holds more than {} JSON values",
            limits.max_json_values
        )));
    }
    Ok(())
}
/// Bytes of text (strings and object keys) inside a value.
pub fn text_bytes(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Array(a) => a.iter().map(text_bytes).sum(),
        Value::Object(m) => m.iter().map(|(k, v)| k.len() + text_bytes(v)).sum(),
        _ => 0,
    }
}

pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
/// Characters of a request value an error message may quote.
pub const EXCERPT: usize = 64;
/// `text` cut to [`EXCERPT`] characters, with an ellipsis if it was longer.
pub fn excerpt(text: &str) -> String {
    let mut out: String = text.chars().take(EXCERPT).collect();
    if out.len() < text.len() {
        out.push('…');
    }
    out
}
/// `v` with every string and key cut to `n` characters and every list and
/// object to `n` entries, so quoting a huge value costs no more than its
/// excerpt. Its repr or JSON text is longer than `n` exactly when `v`'s is.
fn shortened(v: &Value, n: usize) -> Value {
    let cut = |s: &str| s.chars().take(n).collect::<String>();
    match v {
        Value::String(s) => Value::String(cut(s)),
        Value::Array(a) => a.iter().take(n).map(|v| shortened(v, n)).collect(),
        Value::Object(m) => m
            .iter()
            .take(n)
            .map(|(k, v)| (cut(k), shortened(v, n)))
            .collect(),
        v => v.clone(),
    }
}
fn py_string(v: &Value) -> String {
    if let Value::String(s) = v {
        s.clone()
    } else {
        py_repr(v)
    }
}
/// Counts the bytes written, to size a buffer before writing into it.
struct Len(usize);
impl fmt::Write for Len {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}
fn py_repr(v: &Value) -> String {
    // Sized exactly by a first pass: a buffer grown by doubling briefly holds
    // its old and new copy, up to twice the result. Neither writer can fail.
    let mut len = Len(0);
    let _ = write_repr(v, &mut len);
    let mut out = String::with_capacity(len.0);
    let _ = write_repr(v, &mut out);
    out
}
/// Python's repr() of `v`, written into one buffer: nested values are not
/// copied once per level, so the result is the only large allocation.
fn write_repr(v: &Value, out: &mut impl fmt::Write) -> fmt::Result {
    match v {
        Value::String(s) => write_str_repr(s, out),
        Value::Null => out.write_str("None"),
        Value::Bool(true) => out.write_str("True"),
        Value::Bool(false) => out.write_str("False"),
        Value::Array(a) => {
            out.write_char('[')?;
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.write_str(", ")?;
                }
                write_repr(v, out)?;
            }
            out.write_char(']')
        }
        Value::Object(m) => {
            out.write_char('{')?;
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.write_str(", ")?;
                }
                write_str_repr(k, out)?;
                out.write_str(": ")?;
                write_repr(v, out)?;
            }
            out.write_char('}')
        }
        _ => out.write_str(&yskn_text::json_dumps(v, false)),
    }
}
fn write_str_repr(s: &str, out: &mut impl fmt::Write) -> fmt::Result {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.write_char(quote)?;
    for c in s.chars() {
        match c {
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            c if c == quote => {
                out.write_char('\\')?;
                out.write_char(c)?;
            }
            c if c.is_control() && (c as u32) <= 255 => write!(out, "\\x{:02x}", c as u32)?,
            c => out.write_char(c)?,
        }
    }
    out.write_char(quote)
}
fn content(v: &Value, i: usize) -> Result<Value> {
    if let Some(parts) = v.as_array() {
        let mut out = Vec::new();
        for p in parts {
            if !p.is_object()
                || (!p["type"].is_null() && p["type"] != "text")
                || p.get("text").is_none()
            {
                return Err(Error::request(format!(
                    "messages[{i}]: only text content is supported"
                )));
            }
            // Assigned, not interpolated: json! would copy the value.
            let mut part = json!({"type":"text"});
            part["text"] = Value::String(py_string(&p["text"]));
            out.push(part);
        }
        Ok(Value::Array(out))
    } else if v.is_null() {
        Ok(json!(""))
    } else if v.is_string() {
        Ok(v.clone())
    } else {
        Err(Error::request(format!(
            "messages[{i}]: content must be a string or a list of text parts"
        )))
    }
}
fn clean_calls(v: &Value, i: usize) -> Result<Value> {
    let calls = v
        .as_array()
        .ok_or_else(|| Error::request(format!("messages[{i}]: tool_calls must be a list")))?;
    let mut out = Vec::new();
    for (k, c) in calls.iter().enumerate() {
        let f = &c["function"];
        let name = f["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::request(format!(
                    "messages[{i}].tool_calls[{k}]: a function call with a name is expected"
                ))
            })?;
        // A string is parsed in place, not copied first.
        let mut args = match f["arguments"].as_str() {
            Some(s) if s.trim().is_empty() => json!({}),
            Some(s) => serde_json::from_str(s).map_err(|_| {
                Error::request(format!(
                    "messages[{i}].tool_calls[{k}]: arguments is not valid JSON"
                ))
            })?,
            None => f["arguments"].clone(),
        };
        if args.is_null() {
            args = json!({});
        }
        if !args.is_object() {
            return Err(Error::request(format!(
                "messages[{i}].tool_calls[{k}]: arguments must be a JSON object"
            )));
        }
        let mut function = json!({"name":name});
        function["arguments"] = args;
        let mut call = json!({"type":"function"});
        call["function"] = function;
        out.push(call);
    }
    Ok(Value::Array(out))
}
pub fn clean_messages(v: &Value) -> Result<Value> {
    let messages = v
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| Error::request("messages must be a non-empty list"))?;
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let role = match m["role"].as_str() {
            Some("developer") => "system",
            Some(s @ ("system" | "user" | "assistant" | "tool")) => s,
            _ => {
                return Err(Error::request(format!(
                    "messages[{i}]: role must be system, user, assistant or tool"
                )));
            }
        };
        let mut msg = json!({"role":role});
        msg["content"] = content(&m["content"], i)?;
        if role == "assistant" {
            if m["reasoning_content"].is_string() {
                msg["reasoning_content"] = m["reasoning_content"].clone();
            }
            if truthy(&m["tool_calls"]) {
                msg["tool_calls"] = clean_calls(&m["tool_calls"], i)?;
            }
        }
        out.push(msg);
    }
    Ok(Value::Array(out))
}
pub fn clean_tools(v: &Value) -> Result<Option<Value>> {
    if v.is_null() || v.as_array().is_some_and(Vec::is_empty) {
        return Ok(None);
    }
    let tools = v
        .as_array()
        .ok_or_else(|| Error::request("tools must be a list"))?;
    let mut out = Vec::new();
    for (i, t) in tools.iter().enumerate() {
        let f = &t["function"];
        if !t.is_object() || t.get("type").is_some_and(|x| x != "function") || !f.is_object() {
            return Err(Error::request(format!(
                "tools[{i}]: only tools of type \"function\" are supported"
            )));
        }
        let name = f["name"]
            .as_str()
            .filter(|s| !s.is_empty() && !s.contains(['<', '>', '\n']))
            .ok_or_else(|| {
                Error::request(format!(
                    "tools[{i}]: the function needs a name (without <, > or line breaks)"
                ))
            })?;
        if !f["parameters"].is_null() && !f["parameters"].is_object() {
            return Err(Error::request(format!(
                "tools[{i}]: parameters must be a JSON schema object"
            )));
        }
        let mut function = json!({"name":name});
        if f["description"].is_string() {
            function["description"] = f["description"].clone();
        }
        function["parameters"] = if truthy(&f["parameters"]) {
            f["parameters"].clone()
        } else {
            json!({"type":"object","properties":{}})
        };
        let mut tool = json!({"type":"function"});
        tool["function"] = function;
        out.push(tool);
    }
    Ok(Some(Value::Array(out)))
}

#[derive(Debug, Clone)]
pub struct RequestDefaults {
    pub max_tokens: usize,
    pub max_tokens_cap: Option<usize>,
    pub thinking_budget: usize,
}
impl Default for RequestDefaults {
    fn default() -> Self {
        Self {
            max_tokens: 32768,
            max_tokens_cap: None,
            thinking_budget: std::env::var("YASAKANI_THINKING_BUDGET")
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0)
                .max(0) as usize,
        }
    }
}

// Python accepts numeric strings, bools and truncates float -> int. Reject
// non-finite / out-of-i64 values instead of wrapping or panicking.
fn numeric(v: &Value, key: &str, int: bool, lo: Option<f64>, hi: Option<f64>) -> Result<f64> {
    let n = match v {
        Value::Bool(b) => Some(u8::from(*b) as f64),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            if int {
                s.trim().parse::<i64>().ok().map(|n| n as f64)
            } else {
                s.trim().parse::<f64>().ok()
            }
        }
        _ => None,
    }
    .filter(|n| n.is_finite())
    .ok_or_else(|| Error::request(format!("{key} must be a number")))?;
    let n = if int { n.trunc() } else { n };
    if int && !(i64::MIN as f64..(i64::MAX as f64)).contains(&n) {
        return Err(Error::request(format!("{key} must be a number")));
    }
    if lo.is_some_and(|lo| n < lo) || hi.is_some_and(|hi| n > hi) {
        let bound = |v: Option<f64>| match v {
            None => "None".into(),
            Some(n) if int => format!("{}", n as i64),
            Some(0.000001) => "1e-06".into(),
            Some(n) if n.fract() == 0.0 => format!("{n:.1}"),
            Some(n) => n.to_string(),
        };
        return Err(Error::request(format!(
            "{key} must be between {} and {}",
            bound(lo),
            bound(hi)
        )));
    }
    Ok(n)
}
fn integer(v: &Value, key: &str, lo: Option<i64>) -> Result<i64> {
    let n = match v {
        Value::Number(n) if n.is_i64() => n.as_i64(),
        Value::Number(n) if n.is_u64() => n.as_u64().and_then(|n| i64::try_from(n).ok()),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => Some(numeric(v, key, true, None, None)? as i64),
    }
    .ok_or_else(|| Error::request(format!("{key} must be a number")))?;
    if lo.is_some_and(|lo| n < lo) {
        return Err(Error::request(format!(
            "{key} must be between {} and None",
            lo.unwrap_or_default()
        )));
    }
    Ok(n)
}
fn num(
    body: &Value,
    key: &str,
    default: f64,
    int: bool,
    lo: Option<f64>,
    hi: Option<f64>,
) -> Result<f64> {
    if body[key].is_null() {
        Ok(default)
    } else {
        numeric(&body[key], key, int, lo, hi)
    }
}
pub fn parse_chat_request(body: &Value, cfg: &RequestDefaults) -> Result<ChatParams> {
    if !body.is_object() {
        return Err(Error::request("the request body must be a JSON object"));
    }
    if truthy(&body["stream_options"]) && !body["stream_options"].is_object() {
        return Err(Error::request("stream_options must be an object"));
    }
    for key in ["functions", "function_call"] {
        if truthy(&body[key]) {
            return Err(Error::request(format!("{key} is not supported; use tools")));
        }
    }
    let mut tools = clean_tools(&body["tools"])?;
    match body["tool_choice"].as_str() {
        Some("none") => tools = None,
        Some("auto") => {}
        _ if body["tool_choice"].is_null() => {}
        _ => {
            return Err(Error::request(format!(
                "tool_choice {} is not supported; use \"auto\" or \"none\"",
                // As upstream: the first 80 characters of the JSON text.
                yskn_text::json_dumps(&shortened(&body["tool_choice"], 80), true)
                    .chars()
                    .take(80)
                    .collect::<String>()
            )));
        }
    }
    if truthy(&body["logprobs"]) {
        return Err(Error::request("logprobs is not supported"));
    }
    if !body["n"].is_null()
        && body["n"] != json!(1)
        && body["n"] != json!(1.0)
        && body["n"] != json!(true)
    {
        return Err(Error::request("n must be 1 (one answer per request)"));
    }
    let rf = &body["response_format"];
    if rf.is_object() && !rf["type"].is_null() && rf["type"] != "text" {
        return Err(Error::request(format!(
            "response_format {} is not supported",
            excerpt(&py_repr(&shortened(&rf["type"], EXCERPT)))
        )));
    }
    let messages = clean_messages(&body["messages"])?;
    let mt = body
        .get("max_completion_tokens")
        .unwrap_or(&body["max_tokens"]);
    let mut max_tokens = if mt.is_null() {
        cfg.max_tokens
    } else {
        integer(mt, "v", Some(1))? as usize
    };
    if let Some(cap) = cfg.max_tokens_cap.filter(|&n| n > 0) {
        max_tokens = max_tokens.min(cap);
    }
    let stop = if !truthy(&body["stop"]) {
        json!([])
    } else if body["stop"].is_string() {
        json!([body["stop"]])
    } else {
        body["stop"].clone()
    };
    let stop = stop
        .as_array()
        .filter(|a| a.len() <= 4 && a.iter().all(Value::is_string))
        .ok_or_else(|| Error::request("stop must be a string or a list of at most 4 strings"))?
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let mut kw = if !truthy(&body["chat_template_kwargs"]) {
        Map::new()
    } else {
        body["chat_template_kwargs"]
            .as_object()
            .cloned()
            .ok_or_else(|| Error::request("chat_template_kwargs must be an object"))?
    };
    // Upstream passes these on as keyword arguments too: `tools` reaches the
    // template whenever the request has no tools of its own (bypassing the
    // tool limit and tool_choice "none"), the others fail with a TypeError.
    if let Some(key) = kw.keys().find(|k| {
        yskn_text::RESERVED_VARIABLES.contains(&k.as_str()) || *k == "add_generation_prompt"
    }) {
        return Err(Error::request(format!(
            "chat_template_kwargs: {key} is set by the server and cannot be given"
        )));
    }
    if let Some(t @ ("enabled" | "disabled")) = body["thinking"]["type"].as_str() {
        kw.entry("enable_thinking").or_insert(json!(t == "enabled"));
    }
    if body["enable_thinking"].is_boolean() {
        kw.entry("enable_thinking")
            .or_insert(body["enable_thinking"].clone());
    }
    if !body["reasoning_effort"].is_null() {
        let effort = match body["reasoning_effort"].as_str() {
            Some("minimal" | "low") => "low",
            Some("medium") => "medium",
            Some("high" | "xhigh") => "xhigh",
            _ => {
                return Err(Error::request(
                    "reasoning_effort must be one of low, medium, high, xhigh",
                ));
            }
        };
        kw.entry("reasoning_effort").or_insert(json!(effort));
    }
    let nested = kw.remove("thinking_budget").unwrap_or(Value::Null);
    let tb = body.get("thinking_budget").unwrap_or(&nested);
    let thinking_budget = if tb.is_null() {
        cfg.thinking_budget
    } else {
        integer(tb, "thinking_budget", Some(0))? as usize
    };
    Ok(ChatParams {
        messages,
        max_tokens,
        temperature: num(body, "temperature", 1.0, false, Some(0.0), Some(2.0))?,
        top_p: num(body, "top_p", 0.95, false, Some(1e-6), Some(1.0))?,
        top_k: if body["top_k"].is_null() {
            20
        } else {
            integer(&body["top_k"], "top_k", None)?
        },
        min_p: num(body, "min_p", 0.0, false, Some(0.0), Some(1.0))?,
        seed: if body["seed"].is_null() {
            None
        } else {
            Some(integer(&body["seed"], "seed", None)?)
        },
        stop,
        ignore_eos: truthy(&body["ignore_eos"]),
        loop_guard: body["loop_guard"] != json!(false),
        thinking_budget,
        template_kwargs: Value::Object(kw),
        tools,
    })
}
