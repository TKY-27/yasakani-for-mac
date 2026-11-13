use serde_json::{Map, Value, json};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// Function name -> parameter name -> JSON schema, preserving insertion order.
pub fn tool_schemas(tools: Option<&Value>) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(tools) = tools.and_then(Value::as_array) {
        for tool in tools {
            let function = tool
                .get("function")
                .filter(|f| f.is_object())
                .unwrap_or(tool);
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                let props = function.get("parameters").and_then(|p| p.get("properties"));
                out.insert(
                    name.into(),
                    props
                        .filter(|v| v.is_object())
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                );
            }
        }
    }
    out
}

pub fn convert_parameter(value: &str, schema: Option<&Value>) -> Value {
    let string = match schema.and_then(|s| s.get("type")) {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(types)) => {
            types.iter().any(|t| t == "string")
                && types.iter().all(|t| t == "string" || t == "null")
        }
        _ => false,
    };
    if string {
        return Value::String(value.into());
    }
    match serde_json::from_str::<Value>(value) {
        Ok(v) if !v.is_string() || schema.is_some_and(Value::is_object) => v,
        _ => Value::String(value.into()),
    }
}

// Match the upstream <function=([^>\n]+)> / <parameter=...> patterns.
fn tag<'a>(text: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    let mut rest = text;
    while let Some(i) = rest.find(prefix) {
        rest = &rest[i + prefix.len()..];
        if let Some(end) = rest.find('>') {
            let name = &rest[..end];
            if !name.is_empty() && !name.contains('\n') {
                return Some((name, &rest[end + 1..]));
            }
        }
    }
    None
}

pub fn parse_tool_call(block: &str, schemas: &Map<String, Value>) -> Option<Value> {
    if let Some((name, mut rest)) = tag(block, "<function=") {
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let props = schemas.get(name);
        let mut args = Map::new();
        while let Some((key, after)) = tag(rest, "<parameter=") {
            let Some(end) = after.find("</parameter>") else {
                break;
            };
            let value = &after[..end];
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            let key = key.trim();
            args.insert(
                key.into(),
                convert_parameter(value, props.and_then(|p| p.get(key))),
            );
            rest = &after[end + "</parameter>".len()..];
        }
        return Some(json!({"name": name, "arguments": args}));
    }
    let text = block.trim();
    if !text.starts_with('{') {
        return None;
    }
    let object: Value = serde_json::from_str(text).ok()?;
    let name = object.get("name")?.as_str()?;
    if name.is_empty() {
        return None;
    }
    let mut args = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(text) = args.as_str() {
        args = serde_json::from_str(text).ok()?;
    }
    if args.is_null() {
        args = json!({});
    }
    args.is_object()
        .then(|| json!({"name": name, "arguments": args}))
}

/// Stream content and complete tool calls; preserve unreadable/unfinished calls as text.
/// Events have kind `content` (string value) or `tool_call` (object value).
pub struct ToolCallFilter {
    schemas: Map<String, Value>,
    buf: String,
    inside: bool,
    gap: String,
    pub calls: usize,
}

impl ToolCallFilter {
    pub fn new(tools: Option<&Value>) -> Self {
        Self {
            schemas: tool_schemas(tools),
            buf: String::new(),
            inside: false,
            gap: String::new(),
            calls: 0,
        }
    }

    fn text(&self, text: &str, out: &mut Vec<(String, Value)>) {
        if !text.is_empty() && (!text.trim().is_empty() || self.calls == 0) {
            out.push(("content".into(), Value::String(text.into())));
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        self.buf.push_str(text);
        loop {
            if self.inside {
                let Some(j) = self.buf.find(TOOL_CLOSE) else {
                    return out;
                };
                let block = &self.buf[..j];
                if let Some(call) = parse_tool_call(block, &self.schemas) {
                    self.calls += 1;
                    out.push(("tool_call".into(), call));
                } else {
                    out.push((
                        "content".into(),
                        Value::String(format!("{}{TOOL_OPEN}{block}{TOOL_CLOSE}", self.gap)),
                    ));
                }
                self.buf.drain(..j + TOOL_CLOSE.len());
                self.inside = false;
                self.gap.clear();
                continue;
            }
            if let Some(i) = self.buf.find(TOOL_OPEN) {
                let before = &self.buf[..i];
                let trimmed = before.trim_end();
                self.text(trimmed, &mut out);
                self.gap = before[trimmed.len()..].into();
                self.buf.drain(..i + TOOL_OPEN.len());
                self.inside = true;
                continue;
            }
            let keep = (1..TOOL_OPEN.len())
                .rev()
                .find(|&k| self.buf.ends_with(&TOOL_OPEN[..k]))
                .unwrap_or(0);
            let head = &self.buf[..self.buf.len() - keep];
            let cut = head.trim_end().len();
            if cut != 0 {
                self.text(&self.buf[..cut], &mut out);
                self.buf.drain(..cut);
            }
            return out;
        }
    }

    pub fn flush(&mut self) -> Vec<(String, Value)> {
        let rest = std::mem::take(&mut self.buf);
        if self.inside {
            self.inside = false;
            let text = format!("{}{TOOL_OPEN}{rest}", self.gap);
            self.gap.clear();
            vec![("content".into(), Value::String(text))]
        } else if !rest.trim().is_empty() || (!rest.is_empty() && self.calls == 0) {
            vec![("content".into(), Value::String(rest))]
        } else {
            Vec::new()
        }
    }
}
