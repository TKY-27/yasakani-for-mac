use minijinja::{
    Environment, Error, ErrorKind, Value as JValue,
    value::{Kwargs, Rest, StringInput, ValueOrKwargs, from_args},
};
use serde_json::Value;
use std::fmt;

pub(crate) fn error(e: Error) -> crate::Error {
    crate::error(
        e.detail()
            .map(str::to_owned)
            .unwrap_or_else(|| e.to_string()),
    )
}

pub(crate) fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("default delimiters are valid"),
    );
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.set_formatter(|out, state, value| {
        if value.kind() == minijinja::value::ValueKind::Number && !value.is_integer() {
            let number = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
            minijinja::escape_formatter(
                out,
                state,
                &JValue::from(python_float(&number.to_string())),
            )
        } else {
            minijinja::escape_formatter(out, state, value)
        }
    });
    env.add_filter("trim", |value: StringInput<'_>, chars: Option<String>| {
        let trimmed = if let Some(chars) = chars {
            value
                .as_str()
                .trim_matches(|c| chars.contains(c))
                .to_owned()
        } else {
            value
                .as_str()
                .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
                .to_owned()
        };
        value.preserve_safety(trimmed)
    });
    env.add_filter("tojson", tojson);
    env.add_filter("items", |v: JValue| {
        if !v.is_undefined() && v.kind() != minijinja::value::ValueKind::Map {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "Can only get item pairs from a mapping.",
            ));
        }
        minijinja::filters::items(&v)
    });
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env
}

/// Python-compatible JSON spelling for protocol-embedded argument strings.
pub fn json_dumps(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    // Writing into a String cannot fail.
    let _ = write_json(value, ensure_ascii, &mut out);
    out
}

/// [`json_dumps`]'s text written into `out` piece by piece, without an
/// intermediate copy of the value. An error from `out` stops the writing
/// there, so a sink that keeps only a prefix pays only for the prefix.
pub fn write_json(value: &Value, ensure_ascii: bool, out: &mut dyn fmt::Write) -> fmt::Result {
    JsonFormat {
        ensure_ascii,
        indent: None,
        item_sep: ", ".into(),
        key_sep: ": ".into(),
        sort_keys: false,
    }
    .write(value, 0, out)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

fn tojson(value: JValue, args: Rest<ValueOrKwargs>) -> Result<String, Error> {
    let args = args.into_values();
    let (pos, kwargs): (&[JValue], Kwargs) = from_args(&args)?;
    if pos.len() > 4 {
        return Err(invalid("tojson takes at most five arguments"));
    }
    let ensure_ascii = pos.first().map(|v| bool::try_from(v.clone())).transpose()?;
    let indent = pos.get(1).cloned();
    let separators = pos.get(2).cloned();
    let sort_keys = pos.get(3).map(|v| bool::try_from(v.clone())).transpose()?;
    let ensure_ascii = ensure_ascii
        .or(kwargs.get("ensure_ascii")?)
        .unwrap_or(false);
    let indent = indent.or(kwargs.get("indent")?);
    let separators = separators.or(kwargs.get("separators")?);
    let sort_keys = sort_keys.or(kwargs.get("sort_keys")?).unwrap_or(false);
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            if let Some(s) = v.as_str() {
                Some(s.to_owned())
            } else if let Ok(n) = i64::try_from(v) {
                Some(" ".repeat(n.max(0) as usize))
            } else {
                return Err(invalid("indent must be an integer or a string"));
            }
        }
    };
    let (item_sep, key_sep) = if let Some(v) = separators.filter(|v| !v.is_none()) {
        let parts: Vec<String> = v
            .try_iter()?
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("separators must contain strings"))
            })
            .collect::<Result<_, _>>()?;
        if parts.len() != 2 {
            return Err(invalid("separators must have two elements"));
        }
        (parts[0].clone(), parts[1].clone())
    } else {
        (
            if indent.is_some() { "," } else { ", " }.into(),
            ": ".into(),
        )
    };
    let value = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
    let format = JsonFormat {
        ensure_ascii,
        indent,
        item_sep,
        key_sep,
        sort_keys,
    };
    let mut out = String::new();
    format
        .write(&value, 0, &mut out)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(out)
}

struct JsonFormat {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

type Entries<'a> = Box<dyn Iterator<Item = (Option<&'a String>, &'a Value)> + 'a>;

impl JsonFormat {
    /// json.dumps' string escaping: serde_json's (`\"`, `\\`, the short
    /// escapes and `\u00XX` for other controls), plus `\uXXXX` UTF-16 units
    /// for everything from DEL up with `ensure_ascii`.
    fn write_quoted(&self, text: &str, out: &mut dyn fmt::Write) -> fmt::Result {
        out.write_char('"')?;
        let mut plain = 0;
        for (i, c) in text.char_indices() {
            if !(matches!(c, '"' | '\\') || c < ' ' || (self.ensure_ascii && c >= '\u{7f}')) {
                continue;
            }
            // Unescaped runs are written whole, not character by character.
            out.write_str(&text[plain..i])?;
            plain = i + c.len_utf8();
            match c {
                '"' => out.write_str("\\\"")?,
                '\\' => out.write_str("\\\\")?,
                '\u{8}' => out.write_str("\\b")?,
                '\u{c}' => out.write_str("\\f")?,
                '\n' => out.write_str("\\n")?,
                '\r' => out.write_str("\\r")?,
                '\t' => out.write_str("\\t")?,
                c => {
                    for unit in c.encode_utf16(&mut [0; 2]) {
                        write!(out, "\\u{unit:04x}")?;
                    }
                }
            }
        }
        out.write_str(&text[plain..])?;
        out.write_char('"')
    }

    fn write(&self, value: &Value, depth: usize, out: &mut dyn fmt::Write) -> fmt::Result {
        let (open, close, entries): (char, char, Entries<'_>) = match value {
            Value::Array(values) => ('[', ']', Box::new(values.iter().map(|v| (None, v)))),
            Value::Object(values) if self.sort_keys => {
                let mut sorted: Vec<_> = values.iter().collect();
                sorted.sort_by_key(|(k, _)| *k);
                (
                    '{',
                    '}',
                    Box::new(sorted.into_iter().map(|(k, v)| (Some(k), v))),
                )
            }
            Value::Object(values) => ('{', '}', Box::new(values.iter().map(|(k, v)| (Some(k), v)))),
            Value::String(s) => return self.write_quoted(s, out),
            Value::Number(n) if n.is_f64() => return out.write_str(&python_float(&n.to_string())),
            _ => return write!(out, "{value}"),
        };
        let prefix = self
            .indent
            .as_ref()
            .map(|indent| format!("\n{}", indent.repeat(depth + 1)));
        out.write_char(open)?;
        let mut empty = true;
        for (key, v) in entries {
            if !empty {
                out.write_str(&self.item_sep)?;
            }
            empty = false;
            if let Some(prefix) = &prefix {
                out.write_str(prefix)?;
            }
            if let Some(key) = key {
                self.write_quoted(key, out)?;
                out.write_str(&self.key_sep)?;
            }
            self.write(v, depth + 1, out)?;
        }
        if let Some(indent) = self.indent.as_ref().filter(|_| !empty) {
            write!(out, "\n{}", indent.repeat(depth))?;
        }
        out.write_char(close)
    }
}

// Python uses scientific notation below 1e-4 and from 1e16, and pads
// signed exponents to two digits. Keep the serializer's shortest digits.
fn python_float(text: &str) -> String {
    let (sign, unsigned) = text.strip_prefix('-').map_or(("", text), |s| ("-", s));
    let (mantissa, exponent) = unsigned.split_once('e').unwrap_or((unsigned, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let leading = digits.len() - digits.trim_start_matches('0').len();
    let digits = digits.trim_start_matches('0').trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let power = exponent + decimal - leading as i32 - 1;
    if !(-4..16).contains(&power) {
        let mantissa = if digits.len() == 1 {
            digits.to_owned()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        return format!("{sign}{mantissa}e{power:+03}");
    }
    let point = power + 1;
    if point <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_format_and_order() {
        let env = environment();
        let value = json!({"z": ["日本語", true, null], "a": {"b": "<>&", "c": "👍🏽"}});
        for (source, expected) in [
            (
                "{{ value|tojson }}",
                "{\"z\": [\"日本語\", true, null], \"a\": {\"b\": \"<>&\", \"c\": \"👍🏽\"}}",
            ),
            (
                "{{ value|tojson(indent=2) }}",
                "{\n  \"z\": [\n    \"日本語\",\n    true,\n    null\n  ],\n  \"a\": {\n    \"b\": \"<>&\",\n    \"c\": \"👍🏽\"\n  }\n}",
            ),
            (
                "{{ value|tojson(separators=[',', ':'], sort_keys=true) }}",
                "{\"a\":{\"b\":\"<>&\",\"c\":\"👍🏽\"},\"z\":[\"日本語\",true,null]}",
            ),
        ] {
            assert_eq!(
                env.render_str(source, minijinja::value::Serde(json!({"value": value})))
                    .unwrap(),
                expected
            );
        }
        assert_eq!(
            env.render_str("{{ '日👍'|tojson(ensure_ascii=true) }}", ())
                .unwrap(),
            "\"\\u65e5\\ud83d\\udc4d\""
        );
        assert_eq!(
            env.render_str("{{ [1, 2]|tojson(false, '\t', [',', ':'], false) }}", ())
                .unwrap(),
            "[\n\t1,\n\t2\n]"
        );
        assert_eq!(
            env.render_str("{{ [1e-7, 1e20, -0.0, 1e15, 0.00001, 0.0001]|tojson }}", ())
                .unwrap(),
            "[1e-07, 1e+20, -0.0, 1000000000000000.0, 1e-05, 0.0001]"
        );
    }

    #[test]
    fn json_text_is_streamed_and_stops_with_its_sink() {
        // String escaping agrees with serde_json's, which agrees with json.dumps.
        let all: String = ('\0'..='\u{3ff}').chain(['\u{2028}', '😀']).collect();
        assert_eq!(json_dumps(&json!(all), false), json!(all).to_string());
        assert_eq!(
            json_dumps(&json!(["\"\\\u{7f}é😀"]), true),
            "[\"\\\"\\\\\\u007f\\u00e9\\ud83d\\ude00\"]"
        );
        assert_eq!(
            environment()
                .render_str("{{ {'a': [], 'b': {}, 'c': [1]}|tojson(indent=2) }}", ())
                .unwrap(),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1\n  ]\n}"
        );
        // A sink's error ends the writing: the rest of the value is not visited.
        struct Writes(usize);
        impl fmt::Write for Writes {
            fn write_str(&mut self, _: &str) -> fmt::Result {
                self.0 += 1;
                if self.0 == 5 { Err(fmt::Error) } else { Ok(()) }
            }
        }
        let mut sink = Writes(0);
        assert!(write_json(&json!(vec![vec!["ab"; 1000]; 1000]), false, &mut sink).is_err());
        assert_eq!(sink.0, 5);
    }

    #[test]
    fn whitespace_loop_controls_python_methods_and_user_errors() {
        let env = environment();
        let source = "  {% for s in [' a ', 'b', 'c'] %}\n{% if s == 'b' %}{% continue %}{% endif %}{{ s.strip().upper() }}{% if s == 'c' %}{% break %}{% endif %}{% endfor %}";
        assert_eq!(env.render_str(source, ()).unwrap(), "AC");
        assert_eq!(
            error(
                env.render_str("{{ raise_exception('bad request') }}", ())
                    .unwrap_err()
            )
            .to_string(),
            "bad request"
        );
    }
}
