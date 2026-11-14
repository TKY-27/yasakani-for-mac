use minijinja::{
    Environment, Error, ErrorKind, Value as JValue,
    value::{Kwargs, Rest, StringInput, ValueOrKwargs, from_args},
};
use serde_json::Value;

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
    JsonFormat {
        ensure_ascii,
        indent: None,
        item_sep: ", ".into(),
        key_sep: ": ".into(),
        sort_keys: false,
    }
    .render(value, 0)
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
    Ok(format.render(&value, 0))
}

struct JsonFormat {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

impl JsonFormat {
    fn quoted(&self, text: &str) -> String {
        // serde_json's escaping agrees with json.dumps except ensure_ascii.
        let encoded = Value::String(text.into()).to_string();
        if !self.ensure_ascii {
            return encoded;
        }
        let mut result = String::new();
        for c in encoded.chars() {
            if c < '\u{7f}' {
                result.push(c);
            } else {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    result.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
        result
    }

    fn render(&self, value: &Value, depth: usize) -> String {
        let (open, close, items): (&str, &str, Vec<String>) = match value {
            Value::Array(values) => (
                "[",
                "]",
                values.iter().map(|v| self.render(v, depth + 1)).collect(),
            ),
            Value::Object(values) => {
                let mut entries: Vec<_> = values.iter().collect();
                if self.sort_keys {
                    entries.sort_by_key(|(k, _)| *k);
                }
                (
                    "{",
                    "}",
                    entries
                        .into_iter()
                        .map(|(k, v)| {
                            format!(
                                "{}{}{}",
                                self.quoted(k),
                                self.key_sep,
                                self.render(v, depth + 1)
                            )
                        })
                        .collect(),
                )
            }
            Value::String(s) => return self.quoted(s),
            Value::Number(n) if n.is_f64() => return python_float(&n.to_string()),
            _ => return value.to_string(),
        };
        if items.is_empty() {
            return format!("{open}{close}");
        }
        if let Some(indent) = &self.indent {
            let prefix = format!("\n{}", indent.repeat(depth + 1));
            format!(
                "{open}{prefix}{}\n{}{close}",
                items.join(&format!("{}{prefix}", self.item_sep)),
                indent.repeat(depth)
            )
        } else {
            format!("{open}{}{close}", items.join(&self.item_sep))
        }
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
