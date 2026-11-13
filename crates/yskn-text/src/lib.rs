//! Model-owned chat formatting, tokenization and incremental output processing.
#![forbid(unsafe_code)]

mod stream;
mod template;
mod tools;

pub use stream::{OutputStream, StreamDecoder};
pub use tools::{ToolCallFilter, convert_parameter, parse_tool_call, tool_schemas};

use serde_json::Value;
use std::{fmt, fs, path::Path};
use tokenizers::Tokenizer;

pub const EOS_IDS: [u32; 2] = [248046, 248044];
pub const IM_START: u32 = 248045;

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn error(e: impl fmt::Display) -> Error {
    Error(e.to_string())
}

/// Special tokens are retained on decode; encode never adds special tokens.
/// `options` is an object of template variables, including `add_generation_prompt`
/// (default true), `enable_thinking`, and `preserve_thinking`.
pub struct ChatFormatter {
    tokenizer: Tokenizer,
    environment: minijinja::Environment<'static>,
    bos_token: Value,
    eos_token: Value,
    pub im_start: u32,
    pub im_end: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub digit_ids: [u32; 10],
}

impl ChatFormatter {
    pub fn load(folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref();
        let tok_path = folder.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| error(format!("{}: {e}", tok_path.display())))?;
        let cfg_path = folder.join("tokenizer_config.json");
        let cfg: Value = serde_json::from_str(
            &fs::read_to_string(&cfg_path)
                .map_err(|e| error(format!("{}: {e}", cfg_path.display())))?,
        )
        .map_err(|e| error(format!("{}: {e}", cfg_path.display())))?;
        if !cfg.is_object() {
            return Err(error("tokenizer_config.json must be an object"));
        }
        let tpl_path = folder.join("chat_template.jinja");
        let template = if tpl_path.is_file() {
            fs::read_to_string(&tpl_path)
                .map_err(|e| error(format!("{}: {e}", tpl_path.display())))?
        } else {
            let entry = &cfg["chat_template"];
            let entry = if let Some(templates) = entry.as_array() {
                templates
                    .iter()
                    .rev()
                    .find(|t| t["name"] == "default")
                    .map(|t| &t["template"])
                    .unwrap_or(&Value::Null)
            } else {
                entry
            };
            entry.as_str().unwrap_or_default().to_owned()
        };
        if template.trim().is_empty() {
            return Err(error(format!(
                "no chat template found in {}",
                folder.display()
            )));
        }
        let mut environment = template::environment();
        environment
            .add_template_owned("chat", template)
            .map_err(template::error)?;
        let token_id = |text: &str| {
            tokenizer
                .token_to_id(text)
                .ok_or_else(|| error(format!("the tokenizer has no token {text}")))
        };
        let im_start = token_id("<|im_start|>")?;
        let im_end = token_id("<|im_end|>")?;
        let think_start = token_id("<think>")?;
        let think_end = token_id("</think>")?;
        if im_start != IM_START || im_end != EOS_IDS[0] {
            return Err(error(format!(
                "this tokenizer does not match the engine (<|im_start|>={im_start}, <|im_end|>={im_end})"
            )));
        }
        let mut digit_ids = [0; 10];
        for (i, id) in digit_ids.iter_mut().enumerate() {
            *id = token_id(&i.to_string())?;
        }
        Ok(Self {
            tokenizer,
            environment,
            bos_token: cfg["bos_token"].clone(),
            eos_token: cfg["eos_token"].clone(),
            im_start,
            im_end,
            think_start,
            think_end,
            digit_ids,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| error(format!("the tokenizer has no token {token}")))
    }

    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        options: &Value,
    ) -> Result<String> {
        if !messages.is_array() {
            return Err(error("messages must be an array"));
        }
        let mut context = options
            .as_object()
            .cloned()
            .ok_or_else(|| error("template options must be an object"))?;
        context.insert("messages".into(), messages.clone());
        context.insert("bos_token".into(), self.bos_token.clone());
        context.insert("eos_token".into(), self.eos_token.clone());
        context
            .entry("add_generation_prompt")
            .or_insert(Value::Bool(true));
        if let Some(tools) = tools {
            context.insert("tools".into(), tools.clone());
        }
        self.environment
            .get_template("chat")
            .map_err(template::error)?
            .render(minijinja::value::Serde(context))
            .map_err(template::error)
    }

    /// Like upstream `prompt`, always append the generation prompt.
    pub fn prompt(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        options: &Value,
    ) -> Result<(Vec<u32>, bool)> {
        let mut options = options.clone();
        let map = options
            .as_object_mut()
            .ok_or_else(|| error("template options must be an object"))?;
        map.insert("add_generation_prompt".into(), Value::Bool(true));
        let text = self.render(messages, tools, &options)?;
        Ok((self.encode(&text)?, text.trim_end().ends_with("<think>")))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(error)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        decode(&self.tokenizer, ids)
    }
}

fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    for &id in ids {
        if tokenizer.id_to_token(id).is_none() {
            return Err(error(format!("the tokenizer has no token id {id}")));
        }
    }
    tokenizer.decode(ids, false).map_err(error)
}
