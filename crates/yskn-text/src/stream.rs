use crate::{ChatFormatter, EOS_IDS, Result, decode, error};
use tokenizers::Tokenizer;

/// A bounded re-decoding window, matching upstream's UTF-8 holdback algorithm.
/// `flush` releases a final incomplete sequence, just as full decode does.
pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    ids: Vec<u32>,
    read: usize,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            read: 0,
        }
    }

    pub fn push(&mut self, ids: &[u32]) -> Result<String> {
        // Reject unknown ids before mutating the window.
        for &id in ids {
            if self.tokenizer.id_to_token(id).is_none() {
                return Err(error(format!("the tokenizer has no token id {id}")));
            }
        }
        self.ids.extend_from_slice(ids);
        if self.ids.len() == self.read {
            return Ok(String::new());
        }
        let old = decode(self.tokenizer, &self.ids[..self.read])?;
        let new = decode(self.tokenizer, &self.ids)?;
        if new.chars().count() > old.chars().count() && !new.ends_with('\u{fffd}') {
            let piece = new.chars().skip(old.chars().count()).collect();
            self.ids.drain(..self.read);
            self.read = self.ids.len();
            return Ok(piece);
        }
        Ok(String::new())
    }

    pub fn flush(&mut self) -> Result<String> {
        let old = decode(self.tokenizer, &self.ids[..self.read])?;
        let new = decode(self.tokenizer, &self.ids)?;
        self.ids.clear();
        self.read = 0;
        Ok(new.chars().skip(old.chars().count()).collect())
    }
}

/// Feed the growing generation id array. Hidden stop ids are never emitted.
/// Events have kind `reasoning` or `content`, one per id completing text.
pub struct OutputStream<'a> {
    think_end: u32,
    pub in_reasoning: bool,
    hidden: Vec<u32>,
    reason: StreamDecoder<'a>,
    answer: StreamDecoder<'a>,
    seen: usize,
    lead: bool,
    pub think_end_at: Option<usize>,
}

impl<'a> OutputStream<'a> {
    pub fn new(formatter: &'a ChatFormatter, in_reasoning: bool) -> Self {
        Self::with_hidden_ids(formatter, in_reasoning, &EOS_IDS)
    }

    pub fn with_hidden_ids(
        formatter: &'a ChatFormatter,
        in_reasoning: bool,
        hidden: &[u32],
    ) -> Self {
        Self {
            think_end: formatter.think_end,
            in_reasoning,
            hidden: hidden.into(),
            reason: StreamDecoder::new(formatter.tokenizer()),
            answer: StreamDecoder::new(formatter.tokenizer()),
            seen: 0,
            lead: true,
            think_end_at: None,
        }
    }

    fn answer_text(&mut self, text: String) -> String {
        if self.lead {
            let text = text.trim_start_matches('\n').to_owned();
            if !text.is_empty() {
                self.lead = false;
            }
            text
        } else {
            text
        }
    }

    pub fn feed(&mut self, ids: &[u32], final_: bool) -> Result<Vec<(String, String)>> {
        if ids.len() < self.seen {
            return Err(error("generation ids must be a growing array"));
        }
        for &id in &ids[self.seen..] {
            if self.reason.tokenizer.id_to_token(id).is_none() {
                return Err(error(format!("the tokenizer has no token id {id}")));
            }
        }
        let mut events = Vec::new();
        for (index, &id) in ids.iter().enumerate().skip(self.seen) {
            if self.hidden.contains(&id) {
                continue;
            }
            if self.in_reasoning && id == self.think_end {
                emit(&mut events, "reasoning", self.reason.flush()?);
                self.in_reasoning = false;
                self.think_end_at = Some(index);
            } else if self.in_reasoning {
                emit(&mut events, "reasoning", self.reason.push(&[id])?);
            } else {
                let text = self.answer.push(&[id])?;
                emit(&mut events, "content", self.answer_text(text));
            }
        }
        self.seen = ids.len();
        if final_ {
            if self.in_reasoning {
                emit(&mut events, "reasoning", self.reason.flush()?);
            }
            let text = self.answer.flush()?;
            emit(&mut events, "content", self.answer_text(text));
        }
        Ok(events)
    }
}

fn emit(events: &mut Vec<(String, String)>, kind: &str, text: String) {
    if !text.is_empty() {
        events.push((kind.into(), text));
    }
}
