use crate::{
    Error, Result,
    engine::{CONTEXT_MARGIN, Engine, Finish, Generation, Sampling},
    loopguard::{self, LoopHit, LoopWatch},
};
use serde_json::{Value, json};
use yskn_text::{ChatFormatter, EOS_IDS, OutputStream, ToolCallFilter};

#[derive(Debug, Clone)]
pub struct ChatParams {
    pub messages: Value,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub loop_guard: bool,
    pub thinking_budget: usize,
    pub template_kwargs: Value,
    pub tools: Option<Value>,
}
impl ChatParams {
    pub fn new(messages: Value) -> Self {
        Self {
            messages,
            max_tokens: 32768,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            seed: None,
            stop: vec![],
            ignore_eos: false,
            loop_guard: true,
            thinking_budget: 0,
            template_kwargs: json!({}),
            tools: None,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub cached_tokens: usize,
    pub finish: &'static str,
    pub passes: usize,
    pub prefill_ms: f64,
    pub gen_ms: f64,
    pub snap_mode: u8,
    pub loop_guard: Option<Value>,
    pub tool_calls: usize,
}
#[derive(Debug, Clone)]
pub enum Event {
    Reasoning(String),
    Content(String),
    ToolCall(Value),
    Done(Usage),
}

/// Longest proper prefix of `stop` that `buf` ends with. As upstream, only
/// prefixes no longer than the buffer are tried, longest first, so the work is
/// bounded by the held-back text rather than by the stop string. Byte indices
/// are used only at Unicode scalar boundaries.
fn held(buf: &str, stop: &str) -> usize {
    let mut k = (stop.len() - 1).min(buf.len());
    while k > 0 && !(stop.is_char_boundary(k) && buf.ends_with(&stop[..k])) {
        k -= 1;
    }
    k
}
pub struct StopFilter {
    stops: Vec<String>,
    buf: String,
    pub hit: bool,
}
impl StopFilter {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            buf: String::new(),
            hit: false,
        }
    }
    pub fn push(&mut self, text: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.into();
        }
        self.buf.push_str(text);
        let buf = &self.buf;
        // A stop longer than the buffer cannot be inside it.
        let found = self.stops.iter().filter(|s| s.len() <= buf.len());
        if let Some(cut) = found.filter_map(|s| buf.find(s.as_str())).min() {
            let out = self.buf[..cut].to_owned();
            self.buf.clear();
            self.hit = true;
            return out;
        }
        let keep = self.stops.iter().map(|s| held(buf, s)).max().unwrap_or(0);
        let tail = self.buf.split_off(self.buf.len() - keep);
        std::mem::replace(&mut self.buf, tail)
    }
    pub fn flush(&mut self) -> String {
        if self.hit {
            self.buf.clear();
            String::new()
        } else {
            std::mem::take(&mut self.buf)
        }
    }
}

pub struct ChatRunner {
    pub engine: Box<dyn Engine>,
    pub formatter: ChatFormatter,
    pub bridge: Vec<u32>,
    seed: i64,
}
fn too_long(message: String) -> Error {
    let mut e = Error::request(message);
    e.code = Some("context_length_exceeded");
    e
}
impl ChatRunner {
    pub fn new(engine: Box<dyn Engine>, formatter: ChatFormatter, seed0: i64) -> Result<Self> {
        let bridge = formatter.encode(loopguard::BRIDGE)?;
        if !bridge.contains(&formatter.think_end) {
            return Err(Error::engine(
                "tokenizer: </think> is not a single id in the loop-guard bridge",
            ));
        }
        Ok(Self {
            engine,
            formatter,
            bridge,
            seed: seed0,
        })
    }
    pub fn prepare_checked(&self, p: &ChatParams) -> Result<(Vec<u32>, bool)> {
        self.prepare_interruptible(p, &|| Ok(()))
    }
    /// [`Self::prepare_checked`], calling `go_on` before rendering, after
    /// rendering and after tokenization; its error ends the preparation there.
    /// Rendering and tokenization themselves cannot be interrupted.
    pub fn prepare_interruptible(
        &self,
        p: &ChatParams,
        go_on: &dyn Fn() -> Result<()>,
    ) -> Result<(Vec<u32>, bool)> {
        go_on()?;
        // A template or tokenizer error may quote request text: capped at the
        // 500 characters upstream keeps of a generation error.
        let rejected = |e: yskn_text::Error| {
            Error::request(format!(
                "the chat template does not accept these messages: {}",
                e.to_string().chars().take(500).collect::<String>()
            ))
        };
        let capacity = self.engine.context_capacity();
        // Mirrors ChatFormatter::prompt. No size check before tokenization: a
        // normalizer may delete any amount of text, so only the token count is
        // a sound reason to refuse. The body and text limits bound the work.
        let mut options = p.template_kwargs.clone();
        if let Some(map) = options.as_object_mut() {
            map.insert("add_generation_prompt".into(), Value::Bool(true));
        }
        let text = self
            .formatter
            .render(&p.messages, p.tools.as_ref(), &options)
            .map_err(rejected)?;
        drop(options);
        go_on()?;
        let ids = self.formatter.encode(&text).map_err(rejected)?;
        go_on()?;
        if ids.len() >= capacity.saturating_sub(CONTEXT_MARGIN) {
            return Err(too_long(format!(
                "the input is too long: {} tokens exceeds the model's maximum context length of {capacity} tokens (input, thinking and answer together)",
                ids.len()
            )));
        }
        Ok((ids, text.trim_end().ends_with("<think>")))
    }
    pub fn warmup(&mut self) -> Result<()> {
        let plain = self.formatter.encode("warm-up")?;
        let chat = self
            .formatter
            .prompt(
                &json!([{"role":"user","content":"warm-up"}]),
                None,
                &json!({}),
            )?
            .0;
        let sampling = Sampling {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            seed: 0,
        };
        for prompt in [plain, chat] {
            self.engine
                .generate(&prompt, 32, &sampling, Some(&EOS_IDS), &mut |_| Ok(true))?;
        }
        self.engine.reset()
    }
    pub fn run(
        &mut self,
        p: &ChatParams,
        emit: &mut dyn FnMut(Event) -> Result<()>,
    ) -> Result<Usage> {
        let prepared = self.prepare_checked(p)?;
        self.run_prepared(p, prepared, &|| false, emit)
    }
    pub fn run_prepared(
        &mut self,
        p: &ChatParams,
        prepared: (Vec<u32>, bool),
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(Event) -> Result<()>,
    ) -> Result<Usage> {
        let (prompt, in_reasoning) = prepared;
        let max_new = p.max_tokens.max(1);
        let seed = if let Some(s) = p.seed {
            s
        } else {
            let s = self.seed;
            self.seed = self
                .seed
                .checked_add(1)
                .ok_or_else(|| Error::engine("request seed counter exhausted"))?;
            s
        };
        let sampling = Sampling {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            min_p: p.min_p,
            seed,
        }
        .normalized();
        let stops = if p.ignore_eos {
            None
        } else {
            Some(EOS_IDS.as_slice())
        };
        let guard_on = p.loop_guard && !p.ignore_eos;
        let budget = if in_reasoning && !p.ignore_eos {
            p.thinking_budget
        } else {
            0
        };
        let mut delivery = Delivery {
            stop: StopFilter::new(&p.stop),
            tool: p.tools.as_ref().map(|t| ToolCallFilter::new(Some(t))),
            emit,
        };
        let mut stream = OutputStream::new(&self.formatter, in_reasoning);
        let mut watch = LoopWatch::new(
            self.formatter.think_end,
            self.formatter.digit_ids,
            in_reasoning,
            0,
            budget,
            guard_on,
        );
        let (r1, hit) = stage(
            self.engine.as_mut(),
            &prompt,
            max_new,
            &sampling,
            stops,
            &mut stream,
            &mut watch,
            &mut delivery,
            cancelled,
        )?;
        let n_out = r1.ids.len();
        let mut u = Usage {
            prompt_tokens: prompt.len(),
            cached_tokens: r1.cached_tokens,
            snap_mode: r1.snap_mode,
            passes: r1.passes,
            prefill_ms: r1.prefill_ms,
            gen_ms: r1.gen_ms,
            reasoning_tokens: stream.think_end_at.unwrap_or(if stream.in_reasoning {
                n_out
            } else {
                0
            }),
            ..Usage::default()
        };
        let mut finish = r1.finish;
        let mut extra = 0;
        if let Some(hit) = hit {
            let mut info = json!({"where":hit.where_,"kind":hit.kind,"period":hit.period,"run":hit.run,"at":hit.at,"tokens_before":n_out});
            if hit.where_ == "reasoning"
                && r1.finish != Finish::Stop
                && !delivery.stop.hit
                && !cancelled()
            {
                let mut seq = prompt;
                seq.extend_from_slice(&r1.ids);
                seq.extend_from_slice(&self.bridge);
                let n2 = max_new.saturating_sub(1 + n_out + self.bridge.len()).min(
                    self.engine
                        .context_capacity()
                        .saturating_sub(seq.len() + CONTEXT_MARGIN),
                );
                if n2 < 16 {
                    info["answer_tokens"] = json!(0);
                    finish = Finish::Length;
                } else {
                    (delivery.emit)(Event::Reasoning(
                        if hit.kind == "budget" {
                            loopguard::BUDGET_NOTICE
                        } else {
                            loopguard::NOTICE
                        }
                        .into(),
                    ))?;
                    let mut stream = OutputStream::new(&self.formatter, false);
                    let mut watch = LoopWatch::new(
                        self.formatter.think_end,
                        self.formatter.digit_ids,
                        false,
                        0,
                        0,
                        guard_on,
                    );
                    let (r2, hit2) = stage(
                        self.engine.as_mut(),
                        &seq,
                        n2 + 1,
                        &sampling,
                        stops,
                        &mut stream,
                        &mut watch,
                        &mut delivery,
                        cancelled,
                    )?;
                    extra = self.bridge.len() + r2.ids.len();
                    info["answer_tokens"] = json!(r2.ids.len());
                    info["answer_cached"] = json!(r2.cached_tokens);
                    info["extra_tokens"] = json!(extra);
                    u.passes += r2.passes;
                    u.prefill_ms += r2.prefill_ms;
                    u.gen_ms += r2.gen_ms;
                    if let Some(h) = hit2 {
                        info["answer_loop"] = json!({"kind":h.kind,"period":h.period,"at":h.at});
                    }
                    finish = r2.finish;
                }
            } else {
                finish = if delivery.stop.hit {
                    Finish::Stop
                } else {
                    Finish::Length
                };
            }
            u.loop_guard = Some(info);
        }
        if delivery.stop.hit {
            finish = Finish::Stop;
        }
        let tail = delivery.stop.flush();
        delivery.answer(&tail)?;
        if let Some(tool) = &mut delivery.tool {
            for (kind, item) in tool.flush() {
                emit_tool(delivery.emit, &kind, item)?;
            }
            u.tool_calls = tool.calls;
        }
        u.completion_tokens = n_out + extra;
        u.finish = if finish == Finish::Stop {
            if u.tool_calls > 0 {
                "tool_calls"
            } else {
                "stop"
            }
        } else {
            "length"
        };
        (delivery.emit)(Event::Done(u.clone()))?;
        Ok(u)
    }
}
fn emit_tool(emit: &mut dyn FnMut(Event) -> Result<()>, kind: &str, item: Value) -> Result<()> {
    emit(if kind == "tool_call" {
        Event::ToolCall(item)
    } else {
        Event::Content(item.as_str().unwrap_or_default().into())
    })
}
struct Delivery<'a> {
    stop: StopFilter,
    tool: Option<ToolCallFilter>,
    emit: &'a mut dyn FnMut(Event) -> Result<()>,
}
impl Delivery<'_> {
    fn answer(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if let Some(tool) = &mut self.tool {
            for (kind, item) in tool.push(text) {
                emit_tool(self.emit, &kind, item)?;
            }
        } else {
            (self.emit)(Event::Content(text.into()))?;
        }
        Ok(())
    }
    fn deliver(&mut self, events: Vec<(String, String)>) -> Result<()> {
        for (kind, text) in events {
            if kind == "content" {
                let text = self.stop.push(&text);
                self.answer(&text)?;
            } else if !text.is_empty() {
                (self.emit)(Event::Reasoning(text))?;
            }
        }
        Ok(())
    }
}
// A stage owns all callback state; the engine is quiescent before final flush
// or any bridge continuation, including on a failed consumer.
#[allow(clippy::too_many_arguments)]
fn stage(
    engine: &mut dyn Engine,
    prompt: &[u32],
    max_new: usize,
    sampling: &Sampling,
    stops: Option<&[u32]>,
    stream: &mut OutputStream<'_>,
    watch: &mut LoopWatch,
    delivery: &mut Delivery<'_>,
    cancelled: &dyn Fn() -> bool,
) -> Result<(Generation, Option<LoopHit>)> {
    let mut hit = None;
    let r = engine.generate(prompt, max_new, sampling, stops, &mut |ids| {
        if cancelled() {
            return Ok(false);
        }
        delivery.deliver(stream.feed(ids, false)?)?;
        if hit.is_none() {
            hit = watch.check(ids);
        }
        Ok(!cancelled() && !delivery.stop.hit && hit.is_none())
    })?;
    if !cancelled() {
        delivery.deliver(stream.feed(&r.ids, true)?)?;
    }
    Ok((r, hit))
}
