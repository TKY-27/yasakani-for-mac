//! Explicit deterministic demo/test engine. No model files, no inference claims.
use crate::{
    Error, Result,
    engine::{CONTEXT_MARGIN, Engine, Finish, Generation, Sampling},
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};
use yskn_text::{ChatFormatter, EOS_IDS, IM_START};

pub fn formatter() -> Result<ChatFormatter> {
    Ok(ChatFormatter::from_json(
        include_str!("../data/stub-tokenizer.json"),
        "{}",
        include_str!("../data/stub-template.jinja"),
    )?)
}
#[derive(Debug, Clone)]
pub struct Script {
    pub ids: Vec<u32>,
    pub loop_from: Option<usize>,
}
impl Script {
    pub fn once(ids: Vec<u32>) -> Self {
        Self {
            ids,
            loop_from: None,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct Observation {
    pub calls: usize,
    pub stops: usize,
    pub prompt: Vec<u32>,
    pub sampling: Option<Sampling>,
    pub generated: usize,
}
#[derive(Clone, Default)]
pub struct Control(Arc<Mutex<State>>);
#[derive(Default)]
struct State {
    scripts: VecDeque<Script>,
    delay: Duration,
    failure: Option<String>,
    observation: Observation,
}
impl Control {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub fn push(&self, script: Script) {
        self.state().scripts.push_back(script);
    }
    pub fn delay(&self, delay: Duration) {
        self.state().delay = delay;
    }
    pub fn fail_next(&self, message: impl Into<String>) {
        self.state().failure = Some(message.into());
    }
    pub fn observation(&self) -> Observation {
        self.state().observation.clone()
    }
}
pub struct StubEngine {
    capacity: usize,
    control: Control,
    history: Vec<u32>,
    snapshot: Vec<u32>,
    formatter: Option<ChatFormatter>,
}
impl StubEngine {
    pub fn new(capacity: usize) -> Self {
        Self::with_control(capacity, Control::default())
    }
    pub fn with_control(capacity: usize, control: Control) -> Self {
        Self {
            capacity,
            control,
            history: vec![],
            snapshot: vec![],
            formatter: None,
        }
    }
    pub fn using_formatter(mut self, formatter: ChatFormatter) -> Self {
        self.formatter = Some(formatter);
        self
    }
    fn default_script(&self, prompt: &[u32]) -> Result<Script> {
        let fallback;
        let fmt = if let Some(fmt) = &self.formatter {
            fmt
        } else {
            fallback = formatter()?;
            &fallback
        };
        let hash = prompt
            .iter()
            .fold(2166136261_u32, |h, id| (h ^ id).wrapping_mul(16777619));
        let text = format!("Stub response {hash:08x}. This is deterministic demo output.");
        let thinking = prompt.last() == Some(&fmt.think_start)
            || fmt.decode(prompt)?.trim_end().ends_with("<think>");
        let mut ids = if thinking {
            fmt.encode("Demo reasoning.</think>\n\n")?
        } else {
            vec![]
        };
        ids.extend(fmt.encode(&text)?);
        ids.push(EOS_IDS[0]);
        Ok(Script::once(ids))
    }
}
impl Engine for StubEngine {
    fn context_capacity(&self) -> usize {
        self.capacity
    }
    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.snapshot.clear();
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stops: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        let room = self.capacity.saturating_sub(prompt.len() + CONTEXT_MARGIN);
        if prompt.is_empty() || room < 1 {
            return Err(Error::engine("the prompt leaves no room in the context"));
        }
        let max_new = max_new.max(1).min(room + 1);
        let (script, delay, failure) = {
            let mut s = self.control.state();
            s.observation.calls += 1;
            s.observation.prompt = prompt.into();
            s.observation.sampling = Some(sampling.normalized());
            (s.scripts.pop_front(), s.delay, s.failure.take())
        };
        if let Some(f) = failure {
            return Err(Error::engine(f));
        }
        let sc = match script {
            Some(s) => s,
            None => self.default_script(prompt)?,
        };
        let (cached_tokens, snap_mode) =
            if !self.history.is_empty() && prompt.starts_with(&self.history) {
                (self.history.len(), 1)
            } else if !self.snapshot.is_empty() && prompt.starts_with(&self.snapshot) {
                (self.snapshot.len(), 2)
            } else {
                (0, 0)
            };
        let mut ids = Vec::new();
        let mut k = 0;
        let mut passes = 0;
        let mut finish = Finish::Length;
        while ids.len() < max_new {
            let count = if passes == 0 {
                1
            } else {
                [1, 3, 2, 4][(passes - 1) % 4]
            };
            for _ in 0..count {
                if ids.len() == max_new {
                    break;
                }
                if k >= sc.ids.len() {
                    k = sc
                        .loop_from
                        .filter(|&i| i < sc.ids.len())
                        .unwrap_or(sc.ids.len());
                }
                let id = sc.ids.get(k).copied().unwrap_or(EOS_IDS[0]);
                k = k.saturating_add(1);
                ids.push(id);
                if stops.is_some_and(|s| s.contains(&id)) {
                    finish = Finish::Stop;
                    break;
                }
            }
            passes += 1;
            if !delay.is_zero() {
                std::thread::sleep(delay);
            }
            self.control.state().observation.generated = ids.len();
            let go = on_tokens(&ids);
            if !matches!(go, Ok(true)) {
                self.control.state().observation.stops += 1;
                // Preserve a usable sequence even when a consumer fails.
                self.history = prompt.into();
                self.history
                    .extend_from_slice(&ids[..ids.len().saturating_sub(1)]);
                go?;
                if finish != Finish::Stop {
                    finish = Finish::Cancelled;
                }
                break;
            }
            if finish == Finish::Stop {
                break;
            }
        }
        self.history = prompt.into();
        self.history
            .extend_from_slice(&ids[..ids.len().saturating_sub(1)]);
        if let Some(last) = prompt.iter().rposition(|&id| id == IM_START) {
            self.snapshot = prompt[..=last].into();
        }
        self.control.state().observation.generated = ids.len();
        Ok(Generation {
            prefill_ms: 0.01 * (prompt.len() - cached_tokens) as f64,
            gen_ms: 0.1 * ids.len() as f64,
            rows_verified: ids.len(),
            ids,
            finish,
            cached_tokens,
            snap_mode,
            passes,
            guesses_offered: 0,
            neural_guesses_offered: 0,
            ngram_guesses_offered: 0,
            guesses_accepted: 0,
            draft_proposals: 0,
            draft_updates: 0,
        })
    }
}
