//! Product session over the mapped main model and shared lossless generator.
//! The caller must keep every model file unchanged for the engine's lifetime.
use std::{fs, path::PathBuf, str::FromStr, time::Instant};

use serde_json::{Value, json};
use yskn_engine::{Backend, KvMode, LogitRows, Model, SessionMemory, Snapshot};
use yskn_ref::{
    Sampler, SamplingParameters, SamplingVersion,
    draft::{Draft, DraftOptions},
    draft_head::DraftChoice,
    model::CapturedRow,
    speculation::{
        self, CombinedProposer, DraftProposer, GenerationLimits, NgramProposer, NoProposer,
        ProposalOrder, Proposer, SessionSnapshot, StopReason, Verifier,
    },
};

pub use crate::memory::MemoryEstimate;
use crate::{
    Error, Result,
    engine::{CONTEXT_MARGIN, Engine, Finish, Generation, Sampling},
};

pub const MODEL_FILE_STABILITY: &str = "Model files must not change while yskn is running.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftMode {
    None,
    Ngram,
    Neural,
    Both,
}
impl DraftMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ngram => "ngram",
            Self::Neural => "neural",
            Self::Both => "both",
        }
    }
    pub(crate) fn neural(self) -> bool {
        matches!(self, Self::Neural | Self::Both)
    }
}
impl FromStr for DraftMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "ngram" => Ok(Self::Ngram),
            "neural" => Ok(Self::Neural),
            "both" => Ok(Self::Both),
            _ => Err(Error::engine("--draft expects none|ngram|neural|both")),
        }
    }
}
#[derive(Debug, Clone)]
pub struct RealEngineConfig {
    pub model_dir: PathBuf,
    pub kv: KvMode,
    pub backend: Option<Backend>,
    pub draft: DraftMode,
    pub k: usize,
    pub context: usize,
    pub threads: Option<usize>,
    pub prefill_width: usize,
}
impl Default for RealEngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            kv: KvMode::Kv5,
            backend: None,
            draft: DraftMode::Ngram,
            k: 3,
            context: 32_768,
            threads: None,
            prefill_width: 16,
        }
    }
}
impl RealEngineConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.model_dir.is_dir() {
            return Err(Error::engine("--model-dir must name a model folder"));
        }
        if self.context == 0 || self.prefill_width == 0 || self.threads == Some(0) {
            return Err(Error::engine(
                "--ctx, --threads and prefill width must be positive",
            ));
        }
        if self.k > self.context || self.k.checked_add(1).is_none() {
            return Err(Error::engine("--k must not exceed --ctx"));
        }
        if self.draft.neural() && !(1..8).contains(&self.k) {
            return Err(Error::engine("neural --k must be in 1..8"));
        }
        Ok(())
    }
    pub(crate) fn kv_name(&self) -> &'static str {
        match self.kv {
            KvMode::Int8 => "int8",
            KvMode::Int8Profile => "int8-profile",
            KvMode::Kv5 => "kv5",
        }
    }
    pub fn inspect(&self) -> Result<ModelInfo> {
        crate::memory::inspect(self)
    }
    /// Qwen model defaults, overridden by the selected folder's generation config.
    pub fn sampling_defaults(&self) -> Result<Sampling> {
        let value = match fs::read(self.model_dir.join("generation_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes).map_err(Error::engine)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(Error::engine(e)),
        };
        let number = |key: &str, default: f64| -> Result<f64> {
            match value.get(key) {
                None => Ok(default),
                Some(v) => v.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
                    Error::engine(format!("generation_config.{key} must be a finite number"))
                }),
            }
        };
        let sampling = Sampling {
            temperature: number("temperature", 1.0)?,
            top_p: number("top_p", 0.95)?,
            top_k: match value.get("top_k") {
                None => 20,
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| Error::engine("generation_config.top_k must be an integer"))?,
            },
            min_p: number("min_p", 0.0)?,
            seed: 0,
        };
        // Validate with the same fixed-point conversions as a real request.
        sampler(&sampling, 20)?;
        Ok(sampling)
    }
}
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_file: PathBuf,
    pub model_bytes: u64,
    pub code_bytes_per_pass: u64,
    pub draft_code_bytes_per_pass: u64,
    pub layers: usize,
    pub hidden_width: usize,
    pub feed_forward_width: usize,
    pub vocabulary: usize,
    pub context: usize,
    pub model_max_context: usize,
    pub threads: usize,
    pub kv: String,
    pub backend: String,
    pub device: String,
    pub draft: String,
    pub k: usize,
    pub memory: MemoryEstimate,
}
impl ModelInfo {
    pub fn to_json(&self) -> Value {
        json!({"model_file":self.model_file,"model_bytes":self.model_bytes,"code_bytes_per_pass":self.code_bytes_per_pass,"draft_code_bytes_per_pass":self.draft_code_bytes_per_pass,"layers":self.layers,"hidden_width":self.hidden_width,"feed_forward_width":self.feed_forward_width,"vocabulary":self.vocabulary,"context":self.context,"model_max_context":self.model_max_context,"threads":self.threads,"kv":self.kv,"backend":self.backend,"device":self.device,"draft":self.draft,"k":self.k,"memory":self.memory.to_json(),"file_stability":MODEL_FILE_STABILITY})
    }
}

/// Python round() is ties-to-even; the C ABI then narrows to signed 32 bits.
/// engine.py:166-176; engine35.cpp:944-959. Version 3 is the upstream default.
pub fn sampler(sampling: &Sampling, vocabulary: usize) -> Result<Sampler> {
    let s = sampling.normalized();
    if !s.temperature.is_finite() || !s.top_p.is_finite() || !s.min_p.is_finite() {
        return Err(Error::request("sampling parameters must be finite"));
    }
    let fixed = |n: f64| -> Result<i32> {
        let v = (n * 1000.0).round_ties_even();
        if v < i32::MIN as f64 || v > i32::MAX as f64 {
            return Err(Error::request(
                "sampling parameter exceeds native fixed-point range",
            ));
        }
        Ok(v as i32)
    };
    let temp = fixed(s.temperature)?;
    let top_p = fixed(s.top_p)?;
    let min_p = fixed(s.min_p)?;
    if !(1..=1000).contains(&top_p) || !(0..=1000).contains(&min_p) {
        return Err(Error::request(
            "rounded top_p must be in 0.001..=1 and min_p in 0..=1",
        ));
    }
    if temp <= 0 {
        return Ok(Sampler::Greedy);
    }
    let mut p = SamplingParameters::from_thousandths(temp as u32, top_p as u32, min_p as u32)
        .map_err(Error::engine)?;
    p.top_k = (s.top_k as usize).min(vocabulary);
    p.seed = u64::from(s.seed as u32);
    p.version = SamplingVersion::Gumbel;
    Ok(Sampler::Sampled(p))
}

enum Drafts {
    None(NoProposer),
    Ngram(NgramProposer),
    Neural(Box<DraftProposer>, usize),
    Both(Box<CombinedProposer>, usize),
}
impl Drafts {
    fn clear_count(&mut self) {
        match self {
            Self::Neural(_, count) | Self::Both(_, count) => *count = 0,
            _ => {}
        }
    }
    fn count(&self) -> usize {
        match self {
            Self::Neural(_, count) | Self::Both(_, count) => *count,
            _ => 0,
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Neural(p, _) => p.reset(),
            Self::Both(p, _) => p.reset(),
            _ => {}
        }
    }
    fn choice(&mut self, choice: DraftChoice) {
        match self {
            Self::Neural(p, _) => p.set_choice(choice),
            Self::Both(p, _) => p.set_choice(choice),
            _ => {}
        }
    }
    fn snapshot(&mut self, main: &mut Model) -> Result<Checkpoint> {
        match self {
            Self::Neural(p, _) => p
                .snapshot_session(main)
                .map(Checkpoint::Joint)
                .map_err(Error::engine),
            Self::Both(p, _) => p
                .snapshot_session(main)
                .map(Checkpoint::Joint)
                .map_err(Error::engine),
            _ => main.snapshot().map(Checkpoint::Main).map_err(Error::engine),
        }
    }
    fn restore(&mut self, main: &mut Model, snapshot: &Checkpoint) -> Result<()> {
        match (self, snapshot) {
            (Self::Neural(p, _), Checkpoint::Joint(s)) => {
                p.restore_session(main, s).map_err(Error::engine)
            }
            (Self::Both(p, _), Checkpoint::Joint(s)) => {
                p.restore_session(main, s).map_err(Error::engine)
            }
            (Self::None(_) | Self::Ngram(_), Checkpoint::Main(s)) => {
                main.restore(s).map_err(Error::engine)
            }
            _ => Err(Error::engine("checkpoint proposer mismatch")),
        }
    }
}
impl Proposer for Drafts {
    fn propose(&mut self, h: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self {
            Self::None(p) => p.propose(h, last, max),
            Self::Ngram(p) => p.propose(h, last, max),
            Self::Neural(p, _) => p.propose(h, last, max),
            Self::Both(p, _) => p.propose(h, last, max),
        }
    }
    fn synchronize<V: Verifier>(&mut self, v: &mut V) -> yskn_ref::Result<()> {
        match self {
            Self::None(p) => p.synchronize(v),
            Self::Ngram(p) => p.synchronize(v),
            Self::Neural(p, _) => p.synchronize(v),
            Self::Both(p, _) => p.synchronize(v),
        }
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> yskn_ref::Result<()> {
        match self {
            Self::None(p) => p.observe_committed(rows),
            Self::Ngram(p) => p.observe_committed(rows),
            Self::Neural(p, _) => p.observe_committed(rows),
            Self::Both(p, _) => p.observe_committed(rows),
        }
    }
    fn propose_with<V: Verifier>(
        &mut self,
        v: &mut V,
        h: &[u32],
        last: u32,
        max: usize,
    ) -> yskn_ref::Result<Vec<u32>> {
        match self {
            Self::None(p) => p.propose_with(v, h, last, max),
            Self::Ngram(p) => p.propose_with(v, h, last, max),
            Self::Neural(p, count) => {
                if max > 0 {
                    *count += 1;
                }
                p.propose_with(v, h, last, max)
            }
            Self::Both(p, count) => {
                if max > 0 && p.propose(h, last, max).is_empty() {
                    *count += 1;
                }
                p.propose_with(v, h, last, max)
            }
        }
    }
}
enum Checkpoint {
    Main(Snapshot),
    Joint(SessionSnapshot<Snapshot>),
}
struct Saved {
    tokens: Vec<u32>,
    checkpoint: Checkpoint,
}

/// This type is intentionally not Send on macOS; construct it on the server's
/// existing engine thread, never before spawning that thread.
pub struct RealEngine {
    model: Model,
    proposer: Drafts,
    config: RealEngineConfig,
    info: ModelInfo,
    defaults: Sampling,
    history: Vec<u32>,
    saved: Option<Saved>,
}
impl RealEngine {
    pub fn load(config: RealEngineConfig) -> Result<Self> {
        Self::load_with_draft_options(config, DraftOptions::default())
    }
    pub fn load_with_draft_options(
        config: RealEngineConfig,
        options: DraftOptions,
    ) -> Result<Self> {
        let mut info = config.inspect()?;
        info.memory.enforce(info.memory.physical_bytes)?;
        let defaults = config.sampling_defaults()?;
        let backend = config
            .backend
            .map_or_else(Backend::from_env, Ok)
            .map_err(Error::engine)?;
        let threads = config
            .threads
            .unwrap_or_else(yskn_engine::Engine::default_threads);
        let kernel = std::env::var("YSKN_KERNEL")
            .map_or_else(|_| Ok(yskn_engine::Kernel::fastest()), |s| s.parse())
            .map_err(Error::engine)?;
        let pool = yskn_engine::Engine::new(kernel, threads).map_err(Error::engine)?;
        let width = config.prefill_width.max(
            config
                .k
                .checked_add(1)
                .ok_or_else(|| Error::engine("pass width overflow"))?,
        );
        let mut model = map_model(&config, pool, backend, width)?;
        let mut proposer = match config.draft {
            DraftMode::None => Drafts::None(NoProposer),
            DraftMode::Ngram => {
                Drafts::Ngram(NgramProposer::new(3, config.k).map_err(Error::engine)?)
            }
            DraftMode::Neural | DraftMode::Both => {
                let draft =
                    Draft::load_with_options(&config.model_dir, options).map_err(Error::engine)?;
                // A caller may select a smaller shortlist/capture set for a
                // compatible model. Report the actual configured head reads.
                let (_, head_path) =
                    yskn_ref::draft::model_paths(&config.model_dir).map_err(Error::engine)?;
                let layout = yskn_ref::draft_head::HeadLayout::read(
                    &mut fs::File::open(head_path).map_err(Error::engine)?,
                )
                .map_err(Error::engine)?;
                let default_head =
                    crate::memory::reduced_head_code_reads(&layout, 512.min(layout.vocabulary))?;
                let actual_head =
                    crate::memory::reduced_head_code_reads(&layout, draft.options().shortlist)?;
                info.draft_code_bytes_per_pass = info
                    .draft_code_bytes_per_pass
                    .checked_sub(default_head)
                    .and_then(|n| n.checked_add(actual_head))
                    .ok_or_else(|| Error::engine("draft code byte accounting overflow"))?;
                let neural = DraftProposer::new(draft, config.k, DraftChoice::Greedy)
                    .map_err(Error::engine)?;
                model
                    .set_capture(neural.capture_options())
                    .map_err(Error::engine)?;
                if config.draft == DraftMode::Both {
                    Drafts::Both(
                        Box::new(CombinedProposer::new(
                            NgramProposer::new(3, config.k).map_err(Error::engine)?,
                            neural,
                            ProposalOrder::NgramFirst,
                        )),
                        0,
                    )
                } else {
                    Drafts::Neural(Box::new(neural), 0)
                }
            }
        };
        proposer.synchronize(&mut model).map_err(Error::engine)?;
        info.backend = model.backend().name().into();
        info.device = model.device();
        Ok(Self {
            model,
            proposer,
            config,
            info,
            defaults,
            history: Vec::new(),
            saved: None,
        })
    }
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
    pub fn default_sampling(&self) -> &Sampling {
        &self.defaults
    }
    pub fn last_neural_proposals(&self) -> usize {
        self.proposer.count()
    }
    pub fn session_memory(&self) -> SessionMemory {
        self.model.session_memory()
    }
    /// At most one checkpoint is retained, independent of request count.
    pub fn snapshot_bytes(&self) -> u64 {
        if self.saved.is_some() {
            self.info.memory.snapshot_bytes
        } else {
            0
        }
    }
    fn empty_generation(&self, cached_tokens: usize, snap_mode: u8, start: Instant) -> Generation {
        Generation {
            ids: vec![],
            finish: Finish::Cancelled,
            cached_tokens,
            snap_mode,
            passes: 0,
            rows_verified: 0,
            guesses_offered: 0,
            guesses_accepted: 0,
            prefill_ms: start.elapsed().as_secs_f64() * 1000.0,
            gen_ms: 0.0,
        }
    }
}
/// The single mapping boundary in this crate.
#[allow(unsafe_code)]
fn map_model(
    config: &RealEngineConfig,
    pool: yskn_engine::Engine,
    backend: Backend,
    width: usize,
) -> Result<Model> {
    // SAFETY: the product's documented file-stability contract requires all
    // model files to remain unmodified and untruncated until this engine drops.
    unsafe { Model::with_options(&config.model_dir, pool, backend, width, config.kv) }
        .map_err(Error::engine)
}
fn strict_prefix(prefix: &[u32], tokens: &[u32]) -> bool {
    !prefix.is_empty() && prefix.len() < tokens.len() && tokens.starts_with(prefix)
}
impl Engine for RealEngine {
    fn context_capacity(&self) -> usize {
        self.config.context
    }
    fn reset(&mut self) -> Result<()> {
        self.saved = None;
        self.history.clear();
        self.model.reset().map_err(Error::engine)?;
        self.proposer.reset();
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stop_ids: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        let room = self
            .config
            .context
            .checked_sub(prompt.len())
            .and_then(|n| n.checked_sub(CONTEXT_MARGIN))
            .filter(|&n| n > 0);
        if prompt.is_empty() || room.is_none() {
            return Err(Error::request("the prompt leaves no room in the context"));
        }
        if prompt
            .iter()
            .any(|&id| id as usize >= self.model.dimensions().vocabulary_size)
        {
            return Err(Error::request("prompt token exceeds model vocabulary"));
        }
        let mut selector = sampler(sampling, self.model.dimensions().vocabulary_size)?;
        let choice = match &selector {
            Sampler::Greedy => DraftChoice::Greedy,
            Sampler::Sampled(p) => DraftChoice::sampled(p.seed),
        };
        self.proposer.choice(choice);
        self.proposer.clear_count();
        let begin = Instant::now();
        if !on_tokens(&[])? {
            return Ok(self.empty_generation(0, 0, begin));
        }
        let (cached_tokens, snap_mode) = if strict_prefix(&self.history, prompt) {
            (self.history.len(), 1)
        } else if self
            .saved
            .as_ref()
            .is_some_and(|s| strict_prefix(&s.tokens, prompt))
        {
            let saved = self.saved.as_ref().expect("checked saved checkpoint");
            self.proposer.restore(&mut self.model, &saved.checkpoint)?;
            self.history.clone_from(&saved.tokens);
            (saved.tokens.len(), 2)
        } else {
            self.reset()?;
            (0, 0)
        };
        // src/specx.cpp:103-108: last marker in the newly processed suffix;
        // absent marker falls back to n-1, a final marker gives no new snapshot.
        let cut = prompt[cached_tokens..]
            .iter()
            .rposition(|&id| id == 248045)
            .map_or(prompt.len() - 1, |p| cached_tokens + p + 1);
        let cut = (cut > cached_tokens && cut < prompt.len()).then_some(cut);
        let end = prompt.len() - 1; // Shared loop consumes the final prompt row.
        let mut position = cached_tokens;
        while position < end {
            let mut next = (position + self.config.prefill_width.min(end - position)).min(end);
            if let Some(cut) = cut {
                if position < cut && cut < next {
                    next = cut;
                }
            }
            let logits = if next == end {
                LogitRows::Last
            } else {
                LogitRows::None
            };
            self.model
                .prefill_with_logits(&prompt[position..next], logits)
                .map_err(Error::engine)?;
            self.history.extend_from_slice(&prompt[position..next]);
            self.proposer
                .synchronize(&mut self.model)
                .map_err(Error::engine)?;
            position = next;
            if cut == Some(position) {
                // Drop the previous slot before allocating its replacement.
                self.saved = None;
                let checkpoint = self.proposer.snapshot(&mut self.model)?;
                self.saved = Some(Saved {
                    tokens: prompt[..position].to_vec(),
                    checkpoint,
                });
            }
            if !on_tokens(&[])? {
                return Ok(self.empty_generation(cached_tokens, snap_mode, begin));
            }
        }
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.0;
        let generation_start = Instant::now();
        let mut emitted = Vec::new();
        let mut callback_error = None;
        let mut callback = |token| {
            emitted.push(token);
            match on_tokens(&emitted) {
                Ok(go) => go,
                Err(error) => {
                    callback_error = Some(error);
                    false
                }
            }
        };
        let mut limits = GenerationLimits::new(
            max_new.max(1).min(room.expect("checked room") + 1),
            self.config.context - CONTEXT_MARGIN + 1,
        );
        limits.max_draft_tokens = if self.config.draft == DraftMode::None {
            0
        } else {
            self.config.k
        };
        limits.eos_ids = stop_ids.unwrap_or(&[]);
        limits.on_token = Some(&mut callback);
        let result = speculation::generate(
            &mut self.model,
            &mut self.proposer,
            &mut selector,
            prompt,
            limits,
        );
        let gen_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.reset()?;
                return Err(Error::engine(error));
            }
        };
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.history
            .extend_from_slice(&result.tokens[..result.tokens.len().saturating_sub(1)]);
        debug_assert_eq!(self.history.len(), self.model.position());
        if let Some(error) = callback_error {
            return Err(error);
        }
        let finish = match result.stop_reason {
            StopReason::EndOfSequence => Finish::Stop,
            StopReason::Cancelled => Finish::Cancelled,
            _ => Finish::Length,
        };
        Ok(Generation {
            ids: result.tokens,
            finish,
            cached_tokens,
            snap_mode,
            passes: result.statistics.passes,
            rows_verified: result.statistics.rows_verified,
            guesses_offered: result.statistics.guesses_offered,
            guesses_accepted: result.statistics.guesses_accepted,
            prefill_ms,
            gen_ms,
        })
    }
}
