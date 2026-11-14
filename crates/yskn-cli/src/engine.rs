use crate::Result;

pub const CONTEXT_MARGIN: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub seed: i64,
}
impl Sampling {
    pub fn normalized(&self) -> Self {
        if self.temperature <= 0.0 {
            return Self {
                temperature: 0.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                seed: 0,
            };
        }
        Self {
            top_k: if (1..=20).contains(&self.top_k) {
                self.top_k
            } else {
                20
            },
            ..self.clone()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug)]
pub struct Generation {
    pub ids: Vec<u32>,
    pub finish: Finish,
    pub cached_tokens: usize,
    pub snap_mode: u8,
    pub passes: usize,
    pub prefill_ms: f64,
    pub gen_ms: f64,
}

/// Synchronous, object safe and deliberately NOT `Send`.
/// `generate` reuses a compatible prompt or resets its sequence internally.
/// The callback sees the growing committed array, including the final ids.
/// `false` stops at the next commit boundary; errors stop and propagate after
/// generation has quiesced. Call it even for chunks that decode to no text.
/// `max_new` includes the first predicted token and any EOS; reserve the
/// context margin, clamp to available room and reject an empty/oversize prompt.
pub trait Engine {
    fn context_capacity(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stop_ids: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation>;
}
