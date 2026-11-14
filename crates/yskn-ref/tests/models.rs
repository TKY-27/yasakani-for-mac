mod support;

// Group expensive scalar cases so libtest can use the runner's three cores
// across suites, rather than leaving two cores idle at each binary's tail.
#[path = "model_cases/draft_wiring.rs"]
mod draft_wiring;
#[path = "model_cases/kv_attention.rs"]
mod kv_attention;
#[path = "model_cases/model_invariants.rs"]
mod model_invariants;
