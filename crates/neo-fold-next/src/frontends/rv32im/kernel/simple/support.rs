//! Owns small mechanical helpers for Simple Kernel flows.

use std::time::Instant;

pub(super) fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub(super) fn allow_parallel_step_build(count: usize) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && count >= 8
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = count;
        false
    }
}
