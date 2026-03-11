use std::sync::{Mutex, MutexGuard};

pub(super) fn lock_mock_backend_counters() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().expect("lock mock backend counter guard")
}

mod ccs_only_mcs_batched;
mod full_folding_integration;
mod mixed_ccs_route_a_segments;
mod mojo_backend_session;
mod output_binding;
mod rectangular_ccs_e2e;
mod riscv_proof_integration;
mod riscv_trace_wiring_ccs_e2e;
mod riscv_trace_wiring_mode_e2e;
mod riscv_trace_wiring_runner_e2e;
mod shard_continuation_extend_and_fold;
mod streaming_dec_equivalence;
