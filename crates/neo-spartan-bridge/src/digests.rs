//! Digest helpers for binding external proofs/artifacts to the Phase-1 Spartan statement.

use crate::statement::SpartanShardStatement;

/// Compute a compact digest of the verifier-side run context.
///
/// This is intended to be used as a public input for Phase-2 closure proofs (or any other proof
/// that must be bound to the same run context as the Phase-1 Spartan proof).
pub fn compute_context_digest_v1(stmt: &SpartanShardStatement) -> [u8; 32] {
    use blake3::Hasher;

    let mut h = Hasher::new();
    h.update(b"neo/spartan-bridge/context_digest/v1");
    h.update(&(stmt.version as u64).to_le_bytes());
    h.update(&stmt.params_digest);
    h.update(&stmt.ccs_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.vm_digest);
    h.update(&stmt.steps_digest);
    h.update(&stmt.program_io_digest);
    h.update(&stmt.step_linking_digest);
    h.update(&(stmt.step_count as u64).to_le_bytes());
    h.update(&[stmt.mem_enabled as u8]);
    h.update(&[stmt.output_binding_enabled as u8]);

    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}

