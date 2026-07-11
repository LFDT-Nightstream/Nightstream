//! Π_CCS.V in-circuit verifier — SplitNcV1 variant.
//!
//! Owns the byte-for-byte mirror of the verifier path that
//! [`crate::engine::optimized::verify_pi_ccs`] runs natively. That path is:
//!
//! ```text
//! neo_reductions::optimized_engine::optimized_verify_with_cache_and_instance_digest_and_perf
//!   → optimized_verify_with_cache_and_public_instance_digest_impl
//!       → bind_header_and_instance_digest_with_digest    (instance-digest variant)
//!       → bind_me_inputs_accumulator_handle               (verified-parent U_i handle)
//!       → sample_challenges  → sample_beta_m
//!       → FE sumcheck (verify_sumcheck_rounds_poseidon_v3) → FE terminal identity
//!       → NC sumcheck (verify_sumcheck_rounds_poseidon_v3) → NC terminal identity
//!   ← back in `engine::optimized::verify_pi_ccs`:
//!       tr.digest32()  // header_digest catch-up squeeze, advances state
//!                      // before pi_rlc::verify samples ρ
//! ```
//!
//! Every `nifs::prove` proof flows through that exact path on the native
//! side, so F'-side recursion must replicate it bit-for-bit (transcript
//! state) and identity-for-identity (algebraic checks).
//!
//! ## Submodule layout
//!
//! - [`transcript`] — `EngineChallenges`, K-batch sampling, raw absorbs for
//!   header / instance digest / ME-input accumulator handle.
//! - [`digests`] — the three authoritative per-claim digest gadgets
//!   (`enforce_ccs_claim_digest`, `enforce_ce_claim_digest`,
//!   `enforce_pi_ccs_instance_digest`). The retired per-claim ME-input
//!   projection digest is gone; the ME-input transcript binding is now
//!   the single 4-lane accumulator handle absorbed by
//!   [`absorb_engine_me_inputs_accumulator_handle`].
//! - [`fe`] — FE channel: `claimed_initial`, sumcheck driver, sparse-poly
//!   eval, terminal identity.
//! - [`nc`] — NC channel: `range_product`, sumcheck driver, terminal
//!   identity. (Lands in sub-step G/H.)
//!
//! ## Soundness contract
//!
//! For any honest prover output `(fresh, running, proof)`, driving these
//! gadgets on the same transcript and witness wires must produce a satisfied
//! constraint system. Any tampered protocol message must break some
//! constraint either via a transcript-state divergence (squeezed challenges
//! pinned to native values) or via an algebraic identity failure.
//!
//! "SplitNcV1" matches `PiCcsProofVariant::SplitNcV1` in `neo-reductions`.
//! The variant tag is bound into the engine's header bundle, so this module
//! doesn't absorb it separately — it's already inside `header_bundle[0..4]`.

#![allow(dead_code)] // pieces land incrementally; full composition is sub-step J.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::R1csBuilder;

pub mod digests;
pub mod fe;
pub mod nc;
pub mod transcript;
pub mod verifier;

// Re-export the public surface so external paths (tests, NIFS.V) keep using
// `paper::reductions::pi_ccs_split_nc_circuit::*` without touching the
// internal module split.
pub use digests::{
    enforce_accumulator_ce_claim_digest, enforce_ccs_claim_digest, enforce_ce_claim_digest,
    enforce_pi_ccs_instance_digest, enforce_pi_ccs_instance_digest_parent_authority, enforce_pi_ccs_outputs_digest,
    AccumulatorCeClaimDigestInputs, CeClaimDigestInputs, PiCcsOutputClaimDigestInputs,
};
pub use fe::{
    enforce_fe_claimed_initial, enforce_fe_sumcheck_driver, enforce_fe_terminal_identity, FeClaimedInitialInputs,
    FeSumcheckResult, FeTerminalInputs,
};
pub use nc::{
    enforce_nc_range_product, enforce_nc_sumcheck_driver, enforce_nc_terminal_identity, NcSumcheckResult,
    NcTerminalInputs,
};
pub use transcript::{
    absorb_engine_header_bundle_and_instance_digest, absorb_engine_header_bundle_wires_and_instance_digest,
    absorb_engine_me_inputs_accumulator_handle, enforce_header_digest_catch_up, enforce_header_digest_catch_up_wires,
    header_digest_bytes_to_fields, sample_engine_beta_m, sample_engine_challenges, EngineChallenges,
};
pub use verifier::{
    enforce_split_nc_pi_ccs_v, enforce_split_nc_pi_ccs_v_with_header_bundle_wires, SplitNcPiCcsOutputWires,
    SplitNcPiCcsVConfig, SplitNcPiCcsVDerived, SplitNcPiCcsVMessages,
};

/// Errors emitted by the SplitNcV1 in-circuit verifier and its building-block
/// gadgets. Wraps shape-mismatch surfaces that would otherwise become
/// in-gadget panics — soundness for an honest prover is unaffected, but
/// callers (composition + tests) get a structured rejection path.
#[derive(Debug, Error)]
pub enum Error {
    #[error("SplitNcV1 circuit shape mismatch: {0}")]
    Shape(String),
}

// ── Shared low-level helpers (cross-submodule) ────────────────────────────

/// Allocate a fresh wire bound to a constant Goldilocks value via a single
/// equality constraint. Mirrors the private `alloc_constant` in
/// `engine/r1cs_circuit/transcript.rs`; duplicated here because that one is
/// not pub.
pub(super) fn alloc_constant_var(builder: &mut R1csBuilder, c: F) -> Var {
    let v = builder.alloc(c);
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(c));
    v
}

/// Mirror of native `extend_packed_bytes_as_fields`. Pushes `[len, limb0,
/// limb1, …]` where each limb packs 7 bytes little-endian into a u64
/// embedded as F. The trailing partial chunk is zero-padded to 8 bytes
/// before the u64 read, exactly as native does.
pub(super) fn extend_packed_bytes_as_fields_wires(builder: &mut R1csBuilder, dst: &mut Vec<Var>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(alloc_constant_var(builder, F::from_u64(bytes.len() as u64)));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(alloc_constant_var(builder, F::from_u64(u64::from_le_bytes(limb))));
    }
}

/// Mirror of native `extend_f_slice`: pushes `[len(vals), vals...]`. The
/// length is a const wire; the slice contents are passed through verbatim.
pub(super) fn extend_f_slice_wires(builder: &mut R1csBuilder, dst: &mut Vec<Var>, vals: &[Var]) {
    dst.push(alloc_constant_var(builder, F::from_u64(vals.len() as u64)));
    dst.extend_from_slice(vals);
}

/// Number of base-field coefficients in one `K` element. `K = F[X]/(X² - 7)`,
/// so this is `2`. The native digest absorbs `coeffs_len` as a length header
/// before each `K`-slice; we mirror that on the wire side.
pub(super) const K_COEFFS_LEN: u64 = 2;

/// Mirror of native `extend_k_slice`: pushes `[len(vals), K_COEFFS_LEN,
/// v0.c0, v0.c1, v1.c0, v1.c1, …]`. Empty `vals` pushes `[0, 0]` (matches
/// the `unwrap_or(0)` branch in native).
pub(super) fn extend_k_slice_wires(
    builder: &mut R1csBuilder,
    dst: &mut Vec<Var>,
    vals: &[crate::engine::r1cs_circuit::field_ext::KVar],
) {
    dst.push(alloc_constant_var(builder, F::from_u64(vals.len() as u64)));
    let coeffs_len = if vals.is_empty() { 0 } else { K_COEFFS_LEN };
    dst.push(alloc_constant_var(builder, F::from_u64(coeffs_len)));
    for k in vals {
        dst.push(k.c0);
        dst.push(k.c1);
    }
}
