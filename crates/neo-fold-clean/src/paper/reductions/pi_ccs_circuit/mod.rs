//! Recursive verifier for the selected one-joint PiCCS protocol.
//!
//! Owns: the `PaddedRowIdentity` verifier, claim digests, and the PiCCS
//! output message. The verifier uses one zero-padded row cube and one
//! SumCheck.
//!
//! Does not own: native proving, PiRLC verification, or matrix evaluation.
//!
//! Emits constraints: canonical constants and the constraints delegated to
//! the verifier, digest, and output-message modules.
//!
//! | Surface | Constraint owner |
//! | --- | --- |
//! | `enforce_pi_ccs*` | one-joint verifier |
//! | digest helpers | Poseidon2 claim binding |
//! | output helpers | canonical output binding |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::R1csBuilder;

pub mod digests;
pub mod output_message;
pub mod stage;
pub mod verifier;

// Re-export the public surface so external paths (tests, NIFS.V) keep using
// `paper::reductions::pi_ccs_circuit::*` without touching the
// internal module split.
pub use digests::{
    enforce_accumulator_ce_claim_digest, enforce_accumulator_claims_digest, enforce_ccs_claim_digest,
    enforce_pi_ccs_instance_digest_parent_authority, enforce_pi_ccs_outputs_digest,
    enforce_strict_binary_accumulator_family_digest, AccumulatorCeClaimDigestInputs, PiCcsOutputsDigestWires,
};
pub use output_message::{
    audit_pi_ccs_output_sis, encode_pi_ccs_outputs_preimage, PiCcsOutputFieldBinding, PiCcsOutputMessageDigestInputs,
    PiCcsOutputSisAudit, PiCcsOutputSisOwnerAudit, PiCcsOutputsPreimage,
};
pub use verifier::{
    enforce_pi_ccs, enforce_pi_ccs_with_matrix_digest_wires, PiCcsOutputWires, PiCcsVerifierConfig,
    PiCcsVerifierMessages, PiCcsVerifierRelation, PiCcsVerifierResult,
};

/// Shape errors from the selected PiCCS circuit.
#[derive(Debug, Error)]
pub enum Error {
    #[error("PaddedRowIdentity circuit shape mismatch: {0}")]
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
