//! Wire-format `NifsProof` — the three sub-proofs F' re-verifies.
//!
//! Carries no IVC framing (no `x_out`, no chain digests) — those belong
//! to `paper::construction2::StepProof`. NIFS at this layer is a pure
//! folding step.

use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Wire-format NIFS proof: the three sub-proofs `F'` will re-verify
/// in-circuit.
#[derive(Clone, Debug)]
pub struct NifsProof {
    pub pi_ccs: pi_ccs::Proof,
    pub pi_rlc: pi_rlc::Proof,
    pub pi_dec: pi_dec::Proof,
}
