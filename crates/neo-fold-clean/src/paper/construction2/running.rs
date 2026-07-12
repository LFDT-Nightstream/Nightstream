//! `RunningInstance` — the running accumulator U_i (with prover-only
//! witness matrices W_i).
//!
//! Verifier-side reconstructions hold `witnesses = vec![]`; only the
//! prover threads the actual Z matrices. After step 1, `claims.len()`
//! equals `pp.k_rho()`.

use crate::paper::relations::{CeClaim, WitnessMat};

/// Running accumulator: verifier-visible CE claims plus the prover-only
/// witness matrices that justify them.
///
/// `parent_authority` is the Π_RLC parent whose Π_DEC decomposition produced
/// `claims`. The next Π_CCS Fiat-Shamir transcript binds this parent, not the
/// child claims individually; the children are still the algebraic running
/// inputs and are checked against this parent by the prior Π_DEC verifier.
#[derive(Clone, Debug, Default)]
pub struct RunningInstance {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<WitnessMat>,
    pub parent_authority: Option<CeClaim>,
}

impl RunningInstance {
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty() && self.witnesses.is_empty() && self.parent_authority.is_none()
    }

    /// Clone the verifier-visible part only — claims and parent authority,
    /// no witnesses. For verifier-side consumers (NIFS.V replays, image
    /// digests) where cloning the witness `Mat`s would be pure waste.
    pub fn claims_only(&self) -> Self {
        Self {
            claims: self.claims.clone(),
            witnesses: Vec::new(),
            parent_authority: self.parent_authority.clone(),
        }
    }

    /// Sanity: each claim has a witness on the prover side; both empty on the verifier side.
    pub fn shape_ok(&self) -> bool {
        self.claims.len() == self.witnesses.len()
            && if self.claims.is_empty() {
                self.parent_authority.is_none()
            } else {
                self.parent_authority.is_some()
            }
    }
}
