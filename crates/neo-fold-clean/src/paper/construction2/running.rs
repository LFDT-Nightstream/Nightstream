//! `RunningInstance` — the running accumulator U_i (with prover-only
//! witness matrices W_i).
//!
//! Verifier-side reconstructions hold `witnesses = vec![]`; only the
//! prover threads the actual Z matrices. After step 1, `claims.len()`
//! equals `pp.k_rho()`.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::paper::params::Params;
use crate::paper::relations::{CeClaim, Structure, WitnessMat};

#[derive(Debug, Error)]
pub enum RunningInstanceError {
    #[error("canonical CE accumulator public-input length {m_in} exceeds structure.m {structure_m}")]
    PublicInputTooLarge { m_in: usize, structure_m: usize },
}

/// Running accumulator: verifier-visible CE claims plus the prover-only
/// witness matrices that justify them.
///
/// `claims` is the exact ordered Construction-2 accumulator. The legacy-named
/// `parent_authority` field is the Π_RLC recomposition cache whose Π_DEC
/// decomposition produced those claims. The next transcript binds both: the
/// exact child handle and the independently checked parent-cache digest.
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

    /// HyperNova-compatible default for SuperNeo's accumulator relation
    /// `R1 = CE(b, L)^k`.
    ///
    /// This is not an empty vector. It contains exactly `k = pp.k_rho()`
    /// zero CE claims and zero witness matrices. `parent_authority` is the
    /// deterministic radix-`b` recomposition of those children; it is retained
    /// only because the optimized Π_CCS transcript consumes that derived
    /// cache. The formal accumulator instance is `claims` alone.
    pub fn canonical_zero(pp: &Params, structure: &Structure, m_in: usize) -> Result<Self, RunningInstanceError> {
        Self::canonical_zero_for_shape(pp, structure.n, structure.m, structure.t(), m_in)
    }

    pub(crate) fn canonical_zero_for_shape(
        pp: &Params,
        relation_n: usize,
        relation_m: usize,
        relation_t: usize,
        m_in: usize,
    ) -> Result<Self, RunningInstanceError> {
        if m_in > relation_m {
            return Err(RunningInstanceError::PublicInputTooLarge {
                m_in,
                structure_m: relation_m,
            });
        }
        let ell_n = relation_n.next_power_of_two().max(2).trailing_zeros() as usize;
        let ell_m = relation_m.next_power_of_two().max(2).trailing_zeros() as usize;
        let d_pad = D.next_power_of_two();
        let zero_claim = CeClaim {
            c: Commitment::zeros(D, pp.kappa() as usize),
            X: Mat::zero(D, m_in, F::ZERO),
            r: vec![K::ZERO; ell_n],
            s_col: vec![K::ZERO; ell_m],
            y_ring: vec![vec![K::ZERO; d_pad]; relation_t],
            ct: vec![K::ZERO; relation_t],
            aux_openings: Vec::new(),
            y_zcol: vec![K::ZERO; d_pad],
            m_in,
            fold_digest: [0u8; 32],
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
            adv: None,
        };
        let zero_witness = Mat::zero(D, relation_m.div_ceil(D), F::ZERO);
        Ok(Self {
            claims: vec![zero_claim.clone(); pp.k_rho() as usize],
            witnesses: vec![zero_witness; pp.k_rho() as usize],
            parent_authority: Some(zero_claim),
        })
    }

    /// Formal `R1` instance. The parent cache is deliberately excluded.
    pub fn formal_claims(&self) -> &[CeClaim] {
        &self.claims
    }

    /// Deterministic Π_DEC recomposition cache used by the optimized NIFS.
    pub fn decomposition_parent(&self) -> Option<&CeClaim> {
        self.parent_authority.as_ref()
    }
}
