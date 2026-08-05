//! `RunningInstance` — the running accumulator U_i (with prover-only
//! witness matrices W_i).
//!
//! Verifier-side reconstructions hold `witnesses = vec![]`; only the
//! prover threads the actual Z matrices. After step 1, `claims.len()`
//! equals `pp.k_rho()`.

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::paper::params::Params;
use crate::paper::relations::{CeClaim, Structure, WitnessMat};

/// Product-commitment shape of the verifier-selected accumulator relation.
///
/// Plain SuperNeo claims omit the Nebula sidecar. Nebula claims carry a full
/// three-commitment tuple, including for the canonical zero accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneCommitmentMode {
    Plain,
    Nebula,
}

impl LaneCommitmentMode {
    pub const fn from_nebula(enabled: bool) -> Self {
        if enabled {
            Self::Nebula
        } else {
            Self::Plain
        }
    }

    fn zero(self, pp: &Params) -> Option<LaneCommitments<Commitment>> {
        match self {
            Self::Plain => None,
            Self::Nebula => Some(zero_lane_commitments(pp)),
        }
    }
}

pub(crate) fn zero_lane_commitments(pp: &Params) -> LaneCommitments<Commitment> {
    let zero = Commitment::zeros(D, pp.kappa() as usize);
    LaneCommitments {
        ops: zero.clone(),
        is: zero.clone(),
        fs: zero,
    }
}

#[derive(Debug, Error)]
pub enum RunningInstanceError {
    #[error("canonical CE accumulator public-input length {m_in} exceeds structure.m {structure_m}")]
    PublicInputTooLarge { m_in: usize, structure_m: usize },
    #[error("nonempty running accumulator is missing its Pi_RLC parent authority")]
    MissingParentAuthority,
    #[error("empty running accumulator unexpectedly carries a Pi_RLC parent authority")]
    UnexpectedParentAuthority,
}

/// Running accumulator: verifier-visible CE claims plus the prover-only
/// witness matrices that justify them.
///
/// `claims` is the exact ordered Construction-2 accumulator. The legacy-named
/// `parent_authority` field is the Π_RLC recomposition cache whose Π_DEC
/// decomposition produced those claims. The next transcript binds both: the
/// exact child handle and the independently checked parent-cache digest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunningInstance {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<WitnessMat>,
    pub parent_authority: Option<CeClaim>,
}

impl RunningInstance {
    /// Construct an accumulator carrier from explicitly typed public and private
    /// components. This constructor assigns no authority;
    /// protocol entry points must still verify every supplied component.
    pub fn new(claims: Vec<CeClaim>, witnesses: Vec<WitnessMat>, parent_authority: Option<CeClaim>) -> Self {
        Self {
            claims,
            witnesses,
            parent_authority,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty() && self.witnesses.is_empty() && self.parent_authority.is_none()
    }

    /// Clone the verifier-visible part only — claims and parent authority, but
    /// no witnesses. For verifier-side consumers
    /// (NIFS.V replays, image digests) where cloning the witness `Mat`s would
    /// be pure waste.
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
    pub fn canonical_zero(
        pp: &Params,
        structure: &Structure,
        m_in: usize,
        lane_mode: LaneCommitmentMode,
    ) -> Result<Self, RunningInstanceError> {
        Self::canonical_zero_for_shape(pp, structure.n, structure.m, structure.t(), m_in, lane_mode)
    }

    pub(crate) fn canonical_zero_for_shape(
        pp: &Params,
        relation_n: usize,
        relation_m: usize,
        relation_t: usize,
        m_in: usize,
        lane_mode: LaneCommitmentMode,
    ) -> Result<Self, RunningInstanceError> {
        if m_in > relation_m {
            return Err(RunningInstanceError::PublicInputTooLarge {
                m_in,
                structure_m: relation_m,
            });
        }
        let assignment_width = neo_reductions::common::superneo_carrier_width(relation_m);
        let ell_n = relation_n
            .max(assignment_width)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;
        let d_pad = D.next_power_of_two();
        let zero_claim = CeClaim {
            c: Commitment::zeros(D, pp.kappa() as usize),
            X: Mat::virtual_constant(D, m_in, F::ZERO),
            r: vec![K::ZERO; ell_n],
            y_ring: vec![vec![K::ZERO; d_pad]; relation_t + 1],
            ct: vec![K::ZERO; relation_t + 1],
            m_in,
            fold_digest: [0u8; 32],
            adv: lane_mode.zero(pp),
        };
        let zero_witness = Mat::virtual_constant(D, relation_m.div_ceil(D), F::ZERO);
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

    /// Canonical content handle for this running accumulator under the
    /// verifier-selected relation profile.
    ///
    /// The Pi_RLC parent remains a separately checked cache; its presence is
    /// validated here but it is not substituted for the exact child family.
    pub(crate) fn accumulator_digest(&self, structure: &Structure) -> Result<[u8; 32], RunningInstanceError> {
        self.accumulator_digest_for_relation_shape(structure.n, structure.m, structure.t())
    }

    /// Canonical accumulator digest from the verifier-owned relation shape.
    pub(crate) fn accumulator_digest_for_relation_shape(
        &self,
        _relation_rows: usize,
        _relation_columns: usize,
        _matrices: usize,
    ) -> Result<[u8; 32], RunningInstanceError> {
        if self.claims.is_empty() {
            if self.parent_authority.is_some() {
                return Err(RunningInstanceError::UnexpectedParentAuthority);
            }
            return Ok(crate::paper::digest::AccumulatorHandle::empty().digest());
        }
        if self.parent_authority.is_none() {
            return Err(RunningInstanceError::MissingParentAuthority);
        }

        Ok(
            crate::paper::digest::AccumulatorHandle::from_running_parts(&self.claims, self.parent_authority.as_ref())
                .digest(),
        )
    }
}
