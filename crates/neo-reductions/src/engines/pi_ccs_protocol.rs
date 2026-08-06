//! Neutral wire types for the `Pi_CCS` protocol.
//!
//! This module owns protocol messages and their canonical serialization. It
//! does not own either the reference or optimized computation.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::CeClaim;
use neo_math::D;
use neo_math::{KExtensions, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::error::PiCcsError;

/// Fiat--Shamir challenges used by a `Pi_CCS` proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Challenges {
    pub alpha: Vec<K>,
    pub gamma: K,
}

impl Challenges {
    /// The paper's one-joint challenge pair `(alpha, gamma)`.
    pub fn new(alpha: Vec<K>, gamma: K) -> Self {
        Self { alpha, gamma }
    }

    pub fn has_expected_dimension(&self, variables: usize) -> bool {
        self.alpha.len() == variables
    }
}

/// Reject a noncanonical public-input coefficient embedding.
pub fn validate_inactive_x_zero<Ff>(label: &str, claim: &CeClaim<Cmt, Ff, K>) -> Result<(), PiCcsError>
where
    Ff: Field,
{
    let active_columns = neo_ccs::superneo_public_x_cols(claim.m_in);
    if claim.X.rows() != D || claim.X.cols() != active_columns {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: X has shape {}x{}, expected {D}x{}",
            claim.X.rows(),
            claim.X.cols(),
            active_columns
        )));
    }
    Ok(())
}

/// Public proof message for `Pi_CCS`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsProof {
    pub sumcheck_rounds: Vec<Vec<K>>,
}

impl PiCcsProof {
    pub fn new(sumcheck_rounds: Vec<Vec<K>>) -> Self {
        Self { sumcheck_rounds }
    }

    /// Normalize field representatives before serialization.
    pub fn canonicalize(&mut self) {
        canonicalize_rounds(&mut self.sumcheck_rounds);
    }

    /// Canonical bytes used for exact engine cross-checks and transport.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut proof = self.clone();
        proof.canonicalize();
        encode_padded_row_identity(&proof)
    }
}

const PADDED_ROW_IDENTITY_PROOF_TAG: u64 = 1102;
const PADDED_ROW_IDENTITY_CODEC_VERSION: u64 = 1;

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_k(output: &mut Vec<u8>, value: K) {
    let (low, high) = value.to_limbs_u64();
    push_u64(output, low);
    push_u64(output, high);
}

/// Versioned PiCCS message codec. It encodes only the one-joint SumCheck
/// messages. Transcript-derived challenges and terminal values are not wire
/// authority.
fn encode_padded_row_identity(proof: &PiCcsProof) -> Vec<u8> {
    let coefficient_count = proof.sumcheck_rounds.first().map_or(0, Vec::len);
    let mut output = Vec::with_capacity(4 * 8 + proof.sumcheck_rounds.len() * coefficient_count * 16);
    push_u64(&mut output, PADDED_ROW_IDENTITY_PROOF_TAG);
    push_u64(&mut output, PADDED_ROW_IDENTITY_CODEC_VERSION);
    push_u64(&mut output, proof.sumcheck_rounds.len() as u64);
    push_u64(&mut output, coefficient_count as u64);
    for round in &proof.sumcheck_rounds {
        debug_assert_eq!(round.len(), coefficient_count);
        for &coefficient in round {
            push_k(&mut output, coefficient);
        }
    }
    output
}

fn canonicalize_rounds(rounds: &mut [Vec<K>]) {
    for round in rounds {
        canonicalize_vec(round);
    }
}

fn canonicalize_vec(values: &mut [K]) {
    for value in values {
        *value = canonical_k(*value);
    }
}

fn canonical_k(value: K) -> K {
    let (c0, c1) = value.to_limbs_u64();
    neo_math::from_complex(F::from_u64(c0), F::from_u64(c1))
}
