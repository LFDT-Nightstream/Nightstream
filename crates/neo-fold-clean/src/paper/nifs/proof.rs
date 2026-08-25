//! Wire-format `NifsProof` — the three sub-proofs F' re-verifies.
//!
//! Carries no IVC framing (no `x_out`, no chain digests) — those belong
//! to `paper::construction2::StepProof`. NIFS at this layer is a pure
//! folding step.

use neo_ajtai::Commitment;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeField64;

use crate::paper::relations::CeClaim;
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Wire-format NIFS proof: the three sub-proofs `F'` will re-verify
/// in-circuit.
#[derive(Clone, Debug, PartialEq)]
pub struct NifsProof {
    pub pi_ccs: pi_ccs::Proof,
    pub pi_rlc: pi_rlc::Proof,
    pub pi_dec: pi_dec::Proof,
}

impl NifsProof {
    /// Versioned exact encoding used by backend crosschecks and golden tests.
    ///
    /// This is not a digest. It contains every verifier-visible NIFS proof
    /// field and therefore detects a backend that emits different bytes for
    /// the same call.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(b"NS-NIFS-PROOF");
        push_u64(&mut output, 1);

        let pi_ccs = self.pi_ccs.sumcheck.canonical_bytes();
        push_bytes(&mut output, &pi_ccs);
        push_claims(&mut output, &self.pi_ccs.outputs);
        push_claim(&mut output, &self.pi_rlc.combined);
        push_claims(&mut output, &self.pi_dec.children);
        output
    }
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(output: &mut Vec<u8>, value: &[u8]) {
    push_u64(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn push_f(output: &mut Vec<u8>, value: F) {
    push_u64(output, value.as_canonical_u64());
}

fn push_k(output: &mut Vec<u8>, value: K) {
    for coefficient in value.as_coeffs() {
        push_f(output, coefficient);
    }
}

fn push_commitment(output: &mut Vec<u8>, commitment: &Commitment) {
    push_u64(output, commitment.d as u64);
    push_u64(output, commitment.kappa as u64);
    push_u64(output, commitment.data.len() as u64);
    for &value in &commitment.data {
        push_f(output, value);
    }
}

fn push_claims(output: &mut Vec<u8>, claims: &[CeClaim]) {
    push_u64(output, claims.len() as u64);
    for claim in claims {
        push_claim(output, claim);
    }
}

fn push_claim(output: &mut Vec<u8>, claim: &CeClaim) {
    push_commitment(output, &claim.c);

    push_u64(output, claim.X.rows() as u64);
    push_u64(output, claim.X.cols() as u64);
    for row in 0..claim.X.rows() {
        for column in 0..claim.X.cols() {
            push_f(output, claim.X[(row, column)]);
        }
    }

    push_u64(output, claim.r.len() as u64);
    for &value in &claim.r {
        push_k(output, value);
    }

    push_u64(output, claim.eval_k.len() as u64);
    for &value in &claim.eval_k {
        push_k(output, value);
    }
    push_u64(output, claim.eval_a.len() as u64);
    for row in &claim.eval_a {
        push_u64(output, row.len() as u64);
        for &value in row {
            push_k(output, value);
        }
    }

    push_u64(output, claim.m_in as u64);
    output.extend_from_slice(&claim.fold_digest);

    match &claim.adv {
        None => output.push(0),
        Some(adv) => {
            output.push(1);
            push_commitment(output, &adv.ops);
            push_commitment(output, &adv.is);
            push_commitment(output, &adv.fs);
        }
    }
}
