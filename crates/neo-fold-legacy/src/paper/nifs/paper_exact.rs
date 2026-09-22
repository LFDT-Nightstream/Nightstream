//! Independent direct NIFS composition used as the correctness reference.
//!
//! This module spells out PiCCS, PiRLC, and PiDEC in protocol order. It owns
//! no optimized cache or accelerator path.

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeField64;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::work::{chain_witness_refs, split_fresh_instances};
use crate::paper::nifs::{Error, NifsProof, NifsProverAdapter, NifsProverRequest, PaperExactNifsProver};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, CeClaim, DecMixer, LaneScheme, RlcMixer, Structure};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

#[allow(clippy::too_many_arguments)]
pub fn prove_paper_exact(
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    commitment: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    mix_commitments: RlcMixer,
    combine_commitments: DecMixer,
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    let (fresh_claims, fresh_witnesses) = split_fresh_instances(fresh);

    let pi_ccs = pi_ccs::prove_from_parts_paper_exact(
        transcript,
        params,
        structure,
        commitment,
        &fresh_claims,
        &fresh_witnesses,
        running,
    )?;

    let all_witnesses = chain_witness_refs(&fresh_witnesses, &running.witnesses);
    let (rlc_output, pi_rlc) = pi_rlc::prove_refs_paper_exact(
        transcript,
        params,
        structure,
        mix_commitments,
        &pi_ccs.outputs,
        &all_witnesses,
    )?;

    let (dec_output, pi_dec) = pi_dec::prove_paper_exact(
        params,
        structure,
        commitment,
        lanes,
        combine_commitments,
        &rlc_output.claim,
        &rlc_output.witness,
    )?;

    Ok((
        RunningInstance::new(dec_output.claims, dec_output.witnesses, Some(rlc_output.claim)),
        NifsProof { pi_ccs, pi_rlc, pi_dec },
    ))
}

impl NifsProverAdapter for PaperExactNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), Error> {
        prove_paper_exact(
            request.tr,
            request.pp,
            request.s,
            request.log,
            request.lanes,
            request.mix_rhos_commits,
            request.combine_b_pows,
            request.fresh,
            request.running,
        )
    }
}

/// Encode every verifier-visible NIFS field without using the production
/// NIFS or PiCCS codec.
pub(super) fn encode_proof(proof: &NifsProof) -> Result<Vec<u8>, neo_reductions::error::PiCcsError> {
    let mut output = Vec::new();
    output.extend_from_slice(b"NS-NIFS-PROOF");
    push_u64(&mut output, 1);

    let pi_ccs = neo_reductions::engines::paper_exact_engine::encode_proof(&proof.pi_ccs.sumcheck)?;
    push_bytes(&mut output, &pi_ccs);
    push_claims(&mut output, &proof.pi_ccs.outputs);
    push_claim(&mut output, &proof.pi_rlc.combined);
    push_claims(&mut output, &proof.pi_dec.children);
    Ok(output)
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
