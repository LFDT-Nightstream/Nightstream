//! Per-execution export for the selected compact PiCCS verifier path.
//!
//! Owns the canonical statement snapshot, raw SumCheck proof bytes, and full
//! source/matrix/coefficient output carried to an independent checker. It does
//! not claim universal correctness of the Rust prover or verifier.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField64;

use crate::engines::pi_ccs_joint::{build_joint_dims, TraceEvent};
use crate::engines::pi_ccs_joint_protocol::TranscriptBinding;
use crate::engines::pi_ccs_protocol::PiCcsProof;
use crate::error::PiCcsError;

use super::optimized_engine::OptimizedStructureCache;

/// Canonical low/high Goldilocks representatives of one quadratic-extension
/// value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsReceiptK {
    pub low: u64,
    pub high: u64,
}

impl From<K> for PiCcsReceiptK {
    fn from(value: K) -> Self {
        let (low, high) = value.to_limbs_u64();
        Self { low, high }
    }
}

/// Verifier-owned data that fixes one PiCCS execution before proof messages
/// and output values are checked.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsCanonicalStatement {
    /// Poseidon2 matrix-artifact identifier. An independent checker must
    /// compare this value with its verifier-owned expected identifier.
    pub relation_id: [u64; 4],
    /// Exact width-8 transcript state before the PiCCS public input.
    pub transcript_state: [u64; 8],
    pub transcript_absorbed: usize,
    /// Every v1_1 transcript absorption in exact execution order. Framed
    /// blocks include their length word.
    pub transcript_absorptions: Vec<Vec<u64>>,
    /// Shared running-claim point in coordinate order.
    pub prior_point: Vec<PiCcsReceiptK>,
    /// Running `Eval_K` coefficients in coefficient-major, running-major
    /// `I_K` order.
    pub claimed_eval_k: Vec<PiCcsReceiptK>,
    /// Running `Eval_A` coefficients in coefficient-major, matrix-major,
    /// running-major `I_A` order.
    pub claimed_eval_a: Vec<PiCcsReceiptK>,
}

/// Prover messages checked against one canonical statement.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsExecutionProof {
    /// Exact versioned bytes returned by `PiCcsProof::canonical_bytes`.
    pub proof_bytes: Vec<u8>,
    /// Complete paper `Eval_K` output in source-major, coefficient-major order.
    pub output_eval_k: Vec<PiCcsReceiptK>,
    /// Complete paper `Eval_A` output in source-major, matrix-major,
    /// coefficient-major order.
    pub output_eval_a: Vec<PiCcsReceiptK>,
}

/// A statement and proof captured from one accepting production verifier
/// call.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsExecutionReceipt {
    pub statement: PiCcsCanonicalStatement,
    pub proof: PiCcsExecutionProof,
}

fn canonical_fields(fields: &[F]) -> Vec<u64> {
    fields.iter().map(PrimeField64::as_canonical_u64).collect()
}

fn transcript_absorptions(trace: &[TraceEvent]) -> Result<Vec<Vec<u64>>, PiCcsError> {
    let absorptions: Vec<_> = trace
        .iter()
        .filter_map(|event| match event {
            TraceEvent::Absorb(fields) => Some(canonical_fields(fields)),
            TraceEvent::Challenge { .. } => None,
        })
        .collect();
    if absorptions.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "Pi_CCS receipt trace has no v1_1 absorptions".into(),
        ));
    }
    Ok(absorptions)
}

fn claimed_eval_k(running: &[CeClaim<Cmt, F, K>]) -> Result<Vec<PiCcsReceiptK>, PiCcsError> {
    let mut values = Vec::with_capacity(running.len() * D);
    for coefficient in 0..D {
        for claim in running {
            values.push(
                claim
                    .eval_k
                    .get(coefficient)
                    .copied()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS receipt Eval_K is incomplete".into()))?
                    .into(),
            );
        }
    }
    Ok(values)
}

fn claimed_eval_a(running: &[CeClaim<Cmt, F, K>], matrix_count: usize) -> Result<Vec<PiCcsReceiptK>, PiCcsError> {
    let mut values = Vec::with_capacity(running.len() * matrix_count * D);
    for coefficient in 0..D {
        for matrix in 0..matrix_count {
            for claim in running {
                let value = claim
                    .eval_a
                    .get(matrix)
                    .and_then(|row| row.get(coefficient))
                    .copied()
                    .ok_or_else(|| {
                        PiCcsError::InvalidInput("Pi_CCS receipt running coefficient family is incomplete".into())
                    })?;
                values.push(value.into());
            }
        }
    }
    Ok(values)
}

fn output_eval_k(outputs: &[CeClaim<Cmt, F, K>]) -> Result<Vec<PiCcsReceiptK>, PiCcsError> {
    let mut values = Vec::with_capacity(outputs.len() * D);
    for output in outputs {
        for coefficient in 0..D {
            values.push(
                output
                    .eval_k
                    .get(coefficient)
                    .copied()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS receipt output Eval_K is incomplete".into()))?
                    .into(),
            );
        }
    }
    Ok(values)
}

fn output_eval_a(outputs: &[CeClaim<Cmt, F, K>], matrix_count: usize) -> Result<Vec<PiCcsReceiptK>, PiCcsError> {
    let mut values = Vec::with_capacity(outputs.len() * matrix_count * D);
    for output in outputs {
        for matrix in 0..matrix_count {
            for coefficient in 0..D {
                let value = output
                    .eval_a
                    .get(matrix)
                    .and_then(|row| row.get(coefficient))
                    .copied()
                    .ok_or_else(|| {
                        PiCcsError::InvalidInput("Pi_CCS receipt output coefficient family is incomplete".into())
                    })?;
                values.push(value.into());
            }
        }
    }
    Ok(values)
}

/// Run the compact production verifier and export a receipt only when it
/// accepts. This call advances `transcript` exactly as the normal verifier.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_export_pi_ccs_receipt(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsExecutionReceipt, PiCcsError> {
    cache.validate_structure(structure)?;
    let transcript_state = transcript.state().map(|value| value.as_canonical_u64());
    let transcript_absorbed = transcript.absorbed();
    let relation_id = (*cache.matrix_digest()).map(|value| value.as_canonical_u64());
    let binding = TranscriptBinding::digest_only();
    let (accepted, trace) = crate::engines::pi_ccs_joint_protocol::verify_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        binding,
        Some(cache.matrix_digest()),
    )?;
    if !accepted {
        return Err(PiCcsError::ProtocolError(
            "Pi_CCS verifier rejected; no execution receipt was issued".into(),
        ));
    }

    let dims = build_joint_dims(params, structure, fresh_claims.len(), running_claims.len())?;
    let transcript_absorptions = transcript_absorptions(&trace.events)?;
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.variables)?
        .unwrap_or(&[])
        .iter()
        .copied()
        .map(Into::into)
        .collect();

    Ok(PiCcsExecutionReceipt {
        statement: PiCcsCanonicalStatement {
            relation_id,
            transcript_state,
            transcript_absorbed,
            transcript_absorptions,
            prior_point,
            claimed_eval_k: claimed_eval_k(running_claims)?,
            claimed_eval_a: claimed_eval_a(running_claims, dims.matrix_count)?,
        },
        proof: PiCcsExecutionProof {
            proof_bytes: proof.canonical_bytes(),
            output_eval_k: output_eval_k(outputs)?,
            output_eval_a: output_eval_a(outputs, dims.matrix_count)?,
        },
    })
}
