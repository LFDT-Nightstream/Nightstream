//! Owns the Phase 2a same-object same-point collapse surface and its digests.
//!
//! It owns:
//! - canonical collapse from Phase 1 unified claims to reduced opening targets
//! - canonical reduced-claim, group, and result digests
//! - exact Phase 1 to Phase 2a coverage checks against canonical expected output
//!
//! It does not own:
//! - Phase 0 claim emission
//! - Phase 1 point unification
//! - any v2 cross-object accumulation

use std::collections::{BTreeMap, BTreeSet};

use neo_math::{from_complex, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::RawDataSerializable;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::opening_claim_reduction::ClaimReductionResult;
use super::opening_claim_reduction_error::ClaimReductionError;
use super::opening_eval_claims::{
    CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId, FamilyEvalPayload, FamilyEvalSchemaId,
    OpenedAjtaiObjectId, PackedColumnEval,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReducedEvalClaim {
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub point: Vec<K>,
    pub payload: FamilyEvalPayload,
    pub source_claim_ids: Vec<FamilyEvalClaimId>,
    pub reduction_digest: [u8; 32],
}

impl ReducedEvalClaim {
    pub fn validate(&self, index: usize) -> Result<(), Phase2CollapseError> {
        self.payload
            .validate()
            .map_err(|source| Phase2CollapseError::InvalidReducedClaimPayload { index, source })?;

        let expected_object_digest = self.opened_object.expected_digest(&self.commitment_context);
        if self.opened_object.digest != expected_object_digest {
            return Err(Phase2CollapseError::ReducedClaimObjectDigestMismatch {
                index,
                expected: expected_object_digest,
                actual: self.opened_object.digest,
            });
        }

        let expected_schema = FamilyEvalSchemaId::from_family(self.opened_object.family).ok_or(
            Phase2CollapseError::ReducedClaimUnsupportedFamily {
                index,
                family: self.opened_object.family,
            },
        )?;
        if self.payload.schema != expected_schema {
            return Err(Phase2CollapseError::ReducedClaimSchemaMismatch {
                index,
                expected: expected_schema,
                actual: self.payload.schema,
            });
        }

        let expected_point_arity = self.opened_object.row_domain_log_size as usize;
        let actual_point_arity = self.point.len();
        if actual_point_arity != expected_point_arity {
            return Err(Phase2CollapseError::ReducedClaimPointArityMismatch {
                index,
                expected: expected_point_arity,
                actual: actual_point_arity,
            });
        }

        if self.source_claim_ids.is_empty() {
            return Err(Phase2CollapseError::ReducedClaimSourceIdsEmpty { index });
        }
        if self
            .source_claim_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(Phase2CollapseError::ReducedClaimSourceIdsNotCanonical { index });
        }

        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim");
        tr.append_message(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/schema_tag",
            &[self.payload.schema.tag()],
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/point",
            &self.point,
        );
        append_payload(
            &mut tr,
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/payload",
            &self.payload,
        );
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/source_claim_count",
            &[self.source_claim_ids.len() as u64],
        );
        for claim_id in &self.source_claim_ids {
            append_claim_id(
                &mut tr,
                b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/source_claim_id",
                claim_id,
            );
        }
        tr.append_message(
            b"neo.fold.next/rv32im/opening_convergence/phase2/reduced_claim/reduction_digest",
            &self.reduction_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Phase2CollapseRecord {
    pub group_digest: [u8; 32],
    pub rho_2: K,
    pub reduced_claim_digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Phase2CollapseResult {
    pub reduced_claims: Vec<ReducedEvalClaim>,
    pub records: Vec<Phase2CollapseRecord>,
    pub digest: [u8; 32],
}

impl Phase2CollapseResult {
    pub fn validate(&self) -> Result<(), Phase2CollapseError> {
        let expected_record_count = self.reduced_claims.len();
        let actual_record_count = self.records.len();
        if actual_record_count != expected_record_count {
            return Err(Phase2CollapseError::RecordCountMismatch {
                expected: expected_record_count,
                actual: actual_record_count,
            });
        }

        let mut seen_claim_ids = BTreeSet::new();
        for (index, (reduced_claim, record)) in self.reduced_claims.iter().zip(&self.records).enumerate() {
            reduced_claim.validate(index)?;
            for claim_id in &reduced_claim.source_claim_ids {
                if !seen_claim_ids.insert(*claim_id) {
                    return Err(Phase2CollapseError::DuplicateSourceClaimId { claim_id: *claim_id });
                }
            }

            let expected_digest = reduced_claim.expected_digest();
            if record.reduced_claim_digest != expected_digest {
                return Err(Phase2CollapseError::ReducedClaimDigestMismatch {
                    index,
                    expected: expected_digest,
                    actual: record.reduced_claim_digest,
                });
            }
        }

        let expected_digest = self.expected_digest();
        if self.digest != expected_digest {
            return Err(Phase2CollapseError::ResultDigestMismatch {
                expected: expected_digest,
                actual: self.digest,
            });
        }

        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_convergence/phase2/result");
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_convergence/phase2/result/record_count",
            &[self.records.len() as u64],
        );
        for record in &self.records {
            tr.append_message(
                b"neo.fold.next/rv32im/opening_convergence/phase2/result/group_digest",
                &record.group_digest,
            );
            append_k(
                &mut tr,
                b"neo.fold.next/rv32im/opening_convergence/phase2/result/rho_2",
                &record.rho_2,
            );
            tr.append_message(
                b"neo.fold.next/rv32im/opening_convergence/phase2/result/reduced_claim_digest",
                &record.reduced_claim_digest,
            );
        }
        tr.digest32()
    }
}

pub fn build_phase2_collapse_result(
    phase1_results: &[ClaimReductionResult],
) -> Result<Phase2CollapseResult, Phase2CollapseError> {
    let ordered_results = canonical_phase1_results(phase1_results)?;
    let result = build_phase2_collapse_result_from_trusted_phase1_results(ordered_results)?;
    result.validate()?;
    Ok(result)
}

pub(crate) fn build_phase2_collapse_result_trusted_local(
    phase1_results: &[ClaimReductionResult],
) -> Result<Phase2CollapseResult, Phase2CollapseError> {
    let ordered_results = canonical_phase1_results_trusted_local(phase1_results)?;
    build_phase2_collapse_result_from_trusted_phase1_results(ordered_results)
}

fn build_phase2_collapse_result_from_trusted_phase1_results(
    ordered_results: Vec<&ClaimReductionResult>,
) -> Result<Phase2CollapseResult, Phase2CollapseError> {
    let mut reduced_claims = Vec::new();
    let mut records = Vec::new();

    for reduction_result in ordered_results {
        let reduction_digest = reduction_result.trusted_local_digest();
        let mut groups = BTreeMap::<[u8; 32], Vec<&FamilyEvalClaim>>::new();
        for unified_claim in &reduction_result.unified_claims {
            groups
                .entry(unified_claim.opened_object.digest)
                .or_default()
                .push(unified_claim);
        }

        for claims in groups.values() {
            let reduced_claim = build_reduced_claim(claims, reduction_digest)?;
            let group_digest = expected_group_digest(claims, reduction_digest);
            let rho_2 = sample_phase2_rho(group_digest);
            let record = Phase2CollapseRecord {
                group_digest,
                rho_2,
                reduced_claim_digest: reduced_claim.expected_digest(),
            };
            reduced_claims.push(reduced_claim);
            records.push(record);
        }
    }

    let mut result = Phase2CollapseResult {
        reduced_claims,
        records,
        digest: [0; 32],
    };
    result.digest = result.expected_digest();
    Ok(result)
}

pub fn verify_phase2_collapse_result(
    result: &Phase2CollapseResult,
    phase1_results: &[ClaimReductionResult],
) -> Result<(), Phase2CollapseError> {
    result.validate()?;
    let expected = build_phase2_collapse_result(phase1_results)?;

    let expected_reduced_claim_count = expected.reduced_claims.len();
    let actual_reduced_claim_count = result.reduced_claims.len();
    if actual_reduced_claim_count != expected_reduced_claim_count {
        return Err(Phase2CollapseError::ReducedClaimCountMismatch {
            expected: expected_reduced_claim_count,
            actual: actual_reduced_claim_count,
        });
    }

    let expected_record_count = expected.records.len();
    let actual_record_count = result.records.len();
    if actual_record_count != expected_record_count {
        return Err(Phase2CollapseError::RecordCountMismatch {
            expected: expected_record_count,
            actual: actual_record_count,
        });
    }

    for (index, (actual, expected_claim)) in result
        .reduced_claims
        .iter()
        .zip(expected.reduced_claims.iter())
        .enumerate()
    {
        if actual != expected_claim {
            return Err(Phase2CollapseError::UnexpectedReducedClaimAtIndex { index });
        }
    }

    for (index, (actual, expected_record)) in result
        .records
        .iter()
        .zip(expected.records.iter())
        .enumerate()
    {
        if actual != expected_record {
            return Err(Phase2CollapseError::UnexpectedRecordAtIndex { index });
        }
    }

    if result.digest != expected.digest {
        return Err(Phase2CollapseError::ResultDigestMismatch {
            expected: expected.digest,
            actual: result.digest,
        });
    }

    Ok(())
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Phase2CollapseError {
    #[error("phase2 received an invalid phase1 result at index {index}: {source}")]
    InvalidPhase1Result {
        index: usize,
        source: ClaimReductionError,
    },
    #[error("phase2 received duplicate phase1 schema {schema:?}")]
    DuplicatePhase1Schema { schema: FamilyEvalSchemaId },
    #[error("phase2 reduced claim {index} has invalid payload: {source}")]
    InvalidReducedClaimPayload {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase2 reduced claim {index} opened-object digest mismatch: expected {expected:?}, got {actual:?}")]
    ReducedClaimObjectDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase2 reduced claim {index} uses unsupported family {family:?}")]
    ReducedClaimUnsupportedFamily {
        index: usize,
        family: super::canonical_openings::AjtaiFamilyKind,
    },
    #[error("phase2 reduced claim {index} schema mismatch: expected {expected:?}, got {actual:?}")]
    ReducedClaimSchemaMismatch {
        index: usize,
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
    },
    #[error("phase2 reduced claim {index} point arity mismatch: expected {expected}, got {actual}")]
    ReducedClaimPointArityMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("phase2 reduced claim {index} has empty source_claim_ids")]
    ReducedClaimSourceIdsEmpty { index: usize },
    #[error("phase2 reduced claim {index} source_claim_ids are not in canonical order")]
    ReducedClaimSourceIdsNotCanonical { index: usize },
    #[error("phase2 result reduced-claim count mismatch: expected {expected}, got {actual}")]
    ReducedClaimCountMismatch { expected: usize, actual: usize },
    #[error("phase2 result record count mismatch: expected {expected}, got {actual}")]
    RecordCountMismatch { expected: usize, actual: usize },
    #[error("phase2 record {index} reduced-claim digest mismatch: expected {expected:?}, got {actual:?}")]
    ReducedClaimDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase2 result digest mismatch: expected {expected:?}, got {actual:?}")]
    ResultDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase2 result carries duplicate source claim id {claim_id:?}")]
    DuplicateSourceClaimId { claim_id: FamilyEvalClaimId },
    #[error("phase2 result carries an unexpected reduced claim at index {index}")]
    UnexpectedReducedClaimAtIndex { index: usize },
    #[error("phase2 result carries an unexpected record at index {index}")]
    UnexpectedRecordAtIndex { index: usize },
    #[error("phase2 group changes opened object at member {index}")]
    GroupOpenedObjectMismatch { index: usize },
    #[error("phase2 group changes commitment context at member {index}")]
    GroupCommitmentContextMismatch { index: usize },
    #[error("phase2 group changes schema at member {index}: expected {expected:?}, got {actual:?}")]
    GroupSchemaMismatch {
        index: usize,
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
    },
    #[error("phase2 group changes unified point at member {index}")]
    GroupPointMismatch { index: usize },
    #[error("phase2 group changes payload at member {index}")]
    GroupPayloadMismatch { index: usize },
}

fn canonical_phase1_results<'a>(
    phase1_results: &'a [ClaimReductionResult],
) -> Result<Vec<&'a ClaimReductionResult>, Phase2CollapseError> {
    let mut by_schema = BTreeMap::<FamilyEvalSchemaId, &ClaimReductionResult>::new();
    for (index, result) in phase1_results.iter().enumerate() {
        result
            .validate()
            .map_err(|source| Phase2CollapseError::InvalidPhase1Result { index, source })?;
        let schema = result.bucket.schema;
        if by_schema.insert(schema, result).is_some() {
            return Err(Phase2CollapseError::DuplicatePhase1Schema { schema });
        }
    }
    Ok(by_schema.into_values().collect())
}

fn canonical_phase1_results_trusted_local<'a>(
    phase1_results: &'a [ClaimReductionResult],
) -> Result<Vec<&'a ClaimReductionResult>, Phase2CollapseError> {
    let mut by_schema = BTreeMap::<FamilyEvalSchemaId, &ClaimReductionResult>::new();
    for result in phase1_results {
        let schema = result.bucket.schema;
        if by_schema.insert(schema, result).is_some() {
            return Err(Phase2CollapseError::DuplicatePhase1Schema { schema });
        }
    }
    Ok(by_schema.into_values().collect())
}

fn build_reduced_claim(
    claims: &[&FamilyEvalClaim],
    reduction_digest: [u8; 32],
) -> Result<ReducedEvalClaim, Phase2CollapseError> {
    let first = claims[0];
    for (index, claim) in claims.iter().enumerate().skip(1) {
        if claim.opened_object != first.opened_object {
            return Err(Phase2CollapseError::GroupOpenedObjectMismatch { index });
        }
        if claim.commitment_context != first.commitment_context {
            return Err(Phase2CollapseError::GroupCommitmentContextMismatch { index });
        }
        if claim.payload.schema != first.payload.schema {
            return Err(Phase2CollapseError::GroupSchemaMismatch {
                index,
                expected: first.payload.schema,
                actual: claim.payload.schema,
            });
        }
        if claim.point != first.point {
            return Err(Phase2CollapseError::GroupPointMismatch { index });
        }
        if claim.payload.column_evals != first.payload.column_evals {
            return Err(Phase2CollapseError::GroupPayloadMismatch { index });
        }
    }

    let reduced_claim = ReducedEvalClaim {
        opened_object: first.opened_object.clone(),
        commitment_context: first.commitment_context,
        point: first.point.clone(),
        payload: first.payload.clone(),
        source_claim_ids: claims.iter().map(|claim| claim.id).collect(),
        reduction_digest,
    };
    reduced_claim.validate(0)?;
    Ok(reduced_claim)
}

fn expected_group_digest(claims: &[&FamilyEvalClaim], reduction_digest: [u8; 32]) -> [u8; 32] {
    let first = claims[0];
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_convergence/phase2/group");
    tr.append_message(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/opened_object_digest",
        &first.opened_object.digest,
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/schema_tag",
        &[first.payload.schema.tag()],
    );
    tr.append_message(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/pp_seed_digest",
        &first.commitment_context.pp_seed_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/module_shape_digest",
        &first.commitment_context.module_shape_digest,
    );
    append_k_vec(
        &mut tr,
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/point",
        &first.point,
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/claim_count",
        &[claims.len() as u64],
    );
    for claim in claims {
        append_claim_id(
            &mut tr,
            b"neo.fold.next/rv32im/opening_convergence/phase2/group/claim_id",
            &claim.id,
        );
    }
    append_payload(
        &mut tr,
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/payload",
        &first.payload,
    );
    tr.append_message(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group/reduction_digest",
        &reduction_digest,
    );
    tr.digest32()
}

fn sample_phase2_rho(group_digest: [u8; 32]) -> K {
    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_convergence/phase2");
    transcript.append_message(
        b"neo.fold.next/rv32im/opening_convergence/phase2/group_digest",
        &group_digest,
    );
    let challenge = transcript.challenge_fields(b"neo.fold.next/rv32im/opening_convergence/phase2/rho_2", 2);
    from_complex(challenge[0], challenge[1])
}

fn append_payload(tr: &mut Poseidon2Transcript, label: &'static [u8], payload: &FamilyEvalPayload) {
    tr.append_u64s(label, &[payload.schema.tag(), payload.column_evals.len() as u64]);
    for column_eval in &payload.column_evals {
        append_packed_column_eval(tr, label, column_eval);
    }
}

fn append_packed_column_eval(tr: &mut Poseidon2Transcript, label: &'static [u8], value: &PackedColumnEval) {
    append_k_vec(tr, label, value.coeffs.as_slice());
}

fn append_k(tr: &mut Poseidon2Transcript, label: &'static [u8], value: &K) {
    let bytes = (*value).into_bytes().into_iter().collect::<Vec<_>>();
    tr.append_message(label, &bytes);
}

fn append_k_vec(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(label, &[values.len() as u64]);
    for value in values {
        append_k(tr, label, value);
    }
}

fn append_claim_id(tr: &mut Poseidon2Transcript, label: &'static [u8], id: &FamilyEvalClaimId) {
    tr.append_message(label, &id.opened_object_digest);
    tr.append_u64s(label, &[id.slot as u64]);
}
