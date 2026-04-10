//! Owns the terminal v1 opening-convergence bundle over Phase 2 reduced claims.
//!
//! It owns:
//! - the verifier-facing final opened-commitment carrier
//! - the concrete v1 Ajtai opening witness carrier used to close Phase 0 -> Phase 2 below export
//! - canonical final-opening-target and final-bundle digests
//! - self-contained verifier replay from carried final targets back to Phase 0/1/2
//! It does not own:
//! - the published Nightstream carried boundary
//! - any v2 cross-object accumulation
//! - a compact PCS proof system beyond the current witness-backed v1 closure surface

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use neo_math::{D, F};
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::opening_accumulate::{
    build_phase2_collapse_result_trusted_local, verify_phase2_collapse_result, Phase2CollapseError,
    Phase2CollapseResult, ReducedEvalClaim,
};
use super::opening_claim_reduction::{
    build_claim_reduction_results_from_bundle_and_witnesses_trusted_local,
    verify_claim_reduction_results_with_binding_surface, ClaimReductionError, ClaimReductionResult,
};
use super::opening_eval_claim_witness::{
    build_commitment_vector, phase0_commitment_root_digest, FamilyEvalClaimWitness, OpenedAjtaiObjectWitness,
    PackedColumnOracleRef, RealAjtaiCommitmentVector,
};
use super::opening_eval_claims::{
    phase0_family_order, CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId, FamilyEvalSchemaId,
    OpenedAjtaiObjectId, Rv64imEvalClaimBundle,
};
use super::opening_phase0_binding_surface::Rv64imPhase0BindingSurface;
use super::simple::SimpleKernelError;

const FINAL_OPENING_COUNT_V1: usize = 6;

pub type RealAjtaiCommitmentVectorPublic = RealAjtaiCommitmentVector;

impl Rv64imPhase0BindingSurface {
    pub fn validate_canonical_order(&self) -> Result<(), FinalOpeningError> {
        let expected_order = phase0_family_order();
        if self.targets.len() != expected_order.len() {
            return Err(FinalOpeningError::BindingSurfaceTargetCountMismatch {
                expected: expected_order.len(),
                actual: self.targets.len(),
            });
        }
        for (index, (target, expected_schema)) in self.targets.iter().zip(expected_order.iter()).enumerate() {
            if target.schema != *expected_schema {
                return Err(FinalOpeningError::BindingSurfaceSchemaMismatch {
                    index,
                    expected: *expected_schema,
                    actual: target.schema,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imOpeningConvergenceArtifactBuildPerf {
    pub phase1_results_ms: f64,
    pub phase2_ms: f64,
    pub final_openings_ms: f64,
    pub final_openings_witness_map_ms: f64,
    pub final_openings_representative_ms: f64,
    pub final_openings_commitment_validate_ms: f64,
    pub final_openings_opened_commitment_digest_ms: f64,
    pub final_openings_opening_proof_digest_ms: f64,
    pub final_openings_target_build_ms: f64,
    pub digest_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactFinalOpeningTargetsBuildPerf {
    witness_map_ms: f64,
    representative_ms: f64,
    commitment_validate_ms: f64,
    opened_commitment_digest_ms: f64,
    opening_proof_digest_ms: f64,
    target_build_ms: f64,
    total_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OpenedAjtaiCommitmentPublic {
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_vector: RealAjtaiCommitmentVectorPublic,
    pub digest: [u8; 32],
}

impl OpenedAjtaiCommitmentPublic {
    pub fn new(
        opened_object: OpenedAjtaiObjectId,
        commitment_context: &CommitmentContextId,
        commitment_vector: RealAjtaiCommitmentVectorPublic,
        expected_width: usize,
    ) -> Result<Self, FinalOpeningError> {
        let commitment_root_digest = phase0_commitment_root_digest(&commitment_vector);
        Self::new_with_commitment_root_digest(
            opened_object,
            commitment_context,
            commitment_vector,
            expected_width,
            commitment_root_digest,
        )
    }

    fn new_with_commitment_root_digest(
        opened_object: OpenedAjtaiObjectId,
        commitment_context: &CommitmentContextId,
        commitment_vector: RealAjtaiCommitmentVectorPublic,
        expected_width: usize,
        commitment_root_digest: [u8; 32],
    ) -> Result<Self, FinalOpeningError> {
        let mut commitment = Self {
            opened_object,
            commitment_vector,
            digest: [0; 32],
        };
        commitment.digest =
            opened_commitment_public_digest_from_root_digest(&commitment.opened_object, commitment_root_digest);
        commitment.validate_with_context(commitment_context, expected_width, 0, commitment_root_digest)?;
        Ok(commitment)
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        opened_commitment_public_digest_from_root_digest(
            &self.opened_object,
            phase0_commitment_root_digest(&self.commitment_vector),
        )
    }

    fn validate_with_context(
        &self,
        commitment_context: &CommitmentContextId,
        expected_width: usize,
        index: usize,
        commitment_root_digest: [u8; 32],
    ) -> Result<(), FinalOpeningError> {
        validate_opened_commitment_public_with_root_digest(
            &self.opened_object,
            commitment_context,
            &self.commitment_vector,
            commitment_root_digest,
            expected_width,
            index,
        )?;

        let expected_digest =
            opened_commitment_public_digest_from_root_digest(&self.opened_object, commitment_root_digest);
        if self.digest != expected_digest {
            return Err(FinalOpeningError::OpenedCommitmentDigestMismatch {
                index,
                expected: expected_digest,
                actual: self.digest,
            });
        }

        Ok(())
    }
}

fn opened_commitment_public_digest_from_root_digest(
    opened_object: &OpenedAjtaiObjectId,
    commitment_root_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/final/opened_commitment");
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/final/opened_commitment/opened_object_digest",
        &opened_object.digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/final/opened_commitment/commitment_root_digest",
        &commitment_root_digest,
    );
    tr.digest32()
}

fn validate_opened_commitment_public_with_root_digest(
    opened_object: &OpenedAjtaiObjectId,
    commitment_context: &CommitmentContextId,
    commitment_vector: &RealAjtaiCommitmentVectorPublic,
    commitment_root_digest: [u8; 32],
    expected_width: usize,
    index: usize,
) -> Result<(), FinalOpeningError> {
    let actual_width = commitment_vector.len();
    if actual_width != expected_width {
        return Err(FinalOpeningError::OpenedCommitmentVectorCountMismatch {
            index,
            expected: expected_width,
            actual: actual_width,
        });
    }

    let expected_object_digest = opened_object.expected_digest(commitment_context);
    if opened_object.digest != expected_object_digest {
        return Err(FinalOpeningError::OpenedCommitmentOpenedObjectDigestMismatch {
            index,
            expected: expected_object_digest,
            actual: opened_object.digest,
        });
    }

    if opened_object.commitment_root_digest != commitment_root_digest {
        return Err(FinalOpeningError::OpenedCommitmentRootMismatch {
            index,
            expected: commitment_root_digest,
            actual: opened_object.commitment_root_digest,
        });
    }

    Ok(())
}

impl AjtaiOpeningProof {
    pub fn new(packed_columns: Vec<PackedColumnOracleRef>) -> Self {
        let digest = opening_proof_digest_from_oracle_refs(&packed_columns);
        Self::new_with_digest(packed_columns, digest)
    }

    fn new_with_digest(packed_columns: Vec<PackedColumnOracleRef>, digest: [u8; 32]) -> Self {
        let proof = Self {
            packed_columns: packed_columns
                .into_iter()
                .map(AjtaiPackedColumnWitness::from_oracle_ref)
                .collect(),
            digest,
        };
        proof
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        opening_proof_digest_from_witness_rows(&self.packed_columns)
    }

    fn validate(
        &self,
        expected_width: usize,
        expected_row_domain_log_size: u32,
        index: usize,
    ) -> Result<(), FinalOpeningError> {
        let actual_width = self.packed_columns.len();
        if actual_width != expected_width {
            return Err(FinalOpeningError::OpeningProofPackedColumnCountMismatch {
                index,
                expected: expected_width,
                actual: actual_width,
            });
        }

        let expected_time_len = 1usize << (expected_row_domain_log_size as usize);
        for (column_index, column) in self.packed_columns.iter().enumerate() {
            if column.column_index as usize != column_index {
                return Err(FinalOpeningError::OpeningProofColumnIndexMismatch {
                    index,
                    expected: column_index,
                    actual: column.column_index as usize,
                });
            }
            let actual_time_len = column.rows.len();
            if actual_time_len != expected_time_len {
                return Err(FinalOpeningError::OpeningProofRowDomainLengthMismatch {
                    index,
                    column_index,
                    expected: expected_time_len,
                    actual: actual_time_len,
                });
            }
            for (row_index, row) in column.rows.iter().enumerate() {
                let actual_row_width = row.len();
                if actual_row_width != D {
                    return Err(FinalOpeningError::OpeningProofRowWidthMismatch {
                        index,
                        column_index,
                        row_index,
                        expected: D,
                        actual: actual_row_width,
                    });
                }
            }
        }

        let expected_digest = self.expected_digest();
        if self.digest != expected_digest {
            return Err(FinalOpeningError::OpeningProofDigestMismatch {
                index,
                expected: expected_digest,
                actual: self.digest,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AjtaiOpeningProof {
    packed_columns: Vec<AjtaiPackedColumnWitness>,
    pub digest: [u8; 32],
}

fn opening_proof_digest_from_witness_rows(packed_columns: &[AjtaiPackedColumnWitness]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/final/opening_proof");
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_count",
        &[packed_columns.len() as u64],
    );
    for column in packed_columns {
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_meta",
            &[column.column_index as u64, column.rows.len() as u64],
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_rows",
            column.rows.len().saturating_mul(D),
            column.rows.iter().flat_map(|row| row.iter().copied()),
        );
    }
    tr.digest32()
}

fn opening_proof_digest_from_oracle_refs(packed_columns: &[PackedColumnOracleRef]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/final/opening_proof");
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_count",
        &[packed_columns.len() as u64],
    );
    for column in packed_columns {
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_meta",
            &[column.column_index as u64, column.rows.len() as u64],
        );
        tr.append_fields(
            b"neo.fold.next/rv64im/opening_convergence/final/opening_proof/packed_column_rows",
            column.rows.as_flattened(),
        );
    }
    tr.digest32()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FinalOpeningTarget {
    pub reduced_claim: ReducedEvalClaim,
    pub opened_commitment: OpenedAjtaiCommitmentPublic,
    pub opening_proof: AjtaiOpeningProof,
}

impl FinalOpeningTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/final/target");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final/target/reduced_claim_digest",
            &self.reduced_claim.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final/target/opened_commitment_digest",
            &self.opened_commitment.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final/target/opening_proof_digest",
            &self.opening_proof.digest,
        );
        tr.digest32()
    }

    fn validate_and_rebuild_witness(&self, index: usize) -> Result<OpenedAjtaiObjectWitness, FinalOpeningError> {
        self.reduced_claim
            .validate(index)
            .map_err(|source| FinalOpeningError::InvalidReducedClaim { index, source })?;
        self.opening_proof.validate(
            self.reduced_claim.payload.column_evals.len(),
            self.reduced_claim.opened_object.row_domain_log_size,
            index,
        )?;
        let commitment_root_digest = phase0_commitment_root_digest(&self.opened_commitment.commitment_vector);
        self.opened_commitment.validate_with_context(
            &self.reduced_claim.commitment_context,
            self.reduced_claim.payload.column_evals.len(),
            index,
            commitment_root_digest,
        )?;

        if self.opened_commitment.opened_object != self.reduced_claim.opened_object {
            return Err(FinalOpeningError::FinalOpeningObjectMismatch { index });
        }

        let rebuilt_commitment_vector = build_commitment_vector(
            self.reduced_claim.payload.schema,
            &self.opening_proof.try_to_oracle_refs(index)?,
        )
        .map_err(|source| FinalOpeningError::FinalOpeningCommitmentVectorBuildFailed { index, source })?;
        if rebuilt_commitment_vector != self.opened_commitment.commitment_vector {
            return Err(FinalOpeningError::FinalOpeningCommitmentVectorMismatch { index });
        }

        let witness = OpenedAjtaiObjectWitness::new_with_commitment_root_digest(
            self.reduced_claim.opened_object.clone(),
            self.reduced_claim.commitment_context,
            self.opening_proof.try_to_oracle_refs(index)?,
            rebuilt_commitment_vector,
            commitment_root_digest,
        )
        .map_err(|source| FinalOpeningError::FinalOpeningWitnessBuildFailed { index, source })?;
        let expected_payload = witness
            .evaluate_payload(&self.reduced_claim.point)
            .map_err(|source| FinalOpeningError::FinalOpeningPayloadEvaluationFailed { index, source })?;
        if expected_payload != self.reduced_claim.payload.column_evals {
            return Err(FinalOpeningError::FinalOpeningPayloadMismatch { index });
        }

        Ok(witness)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProjectedFinalOpeningTarget {
    pub opened_commitment: OpenedAjtaiCommitmentPublic,
    pub opening_proof: AjtaiOpeningProof,
    pub digest: [u8; 32],
}

impl ProjectedFinalOpeningTarget {
    pub fn expected_digest(&self, reduced_claim: &ReducedEvalClaim) -> [u8; 32] {
        compact_final_opening_target_digest_from_reduced_claim_digest(
            reduced_claim.expected_digest(),
            self.opened_commitment.digest,
            self.opening_proof.digest,
        )
    }

    fn validate_and_rebuild_witness(
        &self,
        reduced_claim: &ReducedEvalClaim,
        index: usize,
    ) -> Result<OpenedAjtaiObjectWitness, FinalOpeningError> {
        reduced_claim
            .validate(index)
            .map_err(|source| FinalOpeningError::InvalidReducedClaim { index, source })?;

        let witness = rebuild_opened_object_witness_from_projection(
            reduced_claim.payload.schema,
            &reduced_claim.commitment_context,
            reduced_claim.payload.column_evals.len(),
            &self.opened_commitment,
            &self.opening_proof,
            index,
        )?;
        if self.opened_commitment.opened_object != reduced_claim.opened_object {
            return Err(FinalOpeningError::FinalOpeningObjectMismatch { index });
        }

        let expected_payload = witness
            .evaluate_payload(&reduced_claim.point)
            .map_err(|source| FinalOpeningError::FinalOpeningPayloadEvaluationFailed { index, source })?;
        if expected_payload != reduced_claim.payload.column_evals {
            return Err(FinalOpeningError::FinalOpeningPayloadMismatch { index });
        }

        let expected_digest = self.expected_digest(reduced_claim);
        if self.digest != expected_digest {
            return Err(FinalOpeningError::ProjectedFinalOpeningDigestMismatch {
                index,
                expected: expected_digest,
                actual: self.digest,
            });
        }

        Ok(witness)
    }
}

fn compact_final_opening_target_digest_from_reduced_claim_digest(
    reduced_claim_digest: [u8; 32],
    opened_commitment_digest: [u8; 32],
    opening_proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/compact_final_target");
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_final_target/reduced_claim_digest",
        &reduced_claim_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_final_target/opened_commitment_digest",
        &opened_commitment_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_final_target/opening_proof_digest",
        &opening_proof_digest,
    );
    tr.digest32()
}

pub(crate) fn rebuild_opened_object_witness_from_projection(
    schema: FamilyEvalSchemaId,
    commitment_context: &CommitmentContextId,
    expected_width: usize,
    opened_commitment: &OpenedAjtaiCommitmentPublic,
    opening_proof: &AjtaiOpeningProof,
    index: usize,
) -> Result<OpenedAjtaiObjectWitness, FinalOpeningError> {
    opening_proof.validate(
        expected_width,
        opened_commitment.opened_object.row_domain_log_size,
        index,
    )?;
    let commitment_root_digest = phase0_commitment_root_digest(&opened_commitment.commitment_vector);
    opened_commitment.validate_with_context(commitment_context, expected_width, index, commitment_root_digest)?;

    let packed_columns = opening_proof.try_to_oracle_refs(index)?;
    let rebuilt_commitment_vector = build_commitment_vector(schema, &packed_columns)
        .map_err(|source| FinalOpeningError::FinalOpeningCommitmentVectorBuildFailed { index, source })?;
    if rebuilt_commitment_vector != opened_commitment.commitment_vector {
        return Err(FinalOpeningError::FinalOpeningCommitmentVectorMismatch { index });
    }

    OpenedAjtaiObjectWitness::new_with_commitment_root_digest(
        opened_commitment.opened_object.clone(),
        *commitment_context,
        packed_columns,
        rebuilt_commitment_vector,
        commitment_root_digest,
    )
    .map_err(|source| FinalOpeningError::FinalOpeningWitnessBuildFailed { index, source })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct AjtaiPackedColumnWitness {
    column_index: u32,
    rows: Vec<Vec<F>>,
}

impl AjtaiPackedColumnWitness {
    fn from_oracle_ref(column: PackedColumnOracleRef) -> Self {
        Self {
            column_index: column.column_index,
            rows: column.rows.into_iter().map(|row| row.to_vec()).collect(),
        }
    }

    fn to_oracle_ref(&self, index: usize) -> Result<PackedColumnOracleRef, FinalOpeningError> {
        let rows = self
            .rows
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                let actual = row.len();
                row.clone()
                    .try_into()
                    .map_err(|_: Vec<F>| FinalOpeningError::OpeningProofRowWidthMismatch {
                        index,
                        column_index: self.column_index as usize,
                        row_index,
                        expected: D,
                        actual,
                    })
            })
            .collect::<Result<Vec<[F; D]>, _>>()?;
        Ok(PackedColumnOracleRef {
            column_index: self.column_index,
            rows,
        })
    }
}

impl AjtaiOpeningProof {
    fn try_to_oracle_refs(&self, index: usize) -> Result<Vec<PackedColumnOracleRef>, FinalOpeningError> {
        self.packed_columns
            .iter()
            .map(|column| column.to_oracle_ref(index))
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imOpeningConvergenceProof {
    pub phase0_binding_surface: Rv64imPhase0BindingSurface,
    pub phase0: Rv64imEvalClaimBundle,
    pub phase1_results: Vec<ClaimReductionResult>,
    pub phase2: Phase2CollapseResult,
    pub final_openings: Vec<FinalOpeningTarget>,
    pub digest: [u8; 32],
}

impl Rv64imOpeningConvergenceProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/final_bundle");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final_bundle/phase0_binding_surface_digest",
            &self.phase0_binding_surface.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final_bundle/phase0_digest",
            &self.phase0.digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/final_bundle/phase1_count",
            &[self.phase1_results.len() as u64],
        );
        for result in &self.phase1_results {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/final_bundle/phase1_result_digest",
                &result.expected_digest(),
            );
        }
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/final_bundle/phase2_digest",
            &self.phase2.digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/final_bundle/final_opening_count",
            &[self.final_openings.len() as u64],
        );
        for target in &self.final_openings {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/final_bundle/final_opening_target_digest",
                &target.expected_digest(),
            );
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imOpeningConvergenceArtifact {
    pub phase0_binding_surface: Rv64imPhase0BindingSurface,
    pub phase0_digest: [u8; 32],
    pub phase1_results: Vec<ClaimReductionResult>,
    pub phase2: Phase2CollapseResult,
    pub final_openings: Vec<ProjectedFinalOpeningTarget>,
    pub digest: [u8; 32],
}

impl Rv64imOpeningConvergenceArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/compact_artifact");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase0_binding_surface_digest",
            &self.phase0_binding_surface.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase0_digest",
            &self.phase0_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase1_count",
            &[self.phase1_results.len() as u64],
        );
        for result in &self.phase1_results {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase1_result_digest",
                &result.expected_digest(),
            );
        }
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase2_digest",
            &self.phase2.digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/final_opening_count",
            &[self.final_openings.len() as u64],
        );
        for target in &self.final_openings {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/compact_artifact/final_opening_target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

fn opening_convergence_artifact_digest_from_trusted_components(
    phase0_binding_surface_digest: [u8; 32],
    phase0_digest: [u8; 32],
    phase1_results: &[ClaimReductionResult],
    phase2: &Phase2CollapseResult,
    final_openings: &[ProjectedFinalOpeningTarget],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/compact_artifact");
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase0_binding_surface_digest",
        &phase0_binding_surface_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase0_digest",
        &phase0_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase1_count",
        &[phase1_results.len() as u64],
    );
    for result in phase1_results {
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase1_result_digest",
            &result.trusted_local_digest(),
        );
    }
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/compact_artifact/phase2_digest",
        &phase2.digest,
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/compact_artifact/final_opening_count",
        &[final_openings.len() as u64],
    );
    for target in final_openings {
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/compact_artifact/final_opening_target_digest",
            &target.digest,
        );
    }
    tr.digest32()
}

pub fn build_rv64im_opening_convergence_proof_from_witnesses(
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imOpeningConvergenceProof, FinalOpeningError> {
    let phase0 = Rv64imEvalClaimBundle::new(
        claim_witnesses
            .iter()
            .map(|witness| witness.claim.clone())
            .collect(),
    )
    .map_err(FinalOpeningError::Phase0BundleBuildFailed)?;
    let phase1_results =
        build_claim_reduction_results_from_bundle_and_witnesses_trusted_local(&phase0, claim_witnesses)
            .map_err(FinalOpeningError::Phase1BuildFailed)?;
    let phase2 =
        build_phase2_collapse_result_trusted_local(&phase1_results).map_err(FinalOpeningError::Phase2BuildFailed)?;
    let final_openings = build_final_opening_targets(&phase2, claim_witnesses)?;
    let proof = Rv64imOpeningConvergenceProof {
        phase0_binding_surface: phase0_binding_surface.clone(),
        phase0,
        phase1_results,
        phase2,
        final_openings,
        digest: [0; 32],
    };
    Ok(Rv64imOpeningConvergenceProof {
        digest: proof.expected_digest(),
        ..proof
    })
}

pub fn build_rv64im_opening_convergence_artifact_from_witnesses(
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imOpeningConvergenceArtifact, FinalOpeningError> {
    let phase0 = Rv64imEvalClaimBundle::new(
        claim_witnesses
            .iter()
            .map(|witness| witness.claim.clone())
            .collect(),
    )
    .map_err(FinalOpeningError::Phase0BundleBuildFailed)?;
    build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local(
        phase0_binding_surface,
        &phase0,
        claim_witnesses,
    )
}

pub(crate) fn build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local(
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
    phase0: &Rv64imEvalClaimBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imOpeningConvergenceArtifact, FinalOpeningError> {
    let (artifact, _) =
        build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf(
            phase0_binding_surface,
            phase0,
            claim_witnesses,
        )?;
    Ok(artifact)
}

pub(crate) fn build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf(
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
    phase0: &Rv64imEvalClaimBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<
    (
        Rv64imOpeningConvergenceArtifact,
        Rv64imOpeningConvergenceArtifactBuildPerf,
    ),
    FinalOpeningError,
> {
    let total_started = Instant::now();

    let started = Instant::now();
    let phase1_results =
        build_claim_reduction_results_from_bundle_and_witnesses_trusted_local(&phase0, claim_witnesses)
            .map_err(FinalOpeningError::Phase1BuildFailed)?;
    let phase1_results_ms = elapsed_ms(started);

    let started = Instant::now();
    let phase2 =
        build_phase2_collapse_result_trusted_local(&phase1_results).map_err(FinalOpeningError::Phase2BuildFailed)?;
    let phase2_ms = elapsed_ms(started);

    let started = Instant::now();
    let (final_openings, final_openings_perf) =
        build_compact_final_opening_targets_with_perf(&phase2, claim_witnesses)?;
    let final_openings_ms = elapsed_ms(started);

    let started = Instant::now();
    let digest = opening_convergence_artifact_digest_from_trusted_components(
        phase0_binding_surface.digest,
        phase0.digest,
        &phase1_results,
        &phase2,
        &final_openings,
    );
    let digest_ms = elapsed_ms(started);

    let artifact = Rv64imOpeningConvergenceArtifact {
        phase0_binding_surface: phase0_binding_surface.clone(),
        phase0_digest: phase0.digest,
        phase1_results,
        phase2,
        final_openings,
        digest,
    };
    let perf = Rv64imOpeningConvergenceArtifactBuildPerf {
        phase1_results_ms,
        phase2_ms,
        final_openings_ms,
        final_openings_witness_map_ms: final_openings_perf.witness_map_ms,
        final_openings_representative_ms: final_openings_perf.representative_ms,
        final_openings_commitment_validate_ms: final_openings_perf.commitment_validate_ms,
        final_openings_opened_commitment_digest_ms: final_openings_perf.opened_commitment_digest_ms,
        final_openings_opening_proof_digest_ms: final_openings_perf.opening_proof_digest_ms,
        final_openings_target_build_ms: final_openings_perf.target_build_ms,
        digest_ms,
        total_ms: elapsed_ms(total_started),
    };
    Ok((artifact, perf))
}

pub fn build_rv64im_opening_convergence_artifact_from_proof(
    proof: &Rv64imOpeningConvergenceProof,
) -> Result<Rv64imOpeningConvergenceArtifact, FinalOpeningError> {
    verify_rv64im_opening_convergence_proof(proof)?;

    let final_openings = proof
        .phase2
        .reduced_claims
        .iter()
        .zip(&proof.final_openings)
        .map(|(reduced_claim, target)| {
            let projected = ProjectedFinalOpeningTarget {
                opened_commitment: target.opened_commitment.clone(),
                opening_proof: target.opening_proof.clone(),
                digest: [0; 32],
            };
            ProjectedFinalOpeningTarget {
                digest: projected.expected_digest(reduced_claim),
                ..projected
            }
        })
        .collect();

    let artifact = Rv64imOpeningConvergenceArtifact {
        phase0_binding_surface: proof.phase0_binding_surface.clone(),
        phase0_digest: proof.phase0.digest,
        phase1_results: proof.phase1_results.clone(),
        phase2: proof.phase2.clone(),
        final_openings,
        digest: [0; 32],
    };
    Ok(Rv64imOpeningConvergenceArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    })
}

pub fn verify_rv64im_opening_convergence_proof(proof: &Rv64imOpeningConvergenceProof) -> Result<(), FinalOpeningError> {
    validate_phase0_binding_surface(&proof.phase0_binding_surface)?;
    let expected_phase0 =
        Rv64imEvalClaimBundle::new(proof.phase0.claims.clone()).map_err(FinalOpeningError::Phase0BundleBuildFailed)?;
    if proof.phase0 != expected_phase0 {
        return Err(FinalOpeningError::Phase0BundleMismatch {
            expected: expected_phase0.digest,
            actual: proof.phase0.digest,
        });
    }
    let phase0_witnesses = rebuild_phase0_witnesses_from_final_openings(&proof.phase0.claims, &proof.final_openings)?;
    verify_claim_reduction_results_with_binding_surface(
        &proof.phase1_results,
        &phase0_witnesses,
        &proof.phase0_binding_surface,
    )
    .map_err(FinalOpeningError::Phase1VerificationFailed)?;
    verify_phase2_collapse_result(&proof.phase2, &proof.phase1_results)
        .map_err(FinalOpeningError::Phase2VerificationFailed)?;

    let actual_final_opening_count = proof.final_openings.len();
    if actual_final_opening_count != FINAL_OPENING_COUNT_V1 {
        return Err(FinalOpeningError::FinalOpeningCountMismatch {
            expected: FINAL_OPENING_COUNT_V1,
            actual: actual_final_opening_count,
        });
    }
    if proof.phase2.reduced_claims.len() != actual_final_opening_count {
        return Err(FinalOpeningError::FinalOpeningCountMismatch {
            expected: proof.phase2.reduced_claims.len(),
            actual: actual_final_opening_count,
        });
    }

    for (index, (target, reduced_claim)) in proof
        .final_openings
        .iter()
        .zip(&proof.phase2.reduced_claims)
        .enumerate()
    {
        if &target.reduced_claim != reduced_claim {
            return Err(FinalOpeningError::UnexpectedReducedClaimAtIndex { index });
        }
    }

    let expected_digest = proof.expected_digest();
    if proof.digest != expected_digest {
        return Err(FinalOpeningError::ProofDigestMismatch {
            expected: expected_digest,
            actual: proof.digest,
        });
    }

    Ok(())
}

pub fn verify_rv64im_opening_convergence_artifact(
    artifact: &Rv64imOpeningConvergenceArtifact,
) -> Result<(), FinalOpeningError> {
    validate_phase0_binding_surface(&artifact.phase0_binding_surface)?;
    let reconstructed_phase0 = Rv64imEvalClaimBundle::new(
        artifact
            .phase1_results
            .iter()
            .flat_map(|result| result.bucket.claims.clone())
            .collect(),
    )
    .map_err(FinalOpeningError::Phase0BundleBuildFailed)?;
    if artifact.phase0_digest != reconstructed_phase0.digest {
        return Err(FinalOpeningError::ArtifactPhase0DigestMismatch {
            expected: reconstructed_phase0.digest,
            actual: artifact.phase0_digest,
        });
    }
    let actual_final_opening_count = artifact.final_openings.len();
    if actual_final_opening_count != FINAL_OPENING_COUNT_V1 {
        return Err(FinalOpeningError::FinalOpeningCountMismatch {
            expected: FINAL_OPENING_COUNT_V1,
            actual: actual_final_opening_count,
        });
    }
    if artifact.phase2.reduced_claims.len() != actual_final_opening_count {
        return Err(FinalOpeningError::FinalOpeningCountMismatch {
            expected: artifact.phase2.reduced_claims.len(),
            actual: actual_final_opening_count,
        });
    }

    let phase0_witnesses = rebuild_phase0_witnesses_from_projected_final_openings(
        &reconstructed_phase0.claims,
        &artifact.phase2,
        &artifact.final_openings,
    )?;
    verify_claim_reduction_results_with_binding_surface(
        &artifact.phase1_results,
        &phase0_witnesses,
        &artifact.phase0_binding_surface,
    )
    .map_err(FinalOpeningError::Phase1VerificationFailed)?;
    verify_phase2_collapse_result(&artifact.phase2, &artifact.phase1_results)
        .map_err(FinalOpeningError::Phase2VerificationFailed)?;

    let expected_digest = artifact.expected_digest();
    if artifact.digest != expected_digest {
        return Err(FinalOpeningError::ArtifactDigestMismatch {
            expected: expected_digest,
            actual: artifact.digest,
        });
    }

    Ok(())
}

pub fn verify_rv64im_opening_convergence_artifact_from_proof(
    artifact: &Rv64imOpeningConvergenceArtifact,
    proof: &Rv64imOpeningConvergenceProof,
) -> Result<(), FinalOpeningError> {
    verify_rv64im_opening_convergence_artifact(artifact)?;
    let expected = build_rv64im_opening_convergence_artifact_from_proof(proof)?;
    if &expected != artifact {
        return Err(FinalOpeningError::ArtifactProjectionMismatch {
            expected: expected.digest,
            actual: artifact.digest,
        });
    }
    Ok(())
}

fn validate_phase0_binding_surface(surface: &Rv64imPhase0BindingSurface) -> Result<(), FinalOpeningError> {
    surface.validate_canonical_order()?;
    for (index, target) in surface.targets.iter().enumerate() {
        let expected_digest = target.expected_digest();
        if target.digest != expected_digest {
            return Err(FinalOpeningError::BindingSurfaceTargetDigestMismatch {
                index,
                expected: expected_digest,
                actual: target.digest,
            });
        }
    }
    let expected_digest = surface.expected_digest();
    if surface.digest != expected_digest {
        return Err(FinalOpeningError::BindingSurfaceDigestMismatch {
            expected: expected_digest,
            actual: surface.digest,
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum FinalOpeningError {
    #[error("final opening phase0 binding surface target count mismatch: expected {expected}, got {actual}")]
    BindingSurfaceTargetCountMismatch { expected: usize, actual: usize },
    #[error(
        "final opening phase0 binding surface schema mismatch at index {index}: expected {expected:?}, got {actual:?}"
    )]
    BindingSurfaceSchemaMismatch {
        index: usize,
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
    },
    #[error("final opening phase0 binding surface is missing target for {schema:?}")]
    BindingSurfaceTargetMissing { schema: FamilyEvalSchemaId },
    #[error("final opening phase0 binding surface target digest mismatch at index {index}: expected {expected:?}, got {actual:?}")]
    BindingSurfaceTargetDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening phase0 binding surface digest mismatch: expected {expected:?}, got {actual:?}")]
    BindingSurfaceDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error(
        "final opening phase0 claim binding digest mismatch at index {index} for {schema:?}: expected {expected:?}, got {actual:?}"
    )]
    ClaimBindingDigestMismatch {
        index: usize,
        schema: FamilyEvalSchemaId,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening phase0 claim point mismatch at index {index} for {schema:?}")]
    ClaimPointBindingMismatch {
        index: usize,
        schema: FamilyEvalSchemaId,
    },
    #[error("final opening could not canonicalize the phase0 claim bundle: {0}")]
    Phase0BundleBuildFailed(EvalClaimError),
    #[error("final opening phase0 bundle digest mismatch: expected {expected:?}, got {actual:?}")]
    Phase0BundleMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening could not build phase1 results: {0}")]
    Phase1BuildFailed(ClaimReductionError),
    #[error("final opening could not build phase2 result: {0}")]
    Phase2BuildFailed(Phase2CollapseError),
    #[error("final opening phase1 verification failed: {0}")]
    Phase1VerificationFailed(ClaimReductionError),
    #[error("final opening phase2 verification failed: {0}")]
    Phase2VerificationFailed(Phase2CollapseError),
    #[error("final opening compact artifact phase0 digest mismatch: expected {expected:?}, got {actual:?}")]
    ArtifactPhase0DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening target {index} carries an invalid reduced claim: {source}")]
    InvalidReducedClaim {
        index: usize,
        #[source]
        source: Phase2CollapseError,
    },
    #[error("final opening target {index} opened-commitment vector width mismatch: expected {expected}, got {actual}")]
    OpenedCommitmentVectorCountMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("final opening target {index} opened-object digest mismatch: expected {expected:?}, got {actual:?}")]
    OpenedCommitmentOpenedObjectDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening target {index} commitment root mismatch: expected {expected:?}, got {actual:?}")]
    OpenedCommitmentRootMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening target {index} opened-commitment digest mismatch: expected {expected:?}, got {actual:?}")]
    OpenedCommitmentDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening target {index} packed-column count mismatch: expected {expected}, got {actual}")]
    OpeningProofPackedColumnCountMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("final opening target {index} packed-column index mismatch: expected {expected}, got {actual}")]
    OpeningProofColumnIndexMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "final opening target {index} packed column {column_index} row-domain length mismatch: expected {expected}, got {actual}"
    )]
    OpeningProofRowDomainLengthMismatch {
        index: usize,
        column_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "final opening target {index} packed column {column_index} row {row_index} width mismatch: expected {expected}, got {actual}"
    )]
    OpeningProofRowWidthMismatch {
        index: usize,
        column_index: usize,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("final opening target {index} opening-proof digest mismatch: expected {expected:?}, got {actual:?}")]
    OpeningProofDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening projected target {index} digest mismatch: expected {expected:?}, got {actual:?}")]
    ProjectedFinalOpeningDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening target {index} opened object does not match the reduced claim")]
    FinalOpeningObjectMismatch { index: usize },
    #[error("final opening target {index} could not rebuild the commitment vector: {source}")]
    FinalOpeningCommitmentVectorBuildFailed {
        index: usize,
        #[source]
        source: SimpleKernelError,
    },
    #[error("final opening target {index} commitment vector does not match the carried opened commitment")]
    FinalOpeningCommitmentVectorMismatch { index: usize },
    #[error("final opening target {index} could not rebuild the opened-object witness: {source}")]
    FinalOpeningWitnessBuildFailed {
        index: usize,
        #[source]
        source: EvalClaimError,
    },
    #[error("final opening target {index} could not evaluate the reduced claim payload: {source}")]
    FinalOpeningPayloadEvaluationFailed {
        index: usize,
        #[source]
        source: EvalClaimError,
    },
    #[error("final opening target {index} payload does not match its packed-column witness")]
    FinalOpeningPayloadMismatch { index: usize },
    #[error("final opening witness set contains duplicate phase0 claim id {claim_id:?}")]
    DuplicatePhase0ClaimId { claim_id: FamilyEvalClaimId },
    #[error("final opening witness set is missing source claim id {claim_id:?}")]
    MissingSourceClaimId { claim_id: FamilyEvalClaimId },
    #[error("final opening target {index} source claim {claim_id:?} does not match the reduced claim object/context")]
    ReducedClaimSourceMismatch {
        index: usize,
        claim_id: FamilyEvalClaimId,
    },
    #[error("final opening witness set contains duplicate opened object {opened_object_digest:?}")]
    DuplicateFinalOpeningObject { opened_object_digest: [u8; 32] },
    #[error("final opening witness set is missing opened object {opened_object_digest:?}")]
    MissingFinalOpeningObject { opened_object_digest: [u8; 32] },
    #[error("final opening count mismatch: expected {expected}, got {actual}")]
    FinalOpeningCountMismatch { expected: usize, actual: usize },
    #[error("final opening target at index {index} does not match the phase2 reduced claim")]
    UnexpectedReducedClaimAtIndex { index: usize },
    #[error("final opening bundle digest mismatch: expected {expected:?}, got {actual:?}")]
    ProofDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("final opening compact artifact digest mismatch: expected {expected:?}, got {actual:?}")]
    ArtifactDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error(
        "final opening compact artifact does not match the projected full proof: expected {expected:?}, got {actual:?}"
    )]
    ArtifactProjectionMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
}

fn build_final_opening_targets(
    phase2: &Phase2CollapseResult,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<FinalOpeningTarget>, FinalOpeningError> {
    let witness_map = build_claim_witness_map(claim_witnesses)?;
    let mut targets = Vec::with_capacity(phase2.reduced_claims.len());

    for (index, reduced_claim) in phase2.reduced_claims.iter().enumerate() {
        let representative = representative_claim_witness(index, reduced_claim, &witness_map)?;
        let commitment_root_digest = representative.witness.opened_object.commitment_root_digest;

        let opened_commitment = OpenedAjtaiCommitmentPublic::new_with_commitment_root_digest(
            representative.witness.opened_object.clone(),
            &reduced_claim.commitment_context,
            representative.witness.commitment_vector.clone(),
            reduced_claim.payload.column_evals.len(),
            commitment_root_digest,
        )?;
        let target = FinalOpeningTarget {
            reduced_claim: reduced_claim.clone(),
            opened_commitment,
            opening_proof: AjtaiOpeningProof::new(representative.witness.packed_columns.clone()),
        };
        target.validate_and_rebuild_witness(index)?;
        targets.push(target);
    }

    Ok(targets)
}

fn build_compact_final_opening_targets_with_perf(
    phase2: &Phase2CollapseResult,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(Vec<ProjectedFinalOpeningTarget>, CompactFinalOpeningTargetsBuildPerf), FinalOpeningError> {
    let total_started = Instant::now();
    let started = Instant::now();
    let witness_map = build_claim_witness_map(claim_witnesses)?;
    let witness_map_ms = elapsed_ms(started);
    let mut targets = Vec::with_capacity(phase2.reduced_claims.len());
    let mut perf = CompactFinalOpeningTargetsBuildPerf {
        witness_map_ms,
        ..CompactFinalOpeningTargetsBuildPerf::default()
    };

    for (index, reduced_claim) in phase2.reduced_claims.iter().enumerate() {
        let reduced_claim_digest = phase2
            .records
            .get(index)
            .map(|record| record.reduced_claim_digest)
            .ok_or(FinalOpeningError::FinalOpeningCountMismatch {
                expected: phase2.reduced_claims.len(),
                actual: phase2.records.len(),
            })?;
        let started = Instant::now();
        let representative = representative_claim_witness(index, reduced_claim, &witness_map)?;
        perf.representative_ms += elapsed_ms(started);
        let commitment_root_digest = representative.witness.opened_object.commitment_root_digest;

        let started = Instant::now();
        let opened_commitment = OpenedAjtaiCommitmentPublic::new_with_commitment_root_digest(
            representative.witness.opened_object.clone(),
            &reduced_claim.commitment_context,
            representative.witness.commitment_vector.clone(),
            reduced_claim.payload.column_evals.len(),
            commitment_root_digest,
        )?;
        perf.commitment_validate_ms += elapsed_ms(started);

        let started = Instant::now();
        let opening_proof = AjtaiOpeningProof::new(representative.witness.packed_columns.clone());
        perf.opening_proof_digest_ms += elapsed_ms(started);

        let started = Instant::now();
        let target = ProjectedFinalOpeningTarget {
            opened_commitment,
            opening_proof,
            digest: [0; 32],
        };
        perf.opened_commitment_digest_ms += 0.0;
        targets.push(ProjectedFinalOpeningTarget {
            digest: compact_final_opening_target_digest_from_reduced_claim_digest(
                reduced_claim_digest,
                target.opened_commitment.digest,
                target.opening_proof.digest,
            ),
            ..target
        });
        targets[index].validate_and_rebuild_witness(reduced_claim, index)?;
        perf.target_build_ms += elapsed_ms(started);
    }

    perf.total_ms = elapsed_ms(total_started);
    Ok((targets, perf))
}

fn rebuild_phase0_witnesses_from_final_openings(
    phase0_claims: &[FamilyEvalClaim],
    final_openings: &[FinalOpeningTarget],
) -> Result<Vec<FamilyEvalClaimWitness>, FinalOpeningError> {
    let mut object_witnesses = BTreeMap::<[u8; 32], Arc<OpenedAjtaiObjectWitness>>::new();
    for (index, target) in final_openings.iter().enumerate() {
        let object_digest = target.reduced_claim.opened_object.digest;
        let witness = Arc::new(target.validate_and_rebuild_witness(index)?);
        if object_witnesses.insert(object_digest, witness).is_some() {
            return Err(FinalOpeningError::DuplicateFinalOpeningObject {
                opened_object_digest: object_digest,
            });
        }
    }

    let mut claim_ids = BTreeSet::new();
    let mut claim_witnesses = Vec::with_capacity(phase0_claims.len());
    for (index, claim) in phase0_claims.iter().enumerate() {
        if !claim_ids.insert(claim.id) {
            return Err(FinalOpeningError::DuplicatePhase0ClaimId { claim_id: claim.id });
        }
        let witness = object_witnesses
            .get(&claim.opened_object.digest)
            .ok_or(FinalOpeningError::MissingFinalOpeningObject {
                opened_object_digest: claim.opened_object.digest,
            })?
            .clone();
        let claim_witness = FamilyEvalClaimWitness::new(claim.clone(), witness)
            .map_err(|source| FinalOpeningError::FinalOpeningWitnessBuildFailed { index, source })?;
        claim_witnesses.push(claim_witness);
    }

    Ok(claim_witnesses)
}

fn rebuild_phase0_witnesses_from_projected_final_openings(
    phase0_claims: &[FamilyEvalClaim],
    phase2: &Phase2CollapseResult,
    final_openings: &[ProjectedFinalOpeningTarget],
) -> Result<Vec<FamilyEvalClaimWitness>, FinalOpeningError> {
    let mut object_witnesses = BTreeMap::<[u8; 32], Arc<OpenedAjtaiObjectWitness>>::new();
    for (index, (target, reduced_claim)) in final_openings
        .iter()
        .zip(&phase2.reduced_claims)
        .enumerate()
    {
        let object_digest = reduced_claim.opened_object.digest;
        let witness = Arc::new(target.validate_and_rebuild_witness(reduced_claim, index)?);
        if object_witnesses.insert(object_digest, witness).is_some() {
            return Err(FinalOpeningError::DuplicateFinalOpeningObject {
                opened_object_digest: object_digest,
            });
        }
    }

    let mut claim_ids = BTreeSet::new();
    let mut claim_witnesses = Vec::with_capacity(phase0_claims.len());
    for (index, claim) in phase0_claims.iter().enumerate() {
        if !claim_ids.insert(claim.id) {
            return Err(FinalOpeningError::DuplicatePhase0ClaimId { claim_id: claim.id });
        }
        let witness = object_witnesses
            .get(&claim.opened_object.digest)
            .ok_or(FinalOpeningError::MissingFinalOpeningObject {
                opened_object_digest: claim.opened_object.digest,
            })?
            .clone();
        let claim_witness = FamilyEvalClaimWitness::new(claim.clone(), witness)
            .map_err(|source| FinalOpeningError::FinalOpeningWitnessBuildFailed { index, source })?;
        claim_witnesses.push(claim_witness);
    }

    Ok(claim_witnesses)
}

fn build_claim_witness_map<'a>(
    claim_witnesses: &'a [FamilyEvalClaimWitness],
) -> Result<BTreeMap<FamilyEvalClaimId, &'a FamilyEvalClaimWitness>, FinalOpeningError> {
    let mut witness_map = BTreeMap::new();
    for claim_witness in claim_witnesses {
        if witness_map
            .insert(claim_witness.claim.id, claim_witness)
            .is_some()
        {
            return Err(FinalOpeningError::DuplicatePhase0ClaimId {
                claim_id: claim_witness.claim.id,
            });
        }
    }
    Ok(witness_map)
}

fn representative_claim_witness<'a>(
    index: usize,
    reduced_claim: &ReducedEvalClaim,
    witness_map: &'a BTreeMap<FamilyEvalClaimId, &'a FamilyEvalClaimWitness>,
) -> Result<&'a FamilyEvalClaimWitness, FinalOpeningError> {
    let first_claim_id =
        *reduced_claim
            .source_claim_ids
            .first()
            .ok_or_else(|| FinalOpeningError::InvalidReducedClaim {
                index,
                source: Phase2CollapseError::ReducedClaimSourceIdsEmpty { index },
            })?;
    let representative = witness_map
        .get(&first_claim_id)
        .copied()
        .ok_or(FinalOpeningError::MissingSourceClaimId {
            claim_id: first_claim_id,
        })?;

    for claim_id in &reduced_claim.source_claim_ids {
        let claim_witness = witness_map
            .get(claim_id)
            .copied()
            .ok_or(FinalOpeningError::MissingSourceClaimId { claim_id: *claim_id })?;
        if claim_witness.claim.opened_object != reduced_claim.opened_object
            || claim_witness.claim.commitment_context != reduced_claim.commitment_context
        {
            return Err(FinalOpeningError::ReducedClaimSourceMismatch {
                index,
                claim_id: *claim_id,
            });
        }
    }

    Ok(representative)
}
