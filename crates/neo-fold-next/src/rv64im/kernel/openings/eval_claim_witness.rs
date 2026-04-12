//! Owns prover-only Phase 0 RV64IM evaluation-claim witnesses and the accepted-artifact claim projection.
//!
//! It owns:
//! - concrete packed-column oracle data for opened Ajtai objects
//! - commitment-root recomputation from real Ajtai commitment vectors
//! - claim/witness consistency checks
//! - the initial Stage1Rows real claim-emission path
//!
//! It does not own:
//! - the published Nightstream carried boundary
//! - synthetic selected-opening rebinding
//! - Phase 1 reduction or Phase 2 accumulation

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use neo_ajtai::{
    get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, set_global_pp_seeded,
    AjtaiSModule, Commitment,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::chip8::poly::build_eq_table;
use crate::finalize::digest32_as_fields;
use crate::rv64im::stage1::{stage1_row_words, Stage1ProofBundle};
use crate::rv64im::stage2::{
    ram_event_words, register_read_words, register_write_words, twist_link_words, Stage2ProofBundle,
};
use crate::rv64im::stage3::{continuity_event_words, Stage3ProofBundle};

use super::opening_eval_claims::{
    CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalPayload, FamilyEvalSchemaId, OpenedAjtaiObjectId,
    OpeningClaimAccumulator, PackedColumnEval, Rv64imEvalClaimBundle,
};
use super::opening_payload_semantics::{encode_words_to_field_evals_f, phase0_full_width_for_schema};
use super::opening_point_derivation::derive_phase0_point;
use super::proof_accepted::Rv64imAcceptedProofArtifact;
use super::simple::{SimpleKernelError, EXACT_STAGE_PP_SEED};
use super::stage_artifacts::{Stage1ArtifactSurface, Stage2ArtifactSurface, Stage3ArtifactSurface};

const PHASE0_OPENED_OBJECT_LAYOUT_V1: u64 = 1;
const STAGE1_PHASE0_SLOT_COUNT: u32 = 4;
const PHASE0_COMMITMENT_BATCH: usize = 256;
static PHASE0_PARALLEL_CLAIM_BUILD_IN_USE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imEvalClaimWitnessBuildPerf {
    pub packed_columns_ms: f64,
    pub commitment_vector_ms: f64,
    pub commitment_params_ms: f64,
    pub commitment_committer_ms: f64,
    pub commitment_mats_ms: f64,
    pub commitment_commit_many_ms: f64,
    pub commitment_root_ms: f64,
    pub opened_object_id_ms: f64,
    pub opened_object_total_ms: f64,
    pub binding_digest_ms: f64,
    pub point_derivation_ms: f64,
    pub payload_eval_ms: f64,
    pub claim_build_ms: f64,
    pub slot_claims_total_ms: f64,
    pub total_ms: f64,
}

impl Rv64imEvalClaimWitnessBuildPerf {
    fn accumulate(&mut self, other: Self) {
        self.packed_columns_ms += other.packed_columns_ms;
        self.commitment_vector_ms += other.commitment_vector_ms;
        self.commitment_params_ms += other.commitment_params_ms;
        self.commitment_committer_ms += other.commitment_committer_ms;
        self.commitment_mats_ms += other.commitment_mats_ms;
        self.commitment_commit_many_ms += other.commitment_commit_many_ms;
        self.commitment_root_ms += other.commitment_root_ms;
        self.opened_object_id_ms += other.opened_object_id_ms;
        self.opened_object_total_ms += other.opened_object_total_ms;
        self.binding_digest_ms += other.binding_digest_ms;
        self.point_derivation_ms += other.point_derivation_ms;
        self.payload_eval_ms += other.payload_eval_ms;
        self.claim_build_ms += other.claim_build_ms;
        self.slot_claims_total_ms += other.slot_claims_total_ms;
        self.total_ms += other.total_ms;
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[derive(Clone, Copy, Debug, Default)]
struct Phase0CommitmentVectorBuildPerf {
    params_ms: f64,
    committer_ms: f64,
    mats_ms: f64,
    commit_many_ms: f64,
}

struct Phase0ParallelClaimBuildGuard;

impl Drop for Phase0ParallelClaimBuildGuard {
    fn drop(&mut self) {
        PHASE0_PARALLEL_CLAIM_BUILD_IN_USE.store(false, Ordering::Release);
    }
}

fn try_acquire_parallel_phase0_claim_build() -> Option<Phase0ParallelClaimBuildGuard> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if rayon::current_num_threads() <= 1 || rayon::current_thread_index().is_some() {
            return None;
        }
        if PHASE0_PARALLEL_CLAIM_BUILD_IN_USE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            Some(Phase0ParallelClaimBuildGuard)
        } else {
            None
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedColumnOracleRef {
    pub column_index: u32,
    pub rows: Vec<[F; D]>,
}

pub type RealAjtaiCommitmentVector = Vec<Commitment>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenedAjtaiObjectWitness {
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub row_domain_log_size: u32,
    pub packed_column_count: u32,
    pub packed_columns: Vec<PackedColumnOracleRef>,
    pub commitment_vector: RealAjtaiCommitmentVector,
}

impl OpenedAjtaiObjectWitness {
    pub fn new(
        opened_object: OpenedAjtaiObjectId,
        commitment_context: CommitmentContextId,
        packed_columns: Vec<PackedColumnOracleRef>,
        commitment_vector: RealAjtaiCommitmentVector,
    ) -> Result<Self, EvalClaimError> {
        let commitment_root_digest = phase0_commitment_root_digest(&commitment_vector);
        Self::new_with_commitment_root_digest(
            opened_object,
            commitment_context,
            packed_columns,
            commitment_vector,
            commitment_root_digest,
        )
    }

    pub(crate) fn new_with_commitment_root_digest(
        opened_object: OpenedAjtaiObjectId,
        commitment_context: CommitmentContextId,
        packed_columns: Vec<PackedColumnOracleRef>,
        commitment_vector: RealAjtaiCommitmentVector,
        commitment_root_digest: [u8; 32],
    ) -> Result<Self, EvalClaimError> {
        Self::validate_with_commitment_root_digest(
            &opened_object,
            &packed_columns,
            &commitment_vector,
            commitment_root_digest,
        )?;
        Ok(Self::new_trusted_local_with_commitment_root_digest(
            opened_object,
            commitment_context,
            packed_columns,
            commitment_vector,
        ))
    }

    fn validate_with_commitment_root_digest(
        opened_object: &OpenedAjtaiObjectId,
        packed_columns: &[PackedColumnOracleRef],
        commitment_vector: &[Commitment],
        commitment_root_digest: [u8; 32],
    ) -> Result<(), EvalClaimError> {
        let packed_column_count = packed_columns.len();
        if packed_column_count != commitment_vector.len() {
            return Err(EvalClaimError::WitnessCommitmentVectorCountMismatch {
                expected: packed_column_count,
                actual: commitment_vector.len(),
            });
        }

        let expected_time_len = 1usize << (opened_object.row_domain_log_size as usize);
        for (column_index, column) in packed_columns.iter().enumerate() {
            if column.rows.len() != expected_time_len {
                return Err(EvalClaimError::WitnessRowDomainLengthMismatch {
                    column_index,
                    expected: expected_time_len,
                    actual: column.rows.len(),
                });
            }
        }

        if commitment_root_digest != opened_object.commitment_root_digest {
            return Err(EvalClaimError::WitnessCommitmentRootMismatch {
                expected_digest: commitment_root_digest,
                actual_digest: opened_object.commitment_root_digest,
            });
        }
        Ok(())
    }

    fn new_trusted_local_with_commitment_root_digest(
        opened_object: OpenedAjtaiObjectId,
        commitment_context: CommitmentContextId,
        packed_columns: Vec<PackedColumnOracleRef>,
        commitment_vector: RealAjtaiCommitmentVector,
    ) -> Self {
        let packed_column_count = packed_columns.len();
        Self {
            row_domain_log_size: opened_object.row_domain_log_size,
            packed_column_count: packed_column_count as u32,
            opened_object,
            commitment_context,
            packed_columns,
            commitment_vector,
        }
    }

    pub fn evaluate_payload(&self, point: &[K]) -> Result<Vec<PackedColumnEval>, EvalClaimError> {
        let expected_point_arity = self.row_domain_log_size as usize;
        if point.len() != expected_point_arity {
            return Err(EvalClaimError::PointArityMismatch {
                expected: expected_point_arity,
                actual: point.len(),
            });
        }
        if self.packed_columns.len() != self.packed_column_count as usize {
            return Err(EvalClaimError::WitnessPackedColumnCountMismatch {
                expected: self.packed_column_count as usize,
                actual: self.packed_columns.len(),
            });
        }

        let weights = build_eq_table(point);
        self.packed_columns
            .iter()
            .map(|column| {
                evaluate_packed_column_rows_with_weights(&column.rows, &weights, column.column_index as usize)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyEvalClaimWitness {
    pub claim: FamilyEvalClaim,
    pub witness: Arc<OpenedAjtaiObjectWitness>,
}

impl FamilyEvalClaimWitness {
    pub fn new<W>(claim: FamilyEvalClaim, witness: W) -> Result<Self, EvalClaimError>
    where
        W: Into<Arc<OpenedAjtaiObjectWitness>>,
    {
        let witness = witness.into();
        claim.validate()?;

        if claim.opened_object.digest != witness.opened_object.digest {
            return Err(EvalClaimError::WitnessOpenedObjectMismatch {
                claim_digest: claim.opened_object.digest,
                witness_digest: witness.opened_object.digest,
            });
        }
        if claim.commitment_context != witness.commitment_context {
            return Err(EvalClaimError::WitnessCommitmentContextMismatch);
        }
        if claim.point.len() != witness.row_domain_log_size as usize {
            return Err(EvalClaimError::PointArityMismatch {
                expected: witness.row_domain_log_size as usize,
                actual: claim.point.len(),
            });
        }
        if claim.payload.column_evals.len() != witness.packed_column_count as usize {
            return Err(EvalClaimError::PayloadWidthMismatch {
                schema: claim.payload.schema,
                expected: witness.packed_column_count as usize,
                actual: claim.payload.column_evals.len(),
            });
        }
        let expected_payload = witness.evaluate_payload(&claim.point)?;
        if expected_payload != claim.payload.column_evals {
            return Err(EvalClaimError::WitnessPayloadMismatch {
                schema: claim.payload.schema,
                slot: claim.id.slot,
            });
        }

        Ok(Self { claim, witness })
    }
}

pub fn build_stage1_claim_witnesses(
    artifact: &Stage1ArtifactSurface,
    proof: &Stage1ProofBundle,
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    let (claims, _) = build_stage1_claim_witnesses_with_perf(artifact, proof)?;
    Ok(claims)
}

fn build_stage1_claim_witnesses_with_perf(
    artifact: &Stage1ArtifactSurface,
    proof: &Stage1ProofBundle,
) -> Result<(Vec<FamilyEvalClaimWitness>, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    assert_matching_digest(
        "phase0/stage1 claim emission rows digest",
        artifact.rows.rows_digest,
        proof.address_correctness.rows_digest,
    )?;
    if proof.row_bindings.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "phase0/stage1 claim emission requires at least one row".into(),
        ));
    }

    let (witness, mut perf) = build_opened_object_witness_with_perf(
        FamilyEvalSchemaId::Stage1Rows,
        &proof.row_bindings,
        stage1_row_words,
        "phase0/stage1",
    )?;
    let (claims, slot_perf) = build_claim_witnesses_for_slots_with_perf(
        FamilyEvalSchemaId::Stage1Rows,
        &witness,
        STAGE1_PHASE0_SLOT_COUNT,
        artifact.rows.rows_digest,
        proof.digest,
        "phase0/stage1",
    )?;
    perf.accumulate(slot_perf);
    Ok((claims, perf))
}

pub fn build_stage2_claim_witnesses(
    artifact: &Stage2ArtifactSurface,
    proof: &Stage2ProofBundle,
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    let (claims, _) = build_stage2_claim_witnesses_with_perf(artifact, proof)?;
    Ok(claims)
}

fn build_stage2_claim_witnesses_with_perf(
    artifact: &Stage2ArtifactSurface,
    proof: &Stage2ProofBundle,
) -> Result<(Vec<FamilyEvalClaimWitness>, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    assert_matching_digest(
        "phase0/stage2 register-reads family digest (semantics)",
        artifact.families.register_reads_digest,
        proof.semantics.register_reads_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 register-reads family digest (linkage)",
        artifact.families.register_reads_digest,
        proof.linkage.register_reads_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 register-writes family digest (semantics)",
        artifact.families.register_writes_digest,
        proof.semantics.register_writes_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 register-writes family digest (linkage)",
        artifact.families.register_writes_digest,
        proof.linkage.register_writes_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 ram-events family digest (semantics)",
        artifact.families.ram_events_digest,
        proof.semantics.ram_events_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 ram-events family digest (linkage)",
        artifact.families.ram_events_digest,
        proof.linkage.ram_events_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 twist-links family digest (semantics)",
        artifact.families.twist_links_digest,
        proof.semantics.twist_links_family_digest,
    )?;
    assert_matching_digest(
        "phase0/stage2 twist-links family digest (linkage)",
        artifact.families.twist_links_digest,
        proof.linkage.twist_links_family_digest,
    )?;

    let mut claims = Vec::with_capacity(4);
    let mut perf = Rv64imEvalClaimWitnessBuildPerf::default();

    if let Some((claim, claim_perf)) = maybe_build_singleton_claim_witness_with_perf(
        FamilyEvalSchemaId::Stage2RegisterReads,
        &proof.register.reads,
        register_read_words,
        artifact.families.register_reads_digest,
        proof.digest,
        "phase0/stage2/register_reads",
    )? {
        claims.push(claim);
        perf.accumulate(claim_perf);
    }

    if let Some((claim, claim_perf)) = maybe_build_singleton_claim_witness_with_perf(
        FamilyEvalSchemaId::Stage2RegisterWrites,
        &proof.register.writes,
        register_write_words,
        artifact.families.register_writes_digest,
        proof.digest,
        "phase0/stage2/register_writes",
    )? {
        claims.push(claim);
        perf.accumulate(claim_perf);
    }

    if let Some((claim, claim_perf)) = maybe_build_singleton_claim_witness_with_perf(
        FamilyEvalSchemaId::Stage2RamEvents,
        &proof.ram.events,
        ram_event_words,
        artifact.families.ram_events_digest,
        proof.digest,
        "phase0/stage2/ram_events",
    )? {
        claims.push(claim);
        perf.accumulate(claim_perf);
    }

    if let Some((claim, claim_perf)) = maybe_build_singleton_claim_witness_with_perf(
        FamilyEvalSchemaId::Stage2TwistLinks,
        &proof.temporal.twist_links,
        twist_link_words,
        artifact.families.twist_links_digest,
        proof.digest,
        "phase0/stage2/twist_links",
    )? {
        claims.push(claim);
        perf.accumulate(claim_perf);
    }

    Ok((claims, perf))
}

pub fn build_stage3_claim_witness(
    artifact: &Stage3ArtifactSurface,
    proof: &Stage3ProofBundle,
) -> Result<FamilyEvalClaimWitness, SimpleKernelError> {
    let (claim, _) = build_stage3_claim_witness_with_perf(artifact, proof)?
        .ok_or_else(|| SimpleKernelError::Bridge("phase0/stage3 continuity family is empty".into()))?;
    Ok(claim)
}

fn build_stage3_claim_witness_with_perf(
    artifact: &Stage3ArtifactSurface,
    proof: &Stage3ProofBundle,
) -> Result<Option<(FamilyEvalClaimWitness, Rv64imEvalClaimWitnessBuildPerf)>, SimpleKernelError> {
    assert_matching_digest(
        "phase0/stage3 continuity family digest (linkage)",
        artifact.continuity.continuity_digest,
        proof.linkage.continuity_family_digest,
    )?;
    maybe_build_singleton_claim_witness_with_perf(
        FamilyEvalSchemaId::Stage3Continuity,
        &proof.bridge.continuity,
        continuity_event_words,
        artifact.continuity.continuity_digest,
        proof.digest,
        "phase0/stage3/continuity",
    )
}

pub fn build_rv64im_eval_claim_witnesses_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    let (claims, _) = build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf(artifact)?;
    Ok(claims)
}

pub(crate) fn build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Vec<FamilyEvalClaimWitness>, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    let ((stage1_claims, stage1_perf), (stage2_claims, stage2_perf), stage3_claim_and_perf) =
        if let Some(_parallel_guard) = try_acquire_parallel_phase0_claim_build() {
            let (stage1_result, (stage2_result, stage3_result)) = rayon::join(
                || build_stage1_claim_witnesses_with_perf(&artifact.stage_claims.claims.stage1, &artifact.stage1),
                || {
                    rayon::join(
                        || {
                            build_stage2_claim_witnesses_with_perf(
                                &artifact.stage_claims.claims.stage2,
                                &artifact.stage2,
                            )
                        },
                        || build_stage3_claim_witness_with_perf(&artifact.stage_claims.claims.stage3, &artifact.stage3),
                    )
                },
            );
            (stage1_result?, stage2_result?, stage3_result?)
        } else {
            (
                build_stage1_claim_witnesses_with_perf(&artifact.stage_claims.claims.stage1, &artifact.stage1)?,
                build_stage2_claim_witnesses_with_perf(&artifact.stage_claims.claims.stage2, &artifact.stage2)?,
                build_stage3_claim_witness_with_perf(&artifact.stage_claims.claims.stage3, &artifact.stage3)?,
            )
        };

    let mut claims = stage1_claims;
    let mut perf = stage1_perf;
    claims.extend(stage2_claims);
    perf.accumulate(stage2_perf);
    if let Some((stage3_claim, stage3_perf)) = stage3_claim_and_perf {
        claims.push(stage3_claim);
        perf.accumulate(stage3_perf);
    }
    Ok((claims, perf))
}

pub fn build_rv64im_eval_claim_bundle_from_claim_witnesses(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imEvalClaimBundle, SimpleKernelError> {
    build_rv64im_eval_claim_bundle_from_claim_witnesses_with_validation(claim_witnesses, true)
}

pub(crate) fn build_rv64im_eval_claim_bundle_from_claim_witnesses_trusted_local(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imEvalClaimBundle, SimpleKernelError> {
    build_rv64im_eval_claim_bundle_from_claim_witnesses_with_validation(claim_witnesses, false)
}

fn build_rv64im_eval_claim_bundle_from_claim_witnesses_with_validation(
    claim_witnesses: &[FamilyEvalClaimWitness],
    validate_claims: bool,
) -> Result<Rv64imEvalClaimBundle, SimpleKernelError> {
    let mut accumulator = OpeningClaimAccumulator::default();
    for claim_witness in claim_witnesses {
        let claim = claim_witness.claim.clone();
        let result = if validate_claims {
            accumulator.insert(claim)
        } else {
            accumulator.insert_trusted_local(claim)
        };
        result.map_err(|err| phase0_build_error("phase0/claim_witness bundle build failed", err))?;
    }
    Ok(accumulator.into_bundle())
}

pub fn build_rv64im_eval_claim_bundle_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imEvalClaimBundle, SimpleKernelError> {
    let claim_witnesses = build_rv64im_eval_claim_witnesses_from_accepted_artifact(artifact)?;
    build_rv64im_eval_claim_bundle_from_claim_witnesses(&claim_witnesses)
}

pub fn verify_rv64im_eval_claim_bundle_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
    bundle: &Rv64imEvalClaimBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Phase 0 eval-claim bundle digest mismatch".into(),
        ));
    }
    let expected = build_rv64im_eval_claim_bundle_from_accepted_artifact(artifact)?;
    if &expected != bundle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Phase 0 eval-claim bundle does not match the accepted artifact".into(),
        ));
    }
    Ok(())
}

fn phase0_build_error(prefix: &str, err: EvalClaimError) -> SimpleKernelError {
    SimpleKernelError::Bridge(format!("{prefix}: {err}"))
}

fn phase0_commitment_context_for_schema(schema: FamilyEvalSchemaId) -> CommitmentContextId {
    CommitmentContextId::new(
        phase0_pp_seed_digest(EXACT_STAGE_PP_SEED),
        phase0_module_shape_digest(D as u64, schema.packed_column_count() as u64),
    )
}

fn phase0_pp_seed_digest(seed: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/pp_seed");
    tr.append_message(b"neo.fold.next/rv64im/opening_convergence/phase0/pp_seed/value", &seed);
    tr.digest32()
}

fn phase0_module_shape_digest(d: u64, packed_column_count: u64) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/module_shape");
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase0/module_shape/value",
        &[d, packed_column_count],
    );
    tr.digest32()
}

pub(crate) fn phase0_binding_digest(
    opened_object: &OpenedAjtaiObjectId,
    schema: FamilyEvalSchemaId,
    slot: u32,
    family_binding_anchor_digest: [u8; 32],
    stage_proof_binding_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/binding");
    tr.append_fields_raw(&digest32_as_fields(opened_object.digest));
    tr.append_fields_raw(&[F::from_u64(schema.tag()), F::from_u64(slot as u64)]);
    tr.append_fields_raw(&digest32_as_fields(family_binding_anchor_digest));
    tr.append_fields_raw(&digest32_as_fields(stage_proof_binding_digest));
    tr.digest32()
}

fn build_opened_object_witness_with_perf<const WORDS: usize, T>(
    schema: FamilyEvalSchemaId,
    rows: &[T],
    words_of: fn(&T) -> [u64; WORDS],
    context_label: &str,
) -> Result<(OpenedAjtaiObjectWitness, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    let commitment_context = phase0_commitment_context_for_schema(schema);
    let total_started = Instant::now();

    let started = Instant::now();
    let packed_columns = build_family_packed_columns(schema, rows, words_of, context_label)?;
    let packed_columns_ms = elapsed_ms(started);

    let started = Instant::now();
    let (commitment_vector, commitment_vector_perf) = build_commitment_vector_with_perf(schema, &packed_columns)?;
    let commitment_vector_ms = elapsed_ms(started);

    let started = Instant::now();
    let commitment_root_digest = phase0_commitment_root_digest(&commitment_vector);
    let commitment_root_ms = elapsed_ms(started);

    let row_domain_log_size = packed_columns
        .first()
        .map(|column| column.rows.len().trailing_zeros())
        .ok_or_else(|| SimpleKernelError::Bridge(format!("{context_label} missing packed columns")))?;

    let started = Instant::now();
    let opened_object = OpenedAjtaiObjectId::new(
        schema.family_kind(),
        &commitment_context,
        commitment_root_digest,
        PHASE0_OPENED_OBJECT_LAYOUT_V1,
        row_domain_log_size,
    );
    let opened_object_id_ms = elapsed_ms(started);

    let witness = OpenedAjtaiObjectWitness::new_trusted_local_with_commitment_root_digest(
        opened_object,
        commitment_context,
        packed_columns,
        commitment_vector,
    );
    Ok((
        witness,
        Rv64imEvalClaimWitnessBuildPerf {
            packed_columns_ms,
            commitment_vector_ms,
            commitment_params_ms: commitment_vector_perf.params_ms,
            commitment_committer_ms: commitment_vector_perf.committer_ms,
            commitment_mats_ms: commitment_vector_perf.mats_ms,
            commitment_commit_many_ms: commitment_vector_perf.commit_many_ms,
            commitment_root_ms,
            opened_object_id_ms,
            opened_object_total_ms: elapsed_ms(total_started),
            total_ms: elapsed_ms(total_started),
            ..Rv64imEvalClaimWitnessBuildPerf::default()
        },
    ))
}

fn build_family_packed_columns<const WORDS: usize, T>(
    schema: FamilyEvalSchemaId,
    rows: &[T],
    words_of: fn(&T) -> [u64; WORDS],
    context_label: &str,
) -> Result<Vec<PackedColumnOracleRef>, SimpleKernelError> {
    let full_width = phase0_full_width_for_schema(schema);
    let packed_column_count = schema.packed_column_count();
    let time_len = rows.len();
    let padded_time_len = time_len.max(1).next_power_of_two();
    let mut packed_columns = (0..packed_column_count)
        .map(|column_index| PackedColumnOracleRef {
            column_index: column_index as u32,
            rows: vec![std::array::from_fn(|_| F::ZERO); padded_time_len],
        })
        .collect::<Vec<_>>();

    for (time_index, row) in rows.iter().enumerate() {
        let field_evals = encode_words_to_field_evals_f(schema, &words_of(row))
            .map_err(|err| phase0_build_error(&format!("{context_label} row field-eval encoding failed"), err))?;
        if field_evals.len() != full_width {
            return Err(SimpleKernelError::Bridge(format!(
                "{context_label} field-eval width {} != frozen full width {}",
                field_evals.len(),
                full_width
            )));
        }
        for (logical_index, &value) in field_evals.iter().enumerate() {
            let column_index = logical_index / D;
            let coeff_index = logical_index % D;
            packed_columns[column_index].rows[time_index][coeff_index] = value;
        }
    }

    Ok(packed_columns)
}

fn build_claim_witnesses_for_slots_with_perf(
    schema: FamilyEvalSchemaId,
    witness: &OpenedAjtaiObjectWitness,
    slot_count: u32,
    family_binding_anchor_digest: [u8; 32],
    stage_proof_binding_digest: [u8; 32],
    context_label: &str,
) -> Result<(Vec<FamilyEvalClaimWitness>, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    let shared_witness = Arc::new(witness.clone());
    let total_started = Instant::now();
    let mut claims = Vec::with_capacity(slot_count as usize);
    let mut perf = Rv64imEvalClaimWitnessBuildPerf::default();
    for slot in 0..slot_count {
        let started = Instant::now();
        let binding_digest = phase0_binding_digest(
            &shared_witness.opened_object,
            schema,
            slot,
            family_binding_anchor_digest,
            stage_proof_binding_digest,
        );
        perf.binding_digest_ms += elapsed_ms(started);

        let started = Instant::now();
        let point = derive_phase0_point(
            &shared_witness.opened_object,
            &shared_witness.commitment_context,
            schema,
            slot,
            binding_digest,
        );
        perf.point_derivation_ms += elapsed_ms(started);

        let started = Instant::now();
        let payload = FamilyEvalPayload::new(
            schema,
            shared_witness
                .evaluate_payload(&point)
                .map_err(|err| phase0_build_error(&format!("{context_label} payload evaluation failed"), err))?,
        )
        .map_err(|err| phase0_build_error(&format!("{context_label} payload build failed"), err))?;
        perf.payload_eval_ms += elapsed_ms(started);

        let started = Instant::now();
        let claim = FamilyEvalClaim::new(
            shared_witness.opened_object.clone(),
            slot,
            shared_witness.commitment_context,
            point,
            payload,
            binding_digest,
        )
        .map_err(|err| phase0_build_error(&format!("{context_label} claim build failed"), err))?;
        perf.claim_build_ms += elapsed_ms(started);
        claims.push(FamilyEvalClaimWitness {
            claim,
            witness: shared_witness.clone(),
        });
    }
    perf.slot_claims_total_ms = elapsed_ms(total_started);
    perf.total_ms = elapsed_ms(total_started);
    Ok((claims, perf))
}

fn build_singleton_claim_witness_with_perf<const WORDS: usize, T>(
    schema: FamilyEvalSchemaId,
    rows: &[T],
    words_of: fn(&T) -> [u64; WORDS],
    family_binding_anchor_digest: [u8; 32],
    stage_proof_binding_digest: [u8; 32],
    context_label: &str,
) -> Result<(FamilyEvalClaimWitness, Rv64imEvalClaimWitnessBuildPerf), SimpleKernelError> {
    let (witness, mut perf) = build_opened_object_witness_with_perf(schema, rows, words_of, context_label)?;
    let (claims, slot_perf) = build_claim_witnesses_for_slots_with_perf(
        schema,
        &witness,
        1,
        family_binding_anchor_digest,
        stage_proof_binding_digest,
        context_label,
    )?;
    perf.accumulate(slot_perf);
    let claim = claims
        .into_iter()
        .next()
        .ok_or_else(|| SimpleKernelError::Bridge(format!("{context_label} missing singleton claim")))?;
    Ok((claim, perf))
}

fn maybe_build_singleton_claim_witness_with_perf<const WORDS: usize, T>(
    schema: FamilyEvalSchemaId,
    rows: &[T],
    words_of: fn(&T) -> [u64; WORDS],
    family_binding_anchor_digest: [u8; 32],
    stage_proof_binding_digest: [u8; 32],
    context_label: &str,
) -> Result<Option<(FamilyEvalClaimWitness, Rv64imEvalClaimWitnessBuildPerf)>, SimpleKernelError> {
    if rows.is_empty() {
        return Ok(None);
    }
    build_singleton_claim_witness_with_perf(
        schema,
        rows,
        words_of,
        family_binding_anchor_digest,
        stage_proof_binding_digest,
        context_label,
    )
    .map(Some)
}

fn assert_matching_digest(label: &str, expected: [u8; 32], actual: [u8; 32]) -> Result<(), SimpleKernelError> {
    if expected != actual {
        return Err(SimpleKernelError::Bridge(format!(
            "{label} mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

pub(crate) fn build_commitment_vector(
    schema: FamilyEvalSchemaId,
    packed_columns: &[PackedColumnOracleRef],
) -> Result<RealAjtaiCommitmentVector, SimpleKernelError> {
    let (commitments, _) = build_commitment_vector_with_perf(schema, packed_columns)?;
    Ok(commitments)
}

fn build_commitment_vector_with_perf(
    schema: FamilyEvalSchemaId,
    packed_columns: &[PackedColumnOracleRef],
) -> Result<(RealAjtaiCommitmentVector, Phase0CommitmentVectorBuildPerf), SimpleKernelError> {
    if packed_columns.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "phase0 commitment-vector build requires at least one packed column".into(),
        ));
    }
    let full_width = phase0_full_width_for_schema(schema);
    let padded_time_len = packed_columns[0].rows.len();

    let mut perf = Phase0CommitmentVectorBuildPerf::default();

    let started = Instant::now();
    let params = phase0_exact_stage_params(full_width)?;
    perf.params_ms = elapsed_ms(started);

    let started = Instant::now();
    let committer = phase0_family_committer(params.kappa as usize, padded_time_len)?;
    perf.committer_ms = elapsed_ms(started);

    let started = Instant::now();
    let mats = packed_columns
        .iter()
        .enumerate()
        .map(|(column_index, column)| packed_column_rows_to_mat(column, padded_time_len, column_index))
        .collect::<Result<Vec<_>, _>>()?;
    perf.mats_ms = elapsed_ms(started);

    let started = Instant::now();
    let mut commitments = Vec::with_capacity(mats.len());
    for chunk in mats.chunks(PHASE0_COMMITMENT_BATCH) {
        let refs: Vec<&Mat<F>> = chunk.iter().collect();
        commitments.extend(committer.commit_many(&refs));
    }
    perf.commit_many_ms = elapsed_ms(started);

    Ok((commitments, perf))
}

fn phase0_exact_stage_params(full_width: usize) -> Result<NeoParams, SimpleKernelError> {
    NeoParams::goldilocks_auto_r1cs_ccs(full_width)
        .map_err(|err| SimpleKernelError::Bridge(format!("phase0 exact-stage params failed: {err}")))
}

fn phase0_family_committer(kappa: usize, padded_time_len: usize) -> Result<AjtaiSModule, SimpleKernelError> {
    if has_global_pp_for_dims(D, padded_time_len) {
        if let Ok((registered_kappa, registered_seed)) = get_global_pp_seeded_params_for_dims(D, padded_time_len) {
            if registered_kappa != kappa || registered_seed != EXACT_STAGE_PP_SEED {
                return Err(SimpleKernelError::Bridge(format!(
                    "phase0 family committer PP mismatch for (d,m)=({D},{padded_time_len})"
                )));
            }
        } else {
            let pp = get_global_pp_for_dims(D, padded_time_len).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "phase0 family committer lookup failed for (d,m)=({D},{padded_time_len}): {err}"
                ))
            })?;
            if pp.kappa != kappa {
                return Err(SimpleKernelError::Bridge(format!(
                    "phase0 family committer kappa mismatch for (d,m)=({D},{padded_time_len})"
                )));
            }
        }
    } else {
        set_global_pp_seeded(D, kappa, padded_time_len, EXACT_STAGE_PP_SEED).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "phase0 family committer seed setup failed for (d,m)=({D},{padded_time_len}): {err}"
            ))
        })?;
    }

    AjtaiSModule::from_global_for_dims(D, padded_time_len).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "phase0 family committer init failed for (d,m)=({D},{padded_time_len}): {err}"
        ))
    })
}

fn packed_column_rows_to_mat(
    column: &PackedColumnOracleRef,
    padded_time_len: usize,
    column_index: usize,
) -> Result<Mat<F>, SimpleKernelError> {
    if column.rows.len() != padded_time_len {
        return Err(SimpleKernelError::Bridge(format!(
            "phase0 packed column {column_index} row length {} != padded time length {padded_time_len}",
            column.rows.len()
        )));
    }
    let mut mat = Mat::zero(D, padded_time_len, F::ZERO);
    for (time_index, coeffs) in column.rows.iter().enumerate() {
        for rho in 0..D {
            mat[(rho, time_index)] = coeffs[rho];
        }
    }
    Ok(mat)
}

pub(crate) fn phase0_commitment_root_digest(commitment_vector: &[Commitment]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/commitment_root");
    tr.append_fields_raw(&[F::from_u64(commitment_vector.len() as u64)]);
    for commitment in commitment_vector {
        tr.append_fields_raw(&commitment.data);
    }
    tr.digest32()
}

fn evaluate_packed_column_rows_with_weights(
    rows: &[[F; D]],
    weights: &[K],
    column_index: usize,
) -> Result<PackedColumnEval, EvalClaimError> {
    if weights.len() != rows.len() {
        return Err(EvalClaimError::WitnessRowDomainLengthMismatch {
            column_index,
            expected: weights.len(),
            actual: rows.len(),
        });
    }

    let mut coeffs = std::array::from_fn(|_| K::ZERO);
    for (weight, row) in weights.iter().zip(rows.iter()) {
        for rho in 0..D {
            coeffs[rho] += *weight * K::from(row[rho]);
        }
    }
    Ok(PackedColumnEval { coeffs })
}
