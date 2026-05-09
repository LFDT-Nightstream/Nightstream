//! Owns RV32IM stage-claim surfaces and compact selected-claim stage packages.

use neo_ajtai::{
    get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, set_global_pp_seeded,
    AjtaiSModule,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::proof::{FoldSchedule, PackagedProof, PublicStep, StepInput};
use crate::rv32im::stage1::Stage1Summary;
use crate::rv32im::stage2::Stage2Summary;
use crate::rv32im::stage3::Stage3Summary;
use crate::session::{prove_and_package, verify_packaged};
use crate::vm::r1cs_builder::R1csBuilder;
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};

use super::{
    artifacts::Rv32imKernelSummary,
    canonical_openings::{AjtaiFamilyKind, AjtaiObjectId, AjtaiOpeningId, SelectedOpeningRef},
    perf_diagnostics::{
        ExactStageVectorBuildPerf, KernelOpeningBundleBuildPerf, KernelOpeningBundleVerifyPerf,
        PackagedOpeningBuildPerf, StageClaimBundleBuildPerf,
    },
    simple::{rv32im_ajtai_mixers, SimpleKernelError, SimpleKernelKernelClaimBundle, EXACT_STAGE_PP_SEED},
    simple_openings::{
        KernelBindingOpeningClaim, KernelBindingOpeningPoints, KernelBindingPackagedOpeningProof,
        KernelPreparedStepOpeningClaim, KernelPreparedStepOpeningPoints, KernelPreparedStepPackagedOpeningProof,
        SimpleKernelOpeningBundle, SimpleKernelOpeningClaim, SimpleKernelStagePackageBundle,
        Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2PackagedOpeningProof, Stage2SelectedOpeningClaim,
        Stage3PackagedOpeningProof, Stage3SelectedOpeningClaim,
    },
    stage1_canonical::{build_stage1_artifact_parts, Stage1CanonicalRowBundle},
    stage2_canonical::{build_stage2_artifact_parts, Stage2CanonicalFamilyBundle},
    stage3_canonical::{build_stage3_artifact_parts, Stage3CanonicalContinuityBundle},
    RootLaneCommitmentArtifact, RootLaneCommitmentSummaryArtifact, VerifiedTranscriptSurface,
};

pub(crate) const RV32IM_SELECTED_OPENING_LAYOUT_V1: u64 = 1;
const EXACT_VECTOR_PACKAGE_LIMB_BITS: u32 = 16;
const EXACT_VECTOR_PACKAGE_LIMB_COUNT: usize = 64 / EXACT_VECTOR_PACKAGE_LIMB_BITS as usize;
const EXACT_VECTOR_PACKAGE_K_RHO: u32 = 24;
const EXACT_VECTOR_PACKAGE_B: u64 = 1 << EXACT_VECTOR_PACKAGE_K_RHO;

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1ClaimSurface {
    pub row_count: usize,
    pub effect_row_count: usize,
    pub commit_row_count: usize,
    pub real_row_count: usize,
    pub preserves_x0_count: usize,
    pub mix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2ClaimSurface {
    pub register_read_count: usize,
    pub register_write_count: usize,
    pub ram_event_count: usize,
    pub twist_link_count: usize,
    pub ram_read_count: usize,
    pub ram_write_count: usize,
    pub reg_mix: u64,
    pub ram_mix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3ClaimSurface {
    pub continuity_count: usize,
    pub final_step_count: usize,
    pub halted: bool,
    pub all_continuity_hold: bool,
    pub continuity_mix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TranscriptClaimSurface {
    pub final_digest: [u8; 32],
    pub event_count: usize,
    pub kernel_final_mix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StageDigestCommitment {
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1ArtifactSurface {
    pub rows: Stage1CanonicalRowBundle,
    pub claim: Stage1ClaimSurface,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2ArtifactSurface {
    pub families: Stage2CanonicalFamilyBundle,
    pub claim: Stage2ClaimSurface,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3ArtifactSurface {
    pub continuity: Stage3CanonicalContinuityBundle,
    pub claim: Stage3ClaimSurface,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TranscriptArtifactSurface {
    pub commitment: StageDigestCommitment,
    pub claim: TranscriptClaimSurface,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelStageClaimBundle {
    pub stage1: Stage1ArtifactSurface,
    pub stage2: Stage2ArtifactSurface,
    pub stage3: Stage3ArtifactSurface,
    pub transcript: TranscriptArtifactSurface,
    pub execution_digest: [u8; 32],
    pub digest: [u8; 32],
}

struct ExactVectorPackageContext {
    params: NeoParams,
    log: AjtaiSModule,
    ccs: neo_ccs::CcsStructure<F>,
}

static EXACT_VECTOR_PACKAGE_CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<ExactVectorPackageContext>>>> = OnceLock::new();

fn split_u64_to_fields(value: u64, out: &mut Vec<F>) {
    const LIMB_MASK: u64 = (1u64 << EXACT_VECTOR_PACKAGE_LIMB_BITS) - 1;
    for shift in (0..64).step_by(EXACT_VECTOR_PACKAGE_LIMB_BITS as usize) {
        out.push(F::from_u64((value >> shift) & LIMB_MASK));
    }
}

fn u64_vector_to_field_limbs(values: &[u64]) -> Vec<F> {
    let mut out = Vec::with_capacity(values.len() * EXACT_VECTOR_PACKAGE_LIMB_COUNT);
    for &value in values {
        split_u64_to_fields(value, &mut out);
    }
    out
}

fn same_public_step(lhs: &PublicStep, rhs: &PublicStep) -> bool {
    lhs.label == rhs.label
        && lhs.mcs.m_in == rhs.mcs.m_in
        && lhs.mcs.x == rhs.mcs.x
        && lhs.mcs.c.d == rhs.mcs.c.d
        && lhs.mcs.c.kappa == rhs.mcs.c.kappa
        && lhs.mcs.c.data == rhs.mcs.c.data
}

impl StageDigestCommitment {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_digest_commitment");
        tr.append_message(b"rv32im/stage_digest_commitment/digest", &self.digest);
        tr.digest32()
    }
}

impl Stage1ClaimSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_claim_surface");
        tr.append_u64s(
            b"rv32im/stage1_claim_surface/counts",
            &[
                self.row_count as u64,
                self.effect_row_count as u64,
                self.commit_row_count as u64,
                self.real_row_count as u64,
                self.preserves_x0_count as u64,
                self.mix,
            ],
        );
        tr.digest32()
    }
}

impl Stage2ClaimSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_claim_surface");
        tr.append_u64s(
            b"rv32im/stage2_claim_surface/counts",
            &[
                self.register_read_count as u64,
                self.register_write_count as u64,
                self.ram_event_count as u64,
                self.twist_link_count as u64,
                self.ram_read_count as u64,
                self.ram_write_count as u64,
                self.reg_mix,
                self.ram_mix,
            ],
        );
        tr.digest32()
    }
}

impl Stage3ClaimSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_claim_surface");
        tr.append_u64s(
            b"rv32im/stage3_claim_surface/counts",
            &[
                self.continuity_count as u64,
                self.final_step_count as u64,
                self.continuity_mix,
            ],
        );
        tr.append_u64s(
            b"rv32im/stage3_claim_surface/flags",
            &[self.halted as u64, self.all_continuity_hold as u64],
        );
        tr.digest32()
    }
}

impl TranscriptClaimSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/transcript_claim_surface");
        tr.append_message(b"rv32im/transcript_claim_surface/final_digest", &self.final_digest);
        tr.append_u64s(
            b"rv32im/transcript_claim_surface/meta",
            &[self.event_count as u64, self.kernel_final_mix],
        );
        tr.digest32()
    }
}

impl Stage1ArtifactSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_artifact_surface");
        tr.append_message(b"rv32im/stage1_artifact_surface/rows", &self.rows.expected_digest());
        tr.append_message(b"rv32im/stage1_artifact_surface/claim", &self.claim.expected_digest());
        tr.digest32()
    }
}

impl Stage2ArtifactSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_artifact_surface");
        tr.append_message(
            b"rv32im/stage2_artifact_surface/families",
            &self.families.expected_digest(),
        );
        tr.append_message(b"rv32im/stage2_artifact_surface/claim", &self.claim.expected_digest());
        tr.digest32()
    }
}

impl Stage3ArtifactSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_artifact_surface");
        tr.append_message(
            b"rv32im/stage3_artifact_surface/continuity",
            &self.continuity.expected_digest(),
        );
        tr.append_message(b"rv32im/stage3_artifact_surface/claim", &self.claim.expected_digest());
        tr.digest32()
    }
}

impl TranscriptArtifactSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/transcript_artifact_surface");
        tr.append_message(
            b"rv32im/transcript_artifact_surface/commitment",
            &self.commitment.expected_digest(),
        );
        tr.append_message(
            b"rv32im/transcript_artifact_surface/claim",
            &self.claim.expected_digest(),
        );
        tr.digest32()
    }
}

impl SimpleKernelStageClaimBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_claim_bundle");
        tr.append_message(b"rv32im/stage_claim_bundle/stage1", &self.stage1.expected_digest());
        tr.append_message(b"rv32im/stage_claim_bundle/stage2", &self.stage2.expected_digest());
        tr.append_message(b"rv32im/stage_claim_bundle/stage3", &self.stage3.expected_digest());
        tr.append_message(
            b"rv32im/stage_claim_bundle/transcript",
            &self.transcript.expected_digest(),
        );
        tr.append_message(b"rv32im/stage_claim_bundle/execution_digest", &self.execution_digest);
        tr.digest32()
    }
}

impl ExactVectorPackageContext {
    fn new(logical_width: usize, seed: [u8; 32], label: &str) -> Result<Self, SimpleKernelError> {
        let full_width = logical_width
            .checked_add(1)
            .ok_or_else(|| SimpleKernelError::Bridge(format!("{label} exact package width overflow")))?;
        let mut params = NeoParams::goldilocks_auto_r1cs_ccs(full_width)
            .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package params failed: {err}")))?;
        params.k_rho = EXACT_VECTOR_PACKAGE_K_RHO;
        params.B = EXACT_VECTOR_PACKAGE_B;
        let m = commit_cols_for_full_width(full_width);
        let want_kappa = params.kappa as usize;
        if has_global_pp_for_dims(D, m) {
            if let Ok((kappa, registered_seed)) = get_global_pp_seeded_params_for_dims(D, m) {
                if kappa != want_kappa || registered_seed != seed {
                    return Err(SimpleKernelError::Bridge(format!(
                        "{label} exact package PP mismatch for (d,m)=({D},{m})"
                    )));
                }
            } else {
                let pp = get_global_pp_for_dims(D, m).map_err(|err| {
                    SimpleKernelError::Bridge(format!(
                        "{label} exact package PP lookup failed for (d,m)=({D},{m}): {err}"
                    ))
                })?;
                if pp.kappa != want_kappa {
                    return Err(SimpleKernelError::Bridge(format!(
                        "{label} exact package PP kappa mismatch for (d,m)=({D},{m})"
                    )));
                }
            }
        } else {
            set_global_pp_seeded(D, want_kappa, m, seed)
                .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package seed setup failed: {err}")))?;
        }
        let log = AjtaiSModule::from_global_for_dims(D, m)
            .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package module failed: {err}")))?;
        let mut builder = R1csBuilder::new(full_width, 0)
            .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package CCS builder failed: {err}")))?;
        builder.push_row([(0, F::ONE)], [(0, F::ONE)], [(0, F::ONE)]);
        let ccs = builder
            .build()
            .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package CCS build failed: {err}")))?;
        Ok(Self { params, log, ccs })
    }

    fn params(&self) -> &NeoParams {
        &self.params
    }

    fn log(&self) -> &AjtaiSModule {
        &self.log
    }

    fn ccs(&self) -> &neo_ccs::CcsStructure<F> {
        &self.ccs
    }
}

fn exact_vector_package_context(
    logical_width: usize,
    label: &str,
) -> Result<Arc<ExactVectorPackageContext>, SimpleKernelError> {
    let contexts = EXACT_VECTOR_PACKAGE_CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut contexts = contexts
        .lock()
        .map_err(|_| SimpleKernelError::Bridge(format!("{label} exact package context cache poisoned")))?;
    if let Some(context) = contexts.get(&logical_width) {
        return Ok(Arc::clone(context));
    }
    let context = Arc::new(ExactVectorPackageContext::new(
        logical_width,
        EXACT_STAGE_PP_SEED,
        label,
    )?);
    contexts.insert(logical_width, Arc::clone(&context));
    Ok(context)
}

fn build_exact_vector_package_step_with_context(
    label: &str,
    logical_values: &[F],
    context: &ExactVectorPackageContext,
) -> Result<StepInput, SimpleKernelError> {
    let mut full_vector = Vec::with_capacity(logical_values.len() + 1);
    full_vector.push(F::ONE);
    full_vector.extend_from_slice(logical_values);
    let packed = encode_vector_for_full_width(context.params(), full_vector.len(), &full_vector)
        .map_err(|err| SimpleKernelError::Bridge(format!("{label} exact package encoding failed: {err}")))?;
    Ok(StepInput {
        label: label.into(),
        mcs: neo_ccs::CcsClaim {
            c: context.log().commit(&packed),
            x: vec![F::ONE],
            m_in: 1,
        },
        witness: neo_ccs::CcsWitness {
            w: logical_values.to_vec(),
            Z: packed,
        },
    })
}

fn build_exact_vector_package_step(label: &str, logical_values: &[F]) -> Result<StepInput, SimpleKernelError> {
    let context = exact_vector_package_context(logical_values.len(), label)?;
    build_exact_vector_package_step_with_context(label, logical_values, context.as_ref())
}

pub(super) fn selected_opening_object(family: AjtaiFamilyKind, commitment_digest: [u8; 32]) -> AjtaiObjectId {
    AjtaiObjectId::new(family, commitment_digest, RV32IM_SELECTED_OPENING_LAYOUT_V1)
}

pub(super) fn selected_opening_ref(
    object: &AjtaiObjectId,
    logical_index: u64,
    value_digest: [u8; 32],
) -> SelectedOpeningRef {
    SelectedOpeningRef::new(AjtaiOpeningId::new(object.clone(), logical_index), value_digest)
}

pub(super) fn first_last_selected_refs<T>(
    items: &[T],
    object: &AjtaiObjectId,
    digest_fn: impl Fn(&T) -> [u8; 32],
) -> (Option<SelectedOpeningRef>, Option<SelectedOpeningRef>) {
    let Some(first) = items.first() else {
        return (None, None);
    };
    let first_ref = selected_opening_ref(object, 0, digest_fn(first));
    let last_index = items.len().saturating_sub(1) as u64;
    let last_ref = selected_opening_ref(object, last_index, digest_fn(items.last().unwrap_or(first)));
    (Some(first_ref), Some(last_ref))
}

fn build_kernel_binding_opening_package_step(
    claim: &KernelBindingOpeningClaim,
) -> Result<StepInput, SimpleKernelError> {
    let words = claim.claim_words();
    let logical_values = u64_vector_to_field_limbs(&words);
    build_exact_vector_package_step("rv32im/kernel_opening_bundle/bindings", &logical_values)
}

fn build_kernel_prepared_step_opening_package_step(
    claim: &KernelPreparedStepOpeningClaim,
) -> Result<StepInput, SimpleKernelError> {
    let words = claim.claim_words();
    let logical_values = u64_vector_to_field_limbs(&words);
    build_exact_vector_package_step("rv32im/kernel_opening_bundle/prepared_steps", &logical_values)
}

pub(crate) fn build_kernel_binding_opening_public_step(
    claim: &KernelBindingOpeningClaim,
) -> Result<PublicStep, SimpleKernelError> {
    Ok(build_kernel_binding_opening_package_step(claim)?.instance())
}

pub(crate) fn build_kernel_prepared_step_opening_public_step(
    claim: &KernelPreparedStepOpeningClaim,
) -> Result<PublicStep, SimpleKernelError> {
    Ok(build_kernel_prepared_step_opening_package_step(claim)?.instance())
}

fn packaged_single_step_matches(packaged: &PackagedProof, expected: &PublicStep) -> bool {
    packaged.statement.chunks.len() == 1
        && packaged.statement.chunks[0].start_index == 0
        && packaged.statement.chunks[0].steps.len() == 1
        && same_public_step(&packaged.statement.chunks[0].steps[0], expected)
}

fn verify_claim_packaged_cryptography(
    label: &str,
    words: &[u64],
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    let context_label = format!("{label}/selected_claim_package");
    let logical_values = u64_vector_to_field_limbs(words);
    let context = exact_vector_package_context(logical_values.len(), &context_label)?;
    verify_packaged(
        FoldingMode::Optimized,
        context.params(),
        context.ccs(),
        packaged,
        rv32im_ajtai_mixers(),
    )?;
    Ok(())
}

pub(crate) fn build_claim_packaged_proof(label: &str, words: &[u64]) -> Result<PackagedProof, SimpleKernelError> {
    let context_label = format!("{label}/selected_claim_package");
    let logical_values = u64_vector_to_field_limbs(words);
    let context = exact_vector_package_context(logical_values.len(), &context_label)?;
    let step = build_exact_vector_package_step_with_context(&context_label, &logical_values, context.as_ref())?;
    Ok(prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        context.params(),
        context.ccs(),
        [step],
        context.log(),
        rv32im_ajtai_mixers(),
    )?)
}

pub(crate) fn build_claim_packaged_public_step(label: &str, words: &[u64]) -> Result<PublicStep, SimpleKernelError> {
    let context_label = format!("{label}/selected_claim_package");
    let logical_values = u64_vector_to_field_limbs(words);
    let context = exact_vector_package_context(logical_values.len(), &context_label)?;
    let step = build_exact_vector_package_step_with_context(&context_label, &logical_values, context.as_ref())?;
    Ok(step.instance())
}

pub(crate) fn verify_claim_packaged_public_step(
    label: &str,
    words: &[u64],
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    let expected_step = build_claim_packaged_public_step(label, words)?;
    if !packaged_single_step_matches(packaged, &expected_step) {
        return Err(SimpleKernelError::Bridge(format!(
            "{label} selected-claim package public step mismatch"
        )));
    }
    Ok(())
}

pub(crate) fn verify_claim_packaged_proof(
    label: &str,
    words: &[u64],
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    verify_claim_packaged_public_step(label, words, packaged)?;
    verify_claim_packaged_cryptography(label, words, packaged)
}

pub(crate) fn verify_stage1_packaged_opening_public_step(
    stage_package: &Stage1PackagedOpeningProof,
    expected_claim: &Stage1SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    if stage_package.digest != stage_package.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage1 selected-claim package digest mismatch".into(),
        ));
    }
    if &stage_package.claim != expected_claim {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage1 selected-claim package claim mismatch".into(),
        ));
    }
    verify_claim_packaged_public_step("rv32im/stage1", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(crate) fn verify_stage2_packaged_opening_public_step(
    stage_package: &Stage2PackagedOpeningProof,
    expected_claim: &Stage2SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    if stage_package.digest != stage_package.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage2 selected-claim package digest mismatch".into(),
        ));
    }
    if &stage_package.claim != expected_claim {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage2 selected-claim package claim mismatch".into(),
        ));
    }
    verify_claim_packaged_public_step("rv32im/stage2", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(crate) fn verify_stage3_packaged_opening_public_step(
    stage_package: &Stage3PackagedOpeningProof,
    expected_claim: &Stage3SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    if stage_package.digest != stage_package.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage3 selected-claim package digest mismatch".into(),
        ));
    }
    if &stage_package.claim != expected_claim {
        return Err(SimpleKernelError::Bridge(
            "rv32im/stage3 selected-claim package claim mismatch".into(),
        ));
    }
    verify_claim_packaged_public_step("rv32im/stage3", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(super) fn verify_stage1_packaged_opening_proof(
    stage_package: &Stage1PackagedOpeningProof,
    expected_claim: &Stage1SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    verify_stage1_packaged_opening_public_step(stage_package, expected_claim)?;
    verify_claim_packaged_cryptography("rv32im/stage1", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(super) fn verify_stage2_packaged_opening_proof(
    stage_package: &Stage2PackagedOpeningProof,
    expected_claim: &Stage2SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    verify_stage2_packaged_opening_public_step(stage_package, expected_claim)?;
    verify_claim_packaged_cryptography("rv32im/stage2", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(super) fn verify_stage3_packaged_opening_proof(
    stage_package: &Stage3PackagedOpeningProof,
    expected_claim: &Stage3SelectedOpeningClaim,
) -> Result<(), SimpleKernelError> {
    verify_stage3_packaged_opening_public_step(stage_package, expected_claim)?;
    verify_claim_packaged_cryptography("rv32im/stage3", &expected_claim.claim_words(), &stage_package.packaged)
}

pub(crate) fn build_kernel_opening_claim_from_parts(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
    prepared_step_count: u64,
    first_prepared_step: Option<SelectedOpeningRef>,
    last_prepared_step: Option<SelectedOpeningRef>,
) -> SimpleKernelOpeningClaim {
    build_kernel_opening_claim_from_compact_stage_package_digests(
        stage_claims,
        stage_packages.digest,
        stage_packages.stage1.digest,
        stage_packages.stage2.digest,
        stage_packages.stage3.digest,
        prepared_step_bindings_digest,
        binding_count,
        first_binding_digest,
        last_binding_digest,
        execution_digest,
        final_state_digest,
        transcript_final_digest,
        final_pc,
        halted,
        prepared_step_count,
        first_prepared_step,
        last_prepared_step,
    )
}

fn stage_package_bundle_digest_from_digests(
    stage1_package_digest: [u8; 32],
    stage2_package_digest: [u8; 32],
    stage3_package_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_package_bundle");
    tr.append_message(b"rv32im/stage_package_bundle/stage1", &stage1_package_digest);
    tr.append_message(b"rv32im/stage_package_bundle/stage2", &stage2_package_digest);
    tr.append_message(b"rv32im/stage_package_bundle/stage3", &stage3_package_digest);
    tr.digest32()
}

fn build_kernel_opening_claim_from_compact_stage_package_digests(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_package_bundle_digest: [u8; 32],
    stage1_package_digest: [u8; 32],
    stage2_package_digest: [u8; 32],
    stage3_package_digest: [u8; 32],
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
    prepared_step_count: u64,
    first_prepared_step: Option<SelectedOpeningRef>,
    last_prepared_step: Option<SelectedOpeningRef>,
) -> SimpleKernelOpeningClaim {
    let binding_object = selected_opening_object(AjtaiFamilyKind::KernelBindings, prepared_step_bindings_digest);
    let first_binding = first_binding_digest.map(|digest| selected_opening_ref(&binding_object, 0, digest));
    let last_binding = last_binding_digest
        .map(|digest| selected_opening_ref(&binding_object, binding_count.saturating_sub(1), digest));
    let binding_claim = KernelBindingOpeningClaim {
        stage_claim_bundle_digest: stage_claims.digest,
        stage_package_bundle_digest,
        stage1_package_digest,
        stage2_package_digest,
        stage3_package_digest,
        prepared_step_bindings_digest,
        binding_count,
        stage1_row_count: stage_claims.stage1.claim.row_count as u64,
        stage2_register_read_count: stage_claims.stage2.claim.register_read_count as u64,
        stage2_register_write_count: stage_claims.stage2.claim.register_write_count as u64,
        stage2_ram_event_count: stage_claims.stage2.claim.ram_event_count as u64,
        stage3_continuity_count: stage_claims.stage3.claim.continuity_count as u64,
        points: KernelBindingOpeningPoints {
            first_binding,
            last_binding,
        },
        digest: [0; 32],
    };
    let prepared_step_claim = KernelPreparedStepOpeningClaim {
        execution_digest,
        final_state_digest,
        transcript_final_digest,
        prepared_step_count,
        final_pc,
        halted,
        points: KernelPreparedStepOpeningPoints {
            first_prepared_step,
            last_prepared_step,
        },
        digest: [0; 32],
    };
    let claim = SimpleKernelOpeningClaim {
        bindings: KernelBindingOpeningClaim {
            digest: binding_claim.expected_digest(),
            ..binding_claim
        },
        prepared_steps: KernelPreparedStepOpeningClaim {
            digest: prepared_step_claim.expected_digest(),
            ..prepared_step_claim
        },
        digest: [0; 32],
    };
    SimpleKernelOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

pub(crate) fn build_public_kernel_opening_claim_from_compact_surfaces(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage1_package_digest: [u8; 32],
    stage2_package_digest: [u8; 32],
    stage3_package_digest: [u8; 32],
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> SimpleKernelOpeningClaim {
    build_kernel_opening_claim_from_compact_stage_package_digests(
        stage_claims,
        stage_package_bundle_digest_from_digests(stage1_package_digest, stage2_package_digest, stage3_package_digest),
        stage1_package_digest,
        stage2_package_digest,
        stage3_package_digest,
        prepared_step_bindings_digest,
        binding_count,
        first_binding_digest,
        last_binding_digest,
        execution_digest,
        final_state_digest,
        transcript_final_digest,
        final_pc,
        halted,
        root_lane_commitment.time_len,
        root_lane_commitment.first_selected_row(),
        root_lane_commitment.last_selected_row(),
    )
}

fn build_kernel_opening_claim(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    prepared_step_count: u64,
    first_prepared_step: Option<SelectedOpeningRef>,
    last_prepared_step: Option<SelectedOpeningRef>,
) -> SimpleKernelOpeningClaim {
    build_kernel_opening_claim_from_parts(
        stage_claims,
        stage_packages,
        kernel_claims.prepared_step_bindings.digest,
        kernel_claims.prepared_step_bindings.binding_count,
        kernel_claims.prepared_step_bindings.first_binding_digest,
        kernel_claims.prepared_step_bindings.last_binding_digest,
        kernel_claims.kernel.execution_digest,
        kernel_claims.kernel.final_state_digest,
        kernel_claims.kernel.transcript_final_digest,
        u64::from(kernel_claims.kernel.final_pc),
        kernel_claims.kernel.halted,
        prepared_step_count,
        first_prepared_step,
        last_prepared_step,
    )
}

fn build_kernel_opening_claim_from_commitment(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentArtifact,
) -> SimpleKernelOpeningClaim {
    build_kernel_opening_claim(
        stage_claims,
        stage_packages,
        kernel_claims,
        root_lane_commitment.time_len,
        root_lane_commitment.first_selected_row(),
        root_lane_commitment.last_selected_row(),
    )
}

fn build_kernel_opening_claim_from_commitment_summary(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> SimpleKernelOpeningClaim {
    build_kernel_opening_claim(
        stage_claims,
        stage_packages,
        kernel_claims,
        root_lane_commitment.time_len,
        root_lane_commitment.first_selected_row(),
        root_lane_commitment.last_selected_row(),
    )
}

fn build_kernel_opening_proof_with_perf(
    claim: SimpleKernelOpeningClaim,
) -> Result<(SimpleKernelOpeningBundle, KernelOpeningBundleBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let bindings_step = build_kernel_binding_opening_package_step(&claim.bindings)?;
    let bindings_claim_words = bindings_step.witness.w.len();
    let bindings_context = ExactVectorPackageContext::new(
        bindings_step.witness.w.len(),
        EXACT_STAGE_PP_SEED,
        "rv32im/kernel_opening_bundle/bindings",
    )?;
    let bindings_started = Instant::now();
    let bindings_packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        bindings_context.params(),
        bindings_context.ccs(),
        [bindings_step],
        bindings_context.log(),
        rv32im_ajtai_mixers(),
    )?;
    let bindings = KernelBindingPackagedOpeningProof {
        claim: claim.bindings.clone(),
        packaged: bindings_packaged,
        digest: [0; 32],
    };
    let bindings = KernelBindingPackagedOpeningProof {
        digest: bindings.expected_digest(),
        ..bindings
    };

    let prepared_steps_step = build_kernel_prepared_step_opening_package_step(&claim.prepared_steps)?;
    let prepared_steps_claim_words = prepared_steps_step.witness.w.len();
    let prepared_steps_context = ExactVectorPackageContext::new(
        prepared_steps_step.witness.w.len(),
        EXACT_STAGE_PP_SEED,
        "rv32im/kernel_opening_bundle/prepared_steps",
    )?;
    let prepared_steps_started = Instant::now();
    let prepared_steps_packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        prepared_steps_context.params(),
        prepared_steps_context.ccs(),
        [prepared_steps_step],
        prepared_steps_context.log(),
        rv32im_ajtai_mixers(),
    )?;
    let prepared_steps = KernelPreparedStepPackagedOpeningProof {
        claim: claim.prepared_steps.clone(),
        packaged: prepared_steps_packaged,
        digest: [0; 32],
    };
    let prepared_steps = KernelPreparedStepPackagedOpeningProof {
        digest: prepared_steps.expected_digest(),
        ..prepared_steps
    };

    let bundle = SimpleKernelOpeningBundle {
        claim,
        bindings,
        prepared_steps,
        digest: [0; 32],
    };
    let bundle = SimpleKernelOpeningBundle {
        digest: bundle.expected_digest(),
        ..bundle
    };
    Ok((
        bundle,
        KernelOpeningBundleBuildPerf {
            bindings: PackagedOpeningBuildPerf {
                selected_labels: 2,
                claim_words: bindings_claim_words,
                package_ms: millis_since(bindings_started),
            },
            prepared_steps: PackagedOpeningBuildPerf {
                selected_labels: 2,
                claim_words: prepared_steps_claim_words,
                package_ms: millis_since(prepared_steps_started),
            },
            total_ms: millis_since(total_started),
        },
    ))
}

pub(crate) fn verify_kernel_opening_proof_public_steps(
    bundle: &SimpleKernelOpeningBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.claim.digest != bundle.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel opening claim digest mismatch".into(),
        ));
    }
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel opening bundle digest mismatch".into(),
        ));
    }
    if bundle.bindings.digest != bundle.bindings.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel binding opening package digest mismatch".into(),
        ));
    }
    if bundle.bindings.claim != bundle.claim.bindings {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel binding opening claim mismatch".into(),
        ));
    }
    let expected_bindings_step = build_kernel_binding_opening_package_step(&bundle.claim.bindings)?;
    if !packaged_single_step_matches(&bundle.bindings.packaged, &expected_bindings_step.instance()) {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel binding opening package public step mismatch".into(),
        ));
    }
    if bundle.prepared_steps.digest != bundle.prepared_steps.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel prepared-step opening package digest mismatch".into(),
        ));
    }
    if bundle.prepared_steps.claim != bundle.claim.prepared_steps {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel prepared-step opening claim mismatch".into(),
        ));
    }
    let expected_prepared_steps_step = build_kernel_prepared_step_opening_package_step(&bundle.claim.prepared_steps)?;
    if !packaged_single_step_matches(
        &bundle.prepared_steps.packaged,
        &expected_prepared_steps_step.instance(),
    ) {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel prepared-step opening package public step mismatch".into(),
        ));
    }
    Ok(())
}

fn verify_kernel_opening_proof_with_perf(
    bundle: &SimpleKernelOpeningBundle,
) -> Result<KernelOpeningBundleVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    verify_kernel_opening_proof_public_steps(bundle)?;
    let expected_bindings_step = build_kernel_binding_opening_package_step(&bundle.claim.bindings)?;
    let bindings_context = ExactVectorPackageContext::new(
        expected_bindings_step.witness.w.len(),
        EXACT_STAGE_PP_SEED,
        "rv32im/kernel_opening_bundle/bindings",
    )?;
    let bindings_started = Instant::now();
    verify_packaged(
        FoldingMode::Optimized,
        bindings_context.params(),
        bindings_context.ccs(),
        &bundle.bindings.packaged,
        rv32im_ajtai_mixers(),
    )?;

    let expected_prepared_steps_step = build_kernel_prepared_step_opening_package_step(&bundle.claim.prepared_steps)?;
    let prepared_steps_context = ExactVectorPackageContext::new(
        expected_prepared_steps_step.witness.w.len(),
        EXACT_STAGE_PP_SEED,
        "rv32im/kernel_opening_bundle/prepared_steps",
    )?;
    let prepared_steps_started = Instant::now();
    verify_packaged(
        FoldingMode::Optimized,
        prepared_steps_context.params(),
        prepared_steps_context.ccs(),
        &bundle.prepared_steps.packaged,
        rv32im_ajtai_mixers(),
    )?;
    Ok(KernelOpeningBundleVerifyPerf {
        claim_rebuild_ms: 0.0,
        bindings_ms: millis_since(bindings_started),
        prepared_steps_ms: millis_since(prepared_steps_started),
        total_ms: millis_since(total_started),
    })
}

pub(super) fn build_kernel_opening_bundle_with_perf(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentArtifact,
) -> Result<(SimpleKernelOpeningBundle, KernelOpeningBundleBuildPerf), SimpleKernelError> {
    build_kernel_opening_proof_with_perf(build_kernel_opening_claim_from_commitment(
        stage_claims,
        stage_packages,
        kernel_claims,
        root_lane_commitment,
    ))
}

pub(super) fn build_public_kernel_opening_bundle_with_perf(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> Result<(SimpleKernelOpeningBundle, KernelOpeningBundleBuildPerf), SimpleKernelError> {
    build_kernel_opening_proof_with_perf(build_kernel_opening_claim_from_commitment_summary(
        stage_claims,
        stage_packages,
        kernel_claims,
        root_lane_commitment,
    ))
}

pub(super) fn build_public_kernel_opening_bundle_from_export_parts_with_perf(
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> Result<(SimpleKernelOpeningBundle, KernelOpeningBundleBuildPerf), SimpleKernelError> {
    build_kernel_opening_proof_with_perf(build_kernel_opening_claim_from_parts(
        stage_claims,
        stage_packages,
        prepared_step_bindings_digest,
        binding_count,
        first_binding_digest,
        last_binding_digest,
        execution_digest,
        final_state_digest,
        transcript_final_digest,
        final_pc,
        halted,
        root_lane_commitment.time_len,
        root_lane_commitment.first_selected_row(),
        root_lane_commitment.last_selected_row(),
    ))
}

pub(super) fn verify_kernel_opening_bundle_with_perf(
    bundle: &SimpleKernelOpeningBundle,
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentArtifact,
) -> Result<KernelOpeningBundleVerifyPerf, SimpleKernelError> {
    let claim_started = Instant::now();
    let expected_claim =
        build_kernel_opening_claim_from_commitment(stage_claims, stage_packages, kernel_claims, root_lane_commitment);
    let claim_rebuild_ms = millis_since(claim_started);
    if bundle.claim != expected_claim {
        return Err(SimpleKernelError::Bridge("RV32IM kernel opening claim mismatch".into()));
    }
    let mut perf = verify_kernel_opening_proof_with_perf(bundle)?;
    perf.claim_rebuild_ms = claim_rebuild_ms;
    perf.total_ms += claim_rebuild_ms;
    Ok(perf)
}

pub(super) fn verify_public_kernel_opening_bundle_with_perf(
    bundle: &SimpleKernelOpeningBundle,
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    kernel_claims: &SimpleKernelKernelClaimBundle,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> Result<KernelOpeningBundleVerifyPerf, SimpleKernelError> {
    let claim_started = Instant::now();
    let expected_claim = build_kernel_opening_claim_from_commitment_summary(
        stage_claims,
        stage_packages,
        kernel_claims,
        root_lane_commitment,
    );
    let claim_rebuild_ms = millis_since(claim_started);
    if bundle.claim != expected_claim {
        return Err(SimpleKernelError::Bridge("RV32IM kernel opening claim mismatch".into()));
    }
    let mut perf = verify_kernel_opening_proof_with_perf(bundle)?;
    perf.claim_rebuild_ms = claim_rebuild_ms;
    perf.total_ms += claim_rebuild_ms;
    Ok(perf)
}

pub(crate) fn verify_public_kernel_opening_bundle_from_export_parts_with_perf(
    bundle: &SimpleKernelOpeningBundle,
    stage_claims: &SimpleKernelStageClaimBundle,
    stage_packages: &SimpleKernelStagePackageBundle,
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> Result<KernelOpeningBundleVerifyPerf, SimpleKernelError> {
    let claim_started = Instant::now();
    let expected_claim = build_kernel_opening_claim_from_parts(
        stage_claims,
        stage_packages,
        prepared_step_bindings_digest,
        binding_count,
        first_binding_digest,
        last_binding_digest,
        execution_digest,
        final_state_digest,
        transcript_final_digest,
        final_pc,
        halted,
        root_lane_commitment.time_len,
        root_lane_commitment.first_selected_row(),
        root_lane_commitment.last_selected_row(),
    );
    let claim_rebuild_ms = millis_since(claim_started);
    if bundle.claim != expected_claim {
        return Err(SimpleKernelError::Bridge("RV32IM kernel opening claim mismatch".into()));
    }
    let mut perf = verify_kernel_opening_proof_with_perf(bundle)?;
    perf.claim_rebuild_ms = claim_rebuild_ms;
    perf.total_ms += claim_rebuild_ms;
    Ok(perf)
}

pub(super) fn build_stage_claim_bundle_from_parts(
    stage1_summary: &Stage1Summary,
    stage2_summary: &Stage2Summary,
    stage3_summary: &Stage3Summary,
    transcript_event_count: usize,
    kernel: &Rv32imKernelSummary,
) -> Result<SimpleKernelStageClaimBundle, SimpleKernelError> {
    Ok(build_stage_claim_bundle_from_parts_with_perf(
        stage1_summary,
        stage2_summary,
        stage3_summary,
        transcript_event_count,
        kernel,
    )?
    .0)
}

fn build_stage_claim_bundle_from_surface_parts_with_perf(
    stage1_summary: &Stage1Summary,
    stage2_summary: &Stage2Summary,
    stage3_summary: &Stage3Summary,
    transcript_event_count: usize,
    execution_digest: [u8; 32],
    stage1_mix: u64,
    stage2_reg_mix: u64,
    stage2_ram_mix: u64,
    stage3_continuity_mix: u64,
    kernel_final_mix: u64,
    transcript_final_digest: [u8; 32],
) -> Result<(SimpleKernelStageClaimBundle, StageClaimBundleBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();

    let stage1_perf = ExactStageVectorBuildPerf::default();
    let (stage1_rows, stage1_claim) = build_stage1_artifact_parts(stage1_summary, stage1_mix);
    let stage1 = Stage1ArtifactSurface {
        rows: stage1_rows,
        claim: stage1_claim,
    };

    let stage2_perf = ExactStageVectorBuildPerf::default();
    let (stage2_families, stage2_claim) = build_stage2_artifact_parts(stage2_summary, stage2_reg_mix, stage2_ram_mix);
    let stage2 = Stage2ArtifactSurface {
        families: stage2_families,
        claim: stage2_claim,
    };

    let (stage3_continuity, stage3_claim) = build_stage3_artifact_parts(stage3_summary, stage3_continuity_mix);
    let stage3 = Stage3ArtifactSurface {
        continuity: stage3_continuity,
        claim: stage3_claim,
    };

    let transcript = TranscriptArtifactSurface {
        commitment: StageDigestCommitment {
            digest: transcript_final_digest,
        },
        claim: TranscriptClaimSurface {
            final_digest: transcript_final_digest,
            event_count: transcript_event_count,
            kernel_final_mix,
        },
    };

    let claims = SimpleKernelStageClaimBundle {
        stage1,
        stage2,
        stage3,
        transcript,
        execution_digest,
        digest: [0; 32],
    };
    let claims = SimpleKernelStageClaimBundle {
        digest: claims.expected_digest(),
        ..claims
    };
    Ok((
        claims,
        StageClaimBundleBuildPerf {
            stage1: stage1_perf,
            stage2: stage2_perf,
            stage3: ExactStageVectorBuildPerf::default(),
            total_ms: millis_since(total_started),
        },
    ))
}

pub(super) fn build_stage_claim_bundle_from_parts_with_perf(
    stage1_summary: &Stage1Summary,
    stage2_summary: &Stage2Summary,
    stage3_summary: &Stage3Summary,
    transcript_event_count: usize,
    kernel: &Rv32imKernelSummary,
) -> Result<(SimpleKernelStageClaimBundle, StageClaimBundleBuildPerf), SimpleKernelError> {
    build_stage_claim_bundle_from_surface_parts_with_perf(
        stage1_summary,
        stage2_summary,
        stage3_summary,
        transcript_event_count,
        kernel.execution_digest,
        kernel.stage1_mix,
        kernel.stage2_reg_mix,
        kernel.stage2_ram_mix,
        kernel.stage3_continuity_mix,
        kernel.kernel_final_mix,
        kernel.transcript_final_digest,
    )
}

pub(super) fn build_stage_claim_bundle_from_export_parts(
    stage1_summary: &Stage1Summary,
    stage2_summary: &Stage2Summary,
    stage3_summary: &Stage3Summary,
    transcript: &VerifiedTranscriptSurface,
    execution_digest: [u8; 32],
) -> Result<SimpleKernelStageClaimBundle, SimpleKernelError> {
    Ok(build_stage_claim_bundle_from_export_parts_with_perf(
        stage1_summary,
        stage2_summary,
        stage3_summary,
        transcript,
        execution_digest,
    )?
    .0)
}

pub(super) fn build_stage_claim_bundle_from_export_parts_with_perf(
    stage1_summary: &Stage1Summary,
    stage2_summary: &Stage2Summary,
    stage3_summary: &Stage3Summary,
    transcript: &VerifiedTranscriptSurface,
    execution_digest: [u8; 32],
) -> Result<(SimpleKernelStageClaimBundle, StageClaimBundleBuildPerf), SimpleKernelError> {
    build_stage_claim_bundle_from_surface_parts_with_perf(
        stage1_summary,
        stage2_summary,
        stage3_summary,
        transcript.event_count,
        execution_digest,
        transcript.challenges.stage1_mix,
        transcript.challenges.stage2_reg_mix,
        transcript.challenges.stage2_ram_mix,
        transcript.challenges.stage3_continuity_mix,
        transcript.challenges.kernel_final_mix,
        transcript.final_digest,
    )
}
