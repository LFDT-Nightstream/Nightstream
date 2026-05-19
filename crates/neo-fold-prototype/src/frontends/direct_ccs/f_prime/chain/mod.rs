//! Owns the internal direct F' accumulator slot.
//!
//! The direct recursive carrier keeps this type private so public callers
//! cannot supply arbitrary exported F' authority. Authority is built only from
//! crate-owned latest-step advice or the size-gated exact verifier body.

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::Mat;
use neo_math::{D, F};

use super::super::adapter::{lower_sparse_r1cs_export_to_low_norm, DirectR1csLowNormLayout};
use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsIvcState, DirectCcsProgram};
use super::super::step::DirectCcsStep;
use super::r1cs::{DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeLowNormSourceR1csShape};
use super::verifier_body::{
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, measure_latest_direct_ccs_f_prime_verifier_body_if_small,
    DirectCcsFPrimeVerifierBodyShape, DIRECT_CCS_F_PRIME_VERIFIER_BODY_DEFAULT_MEASURE_ROW_LIMIT,
};
use super::{DirectCcsFPrimeNifsPayloadShape, DirectCcsNativeFPrimeAdvice};
use crate::prover::CommitmentMixers;

pub(crate) const DIRECT_CCS_F_PRIME_LOW_NORM_ENCODER_BLOCKER: &str =
    "missing low-norm enc(F') builder for the verifier-shaped direct F' body";
pub(crate) const DIRECT_CCS_F_PRIME_EXACT_ENCODER_SIZE_BLOCKER: &str =
    "verifier-shaped direct F' body exceeds the exact low-norm encoder size gate";
pub(crate) const DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS: usize = 8_192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectCcsFPrimeChainSummary {
    pub(crate) folded_r2_steps: u64,
    pub(crate) has_proof_authority: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectCcsFPrimeEncoderStatus {
    pub(crate) native_evaluator_available: bool,
    pub(crate) compact_image_digest: Option<[u8; 32]>,
    pub(crate) low_norm_source_available: bool,
    pub(crate) low_norm_source_len: usize,
    pub(crate) low_norm_source_digest: Option<[u8; 32]>,
    pub(crate) low_norm_source_r1cs_shape: Option<DirectCcsFPrimeLowNormSourceR1csShape>,
    pub(crate) low_norm_source_digest_count: usize,
    pub(crate) low_norm_source_u64_count: usize,
    pub(crate) low_norm_source_encoded_public_input_count: usize,
    pub(crate) low_norm_source_field_lane_count: usize,
    pub(crate) low_norm_source_construction2_commitment_fields: usize,
    pub(crate) nifs_payload_shape: Option<DirectCcsFPrimeNifsPayloadShape>,
    pub(crate) verifier_body_shape: Option<DirectCcsFPrimeVerifierBodyShape>,
    pub(crate) verifier_body_measure_skipped: bool,
    pub(crate) low_norm_relation_available: bool,
    pub(crate) blocker: Option<&'static str>,
}

struct DirectCcsFPrimeNativeArtifacts {
    compact_image_digest: Option<[u8; 32]>,
    low_norm_source_len: usize,
    low_norm_source_digest: Option<[u8; 32]>,
    low_norm_source_r1cs_shape: Option<DirectCcsFPrimeLowNormSourceR1csShape>,
    low_norm_source_digest_count: usize,
    low_norm_source_u64_count: usize,
    low_norm_source_encoded_public_input_count: usize,
    low_norm_source_field_lane_count: usize,
    low_norm_source_construction2_commitment_fields: usize,
    nifs_payload_shape: Option<DirectCcsFPrimeNifsPayloadShape>,
}

struct DirectCcsFPrimeAuthorityAvailability {
    relation_available: bool,
    blocker: Option<&'static str>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsFPrimeChain {
    state: Option<DirectCcsIvcState>,
    summary: Option<DirectCcsFPrimeChainSummary>,
}

impl DirectCcsFPrimeChain {
    pub(crate) fn new() -> Self {
        Self {
            state: None,
            summary: None,
        }
    }

    pub(crate) fn state(&self) -> Option<&DirectCcsIvcState> {
        self.state.as_ref()
    }

    pub(crate) fn summary(&self) -> Option<&DirectCcsFPrimeChainSummary> {
        self.summary.as_ref()
    }

    pub(crate) fn append_latest_step_authority_from_direct_state<MR, MB>(
        &self,
        direct: &DirectCcsIvcState,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let advice = DirectCcsNativeFPrimeAdvice::from_latest_state(direct)?;
        let source_r1cs = DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(
            &advice,
            direct.params().kappa as u64,
            1,
            direct.params().k_rho as u64,
        )?;
        if !source_r1cs.shape.has_proof_authority() {
            if let Some(exact) = self.append_exact_verifier_body_authority_if_small(direct, mixers)? {
                return Ok(exact);
            }
            return Ok(Self {
                state: self.state.clone(),
                summary: self.summary.clone(),
            });
        }
        let program = source_r1cs.to_direct_ccs_program(direct.params())?;
        let source_log = source_ajtai_module(direct.params().kappa as usize, program.structure().m.div_ceil(D))?;
        let step = source_r1cs.to_direct_ccs_step(&program, &source_log, "direct_f_prime_recursive_step")?;
        self.append_f_prime_source_step(program, source_log, step, mixers)
    }

    fn append_exact_verifier_body_authority_if_small<MR, MB>(
        &self,
        direct: &DirectCcsIvcState,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<Option<Self>, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let Some(shape) = measure_latest_direct_ccs_f_prime_verifier_body_if_small(direct)? else {
            return Ok(None);
        };
        if shape.constraints > DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS {
            return Ok(None);
        }

        let export = export_latest_direct_ccs_f_prime_verifier_body_r1cs(direct)?;
        let layout = DirectR1csLowNormLayout::conservative_for_export(&export);
        let lowered = lower_sparse_r1cs_export_to_low_norm(&export, &layout)?;
        let program = lowered.to_direct_ccs_program()?;
        let source_log = source_ajtai_module(direct.params().kappa as usize, program.structure().m.div_ceil(D))?;
        let step = lowered.into_direct_ccs_step(&program, &source_log, "direct_f_prime_exact_verifier_body")?;
        self.append_f_prime_source_step(program, source_log, step, mixers)
            .map(Some)
    }

    fn append_f_prime_source_step<MR, MB>(
        &self,
        program: DirectCcsProgram,
        source_log: AjtaiSModule,
        step: DirectCcsStep,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let base_state = match self.state.as_ref() {
            Some(state) => state.clone(),
            None => DirectCcsIvcState::start(program)?,
        };
        let state = base_state.append_step(step, &source_log, mixers)?;
        let folded_r2_steps = self
            .summary
            .as_ref()
            .map_or(0, |summary| summary.folded_r2_steps)
            + 1;
        Ok(Self {
            state: Some(state),
            summary: Some(DirectCcsFPrimeChainSummary {
                folded_r2_steps,
                has_proof_authority: true,
            }),
        })
    }
}

fn source_ajtai_module(kappa: usize, cols: usize) -> Result<AjtaiSModule, DirectCcsFPrimeSnarkError> {
    if cols == 0 {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' source Ajtai module requires nonzero columns".into(),
        ));
    }
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4443_4353_4650_5255_u64.to_le_bytes());
        set_global_pp_seeded(D, kappa, cols, seed)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("direct F' source Ajtai setup failed: {err}")))?;
    }
    AjtaiSModule::from_global_for_dims(D, cols)
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("direct F' source Ajtai module failed: {err}")))
}

impl DirectCcsFPrimeEncoderStatus {
    pub(crate) fn from_direct_state(state: &DirectCcsIvcState, encoder_required: bool) -> Self {
        Self::from_direct_state_with_verifier_body_measurement(state, encoder_required, false)
    }

    pub(crate) fn from_direct_state_with_verifier_body_measurement(
        state: &DirectCcsIvcState,
        encoder_required: bool,
        measure_verifier_body: bool,
    ) -> Self {
        let native_advice = DirectCcsNativeFPrimeAdvice::from_latest_state(state);
        let verifier_body_shape = if measure_verifier_body {
            measure_latest_direct_ccs_f_prime_verifier_body_if_small(state)
                .ok()
                .flatten()
        } else {
            None
        };
        let verifier_body_measure_skipped = verifier_body_shape.is_none()
            && measure_verifier_body
            && native_advice.is_ok()
            && state.structure().n > DIRECT_CCS_F_PRIME_VERIFIER_BODY_DEFAULT_MEASURE_ROW_LIMIT;
        let native = DirectCcsFPrimeNativeArtifacts::from_advice_result(state, native_advice);
        let authority = DirectCcsFPrimeAuthorityAvailability::from_native_and_verifier_body(
            &native,
            verifier_body_shape.as_ref(),
            encoder_required,
        );
        Self {
            native_evaluator_available: native.compact_image_digest.is_some(),
            compact_image_digest: native.compact_image_digest,
            low_norm_source_available: native.low_norm_source_digest.is_some(),
            low_norm_source_len: native.low_norm_source_len,
            low_norm_source_digest: native.low_norm_source_digest,
            low_norm_source_r1cs_shape: native.low_norm_source_r1cs_shape,
            low_norm_source_digest_count: native.low_norm_source_digest_count,
            low_norm_source_u64_count: native.low_norm_source_u64_count,
            low_norm_source_encoded_public_input_count: native.low_norm_source_encoded_public_input_count,
            low_norm_source_field_lane_count: native.low_norm_source_field_lane_count,
            low_norm_source_construction2_commitment_fields: native.low_norm_source_construction2_commitment_fields,
            nifs_payload_shape: native.nifs_payload_shape,
            verifier_body_shape,
            verifier_body_measure_skipped,
            low_norm_relation_available: authority.relation_available,
            blocker: authority.blocker,
        }
    }
}

impl DirectCcsFPrimeAuthorityAvailability {
    fn from_native_and_verifier_body(
        native: &DirectCcsFPrimeNativeArtifacts,
        verifier_body_shape: Option<&DirectCcsFPrimeVerifierBodyShape>,
        encoder_required: bool,
    ) -> Self {
        let exact_verifier_body_available = verifier_body_shape
            .is_some_and(|shape| shape.constraints <= DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS);
        let relation_available = native.low_norm_source_has_proof_authority() || exact_verifier_body_available;
        let blocker = (encoder_required && !relation_available).then_some(
            if verifier_body_shape
                .is_some_and(|shape| shape.constraints > DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS)
            {
                DIRECT_CCS_F_PRIME_EXACT_ENCODER_SIZE_BLOCKER
            } else {
                DIRECT_CCS_F_PRIME_LOW_NORM_ENCODER_BLOCKER
            },
        );
        Self {
            relation_available,
            blocker,
        }
    }
}

impl DirectCcsFPrimeNativeArtifacts {
    fn low_norm_source_has_proof_authority(&self) -> bool {
        self.low_norm_source_r1cs_shape
            .is_some_and(|shape| shape.has_proof_authority())
    }

    fn from_advice_result(
        state: &DirectCcsIvcState,
        native_advice: Result<DirectCcsNativeFPrimeAdvice, DirectCcsFPrimeSnarkError>,
    ) -> Self {
        let Ok(advice) = native_advice else {
            return Self::empty();
        };

        let compact_image_digest = advice.compact_image().expected_digest().ok();
        let source = advice.low_norm_source_image().ok();
        let low_norm_source_len = source.as_ref().map_or(0, |source| source.len());
        let low_norm_source_digest = source.as_ref().map(|source| source.expected_digest());
        let low_norm_source_r1cs_shape = source
            .as_ref()
            .map(DirectCcsFPrimeLowNormSourceR1csShape::from_source_with_authority_estimate);
        let low_norm_source_digest_count = source.as_ref().map_or(0, |source| source.digest_count());
        let low_norm_source_u64_count = source.as_ref().map_or(0, |source| source.u64_count());
        let low_norm_source_encoded_public_input_count = source
            .as_ref()
            .map_or(0, |source| source.encoded_public_input_count());
        let low_norm_source_field_lane_count = source
            .as_ref()
            .map_or(0, |source| source.field_lane_count());
        let low_norm_source_construction2_commitment_fields = source
            .as_ref()
            .map_or(0, |source| source.construction2_commitment_fields());
        let nifs_payload_shape = DirectCcsFPrimeNifsPayloadShape::from_latest_state(state).ok();

        Self {
            compact_image_digest,
            low_norm_source_len,
            low_norm_source_digest,
            low_norm_source_r1cs_shape,
            low_norm_source_digest_count,
            low_norm_source_u64_count,
            low_norm_source_encoded_public_input_count,
            low_norm_source_field_lane_count,
            low_norm_source_construction2_commitment_fields,
            nifs_payload_shape,
        }
    }

    fn empty() -> Self {
        Self {
            compact_image_digest: None,
            low_norm_source_len: 0,
            low_norm_source_digest: None,
            low_norm_source_r1cs_shape: None,
            low_norm_source_digest_count: 0,
            low_norm_source_u64_count: 0,
            low_norm_source_encoded_public_input_count: 0,
            low_norm_source_field_lane_count: 0,
            low_norm_source_construction2_commitment_fields: 0,
            nifs_payload_shape: None,
        }
    }
}

impl Default for DirectCcsFPrimeChain {
    fn default() -> Self {
        Self::new()
    }
}
