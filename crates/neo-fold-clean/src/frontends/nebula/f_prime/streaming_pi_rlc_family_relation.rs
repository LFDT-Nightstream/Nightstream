//! Exact split relation for one production PiRLC family.
//!
//! The two parity bodies own the family-independent arithmetic, canonical
//! openings, shape pins, residual update, carry, cursor, and Poseidon2 replay.
//! Each of the 110 overlays owns only the family-position-dependent 108-row
//! seeded Phi81 map. The scheduled composer must link every digit field and
//! every commitment output field between the selected body and overlay.
//!
//! The 76,670-column rank-two map is outside the pinned estimator ceiling.
//! Its binding property remains an explicit Module-SIS assumption.

mod opening_rows;
mod retained_algebra;
mod retained_carry;
mod retained_links;
mod retained_overlay;
mod retained_residual;
mod row_ledger;

pub use opening_rows::{production_pi_rlc_family_body_opening_rows_audit, NebulaFPrimePiRlcBodyOpeningRowsAudit};
pub use retained_algebra::{
    production_pi_rlc_family_body_algebra_retained_audit, NebulaFPrimePiRlcBodyAlgebraRetainedAudit,
};
pub use retained_carry::{production_pi_rlc_family_body_carry_retained_audit, NebulaFPrimePiRlcBodyCarryRetainedAudit};
pub use retained_links::{
    production_pi_rlc_family_normalized_link_audit, NebulaFPrimePiRlcFamilyNormalizedLinkAudit,
    NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit,
};
pub use retained_overlay::{
    production_pi_rlc_family_overlay_retained_audit, NebulaFPrimePiRlcFamilyOverlayRetainedAudit,
};
pub use retained_residual::{
    production_pi_rlc_family_body_residual_retained_audit, NebulaFPrimePiRlcBodyResidualRetainedAudit,
};
pub use row_ledger::{
    production_pi_rlc_family_body_row_ledger, NebulaFPrimePiRlcBodyFixedEmittedRun, NebulaFPrimePiRlcBodyFixedFamily,
    NebulaFPrimePiRlcBodyRetainedRun, NebulaFPrimePiRlcBodyRewriteBatch, NebulaFPrimePiRlcBodyRewriteKind,
    NebulaFPrimePiRlcFamilyBodyRowLedger,
};

use std::ops::Range;

use neo_ajtai::{commit_row_major_seeded, seeded_pp_chunk_seeds, Commitment};
use neo_ccs::{Mat, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::lowering::normalized_field_assignment;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix,
    audit_multi_branch_selective_compiler_with_shared_bit_prefix,
    audit_multi_branch_selective_decoder_runs_with_shared_bit_prefix, lower_field_r1cs,
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix,
    project_rows_with_complete_source_provenance_with_alignment, FieldR1csLoweringError, LowNormR1csError,
    MultiBranchLowNormR1cs, OverlayFieldLink, OverlayKindLinks, SelectiveCompactLayoutAudit, SelectiveCompilerAudit,
    SelectiveProjectedDecoderRunProvenance, SelectiveProjectedRowsAudit, SparseR1cs,
};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::StateXOutDigestInputs;
use crate::paper::reductions::accumulator_sis_circuit::{
    alloc_zero_coordinate_word, balanced_ternary_digits, decompose_var_to_balanced_ternary,
    PI_RLC_INPUT_COORDINATE_SIS_CONFIG,
};
use crate::paper::relations::product_commitment_circuit::alloc_commitment;

use super::streaming_phase_envelope::{
    enforce_streaming_carry_phase_semantic_envelope, StreamingCarryPhaseSemanticEnvelope,
};
use super::streaming_pi_rlc_family_replay::{
    append_pi_rlc_family_replay, NebulaFPrimePiRlcFamilyReplayArmKind, NebulaFPrimePiRlcFamilyReplayCallAudit,
    ACTIVE_DIGIT_START, AFTER_CHALLENGE_START, AFTER_CURSOR_COLUMN, AFTER_RESIDUAL_START, ALGEBRA_CHALLENGE_START,
    ALGEBRA_COLUMNS, ALGEBRA_INPUT_START, ALGEBRA_OUTPUT_START, ALGEBRA_PRODUCT_COLUMNS, ALGEBRA_PRODUCT_START,
    BEFORE_CHALLENGE_START, BEFORE_CURSOR_COLUMN, BEFORE_RESIDUAL_START, COMMITMENT_OUTPUT_FIELDS,
    COMMITMENT_OUTPUT_START, DIGIT_COUNT, FAMILY_INPUT_FIELDS, INPUT_REPLAY_BEFORE_START, LANE_COUNT, OPENING_COLUMNS,
    OUTPUT_REPLAY_BEFORE_START, REPLAY_AUXILIARY_START, SHAPE_D_COLUMN, SHAPE_KAPPA_COLUMN, SOURCE_COLUMNS,
    SOURCE_COUNT, ZERO_DIGIT_START,
};
use super::streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit,
    FIRST_PI_RLC_FAMILY_PROGRAM_CURSOR,
};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;

pub const PI_RLC_FAMILY_COUNT: usize = 110;
pub const PI_RLC_GLOBAL_INPUT_FIELDS: usize = PI_RLC_FAMILY_COUNT * FAMILY_INPUT_FIELDS;
pub const PI_RLC_MESSAGE_COLUMNS: usize = PI_RLC_GLOBAL_INPUT_FIELDS * DIGIT_COUNT / D;

const PI_RLC_FAMILY_ALGEBRA_ROWS: usize = ALGEBRA_PRODUCT_COLUMNS + LANE_COUNT;
const PI_RLC_FAMILY_OPENING_ROWS: usize = DIGIT_COUNT + FAMILY_INPUT_FIELDS * 124;
const PI_RLC_FAMILY_CARRY_ROWS: usize = 2 * FAMILY_INPUT_FIELDS + 1;

pub const PI_RLC_FAMILY_BODY_SOURCE_ROWS: usize =
    PI_RLC_FAMILY_ALGEBRA_ROWS + PI_RLC_FAMILY_OPENING_ROWS + 2 + COMMITMENT_OUTPUT_FIELDS + PI_RLC_FAMILY_CARRY_ROWS;
pub const PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS: usize = PI_RLC_FAMILY_BODY_SOURCE_ROWS + 145_200;
pub const PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS: usize = PI_RLC_FAMILY_BODY_SOURCE_ROWS + 146_400;
pub const PI_RLC_FAMILY_BODY_EVEN_ROWS: usize = 1_300_897;
pub const PI_RLC_FAMILY_BODY_ODD_ROWS: usize = 1_302_097;
pub const PI_RLC_FAMILY_BODY_EVEN_COLUMNS: usize = 1_301_126;
pub const PI_RLC_FAMILY_BODY_ODD_COLUMNS: usize = 1_302_326;
pub const PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS: usize = 10 * PUBLIC_WORD_BITS;

pub const PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START: usize = 1;
pub const PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START: usize = PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START + DIGIT_COUNT;
pub const PI_RLC_FAMILY_OVERLAY_OUTPUT_START: usize =
    PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START + FAMILY_INPUT_FIELDS * DIGIT_COUNT;
pub const PI_RLC_FAMILY_OVERLAY_COLUMNS: usize = PI_RLC_FAMILY_OVERLAY_OUTPUT_START + COMMITMENT_OUTPUT_FIELDS;
pub const PI_RLC_FAMILY_OVERLAY_ROWS: usize = COMMITMENT_OUTPUT_FIELDS;
pub const PI_RLC_FAMILY_LINK_FIELDS: usize = DIGIT_COUNT + FAMILY_INPUT_FIELDS * DIGIT_COUNT + COMMITMENT_OUTPUT_FIELDS;

const SPONGE_WIDTH: usize = 8;
const PUBLIC_WORD_BITS: usize = 64;
const FAMILY_STATE_FIELDS: usize = 127 + FAMILY_INPUT_FIELDS;
const DIGEST_PIN_COUNT: usize = 13;
const STATE_X_OUT_PREIMAGE_FIELDS: usize = 32;
const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-pirlc-state/v1";
const STATE_DIGEST_FIELDS_LABEL: &[u8] = b"state";

/// One compact rectangular run of field links between a parity body and a
/// family overlay. Each outer item links `field_count` consecutive fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyOverlayLinkRun {
    phase_field_start: usize,
    overlay_field_start: usize,
    outer_count: usize,
    phase_stride: usize,
    overlay_stride: usize,
    field_count: usize,
}

impl NebulaFPrimePiRlcFamilyOverlayLinkRun {
    pub const fn phase_field_start(self) -> usize {
        self.phase_field_start
    }

    pub const fn overlay_field_start(self) -> usize {
        self.overlay_field_start
    }

    pub const fn outer_count(self) -> usize {
        self.outer_count
    }

    pub const fn phase_stride(self) -> usize {
        self.phase_stride
    }

    pub const fn overlay_stride(self) -> usize {
        self.overlay_stride
    }

    pub const fn field_count(self) -> usize {
        self.field_count
    }

    pub const fn link_count(self) -> usize {
        self.outer_count * self.field_count
    }

    fn fields(self) -> impl Iterator<Item = OverlayFieldLink> {
        (0..self.outer_count).flat_map(move |outer| {
            (0..self.field_count).map(move |field| OverlayFieldLink {
                phase_field: self.phase_field_start + outer * self.phase_stride + field,
                overlay_field: self.overlay_field_start + outer * self.overlay_stride + field,
            })
        })
    }
}

#[derive(Debug, Error)]
pub enum NebulaFPrimePiRlcFamilyRelationError {
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error("production PiRLC family-body row ledger: {0}")]
    RowLedger(&'static str),
    #[error("production PiRLC normalized algebra bridge: {0}")]
    AlgebraRetained(String),
    #[error("production PiRLC normalized carry bridge: {0}")]
    CarryRetained(String),
    #[error("production PiRLC normalized residual bridge: {0}")]
    ResidualRetained(String),
    #[error("production PiRLC normalized family-overlay bridge: {0}")]
    OverlayRetained(String),
    #[error("production PiRLC normalized body-overlay links: {0}")]
    NormalizedLinks(String),
    #[error("production PiRLC normalized opening rows: {0}")]
    OpeningRows(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyBodyShapeAudit {
    pub kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    pub source_rows: usize,
    pub rows: usize,
    pub columns: usize,
    pub source_columns: usize,
    pub public_columns: usize,
    pub replay_poseidon2_permutations: usize,
    pub poseidon2_permutations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyBodyLowNormShapeAudit {
    pub norm_base: u32,
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub total_coordinates: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyOverlayShapeAudit {
    pub family: usize,
    pub rows: usize,
    pub columns: usize,
    pub zero_digits: Range<usize>,
    pub active_digits: Range<usize>,
    pub outputs: Range<usize>,
}

pub struct NebulaFPrimePiRlcFamilyBodySynthesis {
    kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    fixture_family: usize,
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    zero_digit_columns: [usize; DIGIT_COUNT],
    active_digit_columns: Vec<usize>,
    commitment_output_columns: Vec<usize>,
    input_after_columns: [usize; 8],
    output_after_columns: [usize; 8],
    replay_call_audits: Vec<NebulaFPrimePiRlcFamilyReplayCallAudit>,
    before_state_field_columns: Vec<usize>,
    after_state_field_columns: Vec<usize>,
    after_digest_pin_columns: [usize; DIGEST_PIN_COUNT],
    before_digest_pin_columns: [usize; DIGEST_PIN_COUNT],
    after_x_out_preimage_columns: [usize; STATE_X_OUT_PREIMAGE_FIELDS],
    before_x_out_preimage_columns: [usize; STATE_X_OUT_PREIMAGE_FIELDS],
    after_x_out_digest_columns: [usize; 4],
    before_x_out_digest_columns: [usize; 4],
    phase_envelope: StreamingCarryPhaseSemanticEnvelope,
}

impl NebulaFPrimePiRlcFamilyBodySynthesis {
    pub fn production(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> Self {
        let fixture_family = match kind {
            NebulaFPrimePiRlcFamilyReplayArmKind::Even => 0,
            NebulaFPrimePiRlcFamilyReplayArmKind::Odd => 1,
        };
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();

        let challenge_values = (0..FAMILY_INPUT_FIELDS)
            .map(|offset| {
                let source = offset / LANE_COUNT;
                let lane = offset % LANE_COUNT;
                F::from_usize((source + lane) % 5)
            })
            .collect::<Vec<_>>();
        let input_values = production_input_values();
        let (product_values, output_values) = product_witness(&challenge_values, &input_values);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.algebra");
        let challenge_vars = builder.alloc_vec(&challenge_values);
        let input_vars = builder.alloc_vec(&input_values);
        let output_vars = builder.alloc_vec(&output_values);
        let product_vars = builder.alloc_vec(&product_values);
        assert_columns(&challenge_vars, ALGEBRA_CHALLENGE_START);
        assert_columns(&input_vars, ALGEBRA_INPUT_START);
        assert_columns(&output_vars, ALGEBRA_OUTPUT_START);
        assert_columns(&product_vars, ALGEBRA_PRODUCT_START);
        assert_eq!(builder.cols(), ALGEBRA_COLUMNS);
        enforce_algebra_rows(&mut builder, &challenge_vars, &input_vars, &output_vars, &product_vars);
        assert_eq!(builder.rows(), PI_RLC_FAMILY_ALGEBRA_ROWS);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.openings");
        let zero_word = alloc_zero_coordinate_word(&mut builder);
        let zero_digit_columns = zero_word.map(Var::col);
        assert_eq!(zero_digit_columns[0], ZERO_DIGIT_START);
        let mut active_digit_columns = Vec::with_capacity(FAMILY_INPUT_FIELDS * DIGIT_COUNT);
        for (offset, &input) in input_vars.iter().enumerate() {
            let word = decompose_var_to_balanced_ternary(&mut builder, input);
            assert_eq!(word[0].col(), ACTIVE_DIGIT_START + offset * OPENING_COLUMNS);
            active_digit_columns.extend(word.map(Var::col));
        }
        assert_eq!(builder.cols(), SHAPE_D_COLUMN);
        assert_eq!(builder.rows(), PI_RLC_FAMILY_ALGEBRA_ROWS + PI_RLC_FAMILY_OPENING_ROWS);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.commitment_shape");
        let native_commitment = native_family_commitment(fixture_family, &input_values);
        let commitment = alloc_commitment(&mut builder, &native_commitment);
        assert_eq!(commitment.d_var.col(), SHAPE_D_COLUMN);
        assert_eq!(commitment.kappa_var.col(), SHAPE_KAPPA_COLUMN);
        assert_columns(&commitment.data, COMMITMENT_OUTPUT_START);
        let commitment_output_columns = commitment
            .data
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        assert_eq!(builder.cols(), BEFORE_RESIDUAL_START);
        assert_eq!(
            builder.rows(),
            PI_RLC_FAMILY_ALGEBRA_ROWS + PI_RLC_FAMILY_OPENING_ROWS + 2
        );

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.residual");
        let before_residual = builder.alloc_vec(&native_commitment.data);
        let after_residual = builder.alloc_vec(&[F::ZERO; COMMITMENT_OUTPUT_FIELDS]);
        assert_columns(&before_residual, BEFORE_RESIDUAL_START);
        assert_columns(&after_residual, AFTER_RESIDUAL_START);
        for ((&before, &phase), &after) in before_residual
            .iter()
            .zip(&commitment.data)
            .zip(&after_residual)
        {
            let expected = Lc::from_var(phase).add_scaled(&Lc::from_var(after), F::ONE);
            builder.enforce_eq(&Lc::from_var(before), &expected);
        }
        assert_eq!(
            builder.rows(),
            PI_RLC_FAMILY_ALGEBRA_ROWS + PI_RLC_FAMILY_OPENING_ROWS + 2 + COMMITMENT_OUTPUT_FIELDS
        );

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.carry");
        let centered_challenges = challenge_values
            .iter()
            .map(|&symbol| symbol - F::from_u64(2))
            .collect::<Vec<_>>();
        let before_challenges = builder.alloc_vec(&centered_challenges);
        let after_challenges = builder.alloc_vec(&centered_challenges);
        let before_cursor = builder.alloc(F::from_usize(fixture_family));
        let after_cursor = builder.alloc(F::from_usize(fixture_family + 1));
        assert_columns(&before_challenges, BEFORE_CHALLENGE_START);
        assert_columns(&after_challenges, AFTER_CHALLENGE_START);
        assert_eq!(before_cursor.col(), BEFORE_CURSOR_COLUMN);
        assert_eq!(after_cursor.col(), AFTER_CURSOR_COLUMN);

        let input_before_values: [F; 8] = std::array::from_fn(|lane| F::from_usize(20_000 + lane * 23));
        let output_before_values: [F; 8] = std::array::from_fn(|lane| F::from_usize(30_000 + lane * 29));
        let input_before_vars: [Var; 8] = builder
            .alloc_vec(&input_before_values)
            .try_into()
            .expect("eight PiRLC input replay state fields");
        let output_before_vars: [Var; 8] = builder
            .alloc_vec(&output_before_values)
            .try_into()
            .expect("eight PiRLC output replay state fields");
        assert_eq!(input_before_vars[0].col(), INPUT_REPLAY_BEFORE_START);
        assert_eq!(output_before_vars[0].col(), OUTPUT_REPLAY_BEFORE_START);
        assert_eq!(builder.cols(), REPLAY_AUXILIARY_START);

        for ((&before, &symbol), &challenge) in before_challenges
            .iter()
            .zip(&challenge_vars)
            .zip(&challenge_values)
        {
            debug_assert_eq!(builder.witness()[before.col()], challenge - F::from_u64(2));
            let mut centered = Lc::from_var(symbol);
            centered.add_constant(-F::from_u64(2));
            builder.enforce_eq(&Lc::from_var(before), &centered);
        }
        for (&before, &after) in before_challenges.iter().zip(&after_challenges) {
            builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
        }
        let mut next_cursor = Lc::from_var(before_cursor);
        next_cursor.add_constant(F::ONE);
        builder.enforce_eq(&Lc::from_var(after_cursor), &next_cursor);
        assert_eq!(builder.rows(), PI_RLC_FAMILY_BODY_SOURCE_ROWS);

        let input_columns = input_vars.iter().map(|wire| wire.col()).collect::<Vec<_>>();
        let output_columns = output_vars
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let replay = append_pi_rlc_family_replay(
            &mut builder,
            kind,
            &input_columns,
            &output_columns,
            input_before_vars.map(Var::col),
            output_before_vars.map(Var::col),
        );
        let replay_call_audits = replay.call_audits.clone();
        assert_eq!(builder.rows(), expected_source_body_rows(kind));
        assert_eq!(builder.cols(), REPLAY_AUXILIARY_START + kind.rows());

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.public_cursor");
        let before_program_cursor = alloc_offset_word(&mut builder, before_cursor, FIRST_PI_RLC_FAMILY_PROGRAM_CURSOR);
        let after_program_cursor = alloc_offset_word(&mut builder, after_cursor, FIRST_PI_RLC_FAMILY_PROGRAM_CURSOR);
        let before_program_cursor_bits = decompose_var_to_u64_bits(&mut builder, before_program_cursor);
        let after_program_cursor_bits = decompose_var_to_u64_bits(&mut builder, after_program_cursor);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.state_digest");
        let before_state_fields = family_state_fields(
            &mut builder,
            kind.before_absorbed(),
            input_before_vars,
            &before_residual,
            output_before_vars,
            &before_challenges,
            before_cursor,
        );
        let after_state_fields = family_state_fields(
            &mut builder,
            kind.after_absorbed(),
            replay.input_after_columns.map(Var::from_column_for_trace),
            &after_residual,
            replay.output_after_columns.map(Var::from_column_for_trace),
            &after_challenges,
            after_cursor,
        );
        let before_state_field_columns = before_state_fields.iter().map(|wire| wire.col()).collect();
        let after_state_field_columns = after_state_fields.iter().map(|wire| wire.col()).collect();
        let (after_digest, after_digest_pin_columns) = digest_family_state(&mut builder, &after_state_fields);
        let (before_digest, before_digest_pin_columns) = digest_family_state(&mut builder, &before_state_fields);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.phase_envelope");
        let phase_envelope = enforce_streaming_carry_phase_semantic_envelope(&mut builder, before_digest, after_digest);

        builder.begin_encoding_stage("nebula.streaming.pi_rlc.body.state_x_out");
        let verifier_digest = alloc_fixture_digest(&mut builder, 40_000);
        let pi_ccs_header_bundle = alloc_fixture_digest(&mut builder, 40_100);
        let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);
        let before_boundary = alloc_fixture_digest(&mut builder, 40_200);
        let after_boundary = alloc_fixture_digest(&mut builder, 40_300);
        let before_accumulator = alloc_fixture_digest(&mut builder, 40_400);
        let after_accumulator = alloc_fixture_digest(&mut builder, 40_500);
        let nebula_lane_digest = alloc_fixture_digest(&mut builder, 40_600);
        let after_x_out = enforce_streaming_state_x_out(
            &mut builder,
            &StateXOutDigestInputs {
                mode: StateXOutDigestMode::Stateful,
                vk_fs_digest: verifier_digest,
                pi_ccs_header_bundle,
                structure_digest: pi_ccs_header_bundle,
                chunk_count: after_program_cursor,
                step_count: after_program_cursor,
                initial_boundary: verifier_digest,
                current_boundary: after_boundary,
                pc,
                semantic_acc: phase_envelope.after_semantic_digest,
                construction2_acc: after_accumulator,
                public_trace: after_boundary,
            },
            nebula_lane_digest,
        );
        let before_x_out = enforce_streaming_state_x_out(
            &mut builder,
            &StateXOutDigestInputs {
                mode: StateXOutDigestMode::Stateful,
                vk_fs_digest: verifier_digest,
                pi_ccs_header_bundle,
                structure_digest: pi_ccs_header_bundle,
                chunk_count: before_program_cursor,
                step_count: before_program_cursor,
                initial_boundary: verifier_digest,
                current_boundary: before_boundary,
                pc,
                semantic_acc: phase_envelope.before_semantic_digest,
                construction2_acc: before_accumulator,
                public_trace: before_boundary,
            },
            nebula_lane_digest,
        );
        let after_x_out_preimage_columns = after_x_out
            .preimage
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>()
            .try_into()
            .expect("stateful Nebula state_x_out has 32 fields");
        let before_x_out_preimage_columns = before_x_out
            .preimage
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>()
            .try_into()
            .expect("stateful Nebula state_x_out has 32 fields");
        let after_x_out_digest_columns = after_x_out.digest.map(Var::col);
        let before_x_out_digest_columns = before_x_out.digest.map(Var::col);
        let mut public_outputs = Vec::with_capacity(10 * PUBLIC_WORD_BITS);
        public_outputs.extend(after_x_out.public_bits);
        public_outputs.extend(before_x_out.public_bits);
        public_outputs.extend(before_program_cursor_bits);
        public_outputs.extend(after_program_cursor_bits);
        let public_layout = NebulaFPrimeStreamingPublicLayout::production();
        debug_assert_eq!(public_outputs.len() + 1, public_layout.logical_columns());

        assert_eq!(builder.rows(), expected_body_rows(kind));
        assert_eq!(builder.cols(), expected_body_columns(kind));
        assert_eq!(builder.first_unsatisfied_row(), None);

        Self {
            kind,
            fixture_family,
            builder,
            public_outputs,
            zero_digit_columns,
            active_digit_columns,
            commitment_output_columns,
            input_after_columns: replay.input_after_columns,
            output_after_columns: replay.output_after_columns,
            replay_call_audits,
            before_state_field_columns,
            after_state_field_columns,
            after_digest_pin_columns,
            before_digest_pin_columns,
            after_x_out_preimage_columns,
            before_x_out_preimage_columns,
            after_x_out_digest_columns,
            before_x_out_digest_columns,
            phase_envelope,
        }
    }

    pub const fn kind(&self) -> NebulaFPrimePiRlcFamilyReplayArmKind {
        self.kind
    }

    pub const fn fixture_family(&self) -> usize {
        self.fixture_family
    }

    pub fn rows(&self) -> usize {
        self.builder.rows()
    }

    pub fn columns(&self) -> usize {
        self.builder.cols()
    }

    pub fn public_columns(&self) -> usize {
        1 + self.public_outputs.len()
    }

    pub fn is_satisfied(&self) -> bool {
        self.builder.is_satisfied()
    }

    pub fn unconstrained_columns(&self) -> Vec<usize> {
        self.builder.unconstrained_columns()
    }

    pub const fn zero_digit_columns(&self) -> [usize; DIGIT_COUNT] {
        self.zero_digit_columns
    }

    pub fn active_digit_columns(&self) -> &[usize] {
        &self.active_digit_columns
    }

    pub fn commitment_output_columns(&self) -> &[usize] {
        &self.commitment_output_columns
    }

    pub const fn input_after_columns(&self) -> [usize; 8] {
        self.input_after_columns
    }

    pub const fn output_after_columns(&self) -> [usize; 8] {
        self.output_after_columns
    }

    pub fn replay_call_audits(&self) -> &[NebulaFPrimePiRlcFamilyReplayCallAudit] {
        &self.replay_call_audits
    }

    pub fn before_state_field_columns(&self) -> &[usize] {
        &self.before_state_field_columns
    }

    pub fn after_state_field_columns(&self) -> &[usize] {
        &self.after_state_field_columns
    }

    pub const fn before_family_cursor_column(&self) -> usize {
        BEFORE_CURSOR_COLUMN
    }

    pub const fn after_family_cursor_column(&self) -> usize {
        AFTER_CURSOR_COLUMN
    }

    pub const fn after_digest_pin_columns(&self) -> [usize; DIGEST_PIN_COUNT] {
        self.after_digest_pin_columns
    }

    pub const fn before_digest_pin_columns(&self) -> [usize; DIGEST_PIN_COUNT] {
        self.before_digest_pin_columns
    }

    pub fn after_x_out_preimage_columns(&self) -> &[usize] {
        &self.after_x_out_preimage_columns
    }

    pub fn before_x_out_preimage_columns(&self) -> &[usize] {
        &self.before_x_out_preimage_columns
    }

    pub const fn after_x_out_digest_columns(&self) -> [usize; 4] {
        self.after_x_out_digest_columns
    }

    pub const fn before_x_out_digest_columns(&self) -> [usize; 4] {
        self.before_x_out_digest_columns
    }

    pub fn before_phase_local_state_columns(&self) -> [usize; 4] {
        self.phase_envelope.before_local_state_digest.map(Var::col)
    }

    pub fn after_phase_local_state_columns(&self) -> [usize; 4] {
        self.phase_envelope.after_local_state_digest.map(Var::col)
    }

    pub fn before_phase_local_state_source_columns(&self) -> [usize; 4] {
        self.phase_envelope
            .before_local_state_source_digest
            .map(Var::col)
    }

    pub fn after_phase_local_state_source_columns(&self) -> [usize; 4] {
        self.phase_envelope
            .after_local_state_source_digest
            .map(Var::col)
    }

    pub fn before_phase_semantic_digest_columns(&self) -> [usize; 4] {
        self.phase_envelope.before_semantic_digest.map(Var::col)
    }

    pub fn after_phase_semantic_digest_columns(&self) -> [usize; 4] {
        self.phase_envelope.after_semantic_digest.map(Var::col)
    }

    pub fn phase_delayed_payload_columns(&self) -> Vec<usize> {
        self.phase_envelope
            .delayed_payload_bits
            .iter()
            .map(|wire| wire.col())
            .collect()
    }

    pub fn public_output_column(&self, index: usize) -> Option<usize> {
        self.public_outputs.get(index).map(|wire| wire.col())
    }

    pub fn shape_audit(&self) -> NebulaFPrimePiRlcFamilyBodyShapeAudit {
        NebulaFPrimePiRlcFamilyBodyShapeAudit {
            kind: self.kind,
            source_rows: PI_RLC_FAMILY_BODY_SOURCE_ROWS,
            rows: self.rows(),
            columns: self.columns(),
            source_columns: SOURCE_COLUMNS,
            public_columns: self.public_columns(),
            replay_poseidon2_permutations: self.kind.poseidon2_calls(),
            poseidon2_permutations: self.builder.poseidon2_permutation_audits().len(),
        }
    }

    #[doc(hidden)]
    pub fn builder_for_artifact(&self) -> &R1csBuilder {
        &self.builder
    }

    #[doc(hidden)]
    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    #[doc(hidden)]
    pub fn normalized_field_assignment_for_artifact(&self) -> Result<Vec<F>, FieldR1csLoweringError> {
        normalized_field_assignment(&self.builder, &self.public_outputs)
    }

    #[doc(hidden)]
    pub fn normalized_field_column_for_artifact(&self, source: usize) -> Option<usize> {
        if source >= self.builder.cols() {
            return None;
        }
        if source == 0 {
            return Some(0);
        }
        let mut public_before = 0;
        for (index, wire) in self.public_outputs.iter().enumerate() {
            let column = wire.col();
            if column == source {
                return Some(index + 1);
            }
            public_before += usize::from(column < source);
        }
        Some(source + self.public_outputs.len() - public_before)
    }

    #[doc(hidden)]
    pub fn tamper_witness_for_test(&mut self, column: usize, value: F) {
        self.builder.tamper_witness(column, value);
    }

    fn into_sparse(self) -> Result<crate::frontends::r1cs_f_prime::SparseR1cs, FieldR1csLoweringError> {
        Ok(lower_field_r1cs(self.builder, &self.public_outputs)?
            .into_parts()
            .0)
    }
}

pub struct NebulaFPrimePiRlcFamilyOverlaySynthesis {
    family: usize,
    builder: R1csBuilder,
    zero_digit_columns: [usize; DIGIT_COUNT],
    active_digit_columns: Vec<usize>,
    output_columns: Vec<usize>,
}

impl NebulaFPrimePiRlcFamilyOverlaySynthesis {
    pub fn production(family: usize) -> Option<Self> {
        if family >= PI_RLC_FAMILY_COUNT {
            return None;
        }
        let input_values = production_input_values();
        let native_commitment = native_family_commitment(family, &input_values);
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();
        builder.begin_encoding_stage("nebula.streaming.pi_rlc.family_overlay");

        let zero_digit_vars: [Var; DIGIT_COUNT] = builder
            .alloc_vec(&[F::ZERO; DIGIT_COUNT])
            .try_into()
            .expect("one PiRLC overlay zero word");
        for &digit in &zero_digit_vars {
            builder.record_centered_unit(digit);
        }
        let zero_digit_columns = zero_digit_vars.map(Var::col);
        assert_eq!(zero_digit_columns[0], PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START);

        let mut active_digit_columns = Vec::with_capacity(FAMILY_INPUT_FIELDS * DIGIT_COUNT);
        for value in input_values {
            let word = builder.alloc_vec(&balanced_ternary_digits(value));
            for &digit in &word {
                builder.record_centered_unit(digit);
            }
            active_digit_columns.extend(word.iter().map(|wire| wire.col()));
        }
        assert_eq!(
            active_digit_columns.first().copied(),
            Some(PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START)
        );
        assert_eq!(builder.cols(), PI_RLC_FAMILY_OVERLAY_OUTPUT_START);

        let output_vars = builder.alloc_vec(&native_commitment.data);
        let output_columns = output_vars
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        assert_columns(&output_vars, PI_RLC_FAMILY_OVERLAY_OUTPUT_START);
        assert_eq!(builder.cols(), PI_RLC_FAMILY_OVERLAY_COLUMNS);

        let mut word_starts = vec![zero_digit_vars[0].col(); PI_RLC_GLOBAL_INPUT_FIELDS];
        let family_start = family * FAMILY_INPUT_FIELDS;
        for offset in 0..FAMILY_INPUT_FIELDS {
            word_starts[family_start + offset] = PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START + offset * DIGIT_COUNT;
        }
        let (chunk_size, chunk_seeds) = seeded_pp_chunk_seeds(
            PI_RLC_INPUT_COORDINATE_SIS_CONFIG.seed,
            PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
            PI_RLC_MESSAGE_COLUMNS,
        );
        let block = SeededPhi81LinearBlock::new_with_word_width(
            builder.rows(),
            word_starts,
            DIGIT_COUNT,
            PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
            PI_RLC_MESSAGE_COLUMNS,
            chunk_size,
            chunk_seeds,
        )
        .expect("fixed production PiRLC family overlay geometry");
        builder.enforce_seeded_phi81_a_block(block, &output_vars);
        assert_eq!(builder.rows(), PI_RLC_FAMILY_OVERLAY_ROWS);
        assert_eq!(builder.first_unsatisfied_row(), None);

        Some(Self {
            family,
            builder,
            zero_digit_columns,
            active_digit_columns,
            output_columns,
        })
    }

    pub const fn family(&self) -> usize {
        self.family
    }

    pub fn rows(&self) -> usize {
        self.builder.rows()
    }

    pub fn columns(&self) -> usize {
        self.builder.cols()
    }

    pub fn is_satisfied(&self) -> bool {
        self.builder.is_satisfied()
    }

    pub fn unconstrained_columns(&self) -> Vec<usize> {
        self.builder.unconstrained_columns()
    }

    pub const fn zero_digit_columns(&self) -> [usize; DIGIT_COUNT] {
        self.zero_digit_columns
    }

    pub fn active_digit_columns(&self) -> &[usize] {
        &self.active_digit_columns
    }

    pub fn output_columns(&self) -> &[usize] {
        &self.output_columns
    }

    pub fn shape_audit(&self) -> NebulaFPrimePiRlcFamilyOverlayShapeAudit {
        NebulaFPrimePiRlcFamilyOverlayShapeAudit {
            family: self.family,
            rows: self.rows(),
            columns: self.columns(),
            zero_digits: PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START..PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START,
            active_digits: PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START..PI_RLC_FAMILY_OVERLAY_OUTPUT_START,
            outputs: PI_RLC_FAMILY_OVERLAY_OUTPUT_START..PI_RLC_FAMILY_OVERLAY_COLUMNS,
        }
    }

    #[doc(hidden)]
    pub fn builder_for_artifact(&self) -> &R1csBuilder {
        &self.builder
    }

    #[doc(hidden)]
    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    #[doc(hidden)]
    pub fn tamper_witness_for_test(&mut self, column: usize, value: F) {
        self.builder.tamper_witness(column, value);
    }

    fn into_sparse(self) -> Result<crate::frontends::r1cs_f_prime::SparseR1cs, FieldR1csLoweringError> {
        Ok(lower_field_r1cs(self.builder, &[])?.into_parts().0)
    }
}

/// Exact Rust-emitted field-R1CS source arms in even, then odd order.
#[doc(hidden)]
pub fn production_pi_rlc_family_body_source_arms() -> Result<Vec<SparseR1cs>, NebulaFPrimePiRlcFamilyRelationError> {
    Ok([
        NebulaFPrimePiRlcFamilyReplayArmKind::Even,
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd,
    ]
    .into_iter()
    .map(|kind| NebulaFPrimePiRlcFamilyBodySynthesis::production(kind).into_sparse())
    .collect::<Result<Vec<_>, _>>()?)
}

/// Project exact production body rows without materializing the final CCS
/// matrices. The layout uses the same shared-field and shared-bit parameters
/// as the production compiler.
#[doc(hidden)]
pub fn production_pi_rlc_family_body_projected_rows_with_source_provenance(
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    Ok(project_rows_with_complete_source_provenance_with_alignment(
        &arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        selected_rows,
        source_arm,
        source_columns,
        retained_row_pairs,
    )?)
}

/// Measure the frozen Nightstream k16 body compiler without emitting its CCS matrices.
#[doc(hidden)]
pub fn production_pi_rlc_family_body_low_norm_shape_audit(
) -> Result<NebulaFPrimePiRlcFamilyBodyLowNormShapeAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    let prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        crate::config::B_BASE,
    )?;
    let shape = prepared.shape_summary();
    Ok(NebulaFPrimePiRlcFamilyBodyLowNormShapeAudit {
        norm_base: crate::config::B_BASE,
        rows: shape.rows,
        columns: shape.columns,
        public_columns: shape.public_input_len,
        total_coordinates: shape.total_coordinates,
    })
}

pub fn build_production_pi_rlc_family_body_low_norm_r1cs(
) -> Result<MultiBranchLowNormR1cs, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    Ok(
        prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            REPLAY_AUXILIARY_START - 1,
            0,
            D,
            0,
            crate::config::B_BASE,
        )?
        .finish()?,
    )
}

/// Complete source-to-final decoder runs for both exact production parity
/// arms. This is assignment-layout evidence only; it does not claim row or
/// matrix soundness.
pub fn production_pi_rlc_family_body_decoder_runs(
) -> Result<Vec<SelectiveProjectedDecoderRunProvenance>, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    let requests = arms
        .iter()
        .enumerate()
        .map(|(arm, source)| (arm, 1..source.m))
        .collect::<Vec<_>>();
    Ok(audit_multi_branch_selective_decoder_runs_with_shared_bit_prefix(
        &arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        crate::config::B_BASE,
        &requests,
    )?)
}

/// Complete source-row ledger and exact requested source-to-final decoder
/// runs from one prepared Nightstream k16 production layout.
#[doc(hidden)]
pub fn production_pi_rlc_family_body_compact_layout_and_decoder_runs_for_ranges(
    requests: &[(usize, Range<usize>)],
) -> Result<
    (SelectiveCompactLayoutAudit, Vec<SelectiveProjectedDecoderRunProvenance>),
    NebulaFPrimePiRlcFamilyRelationError,
> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    Ok(
        audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
            &arms,
            REPLAY_AUXILIARY_START - 1,
            0,
            D,
            0,
            crate::config::B_BASE,
            requests,
        )?,
    )
}

/// Complete source-row and emitted-row ledger from the exact prepared layout
/// used by the frozen Nightstream k16 production body emitter.
pub fn production_pi_rlc_family_body_compiler_audit(
) -> Result<SelectiveCompilerAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    Ok(audit_multi_branch_selective_compiler_with_shared_bit_prefix(
        &arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        crate::config::B_BASE,
    )?)
}

pub fn build_production_pi_rlc_family_overlay_low_norm_r1cs(
) -> Result<MultiBranchLowNormR1cs, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_overlay_sparse_arms()?;
    Ok(
        prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            PI_RLC_FAMILY_OVERLAY_COLUMNS - 1,
            0,
            1,
            0,
            crate::config::B_BASE,
        )?
        .finish()?,
    )
}

pub(crate) fn production_pi_rlc_family_overlay_sparse_arms(
) -> Result<Vec<crate::frontends::r1cs_f_prime::SparseR1cs>, NebulaFPrimePiRlcFamilyRelationError> {
    Ok((0..PI_RLC_FAMILY_COUNT)
        .map(|family| {
            NebulaFPrimePiRlcFamilyOverlaySynthesis::production(family)
                .expect("bounded PiRLC family")
                .into_sparse()
        })
        .collect::<Result<Vec<_>, _>>()?)
}

pub fn production_pi_rlc_family_overlay_kind_map(noop_kind: usize, first_family_kind: usize) -> Vec<usize> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    program
        .work_items()
        .iter()
        .map(|item| {
            if item.phase() == NebulaFPrimeStreamingPhase::PiRlcFamily {
                first_family_kind + item.index()
            } else {
                noop_kind
            }
        })
        .collect()
}

/// Three exact link runs shared by all 110 family overlays: the constrained
/// zero word, the 918 canonical input words, and the 108 commitment outputs.
pub const fn production_pi_rlc_family_overlay_link_runs() -> [NebulaFPrimePiRlcFamilyOverlayLinkRun; 3] {
    [
        NebulaFPrimePiRlcFamilyOverlayLinkRun {
            phase_field_start: ZERO_DIGIT_START + PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS,
            overlay_field_start: PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START,
            outer_count: 1,
            phase_stride: DIGIT_COUNT,
            overlay_stride: DIGIT_COUNT,
            field_count: DIGIT_COUNT,
        },
        NebulaFPrimePiRlcFamilyOverlayLinkRun {
            phase_field_start: ACTIVE_DIGIT_START + PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS,
            overlay_field_start: PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START,
            outer_count: FAMILY_INPUT_FIELDS,
            phase_stride: OPENING_COLUMNS,
            overlay_stride: DIGIT_COUNT,
            field_count: DIGIT_COUNT,
        },
        NebulaFPrimePiRlcFamilyOverlayLinkRun {
            phase_field_start: COMMITMENT_OUTPUT_START + PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS,
            overlay_field_start: PI_RLC_FAMILY_OVERLAY_OUTPUT_START,
            outer_count: 1,
            phase_stride: COMMITMENT_OUTPUT_FIELDS,
            overlay_stride: COMMITMENT_OUTPUT_FIELDS,
            field_count: COMMITMENT_OUTPUT_FIELDS,
        },
    ]
}

pub fn production_pi_rlc_family_overlay_links(first_overlay_kind: usize) -> Vec<OverlayKindLinks> {
    let link_runs = production_pi_rlc_family_overlay_link_runs();
    (0..PI_RLC_FAMILY_COUNT)
        .map(|family| production_pi_rlc_family_overlay_links_for_family(first_overlay_kind, family, link_runs))
        .collect()
}

fn production_pi_rlc_family_overlay_links_for_family(
    first_overlay_kind: usize,
    family: usize,
    link_runs: [NebulaFPrimePiRlcFamilyOverlayLinkRun; 3],
) -> OverlayKindLinks {
    debug_assert!(family < PI_RLC_FAMILY_COUNT);
    let phase_kind = if family % 2 == 0 {
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyEven
    } else {
        NebulaFPrimeStreamingCircuitKind::PiRlcFamilyOdd
    };
    let fields = link_runs.into_iter().flat_map(|run| run.fields()).collect();
    OverlayKindLinks {
        overlay_kind: first_overlay_kind + family,
        phase_kind: phase_kind.code() as usize,
        fields,
    }
}

const fn expected_body_rows(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> usize {
    match kind {
        NebulaFPrimePiRlcFamilyReplayArmKind::Even => PI_RLC_FAMILY_BODY_EVEN_ROWS,
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd => PI_RLC_FAMILY_BODY_ODD_ROWS,
    }
}

const fn expected_source_body_rows(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> usize {
    match kind {
        NebulaFPrimePiRlcFamilyReplayArmKind::Even => PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS,
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd => PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS,
    }
}

const fn expected_body_columns(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> usize {
    match kind {
        NebulaFPrimePiRlcFamilyReplayArmKind::Even => PI_RLC_FAMILY_BODY_EVEN_COLUMNS,
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd => PI_RLC_FAMILY_BODY_ODD_COLUMNS,
    }
}

fn alloc_offset_word(builder: &mut R1csBuilder, source: Var, offset: usize) -> Var {
    let value = builder.witness()[source.col()] + F::from_usize(offset);
    let output = builder.alloc(value);
    let expected = Lc::from_var(source).add_scaled(&Lc::from_const(F::ONE), F::from_usize(offset));
    builder.enforce_eq(&Lc::from_var(output), &expected);
    output
}

fn alloc_bound_constant(builder: &mut R1csBuilder, value: usize) -> Var {
    let value = F::from_usize(value);
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}

fn alloc_fixture_digest(builder: &mut R1csBuilder, start: usize) -> [Var; 4] {
    std::array::from_fn(|lane| builder.alloc(F::from_usize(start + lane)))
}

fn family_state_fields(
    builder: &mut R1csBuilder,
    absorbed: usize,
    input_replay: [Var; SPONGE_WIDTH],
    residual: &[Var],
    output_replay: [Var; SPONGE_WIDTH],
    challenges: &[Var],
    cursor: Var,
) -> Vec<Var> {
    assert_eq!(residual.len(), COMMITMENT_OUTPUT_FIELDS);
    assert_eq!(challenges.len(), FAMILY_INPUT_FIELDS);
    let mut fields = Vec::with_capacity(FAMILY_STATE_FIELDS);
    fields.extend(input_replay);
    fields.push(alloc_bound_constant(builder, absorbed));
    fields.extend_from_slice(residual);
    fields.extend(output_replay);
    fields.push(alloc_bound_constant(builder, absorbed));
    fields.extend_from_slice(challenges);
    fields.push(cursor);
    debug_assert_eq!(fields.len(), FAMILY_STATE_FIELDS);
    fields
}

fn digest_family_state(builder: &mut R1csBuilder, fields: &[Var]) -> ([Var; 4], [usize; DIGEST_PIN_COUNT]) {
    assert_eq!(fields.len(), FAMILY_STATE_FIELDS);
    let mut transcript = TranscriptGadget::new(builder, STATE_DIGEST_DOMAIN);
    transcript.append_fields(builder, STATE_DIGEST_FIELDS_LABEL, fields);
    let digest = transcript.digest_fields(builder);
    let pin_columns = transcript.constant_bindings()[..DIGEST_PIN_COUNT]
        .iter()
        .map(|(wire, _)| wire.col())
        .collect::<Vec<_>>()
        .try_into()
        .expect("fixed PiRLC family-state digest pin count");
    (digest, pin_columns)
}

fn production_input_values() -> Vec<F> {
    (0..FAMILY_INPUT_FIELDS)
        .map(|offset| {
            let source = offset / LANE_COUNT;
            let lane = offset % LANE_COUNT;
            F::from_usize(1 + source * 3 + lane * 5)
        })
        .collect()
}

fn native_family_commitment(family: usize, input_values: &[F]) -> Commitment {
    assert!(family < PI_RLC_FAMILY_COUNT);
    assert_eq!(input_values.len(), FAMILY_INPUT_FIELDS);
    let mut message = Mat::zero(D, PI_RLC_MESSAGE_COLUMNS, F::ZERO);
    let family_start = family * FAMILY_INPUT_FIELDS;
    for (offset, &value) in input_values.iter().enumerate() {
        for (digit, coefficient) in balanced_ternary_digits(value).into_iter().enumerate() {
            let index = (family_start + offset) * DIGIT_COUNT + digit;
            message.set(
                index / PI_RLC_MESSAGE_COLUMNS,
                index % PI_RLC_MESSAGE_COLUMNS,
                coefficient,
            );
        }
    }
    commit_row_major_seeded(
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.seed,
        D,
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
        PI_RLC_MESSAGE_COLUMNS,
        &message,
    )
}

fn product_witness(challenges: &[F], inputs: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(challenges.len(), FAMILY_INPUT_FIELDS);
    assert_eq!(inputs.len(), FAMILY_INPUT_FIELDS);
    let mut products = vec![F::ZERO; ALGEBRA_PRODUCT_COLUMNS];
    let mut outputs = vec![F::ZERO; LANE_COUNT];
    for source in 0..SOURCE_COUNT {
        for left in 0..LANE_COUNT {
            let challenge = challenges[source * LANE_COUNT + left] - F::from_u64(2);
            for right in 0..LANE_COUNT {
                let index = product_offset(source, left, right);
                let product = challenge * inputs[source * LANE_COUNT + right];
                products[index] = product;
                for (output, coefficient) in reduced_monomial(left + right) {
                    outputs[output] += coefficient * product;
                }
            }
        }
    }
    (products, outputs)
}

fn enforce_algebra_rows(
    builder: &mut R1csBuilder,
    challenges: &[Var],
    inputs: &[Var],
    outputs: &[Var],
    products: &[Var],
) {
    let mut output_sums = (0..LANE_COUNT).map(|_| Lc::zero()).collect::<Vec<_>>();
    for source in 0..SOURCE_COUNT {
        for left in 0..LANE_COUNT {
            let mut challenge = Lc::from_var(challenges[source * LANE_COUNT + left]);
            challenge.add_constant(-F::from_u64(2));
            for right in 0..LANE_COUNT {
                let product = products[product_offset(source, left, right)];
                builder.enforce(
                    &challenge,
                    &Lc::from_var(inputs[source * LANE_COUNT + right]),
                    &Lc::from_var(product),
                );
                for (output, coefficient) in reduced_monomial(left + right) {
                    output_sums[output].add_term(product, coefficient);
                }
            }
        }
    }
    for (sum, &output) in output_sums.iter().zip(outputs) {
        builder.enforce(&Lc::from_const(F::ONE), sum, &Lc::from_var(output));
    }
}

fn product_offset(source: usize, left: usize, right: usize) -> usize {
    (source * LANE_COUNT + left) * LANE_COUNT + right
}

fn reduced_monomial(degree: usize) -> Vec<(usize, F)> {
    if degree < LANE_COUNT {
        vec![(degree, F::ONE)]
    } else if degree < LANE_COUNT + LANE_COUNT / 2 {
        vec![(degree - LANE_COUNT, -F::ONE), (degree - LANE_COUNT / 2, -F::ONE)]
    } else {
        vec![(degree - 3 * LANE_COUNT / 2, F::ONE)]
    }
}

fn assert_columns(vars: &[Var], start: usize) {
    assert!(vars
        .iter()
        .enumerate()
        .all(|(offset, wire)| wire.col() == start + offset));
}

const _: () = assert!(D == LANE_COUNT);
const _: () = assert!(SOURCE_COUNT == 17);
const _: () = assert!(FAMILY_INPUT_FIELDS == 918);
const _: () = assert!(PI_RLC_FAMILY_ALGEBRA_ROWS == 49_626);
const _: () = assert!(PI_RLC_FAMILY_OPENING_ROWS == 113_873);
const _: () = assert!(PI_RLC_FAMILY_BODY_SOURCE_ROWS == 165_446);
const _: () = assert!(PI_RLC_GLOBAL_INPUT_FIELDS == 100_980);
const _: () = assert!(PI_RLC_MESSAGE_COLUMNS == 76_670);
const _: () = assert!(PI_RLC_FAMILY_OVERLAY_COLUMNS == 37_788);
const _: () = assert!(PI_RLC_FAMILY_LINK_FIELDS == 37_787);
