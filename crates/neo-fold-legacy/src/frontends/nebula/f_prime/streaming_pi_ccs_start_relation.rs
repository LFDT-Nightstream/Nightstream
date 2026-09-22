//! Exact handoff from completed claim replay to the first PiCCS round.
//!
//! Owns claim-replay readiness, the full 21,220-field variable-statement
//! opening, its coordinate-preserving Module-SIS check, the fresh-metadata
//! residual, the carried running-metadata binding, the fixed selective
//! statement transcript, alpha and gamma sampling, the complete initial
//! claim, the zero challenge register, and the authenticated phase envelope.
//!
//! The two carried commitments are checked compression. Claim replay derives
//! them from the authoritative frame. This phase recomputes the statement
//! part of the first map and carries only its fresh-metadata residual. A
//! different opening is a named Module-SIS failure; neither commitment is
//! independent authority.
//!
//! Does not own PiCCS rounds or finish, Module-SIS hardness, claim-replay
//! source rows, lifecycle selection, terminal integration, or final fixed-
//! point geometry.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::field_ext::{alloc_klc, enforce_k_mul, KLc};
use crate::engine::r1cs_circuit::sumcheck::gamma_powers;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{KVar, Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, selective_polynomial, FieldR1csLoweringError, SparseR1cs};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::{alloc_constant, StateXOutDigestInputs};
use crate::paper::reductions::accumulator_sis_circuit::{
    commit_coordinate_fields, enforce_commit_coordinate_fields, SisAccumulatorError,
    PI_CCS_RUNNING_COMMITMENTS_COORDINATE_SIS_CONFIG, PI_CCS_RUNNING_PUBLIC_COORDINATE_SIS_CONFIG,
    PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
};
use crate::paper::reductions::pi_ccs_circuit::verifier::{append_pi_ccs_statement, squeeze_pi_ccs_challenge};

use super::streaming_claim_replay::{
    alloc_persistent, digest_persistent_state, enforce_sponge_equal, PersistentState, PersistentStateVars, SpongeState,
    COORDINATE_COMMITMENT_FIELDS, PI_CCS_RUNNING_COMMITMENT_FIELDS, PI_CCS_RUNNING_PUBLIC_FIELDS,
    PI_CCS_STATEMENT_FIELDS, PI_CCS_STATEMENT_FRESH_FIELDS,
};
use super::streaming_phase_envelope::{
    enforce_streaming_carry_phase_semantic_envelope, StreamingCarryPhaseSemanticEnvelope,
};
use super::streaming_pi_ccs_state::{
    digest_streaming_pi_ccs_state, StreamingPiCcsStateVars, PI_CCS_CONTEXT_DIGEST_FIELDS, PI_CCS_POINT_COUNT,
    PI_CCS_SPONGE_WIDTH,
};
use super::streaming_program::{CLAIM_FRAME_FIELDS, FIRST_PI_CCS_ROUND_PROGRAM_CURSOR};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;

const PUBLIC_WORD_BITS: usize = 64;
const FRESH_COUNT: usize = 1;
const RUNNING_COUNT: usize = crate::config::K_RHO as usize;
const MATRIX_COUNT: usize = 14;
const COEFFICIENT_COUNT: usize = D;
const VARIABLE_FIELDS: usize = 2 * PI_CCS_POINT_COUNT + 2 * RUNNING_COUNT * MATRIX_COUNT * COEFFICIENT_COUNT;
const GAMMA_POWER_COUNT: usize = 12_130;
const CLAIM_READY_ABSORBED: usize = CLAIM_FRAME_FIELDS % 4;
const PI_CCS_START_PROGRAM_CURSOR: usize = FIRST_PI_CCS_ROUND_PROGRAM_CURSOR - 1;
const CONTEXT_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-pi-ccs-context/v1";
const CONTEXT_FIELDS_LABEL: &[u8] = b"context";

pub const STREAMING_PI_CCS_START_CLAIM_STATE_FAMILY: &str = "fprime.streaming.pi_ccs.start.claim_state";
pub const STREAMING_PI_CCS_START_VARIABLE_FIELDS_FAMILY: &str = "fprime.streaming.pi_ccs.start.variable_fields";
pub const STREAMING_PI_CCS_START_READY_FAMILY: &str = "fprime.streaming.pi_ccs.start.ready";
pub const STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY: &str = "fprime.streaming.pi_ccs.start.variable_binding";
pub const STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY: &str = "fprime.streaming.pi_ccs.start.transcript";
pub const STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY: &str = "fprime.streaming.pi_ccs.start.initial_claim";
pub const STREAMING_PI_CCS_START_CONTEXT_FAMILY: &str = "fprime.streaming.pi_ccs.start.context";
pub const STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY: &str = "fprime.streaming.pi_ccs.start.lifecycle_carry";

pub const PI_CCS_START_SOURCE_ROWS: usize = 4_115_653;
pub const PI_CCS_START_SOURCE_COLUMNS: usize = 4_091_727;
pub const PI_CCS_START_SOURCE_PUBLIC_COLUMNS: usize = 641;
pub const PI_CCS_START_SOURCE_POSEIDON2_PERMUTATIONS: usize = 1_632;
pub const PI_CCS_START_SOURCE_ARTIFACT_ID: &str = "rust:streaming-pi-ccs-start/source-b2-k16-v3";
pub const PI_CCS_START_SOURCE_HASH_SCHEMA: &str = "nightstream-normalized-sparse-r1cs-compact-v1";
pub const PI_CCS_START_SOURCE_SHA256: &str = "726102be17e658218b03b80755da76867966068a71b867000455c3240b17a270";
pub const PI_CCS_START_PROFILE_ID: &str = "nightstream/goldilocks/b2-k16/streaming-pi-ccs-start/v3";
pub const PI_CCS_START_LIFECYCLE_SCOPE: &str = "recursive transition: claim replay to PiCCS round 0";
pub const PI_CCS_START_BEFORE_PROGRAM_CURSOR: usize = PI_CCS_START_PROGRAM_CURSOR;
pub const PI_CCS_START_AFTER_PROGRAM_CURSOR: usize = FIRST_PI_CCS_ROUND_PROGRAM_CURSOR;
pub const PI_CCS_START_FINAL_COMMON_PUBLIC_COLUMNS: usize = 648;
pub const PI_CCS_START_FINAL_BINDING_STATUS: &str =
    "pending complete 23-kind selective CCS schedule; no final row identity is claimed";

/// One exact source-stage interval in normalized column space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsStartSourceStage {
    path: &'static str,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
}

impl NebulaFPrimePiCcsStartSourceStage {
    pub const fn path(self) -> &'static str {
        self.path
    }

    pub const fn row_start(self) -> usize {
        self.row_start
    }

    pub const fn row_end(self) -> usize {
        self.row_end
    }

    pub const fn column_start(self) -> usize {
        self.column_start
    }

    pub const fn column_end(self) -> usize {
        self.column_end
    }
}

pub const PI_CCS_START_SOURCE_STAGE_SCHEDULE: [NebulaFPrimePiCcsStartSourceStage; 9] = [
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.state_words",
        row_start: 0,
        row_end: 69,
        column_start: 641,
        column_end: 25_235,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.ready",
        row_start: 69,
        row_end: 82,
        column_start: 25_235,
        column_end: 25_235,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.variable_binding",
        row_start: 82,
        row_end: 3_006_597,
        column_start: 25_235,
        column_end: 2_983_262,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.transcript",
        row_start: 3_006_597,
        row_end: 3_203_470,
        column_start: 2_983_262,
        column_end: 3_180_135,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.initial_claim",
        row_start: 3_203_470,
        row_end: 3_324_599,
        column_start: 3_180_135,
        column_end: 3_301_264,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.context",
        row_start: 3_324_599,
        row_end: 3_376_867,
        column_start: 3_301_264,
        column_end: 3_353_532,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.state_digest",
        row_start: 3_376_867,
        row_end: 3_441_097,
        column_start: 3_353_532,
        column_end: 3_417_762,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.phase_envelope",
        row_start: 3_441_097,
        row_end: 4_104_068,
        column_start: 3_417_762,
        column_end: 4_080_733,
    },
    NebulaFPrimePiCcsStartSourceStage {
        path: "nebula.streaming.pi_ccs.start.state_x_out",
        row_start: 4_104_068,
        row_end: PI_CCS_START_SOURCE_ROWS,
        column_start: 4_080_733,
        column_end: PI_CCS_START_SOURCE_COLUMNS,
    },
];

/// Frozen source and common-selective public-column boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsStartColumnLayout {
    constant_one: usize,
    after_x_out_bits: (usize, usize),
    before_x_out_bits: (usize, usize),
    before_cursor_bits: (usize, usize),
    after_cursor_bits: (usize, usize),
    common_public_padding: (usize, usize),
    private_columns: (usize, usize),
}

impl NebulaFPrimePiCcsStartColumnLayout {
    pub const fn constant_one(self) -> usize {
        self.constant_one
    }

    pub const fn after_x_out_bits(self) -> (usize, usize) {
        self.after_x_out_bits
    }

    pub const fn before_x_out_bits(self) -> (usize, usize) {
        self.before_x_out_bits
    }

    pub const fn before_cursor_bits(self) -> (usize, usize) {
        self.before_cursor_bits
    }

    pub const fn after_cursor_bits(self) -> (usize, usize) {
        self.after_cursor_bits
    }

    pub const fn common_public_padding(self) -> (usize, usize) {
        self.common_public_padding
    }

    pub const fn private_columns(self) -> (usize, usize) {
        self.private_columns
    }
}

pub const PI_CCS_START_SOURCE_COLUMN_LAYOUT: NebulaFPrimePiCcsStartColumnLayout = NebulaFPrimePiCcsStartColumnLayout {
    constant_one: 0,
    after_x_out_bits: (1, 257),
    before_x_out_bits: (257, 513),
    before_cursor_bits: (513, 577),
    after_cursor_bits: (577, 641),
    common_public_padding: (641, PI_CCS_START_FINAL_COMMON_PUBLIC_COLUMNS),
    private_columns: (641, PI_CCS_START_SOURCE_COLUMNS),
};

#[derive(Debug, Error)]
pub enum NebulaFPrimePiCcsStartRelationError {
    #[error(transparent)]
    Sis(#[from] SisAccumulatorError),
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsStartShapeAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub variable_fields: usize,
    pub gamma_powers: usize,
    pub poseidon2_permutations: usize,
}

/// One exact Rust-emitted field-R1CS source for circuit kind 5.
pub struct NebulaFPrimePiCcsStartSynthesis {
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    before: PersistentStateVars,
    after: StreamingPiCcsStateVars,
    variable_fields: Vec<Var>,
    computed_statement_commitment: [Var; COORDINATE_COMMITMENT_FIELDS],
    fresh_metadata_residual: [Var; COORDINATE_COMMITMENT_FIELDS],
    alpha: [KVar; PI_CCS_POINT_COUNT],
    gamma: KVar,
    before_program_cursor: Var,
    after_program_cursor: Var,
    before_x_out_preimage_columns: [usize; 32],
    after_x_out_preimage_columns: [usize; 32],
    before_boundary_columns: [usize; 4],
    after_boundary_columns: [usize; 4],
    before_accumulator_columns: [usize; 4],
    after_accumulator_columns: [usize; 4],
    phase_envelope: StreamingCarryPhaseSemanticEnvelope,
}

impl NebulaFPrimePiCcsStartSynthesis {
    pub fn production() -> Result<Self, NebulaFPrimePiCcsStartRelationError> {
        let variable_values = fixture_variable_fields();
        let fresh_metadata_values = fixture_fresh_metadata_fields();
        let running_commitment_values = fixture_running_commitment_fields();
        let running_public_values = fixture_running_public_fields();
        let statement_values = variable_values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        let fresh_metadata_fields = fresh_metadata_values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (PI_CCS_STATEMENT_FIELDS + index, value))
            .collect::<Vec<_>>();
        let native_statement_commitment = commit_coordinate_fields(
            PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
            PI_CCS_STATEMENT_FRESH_FIELDS,
            &statement_values,
        )?;
        let native_fresh_metadata_residual = commit_coordinate_fields(
            PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
            PI_CCS_STATEMENT_FRESH_FIELDS,
            &fresh_metadata_fields,
        )?;
        let statement_fresh_commitment: [F; COORDINATE_COMMITMENT_FIELDS] = native_statement_commitment
            .data
            .iter()
            .zip(&native_fresh_metadata_residual.data)
            .map(|(&statement, &fresh)| statement + fresh)
            .collect::<Vec<_>>()
            .try_into()
            .expect("fixed rank-two statement-and-fresh commitment");
        let running_commitment_fields = running_commitment_values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        let running_commitments_binding: [F; COORDINATE_COMMITMENT_FIELDS] = commit_coordinate_fields(
            PI_CCS_RUNNING_COMMITMENTS_COORDINATE_SIS_CONFIG,
            PI_CCS_RUNNING_COMMITMENT_FIELDS,
            &running_commitment_fields,
        )?
        .data
        .try_into()
        .expect("fixed rank-two running-commitments binding");
        let running_public_fields = running_public_values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        let running_public_binding: [F; COORDINATE_COMMITMENT_FIELDS] = commit_coordinate_fields(
            PI_CCS_RUNNING_PUBLIC_COORDINATE_SIS_CONFIG,
            PI_CCS_RUNNING_PUBLIC_FIELDS,
            &running_public_fields,
        )?
        .data
        .try_into()
        .expect("fixed rank-two running-public binding");
        let ready_state = fixture_ready_state(
            statement_fresh_commitment,
            running_commitments_binding,
            running_public_binding,
        );

        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.state_words");
        let claim_state_start = builder.cols();
        let before = alloc_persistent(&mut builder, ready_state);
        builder.record_column_family(STREAMING_PI_CCS_START_CLAIM_STATE_FAMILY, claim_state_start);
        let header = alloc_fixture_digest(&mut builder, 70_000);
        let variable_start = builder.cols();
        let variable_fields = builder.alloc_vec(&variable_values);
        builder.record_column_family(STREAMING_PI_CCS_START_VARIABLE_FIELDS_FAMILY, variable_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.ready");
        let ready_start = builder.rows();
        enforce_sponge_equal(&mut builder, before.expected, before.runtime);
        enforce_constant(&mut builder, before.expected.absorbed.field, CLAIM_READY_ABSORBED);
        enforce_constant(&mut builder, before.runtime.absorbed.field, CLAIM_READY_ABSORBED);
        enforce_constant(&mut builder, before.frame_cursor.field, CLAIM_FRAME_FIELDS);
        enforce_constant(&mut builder, before.program_cursor.field, PI_CCS_START_PROGRAM_CURSOR);
        builder.record_row_family(STREAMING_PI_CCS_START_READY_FAMILY, ready_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.variable_binding");
        let variable_binding_start = builder.rows();
        let positioned_fields = variable_fields
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        let computed_statement_commitment = enforce_commit_coordinate_fields(
            &mut builder,
            PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
            PI_CCS_STATEMENT_FRESH_FIELDS,
            &positioned_fields,
        )?;
        let computed_statement_commitment: [Var; COORDINATE_COMMITMENT_FIELDS] = computed_statement_commitment
            .data
            .try_into()
            .expect("fixed rank-two PiCCS statement commitment wires");
        let fresh_metadata_residual: [Var; COORDINATE_COMMITMENT_FIELDS] = native_fresh_metadata_residual
            .data
            .iter()
            .map(|&value| builder.alloc(value))
            .collect::<Vec<_>>()
            .try_into()
            .expect("fixed rank-two fresh-metadata residual wires");
        for ((expected, computed), residual) in before
            .statement_fresh_commitment
            .iter()
            .zip(computed_statement_commitment)
            .zip(fresh_metadata_residual)
        {
            let recomposed = Lc::from_var(computed).add_scaled(&Lc::from_var(residual), F::ONE);
            builder.enforce_eq(&Lc::from_var(expected.field), &recomposed);
        }
        builder.record_row_family(STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY, variable_binding_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.transcript");
        let transcript_start = builder.rows();
        let mut transcript =
            TranscriptGadget::from_variable_state(before.runtime.lanes.map(|word| word.field), CLAIM_READY_ABSORBED);
        let polynomial = selective_polynomial();
        append_pi_ccs_statement(
            &mut builder,
            &mut transcript,
            &polynomial,
            PI_CCS_POINT_COUNT,
            FRESH_COUNT,
            RUNNING_COUNT,
            MATRIX_COUNT,
        );
        let alpha = (0..PI_CCS_POINT_COUNT)
            .map(|index| {
                squeeze_pi_ccs_challenge(
                    &mut builder,
                    &mut transcript,
                    neo_reductions::engines::pi_ccs_joint::ALPHA_TAG,
                    Some(index),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("26 fixed alpha challenges");
        let gamma = squeeze_pi_ccs_challenge(
            &mut builder,
            &mut transcript,
            neo_reductions::engines::pi_ccs_joint::GAMMA_TAG,
            None,
        );
        debug_assert_eq!(transcript.absorbed(), 0);
        builder.record_row_family(STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY, transcript_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.initial_claim");
        let initial_claim_start = builder.rows();
        let powers = gamma_powers(&mut builder, gamma, GAMMA_POWER_COUNT);
        let current = enforce_initial_claim(&mut builder, &powers, &variable_fields);
        builder.record_row_family(STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY, initial_claim_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.context");
        let context_start = builder.rows();
        let context_digest = enforce_context_digest(&mut builder, header, before, fresh_metadata_residual);
        let reverse_point = std::array::from_fn(|_| {
            KVar::new(
                alloc_constant(&mut builder, F::ZERO),
                alloc_constant(&mut builder, F::ZERO),
            )
        });
        let round_cursor = alloc_constant(&mut builder, F::ZERO);
        let after = StreamingPiCcsStateVars {
            transcript: transcript.variable_state(),
            current,
            reverse_point,
            round_cursor,
            context_digest,
        };
        builder.record_row_family(STREAMING_PI_CCS_START_CONTEXT_FAMILY, context_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.state_digest");
        let (before_local_state_digest, _) = digest_persistent_state(&mut builder, before);
        let after_local_state_digest = digest_streaming_pi_ccs_state(&mut builder, after);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.phase_envelope");
        let phase_envelope = enforce_streaming_carry_phase_semantic_envelope(
            &mut builder,
            before_local_state_digest,
            after_local_state_digest,
        );

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.start.state_x_out");
        let verifier_digest = alloc_fixture_digest(&mut builder, 71_000);
        let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);
        let before_boundary = alloc_fixture_digest(&mut builder, 71_200);
        let after_boundary = alloc_fixture_digest(&mut builder, 71_200);
        let before_accumulator = alloc_fixture_digest(&mut builder, 71_400);
        let after_accumulator = alloc_fixture_digest(&mut builder, 71_400);
        let nebula_lane_digest = alloc_fixture_digest(&mut builder, 71_600);
        let lifecycle_carry_start = builder.rows();
        enforce_digest_equal(&mut builder, before_boundary, after_boundary);
        enforce_digest_equal(&mut builder, before_accumulator, after_accumulator);
        builder.record_row_family(STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY, lifecycle_carry_start);

        let before_program_cursor = before.program_cursor.field;
        let after_program_cursor = alloc_bound_constant(&mut builder, FIRST_PI_CCS_ROUND_PROGRAM_CURSOR);
        enforce_add_one(&mut builder, before_program_cursor, after_program_cursor);
        let after_program_cursor_bits = decompose_var_to_u64_bits(&mut builder, after_program_cursor);
        let after_x_out = enforce_streaming_state_x_out(
            &mut builder,
            &StateXOutDigestInputs {
                mode: StateXOutDigestMode::Stateful,
                vk_fs_digest: verifier_digest,
                pi_ccs_header_bundle: header,
                structure_digest: header,
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
                pi_ccs_header_bundle: header,
                structure_digest: header,
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

        let mut public_outputs = Vec::with_capacity(10 * PUBLIC_WORD_BITS);
        public_outputs.extend(after_x_out.public_bits);
        public_outputs.extend(before_x_out.public_bits);
        public_outputs.extend(before.program_cursor.bits);
        public_outputs.extend(after_program_cursor_bits);
        let public_layout = NebulaFPrimeStreamingPublicLayout::production();
        debug_assert_eq!(public_outputs.len() + 1, public_layout.logical_columns());
        debug_assert_eq!(builder.first_unsatisfied_row(), None);
        debug_assert_eq!(builder.rows(), PI_CCS_START_SOURCE_ROWS);
        debug_assert_eq!(builder.cols(), PI_CCS_START_SOURCE_COLUMNS);
        debug_assert_eq!(public_outputs.len() + 1, PI_CCS_START_SOURCE_PUBLIC_COLUMNS);
        debug_assert_eq!(
            builder.poseidon2_permutation_audits().len(),
            PI_CCS_START_SOURCE_POSEIDON2_PERMUTATIONS
        );

        Ok(Self {
            builder,
            public_outputs,
            before,
            after,
            variable_fields,
            computed_statement_commitment,
            fresh_metadata_residual,
            alpha,
            gamma,
            before_program_cursor,
            after_program_cursor,
            before_x_out_preimage_columns,
            after_x_out_preimage_columns,
            before_boundary_columns: before_boundary.map(Var::col),
            after_boundary_columns: after_boundary.map(Var::col),
            before_accumulator_columns: before_accumulator.map(Var::col),
            after_accumulator_columns: after_accumulator.map(Var::col),
            phase_envelope,
        })
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

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        self.builder.first_unsatisfied_row()
    }

    pub fn unconstrained_columns(&self) -> Vec<usize> {
        self.builder.unconstrained_columns()
    }

    pub fn variable_field_column(&self, index: usize) -> Option<usize> {
        self.variable_fields.get(index).map(|wire| wire.col())
    }

    pub fn prior_point_columns(&self) -> [[usize; 2]; PI_CCS_POINT_COUNT] {
        std::array::from_fn(|point| {
            [
                self.variable_fields[2 * point].col(),
                self.variable_fields[2 * point + 1].col(),
            ]
        })
    }

    pub fn evaluation_columns(&self, running: usize, matrix: usize, coefficient: usize) -> Option<[usize; 2]> {
        (running < RUNNING_COUNT && matrix < MATRIX_COUNT && coefficient < COEFFICIENT_COUNT).then(|| {
            let start = evaluation_field_index(running, matrix, coefficient, 0);
            [self.variable_fields[start].col(), self.variable_fields[start + 1].col()]
        })
    }

    pub fn expected_statement_fresh_commitment_columns(&self) -> [usize; COORDINATE_COMMITMENT_FIELDS] {
        self.before
            .statement_fresh_commitment
            .map(|word| word.field.col())
    }

    pub fn expected_running_commitments_binding_columns(&self) -> [usize; COORDINATE_COMMITMENT_FIELDS] {
        self.before
            .running_commitments_binding
            .map(|word| word.field.col())
    }

    pub fn expected_running_public_binding_columns(&self) -> [usize; COORDINATE_COMMITMENT_FIELDS] {
        self.before
            .running_public_binding
            .map(|word| word.field.col())
    }

    pub fn before_runtime_columns(&self) -> [usize; PI_CCS_SPONGE_WIDTH] {
        self.before.runtime.lanes.map(|word| word.field.col())
    }

    pub fn before_runtime_absorbed_column(&self) -> usize {
        self.before.runtime.absorbed.field.col()
    }

    pub fn computed_statement_commitment_columns(&self) -> [usize; COORDINATE_COMMITMENT_FIELDS] {
        self.computed_statement_commitment.map(Var::col)
    }

    pub fn fresh_metadata_residual_columns(&self) -> [usize; COORDINATE_COMMITMENT_FIELDS] {
        self.fresh_metadata_residual.map(Var::col)
    }

    pub fn after_transcript_columns(&self) -> [usize; PI_CCS_SPONGE_WIDTH] {
        self.after.transcript.map(Var::col)
    }

    pub fn after_current_columns(&self) -> [usize; 2] {
        [self.after.current.c0.col(), self.after.current.c1.col()]
    }

    pub fn after_reverse_point_columns(&self) -> [[usize; 2]; PI_CCS_POINT_COUNT] {
        self.after
            .reverse_point
            .map(|point| [point.c0.col(), point.c1.col()])
    }

    pub fn after_round_cursor_column(&self) -> usize {
        self.after.round_cursor.col()
    }

    pub fn context_digest_columns(&self) -> [usize; PI_CCS_CONTEXT_DIGEST_FIELDS] {
        self.after.context_digest.map(Var::col)
    }

    pub fn alpha_columns(&self) -> [[usize; 2]; PI_CCS_POINT_COUNT] {
        self.alpha.map(|value| [value.c0.col(), value.c1.col()])
    }

    pub fn gamma_columns(&self) -> [usize; 2] {
        [self.gamma.c0.col(), self.gamma.c1.col()]
    }

    pub fn before_program_cursor_column(&self) -> usize {
        self.before_program_cursor.col()
    }

    pub fn after_program_cursor_column(&self) -> usize {
        self.after_program_cursor.col()
    }

    pub const fn before_x_out_preimage_columns(&self) -> [usize; 32] {
        self.before_x_out_preimage_columns
    }

    pub const fn after_x_out_preimage_columns(&self) -> [usize; 32] {
        self.after_x_out_preimage_columns
    }

    pub const fn before_boundary_columns(&self) -> [usize; 4] {
        self.before_boundary_columns
    }

    pub const fn after_boundary_columns(&self) -> [usize; 4] {
        self.after_boundary_columns
    }

    pub const fn before_accumulator_columns(&self) -> [usize; 4] {
        self.before_accumulator_columns
    }

    pub const fn after_accumulator_columns(&self) -> [usize; 4] {
        self.after_accumulator_columns
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

    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    pub fn shape_audit(&self) -> NebulaFPrimePiCcsStartShapeAudit {
        NebulaFPrimePiCcsStartShapeAudit {
            rows: self.rows(),
            columns: self.columns(),
            public_columns: self.public_columns(),
            variable_fields: self.variable_fields.len(),
            gamma_powers: GAMMA_POWER_COUNT,
            poseidon2_permutations: self.builder.poseidon2_permutation_audits().len(),
        }
    }

    #[doc(hidden)]
    pub fn builder_for_artifact(&self) -> &R1csBuilder {
        &self.builder
    }

    #[doc(hidden)]
    pub fn tamper_witness_for_test(&mut self, column: usize, value: F) {
        self.builder.tamper_witness(column, value);
    }

    fn into_sparse(self) -> Result<SparseR1cs, FieldR1csLoweringError> {
        Ok(lower_field_r1cs(self.builder, &self.public_outputs)?
            .into_parts()
            .0)
    }
}

#[doc(hidden)]
pub fn production_pi_ccs_start_source_arm() -> Result<SparseR1cs, NebulaFPrimePiCcsStartRelationError> {
    Ok(NebulaFPrimePiCcsStartSynthesis::production()?.into_sparse()?)
}

fn fixture_variable_fields() -> Vec<F> {
    (0..VARIABLE_FIELDS)
        .map(|index| F::from_usize(100_000 + 17 * index))
        .collect()
}

fn fixture_fresh_metadata_fields() -> Vec<F> {
    (0..PI_CCS_STATEMENT_FRESH_FIELDS - PI_CCS_STATEMENT_FIELDS)
        .map(|index| F::from_usize(500_000 + 19 * index))
        .collect()
}

fn fixture_running_commitment_fields() -> Vec<F> {
    (0..PI_CCS_RUNNING_COMMITMENT_FIELDS)
        .map(|index| F::from_usize(700_000 + 23 * index))
        .collect()
}

fn fixture_running_public_fields() -> Vec<F> {
    (0..PI_CCS_RUNNING_PUBLIC_FIELDS)
        .map(|index| F::from_usize(700_000 + 23 * (PI_CCS_RUNNING_COMMITMENT_FIELDS + index)))
        .collect()
}

fn fixture_ready_state(
    statement_fresh_commitment: [F; COORDINATE_COMMITMENT_FIELDS],
    running_commitments_binding: [F; COORDINATE_COMMITMENT_FIELDS],
    running_public_binding: [F; COORDINATE_COMMITMENT_FIELDS],
) -> PersistentState {
    let transcript = SpongeState {
        lanes: std::array::from_fn(|lane| F::from_usize(80_000 + 31 * lane)),
        absorbed: CLAIM_READY_ABSORBED as u64,
    };
    PersistentState {
        expected: transcript,
        runtime: transcript,
        frame_cursor: CLAIM_FRAME_FIELDS as u64,
        program_cursor: PI_CCS_START_PROGRAM_CURSOR as u64,
        statement_fresh_commitment,
        running_commitments_binding,
        running_public_binding,
    }
}

fn evaluation_field_index(running: usize, matrix: usize, coefficient: usize, limb: usize) -> usize {
    2 * PI_CCS_POINT_COUNT + 2 * ((running * MATRIX_COUNT + matrix) * COEFFICIENT_COUNT + coefficient) + limb
}

fn carried_exponent(running: usize, matrix: usize, coefficient: usize) -> usize {
    2 * FRESH_COUNT + RUNNING_COUNT + running + RUNNING_COUNT * matrix + RUNNING_COUNT * MATRIX_COUNT * coefficient
}

fn enforce_initial_claim(builder: &mut R1csBuilder, powers: &[KVar], variable_fields: &[Var]) -> KVar {
    assert_eq!(powers.len(), GAMMA_POWER_COUNT);
    assert_eq!(variable_fields.len(), VARIABLE_FIELDS);
    let mut sum = KLc::zero();
    for running in 0..RUNNING_COUNT {
        for matrix in 0..MATRIX_COUNT {
            for coefficient in 0..COEFFICIENT_COUNT {
                let start = evaluation_field_index(running, matrix, coefficient, 0);
                let evaluation = KVar::new(variable_fields[start], variable_fields[start + 1]);
                let term = enforce_k_mul(
                    builder,
                    &KLc::from_var(powers[carried_exponent(running, matrix, coefficient)]),
                    &KLc::from_var(evaluation),
                );
                sum.c0.add_term(term.c0, F::ONE);
                sum.c1.add_term(term.c1, F::ONE);
            }
        }
    }
    alloc_klc(builder, &sum)
}

fn enforce_context_digest(
    builder: &mut R1csBuilder,
    header: [Var; 4],
    claim_state: PersistentStateVars,
    fresh_metadata_residual: [Var; COORDINATE_COMMITMENT_FIELDS],
) -> [Var; PI_CCS_CONTEXT_DIGEST_FIELDS] {
    let mut fields = Vec::with_capacity(4 + PI_CCS_SPONGE_WIDTH + 1 + 3 * COORDINATE_COMMITMENT_FIELDS);
    fields.extend(header);
    fields.extend(claim_state.runtime.lanes.map(|word| word.field));
    fields.push(claim_state.runtime.absorbed.field);
    fields.extend(fresh_metadata_residual);
    fields.extend(
        claim_state
            .running_commitments_binding
            .map(|word| word.field),
    );
    fields.extend(claim_state.running_public_binding.map(|word| word.field));
    let mut transcript = TranscriptGadget::new(builder, CONTEXT_DOMAIN);
    transcript.append_fields(builder, CONTEXT_FIELDS_LABEL, &fields);
    transcript.digest_fields(builder)
}

fn enforce_digest_equal(builder: &mut R1csBuilder, before: [Var; 4], after: [Var; 4]) {
    for (before, after) in before.into_iter().zip(after) {
        builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
    }
}

fn enforce_constant(builder: &mut R1csBuilder, wire: Var, value: usize) {
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_usize(value)));
}

fn enforce_add_one(builder: &mut R1csBuilder, before: Var, after: Var) {
    let expected = Lc::from_var(before).add_scaled(&Lc::from_const(F::ONE), F::ONE);
    builder.enforce_eq(&Lc::from_var(after), &expected);
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

const _: () = assert!(VARIABLE_FIELDS == 24_244);
const _: () = assert!(CLAIM_READY_ABSORBED == 3);
const _: () = assert!(PI_CCS_START_PROGRAM_CURSOR == 193);
const _: () = assert!(GAMMA_POWER_COUNT == 2 * FRESH_COUNT + 2 * RUNNING_COUNT + RUNNING_COUNT * MATRIX_COUNT * D);
