//! Exact local relation for one production PiCCS SumCheck round.
//!
//! Owns the fixed transcript message, Poseidon2 challenge replay, degree-nine
//! SumCheck arithmetic, the 26-slot reverse challenge register, and the
//! authenticated phase envelope. The public schedule cursor selects the
//! round index. One unchanged context digest carries start-phase data that the
//! future finish phase must replay and open.
//!
//! Does not own PiCCS start or finish, the context-digest preimage, lifecycle
//! selection, terminal integration, or the Poseidon2 collision reduction.

use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{enforce_sumcheck_round_phase, KVar, Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError, SparseR1cs};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::{alloc_constant, StateXOutDigestInputs};
use crate::paper::reductions::pi_ccs_circuit::verifier::{PI_CCS_ROUND_CHALLENGE_TAG, PI_CCS_ROUND_MESSAGE_TAG};

use super::streaming_phase_envelope::{
    enforce_streaming_carry_phase_semantic_envelope, StreamingCarryPhaseSemanticEnvelope,
};
use super::streaming_pi_ccs_state::{
    alloc_streaming_pi_ccs_state as alloc_state, digest_streaming_pi_ccs_state as digest_state, PiCcsPair as Pair,
    StreamingPiCcsStateValue as PiCcsRoundStateValue, StreamingPiCcsStateVars as PiCcsRoundStateVars,
    PI_CCS_CONTEXT_DIGEST_FIELDS as CONTEXT_DIGEST_FIELDS, PI_CCS_LOCAL_STATE_FIELDS as LOCAL_STATE_FIELDS,
    PI_CCS_POINT_COUNT as POINT_COUNT, PI_CCS_SPONGE_WIDTH as SPONGE_WIDTH,
};
use super::streaming_program::FIRST_PI_CCS_ROUND_PROGRAM_CURSOR;
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;

const PUBLIC_WORD_BITS: usize = 64;
const COEFFICIENT_COUNT: usize = 10;
const ROUND_FIXTURE_INDEX: usize = 7;

pub const STREAMING_PI_CCS_ROUND_BEFORE_STATE_FAMILY: &str = "fprime.streaming.pi_ccs.round.before_state";
pub const STREAMING_PI_CCS_ROUND_AFTER_STATE_FAMILY: &str = "fprime.streaming.pi_ccs.round.after_state";
pub const STREAMING_PI_CCS_ROUND_COEFFICIENT_FAMILY: &str = "fprime.streaming.pi_ccs.round.coefficients";
pub const STREAMING_PI_CCS_ROUND_TRANSCRIPT_FAMILY: &str = "fprime.streaming.pi_ccs.round.transcript";
pub const STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY: &str = "fprime.streaming.pi_ccs.round.arithmetic";
pub const STREAMING_PI_CCS_ROUND_STATE_TRANSITION_FAMILY: &str = "fprime.streaming.pi_ccs.round.state_transition";
pub const STREAMING_PI_CCS_ROUND_STATE_DIGEST_FAMILY: &str = "fprime.streaming.pi_ccs.round.state_digest";
pub const STREAMING_PI_CCS_ROUND_LIFECYCLE_CARRY_FAMILY: &str = "fprime.streaming.pi_ccs.round.lifecycle_carry";

pub const PI_CCS_ROUND_SOURCE_ROWS: usize = 701_757;
pub const PI_CCS_ROUND_SOURCE_COLUMNS: usize = 701_828;
pub const PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS: usize = 641;
pub const PI_CCS_ROUND_SOURCE_POSEIDON2_PERMUTATIONS: usize = 1_157;
pub const PI_CCS_ROUND_SOURCE_ARTIFACT_ID: &str = "rust:streaming-pi-ccs-round/source-v1";
pub const PI_CCS_ROUND_SOURCE_SHA256: &str = "69479136251584cc11a49ffce566dc3f014a023710918322309a04ba5595fcb5";
pub const PI_CCS_ROUND_PROFILE_ID: &str = "nightstream/goldilocks/streaming-pi-ccs-round/v1";
pub const PI_CCS_ROUND_LIFECYCLE_SCOPE: &str = "recursive carry: PiCCS rounds 0..26";
pub const PI_CCS_ROUND_FIRST_PROGRAM_CURSOR: usize = FIRST_PI_CCS_ROUND_PROGRAM_CURSOR;
pub const PI_CCS_ROUND_AFTER_LAST_PROGRAM_CURSOR: usize = FIRST_PI_CCS_ROUND_PROGRAM_CURSOR + POINT_COUNT;
pub const PI_CCS_ROUND_FINAL_COMMON_PUBLIC_COLUMNS: usize = 648;
pub const PI_CCS_ROUND_COMPACT_ARITHMETIC_ARTIFACT_ID: &str = "lean:f-prime-full-history/pi-ccs-round-selective-ccs/v1";

/// One exact source-stage interval in normalized column space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsRoundSourceStage {
    path: &'static str,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
}

impl NebulaFPrimePiCcsRoundSourceStage {
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

pub const PI_CCS_ROUND_SOURCE_STAGE_SCHEDULE: [NebulaFPrimePiCcsRoundSourceStage; 7] = [
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.state_words",
        row_start: 0,
        row_end: 0,
        column_start: 641,
        column_end: 795,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.transcript",
        row_start: 0,
        row_end: 4_212,
        column_start: 795,
        column_end: 4_999,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.arithmetic",
        row_start: 4_212,
        row_end: 4_243,
        column_start: 4_999,
        column_end: 5_026,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.state_transition",
        row_start: 4_243,
        row_end: 4_442,
        column_start: 5_026,
        column_end: 5_032,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.state_digest",
        row_start: 4_442,
        row_end: 27_272,
        column_start: 5_032,
        column_end: 27_862,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.phase_envelope",
        row_start: 27_272,
        row_end: 690_243,
        column_start: 27_862,
        column_end: 690_833,
    },
    NebulaFPrimePiCcsRoundSourceStage {
        path: "nebula.streaming.pi_ccs.round.state_x_out",
        row_start: 690_243,
        row_end: 701_757,
        column_start: 690_833,
        column_end: 701_828,
    },
];

/// Frozen source and common-selective public-column boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsRoundColumnLayout {
    constant_one: usize,
    after_x_out_bits: (usize, usize),
    before_x_out_bits: (usize, usize),
    before_cursor_bits: (usize, usize),
    after_cursor_bits: (usize, usize),
    common_public_padding: (usize, usize),
    private_columns: (usize, usize),
}

impl NebulaFPrimePiCcsRoundColumnLayout {
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

pub const PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT: NebulaFPrimePiCcsRoundColumnLayout = NebulaFPrimePiCcsRoundColumnLayout {
    constant_one: 0,
    after_x_out_bits: (1, 257),
    before_x_out_bits: (257, 513),
    before_cursor_bits: (513, 577),
    after_cursor_bits: (577, 641),
    common_public_padding: (641, PI_CCS_ROUND_FINAL_COMMON_PUBLIC_COLUMNS),
    private_columns: (641, PI_CCS_ROUND_SOURCE_COLUMNS),
};

/// Exact phase-local arithmetic slice already checked by the generated Lean
/// selective-CCS recipe. Global row placement remains owned by the future
/// complete 23-kind schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsRoundArithmeticBinding {
    source_rows: (usize, usize),
    phase_local_selective_rows: (usize, usize),
    artifact_identity: &'static str,
}

impl NebulaFPrimePiCcsRoundArithmeticBinding {
    pub const fn source_rows(self) -> (usize, usize) {
        self.source_rows
    }

    pub const fn phase_local_selective_rows(self) -> (usize, usize) {
        self.phase_local_selective_rows
    }

    pub const fn artifact_identity(self) -> &'static str {
        self.artifact_identity
    }
}

pub const PI_CCS_ROUND_ARITHMETIC_BINDING: NebulaFPrimePiCcsRoundArithmeticBinding =
    NebulaFPrimePiCcsRoundArithmeticBinding {
        source_rows: (4_212, 4_243),
        phase_local_selective_rows: (0, 31),
        artifact_identity: PI_CCS_ROUND_COMPACT_ARITHMETIC_ARTIFACT_ID,
    };

#[derive(Debug, Error)]
pub enum NebulaFPrimePiCcsRoundRelationError {
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiCcsRoundShapeAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub arithmetic_rows: usize,
    pub poseidon2_permutations: usize,
}

/// One exact Rust-emitted field-R1CS source for circuit kind 6.
pub struct NebulaFPrimePiCcsRoundSynthesis {
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    before: PiCcsRoundStateVars,
    after: PiCcsRoundStateVars,
    coefficients: [KVar; COEFFICIENT_COUNT],
    challenge: KVar,
    before_program_cursor: Var,
    after_program_cursor: Var,
    before_x_out_preimage_columns: [usize; 32],
    after_x_out_preimage_columns: [usize; 32],
    before_x_out_digest_columns: [usize; 4],
    after_x_out_digest_columns: [usize; 4],
    before_boundary_columns: [usize; 4],
    after_boundary_columns: [usize; 4],
    before_accumulator_columns: [usize; 4],
    after_accumulator_columns: [usize; 4],
    phase_envelope: StreamingCarryPhaseSemanticEnvelope,
}

impl NebulaFPrimePiCcsRoundSynthesis {
    pub fn production() -> Self {
        let (before_value, after_value, coefficient_values) = fixture_transition();
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.state_words");
        let before_start = builder.cols();
        let before = alloc_state(&mut builder, before_value);
        builder.record_column_family(STREAMING_PI_CCS_ROUND_BEFORE_STATE_FAMILY, before_start);
        let after_start = builder.cols();
        let after = alloc_state(&mut builder, after_value);
        builder.record_column_family(STREAMING_PI_CCS_ROUND_AFTER_STATE_FAMILY, after_start);
        let coefficient_start = builder.cols();
        let coefficients = coefficient_values.map(|value| KVar::alloc(&mut builder, value[0], value[1]));
        builder.record_column_family(STREAMING_PI_CCS_ROUND_COEFFICIENT_FAMILY, coefficient_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.transcript");
        let transcript_start = builder.rows();
        let mut transcript = TranscriptGadget::from_variable_state(before.transcript, 0);
        let mut message = Vec::with_capacity(3 + 2 * COEFFICIENT_COUNT);
        message.push(alloc_constant(&mut builder, F::from_u64(PI_CCS_ROUND_MESSAGE_TAG)));
        message.push(before.round_cursor);
        message.push(alloc_constant(&mut builder, F::from_usize(COEFFICIENT_COUNT)));
        for coefficient in coefficients {
            message.push(coefficient.c0);
            message.push(coefficient.c1);
        }
        transcript.append_fields_unframed_vars(&mut builder, &message);
        let challenge_frame = [
            alloc_constant(&mut builder, F::from_u64(PI_CCS_ROUND_CHALLENGE_TAG)),
            before.round_cursor,
        ];
        transcript.append_fields_unframed_vars(&mut builder, &challenge_frame);
        let challenge_lanes = transcript.challenge_fields_raw(&mut builder, 2);
        let challenge = KVar::new(challenge_lanes[0], challenge_lanes[1]);
        debug_assert_eq!(transcript.absorbed(), 0);
        for (&declared, computed) in after.transcript.iter().zip(transcript.variable_state()) {
            builder.enforce_eq(&Lc::from_var(declared), &Lc::from_var(computed));
        }
        builder.record_row_family(STREAMING_PI_CCS_ROUND_TRANSCRIPT_FAMILY, transcript_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.arithmetic");
        let arithmetic_start = builder.rows();
        enforce_sumcheck_round_phase(&mut builder, &coefficients, challenge, before.current, after.current);
        builder.record_row_family(STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY, arithmetic_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.state_transition");
        let state_transition_start = builder.rows();
        enforce_pair_eq(&mut builder, after.reverse_point[0], challenge);
        for index in 1..POINT_COUNT {
            enforce_pair_eq(
                &mut builder,
                after.reverse_point[index],
                before.reverse_point[index - 1],
            );
        }
        enforce_pair_zero(&mut builder, before.reverse_point[POINT_COUNT - 1]);
        enforce_add_one(&mut builder, before.round_cursor, after.round_cursor);
        for (&before, &after) in before.context_digest.iter().zip(&after.context_digest) {
            builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
        }
        let before_program_cursor =
            alloc_offset_word(&mut builder, before.round_cursor, FIRST_PI_CCS_ROUND_PROGRAM_CURSOR);
        let after_program_cursor =
            alloc_offset_word(&mut builder, after.round_cursor, FIRST_PI_CCS_ROUND_PROGRAM_CURSOR);
        let before_program_cursor_bits = decompose_var_to_u64_bits(&mut builder, before_program_cursor);
        let after_program_cursor_bits = decompose_var_to_u64_bits(&mut builder, after_program_cursor);
        builder.record_row_family(STREAMING_PI_CCS_ROUND_STATE_TRANSITION_FAMILY, state_transition_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.state_digest");
        let state_digest_start = builder.rows();
        let before_local_state_digest = digest_state(&mut builder, before);
        let after_local_state_digest = digest_state(&mut builder, after);
        builder.record_row_family(STREAMING_PI_CCS_ROUND_STATE_DIGEST_FAMILY, state_digest_start);

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.phase_envelope");
        let phase_envelope = enforce_streaming_carry_phase_semantic_envelope(
            &mut builder,
            before_local_state_digest,
            after_local_state_digest,
        );

        builder.begin_encoding_stage("nebula.streaming.pi_ccs.round.state_x_out");
        let verifier_digest = alloc_fixture_digest(&mut builder, 50_000);
        let pi_ccs_header_bundle = alloc_fixture_digest(&mut builder, 50_100);
        let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);
        let before_boundary = alloc_fixture_digest(&mut builder, 50_200);
        let after_boundary = alloc_fixture_digest(&mut builder, 50_200);
        let before_accumulator = alloc_fixture_digest(&mut builder, 50_400);
        let after_accumulator = alloc_fixture_digest(&mut builder, 50_400);
        let nebula_lane_digest = alloc_fixture_digest(&mut builder, 50_600);
        let lifecycle_carry_start = builder.rows();
        for (&before, &after) in before_boundary.iter().zip(&after_boundary) {
            builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
        }
        for (&before, &after) in before_accumulator.iter().zip(&after_accumulator) {
            builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
        }
        builder.record_row_family(STREAMING_PI_CCS_ROUND_LIFECYCLE_CARRY_FAMILY, lifecycle_carry_start);
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
        debug_assert_eq!(builder.rows(), PI_CCS_ROUND_SOURCE_ROWS);
        debug_assert_eq!(builder.cols(), PI_CCS_ROUND_SOURCE_COLUMNS);
        debug_assert_eq!(public_outputs.len() + 1, PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS);
        debug_assert_eq!(
            builder.poseidon2_permutation_audits().len(),
            PI_CCS_ROUND_SOURCE_POSEIDON2_PERMUTATIONS
        );
        debug_assert_eq!(builder.first_unsatisfied_row(), None);

        Self {
            builder,
            public_outputs,
            before,
            after,
            coefficients,
            challenge,
            before_program_cursor,
            after_program_cursor,
            before_x_out_preimage_columns,
            after_x_out_preimage_columns,
            before_x_out_digest_columns,
            after_x_out_digest_columns,
            before_boundary_columns: before_boundary.map(Var::col),
            after_boundary_columns: after_boundary.map(Var::col),
            before_accumulator_columns: before_accumulator.map(Var::col),
            after_accumulator_columns: after_accumulator.map(Var::col),
            phase_envelope,
        }
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

    pub fn before_transcript_columns(&self) -> [usize; SPONGE_WIDTH] {
        self.before.transcript.map(Var::col)
    }

    pub fn after_transcript_columns(&self) -> [usize; SPONGE_WIDTH] {
        self.after.transcript.map(Var::col)
    }

    pub fn before_current_columns(&self) -> [usize; 2] {
        [self.before.current.c0.col(), self.before.current.c1.col()]
    }

    pub fn after_current_columns(&self) -> [usize; 2] {
        [self.after.current.c0.col(), self.after.current.c1.col()]
    }

    pub fn coefficient_columns(&self) -> [[usize; 2]; COEFFICIENT_COUNT] {
        self.coefficients
            .map(|coefficient| [coefficient.c0.col(), coefficient.c1.col()])
    }

    pub fn challenge_columns(&self) -> [usize; 2] {
        [self.challenge.c0.col(), self.challenge.c1.col()]
    }

    pub fn before_reverse_point_columns(&self) -> [[usize; 2]; POINT_COUNT] {
        self.before
            .reverse_point
            .map(|point| [point.c0.col(), point.c1.col()])
    }

    pub fn after_reverse_point_columns(&self) -> [[usize; 2]; POINT_COUNT] {
        self.after
            .reverse_point
            .map(|point| [point.c0.col(), point.c1.col()])
    }

    pub fn before_round_cursor_column(&self) -> usize {
        self.before.round_cursor.col()
    }

    pub fn after_round_cursor_column(&self) -> usize {
        self.after.round_cursor.col()
    }

    pub fn before_program_cursor_column(&self) -> usize {
        self.before_program_cursor.col()
    }

    pub fn after_program_cursor_column(&self) -> usize {
        self.after_program_cursor.col()
    }

    pub fn before_context_digest_columns(&self) -> [usize; CONTEXT_DIGEST_FIELDS] {
        self.before.context_digest.map(Var::col)
    }

    pub fn after_context_digest_columns(&self) -> [usize; CONTEXT_DIGEST_FIELDS] {
        self.after.context_digest.map(Var::col)
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

    pub fn phase_delayed_payload_columns(&self) -> Vec<usize> {
        self.phase_envelope
            .delayed_payload_bits
            .iter()
            .map(|wire| wire.col())
            .collect()
    }

    pub const fn before_x_out_preimage_columns(&self) -> [usize; 32] {
        self.before_x_out_preimage_columns
    }

    pub const fn after_x_out_preimage_columns(&self) -> [usize; 32] {
        self.after_x_out_preimage_columns
    }

    pub const fn before_x_out_digest_columns(&self) -> [usize; 4] {
        self.before_x_out_digest_columns
    }

    pub const fn after_x_out_digest_columns(&self) -> [usize; 4] {
        self.after_x_out_digest_columns
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

    pub fn public_output_column(&self, index: usize) -> Option<usize> {
        self.public_outputs.get(index).map(|wire| wire.col())
    }

    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    pub fn shape_audit(&self) -> NebulaFPrimePiCcsRoundShapeAudit {
        let arithmetic = self
            .builder
            .row_family_ranges()
            .iter()
            .find(|range| range.name == STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY)
            .expect("one PiCCS round arithmetic family");
        NebulaFPrimePiCcsRoundShapeAudit {
            rows: self.rows(),
            columns: self.columns(),
            public_columns: self.public_columns(),
            arithmetic_rows: arithmetic.row_end - arithmetic.row_start,
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
pub fn production_pi_ccs_round_source_arm() -> Result<SparseR1cs, NebulaFPrimePiCcsRoundRelationError> {
    Ok(NebulaFPrimePiCcsRoundSynthesis::production().into_sparse()?)
}

fn fixture_transition() -> (PiCcsRoundStateValue, PiCcsRoundStateValue, [Pair; COEFFICIENT_COUNT]) {
    let coefficients = std::array::from_fn(|index| [F::from_usize(3 + 5 * index), F::from_usize(7 + 11 * index)]);
    let current = round_initial(&coefficients);
    let before_transcript = std::array::from_fn(|lane| F::from_usize(60_000 + 17 * lane));
    let mut native = Poseidon2Transcript::from_state_and_absorbed(before_transcript, 0);
    let mut message = Vec::with_capacity(3 + 2 * COEFFICIENT_COUNT);
    message.push(F::from_u64(PI_CCS_ROUND_MESSAGE_TAG));
    message.push(F::from_usize(ROUND_FIXTURE_INDEX));
    message.push(F::from_usize(COEFFICIENT_COUNT));
    for coefficient in coefficients {
        message.extend(coefficient);
    }
    native.append_fields_unframed(&message);
    native.append_fields_unframed(&[
        F::from_u64(PI_CCS_ROUND_CHALLENGE_TAG),
        F::from_usize(ROUND_FIXTURE_INDEX),
    ]);
    let challenge_lanes = native.challenge_fields_raw(2);
    let challenge = [challenge_lanes[0], challenge_lanes[1]];
    debug_assert_eq!(native.absorbed(), 0);

    let reverse_point = std::array::from_fn(|index| {
        if index < ROUND_FIXTURE_INDEX {
            [F::from_usize(61_000 + 2 * index), F::from_usize(61_001 + 2 * index)]
        } else {
            [F::ZERO; 2]
        }
    });
    let context_digest = std::array::from_fn(|lane| F::from_usize(62_000 + lane));
    let before = PiCcsRoundStateValue {
        transcript: before_transcript,
        current,
        reverse_point,
        round_cursor: F::from_usize(ROUND_FIXTURE_INDEX),
        context_digest,
    };
    let after = PiCcsRoundStateValue {
        transcript: native.state(),
        current: evaluate(&coefficients, challenge),
        reverse_point: std::array::from_fn(|index| {
            if index == 0 {
                challenge
            } else {
                reverse_point[index - 1]
            }
        }),
        round_cursor: F::from_usize(ROUND_FIXTURE_INDEX + 1),
        context_digest,
    };
    (before, after, coefficients)
}

fn enforce_pair_eq(builder: &mut R1csBuilder, left: KVar, right: KVar) {
    builder.enforce_eq(&Lc::from_var(left.c0), &Lc::from_var(right.c0));
    builder.enforce_eq(&Lc::from_var(left.c1), &Lc::from_var(right.c1));
}

fn enforce_pair_zero(builder: &mut R1csBuilder, value: KVar) {
    builder.enforce_eq(&Lc::from_var(value.c0), &Lc::zero());
    builder.enforce_eq(&Lc::from_var(value.c1), &Lc::zero());
}

fn enforce_add_one(builder: &mut R1csBuilder, before: Var, after: Var) {
    let mut expected = Lc::from_var(before);
    expected.add_constant(F::ONE);
    builder.enforce_eq(&Lc::from_var(after), &expected);
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

fn pair_add(left: Pair, right: Pair) -> Pair {
    [left[0] + right[0], left[1] + right[1]]
}

fn pair_mul(left: Pair, right: Pair) -> Pair {
    [
        left[0] * right[0] + F::from_u64(7) * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

fn evaluate(coefficients: &[Pair], point: Pair) -> Pair {
    coefficients
        .iter()
        .rev()
        .fold([F::ZERO; 2], |suffix, &coefficient| {
            pair_add(coefficient, pair_mul(point, suffix))
        })
}

fn round_initial(coefficients: &[Pair]) -> Pair {
    let sum = coefficients.iter().copied().fold([F::ZERO; 2], pair_add);
    pair_add(coefficients[0], sum)
}

const _: () = assert!(LOCAL_STATE_FIELDS == 67);
const _: () = assert!(COEFFICIENT_COUNT == 10);
const _: () = assert!(POINT_COUNT == 26);
