//! Exact bounded replay of the prior recursive state.
//!
//! Owns the shared 1,024-field arm, the 522-field final arm, the compact
//! ten-field continuation transition, local-state digests, phase envelope,
//! and full before/after XOut recomputation. The final target is an exposed
//! source slice. Its link to the verifier-owned running instance is a later
//! lifecycle-composition obligation, so this module does not treat it as
//! independent authority.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError, SparseR1cs};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::{digest32_as_fields, StateXOutDigestMode};
use crate::paper::f_prime::digest_circuit::StateXOutDigestInputs;

use super::streaming_phase_envelope::{
    enforce_streaming_carry_phase_semantic_envelope, StreamingCarryPhaseSemanticEnvelope,
};
use super::streaming_program::{FIRST_CLAIM_PROGRAM_CURSOR, PRIOR_STATE_FRAME_FIELDS, STATE_CHUNK_FIELDS};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;

const SPONGE_WIDTH: usize = 8;
const RATE: usize = 4;
const DIGEST_FIELDS: usize = 4;
const REPLAY_STATE_FIELDS: usize = SPONGE_WIDTH + 2;
const PUBLIC_WORD_BITS: usize = 64;
const FINAL_FIELDS: usize = PRIOR_STATE_FRAME_FIELDS % STATE_CHUNK_FIELDS;
const FULL_CHUNKS: usize = PRIOR_STATE_FRAME_FIELDS / STATE_CHUNK_FIELDS;
const FIRST_PROGRAM_CURSOR: usize = 1;
const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-prior-state-replay-state/v1";
const STATE_DIGEST_FIELDS_LABEL: &[u8] = b"state";

pub const PRIOR_STATE_REPLAY_CHUNK_FIELDS: usize = STATE_CHUNK_FIELDS;
pub const PRIOR_STATE_REPLAY_FINAL_FIELDS: usize = FINAL_FIELDS;
pub const PRIOR_STATE_REPLAY_FULL_CHUNKS: usize = FULL_CHUNKS;
pub const PRIOR_STATE_REPLAY_CHUNKS: usize = FULL_CHUNKS + 1;
pub const PRIOR_STATE_REPLAY_FRAME_FIELDS: usize = PRIOR_STATE_FRAME_FIELDS;
pub const PRIOR_STATE_REPLAY_FIRST_PROGRAM_CURSOR: usize = FIRST_PROGRAM_CURSOR;
pub const PRIOR_STATE_REPLAY_AFTER_LAST_PROGRAM_CURSOR: usize = FIRST_CLAIM_PROGRAM_CURSOR;
pub const PRIOR_STATE_REPLAY_PROFILE_ID: &str = "nightstream/goldilocks/b2-k16/streaming-prior-state-replay/v1";
pub const PRIOR_STATE_REPLAY_SOURCE_ARTIFACT_ID: &str = "rust:streaming-prior-state-replay/source-b2-k16-v1";
pub const PRIOR_STATE_REPLAY_SOURCE_HASH_SCHEMA: &str = "nightstream-normalized-sparse-r1cs-csc-v1";
pub const PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE: &str = "recursive transition: prior-state replay indices 0..94";
pub const PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS: &str =
    "pending final selective link to the verifier-owned running-instance prior-state digest";
pub const PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS: usize = 833_066;
pub const PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS: usize = 834_087;
pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS: usize = 758_578;
pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS: usize = 759_094;
pub const PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS: usize = 641;
pub const PRIOR_STATE_REPLAY_FINAL_COMMON_PUBLIC_COLUMNS: usize = 648;
pub const PRIOR_STATE_REPLAY_FULL_SOURCE_POSEIDON2_PERMUTATIONS: usize = 1_376;
pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_POSEIDON2_PERMUTATIONS: usize = 1_251;
pub const PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256: &str =
    "7aa0f51acabf8fdae6107b8186a398ae715491cffa963fbe8db22f30f4c829d1";
pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256: &str =
    "78616d01f4038c7eed43f489f3bb67788bea70d63cc9eb9733f0790d03257078";

pub const STREAMING_PRIOR_STATE_REPLAY_BEFORE_STATE_FAMILY: &str = "fprime.streaming.prior_state_replay.before_state";
pub const STREAMING_PRIOR_STATE_REPLAY_AFTER_STATE_FAMILY: &str = "fprime.streaming.prior_state_replay.after_state";
pub const STREAMING_PRIOR_STATE_REPLAY_CHUNK_FAMILY: &str = "fprime.streaming.prior_state_replay.chunk";
pub const STREAMING_PRIOR_STATE_REPLAY_STATE_TRANSITION_FAMILY: &str =
    "fprime.streaming.prior_state_replay.state_transition";
pub const STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY: &str = "fprime.streaming.prior_state_replay.final_target";
pub const STREAMING_PRIOR_STATE_REPLAY_LIFECYCLE_CARRY_FAMILY: &str =
    "fprime.streaming.prior_state_replay.lifecycle_carry";

/// One exact source-stage interval in normalized column space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePriorStateReplaySourceStage {
    path: &'static str,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
}

impl NebulaFPrimePriorStateReplaySourceStage {
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

pub const PRIOR_STATE_REPLAY_FULL_SOURCE_STAGE_SCHEDULE: [NebulaFPrimePriorStateReplaySourceStage; 6] = [
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_words",
        row_start: 0,
        row_end: 138,
        column_start: 641,
        column_end: 667,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.chunk",
        row_start: 138,
        row_end: 138,
        column_start: 667,
        column_end: 1_691,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_transition",
        row_start: 138,
        row_end: 153_751,
        column_start: 1_691,
        column_end: 155_291,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_digest",
        row_start: 153_751,
        row_end: 158_581,
        column_start: 155_291,
        column_end: 160_121,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.phase_envelope",
        row_start: 158_581,
        row_end: 821_552,
        column_start: 160_121,
        column_end: 823_092,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_x_out",
        row_start: 821_552,
        row_end: PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS,
        column_start: 823_092,
        column_end: PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS,
    },
];

pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_STAGE_SCHEDULE: [NebulaFPrimePriorStateReplaySourceStage; 7] = [
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_words",
        row_start: 0,
        row_end: 138,
        column_start: 641,
        column_end: 671,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.chunk",
        row_start: 138,
        row_end: 138,
        column_start: 671,
        column_end: 1_695,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_transition",
        row_start: 138,
        row_end: 78_151,
        column_start: 1_695,
        column_end: 79_695,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.final_target",
        row_start: 78_151,
        row_end: 79_263,
        column_start: 79_695,
        column_end: 80_298,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_digest",
        row_start: 79_263,
        row_end: 84_093,
        column_start: 80_298,
        column_end: 85_128,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.phase_envelope",
        row_start: 84_093,
        row_end: 747_064,
        column_start: 85_128,
        column_end: 748_099,
    },
    NebulaFPrimePriorStateReplaySourceStage {
        path: "nebula.streaming.prior_state_replay.state_x_out",
        row_start: 747_064,
        row_end: PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS,
        column_start: 748_099,
        column_end: PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS,
    },
];

/// Frozen source and common-selective public-column boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePriorStateReplayColumnLayout {
    private_columns: (usize, usize),
}

impl NebulaFPrimePriorStateReplayColumnLayout {
    pub const fn constant_one(self) -> usize {
        0
    }

    pub const fn after_x_out_bits(self) -> (usize, usize) {
        (1, 257)
    }

    pub const fn before_x_out_bits(self) -> (usize, usize) {
        (257, 513)
    }

    pub const fn before_cursor_bits(self) -> (usize, usize) {
        (513, 577)
    }

    pub const fn after_cursor_bits(self) -> (usize, usize) {
        (577, 641)
    }

    pub const fn common_public_padding(self) -> (usize, usize) {
        (641, PRIOR_STATE_REPLAY_FINAL_COMMON_PUBLIC_COLUMNS)
    }

    pub const fn private_columns(self) -> (usize, usize) {
        self.private_columns
    }
}

pub const PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMN_LAYOUT: NebulaFPrimePriorStateReplayColumnLayout =
    NebulaFPrimePriorStateReplayColumnLayout {
        private_columns: (
            PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS,
        ),
    };

pub const PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMN_LAYOUT: NebulaFPrimePriorStateReplayColumnLayout =
    NebulaFPrimePriorStateReplayColumnLayout {
        private_columns: (
            PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS,
        ),
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePriorStateReplayArmKind {
    Full,
    Final,
}

impl NebulaFPrimePriorStateReplayArmKind {
    pub const fn active_fields(self) -> usize {
        match self {
            Self::Full => STATE_CHUNK_FIELDS,
            Self::Final => FINAL_FIELDS,
        }
    }

    const fn fixture_index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::Final => FULL_CHUNKS,
        }
    }
}

#[derive(Clone, Copy)]
struct ReplayState {
    lanes: [F; SPONGE_WIDTH],
    absorbed: u64,
    cursor: u64,
}

#[derive(Clone, Copy)]
struct ReplayStateVars {
    lanes: [Var; SPONGE_WIDTH],
    absorbed: Var,
    cursor: Var,
}

#[derive(Clone, Copy)]
struct CanonicalWord {
    field: Var,
    bits: [Var; PUBLIC_WORD_BITS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePriorStateReplayShapeAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub poseidon2_permutations: usize,
}

#[derive(Debug, Error)]
pub enum NebulaFPrimePriorStateReplayRelationError {
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
}

/// One exact Rust-emitted field-R1CS source for circuit kind 1 or 2.
pub struct NebulaFPrimePriorStateReplaySynthesis {
    kind: NebulaFPrimePriorStateReplayArmKind,
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    before_state_columns: [usize; REPLAY_STATE_FIELDS],
    after_state_columns: [usize; REPLAY_STATE_FIELDS],
    chunk_columns: [usize; STATE_CHUNK_FIELDS],
    target_digest_columns: Option<[usize; DIGEST_FIELDS]>,
    before_program_cursor: CanonicalWord,
    after_program_cursor: CanonicalWord,
    before_x_out_preimage_columns: [usize; 32],
    after_x_out_preimage_columns: [usize; 32],
    before_boundary_columns: [usize; DIGEST_FIELDS],
    after_boundary_columns: [usize; DIGEST_FIELDS],
    before_accumulator_columns: [usize; DIGEST_FIELDS],
    after_accumulator_columns: [usize; DIGEST_FIELDS],
    phase_envelope: StreamingCarryPhaseSemanticEnvelope,
}

impl NebulaFPrimePriorStateReplaySynthesis {
    pub fn production_full() -> Self {
        Self::production(NebulaFPrimePriorStateReplayArmKind::Full)
    }

    pub fn production_final() -> Self {
        Self::production(NebulaFPrimePriorStateReplayArmKind::Final)
    }

    fn production(kind: NebulaFPrimePriorStateReplayArmKind) -> Self {
        let chunk_index = kind.fixture_index();
        let chunk = fixture_chunk(kind, chunk_index);
        let before = ReplayState {
            lanes: std::array::from_fn(|lane| F::from_u64(0x1000 + 17 * chunk_index as u64 + lane as u64)),
            absorbed: 0,
            cursor: (chunk_index * STATE_CHUNK_FIELDS) as u64,
        };
        let after = absorb(before, &chunk[..kind.active_fields()]);
        let before_program_cursor_value = FIRST_PROGRAM_CURSOR + chunk_index;
        let after_program_cursor_value = before_program_cursor_value + 1;

        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.state_words");
        let before_state_start = builder.cols();
        let before_vars = alloc_replay_state(&mut builder, before);
        builder.record_column_family(STREAMING_PRIOR_STATE_REPLAY_BEFORE_STATE_FAMILY, before_state_start);
        let after_state_start = builder.cols();
        let after_vars = alloc_replay_state(&mut builder, after);
        builder.record_column_family(STREAMING_PRIOR_STATE_REPLAY_AFTER_STATE_FAMILY, after_state_start);
        let before_program_cursor = alloc_canonical_word(&mut builder, before_program_cursor_value as u64);
        let after_program_cursor = alloc_canonical_word(&mut builder, after_program_cursor_value as u64);
        let target_digest = (kind == NebulaFPrimePriorStateReplayArmKind::Final)
            .then(|| gated_digest(after).map(|value| builder.alloc(value)));

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.chunk");
        let chunk_start = builder.cols();
        let chunk_vars: [Var; STATE_CHUNK_FIELDS] = builder
            .alloc_vec(&chunk)
            .try_into()
            .expect("fixed prior-state chunk width");
        builder.record_column_family(STREAMING_PRIOR_STATE_REPLAY_CHUNK_FAMILY, chunk_start);

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.state_transition");
        let transition_start = builder.rows();
        enforce_constant(&mut builder, before_vars.absorbed, 0);
        enforce_constant(&mut builder, after_vars.absorbed, (kind.active_fields() % RATE) as u64);
        enforce_cursor_alignment(&mut builder, before_vars.cursor, before_program_cursor.field);
        enforce_add_constant(
            &mut builder,
            before_vars.cursor,
            after_vars.cursor,
            kind.active_fields() as u64,
        );
        enforce_add_constant(&mut builder, before_program_cursor.field, after_program_cursor.field, 1);
        let mut transcript = TranscriptGadget::from_variable_state(before_vars.lanes, 0);
        transcript.append_fields_unframed_vars(&mut builder, &chunk_vars[..kind.active_fields()]);
        for (&declared, computed) in after_vars.lanes.iter().zip(transcript.variable_state()) {
            builder.enforce_eq(&Lc::from_var(declared), &Lc::from_var(computed));
        }
        builder.record_row_family(STREAMING_PRIOR_STATE_REPLAY_STATE_TRANSITION_FAMILY, transition_start);

        if let Some(target_digest) = target_digest {
            builder.begin_encoding_stage("nebula.streaming.prior_state_replay.final_target");
            let final_start = builder.rows();
            enforce_constant(
                &mut builder,
                before_vars.cursor,
                (FULL_CHUNKS * STATE_CHUNK_FIELDS) as u64,
            );
            enforce_constant(
                &mut builder,
                before_program_cursor.field,
                (FIRST_PROGRAM_CURSOR + FULL_CHUNKS) as u64,
            );
            enforce_constant(&mut builder, after_vars.cursor, PRIOR_STATE_FRAME_FIELDS as u64);
            for &padding in &chunk_vars[FINAL_FIELDS..] {
                enforce_constant(&mut builder, padding, 0);
            }
            let mut gate = TranscriptGadget::from_variable_state(after_vars.lanes, FINAL_FIELDS % RATE);
            let computed = gate.digest_fields(&mut builder);
            enforce_digest_equal(&mut builder, target_digest, computed);
            builder.record_row_family(STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY, final_start);
        }

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.state_digest");
        let before_local_state_digest = digest_replay_state(&mut builder, before_vars);
        let after_local_state_digest = digest_replay_state(&mut builder, after_vars);

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.phase_envelope");
        let phase_envelope = enforce_streaming_carry_phase_semantic_envelope(
            &mut builder,
            before_local_state_digest,
            after_local_state_digest,
        );

        builder.begin_encoding_stage("nebula.streaming.prior_state_replay.state_x_out");
        let verifier_digest = alloc_fixture_digest(&mut builder, 80_000);
        let header = alloc_fixture_digest(&mut builder, 80_100);
        let before_boundary = alloc_fixture_digest(&mut builder, 80_200);
        let after_boundary = alloc_fixture_digest(&mut builder, 80_200);
        let before_accumulator = alloc_fixture_digest(&mut builder, 80_300);
        let after_accumulator = alloc_fixture_digest(&mut builder, 80_300);
        let nebula_lane_digest = alloc_fixture_digest(&mut builder, 80_400);
        let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);
        let carry_start = builder.rows();
        enforce_digest_equal(&mut builder, before_boundary, after_boundary);
        enforce_digest_equal(&mut builder, before_accumulator, after_accumulator);
        builder.record_row_family(STREAMING_PRIOR_STATE_REPLAY_LIFECYCLE_CARRY_FAMILY, carry_start);

        let after_x_out = enforce_streaming_state_x_out(
            &mut builder,
            &StateXOutDigestInputs {
                mode: StateXOutDigestMode::Stateful,
                vk_fs_digest: verifier_digest,
                pi_ccs_header_bundle: header,
                structure_digest: header,
                chunk_count: after_program_cursor.field,
                step_count: after_program_cursor.field,
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
                chunk_count: before_program_cursor.field,
                step_count: before_program_cursor.field,
                initial_boundary: verifier_digest,
                current_boundary: before_boundary,
                pc,
                semantic_acc: phase_envelope.before_semantic_digest,
                construction2_acc: before_accumulator,
                public_trace: before_boundary,
            },
            nebula_lane_digest,
        );
        let mut public_outputs = Vec::with_capacity(10 * PUBLIC_WORD_BITS);
        public_outputs.extend(after_x_out.public_bits);
        public_outputs.extend(before_x_out.public_bits);
        public_outputs.extend(before_program_cursor.bits);
        public_outputs.extend(after_program_cursor.bits);
        builder.begin_encoding_stage("complete");
        debug_assert_eq!(
            public_outputs.len() + 1,
            NebulaFPrimeStreamingPublicLayout::production().logical_columns()
        );
        debug_assert_eq!(builder.first_unsatisfied_row(), None);

        Self {
            kind,
            before_state_columns: replay_state_fields(before_vars).map(Var::col),
            after_state_columns: replay_state_fields(after_vars).map(Var::col),
            chunk_columns: chunk_vars.map(Var::col),
            target_digest_columns: target_digest.map(|digest| digest.map(Var::col)),
            before_program_cursor,
            after_program_cursor,
            before_x_out_preimage_columns: before_x_out
                .preimage
                .iter()
                .map(|wire| wire.col())
                .collect::<Vec<_>>()
                .try_into()
                .expect("stateful before XOut has 32 fields"),
            after_x_out_preimage_columns: after_x_out
                .preimage
                .iter()
                .map(|wire| wire.col())
                .collect::<Vec<_>>()
                .try_into()
                .expect("stateful after XOut has 32 fields"),
            before_boundary_columns: before_boundary.map(Var::col),
            after_boundary_columns: after_boundary.map(Var::col),
            before_accumulator_columns: before_accumulator.map(Var::col),
            after_accumulator_columns: after_accumulator.map(Var::col),
            phase_envelope,
            builder,
            public_outputs,
        }
    }

    pub const fn kind(&self) -> NebulaFPrimePriorStateReplayArmKind {
        self.kind
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

    pub const fn before_state_columns(&self) -> [usize; REPLAY_STATE_FIELDS] {
        self.before_state_columns
    }

    pub const fn after_state_columns(&self) -> [usize; REPLAY_STATE_FIELDS] {
        self.after_state_columns
    }

    pub const fn chunk_columns(&self) -> [usize; STATE_CHUNK_FIELDS] {
        self.chunk_columns
    }

    pub const fn target_digest_columns(&self) -> Option<[usize; DIGEST_FIELDS]> {
        self.target_digest_columns
    }

    pub fn before_program_cursor_column(&self) -> usize {
        self.before_program_cursor.field.col()
    }

    pub fn after_program_cursor_column(&self) -> usize {
        self.after_program_cursor.field.col()
    }

    pub const fn before_x_out_preimage_columns(&self) -> [usize; 32] {
        self.before_x_out_preimage_columns
    }

    pub const fn after_x_out_preimage_columns(&self) -> [usize; 32] {
        self.after_x_out_preimage_columns
    }

    pub const fn before_boundary_columns(&self) -> [usize; DIGEST_FIELDS] {
        self.before_boundary_columns
    }

    pub const fn after_boundary_columns(&self) -> [usize; DIGEST_FIELDS] {
        self.after_boundary_columns
    }

    pub const fn before_accumulator_columns(&self) -> [usize; DIGEST_FIELDS] {
        self.before_accumulator_columns
    }

    pub const fn after_accumulator_columns(&self) -> [usize; DIGEST_FIELDS] {
        self.after_accumulator_columns
    }

    pub fn before_phase_local_state_source_columns(&self) -> [usize; DIGEST_FIELDS] {
        self.phase_envelope
            .before_local_state_source_digest
            .map(Var::col)
    }

    pub fn after_phase_local_state_source_columns(&self) -> [usize; DIGEST_FIELDS] {
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

    pub fn shape_audit(&self) -> NebulaFPrimePriorStateReplayShapeAudit {
        NebulaFPrimePriorStateReplayShapeAudit {
            rows: self.rows(),
            columns: self.columns(),
            public_columns: self.public_columns(),
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

    fn into_sparse(self) -> Result<SparseR1cs, NebulaFPrimePriorStateReplayRelationError> {
        Ok(lower_field_r1cs(self.builder, &self.public_outputs)?
            .into_parts()
            .0)
    }
}

pub fn production_prior_state_replay_full_source_arm() -> Result<SparseR1cs, NebulaFPrimePriorStateReplayRelationError>
{
    NebulaFPrimePriorStateReplaySynthesis::production_full().into_sparse()
}

pub fn production_prior_state_replay_final_source_arm() -> Result<SparseR1cs, NebulaFPrimePriorStateReplayRelationError>
{
    NebulaFPrimePriorStateReplaySynthesis::production_final().into_sparse()
}

fn fixture_chunk(kind: NebulaFPrimePriorStateReplayArmKind, chunk_index: usize) -> Vec<F> {
    (0..STATE_CHUNK_FIELDS)
        .map(|offset| {
            if offset < kind.active_fields() {
                let source = (chunk_index * STATE_CHUNK_FIELDS + offset) as u64;
                F::from_u64(source.wrapping_mul(0x9e37_79b9).wrapping_add(0x7f4a_7c15))
            } else {
                F::ZERO
            }
        })
        .collect()
}

fn absorb(before: ReplayState, fields: &[F]) -> ReplayState {
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(before.lanes, before.absorbed as usize);
    transcript.append_fields_unframed(fields);
    ReplayState {
        lanes: transcript.state(),
        absorbed: transcript.absorbed() as u64,
        cursor: before.cursor + fields.len() as u64,
    }
}

fn gated_digest(state: ReplayState) -> [F; DIGEST_FIELDS] {
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state.lanes, state.absorbed as usize);
    digest32_as_fields(transcript.digest32())
}

fn alloc_replay_state(builder: &mut R1csBuilder, state: ReplayState) -> ReplayStateVars {
    ReplayStateVars {
        lanes: state.lanes.map(|value| builder.alloc(value)),
        absorbed: builder.alloc(F::from_u64(state.absorbed)),
        cursor: builder.alloc(F::from_u64(state.cursor)),
    }
}

fn replay_state_fields(state: ReplayStateVars) -> [Var; REPLAY_STATE_FIELDS] {
    let mut fields = [Var::ONE; REPLAY_STATE_FIELDS];
    fields[..SPONGE_WIDTH].copy_from_slice(&state.lanes);
    fields[SPONGE_WIDTH] = state.absorbed;
    fields[SPONGE_WIDTH + 1] = state.cursor;
    fields
}

fn digest_replay_state(builder: &mut R1csBuilder, state: ReplayStateVars) -> [Var; DIGEST_FIELDS] {
    let fields = replay_state_fields(state);
    let mut transcript = TranscriptGadget::new(builder, STATE_DIGEST_DOMAIN);
    transcript.append_fields(builder, STATE_DIGEST_FIELDS_LABEL, &fields);
    transcript.digest_fields(builder)
}

fn alloc_canonical_word(builder: &mut R1csBuilder, value: u64) -> CanonicalWord {
    let field = builder.alloc(F::from_u64(value));
    let bits = decompose_var_to_u64_bits(builder, field);
    CanonicalWord { field, bits }
}

fn alloc_bound_constant(builder: &mut R1csBuilder, value: usize) -> Var {
    let value = F::from_usize(value);
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}

fn alloc_fixture_digest(builder: &mut R1csBuilder, start: usize) -> [Var; DIGEST_FIELDS] {
    std::array::from_fn(|lane| builder.alloc(F::from_usize(start + lane)))
}

fn enforce_constant(builder: &mut R1csBuilder, wire: Var, value: u64) {
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_u64(value)));
}

fn enforce_add_constant(builder: &mut R1csBuilder, before: Var, after: Var, increment: u64) {
    let expected = Lc::from_var(before).add_scaled(&Lc::from_const(F::from_u64(increment)), F::ONE);
    builder.enforce_eq(&Lc::from_var(after), &expected);
}

fn enforce_cursor_alignment(builder: &mut R1csBuilder, frame_cursor: Var, program_cursor: Var) {
    let item_index = Lc::from_var(program_cursor).add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    let expected = Lc::zero().add_scaled(&item_index, F::from_usize(STATE_CHUNK_FIELDS));
    builder.enforce_eq(&Lc::from_var(frame_cursor), &expected);
}

fn enforce_digest_equal(builder: &mut R1csBuilder, left: [Var; DIGEST_FIELDS], right: [Var; DIGEST_FIELDS]) {
    for (left, right) in left.into_iter().zip(right) {
        builder.enforce_eq(&Lc::from_var(left), &Lc::from_var(right));
    }
}

const _: () = assert!(PRIOR_STATE_FRAME_FIELDS == 95_754);
const _: () = assert!(STATE_CHUNK_FIELDS == 1_024);
const _: () = assert!(FULL_CHUNKS == 93);
const _: () = assert!(FINAL_FIELDS == 522);
const _: () = assert!(STATE_CHUNK_FIELDS - FINAL_FIELDS == 502);
const _: () = assert!(FIRST_CLAIM_PROGRAM_CURSOR == 95);
const _: () = assert!(REPLAY_STATE_FIELDS == 10);
