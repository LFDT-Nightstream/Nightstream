//! Canonical start of the phased prior-state replay.
//!
//! Owns the zero-to-initial transition for the compact ten-field replay
//! state, its Poseidon2 local-state digest, and the standard phased public
//! envelope. It does not own prior-state chunks or terminal acceptance.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError, SparseR1cs};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::StateXOutDigestInputs;

use super::streaming_phase_envelope::enforce_streaming_carry_phase_semantic_envelope;
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;

const PUBLIC_WORD_BITS: usize = 64;
const REPLAY_STATE_FIELDS: usize = 10;
const BEFORE_PROGRAM_CURSOR: usize = 0;
const AFTER_PROGRAM_CURSOR: usize = 1;
const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-prior-state-replay-state/v1";
const STATE_DIGEST_FIELDS_LABEL: &[u8] = b"state";

pub const STREAMING_PRELUDE_INITIAL_REPLAY_STATE_FAMILY: &str = "fprime.streaming.prelude.initial_replay_state";
pub const STREAMING_PRELUDE_INITIAL_REPLAY_STATE_ROWS_FAMILY: &str =
    "fprime.streaming.prelude.initial_replay_state.rows";

/// Exact source relation for the first verifier-selected work item.
pub struct NebulaFPrimeStreamingPreludeSynthesis {
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    initial_replay_state_columns: [usize; REPLAY_STATE_FIELDS],
    before_local_state_digest_columns: [usize; 4],
    after_local_state_digest_columns: [usize; 4],
    before_program_cursor_column: usize,
    after_program_cursor_column: usize,
}

impl NebulaFPrimeStreamingPreludeSynthesis {
    pub fn production() -> Self {
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();

        builder.begin_encoding_stage("nebula.streaming.prelude.initial_replay_state");
        let initial_state_start = builder.cols();
        let initial_replay_state = std::array::from_fn(|_| builder.alloc(F::ZERO));
        builder.record_column_family(STREAMING_PRELUDE_INITIAL_REPLAY_STATE_FAMILY, initial_state_start);
        let initial_rows_start = builder.rows();
        for &field in &initial_replay_state {
            builder.enforce_zero(&Lc::from_var(field));
        }
        builder.record_row_family(STREAMING_PRELUDE_INITIAL_REPLAY_STATE_ROWS_FAMILY, initial_rows_start);

        builder.begin_encoding_stage("nebula.streaming.prelude.state_digest");
        let before_local_state_digest = std::array::from_fn(|_| {
            let field = builder.alloc(F::ZERO);
            builder.enforce_zero(&Lc::from_var(field));
            field
        });
        let after_local_state_digest = digest_replay_state(&mut builder, &initial_replay_state);

        builder.begin_encoding_stage("nebula.streaming.prelude.phase_envelope");
        let phase_envelope = enforce_streaming_carry_phase_semantic_envelope(
            &mut builder,
            before_local_state_digest,
            after_local_state_digest,
        );

        builder.begin_encoding_stage("nebula.streaming.prelude.state_x_out");
        let verifier_digest = alloc_fixture_digest(&mut builder, 10_000);
        let header = alloc_fixture_digest(&mut builder, 10_100);
        let initial_boundary = alloc_fixture_digest(&mut builder, 10_200);
        let current_boundary = alloc_fixture_digest(&mut builder, 10_300);
        let accumulator = alloc_fixture_digest(&mut builder, 10_400);
        let nebula_lane_digest = alloc_fixture_digest(&mut builder, 10_500);
        let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);
        let before_program_cursor = alloc_bound_constant(&mut builder, BEFORE_PROGRAM_CURSOR);
        let after_program_cursor = alloc_bound_constant(&mut builder, AFTER_PROGRAM_CURSOR);
        enforce_add_one(&mut builder, before_program_cursor, after_program_cursor);

        let after_x_out = enforce_streaming_state_x_out(
            &mut builder,
            &StateXOutDigestInputs {
                mode: StateXOutDigestMode::Stateful,
                vk_fs_digest: verifier_digest,
                pi_ccs_header_bundle: header,
                structure_digest: header,
                chunk_count: after_program_cursor,
                step_count: after_program_cursor,
                initial_boundary,
                current_boundary,
                pc,
                semantic_acc: phase_envelope.after_semantic_digest,
                construction2_acc: accumulator,
                public_trace: current_boundary,
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
                initial_boundary,
                current_boundary,
                pc,
                semantic_acc: phase_envelope.before_semantic_digest,
                construction2_acc: accumulator,
                public_trace: current_boundary,
            },
            nebula_lane_digest,
        );

        let before_cursor_bits = decompose_var_to_u64_bits(&mut builder, before_program_cursor);
        let after_cursor_bits = decompose_var_to_u64_bits(&mut builder, after_program_cursor);
        let mut public_outputs = Vec::with_capacity(10 * PUBLIC_WORD_BITS);
        public_outputs.extend(after_x_out.public_bits);
        public_outputs.extend(before_x_out.public_bits);
        public_outputs.extend(before_cursor_bits);
        public_outputs.extend(after_cursor_bits);
        builder.begin_encoding_stage("complete");
        let public_layout = NebulaFPrimeStreamingPublicLayout::production();
        debug_assert_eq!(public_outputs.len() + 1, public_layout.logical_columns());
        debug_assert_eq!(builder.first_unsatisfied_row(), None);

        Self {
            initial_replay_state_columns: initial_replay_state.map(Var::col),
            before_local_state_digest_columns: phase_envelope.before_local_state_digest.map(Var::col),
            after_local_state_digest_columns: phase_envelope.after_local_state_digest.map(Var::col),
            before_program_cursor_column: before_program_cursor.col(),
            after_program_cursor_column: after_program_cursor.col(),
            builder,
            public_outputs,
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

    pub fn initial_replay_state_columns(&self) -> &[usize; REPLAY_STATE_FIELDS] {
        &self.initial_replay_state_columns
    }

    pub fn before_local_state_digest_columns(&self) -> &[usize; 4] {
        &self.before_local_state_digest_columns
    }

    pub fn after_local_state_digest_columns(&self) -> &[usize; 4] {
        &self.after_local_state_digest_columns
    }

    pub const fn before_program_cursor_column(&self) -> usize {
        self.before_program_cursor_column
    }

    pub const fn after_program_cursor_column(&self) -> usize {
        self.after_program_cursor_column
    }

    pub fn public_output_column(&self, index: usize) -> Option<usize> {
        self.public_outputs.get(index).map(|wire| wire.col())
    }

    #[doc(hidden)]
    pub fn builder_for_test(&self) -> &R1csBuilder {
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

pub fn production_streaming_prelude_source_arm() -> Result<SparseR1cs, FieldR1csLoweringError> {
    NebulaFPrimeStreamingPreludeSynthesis::production().into_sparse()
}

fn digest_replay_state(builder: &mut R1csBuilder, fields: &[Var; REPLAY_STATE_FIELDS]) -> [Var; 4] {
    let mut transcript = TranscriptGadget::new(builder, STATE_DIGEST_DOMAIN);
    transcript.append_fields(builder, STATE_DIGEST_FIELDS_LABEL, fields);
    transcript.digest_fields(builder)
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

const _: () = assert!(REPLAY_STATE_FIELDS == 8 + 1 + 1);
const _: () = assert!(BEFORE_PROGRAM_CURSOR == 0);
const _: () = assert!(AFTER_PROGRAM_CURSOR == 1);
