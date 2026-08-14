//! Bounded-width Poseidon2 replay for the production Nebula claim frame.
//!
//! Owns one full-chunk arm and one final-chunk arm. Both arms use canonical
//! public bits for the carried state and private field advice for the chunk.
//! The final arm is the only arm that checks the complete expected state.
//!
//! Does not own the prelude, the next PiCCS phase, branch selection, recursive
//! proof integration, or the Poseidon2 collision reduction.

use neo_math::{D, F};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use super::streaming_program::{CLAIM_CHUNK_FIELDS, CLAIM_FRAME_FIELDS};
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError, LowNormR1csError, LoweredFieldR1cs};

const SPONGE_WIDTH: usize = 8;
const RATE: usize = 4;
const PUBLIC_WORD_BITS: usize = 64;
const PERSISTENT_WORDS: usize = 2 * (SPONGE_WIDTH + 1) + 2;
const TRANSITION_PUBLIC_WORDS: usize = 2 * PERSISTENT_WORDS;
const FINAL_CHUNK_FIELDS: usize = CLAIM_FRAME_FIELDS % CLAIM_CHUNK_FIELDS;
const FULL_CHUNKS: usize = CLAIM_FRAME_FIELDS / CLAIM_CHUNK_FIELDS;
const FIRST_CLAIM_PROGRAM_CURSOR: u64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeClaimReplayArmKind {
    Full,
    Final,
}

impl NebulaFPrimeClaimReplayArmKind {
    const fn active_fields(self, geometry: ClaimReplayGeometry) -> usize {
        match self {
            Self::Full => geometry.chunk_fields,
            Self::Final => geometry.final_chunk_fields,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClaimReplayGeometry {
    chunk_fields: usize,
    final_chunk_fields: usize,
    full_chunks: usize,
}

impl ClaimReplayGeometry {
    fn checked(chunk_fields: usize) -> Result<Self, NebulaFPrimeClaimReplayError> {
        if chunk_fields == 0
            || chunk_fields >= CLAIM_FRAME_FIELDS
            || chunk_fields % RATE != 0
            || CLAIM_FRAME_FIELDS % chunk_fields == 0
        {
            return Err(NebulaFPrimeClaimReplayError::InvalidChunkFields(chunk_fields));
        }
        Ok(Self {
            chunk_fields,
            final_chunk_fields: CLAIM_FRAME_FIELDS % chunk_fields,
            full_chunks: CLAIM_FRAME_FIELDS / chunk_fields,
        })
    }

    fn production() -> Self {
        Self::checked(CLAIM_CHUNK_FIELDS).expect("production claim-replay geometry")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpongeState {
    lanes: [F; SPONGE_WIDTH],
    absorbed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PersistentState {
    expected: SpongeState,
    runtime: SpongeState,
    frame_cursor: u64,
    program_cursor: u64,
}

#[derive(Clone, Copy)]
struct PublicWord {
    field: Var,
}

#[derive(Clone, Copy)]
struct SpongeStateVars {
    lanes: [PublicWord; SPONGE_WIDTH],
    absorbed: PublicWord,
}

#[derive(Clone, Copy)]
struct PersistentStateVars {
    expected: SpongeStateVars,
    runtime: SpongeStateVars,
    frame_cursor: PublicWord,
    program_cursor: PublicWord,
}

#[derive(Clone, Copy)]
struct TransitionVars {
    before: PersistentStateVars,
    after: PersistentStateVars,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimReplayFieldArmAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub poseidon2_permutations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimReplayShapeAudit {
    pub chunk_fields: usize,
    pub final_chunk_fields: usize,
    pub full_chunks: usize,
    pub full: NebulaFPrimeClaimReplayFieldArmAudit,
    pub final_chunk: NebulaFPrimeClaimReplayFieldArmAudit,
    pub shared_private_fields: usize,
    pub low_norm_rows: usize,
    pub low_norm_columns: usize,
    pub low_norm_public_columns: usize,
    pub low_norm_total_coordinates: usize,
    pub low_norm_arity: usize,
    pub low_norm_degree: u32,
    pub low_norm_shared_private_coordinates: usize,
    pub low_norm_full_branch_coordinates: usize,
    pub low_norm_final_branch_coordinates: usize,
    pub low_norm_full_poseidon2_coordinates: usize,
    pub low_norm_final_poseidon2_coordinates: usize,
}

#[derive(Debug, Error)]
pub enum NebulaFPrimeClaimReplayError {
    #[error("claim-replay chunk width {0} must be rate-aligned, smaller than the frame, and leave a final chunk")]
    InvalidChunkFields(usize),
    #[error("claim-replay full chunk index {chunk_index} is outside 0..{full_chunks}")]
    FullChunkIndex {
        chunk_index: usize,
        full_chunks: usize,
    },
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
}

/// One synthesized field-native claim-replay arm and its exact public layout.
/// This is an audit surface until the streaming relation enters the lifecycle.
pub struct NebulaFPrimeClaimReplaySynthesis {
    kind: NebulaFPrimeClaimReplayArmKind,
    geometry: ClaimReplayGeometry,
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    private_prefix_fields: usize,
    chunk_columns: Vec<usize>,
    after_runtime_columns: [usize; SPONGE_WIDTH],
}

impl NebulaFPrimeClaimReplaySynthesis {
    pub fn production_full(chunk_index: usize) -> Result<Self, NebulaFPrimeClaimReplayError> {
        let geometry = ClaimReplayGeometry::production();
        if chunk_index >= geometry.full_chunks {
            return Err(NebulaFPrimeClaimReplayError::FullChunkIndex {
                chunk_index,
                full_chunks: geometry.full_chunks,
            });
        }
        Ok(synthesize_fixture(
            NebulaFPrimeClaimReplayArmKind::Full,
            chunk_index,
            geometry,
        ))
    }

    pub fn production_final() -> Self {
        let geometry = ClaimReplayGeometry::production();
        synthesize_fixture(NebulaFPrimeClaimReplayArmKind::Final, geometry.full_chunks, geometry)
    }

    pub const fn kind(&self) -> NebulaFPrimeClaimReplayArmKind {
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

    pub fn poseidon2_permutations(&self) -> usize {
        self.builder.encoding_trace().poseidon_permutations().len()
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

    pub fn chunk_column(&self, index: usize) -> Option<usize> {
        self.chunk_columns.get(index).copied()
    }

    pub const fn after_runtime_column(&self, lane: usize) -> Option<usize> {
        if lane < SPONGE_WIDTH {
            Some(self.after_runtime_columns[lane])
        } else {
            None
        }
    }

    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    #[doc(hidden)]
    pub fn builder_for_artifact(&self) -> &R1csBuilder {
        &self.builder
    }

    #[doc(hidden)]
    pub fn tamper_witness_for_test(&mut self, column: usize, value: F) {
        self.builder.tamper_witness(column, value);
    }

    pub fn field_arm_audit(&self) -> NebulaFPrimeClaimReplayFieldArmAudit {
        NebulaFPrimeClaimReplayFieldArmAudit {
            rows: self.rows(),
            columns: self.columns(),
            public_columns: self.public_columns(),
            poseidon2_permutations: self.poseidon2_permutations(),
        }
    }

    fn into_lowered(self) -> Result<(LoweredFieldR1cs, usize), NebulaFPrimeClaimReplayError> {
        let shared_prefix = self.private_prefix_fields + self.geometry.final_chunk_fields;
        Ok((lower_field_r1cs(self.builder, &self.public_outputs)?, shared_prefix))
    }
}

/// Compute the exact field-native and radix-four selective relation shape.
/// This does not allocate the final CCS matrices.
pub fn production_claim_replay_shape_audit() -> Result<NebulaFPrimeClaimReplayShapeAudit, NebulaFPrimeClaimReplayError>
{
    claim_replay_shape_audit_for_chunk_fields(CLAIM_CHUNK_FIELDS)
}

/// Compute the exact relation shape for one rate-aligned streaming candidate.
/// This is a planning surface. It does not select the production profile.
#[doc(hidden)]
pub fn claim_replay_shape_audit_for_chunk_fields(
    chunk_fields: usize,
) -> Result<NebulaFPrimeClaimReplayShapeAudit, NebulaFPrimeClaimReplayError> {
    let geometry = ClaimReplayGeometry::checked(chunk_fields)?;
    let full = synthesize_fixture(NebulaFPrimeClaimReplayArmKind::Full, 0, geometry);
    let final_chunk = synthesize_fixture(NebulaFPrimeClaimReplayArmKind::Final, geometry.full_chunks, geometry);
    let full_audit = full.field_arm_audit();
    let final_audit = final_chunk.field_arm_audit();
    let (full, full_shared) = full.into_lowered()?;
    let (final_chunk, final_shared) = final_chunk.into_lowered()?;
    debug_assert_eq!(full_shared, final_shared);
    let (full, _) = full.into_parts();
    let (final_chunk, _) = final_chunk.into_parts();
    let arms = vec![full, final_chunk];
    let width =
        crate::frontends::r1cs_f_prime::audit_multi_branch_selective_low_norm_width_for_norm_base_with_alignment(
            &arms,
            full_shared,
            D,
            0,
            4,
        )?;
    let prepared =
        crate::frontends::r1cs_f_prime::prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            full_shared,
            0,
            D,
            0,
            4,
        )?;
    let shape = prepared.shape_summary();
    Ok(NebulaFPrimeClaimReplayShapeAudit {
        chunk_fields: geometry.chunk_fields,
        final_chunk_fields: geometry.final_chunk_fields,
        full_chunks: geometry.full_chunks,
        full: full_audit,
        final_chunk: final_audit,
        shared_private_fields: full_shared,
        low_norm_rows: shape.rows,
        low_norm_columns: shape.columns,
        low_norm_public_columns: shape.public_input_len,
        low_norm_total_coordinates: shape.total_coordinates,
        low_norm_arity: shape.polynomial.arity(),
        low_norm_degree: shape.polynomial.max_degree(),
        low_norm_shared_private_coordinates: width.shared_private_coordinates,
        low_norm_full_branch_coordinates: width.arms[0].total_branch_coordinates,
        low_norm_final_branch_coordinates: width.arms[1].total_branch_coordinates,
        low_norm_full_poseidon2_coordinates: width.arms[0].traces.poseidon2_coordinates,
        low_norm_final_poseidon2_coordinates: width.arms[1].traces.poseidon2_coordinates,
    })
}

fn synthesize_fixture(
    kind: NebulaFPrimeClaimReplayArmKind,
    chunk_index: usize,
    geometry: ClaimReplayGeometry,
) -> NebulaFPrimeClaimReplaySynthesis {
    let chunk = fixture_chunk(kind, chunk_index, geometry);
    let runtime = SpongeState {
        lanes: std::array::from_fn(|lane| F::from_u64(0x1000 + (chunk_index as u64) * 17 + lane as u64)),
        absorbed: 0,
    };
    let advanced = absorb(runtime, &chunk[..kind.active_fields(geometry)]);
    let expected = match kind {
        NebulaFPrimeClaimReplayArmKind::Full => SpongeState {
            lanes: std::array::from_fn(|lane| F::from_u64(0x9000 + lane as u64)),
            absorbed: (geometry.final_chunk_fields % RATE) as u64,
        },
        NebulaFPrimeClaimReplayArmKind::Final => advanced,
    };
    let before = PersistentState {
        expected,
        runtime,
        frame_cursor: (chunk_index * geometry.chunk_fields) as u64,
        program_cursor: FIRST_CLAIM_PROGRAM_CURSOR + chunk_index as u64,
    };
    let after = PersistentState {
        expected,
        runtime: advanced,
        frame_cursor: before.frame_cursor + kind.active_fields(geometry) as u64,
        program_cursor: before.program_cursor + 1,
    };
    synthesize(kind, before, after, &chunk, geometry)
}

fn fixture_chunk(kind: NebulaFPrimeClaimReplayArmKind, chunk_index: usize, geometry: ClaimReplayGeometry) -> Vec<F> {
    let active = kind.active_fields(geometry);
    (0..geometry.chunk_fields)
        .map(|offset| {
            if offset < active {
                let source = (chunk_index * geometry.chunk_fields + offset) as u64;
                F::from_u64(source.wrapping_mul(0x9e37_79b9).wrapping_add(0x7f4a_7c15))
            } else {
                F::ZERO
            }
        })
        .collect()
}

fn absorb(before: SpongeState, fields: &[F]) -> SpongeState {
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(before.lanes, before.absorbed as usize);
    transcript.append_fields_unframed(fields);
    SpongeState {
        lanes: transcript.state(),
        absorbed: transcript.absorbed() as u64,
    }
}

fn synthesize(
    kind: NebulaFPrimeClaimReplayArmKind,
    before: PersistentState,
    after: PersistentState,
    chunk: &[F],
    geometry: ClaimReplayGeometry,
) -> NebulaFPrimeClaimReplaySynthesis {
    assert_eq!(chunk.len(), geometry.chunk_fields, "fixed claim chunk width");
    assert_eq!(before.runtime.absorbed, 0, "claim chunks start at a rate boundary");
    assert_eq!(after.runtime.absorbed, (kind.active_fields(geometry) % RATE) as u64);

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("nebula.streaming.claim_replay.public");
    let mut public_outputs = Vec::with_capacity(TRANSITION_PUBLIC_WORDS * PUBLIC_WORD_BITS);
    let transition = alloc_transition(&mut builder, before, after, &mut public_outputs);
    let private_prefix_fields = builder.cols() - 1 - public_outputs.len();

    builder.begin_encoding_stage("nebula.streaming.claim_replay.chunk");
    let chunk_vars = builder.alloc_vec(chunk);
    let chunk_columns = chunk_vars.iter().map(|var| var.col()).collect::<Vec<_>>();

    builder.begin_encoding_stage("nebula.streaming.claim_replay.state");
    enforce_persistent_carry(&mut builder, transition.before, transition.after);
    enforce_constant(
        &mut builder,
        transition.before.expected.absorbed.field,
        (geometry.final_chunk_fields % RATE) as u64,
    );
    enforce_constant(
        &mut builder,
        transition.after.expected.absorbed.field,
        (geometry.final_chunk_fields % RATE) as u64,
    );
    enforce_constant(&mut builder, transition.before.runtime.absorbed.field, 0);
    enforce_constant(
        &mut builder,
        transition.after.runtime.absorbed.field,
        (kind.active_fields(geometry) % RATE) as u64,
    );
    enforce_cursor_alignment(&mut builder, transition.before, geometry.chunk_fields);
    enforce_add_constant(
        &mut builder,
        transition.before.frame_cursor.field,
        transition.after.frame_cursor.field,
        kind.active_fields(geometry) as u64,
    );
    enforce_add_constant(
        &mut builder,
        transition.before.program_cursor.field,
        transition.after.program_cursor.field,
        1,
    );

    if kind == NebulaFPrimeClaimReplayArmKind::Final {
        enforce_constant(
            &mut builder,
            transition.before.frame_cursor.field,
            (geometry.full_chunks * geometry.chunk_fields) as u64,
        );
        enforce_constant(
            &mut builder,
            transition.before.program_cursor.field,
            FIRST_CLAIM_PROGRAM_CURSOR + geometry.full_chunks as u64,
        );
        for &tail in &chunk_vars[geometry.final_chunk_fields..] {
            enforce_constant(&mut builder, tail, 0);
        }
    }

    builder.begin_encoding_stage("nebula.streaming.claim_replay.poseidon2");
    let runtime_input = transition.before.runtime.lanes.map(|word| word.field);
    let mut transcript = TranscriptGadget::from_variable_state(runtime_input, 0);
    transcript.append_fields_unframed_vars(&mut builder, &chunk_vars[..kind.active_fields(geometry)]);
    let computed = transcript.variable_state();
    debug_assert_eq!(transcript.absorbed(), kind.active_fields(geometry) % RATE);
    for (declared, computed) in transition.after.runtime.lanes.iter().zip(computed) {
        builder.enforce_eq(&Lc::from_var(declared.field), &Lc::from_var(computed));
    }

    if kind == NebulaFPrimeClaimReplayArmKind::Final {
        builder.begin_encoding_stage("nebula.streaming.claim_replay.ready");
        enforce_sponge_equal(&mut builder, transition.after.runtime, transition.after.expected);
        enforce_constant(
            &mut builder,
            transition.after.frame_cursor.field,
            CLAIM_FRAME_FIELDS as u64,
        );
    }

    let after_runtime_columns = transition.after.runtime.lanes.map(|word| word.field.col());
    NebulaFPrimeClaimReplaySynthesis {
        kind,
        geometry,
        builder,
        public_outputs,
        private_prefix_fields,
        chunk_columns,
        after_runtime_columns,
    }
}

fn alloc_transition(
    builder: &mut R1csBuilder,
    before: PersistentState,
    after: PersistentState,
    public_outputs: &mut Vec<Var>,
) -> TransitionVars {
    TransitionVars {
        before: alloc_persistent(builder, before, public_outputs),
        after: alloc_persistent(builder, after, public_outputs),
    }
}

fn alloc_persistent(
    builder: &mut R1csBuilder,
    value: PersistentState,
    public_outputs: &mut Vec<Var>,
) -> PersistentStateVars {
    PersistentStateVars {
        expected: alloc_sponge(builder, value.expected, public_outputs),
        runtime: alloc_sponge(builder, value.runtime, public_outputs),
        frame_cursor: alloc_public_word(builder, value.frame_cursor, public_outputs),
        program_cursor: alloc_public_word(builder, value.program_cursor, public_outputs),
    }
}

fn alloc_sponge(builder: &mut R1csBuilder, value: SpongeState, public_outputs: &mut Vec<Var>) -> SpongeStateVars {
    SpongeStateVars {
        lanes: value
            .lanes
            .map(|lane| alloc_public_word(builder, lane.as_canonical_u64(), public_outputs)),
        absorbed: alloc_public_word(builder, value.absorbed, public_outputs),
    }
}

fn alloc_public_word(builder: &mut R1csBuilder, value: u64, public_outputs: &mut Vec<Var>) -> PublicWord {
    let field = builder.alloc(F::from_u64(value));
    let bits = decompose_var_to_u64_bits(builder, field);
    public_outputs.extend(bits);
    PublicWord { field }
}

fn enforce_persistent_carry(builder: &mut R1csBuilder, before: PersistentStateVars, after: PersistentStateVars) {
    enforce_sponge_equal(builder, before.expected, after.expected);
}

fn enforce_sponge_equal(builder: &mut R1csBuilder, left: SpongeStateVars, right: SpongeStateVars) {
    for (left, right) in left.lanes.iter().zip(right.lanes) {
        builder.enforce_eq(&Lc::from_var(left.field), &Lc::from_var(right.field));
    }
    builder.enforce_eq(&Lc::from_var(left.absorbed.field), &Lc::from_var(right.absorbed.field));
}

fn enforce_constant(builder: &mut R1csBuilder, var: Var, value: u64) {
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(F::from_u64(value)));
}

fn enforce_add_constant(builder: &mut R1csBuilder, before: Var, after: Var, increment: u64) {
    let expected = Lc::from_var(before).add_scaled(&Lc::from_const(F::from_u64(increment)), F::ONE);
    builder.enforce_eq(&Lc::from_var(after), &expected);
}

fn enforce_cursor_alignment(builder: &mut R1csBuilder, before: PersistentStateVars, chunk_fields: usize) {
    let expected = Lc::from_var(before.program_cursor.field).add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    let expected = Lc::zero().add_scaled(&expected, F::from_u64(chunk_fields as u64));
    builder.enforce_eq(&Lc::from_var(before.frame_cursor.field), &expected);
}

const _: () = assert!(FINAL_CHUNK_FIELDS == 983);
const _: () = assert!(FULL_CHUNKS == 85);
const _: () = assert!(TRANSITION_PUBLIC_WORDS == 40);
