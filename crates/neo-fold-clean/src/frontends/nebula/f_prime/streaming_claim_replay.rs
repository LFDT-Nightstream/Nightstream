//! Bounded-width Poseidon2 replay for the production Nebula claim frame.
//!
//! Owns full-chunk and final-chunk arms. Both arms use canonical
//! public bits for Poseidon2 state digests and schedule cursors. The exact
//! carried state and chunk are private field advice. The combined audit arms
//! also derive and add the fixed-position PiCCS coordinate commitment. The
//! production base arms defer that work to the linked coordinate overlay. The
//! final arm is the only arm that checks the complete expected sponge state.
//!
//! Does not own the prelude, the next PiCCS phase, branch selection, recursive
//! proof integration, or the Poseidon2 collision reduction.

mod coordinate_overlay;

pub(crate) use coordinate_overlay::production_claim_coordinate_overlay_sparse_arms;

pub use coordinate_overlay::{
    build_production_claim_coordinate_overlay_low_norm_r1cs, build_production_claim_replay_base_low_norm_r1cs,
    production_claim_coordinate_overlay_kind_count, production_claim_coordinate_overlay_kind_map,
    production_claim_coordinate_overlay_link_runs, production_claim_coordinate_overlay_links,
    production_claim_coordinate_overlay_shape_audit, production_claim_replay_base_shape_audit,
    NebulaFPrimeClaimCoordinateOverlayLinkRun, NebulaFPrimeClaimCoordinateOverlayShapeAudit,
    NebulaFPrimeClaimCoordinateOverlaySynthesis, NebulaFPrimeClaimReplayBaseShapeAudit,
};

use neo_ajtai::Commitment;
use neo_math::{D, F};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use super::streaming_program::{CLAIM_CHUNK_FIELDS, CLAIM_FRAME_FIELDS, FIRST_CLAIM_PROGRAM_CURSOR};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, FieldR1csLoweringError, LowNormR1csError, LoweredFieldR1cs};
use crate::paper::reductions::accumulator_sis_circuit::{
    commit_coordinate_fields, enforce_commit_coordinate_fields, SisAccumulatorConfig,
    PI_CCS_RUNNING_METADATA_COORDINATE_SIS_CONFIG, PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG, PROTOCOL_BINDING_KAPPA,
    PROTOCOL_BINDING_MAX_FIELDS,
};

pub(super) const SPONGE_WIDTH: usize = 8;
const RATE: usize = 4;
const PUBLIC_WORD_BITS: usize = 64;
pub(super) const PI_CCS_STATEMENT_FIELDS: usize = 21_220;
pub(super) const PI_CCS_STATEMENT_FRESH_FIELDS: usize = 25_648;
pub(super) const PI_CCS_RUNNING_METADATA_FIELDS: usize = 61_992;
const PI_CCS_POINT_FIELDS: usize = 52;
const PI_CCS_POINT_FRAME_OFFSET: usize = 383;
const PI_CCS_RUNNING_COMMITMENT_FIELDS: usize = 54_432;
const PI_CCS_RUNNING_COMMITMENT_FRAME_OFFSET: usize = 435;
const PI_CCS_RUNNING_PUBLIC_FRAME_OFFSET: usize = 54_867;
const PI_CCS_EVALUATION_FRAME_OFFSET: usize = 62_427;
const PI_CCS_FRESH_COMMITMENT_FIELDS: usize = 3_888;
const PI_CCS_FRESH_COMMITMENT_FRAME_OFFSET: usize = 83_595;
const PI_CCS_FRESH_PUBLIC_FRAME_OFFSET: usize = 87_483;
pub(super) const COORDINATE_COMMITMENT_FIELDS: usize = D * PROTOCOL_BINDING_KAPPA;
pub(super) const PERSISTENT_WORDS: usize = 2 * (SPONGE_WIDTH + 1) + 2 + 2 * COORDINATE_COMMITMENT_FIELDS;
const TRANSITION_WORDS: usize = 2 * PERSISTENT_WORDS;
const STATE_DIGEST_WORDS: usize = 8;
const PUBLIC_CURSOR_WORDS: usize = 2;
const SHARED_PUBLIC_WORDS: usize = STATE_DIGEST_WORDS + PUBLIC_CURSOR_WORDS;
pub(super) const DIGEST_PIN_COUNT: usize = 13;
const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-claim-state/v1";
const STATE_DIGEST_FIELDS_LABEL: &[u8] = b"state";
const FINAL_CHUNK_FIELDS: usize = CLAIM_FRAME_FIELDS % CLAIM_CHUNK_FIELDS;
const FULL_CHUNKS: usize = CLAIM_FRAME_FIELDS / CLAIM_CHUNK_FIELDS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeClaimReplayArmKind {
    Full,
    Final,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordinateConstraints {
    Complete,
    DeferredOverlay,
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
pub(super) struct SpongeState {
    pub lanes: [F; SPONGE_WIDTH],
    pub absorbed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PersistentState {
    pub expected: SpongeState,
    pub runtime: SpongeState,
    pub frame_cursor: u64,
    pub program_cursor: u64,
    pub statement_fresh_commitment: [F; COORDINATE_COMMITMENT_FIELDS],
    pub running_metadata_commitment: [F; COORDINATE_COMMITMENT_FIELDS],
}

#[derive(Clone, Copy)]
pub(super) struct FieldWord {
    pub field: Var,
}

#[derive(Clone, Copy)]
pub(super) struct CanonicalWord {
    pub field: Var,
    pub bits: [Var; PUBLIC_WORD_BITS],
}

#[derive(Clone, Copy)]
pub(super) struct SpongeStateVars {
    pub lanes: [FieldWord; SPONGE_WIDTH],
    pub absorbed: FieldWord,
}

#[derive(Clone, Copy)]
pub(super) struct PersistentStateVars {
    pub expected: SpongeStateVars,
    pub runtime: SpongeStateVars,
    pub frame_cursor: FieldWord,
    pub program_cursor: CanonicalWord,
    pub statement_fresh_commitment: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
    pub running_metadata_commitment: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
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
    chunk_index: usize,
    builder: R1csBuilder,
    public_outputs: Vec<Var>,
    private_prefix_fields: usize,
    chunk_columns: Vec<usize>,
    state_word_columns: [usize; TRANSITION_WORDS],
    after_runtime_columns: [usize; SPONGE_WIDTH],
    statement_fresh_fields: Vec<(usize, usize)>,
    running_metadata_fields: Vec<(usize, usize)>,
    partial_statement_fresh_columns: Vec<usize>,
    partial_running_metadata_columns: Vec<usize>,
    before_statement_fresh_columns: [usize; COORDINATE_COMMITMENT_FIELDS],
    after_statement_fresh_columns: [usize; COORDINATE_COMMITMENT_FIELDS],
    before_running_metadata_columns: [usize; COORDINATE_COMMITMENT_FIELDS],
    after_running_metadata_columns: [usize; COORDINATE_COMMITMENT_FIELDS],
    after_digest_pin_columns: [usize; DIGEST_PIN_COUNT],
    before_digest_pin_columns: [usize; DIGEST_PIN_COUNT],
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

    #[doc(hidden)]
    pub fn production_base_full(chunk_index: usize) -> Result<Self, NebulaFPrimeClaimReplayError> {
        let geometry = ClaimReplayGeometry::production();
        if chunk_index >= geometry.full_chunks {
            return Err(NebulaFPrimeClaimReplayError::FullChunkIndex {
                chunk_index,
                full_chunks: geometry.full_chunks,
            });
        }
        Ok(synthesize_fixture_with_coordinate_constraints(
            NebulaFPrimeClaimReplayArmKind::Full,
            chunk_index,
            geometry,
            CoordinateConstraints::DeferredOverlay,
        ))
    }

    #[doc(hidden)]
    pub fn production_base_final() -> Self {
        let geometry = ClaimReplayGeometry::production();
        synthesize_fixture_with_coordinate_constraints(
            NebulaFPrimeClaimReplayArmKind::Final,
            geometry.full_chunks,
            geometry,
            CoordinateConstraints::DeferredOverlay,
        )
    }

    pub const fn kind(&self) -> NebulaFPrimeClaimReplayArmKind {
        self.kind
    }

    pub const fn chunk_index(&self) -> usize {
        self.chunk_index
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

    #[doc(hidden)]
    pub const fn state_word_column(&self, index: usize) -> Option<usize> {
        if index < TRANSITION_WORDS {
            Some(self.state_word_columns[index])
        } else {
            None
        }
    }

    pub const fn after_runtime_column(&self, lane: usize) -> Option<usize> {
        if lane < SPONGE_WIDTH {
            Some(self.after_runtime_columns[lane])
        } else {
            None
        }
    }

    /// `(global map field, local chunk offset)` pairs for the statement and
    /// fresh-metadata binding.
    pub fn statement_fresh_fields(&self) -> &[(usize, usize)] {
        &self.statement_fresh_fields
    }

    /// `(global map field, local chunk offset)` pairs for the running-metadata
    /// binding.
    pub fn running_metadata_fields(&self) -> &[(usize, usize)] {
        &self.running_metadata_fields
    }

    pub fn partial_statement_fresh_commitment_column(&self, index: usize) -> Option<usize> {
        self.partial_statement_fresh_columns.get(index).copied()
    }

    pub fn partial_running_metadata_commitment_column(&self, index: usize) -> Option<usize> {
        self.partial_running_metadata_columns.get(index).copied()
    }

    pub const fn before_statement_fresh_commitment_column(&self, index: usize) -> Option<usize> {
        if index < COORDINATE_COMMITMENT_FIELDS {
            Some(self.before_statement_fresh_columns[index])
        } else {
            None
        }
    }

    pub const fn after_statement_fresh_commitment_column(&self, index: usize) -> Option<usize> {
        if index < COORDINATE_COMMITMENT_FIELDS {
            Some(self.after_statement_fresh_columns[index])
        } else {
            None
        }
    }

    pub const fn before_running_metadata_commitment_column(&self, index: usize) -> Option<usize> {
        if index < COORDINATE_COMMITMENT_FIELDS {
            Some(self.before_running_metadata_columns[index])
        } else {
            None
        }
    }

    pub const fn after_running_metadata_commitment_column(&self, index: usize) -> Option<usize> {
        if index < COORDINATE_COMMITMENT_FIELDS {
            Some(self.after_running_metadata_columns[index])
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn normalized_chunk_column(&self, index: usize) -> Option<usize> {
        self.chunk_column(index)
            .and_then(|column| self.normalized_field_column(column))
    }

    #[doc(hidden)]
    pub fn normalized_before_statement_fresh_commitment_column(&self, index: usize) -> Option<usize> {
        self.before_statement_fresh_commitment_column(index)
            .and_then(|column| self.normalized_field_column(column))
    }

    #[doc(hidden)]
    pub fn normalized_after_statement_fresh_commitment_column(&self, index: usize) -> Option<usize> {
        self.after_statement_fresh_commitment_column(index)
            .and_then(|column| self.normalized_field_column(column))
    }

    #[doc(hidden)]
    pub fn normalized_before_running_metadata_commitment_column(&self, index: usize) -> Option<usize> {
        self.before_running_metadata_commitment_column(index)
            .and_then(|column| self.normalized_field_column(column))
    }

    #[doc(hidden)]
    pub fn normalized_after_running_metadata_commitment_column(&self, index: usize) -> Option<usize> {
        self.after_running_metadata_commitment_column(index)
            .and_then(|column| self.normalized_field_column(column))
    }

    #[doc(hidden)]
    pub const fn after_digest_pin_columns(&self) -> [usize; DIGEST_PIN_COUNT] {
        self.after_digest_pin_columns
    }

    #[doc(hidden)]
    pub const fn before_digest_pin_columns(&self) -> [usize; DIGEST_PIN_COUNT] {
        self.before_digest_pin_columns
    }

    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    #[doc(hidden)]
    pub fn public_output_column(&self, index: usize) -> Option<usize> {
        self.public_outputs.get(index).map(|var| var.col())
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
        // Only the state prefix is shared between base arms. The linked
        // overlay keeps separate balanced words and equates them to the exact
        // normalized replay fields with gated low-norm rows.
        let shared_prefix = self.private_prefix_fields;
        Ok((lower_field_r1cs(self.builder, &self.public_outputs)?, shared_prefix))
    }

    fn normalized_field_column(&self, source: usize) -> Option<usize> {
        if source >= self.builder.cols() {
            return None;
        }
        if source == 0 {
            return Some(0);
        }
        if let Some(index) = self
            .public_outputs
            .iter()
            .position(|wire| wire.col() == source)
        {
            return Some(index + 1);
        }
        let private_before = (1..source)
            .filter(|column| !self.public_outputs.iter().any(|wire| wire.col() == *column))
            .count();
        Some(1 + self.public_outputs.len() + private_before)
    }
}

/// Compute the exact field-native and radix-four selective relation shape.
/// This does not allocate the final CCS matrices.
pub fn production_claim_replay_shape_audit() -> Result<NebulaFPrimeClaimReplayShapeAudit, NebulaFPrimeClaimReplayError>
{
    claim_replay_shape_audit_for_chunk_fields(CLAIM_CHUNK_FIELDS)
}

/// Exact verifier-owned statement-and-fresh coordinate map for all 86
/// production claim chunks. Empty entries carry this map unchanged.
pub fn production_claim_statement_fresh_field_map() -> Vec<Vec<(usize, usize)>> {
    let geometry = ClaimReplayGeometry::production();
    (0..=geometry.full_chunks)
        .map(|chunk_index| claim_statement_fresh_positions(chunk_index, geometry))
        .collect()
}

/// Exact verifier-owned running-metadata coordinate map for all 86
/// production claim chunks. Empty entries carry this map unchanged.
pub fn production_claim_running_metadata_field_map() -> Vec<Vec<(usize, usize)>> {
    let geometry = ClaimReplayGeometry::production();
    (0..=geometry.full_chunks)
        .map(|chunk_index| claim_running_metadata_positions(chunk_index, geometry))
        .collect()
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
    synthesize_fixture_with_coordinate_constraints(kind, chunk_index, geometry, CoordinateConstraints::Complete)
}

fn synthesize_fixture_with_coordinate_constraints(
    kind: NebulaFPrimeClaimReplayArmKind,
    chunk_index: usize,
    geometry: ClaimReplayGeometry,
    coordinate_constraints: CoordinateConstraints,
) -> NebulaFPrimeClaimReplaySynthesis {
    let chunk = fixture_chunk(kind, chunk_index, geometry);
    let statement_fresh_positions = claim_statement_fresh_positions(chunk_index, geometry);
    let running_metadata_positions = claim_running_metadata_positions(chunk_index, geometry);
    let partial_statement_fresh = fixture_partial_coordinate_commitment(
        PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
        PI_CCS_STATEMENT_FRESH_FIELDS,
        &chunk,
        &statement_fresh_positions,
    );
    let partial_running_metadata = fixture_partial_coordinate_commitment(
        PI_CCS_RUNNING_METADATA_COORDINATE_SIS_CONFIG,
        PI_CCS_RUNNING_METADATA_FIELDS,
        &chunk,
        &running_metadata_positions,
    );
    let statement_fresh_commitment = fixture_coordinate_accumulator(chunk_index, 0x4000);
    let running_metadata_commitment = fixture_coordinate_accumulator(chunk_index, 0x8000);
    let next_statement_fresh_commitment = add_partial(statement_fresh_commitment, partial_statement_fresh.as_ref());
    let next_running_metadata_commitment = add_partial(running_metadata_commitment, partial_running_metadata.as_ref());
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
        program_cursor: (FIRST_CLAIM_PROGRAM_CURSOR + chunk_index) as u64,
        statement_fresh_commitment,
        running_metadata_commitment,
    };
    let after = PersistentState {
        expected,
        runtime: advanced,
        frame_cursor: before.frame_cursor + kind.active_fields(geometry) as u64,
        program_cursor: before.program_cursor + 1,
        statement_fresh_commitment: next_statement_fresh_commitment,
        running_metadata_commitment: next_running_metadata_commitment,
    };
    synthesize(
        kind,
        chunk_index,
        before,
        after,
        &chunk,
        geometry,
        coordinate_constraints,
    )
}

struct FixtureCoordinateTransition {
    before_statement_fresh: [F; COORDINATE_COMMITMENT_FIELDS],
    after_statement_fresh: [F; COORDINATE_COMMITMENT_FIELDS],
    before_running_metadata: [F; COORDINATE_COMMITMENT_FIELDS],
    after_running_metadata: [F; COORDINATE_COMMITMENT_FIELDS],
    chunk: Vec<F>,
    statement_fresh_positions: Vec<(usize, usize)>,
    running_metadata_positions: Vec<(usize, usize)>,
}

fn fixture_coordinate_transition(chunk_index: usize, geometry: ClaimReplayGeometry) -> FixtureCoordinateTransition {
    let kind = if chunk_index == geometry.full_chunks {
        NebulaFPrimeClaimReplayArmKind::Final
    } else {
        NebulaFPrimeClaimReplayArmKind::Full
    };
    let chunk = fixture_chunk(kind, chunk_index, geometry);
    let statement_fresh_positions = claim_statement_fresh_positions(chunk_index, geometry);
    let running_metadata_positions = claim_running_metadata_positions(chunk_index, geometry);
    let partial_statement_fresh = fixture_partial_coordinate_commitment(
        PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
        PI_CCS_STATEMENT_FRESH_FIELDS,
        &chunk,
        &statement_fresh_positions,
    );
    let partial_running_metadata = fixture_partial_coordinate_commitment(
        PI_CCS_RUNNING_METADATA_COORDINATE_SIS_CONFIG,
        PI_CCS_RUNNING_METADATA_FIELDS,
        &chunk,
        &running_metadata_positions,
    );
    let before_statement_fresh = fixture_coordinate_accumulator(chunk_index, 0x4000);
    let before_running_metadata = fixture_coordinate_accumulator(chunk_index, 0x8000);
    FixtureCoordinateTransition {
        before_statement_fresh,
        after_statement_fresh: add_partial(before_statement_fresh, partial_statement_fresh.as_ref()),
        before_running_metadata,
        after_running_metadata: add_partial(before_running_metadata, partial_running_metadata.as_ref()),
        chunk,
        statement_fresh_positions,
        running_metadata_positions,
    }
}

fn fixture_coordinate_accumulator(chunk_index: usize, base: u64) -> [F; COORDINATE_COMMITMENT_FIELDS] {
    if chunk_index == 0 {
        [F::ZERO; COORDINATE_COMMITMENT_FIELDS]
    } else {
        std::array::from_fn(|coordinate| F::from_u64(base + (chunk_index as u64) * 131 + coordinate as u64))
    }
}

fn add_partial(
    mut accumulator: [F; COORDINATE_COMMITMENT_FIELDS],
    partial: Option<&Commitment>,
) -> [F; COORDINATE_COMMITMENT_FIELDS] {
    if let Some(partial) = partial {
        for (next, value) in accumulator.iter_mut().zip(&partial.data) {
            *next += *value;
        }
    }
    accumulator
}

fn statement_fresh_frame_position(field: usize) -> usize {
    if field < PI_CCS_POINT_FIELDS {
        PI_CCS_POINT_FRAME_OFFSET + field
    } else if field < PI_CCS_STATEMENT_FIELDS {
        PI_CCS_EVALUATION_FRAME_OFFSET + (field - PI_CCS_POINT_FIELDS)
    } else if field < PI_CCS_STATEMENT_FIELDS + PI_CCS_FRESH_COMMITMENT_FIELDS {
        PI_CCS_FRESH_COMMITMENT_FRAME_OFFSET + (field - PI_CCS_STATEMENT_FIELDS)
    } else {
        PI_CCS_FRESH_PUBLIC_FRAME_OFFSET + (field - PI_CCS_STATEMENT_FIELDS - PI_CCS_FRESH_COMMITMENT_FIELDS)
    }
}

fn running_metadata_frame_position(field: usize) -> usize {
    if field < PI_CCS_RUNNING_COMMITMENT_FIELDS {
        PI_CCS_RUNNING_COMMITMENT_FRAME_OFFSET + field
    } else {
        PI_CCS_RUNNING_PUBLIC_FRAME_OFFSET + (field - PI_CCS_RUNNING_COMMITMENT_FIELDS)
    }
}

fn claim_statement_fresh_positions(chunk_index: usize, geometry: ClaimReplayGeometry) -> Vec<(usize, usize)> {
    claim_binding_positions(
        chunk_index,
        geometry,
        PI_CCS_STATEMENT_FRESH_FIELDS,
        statement_fresh_frame_position,
    )
}

fn claim_running_metadata_positions(chunk_index: usize, geometry: ClaimReplayGeometry) -> Vec<(usize, usize)> {
    claim_binding_positions(
        chunk_index,
        geometry,
        PI_CCS_RUNNING_METADATA_FIELDS,
        running_metadata_frame_position,
    )
}

fn claim_binding_positions(
    chunk_index: usize,
    geometry: ClaimReplayGeometry,
    total_fields: usize,
    frame_position: fn(usize) -> usize,
) -> Vec<(usize, usize)> {
    (0..total_fields)
        .filter_map(|field| {
            let position = frame_position(field);
            (position / geometry.chunk_fields == chunk_index).then_some((field, position % geometry.chunk_fields))
        })
        .collect()
}

fn fixture_partial_coordinate_commitment(
    config: SisAccumulatorConfig,
    total_fields: usize,
    chunk: &[F],
    positions: &[(usize, usize)],
) -> Option<Commitment> {
    if positions.is_empty() {
        return None;
    }
    let fields = positions
        .iter()
        .map(|&(field, offset)| (field, chunk[offset]))
        .collect::<Vec<_>>();
    Some(commit_coordinate_fields(config, total_fields, &fields).expect("fixed production PiCCS coordinate map"))
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
    chunk_index: usize,
    before: PersistentState,
    after: PersistentState,
    chunk: &[F],
    geometry: ClaimReplayGeometry,
    coordinate_constraints: CoordinateConstraints,
) -> NebulaFPrimeClaimReplaySynthesis {
    assert_eq!(chunk.len(), geometry.chunk_fields, "fixed claim chunk width");
    assert_eq!(before.runtime.absorbed, 0, "claim chunks start at a rate boundary");
    assert_eq!(after.runtime.absorbed, (kind.active_fields(geometry) % RATE) as u64);

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("nebula.streaming.claim_replay.state_words");
    let transition = alloc_transition(&mut builder, before, after);
    let state_word_columns = transition_state_fields(transition).map(Var::col);
    let cursor_public_fields = PUBLIC_CURSOR_WORDS * PUBLIC_WORD_BITS;
    let private_prefix_fields = builder.cols() - 1 - cursor_public_fields;

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
            (FIRST_CLAIM_PROGRAM_CURSOR + geometry.full_chunks) as u64,
        );
        for &tail in &chunk_vars[geometry.final_chunk_fields..] {
            enforce_constant(&mut builder, tail, 0);
        }
    }

    if coordinate_constraints == CoordinateConstraints::Complete && chunk_index == 0 {
        for coordinate in transition.before.statement_fresh_commitment {
            enforce_constant(&mut builder, coordinate.field, 0);
        }
        for coordinate in transition.before.running_metadata_commitment {
            enforce_constant(&mut builder, coordinate.field, 0);
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

    let statement_fresh_positions = claim_statement_fresh_positions(chunk_index, geometry);
    let running_metadata_positions = claim_running_metadata_positions(chunk_index, geometry);
    let (partial_statement_fresh_columns, partial_running_metadata_columns) = match coordinate_constraints {
        CoordinateConstraints::DeferredOverlay => (Vec::new(), Vec::new()),
        CoordinateConstraints::Complete => {
            builder.begin_encoding_stage("nebula.streaming.claim_replay.coordinate_binding");
            let statement_fresh = enforce_coordinate_map_transition(
                &mut builder,
                PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
                PI_CCS_STATEMENT_FRESH_FIELDS,
                &statement_fresh_positions,
                &chunk_vars,
                transition.before.statement_fresh_commitment,
                transition.after.statement_fresh_commitment,
            );
            let running_metadata = enforce_coordinate_map_transition(
                &mut builder,
                PI_CCS_RUNNING_METADATA_COORDINATE_SIS_CONFIG,
                PI_CCS_RUNNING_METADATA_FIELDS,
                &running_metadata_positions,
                &chunk_vars,
                transition.before.running_metadata_commitment,
                transition.after.running_metadata_commitment,
            );
            (statement_fresh, running_metadata)
        }
    };

    builder.begin_encoding_stage("nebula.streaming.claim_replay.state_digest");
    let (after_digest, after_digest_pin_columns) = digest_persistent_state(&mut builder, transition.after);
    let (before_digest, before_digest_pin_columns) = digest_persistent_state(&mut builder, transition.before);
    let mut public_outputs = Vec::with_capacity(SHARED_PUBLIC_WORDS * PUBLIC_WORD_BITS);
    append_digest_bits(&mut builder, after_digest, &mut public_outputs);
    append_digest_bits(&mut builder, before_digest, &mut public_outputs);
    public_outputs.extend(transition.before.program_cursor.bits);
    public_outputs.extend(transition.after.program_cursor.bits);
    let public_layout = NebulaFPrimeStreamingPublicLayout::production();
    debug_assert_eq!(public_outputs.len() + 1, public_layout.logical_columns());

    let after_runtime_columns = transition.after.runtime.lanes.map(|word| word.field.col());
    let before_statement_fresh_columns = transition
        .before
        .statement_fresh_commitment
        .map(|word| word.field.col());
    let after_statement_fresh_columns = transition
        .after
        .statement_fresh_commitment
        .map(|word| word.field.col());
    let before_running_metadata_columns = transition
        .before
        .running_metadata_commitment
        .map(|word| word.field.col());
    let after_running_metadata_columns = transition
        .after
        .running_metadata_commitment
        .map(|word| word.field.col());
    NebulaFPrimeClaimReplaySynthesis {
        kind,
        chunk_index,
        builder,
        public_outputs,
        private_prefix_fields,
        chunk_columns,
        state_word_columns,
        after_runtime_columns,
        statement_fresh_fields: statement_fresh_positions,
        running_metadata_fields: running_metadata_positions,
        partial_statement_fresh_columns,
        partial_running_metadata_columns,
        before_statement_fresh_columns,
        after_statement_fresh_columns,
        before_running_metadata_columns,
        after_running_metadata_columns,
        after_digest_pin_columns,
        before_digest_pin_columns,
    }
}

fn alloc_transition(builder: &mut R1csBuilder, before: PersistentState, after: PersistentState) -> TransitionVars {
    TransitionVars {
        before: alloc_persistent(builder, before),
        after: alloc_persistent(builder, after),
    }
}

pub(super) fn alloc_persistent(builder: &mut R1csBuilder, value: PersistentState) -> PersistentStateVars {
    let expected = alloc_sponge(builder, value.expected);
    let runtime = alloc_sponge(builder, value.runtime);
    let frame_cursor = alloc_field_word(builder, value.frame_cursor);
    let statement_fresh_commitment = value
        .statement_fresh_commitment
        .map(|coordinate| alloc_field_word(builder, coordinate.as_canonical_u64()));
    let running_metadata_commitment = value
        .running_metadata_commitment
        .map(|coordinate| alloc_field_word(builder, coordinate.as_canonical_u64()));
    let program_cursor = alloc_canonical_word(builder, value.program_cursor);
    PersistentStateVars {
        expected,
        runtime,
        frame_cursor,
        program_cursor,
        statement_fresh_commitment,
        running_metadata_commitment,
    }
}

fn alloc_sponge(builder: &mut R1csBuilder, value: SpongeState) -> SpongeStateVars {
    SpongeStateVars {
        lanes: value
            .lanes
            .map(|lane| alloc_field_word(builder, lane.as_canonical_u64())),
        absorbed: alloc_field_word(builder, value.absorbed),
    }
}

fn alloc_field_word(builder: &mut R1csBuilder, value: u64) -> FieldWord {
    FieldWord {
        field: builder.alloc(F::from_u64(value)),
    }
}

fn alloc_canonical_word(builder: &mut R1csBuilder, value: u64) -> CanonicalWord {
    let field = builder.alloc(F::from_u64(value));
    let bits = decompose_var_to_u64_bits(builder, field);
    CanonicalWord { field, bits }
}

pub(super) fn persistent_state_fields(state: PersistentStateVars) -> Vec<Var> {
    let mut fields = Vec::with_capacity(PERSISTENT_WORDS);
    fields.extend(state.expected.lanes.map(|word| word.field));
    fields.push(state.expected.absorbed.field);
    fields.extend(state.runtime.lanes.map(|word| word.field));
    fields.push(state.runtime.absorbed.field);
    fields.push(state.frame_cursor.field);
    fields.push(state.program_cursor.field);
    fields.extend(state.statement_fresh_commitment.map(|word| word.field));
    fields.extend(state.running_metadata_commitment.map(|word| word.field));
    debug_assert_eq!(fields.len(), PERSISTENT_WORDS);
    fields
}

fn transition_state_fields(transition: TransitionVars) -> [Var; TRANSITION_WORDS] {
    let mut fields = persistent_state_fields(transition.before);
    fields.extend(persistent_state_fields(transition.after));
    fields
        .try_into()
        .expect("two fixed-width persistent claim states")
}

pub(super) fn digest_persistent_state(
    builder: &mut R1csBuilder,
    state: PersistentStateVars,
) -> ([Var; 4], [usize; DIGEST_PIN_COUNT]) {
    let fields = persistent_state_fields(state);
    let mut transcript = TranscriptGadget::new(builder, STATE_DIGEST_DOMAIN);
    transcript.append_fields(builder, STATE_DIGEST_FIELDS_LABEL, &fields);
    let digest = transcript.digest_fields(builder);
    let pin_columns = transcript.constant_bindings()[..DIGEST_PIN_COUNT]
        .iter()
        .map(|(wire, _)| wire.col())
        .collect::<Vec<_>>()
        .try_into()
        .expect("fixed claim-state digest pin count");
    (digest, pin_columns)
}

fn append_digest_bits(builder: &mut R1csBuilder, digest: [Var; 4], public_outputs: &mut Vec<Var>) {
    for lane in digest {
        public_outputs.extend(decompose_var_to_u64_bits(builder, lane));
    }
}

fn enforce_persistent_carry(builder: &mut R1csBuilder, before: PersistentStateVars, after: PersistentStateVars) {
    enforce_sponge_equal(builder, before.expected, after.expected);
}

fn enforce_coordinate_map_transition(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    total_fields: usize,
    positions: &[(usize, usize)],
    chunk: &[Var],
    before: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
    after: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
) -> Vec<usize> {
    if positions.is_empty() {
        enforce_coordinate_commitment_carry(builder, before, after);
        return Vec::new();
    }
    let positioned_fields = positions
        .iter()
        .map(|&(field, offset)| (field, chunk[offset]))
        .collect::<Vec<_>>();
    let partial = enforce_commit_coordinate_fields(builder, config, total_fields, &positioned_fields)
        .expect("fixed production PiCCS coordinate map");
    enforce_coordinate_commitment_update(builder, before, after, &partial.data);
    partial.data.iter().map(|wire| wire.col()).collect()
}

fn enforce_coordinate_commitment_carry(
    builder: &mut R1csBuilder,
    before: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
    after: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
) {
    for (before, after) in before.iter().zip(after.iter()) {
        builder.enforce_eq(&Lc::from_var(after.field), &Lc::from_var(before.field));
    }
}

fn enforce_coordinate_commitment_update(
    builder: &mut R1csBuilder,
    before: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
    after: [FieldWord; COORDINATE_COMMITMENT_FIELDS],
    partial: &[Var],
) {
    assert_eq!(partial.len(), COORDINATE_COMMITMENT_FIELDS);
    for ((before, after), &partial) in before.iter().zip(after.iter()).zip(partial) {
        let expected = Lc::from_var(before.field).add_scaled(&Lc::from_var(partial), F::ONE);
        builder.enforce_eq(&Lc::from_var(after.field), &expected);
    }
}

pub(super) fn enforce_sponge_equal(builder: &mut R1csBuilder, left: SpongeStateVars, right: SpongeStateVars) {
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
    let expected = Lc::from_var(before.program_cursor.field)
        .add_scaled(&Lc::from_const(F::from_usize(FIRST_CLAIM_PROGRAM_CURSOR)), -F::ONE);
    let expected = Lc::zero().add_scaled(&expected, F::from_u64(chunk_fields as u64));
    builder.enforce_eq(&Lc::from_var(before.frame_cursor.field), &expected);
}

const _: () = assert!(FINAL_CHUNK_FIELDS == 983);
const _: () = assert!(FULL_CHUNKS == 85);
const _: () = assert!(COORDINATE_COMMITMENT_FIELDS == 108);
const _: () = assert!(PI_CCS_STATEMENT_FRESH_FIELDS == 25_648);
const _: () = assert!(PI_CCS_RUNNING_METADATA_FIELDS == 61_992);
const _: () = assert!(PI_CCS_STATEMENT_FRESH_FIELDS <= PROTOCOL_BINDING_MAX_FIELDS);
const _: () = assert!(PI_CCS_RUNNING_METADATA_FIELDS <= PROTOCOL_BINDING_MAX_FIELDS);
const _: () = assert!(PI_CCS_FRESH_PUBLIC_FRAME_OFFSET + 540 == CLAIM_FRAME_FIELDS);
const _: () = assert!(TRANSITION_WORDS == 472);
const _: () = assert!(SHARED_PUBLIC_WORDS == 10);
