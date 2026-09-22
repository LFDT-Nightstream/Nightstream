//! Fixed-position PiCCS coordinate overlays for claim replay.
//!
//! Owns one no-op kind and one exact binding kind for each claim
//! chunks. Overlay inputs are private copies. The joint
//! scheduled composer links those copies to the replay body's exact chunk and
//! both before/after commitment fields.
//!
//! Does not own Poseidon2 replay, state digests, schedule selection, or the
//! final equality between the accumulated and authoritative commitment.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{
    fixture_coordinate_transition, ClaimReplayGeometry, NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplayError,
    NebulaFPrimeClaimReplaySynthesis, COORDINATE_COMMITMENT_FIELDS, FIRST_CLAIM_PROGRAM_CURSOR,
    PI_CCS_RUNNING_COMMITMENT_FIELDS, PI_CCS_RUNNING_PUBLIC_FIELDS, PI_CCS_STATEMENT_FRESH_FIELDS, SPONGE_WIDTH,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::f_prime::streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit,
    CLAIM_CHUNK_FIELDS, CLAIM_FRAME_FIELDS,
};
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix,
    build_linked_overlay_low_norm_r1cs, lower_field_r1cs,
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix,
    project_rows_with_complete_source_provenance_with_alignment, LinkedOverlayLowNormR1cs, MultiBranchLowNormR1cs,
    OverlayBaseFieldPin, OverlayFieldLink, OverlayKindLinks, SelectiveCompactLayoutAudit,
    SelectiveProjectedDecoderRunProvenance, SelectiveProjectedRowsAudit, SelectiveRewriteKind,
    SelectiveSourceRowDisposition,
};
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_commit_coordinate_fields, SisAccumulatorConfig, PI_CCS_RUNNING_COMMITMENTS_COORDINATE_SIS_CONFIG,
    PI_CCS_RUNNING_PUBLIC_COORDINATE_SIS_CONFIG, PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
};

const NOOP_KIND: usize = 0;
const FIRST_ACTIVE_KIND: usize = 1;
const BASE_FULL_KIND: usize = 0;
const BASE_FINAL_KIND: usize = 1;
const ACTIVE_CHUNKS: usize = CLAIM_FRAME_FIELDS.div_ceil(CLAIM_CHUNK_FIELDS);
const OVERLAY_KINDS: usize = FIRST_ACTIVE_KIND + ACTIVE_CHUNKS;

pub const fn production_claim_coordinate_overlay_kind_count() -> usize {
    OVERLAY_KINDS
}

/// Compact source-field link contract for one non-no-op coordinate overlay.
/// Active chunk fields are one contiguous range in overlay allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimCoordinateOverlayLinkRun {
    overlay_kind: usize,
    phase_kind: usize,
    chunk_index: usize,
    active_offset_start: usize,
    active_field_count: usize,
}

impl NebulaFPrimeClaimCoordinateOverlayLinkRun {
    pub const fn overlay_kind(self) -> usize {
        self.overlay_kind
    }

    pub const fn phase_kind(self) -> usize {
        self.phase_kind
    }

    pub const fn chunk_index(self) -> usize {
        self.chunk_index
    }

    pub const fn active_offset_start(self) -> usize {
        self.active_offset_start
    }

    pub const fn active_field_count(self) -> usize {
        self.active_field_count
    }
}

/// One synthesized coordinate overlay and its exact source-link columns.
pub struct NebulaFPrimeClaimCoordinateOverlaySynthesis {
    chunk_index: Option<usize>,
    builder: R1csBuilder,
    before_statement_fresh_columns: Vec<usize>,
    after_statement_fresh_columns: Vec<usize>,
    before_running_commitments_columns: Vec<usize>,
    after_running_commitments_columns: Vec<usize>,
    before_running_public_columns: Vec<usize>,
    after_running_public_columns: Vec<usize>,
    chunk_columns: Vec<(usize, usize)>,
}

impl NebulaFPrimeClaimCoordinateOverlaySynthesis {
    #[doc(hidden)]
    pub fn production_kind(kind: usize) -> Option<Self> {
        match kind {
            NOOP_KIND => Some(Self::noop()),
            FIRST_ACTIVE_KIND..OVERLAY_KINDS => Some(Self::active(kind - FIRST_ACTIVE_KIND)),
            _ => None,
        }
    }

    fn noop() -> Self {
        let mut builder = R1csBuilder::new();
        builder.begin_encoding_stage("nebula.streaming.claim_coordinate_overlay.noop");
        let one = Lc::from_const(F::ONE);
        builder.enforce(&one, &one, &one);
        Self {
            chunk_index: None,
            builder,
            before_statement_fresh_columns: Vec::new(),
            after_statement_fresh_columns: Vec::new(),
            before_running_commitments_columns: Vec::new(),
            after_running_commitments_columns: Vec::new(),
            before_running_public_columns: Vec::new(),
            after_running_public_columns: Vec::new(),
            chunk_columns: Vec::new(),
        }
    }

    fn active(chunk_index: usize) -> Self {
        let geometry = ClaimReplayGeometry::production();
        let transition = fixture_coordinate_transition(chunk_index, geometry);
        assert!(
            !transition.statement_fresh_positions.is_empty()
                || !transition.running_commitment_positions.is_empty()
                || !transition.running_public_positions.is_empty(),
            "every production claim chunk owns metadata"
        );

        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();
        builder.begin_encoding_stage("nebula.streaming.claim_coordinate_overlay.state");
        let before_statement_fresh = builder.alloc_vec(&transition.before_statement_fresh);
        let after_statement_fresh = builder.alloc_vec(&transition.after_statement_fresh);
        let before_running_commitments = builder.alloc_vec(&transition.before_running_commitments);
        let after_running_commitments = builder.alloc_vec(&transition.after_running_commitments);
        let before_running_public = builder.alloc_vec(&transition.before_running_public);
        let after_running_public = builder.alloc_vec(&transition.after_running_public);
        let before_statement_fresh_columns = before_statement_fresh
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let after_statement_fresh_columns = after_statement_fresh
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let before_running_commitments_columns = before_running_commitments
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let after_running_commitments_columns = after_running_commitments
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let before_running_public_columns = before_running_public
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();
        let after_running_public_columns = after_running_public
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>();

        builder.begin_encoding_stage("nebula.streaming.claim_coordinate_overlay.chunk");
        let mut chunk_columns = enforce_map_transition(
            &mut builder,
            PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
            PI_CCS_STATEMENT_FRESH_FIELDS,
            &transition.chunk,
            &transition.statement_fresh_positions,
            &before_statement_fresh,
            &after_statement_fresh,
        );
        chunk_columns.extend(enforce_map_transition(
            &mut builder,
            PI_CCS_RUNNING_COMMITMENTS_COORDINATE_SIS_CONFIG,
            PI_CCS_RUNNING_COMMITMENT_FIELDS,
            &transition.chunk,
            &transition.running_commitment_positions,
            &before_running_commitments,
            &after_running_commitments,
        ));
        chunk_columns.extend(enforce_map_transition(
            &mut builder,
            PI_CCS_RUNNING_PUBLIC_COORDINATE_SIS_CONFIG,
            PI_CCS_RUNNING_PUBLIC_FIELDS,
            &transition.chunk,
            &transition.running_public_positions,
            &before_running_public,
            &after_running_public,
        ));
        chunk_columns.sort_unstable_by_key(|&(offset, _)| offset);
        if chunk_index == 0 {
            for &wire in before_statement_fresh
                .iter()
                .chain(&before_running_commitments)
                .chain(&before_running_public)
            {
                builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
            }
        }
        Self {
            chunk_index: Some(chunk_index),
            builder,
            before_statement_fresh_columns,
            after_statement_fresh_columns,
            before_running_commitments_columns,
            after_running_commitments_columns,
            before_running_public_columns,
            after_running_public_columns,
            chunk_columns,
        }
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

    pub fn chunk_index(&self) -> Option<usize> {
        self.chunk_index
    }

    pub fn before_statement_fresh_column(&self, coordinate: usize) -> Option<usize> {
        self.before_statement_fresh_columns.get(coordinate).copied()
    }

    pub fn after_statement_fresh_column(&self, coordinate: usize) -> Option<usize> {
        self.after_statement_fresh_columns.get(coordinate).copied()
    }

    pub fn before_running_commitments_column(&self, coordinate: usize) -> Option<usize> {
        self.before_running_commitments_columns
            .get(coordinate)
            .copied()
    }

    pub fn after_running_commitments_column(&self, coordinate: usize) -> Option<usize> {
        self.after_running_commitments_columns
            .get(coordinate)
            .copied()
    }

    pub fn before_running_public_column(&self, coordinate: usize) -> Option<usize> {
        self.before_running_public_columns.get(coordinate).copied()
    }

    pub fn after_running_public_column(&self, coordinate: usize) -> Option<usize> {
        self.after_running_public_columns.get(coordinate).copied()
    }

    pub fn chunk_columns(&self) -> &[(usize, usize)] {
        &self.chunk_columns
    }

    #[doc(hidden)]
    pub fn builder_for_artifact(&self) -> &R1csBuilder {
        &self.builder
    }

    #[doc(hidden)]
    pub fn normalized_field_assignment_for_artifact(&self) -> Vec<F> {
        self.builder.witness().to_vec()
    }

    #[doc(hidden)]
    pub fn witness_value(&self, column: usize) -> Option<F> {
        self.builder.witness().get(column).copied()
    }

    #[doc(hidden)]
    pub fn tamper_witness_for_test(&mut self, column: usize, value: F) {
        self.builder.tamper_witness(column, value);
    }

    fn into_lowered(self) -> Result<crate::frontends::r1cs_f_prime::LoweredFieldR1cs, NebulaFPrimeClaimReplayError> {
        Ok(lower_field_r1cs(self.builder, &[])?)
    }
}

fn enforce_map_transition(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    total_fields: usize,
    chunk: &[F],
    positions: &[(usize, usize)],
    before: &[Var],
    after: &[Var],
) -> Vec<(usize, usize)> {
    if positions.is_empty() {
        enforce_commitment_carry(builder, before, after);
        return Vec::new();
    }
    let positioned = positions
        .iter()
        .map(|&(field, offset)| {
            let wire = builder.alloc(chunk[offset]);
            ((field, offset), wire)
        })
        .collect::<Vec<_>>();
    builder.begin_encoding_stage("nebula.streaming.claim_coordinate_overlay.binding");
    let commitment_inputs = positioned
        .iter()
        .map(|&((field, _), wire)| (field, wire))
        .collect::<Vec<_>>();
    let partial = enforce_commit_coordinate_fields(builder, config, total_fields, &commitment_inputs)
        .expect("fixed production PiCCS coordinate overlay");
    enforce_commitment_update(builder, before, after, &partial.data);
    positioned
        .into_iter()
        .map(|((_, offset), wire)| (offset, wire.col()))
        .collect()
}

fn enforce_commitment_carry(builder: &mut R1csBuilder, before: &[Var], after: &[Var]) {
    assert_eq!(before.len(), COORDINATE_COMMITMENT_FIELDS);
    assert_eq!(after.len(), COORDINATE_COMMITMENT_FIELDS);
    for (&before, &after) in before.iter().zip(after) {
        builder.enforce_eq(&Lc::from_var(after), &Lc::from_var(before));
    }
}

fn enforce_commitment_update(builder: &mut R1csBuilder, before: &[Var], after: &[Var], partial: &[Var]) {
    assert_eq!(before.len(), COORDINATE_COMMITMENT_FIELDS);
    assert_eq!(after.len(), COORDINATE_COMMITMENT_FIELDS);
    enforce_coordinate_commitment_update_raw(builder, before, after, partial);
}

fn enforce_coordinate_commitment_update_raw(builder: &mut R1csBuilder, before: &[Var], after: &[Var], partial: &[Var]) {
    assert_eq!(partial.len(), COORDINATE_COMMITMENT_FIELDS);
    for ((&before, &after), &partial) in before.iter().zip(after).zip(partial) {
        let expected = Lc::from_var(before).add_scaled(&Lc::from_var(partial), F::ONE);
        builder.enforce_eq(&Lc::from_var(after), &expected);
    }
}

fn production_syntheses() -> Vec<NebulaFPrimeClaimCoordinateOverlaySynthesis> {
    let mut syntheses = Vec::with_capacity(OVERLAY_KINDS);
    syntheses.push(NebulaFPrimeClaimCoordinateOverlaySynthesis::noop());
    syntheses.extend((0..ACTIVE_CHUNKS).map(NebulaFPrimeClaimCoordinateOverlaySynthesis::active));
    syntheses
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimCoordinateOverlayShapeAudit {
    pub kinds: usize,
    pub active_kinds: usize,
    pub active_fields: usize,
    pub source_rows: usize,
    pub source_columns: usize,
    pub low_norm_rows: usize,
    pub low_norm_columns: usize,
    pub low_norm_public_columns: usize,
    pub low_norm_total_coordinates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimReplayBaseShapeAudit {
    pub full_rows: usize,
    pub full_columns: usize,
    pub final_rows: usize,
    pub final_columns: usize,
    pub low_norm_rows: usize,
    pub low_norm_columns: usize,
    pub low_norm_public_columns: usize,
    pub low_norm_total_coordinates: usize,
}

#[doc(hidden)]
pub fn production_claim_replay_base_source_arms(
) -> Result<(Vec<crate::frontends::r1cs_f_prime::SparseR1cs>, usize), NebulaFPrimeClaimReplayError> {
    let full = NebulaFPrimeClaimReplaySynthesis::production_base_full(0)?;
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    let (full, shared) = full.into_lowered()?;
    let (final_chunk, final_shared) = final_chunk.into_lowered()?;
    assert_eq!(shared, final_shared, "claim-replay base arms share one exact prefix");
    Ok((vec![full.into_parts().0, final_chunk.into_parts().0], shared))
}

pub fn production_claim_replay_base_shape_audit(
) -> Result<NebulaFPrimeClaimReplayBaseShapeAudit, NebulaFPrimeClaimReplayError> {
    let full = NebulaFPrimeClaimReplaySynthesis::production_base_full(0)?;
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    let full_rows = full.rows();
    let full_columns = full.columns();
    let final_rows = final_chunk.rows();
    let final_columns = final_chunk.columns();
    let (arms, shared) = production_claim_replay_base_source_arms()?;
    let prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        shared,
        0,
        neo_math::D,
        0,
        crate::config::B_BASE,
    )?;
    let shape = prepared.shape_summary();
    Ok(NebulaFPrimeClaimReplayBaseShapeAudit {
        full_rows,
        full_columns,
        final_rows,
        final_columns,
        low_norm_rows: shape.rows,
        low_norm_columns: shape.columns,
        low_norm_public_columns: shape.public_input_len,
        low_norm_total_coordinates: shape.total_coordinates,
    })
}

pub fn build_production_claim_replay_base_low_norm_r1cs() -> Result<MultiBranchLowNormR1cs, NebulaFPrimeClaimReplayError>
{
    let (arms, shared) = production_claim_replay_base_source_arms()?;
    Ok(
        prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            shared,
            0,
            neo_math::D,
            0,
            crate::config::B_BASE,
        )?
        .finish()?,
    )
}

/// Exact compact row ledger and requested source-to-final decoder runs from
/// the same prepared layout as the production claim-replay base emitter.
#[doc(hidden)]
pub fn production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<(SelectiveCompactLayoutAudit, Vec<SelectiveProjectedDecoderRunProvenance>), NebulaFPrimeClaimReplayError> {
    let (arms, shared) = production_claim_replay_base_source_arms()?;
    Ok(
        audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
            &arms,
            shared,
            0,
            neo_math::D,
            0,
            crate::config::B_BASE,
            requests,
        )?,
    )
}

/// Exact final-row projection for the retained production-base semantics of
/// one claim-replay arm. The state-digest stage is outside this phase owner.
#[doc(hidden)]
pub fn production_claim_replay_base_semantic_row_projection(
    kind: NebulaFPrimeClaimReplayArmKind,
) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimeClaimReplayError> {
    production_claim_replay_base_row_projection(kind, true)
}

/// Exact retained-row projection without the separately certified Poseidon2
/// blocks. This is the compact source for retained-family Lean certificates.
#[doc(hidden)]
pub fn production_claim_replay_base_retained_row_projection(
    kind: NebulaFPrimeClaimReplayArmKind,
) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimeClaimReplayError> {
    production_claim_replay_base_row_projection(kind, false)
}

fn production_claim_replay_base_row_projection(
    kind: NebulaFPrimeClaimReplayArmKind,
    include_poseidon: bool,
) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimeClaimReplayError> {
    let arm = match kind {
        NebulaFPrimeClaimReplayArmKind::Full => BASE_FULL_KIND,
        NebulaFPrimeClaimReplayArmKind::Final => BASE_FINAL_KIND,
    };
    let (arms, shared) = production_claim_replay_base_source_arms()?;
    let expected_stages: &[&str] = match kind {
        NebulaFPrimeClaimReplayArmKind::Full => &[
            "nebula.streaming.claim_replay.state_words",
            "nebula.streaming.claim_replay.chunk",
            "nebula.streaming.claim_replay.state",
            "nebula.streaming.claim_replay.poseidon2",
            "nebula.streaming.claim_replay.state_digest",
        ],
        NebulaFPrimeClaimReplayArmKind::Final => &[
            "nebula.streaming.claim_replay.state_words",
            "nebula.streaming.claim_replay.chunk",
            "nebula.streaming.claim_replay.state",
            "nebula.streaming.claim_replay.poseidon2",
            "nebula.streaming.claim_replay.ready",
            "nebula.streaming.claim_replay.state_digest",
        ],
    };
    let stages = arms[arm].physical_stage_ranges();
    assert_eq!(
        stages.len(),
        expected_stages.len(),
        "claim-replay base physical-stage count"
    );
    for (stage, expected) in stages.iter().zip(expected_stages) {
        assert_eq!(stage.path(), *expected, "claim-replay base physical-stage identity");
    }
    let digest_stage = expected_stages.len() - 1;

    let (layout, _) = audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
        &arms,
        shared,
        0,
        neo_math::D,
        0,
        crate::config::B_BASE,
        &[],
    )?;
    let mut selected_rows = Vec::new();
    let mut retained_row_pairs = Vec::new();
    for run in layout.rows().arms()[arm].source_runs() {
        let stage = run
            .stage_occurrence()
            .expect("claim-replay base source row has one physical-stage owner");
        if stage >= digest_stage || run.disposition() != SelectiveSourceRowDisposition::Retained {
            continue;
        }
        let source_rows = run.source_rows();
        let emitted_start = run
            .emitted_start()
            .expect("retained claim-replay base source run has final-row ownership");
        for offset in 0..source_rows.len() {
            let emitted_row = emitted_start + offset;
            selected_rows.push(emitted_row);
            retained_row_pairs.push((source_rows.start + offset, emitted_row));
        }
    }
    if include_poseidon {
        for rewrite in layout.rows().rewrites() {
            if rewrite.arm() == arm
                && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                && rewrite.source_stage_occurrence() == Some(3)
            {
                selected_rows.extend(rewrite.emitted_rows());
            }
        }
    }
    selected_rows.sort_unstable();
    assert!(
        selected_rows.windows(2).all(|rows| rows[0] != rows[1]),
        "claim-replay base semantic final rows must have exclusive ownership"
    );

    Ok(project_rows_with_complete_source_provenance_with_alignment(
        &arms,
        shared,
        0,
        neo_math::D,
        0,
        &selected_rows,
        arm,
        &[],
        &retained_row_pairs,
    )?)
}

pub fn production_claim_coordinate_overlay_shape_audit(
) -> Result<NebulaFPrimeClaimCoordinateOverlayShapeAudit, NebulaFPrimeClaimReplayError> {
    let syntheses = production_syntheses();
    let active_fields = syntheses
        .iter()
        .map(|synthesis| synthesis.chunk_columns.len())
        .sum();
    let source_rows = syntheses.iter().map(|synthesis| synthesis.rows()).sum();
    let source_columns = syntheses.iter().map(|synthesis| synthesis.columns()).sum();
    let arms = syntheses
        .into_iter()
        .map(|synthesis| {
            synthesis
                .into_lowered()
                .map(|lowered| lowered.into_parts().0)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        0,
        0,
        1,
        0,
        crate::config::B_BASE,
    )?;
    let shape = prepared.shape_summary();
    Ok(NebulaFPrimeClaimCoordinateOverlayShapeAudit {
        kinds: OVERLAY_KINDS,
        active_kinds: ACTIVE_CHUNKS,
        active_fields,
        source_rows,
        source_columns,
        low_norm_rows: shape.rows,
        low_norm_columns: shape.columns,
        low_norm_public_columns: shape.public_input_len,
        low_norm_total_coordinates: shape.total_coordinates,
    })
}

pub(crate) fn production_claim_coordinate_overlay_sparse_arms(
) -> Result<Vec<crate::frontends::r1cs_f_prime::SparseR1cs>, NebulaFPrimeClaimReplayError> {
    production_syntheses()
        .into_iter()
        .map(|synthesis| {
            synthesis
                .into_lowered()
                .map(|lowered| lowered.into_parts().0)
        })
        .collect()
}

pub fn build_production_claim_coordinate_overlay_low_norm_r1cs(
) -> Result<MultiBranchLowNormR1cs, NebulaFPrimeClaimReplayError> {
    let arms = production_claim_coordinate_overlay_sparse_arms()?;
    Ok(
        prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            0,
            0,
            1,
            0,
            crate::config::B_BASE,
        )?
        .finish()?,
    )
}

/// Compile only the 98 active claim-coordinate kinds. Source rows are the
/// exact active suffix of the schedule overlay source, with the no-op omitted.
pub fn build_production_claim_active_coordinate_overlay_low_norm_r1cs(
) -> Result<MultiBranchLowNormR1cs, NebulaFPrimeClaimReplayError> {
    let arms = production_claim_coordinate_overlay_sparse_arms()?
        .into_iter()
        .skip(FIRST_ACTIVE_KIND)
        .collect();
    Ok(
        prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms,
            0,
            0,
            1,
            0,
            crate::config::B_BASE,
        )?
        .finish()?,
    )
}

/// Exact compact row ledger and requested source-to-final decoder runs from
/// the same active-arm layout as the production coordinate-overlay emitter.
#[doc(hidden)]
pub fn production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges(
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<(SelectiveCompactLayoutAudit, Vec<SelectiveProjectedDecoderRunProvenance>), NebulaFPrimeClaimReplayError> {
    let arms = production_claim_coordinate_overlay_sparse_arms()?
        .into_iter()
        .skip(FIRST_ACTIVE_KIND)
        .collect::<Vec<_>>();
    Ok(
        audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
            &arms,
            0,
            0,
            1,
            0,
            crate::config::B_BASE,
            requests,
        )?,
    )
}

/// Exact non-seeded final-row projection for one active coordinate-overlay
/// source arm. Compact seeded rows remain owned by
/// `production_claim_active_coordinate_overlay_seeded_placements`.
/// Retained non-seeded rows and rewrite rows come from the same compact ledger
/// as the production emitter. The caller selects one arm so the audit does not
/// hold projected rows for all 98 arms at once.
#[doc(hidden)]
pub fn production_claim_active_coordinate_overlay_nonseeded_row_projection(
    arm: usize,
) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimeClaimReplayError> {
    let arms = production_claim_coordinate_overlay_sparse_arms()?
        .into_iter()
        .skip(FIRST_ACTIVE_KIND)
        .collect::<Vec<_>>();
    assert!(arm < arms.len(), "active coordinate-overlay arm is in range");

    let (layout, _) = audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
        &arms,
        0,
        0,
        1,
        0,
        crate::config::B_BASE,
        &[],
    )?;
    let arm_rows = &layout.rows().arms()[arm];
    let source_row_count = arm_rows
        .source_runs()
        .iter()
        .map(|run| run.source_rows().len())
        .sum::<usize>();
    assert_eq!(source_row_count, arms[arm].n, "active overlay source-row cover");
    assert!(
        arms[arm].b.seeded_phi81_blocks().is_empty() && arms[arm].c.seeded_phi81_blocks().is_empty(),
        "active overlay compact blocks have one exact A-port owner"
    );
    let seeded_source_rows = arms[arm]
        .a
        .seeded_phi81_blocks()
        .iter()
        .map(|block| block.row_start()..block.row_end())
        .collect::<Vec<_>>();
    let seeded_source_row_count = seeded_source_rows
        .iter()
        .map(std::ops::Range::len)
        .sum::<usize>();

    let mut selected_rows = Vec::new();
    let mut retained_row_pairs = Vec::new();
    let mut omitted_seeded_rows = 0;
    for run in arm_rows.source_runs() {
        if run.disposition() != SelectiveSourceRowDisposition::Retained {
            continue;
        }
        let source_rows = run.source_rows();
        let emitted_start = run
            .emitted_start()
            .expect("retained active-overlay source run has final-row ownership");
        for offset in 0..source_rows.len() {
            let source_row = source_rows.start + offset;
            if seeded_source_rows
                .iter()
                .any(|rows| rows.contains(&source_row))
            {
                omitted_seeded_rows += 1;
                continue;
            }
            let emitted_row = emitted_start + offset;
            selected_rows.push(emitted_row);
            retained_row_pairs.push((source_row, emitted_row));
        }
    }
    assert_eq!(
        omitted_seeded_rows, seeded_source_row_count,
        "active overlay projection omits exactly the compact seeded source rows"
    );
    for rewrite in layout.rows().rewrites() {
        if rewrite.arm() == arm {
            selected_rows.extend(rewrite.emitted_rows());
        }
    }
    selected_rows.sort_unstable();
    assert!(
        selected_rows.windows(2).all(|rows| rows[0] != rows[1]),
        "active overlay final rows must have exclusive ownership"
    );

    Ok(project_rows_with_complete_source_provenance_with_alignment(
        &arms,
        0,
        0,
        1,
        0,
        &selected_rows,
        arm,
        &[],
        &retained_row_pairs,
    )?)
}

/// One affine segment of exact source-to-final seeded word starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimCoordinateOverlayWordStartRun {
    source_start: usize,
    final_start: usize,
    count: usize,
    source_stride: usize,
    final_stride: usize,
}

impl NebulaFPrimeClaimCoordinateOverlayWordStartRun {
    pub const fn source_start(self) -> usize {
        self.source_start
    }

    pub const fn final_start(self) -> usize {
        self.final_start
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub const fn source_stride(self) -> usize {
        self.source_stride
    }

    pub const fn final_stride(self) -> usize {
        self.final_stride
    }
}

/// One exact compact seeded-block placement from an active coordinate-overlay
/// source arm into its retained final selective rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeClaimCoordinateOverlaySeededPlacement {
    arm: usize,
    selector_column: usize,
    source_row_start: usize,
    final_row_start: usize,
    word_start_runs: Vec<NebulaFPrimeClaimCoordinateOverlayWordStartRun>,
    word_count: usize,
    word_width: usize,
    kappa: usize,
    message_columns: usize,
}

impl NebulaFPrimeClaimCoordinateOverlaySeededPlacement {
    pub const fn arm(&self) -> usize {
        self.arm
    }

    pub const fn selector_column(&self) -> usize {
        self.selector_column
    }

    pub const fn source_row_start(&self) -> usize {
        self.source_row_start
    }

    pub const fn final_row_start(&self) -> usize {
        self.final_row_start
    }

    pub fn word_start_runs(&self) -> &[NebulaFPrimeClaimCoordinateOverlayWordStartRun] {
        &self.word_start_runs
    }

    pub const fn word_count(&self) -> usize {
        self.word_count
    }

    pub const fn word_width(&self) -> usize {
        self.word_width
    }

    pub const fn kappa(&self) -> usize {
        self.kappa
    }

    pub const fn message_columns(&self) -> usize {
        self.message_columns
    }
}

fn paired_word_start_runs(
    source: &[usize],
    final_starts: &[usize],
) -> Vec<NebulaFPrimeClaimCoordinateOverlayWordStartRun> {
    assert_eq!(source.len(), final_starts.len());
    let mut runs = Vec::new();
    let mut cursor = 0;
    while cursor < source.len() {
        let mut count = 1;
        let mut source_stride = 0;
        let mut final_stride = 0;
        if cursor + 1 < source.len()
            && source[cursor] <= source[cursor + 1]
            && final_starts[cursor] <= final_starts[cursor + 1]
        {
            source_stride = source[cursor + 1] - source[cursor];
            final_stride = final_starts[cursor + 1] - final_starts[cursor];
            count = 2;
            while cursor + count < source.len()
                && source[cursor + count - 1] <= source[cursor + count]
                && final_starts[cursor + count - 1] <= final_starts[cursor + count]
                && source[cursor + count] - source[cursor + count - 1] == source_stride
                && final_starts[cursor + count] - final_starts[cursor + count - 1] == final_stride
            {
                count += 1;
            }
        }
        runs.push(NebulaFPrimeClaimCoordinateOverlayWordStartRun {
            source_start: source[cursor],
            final_start: final_starts[cursor],
            count,
            source_stride,
            final_stride,
        });
        cursor += count;
    }
    assert_eq!(runs.iter().map(|run| run.count).sum::<usize>(), source.len());
    runs
}

/// Rust-checked compact block placements for all active coordinate calls.
/// The source and final blocks must use one exact common seeded schedule.
#[doc(hidden)]
pub fn production_claim_active_coordinate_overlay_seeded_placements(
) -> Result<Vec<NebulaFPrimeClaimCoordinateOverlaySeededPlacement>, NebulaFPrimeClaimReplayError> {
    let arms = production_claim_coordinate_overlay_sparse_arms()?
        .into_iter()
        .skip(FIRST_ACTIVE_KIND)
        .collect::<Vec<_>>();
    let source_blocks = arms
        .iter()
        .map(|arm| arm.a.seeded_phi81_blocks().to_vec())
        .collect::<Vec<_>>();
    let relation = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        0,
        0,
        1,
        0,
        crate::config::B_BASE,
    )?
    .finish()?;
    let snapshot = relation
        .selective_snapshot()
        .expect("freshly compiled coordinate overlay has one checked snapshot");
    let final_blocks = relation.structure().matrices[2].seeded_phi81_blocks();
    assert!(relation
        .structure()
        .matrices
        .iter()
        .enumerate()
        .all(|(port, matrix)| port == 2 || matrix.seeded_phi81_blocks().is_empty()));

    let mut placements = Vec::new();
    for (arm, blocks) in source_blocks.iter().enumerate() {
        let plan = snapshot
            .arm(arm)
            .expect("active coordinate-overlay arm has one checked slot plan");
        let row_plan = &snapshot.compiler_audit().rows().arms()[arm];
        for source in blocks {
            let source_rows = source.row_start()..source.row_end();
            let run = row_plan
                .source_runs()
                .iter()
                .find(|run| {
                    let owned = run.source_rows();
                    owned.start <= source_rows.start && source_rows.end <= owned.end
                })
                .expect("source seeded block has one compiler row owner");
            assert_eq!(run.disposition(), SelectiveSourceRowDisposition::Retained);
            let owned = run.source_rows();
            let final_row_start = run
                .emitted_start()
                .expect("retained seeded block has final-row ownership")
                + source_rows.start
                - owned.start;

            let final_word_starts = source
                .word_starts()
                .iter()
                .map(|&source_start| {
                    let first = plan
                        .slot(source_start)
                        .expect("seeded word starts in one direct final slot");
                    assert_eq!(first.len(), 1);
                    for offset in 0..source.word_width() {
                        let coordinate = plan
                            .slot(source_start + offset)
                            .expect("every seeded word coordinate has one direct final slot");
                        assert_eq!(coordinate.len(), 1);
                        assert_eq!(coordinate.start(), first.start() + offset);
                    }
                    first.start()
                })
                .collect::<Vec<_>>();
            let final_block = final_blocks
                .iter()
                .find(|block| block.row_start() == final_row_start)
                .expect("retained final seeded block exists at its checked row placement");
            assert_eq!(final_block.word_starts(), final_word_starts);
            assert_eq!(final_block.word_width(), source.word_width());
            assert_eq!(final_block.kappa(), source.kappa());
            assert_eq!(final_block.message_cols(), source.message_cols());
            assert_eq!(final_block.chunk_size(), source.chunk_size());
            assert_eq!(final_block.chunk_seeds_by_row(), source.chunk_seeds_by_row());
            assert_eq!(
                final_block.has_superneo_transformed_columns(),
                source.has_superneo_transformed_columns()
            );
            let word_start_runs = paired_word_start_runs(source.word_starts(), &final_word_starts);
            placements.push(NebulaFPrimeClaimCoordinateOverlaySeededPlacement {
                arm,
                selector_column: snapshot.selector_cols()[arm],
                source_row_start: source.row_start(),
                final_row_start,
                word_start_runs,
                word_count: source.word_starts().len(),
                word_width: source.word_width(),
                kappa: source.kappa(),
                message_columns: source.message_cols(),
            });
        }
    }
    assert_eq!(placements.len(), final_blocks.len());
    Ok(placements)
}

/// Semantic phase code owned by each exact production-base arm.
pub fn production_claim_replay_base_phase_kinds() -> Vec<usize> {
    vec![
        NebulaFPrimeStreamingCircuitKind::ClaimReplayFull.code() as usize,
        NebulaFPrimeStreamingCircuitKind::ClaimReplayFinal.code() as usize,
    ]
}

/// Production-base arm selected by each active coordinate-overlay kind.
pub fn production_claim_active_coordinate_overlay_base_kind_map() -> Vec<usize> {
    (0..ACTIVE_CHUNKS)
        .map(|chunk| {
            if chunk + 1 == ACTIVE_CHUNKS {
                BASE_FINAL_KIND
            } else {
                BASE_FULL_KIND
            }
        })
        .collect()
}

/// Exact active link contracts with the schedule no-op index removed.
pub fn production_claim_active_coordinate_overlay_links() -> Vec<OverlayKindLinks> {
    production_claim_coordinate_overlay_links()
        .into_iter()
        .map(|mut contract| {
            contract.overlay_kind -= FIRST_ACTIVE_KIND;
            contract
        })
        .collect()
}

/// One exact selected assignment for the production claim-replay base and its
/// active coordinate overlay.
pub fn build_production_claim_replay_linked_overlay_low_norm_r1cs(
) -> Result<LinkedOverlayLowNormR1cs, NebulaFPrimeClaimReplayError> {
    Ok(build_linked_overlay_low_norm_r1cs(
        build_production_claim_replay_base_low_norm_r1cs()?,
        build_production_claim_active_coordinate_overlay_low_norm_r1cs()?,
        production_claim_replay_base_phase_kinds(),
        production_claim_active_coordinate_overlay_base_kind_map(),
        production_claim_active_coordinate_overlay_links(),
    )?)
}

pub fn production_claim_coordinate_overlay_kind_map() -> Vec<usize> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    program
        .work_items()
        .iter()
        .map(|item| {
            if item.phase() != NebulaFPrimeStreamingPhase::ClaimReplay {
                return NOOP_KIND;
            }
            FIRST_ACTIVE_KIND + item.index()
        })
        .collect()
}

pub fn production_claim_coordinate_overlay_links() -> Vec<OverlayKindLinks> {
    let syntheses = production_syntheses();
    let mut links = Vec::with_capacity(OVERLAY_KINDS - 1);
    for chunk in 0..ACTIVE_CHUNKS {
        let kind = FIRST_ACTIVE_KIND + chunk;
        if chunk + 1 == ACTIVE_CHUNKS {
            links.push(links_for(
                kind,
                NebulaFPrimeStreamingCircuitKind::ClaimReplayFinal,
                &NebulaFPrimeClaimReplaySynthesis::production_base_final(),
                &syntheses[kind],
            ));
        } else {
            links.push(links_for(
                kind,
                NebulaFPrimeStreamingCircuitKind::ClaimReplayFull,
                &NebulaFPrimeClaimReplaySynthesis::production_base_full(chunk).expect("active claim base"),
                &syntheses[kind],
            ));
        }
    }
    links
}

/// Exact compact form of [`production_claim_coordinate_overlay_links`]. The
/// no-op kind has no link contract and is omitted.
pub fn production_claim_coordinate_overlay_link_runs() -> Vec<NebulaFPrimeClaimCoordinateOverlayLinkRun> {
    let syntheses = production_syntheses();
    let links = production_claim_coordinate_overlay_links();
    links
        .into_iter()
        .map(|contract| {
            let overlay = &syntheses[contract.overlay_kind];
            let chunk_index = overlay
                .chunk_index
                .expect("linked overlay kind has a claim chunk");
            let active_field_count = overlay.chunk_columns.len();
            let active_offset_start = overlay
                .chunk_columns
                .first()
                .map_or(0, |&(offset, _)| offset);
            for (index, &(offset, _)) in overlay.chunk_columns.iter().enumerate() {
                assert_eq!(
                    offset,
                    active_offset_start + index,
                    "active overlay offsets must be contiguous"
                );
            }
            NebulaFPrimeClaimCoordinateOverlayLinkRun {
                overlay_kind: contract.overlay_kind,
                phase_kind: contract.phase_kind,
                chunk_index,
                active_offset_start,
                active_field_count,
            }
        })
        .collect()
}

fn links_for(
    overlay_kind: usize,
    phase_kind: NebulaFPrimeStreamingCircuitKind,
    base: &NebulaFPrimeClaimReplaySynthesis,
    overlay: &NebulaFPrimeClaimCoordinateOverlaySynthesis,
) -> OverlayKindLinks {
    let chunk_index = overlay
        .chunk_index
        .expect("linked overlay kind has one claim chunk");
    let mut fields = Vec::with_capacity(6 * COORDINATE_COMMITMENT_FIELDS + overlay.chunk_columns.len());
    for coordinate in 0..COORDINATE_COMMITMENT_FIELDS {
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_before_statement_fresh_commitment_column(coordinate)
                .expect("base before statement-and-fresh commitment column"),
            overlay_field: overlay
                .before_statement_fresh_column(coordinate)
                .expect("overlay before statement-and-fresh commitment column"),
        });
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_after_statement_fresh_commitment_column(coordinate)
                .expect("base after statement-and-fresh commitment column"),
            overlay_field: overlay
                .after_statement_fresh_column(coordinate)
                .expect("overlay after statement-and-fresh commitment column"),
        });
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_before_running_commitments_binding_column(coordinate)
                .expect("base before running-commitments binding column"),
            overlay_field: overlay
                .before_running_commitments_column(coordinate)
                .expect("overlay before running-commitments binding column"),
        });
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_after_running_commitments_binding_column(coordinate)
                .expect("base after running-commitments binding column"),
            overlay_field: overlay
                .after_running_commitments_column(coordinate)
                .expect("overlay after running-commitments binding column"),
        });
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_before_running_public_binding_column(coordinate)
                .expect("base before running-public binding column"),
            overlay_field: overlay
                .before_running_public_column(coordinate)
                .expect("overlay before running-public binding column"),
        });
        fields.push(OverlayFieldLink {
            phase_field: base
                .normalized_after_running_public_binding_column(coordinate)
                .expect("base after running-public binding column"),
            overlay_field: overlay
                .after_running_public_column(coordinate)
                .expect("overlay after running-public binding column"),
        });
    }
    fields.extend(
        overlay
            .chunk_columns()
            .iter()
            .map(|&(offset, overlay_field)| OverlayFieldLink {
                phase_field: base
                    .normalized_chunk_column(offset)
                    .expect("base linked chunk column"),
                overlay_field,
            }),
    );
    let mut base_pins = vec![OverlayBaseFieldPin {
        phase_field: base
            .normalized_before_program_cursor_column()
            .expect("base before-program-cursor column"),
        value: F::from_usize(FIRST_CLAIM_PROGRAM_CURSOR + chunk_index),
    }];
    if chunk_index == 0 {
        base_pins.extend((0..SPONGE_WIDTH).map(|lane| {
            OverlayBaseFieldPin {
                phase_field: base
                    .normalized_before_runtime_column(lane)
                    .expect("base before-runtime column"),
                value: F::ZERO,
            }
        }));
    }
    OverlayKindLinks {
        overlay_kind,
        phase_kind: phase_kind.code() as usize,
        fields,
        base_pins,
    }
}

const _: () = assert!(ACTIVE_CHUNKS == 98);
const _: () = assert!(OVERLAY_KINDS == 99);
