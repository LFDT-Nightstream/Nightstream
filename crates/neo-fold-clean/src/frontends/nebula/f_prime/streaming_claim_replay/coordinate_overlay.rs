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
    fixture_coordinate_transition, ClaimReplayGeometry, NebulaFPrimeClaimReplayError, NebulaFPrimeClaimReplaySynthesis,
    COORDINATE_COMMITMENT_FIELDS, PI_CCS_RUNNING_COMMITMENT_FIELDS, PI_CCS_RUNNING_PUBLIC_FIELDS,
    PI_CCS_STATEMENT_FRESH_FIELDS,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::f_prime::streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit,
    CLAIM_CHUNK_FIELDS, CLAIM_FRAME_FIELDS,
};
use crate::frontends::r1cs_f_prime::{
    lower_field_r1cs, prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix,
    MultiBranchLowNormR1cs, OverlayFieldLink, OverlayKindLinks,
};
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_commit_coordinate_fields, SisAccumulatorConfig, PI_CCS_RUNNING_COMMITMENTS_COORDINATE_SIS_CONFIG,
    PI_CCS_RUNNING_PUBLIC_COORDINATE_SIS_CONFIG, PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG,
};

const NOOP_KIND: usize = 0;
const FIRST_ACTIVE_KIND: usize = 1;
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

pub fn production_claim_replay_base_shape_audit(
) -> Result<NebulaFPrimeClaimReplayBaseShapeAudit, NebulaFPrimeClaimReplayError> {
    let full = NebulaFPrimeClaimReplaySynthesis::production_base_full(0)?;
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    let full_rows = full.rows();
    let full_columns = full.columns();
    let final_rows = final_chunk.rows();
    let final_columns = final_chunk.columns();
    let (full, shared) = full.into_lowered()?;
    let (final_chunk, final_shared) = final_chunk.into_lowered()?;
    debug_assert_eq!(shared, final_shared);
    let arms = vec![full.into_parts().0, final_chunk.into_parts().0];
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
    let full = NebulaFPrimeClaimReplaySynthesis::production_base_full(0)?;
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    let (full, shared) = full.into_lowered()?;
    let (final_chunk, final_shared) = final_chunk.into_lowered()?;
    debug_assert_eq!(shared, final_shared);
    let arms = vec![full.into_parts().0, final_chunk.into_parts().0];
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
    OverlayKindLinks {
        overlay_kind,
        phase_kind: phase_kind.code() as usize,
        fields,
    }
}

const _: () = assert!(ACTIVE_CHUNKS == 98);
const _: () = assert!(OVERLAY_KINDS == 99);
