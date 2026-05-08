//! Owns the canonical Stage 1 row-family root and selected-opening claims for the production RV32IM path.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::stage1::{stage1_row_digest, Stage1RowBinding, Stage1Summary};

use super::{
    simple::SimpleKernelError,
    simple_openings::{Stage1OpeningPoints, Stage1SelectedOpeningClaim},
    stage_artifacts::{selected_opening_object, selected_opening_ref, Stage1ClaimSurface},
    AjtaiFamilyKind,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1CanonicalRowBundle {
    pub rows_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Stage1CanonicalRowBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_canonical_rows");
        tr.append_message(b"rv32im/stage1_canonical_rows/rows_digest", &self.rows_digest);
        tr.digest32()
    }
}

pub(super) fn build_stage1_artifact_parts(
    stage1: &Stage1Summary,
    mix: u64,
) -> (Stage1CanonicalRowBundle, Stage1ClaimSurface) {
    let mut rows_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_rows_family");
    rows_tr.append_u64s(b"rv32im/stage1_rows_family/len", &[stage1.rows.len() as u64]);
    let mut effect_row_count = 0usize;
    let mut commit_row_count = 0usize;
    let mut real_row_count = 0usize;
    let mut preserves_x0_count = 0usize;
    for row in &stage1.rows {
        rows_tr.append_message(b"rv32im/stage1_rows_family/row_digest", &stage1_row_digest(row));
        effect_row_count += row.is_effect_row as usize;
        commit_row_count += row.is_commit_row as usize;
        real_row_count += row.is_real as usize;
        preserves_x0_count += row.preserves_x0 as usize;
    }
    let rows = Stage1CanonicalRowBundle {
        rows_digest: rows_tr.digest32(),
        digest: [0; 32],
    };
    let rows = Stage1CanonicalRowBundle {
        digest: rows.expected_digest(),
        ..rows
    };
    let claim = Stage1ClaimSurface {
        row_count: stage1.rows.len(),
        effect_row_count,
        commit_row_count,
        real_row_count,
        preserves_x0_count,
        mix,
    };
    (rows, claim)
}

pub(super) fn build_stage1_selected_opening_claim(
    stage1: &Stage1Summary,
    claim: &Stage1ClaimSurface,
    rows: &Stage1CanonicalRowBundle,
) -> Result<Stage1SelectedOpeningClaim, SimpleKernelError> {
    build_stage1_selected_opening_claim_from_rows(&stage1.rows, claim, rows)
}

pub(super) fn build_stage1_selected_opening_claim_from_rows(
    rows: &[Stage1RowBinding],
    claim: &Stage1ClaimSurface,
    row_bundle: &Stage1CanonicalRowBundle,
) -> Result<Stage1SelectedOpeningClaim, SimpleKernelError> {
    let first = rows
        .first()
        .ok_or_else(|| SimpleKernelError::Bridge("rv32im/stage1 selected claim missing first row".into()))?;
    let effect_position = rows
        .iter()
        .position(|row| row.is_effect_row)
        .ok_or_else(|| SimpleKernelError::Bridge("rv32im/stage1 selected claim missing effect row".into()))?;
    let commit_position = rows
        .iter()
        .position(|row| row.is_commit_row)
        .ok_or_else(|| SimpleKernelError::Bridge("rv32im/stage1 selected claim missing commit row".into()))?;
    let effect = &rows[effect_position];
    let commit = &rows[commit_position];
    let last_position = rows.len().saturating_sub(1);
    let last = rows
        .last()
        .ok_or_else(|| SimpleKernelError::Bridge("rv32im/stage1 selected claim missing last row".into()))?;
    let object = selected_opening_object(AjtaiFamilyKind::Stage1Rows, row_bundle.rows_digest);
    let selected = Stage1SelectedOpeningClaim {
        rows_family_digest: row_bundle.rows_digest,
        row_count: claim.row_count as u64,
        effect_row_count: claim.effect_row_count as u64,
        commit_row_count: claim.commit_row_count as u64,
        real_row_count: claim.real_row_count as u64,
        preserves_x0_count: claim.preserves_x0_count as u64,
        first_trace_index: first.trace_index as u64,
        effect_trace_index: effect.trace_index as u64,
        commit_trace_index: commit.trace_index as u64,
        last_trace_index: last.trace_index as u64,
        mix: claim.mix,
        points: Stage1OpeningPoints {
            first: selected_opening_ref(&object, 0, stage1_row_digest(first)),
            effect: selected_opening_ref(&object, effect_position as u64, stage1_row_digest(effect)),
            commit: selected_opening_ref(&object, commit_position as u64, stage1_row_digest(commit)),
            last: selected_opening_ref(&object, last_position as u64, stage1_row_digest(last)),
        },
        digest: [0; 32],
    };
    Ok(Stage1SelectedOpeningClaim {
        digest: selected.expected_digest(),
        ..selected
    })
}
