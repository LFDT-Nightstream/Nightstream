//! Owns the RV32IM root main-lane column-family summary derived from the canonical root-lane witness.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::canonical_openings::{AjtaiFamilyKind, AjtaiObjectId, SelectedOpeningRef};
use super::root_lane_witness::{build_root_lane_witness, RootLanePublicWitness, RootLaneWitness};
use crate::rv32im::ccs::RV32IM_ROOT_ROW_WIDTH;
use crate::rv32im::Rv32ExpandedRow;

pub(crate) const RV32IM_ROOT_LANE_COLUMNS_LAYOUT_V1: u64 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneColumns {
    pub object: AjtaiObjectId,
    pub row_width: u64,
    pub time_len: u64,
    pub column_digests: Vec<[u8; 32]>,
    pub family_digest: [u8; 32],
    pub first_row: Option<SelectedOpeningRef>,
    pub last_row: Option<SelectedOpeningRef>,
    pub digest: [u8; 32],
}

impl RootLaneColumns {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_columns");
        tr.append_message(b"rv32im/root_lane_columns/object_digest", &self.object.digest);
        tr.append_u64s(
            b"rv32im/root_lane_columns/meta",
            &[self.row_width, self.time_len, self.column_digests.len() as u64],
        );
        tr.append_message(b"rv32im/root_lane_columns/family_digest", &self.family_digest);
        for digest in &self.column_digests {
            tr.append_message(b"rv32im/root_lane_columns/column_digest", digest);
        }
        tr.append_u64s(
            b"rv32im/root_lane_columns/first_present",
            &[self.first_row.is_some() as u64],
        );
        if let Some(reference) = &self.first_row {
            tr.append_message(b"rv32im/root_lane_columns/first_digest", &reference.digest);
        }
        tr.append_u64s(
            b"rv32im/root_lane_columns/last_present",
            &[self.last_row.is_some() as u64],
        );
        if let Some(reference) = &self.last_row {
            tr.append_message(b"rv32im/root_lane_columns/last_digest", &reference.digest);
        }
        tr.digest32()
    }

    pub fn opening_refs(&self) -> Vec<&SelectedOpeningRef> {
        let mut refs = Vec::new();
        if let Some(reference) = &self.first_row {
            refs.push(reference);
        }
        if let Some(reference) = &self.last_row {
            refs.push(reference);
        }
        refs
    }
}

pub(crate) fn build_root_lane_columns_from_witness(witness: &RootLaneWitness) -> RootLaneColumns {
    build_root_lane_columns_from_summary_parts(
        witness.time_len(),
        witness.family_digest,
        witness.column_digests.clone(),
        witness.first_row_digest,
        witness.last_row_digest,
    )
}

pub(crate) fn build_root_lane_columns_from_public_witness(witness: &RootLanePublicWitness) -> RootLaneColumns {
    build_root_lane_columns_from_summary_parts(
        witness.time_len,
        witness.family_digest,
        witness.column_digests.clone(),
        witness.first_row_digest,
        witness.last_row_digest,
    )
}

pub(crate) fn build_root_lane_columns_from_summary_parts(
    time_len: usize,
    family_digest: [u8; 32],
    column_digests: Vec<[u8; 32]>,
    first_row_digest: Option<[u8; 32]>,
    last_row_digest: Option<[u8; 32]>,
) -> RootLaneColumns {
    let object = AjtaiObjectId::new(
        AjtaiFamilyKind::RootMainLaneColumns,
        family_digest,
        RV32IM_ROOT_LANE_COLUMNS_LAYOUT_V1,
    );
    let first_row = first_row_digest.map(|digest| {
        SelectedOpeningRef::from_parts(
            AjtaiFamilyKind::RootMainLaneColumns,
            family_digest,
            RV32IM_ROOT_LANE_COLUMNS_LAYOUT_V1,
            0,
            digest,
        )
    });
    let last_row = last_row_digest.map(|digest| {
        SelectedOpeningRef::from_parts(
            AjtaiFamilyKind::RootMainLaneColumns,
            family_digest,
            RV32IM_ROOT_LANE_COLUMNS_LAYOUT_V1,
            time_len.saturating_sub(1) as u64,
            digest,
        )
    });
    let summary = RootLaneColumns {
        object,
        row_width: RV32IM_ROOT_ROW_WIDTH as u64,
        time_len: time_len as u64,
        column_digests,
        family_digest,
        first_row,
        last_row,
        digest: [0; 32],
    };
    RootLaneColumns {
        digest: summary.expected_digest(),
        ..summary
    }
}

pub fn build_root_lane_columns(rows: &[Rv32ExpandedRow]) -> RootLaneColumns {
    build_root_lane_columns_from_witness(&build_root_lane_witness(rows))
}
