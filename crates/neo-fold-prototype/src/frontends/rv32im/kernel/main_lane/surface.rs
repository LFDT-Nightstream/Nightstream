//! Owns the compact public-facing RV32IM main-lane surface derived from the root lane columns.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::canonical_openings::SelectedOpeningRef;
use super::RootLaneColumns;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imMainLaneSurface {
    pub object_digest: [u8; 32],
    pub family_digest: [u8; 32],
    pub row_width: u64,
    pub public_step_count: u64,
    pub first_public_step: Option<SelectedOpeningRef>,
    pub last_public_step: Option<SelectedOpeningRef>,
    pub digest: [u8; 32],
}

impl Rv32imMainLaneSurface {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_lane_surface");
        tr.append_message(b"rv32im/main_lane_surface/object_digest", &self.object_digest);
        tr.append_message(b"rv32im/main_lane_surface/family_digest", &self.family_digest);
        tr.append_u64s(
            b"rv32im/main_lane_surface/meta",
            &[self.row_width, self.public_step_count],
        );
        tr.append_u64s(
            b"rv32im/main_lane_surface/first_present",
            &[self.first_public_step.is_some() as u64],
        );
        if let Some(reference) = &self.first_public_step {
            tr.append_message(b"rv32im/main_lane_surface/first_digest", &reference.digest);
        }
        tr.append_u64s(
            b"rv32im/main_lane_surface/last_present",
            &[self.last_public_step.is_some() as u64],
        );
        if let Some(reference) = &self.last_public_step {
            tr.append_message(b"rv32im/main_lane_surface/last_digest", &reference.digest);
        }
        tr.digest32()
    }
}

pub fn build_main_lane_surface(root_lane_columns: &RootLaneColumns) -> Rv32imMainLaneSurface {
    let surface = Rv32imMainLaneSurface {
        object_digest: root_lane_columns.object.digest,
        family_digest: root_lane_columns.family_digest,
        row_width: root_lane_columns.row_width,
        public_step_count: root_lane_columns.time_len,
        first_public_step: root_lane_columns.first_row.clone(),
        last_public_step: root_lane_columns.last_row.clone(),
        digest: [0; 32],
    };
    Rv32imMainLaneSurface {
        digest: surface.expected_digest(),
        ..surface
    }
}
