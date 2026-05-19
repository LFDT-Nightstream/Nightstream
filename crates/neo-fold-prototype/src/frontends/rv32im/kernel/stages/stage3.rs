//! Owns the canonical Stage 3 continuity-family root and selected-opening claims for the production RV32IM path.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::stage3::{continuity_event_digest, continuity_event_words, Stage3Summary};

use super::{
    simple_openings::{Stage3OpeningPoints, Stage3SelectedOpeningClaim},
    stage_artifacts::{first_last_selected_refs, selected_opening_object, Stage3ClaimSurface},
    AjtaiFamilyKind,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3CanonicalContinuityBundle {
    pub continuity_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Stage3CanonicalContinuityBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_canonical_continuity");
        tr.append_message(
            b"rv32im/stage3_canonical_continuity/continuity_digest",
            &self.continuity_digest,
        );
        tr.digest32()
    }
}

pub(super) fn build_stage3_artifact_parts(
    stage3: &Stage3Summary,
    continuity_mix: u64,
) -> (Stage3CanonicalContinuityBundle, Stage3ClaimSurface) {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_continuity_family");
    let mut final_step_count = 0usize;
    let mut all_continuity_hold = true;
    for event in &stage3.continuity {
        final_step_count += event.final_step as usize;
        all_continuity_hold &= event.continuity_holds;
    }
    tr.append_u64s_iter(
        b"stage3/continuity",
        stage3.continuity.len() * 6 + 2,
        std::iter::once(stage3.halted as u64)
            .chain(std::iter::once(stage3.continuity.len() as u64))
            .chain(
                stage3
                    .continuity
                    .iter()
                    .flat_map(|event| continuity_event_words(event)),
            ),
    );
    let continuity = Stage3CanonicalContinuityBundle {
        continuity_digest: tr.digest32(),
        digest: [0; 32],
    };
    let continuity = Stage3CanonicalContinuityBundle {
        digest: continuity.expected_digest(),
        ..continuity
    };
    let claim = Stage3ClaimSurface {
        continuity_count: stage3.continuity.len(),
        final_step_count,
        halted: stage3.halted,
        all_continuity_hold,
        continuity_mix,
    };
    (continuity, claim)
}

pub(super) fn build_stage3_selected_opening_claim(
    stage3: &Stage3Summary,
    claim: &Stage3ClaimSurface,
    continuity: &Stage3CanonicalContinuityBundle,
) -> Stage3SelectedOpeningClaim {
    let object = selected_opening_object(AjtaiFamilyKind::Stage3Continuity, continuity.continuity_digest);
    let (first_continuity, last_continuity) =
        first_last_selected_refs(&stage3.continuity, &object, continuity_event_digest);
    let selected = Stage3SelectedOpeningClaim {
        continuity_family_digest: continuity.continuity_digest,
        continuity_count: claim.continuity_count as u64,
        final_step_count: claim.final_step_count as u64,
        halted: claim.halted,
        all_continuity_hold: claim.all_continuity_hold,
        continuity_mix: claim.continuity_mix,
        points: Stage3OpeningPoints {
            first_continuity,
            last_continuity,
        },
        digest: [0; 32],
    };
    Stage3SelectedOpeningClaim {
        digest: selected.expected_digest(),
        ..selected
    }
}
