//! Owns Stage 3 continuity and export summaries for the RV32IM parity slice.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::kernel::{RootExecutionBundle, Stage3ArtifactSurface, Stage3PackagedOpeningProof};
use crate::rv32im::lower::Rv32ExpandedRow;

use super::semantics::Stage3SemanticsProof;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContinuityEvent {
    pub step_index: usize,
    pub pc: u32,
    pub next_pc: u32,
    pub successor_pc: Option<u32>,
    pub final_step: bool,
    pub continuity_holds: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3Summary {
    pub continuity: Vec<ContinuityEvent>,
    pub halted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PcAdjacentBridge {
    pub continuity: Vec<ContinuityEvent>,
    pub halted: bool,
    pub continuity_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3LinkageProof {
    pub continuity_family_digest: [u8; 32],
    pub continuity_mix: u64,
    pub packaged_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage3ProofBundle {
    pub bridge: PcAdjacentBridge,
    pub semantics: Stage3SemanticsProof,
    pub linkage: Stage3LinkageProof,
    pub selected_opening: Stage3PackagedOpeningProof,
    pub digest: [u8; 32],
}

pub(crate) fn continuity_event_words(event: &ContinuityEvent) -> [u64; 6] {
    [
        event.step_index as u64,
        u64::from(event.pc),
        u64::from(event.next_pc),
        u64::from(event.successor_pc.unwrap_or(0)),
        event.final_step as u64,
        event.continuity_holds as u64,
    ]
}

pub fn continuity_event_word_width() -> usize {
    continuity_event_words(&ContinuityEvent {
        step_index: 0,
        pc: 0,
        next_pc: 0,
        successor_pc: None,
        final_step: false,
        continuity_holds: true,
    })
    .len()
}

pub(crate) fn continuity_event_digest(event: &ContinuityEvent) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_selected_continuity");
    tr.append_u64s_iter(
        b"stage3/continuity",
        8,
        std::iter::once(event.final_step as u64)
            .chain(std::iter::once(1u64))
            .chain(continuity_event_words(event).into_iter()),
    );
    tr.digest32()
}

pub fn build_stage3_summary(rows: &[Rv32ExpandedRow]) -> Stage3Summary {
    let real_rows = rows.iter().filter(|row| row.is_real).collect::<Vec<_>>();
    let mut continuity = Vec::with_capacity(real_rows.len());

    for (idx, row) in real_rows.iter().enumerate() {
        let successor_pc = real_rows.get(idx + 1).map(|next| next.pc);
        let final_step = idx + 1 == real_rows.len();
        let continuity_holds = successor_pc.is_none_or(|pc| row.next_pc == pc);
        let event = ContinuityEvent {
            step_index: row.step_index,
            pc: row.pc,
            next_pc: row.next_pc,
            successor_pc,
            final_step,
            continuity_holds,
        };
        continuity.push(event);
    }

    Stage3Summary {
        halted: real_rows.last().is_some_and(|row| row.halted),
        continuity,
    }
}

fn continuity_digest(events: &[ContinuityEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_pc_adjacent_bridge");
    tr.append_u64s(b"rv32im/stage3_pc_adjacent_bridge/len", &[events.len() as u64]);
    for event in events {
        tr.append_message(
            b"rv32im/stage3_pc_adjacent_bridge/event",
            &continuity_event_digest(event),
        );
    }
    tr.digest32()
}

impl PcAdjacentBridge {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_pc_adjacent_bridge_bundle");
        tr.append_message(
            b"rv32im/stage3_pc_adjacent_bridge_bundle/continuity_digest",
            &self.continuity_digest,
        );
        tr.append_u64s(
            b"rv32im/stage3_pc_adjacent_bridge_bundle/meta",
            &[self.continuity.len() as u64, self.halted as u64],
        );
        tr.digest32()
    }
}

impl Stage3LinkageProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_linkage_proof");
        tr.append_message(
            b"rv32im/stage3_linkage_proof/continuity_family_digest",
            &self.continuity_family_digest,
        );
        tr.append_message(b"rv32im/stage3_linkage_proof/packaged_digest", &self.packaged_digest);
        tr.append_u64s(b"rv32im/stage3_linkage_proof/meta", &[self.continuity_mix]);
        tr.digest32()
    }
}

impl Stage3ProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_proof_bundle");
        tr.append_message(b"rv32im/stage3_proof_bundle/bridge", &self.bridge.digest);
        tr.append_message(b"rv32im/stage3_proof_bundle/semantics", &self.semantics.digest);
        tr.append_message(b"rv32im/stage3_proof_bundle/linkage", &self.linkage.digest);
        tr.append_message(
            b"rv32im/stage3_proof_bundle/selected_opening",
            &self.selected_opening.digest,
        );
        tr.digest32()
    }
}

pub fn build_stage3_proof_bundle(
    summary: &Stage3Summary,
    artifact: &Stage3ArtifactSurface,
    root_execution: &RootExecutionBundle,
    stage2_temporal_digest: [u8; 32],
    initial_pc: u64,
    final_pc: u64,
    selected_opening: &Stage3PackagedOpeningProof,
) -> Stage3ProofBundle {
    let bridge = PcAdjacentBridge {
        continuity: summary.continuity.clone(),
        halted: summary.halted,
        continuity_digest: continuity_digest(&summary.continuity),
        digest: [0; 32],
    };
    let bridge = PcAdjacentBridge {
        digest: bridge.expected_digest(),
        ..bridge
    };
    let semantics = Stage3SemanticsProof::new(
        bridge.continuity_digest,
        root_execution,
        stage2_temporal_digest,
        initial_pc,
        final_pc,
        &summary.continuity,
    );
    let linkage = Stage3LinkageProof {
        continuity_family_digest: artifact.continuity.continuity_digest,
        continuity_mix: artifact.claim.continuity_mix,
        packaged_digest: selected_opening.digest,
        digest: [0; 32],
    };
    let linkage = Stage3LinkageProof {
        digest: linkage.expected_digest(),
        ..linkage
    };
    let bundle = Stage3ProofBundle {
        bridge,
        semantics,
        linkage,
        selected_opening: selected_opening.clone(),
        digest: [0; 32],
    };
    Stage3ProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}
