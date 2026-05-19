//! Owns canonical Stage 2 family roots and selected-opening claims for the production RV32IM path.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::stage2::{
    ram_event_digest, ram_event_words, register_read_event_digest, register_read_words, register_write_event_digest,
    register_write_words, twist_link_event_digest, twist_link_words, RamAccessKind, RamEvent, RegisterReadEvent,
    RegisterWriteEvent, Stage2Summary, TwistLinkEvent,
};

use super::{
    simple_openings::{Stage2OpeningPoints, Stage2SelectedOpeningClaim},
    stage_artifacts::{first_last_selected_refs, selected_opening_object, Stage2ClaimSurface},
    AjtaiFamilyKind,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2CanonicalFamilyBundle {
    pub register_reads_digest: [u8; 32],
    pub register_writes_digest: [u8; 32],
    pub ram_events_digest: [u8; 32],
    pub twist_links_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Stage2CanonicalFamilyBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_canonical_families");
        tr.append_message(
            b"rv32im/stage2_canonical_families/register_reads",
            &self.register_reads_digest,
        );
        tr.append_message(
            b"rv32im/stage2_canonical_families/register_writes",
            &self.register_writes_digest,
        );
        tr.append_message(b"rv32im/stage2_canonical_families/ram_events", &self.ram_events_digest);
        tr.append_message(
            b"rv32im/stage2_canonical_families/twist_links",
            &self.twist_links_digest,
        );
        tr.digest32()
    }
}

pub(super) fn build_stage2_artifact_parts(
    stage2: &Stage2Summary,
    reg_mix: u64,
    ram_mix: u64,
) -> (Stage2CanonicalFamilyBundle, Stage2ClaimSurface) {
    let mut reads_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_reads_family");
    reads_tr.append_u64s_iter(
        b"stage2/register_reads",
        stage2.register_reads.len() * 5 + 4,
        std::iter::once(stage2.register_reads.len() as u64)
            .chain(
                stage2
                    .register_reads
                    .iter()
                    .flat_map(|event| register_read_words(event)),
            )
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64)),
    );
    let mut writes_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_writes_family");
    writes_tr.append_u64s_iter(
        b"stage2/register_writes",
        stage2.register_writes.len() * 5 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(stage2.register_writes.len() as u64))
            .chain(
                stage2
                    .register_writes
                    .iter()
                    .flat_map(|event| register_write_words(event)),
            )
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64)),
    );
    let mut ram_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_ram_events_family");
    let mut ram_read_count = 0usize;
    let mut ram_write_count = 0usize;
    for event in &stage2.ram_events {
        match event.kind {
            RamAccessKind::Read => ram_read_count += 1,
            RamAccessKind::Write => ram_write_count += 1,
        }
    }
    ram_tr.append_u64s_iter(
        b"stage2/ram_events",
        stage2.ram_events.len() * 6 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(stage2.ram_events.len() as u64))
            .chain(
                stage2
                    .ram_events
                    .iter()
                    .flat_map(|event| ram_event_words(event)),
            )
            .chain(std::iter::once(0u64)),
    );
    let mut twist_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_twist_links_family");
    twist_tr.append_u64s_iter(
        b"stage2/twist_links",
        stage2.twist_links.len() * 6 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(stage2.twist_links.len() as u64))
            .chain(
                stage2
                    .twist_links
                    .iter()
                    .flat_map(|event| twist_link_words(event)),
            ),
    );
    let families = Stage2CanonicalFamilyBundle {
        register_reads_digest: reads_tr.digest32(),
        register_writes_digest: writes_tr.digest32(),
        ram_events_digest: ram_tr.digest32(),
        twist_links_digest: twist_tr.digest32(),
        digest: [0; 32],
    };
    let families = Stage2CanonicalFamilyBundle {
        digest: families.expected_digest(),
        ..families
    };
    let claim = Stage2ClaimSurface {
        register_read_count: stage2.register_reads.len(),
        register_write_count: stage2.register_writes.len(),
        ram_event_count: stage2.ram_events.len(),
        twist_link_count: stage2.twist_links.len(),
        ram_read_count,
        ram_write_count,
        reg_mix,
        ram_mix,
    };
    (families, claim)
}

pub(super) fn build_stage2_selected_opening_claim(
    stage2: &Stage2Summary,
    claim: &Stage2ClaimSurface,
    families: &Stage2CanonicalFamilyBundle,
) -> Stage2SelectedOpeningClaim {
    build_stage2_selected_opening_claim_from_events(
        &stage2.register_reads,
        &stage2.register_writes,
        &stage2.ram_events,
        &stage2.twist_links,
        claim,
        families,
    )
}

pub(super) fn build_stage2_selected_opening_claim_from_events(
    register_reads: &[RegisterReadEvent],
    register_writes: &[RegisterWriteEvent],
    ram_events: &[RamEvent],
    twist_links: &[TwistLinkEvent],
    claim: &Stage2ClaimSurface,
    families: &Stage2CanonicalFamilyBundle,
) -> Stage2SelectedOpeningClaim {
    let read_object = selected_opening_object(AjtaiFamilyKind::Stage2RegisterReads, families.register_reads_digest);
    let write_object = selected_opening_object(AjtaiFamilyKind::Stage2RegisterWrites, families.register_writes_digest);
    let ram_object = selected_opening_object(AjtaiFamilyKind::Stage2RamEvents, families.ram_events_digest);
    let twist_object = selected_opening_object(AjtaiFamilyKind::Stage2TwistLinks, families.twist_links_digest);
    let (first_read, last_read) = first_last_selected_refs(register_reads, &read_object, register_read_event_digest);
    let (first_write, last_write) =
        first_last_selected_refs(register_writes, &write_object, register_write_event_digest);
    let (first_ram, last_ram) = first_last_selected_refs(ram_events, &ram_object, ram_event_digest);
    let (first_twist, last_twist) = first_last_selected_refs(twist_links, &twist_object, twist_link_event_digest);
    let selected = Stage2SelectedOpeningClaim {
        register_reads_family_digest: families.register_reads_digest,
        register_writes_family_digest: families.register_writes_digest,
        ram_events_family_digest: families.ram_events_digest,
        twist_links_family_digest: families.twist_links_digest,
        register_read_count: claim.register_read_count as u64,
        register_write_count: claim.register_write_count as u64,
        ram_event_count: claim.ram_event_count as u64,
        twist_link_count: claim.twist_link_count as u64,
        ram_read_count: claim.ram_read_count as u64,
        ram_write_count: claim.ram_write_count as u64,
        reg_mix: claim.reg_mix,
        ram_mix: claim.ram_mix,
        points: Stage2OpeningPoints {
            first_read,
            last_read,
            first_write,
            last_write,
            first_ram,
            last_ram,
            first_twist,
            last_twist,
        },
        digest: [0; 32],
    };
    Stage2SelectedOpeningClaim {
        digest: selected.expected_digest(),
        ..selected
    }
}
