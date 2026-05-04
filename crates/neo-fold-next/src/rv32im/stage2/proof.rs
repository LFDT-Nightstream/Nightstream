//! Owns Stage 2 register history, RAM history, and Twist-link summaries for the RV32IM parity slice.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::isa::Rv32Opcode;
use crate::rv32im::kernel::{
    family_word, ram_access_kind_word, register_read_role_word, Stage2ArtifactSurface, Stage2PackagedOpeningProof,
};
use crate::rv32im::lower::{Rv32ExpandedRow, Rv32TraceVirtualOpcode};
use crate::rv32im::tables::Rv32FamilyTag;

use super::semantics::Stage2SemanticsProof;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegisterReadRole {
    Rs1,
    Rs2,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisterReadEvent {
    pub trace_index: usize,
    pub step_index: usize,
    pub role: RegisterReadRole,
    pub reg: u8,
    pub value: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisterWriteEvent {
    pub trace_index: usize,
    pub step_index: usize,
    pub reg: u8,
    pub previous: u32,
    pub next: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RamAccessKind {
    Read,
    Write,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RamEvent {
    pub trace_index: usize,
    pub step_index: usize,
    pub kind: RamAccessKind,
    pub addr: u32,
    pub previous: u32,
    pub next: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TwistLinkEvent {
    pub trace_index: usize,
    pub step_index: usize,
    pub family: Rv32FamilyTag,
    pub routed_write_value: Option<u32>,
    pub routed_memory_before: Option<u32>,
    pub routed_memory_after: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2Summary {
    pub register_reads: Vec<RegisterReadEvent>,
    pub register_writes: Vec<RegisterWriteEvent>,
    pub ram_events: Vec<RamEvent>,
    pub twist_links: Vec<TwistLinkEvent>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisterTwistProof {
    pub reads: Vec<RegisterReadEvent>,
    pub writes: Vec<RegisterWriteEvent>,
    pub timeline_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RamTwistProof {
    pub events: Vec<RamEvent>,
    pub timeline_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2TemporalContext {
    pub twist_links: Vec<TwistLinkEvent>,
    pub register_timeline_digest: [u8; 32],
    pub ram_timeline_digest: [u8; 32],
    pub twist_links_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2LinkageProof {
    pub register_reads_family_digest: [u8; 32],
    pub register_writes_family_digest: [u8; 32],
    pub ram_events_family_digest: [u8; 32],
    pub twist_links_family_digest: [u8; 32],
    pub reg_mix: u64,
    pub ram_mix: u64,
    pub packaged_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage2ProofBundle {
    pub register: RegisterTwistProof,
    pub ram: RamTwistProof,
    pub temporal: Stage2TemporalContext,
    pub semantics: Stage2SemanticsProof,
    pub linkage: Stage2LinkageProof,
    pub selected_opening: Stage2PackagedOpeningProof,
    pub digest: [u8; 32],
}

pub(crate) fn canonical_ram_addr(row: &Rv32ExpandedRow, addr: u32) -> u32 {
    match row.trace_opcode {
        Some(
            Rv32Opcode::Lb
            | Rv32Opcode::Lbu
            | Rv32Opcode::Lh
            | Rv32Opcode::Lhu
            | Rv32Opcode::Lw
            | Rv32Opcode::Sb
            | Rv32Opcode::Sh
            | Rv32Opcode::Sw,
        ) => addr & !0x3,
        _ => addr,
    }
}

pub(crate) fn register_read_words(event: &RegisterReadEvent) -> [u64; 5] {
    [
        event.trace_index as u64,
        event.step_index as u64,
        register_read_role_word(event.role),
        event.reg as u64,
        u64::from(event.value),
    ]
}

pub fn register_read_word_width() -> usize {
    register_read_words(&RegisterReadEvent {
        trace_index: 0,
        step_index: 0,
        role: RegisterReadRole::Rs1,
        reg: 0,
        value: 0,
    })
    .len()
}

pub(crate) fn register_write_words(event: &RegisterWriteEvent) -> [u64; 5] {
    [
        event.trace_index as u64,
        event.step_index as u64,
        event.reg as u64,
        u64::from(event.previous),
        u64::from(event.next),
    ]
}

pub fn register_write_word_width() -> usize {
    register_write_words(&RegisterWriteEvent {
        trace_index: 0,
        step_index: 0,
        reg: 0,
        previous: 0,
        next: 0,
    })
    .len()
}

pub(crate) fn ram_event_words(event: &RamEvent) -> [u64; 6] {
    [
        event.trace_index as u64,
        event.step_index as u64,
        ram_access_kind_word(event.kind),
        u64::from(event.addr),
        u64::from(event.previous),
        u64::from(event.next),
    ]
}

pub fn ram_event_word_width() -> usize {
    ram_event_words(&RamEvent {
        trace_index: 0,
        step_index: 0,
        kind: RamAccessKind::Read,
        addr: 0,
        previous: 0,
        next: 0,
    })
    .len()
}

pub(crate) fn twist_link_words(event: &TwistLinkEvent) -> [u64; 6] {
    [
        event.trace_index as u64,
        event.step_index as u64,
        family_word(event.family),
        u64::from(event.routed_write_value.unwrap_or(0)),
        u64::from(event.routed_memory_before.unwrap_or(0)),
        u64::from(event.routed_memory_after.unwrap_or(0)),
    ]
}

pub fn twist_link_word_width() -> usize {
    twist_link_words(&TwistLinkEvent {
        trace_index: 0,
        step_index: 0,
        family: Rv32FamilyTag::NativeAlu,
        routed_write_value: None,
        routed_memory_before: None,
        routed_memory_after: None,
    })
    .len()
}

pub(crate) fn register_read_timeline_words(event: &RegisterReadEvent) -> [u64; 9] {
    let words = register_read_words(event);
    [1u64, words[0], words[1], words[2], words[3], words[4], 0u64, 0u64, 0u64]
}

pub(crate) fn register_write_timeline_words(event: &RegisterWriteEvent) -> [u64; 9] {
    let words = register_write_words(event);
    [0u64, 1u64, words[0], words[1], words[2], words[3], words[4], 0u64, 0u64]
}

pub(crate) fn ram_timeline_words(event: &RamEvent) -> [u64; 10] {
    let words = ram_event_words(event);
    [
        0u64, 0u64, 1u64, words[0], words[1], words[2], words[3], words[4], words[5], 0u64,
    ]
}

pub(crate) fn twist_link_timeline_words(event: &TwistLinkEvent) -> [u64; 10] {
    let words = twist_link_words(event);
    [
        0u64, 0u64, 0u64, 1u64, words[0], words[1], words[2], words[3], words[4], words[5],
    ]
}

pub(crate) fn register_read_event_digest(event: &RegisterReadEvent) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_selected_register_read");
    tr.append_u64s(b"stage2/read", &register_read_timeline_words(event));
    tr.digest32()
}

pub(crate) fn register_write_event_digest(event: &RegisterWriteEvent) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_selected_register_write");
    tr.append_u64s(b"stage2/write", &register_write_timeline_words(event));
    tr.digest32()
}

pub(crate) fn ram_event_digest(event: &RamEvent) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_selected_ram_event");
    tr.append_u64s(b"stage2/ram", &ram_timeline_words(event));
    tr.digest32()
}

pub(crate) fn twist_link_event_digest(event: &TwistLinkEvent) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_selected_twist_link");
    tr.append_u64s(b"stage2/twist", &twist_link_timeline_words(event));
    tr.digest32()
}

pub(super) fn row_reads_rs1(row: &Rv32ExpandedRow) -> bool {
    matches!(
        row.trace_opcode,
        Some(
            Rv32Opcode::Addi
                | Rv32Opcode::Add
                | Rv32Opcode::Sub
                | Rv32Opcode::Andi
                | Rv32Opcode::And
                | Rv32Opcode::Ori
                | Rv32Opcode::Or
                | Rv32Opcode::Xori
                | Rv32Opcode::Xor
                | Rv32Opcode::Slti
                | Rv32Opcode::Slt
                | Rv32Opcode::Sltiu
                | Rv32Opcode::Sltu
                | Rv32Opcode::Slli
                | Rv32Opcode::Sll
                | Rv32Opcode::Srli
                | Rv32Opcode::Srl
                | Rv32Opcode::Srai
                | Rv32Opcode::Sra
                | Rv32Opcode::Mul
                | Rv32Opcode::Mulh
                | Rv32Opcode::Mulhsu
                | Rv32Opcode::Mulhu
                | Rv32Opcode::Div
                | Rv32Opcode::Divu
                | Rv32Opcode::Rem
                | Rv32Opcode::Remu
                | Rv32Opcode::Lb
                | Rv32Opcode::Lbu
                | Rv32Opcode::Lh
                | Rv32Opcode::Lhu
                | Rv32Opcode::Lw
                | Rv32Opcode::Sb
                | Rv32Opcode::Sh
                | Rv32Opcode::Sw
                | Rv32Opcode::Jalr
                | Rv32Opcode::Beq
                | Rv32Opcode::Bne
                | Rv32Opcode::Blt
                | Rv32Opcode::Bge
                | Rv32Opcode::Bltu
                | Rv32Opcode::Bgeu
        )
    ) || row.trace_virtual_opcode.is_some()
}

pub(super) fn row_reads_rs2(row: &Rv32ExpandedRow) -> bool {
    matches!(
        row.trace_opcode,
        Some(
            Rv32Opcode::Add
                | Rv32Opcode::Sub
                | Rv32Opcode::And
                | Rv32Opcode::Or
                | Rv32Opcode::Xor
                | Rv32Opcode::Slt
                | Rv32Opcode::Sltu
                | Rv32Opcode::Sll
                | Rv32Opcode::Srl
                | Rv32Opcode::Sra
                | Rv32Opcode::Mul
                | Rv32Opcode::Mulh
                | Rv32Opcode::Mulhsu
                | Rv32Opcode::Mulhu
                | Rv32Opcode::Div
                | Rv32Opcode::Divu
                | Rv32Opcode::Rem
                | Rv32Opcode::Remu
                | Rv32Opcode::Sb
                | Rv32Opcode::Sh
                | Rv32Opcode::Sw
                | Rv32Opcode::Beq
                | Rv32Opcode::Bne
                | Rv32Opcode::Blt
                | Rv32Opcode::Bge
                | Rv32Opcode::Bltu
                | Rv32Opcode::Bgeu
        )
    ) || matches!(
        row.trace_virtual_opcode,
        Some(
            Rv32TraceVirtualOpcode::Advice
                | Rv32TraceVirtualOpcode::ChangeDivisor
                | Rv32TraceVirtualOpcode::AssertValidDiv0
                | Rv32TraceVirtualOpcode::AssertMulNoOverflow
                | Rv32TraceVirtualOpcode::AssertLte
                | Rv32TraceVirtualOpcode::AssertValidUnsignedRemainder
                | Rv32TraceVirtualOpcode::AssertSignedDivIdentity
                | Rv32TraceVirtualOpcode::AssertSignedRemainderBounds
        )
    )
}

pub fn build_stage2_summary(rows: &[Rv32ExpandedRow]) -> Stage2Summary {
    let mut register_reads = Vec::new();
    let mut register_writes = Vec::new();
    let mut ram_events = Vec::new();
    let mut twist_links = Vec::new();

    for row in rows {
        if row_reads_rs1(row) {
            let event = RegisterReadEvent {
                trace_index: row.trace_index,
                step_index: row.step_index,
                role: RegisterReadRole::Rs1,
                reg: row.rs1,
                value: row.rs1_value,
            };
            register_reads.push(event);
        }
        if row_reads_rs2(row) {
            let event = RegisterReadEvent {
                trace_index: row.trace_index,
                step_index: row.step_index,
                role: RegisterReadRole::Rs2,
                reg: row.rs2,
                value: row.rs2_value,
            };
            register_reads.push(event);
        }

        if row.writes_rd {
            let event = RegisterWriteEvent {
                trace_index: row.trace_index,
                step_index: row.step_index,
                reg: row.rd,
                previous: row.rd_before,
                next: row.rd_after,
            };
            register_writes.push(event);
        }

        if let Some(addr) = row.effective_addr {
            if let Some(before) = row.memory_before {
                let next = row.memory_after.unwrap_or(before);
                let kind = if row.writes_ram {
                    RamAccessKind::Write
                } else {
                    RamAccessKind::Read
                };
                let event = RamEvent {
                    trace_index: row.trace_index,
                    step_index: row.step_index,
                    kind,
                    addr: canonical_ram_addr(row, addr),
                    previous: before,
                    next,
                };
                ram_events.push(event);
            }
        }

        let twist = TwistLinkEvent {
            trace_index: row.trace_index,
            step_index: row.step_index,
            family: row.family,
            routed_write_value: row.writes_rd.then_some(row.rd_after),
            routed_memory_before: row.memory_before,
            routed_memory_after: row.memory_after,
        };
        twist_links.push(twist);
    }

    Stage2Summary {
        register_reads,
        register_writes,
        ram_events,
        twist_links,
    }
}

impl RegisterTwistProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_twist_proof");
        tr.append_message(
            b"rv32im/stage2_register_twist_proof/timeline_digest",
            &self.timeline_digest,
        );
        tr.append_u64s(
            b"rv32im/stage2_register_twist_proof/meta",
            &[self.reads.len() as u64, self.writes.len() as u64],
        );
        tr.digest32()
    }
}

impl RamTwistProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_ram_twist_proof");
        tr.append_message(b"rv32im/stage2_ram_twist_proof/timeline_digest", &self.timeline_digest);
        tr.append_u64s(b"rv32im/stage2_ram_twist_proof/meta", &[self.events.len() as u64]);
        tr.digest32()
    }
}

impl Stage2TemporalContext {
    pub(crate) fn expected_digest_from_parts(
        register_timeline_digest: [u8; 32],
        ram_timeline_digest: [u8; 32],
        twist_links_digest: [u8; 32],
        twist_link_count: usize,
    ) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_temporal_context");
        tr.append_message(
            b"rv32im/stage2_temporal_context/register_timeline_digest",
            &register_timeline_digest,
        );
        tr.append_message(
            b"rv32im/stage2_temporal_context/ram_timeline_digest",
            &ram_timeline_digest,
        );
        tr.append_message(
            b"rv32im/stage2_temporal_context/twist_links_digest",
            &twist_links_digest,
        );
        tr.append_u64s(b"rv32im/stage2_temporal_context/meta", &[twist_link_count as u64]);
        tr.digest32()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        Self::expected_digest_from_parts(
            self.register_timeline_digest,
            self.ram_timeline_digest,
            self.twist_links_digest,
            self.twist_links.len(),
        )
    }
}

impl Stage2LinkageProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_linkage_proof");
        tr.append_message(
            b"rv32im/stage2_linkage_proof/register_reads_family_digest",
            &self.register_reads_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_linkage_proof/register_writes_family_digest",
            &self.register_writes_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_linkage_proof/ram_events_family_digest",
            &self.ram_events_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_linkage_proof/twist_links_family_digest",
            &self.twist_links_family_digest,
        );
        tr.append_message(b"rv32im/stage2_linkage_proof/packaged_digest", &self.packaged_digest);
        tr.append_u64s(b"rv32im/stage2_linkage_proof/meta", &[self.reg_mix, self.ram_mix]);
        tr.digest32()
    }
}

impl Stage2ProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_proof_bundle");
        tr.append_message(b"rv32im/stage2_proof_bundle/register", &self.register.digest);
        tr.append_message(b"rv32im/stage2_proof_bundle/ram", &self.ram.digest);
        tr.append_message(b"rv32im/stage2_proof_bundle/temporal", &self.temporal.digest);
        tr.append_message(b"rv32im/stage2_proof_bundle/semantics", &self.semantics.digest);
        tr.append_message(b"rv32im/stage2_proof_bundle/linkage", &self.linkage.digest);
        tr.append_message(
            b"rv32im/stage2_proof_bundle/selected_opening",
            &self.selected_opening.digest,
        );
        tr.digest32()
    }
}

pub(crate) fn register_timeline_digest(reads: &[RegisterReadEvent], writes: &[RegisterWriteEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_timeline");
    tr.append_u64s(b"meta", &[(reads.len() + writes.len()) as u64]);
    tr.append_u64s_iter(
        b"entries",
        reads.len() * 9 + writes.len() * 9,
        reads
            .iter()
            .flat_map(register_read_timeline_words)
            .chain(writes.iter().flat_map(register_write_timeline_words)),
    );
    tr.digest32()
}

pub(crate) fn ram_timeline_digest(events: &[RamEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_ram_timeline");
    tr.append_u64s(b"meta", &[events.len() as u64]);
    tr.append_u64s_iter(
        b"entries",
        events.len() * 10,
        events.iter().flat_map(ram_timeline_words),
    );
    tr.digest32()
}

pub(crate) fn twist_links_timeline_digest(events: &[TwistLinkEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_twist_links");
    tr.append_u64s(b"meta", &[events.len() as u64]);
    tr.append_u64s_iter(
        b"entries",
        events.len() * 10,
        events.iter().flat_map(twist_link_timeline_words),
    );
    tr.digest32()
}

pub fn build_stage2_proof_bundle(
    summary: &Stage2Summary,
    artifact: &Stage2ArtifactSurface,
    selected_opening: &Stage2PackagedOpeningProof,
) -> Stage2ProofBundle {
    let register_timeline_digest = register_timeline_digest(&summary.register_reads, &summary.register_writes);
    let ram_timeline_digest = ram_timeline_digest(&summary.ram_events);
    let twist_links_digest = twist_links_timeline_digest(&summary.twist_links);

    let register = RegisterTwistProof {
        reads: summary.register_reads.clone(),
        writes: summary.register_writes.clone(),
        timeline_digest: register_timeline_digest,
        digest: [0; 32],
    };
    let register = RegisterTwistProof {
        digest: register.expected_digest(),
        ..register
    };
    let ram = RamTwistProof {
        events: summary.ram_events.clone(),
        timeline_digest: ram_timeline_digest,
        digest: [0; 32],
    };
    let ram = RamTwistProof {
        digest: ram.expected_digest(),
        ..ram
    };
    let temporal = Stage2TemporalContext {
        twist_links: summary.twist_links.clone(),
        register_timeline_digest,
        ram_timeline_digest,
        twist_links_digest,
        digest: [0; 32],
    };
    let temporal = Stage2TemporalContext {
        digest: temporal.expected_digest(),
        ..temporal
    };
    let semantics = Stage2SemanticsProof::from_surface_digests(
        artifact.families.register_reads_digest,
        artifact.families.register_writes_digest,
        artifact.families.ram_events_digest,
        artifact.families.twist_links_digest,
        summary,
    );
    let linkage = Stage2LinkageProof {
        register_reads_family_digest: artifact.families.register_reads_digest,
        register_writes_family_digest: artifact.families.register_writes_digest,
        ram_events_family_digest: artifact.families.ram_events_digest,
        twist_links_family_digest: artifact.families.twist_links_digest,
        reg_mix: artifact.claim.reg_mix,
        ram_mix: artifact.claim.ram_mix,
        packaged_digest: selected_opening.digest,
        digest: [0; 32],
    };
    let linkage = Stage2LinkageProof {
        digest: linkage.expected_digest(),
        ..linkage
    };
    let bundle = Stage2ProofBundle {
        register,
        ram,
        temporal,
        semantics,
        linkage,
        selected_opening: selected_opening.clone(),
        digest: [0; 32],
    };
    Stage2ProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}
