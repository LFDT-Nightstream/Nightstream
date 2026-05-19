//! Owns deterministic Stage-2 register and RAM timeline semantics for accepted-proof verification.

use std::collections::BTreeMap;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::isa::MemoryWord;
use crate::rv32im::layout::RV32_REGISTER_COUNT;
use crate::rv32im::lower::Rv32ExpandedRow;

use super::proof::{
    canonical_ram_addr, ram_event_words, register_read_words, register_write_words, row_reads_rs1, row_reads_rs2,
    twist_link_words, RamAccessKind, RamEvent, RegisterReadEvent, RegisterReadRole, RegisterWriteEvent, Stage2Summary,
    TwistLinkEvent,
};

const REGISTER_TIMELINE_LEN: usize = 128;
const REGISTER_SINK: u8 = 127;
const TEMP_REG_START: u8 = 40;
const TEMP_REG_END: u8 = 47;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2SemanticsProof {
    pub register_reads_family_digest: [u8; 32],
    pub register_writes_family_digest: [u8; 32],
    pub ram_events_family_digest: [u8; 32],
    pub twist_links_family_digest: [u8; 32],
    pub row_count: u64,
    pub register_event_count: u64,
    pub ram_event_count: u64,
    pub digest: [u8; 32],
}

impl Stage2SemanticsProof {
    pub(crate) fn new(summary: &Stage2Summary) -> Self {
        Self::from_surface_digests(
            register_reads_family_digest(&summary.register_reads),
            register_writes_family_digest(&summary.register_writes),
            ram_events_family_digest(&summary.ram_events),
            twist_links_family_digest(&summary.twist_links),
            summary,
        )
    }

    pub(crate) fn from_surface_digests(
        register_reads_family_digest: [u8; 32],
        register_writes_family_digest: [u8; 32],
        ram_events_family_digest: [u8; 32],
        twist_links_family_digest: [u8; 32],
        summary: &Stage2Summary,
    ) -> Self {
        let proof = Self {
            register_reads_family_digest,
            register_writes_family_digest,
            ram_events_family_digest,
            twist_links_family_digest,
            row_count: summary.twist_links.len() as u64,
            register_event_count: (summary.register_reads.len() + summary.register_writes.len()) as u64,
            ram_event_count: summary.ram_events.len() as u64,
            digest: [0; 32],
        };
        Self {
            digest: proof.expected_digest(),
            ..proof
        }
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_semantics_proof");
        tr.append_message(
            b"rv32im/stage2_semantics_proof/register_reads_family_digest",
            &self.register_reads_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_semantics_proof/register_writes_family_digest",
            &self.register_writes_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_semantics_proof/ram_events_family_digest",
            &self.ram_events_family_digest,
        );
        tr.append_message(
            b"rv32im/stage2_semantics_proof/twist_links_family_digest",
            &self.twist_links_family_digest,
        );
        tr.append_u64s(
            b"rv32im/stage2_semantics_proof/meta",
            &[self.row_count, self.register_event_count, self.ram_event_count],
        );
        tr.digest32()
    }
}

pub(crate) fn register_reads_family_digest(events: &[RegisterReadEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_reads_family");
    tr.append_u64s_iter(
        b"stage2/register_reads",
        events.len() * 5 + 4,
        std::iter::once(events.len() as u64)
            .chain(events.iter().flat_map(register_read_words))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64)),
    );
    tr.digest32()
}

pub(crate) fn register_writes_family_digest(events: &[RegisterWriteEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_register_writes_family");
    tr.append_u64s_iter(
        b"stage2/register_writes",
        events.len() * 5 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(events.len() as u64))
            .chain(events.iter().flat_map(register_write_words))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64)),
    );
    tr.digest32()
}

pub(crate) fn ram_events_family_digest(events: &[RamEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_ram_events_family");
    tr.append_u64s_iter(
        b"stage2/ram_events",
        events.len() * 6 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(events.len() as u64))
            .chain(events.iter().flat_map(ram_event_words))
            .chain(std::iter::once(0u64)),
    );
    tr.digest32()
}

pub(crate) fn twist_links_family_digest(events: &[TwistLinkEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage2_twist_links_family");
    tr.append_u64s_iter(
        b"stage2/twist_links",
        events.len() * 6 + 4,
        std::iter::once(0u64)
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(0u64))
            .chain(std::iter::once(events.len() as u64))
            .chain(events.iter().flat_map(twist_link_words)),
    );
    tr.digest32()
}

pub fn verify_stage2_semantics_from_events(
    rows: &[Rv32ExpandedRow],
    register_reads: &[RegisterReadEvent],
    register_writes: &[RegisterWriteEvent],
    ram_events: &[RamEvent],
    twist_links: &[TwistLinkEvent],
    initial_registers: &[u32; RV32_REGISTER_COUNT],
    initial_memory: &[MemoryWord],
) -> Result<(), String> {
    if rows.len() != twist_links.len() {
        return Err("stage2 twist-link row count mismatch".into());
    }

    let mut registers = [None; REGISTER_TIMELINE_LEN];
    for (idx, value) in initial_registers.iter().enumerate() {
        registers[idx] = Some(*value);
    }
    registers[0] = Some(0);

    let mut memory = BTreeMap::new();
    for word in initial_memory {
        memory.insert(word.addr, word.value);
    }

    let mut read_index = 0usize;
    let mut write_index = 0usize;
    let mut ram_index = 0usize;
    let mut current_step = None;
    let mut temp_written_this_step = [false; (TEMP_REG_END - TEMP_REG_START + 1) as usize];

    for (row_index, row) in rows.iter().enumerate() {
        if current_step != Some(row.step_index) {
            current_step = Some(row.step_index);
            temp_written_this_step.fill(false);
            for reg in TEMP_REG_START..=TEMP_REG_END {
                registers[reg as usize] = None;
            }
        }

        let twist = &twist_links[row_index];
        verify_twist_link(row, twist)?;

        if row_reads_rs1(row) {
            let event = register_reads
                .get(read_index)
                .ok_or_else(|| format!("stage2 missing rs1 read event for trace index {}", row.trace_index))?;
            verify_register_read(
                row,
                event,
                RegisterReadRole::Rs1,
                row.rs1,
                row.rs1_value,
                &mut registers,
                &temp_written_this_step,
            )?;
            read_index += 1;
        }

        if row_reads_rs2(row) {
            let event = register_reads
                .get(read_index)
                .ok_or_else(|| format!("stage2 missing rs2 read event for trace index {}", row.trace_index))?;
            verify_register_read(
                row,
                event,
                RegisterReadRole::Rs2,
                row.rs2,
                row.rs2_value,
                &mut registers,
                &temp_written_this_step,
            )?;
            read_index += 1;
        }

        if let Some(addr) = row.effective_addr {
            let event = ram_events
                .get(ram_index)
                .ok_or_else(|| format!("stage2 missing RAM event for trace index {}", row.trace_index))?;
            verify_ram_event(row, event, canonical_ram_addr(row, addr), &mut memory)?;
            ram_index += 1;
        } else if row.memory_before.is_some() || row.memory_after.is_some() {
            return Err(format!(
                "stage2 row {} carries RAM values without an effective address",
                row.trace_index
            ));
        }

        if row.writes_rd {
            let event = register_writes
                .get(write_index)
                .ok_or_else(|| format!("stage2 missing register write for trace index {}", row.trace_index))?;
            verify_register_write(row, event, &mut registers, &mut temp_written_this_step)?;
            write_index += 1;
        }
    }

    if read_index != register_reads.len() || write_index != register_writes.len() || ram_index != ram_events.len() {
        return Err("stage2 event cursors did not consume the full summary".into());
    }

    Ok(())
}

fn verify_twist_link(row: &Rv32ExpandedRow, twist: &TwistLinkEvent) -> Result<(), String> {
    if twist.trace_index != row.trace_index
        || twist.step_index != row.step_index
        || twist.family != row.family
        || twist.routed_write_value != row.writes_rd.then_some(row.rd_after)
        || twist.routed_memory_before != row.memory_before
        || twist.routed_memory_after != row.memory_after
    {
        return Err(format!("stage2 twist-link mismatch at trace index {}", row.trace_index));
    }
    Ok(())
}

fn verify_register_read(
    row: &Rv32ExpandedRow,
    event: &RegisterReadEvent,
    expected_role: RegisterReadRole,
    expected_reg: u8,
    expected_value: u32,
    registers: &mut [Option<u32>; REGISTER_TIMELINE_LEN],
    temp_written_this_step: &[bool; (TEMP_REG_END - TEMP_REG_START + 1) as usize],
) -> Result<(), String> {
    if event.trace_index != row.trace_index
        || event.step_index != row.step_index
        || event.role != expected_role
        || event.reg != expected_reg
        || event.value != expected_value
    {
        return Err(format!(
            "stage2 register read surface mismatch at trace index {}",
            row.trace_index
        ));
    }
    if expected_reg == REGISTER_SINK {
        return Err(format!(
            "stage2 register read used sink register at trace index {}",
            row.trace_index
        ));
    }
    if expected_reg == 0 {
        if expected_value != 0 {
            return Err(format!(
                "stage2 x0 read carried non-zero value at trace index {}",
                row.trace_index
            ));
        }
        return Ok(());
    }
    if is_temporary_virtual_register(expected_reg) && !temp_written_this_step[temp_reg_slot(expected_reg)] {
        return Err(format!(
            "stage2 temporary register x{} was read before a write in step {}",
            expected_reg, row.step_index
        ));
    }
    let slot = &mut registers[expected_reg as usize];
    match *slot {
        Some(current) if current != expected_value => Err(format!(
            "stage2 register history mismatch for x{} at trace index {}",
            expected_reg, row.trace_index
        )),
        Some(_) => Ok(()),
        None => {
            *slot = Some(expected_value);
            Ok(())
        }
    }
}

fn verify_register_write(
    row: &Rv32ExpandedRow,
    event: &RegisterWriteEvent,
    registers: &mut [Option<u32>; REGISTER_TIMELINE_LEN],
    temp_written_this_step: &mut [bool; (TEMP_REG_END - TEMP_REG_START + 1) as usize],
) -> Result<(), String> {
    if event.trace_index != row.trace_index
        || event.step_index != row.step_index
        || event.reg != row.rd
        || event.previous != row.rd_before
        || event.next != row.rd_after
    {
        return Err(format!(
            "stage2 register write surface mismatch at trace index {}",
            row.trace_index
        ));
    }
    if event.reg == 0 || event.reg == REGISTER_SINK {
        return Err(format!(
            "stage2 register write targeted a sinked register at trace index {}",
            row.trace_index
        ));
    }
    if is_temporary_virtual_register(event.reg)
        && !temp_written_this_step[temp_reg_slot(event.reg)]
        && event.previous != 0
    {
        return Err(format!(
            "stage2 temporary register x{} must start each step from zero at trace index {}",
            event.reg, row.trace_index
        ));
    }
    let slot = &mut registers[event.reg as usize];
    match *slot {
        Some(current) if current != event.previous => {
            return Err(format!(
                "stage2 register history mismatch for x{} at trace index {}",
                event.reg, row.trace_index
            ))
        }
        None => *slot = Some(event.previous),
        Some(_) => {}
    }
    *slot = Some(event.next);
    if is_temporary_virtual_register(event.reg) {
        temp_written_this_step[temp_reg_slot(event.reg)] = true;
    }
    Ok(())
}

fn verify_ram_event(
    row: &Rv32ExpandedRow,
    event: &RamEvent,
    addr: u32,
    memory: &mut BTreeMap<u32, u32>,
) -> Result<(), String> {
    let current = memory.get(&addr).copied().unwrap_or(0);
    let expected_kind = if row.writes_ram {
        RamAccessKind::Write
    } else {
        RamAccessKind::Read
    };
    let row_before = row
        .memory_before
        .ok_or_else(|| format!("stage2 memory row {} is missing memory_before", row.trace_index))?;
    if event.trace_index != row.trace_index
        || event.step_index != row.step_index
        || event.kind != expected_kind
        || event.addr != addr
        || event.previous != row_before
    {
        return Err(format!(
            "stage2 RAM event surface mismatch at trace index {}",
            row.trace_index
        ));
    }
    if current != event.previous {
        return Err(format!(
            "stage2 RAM history mismatch at trace index {} for address 0x{addr:016x}",
            row.trace_index
        ));
    }

    if row.writes_ram {
        let row_after = row
            .memory_after
            .ok_or_else(|| format!("stage2 store row {} is missing memory_after", row.trace_index))?;
        if event.next != row_after {
            return Err(format!(
                "stage2 RAM write-next mismatch at trace index {}",
                row.trace_index
            ));
        }
        memory.insert(addr, event.next);
    } else {
        if let Some(row_after) = row.memory_after {
            if row_after != current {
                return Err(format!(
                    "stage2 RAM read carried a changing after-value at trace index {}",
                    row.trace_index
                ));
            }
        }
        if event.next != current {
            return Err(format!(
                "stage2 RAM read-next mismatch at trace index {}",
                row.trace_index
            ));
        }
    }
    Ok(())
}

fn is_temporary_virtual_register(reg: u8) -> bool {
    (TEMP_REG_START..=TEMP_REG_END).contains(&reg)
}

fn temp_reg_slot(reg: u8) -> usize {
    (reg - TEMP_REG_START) as usize
}
