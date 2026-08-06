//! Temporal linear-memory state used while expanding grammar rows.
//!
//! Owns the sparse memory-zero image seen by normalization. Program rows and
//! grammar writes advance it in trace order; grammar reads are derived from it.

use super::NormalizedStep;
use crate::adapters::wasmtime::WasmProgramTables;
use crate::event_grammar::{HostEventGrammar, SlotSource};
use crate::ir::{LinearMemoryAccess, LinearMemoryWordLane, WasmBuildError};
use crate::isa::WasmMemoryAccessKind;
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct LinearMemoryImage {
    words: BTreeMap<u64, u32>,
}

impl LinearMemoryImage {
    pub(super) fn for_grammar(
        grammar: &HostEventGrammar,
        program: &WasmProgramTables,
    ) -> Result<Option<Self>, WasmBuildError> {
        if !grammar_uses_linear_memory(grammar) {
            return Ok(None);
        }
        if program.has_imported_memory {
            return Err(WasmBuildError::Unsupported(
                "grammar memory slots require verifier-known memory-zero initialization; imported memory is unsupported"
                    .to_string(),
            ));
        }
        Ok(Some(Self::from_program(program)))
    }

    pub(super) fn from_program(tables: &WasmProgramTables) -> Self {
        let mut memory = Self::default();
        for &(byte_addr, byte_value) in &tables.linear_memory_init {
            let word_addr = byte_addr / 4;
            let shift = ((byte_addr % 4) * 8) as u32;
            let word = memory.words.entry(word_addr).or_insert(0);
            *word &= !(0xff << shift);
            *word |= u32::from(byte_value) << shift;
        }
        memory
    }

    pub(super) fn read_aligned_word(&self, base: u32, byte_offset: u32) -> Result<u32, WasmBuildError> {
        let word_addr = aligned_word_addr(base, byte_offset)?;
        Ok(self.read_word(word_addr))
    }

    pub(super) fn write_aligned_word(&mut self, base: u32, byte_offset: u32, value: u32) -> Result<(), WasmBuildError> {
        let word_addr = aligned_word_addr(base, byte_offset)?;
        self.write_word(word_addr, value);
        Ok(())
    }

    pub(super) fn apply_program_access(&mut self, step: &NormalizedStep) -> Result<(), WasmBuildError> {
        let Some(access) = step.linear_memory.as_ref() else {
            return Ok(());
        };

        let kind = step
            .opcode
            .memory_access_info()
            .expect("normalized linear-memory access has opcode metadata")
            .kind;

        self.apply_lane(step, kind, access.lane0)?;

        if let Some(lane) = access.lane1 {
            self.apply_lane(step, kind, lane)?;
        }

        if let Some(lane) = access.lane2 {
            self.apply_lane(step, kind, lane)?;
        }

        Ok(())
    }

    fn apply_lane(
        &mut self,
        step: &NormalizedStep,
        kind: WasmMemoryAccessKind,
        lane: LinearMemoryWordLane,
    ) -> Result<(), WasmBuildError> {
        let current = self.read_word(lane.word_addr);
        if lane.value_before != current {
            return Err(WasmBuildError::Trace(format!(
                "linear-memory replay mismatch at cycle {} word {}: trace observed {}, replay has {}",
                step.cycle, lane.word_addr, lane.value_before, current
            )));
        }
        match kind {
            WasmMemoryAccessKind::Load => {
                if lane.value_after != current {
                    return Err(WasmBuildError::Trace(format!(
                        "linear-memory load at cycle {} changes word {} from {} to {}",
                        step.cycle, lane.word_addr, current, lane.value_after
                    )));
                }
            }
            WasmMemoryAccessKind::Store => self.write_word(lane.word_addr, lane.value_after),
        }
        Ok(())
    }

    fn read_word(&self, word_addr: u64) -> u32 {
        self.words.get(&word_addr).copied().unwrap_or(0)
    }

    fn write_word(&mut self, word_addr: u64, value: u32) {
        if value == 0 {
            self.words.remove(&word_addr);
        } else {
            self.words.insert(word_addr, value);
        }
    }
}

fn grammar_uses_linear_memory(grammar: &HostEventGrammar) -> bool {
    grammar
        .imports
        .values()
        .flat_map(|template| &template.events)
        .chain(
            grammar
                .exports
                .values()
                .flat_map(|template| template.entry.iter().chain(&template.exit)),
        )
        .flat_map(|event| &event.block)
        .any(|source| {
            matches!(
                source,
                SlotSource::MemoryRead32 { .. } | SlotSource::MemoryWrite32 { .. }
            )
        })
}

pub(super) fn aligned_word_addr(base: u32, byte_offset: u32) -> Result<u64, WasmBuildError> {
    let effective = base.checked_add(byte_offset).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "grammar memory address overflows wasm32: {base} + {byte_offset}"
        ))
    })?;
    if effective % 4 != 0 {
        return Err(WasmBuildError::Trace(format!(
            "grammar Memory32 address {effective} is not naturally aligned"
        )));
    }
    Ok(u64::from(effective / 4))
}

pub(super) fn memory_word_access(word_addr: u64, value_before: u32, value_after: u32) -> LinearMemoryAccess {
    LinearMemoryAccess {
        width_bytes: 4,
        byte_offset: 0,
        lane0: LinearMemoryWordLane {
            word_addr,
            value_before,
            value_after,
        },
        lane1: None,
        lane2: None,
    }
}
