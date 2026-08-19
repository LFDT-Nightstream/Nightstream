//! Temporal linear-memory state used while expanding host-event rows.
//!
//! Owns the sparse memory-zero image seen by normalization. Program rows and
//! host-event writes advance it in trace order; host-event reads derive from it.

use super::NormalizedStep;
use crate::adapters::wasmtime::WasmProgramTables;
use crate::host_event_bindings::{HostEventBindings, SlotBinding};
use crate::ir::{LinearMemoryAccess, LinearMemoryWordLane, WasmBuildError};
use crate::isa::WasmMemoryAccessKind;
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct LinearMemoryImage {
    words: BTreeMap<u64, u32>,
}

impl LinearMemoryImage {
    pub(super) fn for_host_events(
        bindings: &HostEventBindings,
        program: &WasmProgramTables,
    ) -> Result<Option<Self>, WasmBuildError> {
        if !host_events_use_linear_memory(bindings) {
            return Ok(None);
        }
        if program.has_imported_memory {
            return Err(WasmBuildError::Unsupported(
                "host-event memory slots require verifier-known memory-zero initialization; imported memory is unsupported"
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

    pub(super) fn read_aligned_word(
        &self,
        base: u32,
        byte_offset: u32,
        memory_pages: Option<u32>,
    ) -> Result<(u32, LinearMemoryAccess), WasmBuildError> {
        let word_addr = aligned_word_addr(base, byte_offset)?;
        ensure_word_in_bounds(word_addr, memory_pages)?;
        let value = self.read_word(word_addr);
        Ok((value, memory_access(4, 0, word_addr, value, value)))
    }

    pub(super) fn write_aligned_word(
        &mut self,
        base: u32,
        byte_offset: u32,
        value: u32,
        memory_pages: Option<u32>,
    ) -> Result<LinearMemoryAccess, WasmBuildError> {
        let word_addr = aligned_word_addr(base, byte_offset)?;
        ensure_word_in_bounds(word_addr, memory_pages)?;
        let prior = self.read_word(word_addr);
        self.write_word(word_addr, value);
        Ok(memory_access(4, 0, word_addr, prior, value))
    }

    pub(super) fn read_byte(
        &self,
        base: u32,
        byte_offset: u32,
        memory_pages: Option<u32>,
    ) -> Result<(u8, LinearMemoryAccess), WasmBuildError> {
        self.read_subword(base, byte_offset, memory_pages, |b, i| b[i])
    }

    pub(super) fn read_half(
        &self,
        base: u32,
        byte_offset: u32,
        memory_pages: Option<u32>,
    ) -> Result<(u16, LinearMemoryAccess), WasmBuildError> {
        self.read_subword(base, byte_offset, memory_pages, |b, i| {
            u16::from_le_bytes([b[i], b[i + 1]])
        })
    }

    pub(super) fn read_subword<T>(
        &self,
        base: u32,
        byte_offset: u32,
        memory_pages: Option<u32>,
        f: impl Fn([u8; 4], usize) -> T,
    ) -> Result<(T, LinearMemoryAccess), WasmBuildError> {
        let byte_width = std::mem::size_of::<T>();
        // FIXME: maybe restrict this with a trait instead
        // but probably should be in some helper module or smh
        debug_assert!((1..=2).contains(&byte_width));

        let (word_addr, byte_in_word) = subword_address(base, byte_offset, byte_width as u32)?;
        ensure_word_in_bounds(word_addr, memory_pages)?;
        let word = self.read_word(word_addr);
        let value = f(word.to_le_bytes(), byte_in_word as usize);

        Ok((
            value,
            memory_access(byte_width as u8, byte_in_word, word_addr, word, word),
        ))
    }

    pub(super) fn write_byte(
        &mut self,
        base: u32,
        byte_offset: u32,
        value: u8,
        memory_pages: Option<u32>,
    ) -> Result<LinearMemoryAccess, WasmBuildError> {
        self.write_subword(base, byte_offset, value as u16, 1, memory_pages)
    }

    pub(super) fn write_half(
        &mut self,
        base: u32,
        byte_offset: u32,
        value: u16,
        memory_pages: Option<u32>,
    ) -> Result<LinearMemoryAccess, WasmBuildError> {
        self.write_subword(base, byte_offset, value, 2, memory_pages)
    }

    pub(super) fn write_subword(
        &mut self,
        base: u32,
        byte_offset: u32,
        value: u16,
        byte_width: usize,
        memory_pages: Option<u32>,
    ) -> Result<LinearMemoryAccess, WasmBuildError> {
        let value_le_bytes = &value.to_le_bytes()[0..byte_width];
        let (word_addr, byte_in_word) = subword_address(base, byte_offset, 1)?;
        ensure_word_in_bounds(word_addr, memory_pages)?;
        let prior = self.read_word(word_addr);
        let mut bytes = prior.to_le_bytes();
        let offset = usize::from(byte_in_word);
        bytes[offset..offset + byte_width].copy_from_slice(&value_le_bytes);
        let updated = u32::from_le_bytes(bytes);
        self.write_word(word_addr, updated);

        Ok(memory_access(
            value_le_bytes.len() as u8,
            byte_in_word,
            word_addr,
            prior,
            updated,
        ))
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

fn host_events_use_linear_memory(bindings: &HostEventBindings) -> bool {
    bindings
        .imports
        .values()
        .flat_map(|template| &template.events)
        .chain(
            bindings
                .exports
                .values()
                .flat_map(|template| template.entry.iter().chain(&template.exit)),
        )
        .flat_map(|event| &event.block)
        .any(|source| {
            matches!(
                source,
                SlotBinding::MemoryRead32 { .. }
                    | SlotBinding::MemoryRead16 { .. }
                    | SlotBinding::MemoryRead8 { .. }
                    | SlotBinding::MemoryWrite32 { .. }
                    | SlotBinding::MemoryWrite16 { .. }
                    | SlotBinding::MemoryWrite8 { .. }
            )
        })
}

fn effective_byte_address(base: u32, byte_offset: u32) -> Result<u32, WasmBuildError> {
    base.checked_add(byte_offset).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "host-event memory address overflows wasm32: {base} + {byte_offset}"
        ))
    })
}

fn subword_address(base: u32, byte_offset: u32, alignment: u32) -> Result<(u64, u8), WasmBuildError> {
    let effective = effective_byte_address(base, byte_offset)?;

    if effective % alignment != 0 {
        return Err(WasmBuildError::Trace(format!(
            "host-event Memory16 address {effective} is not naturally aligned"
        )));
    }

    Ok((u64::from(effective / 4), (effective % 4) as u8))
}

fn aligned_word_addr(base: u32, byte_offset: u32) -> Result<u64, WasmBuildError> {
    let effective = effective_byte_address(base, byte_offset)?;
    if effective % 4 != 0 {
        return Err(WasmBuildError::Trace(format!(
            "host-event Memory32 address {effective} is not naturally aligned"
        )));
    }
    Ok(u64::from(effective / 4))
}

fn ensure_word_in_bounds(word_addr: u64, memory_pages: Option<u32>) -> Result<(), WasmBuildError> {
    let pages = memory_pages
        .ok_or_else(|| WasmBuildError::Trace("host-event memory access requires default linear memory".to_string()))?;
    let word_bound = u64::from(pages) * 16_384;
    if word_addr >= word_bound {
        return Err(WasmBuildError::Trace(format!(
            "host-event memory access at byte address {} is out of bounds for {pages} memory pages",
            word_addr * 4
        )));
    }
    Ok(())
}

fn memory_access(
    width_bytes: u8,
    byte_offset: u8,
    word_addr: u64,
    value_before: u32,
    value_after: u32,
) -> LinearMemoryAccess {
    LinearMemoryAccess {
        width_bytes,
        byte_offset,
        lane0: LinearMemoryWordLane {
            word_addr,
            value_before,
            value_after,
        },
        lane1: None,
        lane2: None,
    }
}
