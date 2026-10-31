//! Static import phase boundaries. No gather or opaque operation spans suspension.

use super::{opaque, ImportTemplate, SlotBinding};
use crate::WasmBuildError;

pub(crate) const IMPORT_SCHEDULE_COUNT_BITS: u8 = 16;
pub(crate) const IMPORT_SCHEDULE_COUNT_FACTOR: u32 = 1 << IMPORT_SCHEDULE_COUNT_BITS;
/// Maximum number of eight-slot blocks in either import phase. The packed
/// schedule ROM reserves zero and stores each present phase's count plus one.
pub const MAX_IMPORT_PHASE_BLOCKS: usize = IMPORT_SCHEDULE_COUNT_FACTOR as usize - 2;

fn biased_count(blocks: usize) -> u32 {
    assert!(
        blocks <= MAX_IMPORT_PHASE_BLOCKS,
        "import schedule must be validated before encoding"
    );
    blocks as u32 + 1
}

impl ImportTemplate {
    pub fn is_split(&self) -> bool {
        self.return_events.is_some()
    }

    /// ROM encoding: zero for atomic, one plus the return-block count for split.
    pub(crate) fn return_schedule_count(&self) -> u32 {
        self.return_events
            .as_ref()
            .map_or(0, |events| biased_count(events.len()))
    }

    /// Low half: call blocks + 1. High half: zero for atomic, return blocks + 1
    /// for split (including an empty return phase).
    pub(crate) fn packed_schedule_counts(&self) -> u32 {
        biased_count(self.events.len()) + IMPORT_SCHEDULE_COUNT_FACTOR * self.return_schedule_count()
    }

    pub(super) fn validate_phases(&self) -> Result<(), WasmBuildError> {
        for (phase, blocks) in [
            ("call", self.events.len()),
            ("return", self.return_events.as_ref().map_or(0, Vec::len)),
        ] {
            if blocks > MAX_IMPORT_PHASE_BLOCKS {
                return Err(WasmBuildError::Trace(format!(
                    "import {phase} phase has {blocks} event blocks; maximum is {MAX_IMPORT_PHASE_BLOCKS}"
                )));
            }
        }
        opaque::validate(&self.events)?;
        let Some(return_events) = &self.return_events else {
            return Ok(());
        };
        opaque::validate(return_events)?;
        for slot in self.events.iter().flat_map(|event| &event.block) {
            if matches!(
                slot,
                SlotBinding::ResultElem { .. }
                    | SlotBinding::MemoryWrite8 { .. }
                    | SlotBinding::MemoryWrite16 { .. }
                    | SlotBinding::MemoryWrite32 { .. }
            ) {
                return Err(WasmBuildError::Trace(
                    "split import results and memory writes belong to the return phase".into(),
                ));
            }
        }
        Ok(())
    }
}
