use super::{EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding};
use crate::adapters::wasmtime::WasmProgramTables;
use crate::comm_chain::COMM_CHAIN_BLOCK_WORDS;
use crate::host_event_bindings::events_dense_input_count;
use crate::ir::WasmBuildError;

/// Builds one event block without exposing its zero-padding array.
#[derive(Clone, Debug)]
pub struct EventBlockBuilder {
    block: [SlotBinding; COMM_CHAIN_BLOCK_WORDS],
    assigned: [bool; COMM_CHAIN_BLOCK_WORDS],
    absorb: bool,
    slot_start: usize,
}

impl EventBlockBuilder {
    pub fn absorbing() -> Self {
        Self::new(true)
    }

    pub fn advice() -> Self {
        Self::new(false)
    }

    /// Start the conventional discriminant-led block. [`Self::slot`] then
    /// addresses the seven payload positions after the discriminant.
    pub fn op(discriminant: u64) -> Self {
        let mut out = Self::absorbing();
        out.block[0] = SlotBinding::Const(discriminant);
        out.assigned[0] = true;
        out.slot_start = 1;
        out
    }

    fn new(absorb: bool) -> Self {
        Self {
            block: [SlotBinding::Const(0); COMM_CHAIN_BLOCK_WORDS],
            assigned: [false; COMM_CHAIN_BLOCK_WORDS],
            absorb,
            slot_start: 0,
        }
    }

    /// Set an absolute block word in `0..8`.
    pub fn word(mut self, word_index: usize, binding: SlotBinding) -> Result<Self, WasmBuildError> {
        if word_index >= COMM_CHAIN_BLOCK_WORDS {
            return Err(WasmBuildError::Trace(format!(
                "host-event block word {word_index} out of range for {COMM_CHAIN_BLOCK_WORDS} words"
            )));
        }
        if self.assigned[word_index] {
            return Err(WasmBuildError::Trace(format!(
                "host-event block word {word_index} assigned more than once"
            )));
        }
        self.block[word_index] = binding;
        self.assigned[word_index] = true;
        Ok(self)
    }

    /// Set a block-relative slot. Discriminant-led blocks expose seven slots
    /// after the tag; raw absorbing and advice blocks expose all eight words.
    pub fn slot(self, slot_index: usize, binding: SlotBinding) -> Result<Self, WasmBuildError> {
        let slot_count = COMM_CHAIN_BLOCK_WORDS - self.slot_start;
        if slot_index >= slot_count {
            return Err(WasmBuildError::Trace(format!(
                "host-event block slot {slot_index} out of range for {slot_count} available slots"
            )));
        }
        let word = self.slot_start + slot_index;
        if self.assigned[word] {
            return Err(WasmBuildError::Trace(format!(
                "host-event block slot {slot_index} assigned more than once"
            )));
        }
        let mut out = self;
        out.block[word] = binding;
        out.assigned[word] = true;
        Ok(out)
    }

    pub fn constant_i32(self, slot_index: usize, value: u32) -> Result<Self, WasmBuildError> {
        self.slot(slot_index, SlotBinding::Const(u64::from(value)))
    }

    pub fn constant_i64(self, slot_index: usize, value: u64) -> Result<Self, WasmBuildError> {
        self.slot_pair(
            slot_index,
            SlotBinding::Const(u64::from(value as u32)),
            SlotBinding::Const(u64::from((value >> 32) as u32)),
        )
    }

    pub fn arg_i32(self, slot_index: usize, arg: u8) -> Result<Self, WasmBuildError> {
        self.slot(slot_index, SlotBinding::ArgElem { arg, limb: Limb::Lo })
    }

    pub fn arg_i64(self, slot_index: usize, arg: u8) -> Result<Self, WasmBuildError> {
        self.slot_pair(
            slot_index,
            SlotBinding::ArgElem { arg, limb: Limb::Lo },
            SlotBinding::ArgElem { arg, limb: Limb::Hi },
        )
    }

    /// Bind both lanes of the import's single result. The low slot performs
    /// the stack push; the following high slot binds its high lane.
    pub fn result(self, slot_index: usize) -> Result<Self, WasmBuildError> {
        self.result_pair(slot_index)
    }

    pub fn input_local_i32(self, slot_index: usize, input: u8, local: u8) -> Result<Self, WasmBuildError> {
        self.slot(
            slot_index,
            SlotBinding::InputLocal {
                input,
                local,
                limb: Limb::Lo,
            },
        )
    }

    pub fn input_local_i64(self, slot_index: usize, first_input: u8, local: u8) -> Result<Self, WasmBuildError> {
        let second_input = next_input(first_input)?;
        self.slot_pair(
            slot_index,
            SlotBinding::InputLocal {
                input: first_input,
                local,
                limb: Limb::Lo,
            },
            SlotBinding::InputLocal {
                input: second_input,
                local,
                limb: Limb::Hi,
            },
        )
    }

    pub fn output_i32(self, slot_index: usize) -> Result<Self, WasmBuildError> {
        self.slot(slot_index, SlotBinding::OutputElem { limb: Limb::Lo })
    }

    pub fn output_i64(self, slot_index: usize) -> Result<Self, WasmBuildError> {
        self.slot_pair(
            slot_index,
            SlotBinding::OutputElem { limb: Limb::Lo },
            SlotBinding::OutputElem { limb: Limb::Hi },
        )
    }

    pub fn memory_read_i32(
        self,
        slot_index: usize,
        base: MemoryBase,
        byte_offset: u32,
    ) -> Result<Self, WasmBuildError> {
        self.slot(slot_index, SlotBinding::MemoryRead32 { base, byte_offset })
    }

    pub fn memory_read_i64(
        self,
        slot_index: usize,
        base: MemoryBase,
        byte_offset: u32,
    ) -> Result<Self, WasmBuildError> {
        let high_offset = next_limb_offset(byte_offset)?;
        self.slot_pair(
            slot_index,
            SlotBinding::MemoryRead32 { base, byte_offset },
            SlotBinding::MemoryRead32 {
                base,
                byte_offset: high_offset,
            },
        )
    }

    pub fn memory_write_i32(
        self,
        slot_index: usize,
        input: u8,
        base: MemoryBase,
        byte_offset: u32,
    ) -> Result<Self, WasmBuildError> {
        self.slot(
            slot_index,
            SlotBinding::MemoryWrite32 {
                input,
                base,
                byte_offset,
            },
        )
    }

    pub fn memory_write_i64(
        self,
        slot_index: usize,
        first_input: u8,
        base: MemoryBase,
        byte_offset: u32,
    ) -> Result<Self, WasmBuildError> {
        let second_input = next_input(first_input)?;
        let high_offset = next_limb_offset(byte_offset)?;
        self.slot_pair(
            slot_index,
            SlotBinding::MemoryWrite32 {
                input: first_input,
                base,
                byte_offset,
            },
            SlotBinding::MemoryWrite32 {
                input: second_input,
                base,
                byte_offset: high_offset,
            },
        )
    }

    fn result_pair(self, slot_index: usize) -> Result<Self, WasmBuildError> {
        self.slot_pair(
            slot_index,
            SlotBinding::ResultElem { limb: Limb::Lo },
            SlotBinding::ResultElem { limb: Limb::Hi },
        )
    }

    fn slot_pair(self, slot_index: usize, low: SlotBinding, high: SlotBinding) -> Result<Self, WasmBuildError> {
        let next_slot = slot_index
            .checked_add(1)
            .ok_or_else(|| WasmBuildError::Trace("host-event scalar slot index overflow".to_string()))?;
        self.slot(slot_index, low)?.slot(next_slot, high)
    }

    pub fn finish(self) -> EventBlock {
        EventBlock {
            block: self.block,
            absorb: self.absorb,
        }
    }
}

fn next_input(input: u8) -> Result<u8, WasmBuildError> {
    input.checked_add(1).ok_or_else(|| {
        WasmBuildError::Trace("host-event i64 low input index 255 has no following high-limb input".to_string())
    })
}

fn next_limb_offset(byte_offset: u32) -> Result<u32, WasmBuildError> {
    byte_offset
        .checked_add(4)
        .ok_or_else(|| WasmBuildError::Trace("host-event i64 memory offset overflows wasm32 address space".to_string()))
}

/// Builds and validates per-function bindings against parsed program tables.
pub struct HostEventBindingsBuilder<'a> {
    program: &'a WasmProgramTables,
    bindings: HostEventBindings,
}

impl<'a> HostEventBindingsBuilder<'a> {
    pub fn new(program: &'a WasmProgramTables) -> Self {
        Self {
            program,
            bindings: HostEventBindings::default(),
        }
    }

    /// Bind a host import. Referenced inputs must form a dense zero-based
    /// tuple; its length becomes the per-call input count.
    pub fn import(&mut self, function_ref: u32, events: Vec<EventBlock>) -> Result<&mut Self, WasmBuildError> {
        let input_count = {
            let events: &[EventBlock] = &events;
            events_dense_input_count(events, "host-event")
        }?;

        if self.bindings.imports.contains_key(&function_ref) {
            return Err(WasmBuildError::Trace(format!(
                "host-event import fref {function_ref} bound more than once"
            )));
        }

        self.bindings
            .imports
            .insert(function_ref, ImportTemplate { events, input_count });

        Ok(self)
    }

    /// Bind an export boundary. Entry inputs must form a dense zero-based
    /// tuple.
    pub fn export(
        &mut self,
        function_ref: u32,
        entry: Vec<EventBlock>,
        exit: Vec<EventBlock>,
    ) -> Result<&mut Self, WasmBuildError> {
        let entry_input_count = {
            let events: &[EventBlock] = &entry;
            events_dense_input_count(events, "host-event")
        }?;

        if self.bindings.exports.contains_key(&function_ref) {
            return Err(WasmBuildError::Trace(format!(
                "host-event export fref {function_ref} bound more than once"
            )));
        }

        self.bindings.exports.insert(
            function_ref,
            ExportTemplate {
                entry,
                exit,
                entry_input_count,
            },
        );

        Ok(self)
    }

    pub fn finish(self) -> Result<HostEventBindings, WasmBuildError> {
        self.bindings.validate_against_program(self.program)?;
        Ok(self.bindings)
    }
}
