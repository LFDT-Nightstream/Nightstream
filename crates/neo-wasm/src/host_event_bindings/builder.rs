use super::{EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding};
use crate::adapters::wasmtime::WasmProgramTables;
use crate::comm_chain::COMM_CHAIN_BLOCK_WORDS;
use crate::host_event_bindings::events_dense_input_count;
use crate::ir::WasmBuildError;

/// Builds an outer word stream with opaque roots, lowering it into a ROM schedule.
///
/// Decoders reconstructing the outer transcript must mirror these packing rules:
///
/// - Ordinary source words append consecutively; scalar limbs may cross eight-word blocks.
/// - Before an opaque root, zero-pad to the next block if fewer than four words remain
///   or the current block already contains a root.
/// - Append the four root words, then continue with subsequent sources.
/// - Zero-pad the final partial block. Empty sequences produce no blocks.
///
/// Field order therefore affects root placement and the number of outer blocks.
/// [`Self::op`] adds a tag only at the start, not to continuation blocks.
///
/// The finished vector contains ROM schedule groups, including suspended prefixes
/// and object-internal hashing; its length is not the outer transcript block count.
/// Use [`super::absorbed_blocks`] with matching expanded values to obtain that transcript.
#[derive(Clone, Debug)]
pub struct EventSequenceBuilder {
    words: Vec<SlotBinding>,
    absorb: bool,
    opaques: Vec<OpaquePlacement>,
}

#[derive(Clone, Debug)]
struct OpaquePlacement {
    word: usize,
    schema: [u64; 4],
    sources: Vec<SlotBinding>,
}

impl EventSequenceBuilder {
    pub fn absorbing() -> Self {
        Self::new(true)
    }

    pub fn advice() -> Self {
        Self::new(false)
    }

    /// Start with a discriminant at word zero; continuation blocks receive no extra tags.
    pub fn op(discriminant: u64) -> Self {
        let mut out = Self::absorbing();
        out.words.push(SlotBinding::Const(discriminant));
        out
    }

    fn new(absorb: bool) -> Self {
        Self {
            words: Vec::new(),
            absorb,
            opaques: Vec::new(),
        }
    }

    /// Append one source to the outer stream.
    pub fn push(mut self, binding: SlotBinding) -> Result<Self, WasmBuildError> {
        self.words
            .try_reserve(1)
            .map_err(|err| WasmBuildError::Trace(format!("host-event sequence allocation failed: {err}")))?;
        self.words.push(binding);
        Ok(self)
    }

    pub fn constant_i32(self, value: u32) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().constant_i32(value))
    }

    pub fn constant_i64(self, value: u64) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().constant_i64(value))
    }

    pub fn arg_i32(self, arg: u8) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().arg_i32(arg))
    }

    pub fn arg_i64(self, arg: u8) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().arg_i64(arg))
    }

    /// Append both result lanes: the low lane pushes, then the high lane binds it.
    pub fn result(self) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().result())
    }

    pub fn input_local_i32(self, input: u8, local: u8) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().input_local_i32(input, local))
    }

    pub fn input_local_i64(self, first_input: u8, local: u8) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().input_local_i64(first_input, local)?)
    }

    pub fn output_i32(self) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().output_i32())
    }

    pub fn output_i64(self) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().output_i64())
    }

    pub fn memory_read_i32(self, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().memory_read_i32(base, byte_offset))
    }

    pub fn memory_read_i64(self, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().memory_read_i64(base, byte_offset)?)
    }

    pub fn memory_write_i32(self, input: u8, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().memory_write_i32(input, base, byte_offset))
    }

    pub fn memory_write_i64(self, first_input: u8, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        self.append_sources(EventSources::new().memory_write_i64(first_input, base, byte_offset)?)
    }

    fn append_sources(mut self, value: EventSources) -> Result<Self, WasmBuildError> {
        for source in value.sources {
            self = self.push(source)?;
        }
        Ok(self)
    }

    /// Append an opaque root, padding to the next block if four words will not fit
    /// or the current block already has a root. Suffix appends follow the root.
    /// Opaque payloads cannot themselves contain opaque instructions.
    ///
    /// ```
    /// use neo_wasm::host_event_bindings::{EventSequenceBuilder, EventSources};
    /// # fn main() -> Result<(), neo_wasm::WasmBuildError> {
    /// let events = EventSequenceBuilder::op(1)
    ///     .arg_i32(0)?
    ///     .arg_i32(1)?
    ///     .opaque([1, 0, 0, 0], EventSources::new().arg_i32(2).arg_i64(3))?
    ///     .constant_i64(99)? // Limbs can cross an outer block boundary.
    ///     .opaque([2, 0, 0, 0], EventSources::new())?
    ///     .finish()?;
    /// # assert_eq!(events.len(), 7);
    /// # Ok(())
    /// # }
    /// ```
    pub fn opaque(mut self, schema: [u64; 4], value: EventSources) -> Result<Self, WasmBuildError> {
        if !self.absorb {
            return Err(WasmBuildError::Trace(
                "opaque roots require an absorbing sequence".into(),
            ));
        }
        let offset = self.words.len() % COMM_CHAIN_BLOCK_WORDS;
        let block_has_root = self
            .opaques
            .last()
            .is_some_and(|opaque| opaque.word / COMM_CHAIN_BLOCK_WORDS == self.words.len() / COMM_CHAIN_BLOCK_WORDS);
        if offset + 4 > COMM_CHAIN_BLOCK_WORDS || block_has_root {
            for _ in offset..COMM_CHAIN_BLOCK_WORDS {
                self = self.push(SlotBinding::Const(0))?;
            }
        }
        let word = self.words.len();
        for _ in 0..4 {
            self = self.push(SlotBinding::Const(0))?;
        }
        self.opaques.push(OpaquePlacement {
            word,
            schema,
            sources: value.sources,
        });
        Ok(self)
    }

    /// Pad the final block and return the complete schedule, including opaque work.
    /// An empty sequence produces no blocks. Function-specific validation belongs
    /// to `HostEventBindingsBuilder::finish`.
    pub fn finish(self) -> Result<Vec<EventBlock>, WasmBuildError> {
        let mut events = Vec::new();
        let mut opaques = self.opaques.into_iter().peekable();
        for (index, chunk) in self.words.chunks(COMM_CHAIN_BLOCK_WORDS).enumerate() {
            let mut block = [SlotBinding::Const(0); COMM_CHAIN_BLOCK_WORDS];
            block[..chunk.len()].copy_from_slice(chunk);
            let event = EventBlock {
                block,
                absorb: self.absorb,
            };
            if opaques
                .peek()
                .is_some_and(|opaque| opaque.word / COMM_CHAIN_BLOCK_WORDS == index)
            {
                let opaque = opaques.next().expect("opaque in this block");
                events.extend(event.with_opaque(
                    opaque.word % COMM_CHAIN_BLOCK_WORDS,
                    opaque.schema,
                    opaque.sources,
                )?);
            } else {
                events.push(event);
            }
        }
        Ok(events)
    }
}

/// An ordered, unpadded source list with shared scalar-to-limb expansion.
/// Supplies opaque payloads to `EventSequenceBuilder`; it has no framing or nesting.
#[derive(Clone, Debug, Default)]
pub struct EventSources {
    sources: Vec<SlotBinding>,
}

impl EventSources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, binding: SlotBinding) -> Self {
        self.sources.push(binding);
        self
    }

    pub fn constant_i32(self, value: u32) -> Self {
        self.push(SlotBinding::Const(u64::from(value)))
    }

    pub fn constant_i64(self, value: u64) -> Self {
        self.constant_i32(value as u32)
            .constant_i32((value >> 32) as u32)
    }

    pub fn arg_i32(self, arg: u8) -> Self {
        self.push(SlotBinding::ArgElem { arg, limb: Limb::Lo })
    }

    pub fn arg_i64(self, arg: u8) -> Self {
        self.arg_i32(arg)
            .push(SlotBinding::ArgElem { arg, limb: Limb::Hi })
    }

    /// Append both result lanes in stack-effect order.
    pub fn result(self) -> Self {
        self.push(SlotBinding::ResultElem { limb: Limb::Lo })
            .push(SlotBinding::ResultElem { limb: Limb::Hi })
    }

    pub fn input_local_i32(self, input: u8, local: u8) -> Self {
        self.push(SlotBinding::InputLocal {
            input,
            local,
            limb: Limb::Lo,
        })
    }

    pub fn input_local_i64(self, first_input: u8, local: u8) -> Result<Self, WasmBuildError> {
        let second_input = next_input(first_input)?;
        Ok(self
            .input_local_i32(first_input, local)
            .push(SlotBinding::InputLocal {
                input: second_input,
                local,
                limb: Limb::Hi,
            }))
    }

    pub fn output_i32(self) -> Self {
        self.push(SlotBinding::OutputElem { limb: Limb::Lo })
    }

    pub fn output_i64(self) -> Self {
        self.output_i32()
            .push(SlotBinding::OutputElem { limb: Limb::Hi })
    }

    pub fn memory_read_i32(self, base: MemoryBase, byte_offset: u32) -> Self {
        self.push(SlotBinding::MemoryRead32 { base, byte_offset })
    }

    pub fn memory_read_i64(self, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        let high_offset = next_limb_offset(byte_offset)?;
        Ok(self
            .memory_read_i32(base, byte_offset)
            .memory_read_i32(base, high_offset))
    }

    pub fn memory_write_i32(self, input: u8, base: MemoryBase, byte_offset: u32) -> Self {
        self.push(SlotBinding::MemoryWrite32 {
            input,
            base,
            byte_offset,
        })
    }

    pub fn memory_write_i64(self, first_input: u8, base: MemoryBase, byte_offset: u32) -> Result<Self, WasmBuildError> {
        let second_input = next_input(first_input)?;
        let high_offset = next_limb_offset(byte_offset)?;
        Ok(self
            .memory_write_i32(first_input, base, byte_offset)
            .memory_write_i32(second_input, base, high_offset))
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
