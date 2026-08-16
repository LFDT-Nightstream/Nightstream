//! Verifier-authored host-event bindings: import and export templates expand
//! Wasm interface boundaries into commitment-chain blocks and VM effects.
//!
//! Owns the template schema and its native expansion. Does not own the chain
//! permutation (`comm_chain`), the circuit gather machinery, or any specific
//! binding set: discriminants and slot layouts are embedder data — neo-wasm
//! never interprets them.
//!
//! One event block contains exactly eight words. Templates are static per
//! function reference; slots bind Wasm values, runtime inputs, constants,
//! or static linear-memory accesses.

use crate::comm_chain::{COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_EVENT_ARGS};
use crate::ir::{
    function_call_metadata_shape, WasmBuildError, WasmHostEventMemoryBase, WasmHostEventMemoryWidth,
    WasmHostEventRomVariant,
};
use crate::WasmProgramTables;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use std::collections::BTreeMap;

mod builder;
pub use builder::{EventBlockBuilder, HostEventBindingsBuilder};

/// Which 32-bit limb of a two-limb value feeds a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Limb {
    Lo,
    Hi,
}

/// Runtime source of a wasm32 byte pointer used by a host-event memory slot.
/// Invalid effective addresses make trace construction or proving fail;
/// host-event memory slots do not model Wasm traps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryBase {
    /// An import argument, read from the call's operand-stack argument area.
    /// Its high limb must be zero so the value is a wasm32 pointer.
    Arg(u8),
    /// An export-entry pointer local, read from the locals memory.
    Local(u8),
    /// The captured wasm32 result pointer of a single-result export.
    Output,
}

/// How one event slot obtains its value and any associated VM effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotBinding {
    /// Fixed field element (canonical u64): zero padding, function ids,
    /// interface-id limbs, ref sizes, enum tags.
    Const(u64),
    /// A limb of the call's `arg`-th argument (declared parameter order).
    ArgElem { arg: u8, limb: Limb },
    /// A limb of the call's single flat result.
    ResultElem { limb: Limb },
    /// The indexed input word of the slot's phase (export entry inputs,
    /// per-call host-provided values, exit values): a free absorbed word
    /// in-circuit. The index is input-side structure — expansion resolves
    /// every slot sharing an index from one input entry, and the transcript
    /// check (native fold or the interleaving proof) binds the absorbed
    /// words to that input; nothing is enforced locally.
    Input { index: u8 },
    /// Export entry templates only: the selected entry input word, absorbed
    /// AND written to one 32-bit lane of the entry frame's `locals[local]`
    /// (the word must fit in 32 bits). This is how export inputs reach the
    /// guest: locals start all-zero and the bootstrap writes them. A `Lo`
    /// write zeroes the hi lane (the write is total); an i64 local takes a
    /// `Lo` slot followed by a `Hi` slot.
    InputLocal { input: u8, local: u8, limb: Limb },
    /// Export exit templates only: a limb of the export's captured result
    /// (the carried simple-output value).
    OutputElem { limb: Limb },
    /// Read one naturally aligned 32-bit linear-memory word and stage it as
    /// this event word.
    MemoryRead32 { base: MemoryBase, byte_offset: u32 },
    /// Read one byte from linear memory and stage its zero-extended value as
    /// this event word.
    MemoryRead8 { base: MemoryBase, byte_offset: u32 },
    /// Read one naturally aligned little-endian 16-bit value from linear
    /// memory and stage its zero-extended value as this event word.
    MemoryRead16 { base: MemoryBase, byte_offset: u32 },
    /// Write one naturally aligned 32-bit input word to linear memory and
    /// stage the same word in this event slot. Host memory mutations later
    /// observed by proof-visible execution must be represented by a template
    /// write, or temporal memory replay rejects the inconsistent access.
    MemoryWrite32 {
        input: u8,
        base: MemoryBase,
        byte_offset: u32,
    },
    /// Write one byte-sized input word to linear memory and stage the same
    /// zero-extended value in this event slot.
    MemoryWrite8 {
        input: u8,
        base: MemoryBase,
        byte_offset: u32,
    },
    /// Write one naturally aligned 16-bit input word to linear memory and
    /// stage the same zero-extended value in this event slot.
    MemoryWrite16 {
        input: u8,
        base: MemoryBase,
        byte_offset: u32,
    },
}

pub(crate) const fn memory_rom_arg_variant(
    base: MemoryBase,
    width: WasmHostEventMemoryWidth,
) -> (u8, WasmHostEventRomVariant) {
    let (arg, base) = match base {
        MemoryBase::Arg(arg) => (arg, WasmHostEventMemoryBase::Argument),
        MemoryBase::Local(local) => (local, WasmHostEventMemoryBase::Local),
        MemoryBase::Output => (0, WasmHostEventMemoryBase::Output),
    };
    (arg, WasmHostEventRomVariant::Memory { base, width })
}

/// One host-event block: eight arbitrary slot bindings.
///
/// neo-wasm attaches no meaning to any slot — in particular, "slot 0 is a
/// discriminant" is only the embedder's single-block op convention (see
/// [`EventBlock::op`]). Multi-block ops and discriminant-free
/// continuation blocks (dense payload encodings) are just more events in a
/// template.
///
/// If `absorb` is false, the gather rows and their VM effects still run,
/// but the block is omitted from the transcript. The flag is ROM-bound,
/// and templates may mix absorbing and advice events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventBlock {
    pub block: [SlotBinding; COMM_CHAIN_BLOCK_WORDS],
    pub absorb: bool,
}

impl EventBlock {
    /// The discriminant-led single-block op layout: `[Const(disc) | slots]`.
    pub fn op(discriminant: u64, slots: [SlotBinding; COMM_CHAIN_EVENT_ARGS]) -> Self {
        let mut block = [SlotBinding::Const(0); COMM_CHAIN_BLOCK_WORDS];
        block[0] = SlotBinding::Const(discriminant);
        block[1..].copy_from_slice(&slots);
        Self { block, absorb: true }
    }

    /// An advice block whose slot effects run without transcript absorption.
    pub fn advice(block: [SlotBinding; COMM_CHAIN_BLOCK_WORDS]) -> Self {
        Self { block, absorb: false }
    }
}

/// Static expansion of one imported function into host-event blocks.
///
/// The whole call is one atomic event sequence at the call site. Imports
/// may mix transcript-bound and advice events; in either case, the
/// `ResultElem { limb: Lo }` slot pushes the host result onto the operand
/// stack.
///
/// SLOT-ORDER RULES (validated here, deliberately not in-circuit): the
/// result push lands in argument 0's stack cell, so every
/// slot reading argument 0 (directly or as a memory pointer) must come
/// before the `ResultElem` Lo slot,
/// and the `ResultElem` Hi slot (which writes the pushed cell's hi lane)
/// must come after it. A returning import must contain exactly one Lo slot
/// (the push — a narrow total write, hi lane zeroed) and a Hi slot (an i64
/// result's hi limb arrives only through it; an i32 result stages and
/// writes 0, costing nothing: it replaces a `Const(0)` padding slot).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImportTemplate {
    pub events: Vec<EventBlock>,
    /// Number of per-call input words the template draws from (one array
    /// for the whole call — e.g. a ref id supplied once and referenced in
    /// several events).
    pub input_count: u8,
}

/// Static expansion of one exported function's boundary into host-event
/// blocks: `entry` blocks absorb before the export's first instruction
/// (receiver-side `Enter`/`Activation`/payload reads), `exit` events after
/// the halting row (`Return`/`Yield` and result publication). Single-turn
/// V1: one export invocation per trace.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportTemplate {
    pub entry: Vec<EventBlock>,
    pub exit: Vec<EventBlock>,
    /// Number of entry input words (`Input`/`InputLocal` indices resolve
    /// against this array; `InputLocal` words also bootstrap locals).
    pub entry_input_count: u8,
    /// Number of exit input words.
    pub exit_input_count: u8,
}

#[derive(Clone, Copy)]
enum ExportPhase {
    Entry,
    Exit,
}

impl ExportPhase {
    const fn name(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Exit => "exit",
        }
    }
}

impl ExportTemplate {
    /// Check the template against the export's locals bound. Entry events
    /// source from consts and entry input words (`Input`/`InputLocal`);
    /// exit events from consts, exit input words, and the captured output.
    /// The stack-based import sources (`ArgElem`/`ResultElem`) never apply.
    pub fn validate(&self, local_bound: u32, result_count: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        let mut written = std::collections::BTreeSet::new();
        for (phase, events, input_count) in [
            (ExportPhase::Entry, &self.entry, self.entry_input_count),
            (ExportPhase::Exit, &self.exit, self.exit_input_count),
        ] {
            let phase_name = phase.name();
            for (idx, event) in events.iter().enumerate() {
                let ctx = |what: &str| format!("export template {phase_name} event {idx}: {what}");
                if !event.absorb {
                    return err(ctx("export boundary events must absorb; advice events are import-only"));
                }
                let check_input_index = |index: u8| {
                    if index >= input_count {
                        return err(ctx(&format!(
                            "input index {index} out of range for {input_count} {phase_name} input words"
                        )));
                    }
                    Ok(())
                };
                for slot in &event.block {
                    match *slot {
                        SlotBinding::Const(value) => {
                            if value >= Goldilocks::ORDER_U64 {
                                return err(ctx("constant is not a canonical field element"));
                            }
                        }
                        SlotBinding::Input { index } => check_input_index(index)?,
                        SlotBinding::InputLocal { input, local, limb } => {
                            if matches!(phase, ExportPhase::Exit) {
                                return err(ctx("locals bootstrap only applies to the entry phase"));
                            }
                            check_input_index(input)?;
                            if u32::from(local) >= local_bound {
                                return err(ctx(&format!(
                                    "local index {local} out of range for {local_bound} locals"
                                )));
                            }
                            if !written.insert((local, matches!(limb, Limb::Hi))) {
                                return err(ctx(&format!("local {local} lane written more than once")));
                            }
                            // A Lo write zeroes the hi lane, so an i64
                            // local's Hi slot must come after its Lo slot.
                            if matches!(limb, Limb::Hi) && !written.contains(&(local, false)) {
                                return err(ctx(&format!(
                                    "local {local} hi lane written before (or without) its lo lane"
                                )));
                            }
                        }
                        SlotBinding::OutputElem { .. } => {
                            if matches!(phase, ExportPhase::Entry) {
                                return err(ctx("output reference before the export halts"));
                            }
                            if result_count != 1 {
                                return err(ctx("output reference requires a single-result export"));
                            }
                        }
                        SlotBinding::MemoryRead32 { base, .. }
                        | SlotBinding::MemoryRead16 { base, .. }
                        | SlotBinding::MemoryRead8 { base, .. } => {
                            if matches!(phase, ExportPhase::Entry) {
                                return err(ctx("memory reads only apply to the export exit phase"));
                            }
                            if base != MemoryBase::Output {
                                return err(ctx("export exit memory reads require the captured output pointer"));
                            }
                            if result_count != 1 {
                                return err(ctx("export exit memory reads require a single-result export"));
                            }
                        }
                        SlotBinding::MemoryWrite32 { input, base, .. }
                        | SlotBinding::MemoryWrite16 { input, base, .. }
                        | SlotBinding::MemoryWrite8 { input, base, .. } => {
                            if matches!(phase, ExportPhase::Exit) {
                                return err(ctx("memory writes only apply to the export entry phase"));
                            }
                            check_input_index(input)?;
                            let MemoryBase::Local(local) = base else {
                                return err(ctx("export entry memory writes require a local base"));
                            };
                            if u32::from(local) >= local_bound {
                                return err(ctx(&format!(
                                    "memory base local {local} out of range for {local_bound} locals"
                                )));
                            }
                            if !written.contains(&(local, false)) {
                                return err(ctx(&format!(
                                    "memory base local {local} must be bootstrap-written by an earlier InputLocal Lo slot"
                                )));
                            }
                        }
                        SlotBinding::ArgElem { .. } | SlotBinding::ResultElem { .. } => {
                            return err(ctx("stack-based import sources do not apply to export templates"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Entry and exit input words for one export invocation. Multi-turn traces
/// supply these in invocation order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TurnInputs {
    pub entry: Vec<u64>,
    pub exit: Vec<u64>,
}

/// Per-program bindings: import templates keyed by callee function ref, and
/// export boundary templates keyed by the exported function's ref.
///
/// Import-free traces use [`HostEventBindings::import_free`]. Every executed
/// host import and entered export must have a matching template.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostEventBindings {
    pub imports: BTreeMap<u32, ImportTemplate>,
    pub exports: BTreeMap<u32, ExportTemplate>,
}

impl HostEventBindings {
    /// The canonical bindings of an import-free single-shot program: no import
    /// templates (so host calls are unprovable) and an empty boundary
    /// template for the invoked export (so nothing is absorbed and the
    /// commitment chain provably stays at its initial value).
    pub fn import_free(export_fref: u32) -> Self {
        let mut bindings = Self::default();
        bindings
            .exports
            .insert(export_fref, ExportTemplate::default());
        bindings
    }

    /// Validate every binding against verifier-owned program tables.
    pub fn validate_against_program(&self, program: &WasmProgramTables) -> Result<(), WasmBuildError> {
        for (&function_ref, template) in &self.imports {
            let (param_count, result_count, is_guest) = function_shape(program, function_ref)?;

            if is_guest {
                return Err(WasmBuildError::Trace(format!(
                    "host-event import fref {function_ref} names a guest function"
                )));
            }

            template.validate(param_count, result_count)?;
        }

        for (&function_ref, template) in &self.exports {
            let (_, result_count, is_guest) = function_shape(program, function_ref)?;

            if !is_guest {
                return Err(WasmBuildError::Trace(format!(
                    "host-event export fref {function_ref} names a host import"
                )));
            }

            let local_bound = program
                .function_local_counts
                .iter()
                .find_map(|&(fref, count)| (fref == u64::from(function_ref)).then_some(count))
                .ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "host-event export fref {function_ref} has no function-local-count entry"
                    ))
                })?;

            let local_bound = u32::try_from(local_bound).map_err(|_| {
                WasmBuildError::Trace(format!(
                    "host-event export fref {function_ref} local count does not fit u32"
                ))
            })?;

            template.validate(local_bound, result_count)?;
        }
        Ok(())
    }
}

fn function_shape(program: &WasmProgramTables, function_ref: u32) -> Result<(u8, u8, bool), WasmBuildError> {
    let metadata = program
        .function_call_metadata
        .iter()
        .find_map(|&(fref, metadata)| (fref == u64::from(function_ref)).then_some(metadata))
        .ok_or_else(|| {
            WasmBuildError::Trace(format!(
                "host-event binding fref {function_ref} has no function-call metadata"
            ))
        })?;

    Ok(function_call_metadata_shape(metadata))
}

impl ImportTemplate {
    /// Check the template against the import's declared arity: every slot
    /// source must be resolvable on every call, and the slot ORDER must be
    /// consistent with the stack mechanics (see the struct doc — the
    /// `ResultElem` Lo slot pushes into argument 0's cell, so arg-0 reads
    /// must precede it and the Hi slot's hi-lane write must follow it). Run
    /// at injection time so expansion failures are table bugs, not trace bugs.
    /// These order rules are deliberately out-of-circuit: the bindings are
    /// verifier-authored data, audited here rather than in the relation.
    pub fn validate(&self, param_count: u8, result_count: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        if self.input_count != 0 && !self.events.iter().any(|event| event.absorb) {
            return err(format!(
                "input words are absorbed words; a template with no absorbing events cannot deliver {} of them",
                self.input_count
            ));
        }
        let mut result_lo_seen = false;
        let mut result_hi_seen = false;
        for (idx, event) in self.events.iter().enumerate() {
            let ctx = |what: &str| format!("host-event template block {idx}: {what}");
            for slot in &event.block {
                // Unabsorbed slots must either affect the VM or be padding.
                if !event.absorb && !matches!(*slot, SlotBinding::Const(_) | SlotBinding::ResultElem { .. }) {
                    return err(ctx("advice events may only contain ResultElem and Const slots"));
                }
                match *slot {
                    SlotBinding::Const(value) => {
                        if value >= Goldilocks::ORDER_U64 {
                            return err(ctx("constant is not a canonical field element"));
                        }
                    }
                    SlotBinding::ArgElem { arg, .. } => {
                        if arg >= param_count {
                            return err(ctx(&format!("arg index {arg} out of range for {param_count} params")));
                        }
                        if arg == 0 && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this slot earlier",
                            ));
                        }
                    }
                    SlotBinding::ResultElem { limb } => {
                        if result_count == 0 {
                            return err(ctx("result reference on a resultless import"));
                        }
                        match limb {
                            Limb::Lo => {
                                if result_lo_seen {
                                    return err(ctx("a second ResultElem Lo slot would push the result twice"));
                                }
                                result_lo_seen = true;
                            }
                            Limb::Hi => {
                                if !result_lo_seen {
                                    return err(ctx(
                                        "ResultElem Hi writes the pushed cell's hi lane; it must follow the Lo slot",
                                    ));
                                }
                                if result_hi_seen {
                                    return err(ctx("duplicate ResultElem Hi slot"));
                                }
                                result_hi_seen = true;
                            }
                        }
                    }
                    SlotBinding::Input { index } => {
                        if index >= self.input_count {
                            return err(ctx(&format!(
                                "input index {index} out of range for {} input words",
                                self.input_count
                            )));
                        }
                    }
                    SlotBinding::MemoryRead32 { base, .. }
                    | SlotBinding::MemoryRead16 { base, .. }
                    | SlotBinding::MemoryRead8 { base, .. } => {
                        if base == MemoryBase::Arg(0) && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this memory slot earlier",
                            ));
                        }
                        validate_import_memory_base(base, param_count, &ctx)?;
                    }
                    SlotBinding::MemoryWrite32 { input, base, .. }
                    | SlotBinding::MemoryWrite16 { input, base, .. }
                    | SlotBinding::MemoryWrite8 { input, base, .. } => {
                        if input >= self.input_count {
                            return err(ctx(&format!(
                                "input index {input} out of range for {} input words",
                                self.input_count
                            )));
                        }
                        if base == MemoryBase::Arg(0) && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this memory slot earlier",
                            ));
                        }
                        validate_import_memory_base(base, param_count, &ctx)?;
                    }
                    SlotBinding::InputLocal { .. } | SlotBinding::OutputElem { .. } => {
                        return err(ctx("export-boundary sources do not apply to import templates"));
                    }
                }
            }
        }
        // The template drives the push: without a Lo slot the result never
        // reaches the operand stack and the trace cannot continue. Each
        // lane is written by its own slot (the Lo write zeroes the hi
        // lane), so the Hi slot is required for COMPLETENESS: an i64
        // result without one would leave the hi lane 0 and the trace would
        // diverge from the real execution. Requiring it uniformly (an i32
        // result stages and writes 0, replacing a Const(0) padding slot for
        // free) keeps this check width-agnostic.
        if result_count == 1 && !result_lo_seen {
            return err("a returning import's template must contain exactly one ResultElem Lo slot".to_string());
        }
        if result_count == 1 && !result_hi_seen {
            return err(
                "a returning import's template must carry the result hi limb (a ResultElem Hi slot after the Lo \
                 slot; it stages and writes 0 for an i32 result)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Resolve a template into one gather block per event. `args` are `(lo, hi)`
/// limb pairs in declared parameter order; `inputs` must supply exactly
/// `template.input_count` canonical words.
pub fn expand_import_events(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    inputs: &[u64],
    memory_reads: &[u32],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_inputs("per-call", inputs, template.input_count)?;
    check_memory_reads("per-call", memory_reads, &template.events)?;
    let mut memory_index = 0usize;
    template
        .events
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = resolve_slot(*slot, args, result, inputs, memory_reads, &mut memory_index)?;
            }
            Ok(block)
        })
        .collect()
}

/// Return the absorbing blocks from a matching template expansion.
pub fn absorbed_blocks(
    template: &ImportTemplate,
    blocks: &[[u64; COMM_CHAIN_BLOCK_WORDS]],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    if blocks.len() != template.events.len() {
        return Err(WasmBuildError::Trace(format!(
            "expansion has {} blocks but the template declares {} events; blocks must come from \
             expand_import_events on the same template",
            blocks.len(),
            template.events.len()
        )));
    }
    Ok(template
        .events
        .iter()
        .zip(blocks)
        .filter(|(event, _)| event.absorb)
        .map(|(_, &block)| block)
        .collect())
}

/// Resolve an export template's ENTRY phase against the turn's entry input
/// words. `InputLocal` words must fit the locals lanes (32 bits).
pub fn expand_export_entry(
    template: &ExportTemplate,
    entry_inputs: &[u64],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_inputs("entry", entry_inputs, template.entry_input_count)?;
    template
        .entry
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = match *slot {
                    SlotBinding::Const(value) => value,
                    SlotBinding::Input { index } => entry_inputs[usize::from(index)],
                    SlotBinding::InputLocal { input, local, .. } => {
                        let value = entry_inputs[usize::from(input)];
                        if value > u64::from(u32::MAX) {
                            return Err(WasmBuildError::Trace(format!(
                                "entry input {input} for local {local} does not fit the 32-bit locals lane"
                            )));
                        }
                        value
                    }
                    SlotBinding::MemoryWrite32 { input, .. } => entry_inputs[usize::from(input)],
                    SlotBinding::MemoryWrite8 { input, .. } => {
                        let value = entry_inputs[usize::from(input)];
                        if value > u64::from(u8::MAX) {
                            return Err(WasmBuildError::Trace(format!(
                                "entry input {input} for a byte memory write does not fit u8"
                            )));
                        }
                        value
                    }
                    SlotBinding::MemoryWrite16 { input, .. } => {
                        let value = entry_inputs[usize::from(input)];
                        if value > u64::from(u16::MAX) {
                            return Err(WasmBuildError::Trace(format!(
                                "entry input {input} for a half-word memory write does not fit u16"
                            )));
                        }
                        value
                    }
                    other => {
                        return Err(WasmBuildError::Trace(format!(
                            "slot source {other:?} does not apply to the export entry phase"
                        )))
                    }
                };
            }
            Ok(block)
        })
        .collect()
}

/// Resolve an export template's EXIT phase against the captured output and
/// the exit input words.
pub fn expand_export_exit(
    template: &ExportTemplate,
    output: Option<(u32, u32)>,
    exit_inputs: &[u64],
    memory_reads: &[u32],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_inputs("exit", exit_inputs, template.exit_input_count)?;
    check_memory_reads("exit", memory_reads, &template.exit)?;
    let mut memory_index = 0usize;
    template
        .exit
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = match *slot {
                    SlotBinding::Const(value) => value,
                    SlotBinding::OutputElem { limb } => output
                        .map(|pair| limb_of(pair, limb))
                        .ok_or_else(|| WasmBuildError::Trace("export slot references a missing output".to_string()))?,
                    SlotBinding::Input { index } => exit_inputs[usize::from(index)],
                    SlotBinding::MemoryRead32 { .. }
                    | SlotBinding::MemoryRead16 { .. }
                    | SlotBinding::MemoryRead8 { .. } => {
                        let value = memory_reads[memory_index];
                        memory_index += 1;
                        u64::from(value)
                    }
                    other => {
                        return Err(WasmBuildError::Trace(format!(
                            "slot source {other:?} does not apply to the export exit phase"
                        )))
                    }
                };
            }
            Ok(block)
        })
        .collect()
}

/// Input arrays must match the template's declared count exactly and hold
/// canonical field elements — a non-canonical word would alias under the
/// field reduction, giving two u64 transcripts for one absorbed sequence.
fn check_inputs(phase: &str, inputs: &[u64], declared: u8) -> Result<(), WasmBuildError> {
    if inputs.len() != usize::from(declared) {
        return Err(WasmBuildError::Trace(format!(
            "{phase} expansion expected {declared} input words, got {}",
            inputs.len()
        )));
    }
    if let Some(idx) = inputs.iter().position(|&v| v >= Goldilocks::ORDER_U64) {
        return Err(WasmBuildError::Trace(format!(
            "{phase} input {idx} is not a canonical field element"
        )));
    }
    Ok(())
}

fn check_memory_reads(phase: &str, memory_reads: &[u32], events: &[EventBlock]) -> Result<(), WasmBuildError> {
    let expected = events
        .iter()
        .flat_map(|event| &event.block)
        .filter(|source| {
            matches!(
                source,
                SlotBinding::MemoryRead32 { .. } | SlotBinding::MemoryRead16 { .. } | SlotBinding::MemoryRead8 { .. }
            )
        })
        .count();
    if memory_reads.len() != expected {
        return Err(WasmBuildError::Trace(format!(
            "{phase} expansion expected {expected} host-event memory reads, got {}",
            memory_reads.len()
        )));
    }
    Ok(())
}

fn validate_import_memory_base(
    base: MemoryBase,
    param_count: u8,
    ctx: &impl Fn(&str) -> String,
) -> Result<(), WasmBuildError> {
    match base {
        MemoryBase::Arg(arg) if arg < param_count => Ok(()),
        MemoryBase::Arg(arg) => Err(WasmBuildError::Trace(ctx(&format!(
            "memory base arg {arg} out of range for {param_count} params"
        )))),
        MemoryBase::Local(_) | MemoryBase::Output => Err(WasmBuildError::Trace(ctx(
            "import memory slots require an argument base",
        ))),
    }
}

fn limb_of((lo, hi): (u32, u32), limb: Limb) -> u64 {
    match limb {
        Limb::Lo => u64::from(lo),
        Limb::Hi => u64::from(hi),
    }
}

fn resolve_slot(
    slot: SlotBinding,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    inputs: &[u64],
    memory_reads: &[u32],
    memory_index: &mut usize,
) -> Result<u64, WasmBuildError> {
    match slot {
        SlotBinding::Const(value) => Ok(value),
        SlotBinding::ArgElem { arg, limb } => args
            .get(usize::from(arg))
            .map(|&pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace(format!("event slot references missing arg {arg}"))),
        SlotBinding::ResultElem { limb } => result
            .map(|pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace("event slot references a missing result".to_string())),
        SlotBinding::Input { index } => inputs
            .get(usize::from(index))
            .copied()
            .ok_or_else(|| WasmBuildError::Trace(format!("event slot references missing input {index}"))),
        SlotBinding::MemoryRead32 { .. } | SlotBinding::MemoryRead16 { .. } | SlotBinding::MemoryRead8 { .. } => {
            let value = memory_reads[*memory_index];
            *memory_index += 1;
            Ok(u64::from(value))
        }
        SlotBinding::MemoryWrite32 { input, .. } => inputs
            .get(usize::from(input))
            .copied()
            .ok_or_else(|| WasmBuildError::Trace(format!("event slot references missing input {input}"))),
        SlotBinding::MemoryWrite8 { input, .. } => {
            let value = inputs
                .get(usize::from(input))
                .copied()
                .ok_or_else(|| WasmBuildError::Trace(format!("event slot references missing input {input}")))?;
            if value > u64::from(u8::MAX) {
                return Err(WasmBuildError::Trace(format!(
                    "host-event memory write input {input} does not fit u8"
                )));
            }
            Ok(value)
        }
        SlotBinding::MemoryWrite16 { input, .. } => {
            let value = inputs
                .get(usize::from(input))
                .copied()
                .ok_or_else(|| WasmBuildError::Trace(format!("event slot references missing input {input}")))?;
            if value > u64::from(u16::MAX) {
                return Err(WasmBuildError::Trace(format!(
                    "host-event memory write input {input} does not fit u16"
                )));
            }
            Ok(value)
        }
        SlotBinding::InputLocal { .. } | SlotBinding::OutputElem { .. } => Err(WasmBuildError::Trace(
            "export-boundary sources do not apply to import templates".to_string(),
        )),
    }
}
