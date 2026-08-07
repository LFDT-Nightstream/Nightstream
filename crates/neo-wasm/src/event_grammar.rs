//! Embedder-injected host-event grammar: per-import templates that expand a
//! host call into a static sequence of absorb blocks for the commitment
//! chain, replacing the raw record format (see
//! `docs/host-event-grammar-tables.md`).
//!
//! Owns the template schema and its native expansion. Does not own the chain
//! permutation (`comm_chain`), the circuit gather machinery, or any specific
//! grammar: discriminants and slot layouts are embedder data — neo-wasm
//! never interprets them.
//!
//! One grammar event is exactly one absorb block: `[discriminant | 7 slots]`.
//! Templates are static per import. Slots resolve from flat values, claims,
//! or static linear-memory accesses.

use crate::comm_chain::{COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_EVENT_ARGS};
use crate::ir::{WasmBuildError, WasmGrammarRomVariant};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use std::collections::BTreeMap;

/// Which 32-bit limb of a two-limb value feeds a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Limb {
    Lo,
    Hi,
}

/// Runtime source of a wasm32 byte pointer used by a grammar memory slot.
/// Invalid effective addresses make trace construction or proving fail;
/// grammar memory slots do not model Wasm traps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryBase {
    /// An import argument, read from the call's operand-stack argument area.
    /// Its high limb must be zero so the value is a wasm32 pointer.
    Arg(u8),
    /// An export-frame local, read from the locals memory.
    Local(u8),
}

/// Where one event slot's value comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotSource {
    /// Fixed field element (canonical u64): zero padding, function ids,
    /// interface-id limbs, ref sizes, enum tags.
    Const(u64),
    /// A limb of the call's `arg`-th argument (declared parameter order).
    ArgElem { arg: u8, limb: Limb },
    /// A limb of the call's single flat result.
    ResultElem { limb: Limb },
    /// The `idx`-th claim word of the slot's phase (export entry inputs,
    /// per-call oracle values, exit values): a free absorbed word
    /// in-circuit. The index is claim-side structure — expansion resolves
    /// every slot sharing an `idx` from one claim entry, and the transcript
    /// check (native fold or the interleaving proof) binds the absorbed
    /// words to that claim; nothing is enforced locally.
    Claim { idx: u8 },
    /// Export entry templates only: the `idx`-th entry claim word, absorbed
    /// AND written to one 32-bit lane of the entry frame's `locals[local]`
    /// (the word must fit in 32 bits). This is how export inputs reach the
    /// guest: locals start all-zero and the bootstrap writes them. A `Lo`
    /// write zeroes the hi lane (the write is total); an i64 local takes a
    /// `Lo` slot followed by a `Hi` slot.
    ClaimLocal { idx: u8, local: u8, limb: Limb },
    /// Export exit templates only: a limb of the export's captured result
    /// (the carried simple-output value).
    OutputElem { limb: Limb },
    /// Read one naturally aligned 32-bit linear-memory word and stage it as
    /// this event word.
    MemoryRead32 { base: MemoryBase, byte_offset: u32 },
    /// Read one byte from linear memory and stage its zero-extended value as
    /// this event word.
    MemoryRead8 { base: MemoryBase, byte_offset: u32 },
    /// Write one naturally aligned 32-bit claim word to linear memory and
    /// stage the same word in this event slot. Host memory mutations later
    /// observed by proof-visible execution must be represented by a grammar
    /// write, or temporal memory replay rejects the inconsistent access.
    MemoryWrite32 {
        claim: u8,
        base: MemoryBase,
        byte_offset: u32,
    },
    /// Write one byte-sized claim word to linear memory and stage the same
    /// zero-extended value in this event slot.
    MemoryWrite8 {
        claim: u8,
        base: MemoryBase,
        byte_offset: u32,
    },
}

pub(crate) const fn memory_rom_arg_variant(base: MemoryBase, byte_width: bool) -> (u8, WasmGrammarRomVariant) {
    let (arg, local_base) = match base {
        MemoryBase::Arg(arg) => (arg, false),
        MemoryBase::Local(local) => (local, true),
    };
    (arg, WasmGrammarRomVariant::Memory { local_base, byte_width })
}

/// One grammar event: one block of 8 arbitrary slot sources.
///
/// neo-wasm attaches no meaning to any slot — in particular, "slot 0 is a
/// discriminant" is only the embedder's single-block op convention (see
/// [`GrammarEvent::op`]). Multi-block ops and discriminant-free
/// continuation blocks (dense payload encodings) are just more events in a
/// template.
///
/// If `absorb` is false, the gather rows and their VM effects still run,
/// but the block is omitted from the transcript. The flag is ROM-bound,
/// and templates may mix absorbing and advice events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrammarEvent {
    pub block: [SlotSource; COMM_CHAIN_BLOCK_WORDS],
    pub absorb: bool,
}

impl GrammarEvent {
    /// The discriminant-led single-block op layout: `[Const(disc) | slots]`.
    pub fn op(discriminant: u64, slots: [SlotSource; COMM_CHAIN_EVENT_ARGS]) -> Self {
        let mut block = [SlotSource::Const(0); COMM_CHAIN_BLOCK_WORDS];
        block[0] = SlotSource::Const(discriminant);
        block[1..].copy_from_slice(&slots);
        Self { block, absorb: true }
    }

    /// An advice block whose slot effects run without transcript absorption.
    pub fn advice(block: [SlotSource; COMM_CHAIN_BLOCK_WORDS]) -> Self {
        Self { block, absorb: false }
    }
}

/// Static expansion of one imported function into grammar events.
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
    pub events: Vec<GrammarEvent>,
    /// Number of per-call claim words the template draws from (one array
    /// for the whole call — e.g. a ref id claimed once and referenced in
    /// several events).
    pub claim_count: u8,
}

/// Static expansion of one exported function's boundary into grammar
/// events: `entry` events absorb before the export's first instruction
/// (receiver-side `Enter`/`Activation`/payload reads), `exit` events after
/// the halting row (`Return`/`Yield` and result publication). Single-turn
/// V1: one export invocation per trace.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportTemplate {
    pub entry: Vec<GrammarEvent>,
    pub exit: Vec<GrammarEvent>,
    /// Number of entry claim words (`Claim`/`ClaimLocal` indices resolve
    /// against this array; `ClaimLocal` words also bootstrap locals).
    pub entry_claim_count: u8,
    /// Number of exit claim words.
    pub exit_claim_count: u8,
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
    /// source from consts and entry claim words (`Claim`/`ClaimLocal`);
    /// exit events from consts, exit claim words, and the captured output.
    /// The stack-based import sources (`ArgElem`/`ResultElem`) never apply.
    pub fn validate(&self, local_bound: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        let mut written = std::collections::BTreeSet::new();
        for (phase, events, claim_count) in [
            (ExportPhase::Entry, &self.entry, self.entry_claim_count),
            (ExportPhase::Exit, &self.exit, self.exit_claim_count),
        ] {
            let phase_name = phase.name();
            for (idx, event) in events.iter().enumerate() {
                let ctx = |what: &str| format!("export template {phase_name} event {idx}: {what}");
                if !event.absorb {
                    return err(ctx("export boundary events must absorb; advice events are import-only"));
                }
                let check_claim_idx = |idx: u8| {
                    if idx >= claim_count {
                        return err(ctx(&format!(
                            "claim index {idx} out of range for {claim_count} {phase_name} claim words"
                        )));
                    }
                    Ok(())
                };
                for slot in &event.block {
                    match *slot {
                        SlotSource::Const(value) => {
                            if value >= Goldilocks::ORDER_U64 {
                                return err(ctx("constant is not a canonical field element"));
                            }
                        }
                        SlotSource::Claim { idx } => check_claim_idx(idx)?,
                        SlotSource::ClaimLocal { idx, local, limb } => {
                            if matches!(phase, ExportPhase::Exit) {
                                return err(ctx("locals bootstrap only applies to the entry phase"));
                            }
                            check_claim_idx(idx)?;
                            if local >= local_bound {
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
                        SlotSource::OutputElem { .. } => {
                            if matches!(phase, ExportPhase::Entry) {
                                return err(ctx("output reference before the export halts"));
                            }
                        }
                        SlotSource::MemoryRead32 { base, .. } | SlotSource::MemoryRead8 { base, .. } => {
                            if matches!(phase, ExportPhase::Entry) {
                                return err(ctx("memory reads only apply to the export exit phase"));
                            }
                            validate_export_memory_base(base, local_bound, &ctx)?;
                        }
                        SlotSource::MemoryWrite32 { claim, base, .. }
                        | SlotSource::MemoryWrite8 { claim, base, .. } => {
                            if matches!(phase, ExportPhase::Exit) {
                                return err(ctx("memory writes only apply to the export entry phase"));
                            }
                            check_claim_idx(claim)?;
                            validate_export_memory_base(base, local_bound, &ctx)?;
                            let MemoryBase::Local(local) = base else {
                                unreachable!("validated export memory base")
                            };
                            if !written.contains(&(local, false)) {
                                return err(ctx(&format!(
                                    "memory base local {local} must be bootstrap-written by an earlier ClaimLocal Lo slot"
                                )));
                            }
                        }
                        SlotSource::ArgElem { .. } | SlotSource::ResultElem { .. } => {
                            return err(ctx("stack-based import sources do not apply to export templates"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Entry and exit claim words for one export invocation. Multi-turn traces
/// supply these in invocation order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TurnClaims {
    pub entry: Vec<u64>,
    pub exit: Vec<u64>,
}

/// Per-program grammar: import templates keyed by callee function ref, and
/// export boundary templates keyed by the exported function's ref.
///
/// Absence of a grammar (or of a template for a given fref) means the raw
/// host-call record format applies — the zkVM stays usable with no embedder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostEventGrammar {
    pub imports: BTreeMap<u32, ImportTemplate>,
    pub exports: BTreeMap<u32, ExportTemplate>,
}

impl ImportTemplate {
    /// Check the template against the import's declared arity: every slot
    /// source must be resolvable on every call, and the slot ORDER must be
    /// consistent with the stack mechanics (see the struct doc — the
    /// `ResultElem` Lo slot pushes into argument 0's cell, so arg-0 reads
    /// must precede it and the Hi slot's hi-lane write must follow it). Run
    /// at injection time so expansion failures are table bugs, not trace bugs.
    /// These order rules are deliberately out-of-circuit: the grammar is
    /// verifier-authored data, audited here rather than in the relation.
    pub fn validate(&self, param_count: u8, result_count: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        if self.claim_count != 0 && !self.events.iter().any(|event| event.absorb) {
            return err(format!(
                "claim words are absorbed words; a template with no absorbing events cannot deliver {} of them",
                self.claim_count
            ));
        }
        let mut result_lo_seen = false;
        let mut result_hi_seen = false;
        for (idx, event) in self.events.iter().enumerate() {
            let ctx = |what: &str| format!("grammar template event {idx}: {what}");
            for slot in &event.block {
                // Unabsorbed slots must either affect the VM or be padding.
                if !event.absorb && !matches!(*slot, SlotSource::Const(_) | SlotSource::ResultElem { .. }) {
                    return err(ctx("advice events may only contain ResultElem and Const slots"));
                }
                match *slot {
                    SlotSource::Const(value) => {
                        if value >= Goldilocks::ORDER_U64 {
                            return err(ctx("constant is not a canonical field element"));
                        }
                    }
                    SlotSource::ArgElem { arg, .. } => {
                        if arg >= param_count {
                            return err(ctx(&format!("arg index {arg} out of range for {param_count} params")));
                        }
                        if arg == 0 && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this slot earlier",
                            ));
                        }
                    }
                    SlotSource::ResultElem { limb } => {
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
                    SlotSource::Claim { idx } => {
                        if idx >= self.claim_count {
                            return err(ctx(&format!(
                                "claim index {idx} out of range for {} claim words",
                                self.claim_count
                            )));
                        }
                    }
                    SlotSource::MemoryRead32 { base, .. } | SlotSource::MemoryRead8 { base, .. } => {
                        if base == MemoryBase::Arg(0) && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this memory slot earlier",
                            ));
                        }
                        validate_import_memory_base(base, param_count, &ctx)?;
                    }
                    SlotSource::MemoryWrite32 { claim, base, .. } | SlotSource::MemoryWrite8 { claim, base, .. } => {
                        if claim >= self.claim_count {
                            return err(ctx(&format!(
                                "claim index {claim} out of range for {} claim words",
                                self.claim_count
                            )));
                        }
                        if base == MemoryBase::Arg(0) && result_lo_seen {
                            return err(ctx(
                                "argument 0 is overwritten by the result push; move this memory slot earlier",
                            ));
                        }
                        validate_import_memory_base(base, param_count, &ctx)?;
                    }
                    SlotSource::ClaimLocal { .. } | SlotSource::OutputElem { .. } => {
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
/// limb pairs in declared parameter order; `claims` must supply exactly
/// `template.claim_count` canonical words.
pub fn expand_import_events(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    claims: &[u64],
    memory_reads: &[u32],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_claims("per-call", claims, template.claim_count)?;
    check_memory_reads("per-call", memory_reads, &template.events)?;
    let mut memory_index = 0usize;
    template
        .events
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = resolve_slot(*slot, args, result, claims, memory_reads, &mut memory_index)?;
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

/// Resolve an export template's ENTRY phase against the turn's entry claim
/// words. `ClaimLocal` words must fit the locals lanes (32 bits).
pub fn expand_export_entry(
    template: &ExportTemplate,
    entry_claims: &[u64],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_claims("entry", entry_claims, template.entry_claim_count)?;
    template
        .entry
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = match *slot {
                    SlotSource::Const(value) => value,
                    SlotSource::Claim { idx } => entry_claims[usize::from(idx)],
                    SlotSource::ClaimLocal { idx, local, .. } => {
                        let value = entry_claims[usize::from(idx)];
                        if value > u64::from(u32::MAX) {
                            return Err(WasmBuildError::Trace(format!(
                                "entry claim {idx} for local {local} does not fit the 32-bit locals lane"
                            )));
                        }
                        value
                    }
                    SlotSource::MemoryWrite32 { claim, .. } => entry_claims[usize::from(claim)],
                    SlotSource::MemoryWrite8 { claim, .. } => {
                        let value = entry_claims[usize::from(claim)];
                        if value > u64::from(u8::MAX) {
                            return Err(WasmBuildError::Trace(format!(
                                "entry claim {claim} for a byte memory write does not fit u8"
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
/// the exit claim words.
pub fn expand_export_exit(
    template: &ExportTemplate,
    output: Option<(u32, u32)>,
    exit_claims: &[u64],
    memory_reads: &[u32],
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_claims("exit", exit_claims, template.exit_claim_count)?;
    check_memory_reads("exit", memory_reads, &template.exit)?;
    let mut memory_index = 0usize;
    template
        .exit
        .iter()
        .map(|event| {
            let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
            for (word, slot) in block.iter_mut().zip(&event.block) {
                *word = match *slot {
                    SlotSource::Const(value) => value,
                    SlotSource::OutputElem { limb } => output
                        .map(|pair| limb_of(pair, limb))
                        .ok_or_else(|| WasmBuildError::Trace("export slot references a missing output".to_string()))?,
                    SlotSource::Claim { idx } => exit_claims[usize::from(idx)],
                    SlotSource::MemoryRead32 { .. } | SlotSource::MemoryRead8 { .. } => {
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

/// Claim arrays must match the template's declared count exactly and hold
/// canonical field elements — a non-canonical word would alias under the
/// field reduction, giving two u64 transcripts for one absorbed sequence.
fn check_claims(phase: &str, claims: &[u64], declared: u8) -> Result<(), WasmBuildError> {
    if claims.len() != usize::from(declared) {
        return Err(WasmBuildError::Trace(format!(
            "{phase} expansion expected {declared} claim words, got {}",
            claims.len()
        )));
    }
    if let Some(idx) = claims.iter().position(|&v| v >= Goldilocks::ORDER_U64) {
        return Err(WasmBuildError::Trace(format!(
            "{phase} claim {idx} is not a canonical field element"
        )));
    }
    Ok(())
}

fn check_memory_reads(phase: &str, memory_reads: &[u32], events: &[GrammarEvent]) -> Result<(), WasmBuildError> {
    let expected = events
        .iter()
        .flat_map(|event| &event.block)
        .filter(|source| matches!(source, SlotSource::MemoryRead32 { .. } | SlotSource::MemoryRead8 { .. }))
        .count();
    if memory_reads.len() != expected {
        return Err(WasmBuildError::Trace(format!(
            "{phase} expansion expected {expected} grammar memory reads, got {}",
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
        MemoryBase::Local(_) => Err(WasmBuildError::Trace(ctx(
            "import memory slots require an argument base",
        ))),
    }
}

fn validate_export_memory_base(
    base: MemoryBase,
    local_bound: u8,
    ctx: &impl Fn(&str) -> String,
) -> Result<(), WasmBuildError> {
    match base {
        MemoryBase::Local(local) if local < local_bound => Ok(()),
        MemoryBase::Local(local) => Err(WasmBuildError::Trace(ctx(&format!(
            "memory base local {local} out of range for {local_bound} locals"
        )))),
        MemoryBase::Arg(_) => Err(WasmBuildError::Trace(ctx("export memory slots require a local base"))),
    }
}

fn limb_of((lo, hi): (u32, u32), limb: Limb) -> u64 {
    match limb {
        Limb::Lo => u64::from(lo),
        Limb::Hi => u64::from(hi),
    }
}

fn resolve_slot(
    slot: SlotSource,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    claims: &[u64],
    memory_reads: &[u32],
    memory_index: &mut usize,
) -> Result<u64, WasmBuildError> {
    match slot {
        SlotSource::Const(value) => Ok(value),
        SlotSource::ArgElem { arg, limb } => args
            .get(usize::from(arg))
            .map(|&pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing arg {arg}"))),
        SlotSource::ResultElem { limb } => result
            .map(|pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace("grammar slot references a missing result".to_string())),
        SlotSource::Claim { idx } => claims
            .get(usize::from(idx))
            .copied()
            .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing claim {idx}"))),
        SlotSource::MemoryRead32 { .. } | SlotSource::MemoryRead8 { .. } => {
            let value = memory_reads[*memory_index];
            *memory_index += 1;
            Ok(u64::from(value))
        }
        SlotSource::MemoryWrite32 { claim, .. } => claims
            .get(usize::from(claim))
            .copied()
            .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing claim {claim}"))),
        SlotSource::MemoryWrite8 { claim, .. } => {
            let value = claims
                .get(usize::from(claim))
                .copied()
                .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing claim {claim}")))?;
            if value > u64::from(u8::MAX) {
                return Err(WasmBuildError::Trace(format!(
                    "grammar memory write claim {claim} does not fit u8"
                )));
            }
            Ok(value)
        }
        SlotSource::ClaimLocal { .. } | SlotSource::OutputElem { .. } => Err(WasmBuildError::Trace(
            "export-boundary sources do not apply to import templates".to_string(),
        )),
    }
}
