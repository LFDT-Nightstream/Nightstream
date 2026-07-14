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
//! Templates are static per import (V1 flat-only: every slot resolves to a
//! constant, an argument limb, a result limb, or a per-phase claim word).

use crate::comm_chain::{COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_EVENT_ARGS};
use crate::ir::WasmBuildError;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use std::collections::BTreeMap;

/// Which 32-bit limb of a two-limb value feeds a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Limb {
    Lo,
    Hi,
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
}

/// One grammar event: one absorb block of 8 arbitrary slot sources.
///
/// neo-wasm attaches no meaning to any slot — in particular, "slot 0 is a
/// discriminant" is only the embedder's single-block op convention (see
/// [`GrammarEvent::op`]). Multi-block ops and discriminant-free
/// continuation blocks (dense payload encodings) are just more events in a
/// template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrammarEvent {
    pub block: [SlotSource; COMM_CHAIN_BLOCK_WORDS],
}

impl GrammarEvent {
    /// The discriminant-led single-block op layout: `[Const(disc) | slots]`.
    pub fn op(discriminant: u64, slots: [SlotSource; COMM_CHAIN_EVENT_ARGS]) -> Self {
        let mut block = [SlotSource::Const(0); COMM_CHAIN_BLOCK_WORDS];
        block[0] = SlotSource::Const(discriminant);
        block[1..].copy_from_slice(&slots);
        Self { block }
    }
}

/// Static expansion of one imported function into grammar events.
///
/// Events split into two phases around the host result: `pre_result` events
/// may reference arguments (which the result push overwrites on the operand
/// stack), `post_result` events may additionally reference the result.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImportTemplate {
    pub pre_result: Vec<GrammarEvent>,
    pub post_result: Vec<GrammarEvent>,
    /// Number of per-call claim words the template draws from (both phases
    /// share one array — e.g. a ref id claimed once and referenced in a
    /// pre-result and a post-result event).
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

impl ExportTemplate {
    /// Check the template against the export's locals bound. Entry events
    /// source from consts and entry claim words (`Claim`/`ClaimLocal`);
    /// exit events from consts, exit claim words, and the captured output.
    /// The stack-based import sources (`ArgElem`/`ResultElem`) never apply.
    pub fn validate(&self, local_bound: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        let mut written = std::collections::BTreeSet::new();
        for (phase, events, claim_count) in [
            ("entry", &self.entry, self.entry_claim_count),
            ("exit", &self.exit, self.exit_claim_count),
        ] {
            for (idx, event) in events.iter().enumerate() {
                let ctx = |what: &str| format!("export template {phase} event {idx}: {what}");
                let check_claim_idx = |idx: u8| {
                    if idx >= claim_count {
                        return err(ctx(&format!(
                            "claim index {idx} out of range for {claim_count} {phase} claim words"
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
                            if phase == "exit" {
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
                            if phase == "entry" {
                                return err(ctx("output reference before the export halts"));
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
    /// source must be resolvable on every call. Run at injection time so
    /// expansion failures are table bugs, not trace bugs.
    pub fn validate(&self, param_count: u8, result_count: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        if result_count == 0 && !self.post_result.is_empty() {
            return err("grammar template has post-result events but the import returns nothing".to_string());
        }
        for (phase, events) in [("pre_result", &self.pre_result), ("post_result", &self.post_result)] {
            for (idx, event) in events.iter().enumerate() {
                let ctx = |what: &str| format!("grammar template {phase} event {idx}: {what}");
                for slot in &event.block {
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
                            // The host result is pushed into argument 0's
                            // stack slot, so post-result events would read
                            // the result there, not the argument.
                            if phase == "post_result" && arg == 0 {
                                return err(ctx("argument 0 is overwritten by the result push"));
                            }
                        }
                        SlotSource::ResultElem { .. } => {
                            if phase == "pre_result" {
                                return err(ctx("result reference in a pre-result event"));
                            }
                            if result_count == 0 {
                                return err(ctx("result reference on a resultless import"));
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
                        SlotSource::ClaimLocal { .. } | SlotSource::OutputElem { .. } => {
                            return err(ctx("export-boundary sources do not apply to import templates"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// One host call's grammar events resolved into absorb blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedImportEvents {
    pub pre_result_blocks: Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>,
    pub post_result_blocks: Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>,
}

/// Resolve a template against one call's data. `args` are `(lo, hi)` limb
/// pairs in declared parameter order; `claims` must supply exactly
/// `template.claim_count` canonical words.
pub fn expand_import_events(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    claims: &[u64],
) -> Result<ExpandedImportEvents, WasmBuildError> {
    check_claims("per-call", claims, template.claim_count)?;
    let resolve_phase = |events: &[GrammarEvent]| -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
        events
            .iter()
            .map(|event| {
                let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
                for (word, slot) in block.iter_mut().zip(&event.block) {
                    *word = resolve_slot(*slot, args, result, claims)?;
                }
                Ok(block)
            })
            .collect()
    };
    Ok(ExpandedImportEvents {
        pre_result_blocks: resolve_phase(&template.pre_result)?,
        post_result_blocks: resolve_phase(&template.post_result)?,
    })
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
) -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
    check_claims("exit", exit_claims, template.exit_claim_count)?;
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
        SlotSource::ClaimLocal { .. } | SlotSource::OutputElem { .. } => Err(WasmBuildError::Trace(
            "export-boundary sources do not apply to import templates".to_string(),
        )),
    }
}
