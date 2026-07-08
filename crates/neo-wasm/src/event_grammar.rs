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
//! constant, an argument limb, a result limb, or a per-call oracle value).

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
    /// The `idx`-th per-call oracle value: prover-supplied, constrained only
    /// to be identical across every slot of the template that references it
    /// (ref ids, ret refs, callers, targets). Globally validated by the
    /// consumer of the chain (the interleaving proof), never locally.
    Oracle { idx: u8 },
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
    /// Number of per-call oracle values the template draws from.
    pub oracle_count: u8,
}

/// Per-call oracle cells the circuit carries; templates may not reference
/// more (see `SlotSource::Oracle`).
pub const MAX_ORACLE_CELLS: u8 = 4;

/// Per-program grammar: import templates keyed by callee function ref.
///
/// Absence of a grammar (or of a template for a given fref) means the raw
/// host-call record format applies — the zkVM stays usable with no embedder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostEventGrammar {
    pub imports: BTreeMap<u32, ImportTemplate>,
}

impl ImportTemplate {
    /// Check the template against the import's declared arity: every slot
    /// source must be resolvable on every call. Run at injection time so
    /// expansion failures are table bugs, not trace bugs.
    pub fn validate(&self, param_count: u8, result_count: u8) -> Result<(), WasmBuildError> {
        let err = |msg: String| Err(WasmBuildError::Trace(msg));
        if self.oracle_count > MAX_ORACLE_CELLS {
            return err(format!(
                "grammar template declares {} oracles; the circuit carries {MAX_ORACLE_CELLS} oracle cells",
                self.oracle_count
            ));
        }
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
                        SlotSource::Oracle { idx } => {
                            if idx >= self.oracle_count {
                                return err(ctx(&format!(
                                    "oracle index {idx} out of range for {} oracles",
                                    self.oracle_count
                                )));
                            }
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
/// pairs in declared parameter order; `oracles` must supply exactly
/// `template.oracle_count` values.
pub fn expand_import_events(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    oracles: &[u64],
) -> Result<ExpandedImportEvents, WasmBuildError> {
    if oracles.len() != usize::from(template.oracle_count) {
        return Err(WasmBuildError::Trace(format!(
            "grammar expansion expected {} oracle values, got {}",
            template.oracle_count,
            oracles.len()
        )));
    }
    // Blocks are canonical u64s everywhere (validate() pins constants and
    // discriminants; args/results are 32-bit limbs); a non-canonical oracle
    // would alias under the field reduction, giving two u64 transcripts for
    // the same absorbed event.
    if let Some(idx) = oracles.iter().position(|&v| v >= Goldilocks::ORDER_U64) {
        return Err(WasmBuildError::Trace(format!(
            "grammar oracle {idx} is not a canonical field element"
        )));
    }
    let resolve_phase = |events: &[GrammarEvent]| -> Result<Vec<[u64; COMM_CHAIN_BLOCK_WORDS]>, WasmBuildError> {
        events
            .iter()
            .map(|event| {
                let mut block = [0u64; COMM_CHAIN_BLOCK_WORDS];
                for (word, slot) in block.iter_mut().zip(&event.block) {
                    *word = resolve_slot(*slot, args, result, oracles)?;
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

fn resolve_slot(
    slot: SlotSource,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    oracles: &[u64],
) -> Result<u64, WasmBuildError> {
    let limb_of = |(lo, hi): (u32, u32), limb: Limb| match limb {
        Limb::Lo => u64::from(lo),
        Limb::Hi => u64::from(hi),
    };
    match slot {
        SlotSource::Const(value) => Ok(value),
        SlotSource::ArgElem { arg, limb } => args
            .get(usize::from(arg))
            .map(|&pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing arg {arg}"))),
        SlotSource::ResultElem { limb } => result
            .map(|pair| limb_of(pair, limb))
            .ok_or_else(|| WasmBuildError::Trace("grammar slot references a missing result".to_string())),
        SlotSource::Oracle { idx } => oracles
            .get(usize::from(idx))
            .copied()
            .ok_or_else(|| WasmBuildError::Trace(format!("grammar slot references missing oracle {idx}"))),
    }
}
