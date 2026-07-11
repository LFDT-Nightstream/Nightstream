//! Grammar-event emission: how expanded absorb blocks become trace rows.
//!
//! Owns the gather/perm aux-row plans and their emission — the per-slot
//! gather plans (`GrammarSlotRow`/`GrammarBlockPlan`/`GrammarCallPlan`),
//! the perm-group row plan probed from the native chain, and the shared
//! row context (`GrammarAuxCtx`) every emitted aux row repeats. Does not
//! own the trace state machine or per-opcode normalization — that is
//! `trace_build`, which drives these emitters.

use crate::comm_chain::{perm_row_checkpoints, COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_PERM_ROWS};
use crate::event_grammar::{GrammarEvent, SlotSource};
use crate::ir::{
    StackValueAccess, WasmAuxOpcode, WasmCountdownState, WasmEventAbsorbState, WasmOutputState, WasmPcEdgeKind,
    WasmRowKind, WasmStepState, WasmVmStep,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

/// One gather row's plan: the staged word, its claimed grammar-ROM entry,
/// and, for arg/result slots, the stack slot it must read — or, for
/// input-local slots, the entry-frame local lane `(local, limb_bit, word)`
/// the claim word is written into.
pub(super) struct GrammarSlotRow {
    pub(super) value: u64,
    pub(super) rom: crate::ir::WasmGrammarRomEntry,
    pub(super) read: Option<(u64, (u32, u32))>,
    pub(super) local_write: Option<(u32, u8, u32)>,
}

/// One grammar event's gather plan: the absorb block plus its 8 slot rows.
pub(super) struct GrammarBlockPlan {
    pub(super) block: [u64; 8],
    pub(super) rows: Vec<GrammarSlotRow>,
}

/// One grammar host call's full emission plan.
pub(super) struct GrammarCallPlan {
    pub(super) pre: Vec<GrammarBlockPlan>,
    pub(super) post: Vec<GrammarBlockPlan>,
    pub(super) args_base: u64,
    pub(super) oracles: [u64; 4],
}

pub(super) fn plan_grammar_blocks(
    events: &[GrammarEvent],
    blocks: &[[u64; 8]],
    args_base: u64,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
) -> Vec<GrammarBlockPlan> {
    events
        .iter()
        .zip(blocks)
        .map(|(event, &block)| {
            let rows = event
                .block
                .iter()
                .zip(block)
                .map(|(source, value)| {
                    let limb_bit = |limb| match limb {
                        crate::event_grammar::Limb::Lo => 0,
                        crate::event_grammar::Limb::Hi => 1,
                    };
                    let entry = |kind, arg, limb, const_lo, const_hi| crate::ir::WasmGrammarRomEntry {
                        kind,
                        arg,
                        limb,
                        const_lo,
                        const_hi,
                    };
                    let (rom, read) = match *source {
                        SlotSource::Const(value) => (entry(0, 0, 0, value as u32, (value >> 32) as u32), None),
                        SlotSource::ArgElem { arg, limb } => (
                            entry(1, arg, limb_bit(limb), 0, 0),
                            Some((args_base + u64::from(arg), args[usize::from(arg)])),
                        ),
                        SlotSource::ResultElem { limb } => (
                            entry(2, 0, limb_bit(limb), 0, 0),
                            Some((args_base, result.expect("validated result"))),
                        ),
                        SlotSource::Oracle { idx } => (entry(3, idx, 0, 0, 0), None),
                        SlotSource::Input | SlotSource::InputLocal { .. } | SlotSource::OutputElem { .. } => {
                            unreachable!("validated: export sources never appear in import templates")
                        }
                    };
                    GrammarSlotRow {
                        value,
                        rom,
                        read,
                        local_write: None,
                    }
                })
                .collect();
            GrammarBlockPlan { block, rows }
        })
        .collect()
}

/// The permutation input for one absorbed block premixed with the initial
/// external linear layer (canonical u64 lanes): what the row raising
/// `perm_pending` hands the perm group's first row as `perm_state`.
pub(super) fn absorb_premix(chain: [u64; 4], evbuf: [u64; 8]) -> [u64; 12] {
    let mut state = [Goldilocks::ZERO; 12];
    for (lane, limb) in state.iter_mut().zip(chain.iter().chain(evbuf.iter())) {
        *lane = Goldilocks::from_u64(*limb);
    }
    crate::comm_chain::perm_external_linear(&mut state);
    state.map(|lane| lane.as_canonical_u64())
}

/// Row-level plan of one absorbed block's perm group: the perm state entering
/// each of the [`COMM_CHAIN_PERM_ROWS`] rows (plus the permutation output)
/// and the fed-forward chain the group's last row lands on.
pub(super) fn perm_group_plan(chain: [u64; 4], evbuf: [u64; 8]) -> ([[u64; 12]; COMM_CHAIN_PERM_ROWS + 1], [u64; 4]) {
    let prev = chain.map(Goldilocks::from_u64);
    let mut words = [Goldilocks::ZERO; COMM_CHAIN_BLOCK_WORDS];
    for (word, limb) in words.iter_mut().zip(evbuf) {
        *word = Goldilocks::from_u64(limb);
    }
    let checkpoints = perm_row_checkpoints(prev, words);
    let updated: [u64; 4] =
        core::array::from_fn(|i| (checkpoints[COMM_CHAIN_PERM_ROWS][i] + prev[i]).as_canonical_u64());
    (
        checkpoints.map(|state| state.map(|lane| lane.as_canonical_u64())),
        updated,
    )
}

pub(super) fn plan_export_blocks(events: &[GrammarEvent], blocks: &[[u64; 8]]) -> Vec<GrammarBlockPlan> {
    events
        .iter()
        .zip(blocks)
        .map(|(event, &block)| {
            let rows = event
                .block
                .iter()
                .zip(block)
                .map(|(source, value)| {
                    let limb_bit = |limb| match limb {
                        crate::event_grammar::Limb::Lo => 0,
                        crate::event_grammar::Limb::Hi => 1,
                    };
                    let entry = |kind, arg, limb| crate::ir::WasmGrammarRomEntry {
                        kind,
                        arg,
                        limb,
                        const_lo: 0,
                        const_hi: 0,
                    };
                    let (rom, local_write) = match *source {
                        SlotSource::Const(value) => (
                            crate::ir::WasmGrammarRomEntry {
                                kind: 0,
                                arg: 0,
                                limb: 0,
                                const_lo: value as u32,
                                const_hi: (value >> 32) as u32,
                            },
                            None,
                        ),
                        SlotSource::Oracle { idx } => (entry(3, idx, 0), None),
                        SlotSource::InputLocal { local, limb } => {
                            // expand_export_entry rejects words over 32 bits.
                            let bit = limb_bit(limb);
                            (entry(4, local, bit), Some((u32::from(local), bit, value as u32)))
                        }
                        SlotSource::OutputElem { limb } => (entry(5, 0, limb_bit(limb)), None),
                        SlotSource::Input => (entry(6, 0, 0), None),
                        SlotSource::ArgElem { .. } | SlotSource::ResultElem { .. } => {
                            unreachable!("validated: stack sources never appear in export templates")
                        }
                    };
                    GrammarSlotRow {
                        value,
                        rom,
                        read: None,
                        local_write,
                    }
                })
                .collect();
            GrammarBlockPlan { block, rows }
        })
        .collect()
}

/// Shared shape of the grammar/perm aux rows emitted outside the per-opcode
/// flow: the carried context every such row repeats.
pub(super) struct GrammarAuxCtx {
    pub(super) pc: u64,
    pub(super) sp: u64,
    pub(super) output: WasmOutputState,
    pub(super) call_stack_depth: u64,
    pub(super) memory_pages: Option<u32>,
    pub(super) max_memory_pages: Option<u32>,
    pub(super) locals_fbp: u64,
    pub(super) host_callee_fref: u32,
    pub(super) grammar_mode: bool,
    pub(super) current_function_ref: u32,
    pub(super) current_function_num_locals: u32,
    pub(super) host_args: WasmCountdownState,
    pub(super) host_result_pending: bool,
}

impl GrammarAuxCtx {
    pub(super) fn state(
        &self,
        host_args: WasmCountdownState,
        comm_chain: [u64; 4],
        event_absorb: WasmEventAbsorbState,
        grammar: crate::ir::WasmGrammarState,
    ) -> WasmStepState {
        WasmStepState {
            pc: self.pc,
            sp: self.sp,
            output: self.output,
            call_stack_depth: self.call_stack_depth,
            memory_pages: self.memory_pages,
            max_memory_pages: self.max_memory_pages,
            locals_fbp: self.locals_fbp,
            halted: false,
            trapped: false,
            param_init: WasmCountdownState::ZERO,
            host_args,
            host_result_pending: self.host_result_pending,
            host_callee_fref: self.host_callee_fref,
            comm_chain,
            event_absorb,
            grammar_mode: self.grammar_mode,
            grammar,
        }
    }

    pub(super) fn row(
        &self,
        cycle: u64,
        aux_opcode: WasmAuxOpcode,
        state_before: WasmStepState,
        state_after: WasmStepState,
    ) -> WasmVmStep {
        WasmVmStep {
            cycle,
            row_kind: WasmRowKind::Aux(aux_opcode),
            state_before,
            state_after,
            control_choice: 0,
            pc_edge_kind: WasmPcEdgeKind::Static,
            wide_values_enabled: false,
            opcode: WasmOpcode::Nop,
            info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
            stack_reads_override: Some(0),
            stack_writes_override: Some(0),
            output_captured: false,
            current_function_ref: self.current_function_ref,
            current_function_num_locals: self.current_function_num_locals,
            stack_read0: None,
            stack_read1: None,
            stack_read2: None,
            stack_write0: None,
            linear_memory: None,
            linear_memory_offset: 0,
            local_index: None,
            local_read_value: None,
            local_read_value_hi: None,
            local_write_value: None,
            local_write_value_hi: None,
            global_index: None,
            global_read_value: None,
            global_read_value_hi: None,
            global_write_value: None,
            global_write_value_hi: None,
            table_id: None,
            table_index: None,
            table_value: None,
            function_ref: None,
            target_function_is_guest: false,
            function_type_id: None,
            call_indirect_type_index: None,
            expected_type_id: None,
            table_size: None,
            call_param_count: None,
            call_result_count: None,
            call_stack_push: None,
            call_stack_pop: None,
            grammar_rom_slot: None,
            grammar_pre_count: None,
            grammar_post_count: None,
        }
    }
}

/// Emit the perm-row group absorbing the pending block (see
/// `WasmAuxOpcode::HostEventPerm`): folds the chain forward on the group's
/// last row, clears the buffer on its first, and hands arg mode back when
/// arguments remain.
pub(super) fn emit_perm_group(
    out: &mut Vec<WasmVmStep>,
    ctx: &GrammarAuxCtx,
    comm_chain: &mut [u64; 4],
    absorb: &mut WasmEventAbsorbState,
    grammar: crate::ir::WasmGrammarState,
) {
    debug_assert!(absorb.perm_pending);
    debug_assert!(!ctx.host_args.active, "arg mode must suspend across a perm group");
    let (checkpoints, updated_chain) = perm_group_plan(*comm_chain, absorb.evbuf);
    debug_assert_eq!(absorb.perm_state, checkpoints[0]);
    let resumed_args = WasmCountdownState {
        active: ctx.host_args.remaining > 0,
        remaining: ctx.host_args.remaining,
    };
    for pos in 0..COMM_CHAIN_PERM_ROWS {
        let chain_before = *comm_chain;
        let absorb_before = *absorb;
        let last = pos + 1 == COMM_CHAIN_PERM_ROWS;
        if pos == 0 {
            absorb.evbuf = [0; 8];
            absorb.evbuf_slot = 0;
            absorb.perm_pending = false;
        }
        absorb.perm_round = if last { 0 } else { pos as u8 + 1 };
        absorb.perm_state = checkpoints[pos + 1];
        if last {
            *comm_chain = updated_chain;
        }
        out.push(ctx.row(
            out.len() as u64,
            WasmAuxOpcode::HostEventPerm,
            ctx.state(ctx.host_args, chain_before, absorb_before, grammar),
            ctx.state(
                if last { resumed_args } else { ctx.host_args },
                *comm_chain,
                *absorb,
                grammar,
            ),
        ));
    }
}

/// Emit one grammar event block: 8 gather rows (one word each, with the
/// arg/result stack read or export-param locals read the slot claims),
/// then its perm group. The last slot row premixes the block, raises
/// `pending`, and advances the event schedule.
pub(super) fn emit_block_plan(
    out: &mut Vec<WasmVmStep>,
    ctx: &GrammarAuxCtx,
    comm_chain: &mut [u64; 4],
    absorb: &mut WasmEventAbsorbState,
    gstate: &mut crate::ir::WasmGrammarState,
    plan: &GrammarBlockPlan,
) {
    for (word, slot) in plan.rows.iter().enumerate() {
        let absorb_before = *absorb;
        let gstate_before = *gstate;
        absorb.evbuf[word] = slot.value;
        gstate.slot_cursor = ((word + 1) % 8) as u8;
        if word == 7 {
            absorb.perm_pending = true;
            absorb.perm_state = absorb_premix(*comm_chain, absorb.evbuf);
            gstate.events_remaining -= 1;
            gstate.event_index += 1;
        }
        let read0 = slot.read.map(|(stack_slot, (lo, hi))| {
            StackValueAccess::new(stack_slot.saturating_mul(2), lo).with_optional_hi(Some(hi))
        });
        // Wide rows: a stack read carrying a hi limb, or a hi-lane locals
        // write (the narrow-row rule pins hi columns to zero otherwise).
        let wide =
            slot.read.is_some_and(|(_, (_, hi))| hi != 0) || slot.local_write.is_some_and(|(_, limb, _)| limb == 1);
        out.push(WasmVmStep {
            wide_values_enabled: wide,
            stack_reads_override: Some(u8::from(read0.is_some())),
            stack_read0: read0,
            local_index: slot.local_write.map(|(index, _, _)| index),
            // Lo rows write (word, 0); hi rows write only the hi lane.
            local_write_value: slot
                .local_write
                .map(|(_, limb, value)| if limb == 0 { value } else { 0 }),
            local_write_value_hi: slot
                .local_write
                .map(|(_, limb, value)| if limb == 0 { 0 } else { value }),
            grammar_rom_slot: Some(slot.rom),
            ..ctx.row(
                out.len() as u64,
                WasmAuxOpcode::HostEventGather,
                ctx.state(ctx.host_args, *comm_chain, absorb_before, gstate_before),
                ctx.state(ctx.host_args, *comm_chain, *absorb, *gstate),
            )
        });
    }
    emit_perm_group(out, ctx, comm_chain, absorb, *gstate);
}
