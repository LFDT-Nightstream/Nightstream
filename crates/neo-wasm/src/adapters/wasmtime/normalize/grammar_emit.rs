//! Grammar-event emission: how expanded absorb blocks become trace rows.
//!
//! Owns the gather/perm aux-row plans and their emission — the per-slot
//! gather plans (`GrammarSlotRow`/`GrammarBlockPlan`/`GrammarCallPlan`),
//! the perm-group row plan probed from the native chain, and the shared
//! row context (`GrammarAuxCtx`) every emitted aux row repeats. Does not
//! own the trace state machine or per-opcode normalization — that is
//! `trace_build`, which drives these emitters.

use crate::comm_chain::{perm_row_checkpoints, COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_PERM_ROWS};
use crate::event_grammar::{
    expand_import_events, memory_rom_arg_variant, GrammarEvent, ImportTemplate, Limb, MemoryBase, SlotSource,
};
use crate::ir::{
    LinearMemoryAccess, StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmCountdownState, WasmEventAbsorbState,
    WasmGrammarRomVariant, WasmGrammarSlotKind, WasmOutputState, WasmPcEdgeKind, WasmRowKind, WasmStepState,
    WasmVmStep,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::memory::LinearMemoryImage;

/// One gather row's plan: the staged word, its claimed grammar-ROM entry,
/// and its stack/locals effect — an addressed read `(stack slot, (lo, hi))`
/// for arg slots, a stack WRITE for result slots (the Lo slot is the push,
/// the Hi slot writes only the cell's hi word; the inactive lane is zero),
/// an entry-frame locals access, or one static linear-memory access.
pub(super) struct GrammarSlotRow {
    pub(super) value: u64,
    pub(super) rom: crate::ir::WasmGrammarRomEntry,
    pub(super) read: Option<(u64, (u32, u32))>,
    pub(super) write: Option<(u64, (u32, u32))>,
    pub(super) local_read: Option<(u32, u32)>,
    pub(super) local_write: Option<(u32, u8, u32)>,
    pub(super) linear_memory: Option<LinearMemoryAccess>,
}

/// One grammar event's gather plan: the staged block plus its 8 slot rows.
pub(super) struct GrammarBlockPlan {
    pub(super) block: [u64; 8],
    pub(super) rows: Vec<GrammarSlotRow>,
    /// Whether the staged block enters the transcript.
    pub(super) absorb: bool,
}

pub(super) struct GrammarCallPlan {
    pub(super) blocks: Vec<GrammarBlockPlan>,
    pub(super) args_base: u64,
}

pub(super) struct ResolvedGrammarMemory {
    pub(super) reads: Vec<u32>,
    pub(super) accesses: Vec<LinearMemoryAccess>,
}

pub(super) fn plan_import_call(
    template: &ImportTemplate,
    args_base: u64,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    claims: &[u64],
    memory: &mut LinearMemoryImage,
) -> Result<GrammarCallPlan, WasmBuildError> {
    let resolved_memory = resolve_import_memory(template, args, claims, memory)?;
    let blocks = expand_import_events(template, args, result, claims, &resolved_memory.reads)?;
    Ok(GrammarCallPlan {
        blocks: plan_grammar_blocks(
            &template.events,
            &blocks,
            args_base,
            args,
            result,
            &resolved_memory.accesses,
        )?,
        args_base,
    })
}

fn resolve_import_memory(
    template: &ImportTemplate,
    import_args: &[(u32, u32)],
    claims: &[u64],
    memory: &mut LinearMemoryImage,
) -> Result<ResolvedGrammarMemory, WasmBuildError> {
    let mut reads = Vec::new();
    let mut accesses = Vec::new();
    for source in template.events.iter().flat_map(|event| &event.block) {
        match *source {
            SlotSource::MemoryRead32 { base, byte_offset } => {
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = import_memory_pointer(import_args, arg)?;
                let (value, access) = memory.read_aligned_word(pointer, byte_offset)?;
                reads.push(value);
                accesses.push(access);
            }
            SlotSource::MemoryRead8 { base, byte_offset } => {
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = import_memory_pointer(import_args, arg)?;
                let (value, access) = memory.read_byte(pointer, byte_offset)?;
                reads.push(u32::from(value));
                accesses.push(access);
            }
            SlotSource::MemoryWrite32 {
                claim,
                base,
                byte_offset,
            } => {
                let value = claims.get(usize::from(claim)).copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!("grammar memory write references missing claim {claim}"))
                })?;
                let value = u32::try_from(value)
                    .map_err(|_| WasmBuildError::Trace("grammar memory write value does not fit u32".to_string()))?;
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = import_memory_pointer(import_args, arg)?;
                accesses.push(memory.write_aligned_word(pointer, byte_offset, value)?);
            }
            SlotSource::MemoryWrite8 {
                claim,
                base,
                byte_offset,
            } => {
                let value = claims.get(usize::from(claim)).copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!("grammar memory write references missing claim {claim}"))
                })?;
                let value = u8::try_from(value)
                    .map_err(|_| WasmBuildError::Trace("grammar memory write value does not fit u8".to_string()))?;
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = import_memory_pointer(import_args, arg)?;
                accesses.push(memory.write_byte(pointer, byte_offset, value)?);
            }
            _ => {}
        }
    }
    Ok(ResolvedGrammarMemory { reads, accesses })
}

fn import_memory_pointer(import_args: &[(u32, u32)], arg: u8) -> Result<u32, WasmBuildError> {
    let (lo, hi) = import_args[usize::from(arg)];
    if hi != 0 {
        return Err(WasmBuildError::Trace(format!(
            "grammar memory base arg {arg} is not a wasm32 pointer: high limb is {hi}"
        )));
    }
    Ok(lo)
}

pub(super) fn apply_export_entry_memory(
    events: &[GrammarEvent],
    blocks: &[[u64; 8]],
    locals: &[u32],
    memory: &mut LinearMemoryImage,
) -> Result<Vec<LinearMemoryAccess>, WasmBuildError> {
    let mut accesses = Vec::new();
    for (source, value) in events
        .iter()
        .zip(blocks)
        .flat_map(|(event, block)| event.block.iter().zip(block))
    {
        match *source {
            SlotSource::MemoryWrite32 { base, byte_offset, .. } => {
                let value = u32::try_from(*value)
                    .map_err(|_| WasmBuildError::Trace("grammar memory write value does not fit u32".to_string()))?;
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                accesses.push(memory.write_aligned_word(locals[usize::from(local)], byte_offset, value)?);
            }
            SlotSource::MemoryWrite8 { base, byte_offset, .. } => {
                let value = u8::try_from(*value)
                    .map_err(|_| WasmBuildError::Trace("grammar memory write value does not fit u8".to_string()))?;
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                accesses.push(memory.write_byte(locals[usize::from(local)], byte_offset, value)?);
            }
            _ => {}
        }
    }
    Ok(accesses)
}

pub(super) fn read_export_exit_memory(
    events: &[GrammarEvent],
    locals: &[u32],
    memory: &LinearMemoryImage,
) -> Result<ResolvedGrammarMemory, WasmBuildError> {
    let mut reads = Vec::new();
    let mut accesses = Vec::new();
    for source in events.iter().flat_map(|event| &event.block) {
        let resolved = match *source {
            SlotSource::MemoryRead32 { base, byte_offset } => {
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                let (value, access) = memory.read_aligned_word(locals[usize::from(local)], byte_offset)?;
                Some((value, access))
            }
            SlotSource::MemoryRead8 { base, byte_offset } => {
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                let (value, access) = memory.read_byte(locals[usize::from(local)], byte_offset)?;
                Some((u32::from(value), access))
            }
            _ => None,
        };
        if let Some((value, access)) = resolved {
            reads.push(value);
            accesses.push(access);
        }
    }
    Ok(ResolvedGrammarMemory { reads, accesses })
}

fn plan_grammar_blocks(
    events: &[GrammarEvent],
    blocks: &[[u64; 8]],
    args_base: u64,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    memory_accesses: &[LinearMemoryAccess],
) -> Result<Vec<GrammarBlockPlan>, WasmBuildError> {
    let mut memory_accesses = memory_accesses.iter();
    events
        .iter()
        .zip(blocks)
        .map(|(event, &block)| {
            let rows = event
                .block
                .iter()
                .zip(block)
                .map(|(source, value)| -> Result<_, WasmBuildError> {
                    let limb_variant = |limb| match limb {
                        crate::event_grammar::Limb::Lo => WasmGrammarRomVariant::LowLimb,
                        crate::event_grammar::Limb::Hi => WasmGrammarRomVariant::HighLimb,
                    };
                    let entry = |kind, arg, variant, const_lo, const_hi| crate::ir::WasmGrammarRomEntry {
                        kind,
                        arg,
                        variant,
                        const_lo,
                        const_hi,
                        advice: !event.absorb,
                    };
                    let base_slot_row = |rom| GrammarSlotRow {
                        value,
                        rom,
                        read: None,
                        write: None,
                        local_read: None,
                        local_write: None,
                        linear_memory: None,
                    };
                    Ok(match *source {
                        SlotSource::Const(constant) => base_slot_row(entry(
                            WasmGrammarSlotKind::Const,
                            0,
                            WasmGrammarRomVariant::None,
                            constant as u32,
                            (constant >> 32) as u32,
                        )),
                        SlotSource::ArgElem { arg, limb } => GrammarSlotRow {
                            read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                            ..base_slot_row(entry(WasmGrammarSlotKind::Arg, arg, limb_variant(limb), 0, 0))
                        },
                        // Each result lane is written by the slot absorbing
                        // it: the Lo slot is the push (a narrow total write,
                        // hi lane zeroed), the Hi slot writes only the
                        // pushed cell's hi word. The other lane is zero in
                        // both plans — its port is inactive on that row, so
                        // the value column is mechanically inert.
                        SlotSource::ResultElem { limb: Limb::Lo } => GrammarSlotRow {
                            write: Some((args_base, (result.expect("validated result").0, 0))),
                            ..base_slot_row(entry(
                                WasmGrammarSlotKind::Result,
                                0,
                                WasmGrammarRomVariant::LowLimb,
                                0,
                                0,
                            ))
                        },
                        SlotSource::ResultElem { limb: Limb::Hi } => GrammarSlotRow {
                            write: Some((args_base, (0, result.expect("validated result").1))),
                            ..base_slot_row(entry(
                                WasmGrammarSlotKind::Result,
                                0,
                                WasmGrammarRomVariant::HighLimb,
                                0,
                                0,
                            ))
                        },
                        SlotSource::Claim { idx } => base_slot_row(entry(
                            WasmGrammarSlotKind::Claim,
                            idx,
                            WasmGrammarRomVariant::None,
                            0,
                            0,
                        )),
                        SlotSource::MemoryRead32 { base, byte_offset }
                        | SlotSource::MemoryRead8 { base, byte_offset } => {
                            let (arg, variant) =
                                memory_rom_arg_variant(base, matches!(*source, SlotSource::MemoryRead8 { .. }));
                            let MemoryBase::Arg(_) = base else {
                                unreachable!("validated import memory base")
                            };
                            GrammarSlotRow {
                                read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved grammar memory access"),
                                ),
                                ..base_slot_row(entry(WasmGrammarSlotKind::MemoryRead, arg, variant, byte_offset, 0))
                            }
                        }
                        SlotSource::MemoryWrite32 {
                            claim,
                            base,
                            byte_offset,
                        }
                        | SlotSource::MemoryWrite8 {
                            claim,
                            base,
                            byte_offset,
                        } => {
                            let (arg, variant) =
                                memory_rom_arg_variant(base, matches!(*source, SlotSource::MemoryWrite8 { .. }));
                            let MemoryBase::Arg(_) = base else {
                                unreachable!("validated import memory base")
                            };
                            GrammarSlotRow {
                                read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved grammar memory access"),
                                ),
                                ..base_slot_row(entry(
                                    WasmGrammarSlotKind::MemoryWrite,
                                    arg,
                                    variant,
                                    byte_offset,
                                    u32::from(claim),
                                ))
                            }
                        }
                        SlotSource::ClaimLocal { .. } | SlotSource::OutputElem { .. } => {
                            unreachable!("validated: export sources never appear in import templates")
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GrammarBlockPlan {
                block,
                rows,
                absorb: event.absorb,
            })
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

pub(super) fn plan_export_blocks(
    events: &[GrammarEvent],
    blocks: &[[u64; 8]],
    locals: &[u32],
    memory_accesses: &[LinearMemoryAccess],
) -> Result<Vec<GrammarBlockPlan>, WasmBuildError> {
    let mut memory_accesses = memory_accesses.iter();
    events
        .iter()
        .zip(blocks)
        .map(|(event, &block)| {
            let rows = event
                .block
                .iter()
                .zip(block)
                .map(|(source, value)| -> Result<_, WasmBuildError> {
                    let limb_variant = |limb| match limb {
                        crate::event_grammar::Limb::Lo => WasmGrammarRomVariant::LowLimb,
                        crate::event_grammar::Limb::Hi => WasmGrammarRomVariant::HighLimb,
                    };
                    let entry = |kind, arg, variant| crate::ir::WasmGrammarRomEntry {
                        kind,
                        arg,
                        variant,
                        const_lo: 0,
                        const_hi: 0,
                        advice: false,
                    };
                    let base_slot_row = |rom| GrammarSlotRow {
                        value,
                        rom,
                        read: None,
                        write: None,
                        local_read: None,
                        local_write: None,
                        linear_memory: None,
                    };
                    Ok(match *source {
                        SlotSource::Const(constant) => base_slot_row(crate::ir::WasmGrammarRomEntry {
                            kind: WasmGrammarSlotKind::Const,
                            arg: 0,
                            variant: WasmGrammarRomVariant::None,
                            const_lo: constant as u32,
                            const_hi: (constant >> 32) as u32,
                            advice: false,
                        }),
                        SlotSource::Claim { idx } => {
                            base_slot_row(entry(WasmGrammarSlotKind::Claim, idx, WasmGrammarRomVariant::None))
                        }
                        SlotSource::ClaimLocal { local, limb, .. } => {
                            // expand_export_entry rejects words over 32 bits.
                            let bit = u8::from(matches!(limb, Limb::Hi));
                            GrammarSlotRow {
                                local_write: Some((u32::from(local), bit, value as u32)),
                                ..base_slot_row(entry(WasmGrammarSlotKind::ClaimLocal, local, limb_variant(limb)))
                            }
                        }
                        SlotSource::OutputElem { limb } => {
                            base_slot_row(entry(WasmGrammarSlotKind::Output, 0, limb_variant(limb)))
                        }
                        SlotSource::MemoryRead32 { base, byte_offset }
                        | SlotSource::MemoryRead8 { base, byte_offset } => {
                            let (local, variant) =
                                memory_rom_arg_variant(base, matches!(*source, SlotSource::MemoryRead8 { .. }));
                            let MemoryBase::Local(_) = base else {
                                unreachable!("validated export memory base")
                            };
                            let base_value = locals[usize::from(local)];
                            GrammarSlotRow {
                                local_read: Some((u32::from(local), base_value)),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved grammar memory access"),
                                ),
                                ..base_slot_row(crate::ir::WasmGrammarRomEntry {
                                    kind: WasmGrammarSlotKind::MemoryRead,
                                    arg: local,
                                    variant,
                                    const_lo: byte_offset,
                                    const_hi: 0,
                                    advice: false,
                                })
                            }
                        }
                        SlotSource::MemoryWrite32 {
                            claim,
                            base,
                            byte_offset,
                        }
                        | SlotSource::MemoryWrite8 {
                            claim,
                            base,
                            byte_offset,
                        } => {
                            let (local, variant) =
                                memory_rom_arg_variant(base, matches!(*source, SlotSource::MemoryWrite8 { .. }));
                            let MemoryBase::Local(_) = base else {
                                unreachable!("validated export memory base")
                            };
                            let base_value = locals[usize::from(local)];
                            GrammarSlotRow {
                                local_read: Some((u32::from(local), base_value)),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved grammar memory access"),
                                ),
                                ..base_slot_row(crate::ir::WasmGrammarRomEntry {
                                    kind: WasmGrammarSlotKind::MemoryWrite,
                                    arg: local,
                                    variant,
                                    const_lo: byte_offset,
                                    const_hi: u32::from(claim),
                                    advice: false,
                                })
                            }
                        }
                        SlotSource::ArgElem { .. } | SlotSource::ResultElem { .. } => {
                            unreachable!("validated: stack sources never appear in export templates")
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GrammarBlockPlan {
                block,
                rows,
                absorb: true,
            })
        })
        .collect()
}

/// Shared shape of the grammar/perm aux rows emitted outside the per-opcode
/// flow: the carried context every such row repeats.
pub(super) struct GrammarAuxCtx {
    pub(super) pc: u64,
    pub(super) sp: u64,
    pub(super) stack_frame_base: u64,
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
    /// The carried halted latch on both sides of these aux rows (true for
    /// post-halt exit gathers, false mid-turn).
    pub(super) halted: bool,
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
            stack_frame_base: self.stack_frame_base,
            output: self.output,
            call_stack_depth: self.call_stack_depth,
            memory_pages: self.memory_pages,
            max_memory_pages: self.max_memory_pages,
            locals_fbp: self.locals_fbp,
            halted: self.halted,
            trapped: false,
            param_init: WasmCountdownState::ZERO,
            tail_call_pending: false,
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
/// arg/result stack read, the result-lo stack WRITE that pushes the host
/// result, or the input-local lane write the slot claims). Absorbing blocks
/// then run their permutation group. A result push bumps `ctx.sp` for every
/// subsequent row.
pub(super) fn emit_block_plan(
    out: &mut Vec<WasmVmStep>,
    ctx: &mut GrammarAuxCtx,
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
            // Advice blocks advance the schedule without starting a permutation.
            if plan.absorb {
                absorb.perm_pending = true;
                absorb.perm_state = absorb_premix(*comm_chain, absorb.evbuf);
            }
            gstate.events_remaining -= 1;
            gstate.event_index += 1;
        }
        let read0 = slot.read.map(|(stack_slot, (lo, hi))| {
            StackValueAccess::new(stack_slot.saturating_mul(2), lo).with_optional_hi(Some(hi))
        });
        let write0 = slot.write.map(|(stack_slot, (lo, hi))| {
            StackValueAccess::new(stack_slot.saturating_mul(2), lo).with_optional_hi(Some(hi))
        });
        // Only the result LO slot pushes (the counted write port pair); the
        // HI slot fires the hi-word port alone and leaves sp untouched.
        let pushes = slot.write.is_some() && slot.rom.variant.is_low_limb();
        // Wide rows: a stack read/write carrying a hi limb, or a hi-lane
        // locals write (the narrow-row rule pins hi columns to zero
        // otherwise).
        let wide = slot.read.is_some_and(|(_, (_, hi))| hi != 0)
            || slot.write.is_some_and(|(_, (_, hi))| hi != 0)
            || slot.local_write.is_some_and(|(_, limb, _)| limb == 1);
        let state_before = ctx.state(ctx.host_args, *comm_chain, absorb_before, gstate_before);
        if pushes {
            ctx.sp += 1;
        }
        let state_after = ctx.state(ctx.host_args, *comm_chain, *absorb, *gstate);
        out.push(WasmVmStep {
            wide_values_enabled: wide,
            stack_reads_override: Some(u8::from(read0.is_some())),
            stack_writes_override: Some(u8::from(pushes)),
            stack_read0: read0,
            stack_write0: write0,
            linear_memory: slot.linear_memory,
            local_index: slot
                .local_write
                .map(|(index, _, _)| index)
                .or_else(|| slot.local_read.map(|(index, _)| index)),
            local_read_value: slot.local_read.map(|(_, value)| value),
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
                state_before,
                state_after,
            )
        });
    }
    if plan.absorb {
        emit_perm_group(out, ctx, comm_chain, absorb, *gstate);
    }
}
