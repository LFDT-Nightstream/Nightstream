//! Host-event emission: how expanded blocks become trace rows.
//!
//! Owns the gather aux-row plans and their emission — the per-slot
//! gather plans (`EventSlotRow`/`EventBlockPlan`/`HostCallEventPlan`) and
//! the shared row context (`HostEventRowContext`) every emitted aux row
//! repeats. Does not
//! own the trace state machine or per-opcode normalization — that is
//! `trace_build`, which drives these emitters.

use crate::comm_chain::absorb_block;
use crate::host_event_bindings::{
    expand_import_phase, memory_rom_arg_variant, opaque_control_encoding, EventBlock, Limb, MemoryBase, SlotBinding,
};
use crate::ir::{
    LinearMemoryAccess, StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmCountdownState, WasmEventAbsorbState,
    WasmFrameMoveState, WasmHostEventMemoryWidth, WasmHostEventRomVariant, WasmHostEventSlotKind, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode};

use super::super::memory_address::memory_pointer;
use super::memory::LinearMemoryImage;

/// One gather row's plan: the staged word, its expected host-event ROM entry,
/// and its stack/locals effect — an addressed read `(stack slot, (lo, hi))`
/// for arg slots, a stack WRITE for result slots (the Lo slot is the push,
/// the Hi slot writes only the cell's hi word; the inactive lane is zero),
/// an entry-frame locals access, or one static linear-memory access.
#[derive(Clone)]
pub(super) struct EventSlotRow {
    pub(super) value: u64,
    pub(super) rom: crate::ir::WasmHostEventRomEntry,
    pub(super) read: Option<(u64, (u32, u32))>,
    pub(super) write: Option<(u64, (u32, u32))>,
    pub(super) local_read: Option<(u32, (u32, u32))>,
    pub(super) local_write: Option<(u32, u8, u32)>,
    pub(super) linear_memory: Option<LinearMemoryAccess>,
    pub(super) wide: bool,
}

/// One ROM group's slot plans and absorb policy. Enter ends a prefix group early.
#[derive(Clone)]
pub(super) struct EventBlockPlan {
    pub(super) rows: Vec<EventSlotRow>,
    /// Whether the staged block enters the transcript.
    pub(super) absorb: bool,
}

pub(super) struct HostCallEventPlan {
    pub(super) blocks: Vec<EventBlockPlan>,
    pub(super) args_base: u64,
}

pub(super) struct ResolvedEventMemory {
    pub(super) reads: Vec<u32>,
    pub(super) accesses: Vec<LinearMemoryAccess>,
}

pub(super) fn plan_import_phase(
    events: &[EventBlock],
    args_base: u64,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    result_is_i64: bool,
    inputs: &[u64],
    memory_pages: Option<u32>,
    memory: &mut LinearMemoryImage,
) -> Result<HostCallEventPlan, WasmBuildError> {
    let resolved_memory = resolve_import_memory(events, args, inputs, memory_pages, memory)?;
    let blocks = expand_import_phase(events, args, result, inputs, &resolved_memory.reads)?;
    Ok(HostCallEventPlan {
        blocks: plan_event_blocks(
            events,
            &blocks,
            args_base,
            args,
            result,
            result_is_i64,
            &resolved_memory.accesses,
        )?,
        args_base,
    })
}

fn resolve_import_memory(
    events: &[EventBlock],
    import_args: &[(u32, u32)],
    inputs: &[u64],
    memory_pages: Option<u32>,
    memory: &mut LinearMemoryImage,
) -> Result<ResolvedEventMemory, WasmBuildError> {
    let mut reads = Vec::new();
    let mut accesses = Vec::new();
    for source in events.iter().flat_map(|event| &event.block) {
        match *source {
            SlotBinding::MemoryRead32 { base, byte_offset } => {
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                let (value, access) = memory.read_aligned_word(pointer, byte_offset, memory_pages)?;
                reads.push(value);
                accesses.push(access);
            }
            SlotBinding::MemoryRead8 { base, byte_offset } => {
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                let (value, access) = memory.read_byte(pointer, byte_offset, memory_pages)?;
                reads.push(u32::from(value));
                accesses.push(access);
            }
            SlotBinding::MemoryRead16 { base, byte_offset } => {
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                let (value, access) = memory.read_half(pointer, byte_offset, memory_pages)?;
                reads.push(u32::from(value));
                accesses.push(access);
            }
            SlotBinding::MemoryWrite32 {
                input,
                base,
                byte_offset,
            } => {
                let value = inputs.get(usize::from(input)).copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!("host-event memory write references missing input {input}"))
                })?;
                let value = u32::try_from(value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u32".to_string()))?;
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                accesses.push(memory.write_aligned_word(pointer, byte_offset, value, memory_pages)?);
            }
            SlotBinding::MemoryWrite8 {
                input,
                base,
                byte_offset,
            } => {
                let value = inputs.get(usize::from(input)).copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!("host-event memory write references missing input {input}"))
                })?;
                let value = u8::try_from(value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u8".to_string()))?;
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                accesses.push(memory.write_byte(pointer, byte_offset, value, memory_pages)?);
            }
            SlotBinding::MemoryWrite16 {
                input,
                base,
                byte_offset,
            } => {
                let value = inputs.get(usize::from(input)).copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!("host-event memory write references missing input {input}"))
                })?;
                let value = u16::try_from(value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u16".to_string()))?;
                let MemoryBase::Arg(arg) = base else {
                    unreachable!("validated import memory base")
                };
                let pointer = memory_pointer(import_args, arg, "import argument")?;
                accesses.push(memory.write_half(pointer, byte_offset, value, memory_pages)?);
            }
            _ => {}
        }
    }
    Ok(ResolvedEventMemory { reads, accesses })
}

fn event_memory_width(source: SlotBinding) -> WasmHostEventMemoryWidth {
    match source {
        SlotBinding::MemoryRead8 { .. } | SlotBinding::MemoryWrite8 { .. } => WasmHostEventMemoryWidth::Byte,
        SlotBinding::MemoryRead16 { .. } | SlotBinding::MemoryWrite16 { .. } => WasmHostEventMemoryWidth::Half,
        SlotBinding::MemoryRead32 { .. } | SlotBinding::MemoryWrite32 { .. } => WasmHostEventMemoryWidth::Word,
        _ => unreachable!("non-memory host-event slot"),
    }
}

pub(super) fn apply_export_entry_memory(
    events: &[EventBlock],
    blocks: &[[u64; 8]],
    locals: &[(u32, u32)],
    memory_pages: Option<u32>,
    memory: &mut LinearMemoryImage,
) -> Result<Vec<LinearMemoryAccess>, WasmBuildError> {
    let mut accesses = Vec::new();
    for (source, value) in events
        .iter()
        .zip(blocks)
        .flat_map(|(event, block)| event.block.iter().zip(block))
    {
        match *source {
            SlotBinding::MemoryWrite32 { base, byte_offset, .. } => {
                let value = u32::try_from(*value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u32".to_string()))?;
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = memory_pointer(locals, local, "entry local")?;
                accesses.push(memory.write_aligned_word(pointer, byte_offset, value, memory_pages)?);
            }
            SlotBinding::MemoryWrite8 { base, byte_offset, .. } => {
                let value = u8::try_from(*value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u8".to_string()))?;
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = memory_pointer(locals, local, "entry local")?;
                accesses.push(memory.write_byte(pointer, byte_offset, value, memory_pages)?);
            }
            SlotBinding::MemoryWrite16 { base, byte_offset, .. } => {
                let value = u16::try_from(*value)
                    .map_err(|_| WasmBuildError::Trace("host-event memory write value does not fit u16".to_string()))?;
                let MemoryBase::Local(local) = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = memory_pointer(locals, local, "entry local")?;
                accesses.push(memory.write_half(pointer, byte_offset, value, memory_pages)?);
            }
            _ => {}
        }
    }
    Ok(accesses)
}

pub(super) fn read_export_exit_memory(
    events: &[EventBlock],
    output: Option<(u32, u32)>,
    memory_pages: Option<u32>,
    memory: &LinearMemoryImage,
) -> Result<ResolvedEventMemory, WasmBuildError> {
    let mut reads = Vec::new();
    let mut accesses = Vec::new();

    let output_pointer = events
        .iter()
        .flat_map(|event| &event.block)
        .any(|source| {
            matches!(
                source,
                SlotBinding::MemoryRead32 { .. } | SlotBinding::MemoryRead16 { .. } | SlotBinding::MemoryRead8 { .. }
            )
        })
        .then(|| export_memory_pointer(output))
        .transpose()?;

    for source in events.iter().flat_map(|event| &event.block) {
        let resolved = match *source {
            SlotBinding::MemoryRead32 { base, byte_offset } => {
                let MemoryBase::Output = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = output_pointer.expect("memory read requires resolved output pointer");
                let (value, access) = memory.read_aligned_word(pointer, byte_offset, memory_pages)?;
                Some((value, access))
            }
            SlotBinding::MemoryRead8 { base, byte_offset } => {
                let MemoryBase::Output = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = output_pointer.expect("memory read requires resolved output pointer");
                let (value, access) = memory.read_byte(pointer, byte_offset, memory_pages)?;
                Some((u32::from(value), access))
            }
            SlotBinding::MemoryRead16 { base, byte_offset } => {
                let MemoryBase::Output = base else {
                    unreachable!("validated export memory base")
                };
                let pointer = output_pointer.expect("memory read requires resolved output pointer");
                let (value, access) = memory.read_half(pointer, byte_offset, memory_pages)?;
                Some((u32::from(value), access))
            }
            _ => None,
        };
        if let Some((value, access)) = resolved {
            reads.push(value);
            accesses.push(access);
        }
    }
    Ok(ResolvedEventMemory { reads, accesses })
}

fn export_memory_pointer(output: Option<(u32, u32)>) -> Result<u32, WasmBuildError> {
    let (pointer, high) = output
        .ok_or_else(|| WasmBuildError::Trace("export exit memory requires a captured output pointer".to_string()))?;

    if high != 0 {
        return Err(WasmBuildError::Trace(format!(
            "export exit memory output is not a wasm32 pointer: high limb is {high}"
        )));
    }

    Ok(pointer)
}

fn plan_event_blocks(
    events: &[EventBlock],
    blocks: &[[u64; 8]],
    args_base: u64,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
    result_is_i64: bool,
    memory_accesses: &[LinearMemoryAccess],
) -> Result<Vec<EventBlockPlan>, WasmBuildError> {
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
                        crate::host_event_bindings::Limb::Lo => WasmHostEventRomVariant::LowLimb,
                        crate::host_event_bindings::Limb::Hi => WasmHostEventRomVariant::HighLimb,
                    };
                    let entry = |kind, arg, variant, immediate0, immediate1| crate::ir::WasmHostEventRomEntry {
                        kind,
                        arg,
                        variant,
                        immediate0,
                        immediate1,
                        advice: !event.absorb,
                    };
                    let base_slot_row = |rom| EventSlotRow {
                        value,
                        rom,
                        read: None,
                        write: None,
                        local_read: None,
                        local_write: None,
                        linear_memory: None,
                        wide: false,
                    };
                    Ok(match *source {
                        SlotBinding::EnterOpaque | SlotBinding::OpaqueSaved { .. } | SlotBinding::OpaqueRoot { .. } => {
                            let (kind, arg) = opaque_control_encoding(source).expect("control slot");
                            base_slot_row(entry(kind, arg, WasmHostEventRomVariant::None, 0, 0))
                        }
                        SlotBinding::Const(constant) => base_slot_row(entry(
                            WasmHostEventSlotKind::Const,
                            0,
                            WasmHostEventRomVariant::None,
                            constant as u32,
                            (constant >> 32) as u32,
                        )),
                        SlotBinding::ArgElem { arg, limb } => EventSlotRow {
                            read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                            ..base_slot_row(entry(WasmHostEventSlotKind::Arg, arg, limb_variant(limb), 0, 0))
                        },
                        // Each result lane is written by the slot absorbing
                        // it: the Lo slot is the push (a narrow total write,
                        // hi lane zeroed), the Hi slot writes only the
                        // pushed cell's hi word. The other lane is zero in
                        // both plans — its port is inactive on that row, so
                        // the value column is mechanically inert.
                        SlotBinding::ResultElem { limb: Limb::Lo } => EventSlotRow {
                            write: Some((args_base, (result.expect("validated result").0, 0))),
                            ..base_slot_row(entry(
                                WasmHostEventSlotKind::Result,
                                0,
                                WasmHostEventRomVariant::LowLimb,
                                1,
                                0,
                            ))
                        },
                        SlotBinding::ResultElem { limb: Limb::Hi } => EventSlotRow {
                            write: Some((args_base, (0, result.expect("validated result").1))),
                            wide: result_is_i64,
                            ..base_slot_row(entry(
                                WasmHostEventSlotKind::Result,
                                0,
                                WasmHostEventRomVariant::HighLimb,
                                1,
                                u32::from(result_is_i64),
                            ))
                        },
                        SlotBinding::MemoryRead32 { base, byte_offset }
                        | SlotBinding::MemoryRead16 { base, byte_offset }
                        | SlotBinding::MemoryRead8 { base, byte_offset } => {
                            let (arg, variant) = memory_rom_arg_variant(base, event_memory_width(*source));
                            let MemoryBase::Arg(_) = base else {
                                unreachable!("validated import memory base")
                            };
                            EventSlotRow {
                                read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved host-event memory access"),
                                ),
                                ..base_slot_row(entry(WasmHostEventSlotKind::MemoryRead, arg, variant, byte_offset, 0))
                            }
                        }
                        SlotBinding::MemoryWrite32 {
                            input,
                            base,
                            byte_offset,
                        }
                        | SlotBinding::MemoryWrite16 {
                            input,
                            base,
                            byte_offset,
                        }
                        | SlotBinding::MemoryWrite8 {
                            input,
                            base,
                            byte_offset,
                        } => {
                            let (arg, variant) = memory_rom_arg_variant(base, event_memory_width(*source));
                            let MemoryBase::Arg(_) = base else {
                                unreachable!("validated import memory base")
                            };
                            EventSlotRow {
                                read: Some((args_base + u64::from(arg), args[usize::from(arg)])),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved host-event memory access"),
                                ),
                                ..base_slot_row(entry(
                                    WasmHostEventSlotKind::MemoryWrite,
                                    arg,
                                    variant,
                                    byte_offset,
                                    u32::from(input),
                                ))
                            }
                        }
                        SlotBinding::InputLocal { .. } | SlotBinding::OutputElem { .. } => {
                            unreachable!("validated: export sources never appear in import templates")
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(EventBlockPlan {
                rows,
                absorb: event.absorb,
            })
        })
        .collect()
}

pub(super) fn plan_export_blocks(
    events: &[EventBlock],
    blocks: &[[u64; 8]],
    locals: &[(u32, u32)],
    local_is_i64: &[bool],
    memory_accesses: &[LinearMemoryAccess],
) -> Result<Vec<EventBlockPlan>, WasmBuildError> {
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
                        crate::host_event_bindings::Limb::Lo => WasmHostEventRomVariant::LowLimb,
                        crate::host_event_bindings::Limb::Hi => WasmHostEventRomVariant::HighLimb,
                    };
                    let entry = |kind, arg, variant| crate::ir::WasmHostEventRomEntry {
                        kind,
                        arg,
                        variant,
                        immediate0: 0,
                        immediate1: 0,
                        advice: !event.absorb,
                    };
                    let base_slot_row = |rom| EventSlotRow {
                        value,
                        rom,
                        read: None,
                        write: None,
                        local_read: None,
                        local_write: None,
                        linear_memory: None,
                        wide: false,
                    };
                    Ok(match *source {
                        SlotBinding::EnterOpaque | SlotBinding::OpaqueSaved { .. } | SlotBinding::OpaqueRoot { .. } => {
                            let (kind, arg) = opaque_control_encoding(source).expect("control slot");
                            base_slot_row(entry(kind, arg, WasmHostEventRomVariant::None))
                        }
                        SlotBinding::Const(constant) => base_slot_row(crate::ir::WasmHostEventRomEntry {
                            kind: WasmHostEventSlotKind::Const,
                            arg: 0,
                            variant: WasmHostEventRomVariant::None,
                            immediate0: constant as u32,
                            immediate1: (constant >> 32) as u32,
                            advice: !event.absorb,
                        }),
                        SlotBinding::InputLocal { local, limb, .. } => {
                            // expand_export_entry rejects words over 32 bits.
                            let bit = u8::from(matches!(limb, Limb::Hi));
                            EventSlotRow {
                                local_write: Some((u32::from(local), bit, value as u32)),
                                wide: local_is_i64[usize::from(local)],
                                ..base_slot_row(crate::ir::WasmHostEventRomEntry {
                                    immediate0: 1,
                                    immediate1: u32::from(local_is_i64[usize::from(local)]),
                                    ..entry(WasmHostEventSlotKind::InputLocal, local, limb_variant(limb))
                                })
                            }
                        }
                        SlotBinding::OutputElem { limb } => {
                            base_slot_row(entry(WasmHostEventSlotKind::Output, 0, limb_variant(limb)))
                        }
                        SlotBinding::MemoryRead32 { base, byte_offset }
                        | SlotBinding::MemoryRead16 { base, byte_offset }
                        | SlotBinding::MemoryRead8 { base, byte_offset } => {
                            let (arg, variant) = memory_rom_arg_variant(base, event_memory_width(*source));
                            let MemoryBase::Output = base else {
                                unreachable!("validated export memory base")
                            };
                            EventSlotRow {
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved host-event memory access"),
                                ),
                                ..base_slot_row(crate::ir::WasmHostEventRomEntry {
                                    kind: WasmHostEventSlotKind::MemoryRead,
                                    arg,
                                    variant,
                                    immediate0: byte_offset,
                                    immediate1: 0,
                                    advice: !event.absorb,
                                })
                            }
                        }
                        SlotBinding::MemoryWrite32 {
                            input,
                            base,
                            byte_offset,
                        }
                        | SlotBinding::MemoryWrite16 {
                            input,
                            base,
                            byte_offset,
                        }
                        | SlotBinding::MemoryWrite8 {
                            input,
                            base,
                            byte_offset,
                        } => {
                            let (local, variant) = memory_rom_arg_variant(base, event_memory_width(*source));
                            let MemoryBase::Local(_) = base else {
                                unreachable!("validated export memory base")
                            };
                            memory_pointer(locals, local, "entry local")?;
                            let base_lanes = locals[usize::from(local)];
                            EventSlotRow {
                                local_read: Some((u32::from(local), base_lanes)),
                                linear_memory: Some(
                                    *memory_accesses
                                        .next()
                                        .expect("resolved host-event memory access"),
                                ),
                                ..base_slot_row(crate::ir::WasmHostEventRomEntry {
                                    kind: WasmHostEventSlotKind::MemoryWrite,
                                    arg: local,
                                    variant,
                                    immediate0: byte_offset,
                                    immediate1: u32::from(input),
                                    advice: !event.absorb,
                                })
                            }
                        }
                        SlotBinding::ArgElem { .. } | SlotBinding::ResultElem { .. } => {
                            unreachable!("validated: stack sources never appear in export templates")
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(EventBlockPlan {
                rows,
                absorb: event.absorb,
            })
        })
        .collect()
}

/// Shared shape of the gather rows emitted outside the per-opcode
/// flow: the carried context every such row repeats.
pub(super) struct HostEventRowContext {
    pub(super) pc: u64,
    pub(super) sp: u64,
    pub(super) stack_frame_base: u64,
    pub(super) output: WasmOutputState,
    pub(super) call_stack_depth: u64,
    pub(super) memory_pages: Option<u32>,
    pub(super) max_memory_pages: Option<u32>,
    pub(super) locals_fbp: u64,
    pub(super) host_callee_fref: u32,
    pub(super) current_function_ref: u32,
    /// The carried activation_finished latch on both sides of these aux rows (true for
    /// post-return exit gathers, false mid-turn).
    pub(super) activation_finished: bool,
}

impl HostEventRowContext {
    pub(super) fn state(
        &self,
        comm_chain: [u64; 4],
        event_absorb: WasmEventAbsorbState,
        host_events: crate::ir::WasmHostEventState,
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
            activation_finished: self.activation_finished,
            trapped: false,
            frame_move: WasmFrameMoveState::ZERO,
            local_zero: WasmCountdownState::ZERO,
            host_callee_fref: self.host_callee_fref,
            comm_chain,
            event_absorb,
            host_events,
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
            br_table_len: 0,
            pc_edge_kind: WasmPcEdgeKind::Static,
            wide_values_enabled: false,
            opcode: WasmOpcode::Nop,
            info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
            stack_reads_override: Some(0),
            stack_writes_override: Some(0),
            output_captured: false,
            current_function_ref: self.current_function_ref,
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
            expected_type_id: None,
            table_size: None,
            call_param_count: None,
            call_result_count: None,
            call_stack_push: None,
            call_stack_pop: None,
            host_context: None,
            host_event_return_schedule_count: None,
            host_event_rom_slot: None,
            host_event_word: None,
            host_event_initial_schedule_count: None,
            host_event_exit_schedule_count: None,
        }
    }
}

/// Emit one host-event group: up to eight gather rows (one word each, with the
/// arg/result stack read, the result-lo stack WRITE that pushes the host
/// result, or the input-local lane write the slot inputs). The slot that fills
/// an absorbing block folds it into the chain on its own row and clears the
/// buffer; Enter ends a prefix group without hashing.
/// A result push bumps `ctx.sp` for every subsequent row.
pub(super) fn emit_block_plan(
    out: &mut Vec<WasmVmStep>,
    ctx: &mut HostEventRowContext,
    comm_chain: &mut [u64; 4],
    absorb: &mut WasmEventAbsorbState,
    host_event_state: &mut crate::ir::WasmHostEventState,
    plan: &EventBlockPlan,
) {
    for (word, slot) in plan.rows.iter().enumerate() {
        let chain_before = *comm_chain;
        let absorb_before = *absorb;
        let host_event_state_before = *host_event_state;
        absorb.evbuf[word] = slot.value;
        let enter_opaque = slot.rom.kind == WasmHostEventSlotKind::EnterOpaque;
        if enter_opaque {
            absorb.outer_prefix.copy_from_slice(&absorb.evbuf[..4]);
            absorb.outer_chain = *comm_chain;
            // TODO: Consider a verifier-authored, ROM-bound initial chain for EnterOpaque,
            // with matching circuit and native-root semantics. Const slots only initialize evbuf.
            *comm_chain = [0; 4];
            absorb.object_active = true;
        }
        if slot.rom.kind == WasmHostEventSlotKind::OpaqueRoot {
            debug_assert_eq!(
                slot.value,
                comm_chain[usize::from(slot.rom.arg)],
                "replayed opaque root must match the emitted object chain"
            );
            if slot.rom.arg == 3 {
                *comm_chain = absorb.outer_chain;
                absorb.object_active = false;
            }
        }
        host_event_state.slot_cursor = if enter_opaque { 0 } else { ((word + 1) % 8) as u8 };
        if word == 7 || enter_opaque {
            // Advice blocks advance the schedule without hashing.
            if plan.absorb && !enter_opaque {
                *comm_chain = absorb_block(*comm_chain, absorb.evbuf);
                absorb.evbuf = [0; 8];
            }
            host_event_state.events_remaining -= 1;
            host_event_state.event_index += 1;
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
        // Wide rows: a stack read/write carrying a hi limb, or an input-local
        // row whose verifier-owned static type is i64. The narrow-row rule
        // pins high columns to zero otherwise.
        let wide = slot.wide
            || slot.read.is_some_and(|(_, (_, hi))| hi != 0)
            || slot.write.is_some_and(|(_, (_, hi))| hi != 0);
        let state_before = ctx.state(chain_before, absorb_before, host_event_state_before);
        if pushes {
            ctx.sp += 1;
        }
        let state_after = ctx.state(*comm_chain, *absorb, *host_event_state);
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
            local_read_value: slot.local_read.map(|(_, (lo, _))| lo),
            local_read_value_hi: slot.local_read.map(|(_, (_, hi))| hi),
            // Lo rows write (word, 0); hi rows write only the hi lane.
            local_write_value: slot
                .local_write
                .map(|(_, limb, value)| if limb == 0 { value } else { 0 }),
            local_write_value_hi: slot
                .local_write
                .map(|(_, limb, value)| if limb == 0 { 0 } else { value }),
            host_context: None,
            host_event_return_schedule_count: None,
            host_event_rom_slot: Some(slot.rom),
            host_event_word: Some(slot.value),
            ..ctx.row(
                out.len() as u64,
                WasmAuxOpcode::HostEventGather,
                state_before,
                state_after,
            )
        });
        if enter_opaque {
            break;
        }
    }
}
