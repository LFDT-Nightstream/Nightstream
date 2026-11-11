/// Stable identity shared by relation declarations, preloads, and memory backends.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WasmMemoryId {
    CallStackCallerFbp,
    CallStackCallerSpBase,
    CallStackReturnPc,
    HostContextExport,
    HostContextImport,
    HostContextArgsBase,
    HostContextParentDepth,
    HostContextSp,
    CallTarget,
    FunctionCallMetadata,
    FunctionEntry,
    FunctionLocalCount,
    FunctionType,
    FrameSlideDistance,
    FrameSlideRemaining,
    GlobalLo,
    GlobalHi,
    HostEventBase,
    HostEventExportScheduleCount,
    HostEventImportScheduleCount,
    HostEventSlotArg,
    HostEventSlotImmediate1,
    HostEventSlotImmediate0,
    HostEventSlotKind,
    HostEventSlotVariant,
    LinearMemory,
    PcEdgeKind,
    PcFunctionRef,
    PcRom,
    ProgramCallIndirectExpectedTypeId,
    ProgramGlobalIndex,
    ProgramI32ConstValue,
    ProgramI64ConstValueHi,
    ProgramI64ConstValueLo,
    /// Table of the pc-keyed immediates that no two ports read at one pc.
    ProgramImmediate,
    /// Table of the second immediate of an instruction that has two.
    ProgramImmediateHi,
    ProgramLocalIndex,
    ProgramMemoryOffset,
    ProgramOpcode,
    ProgramBrTableLen,
    ProgramRefFuncRef,
    ProgramTableId,
    Stack,
    TableSize,
    TableElement,
}

impl WasmMemoryId {
    /// The table that holds this memory. Memories of one table have the same
    /// key and never own the same cell (`memory_semantics::table_preload`
    /// checks this for each program), so the catalog gives their ports to one
    /// memory. Each other memory is its own table.
    pub const fn table(self) -> Self {
        match self {
            Self::ProgramLocalIndex
            | Self::ProgramGlobalIndex
            | Self::ProgramTableId
            | Self::ProgramMemoryOffset
            | Self::ProgramI32ConstValue
            | Self::ProgramI64ConstValueLo
            | Self::ProgramRefFuncRef
            | Self::ProgramBrTableLen
            | Self::CallTarget => Self::ProgramImmediate,
            Self::ProgramCallIndirectExpectedTypeId | Self::ProgramI64ConstValueHi => Self::ProgramImmediateHi,
            memory => memory,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::CallStackCallerFbp => "call_stack_caller_fbp",
            Self::CallStackCallerSpBase => "call_stack_caller_sp_base",
            Self::CallStackReturnPc => "call_stack_return_pc",
            Self::HostContextExport => "host_context_export",
            Self::HostContextImport => "host_context_import",
            Self::HostContextArgsBase => "host_context_argsbase",
            Self::HostContextParentDepth => "host_context_parentdepth",
            Self::HostContextSp => "host_context_sp",
            Self::CallTarget => "call_target",
            Self::FunctionCallMetadata => "function_call_metadata",
            Self::FunctionEntry => "function_entry",
            Self::FunctionLocalCount => "function_local_count",
            Self::FunctionType => "function_type",
            Self::FrameSlideDistance => "frame_slide_distance",
            Self::FrameSlideRemaining => "frame_slide_remaining",
            Self::GlobalLo => "global",
            Self::GlobalHi => "global_hi",
            Self::HostEventBase => "host_event_base",
            Self::HostEventExportScheduleCount => "host_event_export_schedule_count",
            Self::HostEventImportScheduleCount => "host_event_import_schedule_count",
            Self::HostEventSlotArg => "host_event_slot_arg",
            Self::HostEventSlotImmediate1 => "host_event_slot_immediate_1",
            Self::HostEventSlotImmediate0 => "host_event_slot_immediate_0",
            Self::HostEventSlotKind => "host_event_slot_kind",
            Self::HostEventSlotVariant => "host_event_slot_variant",
            Self::LinearMemory => "linear_memory",
            Self::PcEdgeKind => "pc_edge_kind",
            Self::PcFunctionRef => "pc_function_ref",
            Self::PcRom => "pc_rom",
            Self::ProgramCallIndirectExpectedTypeId => "program_call_indirect_expected_type_id",
            Self::ProgramGlobalIndex => "program_global_index",
            Self::ProgramI32ConstValue => "program_i32_const_value",
            Self::ProgramI64ConstValueHi => "program_i64_const_value_hi",
            Self::ProgramI64ConstValueLo => "program_i64_const_value_lo",
            Self::ProgramImmediate => "program_immediate",
            Self::ProgramImmediateHi => "program_immediate_hi",
            Self::ProgramLocalIndex => "program_local_index",
            Self::ProgramMemoryOffset => "program_memory_offset",
            Self::ProgramOpcode => "program_opcode",
            Self::ProgramBrTableLen => "program_br_table_len",
            Self::ProgramRefFuncRef => "program_ref_func_ref",
            Self::ProgramTableId => "program_table_id",
            Self::Stack => "stack",
            Self::TableSize => "table_size",
            Self::TableElement => "table_element",
        }
    }
}

impl std::fmt::Display for WasmMemoryId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
