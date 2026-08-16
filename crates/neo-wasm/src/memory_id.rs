/// Stable identity shared by relation declarations, preloads, and memory backends.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WasmMemoryId {
    CallStackCallerFbp,
    CallStackCallerSpBase,
    CallStackReturnPc,
    CallTarget,
    FunctionCallMetadata,
    FunctionEntry,
    FunctionLocalCount,
    FunctionType,
    GlobalLo,
    GlobalHi,
    HostEventExportEntryScheduleCount,
    HostEventExportExitScheduleCount,
    HostEventImportScheduleCount,
    HostEventSlotArg,
    HostEventSlotImmediate1,
    HostEventSlotImmediate0,
    HostEventSlotKind,
    HostEventSlotVariant,
    LinearMemory,
    LocalLo,
    LocalHi,
    ModuleType,
    PcEdgeKind,
    PcFunctionRef,
    PcRom,
    ProgramCallIndirectExpectedTypeId,
    ProgramCallIndirectTypeIndex,
    ProgramGlobalIndex,
    ProgramI32ConstValue,
    ProgramI64ConstValueHi,
    ProgramI64ConstValueLo,
    ProgramLocalIndex,
    ProgramMemoryOffset,
    ProgramOpcode,
    ProgramRefFuncRef,
    ProgramTableId,
    Stack,
    TableSize,
    TableElement,
}

impl WasmMemoryId {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CallStackCallerFbp => "call_stack_caller_fbp",
            Self::CallStackCallerSpBase => "call_stack_caller_sp_base",
            Self::CallStackReturnPc => "call_stack_return_pc",
            Self::CallTarget => "call_target",
            Self::FunctionCallMetadata => "function_call_metadata",
            Self::FunctionEntry => "function_entry",
            Self::FunctionLocalCount => "function_local_count",
            Self::FunctionType => "function_type",
            Self::GlobalLo => "global",
            Self::GlobalHi => "global_hi",
            Self::HostEventExportEntryScheduleCount => "host_event_export_entry_schedule_count",
            Self::HostEventExportExitScheduleCount => "host_event_export_exit_schedule_count",
            Self::HostEventImportScheduleCount => "host_event_import_schedule_count",
            Self::HostEventSlotArg => "host_event_slot_arg",
            Self::HostEventSlotImmediate1 => "host_event_slot_immediate_1",
            Self::HostEventSlotImmediate0 => "host_event_slot_immediate_0",
            Self::HostEventSlotKind => "host_event_slot_kind",
            Self::HostEventSlotVariant => "host_event_slot_variant",
            Self::LinearMemory => "linear_memory",
            Self::LocalLo => "local",
            Self::LocalHi => "local_hi",
            Self::ModuleType => "module_type",
            Self::PcEdgeKind => "pc_edge_kind",
            Self::PcFunctionRef => "pc_function_ref",
            Self::PcRom => "pc_rom",
            Self::ProgramCallIndirectExpectedTypeId => "program_call_indirect_expected_type_id",
            Self::ProgramCallIndirectTypeIndex => "program_call_indirect_type_index",
            Self::ProgramGlobalIndex => "program_global_index",
            Self::ProgramI32ConstValue => "program_i32_const_value",
            Self::ProgramI64ConstValueHi => "program_i64_const_value_hi",
            Self::ProgramI64ConstValueLo => "program_i64_const_value_lo",
            Self::ProgramLocalIndex => "program_local_index",
            Self::ProgramMemoryOffset => "program_memory_offset",
            Self::ProgramOpcode => "program_opcode",
            Self::ProgramRefFuncRef => "program_ref_func_ref",
            Self::ProgramTableId => "program_table_id",
            Self::Stack => "stack",
            Self::TableSize => "table_size",
            Self::TableElement => "table_element",
        }
    }

    pub const fn is_rom(self) -> bool {
        match self {
            Self::CallStackCallerFbp
            | Self::CallStackCallerSpBase
            | Self::CallStackReturnPc
            | Self::GlobalLo
            | Self::GlobalHi
            | Self::LinearMemory
            | Self::LocalLo
            | Self::LocalHi
            | Self::Stack
            | Self::TableSize
            | Self::TableElement => false,
            Self::CallTarget
            | Self::FunctionCallMetadata
            | Self::FunctionEntry
            | Self::FunctionLocalCount
            | Self::FunctionType
            | Self::HostEventExportEntryScheduleCount
            | Self::HostEventExportExitScheduleCount
            | Self::HostEventImportScheduleCount
            | Self::HostEventSlotArg
            | Self::HostEventSlotImmediate1
            | Self::HostEventSlotImmediate0
            | Self::HostEventSlotKind
            | Self::HostEventSlotVariant
            | Self::ModuleType
            | Self::PcEdgeKind
            | Self::PcFunctionRef
            | Self::PcRom
            | Self::ProgramCallIndirectExpectedTypeId
            | Self::ProgramCallIndirectTypeIndex
            | Self::ProgramGlobalIndex
            | Self::ProgramI32ConstValue
            | Self::ProgramI64ConstValueHi
            | Self::ProgramI64ConstValueLo
            | Self::ProgramLocalIndex
            | Self::ProgramMemoryOffset
            | Self::ProgramOpcode
            | Self::ProgramRefFuncRef
            | Self::ProgramTableId => true,
        }
    }
}

impl std::fmt::Display for WasmMemoryId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
