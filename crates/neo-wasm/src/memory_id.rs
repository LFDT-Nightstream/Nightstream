/// Stable identity shared by relation declarations, preloads, and memory backends.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WasmMemoryId {
    CallStackCallerFbp,
    CallStackCallerSpBase,
    CallStackReturnPc,
    HostContextRefs,
    HostContextArgsDepth,
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
    HostEventSlotImmediate1,
    HostEventSlotImmediate0,
    HostEventSlotDescriptor,
    LinearMemory,
    PcRom,
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
    ProgramHeader,
    ProgramBrTableLen,
    ProgramRefFuncRef,
    ProgramTableDescriptor,
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
            | Self::ProgramTableDescriptor
            | Self::ProgramMemoryOffset
            | Self::ProgramI32ConstValue
            | Self::ProgramI64ConstValueLo
            | Self::ProgramRefFuncRef
            | Self::ProgramBrTableLen
            | Self::CallTarget => Self::ProgramImmediate,
            Self::ProgramI64ConstValueHi => Self::ProgramImmediateHi,
            memory => memory,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::CallStackCallerFbp => "call_stack_caller_fbp",
            Self::CallStackCallerSpBase => "call_stack_caller_sp_base",
            Self::CallStackReturnPc => "call_stack_return_pc",
            Self::HostContextRefs => "host_context_refs",
            Self::HostContextArgsDepth => "host_context_args_depth",
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
            Self::HostEventSlotImmediate1 => "host_event_slot_immediate_1",
            Self::HostEventSlotImmediate0 => "host_event_slot_immediate_0",
            Self::HostEventSlotDescriptor => "host_event_slot_descriptor",
            Self::LinearMemory => "linear_memory",
            Self::PcRom => "pc_rom",
            Self::ProgramGlobalIndex => "program_global_index",
            Self::ProgramI32ConstValue => "program_i32_const_value",
            Self::ProgramI64ConstValueHi => "program_i64_const_value_hi",
            Self::ProgramI64ConstValueLo => "program_i64_const_value_lo",
            Self::ProgramImmediate => "program_immediate",
            Self::ProgramImmediateHi => "program_immediate_hi",
            Self::ProgramLocalIndex => "program_local_index",
            Self::ProgramMemoryOffset => "program_memory_offset",
            Self::ProgramHeader => "program_header",
            Self::ProgramBrTableLen => "program_br_table_len",
            Self::ProgramRefFuncRef => "program_ref_func_ref",
            Self::ProgramTableDescriptor => "program_table_descriptor",
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
