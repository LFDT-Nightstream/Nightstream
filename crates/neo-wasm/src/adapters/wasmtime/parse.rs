//! Static parsing of a wasm (or first embedded core module of a component)
//! binary into the artifacts the tracer needs before execution: the
//! `(defined_function_index, pc)` opcode/immediate map, the pc-ROM control
//! graph, and per-function type/arity/locals metadata.
//!
//! Owns the one-time binary walk and the normalized-type-id assignment. Holds
//! no live Wasmtime `Store` and emits no trace rows — it only produces the
//! immutable lookup tables the parent module and `normalize` consult at trace
//! time.

use super::decode::{
    decode_control_opcode, decode_memory_opcode, decode_opcode, ControlFrame, ControlFrameKind, DecodedOpcode,
};
use crate::ir::{WasmBuildError, WasmPcEdgeKind};
use crate::isa::{opcode_code, WasmOpcode};
use crate::layout::CALL_RETURN_PC_CHOICE;
use std::collections::BTreeMap;
use wasmparser::{Parser, Payload};

#[derive(Clone, Debug)]
pub struct WasmProgramArtifacts {
    /// Verifier/proof-bound static tables derived only from the wasm program.
    pub tables: WasmProgramTables,
    // Adapter-only helper state needed to turn Wasmtime debug frames into trace rows.
    pub(crate) trace: WasmTraceLoweringTables,
}

#[derive(Clone, Debug)]
pub struct WasmProgramTables {
    /// Initial page count for default linear memory 0, or `None` when the
    /// module has no default memory. This seeds the VM boundary state; data
    /// segment contents are tracked separately in `linear_memory_init`.
    pub initial_memory_pages: Option<u32>,
    /// Static per-PC decode rows. These bind each real program row to the
    /// opcode and immediate-bearing witness columns that are derived from the
    /// wasm bytes. Non-applicable immediate slots are encoded as zero.
    pub program_decode: Vec<WasmProgramDecodeEntry>,
    /// Static next-PC ROM rows `(pc_before, control_choice, pc_after)`.
    /// Program rows read this table to bind the witnessed PC transition to
    /// the parsed control-flow graph.
    pub pc_rom: Vec<(u64, u64, u64)>,
    /// Static `(pc_before, edge_kind)` rows classifying how a PC chooses its
    /// successor, such as fallthrough, conditional branch, call, or return.
    pub pc_edge_kinds: Vec<(u64, u64)>,
    /// Static `(pc_before, function_ref)` rows binding each code PC to the
    /// normalized function reference of its containing function.
    pub pc_function_refs: Vec<(u64, u64)>,
    /// Static `(function_ref, entry_pc)` rows keyed by the normalized funcref
    /// id space used in tables and direct/indirect calls.
    pub function_entries: Vec<(u64, u64)>,
    /// Static `(function_ref, type_id)` rows for call-indirect signature
    /// checks. `type_id` is a normalized id assigned by signature shape.
    pub function_types: Vec<(u64, u64)>,
    /// Static `(function_ref, param_count)` rows used by call-frame and
    /// parameter-initialization bookkeeping.
    pub function_param_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, result_count)` rows used by call-frame return
    /// bookkeeping.
    pub function_result_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, params_plus_declared_locals)` rows used to bind
    /// frame-base transitions and local-slot addressing.
    pub function_local_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, is_guest)` rows distinguishing guest wasm
    /// functions from imported host functions at call boundaries.
    pub function_guest_flags: Vec<(u64, u64)>,
    /// Static `(pc_before, function_ref)` rows for direct `call` targets.
    pub call_targets: Vec<(u64, u64)>,
    /// Static `(raw_type_index, expected_type_id)` rows mapping module type
    /// section indexes to normalized type ids for `call_indirect`.
    pub module_types: Vec<(u64, u64)>,
    /// Bytes initialized by active `(data ...)` segments at module
    /// instantiation, as `(byte_addr, byte_value)` pairs. Used by the memory
    /// sanity checker to seed `linear_memory` so the RMW Read check at
    /// data-initialized addresses sees the correct prior value (instead of
    /// the zero-default that applies to bytes outside any data segment).
    /// Only active segments targeting memory 0 with an `i32.const` offset are
    /// recorded; passive segments and globalexpr offsets are skipped.
    pub linear_memory_init: Vec<(u64, u8)>,
    /// Initial declared-global values as `(global_index, lo_limb, hi_limb)`.
    /// Imported globals are not represented here.
    pub globals_init: Vec<(u32, u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmProgramDecodeEntry {
    pub pc: u64,
    pub opcode_code: u32,
    pub local_index: u32,
    pub global_index: u32,
    pub table_id: u32,
    pub memory_offset: u32,
    pub call_indirect_type_index: u32,
    /// Normalized type id derived from the same module type-section entry
    /// checked by the `module_types` ROM, but keyed by program PC so the
    /// static opcode/immediate row is bound directly to the wasm bytes.
    pub call_indirect_expected_type_id: u32,
    pub i32_const_value: u32,
    pub i64_const_value_lo: u32,
    pub i64_const_value_hi: u32,
    pub ref_func_ref: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct WasmTraceLoweringTables {
    /// Static decode table keyed by Wasmtime's `(defined_function_index, pc)`
    /// pair so guest-debug frames can recover opcode/immediate metadata
    /// without reparsing at trace time.
    pub(crate) opcode_map: BTreeMap<(u32, u32), DecodedOpcode>,
    /// Count of imported functions used to normalize defined function refs
    /// into the same 1-based funcref id space as runtime table values.
    pub(crate) imported_function_count: u32,
    /// Per-function metadata used by the Wasmtime debug adapter while turning
    /// raw frames into trace rows; verifier-facing function tables are derived
    /// from this map.
    pub(crate) function_metas: BTreeMap<u32, ParsedFunctionMeta>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ParsedFunctionMeta {
    pub(crate) type_id: u32,
    pub(crate) param_count: u8,
    pub(crate) result_count: u8,
    num_locals: u32,
    entry_pc: Option<u64>,
}

pub(super) fn parse_wasm_artifacts(wasm_bytes: &[u8]) -> Result<WasmProgramArtifacts, WasmBuildError> {
    let mut builder = ParsedWasmArtifactsBuilder::default();
    for payload in Parser::new(0).parse_all(wasm_bytes) {
        let payload = payload.map_err(|err| WasmBuildError::Trace(format!("failed to parse wasm payload: {err}")))?;
        builder.consume_payload(payload)?;
    }
    builder.finish()
}

pub(crate) fn parse_first_component_core_module_artifacts(
    component_bytes: &[u8],
) -> Result<WasmProgramArtifacts, WasmBuildError> {
    let mut builder = ParsedWasmArtifactsBuilder::default();
    let mut inside_first_module = false;

    for payload in Parser::new(0).parse_all(component_bytes) {
        let payload =
            payload.map_err(|err| WasmBuildError::Trace(format!("failed to parse component payload: {err}")))?;
        match payload {
            // Current component tests execute the first embedded core module directly. Multi-module
            // components need an explicit resolution step for the actually executed core module.
            Payload::ModuleSection { .. } if !inside_first_module => inside_first_module = true,
            Payload::End(_) if inside_first_module => return builder.finish(),
            _ if inside_first_module => builder.consume_payload(payload)?,
            _ => {}
        }
    }

    Err(WasmBuildError::Unsupported(
        "component did not contain any embedded core module sections".to_string(),
    ))
}

struct ParsedWasmArtifactsBuilder {
    opcode_map: BTreeMap<(u32, u32), DecodedOpcode>,
    pc_rom: Vec<(u64, u64, u64)>,
    program_decode: Vec<WasmProgramDecodeEntry>,
    pc_edge_kinds: Vec<(u64, u64)>,
    pc_function_refs: Vec<(u64, u64)>,
    unresolved_call_edges: Vec<(u64, u32)>,
    function_entries: Vec<(u64, u64)>,
    call_targets: Vec<(u64, u64)>,
    function_metas: BTreeMap<u32, ParsedFunctionMeta>,
    defined_function_index: u32,
    imported_function_count: u32,
    raw_type_id_by_index: BTreeMap<u32, u32>,
    raw_type_shape_by_index: BTreeMap<u32, (u8, u8)>,
    signature_ids: BTreeMap<String, u32>,
    next_type_id: u32,
    defined_function_type_indices: Vec<u32>,
    initial_memory_pages: Option<u32>,
    linear_memory_init: Vec<(u64, u8)>,
    globals_init: Vec<(u32, u32, u32)>,
    next_declared_global_index: u32,
}

impl Default for ParsedWasmArtifactsBuilder {
    fn default() -> Self {
        Self {
            opcode_map: BTreeMap::new(),
            pc_rom: Vec::new(),
            program_decode: Vec::new(),
            pc_edge_kinds: Vec::new(),
            pc_function_refs: Vec::new(),
            unresolved_call_edges: Vec::new(),
            function_entries: Vec::new(),
            call_targets: Vec::new(),
            function_metas: BTreeMap::new(),
            defined_function_index: 0,
            imported_function_count: 0,
            raw_type_id_by_index: BTreeMap::new(),
            raw_type_shape_by_index: BTreeMap::new(),
            signature_ids: BTreeMap::new(),
            next_type_id: 1,
            defined_function_type_indices: Vec::new(),
            initial_memory_pages: None,
            linear_memory_init: Vec::new(),
            globals_init: Vec::new(),
            next_declared_global_index: 0,
        }
    }
}

impl ParsedWasmArtifactsBuilder {
    fn narrow_u32(value: u64, field: &str) -> Result<u32, WasmBuildError> {
        u32::try_from(value).map_err(|_| {
            WasmBuildError::Unsupported(format!(
                "wasm program field `{field}` value {value} does not fit in u32"
            ))
        })
    }

    fn push_pc_rom_edge(&mut self, pc_before: u64, control_choice: u64, pc_after: u64) {
        self.pc_rom.push((pc_before, control_choice, pc_after));
    }

    fn finish(mut self) -> Result<WasmProgramArtifacts, WasmBuildError> {
        let unresolved_call_edges = std::mem::take(&mut self.unresolved_call_edges);
        for (pc_before, function_ref) in unresolved_call_edges {
            let entry_pc = self
                .function_metas
                .get(&function_ref)
                .and_then(|meta| meta.entry_pc)
                .ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing entry pc for direct call target function ref {function_ref}"
                    ))
                })?;
            self.push_pc_rom_edge(pc_before, 0, entry_pc);
        }
        self.pc_rom.sort_unstable();
        self.pc_rom.dedup();
        self.pc_edge_kinds.sort_unstable();
        self.pc_edge_kinds.dedup();
        self.pc_function_refs.sort_unstable();
        self.pc_function_refs.dedup();
        self.function_entries.sort_unstable();
        self.function_entries.dedup();
        self.call_targets.sort_unstable();
        self.call_targets.dedup();
        let mut function_types = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.type_id)))
            .collect::<Vec<_>>();
        function_types.sort_unstable();
        function_types.dedup();
        let mut function_param_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.param_count)))
            .collect::<Vec<_>>();
        function_param_counts.sort_unstable();
        function_param_counts.dedup();
        let mut function_result_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.result_count)))
            .collect::<Vec<_>>();
        function_result_counts.sort_unstable();
        function_result_counts.dedup();
        let mut function_local_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.num_locals)))
            .collect::<Vec<_>>();
        function_local_counts.sort_unstable();
        function_local_counts.dedup();
        let mut function_guest_flags = self
            .function_metas
            .keys()
            .map(|&function_ref| {
                let is_guest = u64::from(function_ref > self.imported_function_count);
                (u64::from(function_ref), is_guest)
            })
            .collect::<Vec<_>>();
        function_guest_flags.sort_unstable();
        function_guest_flags.dedup();
        let mut module_types = self
            .raw_type_id_by_index
            .iter()
            .map(|(&raw_type_index, &type_id)| (u64::from(raw_type_index), u64::from(type_id)))
            .collect::<Vec<_>>();
        module_types.sort_unstable();
        module_types.dedup();
        Ok(WasmProgramArtifacts {
            tables: WasmProgramTables {
                initial_memory_pages: self.initial_memory_pages,
                program_decode: self.program_decode,
                pc_rom: self.pc_rom,
                pc_edge_kinds: self.pc_edge_kinds,
                pc_function_refs: self.pc_function_refs,
                function_entries: self.function_entries,
                function_types,
                function_param_counts,
                function_result_counts,
                function_local_counts,
                function_guest_flags,
                call_targets: self.call_targets,
                module_types,
                linear_memory_init: self.linear_memory_init,
                globals_init: self.globals_init,
            },
            trace: WasmTraceLoweringTables {
                opcode_map: self.opcode_map,
                imported_function_count: self.imported_function_count,
                function_metas: self.function_metas,
            },
        })
    }

    fn consume_payload(&mut self, payload: Payload<'_>) -> Result<(), WasmBuildError> {
        match payload {
            Payload::TypeSection(reader) => {
                for (raw_type_index, func_type_result) in reader.into_iter_err_on_gc_types().enumerate() {
                    let func_type = func_type_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm func type: {err}")))?;
                    let signature = canonical_wasmparser_func_signature(&func_type);
                    let type_id = *self.signature_ids.entry(signature).or_insert_with(|| {
                        let assigned = self.next_type_id;
                        self.next_type_id = self.next_type_id.saturating_add(1);
                        assigned
                    });
                    self.raw_type_id_by_index
                        .insert(raw_type_index as u32, type_id);
                    self.raw_type_shape_by_index.insert(
                        raw_type_index as u32,
                        (func_type.params().len() as u8, func_type.results().len() as u8),
                    );
                }
            }
            Payload::ImportSection(reader) => {
                for import_result in reader {
                    let import = import_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm import: {err}")))?;
                    if matches!(import.ty, wasmparser::TypeRef::Global(_)) {
                        // Imported globals occupy the leading global indexes.
                        self.next_declared_global_index = self.next_declared_global_index.saturating_add(1);
                    }
                    if let wasmparser::TypeRef::Memory(memory) = import.ty {
                        self.set_initial_memory_pages(memory)?;
                    }
                    if let wasmparser::TypeRef::Func(raw_type_index) = import.ty {
                        let function_ref = self.imported_function_count.saturating_add(1);
                        self.imported_function_count = self.imported_function_count.saturating_add(1);
                        let type_id = *self
                            .raw_type_id_by_index
                            .get(&raw_type_index)
                            .ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing normalized type id for imported function type {raw_type_index}"
                                ))
                            })?;
                        let (param_count, result_count) = *self
                            .raw_type_shape_by_index
                            .get(&raw_type_index)
                            .ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing raw type shape for imported function type {raw_type_index}"
                                ))
                            })?;
                        self.function_metas.insert(
                            function_ref,
                            ParsedFunctionMeta {
                                type_id,
                                param_count,
                                result_count,
                                num_locals: u32::from(param_count),
                                entry_pc: None,
                            },
                        );
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for raw_type_index_result in reader {
                    let raw_type_index = raw_type_index_result.map_err(|err| {
                        WasmBuildError::Trace(format!("failed to decode wasm function type index: {err}"))
                    })?;
                    self.defined_function_type_indices.push(raw_type_index);
                }
            }
            Payload::MemorySection(reader) => {
                for memory_result in reader {
                    let memory = memory_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm memory: {err}")))?;
                    self.set_initial_memory_pages(memory)?;
                }
            }
            Payload::CodeSectionEntry(body) => {
                let function_ref = self
                    .imported_function_count
                    .checked_add(self.defined_function_index)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| WasmBuildError::Trace("function ref id overflow".to_string()))?;
                let raw_type_index = *self
                    .defined_function_type_indices
                    .get(self.defined_function_index as usize)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing type index for defined function {}",
                            self.defined_function_index
                        ))
                    })?;
                let (param_count, result_count) = *self
                    .raw_type_shape_by_index
                    .get(&raw_type_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing raw type shape for defined function type {raw_type_index}"
                        ))
                    })?;
                let mut declared_local_count = 0_u32;
                let locals_reader = body
                    .get_locals_reader()
                    .map_err(|err| WasmBuildError::Trace(format!("failed to read wasm locals: {err}")))?;
                for local_result in locals_reader {
                    let (count, _) = local_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm local decl: {err}")))?;
                    declared_local_count = declared_local_count
                        .checked_add(count)
                        .ok_or_else(|| WasmBuildError::Trace("declared local count overflow".to_string()))?;
                }
                let num_locals = u32::from(param_count)
                    .checked_add(declared_local_count)
                    .ok_or_else(|| WasmBuildError::Trace("total local count overflow".to_string()))?;
                let mut reader = body
                    .get_operators_reader()
                    .map_err(|err| WasmBuildError::Trace(format!("failed to read wasm operators: {err}")))?;
                let mut curr_depth = 0_usize;
                let mut control_stack: Vec<ControlFrame> = Vec::new();
                let mut entry_pc = None;
                while !reader.eof() {
                    let offset = reader.original_position() as u32;
                    if entry_pc.is_none() {
                        entry_pc = Some(u64::from(offset));
                    }
                    let pc_before = u64::from(offset);
                    let operator = reader
                        .read()
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm operator: {err}")))?;
                    let pc_after = reader.original_position() as u64;
                    let is_function_end = matches!(&operator, wasmparser::Operator::End) && curr_depth == 0;
                    let decoded = match &operator {
                        wasmparser::Operator::Loop { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::Loop, None))
                        }
                        wasmparser::Operator::Block { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::Block, None))
                        }
                        wasmparser::Operator::If { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::If, None))
                        }
                        wasmparser::Operator::Else => Some((WasmOpcode::Else, None)),
                        wasmparser::Operator::Unreachable => Some((WasmOpcode::Unreachable, None)),
                        wasmparser::Operator::Return => Some((WasmOpcode::Return, None)),
                        wasmparser::Operator::End => {
                            if curr_depth == 0 {
                                Some((WasmOpcode::End, None))
                            } else {
                                curr_depth -= 1;
                                Some((WasmOpcode::End, None))
                            }
                        }
                        _ => decode_opcode(&operator),
                    };
                    let memory = decode_memory_opcode(&operator);
                    let call_return_pc = match &operator {
                        wasmparser::Operator::Call { .. } | wasmparser::Operator::CallIndirect { .. } => {
                            Some(reader.original_position() as u64)
                        }
                        _ => None,
                    };
                    let (call_indirect_type_index, expected_type_id) = match &operator {
                        wasmparser::Operator::CallIndirect { type_index, .. } => {
                            let expected_type_id = *self.raw_type_id_by_index.get(type_index).ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing normalized type id for call_indirect type {type_index}"
                                ))
                            })?;
                            (Some(*type_index), Some(expected_type_id))
                        }
                        _ => (None, None),
                    };
                    if let Some((opcode, immediate)) = decoded {
                        let i64_const = match operator {
                            wasmparser::Operator::I64Const { value } => Some(value as u64),
                            _ => None,
                        };
                        let memory_offset = match memory {
                            Some(memory) => Self::narrow_u32(memory.offset, "program_decode.memory_offset")?,
                            None => 0,
                        };
                        self.program_decode.push(WasmProgramDecodeEntry {
                            pc: pc_before,
                            opcode_code: u32::from(opcode_code(opcode)),
                            local_index: match opcode {
                                WasmOpcode::LocalGet | WasmOpcode::LocalSet | WasmOpcode::LocalTee => {
                                    immediate.unwrap_or(0)
                                }
                                _ => 0,
                            },
                            global_index: match opcode {
                                WasmOpcode::GlobalGet | WasmOpcode::GlobalSet => immediate.unwrap_or(0),
                                _ => 0,
                            },
                            table_id: match opcode {
                                WasmOpcode::TableGet
                                | WasmOpcode::TableSet
                                | WasmOpcode::TableSize
                                | WasmOpcode::CallIndirect => immediate.unwrap_or(0),
                                _ => 0,
                            },
                            memory_offset,
                            call_indirect_type_index: call_indirect_type_index.unwrap_or(0),
                            call_indirect_expected_type_id: expected_type_id.unwrap_or(0),
                            i32_const_value: match opcode {
                                WasmOpcode::I32Const => immediate.unwrap_or(0),
                                _ => 0,
                            },
                            i64_const_value_lo: i64_const.map_or(0, |value| value as u32),
                            i64_const_value_hi: i64_const.map_or(0, |value| (value >> 32) as u32),
                            ref_func_ref: match opcode {
                                WasmOpcode::RefFunc => immediate.unwrap_or(0),
                                _ => 0,
                            },
                        });
                    }
                    let pc_edge_kind = match &operator {
                        wasmparser::Operator::Return => WasmPcEdgeKind::ReturnLike,
                        wasmparser::Operator::End if is_function_end => WasmPcEdgeKind::ReturnLike,
                        wasmparser::Operator::CallIndirect { .. } => WasmPcEdgeKind::DynamicCallIndirect,
                        wasmparser::Operator::Unreachable => WasmPcEdgeKind::Terminal,
                        _ => WasmPcEdgeKind::Static,
                    };
                    self.pc_edge_kinds
                        .push((pc_before, u64::from(pc_edge_kind.as_u32())));
                    self.pc_function_refs
                        .push((pc_before, u64::from(function_ref)));
                    self.opcode_map.insert(
                        (self.defined_function_index, offset),
                        DecodedOpcode {
                            text: format!("{operator:?}"),
                            memory,
                            decoded,
                            control: decode_control_opcode(&operator),
                            pc_edge_kind,
                            call_indirect_type_index,
                            expected_type_id,
                            call_return_pc,
                        },
                    );

                    match &operator {
                        wasmparser::Operator::Loop { .. } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            control_stack.push(ControlFrame::new(ControlFrameKind::Loop, Some(pc_after)));
                        }
                        wasmparser::Operator::Block { .. } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            control_stack.push(ControlFrame::new(ControlFrameKind::Block, None));
                        }
                        wasmparser::Operator::If { .. } => {
                            self.push_pc_rom_edge(pc_before, 1, pc_after);
                            let mut frame = ControlFrame::new(ControlFrameKind::If, None);
                            frame.pending_if_false = Some(pc_before);
                            control_stack.push(frame);
                        }
                        wasmparser::Operator::Else => {
                            let frame = control_stack.last_mut().ok_or_else(|| {
                                WasmBuildError::Trace("encountered else without an open control frame".to_string())
                            })?;
                            if frame.kind != ControlFrameKind::If {
                                return Err(WasmBuildError::Trace(
                                    "encountered else outside an if frame".to_string(),
                                ));
                            }
                            if let Some(if_pc) = frame.pending_if_false.take() {
                                self.push_pc_rom_edge(if_pc, 0, pc_after);
                            }
                            frame.pending_to_end.push(pc_before);
                        }
                        wasmparser::Operator::End => {
                            if let Some(mut frame) = control_stack.pop() {
                                if let Some(if_pc) = frame.pending_if_false.take() {
                                    self.push_pc_rom_edge(if_pc, 0, pc_after);
                                }
                                for edge_pc in frame.pending_to_end.drain(..) {
                                    self.push_pc_rom_edge(edge_pc, 0, pc_after);
                                }
                                self.push_pc_rom_edge(pc_before, 0, pc_after);
                            }
                        }
                        wasmparser::Operator::BrIf { relative_depth } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            let depth = *relative_depth as usize;
                            let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br_if depth {depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 1, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::Br { relative_depth } => {
                            let depth = *relative_depth as usize;
                            let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br depth {depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 0, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::BrTable { targets } => {
                            for (choice_index, depth_result) in targets.targets().enumerate() {
                                let depth = depth_result.map_err(|err| {
                                    WasmBuildError::Trace(format!("failed to decode br_table target: {err}"))
                                })? as usize;
                                let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                    return Err(WasmBuildError::Trace(format!(
                                        "br_table depth {depth} exceeded control nesting"
                                    )));
                                };
                                let target_frame = &mut control_stack[target_index];
                                if let Some(target) = target_frame.branch_target {
                                    self.push_pc_rom_edge(pc_before, choice_index as u64 + 1, target);
                                } else {
                                    target_frame.pending_to_end.push(pc_before);
                                }
                            }
                            let default_depth = targets.default() as usize;
                            let Some(target_index) = control_stack.len().checked_sub(default_depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br_table default depth {default_depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 0, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::Call { function_index } => {
                            let function_ref = function_index.saturating_add(1);
                            self.call_targets.push((pc_before, u64::from(function_ref)));
                            self.push_pc_rom_edge(pc_before, CALL_RETURN_PC_CHOICE, pc_after);
                            if function_ref <= self.imported_function_count {
                                self.push_pc_rom_edge(pc_before, 0, pc_after);
                            } else {
                                match self
                                    .function_metas
                                    .get(&function_ref)
                                    .and_then(|meta| meta.entry_pc)
                                {
                                    Some(entry_pc) => self.push_pc_rom_edge(pc_before, 0, entry_pc),
                                    None => self.unresolved_call_edges.push((pc_before, function_ref)),
                                }
                            }
                        }
                        wasmparser::Operator::CallIndirect { .. } => {
                            self.push_pc_rom_edge(pc_before, CALL_RETURN_PC_CHOICE, pc_after);
                        }
                        wasmparser::Operator::Unreachable | wasmparser::Operator::Return => {}
                        _ => self.push_pc_rom_edge(pc_before, 0, pc_after),
                    }
                }
                let type_id = *self
                    .raw_type_id_by_index
                    .get(&raw_type_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing normalized type id for defined function type {raw_type_index}"
                        ))
                    })?;
                self.function_metas.insert(
                    function_ref,
                    ParsedFunctionMeta {
                        type_id,
                        param_count,
                        result_count,
                        num_locals,
                        entry_pc,
                    },
                );
                if let Some(entry_pc) = entry_pc {
                    self.function_entries
                        .push((u64::from(function_ref), entry_pc));
                }
                self.defined_function_index += 1;
            }
            Payload::DataSection(reader) => {
                for segment_result in reader {
                    let segment = segment_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm data segment: {err}")))?;
                    let (memory_index, offset) = match segment.kind {
                        wasmparser::DataKind::Active {
                            memory_index,
                            offset_expr,
                        } => {
                            // Only `i32.const N` offset expressions are supported; everything
                            // else (e.g., `global.get`) needs runtime evaluation that's out of
                            // scope for static preload.
                            let mut reader = offset_expr.get_operators_reader();
                            let first = reader.read().map_err(|err| {
                                WasmBuildError::Trace(format!("failed to read data segment offset expr: {err}"))
                            })?;
                            let offset = match first {
                                wasmparser::Operator::I32Const { value } => value as u32 as u64,
                                other => {
                                    return Err(WasmBuildError::Unsupported(format!(
                                        "data segment with non-i32.const offset expr: {other:?}"
                                    )));
                                }
                            };
                            (memory_index, offset)
                        }
                        wasmparser::DataKind::Passive => continue,
                    };
                    if memory_index != 0 {
                        return Err(WasmBuildError::Unsupported(format!(
                            "data segment targeting non-default memory {memory_index} not supported"
                        )));
                    }
                    for (i, &byte) in segment.data.iter().enumerate() {
                        self.linear_memory_init.push((offset + i as u64, byte));
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for global_result in reader {
                    let global = global_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm global: {err}")))?;
                    let mut ops = global.init_expr.get_operators_reader();
                    let first = ops
                        .read()
                        .map_err(|err| WasmBuildError::Trace(format!("failed to read wasm global init expr: {err}")))?;
                    let (lo, hi) = match first {
                        wasmparser::Operator::I32Const { value } => (value as u32, 0),
                        wasmparser::Operator::I64Const { value } => {
                            let v = value as u64;
                            (v as u32, (v >> 32) as u32)
                        }
                        wasmparser::Operator::F32Const { value } => (value.bits(), 0),
                        wasmparser::Operator::F64Const { value } => {
                            let v = value.bits();
                            (v as u32, (v >> 32) as u32)
                        }
                        other => {
                            return Err(WasmBuildError::Unsupported(format!(
                                "wasm global with non-const init expr: {other:?}"
                            )));
                        }
                    };
                    let index = self.next_declared_global_index;
                    self.next_declared_global_index = self.next_declared_global_index.saturating_add(1);
                    self.globals_init.push((index, lo, hi));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn set_initial_memory_pages(&mut self, memory: wasmparser::MemoryType) -> Result<(), WasmBuildError> {
        if memory.memory64 {
            return Err(WasmBuildError::Unsupported(
                "memory64 default memory is not supported by the wasm proof tables".to_string(),
            ));
        }
        if self.initial_memory_pages.is_some() {
            return Err(WasmBuildError::Unsupported(
                "multiple memories are not supported by the wasm proof tables".to_string(),
            ));
        }
        self.initial_memory_pages = Some(Self::narrow_u32(memory.initial, "memory.initial")?);
        Ok(())
    }
}

fn canonical_wasmparser_func_signature(ty: &wasmparser::FuncType) -> String {
    let params = ty
        .params()
        .iter()
        .map(|ty| canonical_wasmparser_val_type(*ty))
        .collect::<Vec<_>>()
        .join(",");
    let results = ty
        .results()
        .iter()
        .map(|ty| canonical_wasmparser_val_type(*ty))
        .collect::<Vec<_>>()
        .join(",");
    format!("{params}->{results}")
}

fn canonical_wasmparser_val_type(ty: wasmparser::ValType) -> String {
    format!("{ty:?}")
}
