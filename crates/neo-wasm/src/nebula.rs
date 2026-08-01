//! Authoritative WASM proof path over the Nebula + terminal-induction backend.

use std::collections::BTreeMap;

#[cfg(feature = "perf-timers")]
use neo_fold_clean::frontends::nebula::application::ApplicationSegmentTrace;
use neo_fold_clean::frontends::nebula::application::{
    ApplicationError, MemoryPort, MemoryPortActivation, MemoryPortKind, MemoryPortLayout, MemoryRegion,
    MemoryRegionKind, NebulaApplication,
};
use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeChainBuilder, NebulaFPrimeChainError, NebulaFPrimePreprocessing,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::{NebulaPlan, PlanError};
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::lifecycle::{verify_uncompressed, Uncompressed};
use neo_fold_clean::paper::nifs::NifsProverAdapter;
use neo_fold_clean::paper::params::Params;
use thiserror::Error;

use crate::adapters::wasmtime::WasmProgramArtifacts;
use crate::batch::padding_step_after;
use crate::comm_chain::CommChainState;
use crate::event_grammar::HostEventGrammar;
use crate::ir::{WasmAuxOpcode, WasmRowKind, WasmStepState, WasmVmStep};
use crate::layout::COL_PADDING_ACTIVE;
use crate::lookup_circuit::{extend_witness, LookupCircuitError};
use crate::memory_semantics::preload_grammar_tables;
use crate::preprocess::{
    canonical_wasm_nebula_shape_batched_with_initial_state_digest, grammar_top_level_initial_state_digest,
    semantic_state_digest, top_level_initial_state_digest, WasmPreprocessError,
};
use crate::relation_layout::{build_wasm_relation_layout, WasmMemoryActivation, WasmMemoryColumnKind};
use crate::witness_builder::build_witness_vector;
use crate::{preload_from_program_artifacts, WasmOpcode};

const WASM_NEBULA_PLAN_SEED: [u8; 32] = [0x57; 32];
const WASM32_PAGE_WORDS: u64 = 65_536 / 4;
// Fixed instruction batch used by the WASM/Nebula profiles. Performance tests
// apply their committed-coordinate targets separately.
const WASM_NEBULA_BATCH_SIZE: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmNebulaLimits {
    stack_cells: u64,
    call_stack_cells: u64,
    local_frames: u64,
    locals_per_frame: u64,
    linear_memory_words: u64,
    globals: u64,
    tables: u64,
    table_elements: u64,
}

impl WasmNebulaLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stack_cells: u64,
        call_stack_cells: u64,
        local_frames: u64,
        locals_per_frame: u64,
        linear_memory_words: u64,
        globals: u64,
        tables: u64,
        table_elements: u64,
    ) -> Result<Self, WasmNebulaError> {
        let out = Self {
            stack_cells,
            call_stack_cells,
            local_frames,
            locals_per_frame,
            linear_memory_words,
            globals,
            tables,
            table_elements,
        };
        for (name, value) in out.named_values() {
            if value < 2 || !value.is_power_of_two() {
                return Err(WasmNebulaError::NonPowerOfTwoLimit { name, value });
            }
        }
        Ok(out)
    }

    pub fn production() -> Self {
        Self::new(4096, 256, 256, 16, 32768, 64, 8, 256).expect("production WASM Nebula limits")
    }

    #[doc(hidden)]
    pub fn test_profile() -> Self {
        Self::new(16, 8, 8, 4, 64, 4, 2, 4).expect("test WASM Nebula limits")
    }

    fn named_values(self) -> [(&'static str, u64); 8] {
        [
            ("stack_cells", self.stack_cells),
            ("call_stack_cells", self.call_stack_cells),
            ("local_frames", self.local_frames),
            ("locals_per_frame", self.locals_per_frame),
            ("linear_memory_words", self.linear_memory_words),
            ("globals", self.globals),
            ("tables", self.tables),
            ("table_elements", self.table_elements),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmNebulaProfile {
    memory: NebulaParams,
    limits: WasmNebulaLimits,
    batch_size: usize,
}

impl WasmNebulaProfile {
    pub fn production() -> Self {
        Self::production_with_batch_size(WASM_NEBULA_BATCH_SIZE)
    }

    #[cfg(feature = "perf-timers")]
    #[doc(hidden)]
    /// Build an unbounded production-parameter timing profile with an explicit
    /// instruction batch. This does not change the production default or imply
    /// Road A width-budget compatibility.
    pub fn production_with_profile_batch_size(batch_size: usize) -> Self {
        assert!(batch_size > 0, "WASM Nebula profile batch size must be nonzero");
        Self::production_with_batch_size(batch_size)
    }

    fn production_with_batch_size(batch_size: usize) -> Self {
        let memory = batched_memory_geometry(NebulaParams::v3_targets(), batch_size);
        Self {
            memory,
            limits: WasmNebulaLimits::production(),
            batch_size,
        }
    }

    #[doc(hidden)]
    pub fn test_profile() -> Self {
        let memory = NebulaParams::new(10, 10, 64, 1024, 16).expect("test WASM Nebula geometry");
        Self::test_profile_with_geometry(memory)
    }

    /// Test profile over a caller-chosen memory geometry, for fixtures whose
    /// ROM plan (pc space, grammar tables) outgrows the default `r = 10`.
    #[doc(hidden)]
    pub fn test_profile_with_geometry(memory: NebulaParams) -> Self {
        Self {
            memory: batched_memory_geometry(memory, WASM_NEBULA_BATCH_SIZE),
            limits: WasmNebulaLimits::test_profile(),
            batch_size: WASM_NEBULA_BATCH_SIZE,
        }
    }

    pub fn memory(&self) -> &NebulaParams {
        &self.memory
    }

    pub fn limits(&self) -> WasmNebulaLimits {
        self.limits
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

fn batched_memory_geometry(memory: NebulaParams, batch_size: usize) -> NebulaParams {
    // The slot budget is the relation's own per-step port count: every
    // declared memory column occupies one ops-lane slot per batched step.
    let step_ports: usize = build_wasm_relation_layout()
        .auxiliary
        .memories
        .iter()
        .map(|memory| memory.columns.len())
        .sum();
    NebulaParams::new(
        memory.r,
        memory.mu,
        step_ports * batch_size,
        memory.b_scan,
        memory.seg_max,
    )
    .expect("batched WASM Nebula memory geometry")
}

pub struct WasmNebulaPreprocessing {
    inner: NebulaFPrimePreprocessing,
    profile: WasmNebulaProfile,
    lookup_auxiliary_columns_per_instruction: usize,
    lookup_auxiliary_columns_total: usize,
    has_linear_memory: bool,
    // Grammar-mode preprocessing binds host calls through the event chain,
    // so `prove` accepts host-call rows instead of rejecting them.
    allows_host_calls: bool,
}

impl WasmNebulaPreprocessing {
    pub fn inner(&self) -> &NebulaFPrimePreprocessing {
        &self.inner
    }

    pub fn profile(&self) -> WasmNebulaProfile {
        self.profile
    }

    #[doc(hidden)]
    pub fn lookup_auxiliary_columns_per_instruction(&self) -> usize {
        self.lookup_auxiliary_columns_per_instruction
    }

    #[doc(hidden)]
    pub fn total_lookup_auxiliary_columns(&self) -> usize {
        self.lookup_auxiliary_columns_total
    }
}

pub struct WasmNebulaProof {
    proof: Uncompressed,
}

impl WasmNebulaProof {
    pub fn inner(&self) -> &Uncompressed {
        &self.proof
    }
}

pub fn preprocess(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(params, profile, artifacts, initial_locals, entry_pc, None, None, None)
}

/// Preprocess only when the final F′ relation fits the caller's committed-
/// coordinate limit.
pub fn preprocess_with_coordinate_limit(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    max_coordinates: usize,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        None,
        Some(max_coordinates),
    )
}

#[doc(hidden)]
pub fn preprocess_seeded(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        Some(seed),
        None,
    )
}

#[doc(hidden)]
pub fn preprocess_seeded_with_coordinate_limit(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
    max_coordinates: usize,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        Some(seed),
        Some(max_coordinates),
    )
}

/// Builds a structurally faithful tiny fixture without claiming that its
/// reduced linear-memory domain implements full WASM page capacity.
#[doc(hidden)]
pub fn preprocess_seeded_reduced_memory_test_only(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    reject_host_imports(artifacts)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        Some(seed),
        None,
    )
}

/// Grammar preprocessing with an explicit initial commitment state.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn preprocess_seeded_grammar_test_only(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    grammar: &HostEventGrammar,
    export_fref: u32,
    seed: u64,
    initial_comm_chain: CommChainState,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_grammar_program(artifacts, profile.limits)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        Some((grammar, export_fref, initial_comm_chain)),
        Some(seed),
        None,
    )
}

#[cfg(feature = "perf-timers")]
#[doc(hidden)]
pub fn preprocess_seeded_unbounded_profile(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        Some(seed),
        None,
    )
}

fn preprocess_inner(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    grammar: Option<(&HostEventGrammar, u32, CommChainState)>,
    seed: Option<u64>,
    max_coordinates: Option<usize>,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    let initial_state = match grammar {
        Some((grammar, export_fref, initial_comm_chain)) => grammar_top_level_initial_state_digest(
            &artifacts.tables,
            entry_pc,
            grammar,
            export_fref,
            initial_comm_chain,
        ),
        None => top_level_initial_state_digest(&artifacts.tables, entry_pc),
    };
    let canonical = canonical_wasm_nebula_shape_batched_with_initial_state_digest(profile.batch_size, initial_state)?;
    let backend = build_memory_backend(
        artifacts,
        initial_locals,
        grammar.map(|(grammar, _, _)| grammar),
        &profile,
        canonical.single_step_columns,
    )?;
    let plan = NebulaPlan::new_with_initial_ram(
        profile.memory,
        backend.rom_image,
        backend.ram_image,
        WASM_NEBULA_PLAN_SEED,
        params.kappa() as usize,
    )?;
    let lookup_auxiliary_columns_per_instruction = canonical.lookup_auxiliary_columns_per_instruction;
    let lookup_auxiliary_columns_total = canonical.lookup_auxiliary_columns_total;
    let application = NebulaApplication::new(canonical.sparse_r1cs, canonical.plan, backend.layout)?;
    let inner = match (seed, max_coordinates) {
        (Some(seed), Some(max_coordinates)) => {
            NebulaFPrimePreprocessing::new_seeded_with_application_and_coordinate_limit(
                params,
                plan,
                application,
                seed,
                max_coordinates,
            )?
        }
        (Some(seed), None) => NebulaFPrimePreprocessing::new_seeded_with_application(params, plan, application, seed)?,
        (None, Some(max_coordinates)) => NebulaFPrimePreprocessing::new_with_application_and_coordinate_limit(
            params,
            plan,
            application,
            max_coordinates,
        )?,
        (None, None) => NebulaFPrimePreprocessing::new_with_application(params, plan, application)?,
    };
    Ok(WasmNebulaPreprocessing {
        inner,
        profile,
        lookup_auxiliary_columns_per_instruction,
        lookup_auxiliary_columns_total,
        has_linear_memory: artifacts.tables.initial_memory_pages.is_some(),
        allows_host_calls: grammar.is_some(),
    })
}

#[cfg(feature = "perf-timers")]
#[doc(hidden)]
pub fn build_application_segment_for_profile(
    prep: &WasmNebulaPreprocessing,
    trace: &[WasmVmStep],
) -> Result<ApplicationSegmentTrace, WasmNebulaError> {
    if trace.is_empty() {
        return Err(WasmNebulaError::EmptyTrace);
    }
    reject_host_trace(trace)?;
    let plan = prep.inner.plan();
    let application = prep
        .inner
        .relation()
        .application()
        .ok_or(WasmNebulaError::MissingApplication)?;
    let rows_per_segment = plan.params().steps_per_segment() * prep.profile.batch_size;
    let mut rows = trace[..trace.len().min(rows_per_segment)].to_vec();
    while rows.len() < rows_per_segment {
        let previous = rows.last().expect("profile segment starts nonempty");
        rows.push(padding_step_after(previous));
    }
    let assignments = rows
        .chunks_exact(prep.profile.batch_size)
        .map(compact_batched_assignment)
        .collect::<Result<Vec<_>, _>>()?;
    let mut memory = Memory::new_with_initial_ram(*plan.params(), plan.rom_image(), plan.ram_image())?;
    Ok(application.trace_segment(&mut memory, assignments)?)
}

/// Prove a WASM execution with automatic CUDA, Metal, or CPU selection.
///
/// The proof format and verifier do not depend on the selected prover.
pub fn prove(prep: &WasmNebulaPreprocessing, trace: &[WasmVmStep]) -> Result<WasmNebulaProof, WasmNebulaError> {
    crate::WasmProver::auto().prove(prep, trace)
}

/// Prove a WASM execution while routing recursive and terminal NIFS folds
/// through `adapter`.
#[doc(hidden)]
pub fn prove_with_nifs_adapter(
    prep: &WasmNebulaPreprocessing,
    adapter: &mut dyn NifsProverAdapter,
    trace: &[WasmVmStep],
) -> Result<WasmNebulaProof, WasmNebulaError> {
    prove_inner(prep, trace, adapter)
}

fn prove_inner(
    prep: &WasmNebulaPreprocessing,
    trace: &[WasmVmStep],
    adapter: &mut dyn NifsProverAdapter,
) -> Result<WasmNebulaProof, WasmNebulaError> {
    if trace.is_empty() {
        return Err(WasmNebulaError::EmptyTrace);
    }
    if !trace.last().expect("nonempty").state_after.halted {
        return Err(WasmNebulaError::NonTerminalTrace);
    }
    if !prep.allows_host_calls {
        reject_host_trace(trace)?;
    }

    let plan = prep.inner.plan();
    let mut memory = Memory::new_with_initial_ram(*plan.params(), plan.rom_image(), plan.ram_image())?;
    let application = prep
        .inner
        .relation()
        .application()
        .ok_or(WasmNebulaError::MissingApplication)?;
    let steps_per_segment = plan.params().steps_per_segment();
    let rows_per_segment = steps_per_segment * prep.profile.batch_size;
    let mut chain = NebulaFPrimeChainBuilder::new(&prep.inner);
    for chunk in trace.chunks(rows_per_segment) {
        let mut rows = chunk.to_vec();
        while rows.len() < rows_per_segment {
            let previous = rows.last().expect("each trace chunk is nonempty");
            rows.push(padding_step_after(previous));
        }
        let assignments = rows
            .chunks_exact(prep.profile.batch_size)
            .map(compact_batched_assignment)
            .collect::<Result<Vec<_>, _>>()?;
        debug_assert_eq!(assignments.len(), steps_per_segment);
        let segment = application.trace_segment(&mut memory, assignments)?;
        chain.append_application_segment_with_nifs_adapter(&segment, adapter)?;
    }
    let proof = chain.finish_with_nifs_adapter(adapter)?;
    Ok(WasmNebulaProof { proof })
}

fn compact_batched_assignment(rows: &[WasmVmStep]) -> Result<Vec<neo_math::F>, LookupCircuitError> {
    let mut assignment = Vec::new();
    for row in rows {
        assignment.extend(extend_witness(build_witness_vector(row))?);
    }
    Ok(assignment)
}

pub fn verify(
    prep: &WasmNebulaPreprocessing,
    proof: &WasmNebulaProof,
    claimed_final_state: WasmStepState,
) -> Result<(), WasmNebulaError> {
    if !claimed_final_state.halted {
        return Err(WasmNebulaError::FalseTerminalClaim);
    }
    let pages_present = claimed_final_state.memory_pages.is_some();
    let max_present = claimed_final_state.max_memory_pages.is_some();
    if pages_present != prep.has_linear_memory || max_present != prep.has_linear_memory {
        return Err(WasmNebulaError::MemoryPresenceMismatch {
            expected: prep.has_linear_memory,
            pages_present,
            max_present,
        });
    }
    verify_uncompressed(&prep.inner.prep, &proof.proof)?;
    if proof.proof.state.semantic_state_digest != semantic_state_digest(claimed_final_state) {
        return Err(WasmNebulaError::FinalStateMismatch);
    }
    Ok(())
}

struct MemoryBackend {
    layout: MemoryPortLayout,
    rom_image: Vec<u32>,
    ram_image: Vec<u32>,
}

fn build_memory_backend(
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    grammar: Option<&HostEventGrammar>,
    profile: &WasmNebulaProfile,
    single_step_columns: usize,
) -> Result<MemoryBackend, WasmNebulaError> {
    let relation = build_wasm_relation_layout();
    let mut preload = preload_from_program_artifacts(artifacts, initial_locals);
    if let Some(grammar) = grammar {
        preload_grammar_tables(&mut preload, grammar);
    }
    let entries = preload.entries();
    let mut by_memory: BTreeMap<&str, Vec<(Vec<u32>, u32)>> = BTreeMap::new();
    for (memory, address, value) in &entries {
        by_memory
            .entry(memory)
            .or_default()
            .push((address.clone(), *value));
    }

    let mut regions = Vec::with_capacity(relation.auxiliary.memories.len());
    let mut region_by_name = BTreeMap::new();
    let mut rom_cursor = 0u64;
    let mut ram_cursor = 0u64;
    for memory in &relation.auxiliary.memories {
        let kind = if memory.is_rom {
            MemoryRegionKind::Rom
        } else {
            MemoryRegionKind::Ram
        };
        let component_bits = if memory.is_rom {
            rom_component_bits(
                memory.name,
                memory.columns[0].address_columns.len(),
                by_memory.get(memory.name),
            )?
        } else {
            ram_component_bits(memory.name, profile.limits)?
        };
        let base = match kind {
            MemoryRegionKind::Rom => rom_cursor,
            MemoryRegionKind::Ram => ram_cursor,
        };
        let region = MemoryRegion::new(memory.name, kind, base, component_bits)?;
        match kind {
            MemoryRegionKind::Rom => rom_cursor += region.cells(),
            MemoryRegionKind::Ram => ram_cursor += region.cells(),
        }
        region_by_name.insert(memory.name, regions.len());
        regions.push(region);
    }
    if rom_cursor > profile.memory.rom_cells() || ram_cursor > profile.memory.ram_cells() {
        return Err(WasmNebulaError::MemoryPlanTooLarge {
            rom: rom_cursor,
            rom_capacity: profile.memory.rom_cells(),
            ram: ram_cursor,
            ram_capacity: profile.memory.ram_cells(),
        });
    }

    let mut rom_image = vec![0; profile.memory.rom_cells() as usize];
    let mut ram_image = vec![0; profile.memory.ram_cells() as usize];
    for (memory, address, value) in entries {
        let region = &regions[*region_by_name
            .get(memory)
            .ok_or_else(|| WasmNebulaError::UnknownMemory(memory.to_string()))?];
        let components = address
            .iter()
            .map(|&value| u64::from(value))
            .collect::<Vec<_>>();
        let physical = region.address(&components)? as usize;
        match region.kind() {
            MemoryRegionKind::Rom => rom_image[physical] = value,
            MemoryRegionKind::Ram => ram_image[physical] = value,
        }
    }

    let mut single_ports = Vec::new();
    for memory in &relation.auxiliary.memories {
        let region = *region_by_name
            .get(memory.name)
            .ok_or_else(|| WasmNebulaError::UnknownMemory(memory.name.to_string()))?;
        for column in &memory.columns {
            let kind = match column.kind {
                WasmMemoryColumnKind::Read => MemoryPortKind::Read,
                WasmMemoryColumnKind::Write { value_before_column } => MemoryPortKind::Write {
                    value_before_column: value_before_column.map(|column| column.0),
                },
            };
            let activation = match column.activation {
                WasmMemoryActivation::Always => MemoryPortActivation::UnlessColumn(COL_PADDING_ACTIVE),
                WasmMemoryActivation::BooleanGate(column) => MemoryPortActivation::Column(column.0),
            };
            single_ports.push(MemoryPort::new(
                region,
                column
                    .address_columns
                    .iter()
                    .map(|column| column.0)
                    .collect(),
                column.value_column.0,
                kind,
                activation,
            ));
        }
    }
    let mut ports = Vec::with_capacity(single_ports.len() * profile.batch_size);
    for block in 0..profile.batch_size {
        let offset = block * single_step_columns;
        for port in &single_ports {
            let kind = match port.kind() {
                MemoryPortKind::Read => MemoryPortKind::Read,
                MemoryPortKind::Write { value_before_column } => MemoryPortKind::Write {
                    value_before_column: value_before_column.map(|column| column + offset),
                },
            };
            let activation = match port.activation() {
                MemoryPortActivation::Always => MemoryPortActivation::Always,
                MemoryPortActivation::Column(column) => MemoryPortActivation::Column(column + offset),
                MemoryPortActivation::UnlessColumn(column) => MemoryPortActivation::UnlessColumn(column + offset),
            };
            ports.push(MemoryPort::new(
                port.region(),
                port.address_columns()
                    .iter()
                    .map(|column| column + offset)
                    .collect(),
                port.value_column() + offset,
                kind,
                activation,
            ));
        }
    }
    Ok(MemoryBackend {
        layout: MemoryPortLayout::new(regions, ports)?,
        rom_image,
        ram_image,
    })
}

fn rom_component_bits(
    memory: &str,
    arity: usize,
    entries: Option<&Vec<(Vec<u32>, u32)>>,
) -> Result<Vec<u8>, WasmNebulaError> {
    let mut maxima = vec![1u64; arity];
    for (address, _) in entries.into_iter().flatten() {
        if address.len() != arity {
            return Err(WasmNebulaError::MemoryAddressArity {
                memory: memory.to_string(),
                expected: arity,
                actual: address.len(),
            });
        }
        for (index, &value) in address.iter().enumerate() {
            maxima[index] = maxima[index].max(u64::from(value) + 1);
        }
    }
    Ok(maxima.into_iter().map(bits_for_bound).collect())
}

fn ram_component_bits(memory: &str, limits: WasmNebulaLimits) -> Result<Vec<u8>, WasmNebulaError> {
    let dimensions = match memory {
        "stack" => vec![limits.stack_cells],
        "call_stack_return_pcs" | "call_stack_caller_fbps" | "call_stack_caller_sp_bases" => {
            vec![limits.call_stack_cells]
        }
        "linear_memory" => vec![limits.linear_memory_words],
        "locals" | "locals_hi" => vec![limits.local_frames, limits.locals_per_frame],
        "globals" | "globals_hi" => vec![limits.globals],
        "tables" => vec![limits.tables, limits.table_elements],
        "table_sizes" => vec![limits.tables],
        other => return Err(WasmNebulaError::UnknownMemory(other.to_string())),
    };
    Ok(dimensions.into_iter().map(bits_for_bound).collect())
}

fn bits_for_bound(bound: u64) -> u8 {
    let rounded = bound.max(2).next_power_of_two();
    rounded.ilog2() as u8
}

fn reject_imported_state(artifacts: &WasmProgramArtifacts) -> Result<(), WasmNebulaError> {
    if artifacts.tables.has_imported_memory || artifacts.tables.imported_global_count != 0 {
        return Err(WasmNebulaError::ImportedStateUnsupported);
    }
    Ok(())
}

fn reject_host_imports(artifacts: &WasmProgramArtifacts) -> Result<(), WasmNebulaError> {
    reject_imported_state(artifacts)?;
    if artifacts
        .tables
        .function_guest_flags
        .iter()
        .any(|&(_, is_guest)| is_guest == 0)
    {
        return Err(WasmNebulaError::HostImportsUnsupported);
    }
    Ok(())
}

/// Grammar mode binds host FUNCTION calls through the event chain; imported
/// memories and globals are still verifier-unbound state, and the declared
/// linear-memory limits apply regardless.
fn validate_grammar_program(artifacts: &WasmProgramArtifacts, limits: WasmNebulaLimits) -> Result<(), WasmNebulaError> {
    reject_imported_state(artifacts)?;
    validate_linear_memory_limits(artifacts, limits)
}

fn validate_sound_program(artifacts: &WasmProgramArtifacts, limits: WasmNebulaLimits) -> Result<(), WasmNebulaError> {
    reject_host_imports(artifacts)?;
    validate_linear_memory_limits(artifacts, limits)
}

fn validate_linear_memory_limits(
    artifacts: &WasmProgramArtifacts,
    limits: WasmNebulaLimits,
) -> Result<(), WasmNebulaError> {
    let Some(initial_pages) = artifacts.tables.initial_memory_pages else {
        return Ok(());
    };
    let max_pages = artifacts
        .tables
        .max_memory_pages
        .expect("parsed linear memory always carries a maximum");
    let capacity_pages = limits.linear_memory_words / WASM32_PAGE_WORDS;
    if u64::from(initial_pages) > capacity_pages || u64::from(max_pages) > capacity_pages {
        return Err(WasmNebulaError::DeclaredLinearMemoryTooLarge {
            initial_pages,
            max_pages,
            capacity_pages,
        });
    }
    Ok(())
}

fn reject_host_trace(trace: &[WasmVmStep]) -> Result<(), WasmNebulaError> {
    let has_host_row = trace.iter().any(|row| {
        matches!(
            row.row_kind,
            WasmRowKind::Aux(WasmAuxOpcode::HostCallArg | WasmAuxOpcode::HostCallResult)
        ) || (matches!(row.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && row.function_ref.is_some()
            && !row.target_function_is_guest)
    });
    if has_host_row {
        return Err(WasmNebulaError::HostImportsUnsupported);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum WasmNebulaError {
    #[error(transparent)]
    LookupCircuit(#[from] LookupCircuitError),
    #[error(transparent)]
    Preprocess(#[from] WasmPreprocessError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    Chain(#[from] NebulaFPrimeChainError),
    #[error(transparent)]
    Lifecycle(#[from] neo_fold_clean::lifecycle::Error),
    #[error(transparent)]
    Trace(#[from] neo_fold_clean::frontends::nebula::trace::TraceError),
    #[error("WASM Nebula limit `{name}` must be a power of two >= 2, got {value}")]
    NonPowerOfTwoLimit { name: &'static str, value: u64 },
    #[error("WASM memory plan needs ROM {rom}/{rom_capacity} and RAM {ram}/{ram_capacity} cells")]
    MemoryPlanTooLarge {
        rom: u64,
        rom_capacity: u64,
        ram: u64,
        ram_capacity: u64,
    },
    #[error("WASM memory plan has no resource rule for `{0}`")]
    UnknownMemory(String),
    #[error("WASM memory `{memory}` expects {expected} address components, preload has {actual}")]
    MemoryAddressArity {
        memory: String,
        expected: usize,
        actual: usize,
    },
    #[error("imported host functions are unsupported without grammar templates binding their calls")]
    HostImportsUnsupported,
    #[error("imported memories and globals are unsupported until their state is verifier-bound")]
    ImportedStateUnsupported,
    #[error(
        "declared linear memory ({initial_pages} initial, {max_pages} maximum pages) exceeds the dense proof domain ({capacity_pages} pages)"
    )]
    DeclaredLinearMemoryTooLarge {
        initial_pages: u32,
        max_pages: u32,
        capacity_pages: u64,
    },
    #[error("cannot prove an empty WASM trace")]
    EmptyTrace,
    #[error("WASM trace does not end in a terminal state")]
    NonTerminalTrace,
    #[error("WASM terminal verifier requires halted = true")]
    FalseTerminalClaim,
    #[error(
        "WASM terminal memory presence does not match the program (expected={expected}, pages_present={pages_present}, max_present={max_present})"
    )]
    MemoryPresenceMismatch {
        expected: bool,
        pages_present: bool,
        max_present: bool,
    },
    #[error("WASM proof final semantic state does not match the claimed state")]
    FinalStateMismatch,
    #[error("WASM Nebula preprocessing is missing its application relation")]
    MissingApplication,
    #[error("WASM prover backend `{backend}` is unavailable: {reason}")]
    ProverBackendUnavailable {
        backend: &'static str,
        reason: String,
    },
}
