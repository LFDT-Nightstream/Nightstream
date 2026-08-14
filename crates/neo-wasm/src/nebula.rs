//! Authoritative WASM proof path over the Nebula + terminal-induction backend.

use std::collections::BTreeMap;

#[cfg(feature = "perf-timers")]
use neo_fold_clean::frontends::nebula::application::ApplicationSegmentTrace;
use neo_fold_clean::frontends::nebula::application::{
    ApplicationError, MemoryPortLayout, MemoryRegion, MemoryRegionKind, NebulaApplication,
};
use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeChainBuilder, NebulaFPrimeChainError, NebulaFPrimePreparedProfile, NebulaFPrimePreprocessing,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::{NebulaPlan, PlanError};
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::lifecycle::{
    verify_uncompressed, verify_uncompressed_with_opening_backend, FinalWitnessOpeningBackend, Uncompressed,
};
use neo_fold_clean::paper::nifs::NifsProverAdapter;
use neo_fold_clean::paper::params::Params;
use thiserror::Error;

use crate::adapters::wasmtime::WasmProgramArtifacts;
use crate::batch::padding_step_after;
use crate::comm_chain::CommChainState;
use crate::event_grammar::HostEventGrammar;
use crate::ir::{WasmAuxOpcode, WasmRowKind, WasmStepState, WasmVmStep};
use crate::lookup_circuit::{extend_witness, LookupCircuitError};
use crate::memory_routing::{build_batched_memory_slots, build_single_step_memory_slots};
use crate::memory_semantics::preload_grammar_tables;
use crate::preprocess::{
    canonical_wasm_nebula_shape_batched_with_initial_state_digest, grammar_top_level_initial_state_digest,
    semantic_state_digest, top_level_initial_state_digest, WasmPreprocessError,
};
use crate::relation_layout::build_wasm_relation_layout;
use crate::witness_builder::build_witness_vector;
use crate::{preload_from_program_artifacts, WasmOpcode};

const WASM_NEBULA_PLAN_SEED: [u8; 32] = [0x57; 32];
const WASM32_PAGE_WORDS: u64 = 65_536 / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmNebulaRomLimits {
    program_pc_bound: u64,
    functions: u64,
    module_types: u64,
    control_choices: u64,
    grammar_events_per_function: u64,
    grammar_slots_per_event: u64,
}

impl WasmNebulaRomLimits {
    pub fn new(
        program_pc_bound: u64,
        functions: u64,
        module_types: u64,
        control_choices: u64,
        grammar_events_per_function: u64,
        grammar_slots_per_event: u64,
    ) -> Result<Self, WasmNebulaError> {
        let out = Self {
            program_pc_bound,
            functions,
            module_types,
            control_choices,
            grammar_events_per_function,
            grammar_slots_per_event,
        };
        for (name, value) in out.named_values() {
            if value < 2 || !value.is_power_of_two() {
                return Err(WasmNebulaError::NonPowerOfTwoLimit { name, value });
            }
        }
        Ok(out)
    }

    #[doc(hidden)]
    pub fn test_profile() -> Self {
        Self::new(64, 8, 8, 2, 4, 8).expect("test WASM ROM limits")
    }

    fn named_values(self) -> [(&'static str, u64); 6] {
        [
            ("program_pc_bound", self.program_pc_bound),
            ("functions", self.functions),
            ("module_types", self.module_types),
            ("control_choices", self.control_choices),
            ("grammar_events_per_function", self.grammar_events_per_function),
            ("grammar_slots_per_event", self.grammar_slots_per_event),
        ]
    }
}

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
    rom: WasmNebulaRomLimits,
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
        rom: WasmNebulaRomLimits,
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
            rom,
        };
        for (name, value) in out.named_values() {
            if value < 2 || !value.is_power_of_two() {
                return Err(WasmNebulaError::NonPowerOfTwoLimit { name, value });
            }
        }
        Ok(out)
    }

    #[doc(hidden)]
    pub fn test_profile() -> Self {
        Self::new(16, 8, 8, 4, 64, 4, 2, 4, WasmNebulaRomLimits::test_profile()).expect("test WASM Nebula limits")
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
    /// Build the paper-parameter profile with caller-owned application limits
    /// and instruction batch size.
    pub fn production(limits: WasmNebulaLimits, batch_size: usize) -> Result<Self, WasmNebulaError> {
        if batch_size == 0 {
            return Err(WasmNebulaError::ZeroBatchSize);
        }
        Ok(Self {
            memory: batched_memory_geometry(NebulaParams::v3_targets(), batch_size),
            limits,
            batch_size,
        })
    }

    #[doc(hidden)]
    pub fn test_profile() -> Self {
        let memory = NebulaParams::new(11, 11, 64, 1024, 16).expect("test WASM Nebula geometry");
        Self::test_profile_with_schedule(memory, 3)
    }

    /// Test profile over a caller-chosen memory geometry, for fixtures whose
    /// ROM plan (pc space, grammar tables) outgrows the default `r = 11`.
    #[doc(hidden)]
    pub fn test_profile_with_geometry(memory: NebulaParams) -> Self {
        Self::test_profile_with_schedule(memory, 3)
    }

    /// Build a reduced test profile with an explicit instruction batch.
    /// The supplied memory geometry still owns the complete Nebula scan.
    #[doc(hidden)]
    pub fn test_profile_with_schedule(memory: NebulaParams, batch_size: usize) -> Self {
        assert!(batch_size > 0, "WASM Nebula test batch size must be nonzero");
        Self {
            memory: batched_memory_geometry(memory, batch_size),
            limits: WasmNebulaLimits::test_profile(),
            batch_size,
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
    let step_slots = build_single_step_memory_slots(build_wasm_relation_layout()).len();
    NebulaParams::new(
        memory.r,
        memory.mu,
        step_slots * batch_size,
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

/// One compiled WASM/Nebula relation and evaluator cache for a fixed profile.
///
/// Live construction can write the evaluator and encoder artifacts. A later
/// process restores this object with [`prepare_profile_with_artifacts`].
/// [`Self::bind_program`] then performs only program construction, exact
/// profile comparison, and verifier-policy binding.
pub struct WasmNebulaPreparedProfile {
    inner: NebulaFPrimePreparedProfile,
    profile: WasmNebulaProfile,
    plan_template: NebulaPlan,
    application_template: NebulaApplication,
    single_step_columns: usize,
    lookup_auxiliary_columns_per_instruction: usize,
    lookup_auxiliary_columns_total: usize,
}

impl WasmNebulaPreparedProfile {
    pub fn bind_program(
        &self,
        artifacts: &WasmProgramArtifacts,
        initial_locals: &[u32],
        entry_pc: u64,
    ) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
        validate_sound_program(artifacts, self.profile.limits)?;
        self.bind_program_inner(artifacts, initial_locals, entry_pc)
    }

    #[doc(hidden)]
    pub fn bind_program_reduced_memory_test_only(
        &self,
        artifacts: &WasmProgramArtifacts,
        initial_locals: &[u32],
        entry_pc: u64,
    ) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
        reject_host_imports(artifacts)?;
        self.bind_program_inner(artifacts, initial_locals, entry_pc)
    }

    fn bind_program_inner(
        &self,
        artifacts: &WasmProgramArtifacts,
        initial_locals: &[u32],
        entry_pc: u64,
    ) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
        #[cfg(feature = "perf-timers")]
        let build_started = std::time::Instant::now();
        let program = build_program_binding_from_template(
            self.inner.params(),
            self.profile,
            artifacts,
            initial_locals,
            entry_pc,
            &self.plan_template,
            &self.application_template,
            self.single_step_columns,
            self.lookup_auxiliary_columns_per_instruction,
            self.lookup_auxiliary_columns_total,
        )?;
        #[cfg(feature = "perf-timers")]
        let build_elapsed = build_started.elapsed();
        if program.metadata.lookup_auxiliary_columns_per_instruction != self.lookup_auxiliary_columns_per_instruction
            || program.metadata.lookup_auxiliary_columns_total != self.lookup_auxiliary_columns_total
        {
            return Err(WasmNebulaError::PreparedProfileMismatch);
        }
        let metadata = program.metadata;
        #[cfg(feature = "perf-timers")]
        let bind_started = std::time::Instant::now();
        let inner = self
            .inner
            .bind_application(program.plan, program.application)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[wasm-nebula-profile-bind] program={:.3}s relation+policy={:.3}s total={:.3}s",
            build_elapsed.as_secs_f64(),
            bind_started.elapsed().as_secs_f64(),
            build_started.elapsed().as_secs_f64(),
        );
        Ok(metadata.finish(inner, self.profile))
    }

    pub fn profile(&self) -> WasmNebulaProfile {
        self.profile
    }

    #[doc(hidden)]
    pub fn inner(&self) -> &NebulaFPrimePreparedProfile {
        &self.inner
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

/// Compile one reusable relation and evaluator cache for this WASM profile.
/// `artifacts` supplies a checked reference instance of the profile shape;
/// its program values are not verifier authority for later bindings.
pub fn prepare_profile(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
) -> Result<WasmNebulaPreparedProfile, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    prepare_profile_inner(params, profile, artifacts, initial_locals, entry_pc, None, None)
}

/// Restore a production profile from receipt-checked evaluator and encoder artifacts.
#[allow(clippy::too_many_arguments)]
pub fn prepare_profile_with_artifacts(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    cache_artifact: neo_reductions::superneo_eval::VerifiedSuperneoCacheArtifact,
    encoder_artifact: neo_fold_clean::frontends::nebula::f_prime::VerifiedNebulaFPrimeEncoderArtifact,
) -> Result<WasmNebulaPreparedProfile, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    prepare_profile_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        None,
        Some((cache_artifact, encoder_artifact)),
    )
}

#[doc(hidden)]
pub fn prepare_profile_seeded_reduced_memory_test_only(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
) -> Result<WasmNebulaPreparedProfile, WasmNebulaError> {
    reject_host_imports(artifacts)?;
    prepare_profile_inner(params, profile, artifacts, initial_locals, entry_pc, Some(seed), None)
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_profile_seeded_reduced_memory_with_artifacts_test_only(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: u64,
    cache_artifact: neo_reductions::superneo_eval::VerifiedSuperneoCacheArtifact,
    encoder_artifact: neo_fold_clean::frontends::nebula::f_prime::VerifiedNebulaFPrimeEncoderArtifact,
) -> Result<WasmNebulaPreparedProfile, WasmNebulaError> {
    reject_host_imports(artifacts)?;
    prepare_profile_inner(
        params,
        profile,
        artifacts,
        initial_locals,
        entry_pc,
        Some(seed),
        Some((cache_artifact, encoder_artifact)),
    )
}

pub fn preprocess(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    validate_sound_program(artifacts, profile.limits)?;
    preprocess_inner(params, profile, artifacts, initial_locals, entry_pc, None, None)
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
    preprocess_inner(params, profile, artifacts, initial_locals, entry_pc, None, Some(seed))
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
    preprocess_inner(params, profile, artifacts, initial_locals, entry_pc, None, Some(seed))
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
    preprocess_inner(params, profile, artifacts, initial_locals, entry_pc, None, Some(seed))
}

fn preprocess_inner(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    grammar: Option<(&HostEventGrammar, u32, CommChainState)>,
    seed: Option<u64>,
) -> Result<WasmNebulaPreprocessing, WasmNebulaError> {
    let program = build_program_binding(&params, profile, artifacts, initial_locals, entry_pc, grammar)?;
    let metadata = program.metadata;
    let inner = match seed {
        Some(seed) => {
            NebulaFPrimePreprocessing::new_seeded_with_application(params, program.plan, program.application, seed)?
        }
        None => NebulaFPrimePreprocessing::new_with_application(params, program.plan, program.application)?,
    };
    Ok(metadata.finish(inner, profile))
}

fn prepare_profile_inner(
    params: Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    seed: Option<u64>,
    profile_artifacts: Option<(
        neo_reductions::superneo_eval::VerifiedSuperneoCacheArtifact,
        neo_fold_clean::frontends::nebula::f_prime::VerifiedNebulaFPrimeEncoderArtifact,
    )>,
) -> Result<WasmNebulaPreparedProfile, WasmNebulaError> {
    let program = build_program_binding(&params, profile, artifacts, initial_locals, entry_pc, None)?;
    let inner = match (seed, profile_artifacts) {
        (Some(seed), Some((cache_artifact, encoder_artifact))) => {
            NebulaFPrimePreparedProfile::new_seeded_with_application_artifacts(
                params,
                &program.plan,
                &program.application,
                seed,
                cache_artifact,
                encoder_artifact,
            )?
        }
        (None, Some((cache_artifact, encoder_artifact))) => {
            NebulaFPrimePreparedProfile::new_with_application_artifacts(
                params,
                &program.plan,
                &program.application,
                cache_artifact,
                encoder_artifact,
            )?
        }
        (Some(seed), None) => {
            NebulaFPrimePreparedProfile::new_seeded_with_application(params, &program.plan, &program.application, seed)?
        }
        (None, None) => NebulaFPrimePreparedProfile::new_with_application(params, &program.plan, &program.application)?,
    };
    Ok(WasmNebulaPreparedProfile {
        inner,
        profile,
        plan_template: program.plan,
        application_template: program.application,
        single_step_columns: program.metadata.single_step_columns,
        lookup_auxiliary_columns_per_instruction: program.metadata.lookup_auxiliary_columns_per_instruction,
        lookup_auxiliary_columns_total: program.metadata.lookup_auxiliary_columns_total,
    })
}

struct WasmNebulaProgramBinding {
    plan: NebulaPlan,
    application: NebulaApplication,
    metadata: WasmNebulaProgramMetadata,
}

#[derive(Clone, Copy)]
struct WasmNebulaProgramMetadata {
    single_step_columns: usize,
    lookup_auxiliary_columns_per_instruction: usize,
    lookup_auxiliary_columns_total: usize,
    has_linear_memory: bool,
    allows_host_calls: bool,
}

impl WasmNebulaProgramMetadata {
    fn finish(self, inner: NebulaFPrimePreprocessing, profile: WasmNebulaProfile) -> WasmNebulaPreprocessing {
        WasmNebulaPreprocessing {
            inner,
            profile,
            lookup_auxiliary_columns_per_instruction: self.lookup_auxiliary_columns_per_instruction,
            lookup_auxiliary_columns_total: self.lookup_auxiliary_columns_total,
            has_linear_memory: self.has_linear_memory,
            allows_host_calls: self.allows_host_calls,
        }
    }
}

fn build_program_binding(
    params: &Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    grammar: Option<(&HostEventGrammar, u32, CommChainState)>,
) -> Result<WasmNebulaProgramBinding, WasmNebulaError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    let initial_state = program_initial_state_digest(artifacts, entry_pc, grammar);
    #[cfg(feature = "perf-timers")]
    let canonical_started = std::time::Instant::now();
    let canonical =
        canonical_wasm_nebula_shape_batched_with_initial_state_digest(params, profile.batch_size, initial_state)?;
    #[cfg(feature = "perf-timers")]
    let canonical_elapsed = canonical_started.elapsed();
    let (plan, memory) = build_program_memory_plan(
        params,
        profile,
        artifacts,
        initial_locals,
        grammar.map(|(grammar, _, _)| grammar),
        canonical.single_step_columns,
        None,
    )?;
    #[cfg(feature = "perf-timers")]
    let application_started = std::time::Instant::now();
    let application = NebulaApplication::new(canonical.sparse_r1cs, canonical.plan, memory)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[wasm-nebula-program-bind] canonical={:.3}s application={:.3}s total={:.3}s",
        canonical_elapsed.as_secs_f64(),
        application_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
    );
    Ok(WasmNebulaProgramBinding {
        plan,
        application,
        metadata: WasmNebulaProgramMetadata {
            single_step_columns: canonical.single_step_columns,
            lookup_auxiliary_columns_per_instruction: canonical.lookup_auxiliary_columns_per_instruction,
            lookup_auxiliary_columns_total: canonical.lookup_auxiliary_columns_total,
            has_linear_memory: artifacts.tables.initial_memory_pages.is_some(),
            allows_host_calls: grammar.is_some(),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn build_program_binding_from_template(
    params: &Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    entry_pc: u64,
    plan_template: &NebulaPlan,
    application_template: &NebulaApplication,
    single_step_columns: usize,
    lookup_auxiliary_columns_per_instruction: usize,
    lookup_auxiliary_columns_total: usize,
) -> Result<WasmNebulaProgramBinding, WasmNebulaError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    let initial_state = program_initial_state_digest(artifacts, entry_pc, None);
    let (plan, memory) = build_program_memory_plan(
        params,
        profile,
        artifacts,
        initial_locals,
        None,
        single_step_columns,
        Some(plan_template),
    )?;
    #[cfg(feature = "perf-timers")]
    let application_started = std::time::Instant::now();
    let application = application_template.bind_program_profile(initial_state, memory)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[wasm-nebula-program-bind] canonical=reused application={:.3}s total={:.3}s",
        application_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
    );
    Ok(WasmNebulaProgramBinding {
        plan,
        application,
        metadata: WasmNebulaProgramMetadata {
            single_step_columns,
            lookup_auxiliary_columns_per_instruction,
            lookup_auxiliary_columns_total,
            has_linear_memory: artifacts.tables.initial_memory_pages.is_some(),
            allows_host_calls: false,
        },
    })
}

fn program_initial_state_digest(
    artifacts: &WasmProgramArtifacts,
    entry_pc: u64,
    grammar: Option<(&HostEventGrammar, u32, CommChainState)>,
) -> [u8; 32] {
    match grammar {
        Some((grammar, export_fref, initial_comm_chain)) => grammar_top_level_initial_state_digest(
            &artifacts.tables,
            entry_pc,
            grammar,
            export_fref,
            initial_comm_chain,
        ),
        None => top_level_initial_state_digest(&artifacts.tables, entry_pc),
    }
}

fn build_program_memory_plan(
    params: &Params,
    profile: WasmNebulaProfile,
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
    grammar: Option<&HostEventGrammar>,
    single_step_columns: usize,
    plan_template: Option<&NebulaPlan>,
) -> Result<(NebulaPlan, MemoryPortLayout), WasmNebulaError> {
    #[cfg(feature = "perf-timers")]
    let backend_started = std::time::Instant::now();
    let backend = build_memory_backend(artifacts, initial_locals, grammar, &profile, single_step_columns)?;
    #[cfg(feature = "perf-timers")]
    let backend_elapsed = backend_started.elapsed();
    let MemoryBackend {
        layout,
        rom_image,
        ram_image,
    } = backend;
    #[cfg(feature = "perf-timers")]
    let plan_started = std::time::Instant::now();
    let plan = match plan_template {
        Some(template) => {
            if template.params() != &profile.memory {
                return Err(WasmNebulaError::PreparedProfileMismatch);
            }
            template.bind_initial_memory(rom_image, ram_image)?
        }
        None => NebulaPlan::new_with_initial_ram(
            profile.memory,
            rom_image,
            ram_image,
            WASM_NEBULA_PLAN_SEED,
            params.kappa() as usize,
        )?,
    };
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[wasm-nebula-memory-plan] memory={:.3}s plan={:.3}s",
        backend_elapsed.as_secs_f64(),
        plan_started.elapsed().as_secs_f64(),
    );
    Ok((plan, layout))
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

/// Prove a WASM execution with automatic CUDA, Metal, or optimized CPU selection.
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
    verify_inner(prep, proof, claimed_final_state, None)
}

#[doc(hidden)]
/// Verify with a trusted final-opening accelerator.
///
/// The backend is part of the verifier's trusted computing base. Normal
/// callers must use [`verify`].
pub fn verify_with_witness_opening_backend(
    prep: &WasmNebulaPreprocessing,
    proof: &WasmNebulaProof,
    claimed_final_state: WasmStepState,
    backend: &mut dyn FinalWitnessOpeningBackend,
) -> Result<(), WasmNebulaError> {
    verify_inner(prep, proof, claimed_final_state, Some(backend))
}

fn verify_inner(
    prep: &WasmNebulaPreprocessing,
    proof: &WasmNebulaProof,
    claimed_final_state: WasmStepState,
    opening_backend: Option<&mut dyn FinalWitnessOpeningBackend>,
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
    match opening_backend {
        Some(backend) => verify_uncompressed_with_opening_backend(&prep.inner.prep, &proof.proof, backend)?,
        None => verify_uncompressed(&prep.inner.prep, &proof.proof)?,
    }
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
                profile.limits.rom,
                grammar.is_some(),
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

    let slots = build_batched_memory_slots(relation, profile.batch_size, single_step_columns);
    Ok(MemoryBackend {
        layout: MemoryPortLayout::new(regions, slots)?,
        rom_image,
        ram_image,
    })
}

fn rom_component_bits(
    memory: &str,
    arity: usize,
    limits: WasmNebulaRomLimits,
    grammar_enabled: bool,
    entries: Option<&Vec<(Vec<u32>, u32)>>,
) -> Result<Vec<u8>, WasmNebulaError> {
    for (address, _) in entries.into_iter().flatten() {
        if address.len() != arity {
            return Err(WasmNebulaError::MemoryAddressArity {
                memory: memory.to_string(),
                expected: arity,
                actual: address.len(),
            });
        }
    }
    let program = bits_for_bound(limits.program_pc_bound);
    let functions = bits_for_bound(limits.functions);
    let component_bits = match memory {
        "program_opcodes"
        | "program_local_indices"
        | "program_global_indices"
        | "program_table_ids"
        | "program_memory_offsets"
        | "program_call_indirect_type_indices"
        | "program_call_indirect_expected_type_ids"
        | "program_i32_const_values"
        | "program_i64_const_values_lo"
        | "program_i64_const_values_hi"
        | "program_ref_func_refs"
        | "pc_function_refs"
        | "call_targets"
        | "pc_edge_kinds" => vec![program],
        "pc_rom" => vec![program, bits_for_bound(limits.control_choices)],
        "function_types" | "function_local_counts" | "function_call_metadata" | "function_entries" => {
            vec![functions]
        }
        "module_types" => vec![bits_for_bound(limits.module_types)],
        "grammar_import_pre_counts" | "grammar_export_entry_counts" | "grammar_export_exit_counts" => {
            vec![if grammar_enabled { functions } else { 1 }]
        }
        "grammar_slot_kind"
        | "grammar_slot_arg"
        | "grammar_slot_variant"
        | "grammar_slot_const_lo"
        | "grammar_slot_const_hi" => {
            if grammar_enabled {
                vec![
                    functions,
                    bits_for_bound(limits.grammar_events_per_function),
                    bits_for_bound(limits.grammar_slots_per_event),
                ]
            } else {
                vec![1; arity]
            }
        }
        other => return Err(WasmNebulaError::UnknownMemory(other.to_string())),
    };
    if component_bits.len() != arity {
        return Err(WasmNebulaError::MemoryAddressArity {
            memory: memory.to_string(),
            expected: arity,
            actual: component_bits.len(),
        });
    }
    Ok(component_bits)
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
        .function_call_metadata
        .iter()
        .any(|&(_, metadata)| !crate::ir::function_call_metadata_is_guest(metadata))
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
    #[error("WASM Nebula instruction batch size must be nonzero")]
    ZeroBatchSize,
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
    #[error("WASM program does not match the prepared profile")]
    PreparedProfileMismatch,
    #[error("WASM prover backend `{backend}` is unavailable: {reason}")]
    ProverBackendUnavailable {
        backend: &'static str,
        reason: String,
    },
}

#[cfg(test)]
#[path = "../tests/nebula/profile_binding.rs"]
mod tests;
