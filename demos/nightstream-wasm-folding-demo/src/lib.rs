//! Demo-facing projection of normalized WASM traces.
//!
//! This crate does not define WASM execution or proof semantics. It invokes
//! `neo-wasm` and projects its authoritative normalized rows into JSON DTOs.

use neo_wasm::{
    collect_wasmtime_steps, extract_wasm_program_artifacts, traces_from_wasmtime_steps, StackValueAccess,
    WasmProgramArtifacts, WasmRowKind, WasmVmStep, WasmtimeTraceRun,
};
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

pub const COUNTER_WAT: &str = r#";; Calls a parameterized helper that counts up to a fixed limit.
;; Change the 5 in main to trace a different number of loop iterations.
(module
  (func $count_to (param $limit i32) (result i32)
    (local $counter i32)
    (loop $again
      local.get $counter
      i32.const 1
      i32.add
      local.tee $counter
      local.get $limit
      i32.lt_u
      br_if $again
    )
    local.get $counter
  )
  (func (export "main") (result i32)
    i32.const 5
    call $count_to
  )
)"#;

pub const TABLE_DISPATCH_WAT: &str = r#";; Calls through a function table, then replaces its first entry.
;; The same indirect call increments once and doubles after table.set.
(module
  (type $operation (func (param i32) (result i32)))

  (func $increment (type $operation) (param $value i32) (result i32)
    local.get $value
    i32.const 1
    i32.add
  )

  (func $double (type $operation) (param $value i32) (result i32)
    local.get $value
    i32.const 2
    i32.mul
  )

  (table 2 funcref)
  (elem (i32.const 0) $increment $double)

  (func $apply (param $value i32) (result i32)
    local.get $value
    i32.const 0
    call_indirect (type $operation)
  )

  (func (export "main") (result i32)
    (local $value i32)

    i32.const 5
    call $apply
    local.set $value

    i32.const 0
    ref.func $double
    table.set 0

    local.get $value
    call $apply
  )
)"#;

pub const MUTABLE_GLOBAL_WAT: &str = r#";; Updates module-level state through a parameterized helper function.
;; The global starts at 10, then accumulates 3 and 4.
(module
  (global $total (mut i32) (i32.const 10))

  (func $accumulate (param $value i32)
    global.get $total
    local.get $value
    i32.add
    global.set $total
  )

  (func (export "main") (result i32)
    i32.const 3
    call $accumulate
    i32.const 4
    call $accumulate
    global.get $total
  )
)"#;

pub const TRAPPING_DIVISION_WAT: &str = r#";; Traps while dividing by zero inside a parameterized helper.
;; The terminal trapped state is part of the proved execution outcome.
(module
  (func $divide (param $numerator i32) (param $denominator i32) (result i32)
    local.get $numerator
    local.get $denominator
    i32.div_u
  )

  (func (export "main") (result i32)
    i32.const 42
    i32.const 0
    call $divide
  )
)"#;

pub const BR_TABLE_WAT: &str = r#";; Uses br_table as a three-way integer dispatch.
;; Change the selector in main: 0 returns 10, 1 returns 20, anything else returns 30.
(module
  (func $dispatch (param $selector i32) (result i32)
    (block $default
      (block $case1
        (block $case0
          local.get $selector
          br_table $case0 $case1 $default
        )
        i32.const 10
        return
      )
      i32.const 20
      return
    )
    i32.const 30
  )

  (func (export "main") (result i32)
    i32.const 1
    call $dispatch
  )
)"#;

#[derive(Debug, Deserialize)]
pub struct TraceRequest {
    pub source: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofMode {
    FoldingAudit,
    IvcNoMemory,
    NebulaMemory,
}

#[derive(Debug, Deserialize)]
pub struct PreparationRequest {
    pub trace_id: u64,
    pub mode: ProofMode,
}

#[derive(Debug, Deserialize)]
pub struct ProofRequest {
    pub preparation_id: u64,
}

#[derive(Debug, Serialize)]
pub struct TraceResponse {
    pub program: ProgramSummary,
    pub execution: ExecutionSummary,
    pub rows: Vec<TraceRow>,
}

#[derive(Debug, Serialize)]
pub struct PreparationResponse {
    pub mode: &'static str,
    pub memory_consistency: bool,
    pub verifier_key_digest: String,
    pub memory_plan_digest: Option<[String; 4]>,
    pub initial_ram_digest: Option<[String; 4]>,
    pub security: &'static str,
    pub normalized_rows: usize,
    pub batch_size: usize,
    pub folds: usize,
    pub padded_rows: usize,
    pub preprocess_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct ProofResponse {
    pub mode: &'static str,
    pub memory_consistency: bool,
    pub security: &'static str,
    pub result: Vec<String>,
    pub trapped: bool,
    pub normalized_rows: usize,
    pub batch_size: usize,
    pub folds: usize,
    pub padded_rows: usize,
    pub prove_ms: u128,
    pub verify_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct ProgramSummary {
    pub wasm_bytes: usize,
    pub decoded_instructions: usize,
    pub control_edges: usize,
    pub function_entries: usize,
    pub has_linear_memory: bool,
}

#[derive(Debug, Serialize)]
pub struct ExecutionSummary {
    pub results: Vec<String>,
    pub initial_locals: Vec<u32>,
    pub normalized_rows: usize,
    pub halted: bool,
    pub trapped: bool,
}

#[derive(Debug, Serialize)]
pub struct TraceRow {
    pub cycle: u64,
    pub kind: String,
    pub opcode: String,
    pub pc_before: u64,
    pub pc_after: u64,
    pub sp_before: u64,
    pub sp_after: u64,
    pub call_depth_before: u64,
    pub call_depth_after: u64,
    pub control_choice: u32,
    pub edge: String,
    pub stack_reads: Vec<ValueAccess>,
    pub stack_write: Option<ValueAccess>,
    pub local_index: Option<u32>,
    pub local_read: Option<u32>,
    pub local_write: Option<u32>,
    pub global_index: Option<u32>,
    pub global_read: Option<u32>,
    pub global_write: Option<u32>,
    pub output: Option<u64>,
    pub halted: bool,
    pub trapped: bool,
}

#[derive(Debug, Serialize)]
pub struct ValueAccess {
    pub address: u64,
    pub lo: u32,
    pub hi: Option<u32>,
}

pub fn trace_program(request: TraceRequest) -> Result<(TraceResponse, Arc<TracedProgram>), String> {
    let program = Arc::new(trace_source(&request)?);
    Ok((project_trace_response(&program), program))
}

pub struct PreparedProof {
    program: Arc<TracedProgram>,
    mode: PreparedMode,
    metadata: PreparedMetadata,
}

enum PreparedMode {
    FoldingAudit(neo_fold_clean::frontends::r1cs_f_prime::R1csFPrimePreprocessing),
    IvcNoMemory(neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcPreprocessing),
    NebulaMemory(neo_wasm::WasmNebulaPreprocessing),
}

#[derive(Clone, Copy)]
struct PreparedMetadata {
    mode: &'static str,
    memory_consistency: bool,
    security: &'static str,
    normalized_rows: usize,
    batch_size: usize,
    folds: usize,
    padded_rows: usize,
}

pub fn preprocess_program(
    program: Arc<TracedProgram>,
    mode: ProofMode,
) -> Result<(PreparationResponse, PreparedProof), String> {
    match mode {
        ProofMode::FoldingAudit => preprocess_folding_audit(program),
        ProofMode::IvcNoMemory => preprocess_ivc_without_memory(program),
        ProofMode::NebulaMemory => preprocess_with_nebula_memory(program),
    }
}

pub fn prove_prepared(prepared: &PreparedProof) -> Result<ProofResponse, String> {
    match &prepared.mode {
        PreparedMode::FoldingAudit(prep) => prove_folding_audit(prepared, prep),
        PreparedMode::IvcNoMemory(prep) => prove_ivc_without_memory(prepared, prep),
        PreparedMode::NebulaMemory(prep) => prove_with_nebula_memory(prepared, prep),
    }
}

fn preprocess_folding_audit(program: Arc<TracedProgram>) -> Result<(PreparationResponse, PreparedProof), String> {
    const BATCH_SIZE: usize = 64;
    let entry_pc = program.normalized[0].state_before.pc;
    let initial_digest = neo_wasm::top_level_initial_state_digest(&program.artifacts.tables, entry_pc);
    let started = Instant::now();
    let prep = neo_wasm::preprocess::preprocess_seeded_batched(BATCH_SIZE, initial_digest)
        .map_err(|error| format!("WASM folding-audit preprocessing failed: {error}"))?;
    let preprocess_ms = started.elapsed().as_millis();
    let folds = program.normalized.len().div_ceil(BATCH_SIZE);
    let metadata = PreparedMetadata {
        mode: "folding_audit",
        memory_consistency: false,
        security:
            "demonstration only: reduced parameters; NIFS.V is not constrained and verification replays the full chain",
        normalized_rows: program.normalized.len(),
        batch_size: BATCH_SIZE,
        folds,
        padded_rows: folds * BATCH_SIZE - program.normalized.len(),
    };
    let response = PreparationResponse {
        mode: metadata.mode,
        memory_consistency: metadata.memory_consistency,
        verifier_key_digest: bytes_hex(prep.prep.vk.digest()),
        memory_plan_digest: None,
        initial_ram_digest: None,
        security: metadata.security,
        normalized_rows: metadata.normalized_rows,
        batch_size: metadata.batch_size,
        folds: metadata.folds,
        padded_rows: metadata.padded_rows,
        preprocess_ms,
    };
    let prepared = PreparedProof {
        program,
        mode: PreparedMode::FoldingAudit(prep),
        metadata,
    };
    Ok((response, prepared))
}

fn preprocess_with_nebula_memory(program: Arc<TracedProgram>) -> Result<(PreparationResponse, PreparedProof), String> {
    let entry_pc = program.normalized[0].state_before.pc;
    let profile = neo_wasm::WasmNebulaProfile::demo_no_linear_memory();
    let started = Instant::now();
    // The demo uses a deterministic, process-local Ajtai setup. Production
    // callers should provide/load the canonical setup explicitly.
    let prep = neo_wasm::preprocess_seeded(
        nebula_demo_params()?,
        profile,
        &program.artifacts,
        &program.run.initial_locals,
        entry_pc,
        0xa55e_ca11_ed15_ea,
    )
    .map_err(|error| format!("Nebula preprocessing failed: {error}"))?;
    let preprocess_ms = started.elapsed().as_millis();
    let rows_per_segment = profile.memory().steps_per_segment() * profile.batch_size();
    let segments = program.normalized.len().div_ceil(rows_per_segment);
    let folds = segments * profile.memory().steps_per_segment();
    let metadata = PreparedMetadata {
        mode: "nebula_memory",
        memory_consistency: true,
        security: "demonstration only: reduced geometry and kappa = 1",
        normalized_rows: program.normalized.len(),
        batch_size: profile.batch_size(),
        folds,
        padded_rows: segments * rows_per_segment - program.normalized.len(),
    };
    let response = PreparationResponse {
        mode: metadata.mode,
        memory_consistency: metadata.memory_consistency,
        verifier_key_digest: bytes_hex(prep.inner().prep.vk.digest()),
        memory_plan_digest: Some(digest_hex(prep.inner().plan().plan_digest())),
        initial_ram_digest: Some(digest_hex(prep.inner().plan().d_init())),
        security: metadata.security,
        normalized_rows: metadata.normalized_rows,
        batch_size: metadata.batch_size,
        folds: metadata.folds,
        padded_rows: metadata.padded_rows,
        preprocess_ms,
    };
    let prepared = PreparedProof {
        program,
        mode: PreparedMode::NebulaMemory(prep),
        metadata,
    };
    Ok((response, prepared))
}

fn preprocess_ivc_without_memory(program: Arc<TracedProgram>) -> Result<(PreparationResponse, PreparedProof), String> {
    use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcPreprocessing;
    const BATCH_SIZE: usize = 32;
    let entry_pc = program.normalized[0].state_before.pc;
    let started = Instant::now();
    let initial_digest = neo_wasm::top_level_initial_state_digest(&program.artifacts.tables, entry_pc);
    let canonical = neo_wasm::preprocess::canonical_wasm_lookup_f_prime_shape_batched_with_initial_state_digest(
        BATCH_SIZE,
        initial_digest,
    )
    .map_err(|error| format!("WASM IVC relation construction failed: {error}"))?;
    let prep = R1csIvcPreprocessing::new_seeded(
        ivc_demo_params()?,
        canonical.sparse_r1cs,
        canonical.plan,
        0xa55e_ca11_ed15_1fc0,
    )
    .map_err(|error| format!("WASM IVC preprocessing failed: {error}"))?;
    let preprocess_ms = started.elapsed().as_millis();
    let folds = program.normalized.len().div_ceil(BATCH_SIZE);
    let metadata = PreparedMetadata {
        mode: "ivc_no_memory",
        memory_consistency: false,
        security: "demonstration only: reduced parameters; ROM/RAM consistency is not proven",
        normalized_rows: program.normalized.len(),
        batch_size: BATCH_SIZE,
        folds,
        padded_rows: folds * BATCH_SIZE - program.normalized.len(),
    };
    let response = PreparationResponse {
        mode: metadata.mode,
        memory_consistency: metadata.memory_consistency,
        verifier_key_digest: bytes_hex(prep.prep.vk.digest()),
        memory_plan_digest: None,
        initial_ram_digest: None,
        security: metadata.security,
        normalized_rows: metadata.normalized_rows,
        batch_size: metadata.batch_size,
        folds: metadata.folds,
        padded_rows: metadata.padded_rows,
        preprocess_ms,
    };
    let prepared = PreparedProof {
        program,
        mode: PreparedMode::IvcNoMemory(prep),
        metadata,
    };
    Ok((response, prepared))
}

fn prove_ivc_without_memory(
    prepared: &PreparedProof,
    prep: &neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcPreprocessing,
) -> Result<ProofResponse, String> {
    use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvc;

    let program = &prepared.program;
    let final_state = program
        .normalized
        .last()
        .expect("preprocessing rejects empty traces")
        .state_after;
    let started = Instant::now();
    let mut chain = R1csIvc::new(prep);
    for batch_index in 0..prepared.metadata.folds {
        let base =
            neo_wasm::batch::build_batched_witness(&program.normalized, prepared.metadata.batch_size, batch_index);
        let single_width = neo_wasm::range_checked_witness_width();
        let mut lookup_closed = Vec::with_capacity(prep.app().m());
        for row in base.chunks_exact(single_width) {
            lookup_closed.extend(
                neo_wasm::extend_witness_for_profile(row.to_vec())
                    .map_err(|error| format!("WASM lookup witness construction failed: {error}"))?,
            );
        }
        chain
            .extend(lookup_closed)
            .map_err(|error| format!("WASM IVC fold failed: {error}"))?;
    }
    let proof = chain
        .finish()
        .map_err(|error| format!("WASM IVC finalization failed: {error}"))?;
    let prove_ms = started.elapsed().as_millis();
    let started = Instant::now();
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .map_err(|error| format!("WASM IVC verification failed: {error}"))?;
    if proof.state.semantic_state_digest != neo_wasm::semantic_state_digest(final_state) {
        return Err("WASM IVC final state does not match the traced execution".to_string());
    }
    let verify_ms = started.elapsed().as_millis();
    Ok(proof_response(
        prepared,
        program.run.results.clone(),
        prove_ms,
        verify_ms,
    ))
}

fn prove_folding_audit(
    prepared: &PreparedProof,
    prep: &neo_fold_clean::frontends::r1cs_f_prime::R1csFPrimePreprocessing,
) -> Result<ProofResponse, String> {
    use neo_fold_clean::frontends::r1cs_f_prime::R1csChainBuilder;

    let program = &prepared.program;
    let final_state = program
        .normalized
        .last()
        .expect("preprocessing rejects empty traces")
        .state_after;
    let started = Instant::now();
    let mut chain =
        R1csChainBuilder::new(prep).map_err(|error| format!("WASM folding-audit initialization failed: {error}"))?;
    for batch_index in 0..prepared.metadata.folds {
        let assignment =
            neo_wasm::batch::build_batched_witness(&program.normalized, prepared.metadata.batch_size, batch_index);
        chain
            .append_assignment(assignment)
            .map_err(|error| format!("WASM folding audit failed: {error}"))?;
    }
    let proof = chain
        .finish_with_audit()
        .map_err(|error| format!("WASM folding-audit finalization failed: {error}"))?;
    let prove_ms = started.elapsed().as_millis();
    let started = Instant::now();
    neo_fold_clean::verify_uncompressed_audit(&prep.prep, &proof)
        .map_err(|error| format!("WASM folding-audit replay failed: {error}"))?;
    if proof.proof.state.semantic_state_digest != neo_wasm::semantic_state_digest(final_state) {
        return Err("WASM folding-audit final state does not match the traced execution".to_string());
    }
    let verify_ms = started.elapsed().as_millis();
    Ok(proof_response(
        prepared,
        program.run.results.clone(),
        prove_ms,
        verify_ms,
    ))
}

fn prove_with_nebula_memory(
    prepared: &PreparedProof,
    prep: &neo_wasm::WasmNebulaPreprocessing,
) -> Result<ProofResponse, String> {
    let program = &prepared.program;
    let final_state = program
        .normalized
        .last()
        .expect("preprocessing rejects empty traces")
        .state_after;
    let started = Instant::now();
    let proof = neo_wasm::prove(prep, &program.normalized)
        .map_err(|error| format!("Nebula proof generation failed: {error}"))?;
    let prove_ms = started.elapsed().as_millis();
    let started = Instant::now();
    neo_wasm::verify(prep, &proof, final_state).map_err(|error| format!("Nebula verification failed: {error}"))?;
    let verify_ms = started.elapsed().as_millis();
    Ok(proof_response(
        prepared,
        program.run.results.clone(),
        prove_ms,
        verify_ms,
    ))
}

fn proof_response(prepared: &PreparedProof, result: Vec<String>, prove_ms: u128, verify_ms: u128) -> ProofResponse {
    let metadata = prepared.metadata;
    ProofResponse {
        mode: metadata.mode,
        memory_consistency: metadata.memory_consistency,
        security: metadata.security,
        result,
        trapped: prepared
            .program
            .normalized
            .last()
            .is_some_and(|row| row.state_after.trapped),
        normalized_rows: metadata.normalized_rows,
        batch_size: metadata.batch_size,
        folds: metadata.folds,
        padded_rows: metadata.padded_rows,
        prove_ms,
        verify_ms,
    }
}

pub struct TracedProgram {
    wasm: Vec<u8>,
    artifacts: WasmProgramArtifacts,
    run: WasmtimeTraceRun,
    normalized: Vec<WasmVmStep>,
}

fn trace_source(request: &TraceRequest) -> Result<TracedProgram, String> {
    if request.source.len() > 256 * 1024 {
        return Err("program source exceeds the 256 KiB demo limit".to_string());
    }

    let wasm = wat::parse_str(&request.source).map_err(|error| format!("invalid WAT: {error}"))?;
    let artifacts = extract_wasm_program_artifacts(&wasm)
        .map_err(|error| format!("could not extract program artifacts: {error:?}"))?;
    let run =
        collect_wasmtime_steps(&wasm, "main", &[]).map_err(|error| format!("could not execute main(): {error:?}"))?;
    let normalized =
        traces_from_wasmtime_steps(&run.steps).map_err(|error| format!("could not normalize trace: {error:?}"))?;
    if normalized.is_empty() {
        return Err("normalized trace is empty".to_string());
    }

    Ok(TracedProgram {
        wasm,
        artifacts,
        run,
        normalized,
    })
}

fn project_trace_response(prepared: &TracedProgram) -> TraceResponse {
    let final_state = prepared.normalized.last().map(|row| row.state_after);

    TraceResponse {
        program: ProgramSummary {
            wasm_bytes: prepared.wasm.len(),
            decoded_instructions: prepared.artifacts.tables.program_decode.len(),
            control_edges: prepared.artifacts.tables.pc_rom.len(),
            function_entries: prepared.artifacts.tables.function_entries.len(),
            has_linear_memory: prepared.artifacts.tables.initial_memory_pages.is_some(),
        },
        execution: ExecutionSummary {
            results: prepared.run.results.clone(),
            initial_locals: prepared.run.initial_locals.clone(),
            normalized_rows: prepared.normalized.len(),
            halted: final_state.is_some_and(|state| state.halted),
            trapped: final_state.is_some_and(|state| state.trapped),
        },
        rows: prepared.normalized.iter().map(project_row).collect(),
    }
}

fn nebula_demo_params() -> Result<neo_fold_clean::paper::params::Params, String> {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .map_err(|error| format!("could not construct demo parameters: {error}"))?;
    Ok(neo_fold_clean::paper::params::Params::test_only_from_neo_params(raw))
}

fn ivc_demo_params() -> Result<neo_fold_clean::paper::params::Params, String> {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        2,
        1 << 15,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        40,
    )
    .map_err(|error| format!("could not construct demo parameters: {error}"))?;
    Ok(neo_fold_clean::paper::params::Params::test_only_from_neo_params(raw))
}

fn digest_hex(digest: [neo_math::F; 4]) -> [String; 4] {
    digest.map(|limb| format!("{:016x}", limb.as_canonical_u64()))
}

fn bytes_hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn project_row(row: &WasmVmStep) -> TraceRow {
    let output = row
        .state_after
        .output
        .enabled
        .then_some(u64::from(row.state_after.output.value_lo) | (u64::from(row.state_after.output.value_hi) << 32));
    TraceRow {
        cycle: row.cycle,
        kind: match row.row_kind {
            WasmRowKind::Program => "program".to_string(),
            WasmRowKind::Aux(opcode) => format!("aux::{opcode:?}"),
        },
        opcode: format!("{:?}", row.opcode),
        pc_before: row.state_before.pc,
        pc_after: row.state_after.pc,
        sp_before: row.state_before.sp,
        sp_after: row.state_after.sp,
        call_depth_before: row.state_before.call_stack_depth,
        call_depth_after: row.state_after.call_stack_depth,
        control_choice: row.control_choice,
        edge: format!("{:?}", row.pc_edge_kind),
        stack_reads: [row.stack_read0, row.stack_read1, row.stack_read2]
            .into_iter()
            .flatten()
            .map(project_access)
            .collect(),
        stack_write: row.stack_write0.map(project_access),
        local_index: row.local_index,
        local_read: row.local_read_value,
        local_write: row.local_write_value,
        global_index: row.global_index,
        global_read: row.global_read_value,
        global_write: row.global_write_value,
        output,
        halted: row.state_after.halted,
        trapped: row.state_after.trapped,
    }
}

fn project_access(access: StackValueAccess) -> ValueAccess {
    ValueAccess {
        address: access.addr_lo,
        lo: access.value_lo,
        hi: access.value_hi,
    }
}
