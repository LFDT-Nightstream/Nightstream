//! Owns the WASM semantic kernel proof boundary above Stage 1/2/3.

mod types;

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::lifecycle::verify_uncompressed as clean_verify_uncompressed;
use neo_fold_clean::paper::digest::structure_digest;

use super::builder::WasmTraceBuilder;
use super::ccs::WasmVmSpec;
use super::relation::{prove_wasm_relation, verify_wasm_relation};
use super::step_build::WasmStepBuild;

pub use types::{
    WasmKernelError, WasmKernelOutput, WasmKernelProof, WasmKernelProverInput, WasmKernelPublicInput,
    WasmKernelRunProof, WasmKernelVerifierInput,
};

pub fn prove_simple_kernel(
    input: &WasmKernelProverInput<'_>,
) -> Result<(WasmKernelOutput, WasmKernelProof), WasmKernelError> {
    let prepared_steps = build_prepared(input.trace)?;

    let relation = prove_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
    )
    .map_err(WasmKernelError::Relation)?;
    if relation.boundary_rows.len() != prepared_steps.len() {
        return Err(WasmKernelError::Bridge(format!(
            "wasm relation exported {} boundary rows for {} prepared steps",
            relation.boundary_rows.len(),
            prepared_steps.len()
        )));
    }

    Ok((WasmKernelOutput { prepared_steps }, WasmKernelProof { relation }))
}

pub fn verify_simple_kernel(
    input: &WasmKernelVerifierInput<'_>,
    proof: &WasmKernelProof,
) -> Result<WasmKernelOutput, WasmKernelError> {
    let prepared_steps = build_prepared(input.trace)?;

    verify_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
        &proof.relation,
    )
    .map_err(WasmKernelError::Relation)?;

    if proof.relation.boundary_rows.len() != prepared_steps.len() {
        return Err(WasmKernelError::Bridge(format!(
            "wasm relation exported {} boundary rows for {} prepared steps",
            proof.relation.boundary_rows.len(),
            prepared_steps.len()
        )));
    }

    Ok(WasmKernelOutput { prepared_steps })
}

pub fn prove_kernel_run(
    prep: &R1csFPrimePreprocessing,
    input: &WasmKernelProverInput<'_>,
) -> Result<(WasmKernelOutput, WasmKernelRunProof), WasmKernelError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;
    let (output, kernel) = prove_simple_kernel(input)?;

    let mut chain =
        R1csChainBuilder::new(prep).map_err(|err| WasmKernelError::Bridge(format!("R1csChainBuilder::new: {err}")))?;
    for step in &output.prepared_steps {
        chain
            .append_assignment(step.assignment.clone())
            .map_err(|err| WasmKernelError::Bridge(format!("append_assignment: {err}")))?;
    }
    let main_run = chain
        .finish()
        .map_err(|err| WasmKernelError::Bridge(format!("finish: {err}")))?;

    Ok((output, WasmKernelRunProof { kernel, main_run }))
}

pub fn verify_kernel_run(
    prep: &R1csFPrimePreprocessing,
    input: &WasmKernelVerifierInput<'_>,
    proof: &WasmKernelRunProof,
) -> Result<WasmKernelOutput, WasmKernelError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;
    let output = verify_simple_kernel(input, &proof.kernel)?;
    clean_verify_uncompressed(&prep.prep, &proof.main_run)
        .map_err(|err| WasmKernelError::Bridge(format!("verify_uncompressed: {err}")))?;
    // BINDING-PENDING: `proof.main_run` is verified as a valid IVC chain
    // under `prep`, but nothing here binds it to `input.trace`. An attacker
    // could swap in a `main_run` from a different wasm execution under the
    // same preprocessing and pass both this verify and `verify_simple_kernel`
    // (which only checks the wasm relation against the trace).
    // The trace ↔ chain binding is the shout/twist layer's job: the program
    // ROM is a public input, the shout proof indexes into it via the (pc,
    // opcode) columns to attest the executed opcodes are exactly those of
    // the published program. Until that lands, callers must treat
    // `verify_kernel_run` as "the chain is a valid wasm execution under this
    // preprocessing" rather than "the chain is the specific wasm execution
    // for this trace".
    Ok(output)
}

/// Reject a `prep` whose underlying R1CS shape or public-input split does
/// not match the canonical wasm VM. Without this gate a caller could prove
/// under a different (same-dimension) R1CS and the wasm relation layer
/// would still accept.
///
/// Compared digests are over the *app* R1CS-to-CCS embedding, not the
/// F'-augmented `prep.prep.structure_digest()` — the latter is the wasm
/// R1CS wrapped in F' bit-decomposition + recursive-plan rows, so it
/// never equals the bare wasm CCS digest.
fn validate_wasm_preprocessing(prep: &R1csFPrimePreprocessing, vm: &WasmVmSpec) -> Result<(), WasmKernelError> {
    let core = vm.core_ccs_spec();
    let expected = structure_digest(&core.structure);
    let actual = structure_digest(&prep.r1cs.to_structure());
    if actual != expected {
        return Err(WasmKernelError::Bridge(
            "preprocessing R1CS shape does not match the canonical wasm VM".into(),
        ));
    }
    if prep.r1cs.m_in() != core.m_in {
        return Err(WasmKernelError::Bridge(format!(
            "preprocessing m_in {} does not match wasm m_in {}",
            prep.r1cs.m_in(),
            core.m_in
        )));
    }
    Ok(())
}

fn build_prepared(trace: &[crate::ir::WasmStepTrace]) -> Result<Vec<WasmStepBuild>, WasmKernelError> {
    let vm = crate::ccs::WasmVmSpec::default();
    let builder = WasmTraceBuilder::new();
    builder
        .build_steps(&vm, trace)
        .map_err(|err| WasmKernelError::InvalidWitness(err.to_string()))
}
