//! Owns direct-CCS terminal F' proof verification.

use super::super::public_image::{direct_state_x_out, DIRECT_CCS_TRIVIAL_PC};
use super::super::state::{
    DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkProof, DirectCcsIvcState, DirectCcsTerminalFPrimeCircuit,
};
use super::committed::{
    setup_direct_ccs_terminal_committed_relation_cached, verify_direct_ccs_terminal_committed_relation,
    DirectCcsTerminalCommittedKeyPair, DirectCcsTerminalCommittedRelation,
};

pub fn verify_direct_ccs_terminal_snark_against_state(
    state: &DirectCcsIvcState,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let context = prepare_terminal_verify_context(state)?;
    enforce_terminal_boundary_matches_state(&context.terminal_circuit, proof)?;
    verify_terminal_committed_step(&context, proof)
}

fn prepare_terminal_verify_context(
    state: &DirectCcsIvcState,
) -> Result<TerminalVerifyContext, DirectCcsFPrimeSnarkError> {
    let terminal_circuit = state.latest_circuit()?.terminal_circuit(true);
    let terminal_relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(terminal_circuit.clone())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let terminal_committed_perf = terminal_relation
        .measure()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let keys = setup_direct_ccs_terminal_committed_relation_cached(&terminal_relation, terminal_committed_perf)
        .map_err(|err| DirectCcsFPrimeSnarkError::Setup(err.to_string()))?;
    Ok(TerminalVerifyContext { terminal_circuit, keys })
}

fn enforce_terminal_boundary_matches_state(
    terminal_circuit: &DirectCcsTerminalFPrimeCircuit,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let expected_x_i = direct_state_x_out(
        terminal_circuit.vk_fs_digest,
        &terminal_circuit.mat_digest,
        terminal_circuit.chunk_count_out,
        terminal_circuit.step_count_out,
        terminal_circuit.initial_boundary_digest,
        terminal_circuit.current_boundary_out_digest,
        DIRECT_CCS_TRIVIAL_PC,
        terminal_circuit.accumulator_out_digest,
        terminal_circuit.construction2_accumulator_out_digest,
        terminal_circuit.public_trace_out_digest,
    );
    if proof.construction2_u_i.x_i != expected_x_i {
        return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
    }
    Ok(())
}

fn verify_terminal_committed_step(
    context: &TerminalVerifyContext,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    verify_direct_ccs_terminal_committed_relation(
        &context.keys.verifier,
        &context.terminal_circuit.terminal_public_values(),
        &proof.construction2_u_i,
        &proof.terminal_f_prime_committed_step_proof,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Verify(err.to_string()))
}

struct TerminalVerifyContext {
    terminal_circuit: DirectCcsTerminalFPrimeCircuit,
    keys: DirectCcsTerminalCommittedKeyPair,
}
