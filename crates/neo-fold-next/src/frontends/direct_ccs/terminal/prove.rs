//! Owns direct-CCS terminal F' proof generation.
//!
//! The main prover function is intentionally written as an ordered flow:
//! prepare the terminal relation, measure/setup the committed relation, prove
//! it, then package the public image and performance accounting.

use std::time::Instant;

use super::super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use super::super::state::{
    DirectCcsFPrimeCircuit, DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkProof,
    DirectCcsTerminalFPrimeCircuit,
};
use super::committed::{
    prove_direct_ccs_terminal_committed_relation, setup_direct_ccs_terminal_committed_relation_cached,
    DirectCcsTerminalCommittedKeyPair, DirectCcsTerminalCommittedPerf, DirectCcsTerminalCommittedRelation,
};
use super::final_ce::{measure_direct_final_ce_relation_breakdown, DirectFinalCeRelationBreakdown};
use super::measure::{measure_direct_ccs_f_prime_constraints, DirectCcsFPrimeConstraintBreakdown};

pub(crate) fn prove_direct_ccs_f_prime_circuit(
    circuit: DirectCcsFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<
    (
        DirectCcsIvcSnark,
        DirectCcsIvcSnarkVerifierKey,
        DirectCcsFPrimeSnarkPerf,
    ),
    DirectCcsFPrimeSnarkError,
> {
    let terminal = prepare_terminal_relation(circuit, emit)?;
    let setup = setup_terminal_committed_relation(&terminal, emit)?;
    let proved = prove_terminal_committed_step(&terminal, &setup, emit)?;
    package_terminal_snark(terminal, setup, proved)
}

fn prepare_terminal_relation(
    circuit: DirectCcsFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<TerminalRelationPackage, DirectCcsFPrimeSnarkError> {
    let terminal_circuit = circuit.terminal_circuit(true);
    let terminal_shape = measure_terminal_shape(&terminal_circuit, emit)?;
    let terminal_relation = build_terminal_committed_relation(&terminal_circuit, emit)?;
    let final_ce_shape = measure_final_ce_shape(&circuit, emit)?;
    Ok(TerminalRelationPackage {
        terminal_circuit,
        terminal_shape,
        terminal_relation,
        final_ce_shape,
    })
}

fn measure_terminal_shape(
    terminal_circuit: &DirectCcsTerminalFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<DirectCcsFPrimeConstraintBreakdown, DirectCcsFPrimeSnarkError> {
    emit("direct_ccs_ivc.phase=terminal_shape_measure.start");
    let breakdown = measure_direct_ccs_f_prime_constraints(terminal_circuit)?;
    emit("direct_ccs_ivc.phase=terminal_shape_measure.done");
    Ok(breakdown)
}

fn build_terminal_committed_relation(
    terminal_circuit: &DirectCcsTerminalFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<DirectCcsTerminalCommittedRelation, DirectCcsFPrimeSnarkError> {
    emit("direct_ccs_ivc.phase=terminal_committed_relation.start");
    let relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(terminal_circuit.clone())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_relation.done");
    Ok(relation)
}

fn measure_final_ce_shape(
    circuit: &DirectCcsFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<DirectFinalCeRelationBreakdown, DirectCcsFPrimeSnarkError> {
    emit("direct_ccs_ivc.phase=final_ce_measure.start");
    let breakdown = measure_direct_final_ce_relation_breakdown(
        &circuit.params,
        &circuit.structure,
        &circuit.final_claims,
        &circuit.final_witnesses,
    )?;
    emit("direct_ccs_ivc.phase=final_ce_measure.done");
    Ok(breakdown)
}

fn setup_terminal_committed_relation(
    terminal: &TerminalRelationPackage,
    emit: &mut dyn FnMut(&str),
) -> Result<TerminalSetupPackage, DirectCcsFPrimeSnarkError> {
    let setup_started = Instant::now();
    let terminal_committed_perf = measure_terminal_committed_relation(terminal, emit)?;
    emit("direct_ccs_ivc.phase=terminal_committed_setup.start");
    let keys =
        setup_direct_ccs_terminal_committed_relation_cached(&terminal.terminal_relation, terminal_committed_perf)
            .map_err(|err| DirectCcsFPrimeSnarkError::Setup(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_setup.done");

    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let terminal_committed_perf = keys.perf.clone();
    Ok(TerminalSetupPackage {
        r1cs_sizes: terminal_committed_perf.sizes,
        r1cs_nnz: terminal_committed_perf.nnz,
        terminal_committed_perf,
        keys,
        setup_ms,
    })
}

fn measure_terminal_committed_relation(
    terminal: &TerminalRelationPackage,
    emit: &mut dyn FnMut(&str),
) -> Result<DirectCcsTerminalCommittedPerf, DirectCcsFPrimeSnarkError> {
    emit("direct_ccs_ivc.phase=terminal_committed_measure.start");
    let terminal_committed_perf = terminal
        .terminal_relation
        .measure()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    emit_terminal_measurement_log(&terminal_committed_perf, &terminal.terminal_shape, emit);
    emit("direct_ccs_ivc.phase=terminal_committed_measure.done");
    Ok(terminal_committed_perf)
}

fn emit_terminal_measurement_log(
    terminal_committed_perf: &DirectCcsTerminalCommittedPerf,
    terminal_shape: &DirectCcsFPrimeConstraintBreakdown,
    emit: &mut dyn FnMut(&str),
) {
    let measure_msg = format!(
        "direct_ccs_ivc.terminal_committed_shape constraints={} public_inputs={} committed_width={} source_values={} commitment_words={}",
        terminal_committed_perf.constraints,
        terminal_committed_perf.public_inputs,
        terminal_committed_perf.committed_width,
        terminal_committed_perf.source_values,
        terminal_committed_perf.commitment_words
    );
    emit(&measure_msg);
    let log_lines = terminal_committed_perf
        .breakdown_log_lines()
        .into_iter()
        .chain(terminal_shape.chunk_stage_log_lines())
        .chain(terminal_shape.construction2_fold_breakdown.log_lines());
    for line in log_lines {
        emit(&line);
    }
}

fn prove_terminal_committed_step(
    terminal: &TerminalRelationPackage,
    setup: &TerminalSetupPackage,
    emit: &mut dyn FnMut(&str),
) -> Result<TerminalProofPackage, DirectCcsFPrimeSnarkError> {
    let prove_started = Instant::now();
    emit("direct_ccs_ivc.phase=terminal_committed_prove.start");
    let (proof, pcs_ms) = prove_direct_ccs_terminal_committed_relation(&setup.keys.prover, &terminal.terminal_relation)
        .map_err(|err| DirectCcsFPrimeSnarkError::Prove(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_prove.done");
    Ok(TerminalProofPackage {
        proof,
        pcs_ms,
        prove_ms: prove_started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn package_terminal_snark(
    terminal: TerminalRelationPackage,
    setup: TerminalSetupPackage,
    proved: TerminalProofPackage,
) -> Result<
    (
        DirectCcsIvcSnark,
        DirectCcsIvcSnarkVerifierKey,
        DirectCcsFPrimeSnarkPerf,
    ),
    DirectCcsFPrimeSnarkError,
> {
    let proof = DirectCcsFPrimeSnarkProof {
        construction2_u_i: terminal.terminal_relation.public_boundary().clone(),
        terminal_f_prime_committed_step_proof: proved.proof,
    };
    let public_image = terminal
        .terminal_circuit
        .public_image(proof.construction2_u_i.clone());
    let verifier_key = DirectCcsIvcSnarkVerifierKey::from_terminal_f_prime(setup.keys.verifier.clone());
    let perf = terminal_perf_accounting(&terminal, &setup, proved.pcs_ms, proved.prove_ms, &proof)?;
    Ok((DirectCcsIvcSnark::from_parts(proof, public_image), verifier_key, perf))
}

fn terminal_perf_accounting(
    terminal: &TerminalRelationPackage,
    setup: &TerminalSetupPackage,
    pcs_ms: f64,
    prove_ms: f64,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkError> {
    let prep_ms = 0.0;
    let encode_ms = 0.0;
    let final_proof_bytes = bincode::serialize(proof)
        .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?
        .len();
    Ok(DirectCcsFPrimeSnarkPerf {
        setup_ms: setup.setup_ms,
        prep_ms,
        prove_ms,
        encode_ms,
        total_prove_ms: prep_ms + prove_ms + encode_ms,
        total_verify_ms: 0.0,
        r1cs_sizes: setup.r1cs_sizes,
        r1cs_nnz: setup.r1cs_nnz,
        pcs_ms,
        final_proof_bytes,
        snark_bytes: proof.snark_bytes_len(),
        public_inputs: setup.terminal_committed_perf.public_inputs,
        chunk_constraints_first4: terminal.terminal_shape.chunk_constraints_first4,
        chunk_constraints_by_chunk: terminal.terminal_shape.chunk_constraints_by_chunk.clone(),
        chunk_count: terminal.terminal_shape.chunk_count,
        public_link_constraints: terminal.terminal_shape.public_link_constraints,
        construction2_fold_constraints: terminal.terminal_shape.construction2_fold_constraints,
        construction2_fold_final_ce_consistency_constraints: 0,
        chunk_done_constraints: terminal.terminal_shape.chunk_done_constraints,
        final_ce_relation_constraints: terminal.final_ce_shape.total_relation_constraints,
        final_ce_relation_breakdown: terminal.final_ce_shape.relation_breakdown,
        final_ce_bundle_constraints: 0,
        final_ce_bundle_digest_constraints: 0,
        final_ce_bundle_digest_match_constraints: 0,
        terminal_f_prime_constraints: setup.terminal_committed_perf.constraints,
        terminal_committed_width: setup.terminal_committed_perf.committed_width,
        terminal_commitment_words: setup.terminal_committed_perf.commitment_words,
        terminal_source_values: setup.terminal_committed_perf.source_values,
        terminal_source_bit_values: setup.terminal_committed_perf.source_bit_values,
        terminal_source_u32_values: setup.terminal_committed_perf.source_u32_values,
        terminal_source_u64_values: setup.terminal_committed_perf.source_u64_values,
        terminal_unclassified_private_values: setup.terminal_committed_perf.unclassified_private_values,
        terminal_committed_breakdown: setup.terminal_committed_perf.breakdown,
        final_ce_r1cs_sizes: [0; 10],
    })
}

struct TerminalRelationPackage {
    terminal_circuit: DirectCcsTerminalFPrimeCircuit,
    terminal_shape: DirectCcsFPrimeConstraintBreakdown,
    terminal_relation: DirectCcsTerminalCommittedRelation,
    final_ce_shape: DirectFinalCeRelationBreakdown,
}

struct TerminalSetupPackage {
    keys: DirectCcsTerminalCommittedKeyPair,
    setup_ms: f64,
    terminal_committed_perf: DirectCcsTerminalCommittedPerf,
    r1cs_sizes: [usize; 10],
    r1cs_nnz: usize,
}

struct TerminalProofPackage {
    proof: super::committed::DirectCcsTerminalCommittedProof,
    pcs_ms: f64,
    prove_ms: f64,
}
