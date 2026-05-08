//! Measures the direct terminal F' circuit shape without owning proof generation.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem};

use super::circuit_util::{
    alloc_digest_constant, digest32_as_spartan_fields, direct_accumulator_digest_circuit_from_claims,
    direct_terminal_accumulator_digest_range, direct_terminal_construction2_accumulator_digest_range,
    enforce_digest_eq_constant, enforce_digest_fields_public_io, enforce_direct_current_boundary_transition,
    enforce_direct_public_trace_transition, enforce_direct_state_x_in_digest, enforce_direct_state_x_out_public_digest,
    enforce_direct_terminal_final_ce_consistency, public_digest_input,
};
use super::construction2_fold::{measure_direct_construction2_fold, DirectCcsConstruction2FoldBreakdown};
use super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsTerminalFPrimeCircuit};
use super::ivc_helpers::{alloc_initial_claim_bundle, alloc_initial_transcript};
use super::public_image::DIRECT_CCS_TRIVIAL_PC;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanCircuit};
use crate::superneo_nifs_circuit::{
    synthesize_superneo_nifs_chunk_with_stage_breakdown, SuperNeoNifsChunkFullBreakdown,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirectCcsFPrimeConstraintBreakdown {
    pub(crate) public_inputs: usize,
    pub(crate) chunk_count: usize,
    pub(crate) chunk_constraints_first4: [usize; 4],
    pub(crate) chunk_constraints_by_chunk: Vec<usize>,
    pub(crate) chunk_stage_breakdowns: Vec<SuperNeoNifsChunkFullBreakdown>,
    pub(crate) public_link_constraints: usize,
    pub(crate) construction2_fold_constraints: usize,
    pub(crate) construction2_fold_breakdown: DirectCcsConstruction2FoldBreakdown,
    pub(crate) chunk_done_constraints: usize,
    pub(crate) terminal_f_prime_constraints: usize,
}

impl DirectCcsFPrimeConstraintBreakdown {
    pub(crate) fn chunk_stage_log_lines(&self) -> Vec<String> {
        let mut lines = vec!["direct_ccs_ivc.terminal_nifs_stage_breakdown chunk|stage|rows|primitive".to_owned()];
        for (chunk, breakdown) in self.chunk_stage_breakdowns.iter().enumerate() {
            push_chunk_stage(
                &mut lines,
                chunk,
                "chunk_meta",
                breakdown.stages.chunk_meta,
                "append chunk metadata into the Fiat-Shamir transcript",
            );
            push_chunk_stage(
                &mut lines,
                chunk,
                "pi_ccs",
                breakdown.stages.pi_ccs,
                "verify Pi_CCS sumchecks, fresh/ME outputs, and terminal identities",
            );
            push_chunk_stage(
                &mut lines,
                chunk,
                "pi_rlc",
                breakdown.stages.pi_rlc,
                "sample rho and check RLC parent CE public surface",
            );
            push_chunk_stage(
                &mut lines,
                chunk,
                "pi_dec",
                breakdown.stages.pi_dec,
                "check DEC recomposition into carried CE children",
            );
            push_chunk_stage(
                &mut lines,
                chunk,
                "total",
                breakdown.stages.total,
                "complete SuperNeo NIFS.V chunk verifier body",
            );
        }
        lines.push(
            "direct_ccs_ivc.terminal_pi_ccs_substage_breakdown chunk|substage|rows|aux_columns|public_columns|primitive"
                .to_owned(),
        );
        for (chunk, breakdown) in self.chunk_stage_breakdowns.iter().enumerate() {
            for detail in &breakdown.pi_ccs_details {
                lines.push(format!(
                    "direct_ccs_ivc.terminal_pi_ccs_substage_breakdown {chunk}|{}|{}|{}|{}|{}",
                    detail.stage,
                    detail.rows,
                    detail.aux_columns,
                    detail.public_columns,
                    pi_ccs_substage_primitive(detail.stage)
                ));
            }
        }
        lines
    }
}

fn push_chunk_stage(lines: &mut Vec<String>, chunk: usize, stage: &str, rows: usize, primitive: &str) {
    lines.push(format!(
        "direct_ccs_ivc.terminal_nifs_stage_breakdown {chunk}|{stage}|{rows}|{primitive}"
    ));
}

fn pi_ccs_substage_primitive(stage: &str) -> &'static str {
    match stage {
        "fresh_claim_and_public_chunk" => "allocate fresh CCS public surface and bind public step/chunk digests",
        "bind_header" => "absorb SuperNeo shape, matrix, parameter, and chunk-instance header",
        "bind_me_inputs" => "absorb carried CE accumulator inputs for the Pi_CCS transcript",
        "sample_challenges" => "derive Pi_CCS alpha/gamma/beta transcript challenges",
        "fe_sumcheck" => "verify FE sumcheck transcript and derive r prime",
        "nc_sumcheck" => "verify norm/evaluation sumcheck transcript and derive s column point",
        "fold_digest" => "digest the Pi_CCS transcript state after sumchecks",
        "alloc_output_ce_surfaces" => "allocate Pi_CCS CE output public surfaces",
        "output_binding" => "bind output CE surfaces back to fresh and carried inputs",
        "terminal_fe_identity" => "check final FE sumcheck identity",
        "terminal_nc_identity" => "check final norm/evaluation sumcheck identity",
        _ => "uncategorized Pi_CCS row or variable",
    }
}

pub(crate) fn measure_direct_ccs_f_prime_constraints(
    circuit: &DirectCcsTerminalFPrimeCircuit,
) -> Result<DirectCcsFPrimeConstraintBreakdown, DirectCcsFPrimeSnarkError> {
    let mut cs = ShapeCS::<NeoFoldDeciderEngine>::new();
    let public_values = circuit
        .public_values()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let public_inputs = public_values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let mut out = DirectCcsFPrimeConstraintBreakdown {
        public_inputs: public_inputs.len(),
        chunk_count: circuit.chunks.len(),
        ..DirectCcsFPrimeConstraintBreakdown::default()
    };
    let mut transcript = alloc_initial_transcript(&mut cs, circuit.initial_transcript.as_ref())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let mut carried = alloc_initial_claim_bundle(&mut cs, &circuit.initial_claims)
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let mut last_chunk_digest = None;

    let before_link = cs.num_constraints();
    let accumulator_in_digest = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_accumulator_in_digest"),
        &circuit.params,
        carried.effective_claims(),
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "direct_terminal_accumulator_in_digest_private"),
        &accumulator_in_digest,
        circuit.accumulator_in_digest,
        "direct_terminal_accumulator_in_digest_private",
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let construction2_accumulator_in_digest = alloc_digest_constant(
        &mut cs.namespace(|| "direct_terminal_construction2_accumulator_in_digest"),
        circuit.construction2_accumulator_in_digest,
        "direct_terminal_construction2_accumulator_in_digest",
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    enforce_direct_state_x_in_digest(
        &mut cs.namespace(|| "direct_terminal_x_in_digest"),
        circuit.vk_fs_digest,
        &circuit.mat_digest,
        circuit.chunk_count_in,
        circuit.step_count_in,
        circuit.initial_boundary_digest,
        circuit.current_boundary_in_digest,
        DIRECT_CCS_TRIVIAL_PC,
        &accumulator_in_digest,
        &construction2_accumulator_in_digest,
        circuit.public_trace_in_digest,
        circuit.x_in.bytes(),
        "direct_terminal_x_in_digest",
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("terminal x_in digest failed: {err}")))?;
    out.public_link_constraints += cs.num_constraints() - before_link;

    for (chunk_index, chunk) in circuit.chunks.iter().enumerate() {
        let before_chunk = cs.num_constraints();
        let (next, chunk_digest, stage_breakdown) = synthesize_superneo_nifs_chunk_with_stage_breakdown(
            &circuit.params,
            &circuit.structure,
            circuit.dims,
            &circuit.mat_digest,
            &mut cs,
            chunk_index,
            &chunk.cover,
            &chunk.replay,
            &mut transcript,
            carried,
            Some((
                &accumulator_in_digest,
                digest32_as_spartan_fields(circuit.accumulator_in_digest),
            )),
        )
        .map_err(|err| {
            DirectCcsFPrimeSnarkError::Synthesis(format!("latest NIFS.V chunk {chunk_index} failed: {err}"))
        })?;
        let chunk_constraints = cs.num_constraints() - before_chunk;
        if chunk_index < out.chunk_constraints_first4.len() {
            out.chunk_constraints_first4[chunk_index] = chunk_constraints;
        }
        out.chunk_constraints_by_chunk.push(chunk_constraints);
        out.chunk_stage_breakdowns.push(stage_breakdown);

        let before_done = cs.num_constraints();
        transcript
            .append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )
            .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        out.chunk_done_constraints += cs.num_constraints() - before_done;
        carried = next;
        last_chunk_digest = Some(chunk_digest);
    }

    let before_output_link = cs.num_constraints();
    let accumulator_digest = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_accumulator_digest"),
        &circuit.params,
        carried.effective_claims(),
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| "direct_terminal_accumulator_digest_public"),
        &accumulator_digest,
        &public_inputs,
        direct_terminal_accumulator_digest_range(),
        "direct_terminal_accumulator_digest_public",
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("terminal accumulator digest failed: {err}")))?;
    let last_chunk_digest = last_chunk_digest
        .ok_or_else(|| DirectCcsFPrimeSnarkError::Synthesis("measured direct F' missing latest chunk digest".into()))?;
    let current_boundary_out_digest = enforce_direct_current_boundary_transition(
        &mut cs.namespace(|| "direct_terminal_current_boundary_transition"),
        &public_inputs,
        circuit.current_boundary_in_digest,
        &last_chunk_digest,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("current boundary transition failed: {err}")))?;
    let public_trace_out_digest = enforce_direct_public_trace_transition(
        &mut cs.namespace(|| "direct_terminal_public_trace_transition"),
        &public_inputs,
        circuit.public_trace_in_digest,
        &last_chunk_digest,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("public trace transition failed: {err}")))?;
    let construction2_accumulator_out_digest =
        public_digest_input(&public_inputs, direct_terminal_construction2_accumulator_digest_range())
            .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    out.public_link_constraints += cs.num_constraints() - before_output_link;
    out.construction2_fold_breakdown = measure_direct_construction2_fold(
        &mut cs,
        circuit.construction2_fold.as_ref(),
        &public_inputs,
        circuit.construction2_accumulator_in_digest,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("Construction-2 fold failed: {err}")))?;
    out.construction2_fold_constraints = out.construction2_fold_breakdown.total.rows;
    let before_output_link = cs.num_constraints();
    enforce_direct_state_x_out_public_digest(
        &mut cs.namespace(|| "direct_terminal_x_out_digest"),
        &public_inputs,
        circuit.vk_fs_digest,
        &circuit.mat_digest,
        circuit.chunk_count_out,
        circuit.step_count_out,
        circuit.initial_boundary_digest,
        &current_boundary_out_digest,
        DIRECT_CCS_TRIVIAL_PC,
        &accumulator_digest,
        &construction2_accumulator_out_digest,
        &public_trace_out_digest,
        "direct_terminal_x_out_digest",
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("terminal x_out digest failed: {err}")))?;
    out.public_link_constraints += cs.num_constraints() - before_output_link;
    if circuit.prove_final_ce {
        enforce_direct_terminal_final_ce_consistency(
            &mut cs.namespace(|| "direct_terminal_final_ce"),
            &circuit.params,
            &circuit.structure,
            carried.effective_claims(),
            &circuit.final_witnesses,
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("inline final CE consistency failed: {err}")))?;
    }
    out.terminal_f_prime_constraints = cs.num_constraints();
    Ok(out)
}
