//! Owns the terminal chunk-step compression circuit substrate.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::engines::utils::{build_dims_and_policy, Dims};
use neo_transcript::Transcript;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use crate::rv64im::chunk_relation::rv64im_chunk_replay_witness_digest;
use crate::rv64im::chunk_step_ivc::{
    Rv64imChunkStepIvcPublishedTarget, Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::final_relation::RV64IM_CHUNK_DONE_RAW_TAG;
use crate::rv64im::ivc_snark::{hash_packed_goldilocks_fields, Rv64imDeciderEngine, ShapeCS, SpartanCircuit, SpartanF};
use crate::rv64im::kernel::rv64im_cached_root_main_lane_optimized_cache;
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_with_shared_point, enforce_claim_projection_eq_native, packed_bytes_field_values,
    CeClaimVar,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::{
    alloc_const_field_values, debug_check_rv64im_rlc_public_x_native_values,
    debug_compare_rv64im_pi_ccs_transcript_state, debug_compare_rv64im_pi_rlc_rho_mats,
    debug_locate_rv64im_pi_ccs_late_transcript_stage, debug_measure_rv64im_main_relation_chunk_stage_ranges,
    debug_measure_rv64im_pi_rlc_stage_ranges, debug_measure_rv64im_rlc_public_stage_ranges, digest_const_inputs,
    enforce_digest_eq, next_public_digest, prepare_rv64im_chunk_step_ivc_circuit_inputs,
    synthesize_rv64im_main_relation_chunk, Rv64imChunkBoundaryMode, Rv64imChunkBoundaryPlan, Rv64imChunkStepIvcShape,
    Rv64imChunkStepIvcSpartanError, Rv64imClaimBundle,
};
use crate::rv64im::main_relation_trace::{
    debug_describe_rv64im_main_relation_pi_ccs_terminal_state_mismatch,
    debug_describe_rv64im_main_relation_pi_rlc_parent_mismatch,
    debug_describe_rv64im_main_relation_pi_rlc_x_flat_mismatch,
    debug_replay_rv64im_main_relation_pi_ccs_transcript_state, Rv64imMainCircuitChunkCover,
    Rv64imMainCircuitChunkReplaySurface, Rv64imMainCircuitChunkTrace,
};

#[derive(Clone)]
pub(super) struct Rv64imChunkStepIvcCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    published_target: Rv64imChunkStepIvcPublishedTarget,
    witness: Rv64imChunkStepIvcWitness,
    cover_chunk: Rv64imMainCircuitChunkCover,
    effective_chunk: Rv64imMainCircuitChunkTrace,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Rv64imChunkStepIvcConstraintCheckpoints {
    pub public_bind_end: usize,
    pub chunk_step_end: usize,
    pub state_out_claims_end: usize,
    pub transcript_out_end: usize,
    pub state_out_handle_end: usize,
}

impl Rv64imChunkStepIvcConstraintCheckpoints {
    pub(super) fn phase_for_row(&self, row: usize) -> (&'static str, usize) {
        if row < self.public_bind_end {
            ("public_bind", row)
        } else if row < self.chunk_step_end {
            ("chunk_step", row - self.public_bind_end)
        } else if row < self.state_out_claims_end {
            ("state_out_claims", row - self.chunk_step_end)
        } else if row < self.transcript_out_end {
            ("transcript_out", row - self.state_out_claims_end)
        } else {
            ("state_out_handle", row - self.transcript_out_end)
        }
    }
}

impl Rv64imChunkStepIvcCircuit {
    pub(super) fn expected_public_values(&self) -> Vec<SpartanF> {
        chunk_step_ivc_spartan_public_values(&self.published_target)
    }
}

fn chunk_step_boundary_plan(circuit: &Rv64imChunkStepIvcCircuit) -> Rv64imChunkBoundaryPlan {
    // The chunk-step IVC relation carries the verified next CE claims into
    // `state_out` even on the terminal chunk. The decider must therefore bind
    // terminal children as the next carry rather than preserving the incoming
    // carry.
    Rv64imChunkBoundaryPlan::from_boundary_mode(
        Rv64imChunkBoundaryMode::from_terminal_flags(circuit.witness.terminal_step, true),
        circuit.effective_chunk.fresh_claims.len(),
        circuit.effective_chunk.ccs_trace.ccs_outputs.len(),
    )
}

impl SpartanCircuit<Rv64imDeciderEngine> for Rv64imChunkStepIvcCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public_cursor = 0usize;
        synthesize_chunk_step_ivc_relation_body(
            self,
            &mut cs.namespace(|| "chunk_step_ivc"),
            &public_inputs,
            &mut public_cursor,
        )?;
        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

pub(super) fn build_rv64im_chunk_step_ivc_circuit(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcCircuit, Rv64imChunkStepIvcSpartanError> {
    let (published_target, effective_chunk) = prepare_rv64im_chunk_step_ivc_circuit_inputs(statement, witness)?;
    let cover_chunk = Rv64imMainCircuitChunkCover::from_trace(&effective_chunk);
    let (params, _, structure) =
        crate::rv64im::kernel::rv64im_root_main_lane_context_for_claim_count(witness.state_in.carry.main.claims.len())
            .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let dims = build_dims_and_policy(&params, structure)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let mat_digest_vec = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(
        structure,
        Some(optimized_cache.sparse()),
    );
    let mat_digest = mat_digest_vec
        .try_into()
        .map_err(|_| Rv64imChunkStepIvcSpartanError::Verify("matrix digest length mismatch".into()))?;
    Ok(Rv64imChunkStepIvcCircuit {
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        published_target,
        witness: witness.clone(),
        cover_chunk,
        effective_chunk,
    })
}

fn rv64im_chunk_step_ivc_shape(circuit: &Rv64imChunkStepIvcCircuit) -> Rv64imChunkStepIvcShape {
    Rv64imChunkStepIvcShape {
        terminal_step: circuit.witness.terminal_step,
        state_in_claim_count: circuit.witness.state_in.carry.main.claims.len() as u64,
        state_out_claim_count: circuit.witness.state_out.carry.main.claims.len() as u64,
        fresh_claim_count: circuit.effective_chunk.fresh_claims.len() as u64,
        fresh_witness_count: circuit.effective_chunk.fresh_witnesses.len() as u64,
        ccs_output_count: circuit.effective_chunk.ccs_trace.ccs_outputs.len() as u64,
        child_count: circuit.effective_chunk.ccs_trace.children.len() as u64,
        transcript_in_absorbed: circuit.witness.state_in.transcript.absorbed as u64,
        transcript_out_absorbed: circuit.witness.state_out.transcript.absorbed as u64,
        fe_round_lengths: circuit
            .effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: circuit
            .effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
    }
}

pub(super) fn rv64im_chunk_step_ivc_cache_key(
    circuit: &Rv64imChunkStepIvcCircuit,
) -> Result<[u8; 32], Rv64imChunkStepIvcSpartanError> {
    let mut tr = neo_transcript::Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key");
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/shape",
        &rv64im_chunk_step_ivc_shape(circuit).expected_digest(),
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/published_target_digest",
        &circuit.published_target.expected_digest(),
    );
    let state_in_bytes = bincode::serialize(&circuit.witness.state_in)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/state_in",
        &state_in_bytes,
    );
    let state_out_bytes = bincode::serialize(&circuit.witness.state_out)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/state_out",
        &state_out_bytes,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/public_chunk_digest",
        &circuit.witness.handoff.public_chunk_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/bridge_handoff_digest",
        &circuit.witness.handoff.bridge_handoff.digest,
    );
    for digest in &circuit.witness.handoff.prepared_step_digests {
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/prepared_step_digest",
            digest,
        );
    }
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/replay_witness_digest",
        &rv64im_chunk_replay_witness_digest(&circuit.witness.replay_witness),
    );
    Ok(tr.digest32())
}

pub(super) fn chunk_step_ivc_spartan_public_values(target: &Rv64imChunkStepIvcPublishedTarget) -> Vec<SpartanF> {
    target
        .public_values()
        .into_iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect()
}

pub(super) fn debug_chunk_step_ivc_constraint_checkpoints(
    circuit: &Rv64imChunkStepIvcCircuit,
) -> Result<Rv64imChunkStepIvcConstraintCheckpoints, SynthesisError> {
    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let public_inputs = circuit
        .expected_public_values()
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut public_cursor = 0usize;
    let mut checkpoints = Rv64imChunkStepIvcConstraintCheckpoints {
        public_bind_end: 0,
        chunk_step_end: 0,
        state_out_claims_end: 0,
        transcript_out_end: 0,
        state_out_handle_end: 0,
    };
    let program_digest_input = next_public_digest(&public_inputs, &mut public_cursor, "program_digest")?;
    let chunk_index_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let step_lo_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let step_hi_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let halted_out_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let state_in_input = next_public_digest(&public_inputs, &mut public_cursor, "state_in")?;
    let state_out_input = next_public_digest(&public_inputs, &mut public_cursor, "state_out")?;
    let summary_start_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let summary_step_count_input = next_public_u64(&public_inputs, &mut public_cursor)?;
    let public_chunk_digest_input = next_public_digest(&public_inputs, &mut public_cursor, "public_chunk_digest")?;

    let program_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "program_digest_const"),
        circuit.published_target.program_digest,
        "program_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "program_digest_eq"),
        &program_digest_input,
        &program_digest_const,
        "program_digest_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "chunk_index_eq"),
        &chunk_index_input,
        circuit.published_target.chunk_index,
        "chunk_index_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_lo_eq"),
        &step_lo_input,
        circuit.published_target.step_lo,
        "step_lo_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_hi_eq"),
        &step_hi_input,
        circuit.published_target.step_hi,
        "step_hi_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "halted_out_eq"),
        &halted_out_input,
        u64::from(circuit.published_target.halted_out),
        "halted_out_eq",
    )?;
    let state_in_const = digest_const_inputs(
        &mut cs.namespace(|| "state_in_const"),
        circuit.published_target.state_in,
        "state_in_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_in_eq"),
        &state_in_input,
        &state_in_const,
        "state_in_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_start_eq"),
        &summary_start_input,
        circuit.published_target.summary_start,
        "summary_start_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_step_count_eq"),
        &summary_step_count_input,
        circuit.published_target.summary_step_count,
        "summary_step_count_eq",
    )?;
    let public_chunk_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "public_chunk_digest_const"),
        circuit.published_target.public_chunk_digest,
        "public_chunk_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "public_chunk_digest_eq"),
        &public_chunk_digest_input,
        &public_chunk_digest_const,
        "public_chunk_digest_eq",
    )?;
    checkpoints.public_bind_end = cs.num_constraints();

    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields.clone(),
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )?;

    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let next_claims = synthesize_rv64im_main_relation_chunk(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs.namespace(|| "chunk_step"),
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &public_inputs,
        &mut public_cursor,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        chunk_step_boundary_plan(circuit),
        true,
        false,
    )?;
    checkpoints.chunk_step_end = cs.num_constraints();

    if next_claims.effective_count() != circuit.witness.state_out.carry.main.claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (claim_index, (actual, expected)) in next_claims
        .effective_claims()
        .iter()
        .zip(circuit.witness.state_out.carry.main.claims.iter())
        .enumerate()
    {
        enforce_claim_projection_eq_native(
            &mut cs.namespace(|| format!("state_out_claim_{claim_index}")),
            actual,
            expected,
            &format!("state_out_claim_{claim_index}"),
        )?;
    }
    checkpoints.state_out_claims_end = cs.num_constraints();

    transcript.append_const_fields_raw(
        cs.namespace(|| "chunk_done"),
        &[
            SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
            SpartanF::from_canonical_u64(1),
        ],
    )?;
    let transcript_out = transcript.state_fields(cs.namespace(|| "carried_transcript_out"))?;
    for (lane_index, (actual, expected)) in transcript_out
        .iter()
        .zip(circuit.witness.state_out.transcript.state.iter())
        .enumerate()
    {
        let expected = SpartanF::from_canonical_u64(expected.as_canonical_u64());
        cs.enforce(
            || format!("transcript_out_lane_{lane_index}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + ShapeCS::<Rv64imDeciderEngine>::one(),
            |lc| lc + (expected, ShapeCS::<Rv64imDeciderEngine>::one()),
        );
    }
    checkpoints.transcript_out_end = cs.num_constraints();

    let state_out_handle = chunk_step_handle_circuit(
        &mut cs.namespace(|| "state_out_handle"),
        &state_in_input,
        circuit.witness.handoff.bridge_handoff.chunk_index as u64,
        circuit.effective_chunk.handoff.public_chunk.start_index as u64,
        circuit.effective_chunk.handoff.public_chunk.steps.len() as u64,
        circuit.effective_chunk.handoff.chunk_relation_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_out_eq"),
        &state_out_handle,
        &state_out_input,
        "state_out_eq",
    )?;
    checkpoints.state_out_handle_end = cs.num_constraints();
    Ok(checkpoints)
}

pub(super) fn debug_locate_chunk_step_main_relation_stage(
    circuit: &Rv64imChunkStepIvcCircuit,
    phase_row: usize,
) -> Result<String, String> {
    let boundary_plan = chunk_step_boundary_plan(circuit);
    let build_replay_context = || -> Result<
        (
            ShapeCS<Rv64imDeciderEngine>,
            Vec<AllocatedNum<SpartanF>>,
            usize,
            Poseidon2TranscriptCircuit,
            Rv64imClaimBundle,
            Rv64imMainCircuitChunkReplaySurface,
        ),
        String,
    > {
        let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
        let public_inputs = circuit
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| err.to_string())?;
        let mut public_cursor = 0usize;
        let _ =
            next_public_digest(&public_inputs, &mut public_cursor, "program_digest").map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_digest(&public_inputs, &mut public_cursor, "state_in").map_err(|err| err.to_string())?;
        let _ = next_public_digest(&public_inputs, &mut public_cursor, "state_out").map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_u64(&public_inputs, &mut public_cursor).map_err(|err| err.to_string())?;
        let _ = next_public_digest(&public_inputs, &mut public_cursor, "public_chunk_digest")
            .map_err(|err| err.to_string())?;

        let transcript_in_fields =
            alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)
                .map_err(|err| err.to_string())?;
        let transcript_in_values = circuit
            .witness
            .state_in
            .transcript
            .state
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
        let transcript = Poseidon2TranscriptCircuit::from_state(
            transcript_in_fields,
            transcript_in_values,
            circuit.witness.state_in.transcript.absorbed,
        )
        .map_err(|err| err.to_string())?;
        let carried_claims = alloc_state_in_claims(
            &mut cs.namespace(|| "state_in_claims"),
            &circuit.witness.state_in.carry.main.claims,
        )
        .map_err(|err| err.to_string())?;
        let replay_chunk = circuit
            .effective_chunk
            .replay_surface()
            .map_err(|_| "replay_surface".to_string())?;
        Ok((
            cs,
            public_inputs,
            public_cursor,
            transcript,
            Rv64imClaimBundle::from_effective_claims(carried_claims),
            replay_chunk,
        ))
    };

    let (mut cs, public_inputs, mut public_cursor, mut transcript, carried_claims, replay_chunk) =
        build_replay_context()?;
    let checkpoints = debug_measure_rv64im_main_relation_chunk_stage_ranges(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &public_inputs,
        &mut public_cursor,
        &mut transcript,
        carried_claims,
        boundary_plan,
        true,
        false,
    )
    .map_err(|err| err.to_string())?;
    if phase_row >= checkpoints.total_constraints() {
        return Err(format!(
            "phase_row_out_of_range({phase_row}>={})",
            checkpoints.total_constraints()
        ));
    }
    let (stage_name, stage_row) = checkpoints.phase_for_row(phase_row);
    if stage_name == "state_out_claims" {
        let state_out_diag = debug_check_chunk_step_state_out_claims_surface(circuit, boundary_plan)
            .unwrap_or_else(|err| format!("state_out_diag_err={err}"));
        return Ok(format!("{stage_name} (stage_row={stage_row}, {state_out_diag})"));
    }
    if stage_name != "pi_rlc" {
        return Ok(format!("{stage_name} (stage_row={stage_row})"));
    }

    let (mut rlc_cs, _, _, mut rlc_transcript, rlc_carried_claims, rlc_replay_chunk) = build_replay_context()?;
    let rlc_checkpoints = debug_measure_rv64im_pi_rlc_stage_ranges(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut rlc_cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &rlc_replay_chunk,
        &mut rlc_transcript,
        rlc_carried_claims,
        boundary_plan,
        0,
    )
    .map_err(|err| err.to_string())?;
    if stage_row >= rlc_checkpoints.total_constraints() {
        return Ok(format!(
            "{stage_name} (stage_row={stage_row}, substage_err=phase_row_out_of_range({stage_row}>={}))",
            rlc_checkpoints.total_constraints()
        ));
    }
    let (substage_name, substage_row) = rlc_checkpoints
        .phase_for_row(stage_row)
        .ok_or_else(|| "missing_pi_rlc_substage".to_string())?;
    if substage_name != "rlc_public" {
        return Ok(format!(
            "{stage_name} (stage_row={stage_row}, substage={substage_name}, substage_row={substage_row})"
        ));
    }
    let rlc_public_detail =
        match debug_locate_chunk_step_main_relation_rlc_public_detail(circuit, substage_row, boundary_plan) {
            Ok(detail) => format!(", detail={detail}"),
            Err(err) => format!(", detail_err={err}"),
        };
    Ok(format!(
        "{stage_name} (stage_row={stage_row}, substage={substage_name}, substage_row={substage_row}{rlc_public_detail})"
    ))
}

pub(super) fn debug_locate_chunk_step_state_out_claims_stage(
    circuit: &Rv64imChunkStepIvcCircuit,
) -> Result<String, String> {
    let boundary_plan = chunk_step_boundary_plan(circuit);
    debug_check_chunk_step_state_out_claims_surface(circuit, boundary_plan)
}

fn describe_chunk_step_state_out_claim_mismatch(
    actual: &CeClaimVar,
    expected: &neo_ccs::CeClaim<neo_ajtai::Commitment, F, neo_math::K>,
) -> Option<String> {
    if actual.c_data_values.len() != expected.c.data.len() {
        return Some("c_data_len_mismatch".into());
    }
    if let Some(idx) = actual
        .c_data_values
        .iter()
        .zip(expected.c.data.iter())
        .position(|(lhs, rhs)| lhs != rhs)
    {
        return Some(format!("c_data[{idx}] mismatch"));
    }
    if actual.x_rows != expected.X.rows() || actual.x_cols != expected.X.cols() {
        return Some("x_shape_mismatch".into());
    }
    if let Some(idx) = actual
        .x_values
        .iter()
        .zip(expected.X.as_slice().iter())
        .position(|(lhs, rhs)| lhs != rhs)
    {
        let row = idx / actual.x_cols;
        let col = idx % actual.x_cols;
        return Some(format!("x[{row},{col}] mismatch"));
    }
    if actual.r_values != expected.r {
        let idx = actual
            .r_values
            .iter()
            .zip(expected.r.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("r[{idx}] mismatch"));
    }
    if actual.s_col_values != expected.s_col {
        let idx = actual
            .s_col_values
            .iter()
            .zip(expected.s_col.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("s_col[{idx}] mismatch"));
    }
    if actual.y_ring_values.len() != expected.y_ring.len() {
        return Some("y_ring_row_count_mismatch".into());
    }
    for (row_idx, (actual_row, expected_row)) in actual
        .y_ring_values
        .iter()
        .zip(expected.y_ring.iter())
        .enumerate()
    {
        if actual_row.len() != expected_row.len() {
            return Some(format!("y_ring[{row_idx}]_len_mismatch"));
        }
        if let Some(col_idx) = actual_row
            .iter()
            .zip(expected_row.iter())
            .position(|(lhs, rhs)| lhs != rhs)
        {
            return Some(format!("y_ring[{row_idx}][{col_idx}] mismatch"));
        }
    }
    if actual.ct_values != expected.ct {
        let idx = actual
            .ct_values
            .iter()
            .zip(expected.ct.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("ct[{idx}] mismatch"));
    }
    if actual.aux_openings_values != expected.aux_openings {
        let idx = actual
            .aux_openings_values
            .iter()
            .zip(expected.aux_openings.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("aux_openings[{idx}] mismatch"));
    }
    if actual.y_zcol_values != expected.y_zcol {
        let idx = actual
            .y_zcol_values
            .iter()
            .zip(expected.y_zcol.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("y_zcol[{idx}] mismatch"));
    }
    if actual.c_step_coords_values != expected.c_step_coords {
        let idx = actual
            .c_step_coords_values
            .iter()
            .zip(expected.c_step_coords.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("c_step_coords[{idx}] mismatch"));
    }
    if actual.m_in != expected.m_in {
        return Some("m_in mismatch".into());
    }
    if actual.u_offset != expected.u_offset {
        return Some("u_offset mismatch".into());
    }
    if actual.u_len != expected.u_len {
        return Some("u_len mismatch".into());
    }
    let expected_fold_digest = packed_bytes_field_values(&expected.fold_digest);
    if actual.fold_digest_encoding_values != expected_fold_digest {
        let idx = actual
            .fold_digest_encoding_values
            .iter()
            .zip(expected_fold_digest.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("fold_digest[{idx}] mismatch"));
    }
    None
}

pub(super) fn debug_check_chunk_step_state_out_claims_surface(
    circuit: &Rv64imChunkStepIvcCircuit,
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<String, String> {
    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let public_inputs = circuit
        .expected_public_values()
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    let mut public_cursor = 0usize;
    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)
        .map_err(|err| err.to_string())?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields,
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )
    .map_err(|err| err.to_string())?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| "replay_surface".to_string())?;
    let next_claims = synthesize_rv64im_main_relation_chunk(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs.namespace(|| "chunk_step"),
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &public_inputs,
        &mut public_cursor,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        boundary_plan,
        true,
        false,
    )
    .map_err(|err| err.to_string())?;
    if next_claims.effective_count() != circuit.witness.state_out.carry.main.claims.len() {
        return Ok(format!(
            "state_out_count_mismatch[actual={},expected={}]",
            next_claims.effective_count(),
            circuit.witness.state_out.carry.main.claims.len()
        ));
    }
    for (claim_index, (actual, expected)) in next_claims
        .effective_claims()
        .iter()
        .zip(circuit.witness.state_out.carry.main.claims.iter())
        .enumerate()
    {
        if let Some(mismatch) = describe_chunk_step_state_out_claim_mismatch(actual, expected) {
            return Ok(format!("claim[{claim_index}] {mismatch}"));
        }
    }
    Ok("state_out_claims_match".into())
}

fn debug_locate_chunk_step_main_relation_rlc_public_detail(
    circuit: &Rv64imChunkStepIvcCircuit,
    rlc_public_row: usize,
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<String, String> {
    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)
        .map_err(|err| err.to_string())?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields,
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )
    .map_err(|err| err.to_string())?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| "replay_surface".to_string())?;
    let checkpoints = debug_measure_rv64im_rlc_public_stage_ranges(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        None,
        boundary_plan,
        0,
        None,
    )
    .map_err(|err| err.to_string())?;
    let (detail, detail_row) = checkpoints
        .phase_for_row(rlc_public_row)
        .ok_or_else(|| format!("rlc_public_row_out_of_range({rlc_public_row})"))?;
    if detail == "x" {
        let x_diag = debug_check_chunk_step_main_relation_rlc_public_x_native_values(circuit, boundary_plan)
            .unwrap_or_else(|err| format!("x_diag_err={err}"));
        return Ok(format!("{detail} (detail_row={detail_row}, {x_diag})"));
    }
    Ok(format!("{detail} (detail_row={detail_row})"))
}

fn debug_check_chunk_step_main_relation_rlc_public_x_native_values(
    circuit: &Rv64imChunkStepIvcCircuit,
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<String, String> {
    let expected_transcript = debug_replay_rv64im_main_relation_pi_ccs_transcript_state(
        &circuit.witness.state_in.transcript,
        &circuit.effective_chunk.handoff,
        &circuit.effective_chunk.fresh_claims,
        &circuit.effective_chunk.fresh_witnesses,
        &circuit.witness.state_in.carry.main.claims,
        &circuit.witness.state_in.carry.main.witnesses,
    )
    .map_err(|err| err.to_string())?;

    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)
        .map_err(|err| err.to_string())?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields,
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )
    .map_err(|err| err.to_string())?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| "replay_surface".to_string())?;
    let x_diag = debug_check_rv64im_rlc_public_x_native_values(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        boundary_plan,
        &expected_transcript,
    )
    .map_err(|err| err.to_string())?;
    let mut transcript_cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let transcript_probe_fields =
        alloc_private_transcript_state(&mut transcript_cs.namespace(|| "transcript_in"), &circuit.witness)
            .map_err(|err| err.to_string())?;
    let transcript_probe_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript_probe = Poseidon2TranscriptCircuit::from_state(
        transcript_probe_fields,
        transcript_probe_values,
        circuit.witness.state_in.transcript.absorbed,
    )
    .map_err(|err| err.to_string())?;
    let transcript_probe_claims = alloc_state_in_claims(
        &mut transcript_cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let transcript_diag = debug_compare_rv64im_pi_ccs_transcript_state(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut transcript_cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &mut transcript_probe,
        Rv64imClaimBundle::from_effective_claims(transcript_probe_claims),
        boundary_plan,
        &expected_transcript,
    )
    .map_err(|err| err.to_string())?;
    let terminal_state_diag = debug_describe_rv64im_main_relation_pi_ccs_terminal_state_mismatch(
        &circuit.witness.state_in.transcript,
        &circuit.effective_chunk.handoff,
        &circuit.effective_chunk.fresh_claims,
        &circuit.effective_chunk.fresh_witnesses,
        &circuit.witness.state_in.carry.main.claims,
        &circuit.witness.state_in.carry.main.witnesses,
        &circuit.effective_chunk.ccs_trace.terminal_state,
    )
    .map_err(|err| err.to_string())?;
    let mut late_stage_cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let late_stage_claims = alloc_state_in_claims(
        &mut late_stage_cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let late_stage_diag = debug_locate_rv64im_pi_ccs_late_transcript_stage(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &mut late_stage_cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &circuit.witness.state_in.transcript,
        Rv64imClaimBundle::from_effective_claims(late_stage_claims),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let mut rho_stage_cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let rho_stage_claims = alloc_state_in_claims(
        &mut rho_stage_cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let rho_stage_diag = debug_compare_rv64im_pi_rlc_rho_mats(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &mut rho_stage_cs,
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        &circuit.witness.state_in.transcript,
        Rv64imClaimBundle::from_effective_claims(rho_stage_claims),
        &circuit.witness.state_in.carry.main.claims,
    )
    .map_err(|err| err.to_string())?;
    let parent_diag = debug_describe_rv64im_main_relation_pi_rlc_parent_mismatch(
        &circuit.witness.state_in.transcript,
        &circuit.effective_chunk.handoff,
        &circuit.effective_chunk.fresh_claims,
        &circuit.effective_chunk.fresh_witnesses,
        &circuit.witness.state_in.carry.main.claims,
        &circuit.witness.state_in.carry.main.witnesses,
        &circuit.effective_chunk.ccs_trace.ccs_outputs,
        &circuit.effective_chunk.ccs_trace.parent,
    )
    .map_err(|err| err.to_string())?;
    let x_flat_diag = debug_describe_rv64im_main_relation_pi_rlc_x_flat_mismatch(
        &circuit.witness.state_in.transcript,
        &circuit.effective_chunk.handoff,
        &circuit.effective_chunk.fresh_claims,
        &circuit.effective_chunk.fresh_witnesses,
        &circuit.witness.state_in.carry.main.claims,
        &circuit.witness.state_in.carry.main.witnesses,
        &circuit.effective_chunk.ccs_trace.ccs_outputs,
        &circuit.effective_chunk.ccs_trace.parent,
    )
    .map_err(|err| err.to_string())?;
    Ok(format!(
        "{x_diag}; {transcript_diag}; pi_ccs_terminal_state={terminal_state_diag}; pi_ccs_late={late_stage_diag}; pi_rlc_rhos={rho_stage_diag}; pi_rlc_parent={parent_diag}; pi_rlc_x_flat={x_flat_diag}"
    ))
}

fn synthesize_chunk_step_ivc_relation_body<CS: ConstraintSystem<SpartanF>>(
    circuit: &Rv64imChunkStepIvcCircuit,
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
) -> Result<(), SynthesisError> {
    let program_digest_input = next_public_digest(public_inputs, public_cursor, "program_digest")?;
    let chunk_index_input = next_public_u64(public_inputs, public_cursor)?;
    let step_lo_input = next_public_u64(public_inputs, public_cursor)?;
    let step_hi_input = next_public_u64(public_inputs, public_cursor)?;
    let halted_out_input = next_public_u64(public_inputs, public_cursor)?;
    let state_in_input = next_public_digest(public_inputs, public_cursor, "state_in")?;
    let state_out_input = next_public_digest(public_inputs, public_cursor, "state_out")?;
    let summary_start_input = next_public_u64(public_inputs, public_cursor)?;
    let summary_step_count_input = next_public_u64(public_inputs, public_cursor)?;
    let public_chunk_digest_input = next_public_digest(public_inputs, public_cursor, "public_chunk_digest")?;

    let program_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "program_digest_const"),
        circuit.published_target.program_digest,
        "program_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "program_digest_eq"),
        &program_digest_input,
        &program_digest_const,
        "program_digest_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "chunk_index_eq"),
        &chunk_index_input,
        circuit.published_target.chunk_index,
        "chunk_index_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_lo_eq"),
        &step_lo_input,
        circuit.published_target.step_lo,
        "step_lo_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_hi_eq"),
        &step_hi_input,
        circuit.published_target.step_hi,
        "step_hi_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "halted_out_eq"),
        &halted_out_input,
        u64::from(circuit.published_target.halted_out),
        "halted_out_eq",
    )?;
    let state_in_const = digest_const_inputs(
        &mut cs.namespace(|| "state_in_const"),
        circuit.published_target.state_in,
        "state_in_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_in_eq"),
        &state_in_input,
        &state_in_const,
        "state_in_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_start_eq"),
        &summary_start_input,
        circuit.published_target.summary_start,
        "summary_start_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_step_count_eq"),
        &summary_step_count_input,
        circuit.published_target.summary_step_count,
        "summary_step_count_eq",
    )?;
    let public_chunk_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "public_chunk_digest_const"),
        circuit.published_target.public_chunk_digest,
        "public_chunk_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "public_chunk_digest_eq"),
        &public_chunk_digest_input,
        &public_chunk_digest_const,
        "public_chunk_digest_eq",
    )?;

    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields.clone(),
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )?;

    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let next_claims = synthesize_rv64im_main_relation_chunk(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs.namespace(|| "chunk_step"),
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        public_inputs,
        public_cursor,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        chunk_step_boundary_plan(circuit),
        true,
        false,
    )?;

    if next_claims.effective_count() != circuit.witness.state_out.carry.main.claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (claim_index, (actual, expected)) in next_claims
        .effective_claims()
        .iter()
        .zip(circuit.witness.state_out.carry.main.claims.iter())
        .enumerate()
    {
        enforce_claim_projection_eq_native(
            &mut cs.namespace(|| format!("state_out_claim_{claim_index}")),
            actual,
            expected,
            &format!("state_out_claim_{claim_index}"),
        )?;
    }

    transcript.append_const_fields_raw(
        cs.namespace(|| "chunk_done"),
        &[
            SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
            SpartanF::from_canonical_u64(1),
        ],
    )?;
    let transcript_out = transcript.state_fields(cs.namespace(|| "carried_transcript_out"))?;
    for (lane_index, (actual, expected)) in transcript_out
        .iter()
        .zip(circuit.witness.state_out.transcript.state.iter())
        .enumerate()
    {
        let expected = SpartanF::from_canonical_u64(expected.as_canonical_u64());
        cs.enforce(
            || format!("transcript_out_lane_{lane_index}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (expected, CS::one()),
        );
    }
    if transcript.absorbed() != circuit.witness.state_out.transcript.absorbed {
        return Err(SynthesisError::Unsatisfiable);
    }

    let state_out_handle = chunk_step_handle_circuit(
        &mut cs.namespace(|| "state_out_handle"),
        &state_in_input,
        circuit.witness.handoff.bridge_handoff.chunk_index as u64,
        circuit.effective_chunk.handoff.public_chunk.start_index as u64,
        circuit.effective_chunk.handoff.public_chunk.steps.len() as u64,
        circuit.effective_chunk.handoff.chunk_relation_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_out_eq"),
        &state_out_handle,
        &state_out_input,
        "state_out_eq",
    )?;

    let _ = next_claims.into_effective_claims();
    let _ = transcript_out;
    let _ = state_out_handle;
    let _ = transcript_in_fields;
    Ok(())
}

fn alloc_private_transcript_state<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<[AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH], SynthesisError> {
    let mut out = Vec::with_capacity(neo_params::poseidon2_goldilocks::WIDTH);
    for (lane_index, lane) in witness.state_in.transcript.state.iter().enumerate() {
        out.push(AllocatedNum::alloc(
            cs.namespace(|| format!("transcript_lane_{lane_index}")),
            || Ok(SpartanF::from_canonical_u64(lane.as_canonical_u64())),
        )?);
    }
    out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}

fn alloc_state_in_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, neo_math::K>],
) -> Result<Vec<CeClaimVar>, SynthesisError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(claims.len());
    let first_var = alloc_ce_claim(&mut cs.namespace(|| "claim_0"), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    let shared_s_col = first_var.s_col.clone();
    let shared_s_col_values = first_var.s_col_values.clone();
    out.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        out.push(alloc_ce_claim_with_shared_point(
            &mut cs.namespace(|| format!("claim_{}", idx + 1)),
            claim,
            &shared_r,
            &shared_r_values,
            &shared_s_col,
            &shared_s_col_values,
            &format!("claim_{}", idx + 1),
        )?);
    }
    Ok(out)
}

fn enforce_u64_input_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &AllocatedNum<SpartanF>,
    expected: u64,
    label: &str,
) -> Result<(), SynthesisError> {
    let expected = SpartanF::from_canonical_u64(expected);
    cs.enforce(
        || label,
        |lc| lc + actual.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
    Ok(())
}

fn next_public_u64(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    if *cursor >= public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = public_inputs[*cursor].clone();
    *cursor += 1;
    Ok(out)
}

fn chunk_step_handle_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    previous_handle_digest: &[AllocatedNum<SpartanF>; 4],
    chunk_index: u64,
    chunk_start_index: u64,
    public_step_count: u64,
    chunk_relation_digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(previous_handle_digest.iter().cloned());
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| "chunk_step_meta"),
        &[
            SpartanF::from_canonical_u64(chunk_index),
            SpartanF::from_canonical_u64(chunk_start_index),
            SpartanF::from_canonical_u64(public_step_count),
        ],
        "chunk_step_meta",
    )?);
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| "chunk_step_digest"),
        &packed_bytes_field_values(&chunk_relation_digest),
        "chunk_step_digest",
    )?);
    hash_packed_goldilocks_fields(cs.namespace(|| "chunk_step_handle_hash"), &preimage)
}
