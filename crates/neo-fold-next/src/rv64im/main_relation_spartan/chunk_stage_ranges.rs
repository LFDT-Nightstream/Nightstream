//! Owns ShapeCS stage checkpoints for chunk replay diagnostics.
//!
//! This module exists only to localize failing rows onto stable semantic stage
//! boundaries without relying on `TestConstraintSystem` namespace uniqueness.

use bellpepper_core::{num::AllocatedNum, SynthesisError};
use neo_math::D;
use neo_reductions::engines::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs;
use neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3;
use neo_transcript::{Poseidon2Transcript, Transcript};

use super::*;
use crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot;
use crate::rv64im::ivc_snark::{Rv64imDeciderEngine, ShapeCS, SpartanF};
use crate::rv64im::main_relation_spartan::nifs_v_stages::synthesize_pi_rlc_stage;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Rv64imMainRelationChunkStageCheckpoints {
    pub chunk_meta_end: usize,
    pub pi_ccs_end: usize,
    pub pi_rlc_end: usize,
    pub pi_dec_end: usize,
    pub outer_relation_public_io_end: usize,
    pub chunk_done_end: usize,
}

impl Rv64imMainRelationChunkStageCheckpoints {
    pub(crate) fn phase_for_row(&self, row: usize) -> (&'static str, usize) {
        if row < self.chunk_meta_end {
            ("chunk_meta", row)
        } else if row < self.pi_ccs_end {
            ("pi_ccs", row - self.chunk_meta_end)
        } else if row < self.pi_rlc_end {
            ("pi_rlc", row - self.pi_ccs_end)
        } else if row < self.pi_dec_end {
            ("pi_dec", row - self.pi_rlc_end)
        } else if row < self.outer_relation_public_io_end {
            ("outer_relation_public_io", row - self.pi_dec_end)
        } else {
            ("chunk_done", row - self.outer_relation_public_io_end)
        }
    }

    pub(crate) fn total_constraints(&self) -> usize {
        self.chunk_done_end
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imPiRlcStageCheckpoints {
    stage_ends: Vec<(&'static str, usize)>,
}

impl Rv64imPiRlcStageCheckpoints {
    fn push(&mut self, name: &'static str, end: usize) {
        self.stage_ends.push((name, end));
    }

    pub(crate) fn phase_for_row(&self, row: usize) -> Option<(&'static str, usize)> {
        let mut start = 0usize;
        for (name, end) in &self.stage_ends {
            if row < *end {
                return Some((*name, row - start));
            }
            start = *end;
        }
        None
    }

    pub(crate) fn total_constraints(&self) -> usize {
        self.stage_ends.last().map(|(_, end)| *end).unwrap_or(0)
    }
}

pub(crate) fn debug_measure_rv64im_main_relation_chunk_stage_ranges(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
    enforce_chunk_relation_public_io: bool,
    append_chunk_done: bool,
) -> Result<Rv64imMainRelationChunkStageCheckpoints, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut checkpoints = Rv64imMainRelationChunkStageCheckpoints {
        chunk_meta_end: 0,
        pi_ccs_end: 0,
        pi_rlc_end: 0,
        pi_dec_end: 0,
        outer_relation_public_io_end: 0,
        chunk_done_end: 0,
    };

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    checkpoints.chunk_meta_end = cs.num_constraints();

    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    let pi_ccs = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, None)?;
    checkpoints.pi_ccs_end = cs.num_constraints();

    let pi_rlc = synthesize_pi_rlc_stage(&ctx, cs, transcript, &pi_ccs)?;
    checkpoints.pi_rlc_end = cs.num_constraints();

    let _next_claims = synthesize_pi_dec_stage(&ctx, cs, carried_claims, &pi_ccs, pi_rlc)?;
    checkpoints.pi_dec_end = cs.num_constraints();

    if enforce_chunk_relation_public_io {
        enforce_outer_chunk_relation_public_io(&ctx, cs, &pi_ccs.fold_digest, public_inputs, public_cursor)?;
    }
    checkpoints.outer_relation_public_io_end = cs.num_constraints();

    if append_chunk_done {
        transcript.append_const_fields_raw(
            cs.namespace(|| format!("chunk_done_{chunk_index}")),
            &[
                SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )?;
    }
    checkpoints.chunk_done_end = cs.num_constraints();
    Ok(checkpoints)
}

pub(crate) fn debug_measure_rv64im_pi_rlc_stage_ranges(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<Rv64imPiRlcStageCheckpoints, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    let pi_ccs = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, None)?;
    let pi_rlc_start = cs.num_constraints();
    let mut checkpoints = Rv64imPiRlcStageCheckpoints { stage_ends: Vec::new() };

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            Rv64imChunkChildClaimSource::TerminalFinalClaims,
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))?;
        alloc_ce_claim(
            &mut cs.namespace(|| format!("chunk_{}_terminal_parent_claim", ctx.chunk_index)),
            &claim,
            &format!("chunk_{}_terminal_parent_claim", ctx.chunk_index),
        )?
    } else {
        let claim = cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )?;
        alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| format!("chunk_{}_parent_claim", ctx.chunk_index)),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            &format!("chunk_{}_parent_claim", ctx.chunk_index),
        )?
    };
    checkpoints.push("parent_claim", cs.num_constraints() - pi_rlc_start);

    let child_claim_source = match ctx.boundary_plan.child_claim_source {
        Rv64imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        Rv64imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let padded_rho_count = pi_ccs
        .padded_ccs_outputs
        .len()
        .saturating_sub(pi_ccs.effective_output_count);
    let mut rho_vars = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{}_rlc_rhos", ctx.chunk_index)),
        transcript,
        pi_ccs.effective_output_count,
        &format!("chunk_{}_rlc_rhos", ctx.chunk_index),
    )?;
    checkpoints.push("sample_rhos", cs.num_constraints() - pi_rlc_start);

    if padded_rho_count > 0 {
        rho_vars.extend(alloc_zero_rot_rhos(
            &mut cs.namespace(|| format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index)),
            padded_rho_count,
            &format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index),
        )?);
    }
    checkpoints.push("pad_rhos", cs.num_constraints() - pi_rlc_start);

    match ctx.boundary_plan.rlc_mode {
        Rv64imChunkRlcMode::TerminalLastChunkShortcut => {
            enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                &mut cs.namespace(|| format!("chunk_{}_rlc_public", ctx.chunk_index)),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                child_claim_source,
                &rho_vars,
                ctx.params.b,
                &format!("chunk_{}_rlc_public", ctx.chunk_index),
            )?;
            checkpoints.push("terminal_last_chunk_shortcut", cs.num_constraints() - pi_rlc_start);
        }
        Rv64imChunkRlcMode::Standard { constant_child_prefix } => {
            let mut rho_mats = materialize_goldilocks_rot_matrices(
                &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats", ctx.chunk_index)),
                &rho_vars[..pi_ccs.effective_output_count],
                &format!("chunk_{}_rlc_rho_mats", ctx.chunk_index),
            )?;
            checkpoints.push("materialize_rho_mats", cs.num_constraints() - pi_rlc_start);
            if padded_rho_count > 0 {
                rho_mats.extend(alloc_zero_rot_rho_matrices(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index)),
                    padded_rho_count,
                    &format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index),
                )?);
            }
            checkpoints.push("pad_rho_mats", cs.num_constraints() - pi_rlc_start);
            enforce_rlc_public_with_rho_vars_constant_prefix(
                &mut cs.namespace(|| format!("chunk_{}_rlc_public", ctx.chunk_index)),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                &rho_mats,
                constant_child_prefix,
                &format!("chunk_{}_rlc_public", ctx.chunk_index),
            )?;
            checkpoints.push("rlc_public", cs.num_constraints() - pi_rlc_start);
        }
    }

    Ok(checkpoints)
}

pub(crate) fn debug_measure_rv64im_rlc_public_stage_ranges(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<crate::rv64im::main_relation_circuit::pi_rlc::RlcPublicStageCheckpoints, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    let pi_ccs = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, None)?;

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            Rv64imChunkChildClaimSource::TerminalFinalClaims,
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))?;
        alloc_ce_claim(
            &mut cs.namespace(|| format!("chunk_{}_terminal_parent_claim", ctx.chunk_index)),
            &claim,
            &format!("chunk_{}_terminal_parent_claim", ctx.chunk_index),
        )?
    } else {
        let claim = cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )?;
        alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| format!("chunk_{}_parent_claim", ctx.chunk_index)),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            &format!("chunk_{}_parent_claim", ctx.chunk_index),
        )?
    };

    let padded_rho_count = pi_ccs
        .padded_ccs_outputs
        .len()
        .saturating_sub(pi_ccs.effective_output_count);
    let mut rho_vars = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{}_rlc_rhos", ctx.chunk_index)),
        transcript,
        pi_ccs.effective_output_count,
        &format!("chunk_{}_rlc_rhos", ctx.chunk_index),
    )?;
    if padded_rho_count > 0 {
        rho_vars.extend(alloc_zero_rot_rhos(
            &mut cs.namespace(|| format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index)),
            padded_rho_count,
            &format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index),
        )?);
    }

    match ctx.boundary_plan.rlc_mode {
        Rv64imChunkRlcMode::TerminalLastChunkShortcut => Err(SynthesisError::Unsatisfiable),
        Rv64imChunkRlcMode::Standard { constant_child_prefix } => {
            let mut rho_mats = materialize_goldilocks_rot_matrices(
                &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats", ctx.chunk_index)),
                &rho_vars[..pi_ccs.effective_output_count],
                &format!("chunk_{}_rlc_rho_mats", ctx.chunk_index),
            )?;
            if padded_rho_count > 0 {
                rho_mats.extend(alloc_zero_rot_rho_matrices(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index)),
                    padded_rho_count,
                    &format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index),
                )?);
            }
            crate::rv64im::main_relation_circuit::pi_rlc::debug_measure_rlc_public_with_rho_vars_constant_prefix_stage_ranges(
                cs,
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                &rho_mats,
                constant_child_prefix,
                &format!("chunk_{}_rlc_public", ctx.chunk_index),
            )
        }
    }
}

pub(crate) fn debug_check_rv64im_rlc_public_x_native_values(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
    expected_pi_ccs_transcript: &Rv64imChunkFoldTranscriptSnapshot,
) -> Result<String, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    let pi_ccs = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, None)?;
    if transcript.absorbed() != expected_pi_ccs_transcript.absorbed {
        return Ok(format!(
            "live_pi_ccs_transcript_absorbed_mismatch[expected={},observed={}]",
            expected_pi_ccs_transcript.absorbed,
            transcript.absorbed()
        ));
    }
    for (idx, (expected_value, observed_value)) in expected_pi_ccs_transcript
        .state
        .iter()
        .zip(transcript.state_values().iter())
        .enumerate()
    {
        let expected_value = SpartanF::from_canonical_u64(expected_value.as_canonical_u64());
        if expected_value != *observed_value {
            return Ok(format!(
                "live_pi_ccs_transcript_state_mismatch[lane={idx},expected={},observed={}]",
                expected_value.to_canonical_u64(),
                observed_value.to_canonical_u64()
            ));
        }
    }

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            Rv64imChunkChildClaimSource::TerminalFinalClaims,
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))?;
        alloc_ce_claim(
            &mut cs.namespace(|| format!("chunk_{}_terminal_parent_claim", ctx.chunk_index)),
            &claim,
            &format!("chunk_{}_terminal_parent_claim", ctx.chunk_index),
        )?
    } else {
        let claim = cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )?;
        alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| format!("chunk_{}_parent_claim", ctx.chunk_index)),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            &format!("chunk_{}_parent_claim", ctx.chunk_index),
        )?
    };

    let padded_rho_count = pi_ccs
        .padded_ccs_outputs
        .len()
        .saturating_sub(pi_ccs.effective_output_count);
    let mut rho_vars = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{}_rlc_rhos", ctx.chunk_index)),
        transcript,
        pi_ccs.effective_output_count,
        &format!("chunk_{}_rlc_rhos", ctx.chunk_index),
    )?;
    if padded_rho_count > 0 {
        rho_vars.extend(alloc_zero_rot_rhos(
            &mut cs.namespace(|| format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index)),
            padded_rho_count,
            &format!("chunk_{}_rlc_rhos_pad", ctx.chunk_index),
        )?);
    }

    let rho_mats = match ctx.boundary_plan.rlc_mode {
        Rv64imChunkRlcMode::TerminalLastChunkShortcut => return Ok("terminal_last_chunk_shortcut".into()),
        Rv64imChunkRlcMode::Standard { .. } => {
            let mut rho_mats = materialize_goldilocks_rot_matrices(
                &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats", ctx.chunk_index)),
                &rho_vars[..pi_ccs.effective_output_count],
                &format!("chunk_{}_rlc_rho_mats", ctx.chunk_index),
            )?;
            if padded_rho_count > 0 {
                rho_mats.extend(alloc_zero_rot_rho_matrices(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index)),
                    padded_rho_count,
                    &format!("chunk_{}_rlc_rho_mats_pad", ctx.chunk_index),
                )?);
            }
            rho_mats
        }
    };
    let mut expected_rlc_transcript = Poseidon2Transcript::from_state_and_absorbed(
        expected_pi_ccs_transcript.state,
        expected_pi_ccs_transcript.absorbed,
    );
    let expected_rhos = neo_reductions::sample_rot_rhos_n_typed(
        &mut expected_rlc_transcript,
        params,
        &neo_reductions::RotRing::goldilocks(),
        pi_ccs.effective_output_count,
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;
    for (rho_idx, (observed_rho, expected_rho)) in rho_mats
        .iter()
        .take(pi_ccs.effective_output_count)
        .zip(expected_rhos.iter())
        .enumerate()
    {
        let expected_mat = expected_rho.as_mat();
        for row in 0..D {
            for col in 0..D {
                let observed = observed_rho.entry_value(row, col)?;
                let expected = expected_mat.row(row)[col];
                if observed != expected {
                    return Ok(format!(
                        "live_rho_mat_mismatch[{rho_idx}][{row},{col}][expected={},observed={}]",
                        expected.as_canonical_u64(),
                        observed.as_canonical_u64()
                    ));
                }
            }
        }
    }
    for (rho_idx, observed_rho) in rho_mats
        .iter()
        .enumerate()
        .skip(pi_ccs.effective_output_count)
    {
        for row in 0..D {
            for col in 0..D {
                let observed = observed_rho.entry_value(row, col)?;
                if observed != F::ZERO {
                    return Ok(format!(
                        "padded_rho_mat_nonzero[{rho_idx}][{row},{col}]={}",
                        observed.as_canonical_u64()
                    ));
                }
            }
        }
    }

    let cols = parent_claim.x_cols;
    if parent_claim.x_values.len() != D * cols {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (output_idx, (stage_output, source_output)) in pi_ccs
        .padded_ccs_outputs
        .iter()
        .take(pi_ccs.effective_output_count)
        .zip(ctx.chunk.pi_ccs.ccs_outputs.iter())
        .enumerate()
    {
        if stage_output.x_rows != source_output.X.rows()
            || stage_output.x_cols != source_output.X.cols()
            || stage_output.x_values.len() != source_output.X.as_slice().len()
        {
            return Ok(format!("ccs_output_x_shape_mismatch[{output_idx}]"));
        }
        for (x_idx, (observed, expected)) in stage_output
            .x_values
            .iter()
            .zip(source_output.X.as_slice().iter())
            .enumerate()
        {
            if observed != expected {
                let row = x_idx / stage_output.x_cols;
                let col = x_idx % stage_output.x_cols;
                return Ok(format!("ccs_output_x_surface_mismatch[{output_idx}][{row},{col}]"));
            }
        }
    }
    if parent_claim.x_rows != ctx.chunk.pi_rlc.parent.X.rows()
        || parent_claim.x_cols != ctx.chunk.pi_rlc.parent.X.cols()
        || parent_claim.x_values.len() != ctx.chunk.pi_rlc.parent.X.as_slice().len()
    {
        return Ok("parent_x_shape_mismatch".into());
    }
    for (x_idx, (observed, expected)) in parent_claim
        .x_values
        .iter()
        .zip(ctx.chunk.pi_rlc.parent.X.as_slice().iter())
        .enumerate()
    {
        if observed != expected {
            let row = x_idx / parent_claim.x_cols;
            let col = x_idx % parent_claim.x_cols;
            return Ok(format!("parent_x_surface_mismatch[{row},{col}]"));
        }
    }
    for child in &pi_ccs.padded_ccs_outputs {
        if child.x_values.len() != D * cols {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for (output_idx, child) in pi_ccs
        .padded_ccs_outputs
        .iter()
        .enumerate()
        .skip(pi_ccs.effective_output_count)
    {
        if child.x_rows != D || child.x_cols != cols {
            return Ok(format!("padded_output_x_shape_mismatch[{output_idx}]"));
        }
        for (x_idx, value) in child.x_values.iter().enumerate() {
            if *value != F::ZERO {
                let row = x_idx / cols;
                let col = x_idx % cols;
                return Ok(format!(
                    "padded_output_x_nonzero[{output_idx}][{row},{col}]={}",
                    value.as_canonical_u64()
                ));
            }
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let mut expected = F::ZERO;
            for (child_idx, child) in pi_ccs.padded_ccs_outputs.iter().enumerate() {
                for k in 0..D {
                    let child_idx_flat = k * cols + col;
                    expected += rho_mats[child_idx].entry_value(row, k)? * child.x_values[child_idx_flat];
                }
            }
            let parent_idx = row * cols + col;
            let observed = parent_claim.x_values[parent_idx];
            if expected != observed {
                return Ok(format!(
                    "x_native_mismatch[row={row},col={col},expected={},observed={},live_pi_ccs=match,live_rhos=match,padded_zero=match]",
                    expected.as_canonical_u64(),
                    observed.as_canonical_u64()
                ));
            }
        }
    }

    Ok("x_values_match[live_pi_ccs=match,live_rhos=match,padded_zero=match]".into())
}

pub(crate) fn debug_compare_rv64im_pi_ccs_transcript_state(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
    expected: &crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot,
) -> Result<String, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    let _ = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, None)?;
    if transcript.absorbed() != expected.absorbed {
        return Ok(format!(
            "pi_ccs_transcript_absorbed_mismatch[expected={},observed={}]",
            expected.absorbed,
            transcript.absorbed()
        ));
    }
    let observed_state = transcript
        .state_values()
        .map(|value| F::from_u64(value.to_canonical_u64()));
    for (idx, (expected_value, observed_value)) in expected.state.iter().zip(observed_state.iter()).enumerate() {
        if expected_value != observed_value {
            return Ok(format!(
                "pi_ccs_transcript_state_mismatch[lane={idx},expected={},observed={}]",
                expected_value.as_canonical_u64(),
                observed_value.as_canonical_u64()
            ));
        }
    }
    Ok("pi_ccs_transcript_state_match".into())
}

pub(crate) fn debug_locate_rv64im_pi_ccs_late_transcript_stage(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    carried_claims: Rv64imClaimBundle,
    native_carried_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
) -> Result<String, SynthesisError> {
    fn compare_stage(
        label: &str,
        circuit_tr: &Poseidon2TranscriptCircuit,
        native_tr: &Poseidon2Transcript,
    ) -> Option<String> {
        if circuit_tr.absorbed() != native_tr.absorbed() {
            return Some(format!(
                "{label}: absorbed {} != {}",
                circuit_tr.absorbed(),
                native_tr.absorbed()
            ));
        }
        let native_state = native_tr.state();
        for (idx, (circuit_value, native_value)) in circuit_tr
            .state_values()
            .iter()
            .zip(native_state.iter())
            .enumerate()
        {
            let expected = SpartanF::from_canonical_u64(native_value.as_canonical_u64());
            if *circuit_value != expected {
                return Some(format!(
                    "{label}: lane {idx} {} != {}",
                    circuit_value.to_canonical_u64(),
                    expected.to_canonical_u64()
                ));
            }
        }
        None
    }

    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if carried_claims.effective_count() != native_carried_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let transcript_fields = transcript_in
        .state
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("transcript_in_state_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transcript_fields: [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH] = transcript_fields
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut circuit_tr =
        Poseidon2TranscriptCircuit::from_state(transcript_fields, transcript_values, transcript_in.absorbed)?;
    let mut native_tr = Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_chunk_meta")),
        &mut circuit_tr,
        &chunk.handoff,
    )?;
    if chunk.handoff.public_chunk.steps.len() == 1 {
        native_tr.append_fields_raw(&[
            F::from_u64(STEP_INDEX_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
        ]);
    } else {
        native_tr.append_fields_raw(&[
            F::from_u64(CHUNK_META_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
            F::from_u64(chunk.handoff.public_chunk.steps.len() as u64),
        ]);
    }
    if let Some(mismatch) = compare_stage("chunk_meta", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_bind_header")),
        &mut circuit_tr,
        params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        mat_digest,
        &chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )?;
    neo_reductions::engines::utils::bind_header_and_instance_digest_with_digest(
        &mut native_tr,
        params,
        structure,
        dims,
        &mat_digest.map(|value| F::from_u64(value.as_canonical_u64())),
        &chunk.handoff.public_chunk_instance_digest,
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;
    if let Some(mismatch) = compare_stage("bind_header", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    bind_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_bind_me_inputs")),
        &mut circuit_tr,
        carried_claims.effective_claims(),
        None,
    )?;
    neo_reductions::engines::utils::bind_me_inputs(&mut native_tr, native_carried_claims)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    if let Some(mismatch) = compare_stage("bind_me_inputs", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let public_challenges = sample_challenges(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_sample_challenges")),
        &mut circuit_tr,
        dims,
    )?;
    let mut native_challenges = neo_reductions::engines::utils::sample_challenges(&mut native_tr, dims.ell_d, dims.ell)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    native_challenges.beta_m = neo_reductions::engines::utils::sample_beta_m(&mut native_tr, dims.ell_m)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    if let Some(mismatch) = compare_stage("sample_challenges", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let effective_fresh_claim_count = chunk.fresh_claims.len();
    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_initial_sum_fe")),
        structure,
        &public_challenges.alpha,
        &chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        chunk.pi_ccs.public_challenges.gamma,
        effective_fresh_claim_count,
        carried_claims.effective_claims(),
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_transcript_initial_sum_fe"),
    )?;
    let native_initial_sum = claimed_initial_sum_from_inputs_with_k_mcs(
        structure,
        &native_challenges,
        effective_fresh_claim_count,
        native_carried_claims,
    );

    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_sumcheck_domain")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
    )?;
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        circuit_tr.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_sumcheck_initial_tag")),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )?;
        circuit_tr.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_sumcheck_initial_append")),
            &[
                SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
            ],
        )?;
    } else {
        append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_sumcheck_initial")),
            &mut circuit_tr,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{chunk_index}_transcript_fe_sumcheck_initial"),
        )?;
    }
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native_tr.append_fields_raw(&native_initial_sum.as_coeffs());
    if let Some(mismatch) = compare_stage("fe_sumcheck_initial", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let padded_fe_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_rounds")),
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{chunk_index}_transcript_fe_round"),
    )?;
    let fe_round_values = pad_round_values(
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )?;
    let fe_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.row_chals, &chunk.pi_ccs.alpha_prime);
    let _ = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_fe_sumcheck")),
        &mut circuit_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_transcript_fe_sumcheck"),
    )?;
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (_, _, ok_fe) = verify_sumcheck_rounds_poseidon_v3(
        &mut native_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
        native_initial_sum,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
    );
    if !ok_fe {
        return Ok("fe_sumcheck_native_invalid".into());
    }
    if let Some(mismatch) = compare_stage("fe_sumcheck", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let zero_nc = alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_initial_sum_nc_zero")),
        KNum::from_neo_k(K::ZERO),
        &format!("chunk_{chunk_index}_transcript_initial_sum_nc_zero"),
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_transcript_nc_sumcheck_domain")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_transcript_nc_sumcheck_initial_tag")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_transcript_nc_sumcheck_initial_append")),
        &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
    )?;
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native_tr.append_fields_raw(&[F::ZERO, F::ZERO]);
    if let Some(mismatch) = compare_stage("nc_sumcheck_initial", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let padded_nc_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_nc_rounds")),
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{chunk_index}_transcript_nc_round"),
    )?;
    let nc_round_values = pad_round_values(
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )?;
    let nc_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.s_col, &chunk.pi_ccs.alpha_prime_nc);
    let _ = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_transcript_nc_sumcheck")),
        &mut circuit_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_transcript_nc_sumcheck"),
    )?;
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (_, _, ok_nc) = verify_sumcheck_rounds_poseidon_v3(
        &mut native_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
        K::ZERO,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    );
    if !ok_nc {
        return Ok("nc_sumcheck_native_invalid".into());
    }
    if let Some(mismatch) = compare_stage("nc_sumcheck", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    let _ = circuit_tr.digest32(cs.namespace(|| format!("chunk_{chunk_index}_transcript_fold_digest")))?;
    let _ = native_tr.digest32();
    if let Some(mismatch) = compare_stage("fold_digest", &circuit_tr, &native_tr) {
        return Ok(mismatch);
    }

    Ok("late_match".into())
}

pub(crate) fn debug_compare_rv64im_pi_rlc_rho_mats(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    carried_claims: Rv64imClaimBundle,
    native_carried_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
) -> Result<String, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) || chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if carried_claims.effective_count() != native_carried_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let transcript_fields = transcript_in
        .state
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("rho_transcript_in_state_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transcript_fields: [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH] = transcript_fields
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut circuit_tr =
        Poseidon2TranscriptCircuit::from_state(transcript_fields, transcript_values, transcript_in.absorbed)?;
    let mut native_tr = Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);

    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_chunk_meta")),
        &mut circuit_tr,
        &chunk.handoff,
    )?;
    if chunk.handoff.public_chunk.steps.len() == 1 {
        native_tr.append_fields_raw(&[
            F::from_u64(STEP_INDEX_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
        ]);
    } else {
        native_tr.append_fields_raw(&[
            F::from_u64(CHUNK_META_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
            F::from_u64(chunk.handoff.public_chunk.steps.len() as u64),
        ]);
    }

    bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_bind_header")),
        &mut circuit_tr,
        params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        mat_digest,
        &chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )?;
    neo_reductions::engines::utils::bind_header_and_instance_digest_with_digest(
        &mut native_tr,
        params,
        structure,
        dims,
        &mat_digest.map(|value| F::from_u64(value.as_canonical_u64())),
        &chunk.handoff.public_chunk_instance_digest,
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;

    bind_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_bind_me_inputs")),
        &mut circuit_tr,
        carried_claims.effective_claims(),
        None,
    )?;
    neo_reductions::engines::utils::bind_me_inputs(&mut native_tr, native_carried_claims)
        .map_err(|_| SynthesisError::Unsatisfiable)?;

    let public_challenges = sample_challenges(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_sample_challenges")),
        &mut circuit_tr,
        dims,
    )?;
    let mut native_challenges = neo_reductions::engines::utils::sample_challenges(&mut native_tr, dims.ell_d, dims.ell)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    native_challenges.beta_m = neo_reductions::engines::utils::sample_beta_m(&mut native_tr, dims.ell_m)
        .map_err(|_| SynthesisError::Unsatisfiable)?;

    let effective_fresh_claim_count = chunk.fresh_claims.len();
    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_initial_sum_fe")),
        structure,
        &public_challenges.alpha,
        &chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        chunk.pi_ccs.public_challenges.gamma,
        effective_fresh_claim_count,
        carried_claims.effective_claims(),
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_rho_initial_sum_fe"),
    )?;
    let native_initial_sum = claimed_initial_sum_from_inputs_with_k_mcs(
        structure,
        &native_challenges,
        effective_fresh_claim_count,
        native_carried_claims,
    );

    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_sumcheck_domain")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
    )?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        circuit_tr.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_sumcheck_initial_tag")),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )?;
        circuit_tr.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_sumcheck_initial_append")),
            &[
                SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
            ],
        )?;
    } else {
        append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_sumcheck_initial")),
            &mut circuit_tr,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{chunk_index}_rho_fe_sumcheck_initial"),
        )?;
    }
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native_tr.append_fields_raw(&native_initial_sum.as_coeffs());

    let padded_fe_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_rounds")),
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{chunk_index}_rho_fe_round"),
    )?;
    let fe_round_values = pad_round_values(
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )?;
    let fe_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.row_chals, &chunk.pi_ccs.alpha_prime);
    let _ = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_fe_sumcheck")),
        &mut circuit_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_rho_fe_sumcheck"),
    )?;
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (_, _, ok_fe) = verify_sumcheck_rounds_poseidon_v3(
        &mut native_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
        native_initial_sum,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
    );
    if !ok_fe {
        return Ok("fe_sumcheck_native_invalid".into());
    }

    let zero_nc = alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_initial_sum_nc_zero")),
        KNum::from_neo_k(K::ZERO),
        &format!("chunk_{chunk_index}_rho_initial_sum_nc_zero"),
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_rho_nc_sumcheck_domain")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_rho_nc_sumcheck_initial_tag")),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
    )?;
    circuit_tr.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{chunk_index}_rho_nc_sumcheck_initial_append")),
        &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
    )?;
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native_tr.append_fields_raw(&[F::ZERO, F::ZERO]);

    let padded_nc_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_nc_rounds")),
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{chunk_index}_rho_nc_round"),
    )?;
    let nc_round_values = pad_round_values(
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )?;
    let nc_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.s_col, &chunk.pi_ccs.alpha_prime_nc);
    let _ = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_nc_sumcheck")),
        &mut circuit_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{chunk_index}_rho_nc_sumcheck"),
    )?;
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (_, _, ok_nc) = verify_sumcheck_rounds_poseidon_v3(
        &mut native_tr,
        max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
        K::ZERO,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    );
    if !ok_nc {
        return Ok("nc_sumcheck_native_invalid".into());
    }

    let _ = circuit_tr.digest32(cs.namespace(|| format!("chunk_{chunk_index}_rho_fold_digest")))?;
    let _ = native_tr.digest32();

    let circuit_rhos = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_sample")),
        &mut circuit_tr,
        chunk.pi_ccs.ccs_outputs.len(),
        &format!("chunk_{chunk_index}_rho_sample"),
    )?;
    let circuit_mats = materialize_goldilocks_rot_matrices(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rho_materialize")),
        &circuit_rhos,
        &format!("chunk_{chunk_index}_rho_materialize"),
    )?;
    let native_rhos = neo_reductions::sample_rot_rhos_n_typed(
        &mut native_tr,
        params,
        &neo_reductions::RotRing::goldilocks(),
        chunk.pi_ccs.ccs_outputs.len(),
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;

    for (rho_idx, (circuit_rho, native_rho)) in circuit_mats.iter().zip(native_rhos.iter()).enumerate() {
        let native_mat = native_rho.as_mat();
        for row in 0..D {
            for col in 0..D {
                let observed = circuit_rho.entry_value(row, col)?;
                let expected = native_mat.row(row)[col];
                if observed != expected {
                    return Ok(format!(
                        "rho_mat[{rho_idx}][{row},{col}] mismatch (expected {}, observed {})",
                        expected.as_canonical_u64(),
                        observed.as_canonical_u64()
                    ));
                }
            }
        }
    }

    Ok("rho_mats_match".into())
}
