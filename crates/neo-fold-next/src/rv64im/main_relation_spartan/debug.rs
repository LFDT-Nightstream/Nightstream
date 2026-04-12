//! Debug and sizing helpers for the RV64IM main-relation Spartan circuit.

use bellpepper_core::{
    test_cs::TestConstraintSystem, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use ff::PrimeField;
use neo_math::{from_complex, KExtensions, F, K};
use neo_reductions::common::decompose_balanced_fixed_d_digits_k;
use neo_reductions::engines::utils::{
    bind_header_and_instance_digest_with_digest, PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
};
use neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG;
use neo_transcript::Poseidon2Transcript;
use std::collections::BTreeMap;

use super::*;
use crate::rv64im::main_relation_circuit::claim::packed_bytes_field_values;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationTraceStats {
    pub chunk_count: usize,
    pub final_claim_count: usize,
    pub fresh_claim_count: usize,
    pub ccs_output_count: usize,
    pub child_claim_count: usize,
    pub fe_round_count: usize,
    pub nc_round_count: usize,
    pub fresh_witness_cells: usize,
    pub child_witness_cells: usize,
    pub max_chunk_public_steps: usize,
    pub max_chunk_witness_cells: usize,
    pub fe_round_coeff_count: usize,
    pub nc_round_coeff_count: usize,
    pub output_logical_col_count: usize,
    pub output_digit_slot_count: usize,
    pub output_nonzero_digit_count: usize,
    pub output_y_zcol_slot_count: usize,
    pub output_nonzero_y_zcol_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationCircuitMetrics {
    pub trace: Rv64imMainRelationTraceStats,
    pub surface: Rv64imMainRelationSurfaceMetrics,
    pub public_input_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub linear_constraint_count: usize,
    pub quadratic_constraint_count: usize,
    pub a_term_count: usize,
    pub b_term_count: usize,
    pub c_term_count: usize,
    pub total_term_count: usize,
    pub max_constraint_term_count: usize,
    pub max_claim_digest_constraint_count: usize,
    pub max_claim_digest_namespace: String,
    pub hotspots: Vec<Rv64imMainRelationCountBucket>,
    pub hotspot_details: Vec<Rv64imMainRelationHotspotDetail>,
    pub representative_claim_details: Vec<Rv64imMainRelationHotspotDetail>,
    pub phase_rollup: Vec<Rv64imMainRelationPhaseBucket>,
    pub component_rollup: Vec<Rv64imMainRelationComponentBucket>,
    pub sumcheck_rollup: Vec<Rv64imMainRelationSumcheckBucket>,
    pub rho_rollup: Vec<Rv64imMainRelationRhoBucket>,
    pub claim_family_rollup: Vec<Rv64imMainRelationClaimFamilyBucket>,
    pub family_component_rollup: Vec<Rv64imMainRelationFamilyComponentBucket>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationCountBucket {
    pub namespace: String,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_constraint_term_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationHotspotDetail {
    pub parent_namespace: String,
    pub total_constraint_count: usize,
    pub leaf_buckets: Vec<Rv64imMainRelationCountBucket>,
    pub leaf_coverage_constraint_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationPhaseBucket {
    pub phase: String,
    pub bucket_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_bucket_constraint_count: usize,
    pub max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationComponentBucket {
    pub component: String,
    pub bucket_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_bucket_constraint_count: usize,
    pub max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationSumcheckBucket {
    pub bucket: String,
    pub bucket_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_bucket_constraint_count: usize,
    pub max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationRhoBucket {
    pub bucket: String,
    pub bucket_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_bucket_constraint_count: usize,
    pub max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationFamilyComponentBucket {
    pub family: String,
    pub component: String,
    pub bucket_count: usize,
    pub aux_count: usize,
    pub constraint_count: usize,
    pub total_term_count: usize,
    pub max_bucket_constraint_count: usize,
    pub max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationClaimFamilyBucket {
    pub family: String,
    pub rlc_bucket_count: usize,
    pub rlc_aux_count: usize,
    pub rlc_constraint_count: usize,
    pub rlc_total_term_count: usize,
    pub rlc_max_bucket_constraint_count: usize,
    pub rlc_max_bucket_namespace: String,
    pub final_bucket_count: usize,
    pub final_aux_count: usize,
    pub final_constraint_count: usize,
    pub final_total_term_count: usize,
    pub final_max_bucket_constraint_count: usize,
    pub final_max_bucket_namespace: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationSurfaceMetrics {
    pub rlc_parent_claim_count: usize,
    pub final_claim_count: usize,
    pub rlc_public_field_coords_total: usize,
    pub rlc_public_k_coords_total: usize,
    pub final_claim_field_coords_total: usize,
    pub final_claim_k_coords_total: usize,
    pub families: Vec<Rv64imMainRelationSurfaceFamilyBucket>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rv64imMainRelationSurfaceFamilyBucket {
    pub family: String,
    pub rlc_public_field_coords_total: usize,
    pub rlc_public_k_coords_total: usize,
    pub final_claim_field_coords_total: usize,
    pub final_claim_k_coords_total: usize,
}

pub fn inspect_rv64im_spartan2_decider_trace(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainRelationTraceStats, SimpleKernelError> {
    let circuit = build_main_relation_circuit(statement, proof)?;
    Ok(trace_stats(&circuit))
}

pub fn measure_rv64im_spartan2_decider_circuit(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainRelationCircuitMetrics, SimpleKernelError> {
    let circuit = build_main_relation_circuit(statement, proof)?;
    let trace = trace_stats(&circuit);
    let mut cs = CountingCS::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation metric synthesis failed: {err}")))?;
    Ok(Rv64imMainRelationCircuitMetrics {
        trace,
        surface: surface_metrics(&circuit),
        public_input_count: cs.num_inputs,
        aux_count: cs.num_aux,
        constraint_count: cs.num_constraints,
        linear_constraint_count: cs.linear_constraint_count,
        quadratic_constraint_count: cs.quadratic_constraint_count,
        a_term_count: cs.a_term_count,
        b_term_count: cs.b_term_count,
        c_term_count: cs.c_term_count,
        total_term_count: cs.a_term_count + cs.b_term_count + cs.c_term_count,
        max_constraint_term_count: cs.max_constraint_term_count,
        max_claim_digest_constraint_count: cs.max_claim_digest_constraint_count(),
        max_claim_digest_namespace: cs.max_claim_digest_namespace(),
        hotspots: cs.hotspots(40),
        hotspot_details: cs.hotspot_details(5, 40),
        representative_claim_details: cs.representative_claim_details(16),
        phase_rollup: cs.phase_rollup(),
        component_rollup: cs.component_rollup(),
        sumcheck_rollup: cs.sumcheck_rollup(),
        rho_rollup: cs.rho_rollup(),
        claim_family_rollup: cs.claim_family_rollup(),
        family_component_rollup: cs.family_component_rollup(),
    })
}

fn surface_metrics(circuit: &Rv64imMainRelationCircuit) -> Rv64imMainRelationSurfaceMetrics {
    let mut families = BTreeMap::<String, Rv64imMainRelationSurfaceFamilyBucket>::new();
    let mut rlc_parent_claim_count = 0usize;

    for chunk in &circuit.trace.chunk_traces {
        rlc_parent_claim_count += 1;
        let claim = &chunk.ccs_trace.parent;
        add_rlc_family_field(&mut families, "x", claim.X.as_slice().len());
        add_rlc_family_field(&mut families, "c_data", claim.c.data.len());
        add_rlc_family_k(&mut families, "r", claim.r.len());
        add_rlc_family_k(&mut families, "s_col", claim.s_col.len());
        add_rlc_family_k(&mut families, "y_ring", claim.y_ring.iter().map(Vec::len).sum());
        add_rlc_family_k(&mut families, "ct", claim.ct.len());
        add_rlc_family_k(&mut families, "aux_openings", claim.aux_openings.len());
        add_rlc_family_k(&mut families, "y_zcol", claim.y_zcol.len());
    }

    let final_claims = &circuit
        .trace
        .statement
        .folded
        .final_accumulator
        .final_main_claims;
    for claim in final_claims {
        add_final_family_field(&mut families, "x", claim.X.as_slice().len());
        add_final_family_field(&mut families, "c_data", claim.c.data.len());
        add_final_family_k(&mut families, "r", claim.r.len());
        add_final_family_k(&mut families, "s_col", claim.s_col.len());
        add_final_family_k(&mut families, "y_ring", claim.y_ring.iter().map(Vec::len).sum());
        add_final_family_k(&mut families, "ct", claim.ct.len());
        add_final_family_k(&mut families, "aux_openings", claim.aux_openings.len());
        add_final_family_k(&mut families, "y_zcol", claim.y_zcol.len());
        add_final_family_field(&mut families, "c_step_coords", claim.c_step_coords.len());
        add_final_family_field(
            &mut families,
            "fold_digest_encoding",
            packed_bytes_field_values(&claim.fold_digest).len(),
        );
    }

    let families = families.into_values().collect::<Vec<_>>();
    Rv64imMainRelationSurfaceMetrics {
        rlc_parent_claim_count,
        final_claim_count: final_claims.len(),
        rlc_public_field_coords_total: families
            .iter()
            .map(|bucket| bucket.rlc_public_field_coords_total)
            .sum(),
        rlc_public_k_coords_total: families
            .iter()
            .map(|bucket| bucket.rlc_public_k_coords_total)
            .sum(),
        final_claim_field_coords_total: families
            .iter()
            .map(|bucket| bucket.final_claim_field_coords_total)
            .sum(),
        final_claim_k_coords_total: families
            .iter()
            .map(|bucket| bucket.final_claim_k_coords_total)
            .sum(),
        families,
    }
}

fn add_rlc_family_field(
    families: &mut BTreeMap<String, Rv64imMainRelationSurfaceFamilyBucket>,
    family: &str,
    count: usize,
) {
    families
        .entry(family.to_string())
        .or_insert_with(|| Rv64imMainRelationSurfaceFamilyBucket {
            family: family.to_string(),
            ..Rv64imMainRelationSurfaceFamilyBucket::default()
        })
        .rlc_public_field_coords_total += count;
}

fn add_rlc_family_k(
    families: &mut BTreeMap<String, Rv64imMainRelationSurfaceFamilyBucket>,
    family: &str,
    count: usize,
) {
    families
        .entry(family.to_string())
        .or_insert_with(|| Rv64imMainRelationSurfaceFamilyBucket {
            family: family.to_string(),
            ..Rv64imMainRelationSurfaceFamilyBucket::default()
        })
        .rlc_public_k_coords_total += count;
}

fn add_final_family_field(
    families: &mut BTreeMap<String, Rv64imMainRelationSurfaceFamilyBucket>,
    family: &str,
    count: usize,
) {
    families
        .entry(family.to_string())
        .or_insert_with(|| Rv64imMainRelationSurfaceFamilyBucket {
            family: family.to_string(),
            ..Rv64imMainRelationSurfaceFamilyBucket::default()
        })
        .final_claim_field_coords_total += count;
}

fn add_final_family_k(
    families: &mut BTreeMap<String, Rv64imMainRelationSurfaceFamilyBucket>,
    family: &str,
    count: usize,
) {
    families
        .entry(family.to_string())
        .or_insert_with(|| Rv64imMainRelationSurfaceFamilyBucket {
            family: family.to_string(),
            ..Rv64imMainRelationSurfaceFamilyBucket::default()
        })
        .final_claim_k_coords_total += count;
}

pub fn debug_check_rv64im_spartan2_decider_circuit(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<(), SimpleKernelError> {
    let circuit = build_main_relation_circuit(statement, proof)?;
    debug_check_first_chunk_fe_alignment(&circuit)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation debug synthesis failed: {err}")))?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

fn trace_stats(circuit: &Rv64imMainRelationCircuit) -> Rv64imMainRelationTraceStats {
    let mut stats = Rv64imMainRelationTraceStats {
        chunk_count: circuit.trace.chunk_traces.len(),
        final_claim_count: circuit
            .trace
            .statement
            .folded
            .final_accumulator
            .final_main_claims
            .len(),
        ..Rv64imMainRelationTraceStats::default()
    };
    let mut carried_witnesses: Vec<neo_ccs::Mat<F>> = Vec::new();

    for chunk in &circuit.trace.chunk_traces {
        let fresh_witness_cells = chunk
            .fresh_witnesses
            .iter()
            .map(|witness| witness.Z.as_slice().len())
            .sum::<usize>();
        let child_witness_cells = chunk
            .ccs_trace
            .z_split
            .iter()
            .map(|z| z.as_slice().len())
            .sum::<usize>();
        stats.fresh_claim_count += chunk.fresh_claims.len();
        stats.ccs_output_count += chunk.ccs_trace.ccs_outputs.len();
        stats.child_claim_count += chunk.ccs_trace.children.len();
        stats.fresh_witness_cells += fresh_witness_cells;
        stats.child_witness_cells += child_witness_cells;
        stats.max_chunk_public_steps = stats
            .max_chunk_public_steps
            .max(chunk.handoff.public_chunk.steps.len());
        stats.max_chunk_witness_cells = stats
            .max_chunk_witness_cells
            .max(fresh_witness_cells + child_witness_cells);
        stats.fe_round_coeff_count += chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        stats.fe_round_count += chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds.len();
        stats.nc_round_coeff_count += chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds_nc
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        stats.nc_round_count += chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds_nc.len();

        let output_witness_count = chunk.fresh_witnesses.len() + carried_witnesses.len();
        stats.output_logical_col_count += output_witness_count * circuit.structure.m;
        stats.output_digit_slot_count += output_witness_count * circuit.structure.m * neo_math::D;
        stats.output_nonzero_digit_count += chunk
            .fresh_witnesses
            .iter()
            .map(|witness| count_nonzero_balanced_digits(&witness.Z, circuit.structure.m, circuit.params.b))
            .sum::<usize>();
        stats.output_nonzero_digit_count += carried_witnesses
            .iter()
            .map(|witness| count_nonzero_balanced_digits(witness, circuit.structure.m, circuit.params.b))
            .sum::<usize>();
        stats.output_y_zcol_slot_count += chunk
            .ccs_trace
            .ccs_outputs
            .iter()
            .map(|claim| claim.y_zcol.len())
            .sum::<usize>();
        stats.output_nonzero_y_zcol_count += chunk
            .ccs_trace
            .ccs_outputs
            .iter()
            .map(|claim| {
                claim
                    .y_zcol
                    .iter()
                    .filter(|value| **value != K::ZERO)
                    .count()
            })
            .sum::<usize>();
        carried_witnesses = chunk.ccs_trace.z_split.clone();
    }

    stats
}

fn count_nonzero_balanced_digits(witness: &neo_ccs::Mat<F>, expected_m: usize, base_b: u32) -> usize {
    (0..expected_m)
        .map(|logical_col| {
            let raw = decode_packed_logical_entry(witness, expected_m, logical_col)
                .expect("packed witness should decode for perf stats");
            decompose_balanced_fixed_d_digits_k(raw, base_b)
                .expect("perf trace witness should admit balanced digit decomposition")
                .iter()
                .filter(|digit| digit.as_coeffs()[0] != F::ZERO)
                .count()
        })
        .sum()
}

fn decode_packed_logical_entry(
    witness: &neo_ccs::Mat<F>,
    expected_m: usize,
    logical_col: usize,
) -> Result<F, SynthesisError> {
    if witness.rows() != neo_math::D || expected_m == 0 || logical_col >= expected_m {
        return Err(SynthesisError::Unsatisfiable);
    }
    let block = logical_col / neo_math::D;
    let off = logical_col % neo_math::D;
    witness
        .as_slice()
        .get(off * witness.cols() + block)
        .copied()
        .ok_or(SynthesisError::Unsatisfiable)
}

fn debug_check_first_chunk_fe_alignment(circuit: &Rv64imMainRelationCircuit) -> Result<(), SimpleKernelError> {
    let Some(chunk) = circuit.trace.chunk_traces.first() else {
        return Ok(());
    };

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut transcript = Poseidon2TranscriptCircuit::new_raw_fields(
        cs.namespace(|| "tr"),
        &[SpartanF::from_canonical_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)],
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk transcript init failed: {err}")))?;
    let mut native = Poseidon2Transcript::new_raw_fields(&[F::from_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)]);
    compare_transcript_state("init", &transcript, &native)?;
    append_chunk_meta(&mut cs.namespace(|| "chunk_meta"), &mut transcript, &chunk.handoff)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk meta failed: {err}")))?;
    if chunk.handoff.public_chunk.steps.len() == 1 {
        native.append_fields_raw(&[
            F::from_u64(STEP_INDEX_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
        ]);
    } else {
        native.append_fields_raw(&[
            F::from_u64(CHUNK_META_RAW_TAG),
            F::from_u64(chunk.handoff.public_chunk.start_index as u64),
            F::from_u64(chunk.handoff.public_chunk.steps.len() as u64),
        ]);
    }
    compare_transcript_state("chunk_meta", &transcript, &native)?;

    bind_header_and_instance_digest(
        &mut cs.namespace(|| "bind_header"),
        &mut transcript,
        &circuit.params,
        circuit.structure.n,
        circuit.structure.m,
        circuit.structure.t(),
        &circuit.structure.f,
        circuit.dims,
        &circuit.mat_digest,
        &chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk header binding failed: {err}")))?;
    bind_header_and_instance_digest_with_digest(
        &mut native,
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit
            .mat_digest
            .map(|value| F::from_u64(value.as_canonical_u64())),
        &chunk.handoff.public_chunk_instance_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk native header binding failed: {err}")))?;
    compare_transcript_state("bind_header", &transcript, &native)?;
    bind_me_inputs(&mut cs.namespace(|| "bind_me_inputs"), &mut transcript, &[])
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk circuit ME binding failed: {err}")))?;
    neo_reductions::engines::utils::bind_me_inputs(&mut native, &[])
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk native ME binding failed: {err}")))?;
    compare_transcript_state("bind_me_inputs", &transcript, &native)?;
    let public_challenges = sample_challenges(&mut cs.namespace(|| "sample_public"), &mut transcript, circuit.dims)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM first-chunk public challenge replay failed: {err}"))
        })?;
    let mut native_public_challenges =
        neo_reductions::engines::utils::sample_challenges(&mut native, circuit.dims.ell_d, circuit.dims.ell).map_err(
            |err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM first-chunk native public challenge replay failed: {err}"
                ))
            },
        )?;
    native_public_challenges.beta_m = neo_reductions::engines::utils::sample_beta_m(&mut native, circuit.dims.ell_m)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk native beta_m replay failed: {err}")))?;
    compare_transcript_state("sample_public", &transcript, &native)?;
    if native_public_challenges.alpha != chunk.replay_public_challenges.alpha
        || native_public_challenges.beta_a != chunk.replay_public_challenges.beta_a
        || native_public_challenges.beta_r != chunk.replay_public_challenges.beta_r
        || native_public_challenges.beta_m != chunk.replay_public_challenges.beta_m
        || native_public_challenges.gamma != chunk.replay_public_challenges.gamma
    {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM first-chunk native/traced public challenge mismatch: native alpha={:?}, traced alpha={:?}, terminal alpha={:?}, native beta_a={:?}, traced beta_a={:?}, terminal beta_a={:?}, native beta_r={:?}, traced beta_r={:?}, terminal beta_r={:?}, native beta_m={:?}, traced beta_m={:?}, terminal beta_m={:?}, native gamma={:?}, traced gamma={:?}, terminal gamma={:?}",
            native_public_challenges.alpha,
            chunk.replay_public_challenges.alpha,
            chunk.ccs_trace.terminal_state.challenges_public.alpha,
            native_public_challenges.beta_a,
            chunk.replay_public_challenges.beta_a,
            chunk.ccs_trace.terminal_state.challenges_public.beta_a,
            native_public_challenges.beta_r,
            chunk.replay_public_challenges.beta_r,
            chunk.ccs_trace.terminal_state.challenges_public.beta_r,
            native_public_challenges.beta_m,
            chunk.replay_public_challenges.beta_m,
            chunk.ccs_trace.terminal_state.challenges_public.beta_m,
            native_public_challenges.gamma,
            chunk.replay_public_challenges.gamma,
            chunk.ccs_trace.terminal_state.challenges_public.gamma,
        )));
    }
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "alpha_native_eq"),
            &public_challenges.alpha,
            &native_public_challenges.alpha,
            "alpha_native_eq",
        ),
        "RV64IM first-chunk alpha/native compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_a_native_eq"),
            &public_challenges.beta_a,
            &native_public_challenges.beta_a,
            "beta_a_native_eq",
        ),
        "RV64IM first-chunk beta_a/native compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_r_native_eq"),
            &public_challenges.beta_r,
            &native_public_challenges.beta_r,
            "beta_r_native_eq",
        ),
        "RV64IM first-chunk beta_r/native compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_m_native_eq"),
            &public_challenges.beta_m,
            &native_public_challenges.beta_m,
            "beta_m_native_eq",
        ),
        "RV64IM first-chunk beta_m/native compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "gamma_native_eq"),
            core::slice::from_ref(&public_challenges.gamma),
            core::slice::from_ref(&native_public_challenges.gamma),
            "gamma_native_eq",
        ),
        "RV64IM first-chunk gamma/native compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "alpha_eq"),
            &public_challenges.alpha,
            &chunk.replay_public_challenges.alpha,
            "alpha_eq",
        ),
        "RV64IM first-chunk alpha compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_a_eq"),
            &public_challenges.beta_a,
            &chunk.replay_public_challenges.beta_a,
            "beta_a_eq",
        ),
        "RV64IM first-chunk beta_a compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_r_eq"),
            &public_challenges.beta_r,
            &chunk.replay_public_challenges.beta_r,
            "beta_r_eq",
        ),
        "RV64IM first-chunk beta_r compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "beta_m_eq"),
            &public_challenges.beta_m,
            &chunk.replay_public_challenges.beta_m,
            "beta_m_eq",
        ),
        "RV64IM first-chunk beta_m compare failed",
    )?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "gamma_eq"),
            core::slice::from_ref(&public_challenges.gamma),
            core::slice::from_ref(&chunk.replay_public_challenges.gamma),
            "gamma_eq",
        ),
        "RV64IM first-chunk gamma compare failed",
    )?;

    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| "initial_sum_fe"),
        &circuit.structure,
        &public_challenges.alpha,
        &chunk.replay_public_challenges.alpha,
        &public_challenges.gamma,
        chunk.replay_public_challenges.gamma,
        chunk.fresh_claims.len(),
        &[],
        Rv64imMainRelationCircuit::delta(),
        "initial_sum_fe",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk initial sum failed: {err}")))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| "fe_domain"),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE domain failed: {err}")))?;
    native.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    map_synth(
        append_k_to_transcript(
            &mut cs.namespace(|| "fe_initial"),
            &mut transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            "fe_initial",
        ),
        "RV64IM first-chunk FE initial binding failed",
    )?;
    native.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native.append_fields_raw(&initial_sum_fe_value.as_coeffs());
    compare_transcript_state("fe_initial", &transcript, &native)?;
    let fe_rounds = alloc_rounds(
        &mut cs.namespace(|| "fe_rounds"),
        &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds,
        "fe_round",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE round alloc failed: {err}")))?;
    let fe_challenge_values = chunk_sumcheck_challenges(&chunk.replay_row_chals, &chunk.replay_alpha_prime);
    let first_round = chunk
        .ccs_trace
        .ccs_replay_proof
        .sumcheck_rounds
        .first()
        .ok_or_else(|| SimpleKernelError::Bridge("RV64IM first-chunk FE rounds missing".into()))?;
    let native_initial_sum = {
        let p0 = *first_round
            .first()
            .ok_or_else(|| SimpleKernelError::Bridge("RV64IM first-chunk FE round0 missing p(0)".into()))?;
        let mut p1 = p0;
        for coeff in first_round.iter().skip(1) {
            p1 += *coeff;
        }
        p0 + p1
    };
    if native_initial_sum != initial_sum_fe_value {
        return Err(SimpleKernelError::Bridge(
            "RV64IM first-chunk FE initial sum mismatch: circuit value != native round-derived value".into(),
        ));
    }
    let first_round_vars = fe_rounds
        .first()
        .ok_or_else(|| SimpleKernelError::Bridge("RV64IM first-chunk FE round vars missing".into()))?;
    let mut first_round_transcript = transcript.clone();
    first_round_transcript
        .append_const_fields_raw(
            cs.namespace(|| "fe_v3"),
            &[SpartanF::from_canonical_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE v3 bind failed: {err}")))?;
    let mut native_first_round = native.clone();
    native_first_round.append_fields_raw(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    compare_transcript_state("fe_v3", &first_round_transcript, &native_first_round)?;
    sumcheck_round_gadget(
        &mut cs.namespace(|| "fe_round0_invariant"),
        first_round_vars,
        first_round,
        &initial_sum_fe,
        "fe_round0_invariant",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE round0 invariant failed: {err}")))?;
    map_synth(
        append_round_coeffs_for_debug(
            &mut cs.namespace(|| "fe_round0_append"),
            &mut first_round_transcript,
            first_round,
            "fe_round0_append",
        ),
        "RV64IM first-chunk FE round0 append failed",
    )?;
    native_first_round.append_fields_raw(&neo_reductions::sumcheck::round_coeff_fields(first_round));
    compare_transcript_state("fe_round0_append", &first_round_transcript, &native_first_round)?;
    let native_first_pair = native_first_round.challenge_fields_raw(2);
    let native_first_challenge = from_complex(
        *native_first_pair.first().ok_or_else(|| {
            SimpleKernelError::Bridge("RV64IM first-chunk native FE first challenge missing c0".into())
        })?,
        *native_first_pair.get(1).ok_or_else(|| {
            SimpleKernelError::Bridge("RV64IM first-chunk native FE first challenge missing c1".into())
        })?,
    );
    let first_pair = first_round_transcript
        .challenge_fields_raw(cs.namespace(|| "fe_round0_challenge_pair"), 2)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM first-chunk FE round0 challenge sample failed: {err}"))
        })?;
    let first_challenge = KNumVar {
        c0: first_pair[0].get_variable(),
        c1: first_pair[1].get_variable(),
    };
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "fe_round0_challenge_eq"),
            core::slice::from_ref(&first_challenge),
            core::slice::from_ref(&native_first_challenge),
            "fe_round0_challenge_eq",
        ),
        "RV64IM first-chunk FE round0 challenge compare failed",
    )?;
    let _ = sumcheck_eval_gadget(
        &mut cs.namespace(|| "fe_round0_eval"),
        first_round_vars,
        first_round,
        &first_challenge,
        native_first_challenge,
        Rv64imMainRelationCircuit::delta(),
        "fe_round0_eval",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE round0 eval failed: {err}")))?;
    let (fe_challenges, _) = verify_sumcheck_rounds(
        &mut cs.namespace(|| "fe_replay"),
        &mut transcript,
        max_degree(&chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds),
        &initial_sum_fe,
        &fe_rounds,
        &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds,
        &fe_challenge_values,
        Rv64imMainRelationCircuit::delta(),
        "fe_replay",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM first-chunk FE replay failed: {err}")))?;
    map_synth(
        enforce_k_slice_against_values(
            &mut cs.namespace(|| "fe_challenge_eq"),
            &fe_challenges,
            &fe_challenge_values,
            "fe_challenge_eq",
        ),
        "RV64IM first-chunk FE challenge compare failed",
    )?;

    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM first-chunk FE alignment unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub(super) fn append_k_to_transcript<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    raw_tag: u64,
    value: &KNumVar,
    value_hint: K,
    name: &str,
) -> Result<(), SynthesisError> {
    let coeffs = value_hint.as_coeffs();
    let coeff_fields = [value.c0, value.c1];
    let coeff_values = [
        SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
        SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
    ];
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("{name}_tag")),
        &[SpartanF::from_canonical_u64(raw_tag)],
    )?;
    transcript.append_field_vars_raw(cs.namespace(|| format!("{name}_append")), &coeff_fields, &coeff_values)
}

fn enforce_k_slice_against_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    vars: &[KNumVar],
    values: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    if vars.len() != values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (var, value)) in vars.iter().zip(values.iter()).enumerate() {
        let expected = alloc_constant_k(cs, KNum::from_neo_k(*value), &format!("{label}_{idx}"))?;
        crate::rv64im::main_relation_circuit::k_field::enforce_k_eq(cs, var, &expected, &format!("{label}_{idx}_eq"));
    }
    Ok(())
}

fn append_round_coeffs_for_debug<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    coeff_values: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("{label}_append")),
        &coeff_values
            .iter()
            .flat_map(|value| {
                let parts = value.as_coeffs();
                [
                    SpartanF::from_canonical_u64(parts[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(parts[1].as_canonical_u64()),
                ]
            })
            .collect::<Vec<_>>(),
    )
}

fn compare_transcript_state(
    label: &str,
    circuit: &Poseidon2TranscriptCircuit,
    native: &Poseidon2Transcript,
) -> Result<(), SimpleKernelError> {
    if circuit.absorbed() != native.absorbed() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM transcript mismatch after {label}: absorbed {} != {}",
            circuit.absorbed(),
            native.absorbed()
        )));
    }
    let native_state = native.state();
    for (idx, (circuit_value, native_value)) in circuit
        .state_values()
        .iter()
        .zip(native_state.iter())
        .enumerate()
    {
        let expected = SpartanF::from_canonical_u64(native_value.as_canonical_u64());
        if *circuit_value != expected {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM transcript mismatch after {label} at limb {idx}: {} != {}",
                circuit_value.to_canonical_u64(),
                expected.to_canonical_u64()
            )));
        }
    }
    Ok(())
}

struct CountingCS<Scalar: PrimeField> {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    linear_constraint_count: usize,
    quadratic_constraint_count: usize,
    a_term_count: usize,
    b_term_count: usize,
    c_term_count: usize,
    max_constraint_term_count: usize,
    namespaces: Vec<String>,
    buckets: BTreeMap<String, CountBucketAcc>,
    leaf_buckets: BTreeMap<String, CountBucketAcc>,
    _marker: core::marker::PhantomData<Scalar>,
}

impl<Scalar: PrimeField> CountingCS<Scalar> {
    fn compact_len(lc: &LinearCombination<Scalar>) -> usize {
        lc.iter()
            .filter(|(_, coeff)| !bool::from(coeff.is_zero()))
            .count()
    }

    fn is_unit_lc(lc: &LinearCombination<Scalar>) -> bool {
        let terms = lc
            .iter()
            .filter(|(_, coeff)| !bool::from(coeff.is_zero()))
            .collect::<Vec<_>>();
        let Some((var, coeff)) = terms.first() else {
            return false;
        };
        terms.len() == 1 && *var == Variable::new_unchecked(Index::Input(0)) && **coeff == Scalar::ONE
    }

    fn current_bucket_key(&self) -> String {
        self.namespaces
            .first()
            .cloned()
            .unwrap_or_else(|| "<root>".to_string())
    }

    fn current_leaf_key(&self) -> String {
        if self.namespaces.is_empty() {
            "<root>".to_string()
        } else {
            self.namespaces.join("/")
        }
    }

    fn hotspots(&self, limit: usize) -> Vec<Rv64imMainRelationCountBucket> {
        let mut ranked = self
            .buckets
            .iter()
            .map(|(namespace, bucket)| Rv64imMainRelationCountBucket {
                namespace: namespace.clone(),
                aux_count: bucket.aux_count,
                constraint_count: bucket.constraint_count,
                total_term_count: bucket.total_term_count,
                max_constraint_term_count: bucket.max_constraint_term_count,
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
        });
        ranked.truncate(limit);
        ranked
    }

    fn hotspot_details(&self, parent_limit: usize, child_limit: usize) -> Vec<Rv64imMainRelationHotspotDetail> {
        let mut details = Vec::new();
        for parent in self.hotspots(parent_limit) {
            if let Some(detail) = self.detail_for_namespace(&parent.namespace, child_limit) {
                details.push(detail);
            }
        }
        details
    }

    fn representative_claim_details(&self, child_limit: usize) -> Vec<Rv64imMainRelationHotspotDetail> {
        [
            "chunk_0_carrier_outputs",
            "chunk_0_carrier_children",
            "chunk_0_carrier_parent",
        ]
        .into_iter()
        .filter_map(|namespace| self.detail_for_namespace(namespace, child_limit))
        .collect()
    }

    fn detail_for_namespace(&self, namespace: &str, child_limit: usize) -> Option<Rv64imMainRelationHotspotDetail> {
        let parent = self.buckets.get(namespace)?;
        let prefix = format!("{namespace}/");
        let mut grouped = BTreeMap::<String, CountBucketAcc>::new();
        for (leaf_namespace, bucket) in self
            .leaf_buckets
            .iter()
            .filter(|(leaf_namespace, _)| *leaf_namespace == namespace || leaf_namespace.starts_with(&prefix))
        {
            let relative = leaf_namespace
                .strip_prefix(&prefix)
                .unwrap_or(leaf_namespace)
                .split('/')
                .next()
                .unwrap_or("<self>")
                .to_string();
            let entry = grouped.entry(relative).or_default();
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            entry.max_constraint_term_count = entry
                .max_constraint_term_count
                .max(bucket.max_constraint_term_count);
        }
        let mut leaves = grouped
            .into_iter()
            .map(|(leaf_namespace, bucket)| Rv64imMainRelationCountBucket {
                namespace: leaf_namespace,
                aux_count: bucket.aux_count,
                constraint_count: bucket.constraint_count,
                total_term_count: bucket.total_term_count,
                max_constraint_term_count: bucket.max_constraint_term_count,
            })
            .collect::<Vec<_>>();
        leaves.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
        });
        let leaf_coverage_constraint_count = leaves
            .iter()
            .take(child_limit)
            .map(|bucket| bucket.constraint_count)
            .sum();
        leaves.truncate(child_limit);
        Some(Rv64imMainRelationHotspotDetail {
            parent_namespace: namespace.to_string(),
            total_constraint_count: parent.constraint_count,
            leaf_buckets: leaves,
            leaf_coverage_constraint_count,
        })
    }

    fn phase_rollup(&self) -> Vec<Rv64imMainRelationPhaseBucket> {
        let mut phases = BTreeMap::<String, Rv64imMainRelationPhaseBucket>::new();
        for (namespace, bucket) in &self.buckets {
            let phase = phase_name(namespace).to_string();
            let entry = phases
                .entry(phase.clone())
                .or_insert_with(|| Rv64imMainRelationPhaseBucket {
                    phase,
                    ..Rv64imMainRelationPhaseBucket::default()
                });
            entry.bucket_count += 1;
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            if bucket.constraint_count > entry.max_bucket_constraint_count {
                entry.max_bucket_constraint_count = bucket.constraint_count;
                entry.max_bucket_namespace = namespace.clone();
            }
        }
        let mut phases = phases.into_values().collect::<Vec<_>>();
        phases.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
        });
        phases
    }

    fn component_rollup(&self) -> Vec<Rv64imMainRelationComponentBucket> {
        let mut grouped = BTreeMap::<(String, String), CountBucketAcc>::new();
        for (namespace, bucket) in &self.leaf_buckets {
            let component = component_name(namespace).to_string();
            let canonical_namespace = component_bucket_namespace(namespace, &component);
            let entry = grouped.entry((component, canonical_namespace)).or_default();
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            entry.max_constraint_term_count = entry.max_constraint_term_count.max(bucket.constraint_count);
        }
        let mut components = BTreeMap::<String, Rv64imMainRelationComponentBucket>::new();
        for ((component, canonical_namespace), bucket) in grouped {
            let entry = components
                .entry(component.clone())
                .or_insert_with(|| Rv64imMainRelationComponentBucket {
                    component,
                    ..Rv64imMainRelationComponentBucket::default()
                });
            entry.bucket_count += 1;
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            if bucket.constraint_count > entry.max_bucket_constraint_count {
                entry.max_bucket_constraint_count = bucket.constraint_count;
                entry.max_bucket_namespace = canonical_namespace;
            }
        }
        let mut components = components.into_values().collect::<Vec<_>>();
        components.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
        });
        components
    }

    fn sumcheck_rollup(&self) -> Vec<Rv64imMainRelationSumcheckBucket> {
        let mut buckets = BTreeMap::<String, Rv64imMainRelationSumcheckBucket>::new();
        for (namespace, bucket) in &self.leaf_buckets {
            let Some(bucket_name) = sumcheck_bucket_name(namespace) else {
                continue;
            };
            let entry = buckets
                .entry(bucket_name.to_string())
                .or_insert_with(|| Rv64imMainRelationSumcheckBucket {
                    bucket: bucket_name.to_string(),
                    ..Default::default()
                });
            entry.bucket_count += 1;
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            if bucket.constraint_count > entry.max_bucket_constraint_count {
                entry.max_bucket_constraint_count = bucket.constraint_count;
                entry.max_bucket_namespace = namespace.clone();
            }
        }
        let mut buckets = buckets.into_values().collect::<Vec<_>>();
        buckets.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
                .then_with(|| left.bucket.cmp(&right.bucket))
        });
        buckets
    }

    fn rho_rollup(&self) -> Vec<Rv64imMainRelationRhoBucket> {
        let mut buckets = BTreeMap::<String, Rv64imMainRelationRhoBucket>::new();
        for (namespace, bucket) in &self.leaf_buckets {
            let Some(bucket_name) = rho_bucket_name(namespace) else {
                continue;
            };
            let entry = buckets
                .entry(bucket_name.to_string())
                .or_insert_with(|| Rv64imMainRelationRhoBucket {
                    bucket: bucket_name.to_string(),
                    ..Default::default()
                });
            entry.bucket_count += 1;
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            if bucket.constraint_count > entry.max_bucket_constraint_count {
                entry.max_bucket_constraint_count = bucket.constraint_count;
                entry.max_bucket_namespace = namespace.clone();
            }
        }
        let mut buckets = buckets.into_values().collect::<Vec<_>>();
        buckets.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
                .then_with(|| left.bucket.cmp(&right.bucket))
        });
        buckets
    }

    fn family_component_rollup(&self) -> Vec<Rv64imMainRelationFamilyComponentBucket> {
        let mut grouped = BTreeMap::<(String, String, String), CountBucketAcc>::new();
        for (namespace, bucket) in &self.leaf_buckets {
            let Some(family) = bucket_family_name(namespace) else {
                continue;
            };
            let component = component_name(namespace).to_string();
            let canonical_namespace = component_bucket_namespace(namespace, &component);
            let entry = grouped
                .entry((family.to_string(), component, canonical_namespace))
                .or_default();
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            entry.max_constraint_term_count = entry
                .max_constraint_term_count
                .max(bucket.max_constraint_term_count);
        }

        let mut families = BTreeMap::<(String, String), Rv64imMainRelationFamilyComponentBucket>::new();
        for ((family, component, canonical_namespace), bucket) in grouped {
            let entry = families
                .entry((family.clone(), component.clone()))
                .or_insert_with(|| Rv64imMainRelationFamilyComponentBucket {
                    family,
                    component,
                    ..Rv64imMainRelationFamilyComponentBucket::default()
                });
            entry.bucket_count += 1;
            entry.aux_count += bucket.aux_count;
            entry.constraint_count += bucket.constraint_count;
            entry.total_term_count += bucket.total_term_count;
            if bucket.constraint_count > entry.max_bucket_constraint_count {
                entry.max_bucket_constraint_count = bucket.constraint_count;
                entry.max_bucket_namespace = canonical_namespace;
            }
        }

        let mut families = families.into_values().collect::<Vec<_>>();
        families.sort_by(|left, right| {
            right
                .constraint_count
                .cmp(&left.constraint_count)
                .then_with(|| right.total_term_count.cmp(&left.total_term_count))
                .then_with(|| left.family.cmp(&right.family))
        });
        families
    }

    fn claim_family_rollup(&self) -> Vec<Rv64imMainRelationClaimFamilyBucket> {
        let mut families = BTreeMap::<String, Rv64imMainRelationClaimFamilyBucket>::new();
        for (namespace, bucket) in &self.leaf_buckets {
            let Some(family) = claim_family_name(namespace) else {
                continue;
            };
            let entry = families
                .entry(family.to_string())
                .or_insert_with(|| Rv64imMainRelationClaimFamilyBucket {
                    family: family.to_string(),
                    ..Rv64imMainRelationClaimFamilyBucket::default()
                });
            if namespace.contains("_rlc_public") {
                entry.rlc_bucket_count += 1;
                entry.rlc_aux_count += bucket.aux_count;
                entry.rlc_constraint_count += bucket.constraint_count;
                entry.rlc_total_term_count += bucket.total_term_count;
                if bucket.constraint_count > entry.rlc_max_bucket_constraint_count {
                    entry.rlc_max_bucket_constraint_count = bucket.constraint_count;
                    entry.rlc_max_bucket_namespace = namespace.clone();
                }
            } else if namespace.contains("final_claim_eq_") {
                entry.final_bucket_count += 1;
                entry.final_aux_count += bucket.aux_count;
                entry.final_constraint_count += bucket.constraint_count;
                entry.final_total_term_count += bucket.total_term_count;
                if bucket.constraint_count > entry.final_max_bucket_constraint_count {
                    entry.final_max_bucket_constraint_count = bucket.constraint_count;
                    entry.final_max_bucket_namespace = namespace.clone();
                }
            }
        }
        let mut families = families.into_values().collect::<Vec<_>>();
        families.sort_by(|left, right| {
            (right.rlc_constraint_count + right.final_constraint_count)
                .cmp(&(left.rlc_constraint_count + left.final_constraint_count))
                .then_with(|| right.rlc_total_term_count.cmp(&left.rlc_total_term_count))
                .then_with(|| {
                    right
                        .final_total_term_count
                        .cmp(&left.final_total_term_count)
                })
                .then_with(|| left.family.cmp(&right.family))
        });
        families
    }

    fn max_claim_digest_constraint_count(&self) -> usize {
        self.buckets
            .iter()
            .filter(|(namespace, _)| is_claim_digest_namespace(namespace))
            .map(|(_, bucket)| bucket.constraint_count)
            .max()
            .unwrap_or(0)
    }

    fn max_claim_digest_namespace(&self) -> String {
        self.buckets
            .iter()
            .filter(|(namespace, _)| is_claim_digest_namespace(namespace))
            .max_by_key(|(_, bucket)| bucket.constraint_count)
            .map(|(namespace, _)| namespace.clone())
            .unwrap_or_else(|| "<none>".to_string())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CountBucketAcc {
    aux_count: usize,
    constraint_count: usize,
    total_term_count: usize,
    max_constraint_term_count: usize,
}

fn map_synth<T>(result: Result<T, SynthesisError>, context: &str) -> Result<T, SimpleKernelError> {
    result.map_err(|err| SimpleKernelError::Bridge(format!("{context}: {err}")))
}

fn is_claim_digest_namespace(namespace: &str) -> bool {
    is_ccs_output_digest_namespace(namespace)
        || is_child_digest_namespace(namespace)
        || is_parent_digest_namespace(namespace)
        || is_final_claim_digest_namespace(namespace)
}

fn is_ccs_output_digest_namespace(namespace: &str) -> bool {
    namespace.contains("_ccs_output_digest_")
}

fn is_parent_digest_namespace(namespace: &str) -> bool {
    namespace.contains("_parent_digest")
}

fn is_child_digest_namespace(namespace: &str) -> bool {
    namespace.contains("_child_digest_")
}

fn is_final_claim_digest_namespace(namespace: &str) -> bool {
    namespace.contains("final_claim_digest_")
}

fn phase_name(namespace: &str) -> &'static str {
    if namespace == "session_transcript" || namespace.contains("chunk_meta_") {
        "transcript_core"
    } else if namespace.contains("_carrier_outputs") {
        "carrier_outputs"
    } else if namespace.contains("_carrier_parent") {
        "carrier_parent"
    } else if namespace.contains("_carrier_children") {
        "carrier_children"
    } else if namespace.contains("_rlc_public") || namespace.contains("_mix_witness") {
        "rlc_fold"
    } else if namespace.contains("_dec_public") || namespace.contains("_dec_split") {
        "dec_fold"
    } else if namespace.contains("_ccs_output_ce_") {
        "ce_output"
    } else if namespace.contains("_child_witness_") {
        "ce_child"
    } else if namespace.contains("_parent_witness") {
        "ce_parent"
    } else if namespace.contains("_sample_challenges") {
        "challenge_sampling"
    } else if namespace.contains("_bind_header") || namespace.contains("_bind_me_inputs") {
        "transcript_bind"
    } else if namespace.contains("_initial_sum_") {
        "initial_sum"
    } else if namespace.contains("_fe_sumcheck") {
        "sumcheck_fe"
    } else if namespace.contains("_nc_sumcheck") {
        "sumcheck_nc"
    } else if namespace.contains("_terminal_fe") {
        "terminal_fe"
    } else if namespace.contains("_terminal_nc") {
        "terminal_nc"
    } else if namespace.contains("_output_binding") {
        "output_binding"
    } else if namespace.contains("final_claim_eq_") {
        "final_claim_binding"
    } else if namespace.contains("_ccs_output_") || namespace.contains("_child_") || namespace.contains("_parent") {
        "claim_alloc"
    } else if namespace.contains("_fold_digest") {
        "fold_digest"
    } else {
        "other"
    }
}

fn component_name(namespace: &str) -> &'static str {
    if namespace.ends_with("_digits") || namespace.contains("/digits/") {
        "digits"
    } else if namespace.ends_with("_commitment") || namespace.contains("/commitment/") {
        "commitment"
    } else if namespace.ends_with("_y_zcol") || namespace.contains("/y_zcol/") {
        "y_zcol"
    } else if namespace.contains("_y_ring_") || namespace.contains("/y_ring_") {
        "y_ring"
    } else if namespace.ends_with("_x_projection") || namespace.contains("/x_projection/") {
        "x_projection"
    } else if namespace.contains("_ct_") || namespace.contains("/ct_") {
        "ct"
    } else if namespace.contains("_chi_alpha_prime") {
        "chi_alpha_prime"
    } else if namespace.contains("_y_eval_") {
        "y_eval"
    } else if namespace.contains("_range_") {
        "range"
    } else if namespace.contains("_append_round_") {
        "sumcheck_append"
    } else if namespace.contains("_challenge_") {
        "sumcheck_challenge"
    } else if namespace.contains("_sample_challenges") {
        "sample_challenges"
    } else if namespace.contains("_bind_header") {
        "bind_header"
    } else if namespace.contains("_bind_me_inputs") {
        "bind_me_inputs"
    } else if namespace.contains("_gamma_step_") {
        "gamma_step"
    } else if namespace.contains("_weight_") || namespace.contains("/weighted_") {
        "weight"
    } else if namespace.contains("_acc_") {
        "accumulate"
    } else if namespace.contains("_eq_") {
        "eq_points"
    } else if namespace.contains("_rlc_public") {
        "rlc_public"
    } else if namespace.contains("_dec_public") {
        "dec_public"
    } else if namespace.contains("_dec_split") {
        "dec_split"
    } else {
        "other"
    }
}

fn bucket_family_name(namespace: &str) -> Option<&'static str> {
    if namespace.contains("_carrier_outputs") {
        Some("carrier_output")
    } else if namespace.contains("_carrier_children") {
        Some("carrier_child")
    } else if namespace.contains("_carrier_parent") {
        Some("carrier_parent")
    } else {
        claim_family_name(namespace)
    }
}

fn claim_family_name(namespace: &str) -> Option<&'static str> {
    if namespace.contains("_rlc_public") {
        rlc_public_family_name(namespace)
    } else if namespace.contains("final_claim_eq_") {
        final_claim_family_name(namespace)
    } else {
        None
    }
}

fn sumcheck_bucket_name(namespace: &str) -> Option<&'static str> {
    let lane = if namespace.contains("_fe_sumcheck") {
        "fe"
    } else if namespace.contains("_nc_sumcheck") {
        "nc"
    } else {
        return None;
    };

    let kind = if namespace.contains("_append_round_") {
        "append"
    } else if namespace.contains("_challenge_") {
        "challenge"
    } else if namespace.contains("_eval_") {
        "eval"
    } else if namespace.contains("_round_") {
        "round"
    } else if namespace.contains("_transcript_v3") {
        "transcript_v3"
    } else if namespace.contains("_initial") {
        "initial"
    } else if namespace.contains("_domain") {
        "domain"
    } else {
        return None;
    };

    Some(match (lane, kind) {
        ("fe", "append") => "fe_append",
        ("fe", "challenge") => "fe_challenge",
        ("fe", "eval") => "fe_eval",
        ("fe", "round") => "fe_round",
        ("fe", "transcript_v3") => "fe_transcript_v3",
        ("fe", "initial") => "fe_initial",
        ("fe", "domain") => "fe_domain",
        ("nc", "append") => "nc_append",
        ("nc", "challenge") => "nc_challenge",
        ("nc", "eval") => "nc_eval",
        ("nc", "round") => "nc_round",
        ("nc", "transcript_v3") => "nc_transcript_v3",
        ("nc", "initial") => "nc_initial",
        ("nc", "domain") => "nc_domain",
        _ => unreachable!(),
    })
}

fn rho_bucket_name(namespace: &str) -> Option<&'static str> {
    if !namespace.contains("_rlc_rhos") {
        return None;
    }

    if namespace.contains("_rho_index_") || namespace.contains("_rho_chunk_msg_") || namespace.contains("_rho_digest_")
    {
        Some("rho_transcript")
    } else if namespace.contains("_rho_words_") {
        Some("rho_words")
    } else if namespace.contains("_reject_bit")
        || namespace.contains("_reject_check")
        || namespace.contains("_popcount_low_bits")
    {
        Some("rho_reject")
    } else if namespace.contains("_quotient_bits")
        || namespace.contains("_remainder_bits")
        || namespace.contains("_quotient_range_")
        || namespace.contains("_remainder_range_")
        || namespace.contains("_mod5")
    {
        Some("rho_mod5")
    } else if namespace.contains("_coeff_alloc") || namespace.ends_with("_coeff") || namespace.contains("_coeff/") {
        Some("rho_coeff")
    } else if namespace.contains("_rho_accept_") {
        Some("rho_accept")
    } else {
        Some("rho_other")
    }
}

fn rlc_public_family_name(namespace: &str) -> Option<&'static str> {
    if namespace.contains("_rlc_public_ct_eq_") {
        Some("ct")
    } else if namespace.contains("_rlc_public_y_zcol") {
        Some("y_zcol")
    } else if namespace.contains("_rlc_public_y_") {
        Some("y_ring")
    } else if namespace.contains("_rlc_public_aux_") {
        Some("aux_openings")
    } else if namespace.contains("_rlc_public_s_col_") {
        Some("s_col")
    } else if namespace.contains("_rlc_public_r_") {
        Some("r")
    } else if namespace.contains("_rlc_public_x") {
        Some("x")
    } else if namespace.contains("_rlc_public_c") {
        Some("c_data")
    } else {
        None
    }
}

fn final_claim_family_name(namespace: &str) -> Option<&'static str> {
    if namespace.contains("_c_data_") || namespace.contains("/c_data_") || namespace.ends_with("/c_data") {
        Some("c_data")
    } else if namespace.contains("_c_step_coords_")
        || namespace.contains("/c_step_coords_")
        || namespace.ends_with("/c_step_coords")
    {
        Some("c_step_coords")
    } else if namespace.contains("_fold_digest") || namespace.contains("/fold_digest") {
        Some("fold_digest_encoding")
    } else if namespace.contains("_y_ring_") || namespace.contains("/y_ring_") {
        Some("y_ring")
    } else if namespace.contains("_y_zcol") || namespace.contains("/y_zcol") {
        Some("y_zcol")
    } else if namespace.contains("_aux_openings_")
        || namespace.contains("/aux_openings_")
        || namespace.ends_with("/aux_openings")
    {
        Some("aux_openings")
    } else if namespace.contains("_s_col_") || namespace.contains("/s_col_") || namespace.ends_with("/s_col") {
        Some("s_col")
    } else if namespace.contains("_ct_") || namespace.contains("/ct_") || namespace.ends_with("/ct") {
        Some("ct")
    } else if namespace.contains("_r_") || namespace.contains("/r_") || namespace.ends_with("/r") {
        Some("r")
    } else if namespace.contains("_x_") || namespace.contains("/x_") || namespace.ends_with("/x") {
        Some("x")
    } else {
        None
    }
}

fn component_bucket_namespace(namespace: &str, component: &str) -> String {
    namespace
        .split('/')
        .find(|segment| component_name(segment) == component)
        .unwrap_or(namespace)
        .to_string()
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for CountingCS<Scalar> {
    type Root = Self;

    fn new() -> Self {
        Self {
            num_inputs: 1,
            num_aux: 0,
            num_constraints: 0,
            linear_constraint_count: 0,
            quadratic_constraint_count: 0,
            a_term_count: 0,
            b_term_count: 0,
            c_term_count: 0,
            max_constraint_term_count: 0,
            namespaces: Vec::new(),
            buckets: BTreeMap::new(),
            leaf_buckets: BTreeMap::new(),
            _marker: core::marker::PhantomData,
        }
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = f()?;
        let bucket = self.current_bucket_key();
        let leaf_bucket = self.current_leaf_key();
        let idx = self.num_aux;
        self.num_aux += 1;
        self.buckets.entry(bucket).or_default().aux_count += 1;
        self.leaf_buckets.entry(leaf_bucket).or_default().aux_count += 1;
        Ok(Variable::new_unchecked(Index::Aux(idx)))
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = f()?;
        let idx = self.num_inputs;
        self.num_inputs += 1;
        Ok(Variable::new_unchecked(Index::Input(idx)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());
        let a_len = Self::compact_len(&a);
        let b_len = Self::compact_len(&b);
        let c_len = Self::compact_len(&c);
        let total_len = a_len + b_len + c_len;
        self.num_constraints += 1;
        if Self::is_unit_lc(&a) || Self::is_unit_lc(&b) {
            self.linear_constraint_count += 1;
        } else {
            self.quadratic_constraint_count += 1;
        }
        self.a_term_count += a_len;
        self.b_term_count += b_len;
        self.c_term_count += c_len;
        self.max_constraint_term_count = self.max_constraint_term_count.max(total_len);
        let bucket = self.buckets.entry(self.current_bucket_key()).or_default();
        bucket.constraint_count += 1;
        bucket.total_term_count += total_len;
        bucket.max_constraint_term_count = bucket.max_constraint_term_count.max(total_len);
        let leaf_bucket = self
            .leaf_buckets
            .entry(self.current_leaf_key())
            .or_default();
        leaf_bucket.constraint_count += 1;
        leaf_bucket.total_term_count += total_len;
        leaf_bucket.max_constraint_term_count = leaf_bucket.max_constraint_term_count.max(total_len);
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.namespaces.push(name_fn().into());
    }

    fn pop_namespace(&mut self) {
        let _ = self.namespaces.pop();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
