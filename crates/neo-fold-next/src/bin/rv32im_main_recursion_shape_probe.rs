use std::collections::HashMap;
use std::env;
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::Rv32imCeClaimDigestShape;
use neo_fold_next::rv32im::audit::{
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint,
    debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown,
    debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis,
    debug_measure_rv32im_main_recursion_step_stage_aux_counts,
    debug_measure_rv32im_terminal_f_prime_committed_step_shape,
    debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionStepSpartanShape, Rv32imNamedConstraintDelta,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices, debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint,
    prove_rv32im_accepted_proof_with_options_and_perf, Rv32imProofInput, Rv32imProofProvePerf,
    Rv32imPublicProofOptions,
};
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 2,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeMode {
    Full,
    FastSummary,
    StageAux,
    ConstraintBreakdown,
    TraceShape,
}

fn probe_mode_from_args() -> ProbeMode {
    let mut mode = ProbeMode::Full;
    for arg in env::args().skip(1) {
        if arg.starts_with("--relation-index=") {
            continue;
        }
        match arg.as_str() {
            "--fast-summary" => mode = ProbeMode::FastSummary,
            "--stage-aux" => mode = ProbeMode::StageAux,
            "--constraint-breakdown" => mode = ProbeMode::ConstraintBreakdown,
            "--trace-shape" => mode = ProbeMode::TraceShape,
            "--full" => mode = ProbeMode::Full,
            "--whole-trace" | "--rows-per-chunk-1" | "--last-relation" => {}
            other => panic!("unknown arg: {other}"),
        }
    }
    mode
}

fn root_fold_schedule_from_args() -> FoldSchedule {
    let mut schedule = FoldSchedule::WholeTrace;
    for arg in env::args().skip(1) {
        if arg.starts_with("--relation-index=") {
            continue;
        }
        match arg.as_str() {
            "--whole-trace" => schedule = FoldSchedule::WholeTrace,
            "--rows-per-chunk-1" => schedule = FoldSchedule::RowsPerChunk(1),
            "--fast-summary"
            | "--stage-aux"
            | "--constraint-breakdown"
            | "--trace-shape"
            | "--full"
            | "--last-relation" => {}
            other => panic!("unknown arg: {other}"),
        }
    }
    schedule
}

fn selected_relation_index_from_args(relation_count: usize) -> usize {
    let mut selected = 0usize;
    for arg in env::args().skip(1) {
        if arg == "--last-relation" {
            selected = relation_count.saturating_sub(1);
        } else if let Some(value) = arg.strip_prefix("--relation-index=") {
            selected = value
                .parse::<usize>()
                .expect("--relation-index requires a usize");
        }
    }
    if selected >= relation_count {
        panic!("selected relation index {selected} is out of range for {relation_count} backend relations");
    }
    selected
}

fn unwrap_accepted_artifact_with_schedule_context<T>(
    result: Result<T, impl std::fmt::Display>,
    root_fold_schedule: FoldSchedule,
    context: &str,
) -> T {
    result.unwrap_or_else(|err| match root_fold_schedule {
        FoldSchedule::WholeTrace => {
            panic!(
                "{context}: WholeTrace overflowed the live RV32IM DEC/k_rho budget; retry with --rows-per-chunk-1.\nunderlying error: {err}"
            )
        }
        FoldSchedule::RowsPerChunk(_) => panic!("{context}: {err}"),
    })
}

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:34} {}", label, value);
}

fn print_cumulative_and_delta(label: &str, previous: usize, current: usize) {
    let delta = current.saturating_sub(previous);
    print_kv(label, format!("{current} (+{delta})"));
}

fn print_named_constraint_breakdown(
    title: &str,
    measure_ms: f64,
    cover_round_lengths: &[u64],
    effective_round_lengths: &[usize],
    stages: &[Rv32imNamedConstraintDelta],
) {
    print_section(title);
    print_kv("measure.wall", format!("{measure_ms:.3} ms"));
    print_kv("cover_round_lengths", format!("{cover_round_lengths:?}"));
    print_kv("effective_round_lengths", format!("{effective_round_lengths:?}"));
    for stage in stages {
        print_kv(&stage.name, stage.delta);
    }
}

fn packed_bytes_field_count(byte_len: usize) -> usize {
    1 + byte_len.div_ceil(7)
}

fn per_unit(ms: f64, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ms / units as f64
    }
}

fn format_ms_per_named_unit(ms: f64, units: usize, unit_suffix: &str) -> String {
    format!("{ms:.3} ms ({:.4} ms/{unit_suffix})", per_unit(ms, units))
}

fn format_ms_per_opcode(ms: f64, opcode_count: usize) -> String {
    format_ms_per_named_unit(ms, opcode_count, "op")
}

fn projection_digest_field_count(
    c_data_len: usize,
    x_compact_len: usize,
    r_len: usize,
    y_ring_row_lens: &[usize],
) -> usize {
    let mut total = packed_bytes_field_count(b"neo/ccs/me_input_projection_digest_poseidon/v2".len());
    total += 1 + c_data_len;
    total += 1 + x_compact_len;
    total += 2 + (2 * r_len);
    total += 1;
    for row_len in y_ring_row_lens {
        total += 2 + (2 * row_len);
    }
    total
}

fn accumulator_phi_dec_parent_hash_field_count(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    let parent_commitment_fields = claims
        .first()
        .map(|claim| 1 + claim.c.data.len())
        .unwrap_or(0);
    packed_bytes_field_count(b"neo.fold.next/rv32im/main_recursion_recursive_accumulator_phi_dec_parent/v1".len())
        + 4
        + parent_commitment_fields
}

fn perturb_ce_claim_values(claim: &mut CeClaim<Commitment, F, K>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if claim.X.rows() > 0 && claim.X.cols() > 0 {
        claim.X[(0, 0)] += F::ONE;
    }
    if let Some(first) = claim.r.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.s_col.first_mut() {
        *first += K::ONE;
    }
    if let Some(row) = claim.y_ring.first_mut() {
        if let Some(first) = row.first_mut() {
            *first += K::ONE;
        }
    }
    if let Some(first) = claim.ct.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.aux_openings.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.y_zcol.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.c_step_coords.first_mut() {
        *first += F::ONE;
    }
    claim.fold_digest[0] ^= 1;
}

fn perturb_ccs_claim_values(claim: &mut CcsClaim<Commitment, F>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if let Some(first) = claim.x.first_mut() {
        *first += F::ONE;
    }
}

fn perturb_ccs_witness_values(witness: &mut CcsWitness<F>) {
    if let Some(first) = witness.w.first_mut() {
        *first += F::ONE;
    }
    if witness.Z.rows() > 0 && witness.Z.cols() > 0 {
        witness.Z[(0, 0)] += F::ONE;
    }
}

fn is_zero_f_slice(values: &[F]) -> bool {
    values.iter().all(|value| *value == F::ZERO)
}

fn count_zero_f_slice(values: &[F]) -> usize {
    values.iter().filter(|value| **value == F::ZERO).count()
}

fn count_zero_commitment_children(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    claims
        .iter()
        .filter(|claim| is_zero_f_slice(&claim.c.data))
        .count()
}

fn count_zero_commitment_words(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    claims
        .iter()
        .map(|claim| count_zero_f_slice(&claim.c.data))
        .sum::<usize>()
}

fn zero_commitment_indices(claims: &[CeClaim<Commitment, F, K>]) -> Vec<usize> {
    claims
        .iter()
        .enumerate()
        .filter_map(|(idx, claim)| is_zero_f_slice(&claim.c.data).then_some(idx))
        .collect()
}

fn is_zero_k_slice(values: &[K]) -> bool {
    values.iter().all(|value| *value == K::ZERO)
}

fn is_zero_y_ring(claim: &CeClaim<Commitment, F, K>) -> bool {
    claim.y_ring.iter().all(|row| is_zero_k_slice(row))
}

fn is_zero_ce_projection(claim: &CeClaim<Commitment, F, K>) -> bool {
    is_zero_f_slice(&claim.c.data)
        && claim.X.as_slice().iter().all(|value| *value == F::ZERO)
        && is_zero_k_slice(&claim.r)
        && claim.y_ring.iter().all(|row| is_zero_k_slice(row))
}

fn toom3_chunk_out_term_counts_current() -> Vec<usize> {
    vec![1; 2 * (D / 3) - 1]
}

fn toom3_chunk_out_term_counts_flattened() -> Vec<usize> {
    let split = D / 3;
    let mut counts = vec![0usize; 2 * split - 1];
    for i in 0..split {
        for j in 0..split {
            counts[i + j] += 1;
        }
    }
    counts
}

fn reduce_phi_81_term_counts(offset_chunk_counts: &[(usize, &[usize])]) -> Vec<usize> {
    let mut coeff_counts = vec![0usize; 2 * D - 1];
    for (offset, chunk_counts) in offset_chunk_counts {
        for (idx, count) in chunk_counts.iter().enumerate() {
            coeff_counts[offset + idx] += *count;
        }
    }
    for i in (D..(2 * D - 1)).rev() {
        let moved = coeff_counts[i];
        coeff_counts[i] = 0;
        coeff_counts[i - D] += moved;
        let idx_27 = i - 27;
        if idx_27 < D {
            coeff_counts[idx_27] += moved;
        } else {
            coeff_counts[idx_27 - D] += moved;
            if idx_27 - 27 < D {
                coeff_counts[idx_27 - 27] += moved;
            }
        }
    }
    coeff_counts.truncate(D);
    coeff_counts
}

fn toom3_reduced_product_term_counts(chunk_term_counts: &[usize]) -> Vec<usize> {
    let split = D / 3;
    let mut offsets = Vec::with_capacity(16);
    offsets.push((0, chunk_term_counts));
    for _ in 0..5 {
        offsets.push((split, chunk_term_counts));
    }
    for _ in 0..4 {
        offsets.push((2 * split, chunk_term_counts));
    }
    for _ in 0..5 {
        offsets.push((3 * split, chunk_term_counts));
    }
    offsets.push((4 * split, chunk_term_counts));
    reduce_phi_81_term_counts(&offsets)
}

fn add_probe_term(row_terms: &mut HashMap<(u8, usize), F>, term_id: (u8, usize), scale: F) {
    if scale == F::ZERO {
        return;
    }
    match row_terms.entry(term_id) {
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let updated = *entry.get() + scale;
            if updated == F::ZERO {
                let _ = entry.remove();
            } else {
                *entry.get_mut() = updated;
            }
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(scale);
        }
    }
}

fn reduce_phi_81_term_maps(offset_chunk_scales: &[(usize, u8, F)], chunk_len: usize) -> Vec<HashMap<(u8, usize), F>> {
    let mut coeff_terms = vec![HashMap::<(u8, usize), F>::new(); 2 * D - 1];
    for (offset, chunk_id, scale) in offset_chunk_scales {
        for idx in 0..chunk_len {
            add_probe_term(&mut coeff_terms[offset + idx], (*chunk_id, idx), *scale);
        }
    }
    for i in (D..(2 * D - 1)).rev() {
        let moved = std::mem::take(&mut coeff_terms[i]);
        for (term_id, scale) in moved {
            add_probe_term(&mut coeff_terms[i - D], term_id, -scale);
            let idx_27 = i - 27;
            if idx_27 < D {
                add_probe_term(&mut coeff_terms[idx_27], term_id, -scale);
            } else {
                add_probe_term(&mut coeff_terms[idx_27 - D], term_id, scale);
                if idx_27 - 27 < D {
                    add_probe_term(&mut coeff_terms[idx_27 - 27], term_id, scale);
                }
            }
        }
    }
    coeff_terms.truncate(D);
    coeff_terms
}

fn toom3_reduced_product_unique_term_counts_current() -> Vec<usize> {
    let split = D / 3;
    let half = F::from_u64(2).inverse();
    let third = F::from_u64(3).inverse();
    let sixth = F::from_u64(6).inverse();
    let term_maps = reduce_phi_81_term_maps(
        &[
            (0, 0, F::ONE),
            (split, 0, -half),
            (split, 1, F::ONE),
            (split, 2, -third),
            (split, 3, -sixth),
            (split, 4, F::from_u64(2)),
            (2 * split, 0, -F::ONE),
            (2 * split, 1, half),
            (2 * split, 2, half),
            (2 * split, 4, -F::ONE),
            (3 * split, 0, half),
            (3 * split, 1, -half),
            (3 * split, 2, -sixth),
            (3 * split, 3, sixth),
            (3 * split, 4, -F::from_u64(2)),
            (4 * split, 4, F::ONE),
        ],
        2 * split - 1,
    );
    term_maps.iter().map(|row| row.len()).collect()
}

fn print_pi_rlc_public_child_families(
    first_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    fresh_child_count: usize,
    actual_child_count: usize,
    pi_rlc_parent_shape: &Rv32imCeClaimDigestShape,
) {
    let schoolbook_dense_ring_mul_cost = D * D;
    let toom3_54_dense_ring_mul_cost = 5 * (D / 3) * (D / 3);
    let recursive_field_toom3_54_dense_ring_mul_cost = 5 * 5 * (D / 9) * (D / 9);
    let recursive_k_toom3_54_dense_ring_mul_cost = 5 * 5 * (D / 9) * (D / 9);
    let carried_child_count = actual_child_count.saturating_sub(fresh_child_count);
    let parent_c_data_len = usize::try_from(pi_rlc_parent_shape.c_data_len).expect("parent c_data len");
    let parent_x_embedded_len = D * first_relation.payload.pi_rlc.parent.m_in;
    let modeled_c_constraints_fresh = fresh_child_count * parent_c_data_len * D;
    let modeled_c_constraints_carried = carried_child_count * parent_c_data_len * D;
    let modeled_c_constraints_target_eq = parent_c_data_len;
    let parent_commitment_cols = parent_c_data_len / D;
    let modeled_x_constraints_fresh = fresh_child_count * parent_x_embedded_len;
    let modeled_x_constraints_carried = carried_child_count * parent_x_embedded_len;
    let modeled_x_constraints_target_eq = parent_x_embedded_len;
    let parent_y_ring_entries = first_relation
        .payload
        .pi_rlc
        .parent
        .y_ring
        .iter()
        .map(|row| row.len())
        .sum::<usize>();
    let modeled_y_ring_constraints_per_child = pi_rlc_parent_shape.y_ring_row_count as usize * D * D * 2;
    let modeled_y_ring_constraints_fresh = fresh_child_count * modeled_y_ring_constraints_per_child;
    let modeled_y_ring_constraints_carried = carried_child_count * modeled_y_ring_constraints_per_child;
    let modeled_y_ring_constraints_target_eq = parent_y_ring_entries * 2;
    let fresh_zero_children = first_relation
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .take(fresh_child_count)
        .filter(|claim| is_zero_ce_projection(claim))
        .count();
    let carried_zero_children = first_relation
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .skip(fresh_child_count)
        .take(carried_child_count)
        .filter(|claim| is_zero_ce_projection(claim))
        .count();
    let fresh_nonzero_children = fresh_child_count.saturating_sub(fresh_zero_children);
    let carried_nonzero_children = carried_child_count.saturating_sub(carried_zero_children);
    let fresh_claims = &first_relation.payload.pi_ccs.ccs_outputs[..fresh_child_count];
    let carried_claims =
        &first_relation.payload.pi_ccs.ccs_outputs[fresh_child_count..fresh_child_count + carried_child_count];
    let fresh_commitment_zero_children = fresh_claims
        .iter()
        .filter(|claim| is_zero_f_slice(&claim.c.data))
        .count();
    let carried_commitment_zero_children = carried_claims
        .iter()
        .filter(|claim| is_zero_f_slice(&claim.c.data))
        .count();
    let fresh_y_ring_zero_children = fresh_claims
        .iter()
        .filter(|claim| is_zero_y_ring(claim))
        .count();
    let carried_y_ring_zero_children = carried_claims
        .iter()
        .filter(|claim| is_zero_y_ring(claim))
        .count();
    let fresh_c_data_zero_words = fresh_claims
        .iter()
        .map(|claim| count_zero_f_slice(&claim.c.data))
        .sum::<usize>();
    let carried_c_data_zero_words = carried_claims
        .iter()
        .map(|claim| count_zero_f_slice(&claim.c.data))
        .sum::<usize>();
    let fresh_c_data_words = fresh_child_count * parent_c_data_len;
    let carried_c_data_words = carried_child_count * parent_c_data_len;
    let sparse_modeled_c_constraints_total =
        (fresh_nonzero_children + carried_nonzero_children) * parent_c_data_len * D + modeled_c_constraints_target_eq;
    let commitment_sparse_modeled_c_constraints_total = (fresh_child_count
        .saturating_sub(fresh_commitment_zero_children)
        + carried_child_count.saturating_sub(carried_commitment_zero_children))
        * parent_c_data_len
        * D
        + modeled_c_constraints_target_eq;
    let sparse_modeled_x_constraints_total =
        (fresh_nonzero_children + carried_nonzero_children) * parent_x_embedded_len + modeled_x_constraints_target_eq;
    let sparse_modeled_y_ring_constraints_total = (fresh_nonzero_children + carried_nonzero_children)
        * modeled_y_ring_constraints_per_child
        + modeled_y_ring_constraints_target_eq;
    let y_ring_sparse_modeled_total = (fresh_child_count.saturating_sub(fresh_y_ring_zero_children)
        + carried_child_count.saturating_sub(carried_y_ring_zero_children))
        * modeled_y_ring_constraints_per_child
        + modeled_y_ring_constraints_target_eq;
    let dense_c_ring_products = (fresh_child_count + carried_child_count) * parent_commitment_cols;
    let nonzero_commit_c_ring_products = (fresh_child_count.saturating_sub(fresh_commitment_zero_children)
        + carried_child_count.saturating_sub(carried_commitment_zero_children))
        * parent_commitment_cols;
    let dense_y_ring_products =
        (fresh_child_count + carried_child_count) * pi_rlc_parent_shape.y_ring_row_count as usize;
    let toom3_modeled_c_constraints_total =
        dense_c_ring_products * toom3_54_dense_ring_mul_cost + modeled_c_constraints_target_eq;
    let recursive_field_toom3_zero_commit_c_constraints_total =
        nonzero_commit_c_ring_products * recursive_field_toom3_54_dense_ring_mul_cost + modeled_c_constraints_target_eq;
    let toom3_modeled_y_ring_constraints_total =
        dense_y_ring_products * toom3_54_dense_ring_mul_cost * 2 + modeled_y_ring_constraints_target_eq;
    let recursive_k_toom3_modeled_y_ring_constraints_total =
        dense_y_ring_products * recursive_k_toom3_54_dense_ring_mul_cost * 2 + modeled_y_ring_constraints_target_eq;
    let recursive_k_toom3_zero_y_ring_constraints_total = (fresh_child_count
        .saturating_sub(fresh_y_ring_zero_children)
        + carried_child_count.saturating_sub(carried_y_ring_zero_children))
        * pi_rlc_parent_shape.y_ring_row_count as usize
        * recursive_k_toom3_54_dense_ring_mul_cost
        * 2
        + modeled_y_ring_constraints_target_eq;
    let toom3_current_product_term_counts = toom3_reduced_product_term_counts(&toom3_chunk_out_term_counts_current());
    let toom3_flattened_product_term_counts =
        toom3_reduced_product_term_counts(&toom3_chunk_out_term_counts_flattened());
    let toom3_current_unique_product_term_counts = toom3_reduced_product_unique_term_counts_current();
    let toom3_current_product_term_total = toom3_current_product_term_counts.iter().sum::<usize>();
    let toom3_flattened_product_term_total = toom3_flattened_product_term_counts.iter().sum::<usize>();
    let toom3_current_unique_product_term_total = toom3_current_unique_product_term_counts
        .iter()
        .sum::<usize>();
    let toom3_current_product_term_max = toom3_current_product_term_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let toom3_flattened_product_term_max = toom3_flattened_product_term_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let toom3_current_unique_product_term_max = toom3_current_unique_product_term_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let toom3_current_product_term_avg = toom3_current_product_term_total as f64 / D as f64;
    let toom3_flattened_product_term_avg = toom3_flattened_product_term_total as f64 / D as f64;
    let toom3_current_unique_product_term_avg = toom3_current_unique_product_term_total as f64 / D as f64;

    print_section("Pi RLC Public Child Families");
    print_kv("dense_ring_mul_cost_schoolbook_54", schoolbook_dense_ring_mul_cost);
    print_kv("dense_ring_mul_cost_toom3_54", toom3_54_dense_ring_mul_cost);
    print_kv(
        "dense_ring_mul_cost_recursive_field_toom3_54",
        recursive_field_toom3_54_dense_ring_mul_cost,
    );
    print_kv(
        "dense_ring_mul_cost_recursive_k_toom3_54",
        recursive_k_toom3_54_dense_ring_mul_cost,
    );
    print_kv("fresh_children", fresh_child_count);
    print_kv("fresh_projection_zero_children", fresh_zero_children);
    print_kv("fresh_projection_nonzero_children", fresh_nonzero_children);
    print_kv("fresh_commitment_zero_children", fresh_commitment_zero_children);
    print_kv("fresh_y_ring_zero_children", fresh_y_ring_zero_children);
    print_kv(
        "fresh_c_data_zero_words",
        format!(
            "{}/{} ({:.1}%)",
            fresh_c_data_zero_words,
            fresh_c_data_words,
            if fresh_c_data_words == 0 {
                0.0
            } else {
                100.0 * fresh_c_data_zero_words as f64 / fresh_c_data_words as f64
            }
        ),
    );
    print_kv("carried_children", carried_child_count);
    print_kv("carried_projection_zero_children", carried_zero_children);
    print_kv("carried_projection_nonzero_children", carried_nonzero_children);
    print_kv("carried_commitment_zero_children", carried_commitment_zero_children);
    print_kv("carried_y_ring_zero_children", carried_y_ring_zero_children);
    print_kv(
        "carried_c_data_zero_words",
        format!(
            "{}/{} ({:.1}%)",
            carried_c_data_zero_words,
            carried_c_data_words,
            if carried_c_data_words == 0 {
                0.0
            } else {
                100.0 * carried_c_data_zero_words as f64 / carried_c_data_words as f64
            }
        ),
    );
    print_kv("c_constraints_fresh_muls", modeled_c_constraints_fresh);
    print_kv("c_constraints_carried_muls", modeled_c_constraints_carried);
    print_kv("c_constraints_target_eq", modeled_c_constraints_target_eq);
    print_kv(
        "c_constraints_modeled_total",
        modeled_c_constraints_fresh + modeled_c_constraints_carried + modeled_c_constraints_target_eq,
    );
    print_kv(
        "c_constraints_modeled_total_if_zero_children_skipped",
        sparse_modeled_c_constraints_total,
    );
    print_kv(
        "c_constraints_modeled_total_if_zero_commit_children_skipped",
        commitment_sparse_modeled_c_constraints_total,
    );
    print_kv(
        "c_constraints_modeled_total_if_toom3_54_used",
        toom3_modeled_c_constraints_total,
    );
    print_kv(
        "c_constraints_modeled_total_if_zero_commit_children_skipped_and_recursive_field_toom3_54_used",
        recursive_field_toom3_zero_commit_c_constraints_total,
    );
    print_kv(
        "toom3_current_lc_terms_per_dense_product_total",
        toom3_current_product_term_total,
    );
    print_kv(
        "toom3_current_lc_terms_per_dense_product_avg_row",
        format!("{toom3_current_product_term_avg:.2}"),
    );
    print_kv(
        "toom3_current_lc_terms_per_dense_product_max_row",
        toom3_current_product_term_max,
    );
    print_kv(
        "toom3_current_unique_lc_terms_per_dense_product_total",
        toom3_current_unique_product_term_total,
    );
    print_kv(
        "toom3_current_unique_lc_terms_per_dense_product_avg_row",
        format!("{toom3_current_unique_product_term_avg:.2}"),
    );
    print_kv(
        "toom3_current_unique_lc_terms_per_dense_product_max_row",
        toom3_current_unique_product_term_max,
    );
    print_kv(
        "toom3_flattened_lc_terms_per_dense_product_total",
        toom3_flattened_product_term_total,
    );
    print_kv(
        "toom3_flattened_lc_terms_per_dense_product_avg_row",
        format!("{toom3_flattened_product_term_avg:.2}"),
    );
    print_kv(
        "toom3_flattened_lc_terms_per_dense_product_max_row",
        toom3_flattened_product_term_max,
    );
    print_kv(
        "c_lc_terms_modeled_total_current_toom3",
        dense_c_ring_products * toom3_current_product_term_total,
    );
    print_kv(
        "c_lc_terms_modeled_total_current_toom3_unique",
        dense_c_ring_products * toom3_current_unique_product_term_total,
    );
    print_kv(
        "c_lc_terms_modeled_total_if_flattened",
        dense_c_ring_products * toom3_flattened_product_term_total,
    );
    print_kv("x_constraints_fresh_muls", modeled_x_constraints_fresh);
    print_kv("x_constraints_carried_muls", modeled_x_constraints_carried);
    print_kv("x_constraints_target_eq", modeled_x_constraints_target_eq);
    print_kv(
        "x_constraints_modeled_total",
        modeled_x_constraints_fresh + modeled_x_constraints_carried + modeled_x_constraints_target_eq,
    );
    print_kv(
        "x_constraints_modeled_total_if_zero_children_skipped",
        sparse_modeled_x_constraints_total,
    );
    print_kv("y_ring_constraints_fresh_muls", modeled_y_ring_constraints_fresh);
    print_kv("y_ring_constraints_carried_muls", modeled_y_ring_constraints_carried);
    print_kv("y_ring_constraints_target_eq", modeled_y_ring_constraints_target_eq);
    print_kv(
        "y_ring_constraints_modeled_total",
        modeled_y_ring_constraints_fresh + modeled_y_ring_constraints_carried + modeled_y_ring_constraints_target_eq,
    );
    print_kv(
        "y_ring_constraints_modeled_total_if_zero_children_skipped",
        sparse_modeled_y_ring_constraints_total,
    );
    print_kv(
        "y_ring_constraints_modeled_total_if_zero_y_ring_children_skipped",
        y_ring_sparse_modeled_total,
    );
    print_kv(
        "y_ring_constraints_modeled_total_if_toom3_54_used",
        toom3_modeled_y_ring_constraints_total,
    );
    print_kv(
        "y_ring_constraints_modeled_total_if_recursive_k_toom3_54_used",
        recursive_k_toom3_modeled_y_ring_constraints_total,
    );
    print_kv(
        "y_ring_constraints_modeled_total_if_zero_y_ring_children_skipped_and_recursive_k_toom3_54_used",
        recursive_k_toom3_zero_y_ring_constraints_total,
    );
    print_kv(
        "y_ring_lc_terms_modeled_total_current_toom3",
        dense_y_ring_products * toom3_current_product_term_total * 2,
    );
    print_kv(
        "y_ring_lc_terms_modeled_total_current_toom3_unique",
        dense_y_ring_products * toom3_current_unique_product_term_total * 2,
    );
    print_kv(
        "y_ring_lc_terms_modeled_total_if_flattened",
        dense_y_ring_products * toom3_flattened_product_term_total * 2,
    );
}

fn print_backend_relation_commitment_sparsity(relations: &[Rv32imMainRecursionFPrimeBackendRelation]) {
    print_section("Backend Relation Commitment Sparsity");
    for (relation_index, relation) in relations.iter().enumerate() {
        let fresh_output_count = relation.payload.fresh_claims.len();
        let ccs_outputs = &relation.payload.pi_ccs.ccs_outputs;
        let fresh_outputs = &ccs_outputs[..fresh_output_count.min(ccs_outputs.len())];
        let carried_outputs = &ccs_outputs[fresh_output_count.min(ccs_outputs.len())..];
        let state_in_words = relation
            .payload
            .state_in_claims
            .iter()
            .map(|claim| claim.c.data.len())
            .sum::<usize>();
        let fresh_output_words = fresh_outputs
            .iter()
            .map(|claim| claim.c.data.len())
            .sum::<usize>();
        let carried_output_words = carried_outputs
            .iter()
            .map(|claim| claim.c.data.len())
            .sum::<usize>();
        let dec_child_words = relation
            .payload
            .pi_dec
            .children
            .iter()
            .map(|claim| claim.c.data.len())
            .sum::<usize>();
        let state_in_zero_indices = zero_commitment_indices(&relation.payload.state_in_claims);
        let fresh_output_zero_indices = zero_commitment_indices(fresh_outputs);
        let carried_output_zero_indices = zero_commitment_indices(carried_outputs);
        let dec_child_zero_indices = zero_commitment_indices(&relation.payload.pi_dec.children);
        print_kv(
            &format!("relation_{relation_index}"),
            format!(
                "state_in zero_commit_children={}/{} zero_words={}/{} | \
fresh_outputs zero_commit_children={}/{} zero_words={}/{} | \
carried_outputs zero_commit_children={}/{} zero_words={}/{} | \
dec_children zero_commit_children={}/{} zero_words={}/{}",
                count_zero_commitment_children(&relation.payload.state_in_claims),
                relation.payload.state_in_claims.len(),
                count_zero_commitment_words(&relation.payload.state_in_claims),
                state_in_words,
                count_zero_commitment_children(fresh_outputs),
                fresh_outputs.len(),
                count_zero_commitment_words(fresh_outputs),
                fresh_output_words,
                count_zero_commitment_children(carried_outputs),
                carried_outputs.len(),
                count_zero_commitment_words(carried_outputs),
                carried_output_words,
                count_zero_commitment_children(&relation.payload.pi_dec.children),
                relation.payload.pi_dec.children.len(),
                count_zero_commitment_words(&relation.payload.pi_dec.children),
                dec_child_words,
            ),
        );
        print_kv(
            &format!("relation_{relation_index}.zero_indices"),
            format!(
                "state_in={state_in_zero_indices:?} fresh_outputs={fresh_output_zero_indices:?} carried_outputs={carried_output_zero_indices:?} dec_children={dec_child_zero_indices:?}"
            ),
        );
    }
}

fn perturb_backend_relation_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.state_out_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.pi_ccs.ccs_outputs {
        perturb_ce_claim_values(claim);
    }
    perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
    for child in &mut relation.payload.pi_dec.children {
        perturb_ce_claim_values(child);
    }
    for claim in &mut relation.payload.fresh_claims {
        perturb_ccs_claim_values(claim);
    }
    for witness in &mut relation.payload.fresh_witnesses {
        perturb_ccs_witness_values(witness);
    }
}

fn perturb_state_in_r_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(first) = claim.r.first_mut() {
            *first += K::ONE;
        }
    }
}

fn perturb_state_in_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

fn perturb_state_in_projection_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.state_in_claims.first_mut() {
        if let Some(first) = claim.c.data.first_mut() {
            *first += F::ONE;
        }
        if claim.X.rows() > 0 && claim.X.cols() > 0 {
            claim.X[(0, 0)] += F::ONE;
        }
    }
}

fn perturb_pi_ccs_alpha_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(first) = relation.payload.pi_ccs.public_challenges.alpha.first_mut() {
        *first += K::ONE;
    }
}

fn perturb_pi_ccs_gamma_value(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    relation.payload.pi_ccs.public_challenges.gamma += K::ONE;
}

fn perturb_state_out_projection_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.state_out_claims.first_mut() {
        if let Some(first) = claim.c.data.first_mut() {
            *first += F::ONE;
        }
        if claim.X.rows() > 0 && claim.X.cols() > 0 {
            claim.X[(0, 0)] += F::ONE;
        }
        if let Some(first) = claim.r.first_mut() {
            *first += K::ONE;
        }
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

fn perturb_pi_ccs_output_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_ccs.ccs_outputs.first_mut() {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

fn perturb_pi_ccs_output_y_zcol_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_ccs.ccs_outputs.first_mut() {
        if let Some(first) = claim.y_zcol.first_mut() {
            *first += K::ONE;
        }
    }
}

fn perturb_pi_dec_child_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_dec.children.first_mut() {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

fn perturb_pi_rlc_parent_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
}

fn perturb_fresh_claim_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.fresh_claims.first_mut() {
        perturb_ccs_claim_values(claim);
    }
}

fn perturb_fresh_witness_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(witness) = relation.payload.fresh_witnesses.first_mut() {
        perturb_ccs_witness_values(witness);
    }
}

fn fixed_shape_family_status(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    baseline_fingerprint: &str,
    relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> &'static str {
    match debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(spartan_shape, relation) {
        Ok(shape) => {
            if shape.constraint_fingerprint == baseline_fingerprint {
                "stable"
            } else {
                "drift"
            }
        }
        Err(err) if err.to_string().contains("unsatisfiable") => "unsat",
        Err(_) => "error",
    }
}

fn first_pi_ccs_stage_diff<'a>(
    baseline: &'a neo_fold_next::rv32im::audit::Rv32imPiCcsStageFingerprint,
    perturbed: &'a neo_fold_next::rv32im::audit::Rv32imPiCcsStageFingerprint,
) -> Option<(&'static str, &'a str, &'a str)> {
    let stages = [
        (
            "after_bind_header",
            &baseline.after_bind_header,
            &perturbed.after_bind_header,
        ),
        (
            "after_bind_me_inputs",
            &baseline.after_bind_me_inputs,
            &perturbed.after_bind_me_inputs,
        ),
        (
            "after_sample_challenges",
            &baseline.after_sample_challenges,
            &perturbed.after_sample_challenges,
        ),
        (
            "after_alloc_fresh_claims",
            &baseline.after_alloc_fresh_claims,
            &perturbed.after_alloc_fresh_claims,
        ),
        (
            "after_fe_sumcheck",
            &baseline.after_fe_sumcheck,
            &perturbed.after_fe_sumcheck,
        ),
        (
            "after_nc_sumcheck",
            &baseline.after_nc_sumcheck,
            &perturbed.after_nc_sumcheck,
        ),
        (
            "after_fold_digest",
            &baseline.after_fold_digest,
            &perturbed.after_fold_digest,
        ),
        (
            "after_alloc_outputs",
            &baseline.after_alloc_outputs,
            &perturbed.after_alloc_outputs,
        ),
        (
            "after_output_binding",
            &baseline.after_output_binding,
            &perturbed.after_output_binding,
        ),
        (
            "after_terminal_fe",
            &baseline.after_terminal_fe,
            &perturbed.after_terminal_fe,
        ),
        (
            "after_terminal_nc",
            &baseline.after_terminal_nc,
            &perturbed.after_terminal_nc,
        ),
    ];
    stages.into_iter().find_map(|(name, lhs, rhs)| {
        if lhs != rhs {
            Some((name, lhs.as_str(), rhs.as_str()))
        } else {
            None
        }
    })
}

#[derive(Clone, Debug)]
struct FastSummaryPerf {
    fixture_ms: f64,
    accepted_wall_ms: f64,
    accepted_perf: Rv32imProofProvePerf,
    final_statement_ms: f64,
    relations_ms: f64,
    advices_ms: f64,
    backend_relations_ms: f64,
    shape_only_ms: f64,
}

#[derive(Clone, Copy, Debug)]
struct ProbeWorkUnits {
    non_halt_opcode_count: usize,
    semantic_step_count: usize,
    chunk_count: usize,
    chunk_fold_step_count: usize,
    relation_count: usize,
    backend_relation_count: usize,
    fold_schedule: FoldSchedule,
}

fn accepted_root_rlc_dec_ms(accepted_perf: &Rv32imProofProvePerf) -> f64 {
    accepted_perf.root_main_lane.session.rlc_ms()
        + accepted_perf.root_main_lane.session.dec_split_ms()
        + accepted_perf.root_main_lane.session.dec_commit_ms()
        + accepted_perf.root_main_lane.session.dec_ms()
}

fn print_probe_work_units(title: &str, units: ProbeWorkUnits) {
    print_section(title);
    print_kv("non_halt_opcode_count", units.non_halt_opcode_count);
    print_kv("semantic_step_count", units.semantic_step_count);
    print_kv("chunk_count", units.chunk_count);
    print_kv("chunk_fold_step_count", units.chunk_fold_step_count);
    print_kv("relation_count", units.relation_count);
    print_kv("backend_relation_count", units.backend_relation_count);
    print_kv("fold_schedule", format!("{:?}", units.fold_schedule));
}

fn print_key_per_opcode_summary(title: &str, summary: &FastSummaryPerf, opcode_count: usize) {
    print_section(title);
    let prove_only_wall_ms = summary.accepted_wall_ms;
    let fold_only_ms = summary.final_statement_ms;
    let prove_plus_fold_wall_ms = prove_only_wall_ms + fold_only_ms;
    let recursive_setup_ms = summary.relations_ms + summary.advices_ms + summary.backend_relations_ms;
    let fast_summary_total_ms = summary.fixture_ms + summary.shape_only_ms;
    print_kv(
        "prove_only_native",
        format_ms_per_opcode(prove_only_wall_ms, opcode_count),
    );
    print_kv("fold_only_native", format_ms_per_opcode(fold_only_ms, opcode_count));
    print_kv(
        "prove_plus_fold_native",
        format_ms_per_opcode(prove_plus_fold_wall_ms, opcode_count),
    );
    print_kv(
        "recursive_setup",
        format_ms_per_opcode(recursive_setup_ms, opcode_count),
    );
    print_kv("shape_only", format_ms_per_opcode(summary.shape_only_ms, opcode_count));
    print_kv(
        "fast_summary_total",
        format_ms_per_opcode(fast_summary_total_ms, opcode_count),
    );
}

fn print_key_per_fold_summary(title: &str, summary: &FastSummaryPerf, fold_count: usize) {
    print_section(title);
    let prove_only_wall_ms = summary.accepted_wall_ms;
    let fold_only_ms = summary.final_statement_ms;
    let prove_plus_fold_wall_ms = prove_only_wall_ms + fold_only_ms;
    let recursive_setup_ms = summary.relations_ms + summary.advices_ms + summary.backend_relations_ms;
    let fast_summary_total_ms = summary.fixture_ms + summary.shape_only_ms;
    print_kv(
        "prove_only_native",
        format_ms_per_named_unit(prove_only_wall_ms, fold_count, "fold"),
    );
    print_kv(
        "fold_only_native",
        format_ms_per_named_unit(fold_only_ms, fold_count, "fold"),
    );
    print_kv(
        "prove_plus_fold_native",
        format_ms_per_named_unit(prove_plus_fold_wall_ms, fold_count, "fold"),
    );
    print_kv(
        "recursive_setup",
        format_ms_per_named_unit(recursive_setup_ms, fold_count, "fold"),
    );
    print_kv(
        "shape_only",
        format_ms_per_named_unit(summary.shape_only_ms, fold_count, "fold"),
    );
    print_kv(
        "fast_summary_total",
        format_ms_per_named_unit(fast_summary_total_ms, fold_count, "fold"),
    );
}

fn print_per_opcode_components(title: &str, summary: &FastSummaryPerf, opcode_count: usize) {
    print_section(title);
    print_kv(
        "accepted_proof.wall",
        format_ms_per_opcode(summary.accepted_wall_ms, opcode_count),
    );
    print_kv(
        "accepted_proof",
        format_ms_per_opcode(summary.accepted_perf.total_ms, opcode_count),
    );
    print_kv(
        "accepted_root_session",
        format_ms_per_opcode(summary.accepted_perf.root_main_lane.session.total_ms, opcode_count),
    );
    print_kv(
        "accepted_root_rlc_dec",
        format_ms_per_opcode(accepted_root_rlc_dec_ms(&summary.accepted_perf), opcode_count),
    );
    print_kv(
        "accepted_root_ccs",
        format_ms_per_opcode(summary.accepted_perf.root_main_lane.session.ccs_ms(), opcode_count),
    );
    print_kv(
        "final_statement",
        format_ms_per_opcode(summary.final_statement_ms, opcode_count),
    );
    print_kv(
        "build_relations",
        format_ms_per_opcode(summary.relations_ms, opcode_count),
    );
    print_kv("build_advices", format_ms_per_opcode(summary.advices_ms, opcode_count));
    print_kv(
        "build_backend_relations",
        format_ms_per_opcode(summary.backend_relations_ms, opcode_count),
    );
    print_kv("shape_only", format_ms_per_opcode(summary.shape_only_ms, opcode_count));
    print_kv("fixture_prep", format_ms_per_opcode(summary.fixture_ms, opcode_count));
}

fn measure_fast_summary_perf(input: &Rv32imProofInput) -> FastSummaryPerf {
    let fixture_started = Instant::now();
    let root_fold_schedule = root_fold_schedule_from_args();

    let accepted_started = Instant::now();
    let ((accepted, _), accepted_perf) = unwrap_accepted_artifact_with_schedule_context(
        prove_rv32im_accepted_proof_with_options_and_perf(input, Rv32imPublicProofOptions { root_fold_schedule }),
        root_fold_schedule,
        "prove accepted artifact for fast summary rerun",
    );
    let accepted_wall_ms = millis_since(accepted_started);

    let final_statement_started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement for fast summary rerun");
    let final_statement_ms = millis_since(final_statement_started);

    let relations_started = Instant::now();
    let relations = build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for fast summary rerun");
    let relations_ms = millis_since(relations_started);

    let advices_started = Instant::now();
    let advices =
        build_rv32im_main_recursion_f_prime_advices(&relations).expect("build f-prime advices for fast summary rerun");
    let advices_ms = millis_since(advices_started);

    let backend_relations_started = Instant::now();
    let (spartan_shape, _) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build recursion backend relations for fast summary rerun");
    let backend_relations_ms = millis_since(backend_relations_started);
    let fixture_ms = millis_since(fixture_started);

    let shape_only_started = Instant::now();
    let _ = debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape(&spartan_shape)
        .expect("measure shape-only circuit for fast summary rerun");
    let shape_only_ms = millis_since(shape_only_started);

    FastSummaryPerf {
        fixture_ms,
        accepted_wall_ms,
        accepted_perf,
        final_statement_ms,
        relations_ms,
        advices_ms,
        backend_relations_ms,
        shape_only_ms,
    }
}

fn main() {
    let probe_mode = probe_mode_from_args();
    let root_fold_schedule = root_fold_schedule_from_args();
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let input = Rv32imProofInput {
        max_steps: source.program_words.len(),
        source,
    };

    let fixture_started = Instant::now();
    let accepted_started = Instant::now();
    let ((accepted, _), accepted_perf) = unwrap_accepted_artifact_with_schedule_context(
        prove_rv32im_accepted_proof_with_options_and_perf(&input, Rv32imPublicProofOptions { root_fold_schedule }),
        root_fold_schedule,
        "prove accepted artifact",
    );
    let accepted_ms = millis_since(accepted_started);

    let final_statement_started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement");
    let final_statement_ms = millis_since(final_statement_started);

    let relations_started = Instant::now();
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations");
    let relations_ms = millis_since(relations_started);

    let advices_started = Instant::now();
    let advices = build_rv32im_main_recursion_f_prime_advices(&relations).expect("build f-prime advices");
    let advices_ms = millis_since(advices_started);

    let backend_relations_started = Instant::now();
    let (spartan_shape, backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build recursion backend relations");
    let backend_relations_ms = millis_since(backend_relations_started);
    let fixture_ms = millis_since(fixture_started);

    let selected_relation_index = selected_relation_index_from_args(backend_relations.len());
    let first_relation = backend_relations
        .get(selected_relation_index)
        .expect("shape probe requires the selected backend relation");
    let terminal_relation_index = backend_relations.len().saturating_sub(1);
    let terminal_relation = backend_relations
        .get(terminal_relation_index)
        .expect("shape probe requires a terminal backend relation");

    let shape_only_started = Instant::now();
    let shape_only = debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape(&spartan_shape);
    let shape_only_ms = millis_since(shape_only_started);
    let terminal_committed_shape = if probe_mode == ProbeMode::Full {
        let terminal_committed_started = Instant::now();
        let shape = debug_measure_rv32im_terminal_f_prime_committed_step_shape(&spartan_shape, terminal_relation)
            .expect("measure terminal F' committed-step shape");
        Some((shape, millis_since(terminal_committed_started)))
    } else {
        None
    };

    let step_shape = &first_relation.payload.step_shape;
    let cover_shape = &first_relation.payload.cover_shape;
    let state_in_claim_shape = first_relation
        .payload
        .state_in_claims
        .first()
        .map(Rv32imCeClaimDigestShape::from_claim)
        .expect("state-in claim shape");
    let state_out_claim_shape = first_relation
        .payload
        .state_out_claims
        .first()
        .map(Rv32imCeClaimDigestShape::from_claim)
        .expect("state-out claim shape");
    let pi_rlc_parent_shape = Rv32imCeClaimDigestShape::from_claim(&first_relation.payload.pi_rlc.parent);
    let state_in_projection_fields_total: usize = first_relation
        .payload
        .state_in_claims
        .iter()
        .map(|claim| {
            projection_digest_field_count(
                claim.c.data.len(),
                claim.m_in,
                claim.r.len(),
                &claim.y_ring.iter().map(|row| row.len()).collect::<Vec<_>>(),
            )
        })
        .sum();
    let state_out_projection_fields_total: usize = first_relation
        .payload
        .state_out_claims
        .iter()
        .map(|claim| {
            projection_digest_field_count(
                claim.c.data.len(),
                claim.m_in,
                claim.r.len(),
                &claim.y_ring.iter().map(|row| row.len()).collect::<Vec<_>>(),
            )
        })
        .sum();
    let first_state_in = first_relation
        .payload
        .state_in_claims
        .first()
        .expect("state-in claim");
    let first_state_out = first_relation
        .payload
        .state_out_claims
        .first()
        .expect("state-out claim");
    let first_state_in_y_ring_row_lens = first_state_in
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let first_state_out_y_ring_row_lens = first_state_out
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let pi_rlc_parent_y_ring_row_lens = first_relation
        .payload
        .pi_rlc
        .parent
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let padded_child_count = usize::try_from(cover_shape.ccs_output_count).expect("padded child count");
    let actual_child_count = first_relation.payload.pi_ccs.ccs_outputs.len();
    let state_out_accumulator_phi_fields =
        accumulator_phi_dec_parent_hash_field_count(&first_relation.payload.state_out_claims);
    let work_units = ProbeWorkUnits {
        non_halt_opcode_count: opcode_count,
        semantic_step_count: usize::try_from(final_statement.folded.semantic_step_count)
            .expect("semantic step count fits usize"),
        chunk_count: usize::try_from(final_statement.folded.chunk_count).expect("chunk count fits usize"),
        chunk_fold_step_count: final_proof.steps.len(),
        relation_count: relations.len(),
        backend_relation_count: backend_relations.len(),
        fold_schedule: root_fold_schedule,
    };

    print_section("RV32IM Main Recursion Shape Probe");
    print_kv(
        "mode",
        match probe_mode {
            ProbeMode::Full => "full",
            ProbeMode::FastSummary => "fast-summary",
            ProbeMode::StageAux => "stage-aux",
            ProbeMode::ConstraintBreakdown => "constraint-breakdown",
            ProbeMode::TraceShape => "trace-shape",
        },
    );
    print_kv("mixed_opcode_non_halt_ops", opcode_count);
    print_kv("relation_count", relations.len());
    print_kv("backend_relation_count", backend_relations.len());
    print_kv("selected_relation_index", selected_relation_index);
    print_kv("terminal_relation_index", terminal_relation_index);
    print_kv("fixture_prep", format!("{fixture_ms:.3} ms"));
    print_probe_work_units("Execution Units", work_units);
    print_section("Fixture Breakdown");
    print_kv("accepted_proof.wall", format!("{accepted_ms:.3} ms"));
    print_kv("accepted_proof.total", format!("{:.3} ms", accepted_perf.total_ms));
    print_kv(
        "accepted_root_session",
        format!("{:.3} ms", accepted_perf.root_main_lane.session.total_ms),
    );
    print_kv(
        "accepted_root_rlc_dec",
        format!("{:.3} ms", accepted_root_rlc_dec_ms(&accepted_perf)),
    );
    print_kv(
        "accepted_root_ccs",
        format!("{:.3} ms", accepted_perf.root_main_lane.session.ccs_ms()),
    );
    print_kv("final_statement", format!("{final_statement_ms:.3} ms"));
    print_kv("build_relations", format!("{relations_ms:.3} ms"));
    print_kv("build_advices", format!("{advices_ms:.3} ms"));
    print_kv("build_backend_relations", format!("{backend_relations_ms:.3} ms"));

    print_section("Shape");
    match &shape_only {
        Ok(shape_only) => {
            print_kv("shape_only.wall", format!("{shape_only_ms:.3} ms"));
            print_kv("shape_only.num_inputs", shape_only.num_inputs);
            print_kv("shape_only.num_aux", shape_only.num_aux);
            print_kv("shape_only.num_constraints", shape_only.num_constraints);
            print_kv(
                "total_aux_across_all_relations",
                shape_only.num_aux * backend_relations.len(),
            );
            print_kv(
                "total_constraints_across_all_relations",
                shape_only.num_constraints * backend_relations.len(),
            );
        }
        Err(err) => {
            print_kv("shape_only.wall", format!("{shape_only_ms:.3} ms"));
            print_kv("shape_only.error", err);
        }
    }

    if let Some((terminal_committed_shape, terminal_committed_ms)) = &terminal_committed_shape {
        print_section("Terminal F' Committed-Step Shape");
        print_kv("measure.wall", format!("{terminal_committed_ms:.3} ms"));
        print_kv(
            "r2_source_ccs.rows",
            terminal_committed_shape.terminal_r2_source_ccs_rows,
        );
        print_kv(
            "r2_source_ccs.cols",
            terminal_committed_shape.terminal_r2_source_ccs_cols,
        );
        print_kv("r2_source_ccs.nnz", terminal_committed_shape.terminal_r2_source_ccs_nnz);
        print_kv("r2_public_inputs", terminal_committed_shape.terminal_r2_public_inputs);
        print_kv("r2_witness_inputs", terminal_committed_shape.terminal_r2_witness_inputs);
        print_kv(
            "r2_private_padding_inputs",
            terminal_committed_shape.terminal_r2_private_padding_inputs,
        );
        print_kv(
            "r2_private_low_norm_bits",
            terminal_committed_shape.terminal_r2_private_low_norm_bit_inputs,
        );
        print_kv(
            "r2_committed_low_norm_width",
            terminal_committed_shape.terminal_r2_committed_low_norm_width,
        );
        print_kv(
            "r2_superneo_packed_cols",
            terminal_committed_shape.terminal_r2_superneo_packed_cols,
        );
        print_kv(
            "r2_commitment_words",
            terminal_committed_shape.terminal_r2_commitment_words,
        );
        print_kv(
            "committed_step_public_inputs",
            terminal_committed_shape.terminal_committed_step_public_inputs,
        );
        print_kv(
            "committed_step_constraints",
            terminal_committed_shape.terminal_committed_step_constraints,
        );
        print_kv(
            "r1cs_public_inputs",
            terminal_committed_shape.terminal_f_prime_r1cs_public_inputs,
        );
        print_kv(
            "r1cs_challenges",
            terminal_committed_shape.terminal_f_prime_r1cs_challenges,
        );
        print_kv(
            "r1cs_variables",
            terminal_committed_shape.terminal_f_prime_r1cs_variables,
        );
        print_kv(
            "r1cs_constraints",
            terminal_committed_shape.terminal_f_prime_r1cs_constraints,
        );
        print_kv("r1cs_nnz", terminal_committed_shape.terminal_f_prime_r1cs_nnz);
    }

    if probe_mode == ProbeMode::TraceShape {
        let traced =
            debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize(&spartan_shape, "shape_trace")
                .expect("trace shape-only circuit");
        print_section("Shape Trace");
        print_kv("num_inputs", traced.num_inputs);
        print_kv("num_aux", traced.num_aux);
        print_kv("num_constraints", traced.num_constraints);
        return;
    }

    if probe_mode == ProbeMode::FastSummary {
        print_section("Payload Dimensions");
        print_kv("step_shape.state_in_claim_count", step_shape.state_in_claim_count);
        print_kv("step_shape.state_out_claim_count", step_shape.state_out_claim_count);
        print_kv("step_shape.fresh_claim_count", step_shape.fresh_claim_count);
        print_kv("step_shape.ccs_output_count", step_shape.ccs_output_count);
        print_kv("step_shape.child_count", step_shape.child_count);
        print_kv("cover_shape.ccs_output_count", cover_shape.ccs_output_count);
        print_kv("cover_shape.child_count", cover_shape.child_count);

        print_section("State In Claim Surface");
        print_kv("claim_count", first_relation.payload.state_in_claims.len());
        print_kv("claim.c_data_len", state_in_claim_shape.c_data_len);
        print_kv("claim.x_compact_len", first_state_in.m_in);
        print_kv("claim.r_len", state_in_claim_shape.r_len);
        print_kv("claim.y_ring_rows", state_in_claim_shape.y_ring_row_count);
        print_kv("claim.y_ring_row_lens", format!("{:?}", first_state_in_y_ring_row_lens));
        print_kv(
            "projection_hash_terms_per_claim",
            projection_digest_field_count(
                first_state_in.c.data.len(),
                first_state_in.m_in,
                first_state_in.r.len(),
                &first_state_in_y_ring_row_lens,
            ),
        );
        print_kv("projection_hash_terms_total", state_in_projection_fields_total);

        print_section("State Out Claim Surface");
        print_kv("claim_count", first_relation.payload.state_out_claims.len());
        print_kv("claim.c_data_len", state_out_claim_shape.c_data_len);
        print_kv("claim.x_compact_len", first_state_out.m_in);
        print_kv("claim.r_len", state_out_claim_shape.r_len);
        print_kv("claim.y_ring_rows", state_out_claim_shape.y_ring_row_count);
        print_kv(
            "claim.y_ring_row_lens",
            format!("{:?}", first_state_out_y_ring_row_lens),
        );
        print_kv(
            "projection_hash_terms_per_claim",
            projection_digest_field_count(
                first_state_out.c.data.len(),
                first_state_out.m_in,
                first_state_out.r.len(),
                &first_state_out_y_ring_row_lens,
            ),
        );
        print_kv("projection_hash_terms_total", state_out_projection_fields_total);
        print_kv("accumulator_phi_hash_terms", state_out_accumulator_phi_fields);

        print_section("Pi RLC Public Surface");
        print_kv("actual_child_count", actual_child_count);
        print_kv("padded_child_count", padded_child_count);
        print_kv("parent.c_data_len", pi_rlc_parent_shape.c_data_len);
        print_kv("parent.commitment_rows", D);
        print_kv(
            "parent.commitment_cols",
            usize::try_from(pi_rlc_parent_shape.c_data_len).expect("commitment len") / D,
        );
        print_kv("parent.x_compact_len", first_relation.payload.pi_rlc.parent.m_in);
        print_kv("parent.r_len", pi_rlc_parent_shape.r_len);
        print_kv("parent.y_ring_rows", pi_rlc_parent_shape.y_ring_row_count);
        print_kv("parent.y_ring_row_lens", format!("{:?}", pi_rlc_parent_y_ring_row_lens));
        print_kv("parent.y_zcol_len", pi_rlc_parent_shape.y_zcol_len);
        print_kv(
            "dense_c_scalars_across_children",
            padded_child_count * usize::try_from(pi_rlc_parent_shape.c_data_len).expect("parent c_data len"),
        );
        print_kv(
            "dense_y_ring_k_scalars_per_claim",
            first_relation
                .payload
                .pi_rlc
                .parent
                .y_ring
                .iter()
                .map(|row| row.len())
                .sum::<usize>(),
        );
        let fresh_child_count = usize::try_from(step_shape.fresh_claim_count).expect("fresh child count");
        print_pi_rlc_public_child_families(
            &first_relation,
            fresh_child_count,
            actual_child_count,
            &pi_rlc_parent_shape,
        );
        print_backend_relation_commitment_sparsity(&backend_relations);
        let fast_summary = FastSummaryPerf {
            fixture_ms,
            accepted_wall_ms: accepted_ms,
            accepted_perf: accepted_perf.clone(),
            final_statement_ms,
            relations_ms,
            advices_ms,
            backend_relations_ms,
            shape_only_ms,
        };
        print_probe_work_units("Key Per-Opcode Units", work_units);
        print_key_per_fold_summary("Key Per-Fold Summary", &fast_summary, work_units.chunk_fold_step_count);
        print_key_per_opcode_summary("Key Per-Opcode Summary", &fast_summary, opcode_count);
        print_per_opcode_components("Per-Opcode Components", &fast_summary, opcode_count);
        return;
    }

    if probe_mode == ProbeMode::ConstraintBreakdown {
        let chunk_replay_aux_started = Instant::now();
        let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
            .expect("measure first-step chunk replay aux counts");
        let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);

        let chunk_replay_tail_digest_started = Instant::now();
        let chunk_replay_tail_digest =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
                .expect("measure first-step chunk replay tail digest aux breakdown");
        let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

        let pi_ccs_bind_me_inputs_started = Instant::now();
        let pi_ccs_bind_me_inputs =
            debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown(first_relation)
                .expect("measure first-step pi_ccs bind_me_inputs aux breakdown");
        let pi_ccs_bind_me_inputs_ms = millis_since(pi_ccs_bind_me_inputs_started);

        let pi_ccs_constraints_started = Instant::now();
        let pi_ccs_constraints = debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts(first_relation)
            .expect("measure first-step pi_ccs constraint counts");
        let pi_ccs_constraints_ms = millis_since(pi_ccs_constraints_started);

        let pi_ccs_sumcheck_started = Instant::now();
        let pi_ccs_sumcheck =
            debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown(first_relation)
                .expect("measure first-step pi_ccs sumcheck constraint breakdown");
        let pi_ccs_sumcheck_ms = millis_since(pi_ccs_sumcheck_started);

        let pi_rlc_public_started = Instant::now();
        let pi_rlc_public = debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(first_relation)
            .expect("measure first-step pi_rlc public breakdown");
        let pi_rlc_public_ms = millis_since(pi_rlc_public_started);

        let pi_rlc_public_stage_started = Instant::now();
        let pi_rlc_public_stage =
            debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown(first_relation)
                .expect("measure first-step pi_rlc public stage breakdown");
        let pi_rlc_public_stage_ms = millis_since(pi_rlc_public_stage_started);

        print_section("Chunk NIFS Verifier Aux Hotspots");
        print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
        print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
        print_cumulative_and_delta(
            "after_public_chunk_meta",
            chunk_replay_aux.after_state_cover,
            chunk_replay_aux.after_chunk_meta,
        );
        print_cumulative_and_delta(
            "after_pi_ccs",
            chunk_replay_aux.after_chunk_meta,
            chunk_replay_aux.after_pi_ccs,
        );
        print_cumulative_and_delta(
            "after_synthetic_relation_io",
            chunk_replay_aux.after_pi_ccs,
            chunk_replay_aux.after_synthetic_relation_io,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_parent_claim",
            chunk_replay_aux.after_synthetic_relation_io,
            chunk_replay_aux.after_pi_rlc_parent_claim,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rhos",
            chunk_replay_aux.after_pi_rlc_parent_claim,
            chunk_replay_aux.after_pi_rlc_rhos,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rho_mats",
            chunk_replay_aux.after_pi_rlc_rhos,
            chunk_replay_aux.after_pi_rlc_rho_mats,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_public",
            chunk_replay_aux.after_pi_rlc_rho_mats,
            chunk_replay_aux.after_pi_rlc_public,
        );
        print_cumulative_and_delta(
            "after_pi_rlc",
            chunk_replay_aux.after_pi_rlc_public,
            chunk_replay_aux.after_pi_rlc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_body",
            chunk_replay_aux.after_pi_rlc,
            chunk_replay_aux.after_chunk_body,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_aux.after_chunk_replay,
        );

        let mut tail_claim_digest_deltas = Vec::with_capacity(chunk_replay_tail_digest.claim_after_digests.len());
        let mut tail_digest_prev = chunk_replay_tail_digest.after_header;
        let mut tail_claim_digest_total = 0usize;
        for idx in 0..chunk_replay_tail_digest.claim_after_digests.len() {
            let claim_delta = chunk_replay_tail_digest.claim_after_digests[idx].saturating_sub(tail_digest_prev);
            tail_claim_digest_total += claim_delta;
            tail_claim_digest_deltas.push((idx, claim_delta));
            tail_digest_prev = chunk_replay_tail_digest.claim_after_digests[idx];
        }
        let tail_outer_hash_delta = chunk_replay_tail_digest
            .after_outer_hash
            .saturating_sub(tail_digest_prev);
        print_section("Chunk NIFS Verifier Tail Digest Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
        print_kv(
            "header",
            chunk_replay_tail_digest
                .after_header
                .saturating_sub(chunk_replay_aux.after_chunk_body),
        );
        print_kv("claim_digest_total", tail_claim_digest_total);
        print_kv("outer_hash", tail_outer_hash_delta);
        for (idx, claim_delta) in &tail_claim_digest_deltas {
            print_kv(&format!("claim_{idx}.digest"), *claim_delta);
        }

        let mut pi_ccs_bind_me_input_deltas = Vec::with_capacity(1 + pi_ccs_bind_me_inputs.after_claim_digests.len());
        let mut bind_prev = pi_ccs_bind_me_inputs.after_bind_header;
        for (idx, end) in pi_ccs_bind_me_inputs.after_claim_digests.iter().enumerate() {
            pi_ccs_bind_me_input_deltas.push((format!("claim_digest_{idx}"), end.saturating_sub(bind_prev)));
            bind_prev = *end;
        }
        pi_ccs_bind_me_input_deltas.push((
            "bind_digests".to_string(),
            pi_ccs_bind_me_inputs
                .after_bind_digests
                .saturating_sub(bind_prev),
        ));
        print_section("Pi CCS Bind ME Inputs Aux");
        print_kv("measure.wall", format!("{pi_ccs_bind_me_inputs_ms:.3} ms"));
        for (name, delta) in &pi_ccs_bind_me_input_deltas {
            print_kv(name, *delta);
        }

        print_section("Pi CCS Constraints");
        print_kv("measure.wall", format!("{pi_ccs_constraints_ms:.3} ms"));
        print_cumulative_and_delta("after_bind_header", 0, pi_ccs_constraints.after_bind_header);
        print_cumulative_and_delta(
            "after_bind_me_inputs",
            pi_ccs_constraints.after_bind_header,
            pi_ccs_constraints.after_bind_me_inputs,
        );
        print_cumulative_and_delta(
            "after_sample_challenges",
            pi_ccs_constraints.after_bind_me_inputs,
            pi_ccs_constraints.after_sample_challenges,
        );
        print_cumulative_and_delta(
            "after_alloc_fresh_claims",
            pi_ccs_constraints.after_sample_challenges,
            pi_ccs_constraints.after_alloc_fresh_claims,
        );
        print_cumulative_and_delta(
            "after_fe_sumcheck",
            pi_ccs_constraints.after_alloc_fresh_claims,
            pi_ccs_constraints.after_fe_sumcheck,
        );
        print_cumulative_and_delta(
            "after_nc_sumcheck",
            pi_ccs_constraints.after_fe_sumcheck,
            pi_ccs_constraints.after_nc_sumcheck,
        );
        print_cumulative_and_delta(
            "after_fold_digest",
            pi_ccs_constraints.after_nc_sumcheck,
            pi_ccs_constraints.after_fold_digest,
        );
        print_cumulative_and_delta(
            "after_alloc_outputs",
            pi_ccs_constraints.after_fold_digest,
            pi_ccs_constraints.after_alloc_outputs,
        );
        print_cumulative_and_delta(
            "after_output_binding",
            pi_ccs_constraints.after_alloc_outputs,
            pi_ccs_constraints.after_output_binding,
        );
        print_cumulative_and_delta(
            "after_terminal_fe",
            pi_ccs_constraints.after_output_binding,
            pi_ccs_constraints.after_terminal_fe,
        );
        print_cumulative_and_delta(
            "after_terminal_nc",
            pi_ccs_constraints.after_terminal_fe,
            pi_ccs_constraints.after_terminal_nc,
        );

        print_named_constraint_breakdown(
            "Pi CCS FE Sumcheck Constraints",
            pi_ccs_sumcheck_ms,
            &pi_ccs_sumcheck.fe_cover_round_lengths,
            &pi_ccs_sumcheck.fe_effective_round_lengths,
            &pi_ccs_sumcheck.fe_stages,
        );
        print_named_constraint_breakdown(
            "Pi CCS NC Sumcheck Constraints",
            pi_ccs_sumcheck_ms,
            &pi_ccs_sumcheck.nc_cover_round_lengths,
            &pi_ccs_sumcheck.nc_effective_round_lengths,
            &pi_ccs_sumcheck.nc_stages,
        );

        print_section("Pi RLC Public");
        print_kv("measure.wall", format!("{pi_rlc_public_ms:.3} ms"));
        print_kv("shared_point", pi_rlc_public.shared_point_constraints);
        print_kv("x", pi_rlc_public.x_constraints);
        print_kv("c", pi_rlc_public.c_constraints);
        print_kv("y_ring", pi_rlc_public.y_ring_constraints);
        print_kv("y_zcol", pi_rlc_public.y_zcol_constraints);
        print_kv("aux", pi_rlc_public.aux_constraints);
        print_kv("total", pi_rlc_public.total_constraints);

        print_section("Pi RLC Public Stages");
        print_kv("measure.wall", format!("{pi_rlc_public_stage_ms:.3} ms"));
        for stage in &pi_rlc_public_stage.stages {
            print_kv(&stage.name, stage.delta);
        }
        return;
    }

    let top_level_aux_started = Instant::now();
    let top_level_aux = debug_measure_rv32im_main_recursion_step_stage_aux_counts(&spartan_shape, first_relation)
        .expect("measure first-step stage aux counts");
    let top_level_aux_ms = millis_since(top_level_aux_started);

    if probe_mode == ProbeMode::StageAux {
        let chunk_replay_aux_started = Instant::now();
        let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
            .expect("measure first-step chunk replay aux counts");
        let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);
        let chunk_replay_tail_aux_started = Instant::now();
        let chunk_replay_tail_aux =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts(first_relation)
                .expect("measure first-step chunk replay tail aux counts");
        let chunk_replay_tail_aux_ms = millis_since(chunk_replay_tail_aux_started);
        let chunk_replay_tail_digest_started = Instant::now();
        let chunk_replay_tail_digest =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
                .expect("measure first-step chunk replay tail digest aux breakdown");
        let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

        print_section("Top-Level Aux");
        print_kv("measure.wall", format!("{top_level_aux_ms:.3} ms"));
        print_cumulative_and_delta(
            "after_private_witness_inputs",
            0,
            top_level_aux.after_private_witness_inputs,
        );
        print_cumulative_and_delta(
            "after_alloc_cover_states",
            top_level_aux.after_private_witness_inputs,
            top_level_aux.after_alloc_cover_states,
        );
        print_cumulative_and_delta(
            "after_bind_state_and_pc",
            top_level_aux.after_alloc_cover_states,
            top_level_aux.after_bind_state_and_pc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            top_level_aux.after_bind_state_and_pc,
            top_level_aux.after_chunk_replay,
        );
        print_cumulative_and_delta(
            "after_inactive_side_lane_x_out",
            top_level_aux.after_chunk_replay,
            top_level_aux.after_inactive_side_lane_and_x_out,
        );
        print_cumulative_and_delta(
            "after_public_output_eq",
            top_level_aux.after_inactive_side_lane_and_x_out,
            top_level_aux.after_public_output_eq,
        );
        print_section("Chunk NIFS Verifier Aux");
        print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
        print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
        print_cumulative_and_delta(
            "after_public_chunk_meta",
            chunk_replay_aux.after_state_cover,
            chunk_replay_aux.after_chunk_meta,
        );
        print_cumulative_and_delta(
            "after_pi_ccs",
            chunk_replay_aux.after_chunk_meta,
            chunk_replay_aux.after_pi_ccs,
        );
        print_cumulative_and_delta(
            "after_synthetic_relation_io",
            chunk_replay_aux.after_pi_ccs,
            chunk_replay_aux.after_synthetic_relation_io,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_parent_claim",
            chunk_replay_aux.after_synthetic_relation_io,
            chunk_replay_aux.after_pi_rlc_parent_claim,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rhos",
            chunk_replay_aux.after_pi_rlc_parent_claim,
            chunk_replay_aux.after_pi_rlc_rhos,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rho_mats",
            chunk_replay_aux.after_pi_rlc_rhos,
            chunk_replay_aux.after_pi_rlc_rho_mats,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_public",
            chunk_replay_aux.after_pi_rlc_rho_mats,
            chunk_replay_aux.after_pi_rlc_public,
        );
        print_cumulative_and_delta(
            "after_pi_rlc",
            chunk_replay_aux.after_pi_rlc_public,
            chunk_replay_aux.after_pi_rlc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_body",
            chunk_replay_aux.after_pi_rlc,
            chunk_replay_aux.after_chunk_body,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_aux.after_chunk_replay,
        );
        print_section("Chunk NIFS Verifier Tail Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_aux_ms:.3} ms"));
        print_cumulative_and_delta(
            "after_state_out_projection_eq",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_tail_aux.after_state_out_projection_eq,
        );
        print_cumulative_and_delta(
            "after_expected_digest",
            chunk_replay_tail_aux.after_state_out_projection_eq,
            chunk_replay_tail_aux.after_expected_digest,
        );
        print_cumulative_and_delta(
            "after_chunk_done_tag",
            chunk_replay_tail_aux.after_expected_digest,
            chunk_replay_tail_aux.after_chunk_done,
        );
        print_cumulative_and_delta(
            "after_transcript_state_eq",
            chunk_replay_tail_aux.after_chunk_done,
            chunk_replay_tail_aux.after_transcript_state_eq,
        );
        print_cumulative_and_delta(
            "after_transcript_absorbed_eq",
            chunk_replay_tail_aux.after_transcript_state_eq,
            chunk_replay_tail_aux.after_transcript_absorbed_eq,
        );
        let tail_header_delta = chunk_replay_tail_digest
            .after_header
            .saturating_sub(chunk_replay_aux.after_chunk_body);
        let mut tail_total_claim_digest = 0usize;
        let mut prev = chunk_replay_tail_digest.after_header;
        for after_digest in &chunk_replay_tail_digest.claim_after_digests {
            tail_total_claim_digest += after_digest.saturating_sub(prev);
            prev = *after_digest;
        }
        let tail_outer_hash_delta = chunk_replay_tail_digest
            .after_outer_hash
            .saturating_sub(prev);
        print_section("Chunk NIFS Verifier Tail Digest Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
        print_kv("header", tail_header_delta);
        print_kv("claim_digest_total", tail_total_claim_digest);
        print_kv("outer_hash", tail_outer_hash_delta);
        return;
    }

    let chunk_replay_aux_started = Instant::now();
    let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
        .expect("measure first-step chunk replay aux counts");
    let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);

    let chunk_replay_tail_aux_started = Instant::now();
    let chunk_replay_tail_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts(first_relation)
        .expect("measure first-step chunk replay tail aux counts");
    let chunk_replay_tail_aux_ms = millis_since(chunk_replay_tail_aux_started);

    let pi_ccs_aux_started = Instant::now();
    let pi_ccs_aux = debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts(first_relation)
        .expect("measure first-step pi_ccs aux counts");
    let pi_ccs_aux_ms = millis_since(pi_ccs_aux_started);

    let pi_ccs_constraints_started = Instant::now();
    let pi_ccs_constraints = debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts(first_relation)
        .expect("measure first-step pi_ccs constraint counts");
    let pi_ccs_constraints_ms = millis_since(pi_ccs_constraints_started);

    let pi_ccs_bind_me_inputs_started = Instant::now();
    let pi_ccs_bind_me_inputs =
        debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown(first_relation)
            .expect("measure first-step pi_ccs bind_me_inputs aux breakdown");
    let pi_ccs_bind_me_inputs_ms = millis_since(pi_ccs_bind_me_inputs_started);

    let pi_ccs_sumcheck_started = Instant::now();
    let pi_ccs_sumcheck = debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown(first_relation)
        .expect("measure first-step pi_ccs sumcheck constraint breakdown");
    let pi_ccs_sumcheck_ms = millis_since(pi_ccs_sumcheck_started);

    let pi_rlc_public_started = Instant::now();
    let pi_rlc_public = debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(first_relation)
        .expect("measure first-step pi_rlc public breakdown");
    let pi_rlc_public_ms = millis_since(pi_rlc_public_started);

    let pi_rlc_public_stage_started = Instant::now();
    let pi_rlc_public_stage = debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown(first_relation)
        .expect("measure first-step pi_rlc public stage breakdown");
    let pi_rlc_public_stage_ms = millis_since(pi_rlc_public_stage_started);

    let chunk_replay_tail_digest_started = Instant::now();
    let chunk_replay_tail_digest =
        debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
            .expect("measure first-step chunk replay tail digest aux breakdown");
    let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

    let synth_started = Instant::now();
    let shape_synth = debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis(&spartan_shape, first_relation)
        .expect("measure first-step shape synthesis");
    let synth_ms = millis_since(synth_started);

    let live_shape_started = Instant::now();
    let live_shape = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first_relation)
        .expect("measure first-step circuit shape");
    let live_shape_ms = millis_since(live_shape_started);
    let pi_ccs_fingerprint_started = Instant::now();
    let pi_ccs_fingerprint = debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(first_relation)
        .expect("measure first-step pi_ccs fingerprint");
    let pi_ccs_fingerprint_ms = millis_since(pi_ccs_fingerprint_started);
    let chunk_replay_fingerprint_started = Instant::now();
    let chunk_replay_fingerprint = debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint(first_relation)
        .expect("measure first-step chunk replay fingerprint");
    let chunk_replay_fingerprint_ms = millis_since(chunk_replay_fingerprint_started);

    let fixed_shape_sanity_started = Instant::now();
    let mut perturbed_relation = first_relation.clone();
    perturb_backend_relation_values(&mut perturbed_relation);
    let perturbed_shape =
        debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, &perturbed_relation);
    let perturbed_pi_ccs_fingerprint = debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&perturbed_relation);
    let perturbed_chunk_replay_fingerprint =
        debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint(&perturbed_relation);
    let fixed_shape_sanity_ms = millis_since(fixed_shape_sanity_started);

    let fixed_shape_family_started = Instant::now();
    let mut state_in_r_relation = first_relation.clone();
    perturb_state_in_r_values(&mut state_in_r_relation);
    let state_in_r_status =
        fixed_shape_family_status(&spartan_shape, &live_shape.constraint_fingerprint, &state_in_r_relation);
    let mut state_in_y_ring_relation = first_relation.clone();
    perturb_state_in_y_ring_values(&mut state_in_y_ring_relation);
    let state_in_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_in_y_ring_relation,
    );
    let mut state_in_projection_relation = first_relation.clone();
    perturb_state_in_projection_values(&mut state_in_projection_relation);
    let state_in_projection_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_in_projection_relation,
    );
    let mut pi_ccs_alpha_relation = first_relation.clone();
    perturb_pi_ccs_alpha_values(&mut pi_ccs_alpha_relation);
    let pi_ccs_alpha_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_alpha_relation,
    );
    let mut pi_ccs_gamma_relation = first_relation.clone();
    perturb_pi_ccs_gamma_value(&mut pi_ccs_gamma_relation);
    let pi_ccs_gamma_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_gamma_relation,
    );
    let mut state_out_projection_relation = first_relation.clone();
    perturb_state_out_projection_values(&mut state_out_projection_relation);
    let state_out_projection_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_out_projection_relation,
    );
    let mut pi_ccs_output_y_ring_relation = first_relation.clone();
    perturb_pi_ccs_output_y_ring_values(&mut pi_ccs_output_y_ring_relation);
    let pi_ccs_output_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_output_y_ring_relation,
    );
    let mut pi_ccs_output_y_zcol_relation = first_relation.clone();
    perturb_pi_ccs_output_y_zcol_values(&mut pi_ccs_output_y_zcol_relation);
    let pi_ccs_output_y_zcol_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_output_y_zcol_relation,
    );
    let mut pi_dec_child_y_ring_relation = first_relation.clone();
    perturb_pi_dec_child_y_ring_values(&mut pi_dec_child_y_ring_relation);
    let pi_dec_child_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_dec_child_y_ring_relation,
    );
    let mut pi_rlc_parent_relation = first_relation.clone();
    perturb_pi_rlc_parent_values(&mut pi_rlc_parent_relation);
    let pi_rlc_parent_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_rlc_parent_relation,
    );
    let mut fresh_claim_relation = first_relation.clone();
    perturb_fresh_claim_values(&mut fresh_claim_relation);
    let fresh_claim_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &fresh_claim_relation,
    );
    let mut fresh_witness_relation = first_relation.clone();
    perturb_fresh_witness_values(&mut fresh_witness_relation);
    let fresh_witness_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &fresh_witness_relation,
    );
    let state_in_projection_pi_ccs_fingerprint =
        debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&state_in_projection_relation);
    let fresh_claim_pi_ccs_fingerprint =
        debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&fresh_claim_relation);
    let fixed_shape_family_ms = millis_since(fixed_shape_family_started);

    print_kv("live_shape.wall", format!("{live_shape_ms:.3} ms"));
    print_kv("live_shape.num_inputs", live_shape.num_inputs);
    print_kv("live_shape.num_aux", live_shape.num_aux);
    print_kv("live_shape.num_constraints", live_shape.num_constraints);
    print_kv(
        "live_shape.total_constraints_across_all_relations",
        live_shape.num_constraints * backend_relations.len(),
    );

    print_section("Shape Synthesis");
    print_kv("shape_synth.wall", format!("{synth_ms:.3} ms"));
    print_kv("shape_synth.shared", format!("{:.3} ms", shape_synth.shared_ms));
    print_kv(
        "shape_synth.precommitted",
        format!("{:.3} ms", shape_synth.precommitted_ms),
    );
    print_kv("shape_synth.synthesize", format!("{:.3} ms", shape_synth.synthesize_ms));
    print_kv("shape_synth.num_inputs", shape_synth.num_inputs);
    print_kv("shape_synth.num_aux", shape_synth.num_aux);
    print_kv("shape_synth.num_constraints", shape_synth.num_constraints);
    print_kv(
        "shape_synth.total_constraints_across_all_relations",
        shape_synth.num_constraints * backend_relations.len(),
    );

    print_section("Fixed-Shape Sanity");
    print_kv("measure.wall", format!("{fixed_shape_sanity_ms:.3} ms"));
    print_kv("baseline.constraint_fingerprint", &live_shape.constraint_fingerprint);
    match &perturbed_shape {
        Ok(perturbed_shape) => {
            print_kv(
                "perturbed.constraint_fingerprint",
                &perturbed_shape.constraint_fingerprint,
            );
            print_kv(
                "fingerprint_equal",
                if live_shape.constraint_fingerprint == perturbed_shape.constraint_fingerprint {
                    "yes"
                } else {
                    "no"
                },
            );
            print_kv(
                "num_constraints_equal",
                if live_shape.num_constraints == perturbed_shape.num_constraints {
                    "yes"
                } else {
                    "no"
                },
            );
            print_kv(
                "num_aux_equal",
                if live_shape.num_aux == perturbed_shape.num_aux {
                    "yes"
                } else {
                    "no"
                },
            );
        }
        Err(err) => {
            print_kv("perturbed.constraint_fingerprint", "unsat");
            print_kv("perturbed.error", err);
            print_kv("fingerprint_equal", "n/a");
            print_kv("num_constraints_equal", "n/a");
            print_kv("num_aux_equal", "n/a");
        }
    }

    print_section("Fixed-Shape Families");
    print_kv("measure.wall", format!("{fixed_shape_family_ms:.3} ms"));
    print_kv("state_in_r_only", state_in_r_status);
    print_kv("state_in_y_ring_only", state_in_y_ring_status);
    print_kv("state_in_projection_only", state_in_projection_status);
    print_kv("pi_ccs_alpha_only", pi_ccs_alpha_status);
    print_kv("pi_ccs_gamma_only", pi_ccs_gamma_status);
    print_kv("state_out_projection_only", state_out_projection_status);
    print_kv("pi_ccs_output_y_ring_only", pi_ccs_output_y_ring_status);
    print_kv("pi_ccs_output_y_zcol_only", pi_ccs_output_y_zcol_status);
    print_kv("pi_dec_child_y_ring_only", pi_dec_child_y_ring_status);
    print_kv("pi_rlc_parent_only", pi_rlc_parent_status);
    print_kv("fresh_claim_only", fresh_claim_status);
    print_kv("fresh_witness_only", fresh_witness_status);

    print_section("Fixed-Shape Drift Localizer");
    print_kv("measure.wall", format!("{pi_ccs_fingerprint_ms:.3} ms"));
    match &state_in_projection_pi_ccs_fingerprint {
        Ok(fingerprint) => {
            if let Some((stage, baseline, perturbed)) = first_pi_ccs_stage_diff(&pi_ccs_fingerprint, fingerprint) {
                print_kv("state_in_projection.first_diff_stage", stage);
                print_kv("state_in_projection.baseline", baseline);
                print_kv("state_in_projection.perturbed", perturbed);
            } else {
                print_kv("state_in_projection.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("state_in_projection.first_diff_stage", "unsat");
        }
    }
    match &fresh_claim_pi_ccs_fingerprint {
        Ok(fingerprint) => {
            if let Some((stage, baseline, perturbed)) = first_pi_ccs_stage_diff(&pi_ccs_fingerprint, fingerprint) {
                print_kv("fresh_claim.first_diff_stage", stage);
                print_kv("fresh_claim.baseline", baseline);
                print_kv("fresh_claim.perturbed", perturbed);
            } else {
                print_kv("fresh_claim.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("fresh_claim.first_diff_stage", "unsat");
        }
    }
    match &perturbed_pi_ccs_fingerprint {
        Ok(perturbed_pi_ccs_fingerprint) => {
            if let Some((stage, baseline, perturbed)) =
                first_pi_ccs_stage_diff(&pi_ccs_fingerprint, perturbed_pi_ccs_fingerprint)
            {
                print_kv("full_perturb.pi_ccs_first_diff_stage", stage);
                print_kv("full_perturb.pi_ccs_baseline", baseline);
                print_kv("full_perturb.pi_ccs_perturbed", perturbed);
            } else {
                print_kv("full_perturb.pi_ccs_first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("full_perturb.pi_ccs_first_diff_stage", "unsat");
        }
    }
    print_kv(
        "chunk_nifs_verifier.measure.wall",
        format!("{chunk_replay_fingerprint_ms:.3} ms"),
    );
    match &perturbed_chunk_replay_fingerprint {
        Ok(perturbed_chunk_replay_fingerprint) => {
            let chunk_replay_stages = [
                (
                    "after_state_cover",
                    &chunk_replay_fingerprint.after_state_cover,
                    &perturbed_chunk_replay_fingerprint.after_state_cover,
                ),
                (
                    "after_public_chunk_meta",
                    &chunk_replay_fingerprint.after_chunk_meta,
                    &perturbed_chunk_replay_fingerprint.after_chunk_meta,
                ),
                (
                    "after_pi_ccs",
                    &chunk_replay_fingerprint.after_pi_ccs,
                    &perturbed_chunk_replay_fingerprint.after_pi_ccs,
                ),
                (
                    "after_synthetic_relation_io",
                    &chunk_replay_fingerprint.after_synthetic_relation_io,
                    &perturbed_chunk_replay_fingerprint.after_synthetic_relation_io,
                ),
                (
                    "after_pi_rlc_parent_claim",
                    &chunk_replay_fingerprint.after_pi_rlc_parent_claim,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_parent_claim,
                ),
                (
                    "after_pi_rlc_rhos",
                    &chunk_replay_fingerprint.after_pi_rlc_rhos,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_rhos,
                ),
                (
                    "after_pi_rlc_rho_mats",
                    &chunk_replay_fingerprint.after_pi_rlc_rho_mats,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_rho_mats,
                ),
                (
                    "after_pi_rlc_public",
                    &chunk_replay_fingerprint.after_pi_rlc_public,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_public,
                ),
                (
                    "after_pi_rlc",
                    &chunk_replay_fingerprint.after_pi_rlc,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc,
                ),
                (
                    "after_chunk_nifs_body",
                    &chunk_replay_fingerprint.after_chunk_body,
                    &perturbed_chunk_replay_fingerprint.after_chunk_body,
                ),
                (
                    "after_chunk_nifs_verifier",
                    &chunk_replay_fingerprint.after_chunk_replay,
                    &perturbed_chunk_replay_fingerprint.after_chunk_replay,
                ),
            ];
            if let Some((stage, baseline, perturbed)) = chunk_replay_stages
                .into_iter()
                .find(|(_, baseline, perturbed)| baseline != perturbed)
            {
                print_kv("chunk_nifs_verifier.first_diff_stage", stage);
                print_kv("chunk_nifs_verifier.baseline", baseline);
                print_kv("chunk_nifs_verifier.perturbed", perturbed);
            } else {
                print_kv("chunk_nifs_verifier.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("chunk_nifs_verifier.first_diff_stage", "unsat");
        }
    }

    print_section("Top-Level Aux");
    print_kv("measure.wall", format!("{top_level_aux_ms:.3} ms"));
    print_cumulative_and_delta(
        "after_private_witness_inputs",
        0,
        top_level_aux.after_private_witness_inputs,
    );
    print_cumulative_and_delta(
        "after_alloc_cover_states",
        top_level_aux.after_private_witness_inputs,
        top_level_aux.after_alloc_cover_states,
    );
    print_cumulative_and_delta(
        "after_bind_state_and_pc",
        top_level_aux.after_alloc_cover_states,
        top_level_aux.after_bind_state_and_pc,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_verifier",
        top_level_aux.after_bind_state_and_pc,
        top_level_aux.after_chunk_replay,
    );
    print_cumulative_and_delta(
        "after_inactive_side_lane_x_out",
        top_level_aux.after_chunk_replay,
        top_level_aux.after_inactive_side_lane_and_x_out,
    );
    print_cumulative_and_delta(
        "after_public_output_eq",
        top_level_aux.after_inactive_side_lane_and_x_out,
        top_level_aux.after_public_output_eq,
    );

    print_section("Chunk NIFS Verifier Aux");
    print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
    print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
    print_cumulative_and_delta(
        "after_public_chunk_meta",
        chunk_replay_aux.after_state_cover,
        chunk_replay_aux.after_chunk_meta,
    );
    print_cumulative_and_delta(
        "after_pi_ccs",
        chunk_replay_aux.after_chunk_meta,
        chunk_replay_aux.after_pi_ccs,
    );
    print_cumulative_and_delta(
        "after_synthetic_relation_io",
        chunk_replay_aux.after_pi_ccs,
        chunk_replay_aux.after_synthetic_relation_io,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_parent_claim",
        chunk_replay_aux.after_synthetic_relation_io,
        chunk_replay_aux.after_pi_rlc_parent_claim,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_rhos",
        chunk_replay_aux.after_pi_rlc_parent_claim,
        chunk_replay_aux.after_pi_rlc_rhos,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_rho_mats",
        chunk_replay_aux.after_pi_rlc_rhos,
        chunk_replay_aux.after_pi_rlc_rho_mats,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_public",
        chunk_replay_aux.after_pi_rlc_rho_mats,
        chunk_replay_aux.after_pi_rlc_public,
    );
    print_cumulative_and_delta(
        "after_pi_rlc",
        chunk_replay_aux.after_pi_rlc_public,
        chunk_replay_aux.after_pi_rlc,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_body",
        chunk_replay_aux.after_pi_rlc,
        chunk_replay_aux.after_chunk_body,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_verifier",
        chunk_replay_aux.after_chunk_body,
        chunk_replay_aux.after_chunk_replay,
    );

    print_section("Chunk NIFS Verifier Tail Aux");
    print_kv("measure.wall", format!("{chunk_replay_tail_aux_ms:.3} ms"));
    print_cumulative_and_delta(
        "after_state_out_projection_eq",
        chunk_replay_aux.after_chunk_body,
        chunk_replay_tail_aux.after_state_out_projection_eq,
    );
    print_cumulative_and_delta(
        "after_expected_digest",
        chunk_replay_tail_aux.after_state_out_projection_eq,
        chunk_replay_tail_aux.after_expected_digest,
    );
    print_cumulative_and_delta(
        "after_chunk_done",
        chunk_replay_tail_aux.after_expected_digest,
        chunk_replay_tail_aux.after_chunk_done,
    );
    print_cumulative_and_delta(
        "after_transcript_state_eq",
        chunk_replay_tail_aux.after_chunk_done,
        chunk_replay_tail_aux.after_transcript_state_eq,
    );
    print_cumulative_and_delta(
        "after_transcript_absorbed_eq",
        chunk_replay_tail_aux.after_transcript_state_eq,
        chunk_replay_tail_aux.after_transcript_absorbed_eq,
    );

    let tail_header_delta = chunk_replay_tail_digest
        .after_header
        .saturating_sub(chunk_replay_aux.after_chunk_body);
    let mut tail_claim_digest_deltas = Vec::with_capacity(chunk_replay_tail_digest.claim_after_digests.len());
    let mut prev = chunk_replay_tail_digest.after_header;
    let mut tail_total_claim_digest = 0usize;
    for idx in 0..chunk_replay_tail_digest.claim_after_digests.len() {
        let claim_digest_delta = chunk_replay_tail_digest.claim_after_digests[idx].saturating_sub(prev);
        tail_total_claim_digest += claim_digest_delta;
        tail_claim_digest_deltas.push((idx, claim_digest_delta));
        prev = chunk_replay_tail_digest.claim_after_digests[idx];
    }
    let tail_outer_hash_delta = chunk_replay_tail_digest
        .after_outer_hash
        .saturating_sub(prev);
    print_section("Chunk NIFS Verifier Tail Digest Aux");
    print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
    print_kv("header", tail_header_delta);
    print_kv("claim_digest_total", tail_total_claim_digest);
    print_kv("outer_hash", tail_outer_hash_delta);
    for (idx, claim_digest_delta) in &tail_claim_digest_deltas {
        print_kv(&format!("claim_{idx}.digest"), *claim_digest_delta);
    }

    print_section("Pi CCS Aux");
    print_kv("measure.wall", format!("{pi_ccs_aux_ms:.3} ms"));
    print_cumulative_and_delta("after_bind_header", 0, pi_ccs_aux.after_bind_header);
    print_cumulative_and_delta(
        "after_bind_me_inputs",
        pi_ccs_aux.after_bind_header,
        pi_ccs_aux.after_bind_me_inputs,
    );
    print_cumulative_and_delta(
        "after_sample_challenges",
        pi_ccs_aux.after_bind_me_inputs,
        pi_ccs_aux.after_sample_challenges,
    );
    print_cumulative_and_delta(
        "after_alloc_fresh_claims",
        pi_ccs_aux.after_sample_challenges,
        pi_ccs_aux.after_alloc_fresh_claims,
    );
    print_cumulative_and_delta(
        "after_fe_sumcheck",
        pi_ccs_aux.after_alloc_fresh_claims,
        pi_ccs_aux.after_fe_sumcheck,
    );
    print_cumulative_and_delta(
        "after_nc_sumcheck",
        pi_ccs_aux.after_fe_sumcheck,
        pi_ccs_aux.after_nc_sumcheck,
    );
    print_cumulative_and_delta(
        "after_fold_digest",
        pi_ccs_aux.after_nc_sumcheck,
        pi_ccs_aux.after_fold_digest,
    );
    print_cumulative_and_delta(
        "after_alloc_outputs",
        pi_ccs_aux.after_fold_digest,
        pi_ccs_aux.after_alloc_outputs,
    );
    print_cumulative_and_delta(
        "after_output_binding",
        pi_ccs_aux.after_alloc_outputs,
        pi_ccs_aux.after_output_binding,
    );
    print_cumulative_and_delta(
        "after_terminal_fe",
        pi_ccs_aux.after_output_binding,
        pi_ccs_aux.after_terminal_fe,
    );
    print_cumulative_and_delta(
        "after_terminal_nc",
        pi_ccs_aux.after_terminal_fe,
        pi_ccs_aux.after_terminal_nc,
    );

    let mut pi_ccs_bind_me_input_deltas = Vec::with_capacity(1 + pi_ccs_bind_me_inputs.after_claim_digests.len());
    let mut prev = pi_ccs_bind_me_inputs.after_bind_header;
    for (idx, end) in pi_ccs_bind_me_inputs.after_claim_digests.iter().enumerate() {
        pi_ccs_bind_me_input_deltas.push((format!("claim_digest_{idx}"), end.saturating_sub(prev)));
        prev = *end;
    }
    pi_ccs_bind_me_input_deltas.push((
        "bind_digests".to_string(),
        pi_ccs_bind_me_inputs
            .after_bind_digests
            .saturating_sub(prev),
    ));
    print_section("Pi CCS Bind ME Inputs Aux");
    print_kv("measure.wall", format!("{pi_ccs_bind_me_inputs_ms:.3} ms"));
    for (name, delta) in &pi_ccs_bind_me_input_deltas {
        print_kv(name, *delta);
    }

    print_section("Pi CCS Constraints");
    print_kv("measure.wall", format!("{pi_ccs_constraints_ms:.3} ms"));
    print_cumulative_and_delta("after_bind_header", 0, pi_ccs_constraints.after_bind_header);
    print_cumulative_and_delta(
        "after_bind_me_inputs",
        pi_ccs_constraints.after_bind_header,
        pi_ccs_constraints.after_bind_me_inputs,
    );
    print_cumulative_and_delta(
        "after_sample_challenges",
        pi_ccs_constraints.after_bind_me_inputs,
        pi_ccs_constraints.after_sample_challenges,
    );
    print_cumulative_and_delta(
        "after_alloc_fresh_claims",
        pi_ccs_constraints.after_sample_challenges,
        pi_ccs_constraints.after_alloc_fresh_claims,
    );
    print_cumulative_and_delta(
        "after_fe_sumcheck",
        pi_ccs_constraints.after_alloc_fresh_claims,
        pi_ccs_constraints.after_fe_sumcheck,
    );
    print_cumulative_and_delta(
        "after_nc_sumcheck",
        pi_ccs_constraints.after_fe_sumcheck,
        pi_ccs_constraints.after_nc_sumcheck,
    );
    print_cumulative_and_delta(
        "after_fold_digest",
        pi_ccs_constraints.after_nc_sumcheck,
        pi_ccs_constraints.after_fold_digest,
    );
    print_cumulative_and_delta(
        "after_alloc_outputs",
        pi_ccs_constraints.after_fold_digest,
        pi_ccs_constraints.after_alloc_outputs,
    );
    print_cumulative_and_delta(
        "after_output_binding",
        pi_ccs_constraints.after_alloc_outputs,
        pi_ccs_constraints.after_output_binding,
    );
    print_cumulative_and_delta(
        "after_terminal_fe",
        pi_ccs_constraints.after_output_binding,
        pi_ccs_constraints.after_terminal_fe,
    );
    print_cumulative_and_delta(
        "after_terminal_nc",
        pi_ccs_constraints.after_terminal_fe,
        pi_ccs_constraints.after_terminal_nc,
    );

    print_named_constraint_breakdown(
        "Pi CCS FE Sumcheck Constraints",
        pi_ccs_sumcheck_ms,
        &pi_ccs_sumcheck.fe_cover_round_lengths,
        &pi_ccs_sumcheck.fe_effective_round_lengths,
        &pi_ccs_sumcheck.fe_stages,
    );
    print_named_constraint_breakdown(
        "Pi CCS NC Sumcheck Constraints",
        pi_ccs_sumcheck_ms,
        &pi_ccs_sumcheck.nc_cover_round_lengths,
        &pi_ccs_sumcheck.nc_effective_round_lengths,
        &pi_ccs_sumcheck.nc_stages,
    );

    print_section("Pi RLC Public");
    print_kv("measure.wall", format!("{pi_rlc_public_ms:.3} ms"));
    print_kv("shared_point", pi_rlc_public.shared_point_constraints);
    print_kv("x", pi_rlc_public.x_constraints);
    print_kv("c", pi_rlc_public.c_constraints);
    print_kv("y_ring", pi_rlc_public.y_ring_constraints);
    print_kv("y_zcol", pi_rlc_public.y_zcol_constraints);
    print_kv("aux", pi_rlc_public.aux_constraints);
    print_kv("total", pi_rlc_public.total_constraints);
    print_section("Pi RLC Public Stages");
    print_kv("measure.wall", format!("{pi_rlc_public_stage_ms:.3} ms"));
    for stage in &pi_rlc_public_stage.stages {
        print_kv(&stage.name, stage.delta);
    }

    print_section("Payload Dimensions");
    print_kv("step_shape.state_in_claim_count", step_shape.state_in_claim_count);
    print_kv("step_shape.state_out_claim_count", step_shape.state_out_claim_count);
    print_kv("step_shape.fresh_claim_count", step_shape.fresh_claim_count);
    print_kv("step_shape.ccs_output_count", step_shape.ccs_output_count);
    print_kv("step_shape.child_count", step_shape.child_count);
    print_kv("cover_shape.ccs_output_count", cover_shape.ccs_output_count);
    print_kv("cover_shape.child_count", cover_shape.child_count);

    print_section("State In Claim Surface");
    print_kv("claim_count", first_relation.payload.state_in_claims.len());
    print_kv("claim.c_data_len", state_in_claim_shape.c_data_len);
    print_kv("claim.x_compact_len", first_state_in.m_in);
    print_kv("claim.r_len", state_in_claim_shape.r_len);
    print_kv("claim.y_ring_rows", state_in_claim_shape.y_ring_row_count);
    print_kv("claim.y_ring_row_lens", format!("{:?}", first_state_in_y_ring_row_lens));
    print_kv(
        "projection_hash_terms_per_claim",
        projection_digest_field_count(
            first_state_in.c.data.len(),
            first_state_in.m_in,
            first_state_in.r.len(),
            &first_state_in_y_ring_row_lens,
        ),
    );
    print_kv("projection_hash_terms_total", state_in_projection_fields_total);

    print_section("State Out Claim Surface");
    print_kv("claim_count", first_relation.payload.state_out_claims.len());
    print_kv("claim.c_data_len", state_out_claim_shape.c_data_len);
    print_kv("claim.x_compact_len", first_state_out.m_in);
    print_kv("claim.r_len", state_out_claim_shape.r_len);
    print_kv("claim.y_ring_rows", state_out_claim_shape.y_ring_row_count);
    print_kv(
        "claim.y_ring_row_lens",
        format!("{:?}", first_state_out_y_ring_row_lens),
    );
    print_kv(
        "projection_hash_terms_per_claim",
        projection_digest_field_count(
            first_state_out.c.data.len(),
            first_state_out.m_in,
            first_state_out.r.len(),
            &first_state_out_y_ring_row_lens,
        ),
    );
    print_kv("projection_hash_terms_total", state_out_projection_fields_total);
    print_kv("accumulator_phi_hash_terms", state_out_accumulator_phi_fields);

    print_section("Pi RLC Public Surface");
    print_kv("actual_child_count", actual_child_count);
    print_kv("padded_child_count", padded_child_count);
    print_kv("parent.c_data_len", pi_rlc_parent_shape.c_data_len);
    print_kv("parent.commitment_rows", D);
    print_kv(
        "parent.commitment_cols",
        usize::try_from(pi_rlc_parent_shape.c_data_len).expect("commitment len") / D,
    );
    print_kv("parent.x_compact_len", first_relation.payload.pi_rlc.parent.m_in);
    print_kv("parent.r_len", pi_rlc_parent_shape.r_len);
    print_kv("parent.y_ring_rows", pi_rlc_parent_shape.y_ring_row_count);
    print_kv("parent.y_ring_row_lens", format!("{:?}", pi_rlc_parent_y_ring_row_lens));
    print_kv("parent.y_zcol_len", pi_rlc_parent_shape.y_zcol_len);
    print_kv(
        "dense_c_scalars_across_children",
        padded_child_count * usize::try_from(pi_rlc_parent_shape.c_data_len).expect("parent c_data len"),
    );
    print_kv(
        "dense_y_ring_k_scalars_per_claim",
        first_relation
            .payload
            .pi_rlc
            .parent
            .y_ring
            .iter()
            .map(|row| row.len())
            .sum::<usize>(),
    );
    let fresh_child_count = usize::try_from(step_shape.fresh_claim_count).expect("fresh child count");
    print_pi_rlc_public_child_families(
        &first_relation,
        fresh_child_count,
        actual_child_count,
        &pi_rlc_parent_shape,
    );
    print_backend_relation_commitment_sparsity(&backend_relations);

    let rerun_summary = measure_fast_summary_perf(&input);
    print_probe_work_units("Fast Key Per-Opcode Units (Rerun)", work_units);
    print_key_per_fold_summary(
        "Fast Key Per-Fold Summary (Rerun)",
        &rerun_summary,
        work_units.chunk_fold_step_count,
    );
    print_key_per_opcode_summary("Fast Key Per-Opcode Summary (Rerun)", &rerun_summary, opcode_count);
    print_per_opcode_components("Fast Per-Opcode Components (Rerun)", &rerun_summary, opcode_count);
    print_section("Full-Only Extra Per-Opcode");
    print_kv("live_shape", format_ms_per_opcode(live_shape_ms, opcode_count));
    print_kv("shape_synth", format_ms_per_opcode(synth_ms, opcode_count));
}
