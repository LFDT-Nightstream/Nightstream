use std::collections::HashMap;
use std::env;
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape;
use neo_fold_prototype::rv32im::audit::{
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError,
    Rv32imMainRecursionStepSpartanShape, Rv32imNamedConstraintDelta, Rv32imTerminalFPrimeCommittedStepShape,
};
use neo_fold_prototype::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_prototype::rv32im::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_advices,
    prove_rv32im_accepted_proof_with_options_and_perf, Rv32imProofInput, Rv32imProofProvePerf,
    Rv32imPublicProofOptions,
};
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
pub(crate) fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub(crate) fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 2,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProbeMode {
    Full,
    FastSummary,
    StageAux,
    ConstraintBreakdown,
    TraceShape,
}

pub(crate) fn probe_mode_from_args() -> ProbeMode {
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

pub(crate) fn root_fold_schedule_from_args() -> FoldSchedule {
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

pub(crate) fn selected_relation_index_from_args(relation_count: usize) -> usize {
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

pub(crate) fn unwrap_accepted_artifact_with_schedule_context<T>(
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

pub(crate) fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

pub(crate) fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:34} {}", label, value);
}

pub(crate) fn print_cumulative_and_delta(label: &str, previous: usize, current: usize) {
    let delta = current.saturating_sub(previous);
    print_kv(label, format!("{current} (+{delta})"));
}

pub(crate) fn print_named_constraint_breakdown(
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

pub(crate) fn print_shape_result(
    shape_only: &Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError>,
    shape_only_ms: f64,
    backend_relation_count: usize,
) {
    print_section("Shape");
    match shape_only {
        Ok(shape_only) => {
            print_kv("shape_only.wall", format!("{shape_only_ms:.3} ms"));
            print_kv("shape_only.num_inputs", shape_only.num_inputs);
            print_kv("shape_only.num_aux", shape_only.num_aux);
            print_kv("shape_only.num_constraints", shape_only.num_constraints);
            print_kv(
                "total_aux_across_all_relations",
                shape_only.num_aux * backend_relation_count,
            );
            print_kv(
                "total_constraints_across_all_relations",
                shape_only.num_constraints * backend_relation_count,
            );
        }
        Err(err) => {
            print_kv("shape_only.wall", format!("{shape_only_ms:.3} ms"));
            print_kv("shape_only.error", err);
        }
    }
}

pub(crate) fn print_terminal_committed_shape(
    terminal_committed_shape: &Rv32imTerminalFPrimeCommittedStepShape,
    terminal_committed_ms: f64,
) {
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

pub(crate) fn packed_bytes_field_count(byte_len: usize) -> usize {
    1 + byte_len.div_ceil(7)
}

pub(crate) fn per_unit(ms: f64, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ms / units as f64
    }
}

pub(crate) fn format_ms_per_named_unit(ms: f64, units: usize, unit_suffix: &str) -> String {
    format!("{ms:.3} ms ({:.4} ms/{unit_suffix})", per_unit(ms, units))
}

pub(crate) fn format_ms_per_opcode(ms: f64, opcode_count: usize) -> String {
    format_ms_per_named_unit(ms, opcode_count, "op")
}

pub(crate) fn projection_digest_field_count(
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

pub(crate) fn accumulator_phi_dec_parent_hash_field_count(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    let parent_commitment_fields = claims
        .first()
        .map(|claim| 1 + claim.c.data.len())
        .unwrap_or(0);
    packed_bytes_field_count(b"neo.fold.next/rv32im/main_recursion_recursive_accumulator_phi_dec_parent/v1".len())
        + 4
        + parent_commitment_fields
}

pub(crate) fn perturb_ce_claim_values(claim: &mut CeClaim<Commitment, F, K>) {
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

pub(crate) fn perturb_ccs_claim_values(claim: &mut CcsClaim<Commitment, F>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if let Some(first) = claim.x.first_mut() {
        *first += F::ONE;
    }
}

pub(crate) fn perturb_ccs_witness_values(witness: &mut CcsWitness<F>) {
    if let Some(first) = witness.w.first_mut() {
        *first += F::ONE;
    }
    if witness.Z.rows() > 0 && witness.Z.cols() > 0 {
        witness.Z[(0, 0)] += F::ONE;
    }
}

pub(crate) fn is_zero_f_slice(values: &[F]) -> bool {
    values.iter().all(|value| *value == F::ZERO)
}

pub(crate) fn count_zero_f_slice(values: &[F]) -> usize {
    values.iter().filter(|value| **value == F::ZERO).count()
}

pub(crate) fn count_zero_commitment_children(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    claims
        .iter()
        .filter(|claim| is_zero_f_slice(&claim.c.data))
        .count()
}

pub(crate) fn count_zero_commitment_words(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    claims
        .iter()
        .map(|claim| count_zero_f_slice(&claim.c.data))
        .sum::<usize>()
}

pub(crate) fn zero_commitment_indices(claims: &[CeClaim<Commitment, F, K>]) -> Vec<usize> {
    claims
        .iter()
        .enumerate()
        .filter_map(|(idx, claim)| is_zero_f_slice(&claim.c.data).then_some(idx))
        .collect()
}

pub(crate) fn is_zero_k_slice(values: &[K]) -> bool {
    values.iter().all(|value| *value == K::ZERO)
}

pub(crate) fn is_zero_y_ring(claim: &CeClaim<Commitment, F, K>) -> bool {
    claim.y_ring.iter().all(|row| is_zero_k_slice(row))
}

pub(crate) fn is_zero_ce_projection(claim: &CeClaim<Commitment, F, K>) -> bool {
    is_zero_f_slice(&claim.c.data)
        && claim.X.as_slice().iter().all(|value| *value == F::ZERO)
        && is_zero_k_slice(&claim.r)
        && claim.y_ring.iter().all(|row| is_zero_k_slice(row))
}

pub(crate) fn toom3_chunk_out_term_counts_current() -> Vec<usize> {
    vec![1; 2 * (D / 3) - 1]
}

pub(crate) fn toom3_chunk_out_term_counts_flattened() -> Vec<usize> {
    let split = D / 3;
    let mut counts = vec![0usize; 2 * split - 1];
    for i in 0..split {
        for j in 0..split {
            counts[i + j] += 1;
        }
    }
    counts
}

pub(crate) fn reduce_phi_81_term_counts(offset_chunk_counts: &[(usize, &[usize])]) -> Vec<usize> {
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

pub(crate) fn toom3_reduced_product_term_counts(chunk_term_counts: &[usize]) -> Vec<usize> {
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

pub(crate) fn add_probe_term(row_terms: &mut HashMap<(u8, usize), F>, term_id: (u8, usize), scale: F) {
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

pub(crate) fn reduce_phi_81_term_maps(
    offset_chunk_scales: &[(usize, u8, F)],
    chunk_len: usize,
) -> Vec<HashMap<(u8, usize), F>> {
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

pub(crate) fn toom3_reduced_product_unique_term_counts_current() -> Vec<usize> {
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

pub(crate) fn print_pi_rlc_public_child_families(
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

pub(crate) fn print_backend_relation_commitment_sparsity(relations: &[Rv32imMainRecursionFPrimeBackendRelation]) {
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

pub(crate) fn perturb_backend_relation_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
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

pub(crate) fn perturb_state_in_r_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(first) = claim.r.first_mut() {
            *first += K::ONE;
        }
    }
}

pub(crate) fn perturb_state_in_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

pub(crate) fn perturb_state_in_projection_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.state_in_claims.first_mut() {
        if let Some(first) = claim.c.data.first_mut() {
            *first += F::ONE;
        }
        if claim.X.rows() > 0 && claim.X.cols() > 0 {
            claim.X[(0, 0)] += F::ONE;
        }
    }
}

pub(crate) fn perturb_pi_ccs_alpha_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(first) = relation.payload.pi_ccs.public_challenges.alpha.first_mut() {
        *first += K::ONE;
    }
}

pub(crate) fn perturb_pi_ccs_gamma_value(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    relation.payload.pi_ccs.public_challenges.gamma += K::ONE;
}

pub(crate) fn perturb_state_out_projection_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
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

pub(crate) fn perturb_pi_ccs_output_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_ccs.ccs_outputs.first_mut() {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

pub(crate) fn perturb_pi_ccs_output_y_zcol_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_ccs.ccs_outputs.first_mut() {
        if let Some(first) = claim.y_zcol.first_mut() {
            *first += K::ONE;
        }
    }
}

pub(crate) fn perturb_pi_dec_child_y_ring_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.pi_dec.children.first_mut() {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

pub(crate) fn perturb_pi_rlc_parent_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
}

pub(crate) fn perturb_fresh_claim_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(claim) = relation.payload.fresh_claims.first_mut() {
        perturb_ccs_claim_values(claim);
    }
}

pub(crate) fn perturb_fresh_witness_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    if let Some(witness) = relation.payload.fresh_witnesses.first_mut() {
        perturb_ccs_witness_values(witness);
    }
}

pub(crate) fn fixed_shape_family_status(
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

pub(crate) fn first_pi_ccs_stage_diff<'a>(
    baseline: &'a neo_fold_prototype::rv32im::audit::Rv32imPiCcsStageFingerprint,
    perturbed: &'a neo_fold_prototype::rv32im::audit::Rv32imPiCcsStageFingerprint,
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
pub(crate) struct FastSummaryPerf {
    pub(crate) fixture_ms: f64,
    pub(crate) accepted_wall_ms: f64,
    pub(crate) accepted_perf: Rv32imProofProvePerf,
    pub(crate) final_statement_ms: f64,
    pub(crate) relations_ms: f64,
    pub(crate) advices_ms: f64,
    pub(crate) backend_relations_ms: f64,
    pub(crate) shape_only_ms: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProbeWorkUnits {
    pub(crate) non_halt_opcode_count: usize,
    pub(crate) semantic_step_count: usize,
    pub(crate) chunk_count: usize,
    pub(crate) chunk_fold_step_count: usize,
    pub(crate) relation_count: usize,
    pub(crate) backend_relation_count: usize,
    pub(crate) fold_schedule: FoldSchedule,
}

pub(crate) fn accepted_root_rlc_dec_ms(accepted_perf: &Rv32imProofProvePerf) -> f64 {
    accepted_perf.root_main_lane.session.rlc_ms()
        + accepted_perf.root_main_lane.session.dec_split_ms()
        + accepted_perf.root_main_lane.session.dec_commit_ms()
        + accepted_perf.root_main_lane.session.dec_ms()
}

pub(crate) fn print_probe_work_units(title: &str, units: ProbeWorkUnits) {
    print_section(title);
    print_kv("non_halt_opcode_count", units.non_halt_opcode_count);
    print_kv("semantic_step_count", units.semantic_step_count);
    print_kv("chunk_count", units.chunk_count);
    print_kv("chunk_fold_step_count", units.chunk_fold_step_count);
    print_kv("relation_count", units.relation_count);
    print_kv("backend_relation_count", units.backend_relation_count);
    print_kv("fold_schedule", format!("{:?}", units.fold_schedule));
}

pub(crate) fn print_key_per_opcode_summary(title: &str, summary: &FastSummaryPerf, opcode_count: usize) {
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

pub(crate) fn print_key_per_fold_summary(title: &str, summary: &FastSummaryPerf, fold_count: usize) {
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

pub(crate) fn print_per_opcode_components(title: &str, summary: &FastSummaryPerf, opcode_count: usize) {
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

pub(crate) fn measure_fast_summary_perf(input: &Rv32imProofInput) -> FastSummaryPerf {
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
