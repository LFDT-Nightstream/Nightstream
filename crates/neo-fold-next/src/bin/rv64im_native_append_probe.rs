use std::{env, time::Instant};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices_single_step,
    debug_trace_rv64im_main_recursion_f_prime_advices_single_step_build,
};
use neo_fold_next::rv64im::construction2::build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc::{
    derive_rv64im_ivc_step_cap, Rv64imIvcAppendPerf, Rv64imIvcState, Rv64imIvcVerifyPerf,
};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_claim_digests,
    build_rv64im_main_recursion_construction2_canonical_full_width, prove_rv64im_accepted_proof_with_options,
    rv64im_claim_tree_opening_from_digests, rv64im_claim_tree_root_from_digests, verify_rv64im_claim_tree_opening,
    Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn root_fold_schedule_from_args() -> FoldSchedule {
    let mut schedule = FoldSchedule::RowsPerChunk(1);
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--whole-trace" => schedule = FoldSchedule::WholeTrace,
            "--rows-per-chunk-1" => schedule = FoldSchedule::RowsPerChunk(1),
            other => panic!("unknown arg: {other}"),
        }
    }
    schedule
}

fn unwrap_accepted_artifact_with_schedule_context<T>(
    result: Result<T, impl std::fmt::Display>,
    root_fold_schedule: FoldSchedule,
) -> T {
    result.unwrap_or_else(|err| match root_fold_schedule {
        FoldSchedule::WholeTrace => {
            panic!(
                "prove accepted artifact: WholeTrace overflowed the live RV64IM DEC/k_rho budget; retry with --rows-per-chunk-1.\nunderlying error: {err}"
            )
        }
        FoldSchedule::RowsPerChunk(_) => panic!("prove accepted artifact: {err}"),
    })
}

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 2,
    }
}

fn per_unit(ms: f64, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ms / units as f64
    }
}

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:30} {}", label, value);
}

fn k_coeff_count() -> usize {
    K::ZERO.as_coeffs().len()
}

fn count_commitment_bits(commitment: &Commitment) -> usize {
    commitment.data.len() * 64
}

fn count_f_slice_bits(values: &[F]) -> usize {
    values.len() * 64
}

fn count_f_matrix_bits(values: &neo_ccs::Mat<F>) -> usize {
    values.rows() * values.cols() * 64
}

fn count_compact_x_bits(claim: &CeClaim<Commitment, F, K>) -> usize {
    claim.m_in * 64
}

fn count_k_slice_bits(values: &[K]) -> usize {
    values.len() * k_coeff_count() * 64
}

fn count_k_rows_bits(rows: &[Vec<K>]) -> usize {
    rows.iter().map(|row| count_k_slice_bits(row)).sum()
}

fn pack_binary_columns(bits: &[F]) -> Vec<u64> {
    let cols = bits.len().div_ceil(D);
    let mut column_bits = vec![0u64; cols];
    for (idx, bit) in bits.iter().enumerate() {
        if *bit != F::ZERO {
            let col = idx / D;
            let row = idx % D;
            column_bits[col] |= 1u64 << row;
        }
    }
    column_bits
}

fn count_nonzero_bits(bits: &[F]) -> usize {
    bits.iter().filter(|bit| **bit != F::ZERO).count()
}

fn count_ccs_claim_bits(claim: &CcsClaim<Commitment, F>) -> usize {
    count_commitment_bits(&claim.c) + count_f_slice_bits(&claim.x)
}

fn count_ccs_witness_bits(witness: &CcsWitness<F>) -> usize {
    count_f_slice_bits(&witness.w) + count_f_matrix_bits(&witness.Z)
}

fn count_state_in_claim_bundle_bits(claims: &[CeClaim<Commitment, F, K>]) -> usize {
    let Some((first, rest)) = claims.split_first() else {
        return 0;
    };
    debug_assert!(rest.iter().all(|claim| claim.r == first.r));
    count_k_slice_bits(&first.r)
        + claims
            .iter()
            .map(|claim| {
                count_commitment_bits(&claim.c) + count_compact_x_bits(claim) + count_k_rows_bits(&claim.y_ring)
            })
            .sum::<usize>()
}

fn count_claim_merkle_path_bits(path_len: usize) -> usize {
    path_len * 256
}

fn is_zero_f_matrix(values: &neo_ccs::Mat<F>) -> bool {
    for row in 0..values.rows() {
        for col in 0..values.cols() {
            if values[(row, col)] != F::ZERO {
                return false;
            }
        }
    }
    true
}

fn is_zero_k_rows(rows: &[Vec<K>]) -> bool {
    rows.iter()
        .all(|row| row.iter().all(|value| *value == K::ZERO))
}

fn is_zero_ce_claim(claim: &CeClaim<Commitment, F, K>) -> bool {
    claim.c.data.iter().all(|word| *word == F::ZERO)
        && is_zero_f_matrix(&claim.X)
        && claim.r.iter().all(|value| *value == K::ZERO)
        && claim.s_col.iter().all(|value| *value == K::ZERO)
        && is_zero_k_rows(&claim.y_ring)
        && claim.ct.iter().all(|value| *value == K::ZERO)
        && claim.aux_openings.iter().all(|value| *value == K::ZERO)
        && claim.y_zcol.iter().all(|value| *value == K::ZERO)
        && claim.fold_digest == [0; 32]
        && claim.c_step_coords.iter().all(|value| *value == F::ZERO)
        && claim.u_offset == 0
        && claim.u_len == 0
}

fn print_rows(title: &str, rows: &[(&str, f64)], total_ms: f64, opcode_count: usize) {
    print_section(title);
    for (label, ms) in rows {
        let pct = if total_ms == 0.0 { 0.0 } else { (ms / total_ms) * 100.0 };
        print_kv(
            label,
            format!("{ms:.3} ms ({:.4} ms/op, {pct:.1}%)", per_unit(*ms, opcode_count)),
        );
    }
}

fn append_stage_rows(perfs: &[Rv64imIvcAppendPerf]) -> Vec<(&'static str, f64)> {
    let mut total = Rv64imIvcAppendPerf::default();
    for perf in perfs {
        total.validate_state_surface_ms += perf.validate_state_surface_ms;
        total.validate_relation_surface_ms += perf.validate_relation_surface_ms;
        total.validate_next_relation_surface_ms += perf.validate_next_relation_surface_ms;
        total.verified_step_statement_ms += perf.verified_step_statement_ms;
        total.fixed_shape_chunk_summary_ms += perf.fixed_shape_chunk_summary_ms;
        total.main_circuit_trace_ms += perf.main_circuit_trace_ms;
        total.construction2_pi_fold_ms += perf.construction2_pi_fold_ms;
        total.advice_build_ms += perf.advice_build_ms;
        total.evaluate_f_prime_ms += perf.evaluate_f_prime_ms;
        total.finalize_state_ms += perf.finalize_state_ms;
    }
    vec![
        ("validate_state_surface", total.validate_state_surface_ms),
        ("validate_relation_surface", total.validate_relation_surface_ms),
        ("validate_next_relation", total.validate_next_relation_surface_ms),
        ("verified_step_statement", total.verified_step_statement_ms),
        ("fixed_shape_chunk_summary", total.fixed_shape_chunk_summary_ms),
        ("main_circuit_trace", total.main_circuit_trace_ms),
        ("construction2_pi_fold", total.construction2_pi_fold_ms),
        ("advice_build", total.advice_build_ms),
        ("evaluate_f_prime", total.evaluate_f_prime_ms),
        ("finalize_state", total.finalize_state_ms),
    ]
}

fn verify_stage_rows(perf: Rv64imIvcVerifyPerf) -> Vec<(&'static str, f64)> {
    vec![
        ("validate_state_surface", perf.validate_state_surface_ms),
        ("build_terminal_relation", perf.build_terminal_relation_ms),
        ("verified_step_statement", perf.verified_step_statement_ms),
        ("context_lookup", perf.context_lookup_ms),
        ("replay_step", perf.replay_step_ms),
        ("compare_running_state", perf.compare_running_state_ms),
        ("transcript_snapshot", perf.transcript_snapshot_ms),
        ("compare_step_public", perf.compare_step_public_ms),
    ]
}

fn main() {
    let opcode_count = perf_opcode_count_from_env();
    let schedule = root_fold_schedule_from_args();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };

    let fixture_started = Instant::now();
    let (accepted, _) = unwrap_accepted_artifact_with_schedule_context(
        prove_rv64im_accepted_proof_with_options(
            &input,
            Rv64imPublicProofOptions {
                root_fold_schedule: schedule,
            },
        ),
        schedule,
    );
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations");
    let fixture_ms = millis_since(fixture_started);

    let semantic_step_count = relations
        .last()
        .map(|relation| relation.statement.step_public.step_hi as usize)
        .expect("native append probe requires at least one relation");
    let step_cap = derive_rv64im_ivc_step_cap(schedule, semantic_step_count).expect("derive native step_cap");

    let append_started = Instant::now();
    let mut state = Rv64imIvcState::init_with_step_cap(step_cap).expect("build initial ivc state");
    let mut append_perfs = Vec::with_capacity(relations.len());
    for relation in &relations {
        let (next, perf) = state
            .append_with_perf(relation)
            .expect("append native ivc relation");
        append_perfs.push(perf);
        state = next;
    }
    let append_ms = millis_since(append_started);

    let verify_started = Instant::now();
    let verify_perf = state.verify_with_perf().expect("verify native ivc state");
    let verify_ms = millis_since(verify_started);

    print_section("RV64IM Native Append Probe");
    print_kv("mixed_opcode_non_halt_ops", opcode_count);
    print_kv("fold_schedule", format!("{schedule:?}"));
    print_kv("relation_count", relations.len());
    print_kv("step_cap", step_cap);
    print_kv("fixture_prep", format!("{fixture_ms:.3} ms"));
    print_kv(
        "append_total",
        format!("{append_ms:.3} ms ({:.4} ms/op)", per_unit(append_ms, opcode_count)),
    );
    print_kv(
        "verify_total",
        format!("{verify_ms:.3} ms ({:.4} ms/op)", per_unit(verify_ms, opcode_count)),
    );

    print_rows(
        "Append Breakdown",
        &append_stage_rows(&append_perfs),
        append_ms,
        opcode_count,
    );
    print_rows(
        "Verify Breakdown",
        &verify_stage_rows(verify_perf),
        verify_ms,
        opcode_count,
    );

    let single_step_relations_only = relations
        .iter()
        .all(|relation| relation.statement.chunk_summary.public_step_count == 1);
    if !single_step_relations_only {
        print_section("Construction2 Witness Width");
        print_kv(
            "skipped",
            "single-step witness sizing path only applies when each recursive relation carries exactly one public step",
        );
        print_section("F' Advice Build");
        print_kv(
            "skipped",
            "single-step advice trace only applies when each recursive relation carries exactly one public step",
        );
        return;
    }

    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&relations)
        .expect("build native f-prime advices for witness sizing");
    let canonical_full_width = build_rv64im_main_recursion_construction2_canonical_full_width(
        advices[0].verifier_key_fs(),
        advices[0].phi_side(),
    )
    .expect("derive canonical construction2 full width");
    print_section("Construction2 Witness Width");
    print_kv("canonical_full_width_bits", canonical_full_width);
    for (step_index, advice) in advices.iter().enumerate() {
        let low_norm = build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(
            advice,
            advice
                .construction2_input_fresh_instance()
                .expect("f-prime advice carries construction2 input fresh instance"),
        )
        .expect("build construction2 low-norm witness image");
        let x_i_bits = advice.x_i().field_image();
        let used_bits = x_i_bits.len() + low_norm.binary_values().len();
        let fill_pct = (used_bits as f64 / canonical_full_width as f64) * 100.0;
        let ones = count_nonzero_bits(&x_i_bits) + count_nonzero_bits(low_norm.binary_values());
        let mut full_bits = Vec::with_capacity(used_bits);
        full_bits.extend_from_slice(&x_i_bits);
        full_bits.extend_from_slice(low_norm.binary_values());
        let packed_columns = pack_binary_columns(&full_bits);
        let nonzero_columns = packed_columns.iter().filter(|mask| **mask != 0).count();
        let total_column_popcount: usize = packed_columns
            .iter()
            .map(|mask| mask.count_ones() as usize)
            .sum();
        let max_column_popcount = packed_columns
            .iter()
            .map(|mask| mask.count_ones())
            .max()
            .unwrap_or(0);
        let dense_columns_16 = packed_columns
            .iter()
            .filter(|mask| mask.count_ones() >= 16)
            .count();
        let dense_columns_27 = packed_columns
            .iter()
            .filter(|mask| mask.count_ones() >= 27)
            .count();
        print_kv(
            &format!("step_{step_index}_used_bits"),
            format!("{used_bits} ({fill_pct:.1}%)"),
        );
        print_kv(
            &format!("step_{step_index}_one_bits"),
            format!("{} ({:.1}% density)", ones, (ones as f64 / used_bits as f64) * 100.0),
        );
        print_kv(
            &format!("step_{step_index}_column_popcount"),
            format!(
                "nonzero={} avg={:.2} max={} ge16={} ge27={}",
                nonzero_columns,
                total_column_popcount as f64 / packed_columns.len() as f64,
                max_column_popcount,
                dense_columns_16,
                dense_columns_27,
            ),
        );
        let state_in_claims = advice.running_state().carry.main.claims.as_slice();
        let state_in_claims_bits = count_state_in_claim_bundle_bits(state_in_claims);
        let claim_digests = build_rv64im_claim_digests(state_in_claims);
        let claim_tree_root = rv64im_claim_tree_root_from_digests(&claim_digests);
        let claim_opening =
            rv64im_claim_tree_opening_from_digests(&claim_digests, 0).expect("build claim-tree opening for slot 0");
        assert!(
            verify_rv64im_claim_tree_opening(claim_tree_root, claim_digests[0], &claim_opening),
            "claim-tree opening must verify against its root"
        );
        let shared_state_in_bits = if let Some(first) = state_in_claims.first() {
            count_k_slice_bits(&first.r)
        } else {
            0
        };
        let per_claim_state_in_bits = state_in_claims
            .first()
            .map(|claim| {
                count_commitment_bits(&claim.c) + count_compact_x_bits(claim) + count_k_rows_bits(&claim.y_ring)
            })
            .unwrap_or(0);
        let claim_path_bits = count_claim_merkle_path_bits(claim_opening.siblings().len());
        let zero_like_claim_count = state_in_claims
            .iter()
            .filter(|claim| is_zero_ce_claim(claim))
            .count();
        let nondefault_claim_count = state_in_claims.len().saturating_sub(zero_like_claim_count);
        let externalized_single_claim_bits_estimate =
            count_state_in_claim_bundle_bits(&state_in_claims[..1]) + claim_path_bits;
        let externalized_all_claims_bits_estimate =
            shared_state_in_bits + state_in_claims.len() * (per_claim_state_in_bits + claim_path_bits) + 256;
        let externalized_nondefault_claim_bits_estimate =
            shared_state_in_bits + nondefault_claim_count * (per_claim_state_in_bits + claim_path_bits) + 256;
        let fixed_header_bits = 64 + 256 + 256 + 64;
        let phi_side_bits: usize = advice
            .phi_side()
            .commitment_words()
            .iter()
            .map(|words| words.len() * 64)
            .sum();
        let current_input_bits = count_commitment_bits(
            advice
                .construction2_input_fresh_instance()
                .unwrap()
                .commitment()
                .commitment(),
        ) + advice.x_i().field_image().len();
        let chunk_input_bits: usize = relations[step_index]
            .witness
            .handoff
            .chunk_input
            .steps
            .iter()
            .map(|step| count_ccs_claim_bits(&step.mcs) + count_ccs_witness_bits(&step.witness))
            .sum::<usize>()
            + 128;
        let replay_witness = &relations[step_index].witness.replay_witness;
        let pi_ccs_output_reference_bits: usize = replay_witness
            .ccs_outputs
            .iter()
            .map(|claim| count_k_rows_bits(&claim.y_ring) + count_k_slice_bits(&claim.y_zcol))
            .sum();
        let pi_fold_replay_bits: usize = replay_witness
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .chain(replay_witness.ccs_replay_proof.sumcheck_rounds_nc.iter())
            .map(|round| count_k_slice_bits(round))
            .sum();
        let state_out_claim_reference_bits: usize = relations[step_index]
            .witness
            .state_out
            .carry
            .main
            .claims
            .iter()
            .map(|claim| count_commitment_bits(&claim.c) + count_k_rows_bits(&claim.y_ring))
            .sum();
        let actual_pi_fold_bits = pi_fold_replay_bits;
        print_kv(&format!("step_{step_index}_fixed_header_bits"), fixed_header_bits);
        print_kv(&format!("step_{step_index}_phi_side_bits"), phi_side_bits);
        print_kv(&format!("step_{step_index}_state_in_claims_bits"), state_in_claims_bits);
        print_kv(
            &format!("step_{step_index}_externalized_single_claim_bits_estimate"),
            externalized_single_claim_bits_estimate,
        );
        print_kv(
            &format!("step_{step_index}_externalized_all_claims_bits_estimate"),
            externalized_all_claims_bits_estimate,
        );
        print_kv(
            &format!("step_{step_index}_externalized_nondefault_claim_bits_estimate"),
            externalized_nondefault_claim_bits_estimate,
        );
        print_kv(&format!("step_{step_index}_current_input_bits"), current_input_bits);
        print_kv(&format!("step_{step_index}_chunk_input_bits"), chunk_input_bits);
        print_kv(&format!("step_{step_index}_actual_pi_fold_bits"), actual_pi_fold_bits);
        print_kv(
            &format!("step_{step_index}_claim_tree_path"),
            format!(
                "depth={} auth_path_bits={} root_bits=256 zero_like_claims={} nondefault_claims={}",
                claim_opening.siblings().len(),
                claim_path_bits,
                zero_like_claim_count,
                nondefault_claim_count,
            ),
        );
        print_kv(
            &format!("step_{step_index}_reference_replay_surface_bits"),
            format!(
                "pi_ccs_outputs={} state_out_claims={}",
                pi_ccs_output_reference_bits, state_out_claim_reference_bits
            ),
        );
        if let Some(claim) = advice.running_state().carry.main.claims.first() {
            print_kv(
                &format!("step_{step_index}_state_in_claim_split"),
                format!(
                    "shared_r={} per_claim(commit={} x={} y_ring={})",
                    count_k_slice_bits(&claim.r),
                    count_commitment_bits(&claim.c),
                    count_compact_x_bits(claim),
                    count_k_rows_bits(&claim.y_ring),
                ),
            );
            print_kv(
                &format!("step_{step_index}_externalized_state_in_split"),
                format!(
                    "shared={} per_claim={} auth_path={} root=256",
                    shared_state_in_bits, per_claim_state_in_bits, claim_path_bits
                ),
            );
            print_kv(
                &format!("step_{step_index}_externalized_state_in_modes"),
                format!(
                    "single_open={} all_opens={} nondefault_opens={}",
                    externalized_single_claim_bits_estimate,
                    externalized_all_claims_bits_estimate,
                    externalized_nondefault_claim_bits_estimate
                ),
            );
        }
        if let Some(claim) = replay_witness.ccs_outputs.first() {
            print_kv(
                &format!("step_{step_index}_pi_ccs_output_reference_split"),
                format!(
                    "y_ring={} y_zcol={}",
                    count_k_rows_bits(&claim.y_ring),
                    count_k_slice_bits(&claim.y_zcol),
                ),
            );
        }
        if let Some(claim) = relations[step_index]
            .witness
            .state_out
            .carry
            .main
            .claims
            .first()
        {
            print_kv(
                &format!("step_{step_index}_state_out_claim_reference_split"),
                format!(
                    "commit={} y_ring={}",
                    count_commitment_bits(&claim.c),
                    count_k_rows_bits(&claim.y_ring),
                ),
            );
        }
    }

    let trace_started = Instant::now();
    let (_, f_prime_build_perf) =
        debug_trace_rv64im_main_recursion_f_prime_advices_single_step_build(&relations, "probe.f_prime")
            .expect("trace native f-prime advice build");
    let f_prime_trace_ms = millis_since(trace_started);

    print_section("F' Advice Build");
    print_kv("trace_total", format!("{f_prime_trace_ms:.3} ms"));
    print_kv("verifier_key", format!("{:.3} ms", f_prime_build_perf.verifier_key_ms));
    print_kv(
        "canonical_full_width",
        format!("{:.3} ms", f_prime_build_perf.canonical_full_width_ms),
    );
    print_kv(
        "canonical_u_perp",
        format!("{:.3} ms", f_prime_build_perf.canonical_u_perp_ms),
    );
    print_kv("total", format!("{:.3} ms", f_prime_build_perf.total_ms));

    print_section("F' Advice Steps");
    for (step_index, perf) in f_prime_build_perf.per_step.iter().enumerate() {
        println!(
            "  step {:>3}  build_advice={:>8.3} ms  evaluate_step={:>8.3} ms  apply_step={:>8.3} ms",
            step_index, perf.build_advice_ms, perf.evaluate_step_ms, perf.apply_step_image_ms,
        );
    }
}
