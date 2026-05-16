use std::time::Instant;

use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::{
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown,
};
use neo_fold_prototype::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_prototype::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices, prove_rv32im_accepted_proof_with_options, Rv32imProofInput,
    Rv32imPublicProofOptions,
};

const OPCODE_COUNT: usize = 2;
const MEASURE_REPEATS: usize = 3;

#[test]
#[ignore = "profiling hook for recursive Pi_RLC public shape work"]
fn rv32im_main_recursion_pi_rlc_public_shape_snapshot() {
    let fixture_started = Instant::now();
    let source = build_mixed_opcode_perf_source_case(OPCODE_COUNT);
    let input = Rv32imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let (accepted, _) = prove_rv32im_accepted_proof_with_options(
        &input,
        Rv32imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove accepted artifact");
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement");
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations");
    let advices = build_rv32im_main_recursion_f_prime_advices(&relations).expect("build f-prime advices");
    let (_, backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build recursion backend relations");
    let first_relation = backend_relations
        .first()
        .expect("Pi_RLC public shape snapshot requires at least one backend relation");
    let fixture_ms = fixture_started.elapsed().as_secs_f64() * 1_000.0;

    let measure_started = Instant::now();
    let mut last_breakdown = None;
    for _ in 0..MEASURE_REPEATS {
        let breakdown = debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(first_relation)
            .expect("measure Pi_RLC public constraint breakdown");
        if let Some(previous) = &last_breakdown {
            assert_eq!(
                previous, &breakdown,
                "Pi_RLC public constraint breakdown should be stable across identical repeats"
            );
        }
        last_breakdown = Some(breakdown);
    }
    let measure_ms = measure_started.elapsed().as_secs_f64() * 1_000.0;
    let breakdown = last_breakdown.expect("at least one Pi_RLC public breakdown result");

    println!("opcode_count={OPCODE_COUNT}");
    println!("measure_repeats={MEASURE_REPEATS}");
    println!("relation_count={}", relations.len());
    println!("fixture_ms={fixture_ms:.3}");
    println!("measure_ms={measure_ms:.3}");
    println!("shared_point_constraints={}", breakdown.shared_point_constraints);
    println!("x_constraints={}", breakdown.x_constraints);
    println!("c_constraints={}", breakdown.c_constraints);
    println!("y_ring_constraints={}", breakdown.y_ring_constraints);
    println!("y_zcol_constraints={}", breakdown.y_zcol_constraints);
    println!("aux_constraints={}", breakdown.aux_constraints);
    println!("total_constraints={}", breakdown.total_constraints);
}
