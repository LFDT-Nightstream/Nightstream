use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_advices,
    build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv32imChunkStepIvcRelation,
    Rv32imMainRecursionFPrimeBackendRelation, Rv32imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, prove_rv32im_accepted_proof_with_options, Rv32imProofInput,
    Rv32imPublicProofOptions,
};

fn build_relations(opcode_count: usize, schedule: FoldSchedule, label: &str) -> Vec<Rv32imChunkStepIvcRelation> {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: schedule,
    };
    let (accepted, _) = prove_rv32im_accepted_proof_with_options(&input, options)
        .unwrap_or_else(|err| panic!("prove {label} accepted artifact: {err}"));
    let (final_statement, final_proof) = prove_rv32im_final_statement_from_accepted(&accepted)
        .unwrap_or_else(|err| panic!("prove {label} final statement: {err}"));
    build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .unwrap_or_else(|err| panic!("build {label} chunk-step relations: {err}"))
}

fn build_rows_per_chunk_fixture_with_short_terminal(step_cap: usize, label: &str) -> Vec<Rv32imChunkStepIvcRelation> {
    for opcode_count in step_cap..=(step_cap * 4) {
        let relations = build_relations(
            opcode_count,
            FoldSchedule::RowsPerChunk(step_cap),
            &format!("{label}-n{opcode_count}"),
        );
        if relations.len() >= 2
            && relations.last().is_some_and(|relation| {
                relation.witness.terminal_step && relation.statement.chunk_summary.public_step_count < step_cap as u64
            })
        {
            return relations;
        }
    }
    panic!("{label}: could not find a RowsPerChunk({step_cap}) mixed-opcode fixture with a short terminal chunk");
}

static SINGLE_STEP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(1, FoldSchedule::RowsPerChunk(1), "single-step"));

static SINGLE_STEP_BACKEND_BUNDLE: LazyLock<(
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv32im_main_recursion_f_prime_advices_single_step(&SINGLE_STEP_RELATIONS)
        .expect("build single-step native F' advices");
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &SINGLE_STEP_RELATIONS,
        &advices,
    )
    .expect("build single-step recursive-step backend relations")
});

pub fn single_step_spartan_shape() -> &'static Rv32imMainRecursionStepSpartanShape {
    &SINGLE_STEP_BACKEND_BUNDLE.0
}

pub fn single_step_backend_relations() -> &'static [Rv32imMainRecursionFPrimeBackendRelation] {
    &SINGLE_STEP_BACKEND_BUNDLE.1
}

static TWO_STEP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(2, FoldSchedule::RowsPerChunk(1), "two-step"));

static TWO_STEP_BACKEND_BUNDLE: LazyLock<(
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv32im_main_recursion_f_prime_advices_single_step(&TWO_STEP_RELATIONS)
        .expect("build two-step native F' advices");
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&TWO_STEP_RELATIONS, &advices)
        .expect("build two-step recursive-step backend relations")
});

pub fn two_step_backend_relations() -> &'static [Rv32imMainRecursionFPrimeBackendRelation] {
    &TWO_STEP_BACKEND_BUNDLE.1
}

pub fn two_step_spartan_shape() -> &'static Rv32imMainRecursionStepSpartanShape {
    &TWO_STEP_BACKEND_BUNDLE.0
}

pub fn two_step_relations() -> &'static [Rv32imChunkStepIvcRelation] {
    &TWO_STEP_RELATIONS
}

static FIVE_STEP_CAP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_rows_per_chunk_fixture_with_short_terminal(5, "five-step-cap"));

static FIVE_STEP_CAP_BACKEND_BUNDLE: LazyLock<(
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv32im_main_recursion_f_prime_advices(&FIVE_STEP_CAP_RELATIONS)
        .expect("build five-step-cap native F' advices");
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &FIVE_STEP_CAP_RELATIONS,
        &advices,
    )
    .expect("build five-step-cap recursive-step backend relations")
});

pub fn five_step_cap_backend_relations() -> &'static [Rv32imMainRecursionFPrimeBackendRelation] {
    &FIVE_STEP_CAP_BACKEND_BUNDLE.1
}

pub fn five_step_cap_spartan_shape() -> &'static Rv32imMainRecursionStepSpartanShape {
    &FIVE_STEP_CAP_BACKEND_BUNDLE.0
}
