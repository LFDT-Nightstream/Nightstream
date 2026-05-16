//! Shared fixture for HyperNova Construction-2 F' conformance tests.
//!
//! Builds single-step F' advices once and caches them behind a `LazyLock`, so
//! each conformance test pays the mixed-opcode n=1 fixture cost once.

use std::sync::LazyLock;

use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv32imChunkStepIvcRelation,
    Rv32imMainRecursionFPrimeBackendRelation, Rv32imMainRecursionStepSpartanShape,
};
use neo_fold_prototype::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_prototype::rv32im::{
    build_mixed_opcode_perf_source_case, prove_rv32im_accepted_proof_with_options, Rv32imMainRecursionFPrimeAdvice,
    Rv32imProofInput, Rv32imPublicProofOptions,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv32im_main_recursion_construction2_f_prime_ccs_shape,
};

static SINGLE_STEP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(1);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv32im_accepted_proof_with_options(&input, options).expect("prove single-step accepted artifact");
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement");
    build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations")
});

static SINGLE_STEP_ADVICES: LazyLock<Vec<Rv32imMainRecursionFPrimeAdvice>> = LazyLock::new(|| {
    build_rv32im_main_recursion_f_prime_advices_single_step(&SINGLE_STEP_RELATIONS)
        .expect("build single-step native F' advices")
});

pub fn single_step_advices() -> &'static [Rv32imMainRecursionFPrimeAdvice] {
    &SINGLE_STEP_ADVICES
}

pub fn default_full_width_from_advice(advice: &Rv32imMainRecursionFPrimeAdvice) -> usize {
    let shape = build_rv32im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))
        .expect("derive explicit native F' shape");
    build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape(&shape)
        .expect("derive explicit default width from native shape")
}

static FAST_STRUCTURAL_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(0);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv32im_accepted_proof_with_options(&input, options).expect("prove fast structural accepted artifact");
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove fast structural final statement");
    build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build fast structural chunk-step relations")
});

static FAST_STRUCTURAL_BACKEND_BUNDLE: LazyLock<(
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv32im_main_recursion_f_prime_advices_single_step(&FAST_STRUCTURAL_RELATIONS)
        .expect("build fast structural recursive-step advices");
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &FAST_STRUCTURAL_RELATIONS,
        &advices,
    )
    .expect("build fast structural recursive-step backend relations")
});

pub fn fast_structural_backend_relations() -> &'static [Rv32imMainRecursionFPrimeBackendRelation] {
    &FAST_STRUCTURAL_BACKEND_BUNDLE.1
}

pub fn fast_structural_spartan_shape() -> &'static Rv32imMainRecursionStepSpartanShape {
    &FAST_STRUCTURAL_BACKEND_BUNDLE.0
}
