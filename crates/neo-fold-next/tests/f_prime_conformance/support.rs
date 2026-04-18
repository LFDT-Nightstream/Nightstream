//! Shared fixture for HyperNova Construction-2 F' conformance tests.
//!
//! Builds single-step F' advices once and caches them behind a `LazyLock`, so
//! each conformance test pays the mixed-opcode n=1 fixture cost once.

use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices_single_step,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    rv64im_chunk_step_ivc_initial_state, Rv64imChunkStepIvcRelation, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_accepted_proof_with_options, Rv64imMainRecursionFPrimeAdvice,
    Rv64imMainRecursionPhiSide, Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv64im_main_recursion_construction2_default_full_width_from_relations,
    build_rv64im_main_recursion_construction2_f_prime_ccs_shape,
};

static SINGLE_STEP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(1);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove single-step accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove final statement");
    build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations")
});

static SINGLE_STEP_ADVICES: LazyLock<Vec<Rv64imMainRecursionFPrimeAdvice>> = LazyLock::new(|| {
    build_rv64im_main_recursion_f_prime_advices_single_step(&SINGLE_STEP_RELATIONS)
        .expect("build single-step native F' advices")
});

pub fn single_step_advices() -> &'static [Rv64imMainRecursionFPrimeAdvice] {
    &SINGLE_STEP_ADVICES
}

pub fn single_step_relations() -> &'static [Rv64imChunkStepIvcRelation] {
    &SINGLE_STEP_RELATIONS
}

pub fn default_full_width_from_advice(advice: &Rv64imMainRecursionFPrimeAdvice) -> usize {
    let shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))
        .expect("derive explicit native F' shape");
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(&shape)
        .expect("derive explicit default width from native shape")
}

pub fn default_full_width_from_relations() -> usize {
    let relations = single_step_relations();
    build_rv64im_main_recursion_construction2_default_full_width_from_relations(
        single_step_advices()[0].verifier_key_fs(),
        &rv64im_chunk_step_ivc_initial_state(),
        relations,
        &Rv64imMainRecursionPhiSide::zero(),
    )
    .expect("derive explicit default width from relation-owned native shape cover")
}

static SINGLE_STEP_BACKEND_BUNDLE: LazyLock<(
    Rv64imMainRecursionStepSpartanShape,
    Vec<Rv64imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        single_step_relations(),
        single_step_advices(),
    )
    .expect("build single-step recursive-step backend relations")
});

pub fn single_step_spartan_shape() -> &'static Rv64imMainRecursionStepSpartanShape {
    &SINGLE_STEP_BACKEND_BUNDLE.0
}

static FAST_STRUCTURAL_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(0);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove fast structural accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove fast structural final statement");
    build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build fast structural chunk-step relations")
});

static FAST_STRUCTURAL_BACKEND_BUNDLE: LazyLock<(
    Rv64imMainRecursionStepSpartanShape,
    Vec<Rv64imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&FAST_STRUCTURAL_RELATIONS)
        .expect("build fast structural recursive-step advices");
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &FAST_STRUCTURAL_RELATIONS,
        &advices,
    )
    .expect("build fast structural recursive-step backend relations")
});

pub fn fast_structural_relations() -> &'static [Rv64imChunkStepIvcRelation] {
    &FAST_STRUCTURAL_RELATIONS
}

pub fn fast_structural_backend_relations() -> &'static [Rv64imMainRecursionFPrimeBackendRelation] {
    &FAST_STRUCTURAL_BACKEND_BUNDLE.1
}

pub fn fast_structural_spartan_shape() -> &'static Rv64imMainRecursionStepSpartanShape {
    &FAST_STRUCTURAL_BACKEND_BUNDLE.0
}
