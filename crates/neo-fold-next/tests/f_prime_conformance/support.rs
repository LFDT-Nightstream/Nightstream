//! Shared fixture for HyperNova Construction-2 F' conformance tests.
//!
//! Builds single-step F' advices once and caches them behind a `LazyLock`, so
//! each conformance test pays the mixed-opcode n=2 fixture cost once.

use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices_single_step,
    Rv64imChunkStepIvcRelation,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, prove_rv64im_public_proof_with_options,
    Rv64imMainRecursionFPrimeAdvice, Rv64imMainRecursionPhiSide, Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv64im_main_recursion_construction2_default_full_width_from_relations,
    build_rv64im_main_recursion_construction2_f_prime_ccs_shape,
};

static SINGLE_STEP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove single-step rv64im public proof");
    let accepted = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
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
    let advices = single_step_advices();
    build_rv64im_main_recursion_construction2_default_full_width_from_relations(
        advices[0].verifier_key_fs(),
        advices[0].running_state(),
        single_step_relations(),
        &Rv64imMainRecursionPhiSide::zero(),
    )
    .expect("derive explicit default width from relation-owned native shape cover")
}
