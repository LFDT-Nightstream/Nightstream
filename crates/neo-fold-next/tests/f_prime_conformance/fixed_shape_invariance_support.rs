use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices_single_step,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv64imChunkStepIvcRelation,
    Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_accepted_proof_with_options, Rv64imProofInput,
    Rv64imPublicProofOptions,
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

static SINGLE_STEP_BACKEND_BUNDLE: LazyLock<(
    Rv64imMainRecursionStepSpartanShape,
    Vec<Rv64imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&SINGLE_STEP_RELATIONS)
        .expect("build single-step native F' advices");
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &SINGLE_STEP_RELATIONS,
        &advices,
    )
    .expect("build single-step recursive-step backend relations")
});

pub fn single_step_spartan_shape() -> &'static Rv64imMainRecursionStepSpartanShape {
    &SINGLE_STEP_BACKEND_BUNDLE.0
}

pub fn single_step_backend_relations() -> &'static [Rv64imMainRecursionFPrimeBackendRelation] {
    &SINGLE_STEP_BACKEND_BUNDLE.1
}

static TWO_STEP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove two-step accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove two-step final statement");
    build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build two-step chunk-step relations")
});

static TWO_STEP_BACKEND_BUNDLE: LazyLock<(
    Rv64imMainRecursionStepSpartanShape,
    Vec<Rv64imMainRecursionFPrimeBackendRelation>,
)> = LazyLock::new(|| {
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&TWO_STEP_RELATIONS)
        .expect("build two-step native F' advices");
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&TWO_STEP_RELATIONS, &advices)
        .expect("build two-step recursive-step backend relations")
});

pub fn two_step_backend_relations() -> &'static [Rv64imMainRecursionFPrimeBackendRelation] {
    &TWO_STEP_BACKEND_BUNDLE.1
}

pub fn two_step_spartan_shape() -> &'static Rv64imMainRecursionStepSpartanShape {
    &TWO_STEP_BACKEND_BUNDLE.0
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
