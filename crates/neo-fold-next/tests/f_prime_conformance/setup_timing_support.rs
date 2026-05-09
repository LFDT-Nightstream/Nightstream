use std::sync::LazyLock;

use neo_fold_next::core::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv32imChunkStepIvcRelation,
    Rv32imMainRecursionFPrimeBackendRelation, Rv32imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, prove_rv32im_accepted_proof_with_options, Rv32imProofInput,
    Rv32imPublicProofOptions,
};

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
