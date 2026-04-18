//! HyperNova §6.2 Def. 12: the native `u_perp` width must be a pure function
//! of `(pp, s)`, not of an honest relation chain.
//!
//! This test compares the structural canonical builder against two independent
//! relation-derived cover builders. All three widths must match exactly.

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{build_rv64im_chunk_step_ivc_relations, rv64im_chunk_step_ivc_initial_state};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::main_recursion::build_rv64im_main_recursion_verifier_key_fs;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact,
    build_rv64im_main_recursion_construction2_canonical_full_width,
    build_rv64im_main_recursion_construction2_default_full_width_from_relations,
    prove_rv64im_public_proof_with_options, Rv64imMainRecursionPhiSide, Rv64imProofInput, Rv64imPublicProofOptions,
};

fn relation_derived_full_width_for_perf_case(perf_n: usize) -> usize {
    let source = build_mixed_opcode_perf_source_case(perf_n);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options)
        .unwrap_or_else(|err| panic!("n={perf_n}: prove rv64im public proof: {err}"));
    let accepted = build_rv64im_accepted_proof_artifact(&public_proof)
        .unwrap_or_else(|err| panic!("n={perf_n}: build accepted proof artifact: {err}"));
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&accepted)
        .unwrap_or_else(|err| panic!("n={perf_n}: prove final statement: {err}"));
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .unwrap_or_else(|err| panic!("n={perf_n}: build chunk-step relations: {err}"));
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()
        .unwrap_or_else(|err| panic!("n={perf_n}: build canonical vk_fs: {err}"));
    let initial_state = rv64im_chunk_step_ivc_initial_state();
    build_rv64im_main_recursion_construction2_default_full_width_from_relations(
        &vk_fs,
        &initial_state,
        &relations,
        &Rv64imMainRecursionPhiSide::zero(),
    )
    .unwrap_or_else(|err| panic!("n={perf_n}: relation-derived canonical F' full_width: {err}"))
}

#[test]
fn u_perp_full_width_is_canonical_across_independent_traces() {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs().expect("build canonical vk_fs");
    let canonical =
        build_rv64im_main_recursion_construction2_canonical_full_width(&vk_fs, &Rv64imMainRecursionPhiSide::zero())
            .expect("derive structural canonical full_width");
    let n2 = relation_derived_full_width_for_perf_case(2);
    let n1 = relation_derived_full_width_for_perf_case(1);

    assert_eq!(
        canonical, n2,
        "HyperNova Def. 12 violation: structural canonical full_width {canonical} disagrees with the n=2 relation-derived width {n2}"
    );
    assert_eq!(
        canonical, n1,
        "HyperNova Def. 12 violation: structural canonical full_width {canonical} disagrees with the n=1 relation-derived width {n1}"
    );
}
