use std::sync::OnceLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv64im_main_recursion_x_out_gadget_parity, evaluate_rv64im_main_recursion_f_prime_advice,
    Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, prove_rv64im_public_proof_with_options,
    Rv64imProofInput, Rv64imPublicProofOptions,
};

pub fn assert_backend_relation_exact_surface_contract(
    relation: &Rv64imMainRecursionFPrimeBackendRelation,
    label: &str,
) {
    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(&relation.f_prime_advice)
        .unwrap_or_else(|err| panic!("{label}: native F' advice should evaluate successfully: {err}"));
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native(relation).unwrap_or_else(|err| {
        panic!("{label}: exact-step payload should reconstruct the native chunk replay trace: {err}")
    });
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity(relation).unwrap_or_else(|err| {
        panic!("{label}: live carried claims should hash to the authoritative native ME digests: {err}")
    });
    debug_check_rv64im_main_recursion_x_out_gadget_parity(relation)
        .unwrap_or_else(|err| panic!("{label}: x_out gadget should match the canonical native F' image: {err}"));
    assert_eq!(
        relation.spartan_statement.folded_accumulator_digest,
        step_image.folded_accumulator_digest(),
        "{label}: per-step Spartan statement folded accumulator digest drifted from the authoritative native F' step image"
    );
    assert_eq!(
        relation.payload.fixed_transcript_out(),
        &relation.f_prime_advice.fresh_state_out().transcript,
        "{label}: fixed recursive-step payload transcript drifted from the carried native state_out transcript"
    );
    let state_out_count = relation.payload.step_shape.state_out_claim_count as usize;
    let child_count = relation.payload.step_shape.child_count as usize;
    assert_eq!(
        state_out_count, child_count,
        "{label}: padded payload must carry exactly the replayed child claims"
    );
    for (idx, (state_out, child)) in relation
        .payload
        .state_out_claims
        .iter()
        .take(state_out_count)
        .zip(relation.payload.pi_dec.children.iter().take(child_count))
        .enumerate()
    {
        assert_eq!(
            state_out, child,
            "{label}: state_out claim drifted from replayed child surface at slot {idx}"
        );
    }
}

pub fn single_relation_backend_fixture() -> (
    Rv64imMainRecursionStepSpartanShape,
    Vec<Rv64imMainRecursionFPrimeBackendRelation>,
) {
    static FIXTURE: OnceLock<(
        Rv64imMainRecursionStepSpartanShape,
        Vec<Rv64imMainRecursionFPrimeBackendRelation>,
    )> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(1);
            let input = Rv64imProofInput {
                max_steps: source.program_words.len(),
                source,
            };
            let options = Rv64imPublicProofOptions {
                root_fold_schedule: FoldSchedule::RowsPerChunk(1),
            };
            let public_proof =
                prove_rv64im_public_proof_with_options(&input, options).expect("prove chunked public proof");
            let accepted_artifact =
                build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
            let (final_statement, final_proof) =
                prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
            let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
                .expect("build chunk-step IVC relations");
            assert!(!relations.is_empty(), "expected at least one chunk-step relation");
            build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations[..1])
                .expect("build single-relation recursive-step backend relations")
        })
        .clone()
}
