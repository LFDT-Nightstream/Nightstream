use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv64im_main_recursion_step_spartan_compressed_chain_shape,
    debug_check_rv64im_main_recursion_step_spartan_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity, prove_rv64im_main_recursion_step_spartan,
    prove_rv64im_main_recursion_step_spartan_chain, prove_rv64im_main_recursion_step_spartan_compressed_chain,
    setup_rv64im_main_recursion_step_spartan_cached, verify_rv64im_main_recursion_step_spartan,
    verify_rv64im_main_recursion_step_spartan_chain, verify_rv64im_main_recursion_step_spartan_compressed_chain,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, prove_rv64im_public_proof,
    Rv64imProofInput,
};

fn build_rv64im_n2_final_fixture() -> (
    neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
) {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement/proof")
}

fn build_rv64im_n2_backend_relations() -> (
    neo_fold_next::rv64im::audit::Rv64imMainRecursionStepSpartanShape,
    Vec<neo_fold_next::rv64im::audit::Rv64imMainRecursionFPrimeBackendRelation>,
) {
    let (final_statement, final_proof) = build_rv64im_n2_final_fixture();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step IVC relations");
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations)
        .expect("build recursive-step backend relations")
}

#[test]
#[ignore = "direct n=2 reproducer for the current multi-step compressed-chain prove/verify failure"]
fn rv64im_main_recursion_step_spartan_n2_compressed_chain_round_trip() {
    let (cover_shape, backend_relations) = build_rv64im_n2_backend_relations();
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity(&backend_relations)
        .expect("n=2 compressed-chain native parity should hold before proving");

    let chain_shape = build_rv64im_main_recursion_step_spartan_compressed_chain_shape(&cover_shape, &backend_relations)
        .expect("build compressed-chain shape");
    let statement = backend_relations
        .last()
        .expect("last backend relation")
        .spartan_statement
        .clone();
    let proof = prove_rv64im_main_recursion_step_spartan_compressed_chain(&cover_shape, &backend_relations)
        .expect("prove compressed chain");
    verify_rv64im_main_recursion_step_spartan_compressed_chain(&chain_shape, &statement, &proof)
        .expect("verify compressed chain");
}

#[test]
#[ignore = "direct n=2 discriminator between per-step recursive proofs and compressed-chain proving"]
fn rv64im_main_recursion_step_spartan_n2_chain_round_trip() {
    let (cover_shape, backend_relations) = build_rv64im_n2_backend_relations();
    let proof = prove_rv64im_main_recursion_step_spartan_chain(&cover_shape, &backend_relations)
        .expect("prove recursive-step chain");
    verify_rv64im_main_recursion_step_spartan_chain(&cover_shape, &backend_relations, &proof)
        .expect("verify recursive-step chain");
}

#[test]
#[ignore = "direct n=2 first-step recursive-step proof discriminator"]
fn rv64im_main_recursion_step_spartan_n2_first_step_round_trip() {
    let (cover_shape, backend_relations) = build_rv64im_n2_backend_relations();
    let first = backend_relations.first().expect("first backend relation");
    let keys =
        setup_rv64im_main_recursion_step_spartan_cached(&cover_shape, first).expect("setup first recursive step");
    let (pk, vk) = &*keys;
    let proof = prove_rv64im_main_recursion_step_spartan(pk, &cover_shape, first).expect("prove first recursive step");
    verify_rv64im_main_recursion_step_spartan(vk, &first.spartan_statement, &proof)
        .expect("verify first recursive step");
}

#[test]
#[ignore = "direct n=2 first-step canonical-circuit satisfiability discriminator"]
fn rv64im_main_recursion_step_spartan_n2_first_step_circuit_is_satisfied() {
    let (cover_shape, backend_relations) = build_rv64im_n2_backend_relations();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv64im_main_recursion_step_spartan_circuit(&cover_shape, first)
        .expect("first recursive-step circuit should synthesize cleanly under the shared canonical cover shape");
}

#[test]
#[ignore = "direct n=2 second-step recursive-step proof discriminator"]
fn rv64im_main_recursion_step_spartan_n2_second_step_round_trip() {
    let (cover_shape, backend_relations) = build_rv64im_n2_backend_relations();
    let second = backend_relations.get(1).expect("second backend relation");
    let keys =
        setup_rv64im_main_recursion_step_spartan_cached(&cover_shape, second).expect("setup second recursive step");
    let (pk, vk) = &*keys;
    let proof =
        prove_rv64im_main_recursion_step_spartan(pk, &cover_shape, second).expect("prove second recursive step");
    verify_rv64im_main_recursion_step_spartan(vk, &second.spartan_statement, &proof)
        .expect("verify second recursive step");
}
