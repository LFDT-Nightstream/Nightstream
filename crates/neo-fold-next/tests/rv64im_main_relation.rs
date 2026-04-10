//! Focused tests for the owned RV64IM main relation seam.

use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_decider_relation, build_rv64im_main_relation_backend_relation,
    build_rv64im_main_relation_from_final, parity_source_cases, prove_rv64im_public_proof, verify_rv64im_main_relation,
    Rv64imProofInput,
};

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

#[test]
fn rv64im_main_relation_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let main_relation =
        build_rv64im_main_relation_from_final(&statement, &final_proof).expect("build rv64im main relation");

    assert_eq!(main_relation.statement.digest(), statement.digest);
    assert_eq!(
        main_relation.statement.final_statement.folded.digest,
        statement.folded.digest
    );
    assert_eq!(main_relation.witness.digest(), final_proof.proof_digest);
    assert_ne!(main_relation.digest, [0; 32]);

    verify_rv64im_main_relation(&main_relation.statement, &main_relation.witness).expect("verify rv64im main relation");
}

#[test]
fn rv64im_main_relation_backend_relation_matches_current_adapter_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let main_relation =
        build_rv64im_main_relation_from_final(&statement, &final_proof).expect("build rv64im main relation");
    let backend_relation =
        build_rv64im_main_relation_backend_relation(&main_relation.statement, &main_relation.witness)
            .expect("build rv64im main backend relation");
    let adapter_relation = build_rv64im_decider_relation(&proof).expect("build current adapter relation");

    assert_eq!(backend_relation, adapter_relation);
    assert_eq!(backend_relation.public_statement_digest, statement.digest);
    assert_eq!(backend_relation.relation_digest, statement.folded.digest);
    assert_eq!(backend_relation.final_proof_digest, final_proof.proof_digest);
}

#[test]
fn rv64im_main_relation_rejects_tampered_final_proof_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut main_relation =
        build_rv64im_main_relation_from_final(&statement, &final_proof).expect("build rv64im main relation");
    main_relation.witness.final_proof.proof_digest[0] ^= 1;

    let err = verify_rv64im_main_relation(&main_relation.statement, &main_relation.witness)
        .expect_err("tampered final proof digest must fail");
    assert!(format!("{err}").contains("digest"));
}
