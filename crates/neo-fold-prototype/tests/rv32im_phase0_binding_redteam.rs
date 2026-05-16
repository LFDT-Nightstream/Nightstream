use neo_fold_prototype::public_proof::rv32im::side_bundle::build_rv32im_side_proof_bundle_from_accepted_artifact;
use neo_fold_prototype::public_proof::rv32im::side_eval_claim_relation::{
    build_rv32im_side_eval_claim_artifact_from_accepted_artifact,
    build_rv32im_side_eval_claim_relation_from_accepted_artifact, verify_rv32im_side_eval_claim_artifact,
    verify_rv32im_side_eval_claim_relation,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof, Rv32imProofInput,
};

fn source_case(name: &str) -> neo_fold_prototype::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

#[test]
fn rv32im_side_eval_claim_relation_rejects_tampered_side_bundle_stage1_digest() {
    let proof =
        prove_rv32im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, witness) =
        build_rv32im_side_eval_claim_relation_from_accepted_artifact(&artifact).expect("build side-eval relation");

    statement.side_bundle.stage1.digest[0] ^= 1;
    statement.side_bundle.digest = statement.side_bundle.expected_digest();

    let err = verify_rv32im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered carried stage1 digest must fail");
    assert!(err
        .to_string()
        .contains("stage1 verified-claims digest mismatch"));
}

#[test]
fn rv32im_side_eval_claim_relation_rejects_tampered_side_bundle_stage2_digest() {
    let proof =
        prove_rv32im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, witness) =
        build_rv32im_side_eval_claim_relation_from_accepted_artifact(&artifact).expect("build side-eval relation");

    statement.side_bundle.stage2.digest[0] ^= 1;
    statement.side_bundle.digest = statement.side_bundle.expected_digest();

    let err = verify_rv32im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered carried stage2 digest must fail");
    assert!(err
        .to_string()
        .contains("stage2 verified-claims digest mismatch"));
}

#[test]
fn rv32im_side_eval_claim_relation_rejects_tampered_side_bundle_stage3_digest() {
    let proof =
        prove_rv32im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, witness) =
        build_rv32im_side_eval_claim_relation_from_accepted_artifact(&artifact).expect("build side-eval relation");

    statement.side_bundle.stage3.digest[0] ^= 1;
    statement.side_bundle.digest = statement.side_bundle.expected_digest();

    let err = verify_rv32im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered carried stage3 digest must fail");
    assert!(err
        .to_string()
        .contains("stage3 verified-claims digest mismatch"));
}

#[test]
fn rv32im_side_eval_claim_artifact_rejects_tampered_side_bundle_stage2_digest() {
    let proof =
        prove_rv32im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let mut side_bundle =
        build_rv32im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let phase0_artifact =
        build_rv32im_side_eval_claim_artifact_from_accepted_artifact(&artifact).expect("build phase0 artifact");

    side_bundle.stage2.digest[0] ^= 1;
    side_bundle.digest = side_bundle.expected_digest();

    let err = verify_rv32im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect_err("tampered carried stage2 digest must fail through the artifact verifier");
    assert!(err
        .to_string()
        .contains("stage2 verified-claims digest mismatch"));
}
