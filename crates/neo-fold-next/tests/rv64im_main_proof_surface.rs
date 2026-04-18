#![allow(dead_code)]

use neo_fold_next::rv64im::audit::{
    audit_build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs,
    audit_rv64im_main_recursion_final_relation_public_images_match_against_published_statement,
    audit_rv64im_main_recursion_proof_matches_published_statement,
    audit_rv64im_main_recursion_terminal_published_target_matches_native_witness_against_published_statement,
    audit_rv64im_recursion_verifier_key_matches_published_statement, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_main_recursion_f_prime_advices, build_rv64im_main_recursion_proof_surface_stub_from_relations,
    rv64im_main_recursion_accumulator_witness_final_fold_witness_mut,
    rv64im_main_recursion_accumulator_witness_running_final_mut, rv64im_main_recursion_proof_x_last_mut,
};
use neo_fold_next::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted, Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, build_rv64im_main_proof,
    build_rv64im_main_recursion_accumulator_witness, build_rv64im_main_recursion_final_relation_statement,
    prove_rv64im_public_proof, prove_rv64im_recursion_proof, verify_rv64im_main_proof,
    verify_rv64im_main_recursion_final_relation_native,
    verify_rv64im_main_recursion_final_relation_native_against_statement, verify_rv64im_published_main_proof_with_vk,
    Rv64imProofInput,
};

fn n2_final_case() -> (Rv64imFinalStatement, Rv64imFinalBuildProof) {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted proof artifact");
    prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove rv64im final statement")
}

fn n2_final_pc() -> u64 {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    public_proof.statement.final_pc
}

fn build_main_surface(
    final_statement: &Rv64imFinalStatement,
    final_proof: &Rv64imFinalBuildProof,
) -> Rv64imMainFinalProofSurface {
    Rv64imMainFinalProofSurface::from_final_proof(final_statement, final_proof, n2_final_pc())
}

#[test]
fn rv64im_main_proof_surface_chunk_summary_chain_digest_is_stable() {
    let (final_statement, final_proof) = n2_final_case();
    let baseline = build_main_surface(&final_statement, &final_proof);
    let rebuilt = build_main_surface(&final_statement, &final_proof);

    assert_eq!(baseline.chunk_summary_count(), final_proof.chunk_summaries.len() as u64);
    assert_ne!(baseline.chunk_summary_chain_digest(), [0; 32]);
    assert_eq!(
        baseline.chunk_summary_chain_digest(),
        rebuilt.chunk_summary_chain_digest(),
        "same carried final proof must produce the same chunk-summary chain digest"
    );
    assert_eq!(
        baseline.expected_digest(),
        rebuilt.expected_digest(),
        "same carried final proof must produce the same published surface digest"
    );
}

#[test]
fn rv64im_main_proof_surface_digest_tracks_chunk_summary_chain_digest() {
    let (final_statement, final_proof) = n2_final_case();
    let mut final_surface = build_main_surface(&final_statement, &final_proof);
    let baseline = final_surface.expected_digest();

    final_surface.chunk_summary_chain_digest_mut()[0] ^= 1;

    assert_ne!(
        baseline,
        final_surface.expected_digest(),
        "published surface digest must bind the chunk-summary chain digest"
    );
}

#[test]
fn rv64im_main_proof_surface_matches_last_native_f_prime_step_image() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion advices");
    let last_advice = advices
        .last()
        .expect("expected non-empty n2 recursion advice chain");
    let final_surface = build_main_surface(&final_statement, &final_proof);

    assert_eq!(final_surface.chunk_summary_count(), last_advice.chunk_index() + 1);
}

#[test]
fn rv64im_main_recursion_final_relation_native_accepts_honest_witness() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(&published_statement)
        .expect("build final relation statement from published statement");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    verify_rv64im_main_recursion_final_relation_native_against_statement(
        &final_relation_statement,
        &accumulator_witness,
    )
    .expect("validate native main recursion final relation against the terminal relation statement");
    verify_rv64im_main_recursion_final_relation_native(&published_statement, &accumulator_witness)
        .expect("published-statement wrapper must agree with the terminal relation statement path");
}

#[test]
fn rv64im_main_recursion_final_relation_statement_matches_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(&published_statement)
        .expect("build final relation statement from published statement");

    assert_eq!(
        final_relation_statement.shape_digest(),
        published_statement.shape_digest(),
        "terminal relation statement must carry the same shape binding as the published statement"
    );
    assert_eq!(
        final_relation_statement.vk_fs(),
        published_statement.vk_fs(),
        "terminal relation statement must carry the same recursion verifier key fs as the published statement"
    );
    assert_eq!(
        final_relation_statement.chunk_count(),
        published_statement
            .expected_chunk_count()
            .expect("derive expected chunk count from published statement"),
        "terminal relation statement must close the published chunk schedule"
    );
    assert_eq!(
        final_relation_statement.pc_final(),
        published_statement.pc_final(),
        "terminal relation statement must carry the same final pc as the published statement"
    );
    assert_eq!(
        final_relation_statement.accumulator_final(),
        published_statement.accumulator_final(),
        "terminal relation statement must carry the authoritative published final accumulator"
    );
    assert_eq!(
        final_relation_statement.x_last(),
        published_statement.x_last(),
        "terminal relation statement must carry the authoritative published x_last"
    );
}

#[test]
fn rv64im_main_recursion_accumulator_witness_exposes_terminal_fold_fields() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    assert_eq!(
        accumulator_witness.running_final().terminal_handle.0,
        published_statement.accumulator_final().terminal_handle.0,
        "recursion-spartan accumulator witness must expose the authoritative final terminal handle directly"
    );
    assert_eq!(
        accumulator_witness.step_public().chunk_index + 1,
        published_statement
            .expected_chunk_count()
            .expect("derive expected chunk count from published statement"),
        "recursion-spartan accumulator witness must expose the terminal chunk index directly"
    );
    assert_eq!(
        accumulator_witness.step_public().halted_out,
        accumulator_witness.halted_out(),
        "recursion-spartan accumulator witness halted flag must stay aligned with the terminal chunk public surface"
    );
}

#[test]
fn rv64im_main_recursion_x_last_from_accumulator_matches_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");
    let chunk_count = published_statement
        .expected_chunk_count()
        .expect("derive expected chunk count from published statement");

    let _ = accumulator_witness;
    let rebuilt_x_last = audit_build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
        published_statement.vk_fs(),
        chunk_count,
        published_statement.accumulator_final(),
    )
    .expect("rebuild published x_last from the recursion-spartan owner surface");

    assert_eq!(
        &rebuilt_x_last,
        published_statement.x_last(),
        "recursion-spartan-owned x_last reconstruction must match the carried published statement"
    );
}

#[test]
fn rv64im_main_recursion_final_relation_native_rejects_tampered_pc_final() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let mut published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    *published_statement.pc_final_mut() ^= 1;

    let err = verify_rv64im_main_recursion_final_relation_native(&published_statement, &accumulator_witness)
        .expect_err("tampered pc_final must fail the native main recursion final relation");
    assert!(
        format!("{err}").contains("pc_final"),
        "expected pc_final mismatch, got: {err}"
    );
}

#[test]
fn rv64im_main_recursion_stub_expected_digest_ignores_final_public_image_shell() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let mut recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");
    let baseline_expected = recursion_proof.expected_digest();

    rv64im_main_recursion_proof_x_last_mut(&mut recursion_proof)[0] ^= 1;

    assert_eq!(
        baseline_expected,
        recursion_proof.expected_digest(),
        "recursion proof digest must ignore the carried final-public-image shell and stay anchored to proof bytes plus chain shape"
    );
}

#[test]
fn rv64im_main_recursion_surface_stub_rejects_nonempty_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");

    let err = audit_rv64im_main_recursion_proof_matches_published_statement(&published_statement, &recursion_proof)
        .expect_err("an empty recursion proof-chain stub must fail against a non-empty published statement");

    assert!(
        err.to_string().contains("chunk count")
            || err.to_string().contains("proof surface")
            || err
                .to_string()
                .contains("canonical final-relation statement public image"),
        "expected recursion proof-chain/public-image mismatch, got: {err}"
    );
}

#[test]
fn rv64im_recursion_verifier_key_matches_honest_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let (_, recursion_vk) = neo_fold_next::rv64im::setup_rv64im_recursion().expect("setup recursion verifier key");

    audit_rv64im_recursion_verifier_key_matches_published_statement(&recursion_vk, &published_statement)
        .expect("honest recursion verifier key must match the published statement");
}

#[test]
fn rv64im_published_main_proof_with_vk_rejects_tampered_shape_digest() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");
    let (_, mut recursion_vk) = neo_fold_next::rv64im::setup_rv64im_recursion().expect("setup recursion verifier key");
    recursion_vk.shape_digest[0] ^= 1;

    let err = verify_rv64im_published_main_proof_with_vk(&recursion_vk, &published_statement, &recursion_proof)
        .expect_err("tampered recursion verifier key shape digest must fail");
    assert!(
        err.to_string().contains("shape_digest"),
        "expected shape_digest mismatch, got: {err}"
    );
}

#[test]
fn rv64im_main_recursion_final_relation_public_images_match_between_proof_and_witness() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    audit_rv64im_main_recursion_final_relation_public_images_match_against_published_statement(
        &published_statement,
        &accumulator_witness,
        &recursion_proof,
    )
    .expect("native accumulator witness and recursion proof surface must expose the same final-relation public image");
}

#[test]
fn rv64im_main_recursion_terminal_published_target_matches_native_terminal_witness() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let recursion_proof = prove_rv64im_recursion_proof(&final_statement, &final_proof).expect("prove recursion proof");
    let accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    audit_rv64im_main_recursion_terminal_published_target_matches_native_witness_against_published_statement(
        &published_statement,
        &accumulator_witness,
        &recursion_proof,
    )
    .expect("terminal recursive-step published target must match the native terminal accumulator witness");
}

#[test]
fn rv64im_main_recursion_surface_rejects_tampered_x_last_against_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let mut recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");

    rv64im_main_recursion_proof_x_last_mut(&mut recursion_proof)[0] ^= 1;

    let err = audit_rv64im_main_recursion_proof_matches_published_statement(&published_statement, &recursion_proof)
        .expect_err("tampered x_last must fail published-statement recursion-proof surface validation");
    assert!(
        err.to_string().contains("x_last")
            || err.to_string().contains("published statement")
            || err
                .to_string()
                .contains("canonical final-relation statement public image"),
        "expected x_last/final-relation statement failure, got: {err}"
    );
}

#[test]
fn rv64im_main_recursion_surface_rejects_tampered_accumulator_final_against_published_statement() {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let mut published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let recursion_proof = build_rv64im_main_recursion_proof_surface_stub_from_relations(&relations)
        .expect("build recursion proof surface stub");

    published_statement
        .accumulator_final_mut()
        .terminal_handle
        .0[0] ^= 1;

    let err = audit_rv64im_main_recursion_proof_matches_published_statement(&published_statement, &recursion_proof)
        .expect_err("tampered accumulator_final must fail published-statement recursion-proof surface validation");
    assert!(
        err.to_string().contains("final accumulator")
            || err.to_string().contains("terminal handle")
            || err
                .to_string()
                .contains("canonical final-relation statement public image"),
        "expected accumulator-final/final-relation statement failure, got: {err}"
    );
}

#[test]
fn rv64im_main_recursion_final_relation_native_rejects_tampered_final_fold_witness() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(&published_statement)
        .expect("build final-relation statement");
    let mut accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    rv64im_main_recursion_accumulator_witness_final_fold_witness_mut(&mut accumulator_witness)
        .ccs_replay_proof
        .header_digest[0] ^= 1;

    let err = verify_rv64im_main_recursion_final_relation_native_against_statement(
        &final_relation_statement,
        &accumulator_witness,
    )
    .expect_err("tampered final-fold witness must fail native final-relation verification");
    assert!(
        err.to_string().contains("header digest")
            || err.to_string().contains("final fold replay")
            || err.to_string().contains("replay"),
        "expected final-fold witness replay failure, got: {err}"
    );
}

#[test]
fn rv64im_main_recursion_final_relation_native_rejects_tampered_u_final() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let published_statement =
        neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
            .expect("build published accumulator statement");
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(&published_statement)
        .expect("build final-relation statement");
    let mut accumulator_witness = build_rv64im_main_recursion_accumulator_witness(&final_statement, &final_proof)
        .expect("build accumulator witness");

    rv64im_main_recursion_accumulator_witness_running_final_mut(&mut accumulator_witness)
        .terminal_handle
        .0[0] ^= 1;

    let err = verify_rv64im_main_recursion_final_relation_native_against_statement(
        &final_relation_statement,
        &accumulator_witness,
    )
    .expect_err("tampered U_final must fail native final-relation verification");
    assert!(
        err.to_string().contains("final accumulator")
            || err.to_string().contains("public image")
            || err.to_string().contains("terminal handle"),
        "expected final-accumulator/public-image failure, got: {err}"
    );
}

#[test]
#[ignore = "main-proof build still too expensive for routine serde-boundary regression"]
fn rv64im_main_proof_serde_roundtrip_drops_local_caches_without_changing_published_digests() {
    let (final_statement, final_proof) = n2_final_case();
    let main_proof = build_rv64im_main_proof(&final_statement, &final_proof).expect("build main proof");
    let baseline_expected = main_proof.expected_digest();
    let baseline_binding = main_proof.binding_digest();
    let baseline_statement = main_proof.published_statement();

    assert!(
        main_proof.kernel_export_cache().is_some(),
        "freshly built main proof should retain the local kernel-export cache"
    );
    assert!(
        !main_proof.chunk_summaries().is_empty(),
        "freshly built main proof should retain local chunk-summary cache"
    );

    let bytes = bincode::serialize(&main_proof).expect("serialize main proof");
    let decoded: neo_fold_next::rv64im::Rv64imMainProof = bincode::deserialize(&bytes).expect("deserialize main proof");

    assert!(
        decoded.kernel_export_cache().is_none(),
        "published main-proof serde boundary must drop the local kernel-export cache"
    );
    assert!(
        decoded.chunk_summaries().is_empty(),
        "published main-proof serde boundary must drop the local chunk-summary cache"
    );
    decoded
        .validate_local_build_caches()
        .expect("published main-proof serde boundary without local caches must still satisfy local-cache validation");
    assert_eq!(
        baseline_expected,
        decoded.expected_digest(),
        "published main-proof digest must ignore dropped local caches across serde"
    );
    assert_eq!(
        baseline_binding,
        decoded.binding_digest(),
        "published main-proof binding digest must ignore dropped local caches across serde"
    );
    assert_eq!(
        baseline_statement,
        decoded.published_statement(),
        "published statement must be stable across main-proof serde roundtrip"
    );
}

#[test]
#[ignore = "long-running published-main end-to-end recursion verification round-trip"]
fn rv64im_main_proof_round_trip_through_published_statement_contract() {
    let (final_statement, final_proof) = n2_final_case();
    let main_proof = build_rv64im_main_proof(&final_statement, &final_proof).expect("build main proof");
    let published_statement = main_proof.published_statement();

    published_statement
        .validate()
        .expect("published statement must satisfy the owned recursion contract");
    verify_rv64im_main_proof(&main_proof).expect("verify main proof");
}
