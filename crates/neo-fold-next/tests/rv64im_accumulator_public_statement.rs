#![allow(dead_code)]

use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
    evaluate_rv64im_main_recursion_f_prime_advice, rv64im_bridge_handoff_chain_digest,
    rv64im_recursion_step_statement_chain_digest,
};
use neo_fold_next::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted, Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, build_rv64im_recursion_shape,
    prove_rv64im_public_proof, Rv64imAccumulatorPublicStatement, Rv64imProofInput,
};
use p3_field::PrimeCharacteristicRing;

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
    let relations =
        build_rv64im_chunk_step_ivc_relations(final_statement, final_proof).expect("build chunk-step ivc relations");
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion advices");
    let last_output = advices
        .last()
        .map(|advice| evaluate_rv64im_main_recursion_f_prime_advice(advice).expect("evaluate last advice"));
    Rv64imMainFinalProofSurface::from_final_proof(
        final_statement,
        final_proof,
        n2_final_pc(),
        rv64im_recursion_step_statement_chain_digest(&relations),
        rv64im_bridge_handoff_chain_digest(&relations),
        last_output
            .as_ref()
            .map(|output| output.folded_accumulator_digest())
            .unwrap_or_else(|| panic!("expected non-empty n2 recursion advice chain")),
        last_output
            .as_ref()
            .map(|output| output.terminal_handle_digest())
            .unwrap_or_else(|| panic!("expected non-empty n2 recursion advice chain")),
    )
}

fn published_statement_from_n2_final_case() -> Rv64imAccumulatorPublicStatement {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
        .expect("build accumulator public statement")
}

fn honest_last_output_folded_accumulator_digest() -> [u8; 32] {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion advices");
    evaluate_rv64im_main_recursion_f_prime_advice(
        advices
            .last()
            .expect("expected non-empty n2 recursion advice chain"),
    )
    .expect("evaluate last advice")
    .folded_accumulator_digest()
}

#[test]
fn rv64im_accumulator_public_statement_is_stable_and_shape_bound() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let baseline = Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
        .expect("build accumulator public statement");
    let rebuilt = Rv64imAccumulatorPublicStatement::from_final_surface(&final_statement, &final_surface)
        .expect("rebuild accumulator public statement");

    assert_eq!(
        baseline, rebuilt,
        "same carried main proof must yield the same published accumulator statement"
    );
    assert_eq!(
        baseline.shape_digest(),
        build_rv64im_recursion_shape()
            .expect("build recursion shape")
            .canonical_digest(),
        "published statement must bind the canonical recursion shape digest"
    );
    assert_eq!(
        baseline.vk_fs().main_lane_shape_digest,
        baseline.shape_digest(),
        "published statement shape_digest must match the carried recursion verifier key fs"
    );
    assert_eq!(
        baseline.fold_schedule(),
        final_surface.fold_schedule(),
        "published statement fold schedule must match the carried final surface"
    );
    assert_eq!(
        baseline
            .expected_chunk_count()
            .expect("derive expected chunk count from published statement"),
        final_surface.chunk_summary_count(),
        "published statement step_count + fold_schedule must reconstruct the authoritative chunk count"
    );
    assert_eq!(
        baseline.pc_final(),
        final_surface.final_pc(),
        "published statement must carry the authoritative final pc"
    );
    assert_eq!(
        baseline.accumulator_final(),
        &final_statement.folded.final_accumulator,
        "published statement must carry the authoritative final accumulator, not only its digest"
    );
    assert_ne!(
        baseline.x_last().bytes(),
        [0; 32],
        "published statement must carry a nonzero x_last image for the nonempty case"
    );
}

#[test]
fn rv64im_accumulator_public_statement_digest_tracks_vk_fs() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement.vk_fs_mut().domain_tag_digest[0] ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the carried recursion verifier key fs"
    );
}

#[test]
fn rv64im_accumulator_public_statement_rejects_tampered_vk_fs() {
    let mut published_statement = published_statement_from_n2_final_case();
    published_statement.vk_fs_mut().domain_tag_digest[0] ^= 1;

    let err = published_statement
        .validate()
        .expect_err("tampered recursion verifier key fs must fail");
    assert!(format!("{err}").contains("verifier key fs"));
}

#[test]
fn rv64im_accumulator_public_statement_digest_tracks_x_last() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement.x_last_mut().bytes_mut()[0] ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind x_last"
    );
}

#[test]
fn rv64im_accumulator_public_statement_digest_tracks_pc_final() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    *published_statement.pc_final_mut() ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind pc_final"
    );
}

#[test]
fn rv64im_accumulator_public_statement_digest_tracks_accumulator_final() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement
        .accumulator_final_mut()
        .final_main_claims[0]
        .ct[0] += neo_math::K::ONE;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the carried final accumulator"
    );
}

#[test]
fn rv64im_accumulator_public_statement_canonical_digests_come_from_accumulator_final() {
    let published_statement = published_statement_from_n2_final_case();

    assert_eq!(
        published_statement.canonical_terminal_handle_digest(),
        published_statement.accumulator_final().terminal_handle.0,
        "canonical terminal-handle digest must come from the carried final accumulator"
    );
    assert_eq!(
        published_statement.canonical_folded_accumulator_digest(),
        honest_last_output_folded_accumulator_digest(),
        "canonical folded-accumulator digest must match the honest native final accumulator digest"
    );
}

#[test]
fn rv64im_accumulator_public_statement_expected_digest_tracks_accumulator_terminal_handle() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement
        .accumulator_final_mut()
        .terminal_handle
        .0[0] ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the authoritative accumulator terminal handle through accumulator_final"
    );
}
