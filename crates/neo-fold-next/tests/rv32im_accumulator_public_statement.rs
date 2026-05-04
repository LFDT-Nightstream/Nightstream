#![allow(dead_code)]

use neo_fold_next::rv32im::audit::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_advices,
    evaluate_rv32im_main_recursion_f_prime_advice,
};
use neo_fold_next::rv32im::final_relation::{
    prove_rv32im_final_statement_from_accepted, Rv32imFinalBuildProof, Rv32imFinalStatement,
};
use neo_fold_next::rv32im::ivc::derive_rv32im_ivc_step_cap;
use neo_fold_next::rv32im::main_proof::Rv32imMainFinalProofSurface;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact,
    build_rv32im_recursion_shape_for_step_cap, prove_rv32im_public_proof, Rv32imAccumulatorPublicStatement,
    Rv32imProofInput,
};
use p3_field::PrimeCharacteristicRing;

fn n2_final_case() -> (Rv32imFinalStatement, Rv32imFinalBuildProof) {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let public_proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&public_proof).expect("build accepted proof artifact");
    prove_rv32im_final_statement_from_accepted(&accepted_artifact).expect("prove rv32im final statement")
}

fn n2_final_pc() -> u64 {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let public_proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    public_proof.statement.final_pc
}

fn build_main_surface(
    final_statement: &Rv32imFinalStatement,
    final_proof: &Rv32imFinalBuildProof,
) -> Rv32imMainFinalProofSurface {
    Rv32imMainFinalProofSurface::from_final_proof(final_statement, final_proof, n2_final_pc())
}

fn published_statement_from_n2_final_case() -> Rv32imAccumulatorPublicStatement {
    let (final_statement, final_proof) = n2_final_case();
    Rv32imAccumulatorPublicStatement::from_final_artifacts(&final_statement, &final_proof, n2_final_pc())
        .expect("build accumulator public statement")
}

fn honest_last_output_folded_accumulator_digest() -> [u8; 32] {
    let (final_statement, final_proof) = n2_final_case();
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let advices = build_rv32im_main_recursion_f_prime_advices(&relations).expect("build main recursion advices");
    evaluate_rv32im_main_recursion_f_prime_advice(
        advices
            .last()
            .expect("expected non-empty n2 recursion advice chain"),
    )
    .expect("evaluate last advice")
    .folded_accumulator_digest()
}

#[test]
fn rv32im_accumulator_public_statement_is_stable_and_shape_bound() {
    let (final_statement, final_proof) = n2_final_case();
    let final_surface = build_main_surface(&final_statement, &final_proof);
    let baseline =
        Rv32imAccumulatorPublicStatement::from_final_artifacts(&final_statement, &final_proof, n2_final_pc())
            .expect("build accumulator public statement");
    let rebuilt = Rv32imAccumulatorPublicStatement::from_final_artifacts(&final_statement, &final_proof, n2_final_pc())
        .expect("rebuild accumulator public statement");

    assert_eq!(
        baseline, rebuilt,
        "same carried main proof must yield the same published accumulator statement"
    );
    assert_eq!(
        baseline.shape_digest(),
        build_rv32im_recursion_shape_for_step_cap(
            derive_rv32im_ivc_step_cap(
                final_surface.fold_schedule(),
                final_surface.semantic_step_count() as usize,
            )
            .expect("derive published recursion step_cap")
        )
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
    assert_eq!(
        baseline.terminal_step_statement().step_public.step_hi,
        final_surface.semantic_step_count(),
        "published statement must carry the authoritative terminal step surface"
    );
}

#[test]
fn rv32im_accumulator_public_statement_digest_tracks_vk_fs() {
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
fn rv32im_accumulator_public_statement_rejects_tampered_vk_fs() {
    let mut published_statement = published_statement_from_n2_final_case();
    published_statement.vk_fs_mut().domain_tag_digest[0] ^= 1;

    let err = published_statement
        .validate()
        .expect_err("tampered recursion verifier key fs must fail");
    assert!(format!("{err}").contains("verifier key fs"));
}

#[test]
fn rv32im_accumulator_public_statement_digest_tracks_x_last() {
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
fn rv32im_accumulator_public_statement_digest_tracks_construction2_u_i_boundary() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement
        .construction2_u_i_mut()
        .commitment_digest[0] ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the final Construction-2 committed-instance boundary"
    );
}

#[test]
fn rv32im_accumulator_public_statement_rejects_tampered_construction2_u_i_x() {
    let mut published_statement = published_statement_from_n2_final_case();
    published_statement.construction2_u_i_mut().x_i.bytes_mut()[0] ^= 1;

    let err = published_statement
        .validate()
        .expect_err("tampered Construction-2 u_i.x_i must fail validation");
    assert!(
        err.to_string().contains("Construction-2 u_i.x_i"),
        "expected Construction-2 u_i.x_i mismatch, got: {err}"
    );
}

#[test]
fn rv32im_accumulator_public_statement_rejects_tampered_construction2_u_i_digest() {
    let mut published_statement = published_statement_from_n2_final_case();
    published_statement
        .construction2_u_i_mut()
        .fresh_instance_digest[0] ^= 1;

    let err = published_statement
        .validate()
        .expect_err("tampered Construction-2 public-boundary digest must fail validation");
    assert!(
        err.to_string().contains("Construction-2 u_i digest"),
        "expected Construction-2 u_i digest mismatch, got: {err}"
    );
}

#[test]
fn rv32im_accumulator_public_statement_digest_tracks_pc_final() {
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
fn rv32im_accumulator_public_statement_digest_tracks_accumulator_final() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement
        .accumulator_final_mut()
        .final_main_claims[0]
        .c
        .data[0] += neo_math::F::ONE;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the authoritative carried final accumulator projection"
    );
}

#[test]
fn rv32im_accumulator_public_statement_canonical_digests_come_from_accumulator_final() {
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
fn rv32im_accumulator_public_statement_expected_digest_tracks_accumulator_terminal_handle() {
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

#[test]
fn rv32im_accumulator_public_statement_expected_digest_tracks_terminal_step_statement() {
    let mut published_statement = published_statement_from_n2_final_case();
    let baseline = published_statement.expected_digest();

    published_statement
        .terminal_step_statement_mut()
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;

    assert_ne!(
        baseline,
        published_statement.expected_digest(),
        "published accumulator statement digest must bind the authoritative terminal step statement"
    );
}

#[test]
fn rv32im_accumulator_public_statement_rejects_tampered_terminal_step_statement() {
    let mut published_statement = published_statement_from_n2_final_case();
    published_statement
        .terminal_step_statement_mut()
        .step_public
        .state_out[0] ^= 1;

    let err = published_statement
        .validate()
        .expect_err("tampered terminal step statement must fail validation");
    assert!(
        err.to_string().contains("terminal state_out"),
        "expected terminal-step state_out mismatch, got: {err}"
    );
}
