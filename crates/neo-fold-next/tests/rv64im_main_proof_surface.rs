#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::nightstream::rv64im::audit::rv64im_main_nightstream_proof_digest;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
};
use neo_fold_next::rv64im::final_relation::{Rv64imFinalBuildProof, Rv64imFinalStatement};
use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;

fn n2_final_case() -> (Rv64imFinalStatement, Rv64imFinalBuildProof) {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    (fixture.final_statement, fixture.final_proof)
}

fn n2_final_pc() -> u64 {
    rv64im_n2_support::build_rv64im_n2_fixture()
        .expect("build rv64im n=2 fixture")
        .accepted_artifact
        .statement
        .final_pc
}

fn build_main_surface(
    final_statement: &Rv64imFinalStatement,
    final_proof: &Rv64imFinalBuildProof,
) -> Rv64imMainFinalProofSurface {
    Rv64imMainFinalProofSurface::from_final_proof(final_statement, final_proof, n2_final_pc())
}

fn flip_first_byte(bytes: &mut Vec<u8>) {
    if let Some(first) = bytes.first_mut() {
        *first ^= 1;
    } else {
        bytes.push(1);
    }
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
#[ignore = "expensive: published seam construction includes compressed IVC proof material"]
fn rv64im_main_nightstream_proof_digest_tracks_compact_compressed_main_proof() {
    let compact = rv64im_n2_support::build_rv64im_n2_published_seam()
        .expect("build one-step published seam")
        .main_proof
        .clone();
    let baseline = rv64im_main_nightstream_proof_digest(&compact);

    let mut recursive_proof_tamper = compact.clone();
    flip_first_byte(
        &mut recursive_proof_tamper
            .ivc_recursion_snark_proof_mut()
            .terminal_f_prime_committed_step_proof
            .snark_data,
    );
    assert_ne!(
        baseline,
        rv64im_main_nightstream_proof_digest(&recursive_proof_tamper),
        "Nightstream proof-binding digest must change when the carried IVC recursion SNARK proof bytes change"
    );

    let mut final_ce_proof_tamper = compact.clone();
    flip_first_byte(
        &mut final_ce_proof_tamper
            .ivc_recursion_snark_proof_mut()
            .final_ce_proof
            .snark_data,
    );
    assert_ne!(
        baseline,
        rv64im_main_nightstream_proof_digest(&final_ce_proof_tamper),
        "Nightstream proof-binding digest must change when the carried final CE proof bytes change"
    );

    let mut statement_tamper = compact.clone();
    statement_tamper
        .published_statement_mut()
        .x_last_mut()
        .bytes_mut()[0] ^= 1;
    assert_ne!(
        baseline,
        rv64im_main_nightstream_proof_digest(&statement_tamper),
        "Nightstream proof-binding digest must change when the carried published statement changes"
    );
}

#[test]
#[ignore = "expensive: published seam construction includes compressed IVC proof material"]
fn rv64im_accumulator_public_statement_rejects_x_last_drift() {
    let mut compact = rv64im_n2_support::build_rv64im_n2_published_seam()
        .expect("build one-step published seam")
        .main_proof
        .clone();
    compact
        .published_statement()
        .validate()
        .expect("baseline published accumulator statement must validate");

    compact.published_statement_mut().x_last_mut().bytes_mut()[0] ^= 1;
    let err = compact
        .published_statement()
        .validate()
        .expect_err("published accumulator statement must reject x_last drift");
    assert!(
        format!("{err}").contains("x_last"),
        "expected x_last validation failure, got: {err}"
    );
}
