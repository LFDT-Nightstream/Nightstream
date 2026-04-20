#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::nightstream::rv64im::audit::rv64im_main_nightstream_proof_digest;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
};
use neo_fold_next::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted, Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, prove_rv64im_public_proof,
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
fn rv64im_main_nightstream_proof_digest_tracks_compact_compressed_main_proof() {
    let mut compact = rv64im_n2_support::build_rv64im_n2_published_seam()
        .expect("build one-step published seam")
        .main_proof
        .clone();
    let baseline = rv64im_main_nightstream_proof_digest(&compact);

    compact.terminal_decider_proof_mut().snark_data[0] ^= 1;
    assert_ne!(
        baseline,
        rv64im_main_nightstream_proof_digest(&compact),
        "Nightstream proof-binding digest must change when the carried terminal decider proof bytes change"
    );

    compact.linkage_anchor_digest_mut()[0] ^= 1;
    assert_ne!(
        baseline,
        rv64im_main_nightstream_proof_digest(&compact),
        "Nightstream proof-binding digest must change when the carried linkage anchor changes"
    );
}
