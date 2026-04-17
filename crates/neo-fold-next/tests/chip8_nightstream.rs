#[path = "support/chip8.rs"]
mod chip8_support;

use neo_fold_next::chip8::decider::build_chip8_spartan2_decider_target;
use neo_fold_next::chip8::proof::prove_recursive;
use neo_fold_next::nightstream::chip8::{
    build_chip8_main_decider_proof, build_chip8_main_residual_proof, build_chip8_nightstream_from_recursive_proof,
    chip8_absent_linkage_binding_digest, chip8_absent_side_proof_digest, chip8_nightstream_linkage_root,
    chip8_verifier_context_digest, verify_chip8_main_decider_proof, verify_chip8_main_residual_proof,
    verify_chip8_nightstream_from_recursive_proof, Chip8MainDeciderProof, Chip8MainResidualProof,
};

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_absent_artifact_digests_are_distinct() {
    let side_proof = chip8_absent_side_proof_digest();
    let linkage = chip8_absent_linkage_binding_digest();

    assert_ne!(side_proof, [0; 32]);
    assert_ne!(linkage, [0; 32]);
    assert_ne!(side_proof, linkage);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_verifier_context_digest_is_stable_and_nonzero() {
    let digest_a = chip8_verifier_context_digest();
    let digest_b = chip8_verifier_context_digest();

    assert_eq!(digest_a, digest_b);
    assert_ne!(digest_a, [0; 32]);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_linkage_root_tracks_anchor_digest() {
    let root_a = chip8_nightstream_linkage_root([1; 32]);
    let root_b = chip8_nightstream_linkage_root([2; 32]);

    assert_ne!(root_a, [0; 32]);
    assert_ne!(root_b, [0; 32]);
    assert_ne!(root_a, root_b);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_main_decider_proof_digest_tracks_target_digest() {
    let proof = Chip8MainDeciderProof {
        decider_target_digest: [7; 32],
    };
    let mut tampered = proof.clone();
    tampered.decider_target_digest[0] ^= 1;

    assert_ne!(proof.expected_digest(), [0; 32]);
    assert_ne!(proof.expected_digest(), tampered.expected_digest());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_main_residual_proof_digest_tracks_fields() {
    let proof = Chip8MainResidualProof {
        statement_digest: [1; 32],
        folded_statement_digest: [2; 32],
        final_proof_digest: [3; 32],
        kernel_export_proof_digest: [4; 32],
        chunk_transition_digests: vec![[5; 32], [6; 32]],
    };
    let mut tampered = proof.clone();
    tampered.chunk_transition_digests[1][0] ^= 1;

    assert_ne!(proof.expected_digest(), [0; 32]);
    assert_ne!(proof.expected_digest(), tampered.expected_digest());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_round_trips_against_current_recursive_seam() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let (nightstream_statement, nightstream_proof) =
        build_chip8_nightstream_from_recursive_proof(&statement, &proof).expect("build chip8 nightstream");

    verify_chip8_nightstream_from_recursive_proof(&statement, &proof, &nightstream_statement, &nightstream_proof)
        .expect("verify chip8 nightstream");

    assert_eq!(
        nightstream_statement.verifier_context_digest,
        chip8_verifier_context_digest()
    );
    assert_eq!(
        nightstream_statement.linkage_root,
        chip8_nightstream_linkage_root(proof.kernel_export.digest)
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_decider_target_bridge_matches_current_target() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let main_decider_proof = build_chip8_main_decider_proof(&statement, &proof).expect("build decider proof");

    assert_eq!(
        main_decider_proof.decider_target_digest,
        build_chip8_spartan2_decider_target(&statement, &proof)
            .expect("build chip8 spartan target")
            .digest()
    );
    verify_chip8_main_decider_proof(&statement, &proof, &main_decider_proof).expect("verify decider proof");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_main_residual_follows_recursive_seam() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let residual = build_chip8_main_residual_proof(&statement, &proof).expect("build residual proof");

    assert_eq!(residual.statement_digest, statement.digest);
    assert_eq!(residual.folded_statement_digest, statement.folded.digest);
    assert_eq!(residual.final_proof_digest, proof.proof_digest);
    assert_eq!(residual.kernel_export_proof_digest, proof.kernel_export.digest);
    assert_eq!(residual.chunk_transition_digests.len(), proof.steps.len());
    verify_chip8_main_residual_proof(&statement, &proof, &residual).expect("verify residual proof");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_rejects_tampered_statement_binding() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let (mut nightstream_statement, nightstream_proof) =
        build_chip8_nightstream_from_recursive_proof(&statement, &proof).expect("build chip8 nightstream");

    nightstream_statement.proof_binding_root[0] ^= 1;
    let err =
        verify_chip8_nightstream_from_recursive_proof(&statement, &proof, &nightstream_statement, &nightstream_proof)
            .expect_err("tampered chip8 nightstream statement must fail");
    assert!(format!("{err}").contains("Nightstream statement"));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_nightstream_rejects_tampered_main_residual_proof() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let (nightstream_statement, mut nightstream_proof) =
        build_chip8_nightstream_from_recursive_proof(&statement, &proof).expect("build chip8 nightstream");

    nightstream_proof
        .main_residual_proof
        .chunk_transition_digests[0][0] ^= 1;
    let err =
        verify_chip8_nightstream_from_recursive_proof(&statement, &proof, &nightstream_statement, &nightstream_proof)
            .expect_err("tampered chip8 main residual proof must fail");
    assert!(format!("{err}").contains("main residual proof"));
}
