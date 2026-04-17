use neo_fold_next::nightstream::rv64im::verify_rv64im_main_proof;

use super::common::build_main_soundness_fixture;

#[test]
#[ignore = "known-red soundness sentinel: expected to fail until the main decider binds the full carried relation"]
fn rv64im_main_soundness_known_red_kernel_export_anchor_is_bound_by_spartan() {
    let fixture = build_main_soundness_fixture().expect("build main-lane soundness fixture");
    verify_rv64im_main_proof(&fixture.main_proof).expect("baseline main proof must verify");

    let mut forged_main_proof = fixture.main_proof.clone();
    forged_main_proof
        .final_statement_cache_mut()
        .expect("known-red fixture must retain the local final-statement cache")
        .folded
        .kernel_relation_digest[0] ^= 1;

    assert!(
        verify_rv64im_main_proof(&forged_main_proof).is_err(),
        "main proof should bind the kernel-export anchor / kernel-relation digest"
    );
}

#[test]
#[ignore = "known-red soundness sentinel: expected to fail until the main decider binds the full carried relation"]
fn rv64im_main_soundness_known_red_initial_handle_chain_is_bound_by_spartan() {
    let fixture = build_main_soundness_fixture().expect("build main-lane soundness fixture");
    verify_rv64im_main_proof(&fixture.main_proof).expect("baseline main proof must verify");

    let mut forged_main_proof = fixture.main_proof.clone();
    forged_main_proof
        .final_statement_cache_mut()
        .expect("known-red fixture must retain the local final-statement cache")
        .folded
        .final_accumulator
        .terminal_handle
        .0[0] ^= 1;

    assert!(
        verify_rv64im_main_proof(&forged_main_proof).is_err(),
        "main proof should bind the initial/terminal handle chain, not only public chunk digests"
    );
}
