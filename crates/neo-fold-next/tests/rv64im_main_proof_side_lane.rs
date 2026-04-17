#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_statement_from_final,
    build_rv64im_side_opening_relation_from_accepted_artifact,
    derive_rv64im_kernel_export_source_digest_from_compact_surfaces,
    derive_rv64im_root_execution_digest_from_compact_surfaces, setup_rv64im_side_binding,
    setup_rv64im_side_opening_spartan, validate_rv64im_nightstream_linkage_claims,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_bound_side_opening_public_from_accepted_artifact, rv64im_nightstream_linkage_root,
    rv64im_verifier_context_digest, verify_rv64im_side_proof,
};
use neo_fold_next::rv64im::{
    build_rv64im_main_proof, build_rv64im_main_proof_with_side_opening_public, verify_rv64im_published_main_proof,
};

#[test]
fn rv64im_side_opening_public_from_accepted_artifact_matches_side_proof_public() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let direct_public = build_rv64im_bound_side_opening_public_from_accepted_artifact(
        &fixture.nightstream_statement,
        &fixture.accepted_artifact,
    )
    .expect("derive bound side opening public from accepted artifact");

    assert_eq!(
        direct_public,
        *fixture.side_proof.opening_public(),
        "accepted-artifact side-opening public must match the published Nightstream side proof surface"
    );
}

#[test]
fn rv64im_bound_side_opening_public_tracks_nightstream_statement_core_digest() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let public = build_rv64im_bound_side_opening_public_from_accepted_artifact(
        &fixture.nightstream_statement,
        &fixture.accepted_artifact,
    )
    .expect("derive bound side opening public from accepted artifact");
    let mut wrong_statement = build_rv64im_nightstream_statement_from_final(
        fixture.accepted_artifact.statement.digest,
        rv64im_verifier_context_digest(fixture.accepted_artifact.statement.root_params_id),
        &fixture.final_statement,
        &fixture.final_proof,
        rv64im_nightstream_linkage_root(
            fixture.final_statement.public_statement_digest,
            &build_rv64im_nightstream_linkage_claims(&fixture.final_statement, &fixture.final_proof)
                .expect("build linkage claims"),
        ),
        [0; 32],
    )
    .expect("build provisional Nightstream statement");
    wrong_statement.linkage_root[0] ^= 1;
    let rebound =
        build_rv64im_bound_side_opening_public_from_accepted_artifact(&wrong_statement, &fixture.accepted_artifact)
            .expect("derive rebound side opening public from accepted artifact");

    assert_ne!(
        rebound, public,
        "bound side-opening public must change when the carried Nightstream statement core changes"
    );
}

#[test]
fn rv64im_side_proof_digest_binds_linkage_and_opening_statement_digest_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let baseline = fixture.side_proof.expected_digest();

    let mut tampered_linkage = fixture.side_proof.clone();
    tampered_linkage
        .linkage_mut()
        .transcript_surface_digest_mut()[0] ^= 1;
    assert_ne!(
        baseline,
        tampered_linkage.expected_digest(),
        "Nightstream side-proof digest must change when carried linkage bytes change"
    );

    let mut tampered_opening_statement_digest = fixture.side_proof.clone();
    tampered_opening_statement_digest
        .opening_statement_mut()
        .stage1
        .digest[0] ^= 1;
    assert_ne!(
        baseline,
        tampered_opening_statement_digest.expected_digest(),
        "Nightstream side-proof digest must change when carried opening-statement digest bytes change"
    );
}

#[test]
fn rv64im_side_proof_rejects_self_consistent_transcript_surface_tamper_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let (opening_statement, opening_witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side-opening relation");
    let (_, opening_vk) =
        setup_rv64im_side_opening_spartan(&opening_statement, &opening_witness).expect("setup side opening");
    let side_statement = fixture
        .side_proof
        .binding_statement(&fixture.nightstream_statement)
        .expect("build side binding statement");
    let (_, vk) =
        setup_rv64im_side_binding(&side_statement, fixture.side_proof.opening_public()).expect("setup side binding");

    verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &fixture.side_proof,
    )
    .expect("baseline n=2 side proof must verify");

    let mut tampered_side_proof = fixture.side_proof.clone();
    {
        let linkage = tampered_side_proof.linkage_mut();
        linkage.transcript_surface_digest_mut()[0] ^= 1;
        let expected_digest = linkage.expected_digest();
        *linkage.digest_mut() = expected_digest;
    }

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &tampered_side_proof,
    )
    .expect_err("self-consistent transcript surface tamper must fail");
    assert!(
        err.to_string().contains("kernel-export source surface") || err.to_string().contains("kernel-export bridge"),
        "unexpected linkage tamper rejection error: {err}"
    );
}

#[test]
fn rv64im_side_proof_rejects_self_consistent_semantic_rows_digest_tamper_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let (opening_statement, opening_witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side-opening relation");
    let (_, opening_vk) =
        setup_rv64im_side_opening_spartan(&opening_statement, &opening_witness).expect("setup side opening");
    let side_statement = fixture
        .side_proof
        .binding_statement(&fixture.nightstream_statement)
        .expect("build side binding statement");
    let (_, vk) =
        setup_rv64im_side_binding(&side_statement, fixture.side_proof.opening_public()).expect("setup side binding");

    verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &fixture.side_proof,
    )
    .expect("baseline n=2 side proof must verify");

    let mut tampered_side_proof = fixture.side_proof.clone();
    {
        let linkage = tampered_side_proof.linkage_mut();
        linkage.semantic_rows_digest_mut()[0] ^= 1;
        *linkage.root_execution_digest_mut() = derive_rv64im_root_execution_digest_from_compact_surfaces(
            &fixture.nightstream_statement,
            &fixture.accepted_artifact.statement,
            linkage.semantic_rows_digest(),
            linkage.row_local_ccs_acceptance_digest(),
            linkage.execution_semantics_refinement_digest(),
            linkage.family_digest(),
        )
        .expect("recompute self-consistent root-execution digest");
        linkage
            .kernel_export_bridge_mut()
            .kernel_export_source_digest = derive_rv64im_kernel_export_source_digest_from_compact_surfaces(
            linkage,
            &fixture.accepted_artifact.statement,
        )
        .expect("recompute self-consistent kernel-export source digest");
        linkage.kernel_export_bridge_mut().digest = linkage.kernel_export_bridge().expected_digest();
        *linkage.digest_mut() = linkage.expected_digest();
    }

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &tampered_side_proof,
    )
    .expect_err("self-consistent semantic-rows digest tamper must fail");
    assert!(
        err.to_string().contains("root-execution summary"),
        "unexpected semantic-rows tamper rejection error: {err}"
    );
}

#[test]
fn rv64im_side_proof_rejects_self_consistent_root_execution_digest_tamper_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let (opening_statement, opening_witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side-opening relation");
    let (_, opening_vk) =
        setup_rv64im_side_opening_spartan(&opening_statement, &opening_witness).expect("setup side opening");
    let side_statement = fixture
        .side_proof
        .binding_statement(&fixture.nightstream_statement)
        .expect("build side binding statement");
    let (_, vk) =
        setup_rv64im_side_binding(&side_statement, fixture.side_proof.opening_public()).expect("setup side binding");

    verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &fixture.side_proof,
    )
    .expect("baseline n=2 side proof must verify");

    let mut tampered_side_proof = fixture.side_proof.clone();
    {
        let linkage = tampered_side_proof.linkage_mut();
        linkage.root_execution_digest_mut()[0] ^= 1;
        *linkage.digest_mut() = linkage.expected_digest();
    }

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &tampered_side_proof,
    )
    .expect_err("self-consistent root-execution digest tamper must fail");
    assert!(
        err.to_string().contains("root-execution surface"),
        "unexpected root-execution tamper rejection error: {err}"
    );
}

#[test]
fn rv64im_side_proof_rejects_self_consistent_kernel_export_bridge_tamper_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let (opening_statement, opening_witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side-opening relation");
    let (_, opening_vk) =
        setup_rv64im_side_opening_spartan(&opening_statement, &opening_witness).expect("setup side opening");
    let side_statement = fixture
        .side_proof
        .binding_statement(&fixture.nightstream_statement)
        .expect("build side binding statement");
    let (_, vk) =
        setup_rv64im_side_binding(&side_statement, fixture.side_proof.opening_public()).expect("setup side binding");

    verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &fixture.side_proof,
    )
    .expect("baseline n=2 side proof must verify");

    let mut tampered_side_proof = fixture.side_proof.clone();
    {
        let linkage = tampered_side_proof.linkage_mut();
        linkage.kernel_export_bridge_mut().main_lane_proof_digest[0] ^= 1;
        linkage.kernel_export_bridge_mut().digest = linkage.kernel_export_bridge().expected_digest();
        *linkage.digest_mut() = linkage.expected_digest();
    }

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.accepted_artifact.statement,
        &tampered_side_proof,
    )
    .expect_err("self-consistent kernel-export bridge tamper must fail");
    assert!(
        err.to_string().contains("kernel-export source surface") || err.to_string().contains("kernel-export bridge"),
        "unexpected kernel-export tamper rejection error: {err}"
    );
}

#[test]
fn rv64im_nightstream_linkage_claims_reject_tampered_contents_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut linkage_claims = build_rv64im_nightstream_linkage_claims(&fixture.final_statement, &fixture.final_proof)
        .expect("build Nightstream linkage claims");
    validate_rv64im_nightstream_linkage_claims(&linkage_claims)
        .expect("baseline Nightstream linkage claims must validate");

    linkage_claims.bridge_handoff_digests_mut()[0][0] ^= 1;
    let err = validate_rv64im_nightstream_linkage_claims(&linkage_claims)
        .expect_err("tampered linkage claims with stale digest must fail");
    assert!(
        err.to_string().contains("linkage claims digest mismatch"),
        "unexpected linkage-claims rejection error: {err}"
    );
}

#[test]
fn rv64im_main_proof_surface_is_unchanged_by_authoritative_phi_side() {
    use neo_fold_next::rv64im::audit::{
        build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
        evaluate_rv64im_main_recursion_f_prime_advice, rv64im_bridge_handoff_chain_digest,
        rv64im_recursion_step_statement_chain_digest,
    };
    use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;
    use neo_fold_next::rv64im::{
        build_rv64im_main_recursion_f_prime_advices_with_side_opening_public, Rv64imAccumulatorPublicStatement,
    };

    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let zero_advices =
        build_rv64im_main_recursion_f_prime_advices(&relations).expect("build zero-side recursion advices");
    let side_advices = build_rv64im_main_recursion_f_prime_advices_with_side_opening_public(
        &relations,
        fixture.side_proof.opening_public(),
    )
    .expect("build side-aware recursion advices");

    let build_surface = |advices: &[neo_fold_next::rv64im::Rv64imMainRecursionFPrimeAdvice]| {
        let last_output = advices
            .last()
            .map(|advice| evaluate_rv64im_main_recursion_f_prime_advice(advice).expect("evaluate last advice"))
            .expect("expected non-empty n=2 recursion advice chain");
        Rv64imMainFinalProofSurface::from_final_proof(
            &fixture.final_statement,
            &fixture.final_proof,
            fixture.accepted_artifact.statement.final_pc,
            rv64im_recursion_step_statement_chain_digest(&relations),
            rv64im_bridge_handoff_chain_digest(&relations),
            last_output.folded_accumulator_digest(),
            last_output.terminal_handle_digest(),
        )
    };

    let zero_surface = build_surface(&zero_advices);
    let side_surface = build_surface(&side_advices);

    assert_eq!(
        zero_surface.expected_digest(),
        side_surface.expected_digest(),
        "authoritative phi_side must not change the published RV64IM main-proof final surface digest"
    );

    let zero_statement = Rv64imAccumulatorPublicStatement::from_final_surface(&fixture.final_statement, &zero_surface)
        .expect("build zero-side published statement");
    let side_statement = Rv64imAccumulatorPublicStatement::from_final_surface(&fixture.final_statement, &side_surface)
        .expect("build side-aware published statement");

    assert_eq!(
        zero_statement, side_statement,
        "authoritative phi_side must not change the published RV64IM main-proof accumulator statement"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands; re-enable with the sibling rv64im_main_proof_* and rv64im_main_recursion_* round-trips"]
fn rv64im_main_proof_round_trip_uses_authoritative_phi_side() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let zero_side =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build baseline main proof");
    let side_aware = build_rv64im_main_proof_with_side_opening_public(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.side_proof.opening_public(),
    )
    .expect("build side-aware main proof");

    assert_eq!(
        side_aware.published_statement(),
        zero_side.published_statement(),
        "authoritative phi_side should not change the published RV64IM main-proof statement surface"
    );
    verify_rv64im_published_main_proof(side_aware.published_statement(), side_aware.published_proof())
        .expect("side-aware main proof should verify through the published recursion seam");
}
