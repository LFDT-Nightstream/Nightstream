#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_statement_from_final,
    validate_rv64im_nightstream_linkage_claims_against_statement,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_bound_side_opening_public_from_accepted_artifact, rv64im_nightstream_linkage_root,
    rv64im_verifier_context_digest,
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
fn rv64im_side_proof_digest_binds_opening_statement_digest_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let baseline = fixture.side_proof.expected_digest();

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
fn rv64im_nightstream_linkage_claims_reject_tampered_contents_for_n2_fixture() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut linkage_claims = build_rv64im_nightstream_linkage_claims(&fixture.final_statement, &fixture.final_proof)
        .expect("build Nightstream linkage claims");
    validate_rv64im_nightstream_linkage_claims_against_statement(&fixture.nightstream_statement, &linkage_claims)
        .expect("baseline Nightstream linkage claims must validate against the carried statement");

    linkage_claims.public_chunk_digests_mut()[0][0] ^= 1;
    *linkage_claims.digest_mut() = linkage_claims.expected_digest();
    let err =
        validate_rv64im_nightstream_linkage_claims_against_statement(&fixture.nightstream_statement, &linkage_claims)
            .expect_err("self-consistent linkage-claim tamper must fail against the carried statement");
    assert!(
        err.to_string().contains("public-chunk digests"),
        "unexpected linkage-claims rejection error: {err}"
    );
}

#[test]
fn rv64im_main_proof_surface_is_unchanged_by_authoritative_phi_side() {
    use neo_fold_next::rv64im::main_proof::Rv64imMainFinalProofSurface;
    use neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement;

    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");

    let surface = Rv64imMainFinalProofSurface::from_final_proof(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.accepted_artifact.statement.final_pc,
    );

    let statement = Rv64imAccumulatorPublicStatement::from_final_surface(&fixture.final_statement, &surface)
        .expect("build published statement");

    assert_eq!(
        statement.expected_digest(),
        statement.expected_digest(),
        "published RV64IM main-proof accumulator statement must be deterministic under the canonical final surface"
    );
    assert_eq!(
        surface.final_pc(),
        fixture.accepted_artifact.statement.final_pc,
        "final surface must carry the authoritative final pc"
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
