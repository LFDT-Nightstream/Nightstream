#![allow(dead_code)]

#[path = "support/rv32im_n2.rs"]
mod rv32im_n2_support;

use neo_fold_next::nightstream::rv32im::audit::build_rv32im_nightstream_statement_from_final;
use neo_fold_next::nightstream::rv32im::{
    build_rv32im_bound_side_opening_public_from_accepted_artifact, rv32im_verifier_context_digest,
};
#[test]
fn rv32im_side_opening_public_from_accepted_artifact_matches_side_proof_public() {
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let direct_public = build_rv32im_bound_side_opening_public_from_accepted_artifact(
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
fn rv32im_bound_side_opening_public_tracks_nightstream_statement_core_digest() {
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let public = build_rv32im_bound_side_opening_public_from_accepted_artifact(
        &fixture.nightstream_statement,
        &fixture.accepted_artifact,
    )
    .expect("derive bound side opening public from accepted artifact");
    let published_statement = neo_fold_next::rv32im::Rv32imAccumulatorPublicStatement::from_final_artifacts(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.accepted_artifact.statement.final_pc,
    )
    .expect("build published statement");
    let mut wrong_statement = build_rv32im_nightstream_statement_from_final(
        fixture.accepted_artifact.statement.digest,
        rv32im_verifier_context_digest(
            fixture.accepted_artifact.statement.root_params_id,
            &published_statement,
            &fixture.ivc_recursion_snark_keys.as_ref().1,
        )
        .expect("digest verifier context"),
        &fixture.final_statement,
        &fixture.final_proof,
        [0; 32],
    )
    .expect("build provisional Nightstream statement");
    wrong_statement.verifier_context_digest[0] ^= 1;
    let rebound =
        build_rv32im_bound_side_opening_public_from_accepted_artifact(&wrong_statement, &fixture.accepted_artifact)
            .expect("derive rebound side opening public from accepted artifact");

    assert_ne!(
        rebound, public,
        "bound side-opening public must change when the carried Nightstream statement core changes"
    );
}

#[test]
fn rv32im_verifier_context_digest_tracks_published_main_recursion_shape() {
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let published_statement = neo_fold_next::rv32im::Rv32imAccumulatorPublicStatement::from_final_artifacts(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.accepted_artifact.statement.final_pc,
    )
    .expect("build published statement");
    let baseline = rv32im_verifier_context_digest(
        fixture.accepted_artifact.statement.root_params_id,
        &published_statement,
        &fixture.ivc_recursion_snark_keys.as_ref().1,
    )
    .expect("digest verifier context");

    let mut tampered_published_statement = published_statement.clone();
    tampered_published_statement.shape_digest_mut()[0] ^= 1;
    let tampered_shape = rv32im_verifier_context_digest(
        fixture.accepted_artifact.statement.root_params_id,
        &tampered_published_statement,
        &fixture.ivc_recursion_snark_keys.as_ref().1,
    )
    .expect("digest tampered-shape verifier context");

    assert_ne!(
        baseline, tampered_shape,
        "Nightstream verifier context must bind the published RV32IM main-recursion shape"
    );
}

#[test]
fn rv32im_side_proof_digest_binds_opening_statement_digest_for_n2_fixture() {
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
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
fn rv32im_main_proof_surface_is_unchanged_by_authoritative_phi_side() {
    use neo_fold_next::rv32im::main_proof::Rv32imMainFinalProofSurface;
    use neo_fold_next::rv32im::Rv32imAccumulatorPublicStatement;

    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");

    let surface = Rv32imMainFinalProofSurface::from_final_proof(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.accepted_artifact.statement.final_pc,
    );

    let statement = Rv32imAccumulatorPublicStatement::from_final_artifacts(
        &fixture.final_statement,
        &fixture.final_proof,
        fixture.accepted_artifact.statement.final_pc,
    )
    .expect("build published statement");

    assert_eq!(
        statement.expected_digest(),
        statement.expected_digest(),
        "published RV32IM main-proof accumulator statement must be deterministic under the canonical final surface"
    );
    assert_eq!(
        surface.final_pc(),
        fixture.accepted_artifact.statement.final_pc,
        "final surface must carry the authoritative final pc"
    );
}
