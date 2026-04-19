use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_eval_claim_relation_from_accepted_artifact, verify_rv64im_side_eval_claim_relation,
};
use neo_fold_next::nightstream::rv64im::audit::{
    measure_rv64im_side_binding_circuit_constraints, prove_rv64im_side_binding, setup_rv64im_side_binding,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_side_binding_statement, build_rv64im_side_proof, verify_rv64im_side_proof,
};

use super::common::{
    alternate_case_name, build_side_fixture, mutated_statement_with_new_core, refresh_side_proof, SideFixture,
};

const BASE_CASE: &str = "control_flow_jal_skip_ecall";

fn assert_side_proof_rejected(
    fixture: &SideFixture,
    side_proof: &neo_fold_next::nightstream::rv64im::Rv64imSideProof,
    expected_error_fragments: &[&str],
    context: &str,
) {
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let opening_vk = fixture.side_opening_vk();
    let (_, vk) = setup_rv64im_side_binding(&side_statement, fixture.side_public()).expect("setup side binding");
    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.public_statement,
        side_proof,
    )
    .expect_err(context);
    assert!(
        expected_error_fragments
            .iter()
            .any(|fragment| err.to_string().contains(fragment)),
        "unexpected rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_side_decider_rejects_container_spartan_splice() {
    let fixture_a = build_side_fixture(BASE_CASE);
    let fixture_b = build_side_fixture(&alternate_case_name(BASE_CASE));

    let side_statement_a = fixture_a
        .side_statement()
        .expect("build side binding statement A");
    let (_, vk_a) =
        setup_rv64im_side_binding(&side_statement_a, fixture_a.side_public()).expect("setup side binding A");
    let opening_vk_a = fixture_a.side_opening_vk();
    let proof_a = fixture_a.side_proof.clone();
    let proof_b = fixture_b.side_proof.clone();

    let mut spliced = proof_a.clone();
    *spliced.binding_mut() = proof_b.binding().clone();

    let err = verify_rv64im_side_proof(
        &opening_vk_a,
        &vk_a,
        &fixture_a.nightstream_statement,
        &fixture_a.public_statement,
        &spliced,
    )
    .expect_err("spliced side proof must be rejected");
    assert!(
        err.to_string().contains("side binding")
            || err.to_string().contains("public IO")
            || err.to_string().contains("public"),
        "unexpected splice rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_exact_cover_rejects_missing_or_duplicate_targets() {
    let fixture = build_side_fixture(BASE_CASE);
    let (statement, mut witness) =
        build_rv64im_side_eval_claim_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side eval-claim relation");

    let stage1_positions = witness
        .claim_witnesses
        .iter()
        .enumerate()
        .filter(|(_, claim)| claim.claim.payload.schema == neo_fold_next::rv64im::FamilyEvalSchemaId::Stage1Rows)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert!(stage1_positions.len() >= 2, "expected multiple Stage1Rows claims");
    witness.claim_witnesses[stage1_positions[1]] = witness.claim_witnesses[stage1_positions[0]].clone();

    let err = verify_rv64im_side_eval_claim_relation(&statement, &witness)
        .expect_err("duplicate target coverage must be rejected");
    assert!(
        err.to_string().contains("slot coverage")
            || err.to_string().contains("duplicate")
            || err.to_string().contains("side-eval-claim relation"),
        "unexpected exact-cover rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_cross_object_or_slot_replay_is_rejected() {
    let fixture = build_side_fixture(BASE_CASE);
    let (statement, mut witness) =
        build_rv64im_side_eval_claim_relation_from_accepted_artifact(&fixture.accepted_artifact)
            .expect("build side eval-claim relation");

    let stage1_position = witness
        .claim_witnesses
        .iter()
        .position(|claim| claim.claim.payload.schema == neo_fold_next::rv64im::FamilyEvalSchemaId::Stage1Rows)
        .expect("find Stage1Rows claim");
    let foreign_position = witness
        .claim_witnesses
        .iter()
        .position(|claim| claim.claim.payload.schema != neo_fold_next::rv64im::FamilyEvalSchemaId::Stage1Rows)
        .expect("find foreign-schema claim");
    let foreign_digest = witness.claim_witnesses[foreign_position]
        .claim
        .opened_object
        .digest;
    witness.claim_witnesses[stage1_position]
        .claim
        .id
        .opened_object_digest = foreign_digest;

    let err =
        verify_rv64im_side_eval_claim_relation(&statement, &witness).expect_err("cross-object replay must be rejected");
    assert!(
        err.to_string().contains("internally inconsistent")
            || err.to_string().contains("opened object")
            || err.to_string().contains("side-eval-claim relation"),
        "unexpected cross-object replay rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_linkage_rejects_transcript_final_digest_tamper() {
    let fixture = build_side_fixture(BASE_CASE);
    let mut tampered_public_statement = fixture.public_statement.clone();
    tampered_public_statement.transcript_final_digest[0] ^= 1;
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let opening_vk = fixture.side_opening_vk();
    let (_, vk) = setup_rv64im_side_binding(&side_statement, fixture.side_public()).expect("setup side binding");
    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &tampered_public_statement,
        &fixture.side_proof,
    )
    .expect_err("tampered transcript final digest must be rejected");
    assert!(
        err.to_string().contains("transcript final digest") || err.to_string().contains("public statement"),
        "unexpected transcript-final rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_linkage_rejects_opening_statement_surface_tamper() {
    let fixture = build_side_fixture(BASE_CASE);
    let mut side_proof = fixture.side_proof.clone();
    side_proof.opening_statement_mut().stage1.rows_digest[0] ^= 1;
    assert_side_proof_rejected(
        &fixture,
        &side_proof,
        &["side-opening statement", "surface", "rows digest", "verified claim"],
        "tampered side-opening statement surface must be rejected",
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_canonical_public_instance_rebuild_rejects_reordering() {
    let fixture = build_side_fixture(BASE_CASE);
    let mut side_proof = fixture.side_proof.clone();
    assert!(side_proof.opening_public().evals.len() >= 2, "expected multiple evals");
    side_proof.opening_public_mut().evals.swap(0, 1);
    refresh_side_proof(&mut side_proof);
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let opening_vk = fixture.side_opening_vk();
    let (_, vk) = setup_rv64im_side_binding(&side_statement, fixture.side_public()).expect("setup side binding");

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.public_statement,
        &side_proof,
    )
    .expect_err("reordered canonical public instance must be rejected");
    assert!(
        err.to_string().contains("public") || err.to_string().contains("side-opening"),
        "unexpected canonical rebuild rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_digest_binding_tracks_opening_statement_digest_bytes() {
    let fixture = build_side_fixture(BASE_CASE);
    let mut side_proof = fixture.side_proof.clone();
    let expected_digest_before = side_proof.expected_digest();
    side_proof.opening_statement_mut().stage1.digest[0] ^= 1;
    assert_ne!(
        expected_digest_before,
        side_proof.expected_digest(),
        "side proof digest must bind opening-statement digest bytes on the published boundary"
    );
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let opening_vk = fixture.side_opening_vk();
    let (_, vk) = setup_rv64im_side_binding(&side_statement, fixture.side_public()).expect("setup side binding");

    let err = verify_rv64im_side_proof(
        &opening_vk,
        &vk,
        &fixture.nightstream_statement,
        &fixture.public_statement,
        &side_proof,
    )
    .expect_err("digest binding must not bypass authoritative rebuild checks");
    assert!(
        err.to_string().contains("public") || err.to_string().contains("side-opening"),
        "unexpected digest-chain rejection error: {err}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_statement_digest_is_recomputed_not_trusted() {
    let fixture = build_side_fixture(BASE_CASE);
    let rebound_statement = mutated_statement_with_new_core(&fixture.nightstream_statement);
    let original_binding_statement = fixture
        .side_statement()
        .expect("build original side binding statement");
    let rebound_binding_statement =
        build_rv64im_side_binding_statement(&rebound_statement, fixture.side_proof.opening_public())
            .expect("recompute side binding statement under a new statement digest");
    assert_ne!(
        original_binding_statement.nightstream_statement_core_digest,
        rebound_binding_statement.nightstream_statement_core_digest,
        "side binding statement should recompute the carried Nightstream statement core digest from the external statement"
    );
    assert_ne!(
        original_binding_statement.digest(),
        rebound_binding_statement.digest(),
        "side binding statement digest should change when the carried Nightstream statement core digest changes"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_binding_rejects_forged_public_witness() {
    let fixture = build_side_fixture(BASE_CASE);
    let (_, phase0_witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&fixture.accepted_artifact)
        .expect("build side eval-claim relation");
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let (pk, vk) = setup_rv64im_side_binding(&side_statement, fixture.side_public()).expect("setup side binding");
    let mut forged_public = fixture.side_public().clone();
    forged_public.opened_objects[0].digest = [0x5a; 32];

    let err = prove_rv64im_side_binding(&pk, &side_statement, &forged_public, &phase0_witness.claim_witnesses)
        .expect_err("side binding should reject a forged side-opening public before proof generation");
    assert!(
        err.to_string()
            .contains("does not match the carried public eval")
            || err.to_string().contains("side binding prove path"),
        "unexpected forged-public rejection error: {err}"
    );
    let _ = vk;
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_binding_circuit_has_nonzero_constraints() {
    let fixture = build_side_fixture(BASE_CASE);
    let side_statement = fixture
        .side_statement()
        .expect("build side binding statement");
    let constraint_count = measure_rv64im_side_binding_circuit_constraints(&side_statement, fixture.side_public())
        .expect("measure side binding constraints");

    assert!(
        constraint_count > 0,
        "side binding circuit should carry real linkage constraints, not only public inputs"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_phase0_point_depends_on_statement_digest() {
    let fixture = build_side_fixture(BASE_CASE);
    let rebound_statement = mutated_statement_with_new_core(&fixture.nightstream_statement);
    let rebound_side_proof = build_rv64im_side_proof(&rebound_statement, &fixture.accepted_artifact)
        .expect("rebuild side proof under new statement core");

    let original_points = fixture
        .side_proof
        .opening_public()
        .evals
        .iter()
        .map(|eval| eval.claim.point.clone())
        .collect::<Vec<_>>();
    let rebound_points = rebound_side_proof
        .opening_public()
        .evals
        .iter()
        .map(|eval| eval.claim.point.clone())
        .collect::<Vec<_>>();

    assert_ne!(
        original_points, rebound_points,
        "Phase 0 points should change when the carried Nightstream statement core digest changes"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_soundness_positive_statement_rebinding_without_reopening_is_rejected() {
    let fixture = build_side_fixture(BASE_CASE);
    let rebound_statement = mutated_statement_with_new_core(&fixture.nightstream_statement);
    let rebound_side_proof = build_rv64im_side_proof(&rebound_statement, &fixture.accepted_artifact)
        .expect("rebuild side proof under new statement core");
    let opening_vk = fixture.side_opening_vk();

    assert!(
        verify_rv64im_side_proof(
            &opening_vk,
            &setup_rv64im_side_binding(
                &fixture
                    .side_statement()
                    .expect("build side binding statement"),
                fixture.side_public(),
            )
            .expect("setup side binding")
            .1,
            &rebound_statement,
            &fixture.public_statement,
            &rebound_side_proof,
        )
        .is_err(),
        "rebinding the same opening material to a different statement core should be rejected"
    );
}
