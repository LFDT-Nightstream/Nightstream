use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_opening_relation_witness_from_accepted_artifact,
    build_rv64im_side_proof_bundle_from_accepted_artifact, circuit_continuity_event_digest, circuit_ram_event_digest,
    circuit_register_read_event_digest, circuit_register_write_event_digest, circuit_stage1_row_digest,
    circuit_twist_link_event_digest,
};
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
};
use spartan2::provider::goldi::F as SpartanF;

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    [
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[0..8].try_into().expect("limb0"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[8..16].try_into().expect("limb1"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[16..24].try_into().expect("limb2"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[24..32].try_into().expect("limb3"))),
    ]
}

fn enforce_digest_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: [SpartanF; 4],
    label: &str,
) -> Result<(), bellpepper_core::SynthesisError> {
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let expected_var = AllocatedNum::alloc(cs.namespace(|| format!("{label}_expected_{idx}")), || Ok(*expected))?;
        cs.enforce(
            || format!("{label}_expected_const_{idx}"),
            |lc| lc + expected_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (*expected, CS::one()),
        );
        cs.enforce(
            || format!("{label}_eq_{idx}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected_var.get_variable(),
        );
    }
    Ok(())
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_relation_circuit_selected_opening_digests_match_carried_claims() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&accepted_artifact).expect("build side bundle");
    let witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(&accepted_artifact);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let stage1_first = circuit_stage1_row_digest(
        cs.namespace(|| "stage1_first"),
        &witness.stage1_selected_rows.first,
        "stage1_first",
    )
    .expect("stage1 digest");
    enforce_digest_eq(
        &mut cs.namespace(|| "stage1_first_eq"),
        &stage1_first,
        digest32_as_spartan_fields(side_bundle.stage1.claim.points.first.value_digest),
        "stage1_first_eq",
    )
    .expect("stage1 first eq");

    match (
        witness.stage2_selected_events.first_read.as_ref(),
        side_bundle
            .stage2
            .claim
            .points
            .first_read
            .as_ref()
            .map(|reference| reference.value_digest),
    ) {
        (Some(event), Some(digest)) => {
            let actual =
                circuit_register_read_event_digest(cs.namespace(|| "stage2_first_read"), event, "stage2_first_read")
                    .expect("stage2 first read digest");
            enforce_digest_eq(
                &mut cs.namespace(|| "stage2_first_read_eq"),
                &actual,
                digest32_as_spartan_fields(digest),
                "stage2_first_read_eq",
            )
            .expect("stage2 read eq");
        }
        (None, None) => {}
        _ => panic!("stage2 first_read presence mismatch"),
    }

    match (
        witness.stage2_selected_events.first_write.as_ref(),
        side_bundle
            .stage2
            .claim
            .points
            .first_write
            .as_ref()
            .map(|reference| reference.value_digest),
    ) {
        (Some(event), Some(digest)) => {
            let actual =
                circuit_register_write_event_digest(cs.namespace(|| "stage2_first_write"), event, "stage2_first_write")
                    .expect("stage2 first write digest");
            enforce_digest_eq(
                &mut cs.namespace(|| "stage2_first_write_eq"),
                &actual,
                digest32_as_spartan_fields(digest),
                "stage2_first_write_eq",
            )
            .expect("stage2 write eq");
        }
        (None, None) => {}
        _ => panic!("stage2 first_write presence mismatch"),
    }

    match (
        witness.stage2_selected_events.first_ram.as_ref(),
        side_bundle
            .stage2
            .claim
            .points
            .first_ram
            .as_ref()
            .map(|reference| reference.value_digest),
    ) {
        (Some(event), Some(digest)) => {
            let actual = circuit_ram_event_digest(cs.namespace(|| "stage2_first_ram"), event, "stage2_first_ram")
                .expect("stage2 first ram digest");
            enforce_digest_eq(
                &mut cs.namespace(|| "stage2_first_ram_eq"),
                &actual,
                digest32_as_spartan_fields(digest),
                "stage2_first_ram_eq",
            )
            .expect("stage2 ram eq");
        }
        (None, None) => {}
        _ => panic!("stage2 first_ram presence mismatch"),
    }

    match (
        witness.stage2_selected_events.first_twist.as_ref(),
        side_bundle
            .stage2
            .claim
            .points
            .first_twist
            .as_ref()
            .map(|reference| reference.value_digest),
    ) {
        (Some(event), Some(digest)) => {
            let actual =
                circuit_twist_link_event_digest(cs.namespace(|| "stage2_first_twist"), event, "stage2_first_twist")
                    .expect("stage2 first twist digest");
            enforce_digest_eq(
                &mut cs.namespace(|| "stage2_first_twist_eq"),
                &actual,
                digest32_as_spartan_fields(digest),
                "stage2_first_twist_eq",
            )
            .expect("stage2 twist eq");
        }
        (None, None) => {}
        _ => panic!("stage2 first_twist presence mismatch"),
    }

    match (
        witness.stage3_selected_continuity.first_continuity.as_ref(),
        side_bundle
            .stage3
            .claim
            .points
            .first_continuity
            .as_ref()
            .map(|reference| reference.value_digest),
    ) {
        (Some(event), Some(digest)) => {
            let actual = circuit_continuity_event_digest(cs.namespace(|| "stage3_first"), event, "stage3_first")
                .expect("stage3 first continuity digest");
            enforce_digest_eq(
                &mut cs.namespace(|| "stage3_first_eq"),
                &actual,
                digest32_as_spartan_fields(digest),
                "stage3_first_eq",
            )
            .expect("stage3 first eq");
        }
        (None, None) => {}
        _ => panic!("stage3 first continuity presence mismatch"),
    }

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
