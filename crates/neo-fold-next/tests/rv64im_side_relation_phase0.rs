use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_eval_claim_relation_from_accepted_artifact, circuit_derive_phase0_point,
    circuit_enforce_phase0_commitment_root_and_opened_object_digest, circuit_enforce_phase0_payload_eq,
    circuit_enforce_phase0_point_eq, circuit_evaluate_phase0_payload_from_packed_rows,
};
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
};
use spartan2::provider::goldi::F as SpartanF;
use std::collections::BTreeSet;

fn alloc_digest_vars(
    cs: &mut TestConstraintSystem<SpartanF>,
    digest: [u8; 32],
    label: &str,
) -> [bellpepper_core::num::AllocatedNum<SpartanF>; 4] {
    let values = [
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ];
    core::array::from_fn(|idx| {
        bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(values[idx]))
            .expect("alloc digest vars")
    })
}

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

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_relation_phase0_point_and_payload_match_native_claim_witness() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (_, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&accepted_artifact)
        .expect("build side eval-claim relation");

    let claim_witness = witness
        .claim_witnesses
        .iter()
        .find(|claim| claim.claim.payload.schema == neo_fold_next::rv64im::FamilyEvalSchemaId::Stage2RegisterWrites)
        .expect("stage2 register-writes phase0 claim witness");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let (opened_object_digest_vars, packed_column_matrix_entries) =
        circuit_enforce_phase0_commitment_root_and_opened_object_digest(
            cs.namespace(|| "phase0_commitment"),
            claim_witness.claim.payload.schema,
            &claim_witness.claim.commitment_context,
            &claim_witness.witness.packed_columns,
            &claim_witness.witness.commitment_vector,
            &claim_witness.claim.opened_object,
            "phase0_commitment",
        )
        .expect("enforce phase0 commitment root and opened-object digest");
    let binding_digest_vars = alloc_digest_vars(&mut cs, claim_witness.claim.binding_digest, "phase0_binding_digest");
    let (point_vars, point_values) = circuit_derive_phase0_point(
        cs.namespace(|| "phase0_point"),
        &opened_object_digest_vars,
        claim_witness.claim.opened_object.digest,
        &claim_witness.claim.commitment_context,
        claim_witness.claim.payload.schema,
        claim_witness.claim.id.slot,
        &binding_digest_vars,
        claim_witness.claim.binding_digest,
        claim_witness.claim.opened_object.row_domain_log_size as usize,
        "phase0_point",
    )
    .expect("derive phase0 point");
    circuit_enforce_phase0_point_eq(
        &mut cs.namespace(|| "phase0_point_eq"),
        &point_vars,
        &claim_witness.claim.point,
        "phase0_point_eq",
    )
    .expect("enforce phase0 point eq");
    assert_eq!(point_values, claim_witness.claim.point);

    let expected_payload = claim_witness
        .claim
        .payload
        .column_evals
        .iter()
        .map(|column| column.coeffs.to_vec())
        .collect::<Vec<_>>();
    let (payload_vars, payload_values) = circuit_evaluate_phase0_payload_from_packed_rows(
        &mut cs.namespace(|| "phase0_payload"),
        &claim_witness.witness.packed_columns,
        &packed_column_matrix_entries,
        &point_vars,
        &point_values,
        "phase0_payload",
    )
    .expect("evaluate phase0 payload");
    circuit_enforce_phase0_payload_eq(
        &mut cs.namespace(|| "phase0_payload_eq"),
        &payload_vars,
        &expected_payload,
        "phase0_payload_eq",
    )
    .expect("enforce phase0 payload eq");
    assert_eq!(payload_values, expected_payload);

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "manual diagnostic for the side Phase0 hotspot without the full perf test"]
fn rv64im_side_relation_phase0_component_counts() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (_, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&accepted_artifact)
        .expect("build side eval-claim relation");

    let mut seen = BTreeSet::new();
    for claim_witness in &witness.claim_witnesses {
        let schema = claim_witness.claim.payload.schema;
        if !seen.insert(schema) {
            continue;
        }

        let mut cs = TestConstraintSystem::<SpartanF>::new();
        let before_commitment = cs.num_constraints();
        let (opened_object_digest_vars, packed_column_matrix_entries) =
            circuit_enforce_phase0_commitment_root_and_opened_object_digest(
                cs.namespace(|| format!("{schema:?}_commitment")),
                schema,
                &claim_witness.claim.commitment_context,
                &claim_witness.witness.packed_columns,
                &claim_witness.witness.commitment_vector,
                &claim_witness.claim.opened_object,
                &format!("{schema:?}_commitment"),
            )
            .expect("enforce phase0 commitment root and opened-object digest");
        let after_commitment = cs.num_constraints();

        let binding_digest_vars = alloc_digest_vars(
            &mut cs,
            claim_witness.claim.binding_digest,
            &format!("{schema:?}_binding_digest"),
        );
        let (point_vars, point_values) = circuit_derive_phase0_point(
            cs.namespace(|| format!("{schema:?}_point")),
            &opened_object_digest_vars,
            claim_witness.claim.opened_object.digest,
            &claim_witness.claim.commitment_context,
            schema,
            claim_witness.claim.id.slot,
            &binding_digest_vars,
            claim_witness.claim.binding_digest,
            claim_witness.claim.opened_object.row_domain_log_size as usize,
            &format!("{schema:?}_point"),
        )
        .expect("derive phase0 point");
        let after_point = cs.num_constraints();

        circuit_enforce_phase0_point_eq(
            &mut cs.namespace(|| format!("{schema:?}_point_eq")),
            &point_vars,
            &claim_witness.claim.point,
            &format!("{schema:?}_point_eq"),
        )
        .expect("enforce phase0 point eq");
        let after_point_eq = cs.num_constraints();

        let expected_payload = claim_witness
            .claim
            .payload
            .column_evals
            .iter()
            .map(|column| column.coeffs.to_vec())
            .collect::<Vec<_>>();
        let (payload_vars, _payload_values) = circuit_evaluate_phase0_payload_from_packed_rows(
            &mut cs.namespace(|| format!("{schema:?}_payload")),
            &claim_witness.witness.packed_columns,
            &packed_column_matrix_entries,
            &point_vars,
            &point_values,
            &format!("{schema:?}_payload"),
        )
        .expect("evaluate phase0 payload");
        let after_payload = cs.num_constraints();

        circuit_enforce_phase0_payload_eq(
            &mut cs.namespace(|| format!("{schema:?}_payload_eq")),
            &payload_vars,
            &expected_payload,
            &format!("{schema:?}_payload_eq"),
        )
        .expect("enforce phase0 payload eq");
        let after_payload_eq = cs.num_constraints();

        assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
        println!(
            "{schema:?}: commitment={} point={} point_eq={} payload={} payload_eq={} total={}",
            after_commitment - before_commitment,
            after_point - after_commitment,
            after_point_eq - after_point,
            after_payload - after_point_eq,
            after_payload_eq - after_payload,
            after_payload_eq,
        );
    }
}
