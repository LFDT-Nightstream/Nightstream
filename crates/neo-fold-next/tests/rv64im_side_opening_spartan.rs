use std::sync::Arc;

use bellpepper_core::{
    num::AllocatedNum, test_cs::TestConstraintSystem, Comparable, ConstraintSystem, Delta, SynthesisError,
};
use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_opening_relation_from_accepted_artifact, circuit_stage1_opening_packaged_statement_digest,
    debug_check_rv64im_side_opening_spartan_circuit, debug_compare_rv64im_side_opening_spartan_setup_shape,
    debug_compare_rv64im_side_opening_spartan_without_packaged_final_main_claims_shape,
    debug_compare_rv64im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape,
    debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape,
    debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape,
    debug_native_stage1_packaged_statement_digest,
    debug_round_trip_rv64im_stage1_packaged_opening_digest_with_reduced_setup,
    debug_setup_rv64im_side_opening_spartan_without_packaged_final_main_claims,
    debug_setup_rv64im_side_opening_spartan_without_stage1_packaged_final_main_claims,
    prove_rv64im_side_opening_spartan, setup_rv64im_side_opening_spartan, setup_rv64im_side_opening_spartan_cached,
    verify_rv64im_side_opening_spartan,
};
use neo_fold_next::rv64im::kernel::Stage1SelectedOpeningClaim;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use spartan2::{
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

#[derive(Clone)]
struct Stage1PackagedStatementDigestCircuit {
    claim: Stage1SelectedOpeningClaim,
    final_main_claim_digests: Vec<[F; 4]>,
    expected_statement_digest: [u8; 32],
}

impl Stage1PackagedStatementDigestCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        digest32_as_spartan_fields(self.expected_statement_digest).to_vec()
    }
}

impl SpartanCircuit<GoldilocksP3MerkleMleEngine> for Stage1PackagedStatementDigestCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let digest = circuit_stage1_opening_packaged_statement_digest(
            cs.namespace(|| "stage1_statement_digest"),
            &self.claim,
            &self.final_main_claim_digests,
            "stage1_statement_digest",
        )?;
        for idx in 0..4 {
            cs.enforce(
                || format!("digest_eq_{idx}"),
                |lc| lc + digest[idx].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public_inputs[idx].get_variable(),
            );
        }
        Ok(())
    }
}

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    core::array::from_fn(|idx| {
        SpartanF::from_canonical_u64(u64::from_le_bytes(
            digest[idx * 8..(idx + 1) * 8]
                .try_into()
                .expect("digest limb"),
        ))
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
fn rv64im_side_opening_spartan_round_trip() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");
    debug_check_rv64im_side_opening_spartan_circuit(&statement, &witness)
        .expect("debug-check side-opening spartan circuit");
    let keys = setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("setup side-opening spartan");
    let proof =
        prove_rv64im_side_opening_spartan(&keys.as_ref().0, &statement, &witness).expect("prove side-opening spartan");

    verify_rv64im_side_opening_spartan(&keys.as_ref().1, &statement, &proof).expect("verify side-opening spartan");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_rejects_tampered_stage1_row_witness() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");
    let (pk, _) = setup_rv64im_side_opening_spartan(&statement, &witness).expect("setup side-opening spartan");

    witness.stage1_selected_rows.first.fetched_word ^= 1;

    let err = prove_rv64im_side_opening_spartan(&pk, &statement, &witness)
        .expect_err("tampered stage1 selected row witness must fail");
    assert!(format!("{err}")
        .contains("RV64IM side-opening relation stage1 selected rows do not match the carried opening claim"));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_circuit_rejects_tampered_kernel_packaged_witness() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    witness.bindings_packaged.proof_digest[0] ^= 1;

    let err = debug_check_rv64im_side_opening_spartan_circuit(&statement, &witness)
        .expect_err("tampered kernel packaged witness must fail in-circuit");
    assert!(format!("{err}").contains("bindings_packaged"));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_cached_setup_reuses_same_shape_for_same_case() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let keys_a =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan A");
    let keys_b =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan B");

    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "same side-opening shape should reuse the cached setup keypair"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_cached_setup_ignores_selected_witness_replay() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let keys_a =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan A");

    let mut tampered_selected = witness.clone();
    tampered_selected.stage1_selected_rows.first.fetched_word ^= 1;
    let keys_b = setup_rv64im_side_opening_spartan_cached(&statement, &tampered_selected)
        .expect("cached setup side-opening spartan B");

    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "selected-witness replay should not affect side-opening setup reuse once packaged witnesses match"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_cached_setup_ignores_packaged_step_replay() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let keys_a =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan A");

    let mut tampered_packaged_step = witness.clone();
    tampered_packaged_step
        .stage1_packaged
        .step
        .label
        .push_str("/tampered");
    let keys_b = setup_rv64im_side_opening_spartan_cached(&statement, &tampered_packaged_step)
        .expect("cached setup side-opening spartan B");

    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "packaged step replay should not affect side-opening setup reuse once packaged digests match"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_cached_setup_ignores_packaged_proof_digest_replay() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let keys_a =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan A");

    let mut tampered_packaged_proof = witness.clone();
    tampered_packaged_proof.stage1_packaged.proof_digest[0] ^= 1;
    let keys_b = setup_rv64im_side_opening_spartan_cached(&statement, &tampered_packaged_proof)
        .expect("cached setup side-opening spartan B");

    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "packaged proof digest replay should not affect side-opening setup reuse once final-main-claim digests match"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_cached_setup_ignores_packaged_final_main_claim_digests() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let keys_a =
        setup_rv64im_side_opening_spartan_cached(&statement, &witness).expect("cached setup side-opening spartan A");

    let mut tampered_packaged_final_claims = witness.clone();
    tampered_packaged_final_claims
        .stage1_packaged
        .final_main_claim_digests[0][0] += F::new(1);
    let keys_b = setup_rv64im_side_opening_spartan_cached(&statement, &tampered_packaged_final_claims)
        .expect("cached setup side-opening spartan B");

    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "packaged final-main-claim digests should no longer affect side-opening setup reuse once the packaged statement digest is carried by the statement"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_reduced_setup_without_packaged_final_main_claims_round_trips() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let attempt = (|| {
        let (pk, vk) =
            debug_setup_rv64im_side_opening_spartan_without_packaged_final_main_claims(&statement, &witness)?;
        let proof = prove_rv64im_side_opening_spartan(&pk, &statement, &witness)?;
        verify_rv64im_side_opening_spartan(&vk, &statement, &proof)
    })();

    attempt.expect(
        "reducing setup by zeroing packaged final-main-claim digests should preserve the side-opening proof path",
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_reduced_setup_without_packaged_final_main_claims_verifies_after_prove() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let (pk, vk) = debug_setup_rv64im_side_opening_spartan_without_packaged_final_main_claims(&statement, &witness)
        .expect("reduced setup should still synthesize");

    let proof = prove_rv64im_side_opening_spartan(&pk, &statement, &witness).expect("reduced setup should still prove");
    verify_rv64im_side_opening_spartan(&vk, &statement, &proof)
        .expect("reduced setup should now verify once the packaged statement digest is carried by the statement");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_reduced_setup_without_stage1_packaged_final_main_claims_verifies_after_prove() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let (pk, vk) =
        debug_setup_rv64im_side_opening_spartan_without_stage1_packaged_final_main_claims(&statement, &witness)
            .expect("reduced stage1 setup should still synthesize");

    let proof =
        prove_rv64im_side_opening_spartan(&pk, &statement, &witness).expect("reduced stage1 setup should still prove");
    verify_rv64im_side_opening_spartan(&vk, &statement, &proof).expect(
        "reduced stage1 setup should now verify once the packaged statement digest is carried by the statement",
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_current_setup_shape_matches_real_circuit() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let delta = debug_compare_rv64im_side_opening_spartan_setup_shape(&statement, &witness)
        .expect("compare real side-opening circuit against current setup witness");
    assert!(
        delta.is_none(),
        "current side-opening setup witness should preserve the real circuit shape: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_zeroing_packaged_final_main_claims_preserves_circuit_shape() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let delta =
        debug_compare_rv64im_side_opening_spartan_without_packaged_final_main_claims_shape(&statement, &witness)
            .expect("compare real side-opening circuit against reduced setup witness");
    assert!(
        delta.is_none(),
        "zeroing packaged final-main-claim digests should no longer change the side-opening circuit shape once the packaged statement digest is carried by the statement: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_packaged_statement_digest_r1cs_is_stable_under_final_main_claim_changes() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let mut real_cs = TestConstraintSystem::<SpartanF>::new();
    circuit_stage1_opening_packaged_statement_digest(
        real_cs.namespace(|| "stage1"),
        &statement.stage1.claim,
        &witness.stage1_packaged.final_main_claim_digests,
        "stage1",
    )
    .expect("real stage1 packaged statement digest gadget must synthesize");

    let mut zeroed_final_claims = witness.stage1_packaged.final_main_claim_digests.clone();
    zeroed_final_claims.fill([F::ZERO; 4]);
    let mut zero_cs = TestConstraintSystem::<SpartanF>::new();
    circuit_stage1_opening_packaged_statement_digest(
        zero_cs.namespace(|| "stage1"),
        &statement.stage1.claim,
        &zeroed_final_claims,
        "stage1",
    )
    .expect("zeroed stage1 packaged statement digest gadget must synthesize");

    let delta = real_cs.delta(&zero_cs, false);
    assert!(
        matches!(delta, Delta::Equal),
        "stage1 packaged statement digest gadget should stay shape-stable under final-main-claim value changes: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_packaged_opening_digest_ignores_final_main_claims_in_setup_shape() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let delta = debug_compare_rv64im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape(
        &statement, &witness,
    )
    .expect("compare stage1 packaged-opening digest against reduced setup witness");
    assert!(
        delta.is_none(),
        "stage1 packaged-opening digest wrapper should no longer expose a final-main-claim setup dependency once the packaged statement digest is carried by the statement: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_statement_digest_reduced_setup_still_round_trips() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let expected_statement_digest = debug_native_stage1_packaged_statement_digest(&statement, &witness.stage1_packaged)
        .expect("compute native stage1 packaged statement digest");

    let mut reduced_final_main_claim_digests = witness.stage1_packaged.final_main_claim_digests.clone();
    reduced_final_main_claim_digests.fill([F::ZERO; 4]);

    let setup_circuit = Stage1PackagedStatementDigestCircuit {
        claim: statement.stage1.claim.clone(),
        final_main_claim_digests: reduced_final_main_claim_digests,
        expected_statement_digest,
    };
    let (pk, vk) =
        R1CSSNARK::<GoldilocksP3MerkleMleEngine>::setup(setup_circuit).expect("setup reduced stage1 digest circuit");

    let prove_circuit = Stage1PackagedStatementDigestCircuit {
        claim: statement.stage1.claim.clone(),
        final_main_claim_digests: witness.stage1_packaged.final_main_claim_digests.clone(),
        expected_statement_digest,
    };
    let prep = R1CSSNARK::<GoldilocksP3MerkleMleEngine>::prep_prove(&pk, prove_circuit.clone(), true)
        .expect("prepare prove stage1 digest circuit");
    let proof = R1CSSNARK::<GoldilocksP3MerkleMleEngine>::prove(&pk, prove_circuit, &prep, true)
        .expect("prove stage1 digest circuit");

    let public_values = proof
        .verify(&vk)
        .expect("stage1 packaged statement digest path should still verify under reduced setup");
    assert_eq!(
        public_values,
        digest32_as_spartan_fields(expected_statement_digest).to_vec(),
        "stage1 packaged statement digest proof should recover the expected public digest under reduced setup"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_packaged_opening_digest_reduced_setup_round_trips() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    debug_round_trip_rv64im_stage1_packaged_opening_digest_with_reduced_setup(&statement, &witness)
        .expect("minimal stage1 packaged-opening wrapper should now round-trip under reduced setup");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_packaged_opening_digest_zeroing_only_final_main_claims_preserves_shape() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let delta =
        debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape(&statement, &witness)
            .expect("compare stage1 packaged-opening wrapper against zeroed final-main-claims variant");
    assert!(
        delta.is_none(),
        "zeroing only stage1 packaged final-main-claim digests should no longer change the minimal packaged-opening wrapper once the packaged statement digest is carried by the statement, got: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_stage1_packaged_opening_digest_fixed_statement_stays_shape_stable() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    let delta =
        debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape(
            &statement, &witness,
        )
        .expect("compare stage1 packaged-opening wrapper against zeroed final-main-claims variant with fixed native statement digest");
    assert!(
        delta.is_none(),
        "holding the packaged statement digest fixed should still leave the minimal wrapper shape stable, got: {delta:?}"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_spartan_rejects_wrong_packaged_final_main_claim_width_at_prove() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    witness.bindings_packaged.final_main_claim_digests.pop();

    let err = match setup_rv64im_side_opening_spartan(&statement, &witness) {
        Ok((pk, _)) => {
            prove_rv64im_side_opening_spartan(&pk, &statement, &witness).expect_err("wrong packaged width must fail")
        }
        Err(err) => err,
    };
    assert!(format!("{err}").contains("frozen exact-package width"));
}
