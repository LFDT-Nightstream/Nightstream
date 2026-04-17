use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::{
    check_ce_consistency, CcsStructure, CcsWitness, CeClaim, CeWitness, Mat, SModuleHomomorphism, SparsePoly, Term,
};
use neo_fold_next::rv64im::main_relation_circuit::ce_consistency::enforce_ce_consistency;
use neo_fold_next::rv64im::main_relation_circuit::ce_spartan::{
    prove_rv64im_ce_relation, setup_rv64im_ce_relation, verify_rv64im_ce_relation,
};
use neo_fold_next::rv64im::main_relation_circuit::claim::{alloc_ce_claim, me_digest_poseidon};
use neo_fold_next::rv64im::main_relation_circuit::witness::alloc_packed_witness;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{compute_y_zcol_from_witness_digits, project_x_from_witness_mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn toy_witness() -> CcsWitness<F> {
    let mut z = Mat::zero(D, 1, F::ZERO);
    z[(0, 0)] = F::from_u64(1);
    z[(1, 0)] = F::from_i64(-1);
    z[(2, 0)] = F::ONE;
    CcsWitness {
        w: vec![F::from_i64(-1), F::ONE],
        Z: z,
    }
}

fn toy_structure() -> CcsStructure<F> {
    let matrix = Mat::from_row_major(1, 3, vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)]);
    CcsStructure::new(
        vec![matrix],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1],
            }],
        ),
    )
    .expect("toy structure")
}

fn toy_claim(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    witness: &CcsWitness<F>,
) -> CeClaim<Commitment, F, K> {
    let x = project_x_from_witness_mat(&witness.Z, structure.m, 2).expect("project x");
    let s_col = vec![K::ZERO, K::ZERO];
    let chi_s = neo_ccs::utils::tensor_point::<K>(&s_col);
    let y_zcol = compute_y_zcol_from_witness_digits(params, &witness.Z, structure.m, &chi_s, D.next_power_of_two())
        .expect("compute y_zcol");
    let mut y_digits = vec![K::ZERO; D];
    y_digits[0] = K::from(F::from_u64(3));
    y_digits[1] = K::from(F::from_i64(-5));
    y_digits[2] = K::from(F::from_u64(7));
    let y_ring = vec![y_digits];
    CeClaim {
        c: log.commit(&witness.Z),
        X: x,
        r: Vec::new(),
        s_col,
        y_ring: y_ring.clone(),
        ct: vec![y_ring[0][0]],
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 2,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn ensure_toy_pp() {
    let _ = set_global_pp_seeded(D, 1, 1, [7u8; 32]);
}

fn assert_full_ce_circuit_satisfied(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
) {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let public_inputs = neo_reductions::engines::utils::me_digest_poseidon(claim)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc_input(cs.namespace(|| format!("claim_digest_input_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
            .expect("public input")
        })
        .collect::<Vec<_>>();
    let claim_var = alloc_ce_claim(&mut cs.namespace(|| "claim"), claim, "claim").expect("claim");
    let witness_var = alloc_packed_witness(&mut cs.namespace(|| "witness"), witness, "witness").expect("witness");
    let digest = me_digest_poseidon(&mut cs.namespace(|| "claim_digest"), &claim_var, "claim_digest").expect("digest");
    for (idx, (actual, expected)) in digest.iter().zip(public_inputs.iter()).enumerate() {
        cs.enforce(
            || format!("claim_digest_match_{idx}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + TestConstraintSystem::<SpartanF>::one(),
            |lc| lc + expected.get_variable(),
        );
    }
    enforce_ce_consistency(
        &mut cs.namespace(|| "ce_consistency"),
        params,
        structure,
        &witness_var,
        &claim_var,
        SpartanF::from_canonical_u64(7),
        "ce",
    )
    .expect("ce consistency");
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_ce_spartan_round_trip() {
    ensure_toy_pp();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let witness = toy_witness();
    let log = AjtaiSModule::from_global_for_dims(D, witness.Z.cols()).expect("ajtai log");
    let claim = toy_claim(&params, &structure, &log, &witness);

    check_ce_consistency(&params, &structure, &log, &claim, &CeWitness { Z: witness.Z.clone() })
        .expect("native ce consistency");
    assert_full_ce_circuit_satisfied(&params, &structure, &claim, &witness);

    let (pk, vk) =
        setup_rv64im_ce_relation(&params, &structure, &claim, &witness, F::from_u64(7)).expect("setup ce relation");
    let proof = prove_rv64im_ce_relation(&pk, &params, &structure, &claim, &witness, F::from_u64(7))
        .expect("prove ce relation");
    verify_rv64im_ce_relation(&vk, &claim, &proof).expect("verify ce relation");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_ce_spartan_rejects_tampered_claim() {
    ensure_toy_pp();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let witness = toy_witness();
    let log = AjtaiSModule::from_global_for_dims(D, witness.Z.cols()).expect("ajtai log");
    let claim = toy_claim(&params, &structure, &log, &witness);
    assert_full_ce_circuit_satisfied(&params, &structure, &claim, &witness);

    let (pk, vk) =
        setup_rv64im_ce_relation(&params, &structure, &claim, &witness, F::from_u64(7)).expect("setup ce relation");
    let proof = prove_rv64im_ce_relation(&pk, &params, &structure, &claim, &witness, F::from_u64(7))
        .expect("prove ce relation");

    let mut tampered = claim.clone();
    tampered.c.data[0] += F::ONE;
    let err = verify_rv64im_ce_relation(&vk, &tampered, &proof).expect_err("tampered claim must fail");
    assert!(format!("{err}").contains("public IO"));
}
