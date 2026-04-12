use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::Commitment;
use neo_ccs::{CcsWitness, CeClaim, Mat};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::witness::{
    alloc_balanced_digit_witness, alloc_packed_witness, compute_digit_y_zcol, enforce_claim_y_zcol,
    enforce_x_projection,
};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{compute_y_zcol_from_witness_digits, project_x_from_witness_mat};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn toy_claim_from_witness(
    params: &NeoParams,
    witness: &CcsWitness<F>,
    expected_m: usize,
    m_in: usize,
) -> CeClaim<Commitment, F, K> {
    let x = project_x_from_witness_mat(&witness.Z, expected_m, m_in).expect("project x");
    let s_col = vec![K::ZERO, K::ZERO];
    let chi_s = neo_ccs::utils::tensor_point::<K>(&s_col);
    let y_zcol = compute_y_zcol_from_witness_digits(params, &witness.Z, expected_m, &chi_s, D.next_power_of_two())
        .expect("compute y_zcol");
    CeClaim {
        c: Commitment::zeros(D, 1),
        X: x,
        r: Vec::new(),
        s_col,
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn toy_witness() -> CcsWitness<F> {
    let mut z = Mat::zero(D, 1, F::ZERO);
    z[(0, 0)] = F::from_u64(1);
    z[(1, 0)] = F::from_i64(-1);
    z[(2, 0)] = F::ZERO;
    CcsWitness {
        w: vec![F::from_i64(-1), F::ZERO],
        Z: z,
    }
}

#[test]
fn rv64im_main_relation_witness_gadgets_accept_consistent_projection_range_and_y_zcol() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let expected_m = 3;
    let m_in = 2;
    let witness = toy_witness();
    let claim = toy_claim_from_witness(&params, &witness, expected_m, m_in);
    let chi_s = neo_ccs::utils::tensor_point::<K>(&claim.s_col);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let witness_var = alloc_packed_witness(&mut cs, &witness, "witness").expect("alloc witness");
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");

    enforce_x_projection(&mut cs, &witness_var, &claim_var, expected_m, "x_projection").expect("x projection");
    let digit_witness = alloc_balanced_digit_witness(
        &mut cs,
        &witness_var,
        expected_m,
        &params,
        SpartanF::from_canonical_u64(7),
        "digits",
    )
    .expect("alloc digit witness");
    let computed_y_zcol = compute_digit_y_zcol(
        &mut cs,
        &digit_witness,
        expected_m,
        &chi_s,
        claim.y_zcol.len(),
        SpartanF::from_canonical_u64(7),
        "y_zcol",
    )
    .expect("compute y_zcol");
    enforce_claim_y_zcol(&mut cs, &computed_y_zcol, &claim_var, "y_zcol_eq").expect("enforce y_zcol");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
fn rv64im_main_relation_witness_gadgets_reject_tampered_projection() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let expected_m = 3;
    let m_in = 2;
    let witness = toy_witness();
    let mut claim = toy_claim_from_witness(&params, &witness, expected_m, m_in);
    claim.X[(0, 0)] += F::ONE;
    let chi_s = neo_ccs::utils::tensor_point::<K>(&claim.s_col);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let witness_var = alloc_packed_witness(&mut cs, &witness, "witness").expect("alloc witness");
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");

    enforce_x_projection(&mut cs, &witness_var, &claim_var, expected_m, "x_projection").expect("x projection");
    let digit_witness = alloc_balanced_digit_witness(
        &mut cs,
        &witness_var,
        expected_m,
        &params,
        SpartanF::from_canonical_u64(7),
        "digits",
    )
    .expect("alloc digit witness");
    let computed_y_zcol = compute_digit_y_zcol(
        &mut cs,
        &digit_witness,
        expected_m,
        &chi_s,
        claim.y_zcol.len(),
        SpartanF::from_canonical_u64(7),
        "y_zcol",
    )
    .expect("compute y_zcol");
    enforce_claim_y_zcol(&mut cs, &computed_y_zcol, &claim_var, "y_zcol_eq").expect("enforce y_zcol");

    assert!(!cs.is_satisfied(), "tampered X projection must fail constraints");
}
