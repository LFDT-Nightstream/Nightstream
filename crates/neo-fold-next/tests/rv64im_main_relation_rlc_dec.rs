use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::pi_dec::{
    enforce_dec_public_non_commitment, enforce_dec_public_with_constant_children,
};
use neo_fold_next::rv64im::main_relation_circuit::pi_rlc::{
    enforce_rlc_public_non_commitment, enforce_rlc_public_non_commitment_with_rho_vars,
};
use neo_fold_next::rv64im::main_relation_circuit::rho_sampling::alloc_rot_rho_matrices_from_native;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn claim_from_scalars(x0: i64, x1: i64, y0: i64, y1: i64, aux: i64, yz0: i64, yz1: i64) -> CeClaim<Commitment, F, K> {
    let mut x = Mat::zero(D, 1, F::ZERO);
    x[(0, 0)] = F::from_i64(x0);
    x[(1, 0)] = F::from_i64(x1);

    let mut y_row = vec![K::ZERO; D.next_power_of_two()];
    y_row[0] = K::from(F::from_i64(y0));
    y_row[1] = K::from(F::from_i64(y1));

    let mut y_zcol = vec![K::ZERO; D.next_power_of_two()];
    y_zcol[0] = K::from(F::from_i64(yz0));
    y_zcol[1] = K::from(F::from_i64(yz1));

    let mut commitment = Commitment::zeros(D, 1);
    commitment.data[0] = F::from_i64(x0 + y0);
    commitment.data[1] = F::from_i64(x1 + y1);

    CeClaim {
        c: commitment,
        X: x,
        r: vec![K::from(F::from_u64(11)), K::from(F::from_u64(13))],
        s_col: vec![K::from(F::from_u64(17)), K::from(F::from_u64(19))],
        y_ring: vec![y_row.clone()],
        ct: vec![y_row[0]],
        aux_openings: vec![K::from(F::from_i64(aux))],
        y_zcol,
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn diag_rho(scale: i64) -> Mat<F> {
    let mut rho = Mat::zero(D, D, F::ZERO);
    for idx in 0..D {
        rho[(idx, idx)] = F::from_i64(scale);
    }
    rho
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_rlc_dec_gadgets_accept_public_equalities() {
    let child0 = claim_from_scalars(2, 3, 5, 7, 11, 13, 17);
    let child1 = claim_from_scalars(19, 23, 29, 31, 37, 41, 43);

    let rlc_parent = claim_from_scalars(
        2 + 2 * 19,
        3 + 2 * 23,
        5 + 2 * 29,
        7 + 2 * 31,
        11 + 2 * 37,
        13 + 2 * 41,
        17 + 2 * 43,
    );
    let dec_parent = claim_from_scalars(
        2 + 4 * 19,
        3 + 4 * 23,
        5 + 4 * 29,
        7 + 4 * 31,
        11 + 4 * 37,
        13 + 4 * 41,
        17 + 4 * 43,
    );

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let rlc_parent_var = alloc_ce_claim(&mut cs, &rlc_parent, "rlc_parent").expect("alloc rlc parent");
    let dec_parent_var = alloc_ce_claim(&mut cs, &dec_parent, "dec_parent").expect("alloc dec parent");
    let child0_var = alloc_ce_claim(&mut cs, &child0, "child0").expect("alloc child0");
    let child1_var = alloc_ce_claim(&mut cs, &child1, "child1").expect("alloc child1");

    enforce_rlc_public_non_commitment(
        &mut cs,
        &rlc_parent_var,
        &[child0_var.clone(), child1_var.clone()],
        &[diag_rho(1), diag_rho(2)],
        "rlc",
    )
    .expect("enforce rlc");
    enforce_dec_public_non_commitment(&mut cs, &dec_parent_var, &[child0_var, child1_var], 4, "dec")
        .expect("enforce dec");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_rlc_dec_gadgets_reject_tampered_parent() {
    let child0 = claim_from_scalars(2, 3, 5, 7, 11, 13, 17);
    let child1 = claim_from_scalars(19, 23, 29, 31, 37, 41, 43);
    let mut bad_parent = claim_from_scalars(
        2 + 2 * 19,
        3 + 2 * 23,
        5 + 2 * 29,
        7 + 2 * 31,
        11 + 2 * 37,
        13 + 2 * 41,
        17 + 2 * 43,
    );
    bad_parent.aux_openings[0] += K::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let parent_var = alloc_ce_claim(&mut cs, &bad_parent, "parent").expect("alloc parent");
    let child0_var = alloc_ce_claim(&mut cs, &child0, "child0").expect("alloc child0");
    let child1_var = alloc_ce_claim(&mut cs, &child1, "child1").expect("alloc child1");

    enforce_rlc_public_non_commitment(
        &mut cs,
        &parent_var,
        &[child0_var, child1_var],
        &[diag_rho(1), diag_rho(2)],
        "rlc",
    )
    .expect("enforce rlc");

    assert!(!cs.is_satisfied(), "tampered RLC parent must fail");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_dec_constant_children_accept_public_equalities() {
    let child0 = claim_from_scalars(2, 3, 5, 7, 11, 13, 17);
    let child1 = claim_from_scalars(19, 23, 29, 31, 37, 41, 43);
    let dec_parent = claim_from_scalars(
        2 + 4 * 19,
        3 + 4 * 23,
        5 + 4 * 29,
        7 + 4 * 31,
        11 + 4 * 37,
        13 + 4 * 41,
        17 + 4 * 43,
    );

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let dec_parent_var = alloc_ce_claim(&mut cs, &dec_parent, "dec_parent").expect("alloc dec parent");

    enforce_dec_public_with_constant_children(&mut cs, &dec_parent_var, &[child0, child1], 4, "dec_const")
        .expect("enforce dec const");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_dec_constant_children_reject_tampered_parent() {
    let child0 = claim_from_scalars(2, 3, 5, 7, 11, 13, 17);
    let child1 = claim_from_scalars(19, 23, 29, 31, 37, 41, 43);
    let mut bad_parent = claim_from_scalars(
        2 + 4 * 19,
        3 + 4 * 23,
        5 + 4 * 29,
        7 + 4 * 31,
        11 + 4 * 37,
        13 + 4 * 41,
        17 + 4 * 43,
    );
    bad_parent.aux_openings[0] += K::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let bad_parent_var = alloc_ce_claim(&mut cs, &bad_parent, "bad_parent").expect("alloc bad parent");

    enforce_dec_public_with_constant_children(&mut cs, &bad_parent_var, &[child0, child1], 4, "dec_const")
        .expect("enforce dec const");

    assert!(!cs.is_satisfied(), "tampered constant-child DEC parent must fail");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_rlc_var_gadget_accepts_public_equalities() {
    let child0 = claim_from_scalars(2, 3, 5, 7, 11, 13, 17);
    let child1 = claim_from_scalars(19, 23, 29, 31, 37, 41, 43);
    let parent = claim_from_scalars(
        2 + 2 * 19,
        3 + 2 * 23,
        5 + 2 * 29,
        7 + 2 * 31,
        11 + 2 * 37,
        13 + 2 * 41,
        17 + 2 * 43,
    );

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let parent_var = alloc_ce_claim(&mut cs, &parent, "parent").expect("alloc parent");
    let child0_var = alloc_ce_claim(&mut cs, &child0, "child0").expect("alloc child0");
    let child1_var = alloc_ce_claim(&mut cs, &child1, "child1").expect("alloc child1");
    let rho_vars =
        alloc_rot_rho_matrices_from_native(&mut cs, &[diag_rho(1), diag_rho(2)], "rho").expect("alloc rho mats");

    enforce_rlc_public_non_commitment_with_rho_vars(
        &mut cs,
        &parent_var,
        &[child0_var, child1_var],
        &rho_vars,
        "rlc_var",
    )
    .expect("enforce variable-rho rlc");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
