use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_ccs::Mat;
use neo_fold_next::rv64im::main_relation_circuit::rho_sampling::alloc_rot_rho_matrices_from_native;
use neo_fold_next::rv64im::main_relation_circuit::witness::alloc_packed_mat_witness;
use neo_fold_next::rv64im::main_relation_circuit::witness_transition::{
    alloc_split_children_from_native, enforce_packed_dec_split, mix_packed_witnesses_with_rho_vars,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn diag_rho(scale: i64) -> Mat<F> {
    let mut rho = Mat::zero(D, D, F::ZERO);
    for idx in 0..D {
        rho[(idx, idx)] = F::from_i64(scale);
    }
    rho
}

#[test]
fn rv64im_main_relation_witness_transition_accepts_rlc_and_dec() {
    let mut z0 = Mat::zero(D, 1, F::ZERO);
    z0[(0, 0)] = F::from_u64(2);
    z0[(1, 0)] = F::from_u64(3);
    let mut z1 = Mat::zero(D, 1, F::ZERO);
    z1[(0, 0)] = F::from_u64(5);
    z1[(1, 0)] = F::from_u64(7);

    let mut parent = Mat::zero(D, 1, F::ZERO);
    parent[(0, 0)] = F::from_u64(2 + 2 * 5);
    parent[(1, 0)] = F::from_u64(3 + 2 * 7);

    let mut child0 = Mat::zero(D, 1, F::ZERO);
    child0[(0, 0)] = F::ZERO;
    child0[(1, 0)] = F::ONE;
    let mut child1 = Mat::zero(D, 1, F::ZERO);
    child1[(0, 0)] = F::from_u64(3);
    child1[(1, 0)] = F::from_u64(4);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let z0_var = alloc_packed_mat_witness(&mut cs, &z0, "z0").expect("alloc z0");
    let z1_var = alloc_packed_mat_witness(&mut cs, &z1, "z1").expect("alloc z1");
    let rho_vars = alloc_rot_rho_matrices_from_native(&mut cs, &[diag_rho(1), diag_rho(2)], "rho").expect("alloc rho");
    let mixed = mix_packed_witnesses_with_rho_vars(&mut cs, &[z0_var, z1_var], &rho_vars, "mix").expect("mix");
    let parent_var = alloc_packed_mat_witness(&mut cs, &parent, "parent").expect("alloc parent");
    let child_vars = alloc_split_children_from_native(&mut cs, &[child0, child1], "child").expect("alloc children");

    for row in 0..D {
        for col in 0..mixed.cols() {
            cs.enforce(
                || format!("mixed_eq_{row}_{col}"),
                |lc| lc + mixed.entry(row, col).expect("entry").get_variable(),
                |lc| lc + TestConstraintSystem::<SpartanF>::one(),
                |lc| lc + parent_var.entry(row, col).expect("entry").get_variable(),
            );
        }
    }
    enforce_packed_dec_split(&mut cs, &parent_var, &child_vars, 4, "dec").expect("dec split");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
fn rv64im_main_relation_witness_transition_rejects_bad_split() {
    let mut parent = Mat::zero(D, 1, F::ZERO);
    parent[(0, 0)] = F::from_u64(9);
    let mut child0 = Mat::zero(D, 1, F::ZERO);
    child0[(0, 0)] = F::from_u64(1);
    let mut child1 = Mat::zero(D, 1, F::ZERO);
    child1[(0, 0)] = F::from_u64(1);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let parent_var = alloc_packed_mat_witness(&mut cs, &parent, "parent").expect("alloc parent");
    let child_vars = alloc_split_children_from_native(&mut cs, &[child0, child1], "child").expect("alloc children");
    enforce_packed_dec_split(&mut cs, &parent_var, &child_vars, 4, "dec").expect("dec split");

    assert!(!cs.is_satisfied(), "bad DEC split must fail");
}
