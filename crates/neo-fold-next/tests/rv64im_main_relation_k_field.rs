use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use ff::Field;
use neo_fold_next::rv64im::main_relation_circuit::k_field::{
    alloc_constant_k, alloc_k, enforce_k_eq, k_add, k_lift_from_f, k_mul, KNum,
};
use spartan2::provider::goldi::F as SpartanF;

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_k_field_gadgets_satisfy_basic_add_mul() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);
    let left = KNum::new(SpartanF::from_canonical_u64(3), SpartanF::from_canonical_u64(5));
    let right = KNum::new(SpartanF::from_canonical_u64(11), SpartanF::from_canonical_u64(13));

    let left_var = alloc_k(&mut cs, Some(left.clone()), "left").expect("left");
    let right_var = alloc_k(&mut cs, Some(right.clone()), "right").expect("right");
    let sum = k_add(
        &mut cs,
        &left_var,
        &right_var,
        Some(KNum::new(
            SpartanF::from_canonical_u64(14),
            SpartanF::from_canonical_u64(18),
        )),
        "sum",
    )
    .expect("sum");
    let product = k_mul(
        &mut cs,
        &left_var,
        &right_var,
        left.clone(),
        right.clone(),
        KNum::new(SpartanF::from_canonical_u64(488), SpartanF::from_canonical_u64(94)),
        delta,
        "product",
    )
    .expect("product");

    let expected_sum = alloc_constant_k(
        &mut cs,
        KNum::new(SpartanF::from_canonical_u64(14), SpartanF::from_canonical_u64(18)),
        "expected_sum",
    )
    .expect("expected sum");
    enforce_k_eq(&mut cs, &sum, &expected_sum, "sum_eq");

    let expected_product = alloc_constant_k(
        &mut cs,
        KNum::new(SpartanF::from_canonical_u64(488), SpartanF::from_canonical_u64(94)),
        "expected_product",
    )
    .expect("expected product");
    enforce_k_eq(&mut cs, &product, &expected_product, "product_eq");

    let scalar =
        bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| "scalar"), || Ok(SpartanF::from_canonical_u64(19)))
            .expect("scalar");
    let lifted = k_lift_from_f(&mut cs, scalar.get_variable(), "lifted").expect("lifted");
    let expected_lift = alloc_constant_k(
        &mut cs,
        KNum::new(SpartanF::from_canonical_u64(19), SpartanF::ZERO),
        "expected_lift",
    )
    .expect("expected lift");
    enforce_k_eq(&mut cs, &lifted, &expected_lift, "lift_eq");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
