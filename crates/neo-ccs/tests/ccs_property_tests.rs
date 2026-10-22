//! Property tests for CCS implementation correctness

use neo_ccs::{
    matrix::Mat,
    poly::{SparsePoly, Term},
    r1cs::r1cs_to_ccs,
    relations::check_ccs_rowwise_relaxed,
    utils::{tensor_point, tensor_point_parallel, validate_power_of_two},
    CcsStructure,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_relaxed_rhs_formula() {
    let terms = vec![Term {
        coeff: F::ONE,
        exps: vec![1, 0],
    }];
    let f = SparsePoly::new(2, terms);

    let m0 = Mat::from_row_major(1, 2, vec![F::ONE, F::TWO]);
    let m1 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    let ccs = CcsStructure::new(vec![m0, m1], f).unwrap();

    let x = vec![F::from_u64(3)];
    let w = vec![F::from_u64(4)];

    let u = vec![F::from_u64(7)];
    let e = F::from_u64(2);

    assert!(check_ccs_rowwise_relaxed(&ccs, &x, &w, Some(&u), Some(e)).is_err());

    let u_correct = vec![F::ONE];
    let e_correct = F::from_u64(11);
    assert!(check_ccs_rowwise_relaxed(&ccs, &x, &w, Some(&u_correct), Some(e_correct)).is_ok());

    assert!(check_ccs_rowwise_relaxed(&ccs, &x, &w, None, None).is_err());
}

#[test]
fn test_tensor_point_sum_invariant() {
    for ell in 1..=4 {
        let r: Vec<F> = (0..ell).map(|i| F::from_u64(17 + i as u64)).collect();

        let tensor = tensor_point(&r);
        let n = 1usize << ell;
        assert_eq!(tensor.len(), n);

        let sum = tensor.iter().fold(F::ZERO, |acc, &x| acc + x);
        assert_eq!(sum, F::ONE, "tensor_point sum failed for ell={}", ell);
    }
}

#[test]
fn test_tensor_point_nested_comparison() {
    let r = vec![F::from_u64(3), F::from_u64(5)];
    let tensor = tensor_point(&r);

    let r0 = r[0];
    let r1 = r[1];
    let expected = vec![
        (F::ONE - r0) * (F::ONE - r1),
        r0 * (F::ONE - r1),
        (F::ONE - r0) * r1,
        r0 * r1,
    ];

    assert_eq!(tensor, expected, "tensor_point expansion failed for ell=2");

    let r1 = vec![F::from_u64(7)];
    let tensor1 = tensor_point(&r1);
    let expected1 = vec![F::ONE - r1[0], r1[0]];
    assert_eq!(tensor1, expected1, "tensor_point expansion failed for ell=1");
}

#[test]
fn test_tensor_point_parallel_matches_serial() {
    for ell in [0usize, 1, 2, 5, 14] {
        let r: Vec<F> = (0..ell).map(|i| F::from_u64(19 + 7 * i as u64)).collect();
        assert_eq!(tensor_point_parallel(&r), tensor_point(&r), "ell={ell}");
    }
}

#[test]
fn test_r1cs_ccs_roundtrip() {
    let a_proper = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let b_proper = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);
    let c_proper = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);

    let ccs_proper = r1cs_to_ccs(a_proper, b_proper, c_proper);

    let witness_proper = vec![F::ONE, F::from_u64(5)];
    let x_empty = vec![];

    assert!(neo_ccs::relations::check_ccs_rowwise_zero(&ccs_proper, &x_empty, &witness_proper).is_ok());

    let x_wrong = vec![F::from_u64(2)];
    let w_wrong = vec![F::from_u64(5)];
    assert!(neo_ccs::relations::check_ccs_rowwise_zero(&ccs_proper, &x_wrong, &w_wrong).is_err());
}

#[test]
fn test_power_of_two_validation() {
    assert!(validate_power_of_two(1));
    assert!(validate_power_of_two(2));
    assert!(validate_power_of_two(4));
    assert!(validate_power_of_two(8));
    assert!(validate_power_of_two(16));

    assert!(!validate_power_of_two(0));
    assert!(!validate_power_of_two(3));
    assert!(!validate_power_of_two(5));
    assert!(!validate_power_of_two(6));
    assert!(!validate_power_of_two(7));
    assert!(!validate_power_of_two(9));
}
