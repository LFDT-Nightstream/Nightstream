//! Direct behavior checks for sparse-polynomial evaluation and shape changes.

use neo_ccs::{SparsePoly, Term};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

fn build_test_poly() -> SparsePoly<F> {
    SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::from_u64(2),
                exps: vec![2, 0],
            },
            Term {
                coeff: F::from_u64(3),
                exps: vec![0, 1],
            },
        ],
    )
}

#[test]
fn eval_known_values() {
    let f = build_test_poly();

    assert_eq!(f.eval(&[F::ZERO, F::ZERO]), F::ZERO);

    assert_eq!(f.eval(&[F::ONE, F::ZERO]), F::from_u64(2));

    assert_eq!(f.eval(&[F::ZERO, F::ONE]), F::from_u64(3));

    assert_eq!(f.eval(&[F::ONE, F::ONE]), F::from_u64(5));

    assert_eq!(f.eval(&[F::from_u64(3), F::from_u64(5)]), F::from_u64(33));

    assert_eq!(f.eval(&[F::from_u64(10), F::from_u64(7)]), F::from_u64(221));
}

#[test]
#[should_panic]
fn eval_wrong_arity_panics() {
    let f = build_test_poly();
    let _ = f.eval(&[F::ONE, F::ONE, F::ONE]);
}

#[test]
fn insert_var_at_front_preserves_eval() {
    let f = build_test_poly();

    let f2 = f.insert_var_at_front();
    assert_eq!(f2.arity(), 3, "arity should be 3 after insert_var_at_front");

    assert_eq!(
        f2.eval(&[F::ZERO, F::from_u64(3), F::from_u64(5)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );

    assert_eq!(
        f2.eval(&[F::from_u64(999), F::from_u64(3), F::from_u64(5)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );
}

#[test]
fn append_zero_vars_preserves_eval() {
    let f = build_test_poly();

    let f2 = f.append_zero_vars(2);
    assert_eq!(f2.arity(), 4, "arity should be 4 after appending 2 vars");

    assert_eq!(
        f2.eval(&[F::from_u64(3), F::from_u64(5), F::ZERO, F::ZERO]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );

    assert_eq!(
        f2.eval(&[F::from_u64(3), F::from_u64(5), F::from_u64(100), F::from_u64(200)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );
}

#[test]
fn max_degree_correct() {
    let f = build_test_poly();
    assert_eq!(f.max_degree(), 2, "max degree should be 2");

    let f_const = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::from_u64(7),
            exps: vec![0],
        }],
    );
    assert_eq!(f_const.max_degree(), 0, "constant poly should have degree 0");

    let f_cubic = SparsePoly::new(
        2,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 2],
        }],
    );
    assert_eq!(f_cubic.max_degree(), 3, "x0*x1^2 should have degree 3");

    let f_empty: SparsePoly<F> = SparsePoly::new(2, vec![]);
    assert_eq!(f_empty.max_degree(), 0, "empty poly should have degree 0");

    let f_r1cs = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            },
        ],
    );
    assert_eq!(f_r1cs.max_degree(), 2, "R1CS poly X0*X1 - X2 has degree 2");
}
