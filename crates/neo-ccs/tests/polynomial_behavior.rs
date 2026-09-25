//! Direct behavior checks for sparse-polynomial evaluation and shape changes.

use neo_ccs::{SparsePoly, Term};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

/// Helper: build f(x0, x1) = 2*x0^2 + 3*x1
fn build_test_poly() -> SparsePoly<F> {
    SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::from_u64(2),
                exps: vec![2, 0], // 2 * x0^2
            },
            Term {
                coeff: F::from_u64(3),
                exps: vec![0, 1], // 3 * x1
            },
        ],
    )
}

// ---------------------------------------------------------------------------
// 1. eval_known_values
// ---------------------------------------------------------------------------
#[test]
fn eval_known_values() {
    let f = build_test_poly();

    // f(x0, x1) = 2*x0^2 + 3*x1
    // f(0, 0) = 0
    assert_eq!(f.eval(&[F::ZERO, F::ZERO]), F::ZERO);

    // f(1, 0) = 2*1 + 0 = 2
    assert_eq!(f.eval(&[F::ONE, F::ZERO]), F::from_u64(2));

    // f(0, 1) = 0 + 3 = 3
    assert_eq!(f.eval(&[F::ZERO, F::ONE]), F::from_u64(3));

    // f(1, 1) = 2 + 3 = 5
    assert_eq!(f.eval(&[F::ONE, F::ONE]), F::from_u64(5));

    // f(3, 5) = 2*9 + 3*5 = 18 + 15 = 33
    assert_eq!(f.eval(&[F::from_u64(3), F::from_u64(5)]), F::from_u64(33));

    // f(10, 7) = 2*100 + 3*7 = 200 + 21 = 221
    assert_eq!(f.eval(&[F::from_u64(10), F::from_u64(7)]), F::from_u64(221));
}

// ---------------------------------------------------------------------------
// 2. eval_wrong_arity_panics
// ---------------------------------------------------------------------------
#[test]
#[should_panic]
fn eval_wrong_arity_panics() {
    let f = build_test_poly(); // arity 2
                               // Passing 3 arguments should panic.
    let _ = f.eval(&[F::ONE, F::ONE, F::ONE]);
}

// ---------------------------------------------------------------------------
// 3. insert_var_at_front_preserves_eval
// ---------------------------------------------------------------------------
#[test]
fn insert_var_at_front_preserves_eval() {
    let f = build_test_poly(); // f(x0, x1) = 2*x0^2 + 3*x1

    let f2 = f.insert_var_at_front(); // f2(y, x0, x1) = 2*x0^2 + 3*x1 (y unused)
    assert_eq!(f2.arity(), 3, "arity should be 3 after insert_var_at_front");

    // f2(0, x0, x1) = f(x0, x1) for any x0, x1
    assert_eq!(
        f2.eval(&[F::ZERO, F::from_u64(3), F::from_u64(5)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );

    // Even with non-zero y, the result should be the same (since y is unused).
    assert_eq!(
        f2.eval(&[F::from_u64(999), F::from_u64(3), F::from_u64(5)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );
}

// ---------------------------------------------------------------------------
// 4. append_zero_vars_preserves_eval
// ---------------------------------------------------------------------------
#[test]
fn append_zero_vars_preserves_eval() {
    let f = build_test_poly(); // f(x0, x1) = 2*x0^2 + 3*x1

    let f2 = f.append_zero_vars(2); // f2(x0, x1, y0, y1) = 2*x0^2 + 3*x1
    assert_eq!(f2.arity(), 4, "arity should be 4 after appending 2 vars");

    // f2(x0, x1, 0, 0) = f(x0, x1)
    assert_eq!(
        f2.eval(&[F::from_u64(3), F::from_u64(5), F::ZERO, F::ZERO]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );

    // Non-zero trailing vars should still give the same result (they're unused).
    assert_eq!(
        f2.eval(&[F::from_u64(3), F::from_u64(5), F::from_u64(100), F::from_u64(200)]),
        f.eval(&[F::from_u64(3), F::from_u64(5)])
    );
}

// ---------------------------------------------------------------------------
// 5. max_degree_correct
// ---------------------------------------------------------------------------
#[test]
fn max_degree_correct() {
    let f = build_test_poly(); // terms: 2*x0^2 (degree 2), 3*x1 (degree 1)
    assert_eq!(f.max_degree(), 2, "max degree should be 2");

    // Single constant term.
    let f_const = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::from_u64(7),
            exps: vec![0],
        }],
    );
    assert_eq!(f_const.max_degree(), 0, "constant poly should have degree 0");

    // Cubic term: x0 * x1^2 (degree 3).
    let f_cubic = SparsePoly::new(
        2,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 2],
        }],
    );
    assert_eq!(f_cubic.max_degree(), 3, "x0*x1^2 should have degree 3");

    // Empty polynomial.
    let f_empty: SparsePoly<F> = SparsePoly::new(2, vec![]);
    assert_eq!(f_empty.max_degree(), 0, "empty poly should have degree 0");

    // R1CS polynomial f(X0,X1,X2) = X0*X1 - X2 => max_degree = 2.
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
