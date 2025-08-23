use neo_ccs::*;
use neo_fields::{from_base, ExtF, F};
use neo_sumcheck::fiat_shamir::fiat_shamir_challenge;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn test_satisfiability() {
    let _n = 2; // Num constraints
    let m = 3; // Witness size
    let a = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], m); // A matrix
    let b = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO], m); // B
    let c = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE], m); // C
    let mats = vec![a, b, c];

    // f = X0 + X1 - X2 (changed to multilinear)
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() != 3 {
                ExtF::ZERO
            } else {
                inputs[0] + inputs[1] - inputs[2]
            }
        },
        1,
    );

    let structure = CcsStructure::new(mats, f);

    // Valid witness: z = [1, 2, 3] for both rows (1+2-3=0, 1+2-3=0)
    let witness = CcsWitness {
        z: vec![
            from_base(F::ONE),
            from_base(F::from_u64(2)),
            from_base(F::from_u64(3)),
        ],
    };

    let instance = CcsInstance {
        commitment: vec![], // Stub, not checked here
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    assert!(check_satisfiability(&structure, &instance, &witness));

    // Invalid: change to [1,2,4] (1+2-4=-1≠0)
    let bad_witness = CcsWitness {
        z: vec![
            from_base(F::ONE),
            from_base(F::from_u64(2)),
            from_base(F::from_u64(4)),
        ],
    };
    assert!(!check_satisfiability(&structure, &instance, &bad_witness));
}

#[test]
fn test_linear_f() {
    let m = 1;
    let mat = RowMajorMatrix::new(vec![F::ONE], m);
    let mats = vec![mat];
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 1 {
                inputs[0] // Linear (multilinear)
            } else {
                ExtF::ZERO
            }
        },
        1,
    );
    let structure = CcsStructure::new(mats, f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![from_base(F::from_u64(2))],
    };
    let mut transcript = vec![];
    let msgs = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut transcript,
    )
    .expect("sumcheck");
    let mut current = from_base(F::from_u64(2)); // Expected value since f(x) = x
    for (uni, _) in &msgs {
        current = uni.eval(fiat_shamir_challenge(&mut vec![]));
    }
    assert_eq!(current, from_base(F::from_u64(2)));
}

#[test]
fn test_public_inputs() {
    let m = 1;
    let mat = RowMajorMatrix::new(vec![F::ZERO], m);
    let mats = vec![mat];
    let f = mv_poly(|inputs: &[ExtF]| inputs[0], 1);
    let structure = CcsStructure::new(mats, f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![F::ONE],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness { z: vec![] };
    let mut transcript = vec![];
    let msgs = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut transcript,
    )
    .expect("sumcheck");
    let mut current = from_base(F::ONE);
    for (uni, _) in &msgs {
        current = uni.eval(fiat_shamir_challenge(&mut vec![]));
    }
    assert_eq!(current, from_base(F::ONE));
}

#[test]
fn test_multilinear_sumcheck_valid() {
    // Simple test for multilinear constraints
    let witness_size = 3;
    
    // Simple matrices for a + b = c constraint
    let m0 = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO], witness_size);  // selects a
    let m1 = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO], witness_size);  // selects b  
    let m2 = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE], witness_size);  // selects c
    let mats = vec![m0, m1, m2];

    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] + inputs[1] - inputs[2] // a + b - c = 0
            } else {
                ExtF::ZERO
            }
        },
        1,
    );

    let structure = CcsStructure::new(mats, f);

    // Valid witness: a=3, b=4, c=7 (3+4=7)
    let z_base = vec![F::from_u64(3), F::from_u64(4), F::from_u64(7)];
    let z: Vec<ExtF> = z_base.into_iter().map(from_base).collect();
    let witness = CcsWitness { z };

    let mut transcript = vec![];
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let _msgs = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        0, // No norm bound for this test
        &mut transcript,
    )
    .expect("sumcheck should succeed");
}

#[test]
fn test_multilinear_constraint_detection() {
    // Test that unsatisfied constraints are properly detected
    let witness_size = 3;
    
    // Simple matrices for a + b = c constraint
    let m0 = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO], witness_size);  // selects a
    let m1 = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO], witness_size);  // selects b  
    let m2 = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE], witness_size);  // selects c
    let mats = vec![m0, m1, m2];

    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] + inputs[1] - inputs[2] // a + b - c = 0
            } else {
                ExtF::ZERO
            }
        },
        1,
    );

    let structure = CcsStructure::new(mats, f);

    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    // Valid witness: a=3, b=4, c=7 (3+4=7)
    let z_good_base = vec![F::from_u64(3), F::from_u64(4), F::from_u64(7)];
    let z_good: Vec<ExtF> = z_good_base.into_iter().map(from_base).collect();
    let witness_good = CcsWitness { z: z_good };
    assert!(check_satisfiability(&structure, &instance, &witness_good));

    // Invalid witness: a=3, b=4, c=8 (3+4≠8)
    let z_bad_base = vec![F::from_u64(3), F::from_u64(4), F::from_u64(8)];
    let z_bad: Vec<ExtF> = z_bad_base.into_iter().map(from_base).collect();
    let witness_bad = CcsWitness { z: z_bad };
    assert!(!check_satisfiability(&structure, &instance, &witness_bad));
}

