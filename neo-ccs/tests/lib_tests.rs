use neo_ccs::*;
use neo_fields::{from_base, ExtF, F};
use neo_sumcheck::{fiat_shamir::fiat_shamir_challenge, FnOracle};
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

    // f = X0 * X1 - X2
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() != 3 {
                ExtF::ZERO
            } else {
                inputs[0] * inputs[1] - inputs[2]
            }
        },
        2,
    );

    let structure = CcsStructure::new(mats, f);

    // Valid witness: z = [1, 2, 2] for both rows (1*2 -2 =0, 1*2 -2=0)
    let witness = CcsWitness {
        z: vec![
            from_base(F::ONE),
            from_base(F::from_u64(2)),
            from_base(F::from_u64(2)),
        ],
    };

    let instance = CcsInstance {
        commitment: vec![], // Stub, not checked here
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    assert!(check_satisfiability(&structure, &instance, &witness));

    // Invalid: change to [1,2,3]
    let bad_witness = CcsWitness {
        z: vec![
            from_base(F::ONE),
            from_base(F::from_u64(2)),
            from_base(F::from_u64(3)),
        ],
    };
    assert!(!check_satisfiability(&structure, &instance, &bad_witness));
}

#[test]
fn test_high_deg_f() {
    let m = 1;
    let mat = RowMajorMatrix::new(vec![F::ONE], m);
    let mats = vec![mat];
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 1 {
                inputs[0] * inputs[0] * inputs[0]
            } else {
                ExtF::ZERO
            }
        },
        3,
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
    let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
    let (msgs, _) = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    )
    .expect("sumcheck");
    let mut current = from_base(F::from_u64(8));
    for (uni, _) in &msgs {
        current = uni.eval(fiat_shamir_challenge(&mut vec![]));
    }
    assert_eq!(current, from_base(F::from_u64(8)));
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
    let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
    let (msgs, _) = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    )
    .expect("sumcheck");
    let mut current = from_base(F::ONE);
    for (uni, _) in &msgs {
        current = uni.eval(fiat_shamir_challenge(&mut vec![]));
    }
    assert_eq!(current, from_base(F::ONE));
}
