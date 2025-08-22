use neo_ccs::*;
use neo_fields::{from_base, ExtF, F};
use neo_sumcheck::fiat_shamir::fiat_shamir_challenge;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
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
    let msgs = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
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
fn test_r1cs_sumcheck_valid() {
    let witness_size = 5;

    let m0_data = vec![
        F::ZERO,
        F::from_u64(F::ORDER_U64 - 1),
        F::from_u64(F::ORDER_U64 - 1),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
    ];
    let m0 = RowMajorMatrix::new(m0_data, witness_size);

    let m1_data = vec![
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
        F::ZERO,
    ];
    let m1 = RowMajorMatrix::new(m1_data, witness_size);

    let m2_data = vec![
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
    ];
    let m2 = RowMajorMatrix::new(m2_data, witness_size);

    let mats = vec![m0, m1, m2];

    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] * inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );

    let structure = CcsStructure::new(mats, f);

    let z_base = vec![
        F::ONE,
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(15),
    ];
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
        16,
        &mut transcript,
    )
    .expect("sumcheck");

    let mut tmp = vec![];
    tmp.extend(b"norm_alpha");
    let alpha = fiat_shamir_challenge(&tmp);
    let mut sum = ExtF::ZERO;
    for (i, &w_i) in witness.z.iter().enumerate() {
        let mut prod = w_i;
        for k in 1..=16 {
            let kf = from_base(F::from_u64(k));
            prod *= w_i * w_i - kf * kf;
        }
        let mut alpha_i = ExtF::ONE;
        for _ in 0..i {
            alpha_i *= alpha;
        }
        sum += alpha_i * prod;
    }
    assert_eq!(sum, ExtF::ZERO);
}

#[test]
fn test_r1cs_sumcheck_invalid_norm() {
    let witness_size = 5;

    let m0_data = vec![
        F::ZERO,
        F::from_u64(F::ORDER_U64 - 1),
        F::from_u64(F::ORDER_U64 - 1),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
    ];
    let m0 = RowMajorMatrix::new(m0_data, witness_size);

    let m1_data = vec![
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
        F::ZERO,
    ];
    let m1 = RowMajorMatrix::new(m1_data, witness_size);

    let m2_data = vec![
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
    ];
    let m2 = RowMajorMatrix::new(m2_data, witness_size);

    let mats = vec![m0, m1, m2];

    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] * inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );

    let structure = CcsStructure::new(mats, f);

    let z_base = vec![
        F::ONE,
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(17),
    ];
    let z: Vec<ExtF> = z_base.into_iter().map(from_base).collect();
    let witness = CcsWitness { z };

    let mut transcript = vec![];
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let res = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        16,
        &mut transcript,
    );
    assert!(res.is_err());
}

#[test]
fn test_r1cs_sumcheck_invalid() {
    let witness_size = 5;

    let m0_data = vec![
        F::ZERO,
        F::from_u64(F::ORDER_U64 - 1),
        F::from_u64(F::ORDER_U64 - 1),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
    ];
    let m0 = RowMajorMatrix::new(m0_data, witness_size);

    let m1_data = vec![
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
        F::ZERO,
    ];
    let m1 = RowMajorMatrix::new(m1_data, witness_size);

    let m2_data = vec![
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ONE,
    ];
    let m2 = RowMajorMatrix::new(m2_data, witness_size);

    let mats = vec![m0, m1, m2];

    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] * inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );

    let structure = CcsStructure::new(mats, f);

    let z_bad_base = vec![
        F::ONE,
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(16),
    ];
    let z_bad: Vec<ExtF> = z_bad_base.into_iter().map(from_base).collect();
    let witness_bad = CcsWitness { z: z_bad };

    let mut transcript = vec![];
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let res = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness_bad,
        0,
        &mut transcript,
    );
    assert!(res.is_err());
}
