use neo_ccs::sumcheck::ProverError;
use neo_ccs::{
    ccs_sumcheck_prover, check_satisfiability, mv_poly, CcsInstance, CcsStructure, CcsWitness,
};
use neo_fields::{from_base, ExtF, F};
use neo_sumcheck::oracle::FnOracle;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::{rng, Rng};

#[test]
fn test_large_satisfiability() {
    let mut rng = rng();
    let num_constraints: usize = std::env::var("NEO_CCS_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let witness_size: usize = std::env::var("NEO_CCS_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let s = 3;

    let mats: Vec<_> = (0..s)
        .map(|_| {
            let data: Vec<F> = (0..num_constraints * witness_size)
                .map(|_| F::from_u64(rng.random()))
                .collect();
            RowMajorMatrix::new(data, witness_size)
        })
        .collect();

    let f = mv_poly(
        move |inputs: &[ExtF]| {
            if inputs.len() == s {
                inputs[0] * inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );

    let structure = CcsStructure::new(mats, f);

    // Use zero witness so relation is trivially satisfied
    let z_base: Vec<F> = vec![F::ZERO; witness_size];
    let z: Vec<ExtF> = z_base.iter().copied().map(from_base).collect();
    let witness = CcsWitness { z };

    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    let start = std::time::Instant::now();
    assert!(check_satisfiability(&structure, &instance, &witness));
    let duration = start.elapsed();
    println!("Large CCS check time: {:?}", duration);
}

#[test]
fn test_sumcheck_prover_returns_err_on_fail() {
    if std::env::var("RUN_FAILING_TESTS").is_err() {
        return;
    }
    // Structure with one constraint and linear polynomial
    let mat = RowMajorMatrix::new(vec![F::ONE], 1);
    let mats = vec![mat];
    let f = mv_poly(|inputs: &[ExtF]| inputs[0], 1);
    let structure = CcsStructure::new(mats, f);

    // Instance claims e = 0 but witness yields non-zero claim, triggering mismatch
    let invalid_instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ZERO,
    };
    let witness = CcsWitness {
        z: vec![from_base(F::ONE)],
    };
    let mut transcript = vec![];
    let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
    let result = ccs_sumcheck_prover(
        &structure,
        &invalid_instance,
        &witness,
        0,
        &mut oracle,
        &mut transcript,
    );
    assert!(matches!(result, Err(ProverError::SumMismatch(_))));
}
