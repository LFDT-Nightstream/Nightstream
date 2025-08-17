use neo_ccs::{ccs_sumcheck_prover, mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_sumcheck::{ExtF, FnOracle, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn test_ccs_prover_no_copy_panic() {
    let structure = CcsStructure::new(vec![], mv_poly(|_| ExtF::ZERO, 2));
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness { z: vec![] };
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    );
    assert!(result.is_ok());
}

#[test]
fn test_ccs_multilinear_specialization() {
    let mat = RowMajorMatrix::new(vec![F::ZERO; 2], 2);
    let mats = vec![mat.clone(), mat];
    let structure = CcsStructure::new(mats, mv_poly(|ins| ins[0] + ins[1], 1));
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ONE, ExtF::ZERO],
    };
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    );
    assert!(result.is_ok());
    let (msgs, _) = result.unwrap();
    assert!(!msgs.is_empty());
}

#[test]
fn test_ccs_multilinear_with_norms() {
    let structure = CcsStructure::new(vec![], mv_poly(|_| ExtF::ZERO, 1));
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness { z: vec![ExtF::ONE] }; // Trigger norm check
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    );
    assert!(result.is_ok()); // Norms handled without error
}
