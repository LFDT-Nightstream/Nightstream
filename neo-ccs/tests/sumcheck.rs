use neo_ccs::{ccs_sumcheck_prover, mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_sumcheck::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

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
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
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
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut transcript,
    );
    assert!(result.is_ok());
    let msgs = result.unwrap();
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
    let mut transcript = vec![];
    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut transcript,
    );
    assert!(result.is_ok()); // Norms handled without error
}

#[test]
fn test_ccs_zk_prover_hides() {
    let m = 2;
    let mat = RowMajorMatrix::new(vec![F::ZERO; m], 1);
    let mats = vec![mat];
    let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1);
    let structure = CcsStructure::new(mats, f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![from_base(F::ZERO)],
    };
    let mut prev: Option<neo_sumcheck::Polynomial<ExtF>> = None;
    let mut diff = false;
    let mut rng = rand::rng();
    for _ in 0..5 {
        let mut t = vec![];
        let prefix = rng.random::<u64>().to_be_bytes().to_vec();
        t.extend(prefix);
        let msgs =
            ccs_sumcheck_prover(&structure, &instance, &witness, 1, &mut t)
                .expect("sumcheck");
        if let Some(p) = &prev {
            if *p != msgs[0].0 {
                diff = true;
                break;
            }
        }
        prev = Some(msgs[0].0.clone());
    }
    assert!(diff);
}