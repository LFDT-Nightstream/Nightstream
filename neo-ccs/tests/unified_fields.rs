use neo_ccs::{
    ccs_sumcheck_prover, check_satisfiability, mv_poly, CcsInstance, CcsStructure, CcsWitness,
};
use neo_fields::{embed_base_to_ext, project_ext_to_base, ExtF, F};
use neo_sumcheck::FnOracle;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn test_ccs_witness_ext_roundtrip() {
    let f_base = F::from_u64(42);
    let e = embed_base_to_ext(f_base);
    assert_eq!(project_ext_to_base(e), Some(f_base));

    let e_complex = ExtF::new_complex(f_base, F::ONE);
    assert_eq!(project_ext_to_base(e_complex), None);
}

#[test]
fn test_ccs_sumcheck_no_conversion_needed() {
    let mat = RowMajorMatrix::new(vec![F::ONE], 1);
    let f = mv_poly(|ins: &[ExtF]| ins[0], 1);
    let structure = CcsStructure::new(vec![mat], f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![embed_base_to_ext(F::ONE)],
    };
    let mut transcript = vec![];
    let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
    let (_msgs, comms) = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
        1,
        &mut oracle,
        &mut transcript,
    )
    .expect("sumcheck");
    assert!(comms.is_empty());
}

#[test]
fn test_ccs_structure_lift_no_loss() {
    let base_mat = RowMajorMatrix::new(vec![F::ONE, F::from_u64(2)], 1);
    let f = mv_poly(|ins: &[ExtF]| ins[0], 1);
    let structure = CcsStructure::new(vec![base_mat.clone()], f);
    let lifted_data = &structure.mats[0].values;
    for (&lifted, &base) in lifted_data.iter().zip(&base_mat.values) {
        assert_eq!(project_ext_to_base(lifted), Some(base));
    }
}

#[test]
fn test_sumcheck_prover_lifted_evals() {
    let mat = RowMajorMatrix::new(vec![F::ONE], 1);
    let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1);
    let structure = CcsStructure::new(vec![mat], f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![embed_base_to_ext(F::from_u64(42))],
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
    for (uni, _) in &msgs {
        for &c in uni.coeffs() {
            assert!(project_ext_to_base(c).is_some());
        }
    }
}

#[test]
fn test_check_satisfiability_with_complex_fails() {
    let mat = RowMajorMatrix::new(vec![F::ONE], 1);
    let f = mv_poly(|ins: &[ExtF]| ins[0], 1);
    let structure = CcsStructure::new(vec![mat], f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE,
    };
    let bad_z = vec![ExtF::new_complex(F::ONE, F::ONE)];
    let witness = CcsWitness { z: bad_z };
    assert!(!check_satisfiability(&structure, &instance, &witness));
}
