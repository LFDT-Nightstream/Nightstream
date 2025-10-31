/// Unit tests for GenericCcsOracle to catch Ajtai phase bugs
/// 
/// These tests verify the oracle returns correct values during both
/// row rounds and Ajtai rounds. They FAIL when the bug is present.

use neo_fold::{pi_ccs_prove_simple};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness, SModuleHomomorphism};
use neo_math::{F, D};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use p3_field::PrimeCharacteristicRing;

struct DummyS;

impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let d = z.rows();
        Commitment::zeros(d, 4)
    }
    
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut result = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = z[(r, c)];
            }
        }
        result
    }
}

/// Create a minimal 3-matrix CCS for testing (n=4, m=5, t=3, ell=2)
/// Following paper convention: M_1 (matrices[0]) must be identity
fn create_test_ccs_3mat() -> CcsStructure<F> {
    let n: usize = 4;
    let m: usize = 5;
    let t: usize = 3;
    
    // M_1 = Identity matrix extended to n×m (as required by paper)
    let mut m0 = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        m0[(i, i)] = F::ONE;
    }
    
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1[(0, 1)] = F::ONE;
    
    let mut m2 = Mat::zero(n, m, F::ZERO);
    m2[(0, 2)] = F::ONE;
    
    // Constraint polynomial: f(identity, m1, m2) = identity + m1 - m2
    // This encodes: z[0] + z[1] - z[2] = 0 on first row
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 0, 0] },   // identity term
        Term { coeff: F::ONE, exps: vec![0, 1, 0] },   // m1 term
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // m2 term
    ];
    let f = SparsePoly::new(t, terms);
    
    CcsStructure {
        n,
        m,
        matrices: vec![m0, m1, m2],
        f,
    }
}

#[allow(non_snake_case)]
fn create_mcs_from_witness(
    ccs: &CcsStructure<F>,
    params: &NeoParams,
    z_full: Vec<F>,
    m_in: usize,
) -> (McsInstance<Commitment, F>, McsWitness<F>) {
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, D, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * ccs.m];
    for col in 0..ccs.m {
        for row in 0..D {
            row_major[row * ccs.m + col] = z_digits[col * D + row];
        }
    }
    let Z = Mat::from_row_major(D, ccs.m, row_major);
    
    let l = DummyS;
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit,
        x: z_full[..m_in].to_vec(),
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    (mcs_instance, mcs_witness)
}

#[test]
fn test_oracle_with_3_matrices_t3() {
    // Test with t=3 matrices: should work but currently fails due to Ajtai bug
    let ccs = create_test_ccs_3mat();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    // Valid witness: satisfies z[0] + z[1] - z[2] = 0
    // So if z[0] = 1, z[1] = 3, then z[2] = 4
    let z_full = vec![F::ONE, F::from_u64(3), F::from_u64(4), F::ZERO, F::ZERO];
    let (mcs_inst, mcs_wit) = create_mcs_from_witness(&ccs, &params, z_full, 1);
    
    let l = DummyS;
    let mut tr = Poseidon2Transcript::new(b"test/oracle/t3");
    
    let result = pi_ccs_prove_simple(
        &mut tr,
        &params,
        &ccs,
        &[mcs_inst],
        &[mcs_wit],
        &l,
    );
    
    // This SHOULD succeed for valid witness, but fails due to Ajtai oracle bug
    assert!(result.is_ok(), 
            "Proving with valid witness should succeed. Error: {:?}", 
            result.err());
}

#[test]
fn test_oracle_witness_1_plus_1_equals_2() {
    let ccs = create_test_ccs_3mat();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    // Valid witness: z[0] + z[1] - z[2] = 0, so 1 + 1 - 2 = 0
    let z_full = vec![F::ONE, F::ONE, F::from_u64(2), F::ZERO, F::ZERO];
    
    // Verify witness satisfies constraint
    {
        // Check constraint: z[0] + z[1] - z[2] = 0
        let constraint_val = z_full[0] + z_full[1] - z_full[2];
        eprintln!("Constraint check: {:?} + {:?} - {:?} = {:?}", 
            z_full[0],
            z_full[1], 
            z_full[2],
            constraint_val);
        assert_eq!(constraint_val, F::ZERO, "Witness must satisfy constraint");
    }
    
    let (mcs_inst, mcs_wit) = create_mcs_from_witness(&ccs, &params, z_full, 1);
    
    let l = DummyS;
    let mut tr = Poseidon2Transcript::new(b"test/oracle/1plus1");
    
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_ok(), 
            "Proving 1+1-2=0 should succeed. Error: {:?}", 
            result.err());
}

#[test]
fn test_oracle_witness_larger_numbers() {
    let ccs = create_test_ccs_3mat();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    // Valid witness: z[0] + z[1] - z[2] = 0, so 1 + 7 - 8 = 0
    let z_full = vec![F::ONE, F::from_u64(7), F::from_u64(8), F::ZERO, F::ZERO];
    let (mcs_inst, mcs_wit) = create_mcs_from_witness(&ccs, &params, z_full, 1);
    
    let l = DummyS;
    let mut tr = Poseidon2Transcript::new(b"test/oracle/large");
    
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_ok(), 
            "Proving 1+7-8=0 should succeed. Error: {:?}", 
            result.err());
}

#[test]
fn test_oracle_with_different_transcript_labels() {
    // Verify behavior is consistent across different transcript initializations
    let ccs = create_test_ccs_3mat();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let z_full = vec![F::ONE, F::from_u64(5), F::from_u64(6), F::ZERO, F::ZERO];
    let (mcs_inst, mcs_wit) = create_mcs_from_witness(&ccs, &params, z_full, 1);
    
    let l = DummyS;
    let mut tr = Poseidon2Transcript::new(b"different/label");
    
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_ok(), 
            "Proving should succeed regardless of transcript label. Error: {:?}", 
            result.err());
}

#[test]
fn test_oracle_zero_values() {
    let ccs = create_test_ccs_3mat();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    // Valid witness with zeros: z[0] + z[1] - z[2] = 0, so 1 + 0 - 1 = 0
    let z_full = vec![F::ONE, F::ZERO, F::ONE, F::ZERO, F::ZERO];
    let (mcs_inst, mcs_wit) = create_mcs_from_witness(&ccs, &params, z_full, 1);
    
    let l = DummyS;
    let mut tr = Poseidon2Transcript::new(b"test/oracle/zeros");
    
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_ok(), 
            "Proving 1+0-1=0 should succeed. Error: {:?}", 
            result.err());
}
