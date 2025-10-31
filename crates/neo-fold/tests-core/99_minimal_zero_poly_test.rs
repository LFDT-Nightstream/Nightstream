#![allow(non_snake_case)]

use neo_fold::{pi_ccs_prove_simple};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, SModuleHomomorphism};
use neo_math::{F, D};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use p3_field::PrimeCharacteristicRing;

struct DummyS;
impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        Commitment::zeros(z.rows(), 4)
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

#[test]
fn test_minimal_zero_poly_k1() {
    // Simplest possible test: 1x1 R1CS with 0=0 constraint
    let n = 1;
    let m = 1;
    
    let a = vec![F::ZERO];
    let b = vec![F::ONE];
    let c = vec![F::ZERO];
    
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(n, m, a),
        Mat::from_row_major(n, m, b),
        Mat::from_row_major(n, m, c),
    );
    
    assert_eq!(ccs.n, 1);
    assert_eq!(ccs.m, 1);
    // r1cs_to_ccs adds identity when n = m
    assert_eq!(ccs.t(), 4); 
    
    let params = NeoParams::goldilocks_for_circuit(1, 2, 2);
    let z_full = vec![F::ONE];
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, D, neo_ajtai::DecompStyle::Balanced);
    let Z = Mat::from_row_major(D, z_full.len(), z_digits);
    
    let l = DummyS;
    let mcs_inst = McsInstance {
        c: l.commit(&Z),
        x: vec![z_full[0]],
        m_in,
    };
    
    let mcs_wit = McsWitness {
        w: vec![],
        Z,
    };
    
    let mut tr = Poseidon2Transcript::new(b"test/minimal");
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_err(), "Expected failure for zero polynomial case");
    println!("Error: {:?}", result.err());
}

#[test] 
fn test_minimal_zero_poly_square() {
    // Try with square matrices to get identity matrix
    let n = 2;
    let m = 2;
    
    // Constraint: 0 = 0 on both rows
    let a = vec![F::ZERO; n * m];
    let mut b = vec![F::ZERO; n * m];
    b[0] = F::ONE; // row 0: const = 1
    b[2] = F::ONE; // row 1: const = 1
    let c = vec![F::ZERO; n * m];
    
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(n, m, a),
        Mat::from_row_major(n, m, b),
        Mat::from_row_major(n, m, c),
    );
    
    assert_eq!(ccs.n, 2);
    assert_eq!(ccs.m, 2);
    assert_eq!(ccs.t(), 4); // Should have identity added
    
    let params = NeoParams::goldilocks_for_circuit(1, 2, 2);
    let z_full = vec![F::ONE, F::ZERO]; // [const=1, 0]
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, D, neo_ajtai::DecompStyle::Balanced);
    let Z = Mat::from_row_major(D, z_full.len(), z_digits);
    
    let l = DummyS;
    let mcs_inst = McsInstance {
        c: l.commit(&Z),
        x: vec![z_full[0]],
        m_in,
    };
    
    let mcs_wit = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    let mut tr = Poseidon2Transcript::new(b"test/minimal2");
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    assert!(result.is_err(), "Expected failure for zero polynomial case");
    println!("Error: {:?}", result.err());
}

#[test]
fn test_minimal_nonzero_poly() {
    // Test with non-zero constraint to see if it passes
    let n = 2;
    let m = 2;
    
    // Constraint: row 0: x1 * 1 = 0 (satisfied when x1=0)
    //            row 1: 1 * 1 = 1 (always satisfied)
    let mut a = vec![F::ZERO; n * m];
    a[1] = F::ONE; // row 0: select x1
    a[2] = F::ONE; // row 1: select const
    
    let mut b = vec![F::ZERO; n * m];
    b[0] = F::ONE; // row 0: const = 1
    b[2] = F::ONE; // row 1: const = 1
    
    let mut c = vec![F::ZERO; n * m];
    c[2] = F::ONE; // row 1: result = 1
    
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(n, m, a),
        Mat::from_row_major(n, m, b),
        Mat::from_row_major(n, m, c),
    );
    
    let params = NeoParams::goldilocks_for_circuit(2, 2, 2);
    let z_full = vec![F::ONE, F::ZERO]; // [const=1, x1=0]
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, D, neo_ajtai::DecompStyle::Balanced);
    let Z = Mat::from_row_major(D, z_full.len(), z_digits);
    
    let l = DummyS;
    let mcs_inst = McsInstance {
        c: l.commit(&Z),
        x: vec![z_full[0]], 
        m_in,
    };
    
    let mcs_wit = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    let mut tr = Poseidon2Transcript::new(b"test/nonzero");
    let result = pi_ccs_prove_simple(&mut tr, &params, &ccs, &[mcs_inst], &[mcs_wit], &l);
    
    // This should pass because F is not identically zero
    match result {
        Ok(_) => println!("✓ Non-zero polynomial test passed"),
        Err(e) => println!("✗ Non-zero polynomial test failed: {:?}", e),
    }
}
