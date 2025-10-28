use neo_fold::{pi_ccs_prove_simple, pi_ccs_verify_simple};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, SModuleHomomorphism};
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

#[test]
#[allow(non_snake_case)]
fn pi_ccs_k1_simple_valid_zero_state() {
    let rows: usize = 4;
    let cols: usize = 4;
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    a[0 * cols + 3] = F::ONE;
    a[0 * cols + 1] = -F::ONE;
    a[0 * cols + 2] = -F::ONE;
    b[0 * cols + 0] = F::ONE;
    
    for row in 1..rows {
        b[row * cols + 0] = F::ONE;
    }
    
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    );
    
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let d = D;
    
    let z_full = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO];
    let m = z_full.len();
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            row_major[row * m + col] = z_digits[col * d + row]; 
        } 
    }
    let Z = Mat::from_row_major(d, m, row_major);
    
    let l = DummyS;
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit.clone(),
        x: vec![z_full[0]],
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    let mut tr_p = Poseidon2Transcript::new(b"test/pi-ccs/k1");
    let prove_result = pi_ccs_prove_simple(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs_instance.clone()],
        &[mcs_witness],
        &l,
    );
    
    assert!(prove_result.is_ok(), "Proving should succeed for valid witness");
    let (me_outputs, proof) = prove_result.unwrap();
    
    let mut tr_v = Poseidon2Transcript::new(b"test/pi-ccs/k1");
    let verify_result = pi_ccs_verify_simple(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs_instance],
        &me_outputs,
        &proof,
    );
    
    assert!(verify_result.is_ok(), "Verification should not error");
    let is_valid = verify_result.unwrap();
    assert!(is_valid, "Valid zero-state proof (const1=1, 0+0=0) should verify");
}

#[test]
// Ajtai oracle fix applied - test should now pass
#[allow(non_snake_case)]
fn pi_ccs_k1_simple_addition_circuit() {
    // Create a CCS with t=3 matrices and n=4 constraints (ell=2)
    // This tests: x1 + x2 = out with 3 distinct matrices
    let n: usize = 4;  // 4 constraints → ell = log2(4) = 2
    let m: usize = 5;  // witness length: [const, x1, x2, intermediate, out]
    let t: usize = 3;  // 3 CCS matrices
    
    // Matrix M0: Selects x1
    let mut m0 = Mat::zero(n, m, F::ZERO);
    m0[(0, 1)] = F::ONE;  // row 0: select x1
    
    // Matrix M1: Selects x2  
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1[(0, 2)] = F::ONE;  // row 0: select x2
    
    // Matrix M2: Selects output
    let mut m2 = Mat::zero(n, m, F::ZERO);
    m2[(0, 4)] = F::ONE;  // row 0: select out
    
    // Do not add padding constraints that inject a constant on other rows.
    // Under full Option B, F' is evaluated from the multilinear extension over rows;
    // setting constants here would make f=a+b-c nonzero on padding rows.
    // We keep other rows neutral (all zeros) so only row 0 contributes.
    
    // CCS polynomial: f(a,b,c) = a + b - c
    // This encodes: M0·z + M1·z - M2·z = 0 → x1 + x2 - out = 0
    use neo_ccs::{CcsStructure, SparsePoly, Term};
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 0, 0] },   // +a (M0)
        Term { coeff: F::ONE, exps: vec![0, 1, 0] },   // +b (M1)
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -c (M2)
    ];
    let f = SparsePoly::new(t, terms);
    
    let ccs = CcsStructure {
        n,
        m,
        matrices: vec![m0, m1, m2],
        f,
    };
    
    // Verify we have the right structure
    assert_eq!(ccs.t(), 3, "Should have exactly 3 CCS matrices");
    assert_eq!(ccs.n, 4, "Should have 4 constraints");
    let ell = ccs.n.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(ell, 2, "Should have ell=2");
    
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let d = D;
    
    let in1 = F::from_u64(3);
    let in2 = F::from_u64(5);
    let out = in1 + in2;
    
    // Witness: [const=1, x1=3, x2=5, intermediate=0, out=8]
    let z_full = vec![F::ONE, in1, in2, F::ZERO, out];
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            row_major[row * m + col] = z_digits[col * d + row]; 
        } 
    }
    let Z = Mat::from_row_major(d, m, row_major);
    
    let l = DummyS;
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit.clone(),
        x: vec![z_full[0]],
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    let mut tr_p = Poseidon2Transcript::new(b"test/pi-ccs/add");
    let prove_result = pi_ccs_prove_simple(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs_instance.clone()],
        &[mcs_witness],
        &l,
    );
    
    assert!(prove_result.is_ok(), "Proving should succeed for valid addition witness: {:?}", prove_result.err());
    let (me_outputs, proof) = prove_result.unwrap();
    
    let mut tr_v = Poseidon2Transcript::new(b"test/pi-ccs/add");
    let verify_result = pi_ccs_verify_simple(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs_instance],
        &me_outputs,
        &proof,
    );
    
    assert!(verify_result.is_ok(), "Verification should not error");
    let is_valid = verify_result.unwrap();
    assert!(is_valid, "Valid addition circuit (3+5=8) with t=3 matrices, ell=2 should verify");
}

#[test]
#[allow(non_snake_case)]
fn pi_ccs_k1_detects_invalid_witness() {
    let rows: usize = 4;
    let cols: usize = 4;
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    a[0 * cols + 3] = F::ONE;
    a[0 * cols + 1] = -F::ONE;
    a[0 * cols + 2] = -F::ONE;
    b[0 * cols + 0] = F::ONE;
    
    for row in 1..rows {
        b[row * cols + 0] = F::ONE;
    }
    
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    );
    
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let d = D;
    
    let z_full = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(99)];
    let m = z_full.len();
    let m_in = 1;
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            row_major[row * m + col] = z_digits[col * d + row]; 
        } 
    }
    let Z = Mat::from_row_major(d, m, row_major);
    
    let l = DummyS;
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit.clone(),
        x: vec![z_full[0]],
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    let mut tr_p = Poseidon2Transcript::new(b"test/pi-ccs/invalid");
    let prove_result = pi_ccs_prove_simple(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs_instance.clone()],
        &[mcs_witness],
        &l,
    );
    
    if let Ok((me_outputs, proof)) = prove_result {
        let mut tr_v = Poseidon2Transcript::new(b"test/pi-ccs/invalid");
        let verify_result = pi_ccs_verify_simple(
            &mut tr_v,
            &params,
            &ccs,
            &[mcs_instance],
            &me_outputs,
            &proof,
        );
        
        if let Ok(is_valid) = verify_result {
            assert!(!is_valid, "Invalid witness (2+3≠99) should not verify");
        }
    }
}
