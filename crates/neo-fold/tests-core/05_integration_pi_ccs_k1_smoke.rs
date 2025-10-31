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
// Uses r1cs_to_ccs which properly handles identity matrix
#[allow(non_snake_case)]
fn pi_ccs_k1_simple_addition_circuit() {
    // Create a simple addition circuit: x1 + x2 = out
    // But make it INVALID on purpose to get non-zero F
    let n: usize = 4;  // 4 constraints
    let m: usize = 4;  // 4 variables: [const=1, x1, x2, out]
    
    // R1CS matrices for addition constraint: A*z ∘ B*z = C*z
    // Normal: out = x1 + x2
    // We'll set: A = out, B = 1, C = x1 + x2
    
    // Matrix A: selects out
    let mut a = vec![F::ZERO; n * m];
    a[0 * m + 3] = F::ONE;   // out coefficient
    
    // Matrix B: selects constant 1
    let mut b = vec![F::ZERO; n * m];
    b[0 * m + 0] = F::ONE;   // const coefficient
    
    // Matrix C: selects x1 + x2
    let mut c = vec![F::ZERO; n * m];
    c[0 * m + 1] = F::ONE;   // x1 coefficient
    c[0 * m + 2] = F::ONE;   // x2 coefficient
    
    // Add padding constraints on other rows
    for row in 1..n {
        a[row * m + 0] = F::ONE;  // const on A
        b[row * m + 0] = F::ONE;  // const on B
        c[row * m + 0] = F::ONE;  // const on C (makes 1*1=1, satisfied)
    }
    
    // Convert R1CS to CCS (automatically adds identity matrix when n=m)
    let ccs = r1cs_to_ccs(
        Mat::from_row_major(n, m, a),
        Mat::from_row_major(n, m, b),
        Mat::from_row_major(n, m, c),
    );
    
    // Verify we have the right structure
    assert_eq!(ccs.t(), 4, "r1cs_to_ccs should create 4 matrices (I, A, B, C)");
    assert_eq!(ccs.n, 4, "Should have 4 constraints");
    let ell = ccs.n.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(ell, 2, "Should have ell=2");
    
    println!("CCS polynomial: {:?}", ccs.f);
    println!("CCS has {} matrices", ccs.matrices.len());
    
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let d = D;
    
    let in1 = F::from_u64(3);
    let in2 = F::from_u64(5);
    let out = in1 + in2;
    
    // Witness: [const=1, x1=3, x2=5, out=8]
    let z_full = vec![F::ONE, in1, in2, out];
    let m_in = 1;
    
    // Verify constraint is satisfied
    println!("Checking constraint satisfaction...");
    println!("z_full = [{}, {}, {}, {}]", z_full[0], z_full[1], z_full[2], z_full[3]);
    for row in 0..ccs.n {
        let mut vals = Vec::new();
        for mat in &ccs.matrices {
            let mut val = F::ZERO;
            for col in 0..ccs.m {
                val += mat[(row, col)] * z_full[col];
            }
            vals.push(val);
        }
        let f_val = ccs.f.eval(&vals);
        println!("Row {}: f({:?}) = {:?}", row, vals, f_val);
        if f_val != F::ZERO {
            panic!("Constraint not satisfied at row {}", row);
        }
    }
    println!("✓ All constraints satisfied");
    
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
    println!("ME outputs count: {}", me_outputs.len());
    
    let verify_result = pi_ccs_verify_simple(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs_instance],
        &me_outputs,
        &proof,
    );
    
    match verify_result {
        Ok(true) => println!("✓ Verification passed!"),
        Ok(false) => panic!("Verification returned false"),
        Err(e) => panic!("Verification errored: {:?}", e),
    }
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
        
        match verify_result {
            Ok(true) => panic!("Invalid witness (2+3≠99) verified as valid!"),
            Ok(false) => {} // This is what we expect, but we changed Ok(false) to Err
            Err(e) => eprintln!("✓ Invalid witness rejected with error: {:?}", e),
        }
    }
}
