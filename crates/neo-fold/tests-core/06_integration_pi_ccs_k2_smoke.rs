use neo_fold::{pi_ccs_prove, pi_ccs_verify};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, MeInstance, SModuleHomomorphism};
use neo_math::{F, K, D};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use p3_field::PrimeCharacteristicRing;

/// Dummy S-module for testing
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

/// Create a tiny addition CCS structure for testing
fn create_test_ccs() -> neo_ccs::CcsStructure<F> {
    let rows: usize = 4;
    let cols: usize = 5;
    
    // Simple addition circuit: x1 + x2 = out
    // Constraint: (1·const + (-1)·x1 + (-1)·x2 + 0·_pad + 1·out) * 1 = 0
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    // Row 0: Main constraint
    a[0 * cols + 4] = F::ONE;        // out coefficient
    a[0 * cols + 1] = -F::ONE;       // -x1
    a[0 * cols + 2] = -F::ONE;       // -x2
    b[0 * cols + 0] = F::ONE;        // multiply by constant 1
    
    // Rows 1-3: Padding constraints (always satisfied)
    for row in 1..rows {
        b[row * cols + 0] = F::ONE;
    }
    
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

/// Helper to create MCS instance from witness values
#[allow(non_snake_case)]
fn create_mcs_from_witness(
    params: &NeoParams,
    z_full: Vec<F>,
    m_in: usize,
    l: &DummyS,
) -> (McsInstance<Commitment, F>, McsWitness<F>) {
    let d = D;
    let m = z_full.len();
    
    // Decompose z into digits
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    
    // Convert to row-major matrix Z (d×m)
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    
    // Create commitment
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit.clone(),
        x: z_full[..m_in].to_vec(),
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    (mcs_instance, mcs_witness)
}

/// Helper to create an ME instance from an MCS by performing k=1 fold
#[allow(non_snake_case)]
fn create_me_from_mcs(
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<F>,
    mcs_instance: McsInstance<Commitment, F>,
    mcs_witness: McsWitness<F>,
    l: &DummyS,
) -> (MeInstance<Commitment, F, K>, Mat<F>) {
    // Use pi_ccs_prove_simple (k=1) to get an ME instance
    let mut tr = Poseidon2Transcript::new(b"test/fixture/k1");
    let prove_result = neo_fold::pi_ccs_prove_simple(
        &mut tr,
        params,
        ccs,
        &[mcs_instance],
        &[mcs_witness.clone()],
        l,
    );
    
    assert!(prove_result.is_ok(), "k=1 fold for ME fixture generation failed: {:?}", prove_result.err());
    let (me_outputs, _proof) = prove_result.unwrap();
    
    assert_eq!(me_outputs.len(), 1, "k=1 should produce exactly 1 ME output");
    (me_outputs[0].clone(), mcs_witness.Z)
}

#[test]
#[ignore] // Remove #[ignore] once the Ajtai fixes are in place
fn pi_ccs_k2_honest_fold() {
    // Setup: params, CCS structure, dummy S-module
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = create_test_ccs();
    let l = DummyS;
    
    // Create first witness: 2 + 3 = 5
    let z1_full = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)];
    let m_in = 1;
    let (mcs_inst1, mcs_wit1) = create_mcs_from_witness(&params, z1_full, m_in, &l);
    
    // Create second witness: 7 + 11 = 18 → fold to ME first
    let z2_full = vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)];
    let (mcs_inst2, mcs_wit2) = create_mcs_from_witness(&params, z2_full, m_in, &l);
    let (me_input, me_witness_z) = create_me_from_mcs(&params, &ccs, mcs_inst2, mcs_wit2, &l);
    
    // Now fold k=2: 1 MCS + 1 ME → 2 ME
    let mut tr_p = Poseidon2Transcript::new(b"test/pi-ccs/k2");
    let prove_result = pi_ccs_prove(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs_inst1.clone()],
        &[mcs_wit1],
        &[me_input.clone()],
        &[me_witness_z],
        &l,
    );
    
    assert!(prove_result.is_ok(), "k=2 proving should succeed for valid witnesses");
    let (me_outputs, proof) = prove_result.unwrap();
    
    assert_eq!(me_outputs.len(), 2, "k=2 should produce exactly 2 ME outputs");
    
    // Verify
    let mut tr_v = Poseidon2Transcript::new(b"test/pi-ccs/k2");
    let verify_result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs_inst1],
        &[me_input],
        &me_outputs,
        &proof,
    );
    
    assert!(verify_result.is_ok(), "Verification should not error");
    let is_valid = verify_result.unwrap();
    assert!(is_valid, "Valid k=2 fold (1 MCS + 1 ME → 2 ME) should verify");
}

#[test]
#[ignore] // Remove once Ajtai fixes are in place
fn pi_ccs_k2_multi_step_ivc_simulation() {
    // Simulate a 2-step IVC: Step 0 → ME, Step 1 folds with Step 0's ME
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = create_test_ccs();
    let l = DummyS;
    let m_in = 1;
    
    // Step 0: Initial computation (1 + 1 = 2)
    let z_step0 = vec![F::ONE, F::ONE, F::ONE, F::ZERO, F::from_u64(2)];
    let (mcs0, wit0) = create_mcs_from_witness(&params, z_step0, m_in, &l);
    let (me_step0, z_step0_mat) = create_me_from_mcs(&params, &ccs, mcs0, wit0, &l);
    
    // Step 1: Next computation (5 + 7 = 12) folded with Step 0's ME
    let z_step1 = vec![F::ONE, F::from_u64(5), F::from_u64(7), F::ZERO, F::from_u64(12)];
    let (mcs1, wit1) = create_mcs_from_witness(&params, z_step1, m_in, &l);
    
    // Fold Step 1 with Step 0's ME (k=2)
    let mut tr_p = Poseidon2Transcript::new(b"test/ivc/step1");
    let prove_result = pi_ccs_prove(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs1.clone()],
        &[wit1],
        &[me_step0.clone()],
        &[z_step0_mat],
        &l,
    );
    
    assert!(prove_result.is_ok(), "IVC step 1 proving should succeed");
    let (me_outputs, proof) = prove_result.unwrap();
    assert_eq!(me_outputs.len(), 2, "Should produce 2 ME outputs");
    
    // Verify Step 1
    let mut tr_v = Poseidon2Transcript::new(b"test/ivc/step1");
    let verify_result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs1],
        &[me_step0],
        &me_outputs,
        &proof,
    );
    
    assert!(verify_result.is_ok(), "IVC step 1 verification should not error");
    let is_valid = verify_result.unwrap();
    assert!(is_valid, "IVC step 1 should verify (simulates folding with prior state)");
}

