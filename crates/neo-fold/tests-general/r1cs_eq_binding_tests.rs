use neo_fold::pi_ccs::{pi_ccs_prove, pi_ccs_verify};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness};
use neo_ajtai::{AjtaiSModule, setup as ajtai_setup, set_global_pp, decomp_b, DecompStyle};
use neo_ccs::SModuleHomomorphism; // for AjtaiSModule::commit
use neo_params::NeoParams;
use neo_math::{F, K, ring::D};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;

fn setup_ajtai() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 8, 2).expect("Ajtai setup");
    let _ = set_global_pp(pp);
}

// Build an R1CS-shaped CCS where f = 7·(X2·X0 − X1) with 3 rows (ℓ=2)
fn r1cs_scaled_permuted() -> CcsStructure<F> {
    // 3 rows to ensure ℓ=2 after padding to 4
    // Row 0-1: identity structure (as before)
    // Row 2: zero row for padding (0 * 0 = 0, always satisfied)
    let a = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,   // Row 0 (identity pattern)
        F::ZERO, F::ONE,   // Row 1 (identity pattern)
        F::ZERO, F::ZERO,  // Row 2 (all zeros)
    ]);
    let b = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,   // Row 0 (identity pattern)
        F::ZERO, F::ONE,   // Row 1 (identity pattern)
        F::ZERO, F::ZERO,  // Row 2 (all zeros)
    ]);
    let c = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,   // Row 0 (identity pattern)
        F::ZERO, F::ONE,   // Row 1 (identity pattern)
        F::ZERO, F::ZERO,  // Row 2 (all zeros)
    ]);
    let lambda = F::from_u64(7);
    // Terms: λ·X2·X0 + (−λ)·X1
    let terms = vec![
        Term { coeff: lambda,   exps: vec![1, 0, 1] }, // X0^1 * X1^0 * X2^1
        Term { coeff: -lambda,  exps: vec![0, 1, 0] }, // X1^1
    ];
    let f = SparsePoly::new(3, terms);
    CcsStructure::new(vec![a, b, c], f).unwrap()
}

#[test]
fn r1cs_normalization_detects_and_verifies() {
    setup_ajtai();
    let params = NeoParams::goldilocks_small_circuits();
    let s = r1cs_scaled_permuted();

    // Simple instance/witness: z = [1, 0] with consistent Ajtai digits Z = decomp_b(z)
    let x = vec![F::ONE];
    let w = vec![F::ZERO];
    let mut z_full = x.clone();
    z_full.extend_from_slice(&w);
    let digits = decomp_b(&z_full, 2, D, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * z_full.len()];
    for col in 0..z_full.len() {
        for row in 0..D {
            row_major[row * z_full.len() + col] = digits[col * D + row];
        }
    }
    let z_mat = Mat::from_row_major(D, z_full.len(), row_major);
    let l = AjtaiSModule::from_global().expect("PP");
    let c = l.commit(&z_mat);
    let inst = McsInstance { c, x, m_in: 1 };
    let wit  = McsWitness { w, Z: z_mat };

    // Prove and verify Pi-CCS
    let l = AjtaiSModule::from_global().expect("PP");
    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    let (outs, proof) = pi_ccs_prove(&mut tr, &params, &s, &[inst.clone()], &[wit], &[], &[], &l).expect("prove");

    // Fresh transcript for verify
    let mut tr_v = Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_ccs_verify(&mut tr_v, &params, &s, &[inst], &[], &outs, &proof).expect("verify call");
    assert!(ok, "eq-binding path should accept scaled/permuted R1CS");
}

#[test]
fn r1cs_terminal_fails_on_y_scalar_tamper() {
    setup_ajtai();
    let params = NeoParams::goldilocks_small_circuits();
    let s = r1cs_scaled_permuted();
    let x = vec![F::ONE];
    let w = vec![F::ZERO];
    let mut z_full = x.clone();
    z_full.extend_from_slice(&w);
    let digits = decomp_b(&z_full, 2, D, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * z_full.len()];
    for col in 0..z_full.len() {
        for row in 0..D {
            row_major[row * z_full.len() + col] = digits[col * D + row];
        }
    }
    let z_mat = Mat::from_row_major(D, z_full.len(), row_major);
    let l = AjtaiSModule::from_global().expect("PP");
    let c = l.commit(&z_mat);
    let inst = McsInstance { c, x, m_in: 1 };
    let wit  = McsWitness { w, Z: z_mat };

    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    let (mut outs, proof) = pi_ccs_prove(&mut tr, &params, &s, &[inst.clone()], &[wit], &[], &[], &l).expect("prove");

    // Tamper terminal scalars: this breaks running_sum == eq*Σ α (A·B − C)
    if let Some(first) = outs.first_mut() {
        if !first.y_scalars.is_empty() {
            first.y_scalars[0] += K::ONE; // flip one scalar by +1
        }
    }

    let mut tr_v = Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_ccs_verify(&mut tr_v, &params, &s, &[inst], &[], &outs, &proof).expect("verify");
    assert!(!ok, "eq-binding terminal must reject tampered y_scalars");
}
