//! Π_DEC red-team: recomposition and consistency failures.
//! 
//! Targets: 
//! - #2 (bad digit commitments) - Commitment recomposition without per‑digit openings
//! - #11 (wrong r on a digit) - Wrong `r` on digits  
//! - #8 (base mismatch at verify) - Base mismatch at verify

#![cfg(feature = "redteam")]
#![allow(non_snake_case)] // Allow mathematical notation like X_parent

use neo_fold::{fold_ccs_instances, pi_dec::pi_dec_verify, transcript::FoldTranscript};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness, MeInstance};
use neo_params::NeoParams;
use neo_math::{F, K, ring::D, Rq, cf_inv};
use p3_field::PrimeCharacteristicRing;
use neo_ajtai::{setup as ajtai_setup, set_global_pp, commit as ajtai_commit, s_lincomb};
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

fn dummy_ccs() -> CcsStructure<F> {
    let matrices = vec![Mat::zero(4, 3, F::ZERO)];
    let terms = vec![Term { coeff: F::ONE, exps: vec![1] }];
    let f = SparsePoly::new(1, terms);
    CcsStructure::new(matrices, f).expect("valid CCS")
}

fn ensure_global_ajtai_pp(m: usize) {
    if neo_ajtai::get_global_pp().is_ok() { return; }
    let mut rng = StdRng::from_seed([9u8; 32]);
    let pp = ajtai_setup(&mut rng, D, 16, m).expect("ajtai setup");
    let _ = set_global_pp(pp);
}

/// Recombine digits -> parent ME(B,L) exactly like the verifier does.
fn recombine_parent(params: &NeoParams, digits: &[MeInstance<neo_ajtai::Commitment, F, K>]) 
    -> MeInstance<neo_ajtai::Commitment, F, K>
{
    use neo_ccs::Mat as M;

    assert!(!digits.is_empty());
    let m_in = digits[0].m_in;
    let r_ref = digits[0].r.clone();
    let rows = digits[0].X.rows();
    let cols = digits[0].X.cols();
    let t = digits[0].y.len();

    // c_parent = Σ b^i · c_i (S-linear combination)
    let mut coeffs: Vec<Rq> = Vec::with_capacity(digits.len());
    let mut pow_f = F::ONE;
    for _ in 0..digits.len() {
        let mut coeff = [F::ZERO; D];
        coeff[0] = pow_f;
        coeffs.push(cf_inv(coeff));
        pow_f *= F::from_u64(params.b as u64);
    }
    let cs: Vec<_> = digits.iter().map(|d| d.c.clone()).collect();
    let c_parent = s_lincomb(&coeffs, &cs).expect("s_lincomb");

    // X_parent = Σ b^i * X_i
    let mut X_parent = M::zero(rows, cols, F::ZERO);
    let mut pow = F::ONE;
    for d in digits {
        for r in 0..rows {
            for c in 0..cols {
                X_parent[(r,c)] += d.X[(r,c)] * pow;
            }
        }
        pow *= F::from_u64(params.b as u64);
    }

    // y_parent[j][u] = Σ b^i * y_{i,j}[u]
    let y_dim = digits[0].y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_parent = vec![vec![K::ZERO; y_dim]; t];
    let mut pow_k = K::from(F::ONE);
    let base_k = K::from(F::from_u64(params.b as u64));
    for d in digits {
        for j in 0..t {
            for u in 0..y_dim { 
                y_parent[j][u] += d.y[j][u] * pow_k; 
            }
        }
        pow_k *= base_k;
    }

    // y_scalars_parent[j] = Σ b^i * Y_{j,i}(r)
    let mut y_scalars_parent = vec![K::ZERO; digits[0].y_scalars.len()];
    let mut pk = K::from(F::ONE);
    for d in digits {
        for j in 0..y_scalars_parent.len() { 
            y_scalars_parent[j] += d.y_scalars[j] * pk; 
        }
        pk *= base_k;
    }

    MeInstance {
        c_step_coords: vec![], // TODO: Pattern A - populate with actual pre-commit coords
        u_offset: 0,
        u_len: 0,
        c: c_parent, 
        X: X_parent, 
        r: r_ref,
        y: y_parent, 
        y_scalars: y_scalars_parent,
        m_in, 
        fold_digest: digits[0].fold_digest,
    }
}

fn build_inputs() -> (NeoParams, CcsStructure<F>, Vec<McsInstance<neo_ajtai::Commitment, F>>, Vec<McsWitness<F>>) {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let ccs = dummy_ccs();
    ensure_global_ajtai_pp(ccs.m);

    // Build 2 identical MCS instances/witnesses (k+1 = 2) following pipeline.rs pattern
    let mut instances = Vec::new();
    let mut witnesses = Vec::new();
    
    for _ in 0..2 {
        let z = vec![F::ZERO; ccs.m]; // witness vector
        let z_cols = neo_ajtai::decomp_b(&z, params.b, D, neo_ajtai::DecompStyle::Balanced);
        let pp = neo_ajtai::get_global_pp().unwrap();
        let c = ajtai_commit(&pp, &z_cols);

        // Convert to row-major matrix for McsWitness
        let mut z_mat = Mat::zero(D, ccs.m, F::ZERO);
        for col in 0..ccs.m { 
            for row in 0..D { 
                z_mat[(row, col)] = z_cols[col * D + row]; 
            } 
        }

        instances.push(McsInstance { c, x: vec![], m_in: 0 });
        witnesses.push(McsWitness { w: z, Z: z_mat });
    }
    
    (params, ccs, instances, witnesses)
}

#[test]
#[serial]
fn rt2_digit_commitment_tamper_must_fail_recomposition() {
    let (params, ccs, inputs, witnesses) = build_inputs();
    let (digits, _dwits, proof) = fold_ccs_instances(&params, &ccs, &inputs, &witnesses).expect("prove");

    let parent = recombine_parent(&params, &digits);
    let mut bad_proof = proof.clone();

    // Tamper: replace first digit commitment with zeros of same shape
    let original_commitment = &bad_proof.pi_dec_proof.digit_commitments.as_ref().expect("have digits")[0];
    bad_proof.pi_dec_proof.digit_commitments.as_mut().expect("have digits")[0] =
        neo_ajtai::Commitment::zeros(original_commitment.d, original_commitment.kappa);

    let l = neo_ajtai::AjtaiSModule::from_global().expect("AjtaiSModule");
    let ok = pi_dec_verify(&mut FoldTranscript::default(),
                           &params, &parent, &digits, &bad_proof.pi_dec_proof, &l).expect("verify");
    assert!(!ok, "DEC must fail when c ≠ Σ b^i · c_i");
}

#[test]
#[serial]
fn rt11_wrong_r_on_digit_must_fail() {
    let (params, ccs, inputs, witnesses) = build_inputs();
    let (mut digits, _dwits, proof) = fold_ccs_instances(&params, &ccs, &inputs, &witnesses).expect("prove");

    let parent = recombine_parent(&params, &digits);

    // Tamper: change r on one digit
    digits[0].r[0] = K::from(F::from_u64(7));

    let l = neo_ajtai::AjtaiSModule::from_global().expect("AjtaiSModule");
    let ok = pi_dec_verify(&mut FoldTranscript::default(),
                           &params, &parent, &digits, &proof.pi_dec_proof, &l).expect("verify");
    assert!(!ok, "DEC must fail when a digit's r differs from the parent r");
}

#[test]
#[serial]
fn rt8_base_mismatch_in_verify_must_fail() {
    // Build with params₁ (whose b matches how digits/parent were produced) …
    let (params1, ccs, inputs, witnesses) = build_inputs();
    let (digits, _dwits, proof) = fold_ccs_instances(&params1, &ccs, &inputs, &witnesses).expect("prove");
    let parent = recombine_parent(&params1, &digits);

    // … but verify with params₂ that has a different 'b' (base).
    // If your API only exposes goldilocks_autotuned_s2(k, T, b), pick a different 'b' here.
    let params2 = NeoParams::goldilocks_autotuned_s2(3, 2, 4);

    let l = neo_ajtai::AjtaiSModule::from_global().expect("AjtaiSModule");
    let ok = pi_dec_verify(&mut FoldTranscript::default(),
                           &params2, &parent, &digits, &proof.pi_dec_proof, &l).expect("verify");
    assert!(!ok, "DEC must fail when verifier's base b differs (recomposition checks break)");
}
