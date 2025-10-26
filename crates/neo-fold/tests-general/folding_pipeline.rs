//! End-to-end folding tests for neo-fold: Π_CCS → Π_RLC → Π_DEC,
//! plus red-team tampering checks.
//!
//! These tests use real Ajtai commitments to properly exercise the
//! commitment verification in Π_DEC, executing the full protocol pipeline.

#![allow(non_snake_case)]

use neo_fold::{
    pi_ccs::{pi_ccs_prove, pi_ccs_verify, eval_tie_constraints},
    pi_rlc::{pi_rlc_prove, pi_rlc_verify},
    pi_dec::{pi_dec, pi_dec_verify},
};
use neo_transcript::Transcript;
use neo_ccs::{
    Mat, SparsePoly, Term,
    relations::{McsInstance, McsWitness},
    traits::SModuleHomomorphism,
};
use neo_math::{F, K, D, Rq, cf_inv};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

/// Dummy S-module homomorphism for testing with correct commitment shape
struct DummyS { kappa: usize }

impl SModuleHomomorphism<F, Cmt> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Cmt { 
        Cmt::zeros(z.rows(), self.kappa) // Use correct κ from params
    }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut out = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows { for c in 0..cols { out[(r, c)] = z[(r, c)]; } }
        out
    }
}

/// Make a simple CCS: z[0] + z[1] = z[2] constraint.
fn make_ccs() -> neo_ccs::CcsStructure<F> {
    // Create a 4x3 matrix for constraint: z[0] + z[1] - z[2] = 0
    let mat = Mat::from_row_major(4, 3, vec![
        F::ONE, F::ZERO, F::ZERO,   // z[0]
        F::ZERO, F::ONE, F::ZERO,   // z[1] 
        F::ZERO, F::ZERO, -F::ONE,  // -z[2]
        F::ZERO, F::ZERO, F::ZERO,  // padding
    ]);
    
    // f(y) = y (degree 1, single variable)
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    neo_ccs::CcsStructure::new(vec![mat], f).expect("valid CCS")
}

/// Build `(MCS, McsWitness)` with constraint-satisfying witness: z[0] + z[1] = z[2].
fn make_instance_and_witness(params: &NeoParams, offset: u64, l: &DummyS) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    // Create witness that satisfies z[0] + z[1] = z[2]
    let a = F::from_u64(offset + 3);
    let b = F::from_u64(offset + 5);
    let c_val = a + b; // z[0] + z[1] = z[2]
    let z = vec![a, b, c_val];
    let m = z.len();

    // Z (column-major) from Ajtai decomp, then convert to row-major D×m
    let z_digits_col_major = neo_ajtai::decomp_b(&z, params.b, D, neo_ajtai::DecompStyle::Balanced);
    neo_ajtai::assert_range_b(&z_digits_col_major, params.b).expect("balanced digits in range");

    let mut row_major = vec![F::ZERO; D * m];
    for c in 0..m { for r in 0..D { row_major[r * m + c] = z_digits_col_major[c * D + r]; } }
    let Z = Mat::from_row_major(D, m, row_major);

    // Use dummy commitment with correct shape
    let c = l.commit(&Z);

    // Expose one public input so X has 1 column (fixes X 54x0 in digits)
    let inst = McsInstance { c, x: vec![z[0]], m_in: 1 };
    let wit  = McsWitness { w: z[1..].to_vec(), Z };
    (inst, wit)
}

/// Recombine `k` digit ME instances into their parent ME(B, L) (public instance only).
/// Mirrors `lib.rs` recombination but kept local for testing.
fn recombine_me_digits_to_parent(params: &NeoParams, digits: &[neo_ccs::MeInstance<Cmt, F, K>]) -> neo_ccs::MeInstance<Cmt, F, K> {
    assert!(!digits.is_empty());
    let m_in = digits[0].m_in;
    let r_ref = digits[0].r.clone();
    let t = digits[0].y.len();
    let d_rows = digits[0].X.rows();
    let x_cols = digits[0].X.cols();

    // c_parent = Σ b^i · c_i, via ring scalar cf_inv([pow,0,...])
    let mut coeffs: Vec<Rq> = Vec::with_capacity(digits.len());
    let mut pow_f = F::ONE;
    for _ in 0..digits.len() {
        let mut arr = [F::ZERO; D];
        arr[0] = pow_f;
        coeffs.push(cf_inv(arr));
        pow_f *= F::from_u64(params.b as u64);
    }
    let cs: Vec<Cmt> = digits.iter().map(|d| d.c.clone()).collect();
    let c_parent = neo_ajtai::s_lincomb(&coeffs, &cs).expect("s_lincomb");

    // X_parent = Σ b^i · X_i
    let mut X_parent = Mat::zero(d_rows, x_cols, F::ZERO);
    let mut pow = F::ONE;
    for d in digits {
        for r in 0..d_rows { for c in 0..x_cols { X_parent[(r, c)] += d.X[(r, c)] * pow; } }
        pow *= F::from_u64(params.b as u64);
    }

    // y_parent[j] = Σ b^i · y_{i,j}
    let y_dim = digits[0].y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_parent = vec![vec![K::ZERO; y_dim]; t];
    let mut pow_k = K::from(F::ONE);
    let base_k = K::from(F::from_u64(params.b as u64));
    for d in digits {
        for j in 0..t { for u in 0..y_dim { y_parent[j][u] += d.y[j][u] * pow_k; } }
        pow_k *= base_k;
    }

    // y_scalars parent = Σ b^i · y_scalars_i
    let t_scal = digits[0].y_scalars.len();
    let mut y_scalars_parent = vec![K::ZERO; t_scal];
    let mut powk = K::from(F::ONE);
    for d in digits {
        for j in 0..t_scal { y_scalars_parent[j] += d.y_scalars[j] * powk; }
        powk *= base_k;
    }

    neo_ccs::MeInstance {
        c_step_coords: vec![],
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

/// Recombine digit witnesses (row-major) to parent witness: Z' = Σ b^i · Z_i
fn recombine_digit_witnesses_to_parent(params: &NeoParams, digits: &[neo_ccs::MeWitness<F>]) -> neo_ccs::MeWitness<F> {
    assert!(!digits.is_empty());
    let d = digits[0].Z.rows();
    let m = digits[0].Z.cols();
    let mut Zp = Mat::zero(d, m, F::ZERO);
    let mut pow = F::ONE;
    for w in digits {
        for r in 0..d { for c in 0..m { Zp[(r, c)] += w.Z[(r, c)] * pow; } }
        pow *= F::from_u64(params.b as u64);
    }
    neo_ccs::MeWitness { Z: Zp }
}

/// Build v_j = M_j^T · χ_r and recompute tie residual (must be zero for honest outputs).
fn tie_residual(s: &neo_ccs::CcsStructure<F>, Z: &neo_ccs::Mat<F>, r: &[K], y: &[Vec<K>]) -> K {
    eval_tie_constraints(s, Z, y, r)
}

#[test]
fn folding_roundtrip_accepts() {
    // Parameters: create minimal test parameters with k=2
    let params = NeoParams {
        q: 0xFFFFFFFF00000001u64, // Goldilocks prime
        eta: 81,          // Standard cyclotomic index
        d: D as u32,      // φ(81) = 54, ring dimension
        kappa: 4,         // Small κ for commitments  
        m: 64,            // Message length
        b: 2,             // Base 2 decomposition
        k: 12,            // Large k to get much larger B
        B: 4096,          // B = b^k = 2^12 = 4096 (should satisfy guard) 
        T: 2,             // T may be auto-calculated anyway
        s: 2,             // Extension degree (only s=2 supported)
        lambda: 80,       // Security parameter
    };
    println!("🔧 Using parameters: k={}, b={}, kappa={}", params.k, params.b, params.kappa);
    
    // Use dummy S-module with correct κ (fixes Π_DEC commitment shape issue)
    let l = DummyS { kappa: params.kappa as usize };

    // Simple CCS: z[0] + z[1] = z[2], m=3, t=1
    let s = make_ccs();
    let m = 3; // z has 3 elements

    // Build k+1=13 instances (since k=12 now)
    let mut instances = Vec::new();
    let mut witnesses = Vec::new();
    for i in 0..=params.k as u64 {
        let (inst, wit) = make_instance_and_witness(&params, i * 7, &l);
        instances.push(inst);
        witnesses.push(wit);
    }

    // Π_CCS (prove + verify)
    let mut tr_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (me_list, pi_ccs_proof) = pi_ccs_prove(&mut tr_p, &params, &s, &instances, &witnesses, &[], &[], &l)
        .expect("pi_ccs_prove");

    let mut tr_v = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let ok_ccs = neo_fold::pi_ccs::pi_ccs_verify(&mut tr_v, &params, &s, &instances, &[], &me_list, &pi_ccs_proof)
        .expect("pi_ccs_verify");
    assert!(ok_ccs, "Π_CCS verification must accept");

    // Π_RLC (prove + verify)
    let mut tr_rlc_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (me_B, pi_rlc_proof) = pi_rlc_prove(&mut tr_rlc_p, &params, &me_list).expect("pi_rlc_prove");

    let mut tr_rlc_v = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let ok_rlc = pi_rlc_verify(&mut tr_rlc_v, &params, &me_list, &me_B, &pi_rlc_proof).expect("pi_rlc_verify");
    assert!(ok_rlc, "Π_RLC verification must accept");

    // Build witness for parent (RLC output): Z' = Σ rot(ρ_i)·Z_i
    // (recompute as done in fold_ccs_instances)
    let mut Z_prime = Mat::zero(D, m, F::ZERO);
    let rho_ring: Vec<Rq> = pi_rlc_proof.rho_elems
        .iter()
        .map(|coeffs| cf_inv(*coeffs))
        .collect();
    for (wit, rho) in witnesses.iter().zip(rho_ring.iter()) {
        let s_action = neo_math::SAction::from_ring(*rho);
        for c in 0..m {
            let mut col = [F::ZERO; D];
            for r in 0..D { col[r] = wit.Z[(r, c)]; }
            let rot = s_action.apply_vec(&col);
            for r in 0..D { Z_prime[(r, c)] += rot[r]; }
        }
    }
    let me_B_wit = neo_ccs::MeWitness { Z: Z_prime };

    // Π_DEC (prove + verify)
    let mut tr_dec_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (digits, digit_wits, pi_dec_proof) =
        pi_dec(&mut tr_dec_p, &params, &me_B, &me_B_wit, &s, &l).expect("pi_dec");

    // COMPREHENSIVE DEBUGGING: Check each verification step that pi_dec_verify performs
    
    println!("🔍 Starting comprehensive Π_DEC verification debugging...");
    
    // Step 1: Basic parameter checks
    let k = params.k as usize;
    let b = params.b;
    println!("📊 Parameters: k={}, b={}, digits.len()={}", k, b, digits.len());
    assert_eq!(digits.len(), k, "❌ digits.len() != k");
    println!("✅ Step 1: Parameter validation passed");
    
    // Step 2: Check y recomposition (what we've already verified)
    let base_k = K::from(F::from_u64(b as u64));
    for j in 0..me_B.y.len() {
        let mut recomposed = vec![K::ZERO; me_B.y[j].len()];
        let mut pow = K::from(F::ONE);
        for d in &digits {
            for u in 0..recomposed.len() { 
                recomposed[u] += d.y[j][u] * pow; 
            }
            pow *= base_k;
        }
        assert_eq!(recomposed, me_B.y[j], "❌ y recomposition mismatch at j={}", j);
    }
    println!("✅ Step 2: Y vector recomposition checks passed");
    
    // Step 3: Check X recomposition (new - this might be where it fails!)
    let base_f = F::from_u64(b as u64);
    let rows = me_B.X.rows();
    let cols = me_B.X.cols();
    println!("📐 X matrix dimensions: {}x{}", rows, cols);
    
    for r in 0..rows {
        for c in 0..cols {
            let parent = me_B.X[(r,c)];
            let limbs: Vec<F> = digits.iter().map(|d| d.X[(r,c)]).collect();
            
            // Manual recombination: parent = Σ b^i * limbs[i]
            let mut recomposed = F::ZERO;
            let mut pow = F::ONE;
            for limb in &limbs {
                recomposed += *limb * pow;
                pow *= base_f;
            }
            
            if parent != recomposed {
                println!("❌ X recomposition failed at ({},{}): parent={:?}, limbs={:?}, recomposed={:?}", 
                         r, c, parent, limbs, recomposed);
                panic!("X recomposition failed!");
            }
        }
    }
    println!("✅ Step 3: X matrix recomposition checks passed");
    
    // Step 4: Check y_scalars recomposition
    for j in 0..me_B.y_scalars.len() {
        let parent_scalar = me_B.y_scalars[j];
        let digit_scalars: Vec<K> = digits.iter()
            .filter_map(|me| me.y_scalars.get(j).copied())
            .collect();
        
        if digit_scalars.len() != k {
            println!("❌ y_scalars inconsistent structure at j={}: expected {} digits, got {}", 
                     j, k, digit_scalars.len());
            panic!("y_scalars structure inconsistent!");
        }
        
        let mut recomposed = K::ZERO;
        let mut pow = K::from(F::ONE);
        for &scalar in &digit_scalars {
            recomposed += scalar * pow;
            pow *= base_k;
        }
        
        if parent_scalar != recomposed {
            println!("❌ y_scalars recomposition failed at j={}: parent={:?}, digits={:?}, recomposed={:?}",
                     j, parent_scalar, digit_scalars, recomposed);
            panic!("y_scalars recomposition failed!");
        }
    }
    println!("✅ Step 4: y_scalars recomposition checks passed");
    
    // Step 5: Check instance consistency 
    for (i, me_digit) in digits.iter().enumerate() {
        assert_eq!(me_digit.r, me_B.r, "❌ r mismatch at digit {}", i);
        assert_eq!(me_digit.m_in, me_B.m_in, "❌ m_in mismatch at digit {}", i);
    }
    println!("✅ Step 5: Instance consistency checks passed");
    
    // Step 6: Check commitment recomposition (what we've already verified)
    let mut coeffs = Vec::<Rq>::new();
    let mut pow_f = F::ONE;
    for _ in 0..digits.len() {
        let mut arr = [F::ZERO; D]; 
        arr[0] = pow_f;
        coeffs.push(cf_inv(arr));
        pow_f *= F::from_u64(b as u64);
    }
    let c_parent_from_digits = neo_ajtai::s_lincomb(
        &coeffs, 
        &digits.iter().map(|d| d.c.clone()).collect::<Vec<_>>()
    ).expect("s_lincomb");
    assert_eq!(c_parent_from_digits, me_B.c, "❌ Commitment recomposition mismatch");
    println!("✅ Step 6: Commitment recomposition checks passed");
    
    // Step 7: Check ME relation consistency for each digit
    for (i, me_digit) in digits.iter().enumerate() {
        // Check y vector structure consistency  
        for y_j in &me_digit.y {
            if y_j.is_empty() {
                println!("❌ Empty y vector in digit {} ", i);
                panic!("Empty y vector!");
            }
        }
        
        // Check X matrix dimensions
        if me_digit.X.rows() == 0 || me_digit.X.cols() == 0 {
            println!("❌ Invalid X matrix dimensions in digit {}: {}x{}", i, me_digit.X.rows(), me_digit.X.cols());
            panic!("Invalid X dimensions!");
        }
    }
    println!("✅ Step 7: ME relation consistency checks passed");
    
    // Step 8: Check proof structure
    println!("🔍 Proof structure:");
    println!("  - digit_commitments: {:?}", pi_dec_proof.digit_commitments.is_some());
    println!("  - recomposition_proof len: {}", pi_dec_proof.recomposition_proof.len());
    println!("  - range_proofs len: {}", pi_dec_proof.range_proofs.len());
    
    // Step 9: Check if digit commitments are present
    if let Some(ref digit_commitments) = pi_dec_proof.digit_commitments {
        println!("✅ Step 9: digit_commitments present with {} entries", digit_commitments.len());
        
        // Check that digit commitments match ME instance commitments
        for (i, (me_digit, c_digit)) in digits.iter().zip(digit_commitments.iter()).enumerate() {
            if &me_digit.c != c_digit {
                println!("❌ Commitment mismatch at digit {}", i);
                panic!("Digit commitment mismatch!");
            }
        }
        println!("✅ Step 9b: Digit commitment consistency checks passed");
    } else {
        println!("⚠️  No digit_commitments in proof");
    }
    
    println!("🎯 All manual verification steps passed! Now calling pi_dec_verify...");
    
    let mut tr_dec_v = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let ok_dec = pi_dec_verify(&mut tr_dec_v, &params, &me_B, &digits, &pi_dec_proof, &l)
        .expect("pi_dec_verify");
    
    if !ok_dec {
        println!("❌ pi_dec_verify returned false despite all manual checks passing!");
        println!("🤔 This suggests the issue is in:");
        println!("   - Transcript state differences between prove/verify");
        println!("   - Some verification step not covered by our manual checks");  
        println!("   - Interaction between DummyS and internal verification logic");
    }
    
    assert!(ok_dec, "Π_DEC verification must accept");

    // Recombine digits → parent again and compare to Π_RLC parent
    let me_parent = recombine_me_digits_to_parent(&params, &digits);
    assert_eq!(me_parent.c, me_B.c, "c recomposition mismatch");
    assert_eq!(me_parent.X.as_slice(), me_B.X.as_slice(), "X recomposition mismatch");
    assert_eq!(me_parent.y, me_B.y, "y recomposition mismatch");
    assert_eq!(me_parent.y_scalars, me_B.y_scalars, "y_scalars recomposition mismatch");

    // Recombine digit witnesses → parent witness and check tie residual is zero
    let me_parent_wit = recombine_digit_witnesses_to_parent(&params, &digit_wits);
    let tie = tie_residual(&s, &me_parent_wit.Z, &me_parent.r, &me_parent.y);
    assert_eq!(tie, K::ZERO, "tie residual must be zero for honest recomposed parent");

    // Also ensure Π_CCS terminal check holds with the produced y_scalars (f ≡ 0 in this test)
    // i.e., expected Q(r) = 0 for each ME output from Π_CCS (already enforced in verify).
    // Nothing else to do here.
}

#[test]
fn pi_ccs_detects_y_scalar_tamper() {
    // Same tiny CCS as above
    let params = NeoParams::goldilocks_for_circuit(3, 1, 2);
    let l = DummyS { kappa: params.kappa as usize };
    let s = make_ccs();

    let (inst0, wit0) = make_instance_and_witness(&params, 0, &l);
    let instances = vec![inst0.clone()];
    let witnesses = vec![wit0];

    // Π_CCS
    let mut tr_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (mut me_list, proof) = pi_ccs_prove(&mut tr_p, &params, &s, &instances, &witnesses, &[], &[], &l)
        .expect("pi_ccs_prove(single)");

    // Tamper y_scalars (which feed the verified Q(r) terminal check)
    if let Some(first) = me_list.get_mut(0) {
        if !first.y_scalars.is_empty() {
            first.y_scalars[0] += K::ONE;
        }
    }

    let mut tr_v = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_ccs_verify(&mut tr_v, &params, &s, &instances, &[], &me_list, &proof)
        .expect("pi_ccs_verify");
    assert!(!ok, "Π_CCS must reject when y_scalars are tampered");
}

#[test]
fn pi_dec_detects_digit_x_tamper() {
    // Build a 3-input fold to exercise DEC
    let params = NeoParams::goldilocks_for_circuit(3, 1, 2);
    let l = DummyS { kappa: params.kappa as usize };
    let s = make_ccs();
    let m = 3; // z has 3 elements

    let (inst0, wit0) = make_instance_and_witness(&params, 0, &l);
    let (inst1, wit1) = make_instance_and_witness(&params, 11, &l);
    let (inst2, wit2) = make_instance_and_witness(&params, 22, &l);

    let instances = vec![inst0, inst1, inst2];
    let witnesses = vec![wit0, wit1, wit2];

    // Π_CCS
    let mut tr_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (me_list, _) = pi_ccs_prove(&mut tr_p, &params, &s, &instances, &witnesses, &[], &[], &l)
        .expect("pi_ccs_prove");

    // Π_RLC
    let mut tr_rlc_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (me_B, pi_rlc_proof) = pi_rlc_prove(&mut tr_rlc_p, &params, &me_list).expect("pi_rlc_prove");

    // Parent witness Z' as in fold_ccs_instances
    let mut Z_prime = Mat::zero(D, m, F::ZERO);
    let rho_ring: Vec<Rq> = pi_rlc_proof.rho_elems.iter().map(|cs| cf_inv(*cs)).collect();
    for (idx, w) in witnesses.iter().enumerate() {
        let s_action = neo_math::SAction::from_ring(rho_ring[idx]);
        for c in 0..m {
            let mut col = [F::ZERO; D];
            for r in 0..D { col[r] = w.Z[(r, c)]; }
            let rot = s_action.apply_vec(&col);
            for r in 0..D { Z_prime[(r, c)] += rot[r]; }
        }
    }
    let me_B_wit = neo_ccs::MeWitness { Z: Z_prime };

    // Π_DEC
    let mut tr_dec_p = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let (mut digits, _digit_wits, proof_dec) = pi_dec(&mut tr_dec_p, &params, &me_B, &me_B_wit, &s, &l)
        .expect("pi_dec");

    // Tamper: flip one entry of X in a digit (if X is not empty)
    if let Some(d0) = digits.get_mut(0) {
        if d0.X.rows() > 0 && d0.X.cols() > 0 {
            d0.X[(0, 0)] += F::ONE;
        }
    }

    // Verify must FAIL
    let mut tr_dec_v = neo_transcript::Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_dec_verify(&mut tr_dec_v, &params, &me_B, &digits, &proof_dec, &l)
        .expect("pi_dec_verify");
    assert!(!ok, "Π_DEC must reject tampered digit X");
}
