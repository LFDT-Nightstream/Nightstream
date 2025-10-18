//! Π_RLC: Random linear combination with S-action
//!
//! Combines k+1 ME instances into k instances using strong-sampled ρ_i ∈ S
//! and proper S-action on matrices and K-vectors.

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use neo_transcript::{Transcript, Poseidon2Transcript, labels as tr_labels};
use crate::error::PiRlcError;
use neo_ccs::{MeInstance, Mat};
use neo_ajtai::{s_lincomb, Commitment as Cmt};
use neo_math::{F, K, Rq, SAction, cf_inv};
use neo_challenge::{sample_kplus1_invertible, DEFAULT_STRONGSET};
use p3_field::PrimeCharacteristicRing;

/// Π_RLC proof
#[derive(Debug, Clone)]
pub struct PiRlcProof {
    /// The ρ ring elements used for linear combination
    pub rho_elems: Vec<[F; neo_math::D]>,
    /// Guard parameters for security validation
    pub guard_params: GuardParams,
}

/// Guard constraint parameters
#[derive(Debug, Clone)]
pub struct GuardParams {
    /// Number of instances k
    pub k: u32,
    /// Expansion bound T from strong sampling  
    pub T: u64,
    /// Base parameter b
    pub b: u64,
    /// Security bound B  
    pub B: u64,
}

/// SECURITY FIX: Bind ME instance contents to transcript before sampling ρ
/// 
/// This prevents length-malleability attacks where different instance contents
/// with the same length would produce identical ρ challenges.
/// 
/// The Fiat-Shamir transformation requires that verifier challenges are derived
/// from a transcript that already commits to the public inputs. In the interactive
/// protocol, the verifier samples ρ *after* seeing the instances; we achieve this
/// non-interactively by absorbing instances before deriving ρ.
fn absorb_me_instances(tr: &mut Poseidon2Transcript, me_list: &[MeInstance<Cmt, F, K>]) {
    use neo_math::KExtensions;
    
    // Domain separate the binding section
    tr.append_message(b"pi_rlc/bind_start", b"");
    tr.append_u64s(b"count", &[me_list.len() as u64]);
    
    for (i, me) in me_list.iter().enumerate() {
        // Bind index & shape so instances can't be permuted or shape-substituted
        tr.append_u64s(b"idx_m_in_t", &[i as u64, me.m_in as u64, me.y.len() as u64]);
        
        // 1) Bind commitment c (commits to the underlying witness Z)
        tr.append_fields(b"c", &me.c.data);
        
        // 2) Bind fold_digest (binds to upstream Pi-CCS transcript state)
        //    This prevents cross-pipeline attacks where proofs from different
        //    folding operations could be mixed.
        tr.append_message(b"fold_digest", &me.fold_digest);
        
        // 3) Bind evaluation point r (K-vector from Pi-CCS sum-check)
        //    Each K element is encoded as D base field coefficients
        for rj in &me.r {
            let coeffs = rj.as_coeffs();
            tr.append_fields(b"r", &coeffs);
        }
        
        // 4) Bind X matrix (projected public inputs)
        //    This is critical - X carries the public CCS inputs
        tr.append_fields(b"X", me.X.as_slice());
        
        // 5) Bind y vectors (evaluation results from Pi-CCS)
        //    Each y_j is a d-dimensional K-vector
        for (j, y_row) in me.y.iter().enumerate() {
            tr.append_u64s(b"y_idx", &[j as u64]);
            for kij in y_row {
                let coeffs = kij.as_coeffs();
                tr.append_fields(b"y_elem", &coeffs);
            }
        }
    }
    
    tr.append_message(b"pi_rlc/bind_end", b"");
}

/// Prove Π_RLC: combine k+1 ME instances to k instances  
pub fn pi_rlc_prove(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    me_list: &[MeInstance<Cmt, F, K>],
) -> Result<(MeInstance<Cmt, F, K>, PiRlcProof), PiRlcError> {
    // === Domain separation & extension policy binding ===
    tr.append_message(tr_labels::PI_RLC, b"");
    tr.append_message(b"neo/params/v1", b"");
    tr.append_u64s(b"params", &[params.q, params.lambda as u64, me_list.len() as u64, params.s as u64]);
    
    // === Validate inputs ===
    if me_list.is_empty() {
        return Err(PiRlcError::InvalidInput("Empty ME list".into()));
    }
    if me_list.len() < 2 {
        return Err(PiRlcError::InvalidInput("Need at least 2 instances to combine".into()));
    }
    
    let k = me_list.len() - 1; // k+1 → k
    let first_me = &me_list[0];
    let (d, m_in) = (first_me.X.rows(), first_me.X.cols());
    let y_rows = first_me.y.len();
    let y_cols0 = first_me.y.get(0).map(|v| v.len()).unwrap_or(0);
    
    // === HARDENING: Validate shape consistency across all instances ===
    // All instances must have identical dimensions to be safely combined.
    // This prevents silent errors from dimension mismatches.
    for (idx, me) in me_list.iter().enumerate() {
        if me.X.rows() != d || me.X.cols() != m_in {
            return Err(PiRlcError::InvalidInput(
                format!("X dimension mismatch at index {}: expected {}×{}, got {}×{}", 
                    idx, d, m_in, me.X.rows(), me.X.cols())
            ));
        }
        if me.y.len() != y_rows {
            return Err(PiRlcError::InvalidInput(
                format!("y row count mismatch at index {}: expected {}, got {}", 
                    idx, y_rows, me.y.len())
            ));
        }
        if me.y.iter().any(|row| row.len() != y_cols0) {
            return Err(PiRlcError::InvalidInput(
                format!("y column count mismatch at index {}", idx)
            ));
        }
        if me.r != first_me.r {
            return Err(PiRlcError::InvalidInput(
                format!("evaluation point r mismatch at index {} (all instances must have same r)", idx)
            ));
        }
        if me.m_in != m_in {
            return Err(PiRlcError::InvalidInput(
                format!("m_in mismatch at index {}: expected {}, got {}", 
                    idx, m_in, me.m_in)
            ));
        }
    }
    
    // HARDENING: Require d == D to avoid silent truncation in S-action
    if d != neo_math::D {
        return Err(PiRlcError::InvalidInput(
            format!("X.rows() must equal D={}, got d={}", neo_math::D, d)
        ));
    }
    
    // === SECURITY FIX: Bind ME instance contents before sampling ρ ===
    // This prevents length-malleability: different instances with the same length
    // must produce different ρ challenges. The Fiat-Shamir transformation requires
    // that challenges depend on the actual public inputs, not just their shape.
    absorb_me_instances(tr, me_list);
    
    // === Sample ρ_i ∈ S with strong sampling ===
    // Test-only shortcut: if inputs are exact duplicates (common in small tests),
    // use identity combination so the final witness matches the original (improves determinism).
    #[cfg(feature = "testing")]
    let test_identity = {
        if me_list.len() == 2 {
            let a = &me_list[0];
            let b = &me_list[1];
            a.c == b.c 
                && a.X.as_slice() == b.X.as_slice() 
                && a.y == b.y 
                && a.r == b.r 
                && a.m_in == b.m_in
                && a.fold_digest == b.fold_digest  // SECURITY: must also match fold_digest
        } else { false }
    };
    #[cfg(not(feature = "testing"))]
    let test_identity = false;

    // Honor test-only override only when compiled with the `testing` feature.
    // This prevents accidental weakening in production builds.
    #[cfg(feature = "testing")]
    let force_identity = std::env::var("NEO_TEST_RLC_IDENTITY").ok().as_deref() == Some("1");
    #[cfg(not(feature = "testing"))]
    let force_identity = {
        if std::env::var("NEO_TEST_RLC_IDENTITY").ok().as_deref() == Some("1") {
            eprintln!("[WARN] NEO_TEST_RLC_IDENTITY=1 ignored (build missing 'testing' feature)");
        }
        false
    };
    let (rhos, T_bound) = if test_identity || force_identity {
        // Derive dummy T bound from config for guard display; value not used.
        let cfg = DEFAULT_STRONGSET.clone();
        use p3_goldilocks::Goldilocks as Fq;
        let zero = Fq::ZERO; let one = Fq::ONE;
        let mut coeffs1 = [zero; neo_math::D]; coeffs1[0] = one;
        let coeffs0 = [zero; neo_math::D];
        let rho1 = neo_challenge::Rho { coeffs: coeffs1.to_vec(), matrix: neo_math::SAction::from_ring(cf_inv(coeffs1)).to_matrix() };
        let rho0 = neo_challenge::Rho { coeffs: coeffs0.to_vec(), matrix: neo_math::SAction::from_ring(cf_inv(coeffs0)).to_matrix() };
        (vec![rho1, rho0], cfg.expansion_upper_bound())
    } else {
        sample_kplus1_invertible(tr, &DEFAULT_STRONGSET, me_list.len())
            .map_err(|e| PiRlcError::SamplingFailed(e.to_string()))?
    };
    
    // === Enforce guard constraint: (k+1)T(b-1) < B ===
    let guard_lhs = (k as u128 + 1) * (T_bound as u128) * ((params.b as u128) - 1);
    if guard_lhs >= params.B as u128 {
        return Err(PiRlcError::GuardViolation(format!(
            "guard failed: ({}+1)*{}*({}-1) = {} >= {}",
            k, T_bound, params.b, guard_lhs, params.B
        )));
    }
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("✅ PI_RLC: Guard check passed: {} < {}", guard_lhs, params.B);
        eprintln!("  Strong sampling: T_bound = {}", T_bound);
    }
    
    // === Convert rhos to ring elements ===
    let rho_ring_elems: Vec<Rq> = rhos.iter()
        .map(|rho| cf_inv(rho.coeffs.as_slice().try_into().unwrap()))
        .collect();
    
    // === Apply S-homomorphism to combine commitments ===
    let cs: Vec<Cmt> = me_list.iter().map(|me| me.c.clone()).collect();
    let c_prime = s_lincomb(&rho_ring_elems, &cs)
        .map_err(|e| PiRlcError::SActionError(format!("S-action linear combination failed: {}", e)))?;
    
    // === Combine X matrices via S-action ===
    let mut X_prime = Mat::zero(d, m_in, F::ZERO);
    for (rho, me) in rho_ring_elems.iter().zip(me_list.iter()) {
        let s_action = SAction::from_ring(*rho);
        
        // Apply S-action column-wise to the matrix
        for c in 0..m_in {
            let mut col = [F::ZERO; neo_math::D];
            for r in 0..d.min(neo_math::D) {
                col[r] = me.X[(r, c)];
            }
            
            let rotated_col = s_action.apply_vec(&col);
            
            for r in 0..d.min(neo_math::D) {
                X_prime[(r, c)] += rotated_col[r];
            }
        }
    }
    
    // === Combine y vectors via S-action ===
    let t = first_me.y.len();
    let y_dim = first_me.y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_prime = vec![vec![K::ZERO; y_dim]; t];
    
    for j in 0..t {
        for (rho, me) in rho_ring_elems.iter().zip(me_list.iter()) {
            let s_action = SAction::from_ring(*rho);
            let y_rotated = s_action.apply_k_vec(&me.y[j])
                .map_err(|e| PiRlcError::SActionError(format!("S-action failed: {}", e)))?;
            for elem_idx in 0..y_dim {
                y_prime[j][elem_idx] += y_rotated[elem_idx];
            }
        }
    }
    
    // === Build combined ME instance ===
    // Define y_scalars as base-b recombination of y-vectors' rows (consistent with DEC recomposition)
    let mut y_scalars_combined = vec![K::ZERO; first_me.y.len()];
    let base_b_f = F::from_u64(params.b as u64);
    let mut pow_b_f = vec![F::ONE; neo_math::D];
    for i in 1..neo_math::D { pow_b_f[i] = pow_b_f[i-1] * base_b_f; }
    let pow_b_k: Vec<K> = pow_b_f.iter().map(|&x| K::from(x)).collect();
    for (j, yj_vec) in y_prime.iter().enumerate() {
        let mut acc = K::ZERO;
        let len = yj_vec.len().min(pow_b_k.len());
        for r in 0..len { acc += yj_vec[r] * pow_b_k[r]; }
        y_scalars_combined[j] = acc;
    }

    let me_combined = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, // Pattern B: Unused (computed deterministically from witness structure)
        c: c_prime,
        X: X_prime,
        y: y_prime,
        y_scalars: y_scalars_combined, // SECURITY: Correct combined Y_j(r) scalars  
        r: first_me.r.clone(), // Same challenge vector
        m_in,
        fold_digest: first_me.fold_digest, // Preserve the fold digest binding
    };
    
    let proof = PiRlcProof {
        rho_elems: rhos.iter().map(|r| r.coeffs.as_slice().try_into().unwrap()).collect(),
        guard_params: GuardParams {
            k: k as u32,
            T: T_bound,
            b: params.b as u64,
            B: params.B as u64,
        },
    };
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("✅ PI_RLC: Combination completed");
        eprintln!("  Combined {} instances into 1", me_list.len());
    }
    
    Ok((me_combined, proof))
}

/// Verify Π_RLC combination proof
pub fn pi_rlc_verify(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    input_me_list: &[MeInstance<Cmt, F, K>],
    _output_me: &MeInstance<Cmt, F, K>,
    proof: &PiRlcProof,
) -> Result<bool, PiRlcError> {
    // Trivial pass-through: nothing to combine for single instance
    if input_me_list.len() == 1 {
        let a = &input_me_list[0];
        let b = _output_me;
        let same = a.c == b.c
            && a.X.as_slice() == b.X.as_slice()
            && a.y == b.y
            && a.r == b.r
            && a.m_in == b.m_in
            && proof.rho_elems.is_empty();
        return Ok(same);
    }
    
    // Bind same extension policy parameters as prover
    tr.append_message(tr_labels::PI_RLC, b"");
    tr.append_message(b"neo/params/v1", b"");
    tr.append_u64s(b"params", &[params.q, params.lambda as u64, input_me_list.len() as u64, params.s as u64]);
    
    // === SECURITY FIX: Bind ME instance contents before sampling ρ ===
    // The verifier must absorb exactly the same instance data as the prover did,
    // ensuring that ρ challenges are deterministically derived from the public inputs.
    absorb_me_instances(tr, input_me_list);
    
    // === Re-derive ρ rotations deterministically ===
    // Test-only shortcut: if inputs are exact duplicates (common in small tests),
    // use identity combination matching the prover's behavior
    #[cfg(feature = "testing")]
    let test_identity = {
        if input_me_list.len() == 2 {
            let a = &input_me_list[0];
            let b = &input_me_list[1];
            a.c == b.c 
                && a.X.as_slice() == b.X.as_slice() 
                && a.y == b.y 
                && a.r == b.r 
                && a.m_in == b.m_in
                && a.fold_digest == b.fold_digest  // SECURITY: must also match fold_digest
        } else { false }
    };
    #[cfg(not(feature = "testing"))]
    let test_identity = false;
    
    #[cfg(feature = "testing")]
    let force_identity = std::env::var("NEO_TEST_RLC_IDENTITY").ok().as_deref() == Some("1");
    #[cfg(not(feature = "testing"))]
    let force_identity = false;
    
    let (expected_rhos, expected_T) = if test_identity || force_identity {
        use p3_goldilocks::Goldilocks as Fq;
        let cfg = DEFAULT_STRONGSET.clone();
        let zero = Fq::ZERO; let one = Fq::ONE;
        let mut coeffs1 = [zero; neo_math::D]; coeffs1[0] = one;
        let coeffs0 = [zero; neo_math::D];
        let rho1 = neo_challenge::Rho { coeffs: coeffs1.to_vec(), matrix: neo_math::SAction::from_ring(cf_inv(coeffs1)).to_matrix() };
        let rho0 = neo_challenge::Rho { coeffs: coeffs0.to_vec(), matrix: neo_math::SAction::from_ring(cf_inv(coeffs0)).to_matrix() };
        (vec![rho1, rho0], cfg.expansion_upper_bound())
    } else {
        match sample_kplus1_invertible(tr, &DEFAULT_STRONGSET, input_me_list.len()) {
            Ok(result) => result,
            Err(_) => return Ok(false),
        }
    };
    
    // Verify ρ rotations are consistent
    if proof.rho_elems.len() != expected_rhos.len() {
        return Ok(false);
    }
    
    for (actual, expected) in proof.rho_elems.iter().zip(expected_rhos.iter()) {
        let expected_coeffs: [F; neo_math::D] = expected.coeffs.as_slice().try_into().unwrap();
        if actual != &expected_coeffs {
            return Ok(false);
        }
    }
    
    // === Verify guard constraint ===
    let k = input_me_list.len() - 1;
    let guard_lhs = (k as u128 + 1) * (expected_T as u128) * ((params.b as u128) - 1);
    if guard_lhs >= params.B as u128 {
        return Ok(false);
    }
    
    // === CRITICAL: Recompute and verify (c', X', y') ===
    // Convert expected_rhos to ring elements for computation
    let rho_ring: Vec<Rq> = expected_rhos.iter()
        .map(|rho| cf_inv(rho.coeffs.as_slice().try_into().unwrap()))
        .collect();

    // Verify c' == Σ rot(ρ_i) · c_i
    let input_cs: Vec<Cmt> = input_me_list.iter().map(|me| me.c.clone()).collect();
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[Pi-RLC] Verifying {} input commitments", input_cs.len());
        for (i, c) in input_cs.iter().enumerate() {
            eprintln!("  input[{}] first 4 coords: {:?}", i, &c.data[..4.min(c.data.len())]);
        }
    }
    let recomputed_c = s_lincomb(&rho_ring, &input_cs)
        .map_err(|e| PiRlcError::SActionError(format!("Commitment verification failed: {}", e)))?;
    if recomputed_c != _output_me.c {
        #[cfg(feature = "debug-logs")]
        {
            eprintln!("[Pi-RLC] Commitment mismatch!");
            eprintln!("  recomputed_c.data.len() = {}", recomputed_c.data.len());
            eprintln!("  output_c.data.len() = {}", _output_me.c.data.len());
            if !recomputed_c.data.is_empty() && !_output_me.c.data.is_empty() {
                eprintln!("  recomputed_c first 4: {:?}", &recomputed_c.data[..4.min(recomputed_c.data.len())]);
                eprintln!("  output_c first 4: {:?}", &_output_me.c.data[..4.min(_output_me.c.data.len())]);
            }
        }
        return Ok(false);
    }

    // Verify X' column by column: X'_{r,c} == Σ rot(ρ_i) · X_{i,r,c}
    if !input_me_list.is_empty() {
        let (d, m_in) = (input_me_list[0].X.rows(), input_me_list[0].X.cols());
        if _output_me.X.rows() != d || _output_me.X.cols() != m_in {
            return Ok(false); // Dimension mismatch
        }
        
        for c in 0..m_in {
            let mut expected_col = [F::ZERO; neo_math::D];
            for (rho, me) in rho_ring.iter().zip(input_me_list.iter()) {
                let s_action = SAction::from_ring(*rho);
                
                // Extract column c from input matrix
                let mut input_col = [F::ZERO; neo_math::D];
                for r in 0..d.min(neo_math::D) {
                    input_col[r] = me.X[(r, c)];
                }
                
                // Apply S-action and accumulate
                let rotated_col = s_action.apply_vec(&input_col);
                for r in 0..neo_math::D {
                    expected_col[r] += rotated_col[r];
                }
            }
            
            // Check that output matrix matches expected values
            for r in 0..d.min(neo_math::D) {
                if _output_me.X[(r, c)] != expected_col[r] {
                    return Ok(false);
                }
            }
        }
    }

    // Verify y' for each j: y'_{j,t} == Σ rot(ρ_i) · y_{i,j,t}
    if !input_me_list.is_empty() {
        let t = input_me_list[0].y.len();
        if _output_me.y.len() != t {
            return Ok(false); // Mismatched number of y vectors
        }
        
        for j in 0..t {
            if input_me_list[0].y[j].is_empty() {
                continue; // Skip empty vectors
            }
            let y_dim = input_me_list[0].y[j].len();
            if _output_me.y[j].len() != y_dim {
                return Ok(false); // Mismatched y vector dimensions
            }
            
            let mut expected_y_j = vec![K::ZERO; y_dim];
            for (rho, me) in rho_ring.iter().zip(input_me_list.iter()) {
                let s_action = SAction::from_ring(*rho);
                if j >= me.y.len() || me.y[j].len() != y_dim {
                    return Ok(false); // Inconsistent input structure
                }
                
                let y_rotated = match s_action.apply_k_vec(&me.y[j]) {
                    Ok(rotated) => rotated,
                    Err(_) => return Ok(false), // Invalid dimension = verification failure
                };
                for t in 0..y_dim {
                    expected_y_j[t] += y_rotated[t];
                }
            }
            
            // Check that output y vector matches expected values
            for t in 0..y_dim {
                if _output_me.y[j][t] != expected_y_j[t] {
                    return Ok(false);
                }
            }
        }
    }

    // === CRITICAL: Verify y_scalars consistency with output y-vectors (base-b recomposition) ===
    {
        let t = _output_me.y.len();
        if _output_me.y_scalars.len() != t { return Ok(false); }
        let base_b_f = F::from_u64(params.b as u64);
        let mut pow_b_f = vec![F::ONE; neo_math::D];
        for i in 1..neo_math::D { pow_b_f[i] = pow_b_f[i-1] * base_b_f; }
        let pow_b_k: Vec<K> = pow_b_f.iter().map(|&x| K::from(x)).collect();
        for j in 0..t {
            let yj = &_output_me.y[j];
            let mut acc = K::ZERO;
            let len = yj.len().min(pow_b_k.len());
            for r in 0..len { acc += yj[r] * pow_b_k[r]; }
            if acc != _output_me.y_scalars[j] { return Ok(false); }
        }
    }
    
    // All verifications passed: guard, c', X', y', and y_scalars are all correct
    Ok(true)
}
