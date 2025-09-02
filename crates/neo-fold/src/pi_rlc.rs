//! Π_RLC: Random linear combination with S-action
//!
//! Combines k+1 ME instances into k instances using strong-sampled ρ_i ∈ S
//! and proper S-action on matrices and K-vectors.

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use crate::transcript::{FoldTranscript, Domain};
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

/// Prove Π_RLC: combine k+1 ME instances to k instances  
pub fn pi_rlc_prove(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    me_list: &[MeInstance<Cmt, F, K>],
) -> Result<(MeInstance<Cmt, F, K>, PiRlcProof), PiRlcError> {
    // === Domain separation & extension policy binding ===
    tr.domain(Domain::Rlc);
    tr.absorb_bytes(b"neo/params/v1");
    tr.absorb_u64(&[
        params.q, params.lambda as u64,
        me_list.len() as u64,
        params.s as u64,
    ]);
    
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
    
    // === Sample ρ_i ∈ S with strong sampling ===
    let mut challenger = tr.challenger();
    let (rhos, T_bound) = sample_kplus1_invertible(&mut challenger, &DEFAULT_STRONGSET, me_list.len())
        .map_err(|e| PiRlcError::SamplingFailed(e.to_string()))?;
    
    // === Enforce guard constraint: (k+1)T(b-1) < B ===
    let guard_lhs = (k as u128 + 1) * (T_bound as u128) * ((params.b as u128) - 1);
    if guard_lhs >= params.B as u128 {
        return Err(PiRlcError::GuardViolation(format!(
            "guard failed: ({}+1)*{}*({}-1) = {} >= {}",
            k, T_bound, params.b, guard_lhs, params.B
        )));
    }
    
    eprintln!("✅ PI_RLC: Guard check passed: {} < {}", guard_lhs, params.B);
    eprintln!("  Strong sampling: T_bound = {}", T_bound);
    
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
    let me_combined = MeInstance {
        c: c_prime,
        X: X_prime,
        y: y_prime,
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
    
    eprintln!("✅ PI_RLC: Combination completed");
    eprintln!("  Combined {} instances into 1", me_list.len());
    
    Ok((me_combined, proof))
}

/// Verify Π_RLC combination proof
pub fn pi_rlc_verify(
    tr: &mut FoldTranscript,
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
    tr.domain(Domain::Rlc);
    tr.absorb_bytes(b"neo/params/v1");
    tr.absorb_u64(&[
        params.q, params.lambda as u64,
        input_me_list.len() as u64,
        params.s as u64,
    ]);
    
    // === Re-derive ρ rotations deterministically ===
    let mut challenger = tr.challenger();
    let (expected_rhos, expected_T) = match sample_kplus1_invertible(&mut challenger, &DEFAULT_STRONGSET, input_me_list.len()) {
        Ok(result) => result,
        Err(_) => return Ok(false),
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
    let recomputed_c = s_lincomb(&rho_ring, &input_cs)
        .map_err(|e| PiRlcError::SActionError(format!("Commitment verification failed: {}", e)))?;
    if recomputed_c != _output_me.c {
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
    
    // All verifications passed: guard, c', X', and y' are all correct
    Ok(true)
}

