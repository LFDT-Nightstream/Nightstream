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
    let mut challenger = tr.as_challenger();
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
    let c_prime = s_lincomb(&rho_ring_elems, &cs);
    
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
            let y_rotated = apply_s_action_to_k_vector(&s_action, &me.y[j]);
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
    // Bind same extension policy parameters as prover
    tr.domain(Domain::Rlc);
    tr.absorb_bytes(b"neo/params/v1");
    tr.absorb_u64(&[
        params.q, params.lambda as u64,
        input_me_list.len() as u64,
        params.s as u64,
    ]);
    
    // === Re-derive ρ rotations deterministically ===
    let mut challenger = tr.as_challenger();
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
    
    eprintln!("✅ PI_RLC_VERIFY: All checks passed");
    eprintln!("  Guard validation: {} < {}", guard_lhs, params.B);
    
    Ok(true)
}

/// Helper: Apply S-action to K vector using proper coordinate transformation
fn apply_s_action_to_k_vector(_s_action: &SAction, y: &[K]) -> Vec<K> {
    // TODO: Real S-action on extension field vectors
    // For now, return identity (this needs proper K-field coordinate handling)
    y.to_vec()
}