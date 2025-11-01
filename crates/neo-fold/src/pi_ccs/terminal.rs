//! Terminal module: RHS assembly for verifier terminal check
//!
//! # Paper Reference
//! Section 4.4, Step 4: Terminal identity verification
//!
//! ## Original Paper Formula
//!
//! ```text
//! v ?= eq((α',r'), β)·(F' + Σ_i γ^i·N_i') 
//!      + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
//!
//! where E_{(i,j)} := eq((α',r'), (α,r))·ỹ'_{(i,j)}(α')
//! ```
//!
//! ## Factored Form (Code Implementation)
//!
//! Since eq((α',r'), (α,r)) does NOT depend on indices (i,j), we can factor it out:
//!
//! ```text
//! v ?= eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
//!      + γ^k · eq((α',r'), (α,r)) · [Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ'_{(i,j)}(α')]
//!                                    └─────────────── Eval' ────────────────┘
//! ```
//!
//! **Key:** `Eval'` is the weighted sum of ỹ' values WITHOUT the eq gate.
//! The full Eval block = γ^k · eq((α',r'), (α,r)) · Eval'
//!
//! This is the soundness check that completes the sum-check protocol.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::pi_ccs::transcript::Challenges;
use neo_ccs::{CcsStructure, McsInstance, MeInstance};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Compute eq(p, q) = ∏_i ((1-p_i)(1-q_i) + p_i·q_i)
///
/// Returns an error if the lengths don't match to avoid silently
/// masking wiring bugs by returning zero.
fn eq_points(p: &[K], q: &[K]) -> Result<K, PiCcsError> {
    if p.len() != q.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "eq_points dimension mismatch: {} vs {}",
            p.len(),
            q.len()
        )));
    }
    let mut acc = K::ONE;
    for i in 0..p.len() {
        acc *= (K::ONE - p[i]) * (K::ONE - q[i]) + p[i] * q[i];
    }
    Ok(acc)
}

/// Compute RHS of terminal identity: Q(α', r')
///
/// # Paper Reference
/// Section 4.4, Step 4 (original form):
/// ```text
/// v = eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
///     + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
///
/// where E_{(i,j)} := eq((α',r'), (α,r))·ỹ'_{(i,j)}(α')
/// ```
///
/// This function implements the **factored form**:
/// ```text
/// v = eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
///     + eq((α',r'), (α,r)) · [γ^k · Σ_{j,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ'_{(i,j)}(α')]
/// ```
///
/// Where:
/// - F' = f(y'_{(1,1)}, ..., y'_{(1,t)}) using first output's y_scalars
/// - N_i' = ∏_j (ỹ'_{(i,1)}(α') - j) for j ∈ {-b+1, ..., b-1}
/// - Eval' = Σ_{j,i=|MCS|+1}^{t,k} γ^{i+(j-1)k-1} · ỹ_{(i,j)}(α') (weighted sum WITHOUT eq gate)
/// - The outer γ^k_total multiplies the entire Eval' sum
/// - k_total = |MCS| + |ME inputs|
pub fn rhs_Q_apr(
    s: &CcsStructure<F>,
    ch: &Challenges,
    r_prime: &[K],
    alpha_prime: &[K],
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    out_me: &[MeInstance<Cmt, F, K>],
    params: &neo_params::NeoParams,
) -> Result<K, PiCcsError> {
    // Sanity check: need at least one output for F' and NC'
    if out_me.is_empty() {
        return Err(PiCcsError::InvalidInput("no ME outputs".into()));
    }
    
    let k_total = mcs_list.len() + me_inputs.len();

    // Check dimension consistency for eq gates to avoid silent errors
    if r_prime.len() != ch.beta_r.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "r_prime length {} != beta_r length {}",
            r_prime.len(),
            ch.beta_r.len()
        )));
    }
    if alpha_prime.len() != ch.beta_a.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "alpha_prime length {} != beta_a length {}",
            alpha_prime.len(),
            ch.beta_a.len()
        )));
    }

    // Consistency check: all ME inputs must share the same r
    if !me_inputs.is_empty() {
        let first_r = &me_inputs[0].r;
        if r_prime.len() != first_r.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "r_prime length {} != ME input r length {}",
                r_prime.len(),
                first_r.len()
            )));
        }
        for (idx, mi) in me_inputs.iter().enumerate().skip(1) {
            if mi.r != *first_r {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME input {} has different r than ME input 0 (all ME inputs must share the same r)",
                    idx
                )));
            }
        }
    }

    // Full β gate (row and Ajtai): eq((α',r'), β) = eq_a(α',β_a)·eq_r(r',β_r)
    let eq_beta_r = eq_points(r_prime, &ch.beta_r)?;
    let eq_beta_a = eq_points(alpha_prime, &ch.beta_a)?;
    let eq_aprp_beta = eq_beta_r * eq_beta_a;

    let eq_aprp_ar = if me_inputs.is_empty() {
        K::ZERO
    } else {
        eq_points(alpha_prime, &ch.alpha)? * eq_points(r_prime, &me_inputs[0].r)?
    };

    let me_for_f = &out_me[0];
    // Recompose m_j from base-b digits in y'_{(1,j)} and evaluate f(m_1, ..., m_t)
    let bK = K::from(F::from_u64(params.b as u64));
    let d_len = me_for_f
        .y
        .get(0)
        .ok_or_else(|| PiCcsError::InvalidInput("empty y in out_me[0]".into()))?
        .len();
    for (j, row) in me_for_f.y.iter().enumerate() {
        if row.len() != d_len {
            return Err(PiCcsError::InvalidInput(format!(
                "y row {} length {} != d_len {}",
                j, row.len(), d_len
            )));
        }
    }
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let row = &me_for_f.y[j];
        let mut pow = K::ONE;
        let mut acc = K::ZERO;
        for &yd in row {
            acc += pow * yd;
            pow *= bK;
        }
        m_vals[j] = acc;
    }
    let f_prime = s.f.eval_in_ext::<K>(&m_vals);

    // Compute NC' = Σ_i γ^i · N_i' where
    // N_i' = ∏_{t=-(b-1)}^{b-1} (ỹ'_{(i,1)}(α') - t)
    // Use j=0 (M_1 = I) row of y; evaluate its MLE at α'.
    let chi_alpha_prime: Vec<K> = neo_ccs::utils::tensor_point::<K>(alpha_prime);
    let mut nc_prime = K::ZERO;
    let mut gamma_pow = ch.gamma; // γ^1 for i=1
    for out in out_me {
        // ỹ'_{(i,1)}(α') = ⟨y_{(i,1)}, χ_{α'}⟩
        let y1 = &out.y[0];
        if y1.len() != chi_alpha_prime.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "norm constraint: y row length {} != χ_{{α'}} length {}",
                y1.len(),
                chi_alpha_prime.len()
            )));
        }
        let y_mle: K = y1
            .iter()
            .zip(&chi_alpha_prime)
            .map(|(&y, chi)| y * *chi)
            .sum();

        // Range polynomial: ∏_{t=-(b-1)}^{b-1} (y_mle - t)
        let Ni = crate::pi_ccs::nc_core::range_product::<F>(y_mle, params.b);
        nc_prime += gamma_pow * Ni;
        gamma_pow *= ch.gamma;
    }

    // Compute Eval': weighted sum of ỹ'_{(i,j)}(α') WITHOUT eq gate
    // Eval' = Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ'_{(i,j)}(α')
    //
    // Paper convention: i ∈ {2, ..., k} means outputs[1..k] (skip only output[0] which is i=1).
    // Output 0 is for F' and NC', outputs 1..k are for Eval' block.
    // This is independent of how many are MCS vs ME outputs.
    let mut eval_sum_prime = K::ZERO;

    if !me_inputs.is_empty() {
        // Precompute γ powers for better performance
        let max_exponent = k_total * s.t() + k_total;
        let mut gamma_pows = Vec::with_capacity(max_exponent + 1);
        gamma_pows.push(K::ONE);
        for e in 1..=max_exponent {
            gamma_pows.push(gamma_pows[e - 1] * ch.gamma);
        }

        for j in 0..s.t() {
            // Paper: i ∈ {2, ..., k}, so skip only the first output (i=1)
            for (i_off, out) in out_me.iter().enumerate().skip(1) {
                let y_vec = &out.y[j];
                if y_vec.len() != chi_alpha_prime.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Eval' block: y row length {} != χ_{{α'}} length {}",
                        y_vec.len(),
                        chi_alpha_prime.len()
                    )));
                }
                let y_mle: K = y_vec
                    .iter()
                    .zip(&chi_alpha_prime)
                    .map(|(&y, chi)| y * *chi)
                    .sum();

                // Weight: paper-exact overall uses γ^{(i-1)} · (γ^k)^{j+1}
                // Folded form: exponent = (i-1) + (j+1)*k_total with j starting at 0 here.
                let i_minus_1 = i_off;
                let exponent = i_minus_1 + (j + 1) * k_total;

                let w_pow = gamma_pows[exponent];
                eval_sum_prime += w_pow * y_mle;
            }
        }
    }

    // Final: Q(α',r') = eq·(F' + NC') + eq·Eval' (weights include γ^k effect)
    #[cfg(feature = "debug-logs")]
    {
        use crate::pi_ccs::format_ext;
        eprintln!("[terminal] eq_beta = {}", format_ext(eq_aprp_beta));
        eprintln!("[terminal] eq_ar   = {}", format_ext(eq_aprp_ar));
        eprintln!("[terminal] F'      = {}", format_ext(f_prime));
        eprintln!("[terminal] NC'     = {}", format_ext(nc_prime));
        eprintln!("[terminal] Eval'   = {}", format_ext(eval_sum_prime));
        eprintln!("[terminal] RHS     = {}", format_ext(eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * eval_sum_prime));
    }
    Ok(eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * eval_sum_prime)
}
