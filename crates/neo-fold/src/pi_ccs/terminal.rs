//! Terminal module: RHS assembly for verifier terminal check
//!
//! # Paper Reference
//! Section 4.4, Step 4: Terminal identity verification
//!
//! v ?= Q(α', r') = eq((α',r'), β)·(F' + Σ_i γ^i·N_i') + eq((α',r'), (α,r))·Eval'
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
fn eq_points(p: &[K], q: &[K]) -> K {
    if p.len() != q.len() {
        return K::ZERO;
    }
    let mut acc = K::ONE;
    for i in 0..p.len() {
        acc *= (K::ONE - p[i]) * (K::ONE - q[i]) + p[i] * q[i];
    }
    acc
}

/// Compute RHS of terminal identity: Q(α', r')
///
/// # Paper Reference
/// Section 4.4, Step 4:
/// ```text
/// v = eq((α',r'), β)·(F' + Σ_i γ^i·N_i') + eq((α',r'), (α,r))·Eval'
/// ```
///
/// Where:
/// - F' = f(y'_{(1,1)}, ..., y'_{(1,t)}) using first output's y_scalars
/// - N_i' = ∏_j (ỹ'_{(i,1)}(α') - j) for j ∈ {-b+1, ..., b-1} (Option B: set to 0)
/// - Eval' = Σ_{j,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ_{(i,j)}(α') from ME inputs
pub fn rhs_Q_apr(
    s: &CcsStructure<F>,
    ch: &Challenges,
    r_prime: &[K],
    alpha_prime: &[K],
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    out_me: &[MeInstance<Cmt, F, K>],
) -> Result<K, PiCcsError> {
    let eq_beta_r = eq_points(r_prime, &ch.beta_r);
    let eq_beta_a = eq_points(alpha_prime, &ch.beta_a);
    let eq_aprp_beta = eq_beta_a * eq_beta_r;

    let eq_aprp_ar = if me_inputs.is_empty() {
        K::ZERO
    } else {
        eq_points(alpha_prime, &ch.alpha) * eq_points(r_prime, &me_inputs[0].r)
    };

    let me_for_f = out_me
        .first()
        .ok_or_else(|| PiCcsError::InvalidInput("no ME outputs".into()))?;

    if me_for_f.y_scalars.len() != s.t() {
        return Err(PiCcsError::InvalidInput(format!(
            "y_scalars length {} != t={}",
            me_for_f.y_scalars.len(),
            s.t()
        )));
    }

    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        m_vals[j] = me_for_f.y_scalars[j];
    }

    let f_prime = s.f.eval_in_ext::<K>(&m_vals);

    let nc_prime = K::ZERO;

    let chi_alpha_prime: Vec<K> = neo_ccs::utils::tensor_point::<K>(alpha_prime);

    let mut eval_sum_prime = K::ZERO;
    let k_total = mcs_list.len() + me_inputs.len();

    for j in 0..s.t() {
        for (i_off, inp) in me_inputs.iter().enumerate() {
            let y_vec = &inp.y[j];
            let mut y_mle = K::ZERO;
            for (idx, &val) in y_vec.iter().enumerate() {
                if idx < chi_alpha_prime.len() {
                    y_mle += val * chi_alpha_prime[idx];
                }
            }

            let exponent = (i_off + 2) + j * k_total - 1;
            let mut w_pow = K::ONE;
            for _ in 0..exponent {
                w_pow *= ch.gamma;
            }

            eval_sum_prime += w_pow * y_mle;
        }
    }

    Ok(eq_aprp_beta * (f_prime + nc_prime) + eq_aprp_ar * eval_sum_prime)
}

