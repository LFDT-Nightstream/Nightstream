//! Terminal-identity formulas used by the optimized verifier.
//!
//! These are kept in `optimized_engine` so optimized verify does not depend on
//! `paper_exact_engine` module paths, while remaining formula-equivalent.

#![allow(non_snake_case)]

use crate::Challenges;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::K;
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing};

use super::common::eq_points;

#[inline]
fn range_product<Ff: Field + PrimeCharacteristicRing>(val: K, b: u32) -> K
where
    K: From<Ff>,
{
    let lo = -((b as i64) - 1);
    let hi = (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(Ff::from_i64(t));
    }
    prod
}

/// FE-only terminal identity (Step 4), i.e. the paper RHS with NC removed.
pub fn rhs_terminal_identity_fe_with_k_mcs<Ff>(
    s: &CcsStructure<Ff>,
    _params: &NeoParams,
    ch: &Challenges,
    r_prime: &[K],
    alpha_prime: &[K],
    out_me: &[CeClaim<Cmt, Ff, K>],
    k_mcs: usize,
    me_inputs_r_opt: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!out_me.is_empty(), "terminal: need at least one output");
    let k_total = out_me.len();
    assert!(k_mcs > 0 && k_mcs <= k_total, "terminal: invalid k_mcs");

    let eq_aprp_beta = eq_points(alpha_prime, &ch.beta_a) * eq_points(r_prime, &ch.beta_r);
    let eq_aprp_ar = if k_total > k_mcs {
        let r = me_inputs_r_opt.expect("terminal FE: missing shared ME input r");
        eq_points(alpha_prime, &ch.alpha) * eq_points(r_prime, r)
    } else {
        K::ZERO
    };

    let F_prime = {
        let mut acc_f = K::ZERO;
        let mut g = K::ONE; // γ^{i-1} with 1-based i
        for out in out_me.iter().take(k_mcs) {
            let mut m_vals = vec![K::ZERO; s.t()];
            for j in 0..s.t() {
                m_vals[j] = *out.y_ring[j]
                    .first()
                    .expect("terminal FE: y_ring row missing constant term");
            }
            acc_f += g * s.f.eval_in_ext::<K>(&m_vals);
            g *= ch.gamma;
        }
        acc_f
    };

    let d_sz = 1usize << alpha_prime.len();
    let mut chi_alpha_prime = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for bit in 0..alpha_prime.len() {
            let a = alpha_prime[bit];
            let bit_is_one = ((rho >> bit) & 1) == 1;
            w *= if bit_is_one { a } else { K::ONE - a };
        }
        chi_alpha_prime[rho] = w;
    }

    let mut eval_sum = K::ZERO;
    if k_total > k_mcs {
        let _ = me_inputs_r_opt.expect("terminal FE: missing shared ME input r");
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        for j in 0..s.t() {
            for (i_abs, out) in out_me.iter().enumerate().skip(k_mcs) {
                let y = &out.y_ring[j];
                let mut y_eval = K::ZERO;
                for rho in 0..chi_alpha_prime.len() {
                    y_eval += y.get(rho).copied().unwrap_or(K::ZERO) * chi_alpha_prime[rho];
                }

                let mut weight = K::ONE;
                for _ in 0..i_abs {
                    weight *= ch.gamma;
                }
                for _ in 0..j {
                    weight *= gamma_to_k;
                }

                eval_sum += weight * y_eval;
            }
        }
    }

    let mut gamma_to_k_outer = K::ONE;
    for _ in 0..k_total {
        gamma_to_k_outer *= ch.gamma;
    }

    eq_aprp_beta * F_prime + eq_aprp_ar * (gamma_to_k_outer * eval_sum)
}

#[inline]
pub fn rhs_terminal_identity_fe<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    ch: &Challenges,
    r_prime: &[K],
    alpha_prime: &[K],
    out_me: &[CeClaim<Cmt, Ff, K>],
    me_inputs_r_opt: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    rhs_terminal_identity_fe_with_k_mcs(s, params, ch, r_prime, alpha_prime, out_me, 1, me_inputs_r_opt)
}

/// NC-only terminal identity (Step 4) for split-NC verification.
pub fn rhs_terminal_identity_nc<Ff>(
    params: &NeoParams,
    ch: &Challenges,
    s_col_prime: &[K],
    alpha_prime: &[K],
    out_me: &[CeClaim<Cmt, Ff, K>],
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!out_me.is_empty(), "terminal: need at least one output");

    let eq_apsp_beta = eq_points(alpha_prime, &ch.beta_a) * eq_points(s_col_prime, &ch.beta_m);

    let d_sz = 1usize << alpha_prime.len();
    let mut chi_alpha_prime = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for bit in 0..alpha_prime.len() {
            let a = alpha_prime[bit];
            let bit_is_one = ((rho >> bit) & 1) == 1;
            w *= if bit_is_one { a } else { K::ONE - a };
        }
        chi_alpha_prime[rho] = w;
    }

    let mut nc_prime_sum = K::ZERO;
    {
        let mut g = ch.gamma; // γ^1
        for out in out_me {
            debug_assert!(out.s_col.is_empty() || out.s_col.as_slice() == s_col_prime);

            let y = &out.y_zcol;
            assert!(
                y.len() >= chi_alpha_prime.len(),
                "terminal NC: y_zcol too short for chi(alpha') (len {} < {})",
                y.len(),
                chi_alpha_prime.len()
            );
            let mut y_eval = K::ZERO;
            for rho in 0..chi_alpha_prime.len() {
                y_eval += y[rho] * chi_alpha_prime[rho];
            }
            let Ni = range_product::<Ff>(y_eval, params.b);
            nc_prime_sum += g * Ni;
            g *= ch.gamma;
        }
    }

    eq_apsp_beta * nc_prime_sum
}
