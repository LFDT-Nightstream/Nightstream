//! Paper-exact Π-CCS implementation (Section 4.4).
//!
//! Important: This is intentionally inefficient and meant for correctness/reference.
//! It follows the paper literally, with explicit sums/products and full hypercube loops.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{K, D};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::pi_ccs::transcript::Challenges;

/// --- Utilities -------------------------------------------------------------

#[inline]
pub fn eq_points(p: &[K], q: &[K]) -> K {
    assert_eq!(p.len(), q.len(), "eq_points: length mismatch");
    let mut acc = K::ONE;
    for i in 0..p.len() {
        let (pi, qi) = (p[i], q[i]);
        acc *= (K::ONE - pi) * (K::ONE - qi) + pi * qi;
    }
    acc
}

/// χ_{x}(row) where x ∈ {0,1}^{ℓ_n} is a Boolean assignment encoded as a usize.
/// This is the classic product gate, but since x is Boolean we can short-circuit:
/// χ_x(row) = 1 if row's bits equal x's bits; else 0.
#[inline]
pub fn chi_row_at_bool_point(row: usize, xr_mask: usize, _ell_n: usize) -> K {
    if row == xr_mask { K::ONE } else { K::ZERO }
}

/// χ_{x}(ρ) in the Ajtai dimension (Boolean x).
#[inline]
pub fn chi_ajtai_at_bool_point(rho: usize, xa_mask: usize, _ell_d: usize) -> K {
    if rho == xa_mask { K::ONE } else { K::ZERO }
}

/// Convert base-b digits Z (d×m, row-major) back to `z ∈ F^m`, then lift to K.
pub fn recomposed_z_from_Z<Ff>(params: &NeoParams, Z: &Mat<Ff>) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let d = D; // digit rows
    let m = Z.cols();
    let bK = K::from(Ff::from_u64(params.b as u64));

    // Precompute b^ℓ in K
    let mut pow = vec![K::ONE; d];
    for i in 1..d {
        pow[i] = pow[i - 1] * bK;
    }

    let mut z = vec![K::ZERO; m];
    for c in 0..m {
        let mut acc = K::ZERO;
        for rho in 0..d {
            acc += K::from(Z[(rho, c)]) * pow[rho];
        }
        z[c] = acc;
    }
    z
}

/// Range polynomial: ∏_{t=-(b-1)}^{b-1} (val - t).
#[inline]
fn range_product<Ff: Field + PrimeCharacteristicRing>(val: K, b: u32) -> K
where
    K: From<Ff>,
{
    let lo = -((b as i64) - 1);
    let hi =  (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(Ff::from_i64(t));
    }
    prod
}

/// Safe access with zero-padding when indices are outside the true dimension.
/// - For M_j ∈ F^{n×m}: if row ≥ n or col ≥ m → 0.
/// - For Z   ∈ F^{d×m}: if rho ≥ d or col ≥ m → 0.
#[inline]
fn get_F<Ff: Field + PrimeCharacteristicRing + Copy>(a: &Mat<Ff>, row: usize, col: usize) -> Ff {
    if row < a.rows() && col < a.cols() { a[(row, col)] } else { Ff::ZERO }
}

/// --- Core, literal formulas from the paper --------------------------------

/// Evaluate F at the Boolean row assignment xr (as in §4.4):
///   F(X_[log n]) = f( Ẽ(M_1 z_1)(X_r), …, Ẽ(M_t z_1)(X_r) )
///
/// Since X_r ∈ {0,1}^{ℓ_n}, Ẽ(v)(X_r) = v[xr] (row selection).
fn F_at_bool_row<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    Z1: &Mat<Ff>,
    xr_mask: usize,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    // Recompose z_1 from Z_1 and compute (M_j z_1)[row].
    let z1 = recomposed_z_from_Z(params, Z1); // in K
    let mut m_vals = vec![K::ZERO; s.t()];

    for j in 0..s.t() {
        // (M_j z_1)[xr] = Σ_c M_j[xr, c] · z1[c]
        let mut acc = K::ZERO;
        for c in 0..s.m {
            acc += K::from(get_F(&s.matrices[j], xr_mask, c)) * z1[c];
        }
        m_vals[j] = acc;
    }

    s.f.eval_in_ext::<K>(&m_vals)
}

/// Evaluate NC_i at Boolean X=(xa,xr), literally (§4.4):
///   NC_i(X) = ∏_{t=-(b-1)}^{b-1} ( Ẽ(Z_i M_1^T ẑ_r)(X_a) - t )
/// where ẑ_r is χ_{X_r} (here a one-hot row selector since X_r is Boolean),
/// and Ẽ(·)(X_a) reduces to picking the Ajtai row `xa`.
fn NC_i_at_bool_point<Ff>(
    s: &CcsStructure<Ff>,
    Z_i: &Mat<Ff>,
    M1: &Mat<Ff>,
    xa_mask: usize,
    xr_mask: usize,
    b: u32,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    // y_{i,1}(X) at Ajtai row xa is:
    // y_val = Σ_c Z_i[xa, c] · M_1[xr, c]
    let mut y_val = K::ZERO;
    for c in 0..s.m {
        let z = K::from(get_F(Z_i, xa_mask, c));
        let m = K::from(get_F(M1, xr_mask, c));
        y_val += z * m;
    }
    range_product::<Ff>(y_val, b)
}

/// Evaluate Eval_{(i,j)}(X) at Boolean X=(xa,xr) literally (§4.4):
///   Eval_{(i,j)}(X) = eq(X,(α,r)) · Ẽ(Z_i M_j^T χ_{X_r})(X_a)
/// and with Boolean X, Ẽ(·)(X_a) reduces to picking Ajtai row `xa`.
fn Eval_ij_at_bool_point<Ff>(
    s: &CcsStructure<Ff>,
    Z_i: &Mat<Ff>,
    Mj: &Mat<Ff>,
    xa_mask: usize,
    xr_mask: usize,
    alpha: &[K],
    r: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    // eq((α',r'),(α,r)) with X boolean → eq(X_a, α) * eq(X_r, r)
    let eq_ar = {
        let eq_a = {
            // For Boolean xa_mask, eq(xa, α) = ∏_bit ((xa_bit==0)? 1-α_i : α_i)
            let mut prod = K::ONE;
            for (bit, &a_i) in alpha.iter().enumerate() {
                let is_one = ((xa_mask >> bit) & 1) == 1;
                prod *= if is_one { a_i } else { K::ONE - a_i };
            }
            prod
        };
        let eq_r = if let Some(rbits) = r {
            let mut prod = K::ONE;
            for (bit, &r_i) in rbits.iter().enumerate() {
                let is_one = ((xr_mask >> bit) & 1) == 1;
                prod *= if is_one { r_i } else { K::ONE - r_i };
            }
            prod
        } else {
            K::ZERO
        };
        eq_a * eq_r
    };

    // Ẽ(Z_i M_j^T χ_{X_r})(X_a) at Boolean X:
    // ajtai pick: value = Σ_c Z_i[xa, c] · M_j[xr, c]
    let mut y_val = K::ZERO;
    for c in 0..s.m {
        let z = K::from(get_F(Z_i, xa_mask, c));
        let m = K::from(get_F(Mj, xr_mask, c));
        y_val += z * m;
    }

    eq_ar * y_val
}

/// Evaluate the paper's Q(X) at Boolean X=(xa,xr) literally:
///
/// Q(X) = eq(X,β)·( F(X_r) + Σ_{i∈[k]} γ^i·NC_i(X) )
///        + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · Eval_{(i,j)}(X)
///
/// Assumptions:
/// - M_1 = I_n (identity), m = n, and n, d·n are powers of two (per paper).
pub fn q_at_point_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[McsWitness<Ff>], // provides Z_1 for F term and Z_i for NC/Eval
    me_witnesses: &[Mat<Ff>],         // additional Z_i for i≥|MCS|+1
    alpha: &[K],
    beta_a: &[K],
    beta_r: &[K],
    gamma: K,
    r_for_me: Option<&[K]>, // all ME inputs share same r, or None (k=1)
    xa_mask: usize,
    xr_mask: usize,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let k_total = mcs_witnesses.len() + me_witnesses.len();
    let M1 = &s.matrices[0];

    // eq(X, β) = eq(xa, β_a) * eq(xr, β_r) with Boolean X
    let eq_beta = {
        let mut prod_a = K::ONE;
        for (bit, &b_i) in beta_a.iter().enumerate() {
            let is_one = ((xa_mask >> bit) & 1) == 1;
            prod_a *= if is_one { b_i } else { K::ONE - b_i };
        }
        let mut prod_r = K::ONE;
        for (bit, &b_i) in beta_r.iter().enumerate() {
            let is_one = ((xr_mask >> bit) & 1) == 1;
            prod_r *= if is_one { b_i } else { K::ONE - b_i };
        }
        prod_a * prod_r
    };

    // --- F(X_r) term (uses Z_1 only) ---
    let F_term = F_at_bool_row::<Ff>(s, params, &mcs_witnesses[0].Z, xr_mask);

    // --- Σ γ^i · NC_i(X) over all instances (MCS first, then ME) ---
    let mut nc_sum = K::ZERO;
    {
        let mut g = gamma; // γ^1
        // MCS instances
        for w in mcs_witnesses {
            let ni = NC_i_at_bool_point::<Ff>(s, &w.Z, M1, xa_mask, xr_mask, params.b);
            nc_sum += g * ni;
            g *= gamma;
        }
        // ME witnesses
        for Z in me_witnesses {
            let ni = NC_i_at_bool_point::<Ff>(s, Z, M1, xa_mask, xr_mask, params.b);
            nc_sum += g * ni;
            g *= gamma;
        }
    }

    // First part: eq(X, β) * (F + Σ γ^i NC_i)
    let mut acc = eq_beta * (F_term + nc_sum);

    // --- Eval block: γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · Eval_{(i,j)}(X) ---
    if r_for_me.is_some() && k_total >= 2 {
        // Precompute γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= gamma;
        }

        // Accumulate inner sum first
        let mut inner = K::ZERO;
        // Instances are ordered: all MCS first, then ME. The paper uses i∈[2..k].
        // That means we skip the very first instance (i=1).
        for j in 0..s.t() {
            for (i_abs, Zi) in
                mcs_witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()).enumerate().skip(1)
            {
                // Inner weight: γ^{i-1} * (γ^k)^j (0-based j)
                let mut weight = K::ONE;
                // γ^{i-1}
                for _ in 0..i_abs { weight *= gamma; }
                // (γ^k)^j
                for _ in 0..j { weight *= gamma_to_k; }

                let e_ij = Eval_ij_at_bool_point::<Ff>(
                    s, Zi, &s.matrices[j], xa_mask, xr_mask, alpha, r_for_me
                );
                inner += weight * e_ij;
            }
        }
        // Paper-exact: multiply entire Eval block by outer γ^k
        acc += gamma_to_k * inner;
    }

    acc
}

/// Brute-force hypercube sum: ∑_{X∈{0,1}^{ℓ_d+ℓ_n}} Q(X).
///
/// This is the literal “claimed sum” the SumCheck proves.
/// It requires no precomputations and is O(2^{ℓ_d+ℓ_n} · t · k · m).
pub fn sum_q_over_hypercube_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[McsWitness<Ff>],
    me_witnesses: &[Mat<Ff>],
    ch: &Challenges,
    ell_d: usize,
    ell_n: usize,
    r_for_me: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut total = K::ZERO;
    let d_sz = 1usize << ell_d;
    let n_sz = 1usize << ell_n;

    for xa in 0..d_sz {
        for xr in 0..n_sz {
            total += q_at_point_paper_exact(
                s, params, mcs_witnesses, me_witnesses,
                &ch.alpha, &ch.beta_a, &ch.beta_r, ch.gamma, r_for_me,
                xa, xr
            );
        }
    }
    total
}

/// Evaluate Q at an arbitrary extension point (α', r') directly from witnesses.
///
/// Mirrors the paper's Step 4 LHS using the literal definitions (no factoring),
/// without using the prover outputs. This is useful for testing that the RHS built
/// from outputs matches the true Q(α', r') defined over the witnesses.
pub fn q_eval_at_ext_point_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[McsWitness<Ff>],
    me_witnesses: &[Mat<Ff>],
    ch: &Challenges,
    alpha_prime: &[K],
    r_prime: &[K],
    me_inputs_r_opt: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    // Dimensions
    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;

    // Build χ tables for α' and r'
    let d_sz = 1usize << ell_d;
    let n_sz = 1usize << ell_n;

    let mut chi_a = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for bit in 0..alpha_prime.len() {
            let a = alpha_prime[bit];
            let is_one = ((rho >> bit) & 1) == 1;
            w *= if is_one { a } else { K::ONE - a };
        }
        chi_a[rho] = w;
    }

    let mut chi_r = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for bit in 0..r_prime.len() {
            let r = r_prime[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { r } else { K::ONE - r };
        }
        chi_r[row] = w;
    }

    // eq gates
    let eq_beta = eq_points(alpha_prime, &ch.beta_a) * eq_points(r_prime, &ch.beta_r);
    let eq_ar = if let Some(r) = me_inputs_r_opt {
        eq_points(alpha_prime, &ch.alpha) * eq_points(r_prime, r)
    } else { K::ZERO };

    // F' direct: evaluate Ẽ(M_j z_1)(r') for j=0..t-1 then f(...)
    let z1 = recomposed_z_from_Z::<Ff>(params, &mcs_witnesses[0].Z);
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        // y_row[r] = (M_j z_1)[r] = Σ_c M_j[r,c]·z1[c]
        // then Ẽ(y_row)(r') = Σ_row χ_r[row] · y_row[row]
        let mut y_eval = K::ZERO;
        for row in 0..n_sz {
            let wr = if row < s.n { chi_r[row] } else { K::ZERO };
            if wr == K::ZERO { continue; }
            let mut y_row = K::ZERO;
            for c in 0..s.m {
                y_row += K::from(get_F(&s.matrices[j], row, c)) * z1[c];
            }
            y_eval += wr * y_row;
        }
        m_vals[j] = y_eval;
    }
    let F_prime = s.f.eval_in_ext::<K>(&m_vals);

    // v1 = M_1^T χ_{r'} ∈ K^m
    let mut v1 = vec![K::ZERO; s.m];
    for row in 0..n_sz {
        let wr = if row < s.n { chi_r[row] } else { K::ZERO };
        if wr == K::ZERO { continue; }
        for c in 0..s.m {
            v1[c] += K::from(get_F(&s.matrices[0], row, c)) * wr;
        }
    }

    // Σ γ^i · N_i'
    let mut nc_sum = K::ZERO;
    {
        let mut g = ch.gamma; // γ^1
        for w in mcs_witnesses {
            // y_digits[rho] = Σ_c Z_i[rho, c] · v1[c]
            let mut y_digits = vec![K::ZERO; D];
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += K::from(w.Z[(rho, c)]) * v1[c];
                }
                y_digits[rho] = acc;
            }
            // ẏ'_{(i,1)}(α') = Σ_ρ y_digits[ρ] · χ_{α'}[ρ]
            let mut y_eval = K::ZERO;
            for rho in 0..core::cmp::min(D, d_sz) {
                y_eval += y_digits[rho] * chi_a[rho];
            }
            nc_sum += g * range_product::<Ff>(y_eval, params.b);
            g *= ch.gamma;
        }
        for Z in me_witnesses {
            let mut y_digits = vec![K::ZERO; D];
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += K::from(Z[(rho, c)]) * v1[c];
                }
                y_digits[rho] = acc;
            }
            let mut y_eval = K::ZERO;
            for rho in 0..core::cmp::min(D, d_sz) {
                y_eval += y_digits[rho] * chi_a[rho];
            }
            nc_sum += g * range_product::<Ff>(y_eval, params.b);
            g *= ch.gamma;
        }
    }

    // Eval block (only if k≥2 and have me_inputs_r)
    let mut eval_sum = K::ZERO;
    let k_total = mcs_witnesses.len() + me_witnesses.len();
    if me_inputs_r_opt.is_some() && k_total >= 2 {
        // Precompute γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= ch.gamma; }

        // For each j,i (skip i=1 instance)
        for j in 0..s.t() {
            for (i_abs, Zi) in mcs_witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()).enumerate().skip(1) {
                // vj = M_j^T χ_{r'}
                let mut vj = vec![K::ZERO; s.m];
                for row in 0..n_sz {
                    let wr = if row < s.n { chi_r[row] } else { K::ZERO };
                    if wr == K::ZERO { continue; }
                    for c in 0..s.m {
                        vj[c] += K::from(get_F(&s.matrices[j], row, c)) * wr;
                    }
                }
                // y_digits[rho] = Σ_c Z_i[rho,c] · vj[c]
                let mut y_digits = vec![K::ZERO; D];
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for c in 0..s.m { acc += K::from(Zi[(rho, c)]) * vj[c]; }
                    y_digits[rho] = acc;
                }
                // Ẽ(...)(α')
                let mut y_eval = K::ZERO;
                for rho in 0..core::cmp::min(D, d_sz) { y_eval += y_digits[rho] * chi_a[rho]; }

                // Inner weight: γ^{i-1} * (γ^k)^j (0-based j)
                let mut weight = K::ONE;
                for _ in 0..i_abs { weight *= ch.gamma; }
                for _ in 0..j { weight *= gamma_to_k; }

                eval_sum += weight * y_eval;
            }
        }
    }

    eq_beta * (F_prime + nc_sum) + eq_ar * (if me_inputs_r_opt.is_some() && k_total >= 2 {
        // Paper-exact: multiply entire Eval block by outer γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= ch.gamma; }
        gamma_to_k * eval_sum
    } else {
        eval_sum
    })
}

/// --- Terminal identity (Step 4) -------------------------------------------
///
/// The original paper formula (no factoring):
///
/// v ?= eq((α',r'), β)·(F' + Σ_i γ^i·N_i')
///      + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
///
/// with:
///   E_{(i,j)} := eq((α',r'), (α,r))·ẏ'_{(i,j)}(α')
///
/// Where:
///   - F' uses y' of the first output (i=1) to reconstruct m_j and f.
///   - N_i' = ∏_{t=-(b-1)}^{b-1} ( ẏ'_{(i,1)}(α') - t )
pub fn rhs_terminal_identity_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    ch: &Challenges,          // contains (α, β, γ)
    r_prime: &[K],
    alpha_prime: &[K],
    out_me: &[MeInstance<Cmt, Ff, K>], // outputs y' (i ∈ [k], j ∈ [t])
    me_inputs_r_opt: Option<&[K]>,     // r from inputs, required if k>1
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!out_me.is_empty(), "terminal: need at least one output");
    let k_total = out_me.len();

    // eq((α',r'), β) = eq(α', β_a) * eq(r', β_r)
    let eq_aprp_beta = {
        let e1 = eq_points(alpha_prime, &ch.beta_a);
        let e2 = eq_points(r_prime,     &ch.beta_r);
        e1 * e2
    };

    // eq((α',r'),(α,r)) if we have ME inputs; else 0 (so the Eval block vanishes).
    let eq_aprp_ar = if let Some(r) = me_inputs_r_opt {
        eq_points(alpha_prime, &ch.alpha) * eq_points(r_prime, r)
    } else {
        K::ZERO
    };

    // --- F' ---
    // Recompose m_j from y'_{(1,j)} using base-b digits (Ajtai rows) and evaluate f.
    let bK = K::from(Ff::from_u64(params.b as u64));
    let mut m_vals = vec![K::ZERO; s.t()];
    {
        let y_first = &out_me[0];
        for j in 0..s.t() {
            let row = &y_first.y[j]; // K^d (padded)
            let mut acc = K::ZERO;
            let mut pow = K::ONE;
            for rho in 0..D {
                acc += pow * row.get(rho).copied().unwrap_or(K::ZERO);
                pow *= bK;
            }
            m_vals[j] = acc;
        }
    }
    let F_prime = s.f.eval_in_ext::<K>(&m_vals);

    // --- Σ γ^i · N_i' ---
    // N_i' = ∏_{t} ( ẏ'_{(i,1)}(α') - t ), with ẏ' evaluated at α' as MLE:
    //        ẏ'_{(i,1)}(α') = ⟨ y'_{(i,1)}, χ_{α'} ⟩.
    let chi_alpha_prime = {
        // Build χ_{α'} over Ajtai domain by tensoring the bits explicitly.
        let d_sz = 1usize << alpha_prime.len();
        let mut tbl = vec![K::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut w = K::ONE;
            for bit in 0..alpha_prime.len() {
                let a = alpha_prime[bit];
                let bit_is_one = ((rho >> bit) & 1) == 1;
                w *= if bit_is_one { a } else { K::ONE - a };
            }
            tbl[rho] = w;
        }
        tbl
    };

    let mut nc_prime_sum = K::ZERO;
    {
        let mut g = ch.gamma; // γ^1
        for out in out_me {
            // ẏ'_{(i,1)}(α') = Σ_ρ y'_{(i,1)}[ρ] · χ_{α'}[ρ]
            let y1 = &out.y[0];
            let limit = core::cmp::min(chi_alpha_prime.len(), y1.len());
            let mut y_eval = K::ZERO;
            for rho in 0..limit {
                y_eval += y1[rho] * chi_alpha_prime[rho];
            }
            let Ni = range_product::<Ff>(y_eval, params.b);
            nc_prime_sum += g * Ni;
            g *= ch.gamma;
        }
    }

    // --- Eval' block ---
    // γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)} with
    // E_{(i,j)} = eq((α',r'),(α,r)) · ẏ'_{(i,j)}(α').
    let mut eval_sum = K::ZERO;
    if me_inputs_r_opt.is_some() && k_total >= 2 {
        // Precompute γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= ch.gamma; }

        for j in 0..s.t() {
            for (i_abs, out) in out_me.iter().enumerate().skip(1) {
                // ẏ'_{(i,j)}(α') = Σ_ρ y'_{(i,j)}[ρ] · χ_{α'}[ρ]
                let y = &out.y[j];
                let mut y_eval = K::ZERO;
                let limit = core::cmp::min(chi_alpha_prime.len(), y.len());
                for rho in 0..limit {
                    y_eval += y[rho] * chi_alpha_prime[rho];
                }

                // Inner weight: γ^{i-1} * (γ^k)^j (0-based i_abs, 0-based j)
                let mut weight = K::ONE;
                for _ in 0..i_abs { weight *= ch.gamma; } // γ^{i-1}
                for _ in 0..j     { weight *= gamma_to_k; } // (γ^k)^j

                eval_sum += weight * y_eval;
            }
        }
    }

    // Assemble RHS exactly like the paper:
    // v = eq((α',r'), β)·(F' + Σ γ^i N_i') + γ^k · eq((α',r'), (α,r)) · [Σ ...]
    eq_aprp_beta * (F_prime + nc_prime_sum) + eq_aprp_ar * (if me_inputs_r_opt.is_some() && k_total >= 2 {
        // Paper-exact: multiply entire Eval block by outer γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= ch.gamma; }
        gamma_to_k * eval_sum
    } else {
        eval_sum
    })
}

/// --- Step 3 outputs, literal form -----------------------------------------
///
/// For each i ∈ [k] and j ∈ [t], send:
///   y'_{(i,j)} := Z_i · M_j^T · ẑ_{r'}  ∈ K^d
///
/// where ẑ_{r'} is χ_{r'} over {0,1}^{ℓ_n}, i.e., the row-table weights.
/// This function builds those outputs exactly by literal dense loops.
///
/// Notes:
/// - First `insts.len()` outputs correspond to MCS instances (`mcs_list` order).
/// - Next `me_witnesses.len()` outputs correspond to ME inputs in order.
/// - Each y[j] is padded to 2^{ℓ_d}.
pub fn build_me_outputs_paper_exact<Ff, L>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_list: &[McsInstance<Cmt, Ff>],
    witnesses: &[McsWitness<Ff>],
    me_inputs: &[MeInstance<Cmt, Ff, K>],
    me_witnesses: &[Mat<Ff>],
    r_prime: &[K],
    ell_d: usize,
    fold_digest: [u8; 32],
    l: &L,
) -> Vec<MeInstance<Cmt, Ff, K>>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
    L: neo_ccs::traits::SModuleHomomorphism<Ff, Cmt>,
{
    // Build χ_{r'}(row) table literally.
    let n_sz = 1usize << r_prime.len();
    let mut chi_rp = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for bit in 0..r_prime.len() {
            let rb = r_prime[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { rb } else { K::ONE - rb };
        }
        chi_rp[row] = w;
    }

    // v_j := M_j^T · χ_{r'} ∈ K^m, computed with literal nested loops.
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        for row in 0..n_sz {
            let wr = if row < s.n { chi_rp[row] } else { K::ZERO };
            if wr == K::ZERO { continue; }
            for c in 0..s.m {
                vj[c] += K::from(get_F(&s.matrices[j], row, c)) * wr;
            }
        }
        vjs.push(vj);
    }

    // Pad helper
    let pad_to_pow2 = |mut y: Vec<K>| -> Vec<K> {
        let want = 1usize << ell_d;
        y.resize(want, K::ZERO);
        y
    };

    let base_f = K::from(Ff::from_u64(params.b as u64));
    let mut pow_cache = vec![K::ONE; D];
    for i in 1..D { pow_cache[i] = pow_cache[i - 1] * base_f; }
    let recompose = |y: &[K]| -> K {
        let mut acc = K::ZERO;
        for (rho, &v) in y.iter().enumerate().take(D) {
            acc += v * pow_cache[rho];
        }
        acc
    };

    let mut out = Vec::with_capacity(witnesses.len() + me_witnesses.len());

    // MCS outputs (keep order)
    for (inst, wit) in mcs_list.iter().zip(witnesses.iter()) {
        let X = l.project_x(&wit.Z, inst.m_in);

        let mut y = Vec::with_capacity(s.t());
        // For each j, y_j = Z · v_j
        for vj in &vjs {
            let mut yj = vec![K::ZERO; D];
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += K::from(wit.Z[(rho, c)]) * vj[c];
                }
                yj[rho] = acc;
            }
            y.push(pad_to_pow2(yj));
        }
        let y_scalars: Vec<K> = y.iter().map(|yj| recompose(yj)).collect();

        out.push(MeInstance {
            c_step_coords: vec![], u_offset: 0, u_len: 0,
            c: inst.c.clone(),
            X,
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inst.m_in,
            fold_digest,
        });
    }

    // ME outputs (keep order)
    for (inp, Zi) in me_inputs.iter().zip(me_witnesses.iter()) {
        let mut y = Vec::with_capacity(s.t());
        for vj in &vjs {
            let mut yj = vec![K::ZERO; D];
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += K::from(Zi[(rho, c)]) * vj[c];
                }
                yj[rho] = acc;
            }
            y.push(pad_to_pow2(yj));
        }
        let y_scalars: Vec<K> = y.iter().map(|yj| recompose(yj)).collect();

        out.push(MeInstance {
            c_step_coords: vec![], u_offset: 0, u_len: 0,
            c: inp.c.clone(),
            X: inp.X.clone(),
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inp.m_in,
            fold_digest,
        });
    }

    out
}
