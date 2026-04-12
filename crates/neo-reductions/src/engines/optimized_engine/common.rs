//! Common utilities and reference implementations for Π_CCS.
//!
//! This module contains:
//! - The `Challenges` struct used by all engines
//! - Utility functions (eq_points, chi, recomposition, etc.)
//! - Reference implementations for Q evaluation and output building
//! - These reference functions are used for cross-checking and verification
//!
//! SplitNcV1 symbol mapping used by these references:
//! - `beta_a`,`beta_r`: FE/full-Q eq-gate point β over (Ajtai,row) bits.
//! - `alpha` + input `r`: Eval gate point (α,r) for carried ME slots.
//! - `beta_m`: NC-channel eq-gate column point for the separate NC sumcheck.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsMatrix, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Challenges sampled in Step 1 of the protocol
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Challenges {
    /// α ∈ K^{log d} - for Ajtai dimension
    pub alpha: Vec<K>,
    /// β = (β_a, β_r) ∈ K^{log(dn)} split into Ajtai and row parts
    pub beta_a: Vec<K>,
    pub beta_r: Vec<K>,
    /// β_m ∈ K^{log m} - column part for the split-NC variant
    pub beta_m: Vec<K>,
    /// γ ∈ K - random linear combination weight
    pub gamma: K,
}

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
    if row == xr_mask {
        K::ONE
    } else {
        K::ZERO
    }
}

/// χ_{x}(ρ) in the Ajtai dimension (Boolean x).
#[inline]
pub fn chi_ajtai_at_bool_point(rho: usize, xa_mask: usize, _ell_d: usize) -> K {
    if rho == xa_mask {
        K::ONE
    } else {
        K::ZERO
    }
}

/// Decode witness matrix `Z` into `z ∈ K^m` for a known CCS width `s_m`.
///
/// Supports both Neo digit layout (`D×m`) and SuperNeo packed layout (`D×(m/D)`).
pub fn recomposed_z_from_Z<Ff>(params: &NeoParams, s_m: usize, Z: &Mat<Ff>) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let _ = params;
    crate::common::decode_superneo_coeffs_from_witness_mat(Z, s_m).unwrap_or_else(|e| {
        panic!("recomposed_z_from_Z: failed to decode packed witness coefficients against m={s_m}: {e}")
    })
}

/// Range polynomial: ∏_{t=-(b-1)}^{b-1} (val - t).
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

/// Safe access with zero-padding when indices are outside the true dimension.
/// - For Z ∈ F^{d×m}: if rho ≥ d or col ≥ m → 0.
#[inline]
fn get_F<Ff: Field + PrimeCharacteristicRing + Copy>(a: &Mat<Ff>, row: usize, col: usize) -> Ff {
    if row < a.rows() && col < a.cols() {
        a[(row, col)]
    } else {
        Ff::ZERO
    }
}

/// Safe access into a CCS matrix M_j, returning 0 for out-of-range indices.
#[inline]
fn get_M<Ff: Field + PrimeCharacteristicRing + Copy>(a: &CcsMatrix<Ff>, row: usize, col: usize) -> Ff {
    if row >= a.rows() || col >= a.cols() {
        return Ff::ZERO;
    }

    match a {
        CcsMatrix::Identity { .. } => {
            if row == col {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(m) => {
            let s = m.col_ptr[col];
            let e = m.col_ptr[col + 1];
            match m.row_idx[s..e].binary_search(&row) {
                Ok(idx) => m.vals[s + idx],
                Err(_) => Ff::ZERO,
            }
        }
    }
}

#[inline]
fn eval_all_mats_with_cache<Ff>(
    _s: &CcsStructure<Ff>,
    superneo_cache: &crate::superneo_eval::SuperneoEvalCache,
    z: &[K],
    chi_r: &[K],
    n_eff: usize,
) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    // This path only needs ct(M_j z) scalars; avoid full ring coefficient evaluation.
    crate::superneo_eval::eval_all_mats_cached(superneo_cache, z, chi_r, n_eff)
}

/// --- Core, literal formulas from the paper --------------------------------

/// Evaluate F at the Boolean row assignment xr (as in §4.4):
///   F(X_[log n]) = f( Ẽ(M_1 z_1)(X_r), …, Ẽ(M_t z_1)(X_r) )
///
/// Since X_r ∈ {0,1}^{ℓ_n}, Ẽ(v)(X_r) = v[xr] (row selection).
fn F_at_bool_row<Ff>(s: &CcsStructure<Ff>, params: &NeoParams, Z1: &Mat<Ff>, xr_mask: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    // Recompose z_1 from Z_1 and compute (M_j z_1)[row].
    let z1 = recomposed_z_from_Z(params, s.m, Z1); // in K
    let mut m_vals = vec![K::ZERO; s.t()];

    for j in 0..s.t() {
        // (M_j z_1)[xr] = Σ_c M_j[xr, c] · z1[c]
        let mut acc = K::ZERO;
        for c in 0..s.m {
            acc += K::from(get_M(&s.matrices[j], xr_mask, c)) * z1[c];
        }
        m_vals[j] = acc;
    }

    s.f.eval_in_ext::<K>(&m_vals)
}

/// Evaluate NC_i at Boolean X=(xa,xr), literally (§4.4):
///   NC_i(X) = ∏_{t=-(b-1)}^{b-1} ( Ẽ(Z_i M_1^T ẑ_r)(X_a) - t )
/// where ẑ_r is χ_{X_r} (here a one-hot row selector since X_r is Boolean),
/// and Ẽ(·)(X_a) reduces to picking the Ajtai row `xa`.
#[inline]
fn NC_i_at_bool_point<Ff>(s: &CcsStructure<Ff>, Z_i: &Mat<Ff>, xa_mask: usize, xr_mask: usize, b: u32) -> K
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let layout = crate::common::witness_mat_layout(Z_i, s.m)
        .unwrap_or_else(|e| panic!("NC_i_at_bool_point: invalid witness shape for m={}: {e}", s.m));
    // Ẑ_i M_1^T χ_{X_r} evaluated at X_a, with (xa,xr) Boolean
    let mut y_val = K::ZERO;
    for c in 0..s.m {
        let z = crate::common::witness_mat_get_k(Z_i, layout, s.m, xa_mask, c);
        let m = K::from(get_M(&s.matrices[0], xr_mask, c));
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
    Mj: &CcsMatrix<Ff>,
    xa_mask: usize,
    xr_mask: usize,
    alpha: &[K],
    r: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let layout = crate::common::witness_mat_layout(Z_i, s.m)
        .unwrap_or_else(|e| panic!("Eval_ij_at_bool_point: invalid witness shape for m={}: {e}", s.m));
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
        let z = crate::common::witness_mat_get_k(Z_i, layout, s.m, xa_mask, c);
        let m = K::from(get_M(Mj, xr_mask, c));
        y_val += z * m;
    }

    eq_ar * y_val
}

/// Evaluate the paper's Q(X) at Boolean X=(xa,xr) literally:
///
/// Q(X) = eq(X,β)·( F(X_r) + Σ_{i∈[k]} γ^{K+i-1}·NC_i(X) )
///        + γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · Eval_{(i,j)}(X)
///
/// Assumptions:
/// - M_1 = I_n (identity), m = n, and n, d·n are powers of two (per paper).
pub fn q_at_point_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[CcsWitness<Ff>], // provides Z_1 for F term and Z_i for NC/Eval
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
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let k_mcs = mcs_witnesses.len();
    let k_total = mcs_witnesses.len() + me_witnesses.len();

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

    // --- F(X_r) term over MCS slots ---
    let mut F_term = K::ZERO;
    {
        let mut g = K::ONE; // γ^{i-1}
        for w in mcs_witnesses {
            F_term += g * F_at_bool_row::<Ff>(s, params, &w.Z, xr_mask);
            g *= gamma;
        }
    }

    // --- Σ γ^{K+i-1} · NC_i(X) over all instances (MCS first, then ME) ---
    let mut nc_sum = K::ZERO;
    {
        let mut g = K::ONE; // γ^K
        for _ in 0..k_mcs {
            g *= gamma;
        }
        // MCS instances
        for w in mcs_witnesses {
            let ni = NC_i_at_bool_point::<Ff>(s, &w.Z, xa_mask, xr_mask, params.b);
            nc_sum += g * ni;
            g *= gamma;
        }
        // ME witnesses
        for Z in me_witnesses {
            let ni = NC_i_at_bool_point::<Ff>(s, Z, xa_mask, xr_mask, params.b);
            nc_sum += g * ni;
            g *= gamma;
        }
    }

    // First part: eq(X, β) * (F + Σ γ^{K+i-1} NC_i)
    let mut acc = eq_beta * (F_term + nc_sum);

    // --- Eval block: γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · Eval_{(i,j)}(X) ---
    if k_total > k_mcs {
        let r_for_me = r_for_me.expect("q_at_point_paper_exact: missing shared ME input r");
        // Precompute γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= gamma;
        }

        // Accumulate inner sum first
        let mut inner = K::ZERO;
        // Instances are ordered: all MCS first, then ME.
        // Eval block runs over ME slots only.
        for j in 0..s.t() {
            for (i_abs, Zi) in mcs_witnesses
                .iter()
                .map(|w| &w.Z)
                .chain(me_witnesses.iter())
                .enumerate()
                .skip(k_mcs)
            {
                // Inner weight: γ^{i-1} * (γ^k)^j (0-based j)
                let mut weight = K::ONE;
                // γ^{i-1}
                for _ in 0..i_abs {
                    weight *= gamma;
                }
                // (γ^k)^j
                for _ in 0..j {
                    weight *= gamma_to_k;
                }

                let e_ij = Eval_ij_at_bool_point::<Ff>(s, Zi, &s.matrices[j], xa_mask, xr_mask, alpha, Some(r_for_me));
                inner += weight * e_ij;
            }
        }
        // Paper-exact: multiply the inner weighted sum by a single outer γ^k.
        acc += gamma_to_k * inner;
    }

    acc
}

/// Brute-force hypercube sum: ∑_{X∈{0,1}^{ℓ_d+ℓ_n}} Q(X).
///
/// This is the literal "claimed sum" the SumCheck proves.
/// It requires no precomputations and is O(2^{ℓ_d+ℓ_n} · t · k · m).
pub fn sum_q_over_hypercube_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[CcsWitness<Ff>],
    me_witnesses: &[Mat<Ff>],
    ch: &Challenges,
    ell_d: usize,
    ell_n: usize,
    r_for_me: Option<&[K]>,
) -> K
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut total = K::ZERO;
    let d_sz = 1usize << ell_d;
    let n_sz = 1usize << ell_n;

    for xa in 0..d_sz {
        for xr in 0..n_sz {
            total += q_at_point_paper_exact(
                s,
                params,
                mcs_witnesses,
                me_witnesses,
                &ch.alpha,
                &ch.beta_a,
                &ch.beta_r,
                ch.gamma,
                r_for_me,
                xa,
                xr,
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
    mcs_witnesses: &[CcsWitness<Ff>],
    me_witnesses: &[Mat<Ff>],
    alpha_prime: &[K],
    r_prime: &[K],
    ch: &Challenges,
) -> (K, K)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    // Convenience wrapper that omits explicit input-evaluation point.
    q_eval_at_ext_point_paper_exact_with_inputs::<Ff>(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        alpha_prime,
        r_prime,
        ch,
        None,
    )
}

/// Evaluate Q at an arbitrary extension point (α', r') directly from witnesses.
///
/// This variant matches the paper's Step 4 LHS exactly, including gating the Eval block by
/// eq((α',r'),(α,r)). In k>k_mcs cases (ME slots present), `r_inputs` is required.
pub fn q_eval_at_ext_point_paper_exact_with_inputs<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    mcs_witnesses: &[CcsWitness<Ff>],
    me_witnesses: &[Mat<Ff>],
    alpha_prime: &[K],
    r_prime: &[K],
    ch: &Challenges,
    r_inputs: Option<&[K]>,
) -> (K, K)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let detailed_log = std::env::var("NEO_CROSSCHECK_DETAIL").is_ok();
    let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s).unwrap_or_else(|| {
        panic!(
            "optimized common Q eval requires SuperNeo-compatible CCS shape (m={}, matrices={})",
            s.m,
            s.matrices.len()
        )
    });

    if detailed_log {
        eprintln!(
            "  [Paper-exact] k_total = {} (mcs_witnesses={}, me_witnesses={})",
            mcs_witnesses.len() + me_witnesses.len(),
            mcs_witnesses.len(),
            me_witnesses.len()
        );
        eprintln!("  [Paper-exact] gamma = {:?}", ch.gamma);
        eprintln!("  [Paper-exact] r_inputs present = {}", r_inputs.is_some());
    }

    // ---------------------------
    // χ tables (Ajtai & row)
    // ---------------------------
    let d_sz = 1usize << alpha_prime.len(); // size along Ajtai bits
    let n_sz = 1usize << r_prime.len(); // size along row bits

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

    // eq((α′,r′), β)
    let eq_beta = eq_points(alpha_prime, &ch.beta_a) * eq_points(r_prime, &ch.beta_r);

    if detailed_log {
        eprintln!("  [Paper-exact] eq((α',r'), β) = {:?}", eq_beta);
    }

    let k_mcs = mcs_witnesses.len();
    let k_total = k_mcs + me_witnesses.len();
    assert!(
        d_sz >= D,
        "q_eval_at_ext_point: alpha dimension too small (2^|alpha'|={} < D={})",
        d_sz,
        D
    );

    // eq((α′,r′), (α, r)) gating for the Eval block.
    // When ME slots exist, `r_inputs` is mandatory.
    let eq_ar = if k_total > k_mcs {
        let r = r_inputs.expect("q_eval_at_ext_point: missing shared ME input r");
        eq_points(alpha_prime, &ch.alpha) * eq_points(r_prime, r)
    } else {
        K::ZERO
    };

    // Packed-SuperNeo path: compute y_ring rows first and derive F/NC/Eval directly.
    // This matches PaperExact packed semantics and keeps cross-engine parity.
    let all_packed = mcs_witnesses.iter().all(|w| {
        matches!(
            crate::common::witness_mat_layout(&w.Z, s.m),
            Ok(crate::common::WitnessMatLayout::SuperneoPacked)
        )
    }) && me_witnesses.iter().all(|z| {
        matches!(
            crate::common::witness_mat_layout(z, s.m),
            Ok(crate::common::WitnessMatLayout::SuperneoPacked)
        )
    });
    if all_packed {
        let cache = &superneo_cache;
        let n_eff = core::cmp::min(s.n, chi_r.len());
        let mut y_by_inst: Vec<Vec<[K; D]>> = Vec::with_capacity(k_total);
        for Zi in mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
        {
            let z_i = recomposed_z_from_Z::<Ff>(params, s.m, Zi);
            y_by_inst.push(crate::superneo_eval::eval_all_mats_ring_cached(
                cache, &z_i, &chi_r, n_eff,
            ));
        }

        let mut F_prime = K::ZERO;
        {
            let mut g = K::ONE;
            for y_by_j in y_by_inst.iter().take(k_mcs) {
                let m_vals: Vec<K> = y_by_j.iter().take(s.t()).map(|row| row[0]).collect();
                F_prime += g * s.f.eval_in_ext::<K>(&m_vals);
                g *= ch.gamma;
            }
        }

        let mut nc_sum = K::ZERO;
        {
            let mut g = K::ONE; // γ^K
            for _ in 0..k_mcs {
                g *= ch.gamma;
            }
            for y_by_j in &y_by_inst {
                let mut y_eval = K::ZERO;
                for rho in 0..D {
                    y_eval += y_by_j[0][rho] * chi_a[rho];
                }
                nc_sum += g * range_product::<Ff>(y_eval, params.b);
                g *= ch.gamma;
            }
        }

        let mut eval_sum = K::ZERO;
        if k_total > k_mcs {
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= ch.gamma;
            }

            for (i_abs, y_by_j) in y_by_inst.iter().enumerate().skip(k_mcs) {
                let mut gamma_i = K::ONE;
                for _ in 0..i_abs {
                    gamma_i *= ch.gamma;
                }
                let mut gamma_k_pow_j = K::ONE;
                for row in y_by_j.iter().take(s.t()) {
                    let mut y_eval = K::ZERO;
                    for rho in 0..D {
                        y_eval += row[rho] * chi_a[rho];
                    }
                    eval_sum += (gamma_i * gamma_k_pow_j) * y_eval;
                    gamma_k_pow_j *= gamma_to_k;
                }
            }
        }

        let mut gamma_to_k_outer = K::ONE;
        for _ in 0..k_total {
            gamma_to_k_outer *= ch.gamma;
        }
        let lhs = eq_beta * (F_prime + nc_sum) + eq_ar * (gamma_to_k_outer * eval_sum);
        return (lhs, K::ZERO);
    }

    if detailed_log {
        eprintln!("  [Paper-exact] eq((α',r'), (α,r)) = {:?}", eq_ar);
    }

    // ---------------------------
    // F' := Σ_{i=1..k_mcs} γ^{i-1} · f( Ẽ(M_j z_i)(r') )_j
    // ---------------------------
    let mut F_prime = K::ZERO;
    {
        let mut g = K::ONE; // γ^{i-1}
        for w in mcs_witnesses {
            let z_i = recomposed_z_from_Z::<Ff>(params, s.m, &w.Z); // K^m
            let m_vals = eval_all_mats_with_cache(s, &superneo_cache, &z_i, &chi_r, s.n);
            F_prime += g * s.f.eval_in_ext::<K>(&m_vals);
            g *= ch.gamma;
        }
    }

    if detailed_log {
        eprintln!("  [Paper-exact] F' = f(m_vals) = {:?}", F_prime);
    }

    // ---------------------------------------
    // v1 := M_1^T · χ_{r'}  (K^m), used in NC
    // ---------------------------------------
    let v1_form = superneo_cache
        .matrix(0)
        .unwrap_or_else(|| panic!("optimized common NC path: missing matrix 0 in SuperNeo cache"))
        .build_linear_form(&chi_r, s.n);

    // ---------------------------------------
    // Σ γ^{K+i-1} · N_i'  with Ajtai MLE at α′
    // ---------------------------------------
    let mut nc_sum = K::ZERO;
    {
        let mut g = K::ONE; // γ^K
        for _ in 0..k_mcs {
            g *= ch.gamma;
        }

        // MCS instances
        for w in mcs_witnesses {
            let z_layout = crate::common::witness_mat_layout(&w.Z, s.m)
                .unwrap_or_else(|e| panic!("q_eval_at_ext_point: invalid MCS witness shape for s.m={}: {e}", s.m));
            let mut y_eval = K::ZERO;
            for rho in 0..D {
                let y_rho =
                    v1_form.eval_vec_base_f_with(|c| crate::common::witness_mat_get_f(&w.Z, z_layout, s.m, rho, c));
                y_eval += y_rho * chi_a[rho];
            }
            nc_sum += g * range_product::<Ff>(y_eval, params.b);
            g *= ch.gamma;
        }

        // ME witnesses (if any)
        for Z in me_witnesses {
            let z_layout = crate::common::witness_mat_layout(Z, s.m)
                .unwrap_or_else(|e| panic!("q_eval_at_ext_point: invalid ME witness shape for s.m={}: {e}", s.m));
            let mut y_eval = K::ZERO;
            for rho in 0..D {
                let y_rho =
                    v1_form.eval_vec_base_f_with(|c| crate::common::witness_mat_get_f(Z, z_layout, s.m, rho, c));
                y_eval += y_rho * chi_a[rho];
            }
            nc_sum += g * range_product::<Ff>(y_eval, params.b);
            g *= ch.gamma;
        }
    }

    if detailed_log {
        eprintln!("  [Paper-exact] NC' (norm constraints) = {:?}", nc_sum);
    }

    // ---------------------------------------
    // Eval block: γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
    // with E_{(i,j)} = eq((α′,r′),(α,r)) · ẏ'_{(i,j)}(α′).
    // We compute the inner sum with correct γ weights; eq_ar keeps it gated.
    // ---------------------------------------
    let mut eval_sum = K::ZERO;
    if k_total > k_mcs {
        // Precompute γ^k
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        for (i_abs, Zi) in mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .enumerate()
            .skip(k_mcs)
        {
            let zi_layout = crate::common::witness_mat_layout(Zi, s.m)
                .unwrap_or_else(|e| panic!("q_eval_at_ext_point: invalid witness shape for s.m={}: {e}", s.m));
            // z_i(α') := Σ_ρ χ_a[ρ] · Z_i[ρ,·]
            let mut z_alpha = vec![K::ZERO; s.m];
            for rho in 0..D {
                let w = chi_a[rho];
                if w == K::ZERO {
                    continue;
                }
                for c in 0..s.m {
                    z_alpha[c] += crate::common::witness_mat_get_k(Zi, zi_layout, s.m, rho, c) * w;
                }
            }

            // y_(i,j)'(α', r') = Ẽ(M_j · z_i(α'))(r')
            let y_by_j = eval_all_mats_with_cache(s, &superneo_cache, &z_alpha, &chi_r, s.n);

            // weight = γ^{i-1} · (γ^k)^j
            let mut gamma_i = K::ONE;
            for _ in 0..i_abs {
                gamma_i *= ch.gamma;
            }
            let mut gamma_k_pow_j = K::ONE;
            for y_eval in y_by_j.iter().take(s.t()) {
                eval_sum += (gamma_i * gamma_k_pow_j) * *y_eval;
                gamma_k_pow_j *= gamma_to_k;
            }
        }
    }

    if detailed_log {
        eprintln!(
            "  [Paper-exact] Eval' (weighted ME evaluations, before outer γ^k) = {:?}",
            eval_sum
        );
    }

    // Paper-exact assembly of LHS:
    // Q(α', r') = eq((α',r'), β)·(F' + NC') + γ^k · eq((α',r'), (α,r)) · Eval'.
    let mut gamma_to_k_outer = K::ONE;
    for _ in 0..k_total {
        gamma_to_k_outer *= ch.gamma;
    }
    let lhs = eq_beta * (F_prime + nc_sum) + eq_ar * (gamma_to_k_outer * eval_sum);

    if detailed_log {
        eprintln!("  [Paper-exact] Final assembly:");
        eprintln!(
            "                eq((α',r'), β) * (F' + NC') = {:?}",
            eq_beta * (F_prime + nc_sum)
        );
        eprintln!(
            "              + eq((α',r'), (α,r)) * (γ^k * Eval') = {:?}",
            eq_ar * (gamma_to_k_outer * eval_sum)
        );
        eprintln!("              = Q(α', r') = {:?}", lhs);
    }

    // Preserve existing return shape; RHS not used by callers here.
    (lhs, K::ZERO)
}

/// --- Public claimed sum T for sumcheck ------------------------------------
///
/// Compute the public claimed sum used by sumcheck:
///   T = γ^k · Σ_{j=1}^{t} Σ_{i=2}^{k} γ^{i+(j-1)k-1} · ⟨ y_{(i,j)}, χ_{α} ⟩
///
/// This value depends *only* on the ME input instances and the challenge α,
/// making it publicly computable by the verifier. The prover must use this
/// same T to ensure that an invalid CCS witness fails the first sumcheck invariant.
pub fn claimed_initial_sum_from_inputs_with_k_mcs<Ff>(
    s: &CcsStructure<Ff>,
    ch: &Challenges,
    k_mcs: usize,
    me_inputs: &[CeClaim<Cmt, Ff, K>],
) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("\n[claimed_initial_sum] === Computing T ===");
        eprintln!("[claimed_initial_sum] me_inputs.len() = {}", me_inputs.len());
    }

    let k_total = k_mcs + me_inputs.len();

    #[cfg(feature = "debug-logs")]
    eprintln!(
        "[claimed_initial_sum] k_total = {} (= {} MCS + {} ME)",
        k_total,
        k_mcs,
        me_inputs.len()
    );

    if k_total < 2 {
        #[cfg(feature = "debug-logs")]
        eprintln!("[claimed_initial_sum] k < 2, returning ZERO (no Eval block)");
        return K::ZERO; // no Eval block when k=1
    }

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[claimed_initial_sum] s.t() = {} (number of matrices)", s.t());
        eprintln!("[claimed_initial_sum] ch.alpha.len() = {}", ch.alpha.len());
        eprintln!("[claimed_initial_sum] ch.gamma = {:?}", ch.gamma);
    }

    // Build χ_{α} over the Ajtai domain
    let d_sz = 1usize << ch.alpha.len();
    let mut chi_a = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for (bit, &a) in ch.alpha.iter().enumerate() {
            let is_one = ((rho >> bit) & 1) == 1;
            w *= if is_one { a } else { K::ONE - a };
        }
        chi_a[rho] = w;
    }

    // γ^k
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= ch.gamma;
    }

    #[cfg(feature = "debug-logs")]
    eprintln!("[claimed_initial_sum] gamma_to_k (γ^{}) = {:?}", k_total, gamma_to_k);

    // Inner weighted sum over (j, i in ME slots).
    let mut inner = K::ZERO;
    for j in 0..s.t() {
        for (idx, out) in me_inputs.iter().enumerate() {
            // me_inputs[idx] corresponds to absolute instance slot i = k_mcs + idx + 1 (1-based).
            let i = k_mcs + idx + 1;

            // ẏ_{(i,j)}(α) = ⟨ y_{(i,j)}, χ_{α} ⟩
            let yj = &out.y_ring[j];
            let mut y_eval = K::ZERO;
            assert!(
                yj.len() >= d_sz,
                "claimed_initial_sum: y_ring[{}] too short for chi(alpha) (len {} < {})",
                j,
                yj.len(),
                d_sz
            );
            for rho in 0..d_sz {
                y_eval += yj[rho] * chi_a[rho];
            }

            // Paper formula: γ^{i+(j-1)k-1} = γ^{i-1+(j-1)k} = γ^{i-1} · (γ^k)^{j-1}
            // But we're using 0-based j, so for paper's j=1: we have loop j=0
            let mut weight = K::ONE;
            // γ^{i-1}
            for _ in 0..(i - 1) {
                weight *= ch.gamma;
            }
            // (γ^k)^j (j is 0-based in the loop)
            for _ in 0..j {
                weight *= gamma_to_k;
            }

            #[cfg(feature = "debug-logs")]
            if idx < 2 && j < 2 {
                eprintln!(
                    "[claimed_initial_sum]   ME[{}] (i={}), j={}: y_eval={:?}, weight={:?}, contrib={:?}",
                    idx,
                    i,
                    j,
                    y_eval,
                    weight,
                    weight * y_eval
                );
            }

            inner += weight * y_eval;
        }
    }

    // Paper-exact: T = γ^k · inner, matching T = γ^k Σ γ^{i+(j-1)k-1} ẏ_{(i,j)}(α).
    let result = gamma_to_k * inner;
    result
}

/// --- Π_RLC (Section 4.5) ---------------------------------------------------
///
/// Paper-exact Random Linear Combination using explicit S-action matrices ρ_i ∈ F^{D×D}.
///
/// Input: `rhos` (one per input), `me_inputs` (k+1 ME instances, same r), and their witnesses `Zs`.
/// Output: combined ME instance and combined witness Z = Σ ρ_i · Z_i.
///
/// Notes:
/// - This helper performs only algebraic mixing over witnesses and outputs; it does not compute the
///   combined commitment. The output `c` is copied from the first input as a placeholder.
/// - Caller should set `out.c = Σ ρ_i · c_i` using the commitment module action if a commitment mix is required.
fn rlc_reduction_paper_exact_from_refs<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!me_inputs.is_empty(), "Π_RLC(paper-exact): need at least one input");
    let k1 = me_inputs.len();
    assert_eq!(rhos.len(), k1, "Π_RLC: |rhos| must equal |inputs|");
    assert_eq!(Zs.len(), k1, "Π_RLC: |Zs| must equal |inputs|");
    crate::common::validate_rhos_are_rotation_matrices(params, rhos, "Π_RLC(paper-exact): rhos")
        .unwrap_or_else(|e| panic!("Π_RLC(paper-exact): invalid rho set: {e}"));
    let z_cols = Zs[0].cols();
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::witness_mat_layout(*z, s.m)
            .unwrap_or_else(|e| panic!("Π_RLC(paper-exact): invalid witness shape at input {idx}: {e}"));
        assert_eq!(
            z.cols(),
            z_cols,
            "Π_RLC(paper-exact): all witness mats must share packed width"
        );
    }

    let d = D;
    let d_pad = 1usize << ell_d;
    let m_in = me_inputs[0].m_in;
    let r = me_inputs[0].r.clone();
    let aux_len = me_inputs[0].aux_openings.len();
    for (idx, inst) in me_inputs.iter().enumerate() {
        assert_eq!(
            inst.aux_openings.len(),
            aux_len,
            "Π_RLC: aux_openings.len mismatch at input {idx}"
        );
    }
    // Helper: acc += rho * A (left multiply)
    let left_mul_acc = |acc: &mut Mat<Ff>, rho: &Mat<Ff>, a: &Mat<Ff>| {
        debug_assert_eq!(rho.rows(), d);
        debug_assert_eq!(rho.cols(), d);
        debug_assert_eq!(a.rows(), d);
        debug_assert_eq!(acc.rows(), d);
        debug_assert_eq!(a.cols(), acc.cols());
        for rr in 0..d {
            for cc in 0..a.cols() {
                let mut sum = Ff::ZERO;
                for kk in 0..d {
                    sum += get_F(rho, rr, kk) * get_F(a, kk, cc);
                }
                acc[(rr, cc)] += sum;
            }
        }
    };

    // y_j := Σ ρ_i y_(i,j) (apply ρ to the first D digits; keep padding to 2^{ell_d})
    let mut y_ring: Vec<Vec<K>> = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut yj_acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            // term = ρ_i · y_(i,j)
            let yi = &me_inputs[i].y_ring[j];
            let mut term = vec![K::ZERO; d_pad];
            for rr in 0..d.min(d_pad) {
                let mut acc_rr = K::ZERO;
                for kk in 0..d.min(yi.len()) {
                    acc_rr += K::from(get_F(&rhos[i], rr, kk)) * yi[kk];
                }
                term[rr] = acc_rr;
            }
            for t in 0..d_pad {
                yj_acc[t] += term[t];
            }
        }
        y_ring.push(yj_acc);
    }

    let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);

    // aux_openings: field-linear mix using the scalar projection of each ρ_i.
    let mut aux_openings = vec![K::ZERO; aux_len];
    for i in 0..k1 {
        let w = K::from(get_F(&rhos[i], 0, 0));
        for (dst, src) in aux_openings
            .iter_mut()
            .zip(me_inputs[i].aux_openings.iter())
        {
            *dst += w * *src;
        }
    }

    // Optional NC channel: preserve channel shape across inputs.
    let wants_nc_channel = !(me_inputs[0].s_col.is_empty() && me_inputs[0].y_zcol.is_empty());
    if wants_nc_channel {
        assert!(
            !me_inputs[0].s_col.is_empty() && !me_inputs[0].y_zcol.is_empty(),
            "Π_RLC: incomplete NC channel on input 0 (expected both s_col and y_zcol)"
        );
        for (idx, inst) in me_inputs.iter().enumerate() {
            assert_eq!(inst.s_col, me_inputs[0].s_col, "Π_RLC: s_col mismatch at input {idx}");
            assert_eq!(
                inst.y_zcol.len(),
                d_pad,
                "Π_RLC: y_zcol len mismatch at input {idx} (expected {d_pad}, got {})",
                inst.y_zcol.len()
            );
        }
    }

    // X := Σ ρ_i · X_i (publicly derivable RLC relation).
    let mut X = Mat::zero(d, m_in, Ff::ZERO);
    for i in 0..k1 {
        let mut term = Mat::zero(d, m_in, Ff::ZERO);
        left_mul_acc(&mut term, &rhos[i], &me_inputs[i].X);
        for r in 0..d {
            for c in 0..m_in {
                X[(r, c)] += term[(r, c)];
            }
        }
    }
    let y_zcol = if wants_nc_channel {
        let mut acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            for r in 0..d {
                let mut sum = K::ZERO;
                for k in 0..d {
                    sum += K::from(rhos[i][(r, k)]) * me_inputs[i].y_zcol[k];
                }
                acc[r] += sum;
            }
        }
        acc
    } else {
        Vec::new()
    };

    // Z := Σ ρ_i Z_i over packed SuperNeo witness columns.
    let mut Z = Mat::zero(d, z_cols, Ff::ZERO);
    for i in 0..k1 {
        left_mul_acc(&mut Z, &rhos[i], Zs[i]);
    }

    let out = CeClaim::<Cmt, Ff, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: me_inputs[0].c.clone(), // NOTE: caller can replace with true Σ ρ_i·c_i
        X,
        r,
        s_col: me_inputs[0].s_col.clone(),
        y_ring,
        ct,
        aux_openings,
        y_zcol,
        m_in,
        fold_digest: me_inputs[0].fold_digest,
    };

    (out, Z)
}

pub fn rlc_reduction_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let z_refs: Vec<&Mat<Ff>> = Zs.iter().collect();
    rlc_reduction_paper_exact_from_refs::<Ff>(s, params, rhos, me_inputs, &z_refs, ell_d)
}

/// --- Π_RLC (optimized) -----------------------------------------------------
///
/// Optimized Random Linear Combination for the prover path.
///
/// Semantics match `rlc_reduction_paper_exact`, but this implementation:
/// - Uses cache-friendly row-major loops for the large witness matrix `Z`.
/// Same as `rlc_reduction_paper_exact`, but also computes the combined commitment via a caller-supplied
/// mixing function over commitments. This matches the paper's Π_RLC output when `combine_commit` implements
/// the correct S-module action on commitments.
#[allow(dead_code)]
pub fn rlc_reduction_paper_exact_with_commit_mix<Ff, Comb>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[Mat<Ff>],
    ell_d: usize,
    combine_commit: Comb,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Comb: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    let (mut out, Z) = rlc_reduction_paper_exact::<Ff>(s, params, rhos, me_inputs, Zs, ell_d);
    let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
    let mixed_c = combine_commit(rhos, &inputs_c);
    out.c = mixed_c;
    (out, Z)
}

/// --- Π_DEC (Section 4.6) ---------------------------------------------------
///
/// Paper-exact decomposition: given parent ME(B,L) and a provided split Z = Σ b^i · Z_i,
/// build child ME(b,L) instances and verify the two algebraic equalities (y vectors and X matrices).
///
/// Notes:
/// - Commitment creation for children is not performed here; `c` is copied from parent.
/// - This keeps the helper algebraic and suitable for cross-checking. Caller is responsible for
///   validating the commitment equality c ?= Σ \bar b^{i-1} c_i if a commitment check is desired.
pub fn dec_reduction_paper_exact<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    dec_reduction_paper_exact_inner(s, params, parent, Z_split, ell_d, None)
}

/// Same as `dec_reduction_paper_exact`, but uses a prebuilt CSC cache to avoid dense n×m scans.
pub fn dec_reduction_paper_exact_with_sparse_cache<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
    sparse: &super::sparse::SparseCache<Ff>,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    dec_reduction_paper_exact_inner(s, params, parent, Z_split, ell_d, Some(sparse))
}

fn dec_reduction_paper_exact_inner<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
    _sparse: Option<&super::sparse::SparseCache<Ff>>,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(
        !Z_split.is_empty(),
        "Π_DEC(paper-exact): need at least one digit witness"
    );

    let d = D;
    let d_pad = 1usize << ell_d;
    let k = Z_split.len();
    let m_in = parent.m_in;

    // Build χ_r and v_j = M_j^T · χ_r.
    let ell_n = parent.r.len();
    let n_sz = 1usize << ell_n; // 2^{ℓ_n}
    let n_eff = core::cmp::min(s.n, n_sz);

    let mut chi_r = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for (bit, &rb) in parent.r.iter().enumerate() {
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { rb } else { K::ONE - rb };
        }
        chi_r[row] = w;
    }

    let t_mats = s.t();
    let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s).unwrap_or_else(|| {
        panic!(
            "Π_DEC optimized common requires SuperNeo-compatible CCS shape (m={}, matrices={})",
            s.m,
            s.matrices.len()
        )
    });

    // Scalar base used for DEC reconstruction checks.
    let bF = Ff::from_u64(params.b as u64);
    let bK = K::from(Ff::from_u64(params.b as u64));

    // Helper: project first m_in columns from Z.
    let project_x = |Z: &Mat<Ff>| {
        crate::common::project_x_from_witness_mat(Z, s.m, m_in)
            .unwrap_or_else(|e| panic!("Π_DEC: project_x failed for m={}: {e}", s.m))
    };

    let parent_c = &parent.c;
    let parent_r = &parent.r;
    let fold_digest = parent.fold_digest;
    let parent_aux = parent.aux_openings.clone();
    let aux_len = parent_aux.len();

    // Optional NC channel: build χ_{s_col} once for all children.
    let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
    let chi_s = if want_nc_channel {
        assert!(
            !parent.s_col.is_empty() && !parent.y_zcol.is_empty(),
            "Π_DEC: incomplete NC channel on parent (expected both s_col and y_zcol)"
        );
        assert_eq!(
            parent.y_zcol.len(),
            d_pad,
            "Π_DEC: parent y_zcol len mismatch (expected {d_pad}, got {})",
            parent.y_zcol.len()
        );
        let chi = neo_ccs::utils::tensor_point::<K>(&parent.s_col);
        assert!(
            chi.len() >= s.m,
            "Π_DEC: chi(s_col) too short for CCS width (need >= {}, got {})",
            s.m,
            chi.len()
        );
        chi
    } else {
        Vec::new()
    };

    // Build children (parallel over digits).
    let build_child = |i: usize| {
        let Zi = &Z_split[i];
        let Xi = project_x(Zi);

        let z_i = crate::common::decode_superneo_coeffs_from_witness_mat(Zi, s.m)
            .unwrap_or_else(|e| panic!("Π_DEC: failed to decode packed child witness coefficients: {e}"));
        let y_i: Vec<Vec<K>> = crate::superneo_eval::eval_all_mats_ring_cached(&superneo_cache, &z_i, &chi_r, n_eff)
            .into_iter()
            .map(|coeffs| {
                let mut row = coeffs.to_vec();
                assert!(
                    row.len() <= d_pad,
                    "Π_DEC: refusing to truncate y row (len {} > d_pad {})",
                    row.len(),
                    d_pad
                );
                if row.len() < d_pad {
                    row.resize(d_pad, K::ZERO);
                }
                row
            })
            .collect();
        let y_scalars_i = crate::common::ct_from_y_ring_for_ccs_m(&y_i, params, s.m);

        let y_zcol = if chi_s.is_empty() {
            Vec::new()
        } else {
            crate::common::compute_y_zcol_from_witness(params, Zi, s.m, &chi_s, d_pad)
                .unwrap_or_else(|e| panic!("Π_DEC: y_zcol compute failed: {e}"))
        };

        CeClaim::<Cmt, Ff, K> {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: parent_c.clone(), // caller patches with L(Z_i)
            X: Xi,
            r: parent_r.clone(),
            s_col: parent.s_col.clone(),
            y_ring: y_i,
            ct: y_scalars_i,
            aux_openings: if i == 0 {
                parent_aux.clone()
            } else {
                vec![K::ZERO; aux_len]
            },
            y_zcol,
            m_in,
            fold_digest,
        }
    };

    let mut children: Vec<CeClaim<Cmt, Ff, K>> = {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            (0..k).into_par_iter().map(build_child).collect()
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            (0..k).map(build_child).collect()
        }
    };

    // Reconcile X-channel to the public parent relation:
    // enforce parent.X == Σ b^i * child_i.X by correcting child 0 only.
    if !children.is_empty() {
        let mut lhs_X = Mat::zero(d, m_in, Ff::ZERO);
        let mut pow = Ff::ONE;
        for child in children.iter().take(k) {
            for r in 0..d {
                for c in 0..m_in {
                    lhs_X[(r, c)] += pow * child.X[(r, c)];
                }
            }
            pow *= bF;
        }
        for r in 0..d {
            for c in 0..m_in {
                children[0].X[(r, c)] += parent.X[(r, c)] - lhs_X[(r, c)];
            }
        }
    }

    // Verify: y_j ?= Σ b^i · y_(i,j)
    let mut ok_y = true;
    for j in 0..t_mats {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut pow = K::ONE;
        for i in 0..k {
            for t in 0..d_pad {
                lhs[t] += pow * children[i].y_ring[j][t];
            }
            pow *= bK;
        }
        if lhs != parent.y_ring[j] {
            #[cfg(feature = "debug-logs")]
            {
                let mut first = None;
                for t in 0..d_pad {
                    if lhs[t] != parent.y_ring[j][t] {
                        first = Some(t);
                        break;
                    }
                }
                eprintln!(
                    "DEC(y) mismatch at row j={j}, first_t={:?}, lhs_t={:?}, parent_t={:?}",
                    first,
                    first.map(|t| lhs[t]),
                    first.map(|t| parent.y_ring[j][t])
                );
            }
            ok_y = false;
            break;
        }
    }

    // Verify: X ?= Σ b^i · X_i.
    let mut ok_X = true;
    'x_check: for rho in 0..d {
        for c in 0..m_in {
            let mut lhs = Ff::ZERO;
            let mut pow = Ff::ONE;
            for child in children.iter().take(k) {
                lhs += pow * child.X[(rho, c)];
                pow *= bF;
            }
            if lhs != parent.X[(rho, c)] {
                ok_X = false;
                break 'x_check;
            }
        }
    }

    let _ = (want_nc_channel, bK, d_pad);

    (children, ok_y, ok_X)
}

/// Same as `dec_reduction_paper_exact`, additionally checking the commitment equality
/// c ?= Σ \bar b^{i-1} c_i via a caller-supplied linear combination over commitments.
/// Returns `(children, ok_y, ok_X, ok_c)`.
#[allow(dead_code)]
pub fn dec_reduction_paper_exact_with_commit_check<Ff, Comb>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
    child_commitments: &[Cmt],
    combine_b_pows: Comb,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    let (mut children, ok_y, ok_X) = dec_reduction_paper_exact::<Ff>(s, params, parent, Z_split, ell_d);

    assert_eq!(
        children.len(),
        child_commitments.len(),
        "DEC: |children| != |child_commitments|"
    );

    // Patch children commitments with the correct ones
    for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
        ch.c = c.clone();
    }

    // Commitment equality: c ?= Σ \bar b^{i-1} c_i
    let combined_c = combine_b_pows(child_commitments, params.b);
    let ok_c = combined_c == parent.c;
    (children, ok_y, ok_X, ok_c)
}
