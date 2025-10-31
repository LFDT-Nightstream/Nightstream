//! Precompute module: Prepare Q polynomial components
//!
//! # Paper Reference
//! Section 4.4, Step 2: Building the Q polynomial
//!
//! Q(X) = eq(X,β)·(F + NC) + γ^k·Eval
//!
//! This module computes the components needed for the sum-check oracle:
//! - F(β_r): CCS constraint polynomial evaluated at β_r
//! - NC hypercube sum: Σ_{X∈{0,1}^{log(dn)}} eq(X,β)·NC_i(X) (non-multilinear!)
//! - Eval aggregator G: Weighted sum of matrix-vector products for ME inputs
//! - NC full y matrices: Z_i·M_1^T for exact Ajtai computation
//! - MLE partials: M̃_j·z_1 for F polynomial evaluation

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::pi_ccs::sparse_matrix::{Csr, spmv_csr_ff};
use crate::pi_ccs::eq_weights::{HalfTableEq, RowWeight};
use crate::pi_ccs::nc_constraints::compute_nc_hypercube_sum;
use crate::pi_ccs::transcript::Challenges;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, MatRef, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

/// Per-instance data after validation
pub struct Inst<'a> {
    pub Z: &'a Mat<F>,
    pub m_in: usize,
    pub mz: Vec<Vec<F>>,
    pub c: Cmt,
}

/// MLE partials for F polynomial (M̃_j·z_1)
pub struct MlePartials {
    pub s_per_j: Vec<Vec<K>>,
}

/// Precomputed constants for beta block (F and NC terms)
pub struct BetaBlock {
    pub f_at_beta_r: K,
    pub nc_sum_hypercube: K,
}

/// Pad vector to power of 2 length with zeros
#[inline]
pub fn pad_to_pow2_k(mut v: Vec<K>, ell: usize) -> Result<Vec<K>, PiCcsError> {
    let want = 1usize << ell;
    if v.len() > want {
        return Err(PiCcsError::SumcheckError(format!(
            "Cannot pad: vector length {} exceeds 2^{} = {}",
            v.len(),
            ell,
            want
        )));
    }
    v.resize(want, K::ZERO);
    Ok(v)
}

/// Prepare instance data: validate MCS openings and cache M_j·z
///
/// # Paper Reference
/// Verifies c = L(Z), Z = Decomp_b(z), and caches M_j·z for reuse
pub fn prepare_instances<'a, L>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &'a [McsWitness<F>],
    mats_csr: &[Csr<F>],
    l: &L,
) -> Result<Vec<Inst<'a>>, PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "empty or mismatched mcs_list/witnesses".into(),
        ));
    }

    let mut insts = Vec::with_capacity(mcs_list.len());

    for (inst, wit) in mcs_list.iter().zip(witnesses.iter()) {
        let z = neo_ccs::relations::check_mcs_opening(l, inst, wit)
            .map_err(|e| PiCcsError::InvalidInput(format!("MCS opening failed: {e}")))?;

        if z.len() != s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "z length {} != CCS column count {}",
                z.len(),
                s.m
            )));
        }

        let Z_expected_col = neo_ajtai::decomp_b(&z, params.b, D, neo_ajtai::DecompStyle::Balanced);
        neo_ajtai::assert_range_b(&Z_expected_col, params.b)
            .map_err(|e| PiCcsError::InvalidInput(format!("Range check failed: {e}")))?;

        let d = D;
        let m = Z_expected_col.len() / d;
        let mut Z_expected_row = vec![F::ZERO; d * m];
        for col in 0..m {
            for row in 0..d {
                Z_expected_row[row * m + col] = Z_expected_col[col * d + row];
            }
        }

        if wit.Z.as_slice() != Z_expected_row.as_slice() {
            return Err(PiCcsError::InvalidInput(
                "Z != Decomp_b(z) - inconsistent z and Z".into(),
            ));
        }

        let mz: Vec<Vec<F>> = mats_csr
            .par_iter()
            .map(|csr| spmv_csr_ff::<F>(csr, &z))
            .collect();

        insts.push(Inst {
            Z: &wit.Z,
            m_in: inst.m_in,
            mz,
            c: inst.c.clone(),
        });
    }

    Ok(insts)
}

/// Build MLE partials for F polynomial (first instance only)
///
/// # Paper Reference
/// Section 4.4: F(X_r) = f(M̃_1·z_1,...,M̃_t·z_1)
pub fn build_mle_partials_first_inst(
    s: &CcsStructure<F>,
    ell_n: usize,
    insts: &[Inst],
) -> Result<MlePartials, PiCcsError> {
    if insts.is_empty() {
        return Err(PiCcsError::InvalidInput("no instances".into()));
    }

    let mut s_per_j = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let w_k: Vec<K> = insts[0].mz[j].iter().map(|&x| K::from(x)).collect();
        s_per_j.push(pad_to_pow2_k(w_k, ell_n)?);
    }

    Ok(MlePartials { s_per_j })
}

/// Precompute beta block: F(β_r) and NC hypercube sum
///
/// # Paper Reference
/// Section 4.4, Step 2: Initial sum components
/// - F(β_r) is the CCS polynomial evaluated at the row randomness
/// - NC sum is Σ_{X∈{0,1}^{log(dn)}} eq(X,β)·NC_i(X) (non-multilinear!)
pub fn precompute_beta_block(
    s: &CcsStructure<F>,
    params: &NeoParams,
    insts: &[Inst],
    witnesses: &[McsWitness<F>],
    me_witnesses: &[Mat<F>],
    ch: &Challenges,
    ell_d: usize,
    ell_n: usize,
) -> Result<BetaBlock, PiCcsError> {
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[precompute_beta_block] Called with {} MCS witnesses, {} ME witnesses", 
                 witnesses.len(), me_witnesses.len());
    }
    let beta_r_ht = HalfTableEq::new(&ch.beta_r);
    let chi_beta_r: Vec<K> = (0..(1 << ell_n)).map(|i| beta_r_ht.w(i)).collect();

    // Compute f_at_beta_r = Σ_row χ_{β_r}(row) · f((M_0·z)[row], (M_1·z)[row], ..., (M_t·z)[row])
    // This is the CORRECT "average of f" form, not "f of averages"
    let mut f_at_beta_r = K::ZERO;
    for row in 0..s.n {
        let mut m_vals = vec![K::ZERO; s.t()];
        for j in 0..s.t() {
            let v_f = &insts[0].mz[j];
            m_vals[j] = K::from(v_f[row]);
        }
        let f_row = s.f.eval_in_ext::<K>(&m_vals);
        f_at_beta_r += chi_beta_r[row] * f_row;
    }

    let nc_sum_hypercube = compute_nc_hypercube_sum(
        s,
        witnesses,
        me_witnesses,
        &ch.beta_a,
        &ch.beta_r,
        ch.gamma,
        params,
        ell_d,
        ell_n,
    );

    Ok(BetaBlock {
        f_at_beta_r,
        nc_sum_hypercube,
    })
}

/// Precompute Eval row aggregator G
///
/// # Paper Reference
/// G[row] = Σ_{j,i} γ^{i+(j-1)k-1} · (M_j · u_i)[row]
/// where u_i[c] = Σ_ρ Z_i[ρ,c]·χ_α[ρ] and i starts from 2 (ME inputs)
pub fn precompute_eval_row_partial(
    s: &CcsStructure<F>,
    me_witnesses: &[Mat<F>],
    ch: &Challenges,
    k_total: usize,
    ell_n: usize,
) -> Result<Vec<K>, PiCcsError> {
    if me_witnesses.is_empty() {
        return pad_to_pow2_k(vec![K::ZERO; 1], ell_n);
    }

    let chi_alpha: Vec<K> = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let mut G_eval = vec![K::ZERO; s.n];

    // Precompute γ^{i-1} for ME witnesses (i = i_off+2 ⇒ i-1 = i_off+1)
    let mut gamma_pow_i: Vec<K> = vec![K::ZERO; me_witnesses.len()];
    {
        let mut cur = ch.gamma; // γ^1
        for i_off in 0..me_witnesses.len() {
            gamma_pow_i[i_off] = cur;
            cur *= ch.gamma;
        }
    }
    // γ^{k_total}
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total { gamma_to_k *= ch.gamma; }

    for (i_off, Zi) in me_witnesses.iter().enumerate() {
        let mut u_i = vec![K::ZERO; s.m];
        for c in 0..s.m {
            u_i[c] = (0..D).fold(K::ZERO, |acc, rho| {
                let w = if rho < chi_alpha.len() {
                    chi_alpha[rho]
                } else {
                    K::ZERO
                };
                acc + K::from(Zi[(rho, c)]) * w
            });
        }

        for j in 0..s.t() {
            let mj_ref = MatRef::from_mat(&s.matrices[j]);
            let g_ij =
                neo_ccs::utils::mat_vec_mul_fk::<F, K>(mj_ref.data, mj_ref.rows, mj_ref.cols, &u_i);
            // Weight: γ^{(i-1)+jk} - consistent with Ajtai phase
            // No extra γ^k factor here
            let mut w_pow = gamma_pow_i[i_off];       // γ^{i-1}
            for _ in 0..j { w_pow *= gamma_to_k; }    // × (γ^k)^j

            for r in 0..s.n {
                G_eval[r] += w_pow * g_ij[r];
            }
        }
    }

    pad_to_pow2_k(G_eval, ell_n)
}

/// Precompute NC full y matrices: y_{i,1} = Z_i·M_1^T
///
/// These are d×n matrices used for exact Ajtai-phase NC computation
pub fn precompute_nc_full_rows(
    s: &CcsStructure<F>,
    witnesses: &[McsWitness<F>],
    me_witnesses: &[Mat<F>],
    ell_n: usize,
) -> Result<Vec<Vec<Vec<K>>>, PiCcsError> {
    let mut nc_y_matrices = Vec::new();

    for Zi in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()) {
        let mut y_i_full: Vec<Vec<K>> = Vec::with_capacity(D);
        for rho in 0..D {
            let mut row = vec![K::ZERO; s.n];
            for col in 0..s.n {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    acc += K::from(Zi[(rho, c)]) * K::from(s.matrices[0][(col, c)]);
                }
                row[col] = acc;
            }
            y_i_full.push(pad_to_pow2_k(row, ell_n)?);
        }
        nc_y_matrices.push(y_i_full);
    }

    Ok(nc_y_matrices)
}

/// Compute initial sum T from ME inputs
///
/// # Paper Reference
/// Section 4.4: T = Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · ỹ_{(i,j)}(α)
///            = Σ_{j,i} γ^{(i-1)+jk} · ỹ_{(i,j)}(α)  (using consistent exponent)
pub fn compute_initial_sum_components(
    s: &CcsStructure<F>,
    me_inputs: &[MeInstance<Cmt, F, K>],
    ch: &Challenges,
    k_total: usize,
    beta_block: &BetaBlock,
) -> Result<K, PiCcsError> {
    let chi_alpha: Vec<K> = neo_ccs::utils::tensor_point::<K>(&ch.alpha);

    // γ^k for computing γ^{(i-1)+jk}
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total { gamma_to_k *= ch.gamma; }

    let mut T = K::ZERO;
    for j in 0..s.t() {
        for (i_offset, me_input) in me_inputs.iter().enumerate() {
            let y_mle = me_input.y[j]
                .iter()
                .zip(&chi_alpha)
                .fold(K::ZERO, |acc, (&v, &w)| acc + v * w);

            // Weight: γ^{(i-1)+jk} - consistent with row/Ajtai phase and terminal check
            // ME inputs start at i=2, so i_offset=0 corresponds to i=2, thus i-1=1
            let mut weight_ij = K::ONE;
            for _ in 0..(i_offset + 1) { weight_ij *= ch.gamma; }  // γ^{i-1}
            for _ in 0..j { weight_ij *= gamma_to_k; }             // × (γ^k)^j
            T += weight_ij * y_mle;
        }
    }
    // No outer γ^k multiplication - the weight already includes all factors
    Ok(beta_block.f_at_beta_r + beta_block.nc_sum_hypercube + T)
}
