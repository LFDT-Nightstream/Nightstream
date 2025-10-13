//! TIE check utilities for IVC
//!
//! TIE (Transcript Integrity Enforcement) checks ensure that the folding
//! witness correctly ties the CCS matrices to the folding challenge r.

use super::prelude::*;

// Build χ_r(i) over the prefix 0..n-1 using ell = r.len() (LSB-first bit order).
#[inline]
pub(crate) fn chi_r_prefix(r: &[neo_math::K], n: usize) -> Vec<neo_math::K> {
    let ell = r.len();
    let mut chi = vec![neo_math::K::ZERO; n];
    for i in 0..n {
        let mut w = neo_math::K::ONE;
        let mut ii = i;
        for k in 0..ell {
            let rk = r[k];
            let bit_is_one = (ii & 1) == 1;
            let term = if bit_is_one { rk } else { neo_math::K::ONE - rk };
            w *= term;
            ii >>= 1;
        }
        chi[i] = w;
    }
    chi
}

// Robust tie check at verifier r: y_j ?= Z · (M_j^T · χ_r), supports any n (not just powers of two).
pub(crate) fn tie_check_with_r(
    s: &neo_ccs::CcsStructure<F>,
    me_parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    wit_parent: &neo_ccs::MeWitness<F>,
    r: &[neo_math::K],
) -> Result<(), String> {
    let d = neo_math::D;
    let n = s.n;
    let m = s.m;
    let t = s.t() as usize;

    if wit_parent.Z.rows() != d || wit_parent.Z.cols() != m {
        return Err(format!(
            "wit_parent.Z shape {}x{} != D x m ({} x {})",
            wit_parent.Z.rows(), wit_parent.Z.cols(), d, m
        ));
    }
    if me_parent.y.len() != t {
        return Err(format!("me_parent.y len {} != t {}", me_parent.y.len(), t));
    }
    for (j, yj) in me_parent.y.iter().enumerate() {
        if yj.len() != d {
            return Err(format!("me_parent.y[{}] len {} != D {}", j, yj.len(), d));
        }
    }

    let chi = chi_r_prefix(r, n);

    for j in 0..t {
        let mj = &s.matrices[j];
        // v_j = M_j^T · χ_r  (size m)
        let mut v_j: Vec<neo_math::K> = vec![neo_math::K::ZERO; m];
        for i in 0..n {
            let w_i = chi[i];
            for col in 0..m {
                let m_ij: F = mj[(i, col)];
                if m_ij != F::ZERO { v_j[col] += neo_math::K::from(m_ij) * w_i; }
            }
        }
        // y_pred = Z · v_j  (size D)
        let mut y_pred: Vec<neo_math::K> = vec![neo_math::K::ZERO; d];
        for row in 0..d {
            let mut acc = neo_math::K::ZERO;
            for col in 0..m {
                let z_rc: F = wit_parent.Z[(row, col)];
                if z_rc != F::ZERO { acc += neo_math::K::from(z_rc) * v_j[col]; }
            }
            y_pred[row] = acc;
        }
        if y_pred != me_parent.y[j] {
            return Err(format!("tie mismatch on j={} (Z·(M_j^T·χ_r))", j));
        }
    }

    // Also ensure X matches Z prefix (public slice), a cheap consistency guard.
    let m_in = me_parent.m_in;
    if me_parent.X.rows() != d || me_parent.X.cols() != m_in {
        return Err("me_parent.X shape mismatch".into());
    }
    for row in 0..d { for col in 0..m_in {
        if me_parent.X[(row, col)] != wit_parent.Z[(row, col)] {
            return Err("X != Z[:, :m_in]".into());
        }
    }}

    Ok(())
}

// Public test-only wrapper for integration tests.
#[cfg(feature = "testing")]
pub fn tie_check_with_r_public(
    s: &neo_ccs::CcsStructure<F>,
    me_parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    wit_parent: &neo_ccs::MeWitness<F>,
    r: &[neo_math::K],
) -> Result<(), String> {
    tie_check_with_r(s, me_parent, wit_parent, r)
}

