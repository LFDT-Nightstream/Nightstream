//! Public DEC verification.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use super::ell_m_for_ccs;

/// Check that `parent = sum_i b^i * children[i]` over the public X, y, and
/// commitment surfaces.
pub fn verify_dec_public<MB>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    children: &[CeClaim<Cmt, F, K>],
    combine_b_pows: MB,
    ell_d: usize,
) -> bool
where
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    fn fail(msg: impl core::fmt::Display) -> bool {
        eprintln!("verify_dec_public failed: {msg}");
        false
    }

    if s.m == 0 {
        return fail(format!("SuperNeo-only mode requires m > 0 (got m={})", s.m));
    }
    let k = children.len();
    if k == 0 {
        return fail("no children");
    }

    let shared_children_r = match crate::engines::utils::shared_me_input_r(children, parent.r.len()) {
        Ok(Some(r)) => r,
        Ok(None) => return fail("no children"),
        Err(e) => return fail(e),
    };
    if parent.r.as_slice() != shared_children_r {
        return fail("r mismatch between parent and children");
    }

    if parent.m_in > s.m {
        return fail(format!("parent m_in={} exceeds CCS width m={}", parent.m_in, s.m));
    }
    if parent.X.rows() != D || parent.X.cols() != parent.m_in {
        eprintln!(
            "verify_dec_public failed: parent X has shape {}x{}, expected {}x{}",
            parent.X.rows(),
            parent.X.cols(),
            D,
            parent.m_in
        );
        return false;
    }
    for (idx, ch) in children.iter().enumerate() {
        if ch.m_in > s.m {
            return fail(format!(
                "child {} has m_in={} exceeding CCS width m={}",
                idx, ch.m_in, s.m
            ));
        }
        if ch.m_in != parent.m_in {
            eprintln!(
                "verify_dec_public failed: child m_in mismatch (child {} has {}, expected {})",
                idx, ch.m_in, parent.m_in
            );
            return false;
        }
        if ch.X.rows() != D || ch.X.cols() != parent.m_in {
            eprintln!(
                "verify_dec_public failed: child X shape mismatch (child {} has {}x{}, expected {}x{})",
                idx,
                ch.X.rows(),
                ch.X.cols(),
                D,
                parent.m_in
            );
            return false;
        }
    }
    let wants_nc_point = !parent.s_col.is_empty() || children.iter().any(|ch| !ch.s_col.is_empty());
    if wants_nc_point {
        let expected_s_col = ell_m_for_ccs(s);
        if parent.s_col.len() != expected_s_col {
            return fail(format!(
                "parent s_col length mismatch (expected {}, got {})",
                expected_s_col,
                parent.s_col.len()
            ));
        }
        for (idx, ch) in children.iter().enumerate() {
            if ch.s_col.len() != expected_s_col {
                return fail(format!(
                    "child {} s_col length mismatch (expected {}, got {})",
                    idx,
                    expected_s_col,
                    ch.s_col.len()
                ));
            }
            if ch.s_col != parent.s_col {
                return fail(format!("child {} s_col does not match parent", idx));
            }
        }
    }
    // Optional NC point: s_col is shared by DEC parent and children. The
    // current verifier does not validate the parent's old-point y_zcol here.
    // Terminal child checks alone do not close that authority chain; the
    // parent projection must be state-bound and checked before s_col changes.
    let t = parent.y_ring.len();
    if t < s.t() {
        eprintln!("verify_dec_public failed: parent y.len()={} < s.t()={}", t, s.t());
        return false;
    }
    for (idx, ch) in children.iter().enumerate() {
        if ch.y_ring.len() != t {
            eprintln!(
                "verify_dec_public failed: child y.len mismatch (child {} has {}, expected {})",
                idx,
                ch.y_ring.len(),
                t
            );
            return false;
        }
        if ch.ct.len() != t {
            eprintln!(
                "verify_dec_public failed: child ct.len mismatch (child {} has {}, expected {})",
                idx,
                ch.ct.len(),
                t
            );
            return false;
        }
        if ch.aux_openings.len() != parent.aux_openings.len() {
            eprintln!(
                "verify_dec_public failed: child aux_openings.len mismatch (child {} has {}, expected {})",
                idx,
                ch.aux_openings.len(),
                parent.aux_openings.len()
            );
            return false;
        }
    }
    if parent.ct.len() != t {
        eprintln!(
            "verify_dec_public failed: parent ct.len()={} expected {}",
            parent.ct.len(),
            t
        );
        return false;
    }

    // X and y_ring decomposition are checked over the same radix-b ladder.
    // The parent's old-point y_zcol is intentionally not checked by the
    // current verifier; the delayed-projection authority bridge must close
    // that gap before this omission can be treated as sound.
    let Some(d_pad) = 1usize.checked_shl(ell_d as u32) else {
        eprintln!("verify_dec_public failed: 2^ell_d overflow");
        return false;
    };
    let b_f = F::from_u64(params.b as u64);
    let b_k = K::from(F::from_u64(params.b as u64));
    let mut b_pows_f = Vec::with_capacity(k);
    let mut b_pows_k = Vec::with_capacity(k);
    let mut p_f = F::ONE;
    let mut p_k = K::ONE;
    for _ in 0..k {
        b_pows_f.push(p_f);
        b_pows_k.push(p_k);
        p_f *= b_f;
        p_k *= b_k;
    }

    for rho in 0..D {
        for c in 0..parent.m_in {
            let mut lhs = F::ZERO;
            for (pow, child) in b_pows_f.iter().zip(children.iter()) {
                lhs += *pow * child.X[(rho, c)];
            }
            if lhs != parent.X[(rho, c)] {
                eprintln!("verify_dec_public failed: X check mismatch at ({rho}, {c})");
                return false;
            }
        }
    }

    let mut y_lhs = vec![K::ZERO; d_pad];
    for j in 0..t {
        y_lhs.fill(K::ZERO);
        for (idx, (pow, child)) in b_pows_k.iter().zip(children.iter()).enumerate() {
            if child.y_ring[j].len() != d_pad {
                eprintln!("verify_dec_public failed: child y[{}] len mismatch at j={}", idx, j);
                return false;
            }
            for t in 0..d_pad {
                y_lhs[t] += *pow * child.y_ring[j][t];
            }
        }
        if parent.y_ring[j].len() != d_pad {
            eprintln!("verify_dec_public failed: parent y[j] len mismatch at j={j}");
            return false;
        }
        if parent.ct[j] != crate::common::ct_from_y_digits(&parent.y_ring[j]) {
            eprintln!("verify_dec_public failed: parent ct mismatch at j={j}");
            return false;
        }
        for (idx, child) in children.iter().enumerate() {
            if child.ct[j] != crate::common::ct_from_y_digits(&child.y_ring[j]) {
                eprintln!("verify_dec_public failed: child {idx} ct mismatch at j={j}");
                return false;
            }
        }
        if y_lhs != parent.y_ring[j] {
            eprintln!("verify_dec_public failed: y check mismatch at j={}", j);
            return false;
        }
    }

    for j in 0..parent.aux_openings.len() {
        let mut lhs = K::ZERO;
        for (pow, child) in b_pows_k.iter().zip(children.iter()) {
            lhs += *pow * child.aux_openings[j];
        }
        if lhs != parent.aux_openings[j] {
            eprintln!("verify_dec_public failed: aux_openings check mismatch at j={j}");
            return false;
        }
    }

    let want_c = combine_b_pows(&children.iter().map(|c| c.c.clone()).collect::<Vec<_>>(), params.b);
    if want_c != parent.c {
        eprintln!("verify_dec_public failed: commitment check mismatch");
        return false;
    }

    true
}
