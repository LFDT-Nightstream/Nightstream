//! Public DEC verification.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

/// Check a public Π_DEC transition.
///
/// The verifier requires exactly `params.k_rho` children, computes
/// `split_b(parent.X, params.k_rho, params.b)` itself, and requires each child's
/// public `X` to equal the corresponding canonical digit matrix. The remaining
/// public fields are checked by radix-`b` recomposition.
pub fn verify_dec_public<MB>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    column_point_len: usize,
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
    if params.b < 2 {
        return fail(format!("invalid decomposition base b={}", params.b));
    }
    if let Err(error) = super::validate_ce_claim_shape("verify_dec_public: parent", s, column_point_len, parent) {
        return fail(error);
    }
    let k = children.len();
    if k == 0 {
        return fail("no children");
    }
    if k != params.k_rho as usize {
        return fail(format!(
            "child count mismatch (expected k_rho={}, got {k})",
            params.k_rho
        ));
    }
    for (index, child) in children.iter().enumerate() {
        if let Err(error) = super::validate_ce_claim_shape(
            &format!("verify_dec_public: children[{index}]"),
            s,
            column_point_len,
            child,
        ) {
            return fail(error);
        }
        if child.fold_digest != parent.fold_digest {
            return fail(format!(
                "child {index} fold digest does not match the parent transcript"
            ));
        }
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
    let wants_nc_point = !parent.s_col.is_empty()
        || !parent.y_zcol.is_empty()
        || children
            .iter()
            .any(|child| !child.s_col.is_empty() || !child.y_zcol.is_empty());
    let enforce_y_zcol_recomposition = wants_nc_point && column_point_len < super::ell_m_for_ccs(s);
    if wants_nc_point {
        if parent.s_col.is_empty() || parent.y_zcol.is_empty() {
            return fail("parent has an incomplete NC channel");
        }
        if parent.s_col.len() != column_point_len {
            return fail(format!(
                "parent s_col length mismatch (expected {}, got {})",
                column_point_len,
                parent.s_col.len()
            ));
        }
        for (idx, ch) in children.iter().enumerate() {
            if ch.s_col.is_empty() || ch.y_zcol.is_empty() {
                return fail(format!("child {idx} has an incomplete NC channel"));
            }
            if ch.s_col.len() != column_point_len {
                return fail(format!(
                    "child {} s_col length mismatch (expected {}, got {})",
                    idx,
                    column_point_len,
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

    // The verifier, rather than the prover, determines the public-X split.
    // Failure to represent the parent in exactly k radix-b digits is a rejected
    // transition. The parent's old-point y_zcol is intentionally not checked
    // here; the delayed-projection authority bridge must close that gap before
    // this omission can be treated as sound.
    let expected_child_x = match crate::common::split_b_matrix_k(&parent.X, k, params.b) {
        Ok(split) => split,
        Err(e) => return fail(format!("parent X is not representable by split_b: {e}")),
    };
    for (idx, (expected, child)) in expected_child_x.iter().zip(children.iter()).enumerate() {
        if child.X != *expected {
            return fail(format!(
                "child {idx} X does not equal verifier-computed split_b(parent.X)"
            ));
        }
    }

    let d_pad = match super::checked_power_of_two("verify_dec_public ell_d", ell_d) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    if wants_nc_point {
        if parent.y_zcol.len() != d_pad || parent.y_zcol[D..].iter().any(|&value| value != K::ZERO) {
            return fail(format!(
                "parent y_zcol must contain {D} live lanes and zero padding to length {d_pad}"
            ));
        }
        for (index, child) in children.iter().enumerate() {
            if child.y_zcol.len() != d_pad || child.y_zcol[D..].iter().any(|&value| value != K::ZERO) {
                return fail(format!(
                    "child {index} y_zcol must contain {D} live lanes and zero padding to length {d_pad}"
                ));
            }
        }
    }
    let b_k = K::from(F::from_u64(params.b as u64));
    let mut b_pows_k = Vec::with_capacity(k);
    let mut p_k = K::ONE;
    for _ in 0..k {
        b_pows_k.push(p_k);
        p_k *= b_k;
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

    if enforce_y_zcol_recomposition {
        let mut lhs = vec![K::ZERO; d_pad];
        for (pow, child) in b_pows_k.iter().zip(children) {
            for (dst, value) in lhs.iter_mut().zip(&child.y_zcol) {
                *dst += *pow * *value;
            }
        }
        if lhs != parent.y_zcol {
            return fail("y_zcol radix recomposition mismatch");
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
