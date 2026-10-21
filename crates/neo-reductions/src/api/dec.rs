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
        return false;
    }
    if params.b < 2 {
        return false;
    }
    if let Err(error) = super::validate_ce_claim_shape("verify_dec_public: parent", s, parent) {
        return fail(error);
    }
    if let Err(error) =
        super::validate_pi_ccs_outputs("verify_dec_public: selected parent", s, std::slice::from_ref(parent))
    {
        return fail(error);
    }
    let k = children.len();
    if k == 0 {
        return false;
    }
    if k != params.k_rho as usize {
        return false;
    }
    for (index, child) in children.iter().enumerate() {
        if let Err(error) = super::validate_ce_claim_shape(&format!("verify_dec_public: children[{index}]"), s, child) {
            return fail(error);
        }
        if let Err(error) =
            super::validation::validate_commitment_shape("verify_dec_public: child", &child.c, parent.c.kappa)
        {
            return fail(error);
        }
        if let Err(error) = super::validate_pi_ccs_outputs(
            &format!("verify_dec_public: selected child[{index}]"),
            s,
            std::slice::from_ref(child),
        ) {
            return fail(error);
        }
        if child.fold_digest != parent.fold_digest {
            return false;
        }
    }

    let shared_children_r = match crate::engines::utils::shared_me_input_r(children, parent.r.len()) {
        Ok(Some(r)) => r,
        Ok(None) => return false,
        Err(e) => return fail(e),
    };
    if parent.r.as_slice() != shared_children_r {
        return false;
    }

    if parent.m_in > s.m {
        return false;
    }
    let x_cols = neo_ccs::superneo_public_x_cols(parent.m_in);
    if parent.X.rows() != D || parent.X.cols() != x_cols {
        return false;
    }
    for ch in children {
        if ch.m_in > s.m {
            return false;
        }
        if ch.m_in != parent.m_in {
            return false;
        }
        if ch.X.rows() != D || ch.X.cols() != x_cols {
            return false;
        }
    }
    let d_pad = match super::checked_superneo_d_pad("verify_dec_public ell_d", ell_d) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let matrix_count = parent.eval_a.len();
    if matrix_count != s.t() {
        return false;
    }
    if parent.eval_k.len() != d_pad {
        return false;
    }
    for ch in children {
        if ch.eval_k.len() != d_pad {
            return false;
        }
        if ch.eval_a.len() != matrix_count {
            return false;
        }
    }

    // The verifier determines the public-X split.
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

    let b_k = K::from(F::from_u64(params.b as u64));
    let mut b_pows_k = Vec::with_capacity(k);
    let mut p_k = K::ONE;
    for _ in 0..k {
        b_pows_k.push(p_k);
        p_k *= b_k;
    }

    let recombines = |select: &dyn Fn(&CeClaim<Cmt, F, K>) -> &[K], expected: &[K]| {
        let mut lhs = vec![K::ZERO; d_pad];
        for (pow, child) in b_pows_k.iter().zip(children) {
            let values = select(child);
            if values.len() != d_pad {
                return false;
            }
            for coordinate in 0..d_pad {
                lhs[coordinate] += *pow * values[coordinate];
            }
        }
        lhs == expected
    };
    if !recombines(&|claim| &claim.eval_k, &parent.eval_k) {
        eprintln!("verify_dec_public failed: Eval_K recomposition mismatch");
        return false;
    }
    for matrix in 0..matrix_count {
        if !recombines(&|claim| &claim.eval_a[matrix], &parent.eval_a[matrix]) {
            eprintln!("verify_dec_public failed: Eval_A recomposition mismatch at matrix {matrix}");
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
