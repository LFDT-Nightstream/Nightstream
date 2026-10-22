//! Public DEC verification.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use crate::error::PiCcsError;

/// Computes `split_b(parent.X, params.k_rho, params.b)` itself and requires each child's public
/// `X` to equal its digit matrix.
pub fn verify_dec_public<MB>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    children: &[CeClaim<Cmt, F, K>],
    combine_b_pows: MB,
    ell_d: usize,
) -> Result<(), PiCcsError>
where
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    fn reject(reason: impl core::fmt::Display) -> PiCcsError {
        PiCcsError::ProtocolError(format!("verify_dec_public: {reason}"))
    }

    if s.m == 0 {
        return Err(reject("empty structure"));
    }
    if params.b < 2 {
        return Err(reject("decomposition base below 2"));
    }
    super::validate_ce_claim_shape("verify_dec_public: parent", s, parent)?;
    super::validate_pi_ccs_outputs("verify_dec_public: selected parent", s, std::slice::from_ref(parent))?;
    let k = children.len();
    if k == 0 {
        return Err(reject("no children"));
    }
    if k != params.k_rho as usize {
        return Err(reject("child count differs from k_rho"));
    }
    for (index, child) in children.iter().enumerate() {
        super::validate_ce_claim_shape(&format!("verify_dec_public: children[{index}]"), s, child)?;
        super::validation::validate_commitment_shape("verify_dec_public: child", &child.c, parent.c.kappa)?;
        super::validate_pi_ccs_outputs(
            &format!("verify_dec_public: selected child[{index}]"),
            s,
            std::slice::from_ref(child),
        )?;
        if child.fold_digest != parent.fold_digest {
            return Err(reject("child frame differs from the parent"));
        }
    }

    let shared_children_r = match crate::engines::utils::shared_me_input_r(children, parent.r.len()) {
        Ok(Some(r)) => r,
        Ok(None) => return Err(reject("children share no point")),
        Err(error) => return Err(error),
    };
    if parent.r.as_slice() != shared_children_r {
        return Err(reject("parent point differs from the children"));
    }

    if parent.m_in > s.m {
        return Err(reject("parent public width exceeds the structure"));
    }
    let Some(x_cols) = neo_ccs::superneo_public_x_cols(parent.m_in) else {
        return Err(reject("parent public width has no packed shape"));
    };
    if parent.X.rows() != D || parent.X.cols() != x_cols {
        return Err(reject("parent X shape"));
    }
    for ch in children {
        if ch.m_in > s.m {
            return Err(reject("child public width exceeds the structure"));
        }
        if ch.m_in != parent.m_in {
            return Err(reject("child public width differs from the parent"));
        }
        if ch.X.rows() != D || ch.X.cols() != x_cols {
            return Err(reject("child X shape"));
        }
    }
    let d_pad = super::checked_superneo_d_pad("verify_dec_public ell_d", ell_d)?;
    let matrix_count = parent.eval_a.len();
    if matrix_count != s.t() {
        return Err(reject("parent Eval_A matrix count"));
    }
    if parent.eval_k.len() != d_pad {
        return Err(reject("parent Eval_K width"));
    }
    for ch in children {
        if ch.eval_k.len() != d_pad {
            return Err(reject("child Eval_K width"));
        }
        if ch.eval_a.len() != matrix_count {
            return Err(reject("child Eval_A matrix count"));
        }
    }

    let expected_child_x = match crate::common::split_b_matrix_k(&parent.X, k, params.b) {
        Ok(split) => split,
        Err(error) => return Err(reject(format!("parent X is not representable by split_b: {error}"))),
    };
    for (idx, (expected, child)) in expected_child_x.iter().zip(children.iter()).enumerate() {
        if child.X != *expected {
            return Err(reject(format!(
                "child {idx} X does not equal verifier-computed split_b(parent.X)"
            )));
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
        return Err(reject("Eval_K recomposition mismatch"));
    }
    for matrix in 0..matrix_count {
        if !recombines(&|claim| &claim.eval_a[matrix], &parent.eval_a[matrix]) {
            return Err(reject(format!("Eval_A recomposition mismatch at matrix {matrix}")));
        }
    }

    let want_c = combine_b_pows(&children.iter().map(|c| c.c.clone()).collect::<Vec<_>>(), params.b);
    if want_c != parent.c {
        return Err(reject("commitment check mismatch"));
    }

    Ok(())
}
