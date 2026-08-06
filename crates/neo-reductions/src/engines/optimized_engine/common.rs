//! Optimized PiDEC claim construction for the selected identity-first relation.
//!
//! This file owns child evaluation and radix recomposition. It does not own
//! commitments, transcript messages, or PaperExact computations.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

pub fn dec_reduction_optimized<Ff>(
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
    dec_reduction_optimized_inner(s, params, parent, Z_split, ell_d, None, None, None, None)
}

pub fn dec_reduction_optimized_with_superneo_cache<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
    cache: &crate::superneo_eval::SuperneoEvalCache,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    dec_reduction_optimized_inner(s, params, parent, Z_split, ell_d, Some(cache), None, None, None)
}

pub fn dec_reduction_optimized_with_digit_flags<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    digit_nonzero: &[bool],
    ell_d: usize,
    cache: &crate::superneo_eval::SuperneoEvalCache,
    ring_linear_forms: Option<&[crate::superneo_eval::SuperneoRingLinearForm]>,
    precomputed_y_ring: Option<&[Vec<[K; D]>]>,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert_eq!(
        digit_nonzero.len(),
        Z_split.len(),
        "PiDEC digit flag count must match the child count"
    );
    dec_reduction_optimized_inner(
        s,
        params,
        parent,
        Z_split,
        ell_d,
        Some(cache),
        Some(digit_nonzero),
        ring_linear_forms,
        precomputed_y_ring,
    )
}

fn dec_reduction_optimized_inner<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    ell_d: usize,
    cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
    digit_nonzero: Option<&[bool]>,
    ring_linear_forms: Option<&[crate::superneo_eval::SuperneoRingLinearForm]>,
    precomputed_y_ring: Option<&[Vec<[K; D]>]>,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!Z_split.is_empty(), "PiDEC needs at least one digit witness");
    let d_pad = 1usize << ell_d;
    assert_eq!(d_pad, D.next_power_of_two(), "PiDEC ell_d is not canonical");
    let matrix_count = s.t() + 1;
    assert_eq!(
        parent.y_ring.len(),
        matrix_count,
        "PiDEC requires the identity-first paper matrix count"
    );
    assert_eq!(parent.ct.len(), matrix_count, "PiDEC ct count mismatch");
    if let Some(forms) = ring_linear_forms {
        assert_eq!(forms.len(), s.t(), "PiDEC ring-form count mismatch");
    }
    if let Some(rows) = precomputed_y_ring {
        assert_eq!(rows.len(), Z_split.len(), "PiDEC precomputed child count mismatch");
        assert!(
            rows.iter()
                .all(|child| child.len() == s.t() || child.len() == matrix_count),
            "PiDEC precomputed matrix count mismatch"
        );
    }
    assert!(
        precomputed_y_ring.is_none() || ring_linear_forms.is_none(),
        "PiDEC accepts precomputed rows or ring forms, not both"
    );

    let full_precomputed_rows =
        precomputed_y_ring.is_some_and(|rows| rows.iter().all(|child| child.len() == matrix_count));
    let row_weights = if full_precomputed_rows {
        Vec::new()
    } else {
        neo_ccs::utils::tensor_point_parallel::<K>(&parent.r)
    };
    let streamed = streamed_application_rows(
        s,
        Z_split,
        &row_weights,
        cache,
        digit_nonzero,
        ring_linear_forms,
        precomputed_y_ring,
    );
    let m_in = parent.m_in;
    let build_child = |index: usize| {
        if digit_nonzero.is_some_and(|flags| !flags[index]) {
            return CeClaim {
                adv: None,
                c: parent.c.clone(),
                X: Mat::zero(D, neo_ccs::superneo_public_x_cols(m_in), Ff::ZERO),
                r: parent.r.clone(),
                y_ring: vec![vec![K::ZERO; d_pad]; matrix_count],
                ct: vec![K::ZERO; matrix_count],
                m_in,
                fold_digest: parent.fold_digest,
            };
        }

        let witness = &Z_split[index];
        let X = crate::common::project_x_from_witness_mat(witness, s.m, m_in)
            .unwrap_or_else(|error| panic!("PiDEC X projection failed: {error}"));
        let identity = precomputed_y_ring
            .and_then(|rows| (rows[index].len() == matrix_count).then_some(rows[index][0]))
            .unwrap_or_else(|| {
                let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, s.m)
                    .unwrap_or_else(|error| panic!("PiDEC identity assignment decode failed: {error}"));
                super::paper_joint::identity_ring_mle(&assignment, &row_weights)
            });
        let mut identity = identity.to_vec();
        identity.resize(d_pad, K::ZERO);
        let mut y_ring = Vec::with_capacity(matrix_count);
        y_ring.push(identity);
        y_ring.extend(streamed[index].iter().map(|coefficients| {
            let mut row = coefficients.to_vec();
            row.resize(d_pad, K::ZERO);
            row
        }));
        let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);
        CeClaim {
            adv: None,
            c: parent.c.clone(),
            X,
            r: parent.r.clone(),
            y_ring,
            ct,
            m_in,
            fold_digest: parent.fold_digest,
        }
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let children: Vec<_> = (0..Z_split.len())
        .into_par_iter()
        .map(build_child)
        .collect();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let children: Vec<_> = (0..Z_split.len()).map(build_child).collect();

    let base_f = Ff::from_u64(params.b as u64);
    let base_k = K::from(base_f);
    let y_ok = (0..matrix_count).all(|matrix| {
        let mut sum = vec![K::ZERO; d_pad];
        let mut power = K::ONE;
        for child in &children {
            for (left, right) in sum.iter_mut().zip(&child.y_ring[matrix]) {
                *left += power * *right;
            }
            power *= base_k;
        }
        sum == parent.y_ring[matrix]
    });
    let x_ok = (0..D).all(|row| {
        (0..parent.X.cols()).all(|column| {
            let mut sum = Ff::ZERO;
            let mut power = Ff::ONE;
            for child in &children {
                sum += power * child.X[(row, column)];
                power *= base_f;
            }
            sum == parent.X[(row, column)]
        })
    });
    (children, y_ok, x_ok)
}

#[allow(clippy::too_many_arguments)]
fn streamed_application_rows<Ff>(
    s: &CcsStructure<Ff>,
    Z_split: &[Mat<Ff>],
    row_weights: &[K],
    cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
    digit_nonzero: Option<&[bool]>,
    ring_linear_forms: Option<&[crate::superneo_eval::SuperneoRingLinearForm]>,
    precomputed_y_ring: Option<&[Vec<[K; D]>]>,
) -> Vec<Vec<[K; D]>>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    if let Some(rows) = precomputed_y_ring {
        return rows
            .iter()
            .map(|child| {
                if child.len() == s.t() + 1 {
                    child[1..].to_vec()
                } else {
                    child.clone()
                }
            })
            .collect();
    }
    if let Some(forms) = ring_linear_forms {
        return Z_split
            .iter()
            .enumerate()
            .map(|(index, witness)| {
                if digit_nonzero.is_some_and(|flags| !flags[index]) {
                    return vec![[K::ZERO; D]; s.t()];
                }
                let blocks = crate::superneo_eval::SuperneoZBlocks::from_witness_mat(witness, s.m)
                    .unwrap_or_else(|error| panic!("PiDEC child block view failed: {error}"));
                crate::superneo_eval::eval_ring_linear_forms_real_z_blocks(forms, &blocks)
            })
            .collect();
    }

    let local_cache;
    let cache = match cache {
        Some(cache) => cache,
        None => {
            local_cache = crate::superneo_eval::build_superneo_eval_cache(s)
                .expect("PiDEC requires a SuperNeo-compatible relation");
            &local_cache
        }
    };
    let active: Vec<_> = (0..Z_split.len())
        .filter(|&index| digit_nonzero.is_none_or(|flags| flags[index]))
        .collect();
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let blocks: Vec<_> = active
        .par_iter()
        .map(|&index| {
            crate::superneo_eval::SuperneoZBlocks::from_witness_mat(&Z_split[index], s.m)
                .unwrap_or_else(|error| panic!("PiDEC child block view failed: {error}"))
        })
        .collect();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let blocks: Vec<_> = active
        .iter()
        .map(|&index| {
            crate::superneo_eval::SuperneoZBlocks::from_witness_mat(&Z_split[index], s.m)
                .unwrap_or_else(|error| panic!("PiDEC child block view failed: {error}"))
        })
        .collect();
    let evaluated = cache.eval_ring_linear_forms_for_real_z_blocks(row_weights, s.n.min(row_weights.len()), &blocks);
    let mut rows = vec![vec![[K::ZERO; D]; s.t()]; Z_split.len()];
    for (index, values) in active.into_iter().zip(evaluated) {
        rows[index] = values;
    }
    rows
}
