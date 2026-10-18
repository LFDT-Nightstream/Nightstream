//! Optimized PiDEC claim construction for the SuperNeo v1.2 relation.
//!
//! This file owns child evaluation and radix recomposition. It does not own
//! commitments, transcript messages, or PaperExact computations.

#![allow(non_snake_case)]

use crate::engines::pi_ccs_joint_protocol::V1_2OutputOpening;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

fn extension_block(values: &[K]) -> [K; D] {
    assert_eq!(values.len(), D, "v1_2 opening must contain exactly D coefficients");
    core::array::from_fn(|coefficient| values[coefficient])
}

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
    dec_reduction_optimized_inner(s, params, parent, Z_split, ell_d, None, None, None)
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
    dec_reduction_optimized_inner(s, params, parent, Z_split, ell_d, Some(cache), None, None)
}

pub fn dec_reduction_optimized_with_digit_flags<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    Z_split: &[Mat<Ff>],
    digit_nonzero: &[bool],
    ell_d: usize,
    cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
    precomputed_openings: Option<&[V1_2OutputOpening]>,
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
        cache,
        Some(digit_nonzero),
        precomputed_openings,
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
    precomputed_openings: Option<&[V1_2OutputOpening]>,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!Z_split.is_empty(), "PiDEC needs at least one digit witness");
    let d_pad = 1usize << ell_d;
    assert_eq!(d_pad, D.next_power_of_two(), "PiDEC ell_d is not canonical");
    let matrix_count = s.t();
    assert_eq!(parent.eval_k.len(), d_pad, "PiDEC Eval_K width mismatch");
    assert_eq!(parent.eval_a.len(), matrix_count, "PiDEC Eval_A count mismatch");
    if let Some(openings) = precomputed_openings {
        assert_eq!(openings.len(), Z_split.len(), "PiDEC precomputed child count mismatch");
        assert!(
            openings
                .iter()
                .all(|child| child.eval_k.len() == D && child.eval_a.len() == matrix_count),
            "PiDEC precomputed v1_2 shape mismatch"
        );
    }

    let computed_openings;
    let openings = match precomputed_openings {
        Some(openings) => openings,
        None => {
            computed_openings = witness_openings(s, Z_split, &parent.r, cache, digit_nonzero);
            &computed_openings
        }
    };
    let m_in = parent.m_in;
    let build_child = |index: usize| {
        if digit_nonzero.is_some_and(|flags| !flags[index]) {
            return CeClaim {
                adv: None,
                c: parent.c.clone(),
                X: Mat::zero(D, parent.X.cols(), Ff::ZERO),
                r: parent.r.clone(),
                eval_k: vec![K::ZERO; d_pad],
                eval_a: vec![vec![K::ZERO; d_pad]; matrix_count],
                m_in,
                fold_digest: parent.fold_digest,
            };
        }

        let witness = &Z_split[index];
        let X = crate::common::project_x_from_witness_mat(witness, s.m, m_in)
            .unwrap_or_else(|error| panic!("PiDEC X projection failed: {error}"));
        let mut eval_k = extension_block(&openings[index].eval_k).to_vec();
        eval_k.resize(d_pad, K::ZERO);
        let eval_a = openings[index]
            .eval_a
            .iter()
            .map(|coefficients| {
                let mut row = extension_block(coefficients).to_vec();
                row.resize(d_pad, K::ZERO);
                row
            })
            .collect();
        CeClaim {
            adv: None,
            c: parent.c.clone(),
            X,
            r: parent.r.clone(),
            eval_k,
            eval_a,
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
    let eval_k_ok = {
        let mut sum = vec![K::ZERO; d_pad];
        let mut power = K::ONE;
        for child in &children {
            for (left, right) in sum.iter_mut().zip(&child.eval_k) {
                *left += power * *right;
            }
            power *= base_k;
        }
        sum == parent.eval_k
    };
    let eval_a_ok = (0..matrix_count).all(|matrix| {
        let mut sum = vec![K::ZERO; d_pad];
        let mut power = K::ONE;
        for child in &children {
            for (left, right) in sum.iter_mut().zip(&child.eval_a[matrix]) {
                *left += power * *right;
            }
            power *= base_k;
        }
        sum == parent.eval_a[matrix]
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
    (children, eval_k_ok && eval_a_ok, x_ok)
}

fn witness_openings<Ff>(
    s: &CcsStructure<Ff>,
    Z_split: &[Mat<Ff>],
    point: &[K],
    cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
    digit_nonzero: Option<&[bool]>,
) -> Vec<V1_2OutputOpening>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let local_cache;
    let cache = match cache {
        Some(cache) => cache,
        None => {
            local_cache = crate::superneo_eval::build_superneo_eval_cache(s)
                .expect("PiDEC requires a SuperNeo-compatible relation");
            &local_cache
        }
    };
    assert_eq!(
        cache.relation_shape(),
        Some((s.n, crate::common::superneo_carrier_width(s.m), s.t())),
        "PiDEC opening cache shape mismatch"
    );
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
    let evaluated = cache
        .eval_real_v1_2_openings(point, &blocks)
        .unwrap_or_else(|error| panic!("PiDEC witness openings failed: {error}"));
    let mut rows = (0..Z_split.len())
        .map(|_| V1_2OutputOpening {
            eval_k: vec![K::ZERO; D],
            eval_a: vec![vec![K::ZERO; D]; s.t()],
        })
        .collect::<Vec<_>>();
    for (index, values) in active.into_iter().zip(evaluated) {
        rows[index] = values;
    }
    rows
}
