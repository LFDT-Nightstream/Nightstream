//! RLC/DEC engine trait and implementations.

#![allow(non_snake_case)]

use super::{DecOutcome, RlcMixer};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn canonical_dec_split(split: &[Mat<F>], base: u32) -> bool {
    if split.is_empty() || base < 2 {
        return false;
    }
    let rows = split[0].rows();
    let columns = split[0].cols();
    if split
        .iter()
        .any(|matrix| matrix.rows() != rows || matrix.cols() != columns)
    {
        return false;
    }

    let mut reconstructed = Mat::zero(rows, columns, F::ZERO);
    let base_u32 = base;
    let base = F::from_u64(base_u32 as u64);
    let mut power = F::ONE;
    for digit in split {
        for row in 0..rows {
            for column in 0..columns {
                reconstructed[(row, column)] += power * digit[(row, column)];
            }
        }
        power *= base;
    }

    crate::common::split_b_matrix_k(&reconstructed, split.len(), base_u32).is_ok_and(|expected| expected == split)
}

fn digit_flags_match(split: &[Mat<F>], digit_nonzero: &[bool]) -> bool {
    split.len() == digit_nonzero.len()
        && split.iter().zip(digit_nonzero).all(|(matrix, &flag)| {
            flag == (0..matrix.rows()).any(|row| (0..matrix.cols()).any(|column| matrix[(row, column)] != F::ZERO))
        })
}

pub trait RlcDecOps {
    fn rlc_with_commit(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: RlcMixer,
    ) -> (CeClaim<Cmt, F, K>, Mat<F>);

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> DecOutcome
    where
        Comb: Fn(&[Cmt], u32) -> Cmt;
}

#[derive(Clone, Debug, Default, Copy)]
pub struct OptimizedRlcDec;

impl OptimizedRlcDec {
    pub fn dec_children_with_commit_superneo_cached_with_digit_flags<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        digit_nonzero: &[bool],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
        superneo_cache: &crate::superneo_eval::SuperneoEvalCache,
        precomputed_openings: Option<&[neo_ccs::V1_2Evaluations<K>]>,
    ) -> DecOutcome
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return DecOutcome::rejected();
        }
        let split_valid = canonical_dec_split(Z_split, params.b) && digit_flags_match(Z_split, digit_nonzero);
        let (mut children, ok_y, ok_X) = super::optimized_engine::dec_reduction_optimized_with_digit_flags::<F>(
            s,
            params,
            parent,
            Z_split,
            digit_nonzero,
            ell_d,
            Some(superneo_cache),
            precomputed_openings,
        );

        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        DecOutcome {
            children,
            ok_y: split_valid && ok_y,
            ok_x: split_valid && ok_X,
            ok_c,
        }
    }

    pub fn dec_children_with_commit_superneo_cached_from_trusted_split_digits<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        split: &crate::common::SplitDigits,
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
        superneo_cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
        precomputed_openings: Option<&[neo_ccs::V1_2Evaluations<K>]>,
    ) -> DecOutcome
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let z_split = split.planes();
        if z_split.len() != params.k_rho as usize || child_commitments.len() != z_split.len() {
            return DecOutcome::rejected();
        }
        let (mut children, ok_y, ok_x) = super::optimized_engine::dec_reduction_optimized_with_digit_flags::<F>(
            s,
            params,
            parent,
            z_split,
            split.nonzero(),
            ell_d,
            superneo_cache,
            precomputed_openings,
        );
        for (child, commitment) in children.iter_mut().zip(child_commitments) {
            child.c = commitment.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        DecOutcome {
            children,
            ok_y,
            ok_x,
            ok_c,
        }
    }
}

impl RlcDecOps for OptimizedRlcDec {
    fn rlc_with_commit(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: RlcMixer,
    ) -> (CeClaim<Cmt, F, K>, Mat<F>) {
        let z_refs: Vec<&Mat<F>> = Zs.iter().collect();
        super::optimized_engine::rlc_reduction_optimized_with_commit_mix(
            s,
            params,
            rhos,
            me_inputs,
            &z_refs,
            ell_d,
            mix_commits,
        )
    }

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> DecOutcome
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return DecOutcome::rejected();
        }
        let split_valid = canonical_dec_split(Z_split, params.b);
        let (mut children, ok_y, ok_X) =
            super::optimized_engine::dec_reduction_optimized(s, params, parent, Z_split, ell_d);
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        DecOutcome {
            children,
            ok_y: split_valid && ok_y,
            ok_x: split_valid && ok_X,
            ok_c,
        }
    }
}
