//! RLC/DEC engine trait and implementations.
//!
//! Provides engine implementations for Random Linear Combination (RLC)
//! and Decomposition (DEC) steps that work alongside the Π_CCS engines.

#![allow(non_snake_case)]

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

/// Trait for RLC/DEC algebraic operations over ME instances.
pub trait RlcDecOps {
    /// RLC: compute parent ME and combined witness Z_mix = Σ ρ_i · Z_i.
    /// The `mix_commits` closure must implement the commitment S-action mix: Σ ρ_i · c_i.
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (CeClaim<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt;

    /// DEC: given parent and a provided split Z = Σ b^i · Z_i, build children with correct
    /// commitments and return (children, ok_y, ok_X, ok_c).
    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt;
}

/// Optimized RLC/DEC implementation.
#[derive(Clone, Debug, Default, Copy)]
pub struct OptimizedRlcDec;

impl OptimizedRlcDec {
    /// Optimized DEC that can reuse a caller-provided CSC cache to avoid dense n×m scans.
    pub fn dec_children_with_commit_cached<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
        sparse: Option<&super::optimized_engine::SparseCache<F>>,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return (Vec::new(), false, false, false);
        }
        let split_valid = canonical_dec_split(Z_split, params.b);
        let _ = sparse;
        let (mut children, ok_y, ok_X) =
            super::optimized_engine::dec_reduction_optimized::<F>(s, params, parent, Z_split, ell_d);

        // Patch children commitments and check c relation.
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        (children, split_valid && ok_y, split_valid && ok_X, ok_c)
    }

    /// Optimized DEC that reuses a caller-provided SuperNeo eval cache.
    pub fn dec_children_with_commit_superneo_cached<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
        superneo_cache: &crate::superneo_eval::SuperneoEvalCache,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return (Vec::new(), false, false, false);
        }
        let split_valid = canonical_dec_split(Z_split, params.b);
        let (mut children, ok_y, ok_X) = super::optimized_engine::dec_reduction_optimized_with_superneo_cache::<F>(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            superneo_cache,
        );

        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        (children, split_valid && ok_y, split_valid && ok_X, ok_c)
    }

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
        ring_linear_forms: Option<&[crate::superneo_eval::SuperneoRingLinearForm]>,
        precomputed_openings: Option<&[neo_ccs::V1_1Evaluations<K>]>,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return (Vec::new(), false, false, false);
        }
        let split_valid = canonical_dec_split(Z_split, params.b) && digit_flags_match(Z_split, digit_nonzero);
        let (mut children, ok_y, ok_X) = super::optimized_engine::dec_reduction_optimized_with_digit_flags::<F>(
            s,
            params,
            parent,
            Z_split,
            digit_nonzero,
            ell_d,
            superneo_cache,
            ring_linear_forms,
            precomputed_openings,
        );

        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        (children, split_valid && ok_y, split_valid && ok_X, ok_c)
    }

    /// Build PiDEC children from the exact digit planes and flags returned by
    /// `split_b_matrix_k_with_nonzero_flags`.
    ///
    /// The caller owns the canonical split boundary. This path keeps the
    /// public y, X, and commitment recomposition checks, but it does not
    /// reconstruct and split the full witness a second time.
    #[allow(clippy::too_many_arguments)]
    pub fn dec_children_with_commit_superneo_cached_from_trusted_split_digits<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        z_split: &[Mat<F>],
        digit_nonzero: &[bool],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
        superneo_cache: &crate::superneo_eval::SuperneoEvalCache,
        ring_linear_forms: Option<&[crate::superneo_eval::SuperneoRingLinearForm]>,
        precomputed_openings: Option<&[neo_ccs::V1_1Evaluations<K>]>,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if z_split.len() != params.k_rho as usize
            || digit_nonzero.len() != z_split.len()
            || child_commitments.len() != z_split.len()
        {
            return (Vec::new(), false, false, false);
        }
        let (mut children, ok_y, ok_x) = super::optimized_engine::dec_reduction_optimized_with_digit_flags::<F>(
            s,
            params,
            parent,
            z_split,
            digit_nonzero,
            ell_d,
            superneo_cache,
            ring_linear_forms,
            precomputed_openings,
        );
        for (child, commitment) in children.iter_mut().zip(child_commitments) {
            child.c = commitment.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_x, ok_c)
    }
}

impl RlcDecOps for OptimizedRlcDec {
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (CeClaim<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    {
        let (mut out, Z) = super::optimized_engine::rlc_reduction_optimized(s, params, rhos, me_inputs, Zs, ell_d);
        let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
        out.c = mix_commits(rhos, &inputs_c);
        (out, Z)
    }

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &CeClaim<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        if Z_split.len() != params.k_rho as usize || child_commitments.len() != Z_split.len() {
            return (Vec::new(), false, false, false);
        }
        let split_valid = canonical_dec_split(Z_split, params.b);
        let (mut children, ok_y, ok_X) =
            super::optimized_engine::dec_reduction_optimized(s, params, parent, Z_split, ell_d);
        // Patch children commitments and check c relation
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = split_valid && combine_b_pows(child_commitments, params.b) == parent.c;
        (children, split_valid && ok_y, split_valid && ok_X, ok_c)
    }
}

/// Paper-exact algebraic implementation.
#[cfg(feature = "paper-exact")]
#[derive(Clone, Debug, Default, Copy)]
pub struct PaperExactRlcDec;

#[cfg(feature = "paper-exact")]
impl RlcDecOps for PaperExactRlcDec {
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (CeClaim<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    {
        // Keep PaperExact auditable: route through wrapper entrypoint so paper-core formulas stay CE-free.
        super::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
            s,
            params,
            rhos,
            me_inputs,
            Zs,
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
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        // Keep PaperExact auditable: wrapper applies CE-only patching over paper-core DEC outputs.
        super::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
        )
    }
}
