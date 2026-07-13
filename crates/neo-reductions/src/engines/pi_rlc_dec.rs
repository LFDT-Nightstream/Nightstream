//! RLC/DEC engine trait and implementations.
//!
//! Provides engine implementations for Random Linear Combination (RLC)
//! and Decomposition (DEC) steps that work alongside the Π_CCS engines.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;

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
        sparse: Option<&super::optimized_engine::oracle::SparseCache<F>>,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let (mut children, ok_y, ok_X) = match sparse {
            Some(cache) => super::optimized_engine::dec_reduction_paper_exact_with_sparse_cache::<F>(
                s, params, parent, Z_split, ell_d, cache,
            ),
            None => super::optimized_engine::dec_reduction_paper_exact::<F>(s, params, parent, Z_split, ell_d),
        };

        // Patch children commitments and check c relation.
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_X, ok_c)
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
        let (mut children, ok_y, ok_X) = super::optimized_engine::dec_reduction_paper_exact_with_superneo_cache::<F>(
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
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_X, ok_c)
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
        precomputed_y_ring: Option<&[Vec<[K; neo_math::D]>]>,
    ) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let (mut children, ok_y, ok_X) =
            super::optimized_engine::dec_reduction_paper_exact_with_superneo_cache_and_digit_flags::<F>(
                s,
                params,
                parent,
                Z_split,
                digit_nonzero,
                ell_d,
                superneo_cache,
                ring_linear_forms,
                precomputed_y_ring,
            );

        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_X, ok_c)
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
        // For now, delegate to paper-exact algebra (implemented in optimized_engine).
        let (mut children, ok_y, ok_X) =
            super::optimized_engine::dec_reduction_paper_exact(s, params, parent, Z_split, ell_d);
        // Patch children commitments and check c relation
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_X, ok_c)
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
