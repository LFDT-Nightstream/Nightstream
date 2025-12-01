//! RLC/DEC engine trait and implementations.
//!
//! Provides engine implementations for Random Linear Combination (RLC)
//! and Decomposition (DEC) steps that work alongside the Π_CCS engines.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K};

/// Trait for RLC/DEC algebraic operations over ME instances.
pub trait RlcDecOps {
    /// RLC: compute parent ME and combined witness Z_mix = Σ ρ_i · Z_i.
    /// The `mix_commits` closure must implement the commitment S-action mix: Σ ρ_i · c_i.
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (MeInstance<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt;

    /// DEC: given parent and a provided split Z = Σ b^i · Z_i, build children with correct
    /// commitments and return (children, ok_y, ok_X, ok_c).
    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &MeInstance<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<MeInstance<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt;
}

/// Optimized RLC/DEC implementation.
#[derive(Clone, Debug, Default, Copy)]
pub struct OptimizedRlcDec;

impl RlcDecOps for OptimizedRlcDec {
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (MeInstance<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    {
        let (mut out, Z) = super::optimized_engine::rlc_reduction_paper_exact(
            s, params, rhos, me_inputs, Zs, ell_d,
        );
        let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
        out.c = mix_commits(rhos, &inputs_c);
        (out, Z)
    }

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &MeInstance<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<MeInstance<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let (mut children, ok_y, ok_X) = super::optimized_engine::dec_reduction_paper_exact(
            s, params, parent, Z_split, ell_d,
        );
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
        me_inputs: &[MeInstance<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (MeInstance<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    {
        let (mut out, Z) = super::paper_exact_engine::rlc_reduction_paper_exact(
            s, params, rhos, me_inputs, Zs, ell_d,
        );
        let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
        out.c = mix_commits(rhos, &inputs_c);
        (out, Z)
    }

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &MeInstance<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<MeInstance<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let (mut children, ok_y, ok_X) = super::paper_exact_engine::dec_reduction_paper_exact(
            s, params, parent, Z_split, ell_d,
        );
        // Patch children commitments and check c relation
        for (ch, c) in children.iter_mut().zip(child_commitments.iter()) {
            ch.c = c.clone();
        }
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;
        (children, ok_y, ok_X, ok_c)
    }
}
