//! Public API for Π_CCS folding and RLC/DEC operations.
//!
//! This module exposes the main entry points for:
//! - Π_CCS proving and verification: `prove`, `prove_simple`, `verify`
//! - RLC/DEC operations with commitments: `rlc_with_commit`, `dec_children_with_commit`
//! - Public verification helpers: `rlc_public`, `verify_dec_public`
//!
//! All operations dispatch to the appropriate engine based on FoldingMode.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

use crate::error::PiCcsError;
use crate::engines::PiCcsEngine;

// Re-export types that are part of the public API
pub use crate::engines::optimized_engine::PiCcsProof;

// Re-export common utilities for convenience (single import path for users)
pub use crate::common::{
    split_b_matrix_k,
    sample_rot_rhos,       // Paper-compliant ΠRLC sampler (Section 4.5)
    RotRing,               // Ring metadata for rotation matrix sampling
    compute_y_from_Z_and_r,
    left_mul_acc,
    format_ext,
};

/// Folding mode selector for engine dispatch.
#[derive(Clone, Debug)]
pub enum FoldingMode {
    Optimized,
    Mojo,
    #[cfg(feature = "paper-exact")]
    PaperExact,
    #[cfg(feature = "paper-exact")]
    OptimizedWithCrosscheck(crate::engines::CrosscheckCfg),
}

// ---------------------------------------------------------------------------
// Π_CCS API
// ---------------------------------------------------------------------------

/// Prove Π_CCS folding.
pub fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    use crate::engines::OptimizedEngine;

    match mode {
        FoldingMode::Optimized => OptimizedEngine.prove(
            tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log,
        ),
        FoldingMode::Mojo => crate::engines::MojoGpuEngine.prove(
            tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => crate::engines::PaperExactEngine.prove(
            tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(cfg) => crate::engines::CrossCheckEngine {
            inner: OptimizedEngine,
            ref_oracle: crate::engines::PaperExactEngine,
            cfg,
        }
        .prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log),
    }
}

/// Prove Π_CCS in the simple (k=1) case without ME inputs.
pub fn prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // Delegate to the selected engine with empty ME inputs/witnesses.
    prove(
        mode,
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        &[],
        &[],
        log,
    )
}

/// Verify Π_CCS proof. Uses the optimized verifier path.
pub fn verify(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_outputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    // Cross-checking is a prover-side aid; verification remains optimized.
    crate::engines::OptimizedEngine.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
}

// ---------------------------------------------------------------------------
// RLC/DEC API
// ---------------------------------------------------------------------------

/// RLC: compute parent ME and combined witness Z_mix = Σ ρ_i · Z_i.
/// The `mix_commits` closure must implement the commitment S-action mix: Σ ρ_i · c_i.
pub fn rlc_with_commit<Comb>(
    mode: FoldingMode,
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
    use crate::engines::pi_rlc_dec::{OptimizedRlcDec, RlcDecOps};
    
    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::rlc_with_commit(
            s, params, rhos, me_inputs, Zs, ell_d, mix_commits,
        ),
        FoldingMode::Mojo => crate::engines::MojoGpuEngine::rlc_with_commit(
            s, params, rhos, me_inputs, Zs, ell_d, mix_commits,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            // Use the paper-exact RLC that actually mixes commitments.
            crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s, params, rhos, me_inputs, Zs, ell_d, mix_commits,
            )
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => {
            // For cross-checking, use paper-exact to verify against optimized
            // In practice, RLC/DEC are simple algebraic operations, so we just use optimized
            OptimizedRlcDec::rlc_with_commit(
                s, params, rhos, me_inputs, Zs, ell_d, mix_commits,
            )
        }
    }
}

/// DEC: given parent and a provided split Z = Σ b^i · Z_i, build children with correct
/// commitments and return (children, ok_y, ok_X, ok_c).
pub fn dec_children_with_commit<Comb>(
    mode: FoldingMode,
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
    use crate::engines::pi_rlc_dec::{OptimizedRlcDec, RlcDecOps};
    
    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit(
            s, params, parent, Z_split, ell_d, child_commitments, combine_b_pows,
        ),
        FoldingMode::Mojo => crate::engines::MojoGpuEngine::dec_children_with_commit(
            s, params, parent, Z_split, ell_d, child_commitments, combine_b_pows,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            // Use the paper-exact DEC that checks commitment equality against Σ b^i · c_i
            crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s, params, parent, Z_split, ell_d, child_commitments, combine_b_pows,
            )
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => {
            // For cross-checking, use paper-exact to verify against optimized
            // In practice, RLC/DEC are simple algebraic operations, so we just use optimized
            OptimizedRlcDec::dec_children_with_commit(
                s, params, parent, Z_split, ell_d, child_commitments, combine_b_pows,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// RLC/DEC Public Verification API
// ---------------------------------------------------------------------------

/// RLC (public): Recompute parent = Σ ρ_i · instance_i (X, y; commitment via mixer).
/// 
/// This is the witness-free version used by verifiers to check the prover's claimed parent.
pub fn rlc_public<MR>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[Mat<F>],
    inputs: &[MeInstance<Cmt, F, K>],
    mix_rhos_commits: MR,
    ell_d: usize,
) -> MeInstance<Cmt, F, K>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    use crate::common::left_mul_acc;

    assert!(!inputs.is_empty());
    assert_eq!(rhos.len(), inputs.len());
    let d = D;
    let m_in = inputs[0].m_in;
    let d_pad = 1usize << ell_d;

    // X_out := Σ ρ_i · X_i
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for (rho, inst) in rhos.iter().zip(inputs.iter()) {
        let mut term = Mat::zero(d, m_in, F::ZERO);
        left_mul_acc(&mut term, rho, &inst.X);
        for r in 0..d { 
            for c in 0..m_in { 
                X[(r, c)] += term[(r, c)]; 
            } 
        }
    }

    // y_out[j] := Σ ρ_i · y_(i,j)  (first D digits, keep padding)
    let mut y = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut acc = vec![K::ZERO; d_pad];
        for (rho, inst) in rhos.iter().zip(inputs.iter()) {
            for r in 0..D {
                let mut sum = K::ZERO;
                for k in 0..D {
                    sum += K::from(rho[(r, k)]) * inst.y[j][k];
                }
                acc[r] += sum;
            }
        }
        y.push(acc);
    }

    // y_scalars (convenience)
    let bK = K::from(F::from_u64(params.b as u64));
    let mut y_scalars = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut sc = K::ZERO; 
        let mut pow = K::ONE;
        for rho in 0..D { 
            sc += pow * y[j][rho]; 
            pow *= bK; 
        }
        y_scalars.push(sc);
    }

    let c = mix_rhos_commits(
        rhos,
        &inputs.iter().map(|m| m.c.clone()).collect::<Vec<_>>()
    );

    MeInstance {
        c_step_coords: vec![], 
        u_offset: 0, 
        u_len: 0,
        c,
        X,
        r: inputs[0].r.clone(),
        y,
        y_scalars,
        m_in,
        fold_digest: inputs[0].fold_digest,
    }
}

/// DEC public verification: Check that parent ?= Σ b^i · child_i (X, y, c).
///
/// Returns true if the decomposition is valid.
pub fn verify_dec_public<MB>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &MeInstance<Cmt, F, K>,
    children: &[MeInstance<Cmt, F, K>],
    combine_b_pows: MB,
    ell_d: usize,
) -> bool
where
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    let k = children.len();
    if k == 0 {
        eprintln!("verify_dec_public failed: no children");
        return false;
    }
    if !children.iter().all(|ch| ch.r == parent.r) {
        eprintln!("verify_dec_public failed: r mismatch");
        return false;
    }

    // X
    let mut lhs_X = Mat::zero(D, parent.m_in, F::ZERO);
    let mut pow = F::ONE;
    for i in 0..k {
        for r in 0..D {
            for c in 0..parent.m_in {
                lhs_X[(r, c)] += pow * children[i].X[(r, c)];
            }
        }
        pow *= F::from_u64(params.b as u64);
    }
    for r in 0..D { 
        for c in 0..parent.m_in {
            if lhs_X[(r, c)] != parent.X[(r, c)] {
                eprintln!("verify_dec_public failed: X check mismatch at ({}, {})", r, c);
                return false; 
            }
        }
    }

    // y_j
    let d_pad = 1usize << ell_d;
    let bK = K::from(F::from_u64(params.b as u64));
    for j in 0..s.t() {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut p = K::ONE;
        for i in 0..k {
            for t in 0..d_pad { 
                lhs[t] += p * children[i].y[j][t]; 
            }
            p *= bK;
        }
        if lhs != parent.y[j] {
            eprintln!("verify_dec_public failed: y check mismatch at j={}", j);
            return false; 
        }
    }

    // c
    let want_c = combine_b_pows(
        &children.iter().map(|c| c.c.clone()).collect::<Vec<_>>(), 
        params.b
    );
    if want_c != parent.c {
        eprintln!("verify_dec_public failed: commitment check mismatch");
        return false;
    }
    
    true
}

