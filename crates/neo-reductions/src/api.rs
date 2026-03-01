//! Public API for Π_CCS folding and RLC/DEC operations.
//!
//! This module exposes the main entry points for:
//! - Π_CCS proving and verification: `prove`, `prove_simple`, `verify`
//! - RLC/DEC operations with commitments: `rlc_with_commit`, `dec_children_with_commit`
//! - Public verification helpers: `rlc_public`, `verify_dec_public`
//!
//! All operations dispatch to the appropriate engine based on FoldingMode.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use crate::engines::PiCcsEngine;
use crate::error::PiCcsError;

// Re-export types that are part of the public API
pub use crate::engines::optimized_engine::PiCcsProof;

// Re-export common utilities for convenience (single import path for users)
pub use crate::common::{
    compute_y_from_Z_and_r,
    ct_from_y_ring,
    format_ext,
    left_mul_acc,
    rot_rhos_from_mats,
    rot_rhos_to_mats,
    sample_rot_rhos_n, // Dynamic: samples N rhos with norm bound check
    sample_rot_rhos_n_typed,
    split_b_matrix_k,
    split_b_matrix_k_with_nonzero_flags,
    RotRho,  // typed, validated rotation-matrix challenge
    RotRing, // Ring metadata for rotation matrix sampling
};

#[inline]
fn ensure_superneo_width(s: &CcsStructure<F>) -> Result<(), PiCcsError> {
    if s.m == 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "SuperNeo-only mode requires CCS width m > 0 (got m={})",
            s.m
        )));
    }
    Ok(())
}

#[inline]
fn ell_n_for_ccs(s: &CcsStructure<F>) -> usize {
    s.n.next_power_of_two().max(2).trailing_zeros() as usize
}

#[inline]
fn ell_m_for_ccs(s: &CcsStructure<F>) -> usize {
    s.m.next_power_of_two().max(2).trailing_zeros() as usize
}

fn validate_mcs_claims(label: &str, s: &CcsStructure<F>, mcs_list: &[CcsClaim<Cmt, F>]) -> Result<(), PiCcsError> {
    for (idx, inst) in mcs_list.iter().enumerate() {
        if inst.m_in > s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}].m_in={} exceeds CCS width m={}",
                inst.m_in, s.m
            )));
        }
        if inst.x.len() != inst.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}].x.len()={} does not match m_in={}",
                inst.x.len(),
                inst.m_in
            )));
        }
    }
    Ok(())
}

fn validate_mcs_witnesses(
    label: &str,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
) -> Result<(), PiCcsError> {
    for (idx, (inst, wit)) in mcs_list.iter().zip(mcs_witnesses.iter()).enumerate() {
        let z_len = inst
            .m_in
            .checked_add(wit.w.len())
            .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: mcs_list[{idx}] witness length overflow")))?;
        if z_len != s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}] has m_in + |w| = {} but CCS width is m={}",
                z_len, s.m
            )));
        }
    }
    Ok(())
}

fn validate_ce_claim_shape(label: &str, s: &CcsStructure<F>, ce: &CeClaim<Cmt, F, K>) -> Result<(), PiCcsError> {
    if ce.m_in > s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: m_in={} exceeds CCS width m={}",
            ce.m_in, s.m
        )));
    }
    if ce.X.rows() != D || ce.X.cols() != ce.m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: X has shape {}x{}, expected {}x{}",
            ce.X.rows(),
            ce.X.cols(),
            D,
            ce.m_in
        )));
    }
    let ell_n = ell_n_for_ccs(s);
    if ce.r.len() != ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: r length mismatch (expected {ell_n}, got {})",
            ce.r.len()
        )));
    }
    if ce.y_ring.len() < s.t() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: y_ring.len()={} is smaller than s.t()={}",
            ce.y_ring.len(),
            s.t()
        )));
    }
    if ce.ct.len() < s.t() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: ct.len()={} is smaller than s.t()={}",
            ce.ct.len(),
            s.t()
        )));
    }
    let d_pad = D.next_power_of_two();
    for (j, row) in ce.y_ring.iter().enumerate() {
        if row.len() < D || row.len() > d_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: y_ring[{j}].len()={} must be in [{}, {}]",
                row.len(),
                D,
                d_pad
            )));
        }
    }
    let has_nc_channel = !(ce.s_col.is_empty() && ce.y_zcol.is_empty());
    if has_nc_channel && (ce.s_col.is_empty() || ce.y_zcol.is_empty()) {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: incomplete NC channel, expected both s_col and y_zcol"
        )));
    }
    if has_nc_channel {
        let ell_m = ell_m_for_ccs(s);
        if ce.s_col.len() != ell_m {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: s_col length mismatch (expected {ell_m}, got {})",
                ce.s_col.len()
            )));
        }
        if ce.y_zcol.len() != d_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: y_zcol length mismatch (expected {d_pad}, got {})",
                ce.y_zcol.len()
            )));
        }
    }
    Ok(())
}

fn validate_ce_claims_shape(label: &str, s: &CcsStructure<F>, claims: &[CeClaim<Cmt, F, K>]) -> Result<(), PiCcsError> {
    for (idx, claim) in claims.iter().enumerate() {
        validate_ce_claim_shape(&format!("{label}[{idx}]"), s, claim)?;
    }
    Ok(())
}

fn validate_dec_boundary_inputs(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    z_split: &[Mat<F>],
    child_commitments: &[Cmt],
    ell_d: usize,
) -> Result<(), PiCcsError> {
    ensure_superneo_width(s)?;
    validate_ce_claim_shape("dec_parent", s, parent)?;
    crate::engines::utils::validate_ct_constant_term(s, params, core::slice::from_ref(parent))?;
    if z_split.len() != child_commitments.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC child input mismatch: |Z_split|={} but |child_commitments|={}",
            z_split.len(),
            child_commitments.len()
        )));
    }
    if 1usize.checked_shl(ell_d as u32).is_none() {
        return Err(PiCcsError::InvalidInput(format!("DEC ell_d overflow: ell_d={ell_d}")));
    }
    for (idx, z) in z_split.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(params, z, s.m, &format!("dec: Z_split[{idx}]"))?;
    }
    Ok(())
}

/// Folding mode selector for engine dispatch.
#[derive(Clone, Debug)]
pub enum FoldingMode {
    Optimized,
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
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    use crate::engines::OptimizedEngine;

    ensure_superneo_width(s)?;
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("prove: empty mcs_list".into()));
    }
    if mcs_list.len() != mcs_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "prove: |mcs_list| mismatch (expected {}, got {})",
            mcs_list.len(),
            mcs_witnesses.len()
        )));
    }
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "prove: |me_inputs| mismatch (expected {}, got {})",
            me_inputs.len(),
            me_witnesses.len()
        )));
    }
    validate_mcs_claims("prove", s, mcs_list)?;
    validate_mcs_witnesses("prove", s, mcs_list, mcs_witnesses)?;
    validate_ce_claims_shape("prove: me_inputs", s, me_inputs)?;
    crate::engines::utils::validate_ct_constant_term(s, params, me_inputs)?;
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n_for_ccs(s))?;
    for (idx, wit) in mcs_witnesses.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(
            params,
            &wit.Z,
            s.m,
            &format!("prove: mcs_witnesses[{idx}].Z"),
        )?;
    }
    for (idx, z) in me_witnesses.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(params, z, s.m, &format!("prove: me_witnesses[{idx}]"))?;
    }
    match mode {
        FoldingMode::Optimized => {
            OptimizedEngine.prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log)
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            crate::engines::PaperExactEngine.prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log)
        }
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
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // Delegate to the selected engine with empty ME inputs/witnesses.
    prove(mode, tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}

/// Verify Π_CCS proof using the selected engine mode.
pub fn verify(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    ensure_superneo_width(s)?;
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("verify: empty mcs_list".into()));
    }
    validate_mcs_claims("verify", s, mcs_list)?;
    validate_ce_claims_shape("verify: me_inputs", s, me_inputs)?;
    validate_ce_claims_shape("verify: me_outputs", s, me_outputs)?;
    crate::engines::utils::validate_ct_constant_term(s, params, me_inputs)?;
    crate::engines::utils::validate_ct_constant_term(s, params, me_outputs)?;
    let ell_n = ell_n_for_ccs(s);
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n)?;
    let _ = crate::engines::utils::shared_me_input_r(me_outputs, ell_n)?;
    crate::engines::utils::validate_mcs_output_x_recomposition(params, s.m, mcs_list, me_outputs)?;

    match mode {
        FoldingMode::Optimized => {
            crate::engines::OptimizedEngine.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            crate::engines::PaperExactEngine.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(cfg) => crate::engines::CrossCheckEngine {
            inner: crate::engines::OptimizedEngine,
            ref_oracle: crate::engines::PaperExactEngine,
            cfg,
        }
        .verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof),
    }
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
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    Zs: &[Mat<F>],
    ell_d: usize,
    mix_commits: Comb,
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError>
where
    Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    use crate::engines::pi_rlc_dec::{OptimizedRlcDec, RlcDecOps};

    ensure_superneo_width(s)?;
    if me_inputs.is_empty() {
        return Err(PiCcsError::InvalidInput("rlc_with_commit: empty inputs".into()));
    }
    if rhos.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_with_commit: |rhos| mismatch (expected {}, got {})",
            me_inputs.len(),
            rhos.len()
        )));
    }
    if Zs.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_with_commit: |Zs| mismatch (expected {}, got {})",
            me_inputs.len(),
            Zs.len()
        )));
    }
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);
    validate_ce_claims_shape("rlc_with_commit: me_inputs", s, me_inputs)?;
    crate::engines::utils::validate_ct_constant_term(s, params, me_inputs)?;
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n_for_ccs(s))?;
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(params, z, s.m, &format!("rlc_with_commit: Zs[{idx}]"))?;
    }

    let (out, Z_mix) = match mode {
        FoldingMode::Optimized => {
            OptimizedRlcDec::rlc_with_commit(s, params, &rho_mats, me_inputs, Zs, ell_d, mix_commits)
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            // PaperExact route: call the CE wrapper around paper-core formulas
            // (paper-core stays formula-only; wrapper applies commitment/CE-field patching).
            crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                Zs,
                ell_d,
                mix_commits,
            )
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => {
            // For cross-checking, use paper-exact to verify against optimized
            // In practice, RLC/DEC are simple algebraic operations, so we just use optimized
            OptimizedRlcDec::rlc_with_commit(s, params, &rho_mats, me_inputs, Zs, ell_d, mix_commits)
        }
    };
    Ok((out, Z_mix))
}

/// DEC: given parent and a provided split Z = Σ b^i · Z_i, build children with correct
/// commitments and return (children, ok_y, ok_X, ok_c).
pub fn dec_children_with_commit<Comb>(
    mode: FoldingMode,
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
    use crate::engines::pi_rlc_dec::{OptimizedRlcDec, RlcDecOps};
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            // PaperExact route: call the CE wrapper around paper-core DEC formulas.
            crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                combine_b_pows,
            )
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => {
            // For cross-checking, use paper-exact to verify against optimized
            // In practice, RLC/DEC are simple algebraic operations, so we just use optimized
            OptimizedRlcDec::dec_children_with_commit(
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
}

/// DEC (cached): same as `dec_children_with_commit`, but can reuse a caller-provided CSC cache.
///
/// This is intended for high-level coordinators (e.g. neo-fold) that already build
/// a `SparseCache` for the optimized CCS oracle, and want to avoid re-scanning dense matrices
/// during Π_DEC.
pub fn dec_children_with_commit_cached<Comb>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    Z_split: &[Mat<F>],
    ell_d: usize,
    child_commitments: &[Cmt],
    combine_b_pows: Comb,
    sparse: Option<&crate::engines::optimized_engine::oracle::SparseCache<F>>,
) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_cached input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit_cached(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
            sparse,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => OptimizedRlcDec::dec_children_with_commit_cached(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
            sparse,
        ),
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
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    mix_rhos_commits: MR,
    ell_d: usize,
) -> Result<CeClaim<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    use crate::common::left_mul_acc;

    ensure_superneo_width(s)?;
    if inputs.is_empty() {
        return Err(PiCcsError::InvalidInput("rlc_public: empty inputs".into()));
    }
    if rhos.len() != inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_public: |rhos| mismatch (expected {}, got {})",
            inputs.len(),
            rhos.len()
        )));
    }
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);
    for (idx, inst) in inputs.iter().enumerate() {
        if inst.m_in > s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "rlc_public: inputs[{idx}].m_in={} exceeds CCS width m={}",
                inst.m_in, s.m
            )));
        }
    }
    crate::engines::utils::validate_ct_constant_term(s, params, inputs)?;
    let _ = crate::engines::utils::shared_me_input_r(inputs, inputs[0].r.len())?;
    let d = D;
    let m_in = inputs[0].m_in;
    let d_pad = 1usize
        .checked_shl(ell_d as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("rlc_public: 2^ell_d overflow".into()))?;
    let t = inputs[0].y_ring.len();
    let aux_len = inputs[0].aux_openings.len();
    if t < s.t() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_public: ME input y.len() must be >= s.t() (got {}, s.t()={})",
            t,
            s.t()
        )));
    }
    for (idx, inst) in inputs.iter().enumerate() {
        if inst.m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "rlc_public: m_in mismatch at input {idx} (expected {m_in}, got {})",
                inst.m_in
            )));
        }
        if inst.X.rows() != D || inst.X.cols() != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "rlc_public: X shape mismatch at input {idx} (got {}x{}, expected {}x{})",
                inst.X.rows(),
                inst.X.cols(),
                D,
                m_in
            )));
        }
        if inst.y_ring.len() != t {
            return Err(PiCcsError::InvalidInput(format!(
                "rlc_public: y.len mismatch at input {idx} (expected {t}, got {})",
                inst.y_ring.len()
            )));
        }
        if inst.aux_openings.len() != aux_len {
            return Err(PiCcsError::InvalidInput(format!(
                "rlc_public: aux_openings.len mismatch at input {idx} (expected {aux_len}, got {})",
                inst.aux_openings.len()
            )));
        }
        for (j, row) in inst.y_ring.iter().enumerate() {
            if row.len() < D || row.len() > d_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "rlc_public: y[{j}].len()={} at input {idx}, expected in [{}, {}]",
                    row.len(),
                    D,
                    d_pad
                )));
            }
        }
    }

    // X_out := Σ ρ_i · X_i
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for (rho, inst) in rho_mats.iter().zip(inputs.iter()) {
        left_mul_acc(&mut X, rho, &inst.X);
    }

    // Precompute ρ entries in K once, laid out column-major by logical k:
    // [rho(0,0)..rho(d-1,0), rho(0,1)..rho(d-1,1), ...].
    // This gives contiguous access in the inner r-loop.
    let rho_k_mats: Vec<Vec<K>> = rho_mats
        .iter()
        .map(|rho| {
            let mut flat = Vec::with_capacity(d * d);
            for k in 0..d {
                for r in 0..d {
                    flat.push(K::from(rho[(r, k)]));
                }
            }
            flat
        })
        .collect();

    // y_out[j] := Σ ρ_i · y_(i,j)  (first D digits, keep padding)
    let mut y_ring = vec![vec![K::ZERO; d_pad]; t];
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && t >= 128;
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let allow_parallel = false;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if allow_parallel {
        y_ring.par_iter_mut().enumerate().for_each(|(j, acc)| {
            for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
                let src = &inst.y_ring[j];
                for k in 0..d {
                    let yk = src[k];
                    if yk == K::ZERO {
                        continue;
                    }
                    let col_off = k * d;
                    let col = &rho_k[col_off..col_off + d];
                    for r in 0..d {
                        acc[r] += col[r] * yk;
                    }
                }
            }
        });
    } else {
        for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
            for (j, acc) in y_ring.iter_mut().enumerate() {
                let src = &inst.y_ring[j];
                for k in 0..d {
                    let yk = src[k];
                    if yk == K::ZERO {
                        continue;
                    }
                    let col_off = k * d;
                    let col = &rho_k[col_off..col_off + d];
                    for r in 0..d {
                        acc[r] += col[r] * yk;
                    }
                }
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
            for (j, acc) in y_ring.iter_mut().enumerate() {
                let src = &inst.y_ring[j];
                for k in 0..d {
                    let yk = src[k];
                    if yk == K::ZERO {
                        continue;
                    }
                    let col_off = k * d;
                    let col = &rho_k[col_off..col_off + d];
                    for r in 0..d {
                        acc[r] += col[r] * yk;
                    }
                }
            }
        }
    }

    // Optional NC channel: y_zcol := Σ ρ_i · y_zcol_i (same mixing as y_j, but independent of t).
    let wants_nc_channel = inputs
        .iter()
        .any(|m| !(m.s_col.is_empty() && m.y_zcol.is_empty()));
    let y_zcol = if wants_nc_channel {
        if inputs[0].s_col.is_empty() || inputs[0].y_zcol.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "rlc_public: incomplete NC channel on input 0 (expected both s_col and y_zcol)".into(),
            ));
        }
        for (idx, inst) in inputs.iter().enumerate() {
            if inst.s_col.is_empty() || inst.y_zcol.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "rlc_public: incomplete NC channel at input {idx} (expected both s_col and y_zcol)"
                )));
            }
            if inst.s_col != inputs[0].s_col {
                return Err(PiCcsError::InvalidInput(format!(
                    "rlc_public: s_col mismatch at input {idx}"
                )));
            }
            if inst.y_zcol.len() != d_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "rlc_public: y_zcol len mismatch at input {idx} (expected {d_pad}, got {})",
                    inst.y_zcol.len()
                )));
            }
        }

        let mut acc = vec![K::ZERO; d_pad];
        for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
            for k in 0..d {
                let yk = inst.y_zcol[k];
                if yk == K::ZERO {
                    continue;
                }
                let col_off = k * d;
                let col = &rho_k[col_off..col_off + d];
                for r in 0..d {
                    acc[r] += col[r] * yk;
                }
            }
        }
        acc
    } else {
        // Legacy: NC channel not present.
        Vec::new()
    };

    let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);

    let c = mix_rhos_commits(&rho_mats, &inputs.iter().map(|m| m.c.clone()).collect::<Vec<_>>());

    // aux_openings: field-linear mix using the scalar projection of each ρ_i.
    // We currently use the (0,0) entry, which corresponds to the constant-coefficient action.
    let mut aux_openings = vec![K::ZERO; aux_len];
    for (rho, inst) in rho_mats.iter().zip(inputs.iter()) {
        let w = K::from(rho[(0, 0)]);
        for (dst, src) in aux_openings.iter_mut().zip(inst.aux_openings.iter()) {
            *dst += w * *src;
        }
    }

    Ok(CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r: inputs[0].r.clone(),
        s_col: inputs[0].s_col.clone(),
        y_ring,
        ct,
        aux_openings,
        y_zcol,
        m_in,
        fold_digest: inputs[0].fold_digest,
    })
}

/// DEC public verification: Check that parent ?= Σ b^i · child_i (X, y, c).
///
/// Returns true if the decomposition is valid.
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
        return fail(format!("SuperNeo-only mode requires m > 0 (got m={})", s.m));
    }
    let k = children.len();
    if k == 0 {
        return fail("no children");
    }

    if let Err(e) = crate::engines::utils::validate_ct_constant_term(s, params, core::slice::from_ref(parent)) {
        return fail(e);
    }
    if let Err(e) = crate::engines::utils::validate_ct_constant_term(s, params, children) {
        return fail(e);
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
    // Optional NC channel: enforce consistency + decomposition for (s_col, y_zcol).
    let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
    if want_nc_channel && (parent.s_col.is_empty() || parent.y_zcol.is_empty()) {
        eprintln!("verify_dec_public failed: incomplete NC channel on parent");
        return false;
    }
    if !want_nc_channel
        && children
            .iter()
            .any(|ch| !(ch.s_col.is_empty() && ch.y_zcol.is_empty()))
    {
        eprintln!("verify_dec_public failed: unexpected NC channel on child");
        return false;
    }

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
        if ch.ct.len() != parent.ct.len() {
            eprintln!(
                "verify_dec_public failed: child ct.len mismatch (child {} has {}, expected {})",
                idx,
                ch.ct.len(),
                parent.ct.len()
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
            "verify_dec_public failed: parent ct.len()={} != y.len()={}",
            parent.ct.len(),
            t
        );
        return false;
    }

    // y_j / X / y_zcol decomposition is checked over the same radix-b ladder.
    let Some(d_pad) = 1usize.checked_shl(ell_d as u32) else {
        eprintln!("verify_dec_public failed: 2^ell_d overflow");
        return false;
    };
    let bF = F::from_u64(params.b as u64);
    let bK = K::from(F::from_u64(params.b as u64));

    // X
    for rho in 0..D {
        for c in 0..parent.m_in {
            let mut lhs = F::ZERO;
            let mut p = F::ONE;
            for child in children.iter().take(k) {
                lhs += p * child.X[(rho, c)];
                p *= bF;
            }
            if lhs != parent.X[(rho, c)] {
                eprintln!("verify_dec_public failed: X check mismatch at ({rho}, {c})");
                return false;
            }
        }
    }

    if want_nc_channel {
        if parent.y_zcol.len() != d_pad {
            eprintln!(
                "verify_dec_public failed: parent y_zcol.len()={} != d_pad={}",
                parent.y_zcol.len(),
                d_pad
            );
            return false;
        }
        for (idx, ch) in children.iter().enumerate() {
            if ch.s_col != parent.s_col {
                eprintln!("verify_dec_public failed: s_col mismatch");
                return false;
            }
            if ch.y_zcol.len() != d_pad {
                eprintln!(
                    "verify_dec_public failed: child y_zcol.len() mismatch at child {} (got {}, expected {})",
                    idx,
                    ch.y_zcol.len(),
                    d_pad
                );
                return false;
            }
        }
    }

    for j in 0..t {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut p = K::ONE;
        for i in 0..k {
            if children[i].y_ring[j].len() != d_pad {
                eprintln!("verify_dec_public failed: child y[{}] len mismatch at j={}", i, j);
                return false;
            }
            for t in 0..d_pad {
                lhs[t] += p * children[i].y_ring[j][t];
            }
            p *= bK;
        }
        if parent.y_ring[j].len() != d_pad {
            eprintln!("verify_dec_public failed: parent y[j] len mismatch at j={j}");
            return false;
        }
        if lhs != parent.y_ring[j] {
            eprintln!("verify_dec_public failed: y check mismatch at j={}", j);
            return false;
        }
    }

    // ct_j: scalar decomposition must also hold (covers any extra appended openings).
    for j in 0..t {
        let mut lhs = K::ZERO;
        let mut p = K::ONE;
        for i in 0..k {
            lhs += p * children[i].ct[j];
            p *= bK;
        }
        if lhs != parent.ct[j] {
            eprintln!("verify_dec_public failed: ct check mismatch at j={j}");
            return false;
        }
    }

    let _ = (want_nc_channel, bK, d_pad);

    // aux_openings: field-linear decomposition must hold as well.
    for j in 0..parent.aux_openings.len() {
        let mut lhs = K::ZERO;
        let mut p = K::ONE;
        for child in children.iter().take(k) {
            lhs += p * child.aux_openings[j];
            p *= bK;
        }
        if lhs != parent.aux_openings[j] {
            eprintln!("verify_dec_public failed: aux_openings check mismatch at j={j}");
            return false;
        }
    }

    // c
    let want_c = combine_b_pows(&children.iter().map(|c| c.c.clone()).collect::<Vec<_>>(), params.b);
    if want_c != parent.c {
        eprintln!("verify_dec_public failed: commitment check mismatch");
        return false;
    }

    true
}
