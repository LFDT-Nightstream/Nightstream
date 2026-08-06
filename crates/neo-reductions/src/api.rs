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
use std::time::Instant;

use crate::engines::PiCcsEngine;
use crate::error::PiCcsError;

mod accelerator;
pub use accelerator::{rlc_with_commit_refs_and_resident_witness, rlc_with_commit_refs_and_witness_mix};
mod dec;
pub use dec::verify_dec_public;
mod validation;
pub(crate) use validation::{
    checked_superneo_d_pad, ell_n_for_ccs, ensure_superneo_width, validate_ce_claim_shape, validate_ce_claims_shape,
    validate_dec_boundary_inputs, validate_dec_boundary_inputs_from_trusted_split, validate_mcs_claims,
    validate_mcs_witnesses, validate_pi_ccs_outputs, validate_rlc_batch_compatibility,
};

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

/// Folding mode selector for engine dispatch.
#[derive(Clone, Debug)]
pub enum FoldingMode {
    Optimized,
    #[cfg(feature = "paper-exact")]
    PaperExact,
    #[cfg(feature = "paper-exact")]
    OptimizedWithCrosscheck,
}

fn validate_selected_reduction_claims(
    _mode: &FoldingMode,
    label: &str,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    validate_pi_ccs_outputs(label, structure, claims)
}

#[cfg(feature = "paper-exact")]
fn require_rlc_crosscheck(
    optimized: (CeClaim<Cmt, F, K>, Mat<F>),
    reference: (CeClaim<Cmt, F, K>, Mat<F>),
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError> {
    if optimized != reference {
        return Err(PiCcsError::ProtocolError(
            "PiRLC optimized and PaperExact results differ".into(),
        ));
    }
    Ok(optimized)
}

#[cfg(feature = "paper-exact")]
fn require_dec_crosscheck(
    optimized: (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool),
    reference: (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool),
) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool) {
    assert_eq!(optimized, reference, "PiDEC optimized and PaperExact results differ");
    optimized
}

// ---------------------------------------------------------------------------
// Π_CCS API
// ---------------------------------------------------------------------------

/// Prove Π_CCS folding.
pub fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt> + Sync>(
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
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n_for_ccs(s))?;
    for (idx, wit) in mcs_witnesses.iter().enumerate() {
        crate::common::validate_fresh_witness_tail_zero(&wit.Z, s.m, &format!("prove: mcs_witnesses[{idx}].Z"))?;
        crate::common::validate_packed_witness_nc_alphabet(
            params,
            &wit.Z,
            s.m,
            &format!("prove: mcs_witnesses[{idx}].Z"),
        )?;
    }
    for (idx, z) in me_witnesses.iter().enumerate() {
        crate::common::validate_packed_witness_nc_alphabet(params, z, s.m, &format!("prove: me_witnesses[{idx}]"))?;
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
        FoldingMode::OptimizedWithCrosscheck => crate::engines::CrossCheckEngine {
            inner: OptimizedEngine,
            ref_oracle: crate::engines::PaperExactEngine,
        }
        .prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log),
    }
}

/// Prove Π_CCS in the simple (k=1) case without ME inputs.
pub fn prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt> + Sync>(
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
    validate_pi_ccs_outputs("verify: me_outputs", s, me_outputs)?;
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
        FoldingMode::OptimizedWithCrosscheck => crate::engines::CrossCheckEngine {
            inner: crate::engines::OptimizedEngine,
            ref_oracle: crate::engines::PaperExactEngine,
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
    validate_selected_reduction_claims(&mode, "rlc_with_commit: selected inputs", s, me_inputs)?;
    validate_rlc_batch_compatibility("rlc_with_commit", params, me_inputs)?;
    checked_superneo_d_pad("rlc_with_commit ell_d", ell_d)?;
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::rlc_with_commit(s, params, &rho_mats, me_inputs, Zs, ell_d, &mix_commits);
            let reference = crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                Zs,
                ell_d,
                &mix_commits,
            );
            return require_rlc_crosscheck(optimized, reference);
        }
    };
    Ok((out, Z_mix))
}

/// Borrowed-matrix variant of [`rlc_with_commit`].
///
/// The optimized path keeps witness matrices borrowed all the way through Π_RLC so callers do not
/// need to clone hot packed witness mats just to satisfy the API boundary. Paper-exact callers still
/// materialize owned mats behind this function when necessary.
pub fn rlc_with_commit_refs<Comb>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    Zs: &[&Mat<F>],
    ell_d: usize,
    mix_commits: Comb,
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError>
where
    Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    ensure_superneo_width(s)?;
    if me_inputs.is_empty() {
        return Err(PiCcsError::InvalidInput("rlc_with_commit_refs: empty inputs".into()));
    }
    if rhos.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_with_commit_refs: |rhos| mismatch (expected {}, got {})",
            me_inputs.len(),
            rhos.len()
        )));
    }
    if Zs.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "rlc_with_commit_refs: |Zs| mismatch (expected {}, got {})",
            me_inputs.len(),
            Zs.len()
        )));
    }
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);
    validate_ce_claims_shape("rlc_with_commit_refs: me_inputs", s, me_inputs)?;
    validate_selected_reduction_claims(&mode, "rlc_with_commit_refs: selected inputs", s, me_inputs)?;
    validate_rlc_batch_compatibility("rlc_with_commit_refs", params, me_inputs)?;
    checked_superneo_d_pad("rlc_with_commit_refs ell_d", ell_d)?;
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n_for_ccs(s))?;
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(params, z, s.m, &format!("rlc_with_commit_refs: Zs[{idx}]"))?;
    }

    let (out, Z_mix) = match mode {
        FoldingMode::Optimized => crate::engines::optimized_engine::rlc_reduction_optimized_with_commit_mix(
            s,
            params,
            &rho_mats,
            me_inputs,
            Zs,
            ell_d,
            mix_commits,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => {
            let owned_zs: Vec<Mat<F>> = Zs.iter().map(|z| (*z).clone()).collect();
            crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                &owned_zs,
                ell_d,
                mix_commits,
            )
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = crate::engines::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                Zs,
                ell_d,
                &mix_commits,
            );
            let owned_zs: Vec<Mat<F>> = Zs.iter().map(|z| (*z).clone()).collect();
            let reference = crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                &owned_zs,
                ell_d,
                &mix_commits,
            );
            return require_rlc_crosscheck(optimized, reference);
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
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit input validation failed: {error}");
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::dec_children_with_commit(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            let reference = crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            require_dec_crosscheck(optimized, reference)
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
    sparse: Option<&crate::engines::optimized_engine::SparseCache<F>>,
) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_cached input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_cached: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit_cached input validation failed: {error}");
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::dec_children_with_commit_cached(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
                sparse,
            );
            let reference = crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            require_dec_crosscheck(optimized, reference)
        }
    }
}

/// DEC (SuperNeo-cached): same as `dec_children_with_commit`, but reuses a
/// caller-provided SuperNeo eval cache.
///
/// This is intended for high-level coordinators that already precompute the
/// verifier-owned optimized structure cache and want Π_DEC to avoid rebuilding
/// the SuperNeo transformed-matrix view on every prover call.
pub fn dec_children_with_commit_superneo_cached<Comb>(
    mode: FoldingMode,
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
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_superneo_cached input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_superneo_cached: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit_superneo_cached input validation failed: {error}");
        return (Vec::new(), false, false, false);
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit_superneo_cached(
            s,
            params,
            parent,
            Z_split,
            ell_d,
            child_commitments,
            combine_b_pows,
            superneo_cache,
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::dec_children_with_commit_superneo_cached(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
                superneo_cache,
            );
            let reference = crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            require_dec_crosscheck(optimized, reference)
        }
    }
}

pub fn dec_children_with_commit_superneo_cached_with_digit_flags<Comb>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    Z_split: &[Mat<F>],
    digit_nonzero: &[bool],
    ell_d: usize,
    child_commitments: &[Cmt],
    combine_b_pows: Comb,
    superneo_cache: &crate::superneo_eval::SuperneoEvalCache,
) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if digit_nonzero.len() != Z_split.len() {
        eprintln!(
            "dec_children_with_commit_superneo_cached_with_digit_flags input validation failed: digit flag count {} != split witness count {}",
            digit_nonzero.len(),
            Z_split.len()
        );
        return (Vec::new(), false, false, false);
    }
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_superneo_cached_with_digit_flags input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_superneo_cached_with_digit_flags: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit_superneo_cached_with_digit_flags input validation failed: {error}");
        return (Vec::new(), false, false, false);
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit_superneo_cached_with_digit_flags(
            s,
            params,
            parent,
            Z_split,
            digit_nonzero,
            ell_d,
            child_commitments,
            combine_b_pows,
            superneo_cache,
            None,
            None,
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::dec_children_with_commit_superneo_cached_with_digit_flags(
                s,
                params,
                parent,
                Z_split,
                digit_nonzero,
                ell_d,
                child_commitments,
                &combine_b_pows,
                superneo_cache,
                None,
                None,
            );
            let reference = crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            require_dec_crosscheck(optimized, reference)
        }
    }
}

/// DEC for digit planes produced by [`split_b_matrix_k_with_nonzero_flags`].
///
/// This is the same computation as
/// [`dec_children_with_commit_superneo_cached_with_digit_flags`], but it
/// skips the per-entry NC-range scan over `Z_split`: the split routine
/// has just decomposed the parent witness into balanced base-`b` digit
/// planes, so every child entry is already in range. We still validate
/// parent shape, child count, `ell_d`, and every split matrix's
/// SuperNeo packed shape. Callers with arbitrary `Z_split` must use the
/// checked API above.
pub fn dec_children_with_commit_superneo_cached_from_trusted_split_digits<Comb>(
    mode: FoldingMode,
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
    precomputed_y_ring: Option<&[Vec<[K; D]>]>,
) -> (Vec<CeClaim<Cmt, F, K>>, bool, bool, bool)
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if digit_nonzero.len() != Z_split.len() {
        eprintln!(
            "dec_children_with_commit_superneo_cached_from_trusted_split_digits input validation failed: digit flag count {} != split witness count {}",
            digit_nonzero.len(),
            Z_split.len()
        );
        return (Vec::new(), false, false, false);
    }
    if let Err(e) =
        validate_dec_boundary_inputs_from_trusted_split(s, params, parent, Z_split, child_commitments, ell_d)
    {
        eprintln!("dec_children_with_commit_superneo_cached_from_trusted_split_digits input validation failed: {e}");
        return (Vec::new(), false, false, false);
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_superneo_cached_from_trusted_split_digits: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!(
            "dec_children_with_commit_superneo_cached_from_trusted_split_digits input validation failed: {error}"
        );
        return (Vec::new(), false, false, false);
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            s,
            params,
            parent,
            Z_split,
            digit_nonzero,
            ell_d,
            child_commitments,
            combine_b_pows,
            superneo_cache,
            ring_linear_forms,
            precomputed_y_ring,
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
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
                s,
                params,
                parent,
                Z_split,
                digit_nonzero,
                ell_d,
                child_commitments,
                &combine_b_pows,
                superneo_cache,
                ring_linear_forms,
                precomputed_y_ring,
            );
            let reference = crate::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
                s,
                params,
                parent,
                Z_split,
                ell_d,
                child_commitments,
                &combine_b_pows,
            );
            require_dec_crosscheck(optimized, reference)
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
    validate_ce_claims_shape("rlc_public: inputs", s, inputs)?;
    validate_pi_ccs_outputs("rlc_public: selected inputs", s, inputs)?;
    validate_rlc_batch_compatibility("rlc_public", params, inputs)?;
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);
    let _ = crate::engines::utils::shared_me_input_r(inputs, inputs[0].r.len())?;
    let d = D;
    let m_in = inputs[0].m_in;
    let d_pad = checked_superneo_d_pad("rlc_public ell_d", ell_d)?;
    let t = inputs[0].y_ring.len();

    // X_out := Σ ρ_i · X_i
    let mut X = Mat::zero(d, neo_ccs::superneo_public_x_cols(m_in), F::ZERO);
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
    let _allow_parallel = false;

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

    let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);
    let c = mix_rhos_commits(&rho_mats, &inputs.iter().map(|m| m.c.clone()).collect::<Vec<_>>());

    Ok(CeClaim {
        adv: None,
        c,
        X,
        r: inputs[0].r.clone(),
        y_ring,
        ct,
        m_in,
        fold_digest: inputs[0].fold_digest,
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RlcPublicVerifyPerf {
    pub rho_mats_ms: f64,
    pub rho_k_lift_ms: f64,
    pub x_ms: f64,
    pub y_ms: f64,
    pub commitment_collect_ms: f64,
    pub commitment_mix_ms: f64,
    pub commitment_ms: f64,
    pub total_ms: f64,
}

/// Witness-free RLC verification without materializing the recomputed parent claim.
///
/// This checks the exact same public relation as `rlc_public(...)? == *expected`, but avoids
/// allocating the full mixed claim on the verifier path.
pub fn rlc_public_matches<MR>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    expected: &CeClaim<Cmt, F, K>,
    mix_rhos_commits: MR,
    ell_d: usize,
) -> Result<bool, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    Ok(rlc_public_matches_with_perf(s, params, rhos, inputs, expected, mix_rhos_commits, ell_d)?.0)
}

pub fn rlc_public_matches_with_perf<MR>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    expected: &CeClaim<Cmt, F, K>,
    mix_rhos_commits: MR,
    ell_d: usize,
) -> Result<(bool, RlcPublicVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    rlc_public_matches_with_perf_impl(s, params, rhos, inputs, expected, mix_rhos_commits, ell_d, true)
}

/// Fast verifier path for callers that already established `inputs` as valid Π_CCS outputs.
///
/// This checks the same public RLC relation as `rlc_public_matches_with_perf`, but skips
/// revalidating input-side CE invariants that Π_CCS verification already proved.
pub fn rlc_public_matches_verified_inputs_with_perf<MR>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    expected: &CeClaim<Cmt, F, K>,
    mix_rhos_commits: MR,
    ell_d: usize,
) -> Result<(bool, RlcPublicVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    rlc_public_matches_with_perf_impl(s, params, rhos, inputs, expected, mix_rhos_commits, ell_d, false)
}

fn rlc_public_matches_with_perf_impl<MR>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    expected: &CeClaim<Cmt, F, K>,
    mix_rhos_commits: MR,
    ell_d: usize,
    _validate_input_ce_invariants: bool,
) -> Result<(bool, RlcPublicVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
{
    let total_started = Instant::now();
    let recomputed = match rlc_public(s, params, rhos, inputs, mix_rhos_commits, ell_d) {
        Ok(value) => value,
        Err(_) => return Ok((false, RlcPublicVerifyPerf::default())),
    };
    let matches = recomputed.c == expected.c
        && recomputed.X == expected.X
        && recomputed.r == expected.r
        && recomputed.y_ring == expected.y_ring
        && recomputed.ct == expected.ct
        && recomputed.m_in == expected.m_in
        && recomputed.fold_digest == expected.fold_digest;
    Ok((
        matches,
        RlcPublicVerifyPerf {
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
            ..RlcPublicVerifyPerf::default()
        },
    ))
}
