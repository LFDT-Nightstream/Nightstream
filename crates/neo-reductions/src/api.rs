//! Public API for Π_CCS folding and RLC/DEC operations.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use crate::error::PiCcsError;

mod accelerator;
pub use accelerator::{rlc_with_commit_refs_and_resident_witness, rlc_with_commit_refs_and_witness_mix};
mod dec;
pub use dec::verify_dec_public;
mod validation;
pub(crate) use validation::{
    checked_superneo_d_pad, ensure_superneo_width, validate_ce_claim_shape, validate_ce_claims_shape,
    validate_dec_boundary_inputs, validate_dec_boundary_inputs_from_trusted_split, validate_pi_ccs_outputs,
    validate_rlc_batch_compatibility,
};

pub use crate::engines::optimized_engine::PiCcsProof;
pub use crate::engines::{DecOutcome, RlcMixer};

pub use crate::common::{
    compute_v1_2_evaluations_from_z_and_r, format_ext, left_mul_acc, rot_rhos_from_mats, rot_rhos_to_mats,
    sample_rot_rhos_n_typed, split_b_matrix_k, split_b_matrix_k_with_nonzero_flags, RotRho, RotRing, SplitDigits,
};

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
fn require_dec_crosscheck(optimized: DecOutcome, reference: DecOutcome) -> DecOutcome {
    assert_eq!(optimized, reference, "PiDEC optimized and PaperExact results differ");
    optimized
}

pub fn rlc_with_commit(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    Zs: &[Mat<F>],
    ell_d: usize,
    mix_commits: RlcMixer,
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError> {
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
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, s.cube_variables())?;
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_packed_witness_dec_representable(
            params,
            z,
            s.m,
            &format!("rlc_with_commit: Zs[{idx}]"),
        )?;
    }

    let (out, Z_mix) = match mode {
        FoldingMode::Optimized => {
            OptimizedRlcDec::rlc_with_commit(s, params, &rho_mats, me_inputs, Zs, ell_d, mix_commits)
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
            s,
            params,
            &rho_mats,
            me_inputs,
            Zs,
            ell_d,
            mix_commits,
        ),
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck => {
            let optimized = OptimizedRlcDec::rlc_with_commit(s, params, &rho_mats, me_inputs, Zs, ell_d, mix_commits);
            let reference = crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                Zs,
                ell_d,
                mix_commits,
            );
            return require_rlc_crosscheck(optimized, reference);
        }
    };
    Ok((out, Z_mix))
}

#[allow(clippy::too_many_arguments)]
fn validate_rlc_refs(
    label: &str,
    mode: &FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    Zs: &[&Mat<F>],
    ell_d: usize,
) -> Result<(), PiCcsError> {
    ensure_superneo_width(s)?;
    if me_inputs.is_empty() {
        return Err(PiCcsError::InvalidInput(format!("{label}: empty inputs")));
    }
    if rhos.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: |rhos| mismatch (expected {}, got {})",
            me_inputs.len(),
            rhos.len()
        )));
    }
    if Zs.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: |Zs| mismatch (expected {}, got {})",
            me_inputs.len(),
            Zs.len()
        )));
    }
    validate_ce_claims_shape(&format!("{label}: me_inputs"), s, me_inputs)?;
    validate_selected_reduction_claims(mode, &format!("{label}: selected inputs"), s, me_inputs)?;
    validate_rlc_batch_compatibility(label, params, me_inputs)?;
    checked_superneo_d_pad(&format!("{label} ell_d"), ell_d)?;
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, s.cube_variables())?;
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_packed_witness_dec_representable(params, z, s.m, &format!("{label}: Zs[{idx}]"))?;
    }
    Ok(())
}

pub fn rlc_with_commit_refs(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    Zs: &[&Mat<F>],
    ell_d: usize,
    mix_commits: RlcMixer,
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError> {
    validate_rlc_refs("rlc_with_commit_refs", &mode, s, params, rhos, me_inputs, Zs, ell_d)?;
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);

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
                mix_commits,
            );
            let owned_zs: Vec<Mat<F>> = Zs.iter().map(|z| (*z).clone()).collect();
            let reference = crate::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
                s,
                params,
                &rho_mats,
                me_inputs,
                &owned_zs,
                ell_d,
                mix_commits,
            );
            return require_rlc_crosscheck(optimized, reference);
        }
    };
    Ok((out, Z_mix))
}

pub fn dec_children_with_commit<Comb>(
    mode: FoldingMode,
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
    use crate::engines::pi_rlc_dec::{OptimizedRlcDec, RlcDecOps};
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit input validation failed: {e}");
        return DecOutcome::rejected();
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit input validation failed: {error}");
        return DecOutcome::rejected();
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
) -> DecOutcome
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_superneo_cached input validation failed: {e}");
        return DecOutcome::rejected();
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_superneo_cached: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit_superneo_cached input validation failed: {error}");
        return DecOutcome::rejected();
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
) -> DecOutcome
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
        return DecOutcome::rejected();
    }
    if let Err(e) = validate_dec_boundary_inputs(s, params, parent, Z_split, child_commitments, ell_d) {
        eprintln!("dec_children_with_commit_superneo_cached_with_digit_flags input validation failed: {e}");
        return DecOutcome::rejected();
    }
    if let Err(error) = validate_selected_reduction_claims(
        &mode,
        "dec_children_with_commit_superneo_cached_with_digit_flags: selected parent",
        s,
        std::slice::from_ref(parent),
    ) {
        eprintln!("dec_children_with_commit_superneo_cached_with_digit_flags input validation failed: {error}");
        return DecOutcome::rejected();
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

pub fn dec_children_with_commit_superneo_cached_from_trusted_split_digits<Comb>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    split: &SplitDigits,
    ell_d: usize,
    child_commitments: &[Cmt],
    combine_b_pows: Comb,
    superneo_cache: Option<&crate::superneo_eval::SuperneoEvalCache>,
    precomputed_openings: Option<&[neo_ccs::V1_2Evaluations<K>]>,
) -> DecOutcome
where
    Comb: Fn(&[Cmt], u32) -> Cmt,
{
    use crate::engines::pi_rlc_dec::OptimizedRlcDec;
    let Z_split = split.planes();
    if let Err(e) =
        validate_dec_boundary_inputs_from_trusted_split(s, params, parent, Z_split, child_commitments, ell_d)
    {
        eprintln!("dec_children_with_commit_superneo_cached_from_trusted_split_digits input validation failed: {e}");
        return DecOutcome::rejected();
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
        return DecOutcome::rejected();
    }

    match mode {
        FoldingMode::Optimized => OptimizedRlcDec::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            s,
            params,
            parent,
            split,
            ell_d,
            child_commitments,
            combine_b_pows,
            superneo_cache,
            precomputed_openings,
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
                split,
                ell_d,
                child_commitments,
                &combine_b_pows,
                superneo_cache,
                precomputed_openings,
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

pub fn rlc_public(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    mix_rhos_commits: RlcMixer,
    ell_d: usize,
) -> Result<CeClaim<Cmt, F, K>, PiCcsError> {
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
    let matrix_count = inputs[0].eval_a.len();

    let mut X = Mat::zero(d, inputs[0].X.cols(), F::ZERO);
    for (rho, inst) in rho_mats.iter().zip(inputs.iter()) {
        left_mul_acc(&mut X, rho, &inst.X);
    }

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

    let mut eval_k = vec![K::ZERO; d_pad];
    for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
        for k in 0..d {
            let value = inst.eval_k[k];
            if value == K::ZERO {
                continue;
            }
            let column = &rho_k[k * d..(k + 1) * d];
            for row in 0..d {
                eval_k[row] += column[row] * value;
            }
        }
    }

    let mut eval_a = vec![vec![K::ZERO; d_pad]; matrix_count];
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let allow_parallel =
        rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && matrix_count >= 128;
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let _allow_parallel = false;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if allow_parallel {
        eval_a.par_iter_mut().enumerate().for_each(|(j, acc)| {
            for (rho_k, inst) in rho_k_mats.iter().zip(inputs.iter()) {
                let src = &inst.eval_a[j];
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
            for (j, acc) in eval_a.iter_mut().enumerate() {
                let src = &inst.eval_a[j];
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
            for (j, acc) in eval_a.iter_mut().enumerate() {
                let src = &inst.eval_a[j];
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

    let c = mix_rhos_commits(&rho_mats, &inputs.iter().map(|m| m.c.clone()).collect::<Vec<_>>());

    Ok(CeClaim {
        adv: None,
        c,
        X,
        r: inputs[0].r.clone(),
        eval_k,
        eval_a,
        m_in,
        fold_digest: inputs[0].fold_digest,
    })
}

pub fn rlc_public_matches(
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    inputs: &[CeClaim<Cmt, F, K>],
    expected: &CeClaim<Cmt, F, K>,
    mix_rhos_commits: RlcMixer,
    ell_d: usize,
) -> Result<bool, PiCcsError> {
    Ok(rlc_public(s, params, rhos, inputs, mix_rhos_commits, ell_d).is_ok_and(|claim| claim == *expected))
}
