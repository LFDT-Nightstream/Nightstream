//! Coordinator for repeated folding: for each MCS, run
//!   Π_CCS (via pi_ccs facade) → RLC (public) → DEC (public).
//!
//! Engine-agnostic: CCS goes through `pi_ccs::prove/verify` with FoldingMode.
//! RLC/DEC are performed publicly here, and commitment checks are injected
//! through user-supplied S-action mixers.

#![allow(non_snake_case)]

use neo_transcript::labels as tr_labels;
use neo_transcript::{Poseidon2Transcript, Transcript};

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};

use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::PiCcsError;
use neo_reductions::engines::utils;

// Pull Π_CCS and RLC/DEC via the public facade (engine-agnostic)
use crate::pi_ccs::{self as ccs, FoldingMode, PiCcsProof};

/// One fold step’s artifacts.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Π_CCS outputs (k ME(b,L) instances)
    pub ccs_out: Vec<MeInstance<Cmt, F, K>>,
    /// Π_CCS proof (engine-agnostic re-export)
    pub ccs_proof: PiCcsProof,

    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<Mat<F>>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: MeInstance<Cmt, F, K>,

    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<MeInstance<Cmt, F, K>>,
}

/// Entire multi-fold run (public proof data only - no witnesses).
#[derive(Clone, Debug)]
pub struct FoldRun {
    pub steps: Vec<FoldStep>,
}

impl FoldRun {
    /// Compute the final accumulator from the verified steps.
    /// This is the ground truth - use this instead of any user-provided field.
    pub fn compute_final_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        if self.steps.is_empty() {
            acc_init.to_vec()
        } else {
            self.steps.last().unwrap().dec_children.clone()
        }
    }
}

/// Commitment mixers so the coordinator stays scheme-agnostic.
/// - `mix_rhos_commits(ρ, cs)` returns Σ ρ_i · c_i  (S-action).
/// - `combine_b_pows(cs, b)` returns Σ \bar b^{i-1} c_i  (DEC check).
#[derive(Clone, Copy)]
pub struct CommitMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

/// Plan: number of folds = number of MCS instances.
#[inline]
pub fn plan_num_folds(mcs_count: usize) -> usize {
    mcs_count
}

// ---------------------------------------------------------------------------
// Prover orchestration
// ---------------------------------------------------------------------------

/// Run all folds (one per MCS), returning both the public proof and the final witnesses.
///
/// This is the prover's internal function. Use `fold_many_prove` if you only need the proof.
///
/// Returns: `(FoldRun, final_witnesses)` where `final_witnesses` are the Z matrices
/// for the final accumulator (needed for subsequent merge operations).
pub fn fold_many_prove_with_witnesses<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_in: &CcsStructure<F>,
    mcss: &[(McsInstance<Cmt, F>, McsWitness<F>)],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(FoldRun, Vec<Mat<F>>), PiCcsError>
where
    F: PrimeField64 + PrimeCharacteristicRing + Copy,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    tr.append_message(tr_labels::PI_CCS, b"");
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, .. } = utils::build_dims_and_policy(params, &s)?;

    // Infer initial k from initial accumulator
    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput("initial accumulator size mismatch".into()));
    }

    // Current accumulator (carried inputs)
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut steps = Vec::with_capacity(mcss.len());

    for (idx, (mcs_i, wit_i)) in mcss.iter().enumerate() {
        // Recompute k for this iteration based on current accumulator size
        // k = |accumulator| + 1 (new MCS instance)
        let k = accumulator.len() + 1;

        // --- Π_CCS via facade (chooses engine by `mode`)
        let (ccs_out, ccs_proof) = ccs::prove(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_i),
            core::slice::from_ref(wit_i),
            &accumulator,
            &accumulator_wit,
            l,
        )?;

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}",
                ccs_out.len()
            )));
        }

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...]
        let mut outs_Z: Vec<Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(wit_i.Z.clone());
        outs_Z.extend(accumulator_wit.iter().cloned());

        // --- RLC (via facade): pick ρ, combine k → parent
        // Use paper-compliant rotation matrix sampler (Section 4.5, Definition 14)
        // Sample exactly k rhos (not k_rho+1) - we only need as many as ME claims being merged
        let ring = ccs::RotRing::goldilocks();
        let rhos = ccs::sample_rot_rhos_n(tr, params, &ring, k)?;

        // rhos is exactly k elements (sample_rot_rhos_n enforces norm bound check)
        let rhos_for_rlc = &rhos[..];
        // Compute parent via RLC and combined witness Z_mix = Σ ρ_i·Z_i
        let (parent_pub, Z_mix) = ccs::rlc_with_commit(
            mode.clone(),
            &s,
            params,
            rhos_for_rlc,
            &ccs_out,
            &outs_Z,
            ell_d,
            mixers.mix_rhos_commits,
        );

        // NOTE: y-vectors are already correctly computed by rlc_with_commit via the S-module action.
        // Do NOT recompute from Z_mix as that gives different results due to the digit structure of y.

        // --- DEC (public): split Z_mix into params.k_rho children; compute (c_i, X_i, y_(i,j))
        // Note: DEC uses params.k_rho (decomposition exponent), NOT k_fold (runtime folding count)
        let k_dec = params.k_rho as usize;
        let Z_split = ccs::split_b_matrix_k(&Z_mix, k_dec, params.b)?;
        // Commitments for children via L(Z_i)
        let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
        let (children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
            mode.clone(),
            &s,
            params,
            &parent_pub,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
        );
        // Enforce y/X/commitment equality in all modes by default.
        if !(ok_y && ok_X && ok_c) {
            return Err(PiCcsError::ProtocolError(format!(
                "DEC public check failed (y={}, X={}, c={})",
                ok_y, ok_X, ok_c
            )));
        }

        // Advance accumulator
        accumulator = children.clone();
        accumulator_wit = Z_split;

        steps.push(FoldStep {
            ccs_out,
            ccs_proof,
            rlc_rhos: rhos_for_rlc.to_vec(),
            rlc_parent: parent_pub,
            dec_children: children,
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    let final_witnesses = accumulator_wit;
    Ok((FoldRun { steps }, final_witnesses))
}

/// Convenience wrapper: run all folds, returning only the public proof (discards witnesses).
pub fn fold_many_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_in: &CcsStructure<F>,
    mcss: &[(McsInstance<Cmt, F>, McsWitness<F>)],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<FoldRun, PiCcsError>
where
    F: PrimeField64 + PrimeCharacteristicRing + Copy,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    Ok(fold_many_prove_with_witnesses(mode, tr, params, s_in, mcss, acc_init, acc_wit_init, l, mixers)?.0)
}

// ---------------------------------------------------------------------------
// Verifier orchestration
// ---------------------------------------------------------------------------

/// Verify a multi-fold run end-to-end and return the computed final accumulator.
///
/// The returned accumulator is computed from the verified steps (ground truth),
/// not from any user-provided proof field. Use this for subsequent operations.
pub fn fold_many_verify<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_in: &CcsStructure<F>,
    mcss: &[McsInstance<Cmt, F>],
    acc_init: &[MeInstance<Cmt, F, K>],
    run: &FoldRun,
    mixers: CommitMixers<MR, MB>,
) -> Result<Vec<MeInstance<Cmt, F, K>>, PiCcsError>
where
    F: PrimeField64 + PrimeCharacteristicRing + Copy,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    tr.append_message(tr_labels::PI_CCS, b"");
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, .. } = utils::build_dims_and_policy(params, &s)?;

    if mcss.len() != run.steps.len() {
        return Err(PiCcsError::InvalidInput("mcss.len() must equal run.steps.len()".into()));
    }

    let mut accumulator = acc_init.to_vec(); // public initial accumulator

    for (i, step) in run.steps.iter().enumerate() {
        // 1) Verify Π_CCS via facade (engine-agnostic)
        let ok_ccs = ccs::verify(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(&mcss[i]),
            &accumulator,
            &step.ccs_out,
            &step.ccs_proof,
        )?;
        if !ok_ccs {
            return Err(PiCcsError::SumcheckError("Π_CCS verification failed".into()));
        }

        // 1a) Consume the same digest the prover took at the end of Π_CCS.
        //     This keeps the transcript state synchronized across steps.
        //     Also verify it matches what the prover recorded in the proof.
        let observed_digest = tr.digest32();
        if observed_digest != step.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }

        // 1b) Sample RLC ρ from transcript and verify they match what the prover used.
        //     This binds the ρ to the Fiat-Shamir transcript, preventing adversarial ρ selection.
        //     Sample exactly k rhos (same as prover) to keep transcripts synchronized.
        let ring = ccs::RotRing::goldilocks();
        let k = step.rlc_rhos.len();
        let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, &ring, k)?;

        // Check that all k rhos match exactly
        for (j, (sampled, stored)) in rhos_from_tr.iter().zip(step.rlc_rhos.iter()).enumerate() {
            if sampled.as_slice() != stored.as_slice() {
                return Err(PiCcsError::ProtocolError(format!(
                    "RLC ρ #{} mismatch: transcript vs proof",
                    j
                )));
            }
        }

        // 2) Verify RLC publicly: recompute parent from (stored rhos, ccs_out)
        // Use the rhos from the proof (already committed to via parent_pub)
        let parent_pub = ccs::rlc_public(
            &s,
            params,
            &step.rlc_rhos,
            &step.ccs_out,
            mixers.mix_rhos_commits,
            ell_d,
        );

        // Check ALL public fields of the RLC parent (X, c, r, y, y_scalars)
        if parent_pub.X.as_slice() != step.rlc_parent.X.as_slice() {
            return Err(PiCcsError::ProtocolError("RLC X mismatch".into()));
        }
        if parent_pub.c != step.rlc_parent.c {
            return Err(PiCcsError::ProtocolError("RLC commitment mismatch".into()));
        }
        if parent_pub.r != step.rlc_parent.r {
            return Err(PiCcsError::ProtocolError("RLC r mismatch".into()));
        }
        // y and y_scalars must match (critical - RLC determines these)
        if parent_pub.y != step.rlc_parent.y {
            return Err(PiCcsError::ProtocolError("RLC y mismatch".into()));
        }
        if parent_pub.y_scalars != step.rlc_parent.y_scalars {
            return Err(PiCcsError::ProtocolError("RLC y_scalars mismatch".into()));
        }

        // 3) Verify DEC publicly (X, y, c)
        if !ccs::verify_dec_public(
            &s,
            params,
            &step.rlc_parent,
            &step.dec_children,
            mixers.combine_b_pows,
            ell_d,
        ) {
            return Err(PiCcsError::ProtocolError("DEC public check failed".into()));
        }

        // Advance accumulator
        accumulator = step.dec_children.clone();
        // Use the exact same marker label the prover used so the next step's challenges line up.
        tr.append_message(b"fold/step_done", &(i as u64).to_le_bytes());
    }

    // Return the verified final accumulator (computed from steps, not from proof fields)
    Ok(accumulator)
}
