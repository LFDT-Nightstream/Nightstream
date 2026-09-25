//! Narrow accelerator hooks that preserve the canonical reduction boundary.

use super::*;

/// Borrowed-matrix Π_RLC with an accelerator-owned witness mixer.
///
/// Claim algebra, commitment mixing, validation, and transcript ownership stay
/// with the canonical reduction. Only `sum_i rho_i * Z_i` is delegated.
pub fn rlc_with_commit_refs_and_witness_mix<Comb, MixWitness>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    witnesses: &[&Mat<F>],
    ell_d: usize,
    mix_commits: Comb,
    mix_witnesses: MixWitness,
) -> Result<(CeClaim<Cmt, F, K>, Mat<F>), PiCcsError>
where
    Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MixWitness: Fn(&[Mat<F>], &[&Mat<F>]) -> Mat<F>,
{
    rlc_with_commit_refs_and_resident_witness(
        mode,
        s,
        params,
        rhos,
        me_inputs,
        witnesses,
        ell_d,
        mix_commits,
        mix_witnesses,
    )
}

/// Borrowed-matrix Π_RLC with an accelerator-owned resident witness.
///
/// The returned handle is opaque to the reduction. Public claim algebra and
/// commitment mixing remain canonical, while the accelerator can pass its
/// witness directly to Π_DEC without manufacturing a host matrix.
#[allow(clippy::too_many_arguments)]
pub fn rlc_with_commit_refs_and_resident_witness<Comb, MixWitness, Resident>(
    mode: FoldingMode,
    s: &CcsStructure<F>,
    params: &NeoParams,
    rhos: &[RotRho],
    me_inputs: &[CeClaim<Cmt, F, K>],
    witnesses: &[&Mat<F>],
    ell_d: usize,
    mix_commits: Comb,
    mix_witnesses: MixWitness,
) -> Result<(CeClaim<Cmt, F, K>, Resident), PiCcsError>
where
    Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MixWitness: Fn(&[Mat<F>], &[&Mat<F>]) -> Resident,
{
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    ensure_superneo_width(s)?;
    if me_inputs.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "rlc_with_commit_refs_and_witness_mix: empty inputs".into(),
        ));
    }
    if rhos.len() != me_inputs.len() || witnesses.len() != me_inputs.len() {
        return Err(PiCcsError::InvalidInput(
            "rlc_with_commit_refs_and_witness_mix: input count mismatch".into(),
        ));
    }
    #[cfg(feature = "perf-timers")]
    let shape_started = std::time::Instant::now();
    validate_ce_claims_shape("rlc_with_commit_refs_and_witness_mix: me_inputs", s, me_inputs)?;
    let _ = crate::engines::utils::shared_me_input_r(me_inputs, ell_n_for_ccs(s))?;
    #[cfg(feature = "perf-timers")]
    let shape_elapsed = shape_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let rho_started = std::time::Instant::now();
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);
    #[cfg(feature = "perf-timers")]
    let rho_elapsed = rho_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let range_started = std::time::Instant::now();
    for (idx, witness) in witnesses.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(
            params,
            witness,
            s.m,
            &format!("rlc_with_commit_refs_and_witness_mix: witnesses[{idx}]"),
        )?;
    }
    #[cfg(feature = "perf-timers")]
    let range_elapsed = range_started.elapsed();

    match mode {
        FoldingMode::Optimized => {
            #[cfg(feature = "perf-timers")]
            let witness_mix_started = std::time::Instant::now();
            let resident = mix_witnesses(&rho_mats, witnesses);
            #[cfg(feature = "perf-timers")]
            let witness_mix_elapsed = witness_mix_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let claim_mix_started = std::time::Instant::now();
            let mut out = crate::engines::optimized_engine::rlc_combine_claims(s, params, &rho_mats, me_inputs, ell_d);
            #[cfg(feature = "perf-timers")]
            let claim_mix_elapsed = claim_mix_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let commitment_started = std::time::Instant::now();
            let commitments = me_inputs
                .iter()
                .map(|input| input.c.clone())
                .collect::<Vec<_>>();
            out.c = mix_commits(&rho_mats, &commitments);
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[pi-rlc/resident] shape={:.3}ms rho={:.3}ms range={:.3}ms witness_mix={:.3}ms claim_mix={:.3}ms commitment={:.3}ms total={:.3}ms inputs={} cols={}",
                shape_elapsed.as_secs_f64() * 1_000.0,
                rho_elapsed.as_secs_f64() * 1_000.0,
                range_elapsed.as_secs_f64() * 1_000.0,
                witness_mix_elapsed.as_secs_f64() * 1_000.0,
                claim_mix_elapsed.as_secs_f64() * 1_000.0,
                commitment_started.elapsed().as_secs_f64() * 1_000.0,
                total_started.elapsed().as_secs_f64() * 1_000.0,
                witnesses.len(),
                s.m,
            );
            Ok((out, resident))
        }
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact | FoldingMode::OptimizedWithCrosscheck => Err(PiCcsError::InvalidInput(
            "accelerator witness mixing is available only in FoldingMode::Optimized".into(),
        )),
    }
}
