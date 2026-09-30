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
    validate_rlc_refs(
        "rlc_with_commit_refs_and_witness_mix",
        &mode,
        s,
        params,
        rhos,
        me_inputs,
        witnesses,
        ell_d,
    )?;
    #[cfg(feature = "perf-timers")]
    let validation_elapsed = total_started.elapsed();
    let rho_mats = crate::common::rot_rhos_to_mats(rhos);

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
                "[pi-rlc/resident] validation={:.3}ms witness_mix={:.3}ms claim_mix={:.3}ms commitment={:.3}ms total={:.3}ms inputs={} cols={}",
                validation_elapsed.as_secs_f64() * 1_000.0,
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
