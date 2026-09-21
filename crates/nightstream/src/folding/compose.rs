//! Ordered selected C/R/D calls and public verifier replay.
use super::{
    ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsClaim, CcsInstance, DecMixer, Error, NifsProof,
    Params, RlcMixer, RunningInstance, Structure,
};
use neo_reductions::superneo_eval::SuperneoEvalCache;
pub(crate) fn prove_owned_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &SuperneoEvalCache,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    let c = pi_ccs::prove_from_parts_with_rows(tr, pp, s, cache, &claims, &witnesses, &running)?;
    let all: Vec<_> = witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(tr, pp, s, ajtai_rlc_mixer, &c.outputs, &all)?;
    drop(all);
    drop(witnesses);
    drop(running);
    let (children, d) = pi_dec::prove_with_production_key(pp, s, cache, &parent.claim, parent.witness)?;
    Ok((
        RunningInstance::new(children.claims, children.witnesses, Some(parent.claim)),
        NifsProof {
            pi_ccs: c,
            pi_rlc: r,
            pi_dec: d,
        },
    ))
}
pub(crate) fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    combine: DecMixer,
    fresh: &[CcsClaim],
    running: &RunningInstance,
    proof: &NifsProof,
) -> Result<RunningInstance, Error> {
    validate_running_parent_authority(pp, s, combine, running)?;
    let outputs = pi_ccs::verify(tr, pp, s, fresh, running, &proof.pi_ccs)?;
    let parent = pi_rlc::verify(tr, pp, s, mix, &outputs, &proof.pi_rlc)?;
    let children = pi_dec::verify(pp, s, combine, &parent, &proof.pi_dec)?;
    Ok(RunningInstance::new(children, Vec::new(), Some(parent)))
}
pub(crate) fn validate_running_parent_authority(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    running: &RunningInstance,
) -> Result<(), Error> {
    match (running.claims.is_empty(), running.parent_authority.as_ref()) {
        (true, None) => Ok(()),
        (true, Some(_)) => Err(pi_dec::Error::VerifyRejected.into()),
        (false, None) => Err(pi_dec::Error::VerifyRejected.into()),
        (false, Some(parent)) => {
            let proof = pi_dec::Proof {
                children: running.claims.clone(),
            };
            pi_dec::verify(pp, s, combine, parent, &proof)?;
            Ok(())
        }
    }
}
