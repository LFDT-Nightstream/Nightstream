//! Ordered selected C/R/D calls and public verifier replay.
use super::{
    ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsClaim, CcsInstance, CeClaim, DecMixer, Error,
    NifsProof, Params, RlcMixer, RunningInstance, Structure,
};
use neo_reductions::superneo_eval::MatrixRows;
pub(crate) fn prove_owned_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    let (c, parent, r) = prove_parent_with_rows(tr, pp, s, rows, workspace_bytes, fresh, running)?;
    let (children, d) = pi_dec::prove_with_production_key(pp, s, rows, workspace_bytes, &parent.claim, parent.witness)?;
    Ok((
        RunningInstance::new(children.claims, children.witnesses, Some(parent.claim)),
        NifsProof {
            pi_ccs: c,
            pi_rlc: r,
            pi_dec: d,
        },
    ))
}
/// PiCCS then PiRLC: one CE(B) parent claim and its witness (paper Lemma 1).
/// A normal fold continues with PiDEC; compression proves the parent instead.
pub(crate) fn prove_parent_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(pi_ccs::Proof, pi_rlc::Output, pi_rlc::Proof), Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    let c = pi_ccs::prove_from_parts_with_rows(tr, pp, s, rows, workspace_bytes, &claims, &witnesses, &running)?;
    let all: Vec<_> = witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(tr, pp, s, ajtai_rlc_mixer, &c.outputs, &all)?;
    Ok((c, parent, r))
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
    let parent = verify_parent(tr, pp, s, mix, combine, fresh, running, &proof.pi_ccs, &proof.pi_rlc)?;
    let children = pi_dec::verify(pp, s, combine, &parent, &proof.pi_dec)?;
    Ok(RunningInstance::new(children, Vec::new(), Some(parent)))
}
/// Replay PiCCS and PiRLC; the parent is recomputed from the inputs.
pub(crate) fn verify_parent(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    combine: DecMixer,
    fresh: &[CcsClaim],
    running: &RunningInstance,
    c: &pi_ccs::Proof,
    r: &pi_rlc::Proof,
) -> Result<CeClaim, Error> {
    validate_running_parent_authority(pp, s, combine, running)?;
    let outputs = pi_ccs::verify(tr, pp, s, fresh, running, c)?;
    Ok(pi_rlc::verify(tr, pp, s, mix, &outputs, r)?)
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
