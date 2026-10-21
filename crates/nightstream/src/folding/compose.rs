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
    fresh: CcsInstance,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    let (c, parent, r) = prove_parent_with_rows(tr, pp, s, rows, workspace_bytes, fresh, running)?;
    let (children, d) = pi_dec::prove_with_production_key(pp, s, rows, workspace_bytes, &parent.claim, parent.witness)?;
    Ok((
        RunningInstance::new(children.claims, children.witnesses),
        NifsProof {
            pi_ccs: c,
            pi_rlc: r,
            pi_dec: d,
        },
    ))
}
/// PiCCS then PiRLC: one CE(B) parent claim and its witness (paper Lemma 1).
/// A normal fold continues with PiDEC; compression proves the parent instead.
/// The fresh witness and the running instance drop on return, before PiDEC.
pub(crate) fn prove_parent_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: CcsInstance,
    running: RunningInstance,
) -> Result<(pi_ccs::Proof, pi_rlc::Output, pi_rlc::Proof), Error> {
    let CcsInstance { claim, witness } = fresh;
    let c = pi_ccs::prove_from_parts_with_rows(tr, pp, s, rows, workspace_bytes, &claim, &witness, &running)?;
    let all: Vec<_> = std::iter::once(&witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(tr, pp, s, ajtai_rlc_mixer, &c.outputs, &all)?;
    Ok((c, parent, r))
}
/// Replay the NIFS verifier for one fold. PiCCS absorbs the prior digest that
/// the fresh public input carries, not the running claims, so the caller must
/// first bind that digest to these claims (the lifecycle uses
/// `checked_prior_state`).
pub(crate) fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    combine: DecMixer,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &NifsProof,
) -> Result<RunningInstance, Error> {
    let parent = verify_parent(tr, pp, s, mix, fresh, running, &proof.pi_ccs, &proof.pi_rlc)?;
    let children = pi_dec::verify(pp, s, combine, &parent, &proof.pi_dec)?;
    Ok(RunningInstance::new(children, Vec::new()))
}
/// Replay PiCCS and PiRLC; the parent is recomputed from the inputs.
pub(crate) fn verify_parent(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    fresh: &CcsClaim,
    running: &RunningInstance,
    c: &pi_ccs::Proof,
    r: &pi_rlc::Proof,
) -> Result<CeClaim, Error> {
    validate_running_children(pp, s, running)?;
    let outputs = pi_ccs::verify(tr, pp, s, fresh, running, c)?;
    Ok(pi_rlc::verify(tr, pp, s, mix, &outputs, r)?)
}
/// A nonempty running instance must be one PiDEC child family. The caller
/// binds the claims to the prior digest that PiCCS absorbs.
pub(crate) fn validate_running_children(pp: &Params, s: &Structure, running: &RunningInstance) -> Result<(), Error> {
    if running.claims.is_empty() {
        return Ok(());
    }
    pi_dec::validate_children(pp, s, &running.claims)?;
    Ok(())
}
