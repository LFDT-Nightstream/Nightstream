use std::time::Instant;
use thiserror::Error;

use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, check_satisfiability};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_fold::{FoldState, Proof};

/// Orchestrator errors (kept minimal on purpose)
#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("ccs constraints not satisfied by provided witness")]
    Unsatisfied,
}

/// Minimal timing/size metrics returned alongside the proof.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub prove_ms: f64,
    pub proof_bytes: usize,
}

/// PROVE: run the NARK pipeline over a CCS + (instance, witness).
///
/// - Accepts a *prepared* CCS instance (you already committed in main).
/// - Auto-detects the number of sum-check rounds from the CCS (handled inside neo-fold).
/// - Duplicates (inst,wit) internally because the current `generate_proof` expects two.
pub fn prove(
    ccs: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
) -> Result<(Proof, Metrics), OrchestratorError> {
    if !check_satisfiability(ccs, instance, witness) {
        return Err(OrchestratorError::Unsatisfied);
    }

    // Use the same shape params as main (SECURE_PARAMS). The public matrix A may differ,
    // which is fine in current NARK mode; shape params (n,k,d,…) must match.
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);

    // Fresh fold-state for this proof
    let mut fs = FoldState::new(ccs.clone());

    let t0 = Instant::now();
    // Current `generate_proof` takes two pairs; pass the same pair twice.
    let proof = fs.generate_proof(
        (instance.clone(), witness.clone()),
        (instance.clone(), witness.clone()),
        &committer,
    );
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let proof_bytes = proof.transcript.len();
    Ok((proof, Metrics { prove_ms, proof_bytes }))
}

/// VERIFY: check a transcript against the CCS.
///
/// Returns `true` on success. The committer is re-instantiated with the standard params.
pub fn verify(ccs: &CcsStructure, proof: &Proof) -> bool {
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
    let fs = FoldState::new(ccs.clone());
    fs.verify(&proof.transcript, &committer)
}
