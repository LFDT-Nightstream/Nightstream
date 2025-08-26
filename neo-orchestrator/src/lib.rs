use std::time::Instant;
use thiserror::Error;

use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, check_satisfiability};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_fold::Proof;
#[allow(unused_imports)]
use neo_fold::FoldState;

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

/// PROVE: run the NARK/SNARK pipeline over a CCS + (instance, witness).
///
/// - Accepts a *prepared* CCS instance (you already committed in main).
/// - Auto-detects the number of sum-check rounds from the CCS (handled inside neo-fold).
/// - Duplicates (inst,wit) internally because the current `generate_proof` expects two.
/// - In SNARK mode, uses NeutronNova folding for succinctness.
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

    let t0 = Instant::now();
    
    // Use NeutronNova SNARK mode by default
    use neo_fold::neutronnova_integration::NeutronNovaFoldState;
    let mut fs = NeutronNovaFoldState::new(ccs.clone());
    
    // Current `generate_proof_snark` takes two pairs; pass the same pair twice.
    let proof = fs.generate_proof_snark(
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
    use neo_fold::neutronnova_integration::NeutronNovaFoldState;
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
    let fs = NeutronNovaFoldState::new(ccs.clone());
    fs.verify_snark(&proof.transcript, &committer)
}
