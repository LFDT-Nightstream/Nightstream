//! Parallel reference/CPU proving from identical inputs. No result or transcript
//! is returned until the complete proofs, accumulators, and transcripts agree.

use super::{paper_exact, Engine, EngineError};
use crate::folding::{
    self,
    transcript::{Poseidon2TranscriptSnapshot, Transcript},
    CcsInstance, NifsProof, Params, RunningInstance, Structure,
};
use neo_math::F;
use neo_reductions::{paper_exact_engine::PaperMatrixRows, superneo_eval::SuperneoEvalCache};

pub(crate) fn prove(
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    cache: &SuperneoEvalCache,
    rows: &dyn PaperMatrixRows<F>,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), EngineError> {
    let mut optimized_transcript = transcript.clone();
    let mut reference_transcript = transcript.clone();
    let reference_fresh = fresh.clone();
    let reference_running = running.clone();
    let (optimized, reference) = std::thread::scope(|scope| {
        let reference = scope.spawn(|| {
            paper_exact::prove(
                &mut reference_transcript,
                params,
                structure,
                rows,
                reference_fresh,
                reference_running,
            )
        });
        let optimized =
            folding::prove_owned_with_rows(&mut optimized_transcript, params, structure, cache, fresh, running);
        // Always join the reference, including when the CPU prover rejects.
        let reference = reference.join().map_err(|_| EngineError::Failure {
            engine: Engine::PaperExact,
            reason: "cross-check worker panicked".into(),
        })?;
        Ok((optimized, reference))
    })?;

    let (optimized, reference) = match (optimized, reference) {
        (Ok(optimized), Ok(reference)) => (optimized, reference),
        (Err(error), Err(_)) => {
            if optimized_transcript.snapshot() != reference_transcript.snapshot() {
                return Err(EngineError::CrosscheckMismatch {
                    boundary: "rejected transcript",
                });
            }
            return Err(EngineError::Failure {
                engine: Engine::Optimized,
                reason: error.to_string(),
            });
        }
        (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
            return Err(EngineError::CrosscheckMismatch { boundary: "acceptance" });
        }
    };
    require_match(
        optimized_transcript.snapshot(),
        &optimized.0,
        &optimized.1,
        reference_transcript.snapshot(),
        &reference.0,
        &reference.1,
    )?;
    *transcript = optimized_transcript;
    Ok(optimized)
}

pub(super) fn require_match(
    optimized_transcript: Poseidon2TranscriptSnapshot,
    optimized_running: &RunningInstance,
    optimized_proof: &NifsProof,
    reference_transcript: Poseidon2TranscriptSnapshot,
    reference_running: &RunningInstance,
    reference_proof: &NifsProof,
) -> Result<(), EngineError> {
    if optimized_transcript != reference_transcript {
        return Err(EngineError::CrosscheckMismatch {
            boundary: "prover transcript",
        });
    }
    // Equality includes the parent authority and every child witness value.
    if optimized_running != reference_running {
        return Err(EngineError::CrosscheckMismatch {
            boundary: "running accumulator",
        });
    }
    // Compare all proof fields, not a digest of them.
    if optimized_proof != reference_proof {
        return Err(EngineError::CrosscheckMismatch {
            boundary: "proof fields",
        });
    }
    Ok(())
}
