//! Complete NIFS crosschecks at the materialized proof boundary.

use crate::engine::transcript::{Poseidon2TranscriptSnapshot, Transcript};
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::{
    verify, verify_paper_exact, AcceleratorCrosscheckNifsProver, CrosscheckNifsProver, Error, NifsFPrimeStepContext,
    NifsFreshInstancesRequest, NifsFreshSignedUnitInstancesRequest, NifsProof, NifsProverAdapter, NifsProverOutput,
    NifsProverRequest, OptimizedCpuNifsProver, OptimizedNifsProverAdapter, PaperExactNifsProver,
};

/// Require two complete materialized NIFS executions to be byte-identical.
///
/// Running-state equality includes every PiDEC child witness. Exact radix
/// recomposition therefore also fixes the PiRLC mixed witness. Accelerator
/// crates use the same boundary for optimized CPU parity.
#[doc(hidden)]
pub fn require_nifs_execution_match(
    primary_transcript: Poseidon2TranscriptSnapshot,
    primary_running: &RunningInstance,
    primary_proof: &NifsProof,
    reference_transcript: Poseidon2TranscriptSnapshot,
    reference_running: &RunningInstance,
    reference_proof: &NifsProof,
) -> Result<(), Error> {
    if primary_transcript != reference_transcript {
        return Err(Error::CrosscheckMismatch {
            boundary: "prover transcript",
        });
    }
    if primary_running != reference_running {
        return Err(Error::CrosscheckMismatch {
            boundary: "running accumulator",
        });
    }
    if primary_proof != reference_proof || primary_proof.canonical_bytes() != reference_proof.canonical_bytes() {
        return Err(Error::CrosscheckMismatch {
            boundary: "proof bytes",
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ReferenceBackend {
    Optimized,
    PaperExact,
}

fn prove_with_reference(
    primary: &mut dyn NifsProverAdapter,
    reference_backend: ReferenceBackend,
    request: NifsProverRequest<'_>,
) -> Result<NifsProverOutput, Error> {
    let NifsProverRequest {
        tr,
        pp,
        s,
        cache,
        log,
        lanes,
        mix_rhos_commits,
        combine_b_pows,
        fresh,
        running_carrier,
        running,
        cache_output_for_next_step,
    } = request;

    let initial_transcript = tr.snapshot();
    let fresh_reference = fresh.clone();
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();

    let run_reference = move || {
        let mut reference_transcript = Transcript::session();
        reference_transcript.restore_snapshot(initial_transcript);
        let reference_result = match reference_backend {
            ReferenceBackend::Optimized => OptimizedCpuNifsProver.prove(NifsProverRequest {
                tr: &mut reference_transcript,
                pp,
                s,
                cache,
                log,
                lanes,
                mix_rhos_commits,
                combine_b_pows,
                fresh: fresh_reference,
                running_carrier,
                running,
                cache_output_for_next_step: false,
            }),
            ReferenceBackend::PaperExact => PaperExactNifsProver.prove(NifsProverRequest {
                tr: &mut reference_transcript,
                pp,
                s,
                cache,
                log,
                lanes,
                mix_rhos_commits,
                combine_b_pows,
                fresh: fresh_reference,
                running_carrier,
                running,
                cache_output_for_next_step: false,
            }),
        };
        (reference_result, reference_transcript.snapshot())
    };

    #[cfg(not(target_arch = "wasm32"))]
    let (primary_result, primary_transcript, reference_result, reference_transcript) = std::thread::scope(|scope| {
        let reference = scope.spawn(run_reference);
        let primary_result = primary.prove(NifsProverRequest {
            tr,
            pp,
            s,
            cache,
            log,
            lanes,
            mix_rhos_commits,
            combine_b_pows,
            fresh,
            running_carrier,
            running,
            cache_output_for_next_step,
        });
        let primary_transcript = tr.snapshot();
        let (reference_result, reference_transcript) = reference.join().map_err(|_| Error::CrosscheckWorkerPanic)?;
        Ok::<_, Error>((
            primary_result,
            primary_transcript,
            reference_result,
            reference_transcript,
        ))
    })?;

    #[cfg(target_arch = "wasm32")]
    let (primary_result, primary_transcript, reference_result, reference_transcript) = {
        let primary_result = primary.prove(NifsProverRequest {
            tr,
            pp,
            s,
            cache,
            log,
            lanes,
            mix_rhos_commits,
            combine_b_pows,
            fresh,
            running_carrier,
            running,
            cache_output_for_next_step,
        });
        let primary_transcript = tr.snapshot();
        let (reference_result, reference_transcript) = run_reference();
        (
            primary_result,
            primary_transcript,
            reference_result,
            reference_transcript,
        )
    };

    let (primary_output, reference_output) = match (primary_result, reference_result) {
        (Ok(primary), Ok(reference)) => (primary, reference),
        (Err(primary), Err(_reference)) => {
            if primary_transcript != reference_transcript {
                return Err(Error::CrosscheckMismatch {
                    boundary: "rejected transcript",
                });
            }
            return Err(primary);
        }
        (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
            return Err(Error::CrosscheckMismatch { boundary: "acceptance" });
        }
    };

    let (primary_running, primary_proof, primary_summary) = primary_output.into_materialized_parts_with_summary()?;
    let (reference_running, reference_proof, reference_summary) =
        reference_output.into_materialized_parts_with_summary()?;
    if reference_summary.is_some() {
        return Err(Error::CrosscheckMismatch {
            boundary: "reference post-fold summary",
        });
    }
    require_nifs_execution_match(
        primary_transcript,
        &primary_running,
        &primary_proof,
        reference_transcript,
        &reference_running,
        &reference_proof,
    )?;
    if matches!(reference_backend, ReferenceBackend::PaperExact) {
        let reference_bytes =
            crate::paper::nifs::paper_exact::encode_proof(&reference_proof).map_err(|error| Error::BackendFailure {
                backend: "paper-exact",
                phase: "NIFS proof encoding",
                reason: error.to_string(),
            })?;
        if primary_proof.canonical_bytes() != reference_bytes {
            return Err(Error::CrosscheckMismatch {
                boundary: "independent proof bytes",
            });
        }
    }

    let mut primary_verifier_transcript = Transcript::session();
    primary_verifier_transcript.restore_snapshot(initial_transcript);
    let primary_verified = verify(
        &mut primary_verifier_transcript,
        pp,
        s,
        cache,
        mix_rhos_commits,
        combine_b_pows,
        &fresh_claims,
        running,
        &primary_proof,
    )?;

    let mut reference_verifier_transcript = Transcript::session();
    reference_verifier_transcript.restore_snapshot(initial_transcript);
    let reference_verified = match reference_backend {
        ReferenceBackend::Optimized => verify(
            &mut reference_verifier_transcript,
            pp,
            s,
            cache,
            mix_rhos_commits,
            combine_b_pows,
            &fresh_claims,
            running,
            &reference_proof,
        )?,
        ReferenceBackend::PaperExact => verify_paper_exact(
            &mut reference_verifier_transcript,
            pp,
            s,
            mix_rhos_commits,
            combine_b_pows,
            &fresh_claims,
            running,
            &reference_proof,
        )?,
    };
    if primary_verifier_transcript.snapshot() != primary_transcript
        || reference_verifier_transcript.snapshot() != primary_transcript
        || primary_verified != reference_verified
        || primary_verified.claims != primary_running.claims
        || primary_verified.parent_authority != primary_running.parent_authority
    {
        return Err(Error::CrosscheckMismatch {
            boundary: "verifier replay",
        });
    }

    let mut output = NifsProverOutput::materialized(primary_running, primary_proof);
    if let Some(summary) = primary_summary {
        output = output.with_post_fold_summary(summary);
    }
    Ok(output)
}

impl NifsProverAdapter for CrosscheckNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        prove_with_reference(&mut OptimizedCpuNifsProver, ReferenceBackend::PaperExact, request)
    }
}

impl<A: OptimizedNifsProverAdapter> NifsProverAdapter for AcceleratorCrosscheckNifsProver<A> {
    fn begin_f_prime_step(&mut self, context: NifsFPrimeStepContext) {
        self.accelerator_mut().begin_f_prime_step(context);
    }

    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        prove_with_reference(self.accelerator_mut(), ReferenceBackend::Optimized, request)
    }

    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<crate::paper::relations::CcsInstance>>, Error> {
        self.accelerator_mut().build_fresh_instances(request)
    }

    fn build_fresh_signed_unit_instances(
        &mut self,
        request: NifsFreshSignedUnitInstancesRequest<'_>,
    ) -> Result<Option<Vec<crate::paper::relations::CcsInstance>>, Error> {
        self.accelerator_mut()
            .build_fresh_signed_unit_instances(request)
    }

    fn requires_recursive_compile_reverify(&self) -> bool {
        true
    }
}
