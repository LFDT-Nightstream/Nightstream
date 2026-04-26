//! Owns RV64IM published-seam assembly from accepted proof artifacts.

use std::time::Instant;

use crate::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source, Rv64imFinalBuildOutput,
    Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use crate::rv64im::kernel::{
    accepted_proof_artifact_from_prover_materials, build_rv64im_accepted_proof_artifact,
    build_rv64im_kernel_export_source_from_accepted_artifact, prove_rv64im_public_proof_prover_seam_with_perf,
    verify_rv64im_kernel_export_proof_with_relation_output, Rv64imAcceptedProofArtifact, Rv64imKernelExportSource,
    Rv64imProof, Rv64imProofInput, Rv64imProofProvePerf, Rv64imPublicProofOptions,
};
use crate::rv64im::main_proof::{Rv64imCompressedMainProof, Rv64imLocalFinalSeam};
use crate::rv64im::SimpleKernelError;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imPublishedProofSeamBuildPerf {
    pub accepted_artifact_ms: f64,
    pub kernel_export_source_ms: f64,
    pub final_statement_ms: f64,
    pub main_proof_ms: f64,
    pub final_statement_kernel_export_ms: f64,
    pub final_statement_recursive_proof_ms: f64,
    pub final_statement_recursive_prepare_inputs_ms: f64,
    pub final_statement_recursive_ccs_bind_ms: f64,
    pub final_statement_recursive_ccs_sample_challenges_ms: f64,
    pub final_statement_recursive_ccs_fe_sumcheck_ms: f64,
    pub final_statement_recursive_ccs_nc_sumcheck_ms: f64,
    pub final_statement_recursive_ccs_output_materialize_ms: f64,
    pub final_statement_recursive_ccs_ms: f64,
    pub final_statement_recursive_dims_ms: f64,
    pub final_statement_recursive_rlc_prepare_ms: f64,
    pub final_statement_recursive_rlc_ms: f64,
    pub final_statement_recursive_dec_split_ms: f64,
    pub final_statement_recursive_dec_commit_ms: f64,
    pub final_statement_recursive_dec_ms: f64,
    pub final_statement_folded_digest_ms: f64,
    pub final_statement_final_proof_ms: f64,
    pub final_statement_statement_digest_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug)]
pub struct Rv64imPublishedProofSeam {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub main_proof: Rv64imCompressedMainProof,
    local_final_seam: Rv64imLocalFinalSeam,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imPublicProofAndSeamBuildPerf {
    pub proof: Rv64imProofProvePerf,
    pub seam: Rv64imPublishedProofSeamBuildPerf,
}

impl Rv64imPublishedProofSeam {
    pub fn kernel_export_source(&self) -> &Rv64imKernelExportSource {
        &self.local_final_seam.kernel_export().source
    }

    pub fn kernel_export(&self) -> &crate::rv64im::kernel::Rv64imKernelExportProof {
        self.local_final_seam.kernel_export()
    }

    pub fn rebuild_final_statement(&self) -> Result<Rv64imFinalStatement, SimpleKernelError> {
        self.local_final_seam.rebuild_final_statement()
    }

    pub fn final_proof(&self) -> Result<Rv64imFinalBuildProof, SimpleKernelError> {
        self.local_final_seam.rebuild_final_proof()
    }
}
fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn published_seam_public_proof_options() -> Rv64imPublicProofOptions {
    Rv64imPublicProofOptions::default()
}

pub fn build_rv64im_published_proof_seam(proof: &Rv64imProof) -> Result<Rv64imPublishedProofSeam, SimpleKernelError> {
    let (built, _) = build_rv64im_published_proof_seam_with_perf(proof)?;
    Ok(built)
}

pub fn build_rv64im_published_proof_seam_with_perf(
    proof: &Rv64imProof,
) -> Result<(Rv64imPublishedProofSeam, Rv64imPublishedProofSeamBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();

    let started = Instant::now();
    let accepted_artifact = build_rv64im_accepted_proof_artifact(proof)?;
    let accepted_artifact_ms = elapsed_ms(started);

    let started = Instant::now();
    let kernel_export_source = build_rv64im_kernel_export_source_from_accepted_artifact(&accepted_artifact)?;
    let kernel_export_source_ms = elapsed_ms(started);

    let started = Instant::now();
    let (
        Rv64imFinalBuildOutput {
            statement: final_statement,
            proof: final_proof,
        },
        final_perf,
    ) = prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(
        &accepted_artifact,
        Some(kernel_export_source),
        None,
    )?;
    let final_statement_ms = elapsed_ms(started);

    let started = Instant::now();
    let (_, verified_kernel) = verify_rv64im_kernel_export_proof_with_relation_output(&final_proof.kernel_export)?;
    let main_proof =
        Rv64imCompressedMainProof::from_final_artifacts(&final_statement, &final_proof, verified_kernel.final_pc)?;
    let main_proof_ms = elapsed_ms(started);

    Ok((
        Rv64imPublishedProofSeam {
            accepted_artifact,
            main_proof,
            local_final_seam: Rv64imLocalFinalSeam::new(
                final_proof.proof_digest,
                final_proof.kernel_export.clone(),
                final_proof.steps.clone(),
            ),
        },
        Rv64imPublishedProofSeamBuildPerf {
            accepted_artifact_ms,
            kernel_export_source_ms,
            final_statement_ms,
            main_proof_ms,
            final_statement_kernel_export_ms: final_perf.folded.kernel_export_ms,
            final_statement_recursive_proof_ms: final_perf.folded.recursive.total_ms,
            final_statement_recursive_prepare_inputs_ms: final_perf.folded.recursive.prepare_inputs_ms,
            final_statement_recursive_ccs_bind_ms: final_perf.folded.recursive.ccs_bind_ms,
            final_statement_recursive_ccs_sample_challenges_ms: final_perf.folded.recursive.ccs_sample_challenges_ms,
            final_statement_recursive_ccs_fe_sumcheck_ms: final_perf.folded.recursive.ccs_fe_sumcheck_ms,
            final_statement_recursive_ccs_nc_sumcheck_ms: final_perf.folded.recursive.ccs_nc_sumcheck_ms,
            final_statement_recursive_ccs_output_materialize_ms: final_perf.folded.recursive.ccs_output_materialize_ms,
            final_statement_recursive_ccs_ms: final_perf.folded.recursive.ccs_ms,
            final_statement_recursive_dims_ms: final_perf.folded.recursive.dims_ms,
            final_statement_recursive_rlc_prepare_ms: final_perf.folded.recursive.rlc_prepare_ms,
            final_statement_recursive_rlc_ms: final_perf.folded.recursive.rlc_ms,
            final_statement_recursive_dec_split_ms: final_perf.folded.recursive.dec_split_ms,
            final_statement_recursive_dec_commit_ms: final_perf.folded.recursive.dec_commit_ms,
            final_statement_recursive_dec_ms: final_perf.folded.recursive.dec_ms,
            final_statement_folded_digest_ms: final_perf.folded.folded_digest_ms,
            final_statement_final_proof_ms: final_perf.final_proof_ms,
            final_statement_statement_digest_ms: final_perf.statement_digest_ms,
            total_ms: elapsed_ms(total_started),
        },
    ))
}

pub fn prove_rv64im_public_proof_and_published_seam_with_perf(
    input: &Rv64imProofInput,
) -> Result<
    (
        (Rv64imProof, Rv64imPublishedProofSeam),
        Rv64imPublicProofAndSeamBuildPerf,
    ),
    SimpleKernelError,
> {
    prove_rv64im_public_proof_and_published_seam_with_options_and_perf(input, published_seam_public_proof_options())
}

fn prove_rv64im_public_proof_and_published_seam_with_options_and_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<
    (
        (Rv64imProof, Rv64imPublishedProofSeam),
        Rv64imPublicProofAndSeamBuildPerf,
    ),
    SimpleKernelError,
> {
    let (built, proof_perf) = prove_rv64im_public_proof_prover_seam_with_perf(input, options)?;

    let total_started = Instant::now();

    let started = Instant::now();
    let accepted_artifact = accepted_proof_artifact_from_prover_materials(
        &built.proof.claim,
        &built.proof.statement,
        &built.kernel,
        &built.sidecar,
        &built.proof.kernel.main_lane,
        &built.proof.kernel.stage_claims,
        &built.proof.kernel.stage_packages,
        &built.proof.kernel.kernel_opening,
        &built.proof.kernel.kernel_claims,
        &built.proof.kernel.root_lane_columns,
        &built.proof.kernel.root_lane_commitment,
    )?;
    let accepted_artifact_ms = elapsed_ms(started);

    let started = Instant::now();
    let kernel_export_source = build_rv64im_kernel_export_source_from_accepted_artifact(&accepted_artifact)?;
    let kernel_export_source_ms = elapsed_ms(started);

    let started = Instant::now();
    let (
        Rv64imFinalBuildOutput {
            statement: final_statement,
            proof: final_proof,
        },
        final_perf,
    ) = prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(
        &accepted_artifact,
        Some(kernel_export_source),
        Some(built.main_lane_inputs),
    )?;
    let final_statement_ms = elapsed_ms(started);

    let started = Instant::now();
    let (_, verified_kernel) = verify_rv64im_kernel_export_proof_with_relation_output(&final_proof.kernel_export)?;
    let main_proof =
        Rv64imCompressedMainProof::from_final_artifacts(&final_statement, &final_proof, verified_kernel.final_pc)?;
    let main_proof_ms = elapsed_ms(started);

    let seam = Rv64imPublishedProofSeam {
        accepted_artifact,
        main_proof,
        local_final_seam: Rv64imLocalFinalSeam::new(
            final_proof.proof_digest,
            final_proof.kernel_export.clone(),
            final_proof.steps.clone(),
        ),
    };
    let seam_perf = Rv64imPublishedProofSeamBuildPerf {
        accepted_artifact_ms,
        kernel_export_source_ms,
        final_statement_ms,
        main_proof_ms,
        final_statement_kernel_export_ms: final_perf.folded.kernel_export_ms,
        final_statement_recursive_proof_ms: final_perf.folded.recursive.total_ms,
        final_statement_recursive_prepare_inputs_ms: final_perf.folded.recursive.prepare_inputs_ms,
        final_statement_recursive_ccs_bind_ms: final_perf.folded.recursive.ccs_bind_ms,
        final_statement_recursive_ccs_sample_challenges_ms: final_perf.folded.recursive.ccs_sample_challenges_ms,
        final_statement_recursive_ccs_fe_sumcheck_ms: final_perf.folded.recursive.ccs_fe_sumcheck_ms,
        final_statement_recursive_ccs_nc_sumcheck_ms: final_perf.folded.recursive.ccs_nc_sumcheck_ms,
        final_statement_recursive_ccs_output_materialize_ms: final_perf.folded.recursive.ccs_output_materialize_ms,
        final_statement_recursive_ccs_ms: final_perf.folded.recursive.ccs_ms,
        final_statement_recursive_dims_ms: final_perf.folded.recursive.dims_ms,
        final_statement_recursive_rlc_prepare_ms: final_perf.folded.recursive.rlc_prepare_ms,
        final_statement_recursive_rlc_ms: final_perf.folded.recursive.rlc_ms,
        final_statement_recursive_dec_split_ms: final_perf.folded.recursive.dec_split_ms,
        final_statement_recursive_dec_commit_ms: final_perf.folded.recursive.dec_commit_ms,
        final_statement_recursive_dec_ms: final_perf.folded.recursive.dec_ms,
        final_statement_folded_digest_ms: final_perf.folded.folded_digest_ms,
        final_statement_final_proof_ms: final_perf.final_proof_ms,
        final_statement_statement_digest_ms: final_perf.statement_digest_ms,
        total_ms: elapsed_ms(total_started),
    };

    Ok((
        (built.proof, seam),
        Rv64imPublicProofAndSeamBuildPerf {
            proof: proof_perf,
            seam: seam_perf,
        },
    ))
}
