//! Owns perf-only timing wrappers for selected stage-package construction and verification.

use std::time::Instant;

use super::perf_diagnostics::{PackagedOpeningBuildPerf, StagePackageBundleBuildPerf, StagePackageBundleVerifyPerf};
use super::simple::SimpleKernelError;
use super::simple_openings::{
    SimpleKernelStagePackageBundle, Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2PackagedOpeningProof,
    Stage2SelectedOpeningClaim, Stage3PackagedOpeningProof, Stage3SelectedOpeningClaim,
};
use super::stage1_canonical::build_stage1_selected_opening_claim;
use super::stage2_canonical::build_stage2_selected_opening_claim;
use super::stage3_canonical::build_stage3_selected_opening_claim;
use super::stage_artifacts::{
    build_claim_packaged_proof, verify_stage1_packaged_opening_proof, verify_stage2_packaged_opening_proof,
    verify_stage3_packaged_opening_proof, SimpleKernelStageClaimBundle,
};
use crate::rv32im::stage1::Stage1Summary;
use crate::rv32im::stage2::Stage2Summary;
use crate::rv32im::stage3::Stage3Summary;

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn build_stage1_packaged_opening_proof_with_perf(
    claim: Stage1SelectedOpeningClaim,
) -> Result<(Stage1PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    let package_started = Instant::now();
    let selected_labels = claim.labels().len();
    let claim_words = claim.claim_words().len();
    let packaged = build_claim_packaged_proof("rv32im/stage1", &claim.claim_words())?;
    let proof = Stage1PackagedOpeningProof {
        claim,
        packaged,
        digest: [0; 32],
    };
    let proof = Stage1PackagedOpeningProof {
        digest: proof.expected_digest(),
        ..proof
    };
    Ok((
        proof,
        PackagedOpeningBuildPerf {
            selected_labels,
            claim_words,
            package_ms: millis_since(package_started),
        },
    ))
}

fn build_stage2_packaged_opening_proof_with_perf(
    claim: Stage2SelectedOpeningClaim,
) -> Result<(Stage2PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    let package_started = Instant::now();
    let selected_labels = claim.labels().len();
    let claim_words = claim.claim_words().len();
    let packaged = build_claim_packaged_proof("rv32im/stage2", &claim.claim_words())?;
    let proof = Stage2PackagedOpeningProof {
        claim,
        packaged,
        digest: [0; 32],
    };
    let proof = Stage2PackagedOpeningProof {
        digest: proof.expected_digest(),
        ..proof
    };
    Ok((
        proof,
        PackagedOpeningBuildPerf {
            selected_labels,
            claim_words,
            package_ms: millis_since(package_started),
        },
    ))
}

fn build_stage3_packaged_opening_proof_with_perf(
    claim: Stage3SelectedOpeningClaim,
) -> Result<(Stage3PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    let package_started = Instant::now();
    let selected_labels = claim.labels().len();
    let claim_words = claim.claim_words().len();
    let packaged = build_claim_packaged_proof("rv32im/stage3", &claim.claim_words())?;
    let proof = Stage3PackagedOpeningProof {
        claim,
        packaged,
        digest: [0; 32],
    };
    let proof = Stage3PackagedOpeningProof {
        digest: proof.expected_digest(),
        ..proof
    };
    Ok((
        proof,
        PackagedOpeningBuildPerf {
            selected_labels,
            claim_words,
            package_ms: millis_since(package_started),
        },
    ))
}

pub(super) fn build_stage_package_bundle_with_perf(
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    stage_claims: &SimpleKernelStageClaimBundle,
) -> Result<(SimpleKernelStagePackageBundle, StagePackageBundleBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (stage1, stage1_perf) = build_stage1_packaged_opening_proof_with_perf(build_stage1_selected_opening_claim(
        stage1,
        &stage_claims.stage1.claim,
        &stage_claims.stage1.rows,
    )?)?;
    let (stage2, stage2_perf) = build_stage2_packaged_opening_proof_with_perf(build_stage2_selected_opening_claim(
        stage2,
        &stage_claims.stage2.claim,
        &stage_claims.stage2.families,
    ))?;
    let (stage3, stage3_perf) = build_stage3_packaged_opening_proof_with_perf(build_stage3_selected_opening_claim(
        stage3,
        &stage_claims.stage3.claim,
        &stage_claims.stage3.continuity,
    ))?;
    let bundle = SimpleKernelStagePackageBundle {
        stage1,
        stage2,
        stage3,
        digest: [0; 32],
    };
    let bundle = SimpleKernelStagePackageBundle {
        digest: bundle.expected_digest(),
        ..bundle
    };
    Ok((
        bundle,
        StagePackageBundleBuildPerf {
            stage1: stage1_perf,
            stage2: stage2_perf,
            stage3: stage3_perf,
            total_ms: millis_since(total_started),
        },
    ))
}

fn build_public_stage1_packaged_opening_proof_with_perf(
    claim: Stage1SelectedOpeningClaim,
) -> Result<(Stage1PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    build_stage1_packaged_opening_proof_with_perf(claim)
}

fn build_public_stage2_packaged_opening_proof_with_perf(
    claim: Stage2SelectedOpeningClaim,
) -> Result<(Stage2PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    build_stage2_packaged_opening_proof_with_perf(claim)
}

fn build_public_stage3_packaged_opening_proof_with_perf(
    claim: Stage3SelectedOpeningClaim,
) -> Result<(Stage3PackagedOpeningProof, PackagedOpeningBuildPerf), SimpleKernelError> {
    build_stage3_packaged_opening_proof_with_perf(claim)
}

pub(super) fn build_public_stage_package_bundle_with_perf(
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    stage_claims: &SimpleKernelStageClaimBundle,
) -> Result<(SimpleKernelStagePackageBundle, StagePackageBundleBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (stage1, stage1_perf) = build_public_stage1_packaged_opening_proof_with_perf(
        build_stage1_selected_opening_claim(stage1, &stage_claims.stage1.claim, &stage_claims.stage1.rows)?,
    )?;
    let (stage2, stage2_perf) = build_public_stage2_packaged_opening_proof_with_perf(
        build_stage2_selected_opening_claim(stage2, &stage_claims.stage2.claim, &stage_claims.stage2.families),
    )?;
    let (stage3, stage3_perf) = build_public_stage3_packaged_opening_proof_with_perf(
        build_stage3_selected_opening_claim(stage3, &stage_claims.stage3.claim, &stage_claims.stage3.continuity),
    )?;
    let bundle = SimpleKernelStagePackageBundle {
        stage1,
        stage2,
        stage3,
        digest: [0; 32],
    };
    let bundle = SimpleKernelStagePackageBundle {
        digest: bundle.expected_digest(),
        ..bundle
    };
    Ok((
        bundle,
        StagePackageBundleBuildPerf {
            stage1: stage1_perf,
            stage2: stage2_perf,
            stage3: stage3_perf,
            total_ms: millis_since(total_started),
        },
    ))
}

pub(crate) fn verify_stage_package_bundle_with_perf(
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    stage_packages: &SimpleKernelStagePackageBundle,
    stage_claims: &SimpleKernelStageClaimBundle,
) -> Result<StagePackageBundleVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let stage1_started = Instant::now();
    verify_stage1_packaged_opening_proof(
        &stage_packages.stage1,
        &build_stage1_selected_opening_claim(stage1, &stage_claims.stage1.claim, &stage_claims.stage1.rows)?,
    )?;
    let stage1_ms = millis_since(stage1_started);
    let stage2_started = Instant::now();
    verify_stage2_packaged_opening_proof(
        &stage_packages.stage2,
        &build_stage2_selected_opening_claim(stage2, &stage_claims.stage2.claim, &stage_claims.stage2.families),
    )?;
    let stage2_ms = millis_since(stage2_started);
    let stage3_started = Instant::now();
    verify_stage3_packaged_opening_proof(
        &stage_packages.stage3,
        &build_stage3_selected_opening_claim(stage3, &stage_claims.stage3.claim, &stage_claims.stage3.continuity),
    )?;
    let stage3_ms = millis_since(stage3_started);
    if stage_packages.digest != stage_packages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM stage package bundle digest mismatch".into(),
        ));
    }
    Ok(StagePackageBundleVerifyPerf {
        stage1_ms,
        stage2_ms,
        stage3_ms,
        total_ms: millis_since(total_started),
    })
}
