//! Owns verification for the first published RV32IM opening bundle.

use crate::opening::time::{prove_time_opening_compact, verify_time_opening_compact};
use crate::rv32im::kernel::SimpleKernelError;

use super::opening_batch::Rv32imOpeningBundle;
use super::opening_manifest::opening_claims_from_carriers;
use super::proof_accepted::Rv32imAcceptedProofArtifact;
use super::proof_staged_verify::{derive_stage1_export_proof, derive_stage2_export_proof, derive_stage3_export_proof};

pub fn verify_rv32im_opening_bundle_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
    bundle: &Rv32imOpeningBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM opening bundle digest mismatch".into(),
        ));
    }
    verify_time_opening_compact(&[], &bundle.claims, &bundle.compact_proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM opening summary verify failed: {err}")))?;

    let stage1 = derive_stage1_export_proof(&artifact.root_execution.execution_rows, &artifact.stage_packages);
    let stage2 = derive_stage2_export_proof(&artifact.root_execution.execution_rows, &artifact.stage_packages);
    let stage3 = derive_stage3_export_proof(
        &artifact.root_execution.execution_rows,
        &artifact.root_execution,
        &artifact.stage_packages,
        artifact.statement.initial_pc,
        artifact.statement.final_pc,
        stage2.temporal_digest,
    );
    let mut expected_carriers = Vec::new();
    expected_carriers.extend(super::opening_manifest::stage1_opening_witness_carriers(
        &stage1.selected_opening,
    ));
    expected_carriers.extend(super::opening_manifest::stage2_opening_witness_carriers(
        &stage2.selected_opening,
    ));
    expected_carriers.extend(super::opening_manifest::stage3_opening_witness_carriers(
        &stage3.selected_opening,
    ));
    let expected_claims = opening_claims_from_carriers(&expected_carriers);
    if expected_claims != bundle.claims {
        return Err(SimpleKernelError::Bridge(
            "RV32IM opening bundle claims do not match the accepted artifact".into(),
        ));
    }
    let expected_compact_proof = prove_time_opening_compact(&[], &expected_claims)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM opening summary build failed: {err}")))?;
    if expected_compact_proof != bundle.compact_proof {
        return Err(SimpleKernelError::Bridge(
            "RV32IM opening bundle grouped summary does not match the accepted artifact".into(),
        ));
    }
    Ok(())
}
