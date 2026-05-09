//! Owns the published RV32IM opening bundle built from canonical opening claims and a compact proof.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::opening::time::{prove_time_opening_compact, time_opening_compact_proof_digest};
use crate::opening::OpeningClaim;
use crate::opening::TimeOpeningCompactProof;

use super::opening_manifest::{
    opening_claims_from_carriers, stage1_opening_witness_carriers, stage2_opening_witness_carriers,
    stage3_opening_witness_carriers, Rv32imOpeningWitnessCarrier,
};
use super::proof_accepted::Rv32imAcceptedProofArtifact;
use super::proof_staged_verify::{
    derive_stage1_export_proof, derive_stage2_export_proof, derive_stage3_export_proof, Rv32imStage1ExportProof,
    Rv32imStage2ExportProof, Rv32imStage3ExportProof,
};
use super::simple::SimpleKernelError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imOpeningBundle {
    pub claims: Vec<OpeningClaim>,
    pub compact_proof: TimeOpeningCompactProof,
    pub digest: [u8; 32],
}

impl Rv32imOpeningBundle {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_bundle");
        tr.append_message(b"neo.fold.next/rv32im/opening_bundle/version", b"v1");
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_bundle/claim_count",
            &[self.claims.len() as u64],
        );
        for claim in &self.claims {
            tr.append_message(b"neo.fold.next/rv32im/opening_bundle/claim_digest", &claim.digest);
        }
        tr.append_message(
            b"neo.fold.next/rv32im/opening_bundle/compact_proof_digest",
            &time_opening_compact_proof_digest(&self.compact_proof),
        );
        tr.digest32()
    }
}

pub fn build_rv32im_opening_bundle_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imOpeningBundle, SimpleKernelError> {
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
    build_rv32im_opening_bundle_from_stage_exports(&stage1, &stage2, &stage3)
}

pub(crate) fn build_rv32im_opening_bundle_from_carriers(
    carriers: &[Rv32imOpeningWitnessCarrier],
) -> Result<Rv32imOpeningBundle, SimpleKernelError> {
    let claims = opening_claims_from_carriers(carriers);
    let compact_proof = prove_time_opening_compact(&[], &claims)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM opening bundle prove failed: {err}")))?;
    let mut bundle = Rv32imOpeningBundle {
        claims,
        compact_proof,
        digest: [0; 32],
    };
    bundle.digest = bundle.expected_digest();
    Ok(bundle)
}

pub(crate) fn build_rv32im_opening_bundle_from_stage_exports(
    stage1: &Rv32imStage1ExportProof,
    stage2: &Rv32imStage2ExportProof,
    stage3: &Rv32imStage3ExportProof,
) -> Result<Rv32imOpeningBundle, SimpleKernelError> {
    let mut carriers = Vec::new();
    carriers.extend(stage1_opening_witness_carriers(&stage1.selected_opening));
    carriers.extend(stage2_opening_witness_carriers(&stage2.selected_opening));
    carriers.extend(stage3_opening_witness_carriers(&stage3.selected_opening));
    build_rv32im_opening_bundle_from_carriers(&carriers)
}
