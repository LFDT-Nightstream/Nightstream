//! Owns the below-export RV64IM side-claim theorem seam.
//!
//! The statement is restricted to already-carried Nightstream surfaces.
//! The witness owns compact single-step packaged claim projections used to bind
//! the carried claim-proof bridges back to the exact stage-claim and
//! kernel-claim public steps, without carrying full `PublicStatement`s or
//! widening the published Nightstream boundary.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::finalize::{digest_public_statement_from_digests, final_main_claim_digests, public_chunk_digest};
use crate::proof::{FoldSchedule, PackagedProof, PublicChunk, PublicStep};
use crate::rv64im::kernel::{
    build_kernel_claim_packaged_public_step_from_compact_surfaces, build_stage_claim_packaged_public_step,
    public_step_digest, same_public_step, Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError,
};

use super::side_bridges::validate_rv64im_side_proof_bundle_structure;
use super::Rv64imSideProofBundle;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imSideClaimRelationStatement {
    pub public_statement: Rv64imProofStatement,
    pub side_bundle: Rv64imSideProofBundle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imSingleStepPackagedProofWitness {
    pub step: PublicStep,
    pub final_main_claim_digests: Vec<[F; 4]>,
    pub proof_digest: [u8; 32],
}

impl Rv64imSingleStepPackagedProofWitness {
    pub(super) fn from_packaged(packaged: &PackagedProof) -> Self {
        let step = packaged
            .statement
            .chunks
            .first()
            .and_then(|chunk| chunk.steps.first())
            .cloned()
            .expect("RV64IM compact packaged witness requires a single public step");
        Self {
            step,
            final_main_claim_digests: final_main_claim_digests(&packaged.statement.final_main_claims),
            proof_digest: packaged.proof.proof_digest,
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/single_step_packaged_witness");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/single_step_packaged_witness/step",
            &public_step_digest(&self.step),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/single_step_packaged_witness/final_main_claim_len",
            &[self.final_main_claim_digests.len() as u64],
        );
        let final_main_claim_fields: Vec<F> = self
            .final_main_claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied())
            .collect();
        tr.append_fields(
            b"neo.fold.next/nightstream/rv64im/single_step_packaged_witness/final_main_claim_digests",
            &final_main_claim_fields,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/single_step_packaged_witness/proof_digest",
            &self.proof_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imSideClaimRelationWitness {
    pub stage_claims_packaged: Rv64imSingleStepPackagedProofWitness,
    pub kernel_claims_packaged: Rv64imSingleStepPackagedProofWitness,
}

impl Rv64imSideClaimRelationWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_claim_relation_witness");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_claim_relation_witness/stage_claims_packaged",
            &self.stage_claims_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_claim_relation_witness/kernel_claims_packaged",
            &self.kernel_claims_packaged.digest(),
        );
        tr.digest32()
    }
}

pub fn build_rv64im_side_claim_relation_statement(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<Rv64imSideClaimRelationStatement, SimpleKernelError> {
    if public_statement.digest != public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-claim relation public statement digest mismatch".into(),
        ));
    }
    validate_rv64im_side_proof_bundle_structure(side_bundle)?;
    Ok(Rv64imSideClaimRelationStatement {
        public_statement: public_statement.clone(),
        side_bundle: side_bundle.clone(),
    })
}

pub fn build_rv64im_side_claim_relation_witness_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Rv64imSideClaimRelationWitness {
    Rv64imSideClaimRelationWitness {
        stage_claims_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(&artifact.stage_claims.packaged),
        kernel_claims_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(&artifact.kernel_claims.packaged),
    }
}

pub fn build_rv64im_side_claim_relation_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imSideClaimRelationStatement, Rv64imSideClaimRelationWitness), SimpleKernelError> {
    let side_bundle = super::build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    let statement = build_rv64im_side_claim_relation_statement(&artifact.statement, &side_bundle)?;
    let witness = build_rv64im_side_claim_relation_witness_from_accepted_artifact(artifact);
    Ok((statement, witness))
}

fn verify_single_step_packaged_witness(
    label: &str,
    expected_step: &PublicStep,
    witness: &Rv64imSingleStepPackagedProofWitness,
    carried_statement_digest: [u8; 32],
    carried_proof_digest: [u8; 32],
    bridge_label: &str,
) -> Result<(), SimpleKernelError> {
    if !same_public_step(&witness.step, expected_step) {
        return Err(SimpleKernelError::Bridge(format!(
            "{label} selected-claim package public step mismatch"
        )));
    }
    let statement_digest = single_step_packaged_statement_digest(&witness.step, &witness.final_main_claim_digests);
    if statement_digest != carried_statement_digest || witness.proof_digest != carried_proof_digest {
        return Err(SimpleKernelError::Bridge(bridge_label.into()));
    }
    Ok(())
}

pub(super) fn single_step_packaged_statement_digest(
    step: &PublicStep,
    final_main_claim_digests: &[[F; 4]],
) -> [u8; 32] {
    let chunk_digest = public_chunk_digest(&PublicChunk {
        start_index: 0,
        steps: vec![step.clone()],
    });
    digest_public_statement_from_digests(FoldSchedule::RowsPerChunk(1), &[chunk_digest], final_main_claim_digests)
}

pub(super) fn verify_rv64im_side_claim_witness_against_compact_surfaces(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    witness: &Rv64imSideClaimRelationWitness,
) -> Result<(), SimpleKernelError> {
    super::verify_rv64im_side_stage_claim_proof_surface(side_bundle, public_statement)?;
    let stage_claims =
        super::build_rv64im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    let expected_stage_step = build_stage_claim_packaged_public_step(&stage_claims)?;
    verify_single_step_packaged_witness(
        "rv64im/stage_claim_bundle",
        &expected_stage_step,
        &witness.stage_claims_packaged,
        side_bundle
            .stage_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.stage_claim_proof_bridge.packaged_proof_digest,
        "RV64IM side-claim relation stage-claim witness does not match the carried side bundle",
    )?;

    let main_lane_bundle_digest = super::verify_rv64im_side_main_lane_proof_surface(side_bundle, public_statement)?;
    super::verify_rv64im_side_kernel_claim_surface(side_bundle, public_statement, main_lane_bundle_digest)?;
    super::verify_rv64im_side_kernel_claim_proof_surface(side_bundle, public_statement)?;
    let expected_kernel_step = build_kernel_claim_packaged_public_step_from_compact_surfaces(
        public_statement.prepared_step_bindings_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .binding_count,
        side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .first_binding_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .last_binding_digest,
        side_bundle.kernel_claim_bridge.root0_digest,
        public_statement.execution_digest,
        public_statement.final_state_digest,
        public_statement.transcript_final_digest,
        public_statement.final_pc,
        public_statement.halted,
    )?;
    verify_single_step_packaged_witness(
        "rv64im/kernel_claim_bundle",
        &expected_kernel_step,
        &witness.kernel_claims_packaged,
        side_bundle
            .kernel_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.kernel_claim_proof_bridge.packaged_proof_digest,
        "RV64IM side-claim relation kernel-claim witness does not match the carried side bundle",
    )?;
    Ok(())
}

pub fn verify_rv64im_side_claim_relation(
    statement: &Rv64imSideClaimRelationStatement,
    witness: &Rv64imSideClaimRelationWitness,
) -> Result<(), SimpleKernelError> {
    if statement.public_statement.digest != statement.public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-claim relation public statement digest mismatch".into(),
        ));
    }
    if statement.side_bundle.digest != statement.side_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-claim relation side-proof bundle digest mismatch".into(),
        ));
    }
    verify_rv64im_side_claim_witness_against_compact_surfaces(
        &statement.public_statement,
        &statement.side_bundle,
        witness,
    )
}
