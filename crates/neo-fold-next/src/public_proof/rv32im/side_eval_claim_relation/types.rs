//! Owns Phase 0 eval-claim statement, witness, and artifact data shapes.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::public_proof::rv32im::side_bridges::Rv32imSideProofBundle;
use crate::rv32im::kernel::{
    AjtaiOpeningProof, CommitmentContextId, FamilyEvalClaimWitness, FamilyEvalSchemaId, OpenedAjtaiCommitmentPublic,
    OpenedAjtaiObjectId, Rv32imEvalClaimBundle, SimpleKernelError,
};
use crate::rv32im::Rv32imProofStatement;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPhase0OpenedObjectSummary {
    pub schema: FamilyEvalSchemaId,
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPhase0OpenedObjectBundle {
    pub objects: Vec<Rv32imPhase0OpenedObjectSummary>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPhase0OpeningTarget {
    pub schema: FamilyEvalSchemaId,
    pub opened_commitment: OpenedAjtaiCommitmentPublic,
    pub opening_proof: AjtaiOpeningProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPhase0OpeningTargetBundle {
    pub targets: Vec<Rv32imPhase0OpeningTarget>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv32imSideEvalClaimRelationStatement {
    pub public_statement: Rv32imProofStatement,
    pub side_bundle: Rv32imSideProofBundle,
    pub phase0_opened_objects: Rv32imPhase0OpenedObjectBundle,
    pub eval_claim_bundle: Rv32imEvalClaimBundle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv32imSideEvalClaimRelationWitness {
    pub claim_witnesses: Vec<FamilyEvalClaimWitness>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imSideEvalClaimArtifact {
    pub statement_digest: [u8; 32],
    pub phase0_opening_targets: Rv32imPhase0OpeningTargetBundle,
    pub eval_claim_bundle: Rv32imEvalClaimBundle,
    pub digest: [u8; 32],
}

impl Rv32imPhase0OpenedObjectSummary {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/phase0_opened_object_summary");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/phase0_opened_object_summary/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/phase0_opened_object_summary/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/phase0_opened_object_summary/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/phase0_opened_object_summary/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.digest32()
    }
}

impl Rv32imPhase0OpenedObjectBundle {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/phase0_opened_object_bundle");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/phase0_opened_object_bundle/count",
            &[self.objects.len() as u64],
        );
        for object in &self.objects {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/phase0_opened_object_bundle/object_digest",
                &object.digest,
            );
        }
        tr.digest32()
    }

    pub(super) fn summary_for_schema(
        &self,
        schema: FamilyEvalSchemaId,
    ) -> Result<&Rv32imPhase0OpenedObjectSummary, SimpleKernelError> {
        self.objects
            .iter()
            .find(|object| object.schema == schema)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV32IM side-eval-claim relation is missing the Phase 0 opened object for {:?}",
                    schema
                ))
            })
    }
}

impl Rv32imPhase0OpeningTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/phase0_opening_target");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/phase0_opening_target/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/phase0_opening_target/opened_commitment_digest",
            &self.opened_commitment.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/phase0_opening_target/opening_proof_digest",
            &self.opening_proof.digest,
        );
        tr.digest32()
    }
}

impl Rv32imPhase0OpeningTargetBundle {
    pub(super) fn validate_canonical_order(&self) -> Result<(), SimpleKernelError> {
        for (index, pair) in self.targets.windows(2).enumerate() {
            if pair[0].schema >= pair[1].schema {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM side-eval-claim artifact Phase 0 opening-target bundle is not in strict schema order at index {}: {:?} then {:?}",
                    index,
                    pair[0].schema,
                    pair[1].schema,
                )));
            }
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/phase0_opening_target_bundle");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/phase0_opening_target_bundle/count",
            &[self.targets.len() as u64],
        );
        for target in &self.targets {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/phase0_opening_target_bundle/target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv32imSideEvalClaimArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_eval_claim_artifact");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_eval_claim_artifact/statement_digest",
            &self.statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_eval_claim_artifact/phase0_opening_targets_digest",
            &self.phase0_opening_targets.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_eval_claim_artifact/eval_claim_bundle_digest",
            &self.eval_claim_bundle.digest,
        );
        tr.digest32()
    }
}
