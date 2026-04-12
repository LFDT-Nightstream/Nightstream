//! Owns the below-export RV64IM Phase 0 eval-claim theorem seam.
//!
//! The statement binds compact opened-object summaries and the carried
//! side-proof bundle. Nightstream Phase 0 claims are rebuilt against that
//! theorem-bearing side bundle; this file does not carry a second digest
//! authority for the same per-stage binding seed.

use std::collections::BTreeMap;
use std::sync::Arc;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv64im::kernel::{
    build_rv64im_eval_claim_bundle_from_claim_witnesses,
    build_rv64im_eval_claim_bundle_from_claim_witnesses_trusted_local,
    build_rv64im_eval_claim_witnesses_from_accepted_artifact, derive_phase0_point, phase0_binding_digest,
    rebuild_opened_object_witness_from_projection, AjtaiOpeningProof, CommitmentContextId, FamilyEvalClaim,
    FamilyEvalClaimWitness, FamilyEvalPayload, FamilyEvalSchemaId, OpenedAjtaiCommitmentPublic, OpenedAjtaiObjectId,
    OpenedAjtaiObjectWitness, Rv64imAcceptedProofArtifact, Rv64imEvalClaimBundle, Rv64imPhase0BindingSurface,
    Rv64imPhase0BindingTarget, SimpleKernelError,
};
use crate::rv64im::Rv64imProofStatement;

use super::side_bridges::validate_rv64im_side_proof_bundle_structure;
use super::{
    bind_rv64im_side_proof_bundle_to_statement_core, build_rv64im_side_proof_bundle_from_accepted_artifact,
    Rv64imSideProofBundle,
};

pub(super) fn active_phase0_schemas_from_side_bundle(side_bundle: &Rv64imSideProofBundle) -> Vec<FamilyEvalSchemaId> {
    let mut schemas = vec![FamilyEvalSchemaId::Stage1Rows];
    if side_bundle.stage2.claim.register_read_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterReads);
    }
    if side_bundle.stage2.claim.register_write_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterWrites);
    }
    if side_bundle.stage2.claim.ram_event_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RamEvents);
    }
    if side_bundle.stage2.claim.twist_link_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2TwistLinks);
    }
    if side_bundle.stage3.claim.continuity_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage3Continuity);
    }
    schemas
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0OpenedObjectSummary {
    pub schema: FamilyEvalSchemaId,
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0OpenedObjectBundle {
    pub objects: Vec<Rv64imPhase0OpenedObjectSummary>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0OpeningTarget {
    pub schema: FamilyEvalSchemaId,
    pub opened_commitment: OpenedAjtaiCommitmentPublic,
    pub opening_proof: AjtaiOpeningProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0OpeningTargetBundle {
    pub targets: Vec<Rv64imPhase0OpeningTarget>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv64imSideEvalClaimRelationStatement {
    pub public_statement: Rv64imProofStatement,
    pub side_bundle: Rv64imSideProofBundle,
    pub phase0_opened_objects: Rv64imPhase0OpenedObjectBundle,
    pub eval_claim_bundle: Rv64imEvalClaimBundle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv64imSideEvalClaimRelationWitness {
    pub claim_witnesses: Vec<FamilyEvalClaimWitness>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideEvalClaimArtifact {
    pub statement_digest: [u8; 32],
    pub phase0_opening_targets: Rv64imPhase0OpeningTargetBundle,
    pub eval_claim_bundle: Rv64imEvalClaimBundle,
    pub digest: [u8; 32],
}

impl Rv64imPhase0OpenedObjectSummary {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/phase0_opened_object_summary");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/phase0_opened_object_summary/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/phase0_opened_object_summary/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/phase0_opened_object_summary/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/phase0_opened_object_summary/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.digest32()
    }
}

impl Rv64imPhase0OpenedObjectBundle {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/phase0_opened_object_bundle");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/phase0_opened_object_bundle/count",
            &[self.objects.len() as u64],
        );
        for object in &self.objects {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/phase0_opened_object_bundle/object_digest",
                &object.digest,
            );
        }
        tr.digest32()
    }

    fn summary_for_schema(
        &self,
        schema: FamilyEvalSchemaId,
    ) -> Result<&Rv64imPhase0OpenedObjectSummary, SimpleKernelError> {
        self.objects
            .iter()
            .find(|object| object.schema == schema)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim relation is missing the Phase 0 opened object for {:?}",
                    schema
                ))
            })
    }
}

impl Rv64imPhase0OpeningTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/phase0_opening_target");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/phase0_opening_target/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/phase0_opening_target/opened_commitment_digest",
            &self.opened_commitment.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/phase0_opening_target/opening_proof_digest",
            &self.opening_proof.digest,
        );
        tr.digest32()
    }
}

impl Rv64imPhase0OpeningTargetBundle {
    fn validate_canonical_order(&self) -> Result<(), SimpleKernelError> {
        for (index, pair) in self.targets.windows(2).enumerate() {
            if pair[0].schema >= pair[1].schema {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim artifact Phase 0 opening-target bundle is not in strict schema order at index {}: {:?} then {:?}",
                    index,
                    pair[0].schema,
                    pair[1].schema,
                )));
            }
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/phase0_opening_target_bundle");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/phase0_opening_target_bundle/count",
            &[self.targets.len() as u64],
        );
        for target in &self.targets {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/phase0_opening_target_bundle/target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imSideEvalClaimArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_eval_claim_artifact");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_eval_claim_artifact/statement_digest",
            &self.statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_eval_claim_artifact/phase0_opening_targets_digest",
            &self.phase0_opening_targets.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_eval_claim_artifact/eval_claim_bundle_digest",
            &self.eval_claim_bundle.digest,
        );
        tr.digest32()
    }
}

fn rv64im_side_eval_claim_relation_statement_digest_from_surfaces(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    phase0_opened_objects: &Rv64imPhase0OpenedObjectBundle,
    eval_claim_bundle: &Rv64imEvalClaimBundle,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/public_statement_digest",
        &public_statement.digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/side_bundle_digest",
        &side_bundle.digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/phase0_opened_objects_digest",
        &phase0_opened_objects.digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/eval_claim_bundle_digest",
        &eval_claim_bundle.digest,
    );
    tr.digest32()
}

pub fn rv64im_side_eval_claim_relation_statement_digest(statement: &Rv64imSideEvalClaimRelationStatement) -> [u8; 32] {
    rv64im_side_eval_claim_relation_statement_digest_from_surfaces(
        &statement.public_statement,
        &statement.side_bundle,
        &statement.phase0_opened_objects,
        &statement.eval_claim_bundle,
    )
}

pub fn build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imPhase0OpenedObjectBundle, SimpleKernelError> {
    let mut by_schema = BTreeMap::<FamilyEvalSchemaId, (&OpenedAjtaiObjectId, CommitmentContextId)>::new();
    for claim_witness in claim_witnesses {
        let schema = claim_witness.claim.payload.schema;
        let opened_object = &claim_witness.claim.opened_object;
        let commitment_context = claim_witness.claim.commitment_context;
        if let Some((expected_object, expected_context)) = by_schema.get(&schema) {
            if *expected_object != opened_object || *expected_context != commitment_context {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim relation found inconsistent opened-object identities for {:?}",
                    schema
                )));
            }
            continue;
        }
        by_schema.insert(schema, (opened_object, commitment_context));
    }

    let mut objects = Vec::new();
    for (schema, (opened_object, commitment_context)) in by_schema {
        let mut summary = Rv64imPhase0OpenedObjectSummary {
            schema,
            opened_object: opened_object.clone(),
            commitment_context,
            digest: [0; 32],
        };
        summary.digest = summary.expected_digest();
        objects.push(summary);
    }
    let mut bundle = Rv64imPhase0OpenedObjectBundle {
        objects,
        digest: [0; 32],
    };
    bundle.digest = bundle.expected_digest();
    Ok(bundle)
}

fn build_rv64im_phase0_opening_target_bundle_from_claim_witnesses(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imPhase0OpeningTargetBundle, SimpleKernelError> {
    let mut by_schema = BTreeMap::<FamilyEvalSchemaId, &FamilyEvalClaimWitness>::new();
    for claim_witness in claim_witnesses {
        let schema = claim_witness.claim.payload.schema;
        if let Some(expected) = by_schema.get(&schema) {
            if expected.claim.opened_object != claim_witness.claim.opened_object
                || expected.claim.commitment_context != claim_witness.claim.commitment_context
            {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim artifact found inconsistent opened-object identities for {:?}",
                    schema
                )));
            }
            continue;
        }
        by_schema.insert(schema, claim_witness);
    }

    let mut targets = Vec::new();
    for (index, (schema, representative)) in by_schema.into_iter().enumerate() {
        let opened_commitment = OpenedAjtaiCommitmentPublic::new(
            representative.witness.opened_object.clone(),
            &representative.witness.commitment_context,
            representative.witness.commitment_vector.clone(),
            representative.claim.payload.column_evals.len(),
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact could not build the Phase 0 opened commitment for {:?}: {err}",
                schema
            ))
        })?;
        let opening_proof = AjtaiOpeningProof::new(representative.witness.packed_columns.clone());
        rebuild_opened_object_witness_from_projection(
            schema,
            &representative.witness.commitment_context,
            representative.claim.payload.column_evals.len(),
            &opened_commitment,
            &opening_proof,
            index,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact could not rebuild the Phase 0 witness for {:?}: {err}",
                schema
            ))
        })?;
        let target = Rv64imPhase0OpeningTarget {
            schema,
            opened_commitment,
            opening_proof,
            digest: [0; 32],
        };
        targets.push(Rv64imPhase0OpeningTarget {
            digest: target.expected_digest(),
            ..target
        });
    }

    let bundle = Rv64imPhase0OpeningTargetBundle {
        targets,
        digest: [0; 32],
    };
    Ok(Rv64imPhase0OpeningTargetBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

fn build_rv64im_phase0_opened_object_bundle_from_opening_targets(
    eval_claim_bundle: &Rv64imEvalClaimBundle,
    phase0_opening_targets: &Rv64imPhase0OpeningTargetBundle,
) -> Result<Rv64imPhase0OpenedObjectBundle, SimpleKernelError> {
    phase0_opening_targets.validate_canonical_order()?;
    let mut objects = Vec::new();
    for target in &phase0_opening_targets.targets {
        let schema = target.schema;
        let mut claims = eval_claim_bundle
            .claims
            .iter()
            .filter(|claim| claim.payload.schema == schema);
        let representative = claims.next().ok_or_else(|| {
            SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact is missing claims for {:?}",
                schema
            ))
        })?;
        if representative.opened_object != target.opened_commitment.opened_object {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact {:?} opened-object mismatch between carried claims and opening target",
                schema
            )));
        }
        for claim in claims {
            if claim.opened_object != representative.opened_object
                || claim.commitment_context != representative.commitment_context
            {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim artifact found inconsistent claim contexts for {:?}",
                    schema
                )));
            }
        }
        let summary = Rv64imPhase0OpenedObjectSummary {
            schema,
            opened_object: target.opened_commitment.opened_object.clone(),
            commitment_context: representative.commitment_context,
            digest: [0; 32],
        };
        objects.push(Rv64imPhase0OpenedObjectSummary {
            digest: summary.expected_digest(),
            ..summary
        });
    }

    let bundle = Rv64imPhase0OpenedObjectBundle {
        objects,
        digest: [0; 32],
    };
    Ok(Rv64imPhase0OpenedObjectBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

fn validate_rv64im_side_eval_claim_relation_inputs(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    phase0_opened_objects: &Rv64imPhase0OpenedObjectBundle,
    eval_claim_bundle: &Rv64imEvalClaimBundle,
) -> Result<(), SimpleKernelError> {
    if public_statement.digest != public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation public statement digest mismatch".into(),
        ));
    }
    validate_rv64im_side_proof_bundle_structure(side_bundle)?;
    if phase0_opened_objects.digest != phase0_opened_objects.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation Phase 0 opened-object bundle digest mismatch".into(),
        ));
    }
    if eval_claim_bundle.digest != eval_claim_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation eval-claim bundle digest mismatch".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_side_eval_claim_relation_statement(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    phase0_opened_objects: &Rv64imPhase0OpenedObjectBundle,
    eval_claim_bundle: &Rv64imEvalClaimBundle,
) -> Result<Rv64imSideEvalClaimRelationStatement, SimpleKernelError> {
    validate_rv64im_side_eval_claim_relation_inputs(
        public_statement,
        side_bundle,
        phase0_opened_objects,
        eval_claim_bundle,
    )?;
    Ok(Rv64imSideEvalClaimRelationStatement {
        public_statement: public_statement.clone(),
        side_bundle: side_bundle.clone(),
        phase0_opened_objects: phase0_opened_objects.clone(),
        eval_claim_bundle: eval_claim_bundle.clone(),
    })
}

pub fn build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideEvalClaimRelationWitness, SimpleKernelError> {
    let side_bundle = build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(&side_bundle, artifact)
}

pub(crate) fn build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideEvalClaimRelationWitness, SimpleKernelError> {
    Ok(Rv64imSideEvalClaimRelationWitness {
        claim_witnesses: rebind_phase0_claim_witnesses_to_side_bundle(
            side_bundle,
            &build_rv64im_eval_claim_witnesses_from_accepted_artifact(artifact)?,
        )?,
    })
}

pub fn build_rv64im_side_eval_claim_relation_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imSideEvalClaimRelationStatement, Rv64imSideEvalClaimRelationWitness), SimpleKernelError> {
    let side_bundle = build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    let witness =
        build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(&side_bundle, artifact)?;
    let phase0_opened_objects =
        build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(&witness.claim_witnesses)?;
    let eval_claim_bundle = build_rv64im_eval_claim_bundle_from_claim_witnesses(&witness.claim_witnesses)?;
    let statement = build_rv64im_side_eval_claim_relation_statement(
        &artifact.statement,
        &side_bundle,
        &phase0_opened_objects,
        &eval_claim_bundle,
    )?;
    Ok((statement, witness))
}

pub fn verify_rv64im_side_eval_claim_relation(
    statement: &Rv64imSideEvalClaimRelationStatement,
    witness: &Rv64imSideEvalClaimRelationWitness,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_side_eval_claim_relation_surfaces(statement, &witness.claim_witnesses)
}

fn verify_rv64im_side_eval_claim_relation_surfaces(
    statement: &Rv64imSideEvalClaimRelationStatement,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    if statement.public_statement.digest != statement.public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation public statement digest mismatch".into(),
        ));
    }
    validate_rv64im_side_proof_bundle_structure(&statement.side_bundle)?;
    if statement.phase0_opened_objects.digest != statement.phase0_opened_objects.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation Phase 0 opened-object bundle digest mismatch".into(),
        ));
    }
    if statement.eval_claim_bundle.digest != statement.eval_claim_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation eval-claim bundle digest mismatch".into(),
        ));
    }

    let expected_opened_objects = build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(claim_witnesses)?;
    if expected_opened_objects != statement.phase0_opened_objects {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation opened-object summaries do not match the carried Phase 0 bundle".into(),
        ));
    }

    let expected_claim_bundle = build_rv64im_eval_claim_bundle_from_claim_witnesses(claim_witnesses)?;
    if expected_claim_bundle != statement.eval_claim_bundle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim relation claim witnesses do not match the carried eval-claim bundle".into(),
        ));
    }

    verify_phase0_claim_bindings(statement, claim_witnesses)?;
    verify_phase0_claim_coverage(&statement.side_bundle, claim_witnesses)?;
    Ok(())
}

pub fn build_rv64im_side_eval_claim_artifact(
    statement: &Rv64imSideEvalClaimRelationStatement,
    witness: &Rv64imSideEvalClaimRelationWitness,
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    verify_rv64im_side_eval_claim_relation(statement, witness)?;
    let phase0_opening_targets =
        build_rv64im_phase0_opening_target_bundle_from_claim_witnesses(&witness.claim_witnesses)?;
    let mut artifact = Rv64imSideEvalClaimArtifact {
        statement_digest: rv64im_side_eval_claim_relation_statement_digest(statement),
        phase0_opening_targets,
        eval_claim_bundle: statement.eval_claim_bundle.clone(),
        digest: [0; 32],
    };
    artifact.digest = artifact.expected_digest();
    Ok(artifact)
}

pub fn build_rv64im_side_eval_claim_artifact_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    let (statement, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(artifact)?;
    build_rv64im_side_eval_claim_artifact(&statement, &witness)
}

pub(crate) fn build_rv64im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    let expected_side_bundle = bind_rv64im_side_proof_bundle_to_statement_core(
        &build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?,
        side_bundle.statement_core_digest,
    )?;
    if &expected_side_bundle != side_bundle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact side-proof bundle does not match the accepted artifact".into(),
        ));
    }
    let witness =
        build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(side_bundle, artifact)?;
    build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle(
        public_statement,
        side_bundle,
        &witness.claim_witnesses,
    )
}

pub(super) fn build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    let phase0_opened_objects = build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(claim_witnesses)?;
    let phase0_opening_targets = build_rv64im_phase0_opening_target_bundle_from_claim_witnesses(claim_witnesses)?;
    let eval_claim_bundle = build_rv64im_eval_claim_bundle_from_claim_witnesses(claim_witnesses)?;
    validate_rv64im_side_eval_claim_relation_inputs(
        public_statement,
        side_bundle,
        &phase0_opened_objects,
        &eval_claim_bundle,
    )?;
    build_rv64im_side_eval_claim_artifact_from_trusted_surfaces(
        public_statement,
        side_bundle,
        phase0_opening_targets,
        eval_claim_bundle,
    )
}

pub(super) fn build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_trusted_side_bundle(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    let phase0_opening_targets = build_rv64im_phase0_opening_target_bundle_from_claim_witnesses(claim_witnesses)?;
    let eval_claim_bundle = build_rv64im_eval_claim_bundle_from_claim_witnesses_trusted_local(claim_witnesses)?;
    build_rv64im_side_eval_claim_artifact_from_trusted_surfaces(
        public_statement,
        side_bundle,
        phase0_opening_targets,
        eval_claim_bundle,
    )
}

fn build_rv64im_side_eval_claim_artifact_from_trusted_surfaces(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    phase0_opening_targets: Rv64imPhase0OpeningTargetBundle,
    eval_claim_bundle: Rv64imEvalClaimBundle,
) -> Result<Rv64imSideEvalClaimArtifact, SimpleKernelError> {
    let phase0_opened_objects =
        build_rv64im_phase0_opened_object_bundle_from_opening_targets(&eval_claim_bundle, &phase0_opening_targets)?;
    let mut artifact = Rv64imSideEvalClaimArtifact {
        statement_digest: rv64im_side_eval_claim_relation_statement_digest_from_surfaces(
            public_statement,
            side_bundle,
            &phase0_opened_objects,
            &eval_claim_bundle,
        ),
        phase0_opening_targets,
        eval_claim_bundle,
        digest: [0; 32],
    };
    artifact.digest = artifact.expected_digest();
    Ok(artifact)
}

pub fn build_rv64im_side_eval_claim_relation_statement_from_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imSideEvalClaimArtifact,
) -> Result<Rv64imSideEvalClaimRelationStatement, SimpleKernelError> {
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact digest mismatch".into(),
        ));
    }
    let phase0_opened_objects = build_rv64im_phase0_opened_object_bundle_from_opening_targets(
        &artifact.eval_claim_bundle,
        &artifact.phase0_opening_targets,
    )?;
    build_rv64im_side_eval_claim_relation_statement(
        public_statement,
        side_bundle,
        &phase0_opened_objects,
        &artifact.eval_claim_bundle,
    )
}

pub fn verify_rv64im_side_eval_claim_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imSideEvalClaimArtifact,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_side_eval_claim_artifact_structure(artifact)?;
    let statement =
        build_rv64im_side_eval_claim_relation_statement_from_artifact(public_statement, side_bundle, artifact)?;
    if artifact.statement_digest != rv64im_side_eval_claim_relation_statement_digest(&statement) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact statement digest does not match the carried relation statement".into(),
        ));
    }
    let claim_witnesses = rebuild_phase0_claim_witnesses_from_artifact(artifact)?;
    verify_rv64im_side_eval_claim_relation_surfaces(&statement, &claim_witnesses)
}

fn validate_rv64im_side_eval_claim_artifact_structure(
    artifact: &Rv64imSideEvalClaimArtifact,
) -> Result<(), SimpleKernelError> {
    artifact.phase0_opening_targets.validate_canonical_order()?;
    if artifact.phase0_opening_targets.digest != artifact.phase0_opening_targets.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact Phase 0 opening-target bundle digest mismatch".into(),
        ));
    }
    for target in &artifact.phase0_opening_targets.targets {
        if target.opened_commitment.digest != target.opened_commitment.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact {:?} opened-commitment digest mismatch",
                target.schema
            )));
        }
        if target.opening_proof.digest != target.opening_proof.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact {:?} opening-proof digest mismatch",
                target.schema
            )));
        }
        if target.digest != target.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact {:?} opening target digest mismatch",
                target.schema
            )));
        }
    }
    if artifact.eval_claim_bundle.digest != artifact.eval_claim_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact eval-claim bundle digest mismatch".into(),
        ));
    }
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-eval-claim artifact digest mismatch".into(),
        ));
    }
    Ok(())
}

fn rebuild_phase0_claim_witnesses_from_artifact(
    artifact: &Rv64imSideEvalClaimArtifact,
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    let mut witness_by_schema = BTreeMap::<FamilyEvalSchemaId, Arc<OpenedAjtaiObjectWitness>>::new();
    for (index, target) in artifact.phase0_opening_targets.targets.iter().enumerate() {
        let schema = target.schema;
        let representative = artifact
            .eval_claim_bundle
            .claims
            .iter()
            .find(|claim| claim.payload.schema == schema)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim artifact is missing claims for {:?}",
                    schema
                ))
            })?;
        let witness = rebuild_opened_object_witness_from_projection(
            schema,
            &representative.commitment_context,
            representative.payload.column_evals.len(),
            &target.opened_commitment,
            &target.opening_proof,
            index,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim artifact could not rebuild the Phase 0 witness for {:?}: {err}",
                schema
            ))
        })?;
        witness_by_schema.insert(schema, Arc::new(witness));
    }

    artifact
        .eval_claim_bundle
        .claims
        .iter()
        .map(|claim| {
            let witness = witness_by_schema
                .get(&claim.payload.schema)
                .ok_or_else(|| {
                    SimpleKernelError::Bridge(format!(
                        "RV64IM side-eval-claim artifact is missing a reconstructed witness for {:?}",
                        claim.payload.schema
                    ))
                })?
                .clone();
            FamilyEvalClaimWitness::new(claim.clone(), witness).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM side-eval-claim artifact {:?} claim payload does not match the carried opening target: {err}",
                    claim.payload.schema
                ))
            })
        })
        .collect()
}

fn verify_phase0_claim_bindings(
    statement: &Rv64imSideEvalClaimRelationStatement,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    for claim_witness in claim_witnesses {
        verify_phase0_claim_surface(statement, &claim_witness.claim)?;
    }
    Ok(())
}

fn verify_phase0_claim_surface(
    statement: &Rv64imSideEvalClaimRelationStatement,
    claim: &FamilyEvalClaim,
) -> Result<(), SimpleKernelError> {
    let summary = statement
        .phase0_opened_objects
        .summary_for_schema(claim.payload.schema)?;
    if claim.opened_object != summary.opened_object || claim.commitment_context != summary.commitment_context {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side-eval-claim relation {:?} claim does not match the carried opened-object summary",
            claim.payload.schema
        )));
    }

    let expected_binding_digest = phase0_binding_digest(
        &summary.opened_object,
        claim.payload.schema,
        claim.id.slot,
        family_binding_anchor_digest(&statement.side_bundle, claim.payload.schema),
        phase0_stage_binding_digest_from_side_bundle(&statement.side_bundle, claim.payload.schema),
    );
    if expected_binding_digest != claim.binding_digest {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side-eval-claim relation {:?} binding digest does not match the carried theorem surfaces",
            claim.payload.schema
        )));
    }

    let expected_point = derive_phase0_point(
        &summary.opened_object,
        &summary.commitment_context,
        claim.payload.schema,
        claim.id.slot,
        claim.binding_digest,
    );
    if expected_point != claim.point {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side-eval-claim relation {:?} point does not match the carried Phase 0 derivation",
            claim.payload.schema
        )));
    }

    Ok(())
}

fn verify_phase0_claim_coverage(
    side_bundle: &Rv64imSideProofBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    let mut actual_schemas = claim_witnesses
        .iter()
        .map(|claim| claim.claim.payload.schema)
        .collect::<Vec<_>>();
    actual_schemas.sort_unstable();
    actual_schemas.dedup();
    let expected_schemas = active_phase0_schemas_from_side_bundle(side_bundle);
    if actual_schemas != expected_schemas {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side-eval-claim relation active schema mismatch: expected {:?}, got {:?}",
            expected_schemas, actual_schemas
        )));
    }
    for schema in expected_schemas {
        let expected_slots = expected_slots_for_schema(schema);
        let actual_slots = claim_witnesses
            .iter()
            .filter(|claim| claim.claim.payload.schema == schema)
            .map(|claim| claim.claim.id.slot)
            .collect::<Vec<_>>();
        if actual_slots != expected_slots {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM side-eval-claim relation {:?} slot coverage mismatch: expected {:?}, got {:?}",
                schema, expected_slots, actual_slots
            )));
        }
    }
    Ok(())
}

fn expected_slots_for_schema(schema: FamilyEvalSchemaId) -> Vec<u32> {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => vec![0, 1, 2, 3],
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks
        | FamilyEvalSchemaId::Stage3Continuity => vec![0],
    }
}

fn family_binding_anchor_digest(side_bundle: &Rv64imSideProofBundle, schema: FamilyEvalSchemaId) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => side_bundle.stage1.rows_digest,
        FamilyEvalSchemaId::Stage2RegisterReads => side_bundle.stage2.claim.register_reads_family_digest,
        FamilyEvalSchemaId::Stage2RegisterWrites => side_bundle.stage2.claim.register_writes_family_digest,
        FamilyEvalSchemaId::Stage2RamEvents => side_bundle.stage2.claim.ram_events_family_digest,
        FamilyEvalSchemaId::Stage2TwistLinks => side_bundle.stage2.claim.twist_links_family_digest,
        FamilyEvalSchemaId::Stage3Continuity => side_bundle.stage3.claim.continuity_family_digest,
    }
}

fn phase0_stage_binding_digest_from_side_bundle(
    side_bundle: &Rv64imSideProofBundle,
    schema: FamilyEvalSchemaId,
) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => side_bundle.stage1.digest,
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks => side_bundle.stage2.digest,
        FamilyEvalSchemaId::Stage3Continuity => side_bundle.stage3.digest,
    }
}

pub(crate) fn build_rv64im_phase0_binding_surface_from_side_bundle(
    side_bundle: &Rv64imSideProofBundle,
) -> Rv64imPhase0BindingSurface {
    let targets = active_phase0_schemas_from_side_bundle(side_bundle)
        .into_iter()
        .map(|schema| {
            let mut target = Rv64imPhase0BindingTarget {
                schema,
                family_binding_anchor_digest: family_binding_anchor_digest(side_bundle, schema),
                stage_proof_binding_digest: phase0_stage_binding_digest_from_side_bundle(side_bundle, schema),
                digest: [0; 32],
            };
            target.digest = target.expected_digest();
            target
        })
        .collect();
    let mut surface = Rv64imPhase0BindingSurface {
        targets,
        digest: [0; 32],
    };
    surface.digest = surface.expected_digest();
    surface
}

pub(crate) fn rebind_phase0_claim_witnesses_to_side_bundle(
    side_bundle: &Rv64imSideProofBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    claim_witnesses
        .iter()
        .map(|claim_witness| {
            let schema = claim_witness.claim.payload.schema;
            let binding_digest = phase0_binding_digest(
                &claim_witness.claim.opened_object,
                schema,
                claim_witness.claim.id.slot,
                family_binding_anchor_digest(side_bundle, schema),
                phase0_stage_binding_digest_from_side_bundle(side_bundle, schema),
            );
            let point = derive_phase0_point(
                &claim_witness.claim.opened_object,
                &claim_witness.claim.commitment_context,
                schema,
                claim_witness.claim.id.slot,
                binding_digest,
            );
            let payload = FamilyEvalPayload::new(
                schema,
                claim_witness
                    .witness
                    .evaluate_payload(&point)
                    .map_err(|err| {
                        SimpleKernelError::Bridge(format!(
                            "RV64IM Nightstream Phase 0 {:?} payload evaluation failed while rebinding to the carried side bundle: {err}",
                            schema
                        ))
                    })?,
            )
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Nightstream Phase 0 {:?} payload rebuild failed while rebinding to the carried side bundle: {err}",
                    schema
                ))
            })?;
            let claim = FamilyEvalClaim::new(
                claim_witness.claim.opened_object.clone(),
                claim_witness.claim.id.slot,
                claim_witness.claim.commitment_context,
                point,
                payload,
                binding_digest,
            )
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Nightstream Phase 0 {:?} claim rebuild failed while rebinding to the carried side bundle: {err}",
                    schema
                ))
            })?;
            FamilyEvalClaimWitness::new(claim, claim_witness.witness.clone()).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Nightstream Phase 0 {:?} rebound claim does not match the carried opened-object witness: {err}",
                    schema
                ))
            })
        })
        .collect()
}
