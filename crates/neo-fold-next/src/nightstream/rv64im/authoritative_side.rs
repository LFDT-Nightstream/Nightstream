//! Owns the authoritative RV64IM Nightstream side public instance and its current direct verifier.
//!
//! This module owns:
//! - the theorem-facing side public instance
//! - the compact statement that Spartan binds on-chain
//! - the current direct verifier for the carried side proof container
//!
//! It does not own:
//! - the final compact opening/eval proof backend
//! - raw Phase 0 witness replay inside Spartan

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{
    CommitmentContextId, FamilyEvalClaim, FamilyEvalSchemaId, OpenedAjtaiObjectId, Rv64imAcceptedProofArtifact,
    Rv64imProofStatement, SimpleKernelError,
};

use super::opening_artifact::{
    build_rv64im_opening_artifact_from_accepted_artifact, verify_rv64im_opening_artifact_from_side_proof_bundle,
    Rv64imOpeningArtifact,
};
use super::side_bridges::validate_rv64im_side_proof_bundle_structure;
use super::side_eval_claim_relation::active_phase0_schemas_from_side_bundle;
use super::side_opening_relation::{
    build_rv64im_side_selected_opening_witness_from_accepted_artifact,
    verify_rv64im_side_selected_opening_witness_against_compact_surfaces, Rv64imSideSelectedOpeningWitness,
};
use super::{
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces,
    verify_rv64im_root_execution_surface_against_compact_surfaces, Rv64imSideProofBundle,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideSurfaceTarget {
    pub schema: FamilyEvalSchemaId,
    pub slot: u32,
    pub family_binding_anchor_digest: [u8; 32],
    pub stage_proof_binding_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideSurfacePublic {
    pub targets: Vec<Rv64imSideSurfaceTarget>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imOpenedObjectPublic {
    pub schema: FamilyEvalSchemaId,
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imEvalPublic {
    pub claim: FamilyEvalClaim,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAuthoritativeSidePublicInstance {
    pub nightstream_statement_core_digest: [u8; 32],
    pub side_surface_public: Rv64imSideSurfacePublic,
    pub opened_objects: Vec<Rv64imOpenedObjectPublic>,
    pub evals: Vec<Rv64imEvalPublic>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideProofContainer {
    pub public_instance: Rv64imAuthoritativeSidePublicInstance,
    pub side_bundle: Rv64imSideProofBundle,
    pub opening_witness: Rv64imSideSelectedOpeningWitness,
    pub opening_artifact: Rv64imOpeningArtifact,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAuthoritativeSideStatement {
    pub nightstream_statement_core_digest: [u8; 32],
    pub public_instance_digest: [u8; 32],
}

impl Rv64imSideSurfaceTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_target");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_target/meta",
            &[self.schema.tag(), self.slot as u64],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_target/family_binding_anchor_digest",
            &self.family_binding_anchor_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_target/stage_proof_binding_digest",
            &self.stage_proof_binding_digest,
        );
        tr.digest32()
    }
}

impl Rv64imSideSurfacePublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_public");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_public/count",
            &[self.targets.len() as u64],
        );
        for target in &self.targets {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/authoritative_side/surface_public/target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imOpenedObjectPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/opened_object");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/opened_object/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/opened_object/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/opened_object/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/opened_object/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.digest32()
    }
}

impl Rv64imEvalPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        self.claim.expected_digest()
    }
}

impl Rv64imAuthoritativeSidePublicInstance {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/nightstream_statement_core_digest",
            &self.nightstream_statement_core_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/side_surface_public_digest",
            &self.side_surface_public.digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/counts",
            &[self.opened_objects.len() as u64, self.evals.len() as u64],
        );
        for opened_object in &self.opened_objects {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/opened_object_digest",
                &opened_object.digest,
            );
        }
        for eval in &self.evals {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/eval_digest",
                &eval.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imSideProofContainer {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/proof_container");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/proof_container/public_instance_digest",
            &self.public_instance.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/proof_container/side_bundle_digest",
            &self.side_bundle.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/proof_container/opening_witness_digest",
            &self.opening_witness.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/proof_container/opening_artifact_digest",
            &self.opening_artifact.digest,
        );
        tr.digest32()
    }
}

impl Rv64imAuthoritativeSideStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/nightstream_statement_core_digest",
            &self.nightstream_statement_core_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/public_instance_digest",
            &self.public_instance_digest,
        );
        tr.digest32()
    }
}

fn expected_slots_for_schema(schema: FamilyEvalSchemaId) -> &'static [u32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => &[0, 1, 2, 3],
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks
        | FamilyEvalSchemaId::Stage3Continuity => &[0],
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

fn stage_proof_binding_digest(side_bundle: &Rv64imSideProofBundle, schema: FamilyEvalSchemaId) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => side_bundle.stage1.digest,
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks => side_bundle.stage2.digest,
        FamilyEvalSchemaId::Stage3Continuity => side_bundle.stage3.digest,
    }
}

pub fn build_rv64im_authoritative_side_public_instance(
    nightstream_statement_core_digest: [u8; 32],
    side_bundle: &Rv64imSideProofBundle,
    opening_artifact: &Rv64imOpeningArtifact,
) -> Result<Rv64imAuthoritativeSidePublicInstance, SimpleKernelError> {
    let mut targets = Vec::new();
    for schema in active_phase0_schemas_from_side_bundle(side_bundle) {
        for &slot in expected_slots_for_schema(schema) {
            let target = Rv64imSideSurfaceTarget {
                schema,
                slot,
                family_binding_anchor_digest: family_binding_anchor_digest(side_bundle, schema),
                stage_proof_binding_digest: stage_proof_binding_digest(side_bundle, schema),
                digest: [0; 32],
            };
            targets.push(Rv64imSideSurfaceTarget {
                digest: target.expected_digest(),
                ..target
            });
        }
    }
    let side_surface_public = Rv64imSideSurfacePublic {
        digest: Rv64imSideSurfacePublic {
            targets,
            digest: [0; 32],
        }
        .expected_digest(),
        targets: {
            let mut rebuilt = Vec::new();
            for schema in active_phase0_schemas_from_side_bundle(side_bundle) {
                for &slot in expected_slots_for_schema(schema) {
                    let target = Rv64imSideSurfaceTarget {
                        schema,
                        slot,
                        family_binding_anchor_digest: family_binding_anchor_digest(side_bundle, schema),
                        stage_proof_binding_digest: stage_proof_binding_digest(side_bundle, schema),
                        digest: [0; 32],
                    };
                    rebuilt.push(Rv64imSideSurfaceTarget {
                        digest: target.expected_digest(),
                        ..target
                    });
                }
            }
            rebuilt
        },
    };

    let mut opened_objects = Vec::new();
    for target in &opening_artifact
        .phase0_artifact
        .phase0_opening_targets
        .targets
    {
        let representative = opening_artifact
            .phase0_artifact
            .eval_claim_bundle
            .claims
            .iter()
            .find(|claim| claim.payload.schema == target.schema)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM authoritative side instance is missing an eval claim for {:?}",
                    target.schema
                ))
            })?;
        let opened_object = Rv64imOpenedObjectPublic {
            schema: target.schema,
            opened_object: target.opened_commitment.opened_object.clone(),
            commitment_context: representative.commitment_context,
            digest: [0; 32],
        };
        opened_objects.push(Rv64imOpenedObjectPublic {
            digest: opened_object.expected_digest(),
            ..opened_object
        });
    }

    let mut evals = Vec::new();
    for claim in &opening_artifact.phase0_artifact.eval_claim_bundle.claims {
        let eval = Rv64imEvalPublic {
            claim: claim.clone(),
            digest: claim.expected_digest(),
        };
        evals.push(eval);
    }

    let instance = Rv64imAuthoritativeSidePublicInstance {
        nightstream_statement_core_digest,
        side_surface_public,
        opened_objects,
        evals,
        digest: [0; 32],
    };
    Ok(Rv64imAuthoritativeSidePublicInstance {
        digest: instance.expected_digest(),
        ..instance
    })
}

pub fn build_rv64im_side_proof_container_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProofContainer, SimpleKernelError> {
    let opening_witness = build_rv64im_side_selected_opening_witness_from_accepted_artifact(accepted_artifact);
    let opening_artifact =
        build_rv64im_opening_artifact_from_accepted_artifact(public_statement, side_bundle, accepted_artifact)?;
    build_rv64im_side_proof_container(nightstream_statement, side_bundle, opening_witness, opening_artifact)
}

pub fn build_rv64im_side_proof_container(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    opening_witness: Rv64imSideSelectedOpeningWitness,
    opening_artifact: Rv64imOpeningArtifact,
) -> Result<Rv64imSideProofContainer, SimpleKernelError> {
    let public_instance = build_rv64im_authoritative_side_public_instance(
        nightstream_statement.core_digest(),
        side_bundle,
        &opening_artifact,
    )?;
    let container = Rv64imSideProofContainer {
        public_instance,
        side_bundle: side_bundle.clone(),
        opening_witness,
        opening_artifact,
        digest: [0; 32],
    };
    Ok(Rv64imSideProofContainer {
        digest: container.expected_digest(),
        ..container
    })
}

pub fn build_rv64im_authoritative_side_statement(
    nightstream_statement: &NightstreamStatement,
    public_instance: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<Rv64imAuthoritativeSideStatement, SimpleKernelError> {
    if public_instance.nightstream_statement_core_digest != nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side statement core digest does not match the carried Nightstream statement".into(),
        ));
    }
    Ok(Rv64imAuthoritativeSideStatement {
        nightstream_statement_core_digest: nightstream_statement.core_digest(),
        public_instance_digest: public_instance.digest,
    })
}

pub fn verify_rv64im_side_proof_container(
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    container: &Rv64imSideProofContainer,
) -> Result<(), SimpleKernelError> {
    if container.digest != container.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side proof container digest mismatch".into(),
        ));
    }
    if container.public_instance.digest != container.public_instance.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side public-instance digest mismatch".into(),
        ));
    }
    if container.public_instance.nightstream_statement_core_digest != nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side public instance does not match the carried Nightstream statement core".into(),
        ));
    }
    validate_rv64im_side_proof_bundle_structure(&container.side_bundle)?;
    if container.side_bundle.statement_core_digest != nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side proof container side bundle does not match the carried Nightstream statement core"
                .into(),
        ));
    }
    verify_rv64im_side_selected_opening_witness_against_compact_surfaces(
        public_statement,
        &container.side_bundle,
        &container.opening_witness,
    )?;
    verify_rv64im_opening_artifact_from_side_proof_bundle(
        public_statement,
        &container.side_bundle,
        &container.opening_artifact,
    )?;
    verify_rv64im_root_execution_surface_against_compact_surfaces(
        nightstream_statement,
        &container.side_bundle,
        public_statement,
    )?;
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces(&container.side_bundle, public_statement)?;

    let expected_instance = build_rv64im_authoritative_side_public_instance(
        nightstream_statement.core_digest(),
        &container.side_bundle,
        &container.opening_artifact,
    )?;
    if expected_instance != container.public_instance {
        return Err(SimpleKernelError::Bridge(
            "RV64IM authoritative side public instance does not match the carried side proof material".into(),
        ));
    }
    Ok(())
}
