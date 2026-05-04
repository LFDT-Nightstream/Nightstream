//! Owns the RV32IM side-opening public tuple and its direct theorem checker.
//!
//! This module owns:
//! - the canonical side-opening public tuple carried by Nightstream
//! - the compact binding statement consumed by the side binding proof
//! - the direct native verifier for the current side-opening proof backend
//!
//! It does not own:
//! - root-execution or kernel-export linkage checks
//! - the packaged succinct side-opening backend
//! - the outer Nightstream proof binding

use std::collections::{BTreeMap, BTreeSet};

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::finalize::digest32_as_fields;
use crate::nightstream::NightstreamStatement;
use crate::rv32im::kernel::{
    derive_phase0_point, CommitmentContextId, FamilyEvalClaim, FamilyEvalSchemaId, OpenedAjtaiObjectId,
    SimpleKernelError,
};
use crate::rv32im::{Rv32imProofStatement, Stage1VerifiedClaims, Stage2VerifiedClaims, Stage3VerifiedClaims};

use super::side_eval_claim_relation::{
    active_phase0_schemas_from_side_bundle, build_rv32im_phase0_opened_object_bundle_from_opening_targets,
    nightstream_phase0_binding_digest, rebuild_phase0_claim_witnesses_from_artifact,
    validate_rv32im_side_eval_claim_artifact_structure, Rv32imSideEvalClaimArtifact,
};
use super::side_opening_relation::{Rv32imSideStage1Summary, Rv32imSideStage2Summary, Rv32imSideStage3Summary};
use super::Rv32imSideProofBundle;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imSideSurfaceTarget {
    pub schema: FamilyEvalSchemaId,
    pub slot: u32,
    pub family_binding_anchor_digest: [u8; 32],
    pub stage_proof_binding_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imSideSurfacePublic {
    pub targets: Vec<Rv32imSideSurfaceTarget>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imOpenedObjectPublic {
    pub schema: FamilyEvalSchemaId,
    pub opened_object: OpenedAjtaiObjectId,
    pub commitment_context: CommitmentContextId,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imEvalPublic {
    pub claim: FamilyEvalClaim,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imSideOpeningPublic {
    pub opened_objects: Vec<Rv32imOpenedObjectPublic>,
    pub evals: Vec<Rv32imEvalPublic>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imSideBindingStatement {
    pub nightstream_statement_core_digest: [u8; 32],
    pub public_statement_digest: [u8; 32],
    pub public_instance_digest: [u8; 32],
}

pub type Rv32imSideOpeningProof = Rv32imSideEvalClaimArtifact;

impl Rv32imSideSurfaceTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_target");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_target/meta",
            &[self.schema.tag(), self.slot as u64],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_target/family_binding_anchor_digest",
            &self.family_binding_anchor_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_target/stage_proof_binding_digest",
            &self.stage_proof_binding_digest,
        );
        tr.digest32()
    }
}

impl Rv32imSideSurfacePublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_public");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_public/count",
            &[self.targets.len() as u64],
        );
        for target in &self.targets {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/authoritative_side/surface_public/target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv32imOpenedObjectPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/authoritative_side/opened_object");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/opened_object/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/opened_object/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/opened_object/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/opened_object/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.digest32()
    }
}

impl Rv32imEvalPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        self.claim.expected_digest()
    }
}

impl Rv32imSideOpeningPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/authoritative_side/public_instance");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/public_instance/counts",
            &[self.opened_objects.len() as u64, self.evals.len() as u64],
        );
        for opened_object in &self.opened_objects {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/authoritative_side/public_instance/opened_object_digest",
                &opened_object.digest,
            );
        }
        for eval in &self.evals {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/authoritative_side/public_instance/eval_digest",
                &eval.digest,
            );
        }
        tr.digest32()
    }
}

impl Rv32imSideBindingStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/authoritative_side/statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/statement/version",
            b"v2",
        );
        tr.append_fields(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/statement/nightstream_statement_core_digest",
            &digest32_as_fields(self.nightstream_statement_core_digest),
        );
        tr.append_fields(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/statement/public_statement_digest",
            &digest32_as_fields(self.public_statement_digest),
        );
        tr.append_fields(
            b"neo.fold.next/nightstream/rv32im/authoritative_side/statement/public_instance_digest",
            &digest32_as_fields(self.public_instance_digest),
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

fn active_phase0_schemas_from_verified_claims(
    stage1: &Stage1VerifiedClaims,
    stage2: &Stage2VerifiedClaims,
    stage3: &Stage3VerifiedClaims,
) -> Vec<FamilyEvalSchemaId> {
    let mut schemas = vec![FamilyEvalSchemaId::Stage1Rows];
    if stage2.claim.register_read_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterReads);
    }
    if stage2.claim.register_write_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterWrites);
    }
    if stage2.claim.ram_event_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RamEvents);
    }
    if stage2.claim.twist_link_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2TwistLinks);
    }
    if stage3.claim.continuity_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage3Continuity);
    }
    let _ = stage1;
    schemas
}

fn family_binding_anchor_digest(
    stage1: &Stage1VerifiedClaims,
    stage2: &Stage2VerifiedClaims,
    stage3: &Stage3VerifiedClaims,
    schema: FamilyEvalSchemaId,
) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => stage1.rows_digest,
        FamilyEvalSchemaId::Stage2RegisterReads => stage2.claim.register_reads_family_digest,
        FamilyEvalSchemaId::Stage2RegisterWrites => stage2.claim.register_writes_family_digest,
        FamilyEvalSchemaId::Stage2RamEvents => stage2.claim.ram_events_family_digest,
        FamilyEvalSchemaId::Stage2TwistLinks => stage2.claim.twist_links_family_digest,
        FamilyEvalSchemaId::Stage3Continuity => stage3.claim.continuity_family_digest,
    }
}

fn stage_proof_binding_digest(
    stage1: &Stage1VerifiedClaims,
    stage2: &Stage2VerifiedClaims,
    stage3: &Stage3VerifiedClaims,
    schema: FamilyEvalSchemaId,
) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => stage1.digest,
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks => stage2.digest,
        FamilyEvalSchemaId::Stage3Continuity => stage3.digest,
    }
}

pub(super) fn build_rv32im_side_surface_public_from_verified_claims(
    stage1: &Stage1VerifiedClaims,
    stage2: &Stage2VerifiedClaims,
    stage3: &Stage3VerifiedClaims,
) -> Rv32imSideSurfacePublic {
    let mut targets = Vec::new();
    for schema in active_phase0_schemas_from_verified_claims(stage1, stage2, stage3) {
        for &slot in expected_slots_for_schema(schema) {
            let target = Rv32imSideSurfaceTarget {
                schema,
                slot,
                family_binding_anchor_digest: family_binding_anchor_digest(stage1, stage2, stage3, schema),
                stage_proof_binding_digest: stage_proof_binding_digest(stage1, stage2, stage3, schema),
                digest: [0; 32],
            };
            targets.push(Rv32imSideSurfaceTarget {
                digest: target.expected_digest(),
                ..target
            });
        }
    }
    let public = Rv32imSideSurfacePublic {
        targets,
        digest: [0; 32],
    };
    Rv32imSideSurfacePublic {
        digest: public.expected_digest(),
        ..public
    }
}

fn active_phase0_schemas_from_opening_summaries(
    stage1: &Rv32imSideStage1Summary,
    stage2: &Rv32imSideStage2Summary,
    stage3: &Rv32imSideStage3Summary,
) -> Vec<FamilyEvalSchemaId> {
    let mut schemas = vec![FamilyEvalSchemaId::Stage1Rows];
    if stage2.claim.register_read_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterReads);
    }
    if stage2.claim.register_write_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RegisterWrites);
    }
    if stage2.claim.ram_event_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2RamEvents);
    }
    if stage2.claim.twist_link_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage2TwistLinks);
    }
    if stage3.claim.continuity_count != 0 {
        schemas.push(FamilyEvalSchemaId::Stage3Continuity);
    }
    let _ = stage1;
    schemas
}

fn family_binding_anchor_digest_from_opening_summaries(
    stage1: &Rv32imSideStage1Summary,
    stage2: &Rv32imSideStage2Summary,
    stage3: &Rv32imSideStage3Summary,
    schema: FamilyEvalSchemaId,
) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => stage1.rows_digest,
        FamilyEvalSchemaId::Stage2RegisterReads => stage2.claim.register_reads_family_digest,
        FamilyEvalSchemaId::Stage2RegisterWrites => stage2.claim.register_writes_family_digest,
        FamilyEvalSchemaId::Stage2RamEvents => stage2.claim.ram_events_family_digest,
        FamilyEvalSchemaId::Stage2TwistLinks => stage2.claim.twist_links_family_digest,
        FamilyEvalSchemaId::Stage3Continuity => stage3.claim.continuity_family_digest,
    }
}

fn stage_proof_binding_digest_from_opening_summaries(
    stage1: &Rv32imSideStage1Summary,
    stage2: &Rv32imSideStage2Summary,
    stage3: &Rv32imSideStage3Summary,
    schema: FamilyEvalSchemaId,
) -> [u8; 32] {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => stage1.digest,
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks => stage2.digest,
        FamilyEvalSchemaId::Stage3Continuity => stage3.digest,
    }
}

pub(super) fn build_rv32im_side_surface_public_from_opening_summaries(
    stage1: &Rv32imSideStage1Summary,
    stage2: &Rv32imSideStage2Summary,
    stage3: &Rv32imSideStage3Summary,
) -> Rv32imSideSurfacePublic {
    let mut targets = Vec::new();
    for schema in active_phase0_schemas_from_opening_summaries(stage1, stage2, stage3) {
        for &slot in expected_slots_for_schema(schema) {
            let target = Rv32imSideSurfaceTarget {
                schema,
                slot,
                family_binding_anchor_digest: family_binding_anchor_digest_from_opening_summaries(
                    stage1, stage2, stage3, schema,
                ),
                stage_proof_binding_digest: stage_proof_binding_digest_from_opening_summaries(
                    stage1, stage2, stage3, schema,
                ),
                digest: [0; 32],
            };
            targets.push(Rv32imSideSurfaceTarget {
                digest: target.expected_digest(),
                ..target
            });
        }
    }
    let public = Rv32imSideSurfacePublic {
        targets,
        digest: [0; 32],
    };
    Rv32imSideSurfacePublic {
        digest: public.expected_digest(),
        ..public
    }
}

pub fn build_rv32im_side_surface_public(
    side_bundle: &Rv32imSideProofBundle,
) -> Result<Rv32imSideSurfacePublic, SimpleKernelError> {
    let _ = active_phase0_schemas_from_side_bundle(side_bundle);
    Ok(build_rv32im_side_surface_public_from_verified_claims(
        &side_bundle.stage1,
        &side_bundle.stage2,
        &side_bundle.stage3,
    ))
}

fn build_rv32im_opened_object_publics(
    opening: &Rv32imSideOpeningProof,
) -> Result<Vec<Rv32imOpenedObjectPublic>, SimpleKernelError> {
    let phase0_opened_objects = build_rv32im_phase0_opened_object_bundle_from_opening_targets(
        &opening.eval_claim_bundle,
        &opening.phase0_opening_targets,
    )?;
    Ok(phase0_opened_objects
        .objects
        .into_iter()
        .map(|summary| {
            let public = Rv32imOpenedObjectPublic {
                schema: summary.schema,
                opened_object: summary.opened_object,
                commitment_context: summary.commitment_context,
                digest: [0; 32],
            };
            Rv32imOpenedObjectPublic {
                digest: public.expected_digest(),
                ..public
            }
        })
        .collect())
}

fn build_rv32im_eval_publics(opening: &Rv32imSideOpeningProof) -> Vec<Rv32imEvalPublic> {
    opening
        .eval_claim_bundle
        .claims
        .iter()
        .cloned()
        .map(|claim| {
            let digest = claim.expected_digest();
            Rv32imEvalPublic { claim, digest }
        })
        .collect()
}

pub fn build_rv32im_side_opening_public(
    _side_bundle: &Rv32imSideProofBundle,
    opening: &Rv32imSideOpeningProof,
) -> Result<Rv32imSideOpeningPublic, SimpleKernelError> {
    let public = Rv32imSideOpeningPublic {
        opened_objects: build_rv32im_opened_object_publics(opening)?,
        evals: build_rv32im_eval_publics(opening),
        digest: [0; 32],
    };
    Ok(Rv32imSideOpeningPublic {
        digest: public.expected_digest(),
        ..public
    })
}

pub fn build_rv32im_side_binding_statement(
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv32imProofStatement,
    public: &Rv32imSideOpeningPublic,
) -> Result<Rv32imSideBindingStatement, SimpleKernelError> {
    validate_rv32im_side_opening_public(nightstream_statement, public)?;
    let public_statement_digest = public_statement.recompute_digest();
    if public_statement.digest != public_statement_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried public statement digest does not match the public statement fields".into(),
        ));
    }
    Ok(Rv32imSideBindingStatement {
        nightstream_statement_core_digest: nightstream_statement.core_digest(),
        public_statement_digest,
        public_instance_digest: public.digest,
    })
}

fn validate_opened_objects_public(opened_objects: &[Rv32imOpenedObjectPublic]) -> Result<(), SimpleKernelError> {
    let mut seen = BTreeSet::new();
    let mut previous = None;
    for opened_object in opened_objects {
        if opened_object.digest != opened_object.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?} opened-object digest mismatch",
                opened_object.schema
            )));
        }
        if !seen.insert(opened_object.schema) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public carries duplicate opened objects for {:?}",
                opened_object.schema
            )));
        }
        if let Some(previous_schema) = previous {
            if previous_schema >= opened_object.schema {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM side-opening public opened objects are not in strict canonical schema order".into(),
                ));
            }
        }
        previous = Some(opened_object.schema);
    }
    Ok(())
}

fn validate_evals_public(evals: &[Rv32imEvalPublic]) -> Result<(), SimpleKernelError> {
    let mut seen = BTreeSet::new();
    let mut previous = None;
    for eval in evals {
        eval.claim.validate().map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?}/{} eval claim is internally inconsistent: {err}",
                eval.claim.payload.schema, eval.claim.id.slot
            ))
        })?;
        if eval.digest != eval.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?}/{} eval digest mismatch",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        }
        let key = (eval.claim.payload.schema, eval.claim.id.slot);
        if !seen.insert(key) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public carries duplicate evals for {:?}/{}",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        }
        if let Some(previous_key) = previous {
            if previous_key >= key {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM side-opening public evals are not in strict canonical order".into(),
                ));
            }
        }
        previous = Some(key);
    }
    Ok(())
}

pub(super) fn verify_phase0_public_claims_against_surface(
    nightstream_statement_core_digest: [u8; 32],
    public: &Rv32imSideOpeningPublic,
    surface: &Rv32imSideSurfacePublic,
) -> Result<(), SimpleKernelError> {
    let object_by_schema = public
        .opened_objects
        .iter()
        .map(|opened_object| (opened_object.schema, opened_object))
        .collect::<BTreeMap<_, _>>();
    let target_by_key = surface
        .targets
        .iter()
        .map(|target| ((target.schema, target.slot), target))
        .collect::<BTreeMap<_, _>>();

    let mut covered = BTreeSet::new();
    for eval in &public.evals {
        let claim = &eval.claim;
        let schema = claim.payload.schema;
        let slot = claim.id.slot;

        let Some(object) = object_by_schema.get(&schema) else {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public is missing the opened object for {:?}",
                schema
            )));
        };
        if claim.opened_object != object.opened_object || claim.commitment_context != object.commitment_context {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?} opened-object summary does not match the carried eval claim",
                schema
            )));
        }

        let Some(target) = target_by_key.get(&(schema, slot)) else {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public is missing the target for {:?}/{}",
                schema, slot
            )));
        };

        let expected_binding_digest = nightstream_phase0_binding_digest(
            nightstream_statement_core_digest,
            &claim.opened_object,
            schema,
            slot,
            target.family_binding_anchor_digest,
            target.stage_proof_binding_digest,
        );
        if claim.binding_digest != expected_binding_digest {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?}/{} binding digest does not match the carried side surface",
                schema, slot
            )));
        }

        let expected_point = derive_phase0_point(
            &claim.opened_object,
            &claim.commitment_context,
            schema,
            slot,
            claim.binding_digest,
        );
        if claim.point != expected_point {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM side-opening public {:?}/{} point does not match the verifier-derived Phase 0 point",
                schema, slot
            )));
        }

        covered.insert((schema, slot));
    }

    let expected_cover = surface
        .targets
        .iter()
        .map(|target| (target.schema, target.slot))
        .collect::<BTreeSet<_>>();
    if covered != expected_cover {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side-opening public does not provide exact target cover".into(),
        ));
    }
    Ok(())
}

pub fn validate_rv32im_side_opening_public(
    nightstream_statement: &NightstreamStatement,
    public: &Rv32imSideOpeningPublic,
) -> Result<(), SimpleKernelError> {
    validate_opened_objects_public(&public.opened_objects)?;
    validate_evals_public(&public.evals)?;
    if public.digest != public.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side-opening public digest mismatch".into(),
        ));
    }
    let _ = nightstream_statement;
    Ok(())
}

pub fn verify_rv32im_side_surface_public_against_bundle(
    public: &Rv32imSideOpeningPublic,
    side_bundle: &Rv32imSideProofBundle,
) -> Result<(), SimpleKernelError> {
    let expected_surface = build_rv32im_side_surface_public(side_bundle)?;
    verify_phase0_public_claims_against_surface(side_bundle.statement_core_digest, public, &expected_surface)
}

pub fn verify_rv32im_side_opening_native(
    nightstream_statement: &NightstreamStatement,
    public: &Rv32imSideOpeningPublic,
    opening: &Rv32imSideOpeningProof,
) -> Result<(), SimpleKernelError> {
    validate_rv32im_side_opening_public(nightstream_statement, public)?;

    validate_rv32im_side_eval_claim_artifact_structure(opening)?;
    let claim_witnesses = rebuild_phase0_claim_witnesses_from_artifact(opening)?;

    let expected_opened_objects = build_rv32im_opened_object_publics(opening)?;
    if public.opened_objects != expected_opened_objects {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side-opening public opened objects do not match the carried opening proof".into(),
        ));
    }

    let expected_evals = build_rv32im_eval_publics(opening);
    if public.evals != expected_evals {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side-opening public evals do not match the carried opening proof".into(),
        ));
    }

    if claim_witnesses.len() != public.evals.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side-opening proof rebuilt claim-witness count does not match the carried public eval set".into(),
        ));
    }
    Ok(())
}
