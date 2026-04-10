//! Owns the compact authoritative Phase 0 binding surface carried into standalone convergence checks.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::opening_eval_claims::FamilyEvalSchemaId;
use super::proof_accepted::Rv64imAcceptedProofArtifact;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0BindingTarget {
    pub schema: FamilyEvalSchemaId,
    pub family_binding_anchor_digest: [u8; 32],
    pub stage_proof_binding_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Rv64imPhase0BindingTarget {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0_binding_target");
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase0_binding_target/meta",
            &[self.schema.tag()],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0_binding_target/family_binding_anchor_digest",
            &self.family_binding_anchor_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0_binding_target/stage_proof_binding_digest",
            &self.stage_proof_binding_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPhase0BindingSurface {
    pub targets: Vec<Rv64imPhase0BindingTarget>,
    pub digest: [u8; 32],
}

impl Rv64imPhase0BindingSurface {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0_binding_surface");
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase0_binding_surface/count",
            &[self.targets.len() as u64],
        );
        for target in &self.targets {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/phase0_binding_surface/target_digest",
                &target.digest,
            );
        }
        tr.digest32()
    }
}

pub fn build_rv64im_phase0_binding_surface_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Rv64imPhase0BindingSurface {
    let targets = [
        (
            FamilyEvalSchemaId::Stage1Rows,
            artifact.stage1.address_correctness.rows_digest,
            artifact.stage1.digest,
        ),
        (
            FamilyEvalSchemaId::Stage2RegisterReads,
            artifact.stage2.linkage.register_reads_family_digest,
            artifact.stage2.digest,
        ),
        (
            FamilyEvalSchemaId::Stage2RegisterWrites,
            artifact.stage2.linkage.register_writes_family_digest,
            artifact.stage2.digest,
        ),
        (
            FamilyEvalSchemaId::Stage2RamEvents,
            artifact.stage2.linkage.ram_events_family_digest,
            artifact.stage2.digest,
        ),
        (
            FamilyEvalSchemaId::Stage2TwistLinks,
            artifact.stage2.linkage.twist_links_family_digest,
            artifact.stage2.digest,
        ),
        (
            FamilyEvalSchemaId::Stage3Continuity,
            artifact.stage3.linkage.continuity_family_digest,
            artifact.stage3.digest,
        ),
    ]
    .into_iter()
    .map(|(schema, family_binding_anchor_digest, stage_proof_binding_digest)| {
        let mut target = Rv64imPhase0BindingTarget {
            schema,
            family_binding_anchor_digest,
            stage_proof_binding_digest,
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
