//! Owns the fixed witness-backed RV64IM side-bridge theorem boundary.
//!
//! The statement binds only the published public statements plus the canonical
//! side/opening/handoff handles needed by the bridge theorem. The witness owns
//! the currently shipped side bundle, opening artifact, and the compact
//! claim/opening witness material privately so later compiler layers can
//! target this one relation.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::nightstream::NightstreamStatement;
use crate::rv64im::{Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError};

use super::opening_artifact::{
    build_rv64im_opening_artifact_from_accepted_artifact, verify_rv64im_opening_artifact_against_compact_surfaces,
};
use super::side_claim_relation::{
    build_rv64im_side_claim_relation_witness_from_accepted_artifact,
    verify_rv64im_side_claim_witness_against_compact_surfaces, Rv64imSideClaimRelationWitness,
};
use super::side_opening_relation::{
    build_rv64im_side_opening_relation_witness_from_accepted_artifact,
    verify_rv64im_side_opening_witness_against_compact_surfaces, Rv64imSideOpeningRelationWitness,
};
use super::{
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces,
    verify_rv64im_root_execution_surface_against_compact_surfaces, Rv64imOpeningArtifact, Rv64imSideProofBundle,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imWitnessBackedSideBridgeStatement {
    pub nightstream_statement: NightstreamStatement,
    pub public_statement: Rv64imProofStatement,
    pub side_bundle_digest: [u8; 32],
    pub opening_artifact_digest: [u8; 32],
    pub bridge_handoff_digests: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imWitnessBackedSideBridgeWitness {
    pub side_bundle: Rv64imSideProofBundle,
    pub opening_artifact: Rv64imOpeningArtifact,
    pub claim_witness: Rv64imSideClaimRelationWitness,
    pub opening_witness: Rv64imSideOpeningRelationWitness,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imWitnessBackedSideBridgeArtifact {
    pub witness: Rv64imWitnessBackedSideBridgeWitness,
    pub digest: [u8; 32],
}

impl PartialEq for Rv64imWitnessBackedSideBridgeArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.witness.digest() == other.witness.digest()
    }
}

impl Eq for Rv64imWitnessBackedSideBridgeArtifact {}

impl Rv64imWitnessBackedSideBridgeStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/nightstream_statement_core_digest",
            &self.nightstream_statement.core_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/public_statement_digest",
            &self.public_statement.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/side_bundle_digest",
            &self.side_bundle_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/opening_artifact_digest",
            &self.opening_artifact_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/bridge_handoff_count",
            &[self.bridge_handoff_digests.len() as u64],
        );
        for digest in &self.bridge_handoff_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/bridge_handoff_digest",
                digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imWitnessBackedSideBridgeWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_witness");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_witness/side_bundle_digest",
            &self.side_bundle.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_witness/opening_artifact_digest",
            &self.opening_artifact.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_witness/claim_witness_digest",
            &self.claim_witness.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_witness/opening_witness_digest",
            &self.opening_witness.digest(),
        );
        tr.digest32()
    }
}

impl Rv64imWitnessBackedSideBridgeArtifact {
    pub fn expected_digest(&self, statement_digest: [u8; 32]) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_artifact");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_artifact/statement_digest",
            &statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_artifact/witness_digest",
            &self.witness.digest(),
        );
        tr.digest32()
    }
}

pub(super) fn build_rv64im_witness_backed_side_bridge_statement(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    side_bundle_digest: [u8; 32],
    opening_artifact_digest: [u8; 32],
) -> Result<Rv64imWitnessBackedSideBridgeStatement, SimpleKernelError> {
    if public_statement.digest != public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge public statement digest mismatch".into(),
        ));
    }
    if nightstream_statement.public_io_digest != public_statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge Nightstream public IO does not match the carried RV64IM statement"
                .into(),
        ));
    }
    if nightstream_statement.fold_schedule != public_statement.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge fold schedule does not match the carried Nightstream statement".into(),
        ));
    }
    if nightstream_statement.chunk_summaries.len() as u64 != public_statement.chunk_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge chunk count does not match the carried Nightstream statement".into(),
        ));
    }
    if bridge_handoff_digests.len() != nightstream_statement.chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge handoff count does not match the carried Nightstream chunk summaries"
                .into(),
        ));
    }
    if opening_artifact_digest == [0; 32] {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge opening artifact digest must be nonzero".into(),
        ));
    }
    Ok(Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: public_statement.clone(),
        side_bundle_digest,
        opening_artifact_digest,
        bridge_handoff_digests: bridge_handoff_digests.to_vec(),
    })
}

fn build_rv64im_witness_backed_side_bridge_witness(
    side_bundle: &Rv64imSideProofBundle,
    opening_artifact: &Rv64imOpeningArtifact,
    claim_witness: &Rv64imSideClaimRelationWitness,
    opening_witness: &Rv64imSideOpeningRelationWitness,
) -> Rv64imWitnessBackedSideBridgeWitness {
    Rv64imWitnessBackedSideBridgeWitness {
        side_bundle: side_bundle.clone(),
        opening_artifact: opening_artifact.clone(),
        claim_witness: claim_witness.clone(),
        opening_witness: opening_witness.clone(),
    }
}

fn build_rv64im_witness_backed_side_bridge_witness_from_accepted_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imWitnessBackedSideBridgeWitness, SimpleKernelError> {
    let opening_artifact =
        build_rv64im_opening_artifact_from_accepted_artifact(public_statement, side_bundle, accepted_artifact)?;
    let claim_witness = build_rv64im_side_claim_relation_witness_from_accepted_artifact(accepted_artifact);
    let opening_witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(accepted_artifact);
    Ok(build_rv64im_witness_backed_side_bridge_witness(
        side_bundle,
        &opening_artifact,
        &claim_witness,
        &opening_witness,
    ))
}

fn build_rv64im_witness_backed_side_bridge_artifact(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    witness: &Rv64imWitnessBackedSideBridgeWitness,
) -> Result<Rv64imWitnessBackedSideBridgeArtifact, SimpleKernelError> {
    verify_rv64im_witness_backed_side_bridge_relation(statement, witness)?;
    let mut artifact = Rv64imWitnessBackedSideBridgeArtifact {
        witness: witness.clone(),
        digest: [0; 32],
    };
    artifact.digest = artifact.expected_digest(statement.digest());
    Ok(artifact)
}

pub(crate) fn build_rv64im_witness_backed_side_bridge_artifact_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imWitnessBackedSideBridgeArtifact, SimpleKernelError> {
    let witness = build_rv64im_witness_backed_side_bridge_witness_from_accepted_artifact(
        public_statement,
        side_bundle,
        accepted_artifact,
    )?;
    let statement = build_rv64im_witness_backed_side_bridge_statement(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        witness.side_bundle.digest,
        witness.opening_artifact.digest,
    )?;
    build_rv64im_witness_backed_side_bridge_artifact(&statement, &witness)
}

fn validate_witness_against_statement(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    witness: &Rv64imWitnessBackedSideBridgeWitness,
) -> Result<(), SimpleKernelError> {
    if witness.side_bundle.digest != statement.side_bundle_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge side bundle digest does not match the carried public statement".into(),
        ));
    }
    if witness.opening_artifact.digest != statement.opening_artifact_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge opening artifact digest does not match the carried public statement"
                .into(),
        ));
    }
    if witness.side_bundle.statement_core_digest != statement.nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge side bundle does not match the carried Nightstream statement core"
                .into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_witness_backed_side_bridge_relation(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    witness: &Rv64imWitnessBackedSideBridgeWitness,
) -> Result<(), SimpleKernelError> {
    validate_witness_against_statement(statement, witness)?;

    verify_rv64im_opening_artifact_against_compact_surfaces(
        &statement.public_statement,
        &witness.side_bundle,
        &witness.opening_artifact,
    )?;
    verify_rv64im_side_claim_witness_against_compact_surfaces(
        &statement.public_statement,
        &witness.side_bundle,
        &witness.claim_witness,
    )?;
    verify_rv64im_side_opening_witness_against_compact_surfaces(
        &statement.public_statement,
        &witness.side_bundle,
        &witness.opening_witness,
    )?;
    verify_rv64im_root_execution_surface_against_compact_surfaces(
        &statement.nightstream_statement,
        &witness.side_bundle,
        &statement.public_statement,
    )?;
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces(
        &witness.side_bundle,
        &statement.public_statement,
    )?;
    Ok(())
}

pub(super) fn verify_rv64im_witness_backed_side_bridge_artifact(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    artifact: &Rv64imWitnessBackedSideBridgeArtifact,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_witness_backed_side_bridge_relation(statement, &artifact.witness)?;
    if artifact.digest != artifact.expected_digest(statement.digest()) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM witness-backed side bridge artifact digest mismatch".into(),
        ));
    }
    Ok(())
}
