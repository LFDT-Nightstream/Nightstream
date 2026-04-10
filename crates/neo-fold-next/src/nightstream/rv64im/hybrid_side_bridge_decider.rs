//! Owns the current hybrid RV64IM side-bridge backend adapter.
//!
//! This does not compress the hidden theorem witness yet. It only maps the
//! fixed witness-backed side-bridge contract into the generic Spartan2
//! decider target seam so the eventual compact proof has one exact target.

use neo_transcript::Transcript;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::decider::spartan2::{
    prove_spartan2_decider, setup_spartan2_decider, verify_spartan2_decider, Spartan2DeciderProof,
    Spartan2DeciderProverKey, Spartan2DeciderTarget, Spartan2DeciderVerifierKey,
};
use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError};

use super::hybrid_side_bridge_contract::{Rv64imHybridSideBridgeContract, Rv64imHybridSideBridgeDeciderRelation};
use super::witness_backed_side_bridge::{
    build_rv64im_witness_backed_side_bridge_artifact_from_accepted_artifact, Rv64imWitnessBackedSideBridgeArtifact,
};

#[derive(Clone)]
struct CachedDeciderKeys {
    // Keep the live keys so the verifier key retains its internal digest cache
    // across Nightstream build -> verify within one process.
    pk: Arc<Spartan2DeciderProverKey>,
    vk: Arc<Spartan2DeciderVerifierKey>,
}

static HYBRID_SIDE_BRIDGE_DECIDER_KEYS: OnceLock<Mutex<HashMap<[u8; 32], CachedDeciderKeys>>> = OnceLock::new();

pub type Rv64imHybridSideBridgeCompiledProof = Spartan2DeciderProof;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imHybridSideBridgeArtifact {
    pub bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    pub compiled_proof: Rv64imHybridSideBridgeCompiledProof,
    pub digest: [u8; 32],
}

impl Rv64imHybridSideBridgeArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr =
            neo_transcript::Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact");
        neo_transcript::Transcript::append_message(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/version",
            b"v1",
        );
        neo_transcript::Transcript::append_message(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/bridge_artifact_digest",
            &self.bridge_artifact.digest,
        );
        neo_transcript::Transcript::append_message(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/compiled_proof_digest",
            &self.compiled_proof.digest(),
        );
        neo_transcript::Transcript::digest32(&mut tr)
    }
}

fn rv64im_hybrid_side_bridge_cached_decider_keys(
    shape: &crate::decider::spartan2::Spartan2DeciderShape,
) -> Result<(Arc<Spartan2DeciderProverKey>, Arc<Spartan2DeciderVerifierKey>), SimpleKernelError> {
    let shape_digest = shape.digest();
    let cache = HYBRID_SIDE_BRIDGE_DECIDER_KEYS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM hybrid side-bridge decider key cache lock poisoned".into()))?
        .get(&shape_digest)
        .cloned()
    {
        return Ok((cached.pk, cached.vk));
    }

    let (pk, vk) = setup_spartan2_decider(shape).map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let cached = CachedDeciderKeys {
        pk: Arc::new(pk),
        vk: Arc::new(vk),
    };
    let (pk, vk) = (Arc::clone(&cached.pk), Arc::clone(&cached.vk));
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM hybrid side-bridge decider key cache lock poisoned".into()))?
        .entry(shape_digest)
        .or_insert(cached);
    Ok((pk, vk))
}

pub(super) fn prove_rv64im_hybrid_side_bridge_compiled_proof_from_decider_relation(
    relation: &Rv64imHybridSideBridgeDeciderRelation,
) -> Result<Rv64imHybridSideBridgeCompiledProof, SimpleKernelError> {
    let target = relation.target();
    let shape = target.shape();
    let (pk, _) = rv64im_hybrid_side_bridge_cached_decider_keys(&shape)?;
    prove_spartan2_decider(&pk, &target).map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}

pub(super) fn prewarm_rv64im_hybrid_side_bridge_decider_cache_for_relation(
    relation: &Rv64imHybridSideBridgeDeciderRelation,
) -> Result<(), SimpleKernelError> {
    let shape = relation.target().shape();
    let _ = rv64im_hybrid_side_bridge_cached_decider_keys(&shape)?;
    Ok(())
}

fn rv64im_hybrid_side_bridge_artifact_from_parts(
    bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    compiled_proof: Rv64imHybridSideBridgeCompiledProof,
) -> Rv64imHybridSideBridgeArtifact {
    let mut artifact = Rv64imHybridSideBridgeArtifact {
        bridge_artifact,
        compiled_proof,
        digest: [0; 32],
    };
    artifact.digest = artifact.expected_digest();
    artifact
}

pub(super) fn build_rv64im_hybrid_side_bridge_material_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    side_bundle: &super::Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<
    (
        Rv64imWitnessBackedSideBridgeArtifact,
        Rv64imHybridSideBridgeDeciderRelation,
    ),
    SimpleKernelError,
> {
    let bridge_artifact = build_rv64im_witness_backed_side_bridge_artifact_from_accepted_artifact(
        nightstream_statement,
        public_statement,
        side_bundle,
        accepted_artifact,
    )?;
    let contract = Rv64imHybridSideBridgeContract::from_bridge_artifact(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        &bridge_artifact,
    )?;
    Ok((bridge_artifact, contract.into_relation()))
}

pub(super) fn assemble_rv64im_hybrid_side_bridge_artifact(
    bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    compiled_proof: Rv64imHybridSideBridgeCompiledProof,
) -> Rv64imHybridSideBridgeArtifact {
    rv64im_hybrid_side_bridge_artifact_from_parts(bridge_artifact, compiled_proof)
}

pub fn build_rv64im_hybrid_side_bridge_target_from_artifact(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    artifact: &Rv64imHybridSideBridgeArtifact,
) -> Result<Spartan2DeciderTarget, SimpleKernelError> {
    let contract = Rv64imHybridSideBridgeContract::from_bridge_artifact(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        &artifact.bridge_artifact,
    )?;
    Ok(contract.target())
}

pub fn verify_rv64im_hybrid_side_bridge_artifact(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    artifact: &Rv64imHybridSideBridgeArtifact,
) -> Result<(), SimpleKernelError> {
    let target = build_rv64im_hybrid_side_bridge_target_from_artifact(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        artifact,
    )?;
    let shape = target.shape();
    let (_, vk) = rv64im_hybrid_side_bridge_cached_decider_keys(&shape)?;
    verify_spartan2_decider(&vk, &target, &artifact.compiled_proof)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge artifact digest mismatch".into(),
        ));
    }
    Ok(())
}
