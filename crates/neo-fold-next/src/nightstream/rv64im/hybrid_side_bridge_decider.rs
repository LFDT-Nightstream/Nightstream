//! Owns the current hybrid RV64IM side-bridge backend adapter.
//!
//! This does not compress the hidden theorem witness yet. It only maps the
//! fixed witness-backed side-bridge contract into the generic Spartan2
//! decider target seam so the eventual compact proof has one exact target.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::decider::spartan2::{
    prove_spartan2_backend_binding_shell, setup_spartan2_backend_binding_shell, verify_spartan2_backend_binding_shell,
    Spartan2BackendBindingShellProof, Spartan2BackendBindingShellProverKey, Spartan2BackendBindingShellVerifierKey,
    Spartan2DeciderTarget,
};
use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError};

use super::hybrid_side_bridge_contract::{Rv64imHybridSideBridgeContract, Rv64imHybridSideBridgeDeciderRelation};
use super::witness_backed_side_bridge::Rv64imWitnessBackedSideBridgeArtifact;
use super::Rv64imNightstreamProof;

#[derive(Clone)]
struct CachedBackendBindingShellKeys {
    // Keep the live keys so the verifier key retains its internal digest cache
    // across Nightstream build -> verify within one process.
    pk: Arc<Spartan2BackendBindingShellProverKey>,
    vk: Arc<Spartan2BackendBindingShellVerifierKey>,
}

static HYBRID_SIDE_BRIDGE_BACKEND_BINDING_SHELL_KEYS: OnceLock<
    Mutex<HashMap<[u8; 32], CachedBackendBindingShellKeys>>,
> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imHybridSideBridgeBackendProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imHybridSideBridgeBackendProof {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_backend_proof");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_backend_proof/version",
            b"v1",
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_backend_proof/snark_bytes_len",
            &[self.snark_data.len() as u64],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_backend_proof/snark_bytes",
            &self.snark_data,
        );
        tr.digest32()
    }

    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }

    fn as_shell_proof(&self) -> Spartan2BackendBindingShellProof {
        Spartan2BackendBindingShellProof {
            snark_data: self.snark_data.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imHybridSideBridgeArtifact {
    pub bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    pub backend_proof: Rv64imHybridSideBridgeBackendProof,
    pub digest: [u8; 32],
}

impl Rv64imHybridSideBridgeArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/bridge_artifact_digest",
            &self.bridge_artifact.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/hybrid_side_bridge_artifact/backend_proof_digest",
            &self.backend_proof.digest(),
        );
        tr.digest32()
    }
}

fn rv64im_hybrid_side_bridge_cached_backend_binding_shell_keys(
    shape: &crate::decider::spartan2::Spartan2DeciderShape,
) -> Result<
    (
        Arc<Spartan2BackendBindingShellProverKey>,
        Arc<Spartan2BackendBindingShellVerifierKey>,
    ),
    SimpleKernelError,
> {
    let shape_digest = shape.digest();
    let cache = HYBRID_SIDE_BRIDGE_BACKEND_BINDING_SHELL_KEYS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .map_err(|_| {
            SimpleKernelError::Bridge("RV64IM hybrid side-bridge backend shell key cache lock poisoned".into())
        })?
        .get(&shape_digest)
        .cloned()
    {
        return Ok((cached.pk, cached.vk));
    }

    let (pk, vk) =
        setup_spartan2_backend_binding_shell(shape).map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let cached = CachedBackendBindingShellKeys {
        pk: Arc::new(pk),
        vk: Arc::new(vk),
    };
    let (pk, vk) = (Arc::clone(&cached.pk), Arc::clone(&cached.vk));
    cache
        .lock()
        .map_err(|_| {
            SimpleKernelError::Bridge("RV64IM hybrid side-bridge backend shell key cache lock poisoned".into())
        })?
        .entry(shape_digest)
        .or_insert(cached);
    Ok((pk, vk))
}

pub(super) fn prove_rv64im_hybrid_side_bridge_backend_proof_from_decider_relation(
    relation: &Rv64imHybridSideBridgeDeciderRelation,
) -> Result<Rv64imHybridSideBridgeBackendProof, SimpleKernelError> {
    let shape = relation.backend_shape();
    let backend_relation = relation.backend_relation();
    let (pk, _) = rv64im_hybrid_side_bridge_cached_backend_binding_shell_keys(&shape)?;
    let shell = prove_spartan2_backend_binding_shell(&pk, &backend_relation)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    Ok(Rv64imHybridSideBridgeBackendProof {
        snark_data: shell.snark_data,
    })
}

pub(super) fn prewarm_rv64im_hybrid_side_bridge_backend_shell_cache_for_relation(
    relation: &Rv64imHybridSideBridgeDeciderRelation,
) -> Result<(), SimpleKernelError> {
    let shape = relation.backend_shape();
    let _ = rv64im_hybrid_side_bridge_cached_backend_binding_shell_keys(&shape)?;
    Ok(())
}

fn rv64im_hybrid_side_bridge_artifact_from_parts(
    bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    backend_proof: Rv64imHybridSideBridgeBackendProof,
) -> Rv64imHybridSideBridgeArtifact {
    let mut artifact = Rv64imHybridSideBridgeArtifact {
        bridge_artifact,
        backend_proof,
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
    let (bridge_artifact, contract) = Rv64imHybridSideBridgeContract::from_accepted_artifact(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        side_bundle,
        accepted_artifact,
    )?;
    Ok((bridge_artifact, contract.into_relation()))
}

pub(super) fn assemble_rv64im_hybrid_side_bridge_artifact(
    bridge_artifact: Rv64imWitnessBackedSideBridgeArtifact,
    backend_proof: Rv64imHybridSideBridgeBackendProof,
) -> Rv64imHybridSideBridgeArtifact {
    rv64im_hybrid_side_bridge_artifact_from_parts(bridge_artifact, backend_proof)
}

pub fn verify_rv64im_hybrid_side_bridge_artifact(
    nightstream_statement: &NightstreamStatement,
    bridge_handoff_digests: &[[u8; 32]],
    public_statement: &Rv64imProofStatement,
    artifact: &Rv64imHybridSideBridgeArtifact,
) -> Result<(), SimpleKernelError> {
    let contract = Rv64imHybridSideBridgeContract::from_bridge_artifact(
        nightstream_statement,
        bridge_handoff_digests,
        public_statement,
        &artifact.bridge_artifact,
    )?;
    let relation = contract.relation();
    let shape = relation.backend_shape();
    let backend_relation = relation.backend_relation();
    let (_, vk) = rv64im_hybrid_side_bridge_cached_backend_binding_shell_keys(&shape)?;
    verify_spartan2_backend_binding_shell(&vk, &backend_relation, &artifact.backend_proof.as_shell_proof())
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge artifact digest mismatch".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_hybrid_side_bridge_public_target(
    nightstream_statement: &NightstreamStatement,
    nightstream_proof: &Rv64imNightstreamProof,
    public_statement: &Rv64imProofStatement,
) -> Result<Spartan2DeciderTarget, SimpleKernelError> {
    let contract = Rv64imHybridSideBridgeContract::from_bridge_artifact(
        nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        public_statement,
        &nightstream_proof
            .hybrid_side_bridge_artifact
            .bridge_artifact,
    )?;
    Ok(contract.target())
}
