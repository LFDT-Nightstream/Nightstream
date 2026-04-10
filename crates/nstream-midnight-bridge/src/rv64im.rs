//! Owns the RV64IM Nightstream bridge split: fixed public digest and bridge-private witness.

#[path = "rv64im_contract.rs"]
mod contract;
#[path = "rv64im_contract_submit.rs"]
mod contract_submit;
#[path = "rv64im_payload.rs"]
mod payload;
#[path = "rv64im_proof_server.rs"]
mod proof_server;

pub use contract::*;
pub use contract_submit::*;
pub use proof_server::*;

use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_main_residual_proof, rv64im_nightstream_linkage_root, rv64im_verifier_context_digest,
    verify_rv64im_linkage_artifact, verify_rv64im_main_decider_proof, verify_rv64im_main_residual_proof,
    verify_rv64im_opening_artifact_from_side_proof_bundle, verify_rv64im_side_proof_artifact_from_accepted_artifact,
    verify_rv64im_side_terminal_proof_artifact, Rv64imMainResidualProof, Rv64imNightstreamProof,
};
use neo_fold_next::nightstream::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{build_rv64im_accepted_proof_artifact, Rv64imProof, SimpleKernelError};
use payload::{decode_rv64im_nightstream_proof_fields, encode_rv64im_nightstream_proof_fields};
use std::borrow::Cow;
use std::sync::Arc;
use thiserror::Error;
use transient_crypto::curve::Fr;
use transient_crypto::proofs::{KeyLocation, ProofPreimage};
use zkir::{Instruction, IrSource};

pub type BridgeFieldWord = u64;

pub const RV64IM_NIGHTSTREAM_BRIDGE_VERSION: u32 = 1;
pub const RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION: &str = "nstream-midnight-bridge/rv64im/nightstream/v1";

const BYTES_PER_FIELD_WORD: usize = 7;
const DIGEST32_FIELD_WORDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgePublicInputs {
    pub version: u32,
    pub statement_digest: [u8; 32],
}

impl Rv64imNightstreamBridgePublicInputs {
    pub fn new(statement: &NightstreamStatement) -> Self {
        Self {
            version: RV64IM_NIGHTSTREAM_BRIDGE_VERSION,
            statement_digest: statement.digest(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rv64imNightstreamBridgePrivateWitness<'a> {
    pub statement: &'a NightstreamStatement,
    pub proof: &'a Rv64imNightstreamProof,
    pub proof_complete_transport: &'a Rv64imProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgePreimage {
    pub inputs: Vec<BridgeFieldWord>,
    pub private_transcript: Vec<BridgeFieldWord>,
    pub public_transcript_inputs: Vec<BridgeFieldWord>,
    pub public_transcript_outputs: Vec<BridgeFieldWord>,
    pub binding_input: BridgeFieldWord,
    pub communications_commitment: Option<(BridgeFieldWord, BridgeFieldWord)>,
    pub key_location: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgeLinkageClaims {
    pub kernel_export_anchor_digest: [u8; 32],
    pub linkage_root: [u8; 32],
    pub public_chunk_digests: Vec<[u8; 32]>,
    pub bridge_handoff_digests: Vec<[u8; 32]>,
    pub linkage_claims_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgePrivateClaims {
    pub statement_digest_hint: [u8; 32],
    pub verifier_context_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub proof_binding: Rv64imNightstreamBridgeProofBindingClaims,
    pub linkage: Rv64imNightstreamBridgeLinkageClaims,
    pub chunk_transitions: Vec<Rv64imNightstreamBridgeChunkTransitionClaim>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgeProofBindingClaims {
    pub proof_binding_root: [u8; 32],
    pub main_decider_target_digest: [u8; 32],
    pub main_residual_public_statement_digest: [u8; 32],
    pub main_residual_folded_statement_digest: [u8; 32],
    pub main_residual_final_proof_digest: [u8; 32],
    pub main_residual_kernel_export_proof_digest: [u8; 32],
    pub side_terminal_artifact_digest: [u8; 32],
    pub side_proof_artifact_digest: [u8; 32],
    pub opening_artifact_digest: [u8; 32],
    pub linkage_artifact_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgeChunkTransitionClaim {
    pub chunk_relation_digest: [u8; 32],
    pub transition_witness_digest: [u8; 32],
}

impl<'a> Rv64imNightstreamBridgePrivateWitness<'a> {
    pub fn new(
        statement: &'a NightstreamStatement,
        proof: &'a Rv64imNightstreamProof,
        proof_complete_transport: &'a Rv64imProof,
    ) -> Self {
        Self {
            statement,
            proof,
            proof_complete_transport,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OwnedRv64imNightstreamBridgePrivateWitness {
    pub statement: NightstreamStatement,
    pub proof: Rv64imNightstreamProof,
    pub proof_complete_transport: Rv64imProof,
}

impl OwnedRv64imNightstreamBridgePrivateWitness {
    pub fn borrowed(&self) -> Rv64imNightstreamBridgePrivateWitness<'_> {
        Rv64imNightstreamBridgePrivateWitness {
            statement: &self.statement,
            proof: &self.proof,
            proof_complete_transport: &self.proof_complete_transport,
        }
    }
}

#[derive(Clone, Debug)]
struct OwnedRv64imNightstreamBridgePrivatePayload {
    claims: Rv64imNightstreamBridgePrivateClaims,
    witness: OwnedRv64imNightstreamBridgePrivateWitness,
}

#[derive(Debug, Error)]
pub enum Rv64imBridgeError {
    #[error("unsupported RV64IM Nightstream bridge version: expected {expected}, got {actual}")]
    UnsupportedVersion { expected: u32, actual: u32 },
    #[error("RV64IM Nightstream bridge statement digest mismatch: expected {expected:?}, got {actual:?}")]
    StatementDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("truncated RV64IM Nightstream bridge encoding while reading {0}")]
    Truncated(&'static str),
    #[error("invalid RV64IM Nightstream bridge encoding: {0}")]
    InvalidEncoding(String),
    #[error("RV64IM Nightstream bridge witness encode failed: {0}")]
    WitnessEncode(String),
    #[error("RV64IM Nightstream bridge witness decode failed: {0}")]
    WitnessDecode(String),
    #[error("RV64IM Nightstream proof-server request encode failed: {0}")]
    RequestEncode(String),
    #[error("RV64IM Nightstream proof-server response decode failed: {0}")]
    ResponseDecode(String),
    #[error("RV64IM Nightstream bridge artifact encode failed: {0}")]
    ArtifactEncode(String),
    #[error("RV64IM Nightstream bridge artifact decode failed: {0}")]
    ArtifactDecode(String),
    #[error("RV64IM Nightstream proof-server transport failed: {0}")]
    Transport(String),
    #[error("RV64IM Nightstream bridge verification failed: {0}")]
    Nightstream(#[from] SimpleKernelError),
    #[error("RV64IM Nightstream bridge private claims mismatch: {0}")]
    PrivateClaims(String),
}

pub fn verify_rv64im_nightstream_bridge_input(
    public_inputs: Rv64imNightstreamBridgePublicInputs,
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<(), Rv64imBridgeError> {
    if public_inputs.version != RV64IM_NIGHTSTREAM_BRIDGE_VERSION {
        return Err(Rv64imBridgeError::UnsupportedVersion {
            expected: RV64IM_NIGHTSTREAM_BRIDGE_VERSION,
            actual: public_inputs.version,
        });
    }
    let actual_statement_digest = private_witness.statement.digest();
    if actual_statement_digest != public_inputs.statement_digest {
        return Err(Rv64imBridgeError::StatementDigestMismatch {
            expected: public_inputs.statement_digest,
            actual: actual_statement_digest,
        });
    }
    let expected_context_digest = rv64im_verifier_context_digest(
        private_witness
            .proof_complete_transport
            .statement
            .root_params_id,
    );
    if private_witness.statement.verifier_context_digest != expected_context_digest {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream statement verifier-context digest does not match the legacy public-proof root params"
                .into(),
        )));
    }
    let artifact = build_rv64im_accepted_proof_artifact(private_witness.proof_complete_transport)?;
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&artifact)?;
    verify_rv64im_main_residual_proof(
        &final_statement,
        &final_proof,
        &private_witness.proof.main_residual_proof,
    )?;
    verify_rv64im_main_decider_proof(
        &final_statement,
        &final_proof,
        &private_witness.proof.main_decider_proof,
    )?;
    verify_rv64im_linkage_artifact(&final_statement, &final_proof, &private_witness.proof.linkage_artifact)?;
    verify_rv64im_side_proof_artifact_from_accepted_artifact(
        &artifact,
        private_witness.statement.core_digest(),
        &private_witness.proof.side_proof_artifact,
    )?;
    let side_bundle = private_witness.proof.side_proof_artifact.bundle.clone();
    verify_rv64im_opening_artifact_from_side_proof_bundle(
        &private_witness.proof_complete_transport.statement,
        &side_bundle,
        &private_witness.proof.opening_artifact,
    )?;
    verify_rv64im_side_terminal_proof_artifact(
        private_witness.statement,
        &private_witness
            .proof
            .main_residual_proof
            .bridge_handoff_digests,
        &private_witness.proof_complete_transport.statement,
        &side_bundle,
        &private_witness.proof.opening_artifact,
        &private_witness.proof.side_terminal_artifact,
    )?;
    let private_claims = build_rv64im_nightstream_bridge_private_claims(private_witness)?;
    verify_rv64im_nightstream_bridge_private_claims(&private_claims, private_witness)?;
    Ok(())
}

fn build_rv64im_nightstream_bridge_private_claims(
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<Rv64imNightstreamBridgePrivateClaims, Rv64imBridgeError> {
    let artifact = build_rv64im_accepted_proof_artifact(private_witness.proof_complete_transport)?;
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&artifact)?;
    let main_residual = build_rv64im_main_residual_proof(&final_statement, &final_proof)?;
    let linkage_claims =
        verify_rv64im_linkage_artifact(&final_statement, &final_proof, &private_witness.proof.linkage_artifact)?;
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: private_witness.proof.main_decider_proof.expected_digest(),
        main_residual_proof_digest: private_witness.proof.main_residual_proof.expected_digest(),
        side_terminal_artifact_digest: private_witness.proof.side_terminal_artifact.digest,
        side_proof_artifact_digest: private_witness.proof.side_proof_artifact.digest,
        opening_artifact_digest: private_witness.proof.opening_artifact.digest,
        linkage_artifact_digest: private_witness.proof.linkage_artifact.digest,
    };
    Ok(Rv64imNightstreamBridgePrivateClaims {
        statement_digest_hint: private_witness.statement.digest(),
        verifier_context_digest: rv64im_verifier_context_digest(
            private_witness
                .proof_complete_transport
                .statement
                .root_params_id,
        ),
        fold_schedule: final_statement.folded.fold_schedule,
        proof_binding: Rv64imNightstreamBridgeProofBindingClaims {
            proof_binding_root: nightstream_proof_binding_root(
                private_witness.statement.core_digest(),
                &proof_binding_inputs,
            ),
            main_decider_target_digest: private_witness
                .proof
                .main_decider_proof
                .decider_target_digest,
            main_residual_public_statement_digest: private_witness
                .proof
                .main_residual_proof
                .public_statement_digest,
            main_residual_folded_statement_digest: private_witness
                .proof
                .main_residual_proof
                .decider_relation
                .relation_digest,
            main_residual_final_proof_digest: private_witness
                .proof
                .main_residual_proof
                .decider_relation
                .final_proof_digest,
            main_residual_kernel_export_proof_digest: private_witness
                .proof
                .main_residual_proof
                .decider_relation
                .base_component_digests
                .first()
                .copied()
                .ok_or_else(|| {
                    Rv64imBridgeError::PrivateClaims(
                        "main residual proof is missing the kernel export component digest".into(),
                    )
                })?,
            side_terminal_artifact_digest: private_witness.proof.side_terminal_artifact.digest,
            side_proof_artifact_digest: private_witness.proof.side_proof_artifact.digest,
            opening_artifact_digest: private_witness.proof.opening_artifact.digest,
            linkage_artifact_digest: private_witness.proof.linkage_artifact.digest,
        },
        linkage: Rv64imNightstreamBridgeLinkageClaims {
            kernel_export_anchor_digest: final_proof.kernel_export.digest,
            linkage_root: rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims),
            public_chunk_digests: linkage_claims.public_chunk_digests,
            bridge_handoff_digests: linkage_claims.bridge_handoff_digests,
            linkage_claims_digest: linkage_claims.digest,
        },
        chunk_transitions: main_residual
            .decider_relation
            .chunk_transition_bindings
            .iter()
            .map(|binding| Rv64imNightstreamBridgeChunkTransitionClaim {
                chunk_relation_digest: binding.claimed_chunk_relation_digest,
                transition_witness_digest: binding.transition_witness_digest,
            })
            .collect(),
    })
}

fn verify_rv64im_nightstream_bridge_private_claims(
    claims: &Rv64imNightstreamBridgePrivateClaims,
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<(), Rv64imBridgeError> {
    let expected = build_rv64im_nightstream_bridge_private_claims(private_witness)?;
    if claims != &expected {
        return Err(Rv64imBridgeError::PrivateClaims(
            "bridge private claims do not match the verified final seam".into(),
        ));
    }
    if claims.statement_digest_hint != private_witness.statement.digest() {
        return Err(Rv64imBridgeError::PrivateClaims(
            "statement_digest_hint does not match the carried statement".into(),
        ));
    }
    if claims.verifier_context_digest != private_witness.statement.verifier_context_digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "verifier_context_digest does not match the carried statement".into(),
        ));
    }
    if claims.fold_schedule != private_witness.statement.fold_schedule {
        return Err(Rv64imBridgeError::PrivateClaims(
            "fold_schedule does not match the carried statement".into(),
        ));
    }
    if claims.proof_binding.proof_binding_root != private_witness.statement.proof_binding_root {
        return Err(Rv64imBridgeError::PrivateClaims(
            "proof_binding_root does not match the carried statement".into(),
        ));
    }
    if claims.proof_binding.main_decider_target_digest
        != private_witness
            .proof
            .main_decider_proof
            .decider_target_digest
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_decider_target_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.main_residual_public_statement_digest
        != private_witness
            .proof
            .main_residual_proof
            .public_statement_digest
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_residual_public_statement_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.main_residual_folded_statement_digest
        != private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .relation_digest
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_residual_folded_statement_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.main_residual_final_proof_digest
        != private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .final_proof_digest
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_residual_final_proof_digest does not match the carried proof".into(),
        ));
    }
    if claims
        .proof_binding
        .main_residual_kernel_export_proof_digest
        != private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .base_component_digests
            .first()
            .copied()
            .ok_or_else(|| {
                Rv64imBridgeError::PrivateClaims(
                    "main residual proof is missing the kernel export component digest".into(),
                )
            })?
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_residual_kernel_export_proof_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.side_terminal_artifact_digest != private_witness.proof.side_terminal_artifact.digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "side_terminal_artifact_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.side_proof_artifact_digest != private_witness.proof.side_proof_artifact.digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "side_proof_artifact_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.opening_artifact_digest != private_witness.proof.opening_artifact.digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "opening_artifact_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.linkage_artifact_digest != private_witness.proof.linkage_artifact.digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "linkage_artifact_digest does not match the carried proof".into(),
        ));
    }
    if claims.linkage.kernel_export_anchor_digest
        != private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .base_component_digests
            .first()
            .copied()
            .ok_or_else(|| {
                Rv64imBridgeError::PrivateClaims(
                    "main residual proof is missing the kernel export component digest".into(),
                )
            })?
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "kernel_export_anchor_digest does not match the carried residual proof".into(),
        ));
    }
    if claims.linkage.linkage_root != private_witness.statement.linkage_root {
        return Err(Rv64imBridgeError::PrivateClaims(
            "linkage_root does not match the carried statement".into(),
        ));
    }
    if claims.linkage.linkage_claims_digest != private_witness.proof.linkage_artifact.digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "linkage_claims_digest does not match the carried linkage artifact".into(),
        ));
    }
    let statement_public_chunk_digests: Vec<[u8; 32]> = private_witness
        .statement
        .chunk_summaries
        .iter()
        .map(|summary| summary.public_chunk_digest)
        .collect();
    if claims.linkage.public_chunk_digests != statement_public_chunk_digests {
        return Err(Rv64imBridgeError::PrivateClaims(
            "linkage public_chunk_digests do not match the carried statement".into(),
        ));
    }
    if claims.chunk_transitions.len() != private_witness.statement.chunk_summaries.len() {
        return Err(Rv64imBridgeError::PrivateClaims(
            "chunk transition claims do not match the carried statement chunk count".into(),
        ));
    }
    if claims.chunk_transitions.len()
        != private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .chunk_transition_bindings
            .len()
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "chunk transition claims do not match the carried residual transition count".into(),
        ));
    }
    for (index, ((claim, summary), transition_digest)) in claims
        .chunk_transitions
        .iter()
        .zip(private_witness.statement.chunk_summaries.iter())
        .zip(
            private_witness
                .proof
                .main_residual_proof
                .decider_relation
                .chunk_transition_bindings
                .iter(),
        )
        .enumerate()
    {
        if claim.chunk_relation_digest != summary.chunk_relation_digest {
            return Err(Rv64imBridgeError::PrivateClaims(format!(
                "chunk transition claim {index} does not match the carried chunk relation digest"
            )));
        }
        if claim.transition_witness_digest != transition_digest.transition_witness_digest {
            return Err(Rv64imBridgeError::PrivateClaims(format!(
                "chunk transition claim {index} does not match the carried transition witness digest"
            )));
        }
    }
    Ok(())
}

pub fn encode_rv64im_nightstream_bridge_public_inputs_fields(
    public_inputs: Rv64imNightstreamBridgePublicInputs,
) -> Vec<BridgeFieldWord> {
    let mut out = Vec::with_capacity(1 + DIGEST32_FIELD_WORDS);
    out.push(public_inputs.version as BridgeFieldWord);
    encode_digest32_field_words(&mut out, public_inputs.statement_digest);
    out
}

pub fn decode_rv64im_nightstream_bridge_public_inputs_fields(
    words: &[BridgeFieldWord],
) -> Result<Rv64imNightstreamBridgePublicInputs, Rv64imBridgeError> {
    let mut cursor = 0;
    let version_word = take_word(words, &mut cursor, "bridge version")?;
    let version = u32::try_from(version_word).map_err(|_| {
        Rv64imBridgeError::InvalidEncoding(format!("bridge version {version_word} does not fit into u32"))
    })?;
    let statement_digest = decode_digest32_field_words(words, &mut cursor, "public statement digest")?;
    if cursor != words.len() {
        return Err(Rv64imBridgeError::InvalidEncoding(format!(
            "public input has {} trailing field words",
            words.len() - cursor
        )));
    }
    Ok(Rv64imNightstreamBridgePublicInputs {
        version,
        statement_digest,
    })
}

pub fn rv64im_nightstream_bridge_binding_input(public_inputs: Rv64imNightstreamBridgePublicInputs) -> BridgeFieldWord {
    first_digest32_field_word(public_inputs.statement_digest)
}

pub fn encode_rv64im_nightstream_bridge_private_witness_fields(
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<Vec<BridgeFieldWord>, Rv64imBridgeError> {
    let mut out = Vec::new();
    let private_claims = build_rv64im_nightstream_bridge_private_claims(private_witness)?;
    encode_rv64im_nightstream_bridge_private_claims_fields(&mut out, &private_claims);
    encode_nightstream_statement_fields(&mut out, private_witness.statement);
    encode_rv64im_nightstream_proof_fields(&mut out, private_witness.proof)?;
    let proof_bytes = bincode::serialize(private_witness.proof_complete_transport)
        .map_err(|err| Rv64imBridgeError::WitnessEncode(err.to_string()))?;
    out.extend(encode_bytes_field_words(&proof_bytes));
    Ok(out)
}

fn decode_rv64im_nightstream_bridge_private_payload_fields(
    words: &[BridgeFieldWord],
) -> Result<OwnedRv64imNightstreamBridgePrivatePayload, Rv64imBridgeError> {
    let mut cursor = 0;
    let claims = decode_rv64im_nightstream_bridge_private_claims_fields(words, &mut cursor)?;
    let statement = decode_nightstream_statement_fields(words, &mut cursor)?;
    let proof = decode_rv64im_nightstream_proof_fields(words, &mut cursor)?;
    let proof_bytes = decode_bytes_field_words(words, &mut cursor, "proof-complete transport bytes")?;
    if cursor != words.len() {
        return Err(Rv64imBridgeError::InvalidEncoding(format!(
            "private witness has {} trailing field words",
            words.len() - cursor
        )));
    }
    let proof_complete_transport = bincode::deserialize::<Rv64imProof>(&proof_bytes)
        .map_err(|err| Rv64imBridgeError::WitnessDecode(err.to_string()))?;
    let witness = OwnedRv64imNightstreamBridgePrivateWitness {
        statement,
        proof,
        proof_complete_transport,
    };
    verify_rv64im_nightstream_bridge_private_claims(&claims, witness.borrowed())?;
    Ok(OwnedRv64imNightstreamBridgePrivatePayload { claims, witness })
}

pub fn decode_rv64im_nightstream_bridge_private_witness_fields(
    words: &[BridgeFieldWord],
) -> Result<OwnedRv64imNightstreamBridgePrivateWitness, Rv64imBridgeError> {
    Ok(decode_rv64im_nightstream_bridge_private_payload_fields(words)?.witness)
}

pub fn verify_rv64im_nightstream_bridge_payload(
    public_inputs: &[BridgeFieldWord],
    private_witness: &[BridgeFieldWord],
) -> Result<(), Rv64imBridgeError> {
    let public_inputs = decode_rv64im_nightstream_bridge_public_inputs_fields(public_inputs)?;
    let private_payload = decode_rv64im_nightstream_bridge_private_payload_fields(private_witness)?;
    verify_rv64im_nightstream_bridge_input(public_inputs, private_payload.witness.borrowed())
}

pub fn build_rv64im_nightstream_bridge_preimage(
    public_inputs: Rv64imNightstreamBridgePublicInputs,
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<Rv64imNightstreamBridgePreimage, Rv64imBridgeError> {
    let binding_input = rv64im_nightstream_bridge_binding_input(public_inputs);
    Ok(Rv64imNightstreamBridgePreimage {
        inputs: encode_rv64im_nightstream_bridge_public_inputs_fields(public_inputs),
        private_transcript: encode_rv64im_nightstream_bridge_private_witness_fields(private_witness)?,
        public_transcript_inputs: Vec::new(),
        public_transcript_outputs: Vec::new(),
        binding_input,
        communications_commitment: None,
        key_location: RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION.to_owned(),
    })
}

pub fn verify_rv64im_nightstream_bridge_preimage(
    preimage: &Rv64imNightstreamBridgePreimage,
) -> Result<(), Rv64imBridgeError> {
    let public_inputs = decode_rv64im_nightstream_bridge_public_inputs_fields(&preimage.inputs)?;
    let expected_binding_input = rv64im_nightstream_bridge_binding_input(public_inputs);
    if preimage.binding_input != expected_binding_input {
        return Err(Rv64imBridgeError::InvalidEncoding(format!(
            "RV64IM Nightstream bridge v1 requires binding_input {}",
            expected_binding_input
        )));
    }
    if preimage.key_location != RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION {
        return Err(Rv64imBridgeError::InvalidEncoding(format!(
            "RV64IM Nightstream bridge v1 requires key_location {}",
            RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION
        )));
    }
    if !preimage.public_transcript_inputs.is_empty() {
        return Err(Rv64imBridgeError::InvalidEncoding(
            "RV64IM Nightstream bridge v1 does not use public transcript inputs".into(),
        ));
    }
    if !preimage.public_transcript_outputs.is_empty() {
        return Err(Rv64imBridgeError::InvalidEncoding(
            "RV64IM Nightstream bridge v1 does not use public transcript outputs".into(),
        ));
    }
    if preimage.communications_commitment.is_some() {
        return Err(Rv64imBridgeError::InvalidEncoding(
            "RV64IM Nightstream bridge v1 does not use communications commitments".into(),
        ));
    }
    verify_rv64im_nightstream_bridge_payload(&preimage.inputs, &preimage.private_transcript)
}

pub fn build_rv64im_nightstream_midnight_proof_preimage(
    preimage: &Rv64imNightstreamBridgePreimage,
) -> Result<ProofPreimage, Rv64imBridgeError> {
    verify_rv64im_nightstream_bridge_preimage(preimage)?;
    Ok(ProofPreimage {
        inputs: bridge_field_words_to_midnight_fields(&preimage.inputs),
        private_transcript: bridge_field_words_to_midnight_fields(&preimage.private_transcript),
        public_transcript_inputs: bridge_field_words_to_midnight_fields(&preimage.public_transcript_inputs),
        public_transcript_outputs: bridge_field_words_to_midnight_fields(&preimage.public_transcript_outputs),
        binding_input: Fr::from(preimage.binding_input),
        communications_commitment: preimage
            .communications_commitment
            .map(|(a, b)| (Fr::from(a), Fr::from(b))),
        key_location: KeyLocation(Cow::Owned(preimage.key_location.clone())),
    })
}

struct VerifierIrBuilder {
    next_index: u32,
    private_words_consumed: usize,
    instructions: Vec<Instruction>,
}

impl VerifierIrBuilder {
    fn new(num_inputs: usize) -> Result<Self, Rv64imBridgeError> {
        let next_index = u32::try_from(num_inputs).map_err(|_| {
            Rv64imBridgeError::InvalidEncoding(format!("bridge IR input count {num_inputs} does not fit into u32"))
        })?;
        Ok(Self {
            next_index,
            private_words_consumed: 0,
            instructions: Vec::new(),
        })
    }

    fn load_imm(&mut self, value: BridgeFieldWord) -> u32 {
        let index = self.next_index;
        self.instructions
            .push(Instruction::LoadImm { imm: Fr::from(value) });
        self.next_index += 1;
        index
    }

    fn private_input(&mut self) -> u32 {
        let index = self.next_index;
        self.instructions
            .push(Instruction::PrivateInput { guard: None });
        self.next_index += 1;
        self.private_words_consumed += 1;
        index
    }

    fn add(&mut self, a: u32, b: u32) -> u32 {
        let index = self.next_index;
        self.instructions.push(Instruction::Add { a, b });
        self.next_index += 1;
        index
    }

    fn assert_equal(&mut self, a: u32, b: u32) {
        let eq = self.next_index;
        self.instructions.push(Instruction::TestEq { a, b });
        self.next_index += 1;
        self.instructions.push(Instruction::Assert { cond: eq });
    }

    fn finish(self) -> Arc<Vec<Instruction>> {
        Arc::new(self.instructions)
    }
}

pub fn build_rv64im_nightstream_verifier_ir_v2(
    preimage: &Rv64imNightstreamBridgePreimage,
) -> Result<IrSource, Rv64imBridgeError> {
    let public_inputs = decode_rv64im_nightstream_bridge_public_inputs_fields(&preimage.inputs)?;
    let private_payload = decode_rv64im_nightstream_bridge_private_payload_fields(&preimage.private_transcript)?;
    let private_witness = &private_payload.witness;
    let private_claims = &private_payload.claims;
    let mut builder = VerifierIrBuilder::new(preimage.inputs.len())?;

    let expected_version = builder.load_imm(RV64IM_NIGHTSTREAM_BRIDGE_VERSION as BridgeFieldWord);
    builder.assert_equal(0, expected_version);

    let mut statement_digest_hint_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut statement_digest_hint_indices {
        *index = builder.private_input();
    }
    for (offset, digest_index) in statement_digest_hint_indices.iter().enumerate() {
        let public_index = 1 + offset as u32;
        builder.assert_equal(public_index, *digest_index);
    }
    let mut verifier_context_digest_claim_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut verifier_context_digest_claim_indices {
        *index = builder.private_input();
    }
    let fold_schedule_claim_tag_index = builder.private_input();
    let fold_schedule_claim_value_index = builder.private_input();

    let mut proof_binding_root_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_root_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_main_decider_target_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_main_decider_target_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_residual_public_statement_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_residual_public_statement_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_residual_folded_statement_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_residual_folded_statement_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_residual_final_proof_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_residual_final_proof_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_residual_kernel_export_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_residual_kernel_export_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_side_terminal_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_side_terminal_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_side_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_side_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_opening_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_opening_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_binding_linkage_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_binding_linkage_artifact_digest_indices {
        *index = builder.private_input();
    }

    let mut linkage_kernel_export_anchor_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut linkage_kernel_export_anchor_digest_indices {
        *index = builder.private_input();
    }
    let mut linkage_root_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut linkage_root_indices {
        *index = builder.private_input();
    }
    let mut linkage_claims_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut linkage_claims_digest_indices {
        *index = builder.private_input();
    }
    let linkage_public_chunk_digest_count_index = builder.private_input();
    let mut linkage_public_chunk_digest_indices = Vec::with_capacity(private_claims.linkage.public_chunk_digests.len());
    for _ in &private_claims.linkage.public_chunk_digests {
        let mut digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut digest_indices {
            *index = builder.private_input();
        }
        linkage_public_chunk_digest_indices.push(digest_indices);
    }
    let linkage_bridge_handoff_digest_count_index = builder.private_input();
    for _ in &private_claims.linkage.bridge_handoff_digests {
        for _ in 0..DIGEST32_FIELD_WORDS {
            builder.private_input();
        }
    }
    let chunk_transition_claim_count_index = builder.private_input();
    let mut chunk_transition_claim_relation_digest_indices = Vec::with_capacity(private_claims.chunk_transitions.len());
    let mut chunk_transition_claim_witness_digest_indices = Vec::with_capacity(private_claims.chunk_transitions.len());
    for _ in &private_claims.chunk_transitions {
        let mut relation_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut relation_digest_indices {
            *index = builder.private_input();
        }
        chunk_transition_claim_relation_digest_indices.push(relation_digest_indices);
        let mut witness_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut witness_digest_indices {
            *index = builder.private_input();
        }
        chunk_transition_claim_witness_digest_indices.push(witness_digest_indices);
    }

    let mut statement_public_io_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut statement_public_io_digest_indices {
        *index = builder.private_input();
    }
    let mut statement_verifier_context_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut statement_verifier_context_digest_indices {
        *index = builder.private_input();
    }
    let statement_fold_schedule_tag_index = builder.private_input();
    let statement_fold_schedule_value_index = builder.private_input();
    let semantic_step_count_index = builder.private_input();
    let chunk_summary_count_index = builder.private_input();

    let mut chunk_start_index_indices = Vec::with_capacity(private_witness.statement.chunk_summaries.len());
    let mut chunk_public_step_count_indices = Vec::with_capacity(private_witness.statement.chunk_summaries.len());
    let mut statement_chunk_public_digest_indices = Vec::with_capacity(private_witness.statement.chunk_summaries.len());
    let mut statement_chunk_relation_digest_indices =
        Vec::with_capacity(private_witness.statement.chunk_summaries.len());
    for _ in &private_witness.statement.chunk_summaries {
        chunk_start_index_indices.push(builder.private_input());
        chunk_public_step_count_indices.push(builder.private_input());
        let mut public_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut public_digest_indices {
            *index = builder.private_input();
        }
        statement_chunk_public_digest_indices.push(public_digest_indices);
        let mut relation_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut relation_digest_indices {
            *index = builder.private_input();
        }
        statement_chunk_relation_digest_indices.push(relation_digest_indices);
    }
    let mut statement_linkage_root_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut statement_linkage_root_indices {
        *index = builder.private_input();
    }

    let mut statement_proof_binding_root_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut statement_proof_binding_root_indices {
        *index = builder.private_input();
    }

    let mut proof_main_decider_target_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_main_decider_target_digest_indices {
        *index = builder.private_input();
    }

    let mut residual_public_statement_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut residual_public_statement_digest_indices {
        *index = builder.private_input();
    }
    let mut residual_folded_statement_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut residual_folded_statement_digest_indices {
        *index = builder.private_input();
    }
    let mut residual_final_proof_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut residual_final_proof_digest_indices {
        *index = builder.private_input();
    }
    let mut residual_kernel_export_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut residual_kernel_export_digest_indices {
        *index = builder.private_input();
    }
    let chunk_transition_count_index = builder.private_input();
    let mut residual_chunk_transition_digest_indices = Vec::with_capacity(
        private_witness
            .proof
            .main_residual_proof
            .decider_relation
            .chunk_transition_bindings
            .len(),
    );
    for _ in &private_witness
        .proof
        .main_residual_proof
        .decider_relation
        .chunk_transition_bindings
    {
        let mut digest_indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut digest_indices {
            *index = builder.private_input();
        }
        residual_chunk_transition_digest_indices.push(digest_indices);
    }
    let mut proof_side_terminal_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_side_terminal_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_side_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_side_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_opening_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_opening_artifact_digest_indices {
        *index = builder.private_input();
    }
    let mut proof_linkage_artifact_digest_indices = [0u32; DIGEST32_FIELD_WORDS];
    for index in &mut proof_linkage_artifact_digest_indices {
        *index = builder.private_input();
    }

    for (statement_index, residual_index) in statement_public_io_digest_indices
        .iter()
        .zip(residual_public_statement_digest_indices.iter())
    {
        builder.assert_equal(*statement_index, *residual_index);
    }
    for (statement_index, claim_index) in statement_verifier_context_digest_indices
        .iter()
        .zip(verifier_context_digest_claim_indices.iter())
    {
        builder.assert_equal(*statement_index, *claim_index);
    }
    builder.assert_equal(statement_fold_schedule_tag_index, fold_schedule_claim_tag_index);
    builder.assert_equal(statement_fold_schedule_value_index, fold_schedule_claim_value_index);
    for (statement_index, proof_binding_index) in statement_proof_binding_root_indices
        .iter()
        .zip(proof_binding_root_indices.iter())
    {
        builder.assert_equal(*statement_index, *proof_binding_index);
    }
    for (proof_index, claim_index) in proof_main_decider_target_digest_indices
        .iter()
        .zip(proof_binding_main_decider_target_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in residual_public_statement_digest_indices
        .iter()
        .zip(proof_binding_residual_public_statement_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in residual_folded_statement_digest_indices
        .iter()
        .zip(proof_binding_residual_folded_statement_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in residual_final_proof_digest_indices
        .iter()
        .zip(proof_binding_residual_final_proof_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in residual_kernel_export_digest_indices
        .iter()
        .zip(proof_binding_residual_kernel_export_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in proof_side_terminal_artifact_digest_indices
        .iter()
        .zip(proof_binding_side_terminal_artifact_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in proof_side_artifact_digest_indices
        .iter()
        .zip(proof_binding_side_artifact_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in proof_opening_artifact_digest_indices
        .iter()
        .zip(proof_binding_opening_artifact_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    for (proof_index, claim_index) in proof_linkage_artifact_digest_indices
        .iter()
        .zip(proof_binding_linkage_artifact_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *claim_index);
    }
    builder.assert_equal(chunk_summary_count_index, chunk_transition_count_index);
    builder.assert_equal(chunk_summary_count_index, linkage_public_chunk_digest_count_index);
    builder.assert_equal(chunk_summary_count_index, linkage_bridge_handoff_digest_count_index);
    builder.assert_equal(chunk_summary_count_index, chunk_transition_claim_count_index);
    for (statement_digest, linkage_digest) in statement_chunk_public_digest_indices
        .iter()
        .zip(linkage_public_chunk_digest_indices.iter())
    {
        for (statement_index, linkage_index) in statement_digest.iter().zip(linkage_digest.iter()) {
            builder.assert_equal(*statement_index, *linkage_index);
        }
    }
    for (statement_digest, claim_digest) in statement_chunk_relation_digest_indices
        .iter()
        .zip(chunk_transition_claim_relation_digest_indices.iter())
    {
        for (statement_index, claim_index) in statement_digest.iter().zip(claim_digest.iter()) {
            builder.assert_equal(*statement_index, *claim_index);
        }
    }
    for (residual_index, linkage_index) in residual_kernel_export_digest_indices
        .iter()
        .zip(linkage_kernel_export_anchor_digest_indices.iter())
    {
        builder.assert_equal(*residual_index, *linkage_index);
    }
    for (statement_index, linkage_index) in statement_linkage_root_indices
        .iter()
        .zip(linkage_root_indices.iter())
    {
        builder.assert_equal(*statement_index, *linkage_index);
    }
    for (proof_index, linkage_index) in proof_linkage_artifact_digest_indices
        .iter()
        .zip(linkage_claims_digest_indices.iter())
    {
        builder.assert_equal(*proof_index, *linkage_index);
    }
    for (residual_digest, claim_digest) in residual_chunk_transition_digest_indices
        .iter()
        .zip(chunk_transition_claim_witness_digest_indices.iter())
    {
        for (residual_index, claim_index) in residual_digest.iter().zip(claim_digest.iter()) {
            builder.assert_equal(*residual_index, *claim_index);
        }
    }

    let semantic_step_sum_index = if chunk_public_step_count_indices.is_empty() {
        builder.load_imm(0)
    } else {
        let mut sum = chunk_public_step_count_indices[0];
        for index in chunk_public_step_count_indices.iter().skip(1) {
            sum = builder.add(sum, *index);
        }
        sum
    };
    builder.assert_equal(semantic_step_count_index, semantic_step_sum_index);

    let mut expected_start_index = builder.load_imm(0);
    for (start_index, public_step_count) in chunk_start_index_indices
        .iter()
        .zip(chunk_public_step_count_indices.iter())
    {
        builder.assert_equal(*start_index, expected_start_index);
        expected_start_index = builder.add(expected_start_index, *public_step_count);
    }
    builder.assert_equal(expected_start_index, semantic_step_count_index);

    while builder.private_words_consumed < preimage.private_transcript.len() {
        builder.private_input();
    }
    if public_inputs.statement_digest != private_witness.statement.digest() {
        return Err(Rv64imBridgeError::StatementDigestMismatch {
            expected: public_inputs.statement_digest,
            actual: private_witness.statement.digest(),
        });
    }
    Ok(IrSource {
        num_inputs: preimage.inputs.len() as u32,
        do_communications_commitment: false,
        instructions: builder.finish(),
    })
}

pub fn check_rv64im_nightstream_verifier_ir_v2(
    preimage: &Rv64imNightstreamBridgePreimage,
) -> Result<Vec<Option<usize>>, Rv64imBridgeError> {
    let ir = build_rv64im_nightstream_verifier_ir_v2(preimage)?;
    let proof_preimage = build_rv64im_nightstream_midnight_proof_preimage(preimage)?;
    proof_preimage
        .check(&ir)
        .map_err(|err| Rv64imBridgeError::InvalidEncoding(err.to_string()))
}

fn take_word(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
    label: &'static str,
) -> Result<BridgeFieldWord, Rv64imBridgeError> {
    let word = words
        .get(*cursor)
        .copied()
        .ok_or(Rv64imBridgeError::Truncated(label))?;
    *cursor += 1;
    Ok(word)
}

fn usize_from_word(word: BridgeFieldWord, label: &'static str) -> Result<usize, Rv64imBridgeError> {
    usize::try_from(word)
        .map_err(|_| Rv64imBridgeError::InvalidEncoding(format!("{label} {word} does not fit into usize")))
}

fn bridge_field_words_to_midnight_fields(words: &[BridgeFieldWord]) -> Vec<Fr> {
    words.iter().copied().map(Fr::from).collect()
}

fn encode_bytes_field_words(bytes: &[u8]) -> Vec<BridgeFieldWord> {
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_FIELD_WORD));
    out.push(bytes.len() as BridgeFieldWord);
    for chunk in bytes.chunks(BYTES_PER_FIELD_WORD) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(u64::from_le_bytes(limb));
    }
    out
}

fn decode_bytes_field_words(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
    label: &'static str,
) -> Result<Vec<u8>, Rv64imBridgeError> {
    let byte_len = usize_from_word(take_word(words, cursor, label)?, label)?;
    let limb_count = byte_len.div_ceil(BYTES_PER_FIELD_WORD);
    let mut out = Vec::with_capacity(byte_len);
    for _ in 0..limb_count {
        let limb = take_word(words, cursor, label)?.to_le_bytes();
        let remaining = byte_len - out.len();
        let take = remaining.min(BYTES_PER_FIELD_WORD);
        out.extend_from_slice(&limb[..take]);
    }
    Ok(out)
}

fn encode_digest32_field_words(out: &mut Vec<BridgeFieldWord>, digest: [u8; 32]) {
    for chunk in digest.as_slice().chunks(BYTES_PER_FIELD_WORD) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(u64::from_le_bytes(limb));
    }
}

fn first_digest32_field_word(digest: [u8; 32]) -> BridgeFieldWord {
    let mut limb = [0u8; 8];
    limb[..BYTES_PER_FIELD_WORD].copy_from_slice(&digest[..BYTES_PER_FIELD_WORD]);
    u64::from_le_bytes(limb)
}

fn decode_digest32_field_words(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
    label: &'static str,
) -> Result<[u8; 32], Rv64imBridgeError> {
    let mut out = [0u8; 32];
    let mut offset = 0;
    for _ in 0..DIGEST32_FIELD_WORDS {
        let limb = take_word(words, cursor, label)?.to_le_bytes();
        let take = (32 - offset).min(BYTES_PER_FIELD_WORD);
        out[offset..offset + take].copy_from_slice(&limb[..take]);
        offset += take;
    }
    Ok(out)
}

fn encode_rv64im_nightstream_bridge_private_claims_fields(
    out: &mut Vec<BridgeFieldWord>,
    claims: &Rv64imNightstreamBridgePrivateClaims,
) {
    encode_digest32_field_words(out, claims.statement_digest_hint);
    encode_digest32_field_words(out, claims.verifier_context_digest);
    encode_fold_schedule_fields(out, claims.fold_schedule);
    encode_rv64im_nightstream_bridge_proof_binding_claims_fields(out, &claims.proof_binding);
    encode_rv64im_nightstream_bridge_linkage_claims_fields(out, &claims.linkage);
    encode_rv64im_nightstream_bridge_chunk_transition_claims_fields(out, &claims.chunk_transitions);
}

fn decode_rv64im_nightstream_bridge_private_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamBridgePrivateClaims, Rv64imBridgeError> {
    Ok(Rv64imNightstreamBridgePrivateClaims {
        statement_digest_hint: decode_digest32_field_words(words, cursor, "bridge witness statement_digest_hint")?,
        verifier_context_digest: decode_digest32_field_words(words, cursor, "bridge witness verifier_context_digest")?,
        fold_schedule: decode_fold_schedule_fields(words, cursor)?,
        proof_binding: decode_rv64im_nightstream_bridge_proof_binding_claims_fields(words, cursor)?,
        linkage: decode_rv64im_nightstream_bridge_linkage_claims_fields(words, cursor)?,
        chunk_transitions: decode_rv64im_nightstream_bridge_chunk_transition_claims_fields(words, cursor)?,
    })
}

fn encode_rv64im_nightstream_bridge_proof_binding_claims_fields(
    out: &mut Vec<BridgeFieldWord>,
    claims: &Rv64imNightstreamBridgeProofBindingClaims,
) {
    encode_digest32_field_words(out, claims.proof_binding_root);
    encode_digest32_field_words(out, claims.main_decider_target_digest);
    encode_digest32_field_words(out, claims.main_residual_public_statement_digest);
    encode_digest32_field_words(out, claims.main_residual_folded_statement_digest);
    encode_digest32_field_words(out, claims.main_residual_final_proof_digest);
    encode_digest32_field_words(out, claims.main_residual_kernel_export_proof_digest);
    encode_digest32_field_words(out, claims.side_terminal_artifact_digest);
    encode_digest32_field_words(out, claims.side_proof_artifact_digest);
    encode_digest32_field_words(out, claims.opening_artifact_digest);
    encode_digest32_field_words(out, claims.linkage_artifact_digest);
}

fn decode_rv64im_nightstream_bridge_proof_binding_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamBridgeProofBindingClaims, Rv64imBridgeError> {
    Ok(Rv64imNightstreamBridgeProofBindingClaims {
        proof_binding_root: decode_digest32_field_words(words, cursor, "bridge proof binding root")?,
        main_decider_target_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding main_decider_target_digest",
        )?,
        main_residual_public_statement_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding main_residual_public_statement_digest",
        )?,
        main_residual_folded_statement_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding main_residual_folded_statement_digest",
        )?,
        main_residual_final_proof_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding main_residual_final_proof_digest",
        )?,
        main_residual_kernel_export_proof_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding main_residual_kernel_export_proof_digest",
        )?,
        side_terminal_artifact_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding side_terminal_artifact_digest",
        )?,
        side_proof_artifact_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding side_proof_artifact_digest",
        )?,
        opening_artifact_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding opening_artifact_digest",
        )?,
        linkage_artifact_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding linkage_artifact_digest",
        )?,
    })
}

fn encode_rv64im_nightstream_bridge_linkage_claims_fields(
    out: &mut Vec<BridgeFieldWord>,
    claims: &Rv64imNightstreamBridgeLinkageClaims,
) {
    encode_digest32_field_words(out, claims.kernel_export_anchor_digest);
    encode_digest32_field_words(out, claims.linkage_root);
    encode_digest32_field_words(out, claims.linkage_claims_digest);
    out.push(claims.public_chunk_digests.len() as BridgeFieldWord);
    for digest in &claims.public_chunk_digests {
        encode_digest32_field_words(out, *digest);
    }
    out.push(claims.bridge_handoff_digests.len() as BridgeFieldWord);
    for digest in &claims.bridge_handoff_digests {
        encode_digest32_field_words(out, *digest);
    }
}

fn decode_rv64im_nightstream_bridge_linkage_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamBridgeLinkageClaims, Rv64imBridgeError> {
    let kernel_export_anchor_digest =
        decode_digest32_field_words(words, cursor, "bridge linkage kernel_export_anchor_digest")?;
    let linkage_root = decode_digest32_field_words(words, cursor, "bridge linkage linkage_root")?;
    let linkage_claims_digest = decode_digest32_field_words(words, cursor, "bridge linkage linkage_claims_digest")?;
    let public_chunk_digest_count = usize_from_word(
        take_word(words, cursor, "bridge linkage public_chunk_digest_count")?,
        "bridge linkage public_chunk_digest_count",
    )?;
    let mut public_chunk_digests = Vec::with_capacity(public_chunk_digest_count);
    for _ in 0..public_chunk_digest_count {
        public_chunk_digests.push(decode_digest32_field_words(
            words,
            cursor,
            "bridge linkage public_chunk_digest",
        )?);
    }
    let bridge_handoff_digest_count = usize_from_word(
        take_word(words, cursor, "bridge linkage bridge_handoff_digest_count")?,
        "bridge linkage bridge_handoff_digest_count",
    )?;
    let mut bridge_handoff_digests = Vec::with_capacity(bridge_handoff_digest_count);
    for _ in 0..bridge_handoff_digest_count {
        bridge_handoff_digests.push(decode_digest32_field_words(
            words,
            cursor,
            "bridge linkage bridge_handoff_digest",
        )?);
    }
    Ok(Rv64imNightstreamBridgeLinkageClaims {
        kernel_export_anchor_digest,
        linkage_root,
        public_chunk_digests,
        bridge_handoff_digests,
        linkage_claims_digest,
    })
}

fn encode_rv64im_nightstream_bridge_chunk_transition_claims_fields(
    out: &mut Vec<BridgeFieldWord>,
    claims: &[Rv64imNightstreamBridgeChunkTransitionClaim],
) {
    out.push(claims.len() as BridgeFieldWord);
    for claim in claims {
        encode_digest32_field_words(out, claim.chunk_relation_digest);
        encode_digest32_field_words(out, claim.transition_witness_digest);
    }
}

fn decode_rv64im_nightstream_bridge_chunk_transition_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Vec<Rv64imNightstreamBridgeChunkTransitionClaim>, Rv64imBridgeError> {
    let claim_count = usize_from_word(
        take_word(words, cursor, "bridge chunk transition claim count")?,
        "bridge chunk transition claim count",
    )?;
    let mut claims = Vec::with_capacity(claim_count);
    for _ in 0..claim_count {
        claims.push(Rv64imNightstreamBridgeChunkTransitionClaim {
            chunk_relation_digest: decode_digest32_field_words(
                words,
                cursor,
                "bridge chunk transition chunk_relation_digest",
            )?,
            transition_witness_digest: decode_digest32_field_words(
                words,
                cursor,
                "bridge chunk transition transition_witness_digest",
            )?,
        });
    }
    Ok(claims)
}

fn encode_fold_schedule_fields(out: &mut Vec<BridgeFieldWord>, schedule: FoldSchedule) {
    out.extend_from_slice(&schedule.meta_words());
}

fn decode_fold_schedule_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<FoldSchedule, Rv64imBridgeError> {
    let tag = take_word(words, cursor, "fold schedule tag")?;
    let value = take_word(words, cursor, "fold schedule value")?;
    let schedule = match tag {
        0 if value == 0 => FoldSchedule::WholeTrace,
        0 => {
            return Err(Rv64imBridgeError::InvalidEncoding(format!(
                "WholeTrace fold schedule must carry zero value, got {value}"
            )))
        }
        1 => FoldSchedule::RowsPerChunk(usize_from_word(value, "RowsPerChunk value")?),
        _ => {
            return Err(Rv64imBridgeError::InvalidEncoding(format!(
                "unknown fold schedule tag {tag}"
            )))
        }
    };
    schedule
        .validate()
        .map_err(|err| Rv64imBridgeError::InvalidEncoding(err.to_string()))?;
    Ok(schedule)
}

fn encode_chunk_summary_fields(out: &mut Vec<BridgeFieldWord>, summary: &FixedShapeChunkSummary) {
    out.push(summary.start_index);
    out.push(summary.public_step_count);
    encode_digest32_field_words(out, summary.public_chunk_digest);
    encode_digest32_field_words(out, summary.chunk_relation_digest);
}

fn decode_chunk_summary_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<FixedShapeChunkSummary, Rv64imBridgeError> {
    Ok(FixedShapeChunkSummary {
        start_index: take_word(words, cursor, "chunk summary start_index")?,
        public_step_count: take_word(words, cursor, "chunk summary public_step_count")?,
        public_chunk_digest: decode_digest32_field_words(words, cursor, "chunk summary public_chunk_digest")?,
        chunk_relation_digest: decode_digest32_field_words(words, cursor, "chunk summary chunk_relation_digest")?,
    })
}

fn encode_nightstream_statement_fields(out: &mut Vec<BridgeFieldWord>, statement: &NightstreamStatement) {
    encode_digest32_field_words(out, statement.public_io_digest);
    encode_digest32_field_words(out, statement.verifier_context_digest);
    encode_fold_schedule_fields(out, statement.fold_schedule);
    out.push(statement.semantic_step_count);
    out.push(statement.chunk_summaries.len() as BridgeFieldWord);
    for summary in &statement.chunk_summaries {
        encode_chunk_summary_fields(out, summary);
    }
    encode_digest32_field_words(out, statement.linkage_root);
    encode_digest32_field_words(out, statement.proof_binding_root);
}

fn decode_nightstream_statement_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<NightstreamStatement, Rv64imBridgeError> {
    let public_io_digest = decode_digest32_field_words(words, cursor, "statement public_io_digest")?;
    let verifier_context_digest = decode_digest32_field_words(words, cursor, "statement verifier_context_digest")?;
    let fold_schedule = decode_fold_schedule_fields(words, cursor)?;
    let semantic_step_count = take_word(words, cursor, "statement semantic_step_count")?;
    let chunk_count = usize_from_word(
        take_word(words, cursor, "statement chunk_summary_count")?,
        "statement chunk_summary_count",
    )?;
    let mut chunk_summaries = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        chunk_summaries.push(decode_chunk_summary_fields(words, cursor)?);
    }
    let linkage_root = decode_digest32_field_words(words, cursor, "statement linkage_root")?;
    let proof_binding_root = decode_digest32_field_words(words, cursor, "statement proof_binding_root")?;
    Ok(NightstreamStatement {
        public_io_digest,
        verifier_context_digest,
        fold_schedule,
        semantic_step_count,
        chunk_summaries,
        linkage_root,
        proof_binding_root,
    })
}
