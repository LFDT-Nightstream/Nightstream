//! Owns the generic Spartan2 decider target surface and its backend-binding contract.
//!
//! Ownership:
//! - one reusable public/private target shape for decider adapters
//! - Poseidon2-only target and witness digests
//! - canonical backend-visible public IO and witness layout
//! - one owned Spartan2 decider-backend proof/key seam over that contract
//! - one strictly named public-target shell proof over the generic target
//! - does not own hidden-witness compression yet

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::crypto::poseidon2_goldilocks::{poseidon2_hash, DIGEST_LEN as POSEIDON2_DIGEST_LEN};
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::poseidon2::hash_packed_goldilocks_fields,
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::{SpartanProvePerf, R1CSSNARK},
    traits::circuit::SpartanCircuit,
    traits::snark::R1CSSNARKTrait,
};
use thiserror::Error;

use crate::finalize::{
    digest_fields_as_digest32, digest_fixed_shape_final_proof, fixed_shape_terminal_handle_digest_fields,
    validate_fixed_shape_chunk_layout, FixedShapeChunkSummary, FIXED_SHAPE_DIGEST_FIELD_LEN,
};
use crate::proof::FoldSchedule;

mod public_relation_shell;

pub use public_relation_shell::{
    prove_spartan2_public_relation_shell, setup_spartan2_public_relation_shell, verify_spartan2_public_relation_shell,
    Spartan2PublicRelationShellError, Spartan2PublicRelationShellProof, Spartan2PublicRelationShellProverKey,
    Spartan2PublicRelationShellSnark, Spartan2PublicRelationShellVerifierKey,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderStatement {
    pub public_statement_digest: [u8; 32],
    pub relation_digest: [u8; 32],
    pub final_proof_digest: [u8; 32],
    pub initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub terminal_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub fold_schedule: FoldSchedule,
    pub semantic_step_count: u64,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spartan2ChunkTransitionBinding {
    pub claimed_chunk_relation_digest: [u8; 32],
    pub transition_witness_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderWitness {
    pub base_component_digests: Vec<[u8; 32]>,
    pub chunk_transition_bindings: Vec<Spartan2ChunkTransitionBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderTarget {
    pub statement: Spartan2DeciderStatement,
    pub witness: Spartan2DeciderWitness,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderShape {
    pub base_component_count: usize,
    pub chunk_transition_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderBackendWitness {
    pub base_component_count: u64,
    pub chunk_transition_count: u64,
    pub base_component_digests: Vec<[u8; 32]>,
    pub chunk_transition_bindings: Vec<Spartan2ChunkTransitionBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spartan2DeciderBackendRelation {
    pub statement: Spartan2DeciderStatement,
    pub witness: Spartan2DeciderBackendWitness,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Spartan2DeciderRelation {
    pub public_statement_digest: [u8; 32],
    pub relation_digest: [u8; 32],
    pub final_proof_digest: [u8; 32],
    pub initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub terminal_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub fold_schedule: FoldSchedule,
    pub semantic_step_count: u64,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
    pub base_component_digests: Vec<[u8; 32]>,
    pub chunk_transition_bindings: Vec<Spartan2ChunkTransitionBinding>,
    pub digest: [u8; 32],
}

pub type Spartan2PublicTargetShellEngine = GoldilocksP3MerkleMleEngine;
pub type Spartan2PublicTargetShellSnark = R1CSSNARK<Spartan2PublicTargetShellEngine>;
pub type Spartan2PublicTargetShellProverKey = spartan2::spartan::SpartanProverKey<Spartan2PublicTargetShellEngine>;
pub type Spartan2PublicTargetShellVerifierKey = spartan2::spartan::SpartanVerifierKey<Spartan2PublicTargetShellEngine>;
pub type Spartan2BackendBindingShellEngine = GoldilocksP3MerkleMleEngine;
pub type Spartan2BackendBindingShellSnark = R1CSSNARK<Spartan2BackendBindingShellEngine>;
pub type Spartan2BackendBindingShellProverKey = spartan2::spartan::SpartanProverKey<Spartan2BackendBindingShellEngine>;
pub type Spartan2BackendBindingShellVerifierKey =
    spartan2::spartan::SpartanVerifierKey<Spartan2BackendBindingShellEngine>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spartan2PublicTargetShellProof {
    pub snark_data: Vec<u8>,
}

impl Spartan2PublicTargetShellProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spartan2BackendBindingShellProof {
    pub snark_data: Vec<u8>,
}

impl Spartan2BackendBindingShellProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

pub struct Spartan2DeciderProverKey {
    shape: Spartan2DeciderShape,
    backend: Spartan2BackendBindingShellProverKey,
}

pub struct Spartan2DeciderVerifierKey {
    shape: Spartan2DeciderShape,
    backend: Spartan2BackendBindingShellVerifierKey,
}

impl Spartan2DeciderVerifierKey {
    pub fn shape_digest(&self) -> [u8; 32] {
        self.shape.digest()
    }
}

impl Spartan2DeciderProverKey {
    pub fn shape_digest(&self) -> [u8; 32] {
        self.shape.digest()
    }

    pub fn backend_shape_sizes(&self) -> [usize; 10] {
        self.backend.sizes()
    }

    pub fn backend_shape_debug_stats(&self) -> spartan2::SplitR1CSShapeDebugStats {
        self.backend.shape_debug_stats()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spartan2DeciderProof {
    pub shape_digest: [u8; 32],
    pub snark_data: Vec<u8>,
}

impl Spartan2DeciderProof {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/proof");
        tr.append_message(b"neo.fold.next/decider/spartan2/proof/version", b"v1");
        tr.append_message(b"neo.fold.next/decider/spartan2/proof/shape_digest", &self.shape_digest);
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/proof/snark_bytes_len",
            &[self.snark_data.len() as u64],
        );
        tr.append_message(b"neo.fold.next/decider/spartan2/proof/snark_bytes", &self.snark_data);
        tr.digest32()
    }

    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Debug, Error)]
pub enum Spartan2PublicTargetShellError {
    #[error("spartan2 public-target shell setup failed: {0}")]
    Setup(String),
    #[error("spartan2 public-target shell prepare failed: {0}")]
    Prepare(String),
    #[error("spartan2 public-target shell prove failed: {0}")]
    Prove(String),
    #[error("spartan2 public-target shell verify failed: {0}")]
    Verify(String),
    #[error("spartan2 public-target shell proof encoding failed: {0}")]
    Encode(String),
    #[error("spartan2 public-target shell proof decoding failed: {0}")]
    Decode(String),
    #[error("spartan2 public-target shell public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Debug, Error)]
pub enum Spartan2BackendBindingShellError {
    #[error("spartan2 backend-binding relation surface mismatch: {0}")]
    RelationSurface(String),
    #[error("spartan2 backend-binding shell setup failed: {0}")]
    Setup(String),
    #[error("spartan2 backend-binding shell prepare failed: {0}")]
    Prepare(String),
    #[error("spartan2 backend-binding shell prove failed: {0}")]
    Prove(String),
    #[error("spartan2 backend-binding shell verify failed: {0}")]
    Verify(String),
    #[error("spartan2 backend-binding shell proof encoding failed: {0}")]
    Encode(String),
    #[error("spartan2 backend-binding shell proof decoding failed: {0}")]
    Decode(String),
    #[error("spartan2 backend-binding shell public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Debug, Error)]
pub enum Spartan2DeciderError {
    #[error(transparent)]
    Backend(#[from] Spartan2BackendBindingShellError),
    #[error("spartan2 decider relation surface mismatch: {0}")]
    RelationSurface(String),
    #[error("spartan2 decider relation digest mismatch")]
    RelationDigestMismatch,
    #[error("spartan2 decider final proof digest does not match the carried fixed-shape relation")]
    FinalProofDigestMismatch,
    #[error("spartan2 decider target shape does not match the setup shape")]
    ShapeMismatch,
    #[error("spartan2 decider proof shape digest mismatch")]
    ShapeDigestMismatch,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spartan2PublicTargetShellProvePerf {
    pub prep_ms: f64,
    pub snark_perf: SpartanProvePerf,
    pub encode_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spartan2BackendBindingShellProvePerf {
    pub prep_ms: f64,
    pub snark_perf: SpartanProvePerf,
    pub encode_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spartan2DeciderProvePerf {
    pub relation_surface_ms: f64,
    pub shell: Spartan2BackendBindingShellProvePerf,
    pub total_ms: f64,
}

impl Spartan2ChunkTransitionBinding {
    fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/chunk_transition_binding");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/chunk_transition_binding/chunk_relation_digest",
            &self.claimed_chunk_relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/chunk_transition_binding/transition_witness_digest",
            &self.transition_witness_digest,
        );
        tr.digest32()
    }

    fn packed_fields(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(Self::packed_field_len());
        extend_packed_bytes_as_fields(&mut out, &self.claimed_chunk_relation_digest);
        extend_packed_bytes_as_fields(&mut out, &self.transition_witness_digest);
        out
    }

    const fn packed_digest_field_len() -> usize {
        6
    }

    const fn claimed_chunk_relation_digest_field_offset() -> usize {
        0
    }

    const fn packed_field_len() -> usize {
        2 * Self::packed_digest_field_len()
    }
}

fn build_chunk_transition_bindings(
    chunk_summaries: &[FixedShapeChunkSummary],
    transition_witness_digests: Vec<[u8; 32]>,
) -> Result<Vec<Spartan2ChunkTransitionBinding>, Spartan2DeciderError> {
    if chunk_summaries.len() != transition_witness_digests.len() {
        return Err(Spartan2DeciderError::RelationSurface(
            "chunk summary count does not match carried chunk transition digests".into(),
        ));
    }
    Ok(chunk_summaries
        .iter()
        .zip(transition_witness_digests)
        .map(|(summary, transition_witness_digest)| Spartan2ChunkTransitionBinding {
            claimed_chunk_relation_digest: summary.chunk_relation_digest,
            transition_witness_digest,
        })
        .collect())
}

fn transition_witness_digests(bindings: &[Spartan2ChunkTransitionBinding]) -> Vec<[u8; 32]> {
    bindings
        .iter()
        .map(|binding| binding.transition_witness_digest)
        .collect()
}

fn validate_spartan2_chunk_layout(
    schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: &[FixedShapeChunkSummary],
) -> Result<(), String> {
    let active_chunk_count = chunk_summaries
        .iter()
        .position(|summary| summary.public_step_count == 0)
        .unwrap_or(chunk_summaries.len());
    let (active, padded) = chunk_summaries.split_at(active_chunk_count);
    validate_fixed_shape_chunk_layout(schedule, semantic_step_count, active)?;
    for (idx, summary) in padded.iter().enumerate() {
        if summary.public_step_count != 0 {
            return Err(format!(
                "padded chunk summary {} carries {} public steps; padded fixed-shape tails must be zero",
                active_chunk_count + idx,
                summary.public_step_count
            ));
        }
        if summary.start_index != semantic_step_count {
            return Err(format!(
                "padded chunk summary {} start index {} does not match semantic step count {}",
                active_chunk_count + idx,
                summary.start_index,
                semantic_step_count
            ));
        }
        if summary.public_chunk_digest != [0; 32] {
            return Err(format!(
                "padded chunk summary {} public chunk digest must be zero",
                active_chunk_count + idx
            ));
        }
        if summary.chunk_relation_digest != [0; 32] {
            return Err(format!(
                "padded chunk summary {} chunk relation digest must be zero",
                active_chunk_count + idx
            ));
        }
    }
    Ok(())
}

fn validate_chunk_transition_bindings(
    chunk_summaries: &[FixedShapeChunkSummary],
    chunk_transition_bindings: &[Spartan2ChunkTransitionBinding],
) -> Result<(), String> {
    if chunk_summaries.len() != chunk_transition_bindings.len() {
        return Err("chunk summary count does not match carried chunk transition bindings".into());
    }
    for (idx, (summary, binding)) in chunk_summaries
        .iter()
        .zip(chunk_transition_bindings.iter())
        .enumerate()
    {
        if summary.public_step_count == 0 {
            if binding.claimed_chunk_relation_digest != [0; 32] || binding.transition_witness_digest != [0; 32] {
                return Err(format!(
                    "padded chunk transition binding {} must be canonical zero",
                    idx
                ));
            }
            continue;
        }
        if binding.claimed_chunk_relation_digest != summary.chunk_relation_digest {
            return Err(format!(
                "chunk transition binding {} does not match the carried public chunk relation digest",
                idx
            ));
        }
    }
    Ok(())
}

fn backend_semantic_digest_fields(
    relation_digest: &[u8; 32],
    chunk_summaries: &[FixedShapeChunkSummary],
    witness: &Spartan2DeciderBackendWitness,
) -> [F; POSEIDON2_DIGEST_LEN] {
    let mut preimage = Vec::with_capacity(
        packed_bytes_field_len(32)
            + chunk_summaries.len() * FixedShapeChunkSummary::packed_field_len()
            + POSEIDON2_DIGEST_LEN,
    );
    extend_packed_bytes_as_fields(&mut preimage, relation_digest);
    for summary in chunk_summaries {
        preimage.extend(summary.packed_fields());
    }
    preimage.extend(witness.digest_fields());
    poseidon2_hash(&preimage)
}

impl Spartan2DeciderRelation {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/relation");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/relation_digest",
            &self.relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/final_proof_digest",
            &self.final_proof_digest,
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/relation/initial_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.initial_handle_digest.iter().copied(),
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/relation/terminal_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.terminal_handle_digest.iter().copied(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/chunk_count",
            &[self.chunk_summaries.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/semantic_step_count",
            &[self.semantic_step_count],
        );
        for summary in &self.chunk_summaries {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/relation/chunk_summary",
                &summary.digest(),
            );
        }
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/base_component_count",
            &[self.base_component_digests.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/chunk_transition_count",
            &[self.chunk_transition_bindings.len() as u64],
        );
        for digest in &self.base_component_digests {
            tr.append_message(b"neo.fold.next/decider/spartan2/relation/base_component_digest", digest);
        }
        for binding in &self.chunk_transition_bindings {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/relation/chunk_transition_binding",
                &binding.digest(),
            );
        }
        tr.digest32()
    }

    pub fn target(&self) -> Spartan2DeciderTarget {
        Spartan2DeciderTarget {
            statement: Spartan2DeciderStatement {
                public_statement_digest: self.public_statement_digest,
                relation_digest: self.relation_digest,
                final_proof_digest: self.final_proof_digest,
                initial_handle_digest: self.initial_handle_digest,
                terminal_handle_digest: self.terminal_handle_digest,
                fold_schedule: self.fold_schedule,
                semantic_step_count: self.semantic_step_count,
                chunk_summaries: self.chunk_summaries.clone(),
            },
            witness: Spartan2DeciderWitness {
                base_component_digests: self.base_component_digests.clone(),
                chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            },
        }
    }

    pub fn backend_shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.base_component_digests.len(),
            chunk_transition_count: self.chunk_transition_bindings.len(),
        }
    }

    pub fn backend_relation(&self) -> Spartan2DeciderBackendRelation {
        Spartan2DeciderBackendRelation {
            statement: Spartan2DeciderStatement {
                public_statement_digest: self.public_statement_digest,
                relation_digest: self.relation_digest,
                final_proof_digest: self.final_proof_digest,
                initial_handle_digest: self.initial_handle_digest,
                terminal_handle_digest: self.terminal_handle_digest,
                fold_schedule: self.fold_schedule,
                semantic_step_count: self.semantic_step_count,
                chunk_summaries: self.chunk_summaries.clone(),
            },
            witness: Spartan2DeciderBackendWitness {
                base_component_count: self.base_component_digests.len() as u64,
                chunk_transition_count: self.chunk_transition_bindings.len() as u64,
                base_component_digests: self.base_component_digests.clone(),
                chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            },
        }
    }

    fn expected_final_proof_digest(&self) -> [u8; 32] {
        digest_fixed_shape_final_proof(
            &self.relation_digest,
            self.chunk_summaries.len() as u64,
            &self.chunk_summaries,
            &self.base_component_digests,
            &transition_witness_digests(&self.chunk_transition_bindings),
        )
    }

    fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        fixed_shape_terminal_handle_digest_fields(
            digest_fields_as_digest32(self.initial_handle_digest),
            &self.chunk_summaries,
        )
    }
}

pub fn build_spartan2_decider_relation(
    public_statement_digest: [u8; 32],
    relation_digest: [u8; 32],
    final_proof_digest: [u8; 32],
    initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    terminal_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    base_component_digests: Vec<[u8; 32]>,
    chunk_transition_digests: Vec<[u8; 32]>,
) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
    let chunk_transition_bindings = build_chunk_transition_bindings(&chunk_summaries, chunk_transition_digests)?;
    let mut relation = Spartan2DeciderRelation {
        public_statement_digest,
        relation_digest,
        final_proof_digest,
        initial_handle_digest,
        terminal_handle_digest,
        fold_schedule,
        semantic_step_count,
        chunk_summaries,
        base_component_digests,
        chunk_transition_bindings,
        digest: [0; 32],
    };
    relation.digest = relation.expected_digest();
    Ok(relation)
}

pub fn build_spartan2_self_bound_decider_relation(
    public_statement_digest: [u8; 32],
    relation_digest: [u8; 32],
    initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    base_component_digests: Vec<[u8; 32]>,
    chunk_transition_digests: Vec<[u8; 32]>,
) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
    let mut relation = build_spartan2_decider_relation(
        public_statement_digest,
        relation_digest,
        [0; 32],
        initial_handle_digest,
        [F::ZERO; FIXED_SHAPE_DIGEST_FIELD_LEN],
        fold_schedule,
        semantic_step_count,
        chunk_summaries,
        base_component_digests,
        chunk_transition_digests,
    )?;
    relation.terminal_handle_digest = relation.expected_terminal_handle_digest();
    relation.final_proof_digest = relation.expected_final_proof_digest();
    relation.digest = relation.expected_digest();
    Ok(relation)
}

pub fn validate_spartan2_decider_relation_surface(
    relation: &Spartan2DeciderRelation,
) -> Result<(), Spartan2DeciderError> {
    validate_chunk_transition_bindings(&relation.chunk_summaries, &relation.chunk_transition_bindings)
        .map_err(Spartan2DeciderError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        relation.fold_schedule,
        relation.semantic_step_count,
        &relation.chunk_summaries,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    if relation.terminal_handle_digest != relation.expected_terminal_handle_digest() {
        return Err(Spartan2DeciderError::RelationSurface(
            "terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if relation.digest != relation.expected_digest() {
        return Err(Spartan2DeciderError::RelationDigestMismatch);
    }
    if relation.final_proof_digest != relation.expected_final_proof_digest() {
        return Err(Spartan2DeciderError::FinalProofDigestMismatch);
    }
    Ok(())
}

impl Spartan2DeciderStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/statement");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/relation_digest",
            &self.relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/final_proof_digest",
            &self.final_proof_digest,
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/statement/initial_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.initial_handle_digest.iter().copied(),
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/statement/terminal_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.terminal_handle_digest.iter().copied(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/statement/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/statement/semantic_step_count",
            &[self.semantic_step_count],
        );
        for summary in &self.chunk_summaries {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/statement/chunk_summary",
                &summary.digest(),
            );
        }
        tr.digest32()
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            3 * packed_bytes_field_len(32)
                + 2 * FIXED_SHAPE_DIGEST_FIELD_LEN
                + 3
                + self.chunk_summaries.len() * FixedShapeChunkSummary::packed_field_len(),
        );
        extend_packed_bytes_as_fields(&mut out, &self.public_statement_digest);
        extend_packed_bytes_as_fields(&mut out, &self.relation_digest);
        extend_packed_bytes_as_fields(&mut out, &self.final_proof_digest);
        out.extend(self.initial_handle_digest);
        out.extend(self.terminal_handle_digest);
        let fold_schedule_meta = self.fold_schedule.meta_words();
        out.push(F::from_u64(fold_schedule_meta[0]));
        out.push(F::from_u64(fold_schedule_meta[1]));
        out.push(F::from_u64(self.semantic_step_count));
        for summary in &self.chunk_summaries {
            out.extend(summary.packed_fields());
        }
        out
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        fixed_shape_terminal_handle_digest_fields(
            digest_fields_as_digest32(self.initial_handle_digest),
            &self.chunk_summaries,
        )
    }
}

impl Spartan2DeciderWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/witness");
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/witness/base_component_count",
            &[self.base_component_digests.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/witness/chunk_transition_count",
            &[self.chunk_transition_bindings.len() as u64],
        );
        for digest in &self.base_component_digests {
            tr.append_message(b"neo.fold.next/decider/spartan2/witness/base_component_digest", digest);
        }
        for binding in &self.chunk_transition_bindings {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/witness/chunk_transition_binding",
                &binding.digest(),
            );
        }
        tr.digest32()
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            2 + self.base_component_digests.len() * packed_bytes_field_len(32)
                + self.chunk_transition_bindings.len() * Spartan2ChunkTransitionBinding::packed_field_len(),
        );
        out.push(F::from_u64(self.base_component_digests.len() as u64));
        out.push(F::from_u64(self.chunk_transition_bindings.len() as u64));
        for digest in &self.base_component_digests {
            extend_packed_bytes_as_fields(&mut out, digest);
        }
        for binding in &self.chunk_transition_bindings {
            out.extend(binding.packed_fields());
        }
        out
    }
}

impl Spartan2DeciderShape {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/shape");
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/base_component_count",
            &[self.base_component_count as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/chunk_transition_count",
            &[self.chunk_transition_count as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/public_io_len",
            &[self.public_io_len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/backend_public_io_len",
            &[self.backend_public_io_len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/backend_witness_field_len",
            &[self.backend_witness_field_len() as u64],
        );
        tr.digest32()
    }

    pub fn statement_public_io_len(&self) -> usize {
        3 * packed_bytes_field_len(32)
            + 2 * FIXED_SHAPE_DIGEST_FIELD_LEN
            + 3
            + self.chunk_transition_count * FixedShapeChunkSummary::packed_field_len()
    }

    pub fn witness_public_io_len(&self) -> usize {
        2 + self.base_component_count * packed_bytes_field_len(32)
            + self.chunk_transition_count * Spartan2ChunkTransitionBinding::packed_field_len()
    }

    pub fn public_io_len(&self) -> usize {
        self.statement_public_io_len() + self.witness_public_io_len()
    }

    pub fn backend_public_io_len(&self) -> usize {
        self.statement_public_io_len() + (2 * POSEIDON2_DIGEST_LEN)
    }

    pub fn backend_witness_field_len(&self) -> usize {
        2 + self.base_component_count * packed_bytes_field_len(32)
            + self.chunk_transition_count * Spartan2ChunkTransitionBinding::packed_field_len()
    }
}

impl Spartan2DeciderBackendWitness {
    pub fn digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        poseidon2_hash(&self.packed_fields())
    }

    pub fn packed_fields(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            2 + self.base_component_digests.len() * packed_bytes_field_len(32)
                + self.chunk_transition_bindings.len() * Spartan2ChunkTransitionBinding::packed_field_len(),
        );
        out.push(F::from_u64(self.base_component_count));
        out.push(F::from_u64(self.chunk_transition_count));
        for digest in &self.base_component_digests {
            extend_packed_bytes_as_fields(&mut out, digest);
        }
        for binding in &self.chunk_transition_bindings {
            out.extend(binding.packed_fields());
        }
        out
    }
}

impl Spartan2DeciderBackendRelation {
    pub fn shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.witness.base_component_digests.len(),
            chunk_transition_count: self.witness.chunk_transition_bindings.len(),
        }
    }

    pub fn witness_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.witness.digest_fields()
    }

    pub fn binding_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        let mut preimage = self.statement.public_io();
        preimage.extend(self.semantic_digest_fields());
        preimage.extend(self.witness_digest_fields());
        poseidon2_hash(&preimage)
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = self.statement.public_io();
        out.extend(self.semantic_digest_fields());
        out.extend(self.binding_digest_fields());
        out
    }

    pub fn expected_final_proof_digest(&self) -> [u8; 32] {
        digest_fixed_shape_final_proof(
            &self.statement.relation_digest,
            self.statement.chunk_summaries.len() as u64,
            &self.statement.chunk_summaries,
            &self.witness.base_component_digests,
            &transition_witness_digests(&self.witness.chunk_transition_bindings),
        )
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        self.statement.expected_terminal_handle_digest()
    }

    pub fn semantic_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        backend_semantic_digest_fields(
            &self.statement.relation_digest,
            &self.statement.chunk_summaries,
            &self.witness,
        )
    }
}

pub fn validate_spartan2_backend_relation_surface(
    relation: &Spartan2DeciderBackendRelation,
) -> Result<(), Spartan2BackendBindingShellError> {
    if relation.witness.base_component_count != relation.witness.base_component_digests.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private base component count does not match carried base component digests".into(),
        ));
    }
    if relation.witness.chunk_transition_count != relation.witness.chunk_transition_bindings.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private chunk transition count does not match carried chunk transition bindings".into(),
        ));
    }
    if relation.witness.chunk_transition_count != relation.statement.chunk_summaries.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private chunk transition count does not match carried public chunk summaries".into(),
        ));
    }
    validate_chunk_transition_bindings(
        &relation.statement.chunk_summaries,
        &relation.witness.chunk_transition_bindings,
    )
    .map_err(Spartan2BackendBindingShellError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        relation.statement.fold_schedule,
        relation.statement.semantic_step_count,
        &relation.statement.chunk_summaries,
    )
    .map_err(Spartan2BackendBindingShellError::RelationSurface)?;
    if relation.statement.terminal_handle_digest != relation.expected_terminal_handle_digest() {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "public terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if relation.statement.final_proof_digest != relation.expected_final_proof_digest() {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "public final proof digest does not match the carried fixed-shape backend relation".into(),
        ));
    }
    Ok(())
}

fn validate_spartan2_decider_target_surface(target: &Spartan2DeciderTarget) -> Result<(), Spartan2DeciderError> {
    if target.witness.chunk_transition_bindings.len() != target.statement.chunk_summaries.len() {
        return Err(Spartan2DeciderError::RelationSurface(
            "private chunk transition count does not match carried public chunk summaries".into(),
        ));
    }
    validate_chunk_transition_bindings(
        &target.statement.chunk_summaries,
        &target.witness.chunk_transition_bindings,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        target.statement.fold_schedule,
        target.statement.semantic_step_count,
        &target.statement.chunk_summaries,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    if target.statement.terminal_handle_digest != target.statement.expected_terminal_handle_digest() {
        return Err(Spartan2DeciderError::RelationSurface(
            "terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if target.statement.final_proof_digest != target.expected_final_proof_digest() {
        return Err(Spartan2DeciderError::FinalProofDigestMismatch);
    }
    Ok(())
}

impl Spartan2DeciderTarget {
    pub fn shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.witness.base_component_digests.len(),
            chunk_transition_count: self.witness.chunk_transition_bindings.len(),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/target");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/target/statement_digest",
            &self.statement.digest(),
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/target/witness_digest",
            &self.witness.digest(),
        );
        tr.digest32()
    }

    pub fn backend_witness(&self) -> Spartan2DeciderBackendWitness {
        Spartan2DeciderBackendWitness {
            base_component_count: self.witness.base_component_digests.len() as u64,
            chunk_transition_count: self.witness.chunk_transition_bindings.len() as u64,
            base_component_digests: self.witness.base_component_digests.clone(),
            chunk_transition_bindings: self.witness.chunk_transition_bindings.clone(),
        }
    }

    pub fn backend_relation(&self) -> Spartan2DeciderBackendRelation {
        Spartan2DeciderBackendRelation {
            statement: self.statement.clone(),
            witness: self.backend_witness(),
        }
    }

    pub fn relation(&self) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
        build_spartan2_decider_relation(
            self.statement.public_statement_digest,
            self.statement.relation_digest,
            self.statement.final_proof_digest,
            self.statement.initial_handle_digest,
            self.statement.terminal_handle_digest,
            self.statement.fold_schedule,
            self.statement.semantic_step_count,
            self.statement.chunk_summaries.clone(),
            self.witness.base_component_digests.clone(),
            transition_witness_digests(&self.witness.chunk_transition_bindings),
        )
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = self.statement.public_io();
        out.extend(self.witness.public_io());
        out
    }

    pub fn backend_public_io(&self) -> Vec<F> {
        self.backend_relation().public_io()
    }

    pub fn backend_semantic_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().semantic_digest_fields()
    }

    pub fn backend_witness_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().witness_digest_fields()
    }

    pub fn backend_binding_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().binding_digest_fields()
    }

    pub fn expected_final_proof_digest(&self) -> [u8; 32] {
        self.backend_relation().expected_final_proof_digest()
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        self.statement.expected_terminal_handle_digest()
    }
}

pub fn setup_spartan2_public_target_shell(
    shape: &Spartan2DeciderShape,
) -> Result<(Spartan2PublicTargetShellProverKey, Spartan2PublicTargetShellVerifierKey), Spartan2PublicTargetShellError>
{
    Spartan2PublicTargetShellSnark::setup(Spartan2PublicTargetShellCircuit::from_shape(shape))
        .map_err(|err| Spartan2PublicTargetShellError::Setup(err.to_string()))
}

pub fn prove_spartan2_public_target_shell(
    pk: &Spartan2PublicTargetShellProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<Spartan2PublicTargetShellProof, Spartan2PublicTargetShellError> {
    let (proof, _) = prove_spartan2_public_target_shell_with_perf(pk, target)?;
    Ok(proof)
}

pub fn prove_spartan2_public_target_shell_with_perf(
    pk: &Spartan2PublicTargetShellProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<(Spartan2PublicTargetShellProof, Spartan2PublicTargetShellProvePerf), Spartan2PublicTargetShellError> {
    let total_started = std::time::Instant::now();
    validate_spartan2_decider_target_surface(target)
        .map_err(|err| Spartan2PublicTargetShellError::Prove(err.to_string()))?;
    let circuit = Spartan2PublicTargetShellCircuit::from_target(target);
    let started = std::time::Instant::now();
    let prep = Spartan2PublicTargetShellSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| Spartan2PublicTargetShellError::Prepare(err.to_string()))?;
    let prep_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let (proof, snark_perf) = Spartan2PublicTargetShellSnark::prove_with_perf(pk, circuit, &prep, true)
        .map_err(|err| Spartan2PublicTargetShellError::Prove(err.to_string()))?;
    let mut snark_perf = snark_perf;
    snark_perf.total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Spartan2PublicTargetShellError::Encode(err.to_string()))?;
    let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
        Spartan2PublicTargetShellProof { snark_data },
        Spartan2PublicTargetShellProvePerf {
            prep_ms,
            snark_perf,
            encode_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_public_target_shell(
    vk: &Spartan2PublicTargetShellVerifierKey,
    target: &Spartan2DeciderTarget,
    proof: &Spartan2PublicTargetShellProof,
) -> Result<(), Spartan2PublicTargetShellError> {
    validate_spartan2_decider_target_surface(target)
        .map_err(|err| Spartan2PublicTargetShellError::Verify(err.to_string()))?;
    let proof: Spartan2PublicTargetShellSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Spartan2PublicTargetShellError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Spartan2PublicTargetShellError::Verify(err.to_string()))?
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();
    if public_values != target.public_io() {
        return Err(Spartan2PublicTargetShellError::PublicIoMismatch);
    }
    Ok(())
}

pub fn setup_spartan2_backend_binding_shell(
    shape: &Spartan2DeciderShape,
) -> Result<
    (
        Spartan2BackendBindingShellProverKey,
        Spartan2BackendBindingShellVerifierKey,
    ),
    Spartan2BackendBindingShellError,
> {
    Spartan2BackendBindingShellSnark::setup(Spartan2BackendBindingShellCircuit::from_shape(shape))
        .map_err(|err| Spartan2BackendBindingShellError::Setup(err.to_string()))
}

pub fn prove_spartan2_backend_binding_shell(
    pk: &Spartan2BackendBindingShellProverKey,
    relation: &Spartan2DeciderBackendRelation,
) -> Result<Spartan2BackendBindingShellProof, Spartan2BackendBindingShellError> {
    let (proof, _) = prove_spartan2_backend_binding_shell_with_perf(pk, relation)?;
    Ok(proof)
}

pub fn prove_spartan2_backend_binding_shell_with_perf(
    pk: &Spartan2BackendBindingShellProverKey,
    relation: &Spartan2DeciderBackendRelation,
) -> Result<(Spartan2BackendBindingShellProof, Spartan2BackendBindingShellProvePerf), Spartan2BackendBindingShellError>
{
    let total_started = std::time::Instant::now();
    validate_spartan2_backend_relation_surface(relation)?;
    let circuit = Spartan2BackendBindingShellCircuit::from_relation(relation);
    let started = std::time::Instant::now();
    let prep = Spartan2BackendBindingShellSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| Spartan2BackendBindingShellError::Prepare(err.to_string()))?;
    let prep_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let (proof, snark_perf) = Spartan2BackendBindingShellSnark::prove_with_perf(pk, circuit, &prep, true)
        .map_err(|err| Spartan2BackendBindingShellError::Prove(err.to_string()))?;
    let mut snark_perf = snark_perf;
    snark_perf.total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Spartan2BackendBindingShellError::Encode(err.to_string()))?;
    let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
        Spartan2BackendBindingShellProof { snark_data },
        Spartan2BackendBindingShellProvePerf {
            prep_ms,
            snark_perf,
            encode_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_backend_binding_shell(
    vk: &Spartan2BackendBindingShellVerifierKey,
    relation: &Spartan2DeciderBackendRelation,
    proof: &Spartan2BackendBindingShellProof,
) -> Result<(), Spartan2BackendBindingShellError> {
    validate_spartan2_backend_relation_surface(relation)?;
    let proof: Spartan2BackendBindingShellSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Spartan2BackendBindingShellError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Spartan2BackendBindingShellError::Verify(err.to_string()))?
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();
    if public_values != relation.public_io() {
        return Err(Spartan2BackendBindingShellError::PublicIoMismatch);
    }
    Ok(())
}

pub fn setup_spartan2_decider(
    shape: &Spartan2DeciderShape,
) -> Result<(Spartan2DeciderProverKey, Spartan2DeciderVerifierKey), Spartan2DeciderError> {
    let (pk, vk) = setup_spartan2_backend_binding_shell(shape)?;
    Ok((
        Spartan2DeciderProverKey {
            shape: shape.clone(),
            backend: pk,
        },
        Spartan2DeciderVerifierKey {
            shape: shape.clone(),
            backend: vk,
        },
    ))
}

pub fn prove_spartan2_decider(
    pk: &Spartan2DeciderProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<Spartan2DeciderProof, Spartan2DeciderError> {
    let (proof, _) = prove_spartan2_decider_with_perf(pk, target)?;
    Ok(proof)
}

pub fn prove_spartan2_decider_with_perf(
    pk: &Spartan2DeciderProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<(Spartan2DeciderProof, Spartan2DeciderProvePerf), Spartan2DeciderError> {
    let total_started = std::time::Instant::now();
    if target.shape() != pk.shape {
        return Err(Spartan2DeciderError::ShapeMismatch);
    }
    validate_spartan2_decider_target_surface(target)?;
    let started = std::time::Instant::now();
    let relation = target
        .relation()
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    let backend_relation = relation.backend_relation();
    validate_spartan2_backend_relation_surface(&backend_relation).map_err(Spartan2DeciderError::Backend)?;
    let relation_surface_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (backend, shell_perf) = prove_spartan2_backend_binding_shell_with_perf(&pk.backend, &backend_relation)
        .map_err(Spartan2DeciderError::Backend)?;
    Ok((
        Spartan2DeciderProof {
            shape_digest: pk.shape.digest(),
            snark_data: backend.snark_data,
        },
        Spartan2DeciderProvePerf {
            relation_surface_ms,
            shell: shell_perf,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_decider(
    vk: &Spartan2DeciderVerifierKey,
    target: &Spartan2DeciderTarget,
    proof: &Spartan2DeciderProof,
) -> Result<(), Spartan2DeciderError> {
    if target.shape() != vk.shape {
        return Err(Spartan2DeciderError::ShapeMismatch);
    }
    if proof.shape_digest != vk.shape.digest() {
        return Err(Spartan2DeciderError::ShapeDigestMismatch);
    }
    validate_spartan2_decider_target_surface(target)?;
    let relation = target
        .relation()
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    let backend_relation = relation.backend_relation();
    validate_spartan2_backend_relation_surface(&backend_relation).map_err(Spartan2DeciderError::Backend)?;
    verify_spartan2_backend_binding_shell(
        &vk.backend,
        &backend_relation,
        &Spartan2BackendBindingShellProof {
            snark_data: proof.snark_data.clone(),
        },
    )
    .map_err(Spartan2DeciderError::Backend)?;
    Ok(())
}

fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(PACKED_BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

const PACKED_BYTES_PER_LIMB: usize = 7;
const SPARTAN_GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

fn packed_bytes_field_len(bytes_len: usize) -> usize {
    1 + bytes_len.div_ceil(PACKED_BYTES_PER_LIMB)
}

fn spartan_pow(mut base: SpartanF, mut exp: u64) -> SpartanF {
    let mut acc = SpartanF::from_canonical_u64(1);
    while exp != 0 {
        if (exp & 1) == 1 {
            acc = acc * base;
        }
        base = base * base;
        exp >>= 1;
    }
    acc
}

fn spartan_inverse(value: SpartanF) -> Option<SpartanF> {
    if value.to_canonical_u64() == 0 {
        return None;
    }
    Some(spartan_pow(value, SPARTAN_GOLDILOCKS_MODULUS - 2))
}

#[derive(Clone, Debug)]
struct Spartan2PublicTargetShellCircuit {
    public_values: Vec<SpartanF>,
}

impl Spartan2PublicTargetShellCircuit {
    fn from_shape(shape: &Spartan2DeciderShape) -> Self {
        Self {
            public_values: vec![SpartanF::from_canonical_u64(0); shape.public_io_len()],
        }
    }

    fn from_target(target: &Spartan2DeciderTarget) -> Self {
        Self {
            public_values: target
                .public_io()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct Spartan2BackendBindingShellCircuit {
    public_values: Vec<SpartanF>,
    private_values: Vec<SpartanF>,
    public_semantic_offset: usize,
    public_binding_offset: usize,
    expected_base_component_count: u64,
    expected_chunk_transition_count: u64,
}

impl Spartan2BackendBindingShellCircuit {
    fn from_shape(shape: &Spartan2DeciderShape) -> Self {
        Self {
            public_values: vec![SpartanF::from_canonical_u64(0); shape.backend_public_io_len()],
            private_values: vec![SpartanF::from_canonical_u64(0); shape.backend_witness_field_len()],
            public_semantic_offset: shape.statement_public_io_len(),
            public_binding_offset: shape.statement_public_io_len() + POSEIDON2_DIGEST_LEN,
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }

    fn from_relation(relation: &Spartan2DeciderBackendRelation) -> Self {
        let shape = relation.shape();
        Self {
            public_values: relation
                .public_io()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
            private_values: relation
                .witness
                .packed_fields()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
            public_semantic_offset: relation.statement.public_io().len(),
            public_binding_offset: relation.statement.public_io().len() + POSEIDON2_DIGEST_LEN,
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }
}

impl SpartanCircuit<Spartan2PublicTargetShellEngine> for Spartan2PublicTargetShellCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.public_values.clone())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        for (idx, value) in self.public_values.iter().copied().enumerate() {
            let witness = AllocatedNum::alloc(cs.namespace(|| format!("public_target_witness_{idx}")), || Ok(value))?;
            let public =
                AllocatedNum::alloc_input(cs.namespace(|| format!("public_target_input_{idx}")), || Ok(value))?;
            cs.enforce(
                || format!("public_target_match_{idx}"),
                |lc| lc + witness.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        Ok(())
    }
}

impl SpartanCircuit<Spartan2BackendBindingShellEngine> for Spartan2BackendBindingShellCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.public_values.clone())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let mut public_inputs = Vec::with_capacity(self.public_values.len());
        for (idx, value) in self.public_values.iter().copied().enumerate() {
            public_inputs.push(AllocatedNum::alloc_input(
                cs.namespace(|| format!("backend_public_input_{idx}")),
                || Ok(value),
            )?);
        }

        let mut private_witness = Vec::with_capacity(self.private_values.len());
        for (idx, value) in self.private_values.iter().copied().enumerate() {
            private_witness.push(AllocatedNum::alloc(
                cs.namespace(|| format!("backend_private_witness_{idx}")),
                || Ok(value),
            )?);
        }

        let digest = hash_packed_goldilocks_fields(cs.namespace(|| "backend_witness_digest"), &private_witness)?;
        let private_base_count = &private_witness[0];
        let private_chunk_count = &private_witness[1];
        let packed_digest_len = packed_bytes_field_len(32);
        let base_digest_offset = 2;
        let chunk_binding_offset = base_digest_offset + self.expected_base_component_count as usize * packed_digest_len;
        let relation_digest_offset = packed_digest_len;
        let relation_digest_end = relation_digest_offset + packed_digest_len;
        let initial_handle_offset = 3 * packed_digest_len;
        let initial_handle_end = initial_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let terminal_handle_offset = initial_handle_end;
        let terminal_handle_end = terminal_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let fold_schedule_offset = terminal_handle_end;
        let semantic_step_count_offset = fold_schedule_offset + 2;
        let public_semantic_step_count = &public_inputs[semantic_step_count_offset];
        let summary_offset = semantic_step_count_offset + 1;
        let summary_end = self.public_semantic_offset;
        cs.enforce(
            || "backend_base_component_count_matches_shape",
            |lc| lc + private_base_count.get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_base_component_count),
                    CS::one(),
                )
            },
        );
        cs.enforce(
            || "backend_chunk_transition_count_matches_shape",
            |lc| lc + private_chunk_count.get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_chunk_transition_count),
                    CS::one(),
                )
            },
        );
        let mut current_handle = public_inputs[initial_handle_offset..initial_handle_end].to_vec();
        let summary_len = FixedShapeChunkSummary::packed_field_len();
        let chunk_relation_offset = FixedShapeChunkSummary::chunk_relation_digest_field_offset();
        let private_chunk_relation_offset =
            Spartan2ChunkTransitionBinding::claimed_chunk_relation_digest_field_offset();
        let private_chunk_relation_end =
            private_chunk_relation_offset + Spartan2ChunkTransitionBinding::packed_digest_field_len();
        if self.expected_chunk_transition_count == 0 {
            cs.enforce(
                || "backend_semantic_step_count_zero_when_no_chunks",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            cs.enforce(
                || "backend_first_chunk_start_index_zero",
                |lc| lc + public_inputs[summary_offset].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
            for chunk_index in 1..self.expected_chunk_transition_count as usize {
                let previous_base = summary_offset + (chunk_index - 1) * summary_len;
                let current_base = summary_offset + chunk_index * summary_len;
                cs.enforce(
                    || format!("backend_chunk_start_contiguous_{chunk_index}"),
                    |lc| lc + public_inputs[current_base].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| {
                        lc + public_inputs[previous_base].get_variable()
                            + public_inputs[previous_base + 1].get_variable()
                    },
                );
            }
            let last_base = summary_offset + (self.expected_chunk_transition_count as usize - 1) * summary_len;
            cs.enforce(
                || "backend_semantic_step_count_matches_coverage",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public_inputs[last_base].get_variable() + public_inputs[last_base + 1].get_variable(),
            );
        }
        for chunk_index in 0..self.expected_chunk_transition_count as usize {
            let chunk_index_num =
                AllocatedNum::alloc(cs.namespace(|| format!("backend_chunk_index_{chunk_index}")), || {
                    Ok(SpartanF::from_canonical_u64(chunk_index as u64))
                })?;
            cs.enforce(
                || format!("backend_chunk_index_matches_shape_{chunk_index}"),
                |lc| lc + chunk_index_num.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (SpartanF::from_canonical_u64(chunk_index as u64), CS::one()),
            );
            let summary_base = summary_offset + chunk_index * summary_len;
            let private_binding_base =
                chunk_binding_offset + chunk_index * Spartan2ChunkTransitionBinding::packed_field_len();
            let start_index = public_inputs[summary_base].clone();
            let public_step_count = public_inputs[summary_base + 1].clone();
            for digest_idx in 0..Spartan2ChunkTransitionBinding::packed_digest_field_len() {
                cs.enforce(
                    || format!("backend_chunk_relation_binding_match_{chunk_index}_{digest_idx}"),
                    |lc| {
                        lc + private_witness[private_binding_base + private_chunk_relation_offset + digest_idx]
                            .get_variable()
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + public_inputs[summary_base + chunk_relation_offset + digest_idx].get_variable(),
                );
            }
            let mut handle_preimage = Vec::with_capacity(
                FIXED_SHAPE_DIGEST_FIELD_LEN + 3 + FixedShapeChunkSummary::packed_digest_field_len(),
            );
            handle_preimage.extend(current_handle.iter().cloned());
            handle_preimage.push(chunk_index_num);
            handle_preimage.push(start_index);
            handle_preimage.push(public_step_count);
            handle_preimage.extend(
                private_witness[private_binding_base + private_chunk_relation_offset
                    ..private_binding_base + private_chunk_relation_end]
                    .iter()
                    .cloned(),
            );
            current_handle = hash_packed_goldilocks_fields(
                cs.namespace(|| format!("backend_terminal_handle_step_{chunk_index}")),
                &handle_preimage,
            )?
            .into_iter()
            .collect();
        }
        for (idx, handle_value) in current_handle.into_iter().enumerate() {
            let public = &public_inputs[terminal_handle_offset + idx];
            cs.enforce(
                || format!("backend_terminal_handle_match_{idx}"),
                |lc| lc + handle_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        let mut semantic_preimage = Vec::with_capacity(
            (relation_digest_end - relation_digest_offset) + (summary_end - summary_offset) + digest.len(),
        );
        semantic_preimage.extend(
            public_inputs[relation_digest_offset..relation_digest_end]
                .iter()
                .cloned(),
        );
        semantic_preimage.extend(public_inputs[summary_offset..summary_end].iter().cloned());
        semantic_preimage.extend(digest.iter().cloned());
        let semantic_digest =
            hash_packed_goldilocks_fields(cs.namespace(|| "backend_semantic_digest"), &semantic_preimage)?;
        for (idx, digest_value) in semantic_digest.into_iter().enumerate() {
            let public = &public_inputs[self.public_semantic_offset + idx];
            cs.enforce(
                || format!("backend_semantic_match_{idx}"),
                |lc| lc + digest_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        let mut binding_preimage = Vec::with_capacity(self.public_binding_offset + digest.len());
        binding_preimage.extend(public_inputs[..self.public_binding_offset].iter().cloned());
        binding_preimage.extend(digest);
        let binding_digest =
            hash_packed_goldilocks_fields(cs.namespace(|| "backend_binding_digest"), &binding_preimage)?;
        for (idx, digest_value) in binding_digest.into_iter().enumerate() {
            let public = &public_inputs[self.public_binding_offset + idx];
            cs.enforce(
                || format!("backend_binding_match_{idx}"),
                |lc| lc + digest_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }

        Ok(())
    }
}
