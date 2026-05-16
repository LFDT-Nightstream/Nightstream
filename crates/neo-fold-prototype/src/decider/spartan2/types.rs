use super::*;

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
    pub(crate) shape: Spartan2DeciderShape,
    pub(crate) backend: Spartan2BackendBindingShellProverKey,
}

pub struct Spartan2DeciderVerifierKey {
    pub(crate) shape: Spartan2DeciderShape,
    pub(crate) backend: Spartan2BackendBindingShellVerifierKey,
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
