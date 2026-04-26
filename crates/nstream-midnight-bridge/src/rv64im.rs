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

use neo_fold_next::nightstream::rv64im::audit::rv64im_main_nightstream_proof_digest;
use neo_fold_next::nightstream::rv64im::{rv64im_verifier_context_digest, Rv64imNightstreamProof};
use neo_fold_next::nightstream::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::SimpleKernelError;
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
    pub trusted_root_params_id: [u8; 32],
    pub public_statement_digest: [u8; 32],
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
pub struct Rv64imNightstreamBridgePrivateClaims {
    pub statement_digest_hint: [u8; 32],
    pub verifier_context_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub semantic_step_count: u64,
    pub proof_binding: Rv64imNightstreamBridgeProofBindingClaims,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imNightstreamBridgeProofBindingClaims {
    pub proof_binding_root: [u8; 32],
    pub main_proof_digest: [u8; 32],
    pub side_proof_digest: [u8; 32],
    pub public_statement_digest: [u8; 32],
    pub published_statement_digest: [u8; 32],
}

impl<'a> Rv64imNightstreamBridgePrivateWitness<'a> {
    pub fn new(
        statement: &'a NightstreamStatement,
        proof: &'a Rv64imNightstreamProof,
        trusted_root_params_id: [u8; 32],
        public_statement_digest: [u8; 32],
    ) -> Self {
        Self {
            statement,
            proof,
            trusted_root_params_id,
            public_statement_digest,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OwnedRv64imNightstreamBridgePrivateWitness {
    pub statement: NightstreamStatement,
    pub proof: Rv64imNightstreamProof,
    pub trusted_root_params_id: [u8; 32],
    pub public_statement_digest: [u8; 32],
}

impl OwnedRv64imNightstreamBridgePrivateWitness {
    pub fn borrowed(&self) -> Rv64imNightstreamBridgePrivateWitness<'_> {
        Rv64imNightstreamBridgePrivateWitness {
            statement: &self.statement,
            proof: &self.proof,
            trusted_root_params_id: self.trusted_root_params_id,
            public_statement_digest: self.public_statement_digest,
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
    verify_rv64im_nightstream_bridge_public_inputs(public_inputs, private_witness)?;
    verify_rv64im_nightstream_bridge_expected_boundary(private_witness)?;
    let private_claims = build_rv64im_nightstream_bridge_private_claims_unchecked(private_witness)?;
    verify_rv64im_nightstream_bridge_private_claims_unchecked(&private_claims, private_witness)?;
    Ok(())
}

fn verify_rv64im_nightstream_bridge_public_inputs(
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
        private_witness.trusted_root_params_id,
        private_witness.proof.main_proof().published_statement(),
    );
    if private_witness.statement.verifier_context_digest != expected_context_digest {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream statement verifier-context digest does not match the trusted root params".into(),
        )));
    }
    Ok(())
}

fn verify_rv64im_nightstream_bridge_expected_boundary(
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<(), Rv64imBridgeError> {
    let published_statement = private_witness.proof.main_proof().published_statement();
    if private_witness.statement.public_io_digest != published_statement.expected_digest() {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream bridge statement public IO digest does not match the compressed main proof".into(),
        )));
    }
    if private_witness.statement.fold_schedule != published_statement.fold_schedule() {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream bridge statement fold schedule does not match the compressed main proof".into(),
        )));
    }
    if private_witness.statement.semantic_step_count != published_statement.step_count() {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream bridge statement step count does not match the compressed main proof".into(),
        )));
    }
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: rv64im_main_nightstream_proof_digest(private_witness.proof.main_proof()),
        side_proof_digest: private_witness.proof.side_proof().expected_digest(),
        public_statement_digest: private_witness.public_statement_digest,
    };
    let expected_proof_binding_root =
        nightstream_proof_binding_root(private_witness.statement.core_digest(), &proof_binding_inputs);
    if private_witness.statement.proof_binding_root != expected_proof_binding_root {
        return Err(Rv64imBridgeError::Nightstream(SimpleKernelError::Bridge(
            "RV64IM Nightstream bridge statement proof binding root does not match the carried proof digests".into(),
        )));
    }
    Ok(())
}

fn build_rv64im_nightstream_bridge_private_claims(
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<Rv64imNightstreamBridgePrivateClaims, Rv64imBridgeError> {
    verify_rv64im_nightstream_bridge_expected_boundary(private_witness)?;
    build_rv64im_nightstream_bridge_private_claims_unchecked(private_witness)
}

fn build_rv64im_nightstream_bridge_private_claims_unchecked(
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<Rv64imNightstreamBridgePrivateClaims, Rv64imBridgeError> {
    let main_proof_digest = rv64im_main_nightstream_proof_digest(private_witness.proof.main_proof());
    let side_proof_digest = private_witness.proof.side_proof().expected_digest();
    let public_statement_digest = private_witness.public_statement_digest;
    let published_statement_digest = private_witness
        .proof
        .main_proof()
        .published_statement()
        .expected_digest();
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest,
        side_proof_digest,
        public_statement_digest,
    };
    Ok(Rv64imNightstreamBridgePrivateClaims {
        statement_digest_hint: private_witness.statement.digest(),
        verifier_context_digest: rv64im_verifier_context_digest(
            private_witness.trusted_root_params_id,
            private_witness.proof.main_proof().published_statement(),
        ),
        fold_schedule: private_witness.statement.fold_schedule,
        semantic_step_count: private_witness.statement.semantic_step_count,
        proof_binding: Rv64imNightstreamBridgeProofBindingClaims {
            proof_binding_root: nightstream_proof_binding_root(
                private_witness.statement.core_digest(),
                &proof_binding_inputs,
            ),
            main_proof_digest,
            side_proof_digest,
            public_statement_digest,
            published_statement_digest,
        },
    })
}

fn verify_rv64im_nightstream_bridge_private_claims(
    claims: &Rv64imNightstreamBridgePrivateClaims,
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<(), Rv64imBridgeError> {
    verify_rv64im_nightstream_bridge_expected_boundary(private_witness)?;
    verify_rv64im_nightstream_bridge_private_claims_unchecked(claims, private_witness)
}

fn verify_rv64im_nightstream_bridge_private_claims_unchecked(
    claims: &Rv64imNightstreamBridgePrivateClaims,
    private_witness: Rv64imNightstreamBridgePrivateWitness<'_>,
) -> Result<(), Rv64imBridgeError> {
    let expected = build_rv64im_nightstream_bridge_private_claims_unchecked(private_witness)?;
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
    if claims.semantic_step_count != private_witness.statement.semantic_step_count {
        return Err(Rv64imBridgeError::PrivateClaims(
            "semantic_step_count does not match the carried statement".into(),
        ));
    }
    if claims.proof_binding.proof_binding_root != private_witness.statement.proof_binding_root {
        return Err(Rv64imBridgeError::PrivateClaims(
            "proof_binding_root does not match the carried statement".into(),
        ));
    }
    if claims.proof_binding.main_proof_digest
        != rv64im_main_nightstream_proof_digest(private_witness.proof.main_proof())
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "main_proof_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.side_proof_digest != private_witness.proof.side_proof().expected_digest() {
        return Err(Rv64imBridgeError::PrivateClaims(
            "side_proof_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.public_statement_digest != private_witness.public_statement_digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "public_statement_digest does not match the bridge witness".into(),
        ));
    }
    if claims.proof_binding.published_statement_digest
        != private_witness
            .proof
            .main_proof()
            .published_statement()
            .expected_digest()
    {
        return Err(Rv64imBridgeError::PrivateClaims(
            "published_statement_digest does not match the carried proof".into(),
        ));
    }
    if claims.proof_binding.published_statement_digest != private_witness.statement.public_io_digest {
        return Err(Rv64imBridgeError::PrivateClaims(
            "published_statement_digest does not match the carried statement public_io_digest".into(),
        ));
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
    encode_digest32_field_words(&mut out, private_witness.trusted_root_params_id);
    encode_digest32_field_words(&mut out, private_witness.public_statement_digest);
    Ok(out)
}

fn decode_rv64im_nightstream_bridge_private_payload_fields(
    words: &[BridgeFieldWord],
) -> Result<OwnedRv64imNightstreamBridgePrivatePayload, Rv64imBridgeError> {
    let mut cursor = 0;
    let claims = decode_rv64im_nightstream_bridge_private_claims_fields(words, &mut cursor)?;
    let statement = decode_nightstream_statement_fields(words, &mut cursor)?;
    let proof = decode_rv64im_nightstream_proof_fields(words, &mut cursor)?;
    let trusted_root_params_id = decode_digest32_field_words(words, &mut cursor, "trusted root params id")?;
    let public_statement_digest = decode_digest32_field_words(words, &mut cursor, "public statement digest")?;
    if cursor != words.len() {
        return Err(Rv64imBridgeError::InvalidEncoding(format!(
            "private witness has {} trailing field words",
            words.len() - cursor
        )));
    }
    let witness = OwnedRv64imNightstreamBridgePrivateWitness {
        statement,
        proof,
        trusted_root_params_id,
        public_statement_digest,
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
    verify_rv64im_nightstream_bridge_public_inputs(public_inputs, private_payload.witness.borrowed())
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

    fn assert_equal(&mut self, a: u32, b: u32) {
        let eq = self.next_index;
        self.instructions.push(Instruction::TestEq { a, b });
        self.next_index += 1;
        self.instructions.push(Instruction::Assert { cond: eq });
    }

    fn private_digest(&mut self) -> [u32; DIGEST32_FIELD_WORDS] {
        let mut indices = [0u32; DIGEST32_FIELD_WORDS];
        for index in &mut indices {
            *index = self.private_input();
        }
        indices
    }

    fn assert_digest_equal(&mut self, a: &[u32; DIGEST32_FIELD_WORDS], b: &[u32; DIGEST32_FIELD_WORDS]) {
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            self.assert_equal(*lhs, *rhs);
        }
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
    if private_payload.claims.statement_digest_hint != public_inputs.statement_digest {
        return Err(Rv64imBridgeError::StatementDigestMismatch {
            expected: public_inputs.statement_digest,
            actual: private_payload.claims.statement_digest_hint,
        });
    }
    let mut builder = VerifierIrBuilder::new(preimage.inputs.len())?;

    let expected_version = builder.load_imm(RV64IM_NIGHTSTREAM_BRIDGE_VERSION as BridgeFieldWord);
    builder.assert_equal(0, expected_version);

    let statement_digest_hint_indices = builder.private_digest();
    for (offset, digest_index) in statement_digest_hint_indices.iter().enumerate() {
        let public_index = 1 + offset as u32;
        builder.assert_equal(public_index, *digest_index);
    }
    let verifier_context_digest_claim_indices = builder.private_digest();
    let fold_schedule_claim_tag_index = builder.private_input();
    let fold_schedule_claim_value_index = builder.private_input();
    let semantic_step_count_claim_index = builder.private_input();

    let proof_binding_root_indices = builder.private_digest();
    let proof_binding_main_digest_indices = builder.private_digest();
    let proof_binding_side_digest_indices = builder.private_digest();
    let _proof_binding_public_statement_digest_indices = builder.private_digest();
    let proof_binding_published_statement_digest_indices = builder.private_digest();

    let statement_public_io_digest_indices = builder.private_digest();
    let statement_verifier_context_digest_indices = builder.private_digest();
    let statement_fold_schedule_tag_index = builder.private_input();
    let statement_fold_schedule_value_index = builder.private_input();
    let semantic_step_count_index = builder.private_input();
    let statement_proof_binding_root_indices = builder.private_digest();

    let proof_main_digest_indices = builder.private_digest();
    let proof_side_digest_indices = builder.private_digest();
    let proof_published_statement_digest_indices = builder.private_digest();

    builder.assert_digest_equal(
        &statement_verifier_context_digest_indices,
        &verifier_context_digest_claim_indices,
    );
    builder.assert_equal(statement_fold_schedule_tag_index, fold_schedule_claim_tag_index);
    builder.assert_equal(statement_fold_schedule_value_index, fold_schedule_claim_value_index);
    builder.assert_equal(semantic_step_count_index, semantic_step_count_claim_index);
    builder.assert_digest_equal(&statement_proof_binding_root_indices, &proof_binding_root_indices);
    builder.assert_digest_equal(
        &statement_public_io_digest_indices,
        &proof_published_statement_digest_indices,
    );
    builder.assert_digest_equal(&proof_main_digest_indices, &proof_binding_main_digest_indices);
    builder.assert_digest_equal(&proof_side_digest_indices, &proof_binding_side_digest_indices);
    builder.assert_digest_equal(
        &proof_binding_published_statement_digest_indices,
        &proof_published_statement_digest_indices,
    );

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
    out.push(claims.semantic_step_count);
    encode_rv64im_nightstream_bridge_proof_binding_claims_fields(out, &claims.proof_binding);
}

fn decode_rv64im_nightstream_bridge_private_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamBridgePrivateClaims, Rv64imBridgeError> {
    Ok(Rv64imNightstreamBridgePrivateClaims {
        statement_digest_hint: decode_digest32_field_words(words, cursor, "bridge witness statement_digest_hint")?,
        verifier_context_digest: decode_digest32_field_words(words, cursor, "bridge witness verifier_context_digest")?,
        fold_schedule: decode_fold_schedule_fields(words, cursor)?,
        semantic_step_count: take_word(words, cursor, "bridge witness semantic_step_count")?,
        proof_binding: decode_rv64im_nightstream_bridge_proof_binding_claims_fields(words, cursor)?,
    })
}

fn encode_rv64im_nightstream_bridge_proof_binding_claims_fields(
    out: &mut Vec<BridgeFieldWord>,
    claims: &Rv64imNightstreamBridgeProofBindingClaims,
) {
    encode_digest32_field_words(out, claims.proof_binding_root);
    encode_digest32_field_words(out, claims.main_proof_digest);
    encode_digest32_field_words(out, claims.side_proof_digest);
    encode_digest32_field_words(out, claims.public_statement_digest);
    encode_digest32_field_words(out, claims.published_statement_digest);
}

fn decode_rv64im_nightstream_bridge_proof_binding_claims_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamBridgeProofBindingClaims, Rv64imBridgeError> {
    Ok(Rv64imNightstreamBridgeProofBindingClaims {
        proof_binding_root: decode_digest32_field_words(words, cursor, "bridge proof binding root")?,
        main_proof_digest: decode_digest32_field_words(words, cursor, "bridge proof binding main_proof_digest")?,
        side_proof_digest: decode_digest32_field_words(words, cursor, "bridge proof binding side_proof_digest")?,
        public_statement_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding public_statement_digest",
        )?,
        published_statement_digest: decode_digest32_field_words(
            words,
            cursor,
            "bridge proof binding published_statement_digest",
        )?,
    })
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

fn encode_nightstream_statement_fields(out: &mut Vec<BridgeFieldWord>, statement: &NightstreamStatement) {
    encode_digest32_field_words(out, statement.public_io_digest);
    encode_digest32_field_words(out, statement.verifier_context_digest);
    encode_fold_schedule_fields(out, statement.fold_schedule);
    out.push(statement.semantic_step_count);
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
    let proof_binding_root = decode_digest32_field_words(words, cursor, "statement proof_binding_root")?;
    Ok(NightstreamStatement {
        public_io_digest,
        verifier_context_digest,
        fold_schedule,
        semantic_step_count,
        proof_binding_root,
    })
}
