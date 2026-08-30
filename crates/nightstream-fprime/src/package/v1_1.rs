//! Exact SuperNeo v1.1 PiCCS output boundary in the Lean-emitted layout.
//!
//! This module reads package-owned segments. It does not define PiCCS
//! semantics, choose offsets, or merge the Pad and CCS evaluation families.

use super::{canonical_field, LoadedPackage, PackageError, Segment};
use crate::identity::{pi_ccs_v1_1_verifier_context, PiCcsV1_1VerifierContext};

const PRIOR_PREIMAGE_ROLE: u64 = 1;
const OUTPUT_PREIMAGE_ROLE: u64 = 2;
const WITNESS_ROLE: u64 = 3;
const FRESH_COMMITMENT_ROLE: u64 = 6;
const ROUND_MESSAGES_ROLE: u64 = 7;
const OUTPUT_EVAL_K_ROLE: u64 = 8;
const OUTPUT_EVAL_A_ROLE: u64 = 9;
const VERIFIER_CONTEXT_ROLE: u64 = 10;
const PI_DEC_COMMITMENTS_ROLE: u64 = 11;
const PI_DEC_EVAL_K_ROLE: u64 = 12;
const PI_DEC_EVAL_A_ROLE: u64 = 13;
const PI_DEC_CHILD_PUBLIC_INPUT_ROLE: u64 = 14;
const PI_DEC_WITNESS_ROLE: u64 = 15;
const RUNNING_TRANSITION_WITNESS_ROLE: u64 = 16;

pub const PI_CCS_V1_1_SOURCE_COUNT: usize = 17;
pub const PI_CCS_V1_1_COEFFICIENT_COUNT: usize = 54;
pub const PI_CCS_V1_1_MATRIX_COUNT: usize = 14;
pub const PI_CCS_V1_1_ROUND_COUNT: usize = 28;
pub const PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT: usize = 10;
pub const PI_CCS_V1_1_STATE_PREIMAGE_WORDS: usize = 45_937;
pub const PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS: usize = 270;
pub const PI_CCS_V1_1_FRESH_COMMITMENT_WORDS: usize = 972;
pub const PI_CCS_V1_1_VERIFIER_CONTEXT_WORDS: usize = 4;
pub const PI_DEC_V1_1_CHILD_COUNT: usize = 16;
pub const PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD: usize = 972;
pub const PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD: usize = PI_CCS_V1_1_COEFFICIENT_COUNT;
pub const PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD: usize = PI_CCS_V1_1_MATRIX_COUNT;
pub const PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD: usize = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS;

const EXTENSION_WORDS: usize = 2;
const ROUND_MESSAGE_WORDS: usize = PI_CCS_V1_1_ROUND_COUNT * PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT * EXTENSION_WORDS;
const EVAL_K_WORDS: usize = PI_CCS_V1_1_COEFFICIENT_COUNT * EXTENSION_WORDS;
const EVAL_A_WORDS: usize = PI_CCS_V1_1_MATRIX_COUNT * PI_CCS_V1_1_COEFFICIENT_COUNT * EXTENSION_WORDS;
const PI_DEC_COMMITMENT_WORDS: usize = PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD;
const PI_DEC_EVAL_K_WORDS: usize = PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD * EXTENSION_WORDS;
const PI_DEC_EVAL_A_WORDS: usize =
    PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD * PI_CCS_V1_1_COEFFICIENT_COUNT * EXTENSION_WORDS;
const PI_DEC_CHILD_PUBLIC_INPUT_WORDS: usize = PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD;
const PI_DEC_WITNESS_WORDS: usize = 18_090;
const RUNNING_TRANSITION_WITNESS_WORDS: usize = 275_402;

pub(super) fn private_segment_roles() -> Vec<u64> {
    let mut roles = Vec::with_capacity(10 + 2 * PI_CCS_V1_1_SOURCE_COUNT);
    roles.extend([
        PRIOR_PREIMAGE_ROLE,
        OUTPUT_PREIMAGE_ROLE,
        FRESH_COMMITMENT_ROLE,
        ROUND_MESSAGES_ROLE,
    ]);
    for _ in 0..PI_CCS_V1_1_SOURCE_COUNT {
        roles.extend([OUTPUT_EVAL_K_ROLE, OUTPUT_EVAL_A_ROLE]);
    }
    roles.push(WITNESS_ROLE);
    roles.extend([
        PI_DEC_COMMITMENTS_ROLE,
        PI_DEC_EVAL_K_ROLE,
        PI_DEC_EVAL_A_ROLE,
        PI_DEC_CHILD_PUBLIC_INPUT_ROLE,
        PI_DEC_WITNESS_ROLE,
        RUNNING_TRANSITION_WITNESS_ROLE,
    ]);
    roles
}

pub(super) fn is_witness_role(role: u64) -> bool {
    role == WITNESS_ROLE || role == PI_DEC_WITNESS_ROLE || role == RUNNING_TRANSITION_WITNESS_ROLE
}

pub(super) fn validate_private_segments(segments: &[Segment]) -> Result<(), PackageError> {
    if segments[0].length != PI_CCS_V1_1_STATE_PREIMAGE_WORDS
        || segments[1].length != PI_CCS_V1_1_STATE_PREIMAGE_WORDS
        || segments[2].length != PI_CCS_V1_1_FRESH_COMMITMENT_WORDS
        || segments[3].length != ROUND_MESSAGE_WORDS
    {
        return Err(PackageError::Invalid("PiCCS v1_1 input segments"));
    }
    let output = &segments[4..4 + 2 * PI_CCS_V1_1_SOURCE_COUNT];
    for pair in output.chunks_exact(2) {
        if pair[0].length != EVAL_K_WORDS || pair[1].length != EVAL_A_WORDS {
            return Err(PackageError::Invalid("PiCCS v1_1 output segments"));
        }
    }
    let suffix = &segments[4 + 2 * PI_CCS_V1_1_SOURCE_COUNT..];
    if suffix[1].length != PI_DEC_COMMITMENT_WORDS
        || suffix[2].length != PI_DEC_EVAL_K_WORDS
        || suffix[3].length != PI_DEC_EVAL_A_WORDS
        || suffix[4].length != PI_DEC_CHILD_PUBLIC_INPUT_WORDS
        || suffix[5].length != PI_DEC_WITNESS_WORDS
        || suffix[6].length != RUNNING_TRANSITION_WITNESS_WORDS
    {
        return Err(PackageError::Invalid("PiDEC v1_1 private segments"));
    }
    Ok(())
}

pub(super) fn validate_public_segments(segments: &[Segment]) -> Result<(), PackageError> {
    if segments[0].length != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
        || segments[1].length != 4
        || segments[2].role != VERIFIER_CONTEXT_ROLE
        || segments[2].length != PI_CCS_V1_1_VERIFIER_CONTEXT_WORDS
    {
        return Err(PackageError::Invalid("PiCCS v1_1 public segments"));
    }
    Ok(())
}

/// Exact loaded PiCCS v1_1 output message.
///
/// `eval_k[source][coefficient]` is the Pad family. The separate
/// `eval_a[source][matrix][coefficient]` value is the CCS-matrix family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1OutputEvaluations {
    eval_k: Vec<Vec<[u64; EXTENSION_WORDS]>>,
    eval_a: Vec<Vec<Vec<[u64; EXTENSION_WORDS]>>>,
}

impl PiCcsV1_1OutputEvaluations {
    pub fn new(
        eval_k: Vec<Vec<[u64; EXTENSION_WORDS]>>,
        eval_a: Vec<Vec<Vec<[u64; EXTENSION_WORDS]>>>,
    ) -> Result<Self, PackageError> {
        if eval_k.len() != PI_CCS_V1_1_SOURCE_COUNT || eval_a.len() != PI_CCS_V1_1_SOURCE_COUNT {
            return Err(PackageError::Invalid("PiCCS v1_1 output source count"));
        }
        for source in 0..PI_CCS_V1_1_SOURCE_COUNT {
            if eval_k[source].len() != PI_CCS_V1_1_COEFFICIENT_COUNT
                || eval_a[source].len() != PI_CCS_V1_1_MATRIX_COUNT
                || eval_a[source]
                    .iter()
                    .any(|matrix| matrix.len() != PI_CCS_V1_1_COEFFICIENT_COUNT)
            {
                return Err(PackageError::Invalid("PiCCS v1_1 output family shape"));
            }
            for value in &eval_k[source] {
                validate_extension(*value, "PiCCS v1_1 Eval_K")?;
            }
            for matrix in &eval_a[source] {
                for value in matrix {
                    validate_extension(*value, "PiCCS v1_1 Eval_A")?;
                }
            }
        }
        Ok(Self { eval_k, eval_a })
    }

    pub fn eval_k(&self) -> &[Vec<[u64; EXTENSION_WORDS]>] {
        &self.eval_k
    }

    pub fn eval_a(&self) -> &[Vec<Vec<[u64; EXTENSION_WORDS]>>] {
        &self.eval_a
    }
}

/// Caller-owned fields for the current Stage 1 PiCCS package prefix.
///
/// The package owns the physical offsets. This value keeps only the semantic
/// segment order that Lean emits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1PackageInputs {
    prior_preimage: Vec<u64>,
    output_preimage: Vec<u64>,
    fresh_commitment: Vec<u64>,
    round_messages: Vec<Vec<[u64; EXTENSION_WORDS]>>,
    output_evaluations: PiCcsV1_1OutputEvaluations,
    prior_public_input: Vec<u64>,
    output_digest: [u64; 4],
    verifier_context: PiCcsV1_1VerifierContext,
}

impl PiCcsV1_1PackageInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prior_preimage: Vec<u64>,
        output_preimage: Vec<u64>,
        fresh_commitment: Vec<u64>,
        round_messages: Vec<Vec<[u64; EXTENSION_WORDS]>>,
        output_evaluations: PiCcsV1_1OutputEvaluations,
        prior_public_input: Vec<u64>,
        output_digest: [u64; 4],
        verifier_context: PiCcsV1_1VerifierContext,
    ) -> Result<Self, PackageError> {
        if prior_preimage.len() != PI_CCS_V1_1_STATE_PREIMAGE_WORDS
            || output_preimage.len() != PI_CCS_V1_1_STATE_PREIMAGE_WORDS
        {
            return Err(PackageError::Invalid("PiCCS v1_1 state preimage shape"));
        }
        if fresh_commitment.len() != PI_CCS_V1_1_FRESH_COMMITMENT_WORDS {
            return Err(PackageError::Invalid("PiCCS v1_1 fresh commitment shape"));
        }
        if round_messages.len() != PI_CCS_V1_1_ROUND_COUNT
            || round_messages
                .iter()
                .any(|round| round.len() != PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
        {
            return Err(PackageError::Invalid("PiCCS v1_1 round-message shape"));
        }
        if prior_public_input.len() != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS {
            return Err(PackageError::Invalid("PiCCS v1_1 prior public-input shape"));
        }
        validate_words(&prior_preimage, "PiCCS v1_1 prior preimage")?;
        validate_words(&output_preimage, "PiCCS v1_1 output preimage")?;
        validate_words(&fresh_commitment, "PiCCS v1_1 fresh commitment")?;
        for round in &round_messages {
            for value in round {
                validate_extension(*value, "PiCCS v1_1 round message")?;
            }
        }
        validate_words(&prior_public_input, "PiCCS v1_1 prior public input")?;
        validate_words(&output_digest, "PiCCS v1_1 output digest")?;
        validate_words(&verifier_context.digest(), "PiCCS v1_1 verifier context")?;
        Ok(Self {
            prior_preimage,
            output_preimage,
            fresh_commitment,
            round_messages,
            output_evaluations,
            prior_public_input,
            output_digest,
            verifier_context,
        })
    }
}

/// Caller-owned PiDEC v1.1 messages and verifier-computed child public inputs.
///
/// The four fields follow the exact segment order emitted by Lean. Eval_K and
/// the 14-matrix Eval_A family remain separate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecV1_1PackageInputs {
    child_commitments: Vec<Vec<u64>>,
    child_eval_k: Vec<Vec<[u64; EXTENSION_WORDS]>>,
    child_eval_a: Vec<Vec<Vec<[u64; EXTENSION_WORDS]>>>,
    child_public_inputs: Vec<Vec<u64>>,
}

impl PiDecV1_1PackageInputs {
    pub fn new(
        child_commitments: Vec<Vec<u64>>,
        child_eval_k: Vec<Vec<[u64; EXTENSION_WORDS]>>,
        child_eval_a: Vec<Vec<Vec<[u64; EXTENSION_WORDS]>>>,
        child_public_inputs: Vec<Vec<u64>>,
    ) -> Result<Self, PackageError> {
        if child_commitments.len() != PI_DEC_V1_1_CHILD_COUNT
            || child_eval_k.len() != PI_DEC_V1_1_CHILD_COUNT
            || child_eval_a.len() != PI_DEC_V1_1_CHILD_COUNT
            || child_public_inputs.len() != PI_DEC_V1_1_CHILD_COUNT
        {
            return Err(PackageError::Invalid("PiDEC v1_1 child count"));
        }
        for child in 0..PI_DEC_V1_1_CHILD_COUNT {
            if child_commitments[child].len() != PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD
                || child_eval_k[child].len() != PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD
                || child_eval_a[child].len() != PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD
                || child_eval_a[child]
                    .iter()
                    .any(|matrix| matrix.len() != PI_CCS_V1_1_COEFFICIENT_COUNT)
                || child_public_inputs[child].len() != PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD
            {
                return Err(PackageError::Invalid("PiDEC v1_1 child shape"));
            }
            validate_words(&child_commitments[child], "PiDEC v1_1 child commitment")?;
            for value in &child_eval_k[child] {
                validate_extension(*value, "PiDEC v1_1 child Eval_K")?;
            }
            for matrix in &child_eval_a[child] {
                for value in matrix {
                    validate_extension(*value, "PiDEC v1_1 child Eval_A")?;
                }
            }
            validate_words(&child_public_inputs[child], "PiDEC v1_1 child public input")?;
        }
        Ok(Self {
            child_commitments,
            child_eval_k,
            child_eval_a,
            child_public_inputs,
        })
    }

    pub fn child_commitments(&self) -> &[Vec<u64>] {
        &self.child_commitments
    }

    pub fn child_eval_k(&self) -> &[Vec<[u64; EXTENSION_WORDS]>] {
        &self.child_eval_k
    }

    pub fn child_eval_a(&self) -> &[Vec<Vec<[u64; EXTENSION_WORDS]>>] {
        &self.child_eval_a
    }

    pub fn child_public_inputs(&self) -> &[Vec<u64>] {
        &self.child_public_inputs
    }

    fn append_private_values(&self, values: &mut Vec<u64>) {
        for commitment in &self.child_commitments {
            values.extend_from_slice(commitment);
        }
        for eval_k in &self.child_eval_k {
            for value in eval_k {
                values.extend_from_slice(value);
            }
        }
        for eval_a in &self.child_eval_a {
            for matrix in eval_a {
                for value in matrix {
                    values.extend_from_slice(value);
                }
            }
        }
        for public_input in &self.child_public_inputs {
            values.extend_from_slice(public_input);
        }
    }
}

/// Flat values accepted by [`LoadedPackage::execute_witness`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1EncodedInputs {
    private_values: Vec<u64>,
    public_values: Vec<u64>,
}

impl PiCcsV1_1EncodedInputs {
    pub fn private_values(&self) -> &[u64] {
        &self.private_values
    }

    pub fn public_values(&self) -> &[u64] {
        &self.public_values
    }
}

impl LoadedPackage {
    /// Derive the only context value accepted by this loaded package. The
    /// caller supplies canonical commitment-setup words, never a digest.
    pub fn derive_pi_ccs_v1_1_verifier_context(
        &self,
        commitment_key_words: &[u64],
    ) -> Result<PiCcsV1_1VerifierContext, PackageError> {
        pi_ccs_v1_1_verifier_context(self.relation_identifier, commitment_key_words)
    }

    /// Encode typed v1_1 values in the exact segment order owned by this
    /// verifier-checked package.
    pub fn encode_pi_ccs_v1_1_inputs(
        &self,
        inputs: &PiCcsV1_1PackageInputs,
    ) -> Result<PiCcsV1_1EncodedInputs, PackageError> {
        if inputs.verifier_context.package_identity() != self.relation_identifier {
            return Err(PackageError::Invalid("PiCCS v1_1 verifier-context package identity"));
        }
        let pi_ccs_input_count = self
            .layout
            .private_segments
            .iter()
            .find(|segment| segment.role == WITNESS_ROLE)
            .ok_or(PackageError::Invalid("witness segment"))?
            .start;
        let mut private_values = Vec::with_capacity(pi_ccs_input_count);
        private_values.extend_from_slice(&inputs.prior_preimage);
        private_values.extend_from_slice(&inputs.output_preimage);
        private_values.extend_from_slice(&inputs.fresh_commitment);
        for round in &inputs.round_messages {
            for value in round {
                private_values.extend_from_slice(value);
            }
        }
        for source in 0..PI_CCS_V1_1_SOURCE_COUNT {
            for value in &inputs.output_evaluations.eval_k[source] {
                private_values.extend_from_slice(value);
            }
            for matrix in &inputs.output_evaluations.eval_a[source] {
                for value in matrix {
                    private_values.extend_from_slice(value);
                }
            }
        }
        if private_values.len() != pi_ccs_input_count {
            return Err(PackageError::Invalid("PiCCS v1_1 encoded private-input length"));
        }

        let mut public_values = Vec::with_capacity(self.layout.public_column_count);
        public_values.extend_from_slice(&inputs.prior_public_input);
        public_values.extend_from_slice(&inputs.output_digest);
        public_values.extend_from_slice(&inputs.verifier_context.digest());
        if public_values.len() != self.layout.public_column_count {
            return Err(PackageError::Invalid("PiCCS v1_1 encoded public-input length"));
        }
        Ok(PiCcsV1_1EncodedInputs {
            private_values,
            public_values,
        })
    }

    /// Encode the complete current Stage 1 caller input, including PiDEC.
    pub fn encode_stage1_v1_1_inputs(
        &self,
        pi_ccs: &PiCcsV1_1PackageInputs,
        pi_dec: &PiDecV1_1PackageInputs,
    ) -> Result<PiCcsV1_1EncodedInputs, PackageError> {
        let encoded = self.encode_pi_ccs_v1_1_inputs(pi_ccs)?;
        let mut private_values = encoded.private_values;
        private_values.reserve(
            PI_DEC_COMMITMENT_WORDS + PI_DEC_EVAL_K_WORDS + PI_DEC_EVAL_A_WORDS + PI_DEC_CHILD_PUBLIC_INPUT_WORDS,
        );
        pi_dec.append_private_values(&mut private_values);
        if private_values.len() != self.private_input_count() {
            return Err(PackageError::Invalid("Stage 1 v1_1 encoded private-input length"));
        }
        Ok(PiCcsV1_1EncodedInputs {
            private_values,
            public_values: encoded.public_values,
        })
    }

    /// Decode the caller-owned PiCCS output words according to the exact
    /// segment order emitted by Lean.
    pub fn pi_ccs_v1_1_output_evaluations(
        &self,
        private_inputs: &[u64],
    ) -> Result<PiCcsV1_1OutputEvaluations, PackageError> {
        let witness_start = self
            .layout
            .private_segments
            .iter()
            .find(|segment| segment.role == WITNESS_ROLE)
            .ok_or(PackageError::Invalid("witness segment"))?
            .start;
        if private_inputs.len() != witness_start {
            return Err(PackageError::Invalid("private input length"));
        }

        let mut eval_k = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);
        let mut eval_a = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);
        let output = &self.layout.private_segments[4..4 + 2 * PI_CCS_V1_1_SOURCE_COUNT];
        for pair in output.chunks_exact(2) {
            let k_words = segment_words(private_inputs, pair[0])?;
            eval_k.push(
                k_words
                    .chunks_exact(EXTENSION_WORDS)
                    .map(|value| [value[0], value[1]])
                    .collect(),
            );

            let a_words = segment_words(private_inputs, pair[1])?;
            let matrices = a_words
                .chunks_exact(PI_CCS_V1_1_COEFFICIENT_COUNT * EXTENSION_WORDS)
                .map(|matrix| {
                    matrix
                        .chunks_exact(EXTENSION_WORDS)
                        .map(|value| [value[0], value[1]])
                        .collect()
                })
                .collect();
            eval_a.push(matrices);
        }
        PiCcsV1_1OutputEvaluations::new(eval_k, eval_a)
    }
}

fn validate_extension(value: [u64; EXTENSION_WORDS], location: &'static str) -> Result<(), PackageError> {
    validate_words(&value, location)
}

fn validate_words(values: &[u64], location: &'static str) -> Result<(), PackageError> {
    for value in values {
        canonical_field(*value, location)?;
    }
    Ok(())
}

fn segment_words(private_inputs: &[u64], segment: Segment) -> Result<&[u64], PackageError> {
    let end = segment
        .start
        .checked_add(segment.length)
        .ok_or(PackageError::Invalid("PiCCS v1_1 segment end"))?;
    let words = private_inputs
        .get(segment.start..end)
        .ok_or(PackageError::Invalid("PiCCS v1_1 segment range"))?;
    for word in words {
        canonical_field(*word, "PiCCS v1_1 output")?;
    }
    Ok(words)
}
