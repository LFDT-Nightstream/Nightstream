//! Exact verifier-key relation payloads.
//!
//! Owns: deterministic JSON export of the complete verifier-owned CCS
//! structure and exact validation against a live [`Preprocessing`].
//!
//! Does not own: application selection, compiler semantics, deployment-key
//! selection, or cryptographic authority for a carried digest.

use std::io::Write;

use neo_ccs::{CcsMatrix, CscMat, SparsePoly};
use neo_math::{D, F};
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::lifecycle::Preprocessing;
use crate::paper::construction2::SemanticStateMode;
use crate::paper::relations::Structure;

pub const RELATION_ARTIFACT_FORMAT: &str = "nightstream/verifier-key-relation";
pub const RELATION_ARTIFACT_SCHEMA_VERSION: u32 = 1;

const MATRIX_PAYLOAD_ENCODING: &str = "rust-ccs-structure-serde-json-v1";
const PADDED_IDENTITY: &str = "implicit-[I_m;0]";
const PADDING_MAP: &str = "logical-prefix-then-zero";
const ASSIGNMENT_LAYOUT: &str = "z=x||w";
const PUBLIC_LAYOUT: &str = "x-is-assignment-prefix";

#[derive(Debug, Error)]
pub enum RelationArtifactError {
    #[error("relation artifact preprocessing is invalid: {0}")]
    Preprocessing(String),
    #[error("relation artifact shape is invalid: {0}")]
    Shape(&'static str),
    #[error("relation artifact profile is invalid: {0}")]
    Profile(String),
    #[error("cannot encode or decode relation artifact: {0}")]
    Json(#[from] serde_json::Error),
    #[error("relation artifact is not in the canonical JSON encoding")]
    NonCanonical,
    #[error("relation artifact differs from verifier-owned {0}")]
    Mismatch(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationArtifactReceipt {
    logical_rows: u64,
    assignment_fields: u64,
    padded_rows: u64,
    row_variables: u32,
    semantic_matrix_count: u32,
    joint_matrix_count: u32,
    polynomial_degree: u32,
    public_field_width: Option<u64>,
    structure_digest: [u64; 4],
    matrix_digest: [u64; 4],
    verifier_key_digest: [u8; 32],
}

impl RelationArtifactReceipt {
    pub fn logical_rows(&self) -> u64 {
        self.logical_rows
    }

    pub fn assignment_fields(&self) -> u64 {
        self.assignment_fields
    }

    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    pub fn row_variables(&self) -> u32 {
        self.row_variables
    }

    pub fn semantic_matrix_count(&self) -> u32 {
        self.semantic_matrix_count
    }

    pub fn joint_matrix_count(&self) -> u32 {
        self.joint_matrix_count
    }

    pub fn polynomial_degree(&self) -> u32 {
        self.polynomial_degree
    }

    pub fn public_field_width(&self) -> Option<u64> {
        self.public_field_width
    }

    pub fn structure_digest(&self) -> [u64; 4] {
        self.structure_digest
    }

    pub fn matrix_digest(&self) -> [u64; 4] {
        self.matrix_digest
    }

    pub fn verifier_key_digest(&self) -> [u8; 32] {
        self.verifier_key_digest
    }
}

/// Deterministic export and exact verifier-owned validation for one CCS key.
pub struct VerifierKeyRelationArtifact;

impl VerifierKeyRelationArtifact {
    /// Write one canonical generic CCS relation artifact.
    pub fn write(prep: &Preprocessing, writer: impl Write) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        write_with_source(prep, SourceWire::VerifierOwnedCcs, writer)
    }

    /// Return one canonical generic CCS relation artifact.
    pub fn to_json_vec(prep: &Preprocessing) -> Result<Vec<u8>, RelationArtifactError> {
        let mut bytes = Vec::new();
        Self::write(prep, &mut bytes)?;
        Ok(bytes)
    }

    /// Decode canonical bytes and compare every field with the live verifier
    /// key. The carried digests are checked as data, but exact structure
    /// equality is the authority.
    pub fn validate_json(prep: &Preprocessing, bytes: &[u8]) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        validate_with_source(prep, SourceWire::VerifierOwnedCcs, bytes)
    }

    /// Validate the live preprocessing and return its derived relation facts
    /// without serializing the matrix payload.
    pub fn receipt(prep: &Preprocessing) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        receipt_for(prep)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum SourceWire {
    VerifierOwnedCcs,
    R1csFPrimeFixedPoint {
        contract_id: String,
        profile_id: String,
        compiler_id: String,
        app: ApplicationWire,
        provenance_only_plan_digest: [u64; 4],
        fixed_point_rounds: Vec<FixedPointRoundWire>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ApplicationWire {
    pub rows: u64,
    pub assignment_fields: u64,
    pub public_fields: u64,
    pub provenance_only_structure_digest: [u64; 4],
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FixedPointRoundWire {
    pub input_rows: u64,
    pub input_assignment_fields: u64,
    pub output_rows: u64,
    pub output_assignment_fields: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct ParamsWire {
    q: u64,
    eta: u32,
    ring_degree: u32,
    kappa: u32,
    row_domain_bound: u64,
    norm_base: u32,
    decomposition_exponent: u32,
    norm_bound: u64,
    expansion_factor: u32,
    extension_degree: u32,
    effective_lambda: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct RelationWire {
    logical_rows: u64,
    assignment_fields: u64,
    padded_rows: u64,
    row_variables: u32,
    public_layout: Option<PublicLayoutWire>,
    semantic_matrix_count: u32,
    joint_matrix_count: u32,
    polynomial_degree: u32,
    padded_identity: String,
    padding_map: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct PublicLayoutWire {
    assignment_layout: String,
    kind: String,
    start_field: u64,
    field_count: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    structure_digest: [u64; 4],
    matrix_digest: [u64; 4],
    ajtai_public_parameters_digest: [u64; 4],
    verifier_key_digest: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct PolicyWire {
    stateful: bool,
    f_prime_recursive_link: bool,
    terminal_induction: bool,
    initial_semantic_state_digest: [u8; 32],
}

#[derive(Serialize)]
struct ArtifactRef<'a> {
    format: &'static str,
    schema: u32,
    matrix_payload_encoding: &'static str,
    source: &'a SourceWire,
    params: &'a ParamsWire,
    relation: &'a RelationWire,
    binding: &'a BindingWire,
    policy: &'a PolicyWire,
    structure: &'a Structure,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactOwned {
    format: String,
    schema: u32,
    matrix_payload_encoding: String,
    source: SourceWire,
    params: ParamsWire,
    relation: RelationWire,
    binding: BindingWire,
    policy: PolicyWire,
    structure: Structure,
}

struct ArtifactParts {
    params: ParamsWire,
    relation: RelationWire,
    binding: BindingWire,
    policy: PolicyWire,
}

pub(crate) fn write_with_source(
    prep: &Preprocessing,
    source: SourceWire,
    writer: impl Write,
) -> Result<RelationArtifactReceipt, RelationArtifactError> {
    let parts = artifact_parts(prep)?;
    let artifact = artifact_ref(prep, &source, &parts);
    serde_json::to_writer(writer, &artifact)?;
    Ok(receipt(&parts))
}

pub(crate) fn validate_with_source(
    prep: &Preprocessing,
    source: SourceWire,
    bytes: &[u8],
) -> Result<RelationArtifactReceipt, RelationArtifactError> {
    let decoded: ArtifactOwned = serde_json::from_slice(bytes)?;
    let mut canonical = CanonicalWriter::new(bytes);
    serde_json::to_writer(&mut canonical, &decoded)?;
    if !canonical.is_exact() {
        return Err(RelationArtifactError::NonCanonical);
    }

    let expected = artifact_parts(prep)?;
    if decoded.format != RELATION_ARTIFACT_FORMAT {
        return Err(RelationArtifactError::Mismatch("format"));
    }
    if decoded.schema != RELATION_ARTIFACT_SCHEMA_VERSION {
        return Err(RelationArtifactError::Mismatch("schema"));
    }
    if decoded.matrix_payload_encoding != MATRIX_PAYLOAD_ENCODING {
        return Err(RelationArtifactError::Mismatch("matrix payload encoding"));
    }
    if decoded.source != source {
        return Err(RelationArtifactError::Mismatch("compiler source"));
    }
    if decoded.params != expected.params {
        return Err(RelationArtifactError::Mismatch("parameters"));
    }
    if decoded.relation != expected.relation {
        return Err(RelationArtifactError::Mismatch("relation header"));
    }
    if decoded.binding != expected.binding {
        return Err(RelationArtifactError::Mismatch("key binding"));
    }
    if decoded.policy != expected.policy {
        return Err(RelationArtifactError::Mismatch("verification policy"));
    }
    decoded
        .structure
        .validate()
        .map_err(|error| RelationArtifactError::Preprocessing(error.to_string()))?;
    if !same_structure(&decoded.structure, prep.structure()) {
        return Err(RelationArtifactError::Mismatch("complete matrix payload"));
    }
    Ok(receipt(&expected))
}

pub(crate) fn receipt_for(prep: &Preprocessing) -> Result<RelationArtifactReceipt, RelationArtifactError> {
    artifact_parts(prep).map(|parts| receipt(&parts))
}

fn artifact_ref<'a>(prep: &'a Preprocessing, source: &'a SourceWire, parts: &'a ArtifactParts) -> ArtifactRef<'a> {
    ArtifactRef {
        format: RELATION_ARTIFACT_FORMAT,
        schema: RELATION_ARTIFACT_SCHEMA_VERSION,
        matrix_payload_encoding: MATRIX_PAYLOAD_ENCODING,
        source,
        params: &parts.params,
        relation: &parts.relation,
        binding: &parts.binding,
        policy: &parts.policy,
        structure: prep.structure(),
    }
}

fn artifact_parts(prep: &Preprocessing) -> Result<ArtifactParts, RelationArtifactError> {
    prep.validate_cached_structure()
        .map_err(|error| RelationArtifactError::Preprocessing(error.to_string()))?;
    prep.validate_verifier_key_binding()
        .map_err(|error| RelationArtifactError::Preprocessing(error.to_string()))?;
    let structure = prep.structure();
    structure
        .validate()
        .map_err(|error| RelationArtifactError::Preprocessing(error.to_string()))?;
    prep.params
        .validate_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .map_err(|error| RelationArtifactError::Preprocessing(error.to_string()))?;

    let logical_rows = u64::try_from(structure.n).map_err(|_| RelationArtifactError::Shape("row count"))?;
    let assignment_fields = u64::try_from(structure.m).map_err(|_| RelationArtifactError::Shape("assignment width"))?;
    let semantic_matrix_count =
        u32::try_from(structure.t()).map_err(|_| RelationArtifactError::Shape("matrix count"))?;
    let joint_matrix_count = semantic_matrix_count
        .checked_add(1)
        .ok_or(RelationArtifactError::Shape("joint matrix count"))?;
    let public_layout = prep
        .public_input_len
        .map(u64::try_from)
        .transpose()
        .map_err(|_| RelationArtifactError::Shape("public width"))?
        .map(|field_count| PublicLayoutWire {
            assignment_layout: ASSIGNMENT_LAYOUT.to_owned(),
            kind: PUBLIC_LAYOUT.to_owned(),
            start_field: 0,
            field_count,
        });
    let packed_assignment = structure
        .m
        .checked_add(D - 1)
        .and_then(|value| value.checked_div(D))
        .and_then(|value| value.checked_mul(D))
        .ok_or(RelationArtifactError::Shape("packed assignment width"))?;
    let padded_rows_native = structure
        .n
        .max(packed_assignment)
        .max(2)
        .checked_next_power_of_two()
        .ok_or(RelationArtifactError::Shape("padded row domain"))?;
    let padded_rows =
        u64::try_from(padded_rows_native).map_err(|_| RelationArtifactError::Shape("padded row domain"))?;

    Ok(ArtifactParts {
        params: ParamsWire {
            q: prep.params.q(),
            eta: prep.params.eta(),
            ring_degree: prep.params.d(),
            kappa: prep.params.kappa(),
            row_domain_bound: prep.params.m(),
            norm_base: prep.params.b(),
            decomposition_exponent: prep.params.k_rho(),
            norm_bound: prep.params.big_b(),
            expansion_factor: prep.params.T(),
            extension_degree: prep.params.extension_degree(),
            effective_lambda: prep.params.lambda(),
        },
        relation: RelationWire {
            logical_rows,
            assignment_fields,
            padded_rows,
            row_variables: padded_rows_native.trailing_zeros(),
            public_layout,
            semantic_matrix_count,
            joint_matrix_count,
            polynomial_degree: structure.max_degree(),
            padded_identity: PADDED_IDENTITY.to_owned(),
            padding_map: PADDING_MAP.to_owned(),
        },
        binding: BindingWire {
            structure_digest: field_digest_words(*prep.structure_digest()),
            matrix_digest: field_digest_words(prep.pi_ccs_header_bundle()),
            ajtai_public_parameters_digest: field_digest_words(prep.ajtai_pp_digest()),
            verifier_key_digest: prep.vk.digest(),
        },
        policy: PolicyWire {
            stateful: matches!(prep.semantic_state_mode(), SemanticStateMode::Stateful),
            f_prime_recursive_link: prep.enforces_f_prime_recursive_link(),
            terminal_induction: prep.enforces_terminal_induction(),
            initial_semantic_state_digest: prep.initial_semantic_state_digest(),
        },
    })
}

fn receipt(parts: &ArtifactParts) -> RelationArtifactReceipt {
    RelationArtifactReceipt {
        logical_rows: parts.relation.logical_rows,
        assignment_fields: parts.relation.assignment_fields,
        padded_rows: parts.relation.padded_rows,
        row_variables: parts.relation.row_variables,
        semantic_matrix_count: parts.relation.semantic_matrix_count,
        joint_matrix_count: parts.relation.joint_matrix_count,
        polynomial_degree: parts.relation.polynomial_degree,
        public_field_width: parts
            .relation
            .public_layout
            .as_ref()
            .map(|layout| layout.field_count),
        structure_digest: parts.binding.structure_digest,
        matrix_digest: parts.binding.matrix_digest,
        verifier_key_digest: parts.binding.verifier_key_digest,
    }
}

pub(crate) fn field_digest_words(digest: [F; 4]) -> [u64; 4] {
    digest.map(|field| field.as_canonical_u64())
}

pub(crate) fn same_structure(left: &Structure, right: &Structure) -> bool {
    left.n == right.n
        && left.m == right.m
        && same_polynomial(&left.f, &right.f)
        && left.matrices.len() == right.matrices.len()
        && left
            .matrices
            .iter()
            .zip(&right.matrices)
            .all(|(left, right)| same_matrix(left, right))
}

fn same_polynomial(left: &SparsePoly<F>, right: &SparsePoly<F>) -> bool {
    left.arity() == right.arity()
        && left.terms().len() == right.terms().len()
        && left
            .terms()
            .iter()
            .zip(right.terms())
            .all(|(left, right)| left.coeff == right.coeff && left.exps == right.exps)
}

fn same_matrix(left: &CcsMatrix<F>, right: &CcsMatrix<F>) -> bool {
    match (left, right) {
        (CcsMatrix::Identity { n: left }, CcsMatrix::Identity { n: right }) => left == right,
        (CcsMatrix::Csc(left), CcsMatrix::Csc(right)) => same_csc(left, right),
        (
            CcsMatrix::CscWithSeededPhi81 {
                csc: left_csc,
                blocks: left_blocks,
                geometric_runs: left_runs,
            },
            CcsMatrix::CscWithSeededPhi81 {
                csc: right_csc,
                blocks: right_blocks,
                geometric_runs: right_runs,
            },
        ) => same_csc(left_csc, right_csc) && left_blocks == right_blocks && left_runs == right_runs,
        _ => false,
    }
}

fn same_csc(left: &CscMat<F>, right: &CscMat<F>) -> bool {
    left.nrows == right.nrows
        && left.ncols == right.ncols
        && left.col_ptr == right.col_ptr
        && left.row_idx == right.row_idx
        && left.vals == right.vals
}

struct CanonicalWriter<'a> {
    expected: &'a [u8],
    position: usize,
    mismatch: bool,
}

impl<'a> CanonicalWriter<'a> {
    fn new(expected: &'a [u8]) -> Self {
        Self {
            expected,
            position: 0,
            mismatch: false,
        }
    }

    fn is_exact(&self) -> bool {
        !self.mismatch && self.position == self.expected.len()
    }
}

impl Write for CanonicalWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let end = self.position.saturating_add(bytes.len());
        if end > self.expected.len() || self.expected.get(self.position..end) != Some(bytes) {
            self.mismatch = true;
        }
        self.position = end;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
