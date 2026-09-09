//! Exact active aggregate-acceptance leaf artifact and drift gate.
//!
//! Owns: nine role-normalized lowered rows, their production matrix bindings,
//! and the exact sparse-polynomial specialization for chunk acceptance.
//!
//! Does not own: singleton source rows, global fixture geometry, the fixed-F'
//! `ChunkBitOuterImage`/960-role census, decoded-LC inputs, the Lean proof,
//! inactive materialization, transcript authority, or enough-accepts logic.
//!
//! Emits constraints: no. It reads the production source and lowered CCS.
//!
//! Authority boundary: the singleton fixture is used only to extract a
//! role-normalized leaf. Physical placement remains a separate obligation.
//! Rows and polynomial terms are compared directly; no digest authorizes them.
//!
//! | Artifact branch | Mathematical obligation | Rust owner | Lean owner |
//! |---|---|---|---|
//! | `activeRows` | Seven bit-pair rows, one radix-3 aggregate, one root binding | `gadget_native::acceptance` | `AggregateAcceptanceRows` |
//! | `matrixBindings` / `polynomialTerms` | Arity-56 production CCS roles and coefficients are exact | `gadget_native::gates` | aggregate artifact refinement |
//! | Outer chunk-bit image | Fixed F' may substitute decoded linear combinations | `ChunkBitOuterImage` plus 960-role census | open, not this artifact |

#[path = "aggregate_acceptance_lean_artifact/lowered.rs"]
mod lowered;
#[path = "aggregate_acceptance_lean_artifact/mutations.rs"]
mod mutations;
#[path = "aggregate_acceptance_lean_artifact/render.rs"]
mod render;
#[path = "aggregate_acceptance_lean_artifact/source.rs"]
mod source;

use std::sync::OnceLock;

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_alphabet_sample_5_d;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{encode_r1cs_gadget_native, GadgetNativeCoordinateGateRoles};
use neo_math::F;
use p3_field::PrimeField64;

const APP: &[u8] = b"aggregate-acceptance-gadget-native-artifact";
const CHUNKS: usize = 64;
const SOURCE_ROWS_PER_CHUNK: usize = 4;
const SOURCE_COLUMNS_PER_CHUNK: usize = 2;
const SOURCE_INPUTS_PER_CHUNK: usize = 16;
const ACCEPTANCE_COORDINATES_PER_CHUNK: usize = 15;
const ACTIVE_ROWS_PER_CHUNK: usize = 9;
const GATE_ARITY: usize = GadgetNativeCoordinateGateRoles::ARITY;
const ARTIFACT_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/PiRlcChallenge/Generated/AggregateAcceptanceArtifactData.lean";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SourceRole {
    One,
    ChunkBit(usize),
    Accept,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CoordinateRole {
    One,
    ChunkBit(usize),
    Accept,
    TreeOutput(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MatrixRole {
    Selector,
    ProductLeft(usize),
    ProductRight(usize),
    ProductOut,
    QuadraticBitLeft,
    QuadraticBitRight,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoleTerm<Role> {
    role: Role,
    coefficient: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceRow {
    a: Vec<RoleTerm<SourceRole>>,
    b: Vec<RoleTerm<SourceRole>>,
    c: Vec<RoleTerm<SourceRole>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CanonicalInverseDecoder {
    output: SourceRole,
    difference: Vec<RoleTerm<SourceRole>>,
    owned_row_offsets: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatrixLinearCombination {
    role: MatrixRole,
    terms: Vec<RoleTerm<CoordinateRole>>,
}

type ActiveRow = Vec<MatrixLinearCombination>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatrixBinding {
    role: MatrixRole,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VariablePower {
    role: MatrixRole,
    power: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PolynomialTerm {
    coefficient: i128,
    powers: Vec<VariablePower>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChunkGeometry {
    source_row_start: usize,
    source_row_end: usize,
    source_column_start: usize,
    source_column_end: usize,
    source_input_columns: Vec<usize>,
    source_accept_column: usize,
    source_inverse_column: usize,
    encoded_input_columns: Vec<usize>,
    encoded_acceptance_columns: Vec<usize>,
    active_row_start: usize,
    active_row_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArtifactAudit {
    schema_version: usize,
    gate_arity: usize,
    matrix_bindings: Vec<MatrixBinding>,
    active_rows: Vec<ActiveRow>,
    polynomial_terms: Vec<PolynomialTerm>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactDrift {
    SchemaVersion,
    GateArity,
    MatrixBindings,
    ActiveRows,
    PolynomialTerms,
}

fn signed(coefficient: F) -> i128 {
    let canonical = coefficient.as_canonical_u64() as i128;
    let modulus = F::ORDER_U64 as i128;
    if canonical > modulus / 2 {
        canonical - modulus
    } else {
        canonical
    }
}

fn sampler_builder() -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.aggregate_acceptance_artifact");
    let mut transcript = TranscriptGadget::new(&mut builder, APP);
    let _symbols = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, 7);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied(), "production sampler source relation");
    builder
}

fn build_production_artifact() -> ArtifactAudit {
    let builder = sampler_builder();
    let source_snapshot = builder.snapshot();
    let trace = builder.encoding_trace();
    assert_eq!(trace.acceptance_chunks().len(), CHUNKS);
    let chunk_inputs = trace
        .acceptance_chunks()
        .iter()
        .flat_map(|event| event.chunk_bits)
        .map(Var::col)
        .collect::<Vec<_>>();
    let encoded =
        encode_r1cs_gadget_native(&source_snapshot, trace, &chunk_inputs).expect("exact aggregate-acceptance lowering");
    assert!(encoded.is_satisfied());
    assert_eq!(
        encoded.decode_source().expect("exact source decoder"),
        source_snapshot.witness()
    );

    let mut source_audit = source::extract(&source_snapshot, trace, &encoded);
    let lowered = lowered::extract(&encoded, trace, &mut source_audit.chunks);
    ArtifactAudit {
        schema_version: 2,
        gate_arity: encoded.structure.f.arity(),
        matrix_bindings: lowered.matrix_bindings,
        active_rows: lowered.active_rows,
        polynomial_terms: lowered.polynomial_terms,
    }
}

fn production_artifact() -> &'static ArtifactAudit {
    static ARTIFACT: OnceLock<ArtifactAudit> = OnceLock::new();
    ARTIFACT.get_or_init(build_production_artifact)
}

fn validate_artifact(candidate: &ArtifactAudit, production: &ArtifactAudit) -> Result<(), ArtifactDrift> {
    macro_rules! check {
        ($field:ident, $error:ident) => {
            if candidate.$field != production.$field {
                return Err(ArtifactDrift::$error);
            }
        };
    }
    check!(schema_version, SchemaVersion);
    check!(gate_arity, GateArity);
    check!(matrix_bindings, MatrixBindings);
    check!(active_rows, ActiveRows);
    check!(polynomial_terms, PolynomialTerms);
    Ok(())
}

#[test]
fn active_singleton_aggregate_acceptance_artifact_mutations_fail_closed() {
    mutations::assert_all_fail_closed(production_artifact());
}

#[test]
fn active_singleton_aggregate_acceptance_lean_artifact_matches_exact_production() {
    let artifact = production_artifact();
    validate_artifact(artifact, artifact).expect("production artifact self-consistency");
    assert_eq!(artifact.active_rows.len(), ACTIVE_ROWS_PER_CHUNK);
    assert_eq!(artifact.gate_arity, GATE_ARITY);

    let rendered = render::render(artifact);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
