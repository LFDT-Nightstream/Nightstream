//! Fail-closed consumer for the Lean-owned canonical F′ manifest.
//!
//! Owns: schema-v1 decoding and structural validation before Rust may use a
//! manifest. Validation recomputes the complete program costs, sparse
//! support, receipt ownership, scope, application-cost split, and fixed
//! Step/Terminal ABI from the serialized data.
//!
//! Does not own: manifest generation, application selection, witness
//! generation, or a claim that the current Rust circuit equals the manifest.
//!
//! Emits constraints: yes, but only after successful validation. Each
//! manifest row is appended unchanged under the returned Lean-column to
//! Rust-column map.

use std::collections::{HashMap, HashSet};

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use thiserror::Error;

use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

pub const LEAN_CANONICAL_MANIFEST_SCHEMA_VERSION: u64 = 1;
pub const LEAN_CANONICAL_MANIFEST_FORMAT: &str = "nightstream/fprime-canonical-manifest";
pub const GOLDILOCKS_MODULUS: u64 = 18_446_744_069_414_584_321;

#[derive(Debug, Error)]
pub enum LeanManifestError {
    #[error("cannot decode Lean canonical manifest: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("unsupported Lean canonical manifest schema {found}; expected {expected}")]
    UnsupportedSchema { found: u64, expected: u64 },
    #[error("invalid Lean canonical manifest at {path}: {detail}")]
    Invalid { path: String, detail: String },
}

#[derive(Debug, Error)]
pub enum LeanManifestEmissionError {
    #[error("Lean canonical manifest requires a fresh R1CS builder; got {rows} rows and {columns} columns")]
    BuilderNotFresh { rows: usize, columns: usize },
    #[error("missing witness value for Lean manifest column {column:?}")]
    MissingValue { column: ColumnId },
    #[error("validated Lean manifest row refers to an unknown column {column:?}")]
    UnknownColumn { column: ColumnId },
}

pub(super) fn invalid(path: impl Into<String>, detail: impl Into<String>) -> LeanManifestError {
    LeanManifestError::Invalid {
        path: path.into(),
        detail: detail.into(),
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OwnerPathStep {
    Rest,
    TrueArm,
    FalseArm,
    Continuation,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedOwner {
    Input { slot: usize },
    Instruction { path: Vec<OwnerPathStep> },
    Branch { path: Vec<OwnerPathStep> },
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhysicalOwner {
    Prelude,
    Typed {
        owner: TypedOwner,
    },
    BranchActivation {
        path: Vec<OwnerPathStep>,
        selected: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ColumnId {
    pub owner: PhysicalOwner,
    pub bundle_index: usize,
    pub coordinate_index: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RowId {
    pub owner: PhysicalOwner,
    pub ordinal: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    Committed,
    Public,
    Auxiliary,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OwnedColumn {
    pub id: ColumnId,
    pub ownership: Ownership,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestTerm {
    pub column: ColumnId,
    pub coefficient: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestRow {
    pub id: RowId,
    pub a: Vec<ManifestTerm>,
    pub b: Vec<ManifestTerm>,
    pub c: Vec<ManifestTerm>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InstructionKind {
    Prelude,
    Input,
    Literal,
    Affine,
    Product,
    Bit,
    Call,
    Assertion,
    BranchControl,
    BranchJoin,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestReceipt {
    pub owner: PhysicalOwner,
    pub kind: InstructionKind,
    pub allocations: Vec<OwnedColumn>,
    pub rows: Vec<ManifestRow>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestProgram {
    pub one: ColumnId,
    pub receipts: Vec<ManifestReceipt>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestCost {
    pub(super) recurring_rows: usize,
    pub(super) committed_columns: usize,
    pub(super) public_columns: usize,
    pub(super) auxiliary_columns: usize,
}

impl ManifestCost {
    fn add(self, other: Self, path: &str) -> Result<Self, LeanManifestError> {
        let add = |left: usize, right: usize, field: &str| {
            left.checked_add(right)
                .ok_or_else(|| invalid(format!("{path}.{field}"), "count overflow"))
        };
        Ok(Self {
            recurring_rows: add(self.recurring_rows, other.recurring_rows, "recurring_rows")?,
            committed_columns: add(self.committed_columns, other.committed_columns, "committed_columns")?,
            public_columns: add(self.public_columns, other.public_columns, "public_columns")?,
            auxiliary_columns: add(self.auxiliary_columns, other.auxiliary_columns, "auxiliary_columns")?,
        })
    }

    pub fn recurring_rows(self) -> usize {
        self.recurring_rows
    }

    pub fn committed_columns(self) -> usize {
        self.committed_columns
    }

    pub fn public_columns(self) -> usize {
        self.public_columns
    }

    pub fn auxiliary_columns(self) -> usize {
        self.auxiliary_columns
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManifestStatistics {
    a_nonzeros: usize,
    b_nonzeros: usize,
    c_nonzeros: usize,
    max_row_support: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
enum ProfileName {
    #[serde(rename = "fixed_one_plain_270")]
    FixedOnePlain270,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProfileIdentifier {
    name: ProfileName,
    matrix_count: usize,
    fresh_source_count: usize,
    running_source_count: usize,
    public_carrier_width: usize,
    fresh_legacy_width: usize,
    fresh_completion_width: usize,
    running_carrier_width: usize,
    poseidon_width: usize,
    poseidon_rate: usize,
    poseidon_capacity: usize,
    poseidon_digest_width: usize,
    binding_preimage_width: usize,
    decomposition_base: usize,
    decomposition_children: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Widths {
    iteration: usize,
    state: usize,
    witness: usize,
    running: usize,
    fresh: usize,
    nifs_proof: usize,
    digest: usize,
    encoded: usize,
    running_witness: usize,
    fresh_witness: usize,
    bit: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum SegmentRole {
    Iteration,
    InitialState,
    CurrentState,
    Running,
    Fresh,
    Witness,
    NifsProof,
    NextState,
    NextRunning,
    Digest,
    RunningWitness,
    FreshWitness,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CodecSegment {
    role: SegmentRole,
    width: usize,
    ownership: Ownership,
    offset: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema: u64,
    format: String,
    goldilocks_modulus: u64,
    profile: ProfileIdentifier,
    widths: Widths,
    step_input: Vec<CodecSegment>,
    step_result: Vec<CodecSegment>,
    terminal_input: Vec<CodecSegment>,
    step_program: ManifestProgram,
    terminal_program: ManifestProgram,
    step_result_columns: Vec<OwnedColumn>,
    step_selector: ColumnId,
    terminal_selector: ColumnId,
    step_activations: Vec<ColumnId>,
    terminal_activations: Vec<ColumnId>,
    step_cost: ManifestCost,
    terminal_cost: ManifestCost,
    fixed_protocol_cost: ManifestCost,
    application_step_cost: ManifestCost,
    step_statistics: ManifestStatistics,
    terminal_statistics: ManifestStatistics,
}

#[derive(Debug)]
pub(super) struct ProgramSummary {
    pub(super) cost: ManifestCost,
    pub(super) statistics: ManifestStatistics,
    pub(super) columns: HashMap<ColumnId, Ownership>,
}

/// A canonical manifest that passed all schema and structural checks.
///
/// This type does not implement `Deserialize`. Call
/// [`Self::from_json_slice`] so callers cannot bypass validation.
///
/// Structural validation does not prove that Lean produced the source bytes.
/// A production verifier must pin or regenerate the selected Lean export and
/// compare it exactly. It must never accept operator-selected or
/// prover-selected manifest bytes as the verifier relation.
#[derive(Clone, Debug)]
pub struct LeanCanonicalManifest {
    wire: ManifestWire,
}

/// Physical Rust variables allocated for one validated Lean program.
#[derive(Clone, Debug)]
pub struct ManifestEmission {
    variables: HashMap<ColumnId, Var>,
    public_columns: Vec<ColumnId>,
    committed_columns: Vec<ColumnId>,
    auxiliary_columns: Vec<ColumnId>,
}

impl ManifestEmission {
    pub fn variable(&self, column: &ColumnId) -> Option<Var> {
        self.variables.get(column).copied()
    }

    /// Public R1CS columns in physical order. The constant-one column is
    /// always first.
    pub fn public_columns(&self) -> &[ColumnId] {
        &self.public_columns
    }

    pub fn committed_columns(&self) -> &[ColumnId] {
        &self.committed_columns
    }

    pub fn auxiliary_columns(&self) -> &[ColumnId] {
        &self.auxiliary_columns
    }

    pub fn public_input_len(&self) -> usize {
        self.public_columns.len()
    }
}

impl LeanCanonicalManifest {
    /// Decode and validate one complete Lean-owned manifest.
    ///
    /// Unknown fields, unsupported schema versions, noncanonical
    /// coefficients, missing allocations, ownership conflicts, count drift,
    /// and statistics drift all reject.
    ///
    /// This function validates structure only. It does not establish Lean
    /// provenance for `bytes`.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, LeanManifestError> {
        let wire: ManifestWire = serde_json::from_slice(bytes)?;
        wire.validate()?;
        Ok(Self { wire })
    }

    pub fn matrix_count(&self) -> usize {
        self.wire.profile.matrix_count
    }

    pub fn step_cost(&self) -> ManifestCost {
        self.wire.step_cost
    }

    pub fn terminal_cost(&self) -> ManifestCost {
        self.wire.terminal_cost
    }

    pub fn step_program(&self) -> &ManifestProgram {
        &self.wire.step_program
    }

    pub fn terminal_program(&self) -> &ManifestProgram {
        &self.wire.terminal_program
    }

    pub fn step_result_columns(&self) -> &[OwnedColumn] {
        &self.wire.step_result_columns
    }

    /// Emit the exact validated Lean Step rows into a fresh Rust builder.
    ///
    /// The callback supplies every nonconstant witness coordinate. Rust
    /// groups public columns first, then committed and auxiliary columns,
    /// while row terms retain their exact Lean identities and coefficients.
    pub fn emit_step(
        &self,
        builder: &mut R1csBuilder,
        values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<ManifestEmission, LeanManifestEmissionError> {
        emit_program(builder, &self.wire.step_program, values)
    }

    /// Emit the exact validated Lean Terminal rows into a fresh Rust builder.
    pub fn emit_terminal(
        &self,
        builder: &mut R1csBuilder,
        values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<ManifestEmission, LeanManifestEmissionError> {
        emit_program(builder, &self.wire.terminal_program, values)
    }
}

impl ManifestWire {
    fn validate(&self) -> Result<(), LeanManifestError> {
        if self.schema != LEAN_CANONICAL_MANIFEST_SCHEMA_VERSION {
            return Err(LeanManifestError::UnsupportedSchema {
                found: self.schema,
                expected: LEAN_CANONICAL_MANIFEST_SCHEMA_VERSION,
            });
        }
        if self.format != LEAN_CANONICAL_MANIFEST_FORMAT {
            return Err(invalid("format", format!("unsupported format {:?}", self.format)));
        }
        if self.goldilocks_modulus != GOLDILOCKS_MODULUS {
            return Err(invalid(
                "goldilocks_modulus",
                format!("got {}, expected {GOLDILOCKS_MODULUS}", self.goldilocks_modulus),
            ));
        }
        self.validate_profile()?;
        validate_segments(
            "step_input",
            &self.step_input,
            &[
                (SegmentRole::Iteration, self.widths.iteration, Ownership::Committed),
                (SegmentRole::InitialState, self.widths.state, Ownership::Committed),
                (SegmentRole::CurrentState, self.widths.state, Ownership::Committed),
                (SegmentRole::Running, self.widths.running, Ownership::Committed),
                (SegmentRole::Fresh, self.widths.fresh, Ownership::Committed),
                (SegmentRole::Witness, self.widths.witness, Ownership::Committed),
                (SegmentRole::NifsProof, self.widths.nifs_proof, Ownership::Committed),
            ],
        )?;
        validate_segments(
            "step_result",
            &self.step_result,
            &[
                (SegmentRole::NextState, self.widths.state, Ownership::Committed),
                (SegmentRole::NextRunning, self.widths.running, Ownership::Committed),
                (SegmentRole::Digest, self.widths.digest, Ownership::Public),
            ],
        )?;
        validate_segments(
            "terminal_input",
            &self.terminal_input,
            &[
                (SegmentRole::Iteration, self.widths.iteration, Ownership::Public),
                (SegmentRole::InitialState, self.widths.state, Ownership::Public),
                (SegmentRole::CurrentState, self.widths.state, Ownership::Public),
                (SegmentRole::Running, self.widths.running, Ownership::Committed),
                (
                    SegmentRole::RunningWitness,
                    self.widths.running_witness,
                    Ownership::Committed,
                ),
                (SegmentRole::Fresh, self.widths.fresh, Ownership::Committed),
                (
                    SegmentRole::FreshWitness,
                    self.widths.fresh_witness,
                    Ownership::Committed,
                ),
            ],
        )?;
        let step = validate_program("step_program", &self.step_program)?;
        let terminal = validate_program("terminal_program", &self.terminal_program)?;
        let application_receipt = validate_program_prefix("step_program", &self.step_program, &self.step_input)?;
        validate_program_prefix("terminal_program", &self.terminal_program, &self.terminal_input)?;
        validate_input_columns("step_input", &self.step_input, &step.columns)?;
        validate_input_columns("terminal_input", &self.terminal_input, &terminal.columns)?;
        if step.cost != self.step_cost {
            return Err(invalid("step_cost", "does not match the Step receipt fold"));
        }
        if terminal.cost != self.terminal_cost {
            return Err(invalid("terminal_cost", "does not match the Terminal receipt fold"));
        }
        if step.statistics != self.step_statistics {
            return Err(invalid("step_statistics", "does not match the Step rows"));
        }
        if terminal.statistics != self.terminal_statistics {
            return Err(invalid("terminal_statistics", "does not match the Terminal rows"));
        }
        let actual_application_cost = manifest_receipt_cost(application_receipt, "step_program.application_receipt")?;
        if actual_application_cost != self.application_step_cost {
            return Err(invalid(
                "application_step_cost",
                "does not match the canonical application Step receipt",
            ));
        }
        let actual_fixed_protocol_cost = self.step_program.receipts.iter().enumerate().try_fold(
            ManifestCost::default(),
            |cost, (index, receipt)| {
                if index == 1 + self.step_input.len() {
                    Ok(cost)
                } else {
                    cost.add(
                        manifest_receipt_cost(receipt, &format!("step_program.receipts[{index}]"))?,
                        "fixed_protocol_cost",
                    )
                }
            },
        )?;
        if actual_fixed_protocol_cost != self.fixed_protocol_cost {
            return Err(invalid(
                "fixed_protocol_cost",
                "does not match the Step receipt fold with the application receipt removed",
            ));
        }
        if self
            .fixed_protocol_cost
            .add(self.application_step_cost, "step_cost_split")?
            != self.step_cost
        {
            return Err(invalid(
                "step_cost_split",
                "fixed protocol plus application Step does not equal Step cost",
            ));
        }
        let expected_results = expected_step_result_columns(&self.widths);
        if self.step_result_columns != expected_results {
            return Err(invalid(
                "step_result_columns",
                "does not match the canonical Step result ABI",
            ));
        }
        validate_owned_columns("step_result_columns", &self.step_result_columns, &step.columns)?;
        validate_result_ownership("step_result_columns", &self.step_result_columns, &self.step_result)?;
        let expected_step_selector = instruction_column(&[OwnerPathStep::Rest], 0, 0);
        if self.step_selector != expected_step_selector {
            return Err(invalid("step_selector", "does not match the canonical Step selector"));
        }
        let expected_terminal_selector = instruction_column(&[], 0, 0);
        if self.terminal_selector != expected_terminal_selector {
            return Err(invalid(
                "terminal_selector",
                "does not match the canonical Terminal selector",
            ));
        }
        validate_declared_columns(
            "step_selector",
            std::iter::once(&self.step_selector),
            &step.columns,
            Ownership::Auxiliary,
        )?;
        validate_declared_columns(
            "terminal_selector",
            std::iter::once(&self.terminal_selector),
            &terminal.columns,
            Ownership::Auxiliary,
        )?;
        validate_activation_pair(
            "step_activations",
            &self.step_activations,
            &step.columns,
            &[OwnerPathStep::Rest, OwnerPathStep::Rest],
        )?;
        validate_activation_pair(
            "terminal_activations",
            &self.terminal_activations,
            &terminal.columns,
            &[OwnerPathStep::Rest],
        )?;
        Ok(())
    }

    fn validate_profile(&self) -> Result<(), LeanManifestError> {
        let profile = &self.profile;
        let width_exact = [
            ("iteration", self.widths.iteration, 1),
            ("digest", self.widths.digest, 5),
            ("bit", self.widths.bit, 1),
        ];
        for (field, actual, expected) in width_exact {
            if actual != expected {
                return Err(invalid(
                    format!("widths.{field}"),
                    format!("got {actual}, expected {expected}"),
                ));
            }
        }
        let exact = [
            ("fresh_source_count", profile.fresh_source_count, 1),
            ("running_source_count", profile.running_source_count, 14),
            ("public_carrier_width", profile.public_carrier_width, 270),
            ("fresh_legacy_width", profile.fresh_legacy_width, 257),
            ("fresh_completion_width", profile.fresh_completion_width, 13),
            ("running_carrier_width", profile.running_carrier_width, 270),
            ("poseidon_width", profile.poseidon_width, 8),
            ("poseidon_rate", profile.poseidon_rate, 4),
            ("poseidon_capacity", profile.poseidon_capacity, 4),
            ("poseidon_digest_width", profile.poseidon_digest_width, 4),
            ("binding_preimage_width", profile.binding_preimage_width, 23),
            ("decomposition_base", profile.decomposition_base, 2),
            ("decomposition_children", profile.decomposition_children, 14),
        ];
        for (field, actual, expected) in exact {
            if actual != expected {
                return Err(invalid(
                    format!("profile.{field}"),
                    format!("got {actual}, expected {expected}"),
                ));
            }
        }
        Ok(())
    }
}

fn instruction_column(path: &[OwnerPathStep], bundle_index: usize, coordinate_index: usize) -> ColumnId {
    ColumnId {
        owner: PhysicalOwner::Typed {
            owner: TypedOwner::Instruction { path: path.to_vec() },
        },
        bundle_index,
        coordinate_index,
    }
}

fn branch_column(path: &[OwnerPathStep], bundle_index: usize, coordinate_index: usize) -> ColumnId {
    ColumnId {
        owner: PhysicalOwner::Typed {
            owner: TypedOwner::Branch { path: path.to_vec() },
        },
        bundle_index,
        coordinate_index,
    }
}

fn expected_step_result_columns(widths: &Widths) -> Vec<OwnedColumn> {
    let mut columns = Vec::with_capacity(widths.state + widths.running + widths.digest);
    columns.extend((0..widths.state).map(|coordinate_index| OwnedColumn {
        id: instruction_column(&[], 0, coordinate_index),
        ownership: Ownership::Committed,
    }));
    columns.extend((0..widths.running).map(|coordinate_index| OwnedColumn {
        id: branch_column(&[OwnerPathStep::Rest, OwnerPathStep::Rest], 0, coordinate_index),
        ownership: Ownership::Committed,
    }));
    columns.extend((0..widths.digest).map(|coordinate_index| OwnedColumn {
        id: instruction_column(
            &[OwnerPathStep::Rest, OwnerPathStep::Rest, OwnerPathStep::Continuation],
            0,
            coordinate_index,
        ),
        ownership: Ownership::Public,
    }));
    columns
}

fn validate_segments(
    path: &str,
    actual: &[CodecSegment],
    expected: &[(SegmentRole, usize, Ownership)],
) -> Result<(), LeanManifestError> {
    if actual.len() != expected.len() {
        return Err(invalid(
            path,
            format!("has {} segments, expected {}", actual.len(), expected.len()),
        ));
    }
    let mut offset = 0usize;
    for (index, (segment, expected)) in actual.iter().zip(expected).enumerate() {
        if segment.role != expected.0 || segment.width != expected.1 || segment.ownership != expected.2 {
            return Err(invalid(
                format!("{path}[{index}]"),
                "role, width, or ownership differs from the Lean profile",
            ));
        }
        if segment.offset != offset {
            return Err(invalid(
                format!("{path}[{index}].offset"),
                format!("got {}, expected {offset}", segment.offset),
            ));
        }
        offset = offset
            .checked_add(segment.width)
            .ok_or_else(|| invalid(format!("{path}[{index}].width"), "offset overflow"))?;
    }
    Ok(())
}

fn validate_program_prefix<'a>(
    path: &str,
    program: &'a ManifestProgram,
    input_segments: &[CodecSegment],
) -> Result<&'a ManifestReceipt, LeanManifestError> {
    let required_len = 2usize
        .checked_add(input_segments.len())
        .ok_or_else(|| invalid(path, "receipt prefix length overflow"))?;
    if program.receipts.len() < required_len {
        return Err(invalid(
            format!("{path}.receipts"),
            "does not contain the prelude, all inputs, and the root call receipt",
        ));
    }
    let prelude = &program.receipts[0];
    let expected_prelude = ManifestReceipt {
        owner: PhysicalOwner::Prelude,
        kind: InstructionKind::Prelude,
        allocations: vec![OwnedColumn {
            id: program.one.clone(),
            ownership: Ownership::Public,
        }],
        rows: vec![],
    };
    if prelude != &expected_prelude {
        return Err(invalid(
            format!("{path}.receipts[0]"),
            "is not the canonical constant-one prelude",
        ));
    }
    for (slot, segment) in input_segments.iter().enumerate() {
        let owner = PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot },
        };
        let expected_allocations = (0..segment.width)
            .map(|coordinate_index| OwnedColumn {
                id: ColumnId {
                    owner: owner.clone(),
                    bundle_index: slot,
                    coordinate_index,
                },
                ownership: segment.ownership,
            })
            .collect();
        let expected = ManifestReceipt {
            owner,
            kind: InstructionKind::Input,
            allocations: expected_allocations,
            rows: vec![],
        };
        if program.receipts[slot + 1] != expected {
            return Err(invalid(
                format!("{path}.receipts[{}]", slot + 1),
                "is not the canonical input receipt",
            ));
        }
    }
    let application_index = 1 + input_segments.len();
    let application = &program.receipts[application_index];
    if application.owner
        != (PhysicalOwner::Typed {
            owner: TypedOwner::Instruction { path: vec![] },
        })
        || application.kind != InstructionKind::Call
    {
        return Err(invalid(
            format!("{path}.receipts[{application_index}]"),
            "is not the canonical root application-call receipt",
        ));
    }
    Ok(application)
}

fn manifest_receipt_cost(receipt: &ManifestReceipt, path: &str) -> Result<ManifestCost, LeanManifestError> {
    let mut cost = ManifestCost {
        recurring_rows: receipt.rows.len(),
        ..ManifestCost::default()
    };
    for allocation in &receipt.allocations {
        let unit = match allocation.ownership {
            Ownership::Committed => ManifestCost {
                committed_columns: 1,
                ..ManifestCost::default()
            },
            Ownership::Public => ManifestCost {
                public_columns: 1,
                ..ManifestCost::default()
            },
            Ownership::Auxiliary => ManifestCost {
                auxiliary_columns: 1,
                ..ManifestCost::default()
            },
        };
        cost = cost.add(unit, path)?;
    }
    Ok(cost)
}

pub(super) fn validate_program(path: &str, program: &ManifestProgram) -> Result<ProgramSummary, LeanManifestError> {
    let expected_one = ColumnId {
        owner: PhysicalOwner::Prelude,
        bundle_index: 0,
        coordinate_index: 0,
    };
    if program.one != expected_one {
        return Err(invalid(
            format!("{path}.one"),
            "constant-one column does not have the canonical prelude identity",
        ));
    }
    let mut columns = HashMap::new();
    let mut rows = HashSet::new();
    let mut cost = ManifestCost::default();
    let mut statistics = ManifestStatistics::default();
    for (receipt_index, receipt) in program.receipts.iter().enumerate() {
        let receipt_path = format!("{path}.receipts[{receipt_index}]");
        for (column_index, column) in receipt.allocations.iter().enumerate() {
            if column.id.owner != receipt.owner {
                return Err(invalid(
                    format!("{receipt_path}.allocations[{column_index}].id.owner"),
                    "does not match receipt owner",
                ));
            }
            if columns
                .insert(column.id.clone(), column.ownership)
                .is_some()
            {
                return Err(invalid(
                    format!("{receipt_path}.allocations[{column_index}].id"),
                    "duplicate physical column identity",
                ));
            }
            let unit = match column.ownership {
                Ownership::Committed => ManifestCost {
                    committed_columns: 1,
                    ..ManifestCost::default()
                },
                Ownership::Public => ManifestCost {
                    public_columns: 1,
                    ..ManifestCost::default()
                },
                Ownership::Auxiliary => ManifestCost {
                    auxiliary_columns: 1,
                    ..ManifestCost::default()
                },
            };
            cost = cost.add(unit, &receipt_path)?;
        }
        for (row_index, row) in receipt.rows.iter().enumerate() {
            let row_path = format!("{receipt_path}.rows[{row_index}]");
            if row.id.owner != receipt.owner {
                return Err(invalid(format!("{row_path}.id.owner"), "does not match receipt owner"));
            }
            if !rows.insert(row.id.clone()) {
                return Err(invalid(format!("{row_path}.id"), "duplicate physical row identity"));
            }
            validate_combination(&format!("{row_path}.a"), &row.a, &columns)?;
            validate_combination(&format!("{row_path}.b"), &row.b, &columns)?;
            validate_combination(&format!("{row_path}.c"), &row.c, &columns)?;
            cost = cost.add(
                ManifestCost {
                    recurring_rows: 1,
                    ..ManifestCost::default()
                },
                &row_path,
            )?;
            statistics.a_nonzeros = statistics
                .a_nonzeros
                .checked_add(row.a.len())
                .ok_or_else(|| invalid(format!("{row_path}.a"), "nonzero count overflow"))?;
            statistics.b_nonzeros = statistics
                .b_nonzeros
                .checked_add(row.b.len())
                .ok_or_else(|| invalid(format!("{row_path}.b"), "nonzero count overflow"))?;
            statistics.c_nonzeros = statistics
                .c_nonzeros
                .checked_add(row.c.len())
                .ok_or_else(|| invalid(format!("{row_path}.c"), "nonzero count overflow"))?;
            let support = row
                .a
                .len()
                .checked_add(row.b.len())
                .and_then(|value| value.checked_add(row.c.len()))
                .ok_or_else(|| invalid(&row_path, "row support overflow"))?;
            statistics.max_row_support = statistics.max_row_support.max(support);
        }
    }
    if columns.get(&program.one) != Some(&Ownership::Public) {
        return Err(invalid(
            format!("{path}.one"),
            "constant-one column is not allocated as a public column",
        ));
    }
    Ok(ProgramSummary {
        cost,
        statistics,
        columns,
    })
}

fn validate_combination(
    path: &str,
    combination: &[ManifestTerm],
    available: &HashMap<ColumnId, Ownership>,
) -> Result<(), LeanManifestError> {
    let mut columns = HashSet::new();
    for (index, term) in combination.iter().enumerate() {
        if term.coefficient == 0 || term.coefficient >= GOLDILOCKS_MODULUS {
            return Err(invalid(
                format!("{path}[{index}].coefficient"),
                "coefficient is zero or is not a canonical Goldilocks residue",
            ));
        }
        if !columns.insert(&term.column) {
            return Err(invalid(
                format!("{path}[{index}].column"),
                "duplicate column in normalized sparse combination",
            ));
        }
        if !available.contains_key(&term.column) {
            return Err(invalid(
                format!("{path}[{index}].column"),
                "column was not allocated by this or an earlier receipt",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_declared_columns<'a>(
    path: &str,
    declared: impl Iterator<Item = &'a ColumnId>,
    allocated: &HashMap<ColumnId, Ownership>,
    expected_ownership: Ownership,
) -> Result<(), LeanManifestError> {
    let mut seen = HashSet::new();
    for (index, column) in declared.enumerate() {
        if !seen.insert(column) {
            return Err(invalid(format!("{path}[{index}]"), "duplicate declared column"));
        }
        if allocated.get(column) != Some(&expected_ownership) {
            return Err(invalid(
                format!("{path}[{index}]"),
                "declared column is missing or has the wrong allocation class",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_owned_columns(
    path: &str,
    declared: &[OwnedColumn],
    allocated: &HashMap<ColumnId, Ownership>,
) -> Result<(), LeanManifestError> {
    let mut seen = HashSet::new();
    for (index, column) in declared.iter().enumerate() {
        if !seen.insert(&column.id) {
            return Err(invalid(format!("{path}[{index}]"), "duplicate declared column"));
        }
        if allocated.get(&column.id) != Some(&column.ownership) {
            return Err(invalid(
                format!("{path}[{index}]"),
                "declared column is missing or has the wrong allocation class",
            ));
        }
    }
    Ok(())
}

fn validate_result_ownership(
    path: &str,
    columns: &[OwnedColumn],
    segments: &[CodecSegment],
) -> Result<(), LeanManifestError> {
    let expected_len = segments.iter().try_fold(0usize, |sum, segment| {
        sum.checked_add(segment.width)
            .ok_or_else(|| invalid(path, "result width overflow"))
    })?;
    if columns.len() != expected_len {
        return Err(invalid(
            path,
            format!("has {} columns, expected {expected_len}", columns.len()),
        ));
    }
    let expected_ownership = segments
        .iter()
        .flat_map(|segment| std::iter::repeat_n(segment.ownership, segment.width));
    for (index, (column, expected)) in columns.iter().zip(expected_ownership).enumerate() {
        if column.ownership != expected {
            return Err(invalid(
                format!("{path}[{index}].ownership"),
                "does not match the result codec segment",
            ));
        }
    }
    Ok(())
}

fn validate_input_columns(
    path: &str,
    segments: &[CodecSegment],
    allocated: &HashMap<ColumnId, Ownership>,
) -> Result<(), LeanManifestError> {
    let actual: HashMap<_, _> = allocated
        .iter()
        .filter_map(|(column, ownership)| match &column.owner {
            PhysicalOwner::Typed {
                owner: TypedOwner::Input { .. },
            } => Some((column.clone(), *ownership)),
            _ => None,
        })
        .collect();
    let mut expected = HashMap::new();
    for (slot, segment) in segments.iter().enumerate() {
        for coordinate_index in 0..segment.width {
            let column = ColumnId {
                owner: PhysicalOwner::Typed {
                    owner: TypedOwner::Input { slot },
                },
                bundle_index: slot,
                coordinate_index,
            };
            expected.insert(column, segment.ownership);
        }
    }
    if actual != expected {
        return Err(invalid(
            path,
            "codec segments do not match the program input allocations",
        ));
    }
    Ok(())
}

pub(super) fn validate_activation_pair(
    path: &str,
    columns: &[ColumnId],
    allocated: &HashMap<ColumnId, Ownership>,
    expected_path: &[OwnerPathStep],
) -> Result<(), LeanManifestError> {
    if columns.len() != 2 {
        return Err(invalid(path, "must contain the two branch activations"));
    }
    validate_declared_columns(path, columns.iter(), allocated, Ownership::Auxiliary)?;
    let expected_shape = |column: &ColumnId, selected: bool| {
        column.bundle_index == 0
            && column.coordinate_index == 0
            && matches!(
                &column.owner,
                PhysicalOwner::BranchActivation {
                    selected: actual,
                    ..
                } if *actual == selected
            )
    };
    if !expected_shape(&columns[0], true) || !expected_shape(&columns[1], false) {
        return Err(invalid(
            path,
            "must contain canonical true then false activation columns",
        ));
    }
    let same_path = match (&columns[0].owner, &columns[1].owner) {
        (PhysicalOwner::BranchActivation { path: left, .. }, PhysicalOwner::BranchActivation { path: right, .. }) => {
            left == right && left == expected_path
        }
        _ => false,
    };
    if !same_path {
        return Err(invalid(
            path,
            "activation columns do not use the canonical shared branch path",
        ));
    }
    Ok(())
}

pub(super) fn emit_program(
    builder: &mut R1csBuilder,
    program: &ManifestProgram,
    mut values: impl FnMut(&ColumnId) -> Option<F>,
) -> Result<ManifestEmission, LeanManifestEmissionError> {
    if builder.rows() != 0 || builder.cols() != 1 {
        return Err(LeanManifestEmissionError::BuilderNotFresh {
            rows: builder.rows(),
            columns: builder.cols(),
        });
    }
    let allocations: Vec<_> = program
        .receipts
        .iter()
        .flat_map(|receipt| receipt.allocations.iter())
        .collect();
    let mut variables = HashMap::with_capacity(allocations.len());
    variables.insert(program.one.clone(), Var::ONE);
    let mut public_columns = vec![program.one.clone()];
    let mut committed_columns = Vec::new();
    let mut auxiliary_columns = Vec::new();
    for ownership in [Ownership::Public, Ownership::Committed, Ownership::Auxiliary] {
        for allocation in allocations
            .iter()
            .filter(|allocation| allocation.ownership == ownership && allocation.id != program.one)
        {
            let value = values(&allocation.id).ok_or_else(|| LeanManifestEmissionError::MissingValue {
                column: allocation.id.clone(),
            })?;
            let variable = builder.alloc(value);
            variables.insert(allocation.id.clone(), variable);
            match ownership {
                Ownership::Public => public_columns.push(allocation.id.clone()),
                Ownership::Committed => committed_columns.push(allocation.id.clone()),
                Ownership::Auxiliary => auxiliary_columns.push(allocation.id.clone()),
            }
        }
    }
    for receipt in &program.receipts {
        for row in &receipt.rows {
            let a = emit_combination(&row.a, &variables)?;
            let b = emit_combination(&row.b, &variables)?;
            let c = emit_combination(&row.c, &variables)?;
            builder.enforce(&a, &b, &c);
        }
    }
    Ok(ManifestEmission {
        variables,
        public_columns,
        committed_columns,
        auxiliary_columns,
    })
}

fn emit_combination(
    combination: &[ManifestTerm],
    variables: &HashMap<ColumnId, Var>,
) -> Result<Lc, LeanManifestEmissionError> {
    let mut result = Lc::zero();
    for term in combination {
        let variable =
            variables
                .get(&term.column)
                .copied()
                .ok_or_else(|| LeanManifestEmissionError::UnknownColumn {
                    column: term.column.clone(),
                })?;
        result.add_term(variable, F::from_u64(term.coefficient));
    }
    Ok(result)
}
