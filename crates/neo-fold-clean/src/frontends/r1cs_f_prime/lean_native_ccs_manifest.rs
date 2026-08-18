//! Fail-closed consumer for the Lean-owned native CCS F′ manifest.
//!
//! Owns: schema-v3 decoding, full structural validation, exact physical
//! column placement, direct construction of the four sparse matrices
//! `[A, B, C, S]`, the seeded Ajtai setup identity, and validation of the
//! compact terminal-R1CS descriptor.
//!
//! Does not own: manifest generation, witness generation, deployment
//! selection, or authority for operator-selected bytes.
//!
//! Emits constraints: yes. One validated Lean source row becomes one CCS row.
//! No activation residual witness or second R1CS row is created.

use std::collections::{HashMap, HashSet};

use neo_ccs::{check_ccs_rowwise_zero, sparse_selected_r1cs_to_ccs, CcsMatrix, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use thiserror::Error;

use crate::config::K_RHO;
use crate::paper::relations::Structure;

use super::lean_manifest::{
    emit_program, invalid, validate_activation_pair, validate_declared_columns, validate_owned_columns,
    validate_program, ColumnId, InstructionKind, LeanManifestEmissionError, LeanManifestError, ManifestCost,
    ManifestEmission, ManifestProgram, ManifestReceipt, ManifestRow, OwnedColumn, OwnerPathStep, Ownership,
    PhysicalOwner, TypedOwner, GOLDILOCKS_MODULUS,
};
use crate::engine::r1cs_circuit::R1csBuilder;

pub const LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION: u64 = 3;
pub const LEAN_NATIVE_CCS_MANIFEST_FORMAT: &str = "nightstream/fprime-native-ccs-manifest";
pub const LEAN_NATIVE_CCS_AJTAI_ALGORITHM: &str = "chacha8_phi81_rejection_v1";
pub const NATIVE_SELECTOR_MATRIX_COUNT: usize = 4;
pub const NATIVE_SELECTOR_POLYNOMIAL_DEGREE: usize = 3;
const PHI81_RING_DEGREE: usize = 54;
const TERMINAL_COMMITMENT_ROWS: usize = 18;

#[derive(Debug, Error)]
pub enum LeanNativeCcsEmissionError {
    #[error("missing witness value for Lean native CCS column {column:?}")]
    MissingValue { column: ColumnId },
    #[error("validated Lean native CCS row refers to an unknown column {column:?}")]
    UnknownColumn { column: ColumnId },
    #[error("cannot construct the validated native CCS relation: {0}")]
    InvalidStructure(String),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum PolynomialSign {
    Positive,
    Negative,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PolynomialTerm {
    sign: PolynomialSign,
    exponents: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeManifestReceipt {
    pub owner: PhysicalOwner,
    pub kind: InstructionKind,
    pub allocations: Vec<OwnedColumn>,
    pub selector: ColumnId,
    pub rows: Vec<ManifestRow>,
}

impl NativeManifestReceipt {
    fn as_canonical(&self) -> ManifestReceipt {
        ManifestReceipt {
            owner: self.owner.clone(),
            kind: self.kind,
            allocations: self.allocations.clone(),
            rows: self.rows.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeManifestProgram {
    pub one: ColumnId,
    matrix_count: usize,
    polynomial_degree: usize,
    polynomial: Vec<PolynomialTerm>,
    pub receipts: Vec<NativeManifestReceipt>,
}

impl NativeManifestProgram {
    fn as_canonical(&self) -> ManifestProgram {
        ManifestProgram {
            one: self.one.clone(),
            receipts: self
                .receipts
                .iter()
                .map(NativeManifestReceipt::as_canonical)
                .collect(),
        }
    }

    pub(crate) fn row_count(&self) -> usize {
        self.receipts.iter().map(|receipt| receipt.rows.len()).sum()
    }
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
enum AjtaiSetupAlgorithm {
    #[serde(rename = "chacha8_phi81_rejection_v1")]
    ChaCha8Phi81RejectionV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct AjtaiSetupDescriptor {
    algorithm: AjtaiSetupAlgorithm,
    seed: [u8; 32],
    rejection_fuel: usize,
}

/// Static Lean-owned dimensions for reconstructing the terminal R1CS from
/// the verifier's authoritative key and terminal statements.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TerminalR1csDescriptor {
    row_variables: usize,
    logical_width: usize,
    recursive_rows: usize,
    fresh_relation_rows: usize,
    fresh_relation_auxiliary_columns: usize,
    matrix_count: usize,
    public_ring_columns: usize,
    verifier_rows: usize,
    cost: ManifestCost,
}

impl TerminalR1csDescriptor {
    pub fn row_variables(self) -> usize {
        self.row_variables
    }

    pub fn logical_width(self) -> usize {
        self.logical_width
    }

    pub fn recursive_rows(self) -> usize {
        self.recursive_rows
    }

    pub fn fresh_relation_rows(self) -> usize {
        self.fresh_relation_rows
    }

    pub fn fresh_relation_auxiliary_columns(self) -> usize {
        self.fresh_relation_auxiliary_columns
    }

    pub fn matrix_count(self) -> usize {
        self.matrix_count
    }

    pub fn public_ring_columns(self) -> usize {
        self.public_ring_columns
    }

    pub fn verifier_rows(self) -> usize {
        self.verifier_rows
    }

    pub fn cost(self) -> ManifestCost {
        self.cost
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeManifestWire {
    schema: u64,
    format: String,
    goldilocks_modulus: u64,
    ajtai_setup: AjtaiSetupDescriptor,
    profile: ProfileIdentifier,
    widths: Widths,
    step_input: Vec<CodecSegment>,
    step_result: Vec<CodecSegment>,
    terminal_input: Vec<CodecSegment>,
    step_program: NativeManifestProgram,
    terminal_program: ManifestProgram,
    terminal_r1cs: TerminalR1csDescriptor,
    step_result_columns: Vec<OwnedColumn>,
    step_selector: ColumnId,
    terminal_selector: ColumnId,
    step_activations: Vec<ColumnId>,
    terminal_activations: Vec<ColumnId>,
    step_cost: ManifestCost,
    terminal_cost: ManifestCost,
}

/// A native CCS manifest that passed every schema and structural check.
///
/// This type does not implement `Deserialize`. Call
/// [`Self::from_json_slice`] so that callers cannot bypass validation.
#[derive(Clone, Debug)]
pub struct LeanNativeCcsManifest {
    wire: NativeManifestWire,
}

/// Exact native CCS relation and assignment emitted from one validated
/// Lean Step manifest.
#[derive(Clone, Debug)]
pub struct NativeStepEmission {
    structure: Structure,
    public_values: Vec<F>,
    witness_values: Vec<F>,
    column_indices: HashMap<ColumnId, usize>,
    public_columns: Vec<ColumnId>,
    committed_columns: Vec<ColumnId>,
    auxiliary_columns: Vec<ColumnId>,
}

/// Phi81-completed native Step relation in Lean receipt order.
///
/// Logical columns retain `program.columnIds` order. Canonical zero
/// coordinates complete the final 54-lane block. This is the relation that
/// SuperNeo folds and that the Lean terminal R1CS opens.
#[derive(Clone, Debug)]
pub struct NativePhi81StepEmission {
    structure: Structure,
    assignment: Vec<F>,
    column_indices: HashMap<ColumnId, usize>,
    logical_width: usize,
    public_width: usize,
}

impl NativePhi81StepEmission {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn assignment(&self) -> &[F] {
        &self.assignment
    }

    pub fn logical_width(&self) -> usize {
        self.logical_width
    }

    pub fn public_width(&self) -> usize {
        self.public_width
    }

    pub fn column_index(&self, column: &ColumnId) -> Option<usize> {
        self.column_indices.get(column).copied()
    }

    pub fn is_satisfied(&self) -> bool {
        check_ccs_rowwise_zero(
            &self.structure,
            &self.assignment[..self.public_width],
            &self.assignment[self.public_width..],
        )
        .is_ok()
    }
}

impl NativeStepEmission {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn public_values(&self) -> &[F] {
        &self.public_values
    }

    pub fn witness_values(&self) -> &[F] {
        &self.witness_values
    }

    pub fn column_index(&self, column: &ColumnId) -> Option<usize> {
        self.column_indices.get(column).copied()
    }

    pub fn public_columns(&self) -> &[ColumnId] {
        &self.public_columns
    }

    pub fn committed_columns(&self) -> &[ColumnId] {
        &self.committed_columns
    }

    pub fn auxiliary_columns(&self) -> &[ColumnId] {
        &self.auxiliary_columns
    }

    pub fn is_satisfied(&self) -> bool {
        check_ccs_rowwise_zero(&self.structure, &self.public_values, &self.witness_values).is_ok()
    }
}

impl LeanNativeCcsManifest {
    /// Decode and validate one complete Lean-owned native manifest.
    ///
    /// Unknown fields, shape drift, polynomial drift, selector
    /// substitution, missing allocation, duplicate identity, cost drift,
    /// and ABI drift all reject.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, LeanManifestError> {
        let wire: NativeManifestWire = serde_json::from_slice(bytes)?;
        wire.validate()?;
        Ok(Self { wire })
    }

    pub fn matrix_count(&self) -> usize {
        self.wire.step_program.matrix_count
    }

    pub fn polynomial_degree(&self) -> usize {
        self.wire.step_program.polynomial_degree
    }

    pub fn step_cost(&self) -> ManifestCost {
        self.wire.step_cost
    }

    pub fn terminal_cost(&self) -> ManifestCost {
        self.wire.terminal_cost
    }

    pub fn terminal_r1cs(&self) -> TerminalR1csDescriptor {
        self.wire.terminal_r1cs
    }

    /// Exact verifier-owned setup seed selected by the Lean manifest.
    pub fn ajtai_setup_seed(&self) -> [u8; 32] {
        self.wire.ajtai_setup.seed
    }

    /// Bound used by Lean to prove that rejection sampling succeeds. Rust
    /// samples without this proof bound, but retains it in the wire contract.
    pub fn ajtai_rejection_fuel(&self) -> usize {
        self.wire.ajtai_setup.rejection_fuel
    }

    pub fn running_claim_count(&self) -> usize {
        self.wire.profile.running_source_count
    }

    pub fn fresh_claim_count(&self) -> usize {
        self.wire.profile.fresh_source_count
    }

    pub fn public_carrier_width(&self) -> usize {
        self.wire.profile.public_carrier_width
    }

    pub fn step_program(&self) -> &NativeManifestProgram {
        &self.wire.step_program
    }

    pub fn outer_terminal_program(&self) -> &ManifestProgram {
        &self.wire.terminal_program
    }

    /// Emit the unchanged Lean-owned outer Terminal program.
    ///
    /// This is the F-prime outer control relation. It is not the direct
    /// terminal claim R1CS consumed by Spartan.
    pub fn emit_outer_terminal(
        &self,
        builder: &mut R1csBuilder,
        values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<ManifestEmission, LeanManifestEmissionError> {
        emit_program(builder, &self.wire.terminal_program, values)
    }

    /// Emit one exact four-matrix CCS relation. Physical columns are ordered
    /// as public, committed, then auxiliary. The constant one is public
    /// column zero.
    pub fn emit_step(
        &self,
        mut values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<NativeStepEmission, LeanNativeCcsEmissionError> {
        emit_native_program(&self.wire.step_program, &mut values)
    }

    /// Emit the completed relation in the exact Lean logical-column order.
    pub fn emit_phi81_step(
        &self,
        mut values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<NativePhi81StepEmission, LeanNativeCcsEmissionError> {
        emit_phi81_program(&self.wire.step_program, self.wire.terminal_r1cs, &mut values)
    }
}

impl NativeManifestWire {
    fn validate(&self) -> Result<(), LeanManifestError> {
        if self.schema != LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION {
            return Err(LeanManifestError::UnsupportedSchema {
                found: self.schema,
                expected: LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION,
            });
        }
        if self.format != LEAN_NATIVE_CCS_MANIFEST_FORMAT {
            return Err(invalid("format", format!("unsupported format {:?}", self.format)));
        }
        if self.goldilocks_modulus != GOLDILOCKS_MODULUS {
            return Err(invalid(
                "goldilocks_modulus",
                format!("got {}, expected {GOLDILOCKS_MODULUS}", self.goldilocks_modulus),
            ));
        }
        self.validate_ajtai_setup()?;
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

        validate_native_polynomial(&self.step_program)?;
        let canonical_step = self.step_program.as_canonical();
        let step = validate_program("step_program", &canonical_step)?;
        let terminal = validate_program("terminal_program", &self.terminal_program)?;
        self.validate_terminal_r1cs(&step)?;
        validate_native_selectors(&self.step_program, &step.columns, &self.step_activations)?;
        validate_program_prefix(
            "step_program",
            &canonical_step,
            &self.step_input,
            Some(&self.step_program),
        )?;
        validate_program_prefix("terminal_program", &self.terminal_program, &self.terminal_input, None)?;
        validate_input_columns("step_input", &self.step_input, &step.columns)?;
        validate_input_columns("terminal_input", &self.terminal_input, &terminal.columns)?;
        if step.cost != self.step_cost {
            return Err(invalid("step_cost", "does not match the native Step receipt fold"));
        }
        if terminal.cost != self.terminal_cost {
            return Err(invalid("terminal_cost", "does not match the Terminal receipt fold"));
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

        let expected_step_selector = instruction_column(&[OwnerPathStep::Rest]);
        if self.step_selector != expected_step_selector {
            return Err(invalid("step_selector", "does not match the canonical Step selector"));
        }
        let expected_terminal_selector = instruction_column(&[]);
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

    fn validate_ajtai_setup(&self) -> Result<(), LeanManifestError> {
        if self.ajtai_setup.algorithm != AjtaiSetupAlgorithm::ChaCha8Phi81RejectionV1 {
            return Err(invalid("ajtai_setup.algorithm", "unsupported setup sampler"));
        }
        if self.ajtai_setup.rejection_fuel == 0 {
            return Err(invalid(
                "ajtai_setup.rejection_fuel",
                "must be positive for the selected nonempty verifier key",
            ));
        }
        Ok(())
    }

    fn validate_profile(&self) -> Result<(), LeanManifestError> {
        if self.profile.name != ProfileName::FixedOnePlain270 {
            return Err(invalid("profile.name", "unsupported profile"));
        }
        if self.profile.matrix_count != NATIVE_SELECTOR_MATRIX_COUNT
            || self.profile.matrix_count != self.step_program.matrix_count
        {
            return Err(invalid(
                "profile.matrix_count",
                "must equal the native Step matrix count four",
            ));
        }
        let widths = [
            ("iteration", self.widths.iteration, 1),
            ("digest", self.widths.digest, 5),
            ("bit", self.widths.bit, 1),
        ];
        for (field, actual, expected) in widths {
            if actual != expected {
                return Err(invalid(
                    format!("widths.{field}"),
                    format!("got {actual}, expected {expected}"),
                ));
            }
        }
        let exact = [
            ("fresh_source_count", self.profile.fresh_source_count, 1),
            (
                "running_source_count",
                self.profile.running_source_count,
                K_RHO as usize,
            ),
            ("public_carrier_width", self.profile.public_carrier_width, 270),
            ("fresh_legacy_width", self.profile.fresh_legacy_width, 257),
            ("fresh_completion_width", self.profile.fresh_completion_width, 13),
            ("running_carrier_width", self.profile.running_carrier_width, 270),
            ("poseidon_width", self.profile.poseidon_width, 8),
            ("poseidon_rate", self.profile.poseidon_rate, 4),
            ("poseidon_capacity", self.profile.poseidon_capacity, 4),
            ("poseidon_digest_width", self.profile.poseidon_digest_width, 4),
            ("binding_preimage_width", self.profile.binding_preimage_width, 23),
            ("decomposition_base", self.profile.decomposition_base, 2),
            (
                "decomposition_children",
                self.profile.decomposition_children,
                K_RHO as usize,
            ),
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

    fn validate_terminal_r1cs(&self, step: &super::lean_manifest::ProgramSummary) -> Result<(), LeanManifestError> {
        let descriptor = self.terminal_r1cs;
        if descriptor.logical_width != step.columns.len() {
            return Err(invalid(
                "terminal_r1cs.logical_width",
                "does not match the native Step allocation count",
            ));
        }
        if descriptor.recursive_rows != step.cost.recurring_rows() {
            return Err(invalid(
                "terminal_r1cs.recursive_rows",
                "does not match the native Step row count",
            ));
        }
        let expected_fresh_rows = checked_mul("terminal_r1cs.fresh_relation_rows", 2, descriptor.recursive_rows)?;
        if descriptor.fresh_relation_rows != expected_fresh_rows {
            return Err(invalid(
                "terminal_r1cs.fresh_relation_rows",
                "does not equal the exact native selected-R1CS lowering",
            ));
        }
        if descriptor.fresh_relation_auxiliary_columns != descriptor.recursive_rows {
            return Err(invalid(
                "terminal_r1cs.fresh_relation_auxiliary_columns",
                "does not equal the exact native residual-column count",
            ));
        }
        if descriptor.matrix_count != self.step_program.matrix_count {
            return Err(invalid(
                "terminal_r1cs.matrix_count",
                "does not match the native Step matrix count",
            ));
        }
        let public_width = checked_mul(
            "terminal_r1cs.public_ring_columns",
            descriptor.public_ring_columns,
            PHI81_RING_DEGREE,
        )?;
        if public_width != self.profile.public_carrier_width {
            return Err(invalid(
                "terminal_r1cs.public_ring_columns",
                "does not reconstruct the fixed 270-field public carrier",
            ));
        }
        if descriptor.verifier_rows != TERMINAL_COMMITMENT_ROWS {
            return Err(invalid(
                "terminal_r1cs.verifier_rows",
                "does not match the selected commitment-key row count",
            ));
        }
        let carrier_width = phi81_carrier_width(descriptor.logical_width)?;
        if public_width > carrier_width {
            return Err(invalid(
                "terminal_r1cs.public_ring_columns",
                format!("public width {public_width} exceeds completed carrier width {carrier_width}"),
            ));
        }
        validate_minimal_row_domain(descriptor.row_variables, descriptor.recursive_rows)?;
        let expected = terminal_r1cs_cost(
            descriptor,
            self.profile.running_source_count,
            self.profile.fresh_source_count,
        )?;
        if descriptor.cost != expected {
            return Err(invalid(
                "terminal_r1cs.cost",
                "does not match the Lean terminal-R1CS cost formula",
            ));
        }
        Ok(())
    }
}

fn checked_add(path: &str, left: usize, right: usize) -> Result<usize, LeanManifestError> {
    left.checked_add(right)
        .ok_or_else(|| invalid(path, "count overflow"))
}

fn checked_mul(path: &str, left: usize, right: usize) -> Result<usize, LeanManifestError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(path, "count overflow"))
}

fn validate_minimal_row_domain(row_variables: usize, rows: usize) -> Result<(), LeanManifestError> {
    let shift =
        u32::try_from(row_variables).map_err(|_| invalid("terminal_r1cs.row_variables", "row domain overflow"))?;
    let capacity = 1usize
        .checked_shl(shift)
        .ok_or_else(|| invalid("terminal_r1cs.row_variables", "row domain overflow"))?;
    if rows > capacity {
        return Err(invalid(
            "terminal_r1cs.row_variables",
            "row domain is too small for the recursive relation",
        ));
    }
    if row_variables > 0 {
        let previous = 1usize << (shift - 1);
        if rows <= previous {
            return Err(invalid(
                "terminal_r1cs.row_variables",
                "row domain is not the least power-of-two domain",
            ));
        }
    }
    Ok(())
}

fn phi81_carrier_width(logical_width: usize) -> Result<usize, LeanManifestError> {
    let blocks = checked_add("terminal_r1cs.logical_width", logical_width, PHI81_RING_DEGREE - 1)? / PHI81_RING_DEGREE;
    checked_mul("terminal_r1cs.logical_width", blocks, PHI81_RING_DEGREE)
}

fn terminal_r1cs_cost(
    descriptor: TerminalR1csDescriptor,
    running_claims: usize,
    fresh_claims: usize,
) -> Result<ManifestCost, LeanManifestError> {
    if fresh_claims != 1 {
        return Err(invalid(
            "profile.fresh_source_count",
            "terminal R1CS requires exactly one fresh claim",
        ));
    }
    let carrier = phi81_carrier_width(descriptor.logical_width)?;
    let public_width = checked_mul("terminal_r1cs.cost", descriptor.public_ring_columns, PHI81_RING_DEGREE)?;
    let verifier_width = checked_mul("terminal_r1cs.cost", descriptor.verifier_rows, PHI81_RING_DEGREE)?;
    let evaluations = checked_mul(
        "terminal_r1cs.cost",
        checked_add("terminal_r1cs.cost", descriptor.matrix_count, 1)?,
        PHI81_RING_DEGREE,
    )?;
    let two_carriers = checked_mul("terminal_r1cs.cost", 2, carrier)?;
    let two_evaluations = checked_mul("terminal_r1cs.cost", 2, evaluations)?;
    let running_statement = checked_add(
        "terminal_r1cs.cost",
        checked_add("terminal_r1cs.cost", verifier_width, public_width)?,
        two_evaluations,
    )?;
    let running_rows = checked_add("terminal_r1cs.cost", running_statement, two_carriers)?;
    let fresh_statement = checked_add("terminal_r1cs.cost", verifier_width, public_width)?;
    let fresh_rows = checked_add(
        "terminal_r1cs.cost",
        checked_add("terminal_r1cs.cost", fresh_statement, two_carriers)?,
        descriptor.fresh_relation_rows,
    )?;
    let claims = checked_add("terminal_r1cs.cost", running_claims, fresh_claims)?;
    Ok(ManifestCost {
        recurring_rows: checked_add(
            "terminal_r1cs.cost",
            checked_mul("terminal_r1cs.cost", running_claims, running_rows)?,
            fresh_rows,
        )?,
        committed_columns: checked_mul("terminal_r1cs.cost", claims, carrier)?,
        public_columns: checked_add(
            "terminal_r1cs.cost",
            checked_add(
                "terminal_r1cs.cost",
                1,
                checked_mul("terminal_r1cs.cost", running_claims, running_statement)?,
            )?,
            fresh_statement,
        )?,
        auxiliary_columns: checked_add(
            "terminal_r1cs.cost",
            checked_mul("terminal_r1cs.cost", claims, carrier)?,
            descriptor.fresh_relation_auxiliary_columns,
        )?,
    })
}

fn validate_native_polynomial(program: &NativeManifestProgram) -> Result<(), LeanManifestError> {
    let expected = vec![
        PolynomialTerm {
            sign: PolynomialSign::Positive,
            exponents: vec![1, 1, 0, 1],
        },
        PolynomialTerm {
            sign: PolynomialSign::Negative,
            exponents: vec![0, 0, 1, 1],
        },
    ];
    if program.matrix_count != NATIVE_SELECTOR_MATRIX_COUNT {
        return Err(invalid(
            "step_program.matrix_count",
            "native selector requires four matrices",
        ));
    }
    if program.polynomial_degree != NATIVE_SELECTOR_POLYNOMIAL_DEGREE {
        return Err(invalid(
            "step_program.polynomial_degree",
            "native selector requires degree three",
        ));
    }
    if program.polynomial != expected {
        return Err(invalid(
            "step_program.polynomial",
            "must equal A * B * S - C * S in matrix order [A, B, C, S]",
        ));
    }
    Ok(())
}

fn validate_native_selectors(
    program: &NativeManifestProgram,
    columns: &HashMap<ColumnId, Ownership>,
    activations: &[ColumnId],
) -> Result<(), LeanManifestError> {
    if activations.len() != 2 {
        return Err(invalid(
            "step_activations",
            "must contain the true and false branch activations",
        ));
    }
    let expected_target_path = vec![
        OwnerPathStep::Rest,
        OwnerPathStep::Rest,
        OwnerPathStep::FalseArm,
        OwnerPathStep::Rest,
        OwnerPathStep::Rest,
        OwnerPathStep::Rest,
        OwnerPathStep::Rest,
        OwnerPathStep::Rest,
    ];
    let expected_target = PhysicalOwner::Typed {
        owner: TypedOwner::Instruction {
            path: expected_target_path,
        },
    };
    let mut selected_count = 0usize;
    let mut available = HashSet::new();
    for (receipt_index, receipt) in program.receipts.iter().enumerate() {
        for allocation in &receipt.allocations {
            available.insert(allocation.id.clone());
        }
        if !available.contains(&receipt.selector) {
            return Err(invalid(
                format!("step_program.receipts[{receipt_index}].selector"),
                "selector is not allocated by this or an earlier receipt",
            ));
        }
        if receipt.selector == program.one {
            continue;
        }
        selected_count += 1;
        if receipt.owner != expected_target {
            return Err(invalid(
                format!("step_program.receipts[{receipt_index}].owner"),
                "the only native-selected receipt must be recursive NIFS",
            ));
        }
        if receipt.selector != activations[1] {
            return Err(invalid(
                format!("step_program.receipts[{receipt_index}].selector"),
                "recursive NIFS must use the false-arm activation",
            ));
        }
        if receipt.rows.is_empty() {
            return Err(invalid(
                format!("step_program.receipts[{receipt_index}].rows"),
                "the native-selected NIFS receipt must contain rows",
            ));
        }
    }
    if selected_count != 1 {
        return Err(invalid(
            "step_program.receipts",
            format!("contains {selected_count} selected receipts; expected one"),
        ));
    }
    if columns.get(&program.one) != Some(&Ownership::Public) {
        return Err(invalid(
            "step_program.one",
            "constant one is not an allocated public column",
        ));
    }
    Ok(())
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

fn validate_program_prefix(
    path: &str,
    program: &ManifestProgram,
    input_segments: &[CodecSegment],
    native: Option<&NativeManifestProgram>,
) -> Result<(), LeanManifestError> {
    let required_len = 2usize
        .checked_add(input_segments.len())
        .ok_or_else(|| invalid(path, "receipt prefix length overflow"))?;
    if program.receipts.len() < required_len {
        return Err(invalid(
            format!("{path}.receipts"),
            "does not contain the prelude, all inputs, and root application receipt",
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
        let expected = ManifestReceipt {
            owner: owner.clone(),
            kind: InstructionKind::Input,
            allocations: (0..segment.width)
                .map(|coordinate_index| OwnedColumn {
                    id: ColumnId {
                        owner: owner.clone(),
                        bundle_index: slot,
                        coordinate_index,
                    },
                    ownership: segment.ownership,
                })
                .collect(),
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
    if let Some(native) = native {
        for index in 0..=application_index {
            if native.receipts[index].selector != native.one {
                return Err(invalid(
                    format!("{path}.receipts[{index}].selector"),
                    "prelude, inputs, and application Step must be unconditional",
                ));
            }
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
            expected.insert(
                ColumnId {
                    owner: PhysicalOwner::Typed {
                        owner: TypedOwner::Input { slot },
                    },
                    bundle_index: slot,
                    coordinate_index,
                },
                segment.ownership,
            );
        }
    }
    if actual != expected {
        return Err(invalid(path, "codec segments do not match program input allocations"));
    }
    Ok(())
}

fn instruction_column(path: &[OwnerPathStep]) -> ColumnId {
    ColumnId {
        owner: PhysicalOwner::Typed {
            owner: TypedOwner::Instruction { path: path.to_vec() },
        },
        bundle_index: 0,
        coordinate_index: 0,
    }
}

fn branch_column(path: &[OwnerPathStep], coordinate_index: usize) -> ColumnId {
    ColumnId {
        owner: PhysicalOwner::Typed {
            owner: TypedOwner::Branch { path: path.to_vec() },
        },
        bundle_index: 0,
        coordinate_index,
    }
}

fn expected_step_result_columns(widths: &Widths) -> Vec<OwnedColumn> {
    let mut columns = Vec::with_capacity(widths.state + widths.running + widths.digest);
    columns.extend((0..widths.state).map(|coordinate_index| OwnedColumn {
        id: ColumnId {
            owner: PhysicalOwner::Typed {
                owner: TypedOwner::Instruction { path: vec![] },
            },
            bundle_index: 0,
            coordinate_index,
        },
        ownership: Ownership::Committed,
    }));
    columns.extend((0..widths.running).map(|coordinate_index| OwnedColumn {
        id: branch_column(&[OwnerPathStep::Rest, OwnerPathStep::Rest], coordinate_index),
        ownership: Ownership::Committed,
    }));
    columns.extend((0..widths.digest).map(|coordinate_index| OwnedColumn {
        id: ColumnId {
            owner: PhysicalOwner::Typed {
                owner: TypedOwner::Instruction {
                    path: vec![OwnerPathStep::Rest, OwnerPathStep::Rest, OwnerPathStep::Continuation],
                },
            },
            bundle_index: 0,
            coordinate_index,
        },
        ownership: Ownership::Public,
    }));
    columns
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
    let expected = segments
        .iter()
        .flat_map(|segment| std::iter::repeat_n(segment.ownership, segment.width));
    for (index, (column, ownership)) in columns.iter().zip(expected).enumerate() {
        if column.ownership != ownership {
            return Err(invalid(
                format!("{path}[{index}].ownership"),
                "does not match the result codec segment",
            ));
        }
    }
    Ok(())
}

fn emit_native_program(
    program: &NativeManifestProgram,
    values: &mut impl FnMut(&ColumnId) -> Option<F>,
) -> Result<NativeStepEmission, LeanNativeCcsEmissionError> {
    let allocations: Vec<_> = program
        .receipts
        .iter()
        .flat_map(|receipt| receipt.allocations.iter())
        .collect();
    let mut column_indices = HashMap::with_capacity(allocations.len());
    let mut ordered_values = Vec::with_capacity(allocations.len());
    let mut public_columns = Vec::new();
    let mut committed_columns = Vec::new();
    let mut auxiliary_columns = Vec::new();

    let mut append = |allocation: &OwnedColumn| -> Result<(), LeanNativeCcsEmissionError> {
        let index = ordered_values.len();
        let value = if allocation.id == program.one {
            F::ONE
        } else {
            values(&allocation.id).ok_or_else(|| LeanNativeCcsEmissionError::MissingValue {
                column: allocation.id.clone(),
            })?
        };
        column_indices.insert(allocation.id.clone(), index);
        ordered_values.push(value);
        match allocation.ownership {
            Ownership::Public => public_columns.push(allocation.id.clone()),
            Ownership::Committed => committed_columns.push(allocation.id.clone()),
            Ownership::Auxiliary => auxiliary_columns.push(allocation.id.clone()),
        }
        Ok(())
    };
    for ownership in [Ownership::Public, Ownership::Committed, Ownership::Auxiliary] {
        for allocation in allocations
            .iter()
            .filter(|allocation| allocation.ownership == ownership)
        {
            append(allocation)?;
        }
    }

    let rows = program.row_count();
    let columns = ordered_values.len();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut selector = Vec::with_capacity(rows);
    let mut row_index = 0usize;
    for receipt in &program.receipts {
        let selector_column =
            *column_indices
                .get(&receipt.selector)
                .ok_or_else(|| LeanNativeCcsEmissionError::UnknownColumn {
                    column: receipt.selector.clone(),
                })?;
        for row in &receipt.rows {
            append_terms(&mut a, row_index, &row.a, &column_indices)?;
            append_terms(&mut b, row_index, &row.b, &column_indices)?;
            append_terms(&mut c, row_index, &row.c, &column_indices)?;
            selector.push((row_index, selector_column, F::ONE));
            row_index += 1;
        }
    }
    debug_assert_eq!(row_index, rows);
    let structure = sparse_selected_r1cs_to_ccs(
        CcsMatrix::Csc(CscMat::from_triplets(a, rows, columns)),
        CcsMatrix::Csc(CscMat::from_triplets(b, rows, columns)),
        CcsMatrix::Csc(CscMat::from_triplets(c, rows, columns)),
        CcsMatrix::Csc(CscMat::from_triplets(selector, rows, columns)),
    )
    .map_err(|error| LeanNativeCcsEmissionError::InvalidStructure(error.to_string()))?;
    let public_len = public_columns.len();
    Ok(NativeStepEmission {
        structure,
        public_values: ordered_values[..public_len].to_vec(),
        witness_values: ordered_values[public_len..].to_vec(),
        column_indices,
        public_columns,
        committed_columns,
        auxiliary_columns,
    })
}

fn emit_phi81_program(
    program: &NativeManifestProgram,
    descriptor: TerminalR1csDescriptor,
    values: &mut impl FnMut(&ColumnId) -> Option<F>,
) -> Result<NativePhi81StepEmission, LeanNativeCcsEmissionError> {
    let allocations: Vec<_> = program
        .receipts
        .iter()
        .flat_map(|receipt| receipt.allocations.iter())
        .collect();
    let logical_width = allocations.len();
    if logical_width != descriptor.logical_width {
        return Err(LeanNativeCcsEmissionError::InvalidStructure(
            "validated terminal descriptor no longer matches Step allocations".into(),
        ));
    }

    let mut column_indices = HashMap::with_capacity(logical_width);
    let mut assignment = Vec::with_capacity(
        phi81_carrier_width(logical_width)
            .map_err(|error| LeanNativeCcsEmissionError::InvalidStructure(error.to_string()))?,
    );
    for allocation in allocations {
        let index = assignment.len();
        let value = if allocation.id == program.one {
            F::ONE
        } else {
            values(&allocation.id).ok_or_else(|| LeanNativeCcsEmissionError::MissingValue {
                column: allocation.id.clone(),
            })?
        };
        column_indices.insert(allocation.id.clone(), index);
        assignment.push(value);
    }

    let carrier_width = phi81_carrier_width(logical_width)
        .map_err(|error| LeanNativeCcsEmissionError::InvalidStructure(error.to_string()))?;
    assignment.resize(carrier_width, F::ZERO);
    let row_domain = 1usize
        .checked_shl(descriptor.row_variables as u32)
        .ok_or_else(|| LeanNativeCcsEmissionError::InvalidStructure("row domain overflow".into()))?;
    let source_rows = program.row_count();
    if source_rows > row_domain {
        return Err(LeanNativeCcsEmissionError::InvalidStructure(
            "Step rows exceed the validated row domain".into(),
        ));
    }

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut selector = Vec::with_capacity(source_rows);
    let mut row_index = 0usize;
    for receipt in &program.receipts {
        let selector_column =
            *column_indices
                .get(&receipt.selector)
                .ok_or_else(|| LeanNativeCcsEmissionError::UnknownColumn {
                    column: receipt.selector.clone(),
                })?;
        for row in &receipt.rows {
            append_terms(&mut a, row_index, &row.a, &column_indices)?;
            append_terms(&mut b, row_index, &row.b, &column_indices)?;
            append_terms(&mut c, row_index, &row.c, &column_indices)?;
            selector.push((row_index, selector_column, F::ONE));
            row_index += 1;
        }
    }
    debug_assert_eq!(row_index, source_rows);
    let structure = sparse_selected_r1cs_to_ccs(
        CcsMatrix::Csc(CscMat::from_triplets(a, row_domain, carrier_width)),
        CcsMatrix::Csc(CscMat::from_triplets(b, row_domain, carrier_width)),
        CcsMatrix::Csc(CscMat::from_triplets(c, row_domain, carrier_width)),
        CcsMatrix::Csc(CscMat::from_triplets(selector, row_domain, carrier_width)),
    )
    .map_err(|error| LeanNativeCcsEmissionError::InvalidStructure(error.to_string()))?;
    let public_width = descriptor.public_ring_columns * PHI81_RING_DEGREE;
    Ok(NativePhi81StepEmission {
        structure,
        assignment,
        column_indices,
        logical_width,
        public_width,
    })
}

fn append_terms(
    target: &mut Vec<(usize, usize, F)>,
    row: usize,
    terms: &[super::lean_manifest::ManifestTerm],
    indices: &HashMap<ColumnId, usize>,
) -> Result<(), LeanNativeCcsEmissionError> {
    for term in terms {
        let column = *indices
            .get(&term.column)
            .ok_or_else(|| LeanNativeCcsEmissionError::UnknownColumn {
                column: term.column.clone(),
            })?;
        target.push((row, column, F::from_u64(term.coefficient)));
    }
    Ok(())
}
