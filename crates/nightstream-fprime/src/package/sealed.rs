//! Phase-local loading of one Lean-authored per-application envelope.
//!
//! This boundary checks the physical package, matrix program, and exact
//! application plan under one structural identity. It is not the final
//! production verifier path, which must also bind the commitment setup and
//! verification key.

use std::ops::Range;

use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use serde_json::Value;

use super::assignment_transport;
use super::matrix_program::{MatrixProgram, RowForms, MEANINGFUL_PORTS};
use super::{
    relation_identifier, validate_per_application_package_schema, Layout, LoadedAssignmentPlan, LoadedPackage,
    LoadedTerminalLayout, LogicalAssignment, PackageError, PackageR1cs, PiCcsV1_1EncodedInputs,
    PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs, RawPackage,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
};
use crate::identity::{
    stage1_verifier_binding, value_preimage_words, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
use crate::Stage1VerifierBinding;
use crate::WitnessAssignment;

const SEALED_PACKAGE_SCHEMA: u64 = 6;
const INNER_PACKAGE_SCHEMA: u64 = 8;
const MATRIX_COUNT: usize = 14;
const APPLICATION_PLAN_SCHEMA: u64 = 1;
const APPLICATION_STATE_WORDS: usize = 4;
const NEXT_PREIMAGE_ROW_COUNT: usize = 5;
pub(super) const APPLICATION_WITNESS_ROLE: u64 = 17;
pub(super) const APPLICATION_LOCAL_ROLE: u64 = 18;
const CIRCUIT_WITNESS_BATCHES: usize = 10;
const CIRCUIT_WITNESS_INSTRUCTIONS: usize = 11;
const CIRCUIT_ASSERTION_ROWS: usize = 12;

#[derive(Debug, Deserialize)]
struct RawSealedPackage(u64, RawPackage, Value, Value, Value, RawRowRange, u64);

#[derive(Debug, Deserialize)]
struct RawRowRange(u64, u64);

#[derive(Debug, Deserialize)]
struct RawApplicationPlan(
    u64,
    u64,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    u64,
    u64,
    u64,
    u64,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
);

/// One canonical sparse entry in a Lean-derived SuperNeo matrix row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalMatrixEntry {
    column: usize,
    coefficient: u64,
}

impl LogicalMatrixEntry {
    pub fn column(&self) -> usize {
        self.column
    }

    pub fn coefficient(&self) -> u64 {
        self.coefficient
    }
}

/// The fourteen matrix forms at one Boolean-row ordinal. Slot 13 is always
/// the canonical zero form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalMatrixRow {
    matrices: [Vec<LogicalMatrixEntry>; MATRIX_COUNT],
}

/// Exact application ABI and owned physical suffix decoded from Lean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedApplicationPlan {
    witness_word_count: usize,
    input_columns: [usize; APPLICATION_STATE_WORDS],
    witness_columns: Vec<usize>,
    output_columns: [usize; APPLICATION_STATE_WORDS],
    private_start: usize,
    private_count: usize,
    row_start: usize,
    row_count: usize,
}

impl LoadedApplicationPlan {
    pub fn witness_word_count(&self) -> usize {
        self.witness_word_count
    }

    pub fn input_columns(&self) -> [usize; APPLICATION_STATE_WORDS] {
        self.input_columns
    }

    pub fn witness_columns(&self) -> &[usize] {
        &self.witness_columns
    }

    pub fn output_columns(&self) -> [usize; APPLICATION_STATE_WORDS] {
        self.output_columns
    }

    pub fn private_range(&self) -> std::ops::Range<usize> {
        self.private_start..self.private_start + self.private_count
    }

    pub fn row_range(&self) -> std::ops::Range<usize> {
        self.row_start..self.row_start + self.row_count
    }
}

impl LogicalMatrixRow {
    pub fn matrix(&self, slot: usize) -> Option<&[LogicalMatrixEntry]> {
        self.matrices.get(slot).map(Vec::as_slice)
    }
}

fn logical_matrix_row(forms: RowForms) -> Result<LogicalMatrixRow, PackageError> {
    let mut matrices = forms
        .into_iter()
        .map(|form| {
            form.into_entries()
                .into_iter()
                .map(|entry| LogicalMatrixEntry {
                    column: entry.column,
                    coefficient: entry.coefficient.as_canonical_u64(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    debug_assert_eq!(matrices.len(), MEANINGFUL_PORTS);
    matrices.push(Vec::new());
    Ok(LogicalMatrixRow {
        matrices: matrices
            .try_into()
            .map_err(|_| PackageError::Invalid("logical matrix port count"))?,
    })
}

/// A structurally identity-bound per-application package. Final production
/// acceptance remains unavailable until the concrete key binding is checked.
#[derive(Clone, Debug)]
pub struct LoadedPerApplicationPackage {
    circuit: LoadedPackage,
    matrix_program: MatrixProgram,
    application: LoadedApplicationPlan,
    assignment_plan: LoadedAssignmentPlan,
    next_preimage_rows: std::ops::Range<usize>,
    logical_public_input_count: usize,
    structural_identifier: [u64; 4],
    relation_value_words: Vec<u64>,
    application_words: Vec<u64>,
}

impl LoadedPerApplicationPackage {
    pub fn structural_identifier(&self) -> [u64; 4] {
        self.structural_identifier
    }

    pub fn row_count(&self) -> usize {
        self.circuit.relation.row_count()
    }

    pub fn physical_row_count(&self) -> usize {
        self.circuit.row_count()
    }

    pub fn logical_column_count(&self) -> usize {
        self.circuit.relation.column_count()
    }

    pub fn logical_public_input_count(&self) -> usize {
        self.logical_public_input_count
    }

    pub fn ccs_relation(&self) -> &super::PackageCcsRelation {
        self.circuit.ccs_relation()
    }

    /// Construct the matrix-content-free CCS header for the separately
    /// verified cache. Every dimension and polynomial term comes from this
    /// identity-checked Lean package.
    pub fn ccs_structure_header(&self) -> Result<CcsStructure<Goldilocks>, PackageError> {
        let relation = self.ccs_relation();
        let terms = relation
            .terms()
            .iter()
            .map(|term| {
                let exps = term
                    .exponents()
                    .iter()
                    .copied()
                    .map(|exponent| {
                        u32::try_from(exponent)
                            .map_err(|_| PackageError::Invalid("CCS polynomial exponent exceeds u32"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Term {
                    coeff: Goldilocks::from_u64(term.coefficient()),
                    exps,
                })
            })
            .collect::<Result<Vec<_>, PackageError>>()?;
        let polynomial = SparsePoly::new(relation.matrix_sources().len(), terms);
        let structure = CcsStructure::new_verifier_artifact_header(
            relation.row_count(),
            relation.column_count(),
            relation.matrix_sources().len(),
            polynomial,
        )
        .map_err(|_| PackageError::Invalid("logical CCS relation header"))?;
        if (structure.max_degree() as usize).checked_add(1) != Some(relation.degree_bound()) {
            return Err(PackageError::Invalid("CCS polynomial strict degree bound"));
        }
        Ok(structure)
    }

    pub fn private_input_count(&self) -> usize {
        self.circuit.private_input_count()
    }

    pub fn public_input_count(&self) -> usize {
        self.circuit.public_column_count()
    }

    pub fn total_column_count(&self) -> usize {
        self.circuit.total_column_count()
    }

    pub fn application(&self) -> &LoadedApplicationPlan {
        &self.application
    }

    pub fn assignment_plan(&self) -> &LoadedAssignmentPlan {
        &self.assignment_plan
    }

    pub fn terminal(&self) -> Option<&LoadedTerminalLayout> {
        self.circuit.terminal()
    }

    pub fn next_preimage_row_range(&self) -> std::ops::Range<usize> {
        self.next_preimage_rows.clone()
    }

    /// Recompute the complete final package and verification-key binding
    /// from this identity-checked package and the fixed production setup.
    pub fn production_verifier_binding(&self) -> Result<Stage1VerifierBinding, PackageError> {
        stage1_verifier_binding(
            self.structural_identifier,
            &self.relation_value_words,
            &self.application_words,
        )
    }

    /// Execute only the witness program carried by this identity-bound
    /// package. The four application message words are caller-owned inputs;
    /// all application-local columns are generated by the package IR.
    pub fn execute_witness(
        &self,
        private_inputs: &[u64],
        public_values: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        self.circuit.execute_witness(private_inputs, public_values)
    }

    /// Lower one package-produced physical witness through the exact
    /// Lean-authored final-assignment transport retained by this package.
    pub fn execute_logical_assignment(&self, physical: &WitnessAssignment) -> Result<LogicalAssignment, PackageError> {
        self.assignment_plan.execute(&self.circuit.layout, physical)
    }

    /// Encode the typed PiCCS input through this verifier-owned package.
    ///
    /// The generic prefix validates the caller's package context. This final
    /// application package replaces the public context slot with the digest
    /// recomputed from the complete production package binding.
    pub fn encode_pi_ccs_v1_1_inputs(
        &self,
        inputs: &PiCcsV1_1PackageInputs,
    ) -> Result<PiCcsV1_1EncodedInputs, PackageError> {
        let encoded = self.circuit.encode_pi_ccs_v1_1_inputs(inputs)?;
        let mut public_values = encoded.public_values().to_vec();
        let verifier_context = self
            .production_verifier_binding()?
            .verifier_context()
            .digest();
        let context_start = public_values
            .len()
            .checked_sub(verifier_context.len())
            .ok_or(PackageError::Invalid("Stage 1 verifier-context public slot"))?;
        let context_end = context_start + verifier_context.len();
        public_values
            .get_mut(context_start..context_end)
            .ok_or(PackageError::Invalid("Stage 1 verifier-context public slot"))?
            .copy_from_slice(&verifier_context);
        Ok(PiCcsV1_1EncodedInputs::from_parts(
            encoded.private_values().to_vec(),
            public_values,
        ))
    }

    /// Encode every caller-owned Stage 1 input in the exact package order.
    pub fn encode_stage1_v1_1_inputs(
        &self,
        pi_ccs: &PiCcsV1_1PackageInputs,
        pi_dec: &PiDecV1_1PackageInputs,
        application_witness: &[u64],
    ) -> Result<PiCcsV1_1EncodedInputs, PackageError> {
        if application_witness.len() != self.application.witness_word_count
            || application_witness
                .iter()
                .any(|value| *value >= Goldilocks::ORDER_U64)
        {
            return Err(PackageError::Invalid("application witness"));
        }
        let encoded = self.encode_pi_ccs_v1_1_inputs(pi_ccs)?;
        let mut private_values = encoded.private_values().to_vec();
        pi_dec.append_private_values(&mut private_values);
        private_values.extend_from_slice(application_witness);
        if private_values.len() != self.circuit.private_input_count() {
            return Err(PackageError::Invalid("Stage 1 v1_1 encoded private-input length"));
        }
        Ok(PiCcsV1_1EncodedInputs::from_parts(
            private_values,
            encoded.public_values().to_vec(),
        ))
    }

    /// Execute the package witness program from typed Stage 1 inputs.
    pub fn execute_stage1_v1_1_witness(
        &self,
        pi_ccs: &PiCcsV1_1PackageInputs,
        pi_dec: &PiDecV1_1PackageInputs,
        application_witness: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        let encoded = self.encode_stage1_v1_1_inputs(pi_ccs, pi_dec, application_witness)?;
        self.circuit
            .execute_witness(encoded.private_values(), encoded.public_values())
    }

    /// Decode the PiCCS output segments through this verifier-owned package.
    pub fn pi_ccs_v1_1_output_evaluations(
        &self,
        private_inputs: &[u64],
    ) -> Result<PiCcsV1_1OutputEvaluations, PackageError> {
        self.circuit.pi_ccs_v1_1_output_evaluations(private_inputs)
    }

    /// Build the exact final padded A/B/C matrices from this sealed package.
    pub fn r1cs_matrices(&self) -> Result<PackageR1cs, PackageError> {
        self.circuit.r1cs_matrices()
    }

    /// Execute and validate every live Lean-authored logical matrix row.
    ///
    /// This avoids constructing a second public row representation while it
    /// checks canonical entry order, nonzero coefficients, column bounds, and
    /// the implicit zero matrix. The returned counts are deterministic.
    pub fn validate_all_matrix_rows(&self) -> Result<[u64; MATRIX_COUNT], PackageError> {
        let meaningful = self
            .matrix_program
            .validate_all_rows(self.logical_column_count(), &|source| self.circuit.source_row(source))?;
        let mut counts = [0u64; MATRIX_COUNT];
        counts[..MEANINGFUL_PORTS].copy_from_slice(&meaningful);
        Ok(counts)
    }

    /// Visit an active logical-row range in ascending Lean-authored order.
    /// Boolean rows after `row_count()` are implicit zero padding and are not
    /// accepted by this active-row interface.
    pub fn visit_matrix_rows(
        &self,
        rows: Range<usize>,
        mut visit: impl FnMut(usize, LogicalMatrixRow) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        if rows.start > rows.end || rows.end > self.row_count() {
            return Err(PackageError::Invalid("logical matrix row range"));
        }
        self.matrix_program.visit_rows(
            self.logical_column_count(),
            rows.start,
            rows.end,
            &|source| self.circuit.source_row(source),
            |ordinal, forms| visit(ordinal, logical_matrix_row(forms)?),
        )
    }
}

/// Strictly decode one canonical Lean sealed value and pin its complete
/// circuit-and-matrix structural identity. This is a phase-local conformance
/// boundary, not the final verifier-key binding.
pub fn load_per_application_package(
    bytes: &[u8],
    expected_structural_identifier: [u64; 4],
) -> Result<LoadedPerApplicationPackage, PackageError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let mut canonical = serde_json::to_vec(&value)?;
    canonical.push(b'\n');
    if bytes != canonical {
        return Err(PackageError::NonCanonicalBytes);
    }

    let computed = relation_identifier(&value)?;
    if computed != expected_structural_identifier {
        return Err(PackageError::ExpectedIdentityMismatch {
            expected: expected_structural_identifier,
            computed,
        });
    }

    let circuit_value = value
        .as_array()
        .and_then(|sealed| sealed.get(1))
        .ok_or(PackageError::Invalid("sealed circuit package"))?
        .clone();
    let RawSealedPackage(
        schema,
        raw_circuit,
        raw_matrix,
        raw_application,
        raw_assignment_plan,
        raw_next_preimage,
        raw_logical_public_input_count,
    ): RawSealedPackage = serde_json::from_value(value)?;
    if schema != SEALED_PACKAGE_SCHEMA {
        return Err(PackageError::Invalid("sealed package schema version"));
    }
    let relation_value = circuit_value
        .as_array()
        .and_then(|circuit| circuit.get(4))
        .ok_or(PackageError::Invalid("sealed relation authority"))?;
    let relation_value_words = value_preimage_words(relation_value)?;
    let application_words = value_preimage_words(&raw_application)?;
    let circuit = validate_per_application_package_schema(raw_circuit, computed, INNER_PACKAGE_SCHEMA)?;
    let matrix_program = MatrixProgram::decode(&raw_matrix)?;
    matrix_program.validate(circuit.layout.row_count)?;
    if matrix_program.row_count()? != circuit.relation.row_count() {
        return Err(PackageError::Invalid("matrix program relation row count"));
    }
    let next_preimage_rows = decode_next_preimage_range(raw_next_preimage, &circuit.layout)?;
    let application = decode_application_plan(&raw_application, &circuit_value, &circuit.layout, &next_preimage_rows)?;
    validate_next_preimage_assertion_suffix(&circuit_value, &next_preimage_rows)?;
    let logical_public_input_count = usize::try_from(raw_logical_public_input_count)
        .map_err(|_| PackageError::Invalid("logical public input count"))?;
    if logical_public_input_count != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
        || logical_public_input_count > circuit.relation.column_count()
    {
        return Err(PackageError::Invalid("logical public input count"));
    }
    let assignment_plan = assignment_transport::decode(
        &raw_assignment_plan,
        circuit.layout.total_column_count,
        logical_public_input_count,
        circuit.relation.column_count(),
    )?;

    Ok(LoadedPerApplicationPackage {
        circuit,
        matrix_program,
        application,
        assignment_plan,
        next_preimage_rows,
        logical_public_input_count,
        structural_identifier: computed,
        relation_value_words,
        application_words,
    })
}

/// Load the sole verifier-owned Stage 1 package for
/// `Poseidon2HashChainV1`. No prover-selected identity or setup enters this
/// path.
pub fn load_poseidon2_hash_chain_v1_package(bytes: &[u8]) -> Result<LoadedPerApplicationPackage, PackageError> {
    let package = load_per_application_package(bytes, POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER)?;
    let binding = package.production_verifier_binding()?;
    if binding.package_identity() != POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY {
        return Err(PackageError::ExpectedPackageIdentityMismatch {
            expected: POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
            computed: binding.package_identity(),
        });
    }
    if binding.verification_key_digest() != POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST {
        return Err(PackageError::ExpectedVerificationKeyBindingMismatch {
            expected: POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
            computed: binding.verification_key_digest(),
        });
    }
    Ok(package)
}

fn decode_next_preimage_range(raw: RawRowRange, layout: &Layout) -> Result<std::ops::Range<usize>, PackageError> {
    let RawRowRange(row_start, row_count) = raw;
    let row_start = word_to_usize(row_start, "next preimage row start")?;
    let row_count = word_to_usize(row_count, "next preimage row count")?;
    let row_end = row_start
        .checked_add(row_count)
        .ok_or(PackageError::Invalid("next preimage row range"))?;
    if row_count != NEXT_PREIMAGE_ROW_COUNT || row_end != layout.row_count {
        return Err(PackageError::Invalid("next preimage row range"));
    }
    Ok(row_start..row_end)
}

fn decode_application_plan(
    value: &Value,
    circuit_value: &Value,
    layout: &Layout,
    next_preimage_rows: &std::ops::Range<usize>,
) -> Result<LoadedApplicationPlan, PackageError> {
    let RawApplicationPlan(
        schema,
        witness_word_count,
        input_columns,
        witness_columns,
        output_columns,
        private_start,
        private_count,
        row_start,
        row_count,
        hash_chains,
        permutation_invocations,
        compact_templates,
        compact_invocations,
        witness_batches,
        witness_instructions,
        assertion_rows,
    ) = serde_json::from_value(value.clone())?;
    if schema != APPLICATION_PLAN_SCHEMA {
        return Err(PackageError::Invalid("application plan schema version"));
    }
    if !hash_chains.is_empty()
        || !permutation_invocations.is_empty()
        || !compact_templates.is_empty()
        || !compact_invocations.is_empty()
    {
        return Err(PackageError::Invalid("application plan row family"));
    }

    let witness_word_count = word_to_usize(witness_word_count, "application witness width")?;
    let input_columns = fixed_columns(input_columns, "application input width")?;
    let witness_columns = columns(witness_columns, "application witness column")?;
    let output_columns = fixed_columns(output_columns, "application output width")?;
    let private_start = word_to_usize(private_start, "application private start")?;
    let private_count = word_to_usize(private_count, "application private count")?;
    let row_start = word_to_usize(row_start, "application row start")?;
    let row_count = word_to_usize(row_count, "application row count")?;

    if witness_columns.len() != witness_word_count {
        return Err(PackageError::Invalid("application witness width"));
    }
    let private_end = private_start
        .checked_add(private_count)
        .ok_or(PackageError::Invalid("application private range"))?;
    let row_end = row_start
        .checked_add(row_count)
        .ok_or(PackageError::Invalid("application row range"))?;
    if private_end > layout.constant_column || row_end != next_preimage_rows.start {
        return Err(PackageError::Invalid("application plan range"));
    }

    let witness_segment = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_WITNESS_ROLE)
        .ok_or(PackageError::Invalid("application witness segment"))?;
    let local_segment = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_LOCAL_ROLE)
        .ok_or(PackageError::Invalid("application local segment"))?;
    let expected_witness_columns = (witness_segment.start
        ..witness_segment
            .start
            .checked_add(witness_segment.length)
            .ok_or(PackageError::Invalid("application witness segment"))?)
        .collect::<Vec<_>>();
    if witness_segment.length != witness_word_count
        || witness_columns != expected_witness_columns
        || local_segment.start != private_start
        || local_segment.length != private_count
        || input_columns
            .iter()
            .chain(&output_columns)
            .any(|column| *column >= witness_segment.start)
    {
        return Err(PackageError::Invalid("application plan column ownership"));
    }

    let ordinary_row_count = witness_instructions
        .len()
        .checked_add(assertion_rows.len())
        .ok_or(PackageError::Invalid("application ordinary row count"))?;
    if ordinary_row_count != row_count {
        return Err(PackageError::Invalid("application ordinary row count"));
    }
    let circuit = circuit_value
        .as_array()
        .ok_or(PackageError::Invalid("sealed circuit package"))?;
    if !is_suffix(circuit, CIRCUIT_WITNESS_BATCHES, &witness_batches)
        || !is_suffix(circuit, CIRCUIT_WITNESS_INSTRUCTIONS, &witness_instructions)
        || !is_suffix_before(
            circuit,
            CIRCUIT_ASSERTION_ROWS,
            &assertion_rows,
            next_preimage_rows.len(),
        )
    {
        return Err(PackageError::Invalid("application plan package suffix"));
    }

    Ok(LoadedApplicationPlan {
        witness_word_count,
        input_columns,
        witness_columns,
        output_columns,
        private_start,
        private_count,
        row_start,
        row_count,
    })
}

fn fixed_columns(raw: Vec<u64>, location: &'static str) -> Result<[usize; APPLICATION_STATE_WORDS], PackageError> {
    columns(raw, location)?
        .try_into()
        .map_err(|_| PackageError::Invalid(location))
}

fn columns(raw: Vec<u64>, location: &'static str) -> Result<Vec<usize>, PackageError> {
    raw.into_iter()
        .map(|column| word_to_usize(column, location))
        .collect()
}

fn is_suffix(circuit: &[Value], field: usize, expected: &[Value]) -> bool {
    circuit
        .get(field)
        .and_then(Value::as_array)
        .is_some_and(|actual| actual.ends_with(expected))
}

fn is_suffix_before(circuit: &[Value], field: usize, expected: &[Value], trailing: usize) -> bool {
    circuit
        .get(field)
        .and_then(Value::as_array)
        .and_then(|actual| actual.len().checked_sub(trailing).map(|end| &actual[..end]))
        .is_some_and(|prefix| prefix.ends_with(expected))
}

fn validate_next_preimage_assertion_suffix(
    circuit_value: &Value,
    expected: &std::ops::Range<usize>,
) -> Result<(), PackageError> {
    let rows = circuit_value
        .as_array()
        .and_then(|circuit| circuit.get(CIRCUIT_ASSERTION_ROWS))
        .and_then(Value::as_array)
        .ok_or(PackageError::Invalid("next preimage package suffix"))?;
    let suffix_start = rows
        .len()
        .checked_sub(expected.len())
        .ok_or(PackageError::Invalid("next preimage package suffix"))?;
    let valid = rows[suffix_start..]
        .iter()
        .zip(expected.clone())
        .all(|(row, expected_index)| {
            row.as_array()
                .and_then(|fields| fields.first())
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                == Some(expected_index)
        });
    if !valid {
        return Err(PackageError::Invalid("next preimage package suffix"));
    }
    Ok(())
}

fn word_to_usize(value: u64, location: &'static str) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| PackageError::Invalid(location))
}

#[cfg(test)]
#[path = "../../tests/unit/sealed.rs"]
mod sealed_tests;

#[cfg(test)]
#[path = "../../tests/unit/pi_ccs_prefix_assignment.rs"]
mod pi_ccs_prefix_assignment_tests;

#[cfg(test)]
#[path = "../../tests/unit/pilot_prefix_assignment.rs"]
mod pilot_prefix_assignment_tests;
