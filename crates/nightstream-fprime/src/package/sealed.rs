//! Phase-local loading of one Lean-authored per-application envelope.
//!
//! This boundary checks the physical package, matrix program, and exact
//! application plan under one structural identity. It is not the final
//! production verifier path, which must also bind the commitment setup and
//! verification key.

use std::{
    ops::{ControlFlow, Range},
    sync::Arc,
};

use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use serde_json::Value;

use super::assignment_transport;
use super::matrix_program::{MatrixProgram, MatrixRun, RowForms, MEANINGFUL_PORTS};
use super::native_application::PreparedApplication;
use super::wire::RawPackage;
use super::{
    canonical_field, relation_identifier, require_input_length, word_to_usize, IdentityKind, Layout,
    LoadedAssignmentPlan, LoadedPackage, LoadedTerminalLayout, LogicalAssignment, PackageError, PackageR1cs,
    PiCcsV1_2EncodedInputs, PiCcsV1_2OutputEvaluations, PiCcsV1_2PackageInputs, PiDecV1_2PackageInputs,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS,
};
use crate::application_records::{ApplicationRecords, PrivateSnapshot};
use crate::identity::{
    native_application_identity, native_relation_identifier, stage1_verifier_binding, value_preimage_words,
    visit_native_application_words, ApplicationIdentity, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
use crate::Stage1VerifierBinding;
use crate::WitnessAssignment;

const SEALED_PACKAGE_SCHEMA: u64 = 6;
pub(super) const INNER_PACKAGE_SCHEMA: u64 = 8;
const MATRIX_COUNT: usize = super::PI_CCS_V1_2_MATRIX_COUNT;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalMatrixRow {
    matrices: [Vec<LogicalMatrixEntry>; MATRIX_COUNT],
}

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
    let matrices = forms
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
    Ok(LogicalMatrixRow {
        matrices: matrices
            .try_into()
            .map_err(|_| PackageError::Malformed("logical matrix port count"))?,
    })
}

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
    application_identity: ApplicationIdentity,
    application_source: ApplicationSource,
}

/// Where the application words come from.
#[derive(Clone, Debug)]
enum ApplicationSource {
    /// The application value of a sealed JSON envelope, as preimage words.
    Words(Vec<u64>),
    /// Native application records and the fixed envelope they were sealed with.
    Native {
        application: Arc<PreparedApplication>,
        source: Arc<PrivateSnapshot>,
    },
}

impl LoadedPerApplicationPackage {
    pub fn write_prepared(&self, output: impl std::io::Write) -> Result<(), PackageError> {
        let ApplicationSource::Native { application, source } = &self.application_source else {
            return Err(PackageError::Malformed("package has no prepared native source"));
        };
        super::prepared::write(
            output,
            source,
            application.records_arc(),
            self.structural_identifier,
            &self.application_identity,
        )
    }

    pub fn application_records(&self) -> Option<&Arc<ApplicationRecords>> {
        match &self.application_source {
            ApplicationSource::Words(_) => None,
            ApplicationSource::Native { application, .. } => Some(application.records_arc()),
        }
    }

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
                            .map_err(|_| PackageError::TooLarge("CCS polynomial exponent exceeds u32"))
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
        .and_then(|header| header.with_domain_variables(relation.cube_variables()))
        .map_err(|_| PackageError::Malformed("logical CCS relation header"))?;
        if (structure.max_degree() as usize).checked_add(1) != Some(relation.degree_bound()) {
            return Err(PackageError::Malformed("CCS polynomial strict degree bound"));
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

    pub fn production_verifier_binding(&self) -> Result<Stage1VerifierBinding, PackageError> {
        stage1_verifier_binding(
            self.structural_identifier,
            self.logical_column_count(),
            &self.relation_value_words,
            &self.application_identity,
            super::matrix_program::formula_digest()?,
        )
    }

    pub fn visit_application_words(
        &self,
        visit: &mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        match &self.application_source {
            ApplicationSource::Words(words) => visit(words),
            ApplicationSource::Native { application, .. } => visit_native_application_words(application, visit),
        }
    }

    pub fn execute_witness(
        &self,
        private_inputs: &[u64],
        public_values: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        self.circuit.execute_witness(private_inputs, public_values)
    }

    pub fn execute_logical_assignment(&self, physical: &WitnessAssignment) -> Result<LogicalAssignment, PackageError> {
        self.assignment_plan.execute(&self.circuit.layout, physical)
    }

    pub fn encode_pi_ccs_v1_2_inputs(
        &self,
        inputs: &PiCcsV1_2PackageInputs,
    ) -> Result<PiCcsV1_2EncodedInputs, PackageError> {
        let encoded = self
            .circuit
            .encode_pi_ccs_v1_2_inputs(inputs, self.structural_identifier)?;
        let mut public_values = encoded.public_values().to_vec();
        let verifier_context = self
            .production_verifier_binding()?
            .verifier_context()
            .digest();
        let context_start = public_values
            .len()
            .checked_sub(verifier_context.len())
            .ok_or(PackageError::Malformed("Stage 1 verifier-context public slot"))?;
        let context_end = context_start + verifier_context.len();
        public_values
            .get_mut(context_start..context_end)
            .ok_or(PackageError::Malformed("Stage 1 verifier-context public slot"))?
            .copy_from_slice(&verifier_context);
        Ok(PiCcsV1_2EncodedInputs::from_parts(
            encoded.private_values().to_vec(),
            public_values,
        ))
    }

    pub fn encode_stage1_v1_2_inputs(
        &self,
        pi_ccs: &PiCcsV1_2PackageInputs,
        pi_dec: &PiDecV1_2PackageInputs,
        application_witness: &[u64],
    ) -> Result<PiCcsV1_2EncodedInputs, PackageError> {
        require_input_length(
            "application witness length",
            self.application.witness_word_count,
            application_witness.len(),
        )?;
        for value in application_witness {
            canonical_field(*value, "application witness")?;
        }
        let encoded = self.encode_pi_ccs_v1_2_inputs(pi_ccs)?;
        let mut private_values = encoded.private_values().to_vec();
        pi_dec.append_private_values(&mut private_values);
        private_values.extend_from_slice(application_witness);
        if private_values.len() != self.circuit.private_input_count() {
            return Err(PackageError::Malformed("Stage 1 v1_2 encoded private-input length"));
        }
        Ok(PiCcsV1_2EncodedInputs::from_parts(
            private_values,
            encoded.public_values().to_vec(),
        ))
    }

    pub fn execute_stage1_v1_2_witness(
        &self,
        pi_ccs: &PiCcsV1_2PackageInputs,
        pi_dec: &PiDecV1_2PackageInputs,
        application_witness: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        let encoded = self.encode_stage1_v1_2_inputs(pi_ccs, pi_dec, application_witness)?;
        self.circuit
            .execute_witness(encoded.private_values(), encoded.public_values())
    }

    pub fn execute_stage1_v1_2_witness_with_application_values(
        &self,
        pi_ccs: &PiCcsV1_2PackageInputs,
        pi_dec: &PiDecV1_2PackageInputs,
        application_witness: &[u64],
        application_values: &[Goldilocks],
    ) -> Result<WitnessAssignment, PackageError> {
        let encoded = self.encode_stage1_v1_2_inputs(pi_ccs, pi_dec, application_witness)?;
        self.circuit.execute_witness_with_application(
            encoded.private_values(),
            encoded.public_values(),
            Some(application_values),
        )
    }

    pub fn pi_ccs_v1_2_output_evaluations(
        &self,
        private_inputs: &[u64],
    ) -> Result<PiCcsV1_2OutputEvaluations, PackageError> {
        self.circuit.pi_ccs_v1_2_output_evaluations(private_inputs)
    }

    pub fn r1cs_matrices(&self) -> Result<PackageR1cs, PackageError> {
        self.circuit.r1cs_matrices()
    }

    pub fn validate_all_matrix_rows(&self) -> Result<[u64; MATRIX_COUNT], PackageError> {
        let meaningful = self
            .matrix_program
            .validate_all_rows(self.logical_column_count(), &|source| self.circuit.source_row(source))?;
        let mut counts = [0u64; MATRIX_COUNT];
        counts[..MEANINGFUL_PORTS].copy_from_slice(&meaningful);
        Ok(counts)
    }

    pub fn visit_matrix_rows(
        &self,
        rows: Range<usize>,
        mut visit: impl FnMut(usize, LogicalMatrixRow) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        if rows.start > rows.end || rows.end > self.row_count() {
            return Err(PackageError::Input("logical matrix row range"));
        }
        self.matrix_program.visit_rows(
            self.logical_column_count(),
            rows.start,
            rows.end,
            &|source| self.circuit.source_row(source),
            |ordinal, forms| visit(ordinal, logical_matrix_row(forms)?),
        )
    }

    pub fn visit_matrix_runs_until(
        &self,
        rows: Range<usize>,
        mut visit: impl FnMut(usize, [&[MatrixRun]; MATRIX_COUNT]) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        if rows.start > rows.end || rows.end > self.row_count() {
            return Err(PackageError::Input("logical matrix row range"));
        }
        self.matrix_program.visit_rows_until(
            self.logical_column_count(),
            rows.start,
            rows.end,
            &|source| self.circuit.source_row(source),
            |ordinal, forms| {
                visit(
                    ordinal,
                    std::array::from_fn(|matrix| forms.get(matrix).copied().unwrap_or(&[])),
                )
            },
        )
    }
}

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
        return Err(PackageError::IdentityMismatch {
            kind: IdentityKind::Relation,
            expected: expected_structural_identifier,
            computed,
        });
    }

    decode_per_application_value(&value, computed)
}

pub fn load_prepared_application_records(
    fixed_envelope: Value,
    records: Arc<ApplicationRecords>,
) -> Result<LoadedPerApplicationPackage, PackageError> {
    let decoded = decode_native_application_records(&fixed_envelope, records)?;
    let computed = native_relation_identifier(&fixed_envelope, decoded.application())?;
    let application_identity = native_application_identity(decoded.application())?;
    let source = super::prepared::snapshot(&fixed_envelope)?;
    Ok(decoded.bind(computed, application_identity, source))
}

/// The checked parts of a sealed envelope that both application sources share.
struct SealedEnvelope {
    circuit: LoadedPackage,
    matrix_program: MatrixProgram,
    application: LoadedApplicationPlan,
    assignment_plan: LoadedAssignmentPlan,
    next_preimage_rows: Range<usize>,
    logical_public_input_count: usize,
    relation_value_words: Vec<u64>,
}

impl SealedEnvelope {
    fn bind(
        self,
        structural_identifier: [u64; 4],
        application_identity: ApplicationIdentity,
        application_source: ApplicationSource,
    ) -> LoadedPerApplicationPackage {
        LoadedPerApplicationPackage {
            circuit: self.circuit,
            matrix_program: self.matrix_program,
            application: self.application,
            assignment_plan: self.assignment_plan,
            next_preimage_rows: self.next_preimage_rows,
            logical_public_input_count: self.logical_public_input_count,
            structural_identifier,
            relation_value_words: self.relation_value_words,
            application_identity,
            application_source,
        }
    }
}

pub(super) struct NativePackage {
    sealed: SealedEnvelope,
    application: Arc<PreparedApplication>,
}

impl NativePackage {
    pub(super) fn application(&self) -> &PreparedApplication {
        &self.application
    }

    pub(super) fn bind(
        self,
        structural_identifier: [u64; 4],
        application_identity: ApplicationIdentity,
        source: PrivateSnapshot,
    ) -> LoadedPerApplicationPackage {
        self.sealed.bind(
            structural_identifier,
            application_identity,
            ApplicationSource::Native {
                application: self.application,
                source: Arc::new(source),
            },
        )
    }
}

pub(super) fn decode_native_application_records(
    fixed_envelope: &Value,
    records: Arc<ApplicationRecords>,
) -> Result<NativePackage, PackageError> {
    let application = Arc::new(PreparedApplication::from_metadata(
        fixed_envelope
            .get(3)
            .ok_or(PackageError::Malformed("native application metadata"))?,
        records,
    )?);
    let source = fixed_envelope
        .get(1)
        .and_then(Value::as_array)
        .ok_or(PackageError::Malformed("sealed circuit package"))?;
    for (field, range) in [
        (CIRCUIT_ASSERTION_ROWS, application.plan().row_range()),
        (CIRCUIT_WITNESS_INSTRUCTIONS, application.plan().row_range()),
        (CIRCUIT_WITNESS_BATCHES, application.plan().private_range()),
    ] {
        let entries = source
            .get(field)
            .and_then(Value::as_array)
            .ok_or(PackageError::Malformed("native fixed source array"))?;
        for entry in entries {
            let start = entry
                .get(0)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(PackageError::Malformed("native fixed source index"))?;
            if range.contains(&start) {
                return Err(PackageError::Malformed(
                    "native application dynamic source slots are not empty",
                ));
            }
        }
    }
    Ok(NativePackage {
        sealed: decode_sealed(fixed_envelope, Some(&application))?,
        application,
    })
}

/// Decode and check a sealed envelope. A native application supplies the
/// application plan; otherwise the envelope's application value does.
fn decode_sealed(
    envelope: &Value,
    native_application: Option<&Arc<PreparedApplication>>,
) -> Result<SealedEnvelope, PackageError> {
    let circuit_value = envelope
        .get(1)
        .ok_or(PackageError::Malformed("sealed circuit package"))?;
    let RawSealedPackage(
        schema,
        raw_circuit,
        raw_matrix,
        raw_application,
        raw_assignment,
        raw_next_preimage,
        raw_public,
    ) = RawSealedPackage::deserialize(envelope)?;
    if schema != SEALED_PACKAGE_SCHEMA {
        return Err(PackageError::Malformed("sealed package schema version"));
    }
    let relation_value_words = value_preimage_words(
        circuit_value
            .get(4)
            .ok_or(PackageError::Malformed("sealed relation authority"))?,
    )?;
    let circuit = super::validate::validate_package(raw_circuit, native_application.cloned())?;
    let matrix_program = MatrixProgram::decode(&raw_matrix)?;
    matrix_program.validate(circuit.layout.row_count)?;
    if matrix_program.row_count()? != circuit.relation.row_count() {
        return Err(PackageError::Malformed("matrix program relation row count"));
    }
    let next_preimage_rows = decode_next_preimage_range(raw_next_preimage, &circuit.layout)?;
    let application = match native_application {
        Some(application) => {
            if application.plan().row_range().end != next_preimage_rows.start {
                return Err(PackageError::Malformed("application plan range"));
            }
            application.plan().clone()
        }
        None => decode_application_plan(&raw_application, circuit_value, &circuit.layout, &next_preimage_rows)?,
    };
    validate_next_preimage_assertion_suffix(circuit_value, &next_preimage_rows)?;
    let logical_public_input_count = word_to_usize(raw_public, "logical public input count")?;
    if logical_public_input_count != PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS
        || logical_public_input_count > circuit.relation.column_count()
    {
        return Err(PackageError::Malformed("logical public input count"));
    }
    let assignment_plan = assignment_transport::decode(
        &raw_assignment,
        circuit.layout.total_column_count,
        logical_public_input_count,
        circuit.relation.column_count(),
    )?;
    Ok(SealedEnvelope {
        circuit,
        matrix_program,
        application,
        assignment_plan,
        next_preimage_rows,
        logical_public_input_count,
        relation_value_words,
    })
}

pub(super) fn prepared_record_counts(fixed: &Value) -> Result<(usize, usize), PackageError> {
    let envelope = fixed
        .as_array()
        .filter(|values| values.len() == 7 && values[0].as_u64() == Some(SEALED_PACKAGE_SCHEMA))
        .ok_or(PackageError::Malformed("prepared package envelope"))?;
    let source = envelope[1]
        .as_array()
        .filter(|values| values.len() == 14 && values[0].as_u64() == Some(INNER_PACKAGE_SCHEMA))
        .ok_or(PackageError::Malformed("prepared source package"))?;
    super::validate::validate_profile(Deserialize::deserialize(&source[1])?)?;
    super::validate::validate_poseidon(Deserialize::deserialize(&source[2])?)?;
    let layout = super::validate::validate_layout(Deserialize::deserialize(&source[3])?)?;
    let application = decode_native_application_plan(&envelope[3])?;
    let witness = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_WITNESS_ROLE)
        .ok_or(PackageError::Malformed("application witness segment"))?;
    let local = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_LOCAL_ROLE)
        .ok_or(PackageError::Malformed("application local segment"))?;
    if application.witness_word_count() != witness.length
        || application.private_range().start != local.start
        || application.private_range().len() != local.length
        || application.private_range().end > layout.constant_column
        || application.row_range().end > layout.row_count
    {
        return Err(PackageError::Malformed("prepared application record dimensions"));
    }
    Ok((application.row_range().len(), application.private_range().len()))
}

fn decode_per_application_value(
    value: &Value,
    structural_identifier: [u64; 4],
) -> Result<LoadedPerApplicationPackage, PackageError> {
    let sealed = decode_sealed(value, None)?;
    let application_words = value_preimage_words(&value[3])?;
    let application_identity = ApplicationIdentity::from_words(&application_words)?;
    Ok(sealed.bind(
        structural_identifier,
        application_identity,
        ApplicationSource::Words(application_words),
    ))
}

pub fn load_poseidon2_hash_chain_v1_package(bytes: &[u8]) -> Result<LoadedPerApplicationPackage, PackageError> {
    let package = load_per_application_package(bytes, POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER)?;
    let binding = package.production_verifier_binding()?;
    if binding.package_identity() != POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY {
        return Err(PackageError::IdentityMismatch {
            kind: IdentityKind::Package,
            expected: POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
            computed: binding.package_identity(),
        });
    }
    if binding.verification_key_digest() != POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST {
        return Err(PackageError::IdentityMismatch {
            kind: IdentityKind::VerificationKeyBinding,
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
        .ok_or(PackageError::Malformed("next preimage row range"))?;
    if row_count != NEXT_PREIMAGE_ROW_COUNT || row_end != layout.row_count {
        return Err(PackageError::Malformed("next preimage row range"));
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
    ) = RawApplicationPlan::deserialize(value)?;
    if schema != APPLICATION_PLAN_SCHEMA {
        return Err(PackageError::Malformed("application plan schema version"));
    }
    if !hash_chains.is_empty()
        || !permutation_invocations.is_empty()
        || !compact_templates.is_empty()
        || !compact_invocations.is_empty()
    {
        return Err(PackageError::Malformed("application plan row family"));
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
        return Err(PackageError::Malformed("application witness width"));
    }
    let private_end = private_start
        .checked_add(private_count)
        .ok_or(PackageError::Malformed("application private range"))?;
    let row_end = row_start
        .checked_add(row_count)
        .ok_or(PackageError::Malformed("application row range"))?;
    if private_end > layout.constant_column || row_end != next_preimage_rows.start {
        return Err(PackageError::Malformed("application plan range"));
    }

    let witness_segment = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_WITNESS_ROLE)
        .ok_or(PackageError::Malformed("application witness segment"))?;
    let local_segment = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == APPLICATION_LOCAL_ROLE)
        .ok_or(PackageError::Malformed("application local segment"))?;
    let expected_witness_columns = (witness_segment.start
        ..witness_segment
            .start
            .checked_add(witness_segment.length)
            .ok_or(PackageError::Malformed("application witness segment"))?)
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
        return Err(PackageError::Malformed("application plan column ownership"));
    }

    let ordinary_row_count = witness_instructions
        .len()
        .checked_add(assertion_rows.len())
        .ok_or(PackageError::Malformed("application ordinary row count"))?;
    if ordinary_row_count != row_count {
        return Err(PackageError::Malformed("application ordinary row count"));
    }
    let circuit = circuit_value
        .as_array()
        .ok_or(PackageError::Malformed("sealed circuit package"))?;
    if !is_suffix(circuit, CIRCUIT_WITNESS_BATCHES, &witness_batches)
        || !is_suffix(circuit, CIRCUIT_WITNESS_INSTRUCTIONS, &witness_instructions)
        || !is_suffix_before(
            circuit,
            CIRCUIT_ASSERTION_ROWS,
            &assertion_rows,
            next_preimage_rows.len(),
        )
    {
        return Err(PackageError::Malformed("application plan package suffix"));
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

pub(super) fn decode_native_application_plan(value: &Value) -> Result<LoadedApplicationPlan, PackageError> {
    let fields = value
        .as_array()
        .filter(|fields| fields.len() == 16)
        .ok_or(PackageError::Malformed("native application metadata"))?;
    if fields[9..]
        .iter()
        .any(|value| value.as_array().is_none_or(|values| !values.is_empty()))
    {
        return Err(PackageError::Malformed(
            "native application dynamic plan slots are not empty",
        ));
    }
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
        permutations,
        templates,
        invocations,
        batches,
        instructions,
        rows,
    ) = RawApplicationPlan::deserialize(value)?;
    if schema != APPLICATION_PLAN_SCHEMA
        || [
            &hash_chains,
            &permutations,
            &templates,
            &invocations,
            &batches,
            &instructions,
            &rows,
        ]
        .iter()
        .any(|values| !values.is_empty())
    {
        return Err(PackageError::Malformed(
            "native application dynamic plan slots are not empty",
        ));
    }
    let private_start = word_to_usize(private_start, "application private start")?;
    let private_count = word_to_usize(private_count, "application private count")?;
    let row_start = word_to_usize(row_start, "application row start")?;
    let row_count = word_to_usize(row_count, "application row count")?;
    private_start
        .checked_add(private_count)
        .ok_or(PackageError::Malformed("application private range"))?;
    row_start
        .checked_add(row_count)
        .ok_or(PackageError::Malformed("application row range"))?;
    let witness_word_count = word_to_usize(witness_word_count, "application witness width")?;
    let witness_columns = columns(witness_columns, "application witness column")?;
    if witness_columns.len() != witness_word_count {
        return Err(PackageError::Malformed("application witness width"));
    }
    Ok(LoadedApplicationPlan {
        witness_word_count,
        input_columns: fixed_columns(input_columns, "application input width")?,
        witness_columns,
        output_columns: fixed_columns(output_columns, "application output width")?,
        private_start,
        private_count,
        row_start,
        row_count,
    })
}

fn fixed_columns(raw: Vec<u64>, location: &'static str) -> Result<[usize; APPLICATION_STATE_WORDS], PackageError> {
    columns(raw, location)?
        .try_into()
        .map_err(|_| PackageError::Malformed(location))
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
        .ok_or(PackageError::Malformed("next preimage package suffix"))?;
    let suffix_start = rows
        .len()
        .checked_sub(expected.len())
        .ok_or(PackageError::Malformed("next preimage package suffix"))?;
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
        return Err(PackageError::Malformed("next preimage package suffix"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/sealed.rs"]
mod sealed_tests;
