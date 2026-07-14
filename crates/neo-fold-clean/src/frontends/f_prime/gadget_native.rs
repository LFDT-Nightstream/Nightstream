//! Exact gadget-native low-norm lowering for field-valued R1CS.
//!
//! The source R1CS is the semantic authority. Recorded gadget provenance
//! permits three local, algebraically exact replacements: Poseidon2 `x^7`,
//! quadratic-extension multiplication, and production Toom-3 ring action.
//! Every source row outside those ranges is lowered generically.

use std::collections::BTreeMap;

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::{
    KMulTraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, RingMulToom3TraceEntry, Sbox7TraceEntry, Var,
};

mod profile;
pub use profile::{
    profile_r1cs_gadget_native_stages, GadgetNativeStageEstimate, GadgetNativeStageProfile,
    GadgetNativeStageProfileError,
};

const FIELD_BITS: usize = 64;
const HIGH_BITS_START: usize = 32;
const CANONICAL_PREFIX_AUX: usize = 31;
const CANONICAL_SLOT_WIDTH: usize = FIELD_BITS + CANONICAL_PREFIX_AUX;
const TOOM_SPLIT: usize = 18;
const TOOM_COEFFICIENTS: usize = 2 * TOOM_SPLIT - 1;
const TOOM_EVALUATIONS: usize = 5;
const MAX_PRODUCT_TERMS: usize = TOOM_SPLIT;

/// Exact size of the gadget-native lowering without materializing its sparse
/// matrices or bit assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GadgetNativeEstimate {
    pub source_rows: usize,
    pub source_cols: usize,
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub max_degree: u32,
    pub one_bit_source_cols: usize,
    pub canonical_field_source_cols: usize,
    pub synthetic_ring_fields: usize,
    pub linearly_derived_source_cols: usize,
    pub gadget_derived_source_cols: usize,
    pub fallback_source_rows: usize,
}

/// Exact fixed-size direct CCS estimate for an internal base/recursive
/// selector. Public bits are shared; private slots are branch-local and forced
/// to their canonical zero encoding while inactive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectorGatedGadgetNativeEstimate {
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub max_degree: u32,
    pub canonical_field_slots: usize,
    pub one_bit_slots: usize,
    pub inactive_zero_rows: usize,
    pub base: GadgetNativeEstimate,
    pub recursive: GadgetNativeEstimate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueSlot {
    start: usize,
    bits: usize,
    canonical_aux_start: Option<usize>,
}

#[derive(Clone, Debug)]
struct LinearDefinition {
    terms: Vec<(usize, F)>,
}

#[derive(Clone, Debug)]
struct ProductDefinition {
    left: Lc,
    right: Lc,
}

#[derive(Clone, Debug)]
enum SourceColumn {
    One,
    Encoded(ValueSlot),
    Linear(LinearDefinition),
    Product(ProductDefinition),
}

#[derive(Clone, Debug)]
struct RingSyntheticSlots {
    coefficients: Vec<ValueSlot>,
}

impl RingSyntheticSlots {
    fn coefficient(&self, evaluation: usize, coefficient: usize) -> ValueSlot {
        self.coefficients[evaluation * TOOM_COEFFICIENTS + coefficient]
    }
}

/// Inverse map from the committed low-norm assignment to the complete source
/// R1CS witness, including every product wire projected out by the lowering.
#[derive(Clone, Debug)]
pub struct GadgetNativePlan {
    source_columns: Vec<SourceColumn>,
    ring_slots: Vec<RingSyntheticSlots>,
    public_columns: Vec<usize>,
    public_input_len: usize,
    encoded_cols: usize,
}

impl GadgetNativePlan {
    pub fn public_columns(&self) -> &[usize] {
        &self.public_columns
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn encoded_cols(&self) -> usize {
        self.encoded_cols
    }

    pub fn encoded_range_for_source_column(&self, column: usize) -> Option<std::ops::Range<usize>> {
        match self.source_columns.get(column) {
            Some(SourceColumn::Encoded(slot)) => Some(slot.start..slot.start + slot.bits),
            _ => None,
        }
    }

    pub fn is_gadget_derived(&self, column: usize) -> bool {
        matches!(self.source_columns.get(column), Some(SourceColumn::Product(_)))
    }

    pub fn synthetic_ring_coefficient_range(
        &self,
        ring: usize,
        evaluation: usize,
        coefficient: usize,
    ) -> Option<std::ops::Range<usize>> {
        let slot = self.ring_slots.get(ring)?.coefficients.get(
            evaluation
                .checked_mul(TOOM_COEFFICIENTS)?
                .checked_add(coefficient)?,
        )?;
        Some(slot.start..slot.start + slot.bits)
    }

    /// Decode the exact source witness. This reconstructs projected gadget
    /// temporaries in allocation order rather than trusting sidecar values.
    pub fn decode_source(&self, encoded: &[F]) -> Result<Vec<F>, GadgetNativeError> {
        if encoded.len() != self.encoded_cols {
            return Err(GadgetNativeError::EncodedLength {
                expected: self.encoded_cols,
                got: encoded.len(),
            });
        }
        if encoded.first().copied() != Some(F::ONE) {
            return Err(GadgetNativeError::EncodedConstantOne);
        }
        let mut source = vec![F::ZERO; self.source_columns.len()];
        for (column, definition) in self.source_columns.iter().enumerate() {
            source[column] = match definition {
                SourceColumn::One => F::ONE,
                SourceColumn::Encoded(slot) => decode_slot(*slot, column, encoded)?,
                SourceColumn::Linear(definition) => definition
                    .terms
                    .iter()
                    .fold(F::ZERO, |value, &(input, coefficient)| {
                        value + coefficient * source[input]
                    }),
                SourceColumn::Product(definition) => {
                    eval_lc_from_source(&definition.left, &source) * eval_lc_from_source(&definition.right, &source)
                }
            };
        }
        Ok(source)
    }
}

/// One materialized gadget-native CCS instance before Ajtai commitment.
#[derive(Clone, Debug)]
pub struct EncodedGadgetNativeR1cs {
    pub structure: CcsStructure<F>,
    pub assignment: Vec<F>,
    pub plan: GadgetNativePlan,
}

impl EncodedGadgetNativeR1cs {
    pub fn public_input(&self) -> &[F] {
        &self.assignment[..self.plan.public_input_len]
    }

    pub fn private_witness(&self) -> &[F] {
        &self.assignment[self.plan.public_input_len..]
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        evaluate_ccs(&self.structure, &self.assignment)
            .into_iter()
            .position(|value| value != F::ZERO)
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }

    pub fn decode_source(&self) -> Result<Vec<F>, GadgetNativeError> {
        if let Some(row) = self.first_unsatisfied_row() {
            return Err(GadgetNativeError::UnsatisfiedEncoding { row });
        }
        self.plan.decode_source(&self.assignment)
    }
}

#[derive(Debug, Error)]
pub enum GadgetNativeError {
    #[error("gadget-native encoding requires source column zero to equal one")]
    SourceConstantOne,
    #[error("public source column {column} is out of range for {cols} columns")]
    PublicColumnOutOfRange { column: usize, cols: usize },
    #[error("public source column {column} appears more than once")]
    DuplicatePublicColumn { column: usize },
    #[error("public source column {column} is not explicitly Boolean")]
    PublicColumnNotBoolean { column: usize },
    #[error("selector-gated branches expose different public bit counts ({base} versus {recursive})")]
    BranchPublicInputLength { base: usize, recursive: usize },
    #[error("encoding trace {gadget} row range {start}..{end} is malformed for a {rows}-row source")]
    TraceRowRange {
        gadget: &'static str,
        start: usize,
        end: usize,
        rows: usize,
    },
    #[error("encoding trace row {row} is claimed by more than one gadget")]
    OverlappingTraceRow { row: usize },
    #[error("encoding trace for {gadget} does not match source R1CS row {row}")]
    TraceRowMismatch { gadget: &'static str, row: usize },
    #[error("encoding trace {gadget} has malformed fixed arity")]
    TraceArity { gadget: &'static str },
    #[error("source column {column} is projected by more than one gadget")]
    DuplicateGadgetDefinition { column: usize },
    #[error("gadget-derived source column {column} is not topological")]
    NonTopologicalDefinition { column: usize },
    #[error("gadget-derived source column {column} escapes its recorded source rows")]
    GadgetTemporaryEscapes { column: usize },
    #[error("public source column {column} is an internal gadget temporary")]
    PublicGadgetTemporary { column: usize },
    #[error("source column {column} has no linear low-norm representation in row {row}")]
    MissingDecodedColumn { column: usize, row: usize },
    #[error("ring trace output {output} has inconsistent coefficient for convolution {evaluation}:{coefficient}")]
    RingCoefficientMismatch {
        output: usize,
        evaluation: usize,
        coefficient: usize,
    },
    #[error("ring trace output expression references non-product source column {column}")]
    RingOutputUnknownColumn { column: usize },
    #[error("source Boolean column {column} has non-Boolean witness value")]
    BooleanWitness { column: usize },
    #[error("encoded assignment length {got} does not match plan length {expected}")]
    EncodedLength { expected: usize, got: usize },
    #[error("encoded assignment's constant column is not one")]
    EncodedConstantOne,
    #[error("encoded bit for source column {column} is not zero or one")]
    NonBinaryDigit { column: usize },
    #[error("64-bit encoding for source column {column} is noncanonical ({value} >= p)")]
    NonCanonicalField { column: usize, value: u64 },
    #[error("canonicality auxiliary {offset} for source column {column} is inconsistent")]
    CanonicalAuxMismatch { column: usize, offset: usize },
    #[error("gadget-native low-norm relation is unsatisfied at row {row}")]
    UnsatisfiedEncoding { row: usize },
}

struct TraceMarks {
    covered_rows: Vec<bool>,
    gadget_columns: Vec<bool>,
}

/// Compute exact production dimensions without allocating the bit witness or
/// sparse CCS matrices.
pub fn estimate_r1cs_gadget_native(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<GadgetNativeEstimate, GadgetNativeError> {
    validate_source_one(source)?;
    let (is_public, explicit_bits) = validate_public_columns(source, public_bit_columns)?;
    let marks = validate_and_mark_trace(source, trace)?;
    reject_public_gadget_columns(&marks.gadget_columns, &is_public)?;
    let (linearly_derived, linearly_derived_source_cols) = select_linear_definition_columns(source, &is_public, &marks);

    let mut one_bit_source_cols = public_bit_columns.len();
    let mut canonical_field_source_cols = 0usize;
    for column in 1..source.cols() {
        if is_public[column] || marks.gadget_columns[column] || linearly_derived[column] {
            continue;
        }
        if explicit_bits[column] {
            one_bit_source_cols += 1;
        } else {
            canonical_field_source_cols += 1;
        }
    }
    let synthetic_ring_fields = trace
        .ring_muls_toom3()
        .len()
        .saturating_mul(TOOM_EVALUATIONS * TOOM_COEFFICIENTS);
    let encoded_cols = 1usize.saturating_add(one_bit_source_cols).saturating_add(
        canonical_field_source_cols
            .saturating_add(synthetic_ring_fields)
            .saturating_mul(CANONICAL_SLOT_WIDTH),
    );
    let covered_count = marks
        .covered_rows
        .iter()
        .filter(|&&covered| covered)
        .count();
    let fallback_source_rows = source
        .rows()
        .saturating_sub(covered_count)
        .saturating_sub(linearly_derived_source_cols);
    let custom_rows = trace
        .sbox7()
        .len()
        .saturating_add(trace.k_muls().len().saturating_mul(2))
        .saturating_add(
            trace
                .ring_muls_toom3()
                .len()
                .saturating_mul(TOOM_EVALUATIONS * TOOM_COEFFICIENTS + 54),
        );
    let encoded_rows = (encoded_cols - 1)
        .saturating_add(
            canonical_field_source_cols
                .saturating_add(synthetic_ring_fields)
                .saturating_mul(32),
        )
        .saturating_add(fallback_source_rows)
        .saturating_add(custom_rows);
    Ok(GadgetNativeEstimate {
        source_rows: source.rows(),
        source_cols: source.cols(),
        public_input_len: 1 + public_bit_columns.len(),
        encoded_cols,
        encoded_rows,
        max_degree: 8,
        one_bit_source_cols,
        canonical_field_source_cols,
        synthetic_ring_fields,
        linearly_derived_source_cols,
        gadget_derived_source_cols: marks
            .gadget_columns
            .iter()
            .filter(|&&derived| derived)
            .count(),
        fallback_source_rows,
    })
}

/// Estimate one fixed relation whose internal Boolean selects the base or
/// recursive traced R1CS. Branch equations are gated directly in the CCS
/// polynomial, so no R1CS residual columns are introduced.
pub fn estimate_selector_gated_r1cs_gadget_native(
    base_source: &R1csSnapshot,
    base_trace: &R1csEncodingTrace,
    base_public_bit_columns: &[usize],
    recursive_source: &R1csSnapshot,
    recursive_trace: &R1csEncodingTrace,
    recursive_public_bit_columns: &[usize],
) -> Result<SelectorGatedGadgetNativeEstimate, GadgetNativeError> {
    if base_public_bit_columns.len() != recursive_public_bit_columns.len() {
        return Err(GadgetNativeError::BranchPublicInputLength {
            base: base_public_bit_columns.len(),
            recursive: recursive_public_bit_columns.len(),
        });
    }
    let base = estimate_r1cs_gadget_native(base_source, base_trace, base_public_bit_columns)?;
    let recursive = estimate_r1cs_gadget_native(recursive_source, recursive_trace, recursive_public_bit_columns)?;
    let public_bits = base_public_bit_columns.len();
    let base_private_bits = base.one_bit_source_cols - public_bits;
    let recursive_private_bits = recursive.one_bit_source_cols - public_bits;
    let canonical_field_slots = base
        .canonical_field_source_cols
        .saturating_add(base.synthetic_ring_fields)
        .saturating_add(recursive.canonical_field_source_cols)
        .saturating_add(recursive.synthetic_ring_fields);
    let one_bit_slots = public_bits
        .saturating_add(1) // internal is_base selector
        .saturating_add(base_private_bits)
        .saturating_add(recursive_private_bits);
    let encoded_cols = 1usize
        .saturating_add(one_bit_slots)
        .saturating_add(canonical_field_slots.saturating_mul(CANONICAL_SLOT_WIDTH));
    let base_canonical = base
        .canonical_field_source_cols
        .saturating_add(base.synthetic_ring_fields);
    let recursive_canonical = recursive
        .canonical_field_source_cols
        .saturating_add(recursive.synthetic_ring_fields);
    let base_semantic_rows = base
        .encoded_rows
        .saturating_sub(base.encoded_cols - 1)
        .saturating_sub(base_canonical.saturating_mul(32));
    let recursive_semantic_rows = recursive
        .encoded_rows
        .saturating_sub(recursive.encoded_cols - 1)
        .saturating_sub(recursive_canonical.saturating_mul(32));
    let inactive_zero_rows = base_private_bits
        .saturating_add(recursive_private_bits)
        .saturating_add(canonical_field_slots);
    let encoded_rows = (encoded_cols - 1)
        .saturating_add(canonical_field_slots.saturating_mul(32))
        .saturating_add(base_semantic_rows)
        .saturating_add(recursive_semantic_rows)
        .saturating_add(inactive_zero_rows);
    Ok(SelectorGatedGadgetNativeEstimate {
        public_input_len: 1 + public_bits,
        encoded_cols,
        encoded_rows,
        max_degree: 8,
        canonical_field_slots,
        one_bit_slots,
        inactive_zero_rows,
        base,
        recursive,
    })
}

/// Materialize the exact gadget-native lowering. Intended for differential
/// tests and bounded relations; use [`estimate_r1cs_gadget_native`] before
/// attempting a production-sized allocation.
pub fn encode_r1cs_gadget_native(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<EncodedGadgetNativeR1cs, GadgetNativeError> {
    validate_source_one(source)?;
    let (is_public, explicit_bits) = validate_public_columns(source, public_bit_columns)?;
    let marks = validate_and_mark_trace(source, trace)?;
    reject_public_gadget_columns(&marks.gadget_columns, &is_public)?;
    let nonlinear = build_product_definitions(source, trace)?;
    let (linear, removed_definition_rows) = build_linear_definitions(source, &is_public, &marks);

    let mut assignment = vec![F::ONE];
    let mut source_columns = vec![None; source.cols()];
    source_columns[0] = Some(SourceColumn::One);
    let mut canonical_slots = Vec::new();

    for &column in public_bit_columns {
        let slot = push_boolean_slot(&mut assignment, source.witness()[column], column)?;
        source_columns[column] = Some(SourceColumn::Encoded(slot));
    }
    let public_input_len = assignment.len();

    for column in 1..source.cols() {
        if is_public[column] {
            continue;
        }
        if let Some(definition) = nonlinear[column].clone() {
            source_columns[column] = Some(SourceColumn::Product(definition));
        } else if let Some(definition) = linear[column].clone() {
            source_columns[column] = Some(SourceColumn::Linear(definition));
        } else if explicit_bits[column] {
            let slot = push_boolean_slot(&mut assignment, source.witness()[column], column)?;
            source_columns[column] = Some(SourceColumn::Encoded(slot));
        } else {
            let slot = push_field_slot(&mut assignment, source.witness()[column]);
            canonical_slots.push(slot);
            source_columns[column] = Some(SourceColumn::Encoded(slot));
        }
    }

    let mut ring_slots = Vec::with_capacity(trace.ring_muls_toom3().len());
    for ring in trace.ring_muls_toom3() {
        let mut coefficients = Vec::with_capacity(TOOM_EVALUATIONS * TOOM_COEFFICIENTS);
        for convolution in &ring.convolutions {
            for coefficient in 0..TOOM_COEFFICIENTS {
                let value = (0..TOOM_SPLIT)
                    .filter_map(|left| {
                        let right = coefficient.checked_sub(left)?;
                        (right < TOOM_SPLIT)
                            .then_some(source.witness()[convolution.products[left * TOOM_SPLIT + right].col()])
                    })
                    .fold(F::ZERO, |sum, product| sum + product);
                let slot = push_field_slot(&mut assignment, value);
                canonical_slots.push(slot);
                coefficients.push(slot);
            }
        }
        ring_slots.push(RingSyntheticSlots { coefficients });
    }

    let source_columns = source_columns
        .into_iter()
        .map(|column| column.expect("every source column has one encoding definition"))
        .collect::<Vec<_>>();
    let decoded_terms = build_source_terms(&source_columns)?;
    let mut gates = TraceGateBuilder::with_estimated_rows(assignment.len() + source.rows());
    for column in 1..assignment.len() {
        gates.bitness(column);
    }
    for &slot in &canonical_slots {
        emit_goldilocks_canonicality(&mut gates, slot);
    }

    for row in 0..source.rows() {
        if marks.covered_rows[row] || removed_definition_rows[row] {
            continue;
        }
        gates.product_sum(
            one_selector(),
            vec![(
                translate_source_row(source.a_row(row), &decoded_terms, row)?,
                translate_source_row(source.b_row(row), &decoded_terms, row)?,
            )],
            translate_source_row(source.c_row(row), &decoded_terms, row)?,
        );
    }
    for event in trace.sbox7() {
        gates.sbox7(
            one_selector(),
            translate_event_lc(&event.input, &decoded_terms, event.source_rows.start)?,
            source_terms(event.output.col(), &decoded_terms, event.source_rows.start)?,
        );
    }
    for event in trace.k_muls() {
        emit_k_mul(event, &decoded_terms, &mut gates)?;
    }
    for (event, slots) in trace.ring_muls_toom3().iter().zip(ring_slots.iter()) {
        emit_ring_mul(event, slots, &decoded_terms, &mut gates)?;
    }

    let structure = gates.finish(assignment.len());
    let plan = GadgetNativePlan {
        source_columns,
        ring_slots,
        public_columns: public_bit_columns.to_vec(),
        public_input_len,
        encoded_cols: assignment.len(),
    };
    Ok(EncodedGadgetNativeR1cs {
        structure,
        assignment,
        plan,
    })
}

fn validate_source_one(source: &R1csSnapshot) -> Result<(), GadgetNativeError> {
    if source.witness().first().copied() != Some(F::ONE) {
        return Err(GadgetNativeError::SourceConstantOne);
    }
    Ok(())
}

fn validate_public_columns(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
) -> Result<(Vec<bool>, Vec<bool>), GadgetNativeError> {
    let explicit_bits = source.explicitly_boolean_columns();
    let mut is_public = vec![false; source.cols()];
    for &column in public_bit_columns {
        if column == 0 || column >= source.cols() {
            return Err(GadgetNativeError::PublicColumnOutOfRange {
                column,
                cols: source.cols(),
            });
        }
        if is_public[column] {
            return Err(GadgetNativeError::DuplicatePublicColumn { column });
        }
        if !explicit_bits[column] {
            return Err(GadgetNativeError::PublicColumnNotBoolean { column });
        }
        is_public[column] = true;
    }
    Ok((is_public, explicit_bits))
}

fn reject_public_gadget_columns(gadget_columns: &[bool], is_public: &[bool]) -> Result<(), GadgetNativeError> {
    if let Some(column) = (1..gadget_columns.len()).find(|&column| gadget_columns[column] && is_public[column]) {
        return Err(GadgetNativeError::PublicGadgetTemporary { column });
    }
    Ok(())
}

fn validate_and_mark_trace(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> Result<TraceMarks, GadgetNativeError> {
    let mut covered_rows = vec![false; source.rows()];
    let mut gadget_columns = vec![false; source.cols()];
    for event in trace.sbox7() {
        validate_sbox(source, event)?;
        claim_rows(source, "Poseidon2 sbox7", &event.source_rows, &mut covered_rows)?;
        for variable in event.intermediates {
            claim_gadget_column(variable.col(), &mut gadget_columns)?;
        }
    }
    for event in trace.k_muls() {
        validate_k_mul(source, event)?;
        claim_rows(source, "K multiplication", &event.source_rows, &mut covered_rows)?;
        for variable in event.intermediates {
            claim_gadget_column(variable.col(), &mut gadget_columns)?;
        }
    }
    for event in trace.ring_muls_toom3() {
        validate_ring_mul(source, event)?;
        claim_rows(
            source,
            "Toom-3 ring multiplication",
            &event.source_rows,
            &mut covered_rows,
        )?;
        for convolution in &event.convolutions {
            for variable in &convolution.products {
                claim_gadget_column(variable.col(), &mut gadget_columns)?;
            }
        }
    }
    for row in 0..source.rows() {
        if covered_rows[row] {
            continue;
        }
        for &(column, _) in source
            .a_row(row)
            .iter()
            .chain(source.b_row(row))
            .chain(source.c_row(row))
        {
            if gadget_columns[column] {
                return Err(GadgetNativeError::GadgetTemporaryEscapes { column });
            }
        }
    }
    Ok(TraceMarks {
        covered_rows,
        gadget_columns,
    })
}

fn claim_rows(
    source: &R1csSnapshot,
    gadget: &'static str,
    range: &std::ops::Range<usize>,
    covered: &mut [bool],
) -> Result<(), GadgetNativeError> {
    if range.start >= range.end || range.end > source.rows() {
        return Err(GadgetNativeError::TraceRowRange {
            gadget,
            start: range.start,
            end: range.end,
            rows: source.rows(),
        });
    }
    for row in range.clone() {
        if std::mem::replace(&mut covered[row], true) {
            return Err(GadgetNativeError::OverlappingTraceRow { row });
        }
    }
    Ok(())
}

fn claim_gadget_column(column: usize, claimed: &mut [bool]) -> Result<(), GadgetNativeError> {
    if column == 0 || column >= claimed.len() || std::mem::replace(&mut claimed[column], true) {
        return Err(GadgetNativeError::DuplicateGadgetDefinition { column });
    }
    Ok(())
}

fn validate_sbox(source: &R1csSnapshot, event: &Sbox7TraceEntry) -> Result<(), GadgetNativeError> {
    if event
        .source_rows
        .end
        .saturating_sub(event.source_rows.start)
        != 4
    {
        return Err(GadgetNativeError::TraceArity {
            gadget: "Poseidon2 sbox7",
        });
    }
    let [x2, x4, x6] = event.intermediates;
    let rows = [
        (event.input.clone(), event.input.clone(), Lc::from_var(x2)),
        (Lc::from_var(x2), Lc::from_var(x2), Lc::from_var(x4)),
        (Lc::from_var(x2), Lc::from_var(x4), Lc::from_var(x6)),
        (event.input.clone(), Lc::from_var(x6), Lc::from_var(event.output)),
    ];
    validate_expected_rows(source, "Poseidon2 sbox7", event.source_rows.start, &rows)
}

fn validate_k_mul(source: &R1csSnapshot, event: &KMulTraceEntry) -> Result<(), GadgetNativeError> {
    if event
        .source_rows
        .end
        .saturating_sub(event.source_rows.start)
        != 5
    {
        return Err(GadgetNativeError::TraceArity {
            gadget: "K multiplication",
        });
    }
    let [p, q, r] = event.intermediates;
    let sum_a = event.a[0].clone().add_scaled(&event.a[1], F::ONE);
    let sum_b = event.b[0].clone().add_scaled(&event.b[1], F::ONE);
    let w = <Fq as BinomiallyExtendable<2>>::W;
    let out0_diff = Lc::from_var(event.output[0])
        .add_scaled(&Lc::from_var(p), -F::ONE)
        .add_scaled(&Lc::from_var(q), -w);
    let out1_diff = Lc::from_var(event.output[1])
        .add_scaled(&Lc::from_var(r), -F::ONE)
        .add_scaled(&Lc::from_var(p), F::ONE)
        .add_scaled(&Lc::from_var(q), F::ONE);
    let rows = [
        (event.a[0].clone(), event.b[0].clone(), Lc::from_var(p)),
        (event.a[1].clone(), event.b[1].clone(), Lc::from_var(q)),
        (sum_a, sum_b, Lc::from_var(r)),
        (out0_diff, Lc::from_var(Var::ONE), Lc::zero()),
        (out1_diff, Lc::from_var(Var::ONE), Lc::zero()),
    ];
    validate_expected_rows(source, "K multiplication", event.source_rows.start, &rows)
}

fn validate_ring_mul(source: &R1csSnapshot, event: &RingMulToom3TraceEntry) -> Result<(), GadgetNativeError> {
    let product_rows = TOOM_EVALUATIONS * TOOM_SPLIT * TOOM_SPLIT;
    if event.rho.len() != 54
        || event.c.len() != 54
        || event.convolutions.len() != TOOM_EVALUATIONS
        || event.reduced_output_lcs.len() != 54
        || event.output.len() != 54
        || event
            .source_rows
            .end
            .saturating_sub(event.source_rows.start)
            != product_rows + 54
        || event.convolutions.iter().any(|convolution| {
            convolution.lhs.len() != TOOM_SPLIT
                || convolution.rhs.len() != TOOM_SPLIT
                || convolution.products.len() != TOOM_SPLIT * TOOM_SPLIT
        })
    {
        return Err(GadgetNativeError::TraceArity {
            gadget: "Toom-3 ring multiplication",
        });
    }
    let mut row = event.source_rows.start;
    for convolution in &event.convolutions {
        for left in 0..TOOM_SPLIT {
            for right in 0..TOOM_SPLIT {
                validate_row(
                    source,
                    "Toom-3 ring multiplication",
                    row,
                    &convolution.lhs[left],
                    &convolution.rhs[right],
                    &Lc::from_var(convolution.products[left * TOOM_SPLIT + right]),
                )?;
                row += 1;
            }
        }
    }
    for output in 0..54 {
        let diff = Lc::from_var(event.output[output]).add_scaled(&event.reduced_output_lcs[output], -F::ONE);
        validate_row(
            source,
            "Toom-3 ring multiplication",
            row,
            &diff,
            &Lc::from_var(Var::ONE),
            &Lc::zero(),
        )?;
        row += 1;
    }
    Ok(())
}

fn validate_expected_rows(
    source: &R1csSnapshot,
    gadget: &'static str,
    first: usize,
    rows: &[(Lc, Lc, Lc)],
) -> Result<(), GadgetNativeError> {
    for (offset, (a, b, c)) in rows.iter().enumerate() {
        validate_row(source, gadget, first + offset, a, b, c)?;
    }
    Ok(())
}

fn validate_row(
    source: &R1csSnapshot,
    gadget: &'static str,
    row: usize,
    a: &Lc,
    b: &Lc,
    c: &Lc,
) -> Result<(), GadgetNativeError> {
    if row >= source.rows()
        || source.a_row(row) != normalize_lc(a)
        || source.b_row(row) != normalize_lc(b)
        || source.c_row(row) != normalize_lc(c)
    {
        return Err(GadgetNativeError::TraceRowMismatch { gadget, row });
    }
    Ok(())
}

fn normalize_lc(lc: &Lc) -> Vec<(usize, F)> {
    let mut terms = BTreeMap::<usize, F>::new();
    for &(column, coefficient) in &lc.terms {
        *terms.entry(column).or_insert(F::ZERO) += coefficient;
    }
    *terms.entry(0).or_insert(F::ZERO) += lc.constant;
    terms
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn build_product_definitions(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<Vec<Option<ProductDefinition>>, GadgetNativeError> {
    let mut definitions = vec![None; source.cols()];
    for event in trace.sbox7() {
        let [x2, x4, x6] = event.intermediates;
        set_product_definition(&mut definitions, x2, event.input.clone(), event.input.clone())?;
        set_product_definition(&mut definitions, x4, Lc::from_var(x2), Lc::from_var(x2))?;
        set_product_definition(&mut definitions, x6, Lc::from_var(x2), Lc::from_var(x4))?;
    }
    for event in trace.k_muls() {
        let [p, q, r] = event.intermediates;
        set_product_definition(&mut definitions, p, event.a[0].clone(), event.b[0].clone())?;
        set_product_definition(&mut definitions, q, event.a[1].clone(), event.b[1].clone())?;
        set_product_definition(
            &mut definitions,
            r,
            event.a[0].clone().add_scaled(&event.a[1], F::ONE),
            event.b[0].clone().add_scaled(&event.b[1], F::ONE),
        )?;
    }
    for event in trace.ring_muls_toom3() {
        for convolution in &event.convolutions {
            for left in 0..TOOM_SPLIT {
                for right in 0..TOOM_SPLIT {
                    set_product_definition(
                        &mut definitions,
                        convolution.products[left * TOOM_SPLIT + right],
                        convolution.lhs[left].clone(),
                        convolution.rhs[right].clone(),
                    )?;
                }
            }
        }
    }
    Ok(definitions)
}

fn set_product_definition(
    definitions: &mut [Option<ProductDefinition>],
    variable: Var,
    left: Lc,
    right: Lc,
) -> Result<(), GadgetNativeError> {
    let column = variable.col();
    if column == 0 || column >= definitions.len() || definitions[column].is_some() {
        return Err(GadgetNativeError::DuplicateGadgetDefinition { column });
    }
    if left
        .terms
        .iter()
        .chain(right.terms.iter())
        .any(|&(input, _)| input >= column)
    {
        return Err(GadgetNativeError::NonTopologicalDefinition { column });
    }
    definitions[column] = Some(ProductDefinition { left, right });
    Ok(())
}

fn select_linear_definition_columns(
    source: &R1csSnapshot,
    is_public: &[bool],
    marks: &TraceMarks,
) -> (Vec<bool>, usize) {
    let mut defined = marks.gadget_columns.clone();
    let mut selected = vec![false; source.cols()];
    let mut count = 0usize;
    for row in 0..source.rows() {
        if marks.covered_rows[row] {
            continue;
        }
        let Some((column, _)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        defined[column] = true;
        selected[column] = true;
        count += 1;
    }
    (selected, count)
}

fn build_linear_definitions(
    source: &R1csSnapshot,
    is_public: &[bool],
    marks: &TraceMarks,
) -> (Vec<Option<LinearDefinition>>, Vec<bool>) {
    let mut definitions = vec![None; source.cols()];
    let mut removed_rows = vec![false; source.rows()];
    let mut defined = marks.gadget_columns.clone();
    for row in 0..source.rows() {
        if marks.covered_rows[row] {
            continue;
        }
        let Some((column, coefficient)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        let (positive, negative) = linear_difference(source, row).expect("candidate row is linear");
        let inverse = coefficient.inverse();
        let mut terms = Vec::new();
        visit_difference_terms(positive, negative, |input, input_coefficient| {
            if input != column {
                let derived = -input_coefficient * inverse;
                if derived != F::ZERO {
                    terms.push((input, derived));
                }
            }
        });
        definitions[column] = Some(LinearDefinition { terms });
        removed_rows[row] = true;
        defined[column] = true;
    }
    (definitions, removed_rows)
}

fn linear_definition_candidate(
    source: &R1csSnapshot,
    row: usize,
    is_public: &[bool],
    defined: &[bool],
) -> Option<(usize, F)> {
    let (positive, negative) = linear_difference(source, row)?;
    let mut highest = None;
    visit_difference_terms(positive, negative, |column, coefficient| {
        if column != 0 {
            highest = Some((column, coefficient));
        }
    });
    let (column, coefficient) = highest?;
    if is_public[column] || defined[column] {
        return None;
    }
    Some((column, coefficient))
}

fn linear_difference(source: &R1csSnapshot, row: usize) -> Option<(&[(usize, F)], &[(usize, F)])> {
    if source.b_row(row) == [(0, F::ONE)] {
        Some((source.a_row(row), source.c_row(row)))
    } else if source.a_row(row) == [(0, F::ONE)] {
        Some((source.b_row(row), source.c_row(row)))
    } else {
        None
    }
}

fn visit_difference_terms(positive: &[(usize, F)], negative: &[(usize, F)], mut visit: impl FnMut(usize, F)) {
    let mut left = 0usize;
    let mut right = 0usize;
    while left < positive.len() || right < negative.len() {
        let column = match (positive.get(left), negative.get(right)) {
            (Some(&(a, _)), Some(&(b, _))) => a.min(b),
            (Some(&(a, _)), None) => a,
            (None, Some(&(b, _))) => b,
            (None, None) => unreachable!(),
        };
        let mut coefficient = F::ZERO;
        if positive
            .get(left)
            .is_some_and(|&(candidate, _)| candidate == column)
        {
            coefficient += positive[left].1;
            left += 1;
        }
        if negative
            .get(right)
            .is_some_and(|&(candidate, _)| candidate == column)
        {
            coefficient -= negative[right].1;
            right += 1;
        }
        if coefficient != F::ZERO {
            visit(column, coefficient);
        }
    }
}

fn push_boolean_slot(out: &mut Vec<F>, value: F, column: usize) -> Result<ValueSlot, GadgetNativeError> {
    if value != F::ZERO && value != F::ONE {
        return Err(GadgetNativeError::BooleanWitness { column });
    }
    let start = out.len();
    out.push(value);
    Ok(ValueSlot {
        start,
        bits: 1,
        canonical_aux_start: None,
    })
}

fn push_field_slot(out: &mut Vec<F>, value: F) -> ValueSlot {
    let start = out.len();
    let value = value.as_canonical_u64();
    out.extend((0..FIELD_BITS).map(|bit| F::from_u64((value >> bit) & 1)));
    let canonical_aux_start = out.len();
    let aux = canonical_prefix_values(&out[start..start + FIELD_BITS]);
    out.extend(aux);
    ValueSlot {
        start,
        bits: FIELD_BITS,
        canonical_aux_start: Some(canonical_aux_start),
    }
}

fn decode_slot(slot: ValueSlot, column: usize, encoded: &[F]) -> Result<F, GadgetNativeError> {
    let mut value = 0u64;
    for bit in 0..slot.bits {
        let digit = encoded[slot.start + bit];
        if digit != F::ZERO && digit != F::ONE {
            return Err(GadgetNativeError::NonBinaryDigit { column });
        }
        if digit == F::ONE {
            value |= 1u64 << bit;
        }
    }
    if slot.bits == FIELD_BITS && value >= F::ORDER_U64 {
        return Err(GadgetNativeError::NonCanonicalField { column, value });
    }
    if let Some(aux_start) = slot.canonical_aux_start {
        let expected = canonical_prefix_values(&encoded[slot.start..slot.start + FIELD_BITS]);
        for (offset, expected_bit) in expected.into_iter().enumerate() {
            if encoded[aux_start + offset] != expected_bit {
                return Err(GadgetNativeError::CanonicalAuxMismatch { column, offset });
            }
        }
    }
    Ok(F::from_u64(value))
}

fn canonical_prefix_values(bits: &[F]) -> Vec<F> {
    let mut out = Vec::with_capacity(CANONICAL_PREFIX_AUX);
    let mut prefix = bits[HIGH_BITS_START] * bits[HIGH_BITS_START + 1];
    out.push(prefix);
    for &bit in &bits[HIGH_BITS_START + 2..] {
        prefix *= bit;
        out.push(prefix);
    }
    out
}

fn slot_terms(slot: ValueSlot) -> Vec<(usize, F)> {
    (0..slot.bits)
        .map(|bit| (slot.start + bit, F::from_u64(1u64 << bit)))
        .collect()
}

fn build_source_terms(columns: &[SourceColumn]) -> Result<Vec<Option<Vec<(usize, F)>>>, GadgetNativeError> {
    let mut decoded = vec![None; columns.len()];
    decoded[0] = Some(vec![(0, F::ONE)]);
    for column in 1..columns.len() {
        decoded[column] = match &columns[column] {
            SourceColumn::One => unreachable!("only source column zero is ONE"),
            SourceColumn::Encoded(slot) => Some(slot_terms(*slot)),
            SourceColumn::Product(_) => None,
            SourceColumn::Linear(definition) => {
                let mut combined = BTreeMap::<usize, F>::new();
                for &(input, scale) in &definition.terms {
                    let Some(input_terms) = &decoded[input] else {
                        return Err(GadgetNativeError::MissingDecodedColumn { column: input, row: 0 });
                    };
                    for &(encoded_column, coefficient) in input_terms {
                        *combined.entry(encoded_column).or_insert(F::ZERO) += scale * coefficient;
                    }
                }
                Some(
                    combined
                        .into_iter()
                        .filter(|(_, coefficient)| *coefficient != F::ZERO)
                        .collect(),
                )
            }
        };
    }
    Ok(decoded)
}

fn translate_source_row(
    row_terms: &[(usize, F)],
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let mut out = Vec::new();
    for &(source_column, scale) in row_terms {
        let terms = source_terms(source_column, decoded, row)?;
        out.extend(
            terms
                .into_iter()
                .map(|(column, coefficient)| (column, scale * coefficient)),
        );
    }
    Ok(out)
}

fn translate_event_lc(
    lc: &Lc,
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let mut source_row = lc.terms.clone();
    if lc.constant != F::ZERO {
        source_row.push((0, lc.constant));
    }
    translate_source_row(&source_row, decoded, row)
}

fn source_terms(
    column: usize,
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    decoded[column]
        .clone()
        .ok_or(GadgetNativeError::MissingDecodedColumn { column, row })
}

fn emit_k_mul(
    event: &KMulTraceEntry,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    let row = event.source_rows.start;
    let a0 = translate_event_lc(&event.a[0], decoded, row)?;
    let a1 = translate_event_lc(&event.a[1], decoded, row)?;
    let b0 = translate_event_lc(&event.b[0], decoded, row)?;
    let b1 = translate_event_lc(&event.b[1], decoded, row)?;
    let out0 = source_terms(event.output[0].col(), decoded, row)?;
    let out1 = source_terms(event.output[1].col(), decoded, row)?;
    let w = <Fq as BinomiallyExtendable<2>>::W;
    gates.product_sum(
        one_selector(),
        vec![(a0.clone(), b0.clone()), (scaled_terms(a1.clone(), w), b1.clone())],
        out0,
    );
    gates.product_sum(one_selector(), vec![(a0, b1), (a1, b0)], out1);
    Ok(())
}

fn emit_ring_mul(
    event: &RingMulToom3TraceEntry,
    slots: &RingSyntheticSlots,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    let row = event.source_rows.start;
    for (evaluation, convolution) in event.convolutions.iter().enumerate() {
        for coefficient in 0..TOOM_COEFFICIENTS {
            let mut products = Vec::new();
            for left in 0..TOOM_SPLIT {
                let Some(right) = coefficient.checked_sub(left) else {
                    continue;
                };
                if right >= TOOM_SPLIT {
                    continue;
                }
                products.push((
                    translate_event_lc(&convolution.lhs[left], decoded, row)?,
                    translate_event_lc(&convolution.rhs[right], decoded, row)?,
                ));
            }
            gates.product_sum(
                one_selector(),
                products,
                slot_terms(slots.coefficient(evaluation, coefficient)),
            );
        }
    }

    let product_groups = ring_product_groups(event);
    for output in 0..54 {
        let rhs = collapse_ring_output(output, &event.reduced_output_lcs[output], &product_groups, slots)?;
        gates.linear(
            one_selector(),
            source_terms(event.output[output].col(), decoded, row)?,
            rhs,
        );
    }
    Ok(())
}

fn ring_product_groups(event: &RingMulToom3TraceEntry) -> BTreeMap<usize, (usize, usize)> {
    let mut groups = BTreeMap::new();
    for (evaluation, convolution) in event.convolutions.iter().enumerate() {
        for left in 0..TOOM_SPLIT {
            for right in 0..TOOM_SPLIT {
                groups.insert(
                    convolution.products[left * TOOM_SPLIT + right].col(),
                    (evaluation, left + right),
                );
            }
        }
    }
    groups
}

fn collapse_ring_output(
    output: usize,
    lc: &Lc,
    product_groups: &BTreeMap<usize, (usize, usize)>,
    slots: &RingSyntheticSlots,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let normalized = normalize_lc(lc);
    let by_column = normalized.into_iter().collect::<BTreeMap<_, _>>();
    let mut group_coefficients = BTreeMap::<(usize, usize), F>::new();
    for (&column, &coefficient) in &by_column {
        if column == 0 {
            continue;
        }
        let Some(&group) = product_groups.get(&column) else {
            return Err(GadgetNativeError::RingOutputUnknownColumn { column });
        };
        if let Some(previous) = group_coefficients.insert(group, coefficient) {
            if previous != coefficient {
                return Err(GadgetNativeError::RingCoefficientMismatch {
                    output,
                    evaluation: group.0,
                    coefficient: group.1,
                });
            }
        }
    }
    for (&column, &(evaluation, coefficient)) in product_groups {
        let actual = by_column.get(&column).copied().unwrap_or(F::ZERO);
        let expected = group_coefficients
            .get(&(evaluation, coefficient))
            .copied()
            .unwrap_or(F::ZERO);
        if actual != expected {
            return Err(GadgetNativeError::RingCoefficientMismatch {
                output,
                evaluation,
                coefficient,
            });
        }
    }
    let mut out = Vec::new();
    if let Some(&constant) = by_column.get(&0) {
        out.push((0, constant));
    }
    for ((evaluation, coefficient), scale) in group_coefficients {
        out.extend(scaled_terms(
            slot_terms(slots.coefficient(evaluation, coefficient)),
            scale,
        ));
    }
    Ok(out)
}

fn scaled_terms(terms: Vec<(usize, F)>, scale: F) -> Vec<(usize, F)> {
    terms
        .into_iter()
        .map(|(column, coefficient)| (column, coefficient * scale))
        .collect()
}

fn eval_lc_from_source(lc: &Lc, source: &[F]) -> F {
    lc.terms
        .iter()
        .fold(lc.constant, |value, &(column, coefficient)| {
            value + coefficient * source[column]
        })
}

fn emit_goldilocks_canonicality(gates: &mut TraceGateBuilder, slot: ValueSlot) {
    let aux_start = slot
        .canonical_aux_start
        .expect("field slot has canonical auxiliaries");
    gates.product_sum(
        one_selector(),
        vec![(
            vec![(slot.start + HIGH_BITS_START, F::ONE)],
            vec![(slot.start + HIGH_BITS_START + 1, F::ONE)],
        )],
        vec![(aux_start, F::ONE)],
    );
    for high_offset in 2..32 {
        gates.product_sum(
            one_selector(),
            vec![(
                vec![(aux_start + high_offset - 2, F::ONE)],
                vec![(slot.start + HIGH_BITS_START + high_offset, F::ONE)],
            )],
            vec![(aux_start + high_offset - 1, F::ONE)],
        );
    }
    gates.product_sum(
        one_selector(),
        vec![(
            vec![(aux_start + CANONICAL_PREFIX_AUX - 1, F::ONE)],
            (0..32)
                .map(|bit| (slot.start + bit, F::from_u64(1u64 << bit)))
                .collect(),
        )],
        Vec::new(),
    );
}

fn one_selector() -> Vec<(usize, F)> {
    vec![(0, F::ONE)]
}

mod gate {
    pub const SELECTOR: usize = 0;
    pub const BITNESS: usize = 1;
    pub const PRODUCT_LEFT: usize = 2;
    pub const PRODUCT_RIGHT: usize = PRODUCT_LEFT + super::MAX_PRODUCT_TERMS;
    pub const PRODUCT_OUT: usize = PRODUCT_RIGHT + super::MAX_PRODUCT_TERMS;
    pub const SBOX_IN: usize = PRODUCT_OUT + 1;
    pub const SBOX_OUT: usize = SBOX_IN + 1;
    pub const LINEAR_LHS: usize = SBOX_OUT + 1;
    pub const LINEAR_RHS: usize = LINEAR_LHS + 1;
    pub const ARITY: usize = LINEAR_RHS + 1;
}

struct TraceGateBuilder {
    trips: Vec<Vec<(usize, usize, F)>>,
    rows: usize,
}

impl TraceGateBuilder {
    fn with_estimated_rows(estimated_rows: usize) -> Self {
        Self {
            trips: (0..gate::ARITY)
                .map(|_| Vec::with_capacity(estimated_rows))
                .collect(),
            rows: 0,
        }
    }

    fn bitness(&mut self, column: usize) {
        let row = self.begin_row(one_selector());
        self.trips[gate::BITNESS].push((row, column, F::ONE));
    }

    fn product_sum(
        &mut self,
        selector: Vec<(usize, F)>,
        products: Vec<(Vec<(usize, F)>, Vec<(usize, F)>)>,
        out: Vec<(usize, F)>,
    ) {
        assert!(products.len() <= MAX_PRODUCT_TERMS);
        let row = self.begin_row(selector);
        for (index, (left, right)) in products.into_iter().enumerate() {
            self.push_terms(gate::PRODUCT_LEFT + index, row, left);
            self.push_terms(gate::PRODUCT_RIGHT + index, row, right);
        }
        self.push_terms(gate::PRODUCT_OUT, row, out);
    }

    fn sbox7(&mut self, selector: Vec<(usize, F)>, input: Vec<(usize, F)>, out: Vec<(usize, F)>) {
        let row = self.begin_row(selector);
        self.push_terms(gate::SBOX_IN, row, input);
        self.push_terms(gate::SBOX_OUT, row, out);
    }

    fn linear(&mut self, selector: Vec<(usize, F)>, lhs: Vec<(usize, F)>, rhs: Vec<(usize, F)>) {
        let row = self.begin_row(selector);
        self.push_terms(gate::LINEAR_LHS, row, lhs);
        self.push_terms(gate::LINEAR_RHS, row, rhs);
    }

    fn begin_row(&mut self, selector: Vec<(usize, F)>) -> usize {
        let row = self.rows;
        self.rows += 1;
        self.push_terms(gate::SELECTOR, row, selector);
        row
    }

    fn push_terms(&mut self, matrix: usize, row: usize, terms: Vec<(usize, F)>) {
        self.trips[matrix].extend(
            terms
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != F::ZERO)
                .map(|(column, coefficient)| (row, column, coefficient)),
        );
    }

    fn finish(self, columns: usize) -> CcsStructure<F> {
        let matrices = self
            .trips
            .into_iter()
            .map(|trips| CcsMatrix::Csc(CscMat::from_triplets(trips, self.rows, columns)))
            .collect();
        let mut terms = Vec::with_capacity(MAX_PRODUCT_TERMS + 8);
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::BITNESS, 2)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::BITNESS, 1)]));
        for index in 0..MAX_PRODUCT_TERMS {
            terms.push(poly_term(
                F::ONE,
                &[
                    (gate::SELECTOR, 1),
                    (gate::PRODUCT_LEFT + index, 1),
                    (gate::PRODUCT_RIGHT + index, 1),
                ],
            ));
        }
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::PRODUCT_OUT, 1)]));
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::SBOX_IN, 7)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::SBOX_OUT, 1)]));
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::LINEAR_LHS, 1)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::LINEAR_RHS, 1)]));
        let polynomial = SparsePoly::new(gate::ARITY, terms);
        CcsStructure::new_sparse(matrices, polynomial).expect("gadget-native CCS is well formed")
    }
}

fn poly_term(coefficient: F, powers: &[(usize, u32)]) -> Term<F> {
    let mut exps = vec![0u32; gate::ARITY];
    for &(matrix, power) in powers {
        exps[matrix] = power;
    }
    Term {
        coeff: coefficient,
        exps,
    }
}

fn evaluate_ccs(structure: &CcsStructure<F>, assignment: &[F]) -> Vec<F> {
    let mut matrix_z = vec![vec![F::ZERO; structure.n]; structure.matrices.len()];
    for (index, matrix) in structure.matrices.iter().enumerate() {
        matrix.add_mul_into(assignment, &mut matrix_z[index], structure.n);
    }
    (0..structure.n)
        .map(|row| {
            let point = matrix_z
                .iter()
                .map(|values| values[row])
                .collect::<Vec<_>>();
            structure.f.eval(&point)
        })
        .collect()
}
