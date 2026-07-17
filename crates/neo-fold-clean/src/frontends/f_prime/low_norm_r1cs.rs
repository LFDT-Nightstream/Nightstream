//! Deterministic low-norm CCS encoding of a field-valued R1CS.
//!
//! Owns: canonical and derived single-branch witness plans, Goldilocks
//! reconstruction, source-row replay, and estimate-only arithmetic for a
//! proposed selector-gated branch encoding.
//!
//! Does not own: source-R1CS correctness, the outer `F'` relation, or folding
//! proof verification.
//!
//! Emits constraints: the single-branch encoders emit CCS rows that decode
//! committed bits and replay the source relation. The selector-gated surface
//! emits no constraints; it only returns a cost estimate.
//!
//! Authority boundary: the source R1CS and declared public columns define the
//! relation; canonical bits and proved linear definitions determine retained
//! values, while encoding estimates carry no authority.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Canonical oracle | [`encode_r1cs_oracle`] | yes | Canonical Goldilocks bits |
//! | Derived encoding | [`encode_r1cs_derived`] | yes | Source Boolean and acyclic-linear facts |
//! | Branch selection | selector-gated estimator | no | Formula only; materializer and soundness proof absent |

use std::collections::BTreeMap;

use neo_ajtai::AjtaiSModule;
use neo_ccs::CcsStructure;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::R1csSnapshot;
use crate::frontends::f_prime::structure::{gate, MixedGateBuilder};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, RelationError};

const FIELD_BITS: usize = 64;
const HIGH_BITS_START: usize = 32;
const CANONICAL_PREFIX_AUX: usize = 31;

/// Encoding policy for private source-R1CS wires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowNormR1csEncodingKind {
    /// Reference encoding: every non-public field wire uses 64 canonical bits.
    CanonicalOracle,
    /// Safe derived encoding: linearly determined wires are reconstructed,
    /// explicitly Boolean wires use one bit, and remaining wires retain the
    /// reference 64-bit representation.
    Derived,
}

/// Allocation/constraint count without materializing the encoded CCS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowNormR1csEstimate {
    pub source_rows: usize,
    pub source_cols: usize,
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub one_bit_source_cols: usize,
    pub canonical_field_source_cols: usize,
    pub linearly_derived_source_cols: usize,
}

/// Formula dimensions for a proposed direct selector-gated CCS lowering of
/// two R1CS branches. No production relation materializer currently audits
/// this estimate. Unlike selector composition back into R1CS, the proposed
/// design uses a degree-three CCS term and allocates no per-row residual
/// witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectorGatedR1csEstimate {
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub canonical_field_source_cols: usize,
    pub one_bit_source_cols: usize,
    pub linearly_derived_source_cols: usize,
    pub inactive_zero_rows: usize,
    pub max_degree: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VariableSlot {
    start: usize,
    bits: usize,
    canonical_aux_start: Option<usize>,
}

/// A source wire reconstructed from lower-index source wires.
#[derive(Clone, Debug)]
struct LinearDefinition {
    terms: Vec<(usize, F)>,
}

/// Witness-independent description of `enc_str(R)` and `enc_inst`.
#[derive(Clone, Debug)]
pub struct LowNormR1csPlan {
    kind: LowNormR1csEncodingKind,
    original_cols: usize,
    public_columns: Vec<usize>,
    public_input_len: usize,
    slots: Vec<Option<VariableSlot>>,
    linear_definitions: Vec<Option<LinearDefinition>>,
    encoded_cols: usize,
}

impl LowNormR1csPlan {
    pub fn kind(&self) -> LowNormR1csEncodingKind {
        self.kind
    }

    pub fn original_cols(&self) -> usize {
        self.original_cols
    }

    pub fn encoded_cols(&self) -> usize {
        self.encoded_cols
    }

    pub fn public_columns(&self) -> &[usize] {
        &self.public_columns
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn encoded_bits_for_column(&self, column: usize) -> Option<usize> {
        self.slots
            .get(column)
            .and_then(|slot| slot.map(|slot| slot.bits))
    }

    pub fn encoded_range_for_column(&self, column: usize) -> Option<std::ops::Range<usize>> {
        self.slots
            .get(column)
            .and_then(|slot| slot.map(|slot| slot.start..slot.start + slot.bits))
    }

    pub fn canonical_aux_range_for_column(&self, column: usize) -> Option<std::ops::Range<usize>> {
        self.slots.get(column).and_then(|slot| {
            slot.and_then(|slot| {
                slot.canonical_aux_start
                    .map(|start| start..start + CANONICAL_PREFIX_AUX)
            })
        })
    }

    pub fn is_linearly_derived(&self, column: usize) -> bool {
        self.linear_definitions
            .get(column)
            .is_some_and(Option::is_some)
    }

    /// Invert the canonical value encoding. This also validates all derived
    /// canonicality auxiliaries, so the map is genuinely one-to-one rather
    /// than merely field-congruent modulo Goldilocks.
    pub fn decode(&self, encoded: &[F]) -> Result<Vec<F>, LowNormR1csError> {
        if encoded.len() != self.encoded_cols {
            return Err(LowNormR1csError::EncodedLength {
                expected: self.encoded_cols,
                got: encoded.len(),
            });
        }
        if encoded.first().copied() != Some(F::ONE) {
            return Err(LowNormR1csError::EncodedConstantOne);
        }

        let mut decoded = vec![F::ZERO; self.original_cols];
        decoded[0] = F::ONE;
        for column in 1..self.original_cols {
            if let Some(slot) = self.slots[column] {
                let bits = &encoded[slot.start..slot.start + slot.bits];
                let value = decode_bits(column, bits)?;
                if slot.bits == FIELD_BITS && value >= F::ORDER_U64 {
                    return Err(LowNormR1csError::NonCanonicalField { column, value });
                }
                decoded[column] = F::from_u64(value);

                if let Some(aux_start) = slot.canonical_aux_start {
                    let expected = canonical_prefix_values(bits);
                    for (offset, expected_bit) in expected.into_iter().enumerate() {
                        let actual = encoded[aux_start + offset];
                        if actual != expected_bit {
                            return Err(LowNormR1csError::CanonicalAuxMismatch { column, offset });
                        }
                    }
                }
            } else {
                let definition = self.linear_definitions[column]
                    .as_ref()
                    .expect("every nonconstant source column is encoded or derived");
                decoded[column] = definition
                    .terms
                    .iter()
                    .fold(F::ZERO, |value, &(input, coefficient)| {
                        value + coefficient * decoded[input]
                    });
            }
        }
        Ok(decoded)
    }
}

/// Complete low-norm relation instance before Ajtai commitment.
#[derive(Clone, Debug)]
pub struct EncodedLowNormR1cs {
    pub structure: CcsStructure<F>,
    pub assignment: Vec<F>,
    pub plan: LowNormR1csPlan,
}

impl EncodedLowNormR1cs {
    pub fn public_input(&self) -> &[F] {
        &self.assignment[..self.plan.public_input_len]
    }

    pub fn private_witness(&self) -> &[F] {
        &self.assignment[self.plan.public_input_len..]
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        evaluate_mixed_gate(&self.structure, &self.assignment)
            .into_iter()
            .position(|value| value != F::ZERO)
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }

    pub fn decode(&self) -> Result<Vec<F>, LowNormR1csError> {
        if let Some(row) = self.first_unsatisfied_row() {
            return Err(LowNormR1csError::UnsatisfiedEncoding { row });
        }
        self.plan.decode(&self.assignment)
    }

    pub fn to_ccs_instance(&self, params: &Params, log: &AjtaiSModule) -> Result<CcsInstance, RelationError> {
        CcsInstance::from_low_norm_assignment(
            params,
            log,
            &self.structure,
            &self.assignment,
            self.plan.public_input_len,
        )
    }
}

#[derive(Debug, Error)]
pub enum LowNormR1csError {
    #[error("low-norm R1CS encoding requires source column 0 to equal one")]
    SourceConstantOne,
    #[error("public source column {column} is out of range for {cols} columns")]
    PublicColumnOutOfRange { column: usize, cols: usize },
    #[error("public source column {column} appears more than once")]
    DuplicatePublicColumn { column: usize },
    #[error("public source column {column} is not explicitly Boolean in the source relation")]
    PublicColumnNotBoolean { column: usize },
    #[error("selector-gated R1CS branches expose different public bit counts ({base} versus {recursive})")]
    BranchPublicInputLength { base: usize, recursive: usize },
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
    #[error("encoded low-norm relation is unsatisfied at row {row}")]
    UnsatisfiedEncoding { row: usize },
}

/// Build the canonical all-field reference encoding.
pub fn encode_r1cs_oracle(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
) -> Result<EncodedLowNormR1cs, LowNormR1csError> {
    encode_r1cs(source, public_bit_columns, LowNormR1csEncodingKind::CanonicalOracle)
}

/// Build the production-derived encoding and compare it against the same
/// source relation used by [`encode_r1cs_oracle`].
pub fn encode_r1cs_derived(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
) -> Result<EncodedLowNormR1cs, LowNormR1csError> {
    encode_r1cs(source, public_bit_columns, LowNormR1csEncodingKind::Derived)
}

/// Compute the exact row/column counts the selected lowering will emit.
pub fn estimate_r1cs_encoding(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
    kind: LowNormR1csEncodingKind,
) -> Result<LowNormR1csEstimate, LowNormR1csError> {
    if source.witness().first().copied() != Some(F::ONE) {
        return Err(LowNormR1csError::SourceConstantOne);
    }
    let explicit_bits = source.explicitly_boolean_columns();
    let mut is_public = vec![false; source.cols()];
    for &column in public_bit_columns {
        if column == 0 || column >= source.cols() {
            return Err(LowNormR1csError::PublicColumnOutOfRange {
                column,
                cols: source.cols(),
            });
        }
        if is_public[column] {
            return Err(LowNormR1csError::DuplicatePublicColumn { column });
        }
        if !explicit_bits[column] {
            return Err(LowNormR1csError::PublicColumnNotBoolean { column });
        }
        is_public[column] = true;
    }
    let (linearly_derived, linearly_derived_source_cols) = if kind == LowNormR1csEncodingKind::Derived {
        select_linear_definition_columns(source, &is_public)
    } else {
        (vec![false; source.cols()], 0)
    };
    let mut one_bit_source_cols = public_bit_columns.len();
    let mut canonical_field_source_cols = 0usize;
    let mut encoded_cols = 1usize + public_bit_columns.len();
    for column in 1..source.cols() {
        if is_public[column] || linearly_derived[column] {
            continue;
        }
        if kind == LowNormR1csEncodingKind::Derived && explicit_bits[column] {
            one_bit_source_cols += 1;
            encoded_cols += 1;
        } else {
            canonical_field_source_cols += 1;
            encoded_cols += FIELD_BITS + CANONICAL_PREFIX_AUX;
        }
    }
    let encoded_rows = (encoded_cols - 1)
        .saturating_add(canonical_field_source_cols.saturating_mul(32))
        .saturating_add(source.rows().saturating_sub(linearly_derived_source_cols));
    Ok(LowNormR1csEstimate {
        source_rows: source.rows(),
        source_cols: source.cols(),
        public_input_len: 1 + public_bit_columns.len(),
        encoded_cols,
        encoded_rows,
        one_bit_source_cols,
        canonical_field_source_cols,
        linearly_derived_source_cols,
    })
}

/// Estimate a proposed direct CCS representation of one relation with base
/// and recursive branches. Branch equations would be multiplied by `is_base`
/// or `1-is_base` in the CCS polynomial itself. Private slots are kept
/// separate; one decoded-value equation is budgeted for every inactive source
/// slot. This function is cost arithmetic only, not a materializer or a
/// selector-soundness audit.
pub fn estimate_selector_gated_r1cs_encoding(
    base: &R1csSnapshot,
    base_public_bit_columns: &[usize],
    recursive: &R1csSnapshot,
    recursive_public_bit_columns: &[usize],
    kind: LowNormR1csEncodingKind,
) -> Result<SelectorGatedR1csEstimate, LowNormR1csError> {
    if base_public_bit_columns.len() != recursive_public_bit_columns.len() {
        return Err(LowNormR1csError::BranchPublicInputLength {
            base: base_public_bit_columns.len(),
            recursive: recursive_public_bit_columns.len(),
        });
    }
    let base_estimate = estimate_r1cs_encoding(base, base_public_bit_columns, kind)?;
    let recursive_estimate = estimate_r1cs_encoding(recursive, recursive_public_bit_columns, kind)?;
    let public_bits = base_public_bit_columns.len();
    let base_private_one_bit = base_estimate.one_bit_source_cols - public_bits;
    let recursive_private_one_bit = recursive_estimate.one_bit_source_cols - public_bits;
    let one_bit_source_cols = public_bits + 1 + base_private_one_bit + recursive_private_one_bit;
    let canonical_field_source_cols =
        base_estimate.canonical_field_source_cols + recursive_estimate.canonical_field_source_cols;
    let private_encoded_cols = (base_estimate.encoded_cols - base_estimate.public_input_len)
        + (recursive_estimate.encoded_cols - recursive_estimate.public_input_len);
    let public_input_len = 1 + public_bits;
    let encoded_cols = public_input_len + 1 + private_encoded_cols;
    let inactive_zero_rows = base_private_one_bit + recursive_private_one_bit + canonical_field_source_cols;
    let branch_rows = base
        .rows()
        .saturating_sub(base_estimate.linearly_derived_source_cols)
        + recursive
            .rows()
            .saturating_sub(recursive_estimate.linearly_derived_source_cols);
    let encoded_rows = (encoded_cols - 1)
        .saturating_add(canonical_field_source_cols.saturating_mul(32))
        .saturating_add(inactive_zero_rows)
        .saturating_add(branch_rows);
    Ok(SelectorGatedR1csEstimate {
        public_input_len,
        encoded_cols,
        encoded_rows,
        canonical_field_source_cols,
        one_bit_source_cols,
        linearly_derived_source_cols: base_estimate.linearly_derived_source_cols
            + recursive_estimate.linearly_derived_source_cols,
        inactive_zero_rows,
        max_degree: 3,
    })
}

fn encode_r1cs(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
    kind: LowNormR1csEncodingKind,
) -> Result<EncodedLowNormR1cs, LowNormR1csError> {
    if source.witness().first().copied() != Some(F::ONE) {
        return Err(LowNormR1csError::SourceConstantOne);
    }
    let explicit_bits = source.explicitly_boolean_columns();
    let mut is_public = vec![false; source.cols()];
    for &column in public_bit_columns {
        if column == 0 || column >= source.cols() {
            return Err(LowNormR1csError::PublicColumnOutOfRange {
                column,
                cols: source.cols(),
            });
        }
        if is_public[column] {
            return Err(LowNormR1csError::DuplicatePublicColumn { column });
        }
        if !explicit_bits[column] {
            return Err(LowNormR1csError::PublicColumnNotBoolean { column });
        }
        is_public[column] = true;
    }

    let mut assignment = vec![F::ONE];
    let mut slots = vec![None; source.cols()];
    let (linear_definitions, removed_definition_rows) = if kind == LowNormR1csEncodingKind::Derived {
        build_linear_definitions(source, &is_public)
    } else {
        (vec![None; source.cols()], vec![false; source.rows()])
    };

    // `enc_inst`: public bits are exactly the first columns after ONE and
    // remain in caller-specified digest-bit order.
    for &column in public_bit_columns {
        let start = assignment.len();
        push_boolean_value(&mut assignment, source.witness()[column], column)?;
        slots[column] = Some(VariableSlot {
            start,
            bits: 1,
            canonical_aux_start: None,
        });
    }
    let public_input_len = assignment.len();

    // `enc`: all remaining source variables are private.
    for column in 1..source.cols() {
        if is_public[column] || linear_definitions[column].is_some() {
            continue;
        }
        let bits = if kind == LowNormR1csEncodingKind::Derived && explicit_bits[column] {
            1
        } else {
            FIELD_BITS
        };
        let start = assignment.len();
        if bits == 1 {
            push_boolean_value(&mut assignment, source.witness()[column], column)?;
        } else {
            push_field_bits(&mut assignment, source.witness()[column]);
        }
        slots[column] = Some(VariableSlot {
            start,
            bits,
            canonical_aux_start: None,
        });
    }

    // Canonicality auxiliaries are deterministic prefix products of the high
    // 32 bits. They are private and do not change `enc_inst`.
    for column in 1..source.cols() {
        let Some(mut slot) = slots[column] else { continue };
        if slot.bits != FIELD_BITS {
            continue;
        }
        slot.canonical_aux_start = Some(assignment.len());
        let values = canonical_prefix_values(&assignment[slot.start..slot.start + FIELD_BITS]);
        assignment.extend(values);
        slots[column] = Some(slot);
    }

    let mut gate_builder =
        MixedGateBuilder::with_estimated_rows(source.rows() + assignment.len() + source.cols().saturating_mul(32));
    for column in 1..assignment.len() {
        gate_builder.bitness(column);
    }
    for column in 1..source.cols() {
        let Some(slot) = slots[column] else { continue };
        if slot.bits == FIELD_BITS {
            emit_goldilocks_canonicality(&mut gate_builder, slot);
        }
    }
    let decoded_terms = build_decoded_terms(&slots, &linear_definitions);
    for row in 0..source.rows() {
        if removed_definition_rows[row] {
            continue;
        }
        gate_builder.product(
            translate_lc(source.a_row(row), &decoded_terms),
            translate_lc(source.b_row(row), &decoded_terms),
            translate_lc(source.c_row(row), &decoded_terms),
        );
    }

    let structure = gate_builder.finish(assignment.len());
    let plan = LowNormR1csPlan {
        kind,
        original_cols: source.cols(),
        public_columns: public_bit_columns.to_vec(),
        public_input_len,
        slots,
        linear_definitions,
        encoded_cols: assignment.len(),
    };
    Ok(EncodedLowNormR1cs {
        structure,
        assignment,
        plan,
    })
}

fn push_boolean_value(out: &mut Vec<F>, value: F, column: usize) -> Result<(), LowNormR1csError> {
    if value != F::ZERO && value != F::ONE {
        return Err(LowNormR1csError::BooleanWitness { column });
    }
    out.push(value);
    Ok(())
}

fn push_field_bits(out: &mut Vec<F>, value: F) {
    let value = value.as_canonical_u64();
    out.extend((0..FIELD_BITS).map(|bit| F::from_u64((value >> bit) & 1)));
}

fn decode_bits(column: usize, bits: &[F]) -> Result<u64, LowNormR1csError> {
    let mut value = 0u64;
    for (bit, &digit) in bits.iter().enumerate() {
        if digit != F::ZERO && digit != F::ONE {
            return Err(LowNormR1csError::NonBinaryDigit { column });
        }
        if digit == F::ONE {
            value |= 1u64 << bit;
        }
    }
    Ok(value)
}

fn canonical_prefix_values(bits: &[F]) -> Vec<F> {
    debug_assert_eq!(bits.len(), FIELD_BITS);
    let mut out = Vec::with_capacity(CANONICAL_PREFIX_AUX);
    let mut prefix = bits[HIGH_BITS_START] * bits[HIGH_BITS_START + 1];
    out.push(prefix);
    for &bit in &bits[HIGH_BITS_START + 2..] {
        prefix *= bit;
        out.push(prefix);
    }
    out
}

fn emit_goldilocks_canonicality(builder: &mut MixedGateBuilder, slot: VariableSlot) {
    let aux_start = slot
        .canonical_aux_start
        .expect("64-bit slot must reserve canonicality auxiliaries");
    builder.product(
        [(slot.start + HIGH_BITS_START, F::ONE)],
        [(slot.start + HIGH_BITS_START + 1, F::ONE)],
        [(aux_start, F::ONE)],
    );
    for high_offset in 2..32 {
        builder.product(
            [(aux_start + high_offset - 2, F::ONE)],
            [(slot.start + HIGH_BITS_START + high_offset, F::ONE)],
            [(aux_start + high_offset - 1, F::ONE)],
        );
    }
    let low = (0..32).map(|bit| (slot.start + bit, F::from_u64(1u64 << bit)));
    builder.product(
        [(aux_start + CANONICAL_PREFIX_AUX - 1, F::ONE)],
        low,
        std::iter::empty(),
    );
}

fn select_linear_definition_columns(source: &R1csSnapshot, is_public: &[bool]) -> (Vec<bool>, usize) {
    let mut defined = vec![false; source.cols()];
    let mut count = 0usize;
    for row in 0..source.rows() {
        let Some((column, _)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        defined[column] = true;
        count += 1;
    }
    (defined, count)
}

fn build_linear_definitions(source: &R1csSnapshot, is_public: &[bool]) -> (Vec<Option<LinearDefinition>>, Vec<bool>) {
    let mut definitions = vec![None; source.cols()];
    let mut removed_rows = vec![false; source.rows()];
    let mut defined = vec![false; source.cols()];
    for row in 0..source.rows() {
        let Some((column, coefficient)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        let (positive, negative) = linear_difference(source, row).expect("candidate row is linear");
        let inverse = coefficient.inverse();
        let mut terms = Vec::new();
        visit_difference_terms(positive, negative, |input, input_coefficient| {
            if input != column {
                let derived_coefficient = -input_coefficient * inverse;
                if derived_coefficient != F::ZERO {
                    terms.push((input, derived_coefficient));
                }
            }
        });
        debug_assert!(terms.iter().all(|&(input, _)| input < column));
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
    if is_constant_one_row(source.b_row(row)) {
        Some((source.a_row(row), source.c_row(row)))
    } else if is_constant_one_row(source.a_row(row)) {
        Some((source.b_row(row), source.c_row(row)))
    } else {
        None
    }
}

fn is_constant_one_row(row: &[(usize, F)]) -> bool {
    row == [(0, F::ONE)]
}

fn visit_difference_terms(positive: &[(usize, F)], negative: &[(usize, F)], mut visit: impl FnMut(usize, F)) {
    let mut left = 0usize;
    let mut right = 0usize;
    while left < positive.len() || right < negative.len() {
        let column = match (positive.get(left), negative.get(right)) {
            (Some(&(left_column, _)), Some(&(right_column, _))) => left_column.min(right_column),
            (Some(&(left_column, _)), None) => left_column,
            (None, Some(&(right_column, _))) => right_column,
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

fn build_decoded_terms(
    slots: &[Option<VariableSlot>],
    definitions: &[Option<LinearDefinition>],
) -> Vec<Vec<(usize, F)>> {
    let mut decoded = vec![Vec::new(); slots.len()];
    decoded[0].push((0, F::ONE));
    for column in 1..slots.len() {
        if let Some(slot) = slots[column] {
            decoded[column].extend((0..slot.bits).map(|bit| (slot.start + bit, F::from_u64(1u64 << bit))));
            continue;
        }
        let definition = definitions[column]
            .as_ref()
            .expect("every nonconstant source column is encoded or derived");
        let mut combined = BTreeMap::<usize, F>::new();
        for &(input, scale) in &definition.terms {
            debug_assert!(input < column);
            for &(encoded_column, coefficient) in &decoded[input] {
                *combined.entry(encoded_column).or_insert(F::ZERO) += scale * coefficient;
            }
        }
        decoded[column] = combined
            .into_iter()
            .filter(|(_, coefficient)| *coefficient != F::ZERO)
            .collect();
    }
    decoded
}

fn translate_lc(row: &[(usize, F)], decoded_terms: &[Vec<(usize, F)>]) -> Vec<(usize, F)> {
    let mut out = Vec::new();
    for &(source_column, coeff) in row {
        for &(encoded_column, source_coefficient) in &decoded_terms[source_column] {
            out.push((encoded_column, coeff * source_coefficient));
        }
    }
    out
}

fn evaluate_mixed_gate(structure: &CcsStructure<F>, assignment: &[F]) -> Vec<F> {
    assert_eq!(structure.m, assignment.len());
    assert_eq!(structure.matrices.len(), gate::ARITY);
    let mut matrix_z: [Vec<F>; gate::ARITY] = std::array::from_fn(|_| vec![F::ZERO; structure.n]);
    for (matrix_index, matrix) in structure.matrices.iter().enumerate() {
        matrix.add_mul_into(assignment, &mut matrix_z[matrix_index], structure.n);
    }
    (0..structure.n)
        .map(|row| {
            let point: [F; gate::ARITY] = std::array::from_fn(|matrix_index| matrix_z[matrix_index][row]);
            structure.f.eval(&point)
        })
        .collect()
}
