//! Exact packed lowering for one sampler chunk's mod-5 source block.
//!
//! Owns: 20-row/19-column trace validation, the fifteen-coordinate witness,
//! projected source-wire reconstruction, and eight packed CCS equations.
//!
//! Does not own: chunk-bit derivation, sampler acceptance, symbol selection,
//! or stage-level cost aggregation.
//!
//! Emits constraints: yes, seven packed bit-pair rows and one packed residue
//! row per traced chunk. Its fifteen coordinates receive no common gates.
//!
//! Authority boundary: source rows remain the local implementation arithmetic
//! reference until replayed exactly. The packed equations use the fixed
//! Goldilocks nonresidue seven; no digest or prover-supplied summary substitutes
//! for source arithmetic. Sampler-level semantic necessity is proved elsewhere.
//!
//! | Stage path | Source obligation | Encoded coordinates | Packed rows | Lean theorem |
//! |---|---|---:|---:|---|
//! | `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.low_bit_pairs` | low quotient bits 0 through 11 are Boolean | 12 | 6 | `Mod5.packedRows_iff_directRows` |
//! | `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.high_bit_pair` | low bit 12 and one derived high bit are Boolean | 1 | 1 | `Mod5.packedRows_iff_directRows` |
//! | `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.residue_pair` | two centered coordinates select residues 0 through 4 | 2 | 1 | `Mod5.packedRows_iff_directRows` |

use std::collections::BTreeMap;
use std::ops::Range;

use neo_ccs::Term;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::{Lc, Mod5TraceEntry, R1csEncodingTrace, R1csSnapshot, Var};

use super::gates::{gate, poly_term, TraceGateBuilder};
use super::slots::{slot_terms, ValueEncoding, ValueSlot};
use super::{
    claim_gadget_column, claim_rows, source_terms, validate_row, GadgetNativeError, GadgetNativePlan,
    ProductDefinition, SourceColumn,
};

const GADGET: &str = "sampler mod-5 chunk";
const SOURCE_ROWS_PER_CHUNK: usize = 20;
const SOURCE_COLUMNS_PER_CHUNK: usize = 19;
const QUOTIENT_BITS: usize = 14;
pub(super) const LOW_QUOTIENT_BITS: usize = QUOTIENT_BITS - 1;
const ALPHABET_SIZE: u64 = 5;
const HIGH_WEIGHT: u64 = 1 << LOW_QUOTIENT_BITS;
const HIGH_DENOMINATOR: u64 = ALPHABET_SIZE * HIGH_WEIGHT;
const NONRESIDUE: u64 = super::gates::GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE;

pub(super) const LOW_BIT_COORDINATES_PER_CHUNK: usize = LOW_QUOTIENT_BITS;
pub(super) const RESIDUE_COORDINATES_PER_CHUNK: usize = 2;
pub(super) const ENCODED_COORDINATES_PER_CHUNK: usize = LOW_BIT_COORDINATES_PER_CHUNK + RESIDUE_COORDINATES_PER_CHUNK;
pub(super) const LOW_BIT_PAIR_ROWS_PER_CHUNK: usize = 6;
pub(super) const HIGH_BIT_PAIR_ROWS_PER_CHUNK: usize = 1;
pub(super) const RESIDUE_PAIR_ROWS_PER_CHUNK: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProjectedRole {
    EncodedLinear { permits_escape: bool },
    Product,
}

/// Exact source ownership established before any mod-5 row is removed.
pub(super) struct ValidatedMod5 {
    projected_roles: Vec<Option<ProjectedRole>>,
    chunks: usize,
}

impl ValidatedMod5 {
    pub(super) fn validate_and_claim(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        covered_rows: &mut [bool],
        gadget_columns: &mut [bool],
    ) -> Result<Self, GadgetNativeError> {
        let mut projected_roles = vec![None; source.cols()];
        let mut owned_columns = vec![false; source.cols()];
        let mut strict_owner = vec![None; source.cols()];

        for (chunk, event) in trace.mod5_chunks().iter().enumerate() {
            validate_geometry(source, chunk, event, &mut owned_columns, gadget_columns)?;
            validate_rows(source, event)?;
            claim_rows(source, GADGET, &event.source_rows, covered_rows)?;

            install_projected_role(
                chunk,
                event.index,
                ProjectedRole::EncodedLinear { permits_escape: true },
                &mut projected_roles,
                gadget_columns,
            )?;
            for variable in [event.quotient, event.quotient_bits[LOW_QUOTIENT_BITS]] {
                install_projected_role(
                    chunk,
                    variable,
                    ProjectedRole::EncodedLinear { permits_escape: false },
                    &mut projected_roles,
                    gadget_columns,
                )?;
                strict_owner[variable.col()] = Some(chunk);
            }
            for variable in event.index_products {
                install_projected_role(
                    chunk,
                    variable,
                    ProjectedRole::Product,
                    &mut projected_roles,
                    gadget_columns,
                )?;
                strict_owner[variable.col()] = Some(chunk);
            }
        }

        // A quotient, high bit, or index-product temporary may occur only in
        // its exact 20-row source block. Index is deliberately excluded: the
        // following symbol row consumes its checked linear representation.
        for row in 0..source.rows() {
            for &(column, _) in source
                .a_row(row)
                .iter()
                .chain(source.b_row(row))
                .chain(source.c_row(row))
            {
                if let Some(chunk) = strict_owner[column] {
                    if !trace.mod5_chunks()[chunk].source_rows.contains(&row) {
                        return Err(GadgetNativeError::GadgetTemporaryEscapes { column });
                    }
                }
            }
        }

        Ok(Self {
            projected_roles,
            chunks: trace.mod5_chunks().len(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.chunks
    }

    pub(super) fn projected_role(&self, column: usize) -> Option<ProjectedRole> {
        self.projected_roles[column]
    }

    pub(super) fn permits_escape(&self, column: usize) -> bool {
        matches!(
            self.projected_roles[column],
            Some(ProjectedRole::EncodedLinear { permits_escape: true })
        )
    }

    pub(super) fn linear_column_count(&self) -> usize {
        self.projected_roles
            .iter()
            .filter(|role| matches!(role, Some(ProjectedRole::EncodedLinear { .. })))
            .count()
    }
}

fn validate_geometry(
    source: &R1csSnapshot,
    chunk: usize,
    event: &Mod5TraceEntry,
    owned_columns: &mut [bool],
    gadget_columns: &[bool],
) -> Result<(), GadgetNativeError> {
    if event.source_rows.len() != SOURCE_ROWS_PER_CHUNK || event.source_rows.end > source.rows() {
        return Err(geometry(chunk, "20-row source interval"));
    }
    if event.allocated_columns.len() != SOURCE_COLUMNS_PER_CHUNK
        || event.allocated_columns.start == 0
        || event.allocated_columns.end > source.cols()
    {
        return Err(geometry(chunk, "19-column allocation interval"));
    }

    let first = event.allocated_columns.start;
    let expected = [
        event.index.col(),
        event.quotient.col(),
        event.index_products[0].col(),
        event.index_products[1].col(),
        event.index_products[2].col(),
    ];
    if expected != [first, first + 1, first + 2, first + 3, first + 4]
        || event
            .quotient_bits
            .iter()
            .enumerate()
            .any(|(offset, variable)| variable.col() != first + 5 + offset)
    {
        return Err(geometry(chunk, "production column role order"));
    }
    if event
        .chunk_bits
        .iter()
        .any(|variable| variable.col() == 0 || variable.col() >= first)
    {
        return Err(geometry(chunk, "topological chunk-bit inputs"));
    }

    for column in event.allocated_columns.clone() {
        if std::mem::replace(&mut owned_columns[column], true) {
            return Err(geometry(chunk, "overlapping source columns"));
        }
    }
    for variable in &event.quotient_bits[..LOW_QUOTIENT_BITS] {
        if gadget_columns[variable.col()] {
            return Err(geometry(chunk, "low quotient bit owned by another gadget"));
        }
    }
    Ok(())
}

fn validate_rows(source: &R1csSnapshot, event: &Mod5TraceEntry) -> Result<(), GadgetNativeError> {
    let index = Lc::from_var(event.index);
    let factor = |value: u64| {
        index
            .clone()
            .add_scaled(&Lc::from_const(F::from_u64(value)), -F::ONE)
    };
    validate_row(
        source,
        GADGET,
        event.source_rows.start,
        &index,
        &factor(1),
        &Lc::from_var(event.index_products[0]),
    )?;
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 1,
        &Lc::from_var(event.index_products[0]),
        &factor(2),
        &Lc::from_var(event.index_products[1]),
    )?;
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 2,
        &Lc::from_var(event.index_products[1]),
        &factor(3),
        &Lc::from_var(event.index_products[2]),
    )?;
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 3,
        &Lc::from_var(event.index_products[2]),
        &factor(4),
        &Lc::zero(),
    )?;

    for (offset, &bit) in event.quotient_bits.iter().enumerate() {
        let bit_lc = Lc::from_var(bit);
        let minus_one = bit_lc.clone().add_scaled(&Lc::from_const(F::ONE), -F::ONE);
        validate_row(
            source,
            GADGET,
            event.source_rows.start + 4 + offset,
            &bit_lc,
            &minus_one,
            &Lc::zero(),
        )?;
    }

    let quotient_bits = little_endian_lc(&event.quotient_bits);
    let quotient_difference = Lc::from_var(event.quotient).add_scaled(&quotient_bits, -F::ONE);
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 18,
        &quotient_difference,
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
    )?;

    let chunk_lc = little_endian_lc(&event.chunk_bits);
    let mut decomposition_difference = chunk_lc;
    decomposition_difference.add_term(event.quotient, -F::from_u64(ALPHABET_SIZE));
    decomposition_difference.add_term(event.index, -F::ONE);
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 19,
        &decomposition_difference,
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
    )
}

fn little_endian_lc<const N: usize>(bits: &[Var; N]) -> Lc {
    let mut out = Lc::zero();
    let mut power = F::ONE;
    for &bit in bits {
        out.add_term(bit, power);
        power += power;
    }
    out
}

fn install_projected_role(
    chunk: usize,
    variable: Var,
    role: ProjectedRole,
    roles: &mut [Option<ProjectedRole>],
    gadget_columns: &mut [bool],
) -> Result<(), GadgetNativeError> {
    let column = variable.col();
    if roles[column].replace(role).is_some() {
        return Err(geometry(chunk, "duplicate projected role"));
    }
    claim_gadget_column(column, gadget_columns)
}

fn geometry(chunk: usize, detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::PackedMod5Geometry { chunk, detail }
}

#[derive(Clone, Debug)]
struct PackedMod5ChunkSlots {
    low_bits: [ValueSlot; LOW_QUOTIENT_BITS],
    residue_left: ValueSlot,
    residue_right: ValueSlot,
    decoder_columns: [usize; 3],
    product_columns: [usize; 3],
}

/// One exact source-product definition projected out by packed Mod-5 lowering.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct PackedMod5ProductDecoderAudit<'a> {
    pub output: usize,
    pub left_terms: &'a [(usize, F)],
    pub left_constant: F,
    pub right_terms: &'a [(usize, F)],
    pub right_constant: F,
}

/// Narrow exact decoder view consumed by the generated Lean artifact bridge.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct PackedMod5DecoderAudit<'a> {
    pub index: &'a [(usize, F)],
    pub high: &'a [(usize, F)],
    pub quotient: &'a [(usize, F)],
    pub products: [PackedMod5ProductDecoderAudit<'a>; 3],
}

/// Assignment coordinates omitted from the common coordinate-gate schedule.
#[derive(Clone, Debug, Default)]
pub(super) struct PackedMod5Slots {
    chunks: Vec<PackedMod5ChunkSlots>,
    omitted_coordinates: Vec<bool>,
}

impl PackedMod5Slots {
    pub(super) fn low_bit_range(&self, chunk: usize) -> Option<Range<usize>> {
        let chunk = self.chunks.get(chunk)?;
        let start = chunk.low_bits[0].start;
        chunk
            .low_bits
            .iter()
            .enumerate()
            .all(|(offset, slot)| slot.start == start + offset && slot.width == 1)
            .then_some(start..start + LOW_QUOTIENT_BITS)
    }

    pub(super) fn residue_range(&self, chunk: usize) -> Option<Range<usize>> {
        let chunk = self.chunks.get(chunk)?;
        (chunk.residue_left.start + 1 == chunk.residue_right.start)
            .then_some(chunk.residue_left.start..chunk.residue_right.start + 1)
    }

    pub(super) fn omits_coordinate(&self, column: usize) -> bool {
        self.omitted_coordinates
            .get(column)
            .copied()
            .unwrap_or(false)
    }
}

impl GadgetNativePlan {
    pub fn packed_mod5_low_bit_range(&self, chunk: usize) -> Option<Range<usize>> {
        self.mod5_slots.low_bit_range(chunk)
    }

    pub fn packed_mod5_residue_range(&self, chunk: usize) -> Option<Range<usize>> {
        self.mod5_slots.residue_range(chunk)
    }

    #[doc(hidden)]
    pub fn packed_mod5_decoder_audit(&self, chunk: usize) -> Option<PackedMod5DecoderAudit<'_>> {
        let slots = self.mod5_slots.chunks.get(chunk)?;
        let [index, high, quotient] = slots.decoder_columns;
        let SourceColumn::EncodedLinear(index) = self.source_columns.get(index)? else {
            return None;
        };
        let SourceColumn::EncodedLinear(high) = self.source_columns.get(high)? else {
            return None;
        };
        let SourceColumn::EncodedLinear(quotient) = self.source_columns.get(quotient)? else {
            return None;
        };
        let [product0, product1, product2] = slots.product_columns;
        let product0 = packed_mod5_product_decoder_audit(&self.source_columns, product0)?;
        let product1 = packed_mod5_product_decoder_audit(&self.source_columns, product1)?;
        let product2 = packed_mod5_product_decoder_audit(&self.source_columns, product2)?;
        Some(PackedMod5DecoderAudit {
            index,
            high,
            quotient,
            products: [product0, product1, product2],
        })
    }
}

fn packed_mod5_product_decoder_audit(
    source_columns: &[SourceColumn],
    output: usize,
) -> Option<PackedMod5ProductDecoderAudit<'_>> {
    let SourceColumn::Product(definition) = source_columns.get(output)? else {
        return None;
    };
    Some(PackedMod5ProductDecoderAudit {
        output,
        left_terms: &definition.left.terms,
        left_constant: definition.left.constant,
        right_terms: &definition.right.terms,
        right_constant: definition.right.constant,
    })
}

pub(super) fn allocate_and_install(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    validated: &ValidatedMod5,
    assignment: &mut Vec<F>,
    source_columns: &mut [Option<SourceColumn>],
) -> Result<PackedMod5Slots, GadgetNativeError> {
    if validated.len() != trace.mod5_chunks().len() {
        return Err(geometry(0, "validated trace census"));
    }
    let mut chunks = Vec::with_capacity(validated.len());
    for (chunk, event) in trace.mod5_chunks().iter().enumerate() {
        let low_bits = event.quotient_bits[..LOW_QUOTIENT_BITS]
            .iter()
            .map(|variable| encoded_singleton(source_columns, variable.col(), chunk))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("thirteen low quotient bits");
        let index_value = source.witness()[event.index.col()].as_canonical_u64();
        let Some((left, right)) = centered_residue_pair(index_value) else {
            return Err(GadgetNativeError::PackedMod5Witness {
                chunk,
                column: event.index.col(),
            });
        };
        let residue_left = push_centered(assignment, left);
        let residue_right = push_centered(assignment, right);
        let chunk_slots = PackedMod5ChunkSlots {
            low_bits,
            residue_left,
            residue_right,
            decoder_columns: [
                event.index.col(),
                event.quotient_bits[LOW_QUOTIENT_BITS].col(),
                event.quotient.col(),
            ],
            product_columns: event.index_products.map(Var::col),
        };
        install_source_definitions(source, event, chunk, &chunk_slots, assignment, source_columns)?;
        chunks.push(chunk_slots);
    }

    let mut omitted_coordinates = vec![false; assignment.len()];
    for chunk in &chunks {
        for slot in chunk
            .low_bits
            .iter()
            .chain([&chunk.residue_left, &chunk.residue_right])
        {
            if slot.width != 1 || std::mem::replace(&mut omitted_coordinates[slot.start], true) {
                return Err(geometry(0, "overlapping packed coordinate"));
            }
        }
    }
    Ok(PackedMod5Slots {
        chunks,
        omitted_coordinates,
    })
}

fn encoded_singleton(
    source_columns: &[Option<SourceColumn>],
    column: usize,
    chunk: usize,
) -> Result<ValueSlot, GadgetNativeError> {
    match source_columns.get(column).and_then(Option::as_ref) {
        Some(SourceColumn::Encoded(slot)) if slot.width == 1 && matches!(slot.encoding, ValueEncoding::Boolean) => {
            Ok(*slot)
        }
        _ => Err(geometry(chunk, "low quotient bit encoding")),
    }
}

fn push_centered(assignment: &mut Vec<F>, value: F) -> ValueSlot {
    let start = assignment.len();
    assignment.push(value);
    ValueSlot {
        start,
        width: 1,
        encoding: ValueEncoding::CenteredUnit,
    }
}

fn centered_residue_pair(index: u64) -> Option<(F, F)> {
    match index {
        0 => Some((-F::ONE, -F::ONE)),
        1 => Some((-F::ONE, F::ZERO)),
        2 => Some((F::ZERO, F::ZERO)),
        3 => Some((F::ONE, F::ZERO)),
        4 => Some((F::ONE, F::ONE)),
        _ => None,
    }
}

fn install_source_definitions(
    source: &R1csSnapshot,
    event: &Mod5TraceEntry,
    chunk: usize,
    slots: &PackedMod5ChunkSlots,
    assignment: &[F],
    source_columns: &mut [Option<SourceColumn>],
) -> Result<(), GadgetNativeError> {
    let index_terms = normalize_terms(
        slot_terms(slots.residue_left)
            .into_iter()
            .chain(slot_terms(slots.residue_right))
            .chain([(0, F::from_u64(2))]),
    );
    install_encoded_linear(
        source,
        event.index,
        chunk,
        index_terms.clone(),
        assignment,
        source_columns,
    )?;

    let mut high_terms = Vec::new();
    let mut power = F::ONE;
    for &bit in &event.chunk_bits {
        high_terms.extend(scale_terms(
            encoded_source_terms(source_columns, bit.col(), chunk)?,
            power,
        ));
        power += power;
    }
    let mut power = F::ONE;
    for slot in &slots.low_bits {
        high_terms.extend(scale_terms(slot_terms(*slot), -F::from_u64(ALPHABET_SIZE) * power));
        power += power;
    }
    high_terms.extend(scale_terms(index_terms, -F::ONE));
    let denominator_inverse = F::from_u64(HIGH_DENOMINATOR).inverse();
    let high_terms = normalize_terms(scale_terms(high_terms, denominator_inverse));

    let high = event.quotient_bits[LOW_QUOTIENT_BITS];
    install_encoded_linear(source, high, chunk, high_terms.clone(), assignment, source_columns)?;

    let mut quotient_terms = Vec::new();
    let mut power = F::ONE;
    for slot in &slots.low_bits {
        quotient_terms.extend(scale_terms(slot_terms(*slot), power));
        power += power;
    }
    quotient_terms.extend(scale_terms(high_terms, F::from_u64(HIGH_WEIGHT)));
    install_encoded_linear(
        source,
        event.quotient,
        chunk,
        normalize_terms(quotient_terms),
        assignment,
        source_columns,
    )?;

    let index = Lc::from_var(event.index);
    let factors = [1u64, 2, 3];
    for (offset, (&product, factor)) in event.index_products.iter().zip(factors).enumerate() {
        let left = if offset == 0 {
            index.clone()
        } else {
            Lc::from_var(event.index_products[offset - 1])
        };
        let right = index
            .clone()
            .add_scaled(&Lc::from_const(F::from_u64(factor)), -F::ONE);
        let definition = ProductDefinition { left, right };
        let value = super::eval_lc_from_source(&definition.left, source.witness())
            * super::eval_lc_from_source(&definition.right, source.witness());
        if value != source.witness()[product.col()] {
            return Err(GadgetNativeError::PackedMod5Witness {
                chunk,
                column: product.col(),
            });
        }
        if source_columns[product.col()]
            .replace(SourceColumn::Product(definition))
            .is_some()
        {
            return Err(geometry(chunk, "projected source definition overlap"));
        }
    }
    Ok(())
}

fn install_encoded_linear(
    source: &R1csSnapshot,
    variable: Var,
    chunk: usize,
    terms: Vec<(usize, F)>,
    assignment: &[F],
    source_columns: &mut [Option<SourceColumn>],
) -> Result<(), GadgetNativeError> {
    let value = terms.iter().fold(F::ZERO, |sum, &(column, coefficient)| {
        sum + assignment[column] * coefficient
    });
    if value != source.witness()[variable.col()] {
        return Err(GadgetNativeError::PackedMod5Witness {
            chunk,
            column: variable.col(),
        });
    }
    if source_columns[variable.col()]
        .replace(SourceColumn::EncodedLinear(terms))
        .is_some()
    {
        return Err(geometry(chunk, "projected source definition overlap"));
    }
    Ok(())
}

fn encoded_source_terms(
    source_columns: &[Option<SourceColumn>],
    column: usize,
    chunk: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    match source_columns.get(column).and_then(Option::as_ref) {
        Some(SourceColumn::Encoded(slot)) => Ok(slot_terms(*slot)),
        Some(SourceColumn::EncodedLinear(terms)) => Ok(terms.clone()),
        Some(SourceColumn::Linear(definition)) | Some(SourceColumn::GadgetLinear(definition)) => {
            let mut terms = Vec::new();
            for &(input, coefficient) in &definition.terms {
                terms.extend(scale_terms(
                    encoded_source_terms(source_columns, input, chunk)?,
                    coefficient,
                ));
            }
            Ok(normalize_terms(terms))
        }
        _ => Err(geometry(chunk, "chunk-bit linear representation")),
    }
}

fn scale_terms(terms: impl IntoIterator<Item = (usize, F)>, scale: F) -> Vec<(usize, F)> {
    terms
        .into_iter()
        .map(|(column, coefficient)| (column, coefficient * scale))
        .collect()
}

fn normalize_terms(terms: impl IntoIterator<Item = (usize, F)>) -> Vec<(usize, F)> {
    let mut normalized = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *normalized.entry(column).or_insert(F::ZERO) += coefficient;
    }
    normalized
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

pub(super) fn emit(
    trace: &R1csEncodingTrace,
    slots: &PackedMod5Slots,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    if trace.mod5_chunks().len() != slots.chunks.len() {
        return Err(geometry(0, "emission trace census"));
    }
    for (event, slots) in trace.mod5_chunks().iter().zip(&slots.chunks) {
        for pair in 0..6 {
            gates.quadratic_bit_pair(
                source_terms(event.quotient_bits[2 * pair].col(), decoded, event.source_rows.start)?,
                source_terms(
                    event.quotient_bits[2 * pair + 1].col(),
                    decoded,
                    event.source_rows.start,
                )?,
            );
        }
        gates.quadratic_bit_pair(
            source_terms(event.quotient_bits[12].col(), decoded, event.source_rows.start)?,
            source_terms(event.quotient_bits[13].col(), decoded, event.source_rows.start)?,
        );
        gates.mod5_residue_pair(slot_terms(slots.residue_left), slot_terms(slots.residue_right));
    }
    Ok(())
}

/// Add the exact expanded centered-residue residual to the common CCS polynomial.
pub(super) fn append_residue_polynomial_terms(terms: &mut Vec<Term<F>>) {
    let selector = gate::SELECTOR;
    let left = gate::MOD5_RESIDUE_LEFT;
    let right = gate::MOD5_RESIDUE_RIGHT;
    terms.extend([
        poly_term(F::ONE, &[(selector, 1), (left, 6)]),
        poly_term(-F::from_u64(2), &[(selector, 1), (left, 4)]),
        poly_term(F::ONE, &[(selector, 1), (left, 2)]),
        poly_term(-F::from_u64(NONRESIDUE), &[(selector, 1), (left, 2), (right, 2)]),
        poly_term(F::from_u64(2 * NONRESIDUE), &[(selector, 1), (left, 1), (right, 3)]),
        poly_term(-F::from_u64(NONRESIDUE), &[(selector, 1), (right, 4)]),
    ]);
}
