//! Constraint leaves for one little-endian 16-bit sampler chunk.
//!
//! Owns: acceptance, mod-5 decomposition, centered-symbol mapping, and prefix
//! recurrence for one fixed chunk.
//!
//! Does not own: digest-lane bit decomposition or output selection.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: the 16 canonical lane bits define `chunk`; quotient,
//! residue, and prefix witnesses are accepted only through these equations.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `challenge.sampler.chunk.accept` | `enforce_accept` | `accept = 1 ↔ chunk != 65535`, canonical inverse | 64 per rho | four source rows, two columns | aggregate lowering | `canonicalAcceptanceSourceRows_exists_iff` |
//! | `challenge.sampler.chunk.mod5` | `enforce_mod5` | `chunk = 5*q + index`, `index < 5` | 64 per rho | quotient/residue range rows | generic R1CS | `chunk_decomposition` |
//! | `challenge.sampler.chunk.symbol_and_prefix` | `enforce_symbol_and_prefix` | `symbol=index-2`, `next=prior+accept` | 64 per rho | two equalities | generic R1CS | `symbol_mem_alphabet` |

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::encoding_trace::{AcceptanceTraceEntry, Mod5TraceEntry};

use super::pi_rlc_challenge_stage;

const BUCKET: u64 = 65_535;
const ALPHABET_SIZE: u64 = 5;
const QUOTIENT_BITS: usize = 14;

pub(super) struct ChunkRecord {
    pub(super) accept: Var,
    pub(super) symbol: Var,
    pub(super) cumulative: Var,
}

struct AcceptResult {
    wire: Var,
    value: F,
}

struct Mod5Result {
    index: Var,
    value: u64,
}

pub(super) fn process_chunk(builder: &mut R1csBuilder, bits: &[Var], cumulative: Var) -> ChunkRecord {
    debug_assert_eq!(bits.len(), 16);
    builder.begin_encoding_stage(pi_rlc_challenge_stage::CHUNK);
    let chunk = chunk_linear_combination(bits);
    let chunk_value = builder.eval(&chunk);
    let chunk_u64 = canonical_u64(chunk_value);
    let chunk_bits: [Var; 16] = bits.try_into().expect("sampler chunk width");
    let accept = enforce_accept(builder, chunk_bits, &chunk, chunk_value);
    let residue = enforce_mod5(builder, bits, &chunk, chunk_u64);
    enforce_symbol_and_prefix(builder, residue, cumulative, accept)
}

fn chunk_linear_combination(bits: &[Var]) -> Lc {
    // The low-word distribution has one extra raw zero. Complement maps that
    // value to the existing rejected candidate 65535 without extra wires.
    let mut chunk = Lc::from_const(F::from_u64(BUCKET));
    let mut power = F::ONE;
    for &bit in bits {
        chunk.add_term(bit, -power);
        power += power;
    }
    chunk
}

fn enforce_accept(builder: &mut R1csBuilder, chunk_bits: [Var; 16], chunk: &Lc, chunk_value: F) -> AcceptResult {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::CHUNK_ACCEPT);
    let source_row_start = builder.rows();
    let allocated_column_start = builder.cols();
    let difference_value = chunk_value - F::from_u64(BUCKET);
    let value = if difference_value == F::ZERO { F::ZERO } else { F::ONE };
    let wire = builder.alloc(value);
    enforce_bit(builder, wire);

    let inverse_value = if difference_value == F::ZERO {
        F::ZERO
    } else {
        difference_value.inverse()
    };
    let inverse = builder.alloc(inverse_value);
    let difference = chunk
        .clone()
        .add_scaled(&Lc::from_const(F::from_u64(BUCKET)), -F::ONE);
    let mut one_minus_accept = Lc::from_const(F::ONE);
    one_minus_accept.add_term(wire, -F::ONE);
    builder.enforce(&one_minus_accept, &difference, &Lc::zero());
    builder.enforce(&difference, &Lc::from_var(inverse), &Lc::from_var(wire));
    builder.enforce(&one_minus_accept, &Lc::from_var(inverse), &Lc::zero());
    debug_assert_eq!(builder.rows() - source_row_start, 4);
    debug_assert_eq!(builder.cols() - allocated_column_start, 2);
    builder.record_acceptance_chunk_encoding(AcceptanceTraceEntry {
        chunk_bits,
        accept: wire,
        inverse,
        source_rows: source_row_start..builder.rows(),
        allocated_columns: allocated_column_start..builder.cols(),
    });
    AcceptResult { wire, value }
}

fn enforce_mod5(builder: &mut R1csBuilder, chunk_bits: &[Var], chunk: &Lc, chunk_value: u64) -> Mod5Result {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::CHUNK_MOD5);
    let source_row_start = builder.rows();
    let allocated_column_start = builder.cols();
    let value = chunk_value % ALPHABET_SIZE;
    let quotient_value = chunk_value / ALPHABET_SIZE;
    let index = builder.alloc(F::from_u64(value));
    let quotient = builder.alloc(F::from_u64(quotient_value));
    let index_products = enforce_mod5_index(builder, index);

    let mut quotient_bits = Lc::zero();
    let mut power = F::ONE;
    let mut quotient_bit_vars = Vec::with_capacity(QUOTIENT_BITS);
    for offset in 0..QUOTIENT_BITS {
        let bit = builder.alloc(F::from_u64((quotient_value >> offset) & 1));
        enforce_bit(builder, bit);
        quotient_bits.add_term(bit, power);
        quotient_bit_vars.push(bit);
        power += power;
    }
    builder.enforce_eq(&Lc::from_var(quotient), &quotient_bits);

    let mut decomposition = Lc::zero();
    decomposition.add_term(quotient, F::from_u64(ALPHABET_SIZE));
    decomposition.add_term(index, F::ONE);
    builder.enforce_eq(chunk, &decomposition);
    let chunk_bits: [Var; 16] = chunk_bits.try_into().expect("mod-5 chunk width");
    let quotient_bits: [Var; QUOTIENT_BITS] = quotient_bit_vars.try_into().expect("mod-5 quotient width");
    debug_assert_eq!(builder.rows() - source_row_start, 20);
    debug_assert_eq!(builder.cols() - allocated_column_start, 19);
    builder.record_mod5_chunk_encoding(Mod5TraceEntry {
        chunk_bits,
        index,
        quotient,
        index_products,
        quotient_bits,
        source_rows: source_row_start..builder.rows(),
        allocated_columns: allocated_column_start..builder.cols(),
    });
    Mod5Result { index, value }
}

fn enforce_mod5_index(builder: &mut R1csBuilder, index: Var) -> [Var; 3] {
    let mut product = Lc::from_var(index);
    let mut products = Vec::with_capacity(3);
    for value in 1..=4 {
        let mut factor = Lc::from_var(index);
        factor.add_constant(-F::from_u64(value));
        if value == 4 {
            builder.enforce(&product, &factor, &Lc::zero());
        } else {
            let variable = builder.alloc_mul(&product, &factor);
            products.push(variable);
            product = Lc::from_var(variable);
        }
    }
    products.try_into().expect("mod-5 index product count")
}

fn enforce_symbol_and_prefix(
    builder: &mut R1csBuilder,
    residue: Mod5Result,
    cumulative: Var,
    accept: AcceptResult,
) -> ChunkRecord {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::CHUNK_SYMBOL_AND_PREFIX);
    let symbol_value = if residue.value < 2 {
        -F::from_u64(2 - residue.value)
    } else {
        F::from_u64(residue.value - 2)
    };
    let symbol = builder.alloc(symbol_value);
    let mut expected_symbol = Lc::from_var(residue.index);
    expected_symbol.add_constant(-F::from_u64(2));
    builder.enforce_eq(&Lc::from_var(symbol), &expected_symbol);

    let cumulative_value = builder.eval(&Lc::from_var(cumulative)) + accept.value;
    let next_cumulative = builder.alloc(cumulative_value);
    let mut expected_cumulative = Lc::from_var(cumulative);
    expected_cumulative.add_term(accept.wire, F::ONE);
    builder.enforce_eq(&Lc::from_var(next_cumulative), &expected_cumulative);
    ChunkRecord {
        accept: accept.wire,
        symbol,
        cumulative: next_cumulative,
    }
}

#[inline]
fn canonical_u64(value: F) -> u64 {
    use p3_field::PrimeField64;
    value.as_canonical_u64()
}
