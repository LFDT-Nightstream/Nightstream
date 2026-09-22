//! First-accepted selection branch for the fixed rejection sampler.
//!
//! Owns: the fixed 11-candidate window and exact first-accepted output binding.
//!
//! Does not own: chunk validity or the low-norm aggregate replacement.
//!
//! Emits constraints: yes; product rows are subsequently replaced only after
//! exact trace validation.
//!
//! Authority boundary: selectors are Boolean and one-hot; candidate values
//! remain the checked chunk wires rather than prover-supplied digests.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `challenge.sampler.selection.one_hot` | `allocate_one_hot` | exactly one of `j..j+10` | 54 per rho | 11 Boolean rows plus one sum row | generic R1CS | `selectionValue_iff_windowed` |
//! | `challenge.sampler.selection.products` | `allocate_products` | three selector products per candidate | `54*11*3` per rho | one product row each | traced product | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind.accept` | `bind_selection` | selected accept aggregate equals one | one per output | one linear binding | traced aggregate | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind.prefix` | `bind_selection` | selected prefix aggregate equals the output index | one per output | one linear binding | traced aggregate | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind.symbol` | `bind_selection` | selected symbol aggregate equals the output | one per output | one linear binding | traced aggregate | `currentSelectionBlock_iff_aggregate` |
//!
//! This branch is isolated because it was the sampler's dominant accidental
//! cost before traced lowering. The compact aggregate encoding preserves these
//! three source bindings after eliminating their product temporaries.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{
    Lc, ProductFactorTrace, ProductSumBatchTrace, ProductSumIdentityTrace, R1csBuilder, Var,
};
use crate::engine::r1cs_circuit::encoding_trace::{FirstAcceptedSelectionProducts, FirstAcceptedSelectionTraceEntry};
use crate::engine::r1cs_circuit::row_formula::{equality_constraint_row, multiplication_constraint_row};

use super::chunk::ChunkRecord;
use super::{pi_rlc_challenge_stage, SELECTION_WINDOW};

struct SelectionProducts {
    symbol: Lc,
    accepted: Lc,
    prefix: Lc,
    trace: Vec<FirstAcceptedSelectionProducts>,
}

struct OneHotWindow {
    start: usize,
    selectors: Vec<Var>,
}

pub(super) fn select_first_n_accepts(builder: &mut R1csBuilder, chunks: &[ChunkRecord], count: usize) -> Vec<Var> {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::SELECTION);
    builder.begin_encoding_stage(pi_rlc_challenge_stage::SELECT_INITIALIZE);
    let zero = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(zero), &Lc::zero());
    let prefixes = std::iter::once(zero)
        .chain(
            chunks[..chunks.len() - 1]
                .iter()
                .map(|chunk| chunk.cumulative),
        )
        .collect::<Vec<_>>();
    debug_assert_eq!(prefixes.len(), chunks.len());

    (0..count)
        .map(|position| {
            let first_one_hot_row = builder.rows();
            let one_hot = allocate_one_hot(builder, chunks, &prefixes, position);
            let first_product_row = builder.rows();
            let first_product_column = builder.cols();
            let products = allocate_products(builder, chunks, &prefixes, &one_hot);
            let first_bind_row = builder.rows();
            let output = bind_selection(builder, &products, position);
            record_selective_product_sum_batch(
                builder,
                chunks,
                &prefixes,
                &one_hot,
                &products,
                position,
                output,
                first_product_row,
                first_product_column,
            );
            builder.record_first_accepted_selection_encoding(FirstAcceptedSelectionTraceEntry {
                one_hot: one_hot.selectors,
                symbols: chunks[one_hot.start..one_hot.start + SELECTION_WINDOW]
                    .iter()
                    .map(|chunk| chunk.symbol)
                    .collect(),
                accepts: chunks[one_hot.start..one_hot.start + SELECTION_WINDOW]
                    .iter()
                    .map(|chunk| chunk.accept)
                    .collect(),
                prefixes: prefixes[one_hot.start..one_hot.start + SELECTION_WINDOW].to_vec(),
                products: products.trace,
                output,
                position,
                one_hot_rows: first_one_hot_row..first_product_row,
                product_rows: first_product_row..first_bind_row,
                bind_rows: first_bind_row..builder.rows(),
            });
            output
        })
        .collect()
}

fn allocate_one_hot(
    builder: &mut R1csBuilder,
    chunks: &[ChunkRecord],
    prefixes: &[Var],
    position: usize,
) -> OneHotWindow {
    let start = position;
    let end = start + SELECTION_WINDOW;
    debug_assert!(end <= chunks.len());
    let target = F::from_u64(position as u64);
    let selected = chunks[start..end]
        .iter()
        .enumerate()
        .find(|(offset, chunk)| {
            builder.witness()[prefixes[start + *offset].col()] == target
                && builder.witness()[chunk.accept.col()] == F::ONE
        })
        .map(|(offset, _)| offset)
        // The acceptance-bound branch already makes a short sample
        // unsatisfiable. Use a deterministic placeholder so synthesis remains
        // total and the relation rejects through constraints instead of panic.
        .unwrap_or(0);

    builder.begin_encoding_stage(pi_rlc_challenge_stage::SELECT_ONE_HOT);
    let selectors = (0..SELECTION_WINDOW)
        .map(|offset| {
            let bit = builder.alloc(if offset == selected { F::ONE } else { F::ZERO });
            enforce_bit(builder, bit);
            bit
        })
        .collect::<Vec<_>>();
    let mut sum = Lc::zero();
    for &bit in &selectors {
        sum.add_term(bit, F::ONE);
    }
    builder.enforce_eq(&sum, &Lc::from_const(F::ONE));
    OneHotWindow { start, selectors }
}

fn allocate_products(
    builder: &mut R1csBuilder,
    chunks: &[ChunkRecord],
    prefixes: &[Var],
    one_hot: &OneHotWindow,
) -> SelectionProducts {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::SELECT_PRODUCTS);
    let mut products = SelectionProducts {
        symbol: Lc::zero(),
        accepted: Lc::zero(),
        prefix: Lc::zero(),
        trace: Vec::with_capacity(chunks.len()),
    };
    for (offset, &selector) in one_hot.selectors.iter().enumerate() {
        let index = one_hot.start + offset;
        let chunk = &chunks[index];
        let symbol = builder.alloc_mul(&Lc::from_var(selector), &Lc::from_var(chunk.symbol));
        let accepted = builder.alloc_mul(&Lc::from_var(selector), &Lc::from_var(chunk.accept));
        let prefix = builder.alloc_mul(&Lc::from_var(selector), &Lc::from_var(prefixes[index]));
        products.symbol.add_term(symbol, F::ONE);
        products.accepted.add_term(accepted, F::ONE);
        products.prefix.add_term(prefix, F::ONE);
        products.trace.push(FirstAcceptedSelectionProducts {
            symbol,
            accepted,
            prefix,
        });
    }
    products
}

fn bind_selection(builder: &mut R1csBuilder, products: &SelectionProducts, position: usize) -> Var {
    builder.begin_nested_encoding_stage(pi_rlc_challenge_stage::SELECT_BIND);
    builder.begin_nested_encoding_stage(pi_rlc_challenge_stage::SELECT_BIND_ACCEPT);
    builder.enforce_eq(&products.accepted, &Lc::from_const(F::ONE));
    builder.begin_nested_encoding_stage(pi_rlc_challenge_stage::SELECT_BIND_PREFIX);
    builder.enforce_eq(&products.prefix, &Lc::from_const(F::from_u64(position as u64)));
    builder.begin_nested_encoding_stage(pi_rlc_challenge_stage::SELECT_BIND_SYMBOL);
    let symbol = builder.alloc(builder.eval(&products.symbol));
    builder.enforce_eq(&Lc::from_var(symbol), &products.symbol);
    symbol
}

#[allow(clippy::too_many_arguments)]
fn record_selective_product_sum_batch(
    builder: &mut R1csBuilder,
    chunks: &[ChunkRecord],
    prefixes: &[Var],
    one_hot: &OneHotWindow,
    products: &SelectionProducts,
    position: usize,
    output: Var,
    row_start: usize,
    column_start: usize,
) {
    let mut source_rows = Vec::with_capacity(SELECTION_WINDOW * 3 + 3);
    for (offset, &selector) in one_hot.selectors.iter().enumerate() {
        let index = one_hot.start + offset;
        source_rows.push(multiplication_constraint_row(
            &Lc::from_var(selector),
            &Lc::from_var(chunks[index].symbol),
            products.trace[offset].symbol,
        ));
        source_rows.push(multiplication_constraint_row(
            &Lc::from_var(selector),
            &Lc::from_var(chunks[index].accept),
            products.trace[offset].accepted,
        ));
        source_rows.push(multiplication_constraint_row(
            &Lc::from_var(selector),
            &Lc::from_var(prefixes[index]),
            products.trace[offset].prefix,
        ));
    }
    source_rows.push(equality_constraint_row(&products.accepted, &Lc::from_const(F::ONE)));
    source_rows.push(equality_constraint_row(
        &products.prefix,
        &Lc::from_const(F::from_u64(position as u64)),
    ));
    source_rows.push(equality_constraint_row(&Lc::from_var(output), &products.symbol));
    builder.assert_recent_rows_equal(row_start, &source_rows);

    let accepted = one_hot
        .selectors
        .iter()
        .enumerate()
        .map(|(offset, &selector)| ProductFactorTrace {
            left: Lc::from_var(selector),
            right: Lc::from_var(chunks[one_hot.start + offset].accept),
            coefficient: F::ONE,
        })
        .collect();
    let prefix = one_hot
        .selectors
        .iter()
        .enumerate()
        .map(|(offset, &selector)| ProductFactorTrace {
            left: Lc::from_var(selector),
            right: Lc::from_var(prefixes[one_hot.start + offset]),
            coefficient: F::ONE,
        })
        .collect();
    let symbol = one_hot
        .selectors
        .iter()
        .enumerate()
        .map(|(offset, &selector)| ProductFactorTrace {
            left: Lc::from_var(selector),
            right: Lc::from_var(chunks[one_hot.start + offset].symbol),
            coefficient: F::ONE,
        })
        .collect();
    builder.record_selective_product_sum_batch(ProductSumBatchTrace {
        row_start,
        row_end: builder.rows(),
        allocated_columns: (column_start..builder.cols()).collect(),
        retained_columns: vec![output.col()],
        identities: vec![
            ProductSumIdentityTrace {
                factors: accepted,
                result: Lc::from_const(F::ONE),
            },
            ProductSumIdentityTrace {
                factors: prefix,
                result: Lc::from_const(F::from_u64(position as u64)),
            },
            ProductSumIdentityTrace {
                factors: symbol,
                result: Lc::from_var(output),
            },
        ],
    });
}
