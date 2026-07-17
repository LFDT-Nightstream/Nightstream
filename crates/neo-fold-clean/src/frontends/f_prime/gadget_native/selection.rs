//! Exact low-norm lowering for one first-accepted selection block.
//!
//! Owns: exact validation, reconstruction, and aggregate lowering of one traced
//! first-accepted selection block.
//!
//! Does not own: sampler acceptance arithmetic or selector one-hotness.
//!
//! Emits constraints: yes, three aggregate product-sum equations per output.
//!
//! Authority boundary: Booleanity and `sum(one_hot) = 1` remain authoritative
//! generic source rows; the replacement is valid only after those rows and the
//! traced product/binding rows match exactly.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `challenge.sampler.selection.products` | `validate` | `p = selector * candidate` | three per candidate | none | none | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind` | `emit` | `sum_i selector_i * accept_i = 1` | one per output | 1 | product-sum | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind` | `emit` | `sum_i selector_i * prefix_i = j` | one per output | 1 | product-sum | `currentSelectionBlock_iff_aggregate` |
//! | `challenge.sampler.selection.bind` | `emit` | `sum_i selector_i * symbol_i = output` | one per output | 1 | product-sum | `currentSelectionBlock_iff_aggregate` |
//!
//! The matching Lean proof lives in
//! `PiRlcChallenge/Refinement/SelectionRows.lean`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{FirstAcceptedSelectionTraceEntry, Lc, R1csSnapshot, Var};

use super::{
    claim_gadget_column, one_selector, set_product_definition, source_terms, validate_expected_rows, GadgetNativeError,
    ProductDefinition, TraceGateBuilder,
};

const GADGET: &str = "first-accepted selection";
const BINDING_ROWS: usize = 3;
const PRODUCT_ROWS_PER_CANDIDATE: usize = 3;
const AGGREGATE_ROWS_PER_OUTPUT: usize = 3;

pub(super) fn validate(
    source: &R1csSnapshot,
    event: &FirstAcceptedSelectionTraceEntry,
) -> Result<(), GadgetNativeError> {
    let width = event.one_hot.len();
    if width == 0
        || event.symbols.len() != width
        || event.accepts.len() != width
        || event.prefixes.len() != width
        || event.products.len() != width
        || event.one_hot_rows.len() != width + 1
        || event.product_rows.len() != PRODUCT_ROWS_PER_CANDIDATE * width
        || event.bind_rows.len() != BINDING_ROWS
    {
        return Err(GadgetNativeError::TraceArity { gadget: GADGET });
    }

    let mut one_hot_rows = Vec::with_capacity(event.one_hot_rows.len());
    for &selector in &event.one_hot {
        let selector_lc = Lc::from_var(selector);
        one_hot_rows.push((
            selector_lc.clone(),
            selector_lc.add_scaled(&Lc::from_const(F::ONE), -F::ONE),
            Lc::zero(),
        ));
    }
    let mut selector_sum = Lc::from_const(-F::ONE);
    for &selector in &event.one_hot {
        selector_sum.add_term(selector, F::ONE);
    }
    one_hot_rows.push((selector_sum, Lc::from_var(Var::ONE), Lc::zero()));
    validate_expected_rows(source, GADGET, event.one_hot_rows.start, &one_hot_rows)?;

    let mut product_rows = Vec::with_capacity(event.product_rows.len());
    for index in 0..width {
        product_rows.push((
            Lc::from_var(event.one_hot[index]),
            Lc::from_var(event.symbols[index]),
            Lc::from_var(event.products[index].symbol),
        ));
        product_rows.push((
            Lc::from_var(event.one_hot[index]),
            Lc::from_var(event.accepts[index]),
            Lc::from_var(event.products[index].accepted),
        ));
        product_rows.push((
            Lc::from_var(event.one_hot[index]),
            Lc::from_var(event.prefixes[index]),
            Lc::from_var(event.products[index].prefix),
        ));
    }

    validate_expected_rows(source, GADGET, event.product_rows.start, &product_rows)?;

    let mut accepted_sum = Lc::from_const(-F::ONE);
    let mut prefix_sum = Lc::from_const(-F::from_u64(event.position as u64));
    let mut output_difference = Lc::from_var(event.output);
    for product in &event.products {
        accepted_sum.add_term(product.accepted, F::ONE);
        prefix_sum.add_term(product.prefix, F::ONE);
        output_difference.add_term(product.symbol, -F::ONE);
    }
    let bind_rows = [
        (accepted_sum, Lc::from_var(Var::ONE), Lc::zero()),
        (prefix_sum, Lc::from_var(Var::ONE), Lc::zero()),
        (output_difference, Lc::from_var(Var::ONE), Lc::zero()),
    ];
    validate_expected_rows(source, GADGET, event.bind_rows.start, &bind_rows)
}

pub(super) fn claim_products(
    event: &FirstAcceptedSelectionTraceEntry,
    claimed: &mut [bool],
) -> Result<(), GadgetNativeError> {
    for products in &event.products {
        for variable in [products.symbol, products.accepted, products.prefix] {
            claim_gadget_column(variable.col(), claimed)?;
        }
    }
    Ok(())
}

pub(super) fn define_products(
    definitions: &mut [Option<ProductDefinition>],
    event: &FirstAcceptedSelectionTraceEntry,
) -> Result<(), GadgetNativeError> {
    for index in 0..event.one_hot.len() {
        let selector = Lc::from_var(event.one_hot[index]);
        set_product_definition(
            definitions,
            event.products[index].symbol,
            selector.clone(),
            Lc::from_var(event.symbols[index]),
        )?;
        set_product_definition(
            definitions,
            event.products[index].accepted,
            selector.clone(),
            Lc::from_var(event.accepts[index]),
        )?;
        set_product_definition(
            definitions,
            event.products[index].prefix,
            selector,
            Lc::from_var(event.prefixes[index]),
        )?;
    }
    Ok(())
}

pub(super) fn encoded_rows(event: &FirstAcceptedSelectionTraceEntry) -> usize {
    debug_assert!(!event.one_hot.is_empty());
    AGGREGATE_ROWS_PER_OUTPUT
}

pub(super) fn aggregate_rows_per_family(event: &FirstAcceptedSelectionTraceEntry) -> usize {
    debug_assert!(!event.one_hot.is_empty());
    1
}

pub(super) fn emit(
    event: &FirstAcceptedSelectionTraceEntry,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    let row = event.bind_rows.start;
    let mut accepted_products = Vec::with_capacity(event.one_hot.len());
    let mut prefix_products = Vec::with_capacity(event.one_hot.len());
    let mut symbol_products = Vec::with_capacity(event.one_hot.len());
    for index in 0..event.one_hot.len() {
        let selector = source_terms(event.one_hot[index].col(), decoded, row)?;
        accepted_products.push((
            selector.clone(),
            source_terms(event.accepts[index].col(), decoded, row)?,
        ));
        prefix_products.push((
            selector.clone(),
            source_terms(event.prefixes[index].col(), decoded, row)?,
        ));
        symbol_products.push((selector, source_terms(event.symbols[index].col(), decoded, row)?));
    }

    gates.product_sum(one_selector(), accepted_products, vec![(0, F::ONE)]);
    gates.product_sum(
        one_selector(),
        prefix_products,
        vec![(0, F::from_u64(event.position as u64))],
    );
    gates.product_sum(
        one_selector(),
        symbol_products,
        source_terms(event.output.col(), decoded, row)?,
    );
    Ok(())
}
