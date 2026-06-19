#[test]
fn column_specs_are_dense_and_in_order() {
    use neo_wasm::layout::{COLUMN_SPECS, WITNESS_WIDTH};

    assert_eq!(
        COLUMN_SPECS.len(),
        WITNESS_WIDTH,
        "macro must emit one spec per witness column"
    );
    for (i, spec) in COLUMN_SPECS.iter().enumerate() {
        assert_eq!(spec.index, i, "COLUMN_SPECS must be index-sequential starting at 0");
    }
}

#[test]
fn every_selector_column_is_declared_boolean() {
    // The opcode-one-hot row in `ccs.rs` only constrains the *sum* of
    // selectors; it does NOT force each selector to be 0 or 1. Per-selector
    // booleanity now comes from `ColumnWidth::Boolean` driving the unified
    // booleanity loop. If a future selector is added without that width
    // annotation, the booleanity row is silently omitted and per-opcode
    // gating becomes unsound (a prover can split a selector's "1" across
    // canceling field values). This test pins that contract.
    use neo_wasm::layout::{ColumnWidth, COLUMN_SPECS, SELECTOR_COLS};

    let undeclared: Vec<&'static str> = SELECTOR_COLS
        .iter()
        .filter(|&&col| COLUMN_SPECS[col].width != ColumnWidth::Boolean)
        .map(|&col| COLUMN_SPECS[col].name)
        .collect();
    assert!(
        undeclared.is_empty(),
        "every SELECTOR_COLS entry must be ColumnWidth::Boolean; missing on: {undeclared:?}"
    );
}
