#[test]
fn column_specs_are_dense_and_in_order() {
    use neo_wasm::layout::{COLUMN_SPECS, NAMED_COLUMN_COUNT};

    let mut next = 0;
    for spec in COLUMN_SPECS {
        assert_eq!(spec.region, "wasm_named");
        assert_eq!(spec.start, next, "COLUMN_SPECS must be dense and ordered");
        next = spec.end();
    }
    assert_eq!(next, NAMED_COLUMN_COUNT);
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
    use neo_wasm::layout::{column_spec, ColumnWidth, SELECTOR_COLS};

    let undeclared: Vec<&'static str> = SELECTOR_COLS
        .iter()
        .filter(|&&col| column_spec(col).expect("declared selector column").width != ColumnWidth::Boolean)
        .map(|&col| column_spec(col).expect("declared selector column").name)
        .collect();
    assert!(
        undeclared.is_empty(),
        "every SELECTOR_COLS entry must be ColumnWidth::Boolean; missing on: {undeclared:?}"
    );
}
