#[test]
fn column_families_are_dense_and_in_order() {
    use neo_wasm::layout::{
        column_families, HOST_EVENT_COLUMN_COUNT, NAMED_COLUMN_COUNT, NAMED_COLUMN_FAMILY_REGIONS, WASM_COLUMN_COUNT,
    };

    assert_eq!(NAMED_COLUMN_FAMILY_REGIONS.len(), 2);
    assert!(NAMED_COLUMN_FAMILY_REGIONS[0]
        .iter()
        .all(|family| family.region == "wasm_named"));
    assert!(NAMED_COLUMN_FAMILY_REGIONS[1]
        .iter()
        .all(|family| family.region == "host_event_interface"));
    assert_eq!(NAMED_COLUMN_FAMILY_REGIONS[1][0].start, WASM_COLUMN_COUNT);
    assert_eq!(NAMED_COLUMN_COUNT, WASM_COLUMN_COUNT + HOST_EVENT_COLUMN_COUNT);

    let mut next = 0;
    for family in column_families() {
        assert!(matches!(family.region, "wasm_named" | "host_event_interface"));
        assert_eq!(family.start, next, "column families must be dense and ordered");
        next = family.end();
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
    use neo_wasm::layout::{named_column_family, ColumnWidth, SELECTOR_COLS};

    let undeclared: Vec<&'static str> = SELECTOR_COLS
        .iter()
        .filter(|&&col| {
            named_column_family(col)
                .expect("declared selector column")
                .width
                != ColumnWidth::Boolean
        })
        .map(|&col| {
            named_column_family(col)
                .expect("declared selector column")
                .name
        })
        .collect();
    assert!(
        undeclared.is_empty(),
        "every SELECTOR_COLS entry must be ColumnWidth::Boolean; missing on: {undeclared:?}"
    );
}
