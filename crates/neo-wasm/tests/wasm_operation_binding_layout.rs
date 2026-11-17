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
    use neo_wasm::layout::{named_column_family, SELECTOR_COLS};

    let undeclared: Vec<&'static str> = SELECTOR_COLS
        .iter()
        .filter(|&&col| {
            !named_column_family(col)
                .expect("declared selector column")
                .width
                .is_boolean()
        })
        .map(|&col| {
            named_column_family(col)
                .expect("declared selector column")
                .name
                .as_ref()
        })
        .collect();
    assert!(
        undeclared.is_empty(),
        "every SELECTOR_COLS entry must be ColumnWidth::Boolean; missing on: {undeclared:?}"
    );
}
