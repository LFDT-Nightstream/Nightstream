use neo_wasm::{build_wasm_lookup_binding_layout, WasmLookupFamilyKind, WasmLookupFamilySpec, WasmMemoryActivation};

fn family_named<'a>(families: &'a [WasmLookupFamilySpec], name: &str) -> &'a WasmLookupFamilySpec {
    families
        .iter()
        .find(|family| family.name == name)
        .expect("lookup family")
}

#[test]
fn layout_describes_lookup_families_and_memory_bindings() {
    let layout = build_wasm_lookup_binding_layout();
    let bounds_family = family_named(&layout.lookup_families, "linear_memory_bounds");
    assert!(matches!(bounds_family.kind, WasmLookupFamilyKind::LinearMemoryBounds));

    let shout_bindings: Vec<_> = layout
        .lookup_bindings
        .iter()
        .filter(|binding| binding.role == "shout row binding")
        .collect();
    assert!(!shout_bindings.is_empty());
    assert!(shout_bindings.iter().all(|binding| binding.gate.is_some()));

    let bounds_binding = layout
        .lookup_bindings
        .iter()
        .find(|binding| binding.family == "linear_memory_bounds")
        .expect("linear-memory bounds binding");
    assert_eq!(bounds_binding.columns.len(), 4);
    assert!(bounds_binding.gate.is_some());

    let globals_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "globals")
        .expect("globals memory family");
    assert_eq!(globals_memory.columns.len(), 2);
    assert!(matches!(
        globals_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));
    assert!(matches!(
        globals_memory.columns[1].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let tables_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "tables")
        .expect("tables memory family");
    assert_eq!(tables_memory.columns.len(), 2);
    assert_eq!(tables_memory.columns[0].address_columns.len(), 2);
    assert!(matches!(
        tables_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));
    assert!(matches!(
        tables_memory.columns[1].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let locals_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "locals")
        .expect("locals memory family");
    assert_eq!(locals_memory.columns.len(), 3);
    assert!(matches!(
        locals_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));
    assert!(matches!(
        locals_memory.columns[1].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));
    assert!(matches!(
        locals_memory.columns[2].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let table_sizes_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "table_sizes")
        .expect("table_sizes memory family");
    assert_eq!(table_sizes_memory.columns.len(), 1);
    assert_eq!(table_sizes_memory.columns[0].address_columns.len(), 1);
    assert!(matches!(
        table_sizes_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let function_types_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "function_types")
        .expect("function_types memory family");
    assert_eq!(function_types_memory.columns.len(), 1);
    assert_eq!(function_types_memory.columns[0].address_columns.len(), 1);
    assert!(function_types_memory.is_rom);
    assert!(matches!(
        function_types_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let function_local_counts_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "function_local_counts")
        .expect("function_local_counts memory family");
    assert_eq!(function_local_counts_memory.columns.len(), 1);
    assert_eq!(
        function_local_counts_memory.columns[0]
            .address_columns
            .len(),
        1
    );
    assert!(function_local_counts_memory.is_rom);
    assert!(matches!(
        function_local_counts_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let pc_function_refs_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "pc_function_refs")
        .expect("pc_function_refs memory family");
    assert_eq!(pc_function_refs_memory.columns.len(), 1);
    assert!(pc_function_refs_memory.is_rom);
    assert!(matches!(
        pc_function_refs_memory.columns[0].activation,
        WasmMemoryActivation::Always
    ));

    let function_guest_flags_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "function_guest_flags")
        .expect("function_guest_flags memory family");
    assert_eq!(function_guest_flags_memory.columns.len(), 2);
    assert!(function_guest_flags_memory.is_rom);
    assert!(function_guest_flags_memory
        .columns
        .iter()
        .all(|column| matches!(column.activation, WasmMemoryActivation::BooleanGate(_))));

    let function_entries_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "function_entries")
        .expect("function_entries memory family");
    assert_eq!(function_entries_memory.columns.len(), 1);
    assert_eq!(function_entries_memory.columns[0].address_columns.len(), 1);
    assert!(function_entries_memory.is_rom);
    assert!(matches!(
        function_entries_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let module_types_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "module_types")
        .expect("module_types memory family");
    assert_eq!(module_types_memory.columns.len(), 1);
    assert_eq!(module_types_memory.columns[0].address_columns.len(), 1);
    assert!(module_types_memory.is_rom);
    assert!(matches!(
        module_types_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let pc_rom_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "pc_rom")
        .expect("pc_rom memory");
    assert_eq!(pc_rom_memory.columns.len(), 2);
    assert_eq!(pc_rom_memory.columns[0].address_columns.len(), 2);
    assert!(matches!(
        pc_rom_memory.columns[0].activation,
        WasmMemoryActivation::BooleanGate(_)
    ));

    let linear_memory = layout
        .memories
        .iter()
        .find(|memory| memory.name == "linear_memory")
        .expect("linear_memory memory family");
    // 3 Read entries (loads, one per lane) + 3 Write+RMW entries (stores).
    // Each pair shares the same `address_columns` and `value_column` but
    // differs in kind, gate (`laneN_load_active` vs `laneN_store_active`),
    // and whether `value_before_column` is set.
    assert_eq!(linear_memory.columns.len(), 6);
    assert!(linear_memory
        .columns
        .iter()
        .all(|column| matches!(column.activation, WasmMemoryActivation::BooleanGate(_))));
    let load_specs: Vec<_> = linear_memory
        .columns
        .iter()
        .filter(|c| matches!(c.kind, neo_wasm::WasmMemoryColumnKind::Read))
        .collect();
    let store_specs: Vec<_> = linear_memory
        .columns
        .iter()
        .filter(|c| matches!(c.kind, neo_wasm::WasmMemoryColumnKind::Write))
        .collect();
    assert_eq!(
        load_specs.len(),
        3,
        "expected 3 lane-Read specs (one per lane) for loads"
    );
    assert_eq!(
        store_specs.len(),
        3,
        "expected 3 lane-Write+RMW specs (one per lane) for stores"
    );
    assert!(
        load_specs.iter().all(|c| c.value_before_column.is_none()),
        "load specs must not name a value_before column (loads emit only Read tuples)",
    );
    assert!(
        store_specs.iter().all(|c| c.value_before_column.is_some()),
        "store specs must name a value_before column (RMW with paired Read + Write)",
    );
}

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
