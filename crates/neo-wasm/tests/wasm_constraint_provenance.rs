use neo_wasm::layout::SELECTOR_COLS;
use neo_wasm::tagged_r1cs_builder::{
    always_rows, count_always_rows, count_host_event_rows, count_owned_by_opcode, count_shared_rows_for_opcode,
    duplicate_bodies_ignoring_selectors, host_event_rows, rows_owned_by_opcode, shared_rows_for_opcode,
};
use neo_wasm::{build_wasm_relation, WasmConstraintScope, WasmOpcode};
use std::collections::BTreeMap;

#[test]
#[ignore]
fn dump_constraint_catalog_by_opcode() {
    let relation = build_wasm_relation().expect("valid WASM relation");
    let catalog = relation.r1cs().catalog();

    println!("total rows: {}", catalog.len());
    let always_rows = always_rows(catalog);
    println!("always rows: count={}", count_always_rows(catalog));
    print_label_counts(catalog, &always_rows, "  ");
    for row in always_rows {
        let tag = catalog.rows()[row].tag();
        println!("  row={row} label={}", tag.label());
    }
    println!();

    let host_event_rows = host_event_rows(catalog);
    println!("host-event rows: count={}", count_host_event_rows(catalog));
    print_label_counts(catalog, &host_event_rows, "  ");
    println!();

    for opcode in WasmOpcode::supported() {
        println!("opcode={}:", opcode.name());
        let owned_rows = rows_owned_by_opcode(catalog, opcode);
        let shared_rows = shared_rows_for_opcode(catalog, opcode);
        println!("  owned ({})", count_owned_by_opcode(catalog, opcode));
        print_label_counts(catalog, &owned_rows, "    ");
        for row in owned_rows {
            let tag = catalog.rows()[row].tag();
            println!("        row={row} label={}", tag.label());
        }
        println!("  shared ({})", count_shared_rows_for_opcode(catalog, opcode));
        print_label_counts(catalog, &shared_rows, "    ");
        for row in shared_rows {
            let tag = catalog.rows()[row].tag();
            println!("        row={row} label={}", tag.label());
        }
        println!();
    }
}

#[test]
fn host_event_constraints_have_semantic_scope() {
    let relation = build_wasm_relation().expect("valid WASM relation");
    let catalog = relation.r1cs().catalog();
    let rows = host_event_rows(catalog);

    assert!(
        !rows.is_empty(),
        "host-event constraints must expose semantic ownership"
    );
    assert_eq!(rows.len(), count_host_event_rows(catalog));
    assert!(rows
        .iter()
        .all(|&row| catalog.rows()[row].tag().owner() == &WasmConstraintScope::HostEvent));

    for label in [
        "host event interface",
        "host-event gather binding",
        "host event buffer write",
        "host event perm full round",
        "host event chain update",
    ] {
        assert!(
            rows.iter()
                .any(|&row| catalog.rows()[row].tag().label() == label),
            "missing HostEvent-tagged constraint family `{label}`"
        );
    }
}

#[test]
#[ignore]
fn dump_duplicate_constraint_bodies_by_selector() {
    let relation = build_wasm_relation().expect("valid WASM relation");
    let catalog = relation.r1cs().catalog();
    let mut groups = duplicate_bodies_ignoring_selectors(catalog, &SELECTOR_COLS);
    groups.sort_by(|lhs, rhs| {
        rhs.rows
            .len()
            .cmp(&lhs.rows.len())
            .then_with(|| lhs.fingerprint.a_terms.cmp(&rhs.fingerprint.a_terms))
            .then_with(|| lhs.fingerprint.b_terms.cmp(&rhs.fingerprint.b_terms))
            .then_with(|| lhs.fingerprint.c_terms.cmp(&rhs.fingerprint.c_terms))
    });

    println!("total rows: {}", catalog.len());
    println!("duplicate bodies with differing selectors: {}", groups.len());
    for (group_idx, group) in groups.iter().enumerate() {
        println!(
            "group {group_idx}: rows={} estimated_savings={}",
            group.rows.len(),
            group.rows.len().saturating_sub(1)
        );
        println!("  A={:?}", group.fingerprint.a_terms.0);
        println!("  B_non_selector={:?}", group.fingerprint.b_terms.0);
        println!("  C={:?}", group.fingerprint.c_terms.0);
        for (((&row_idx, selector_a_terms), selector_b_terms), tag) in group
            .rows
            .iter()
            .zip(group.selector_a_terms_by_row.iter())
            .zip(group.selector_b_terms_by_row.iter())
            .zip(group.rows.iter().map(|&row| catalog.rows()[row].tag()))
        {
            println!(
                "  row={row_idx} label={} scope={:?} selectors_a={selector_a_terms:?} selectors_b={selector_b_terms:?}",
                tag.label(),
                tag.owner()
            );
        }
        println!();
    }
}

fn print_label_counts(catalog: &neo_wasm::WasmConstraintCatalog, rows: &[usize], indent: &str) {
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for &row in rows {
        *counts.entry(catalog.rows()[row].tag().label()).or_default() += 1;
    }
    for (label, count) in counts {
        println!("{indent}label={label} count={count}");
    }
}
