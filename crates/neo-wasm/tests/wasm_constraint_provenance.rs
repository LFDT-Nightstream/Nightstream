use neo_wasm::layout::SELECTOR_COLS;
use neo_wasm::{WasmConstraintScope, WasmOpcode, WasmVmSpec};
use std::collections::BTreeMap;

#[test]
#[ignore]
fn dump_constraint_catalog_by_opcode() {
    let vm = WasmVmSpec::default();
    let catalog = vm.constraint_catalog();

    println!("total rows: {}", catalog.row_tags.len());
    let always_rows = catalog.always_rows();
    println!("always rows: count={}", catalog.count_always_rows());
    print_label_counts(catalog, &always_rows, "  ");
    for row in always_rows {
        let tag = &catalog.row_tags[row];
        println!("  row={row} label={}", tag.label);
    }
    println!();

    let host_event_rows = catalog.host_event_rows();
    println!("host-event rows: count={}", catalog.count_host_event_rows());
    print_label_counts(catalog, &host_event_rows, "  ");
    println!();

    for opcode in WasmOpcode::supported() {
        println!("opcode={}:", opcode.name());
        let owned_rows = catalog.rows_owned_by_opcode(opcode);
        let shared_rows = catalog.shared_rows_for_opcode(opcode);
        println!("  owned ({})", catalog.count_owned_by_opcode(opcode));
        print_label_counts(catalog, &owned_rows, "    ");
        for row in owned_rows {
            let tag = &catalog.row_tags[row];
            println!("        row={row} label={}", tag.label);
        }
        println!("  shared ({})", catalog.count_shared_rows_for_opcode(opcode));
        print_label_counts(catalog, &shared_rows, "    ");
        for row in shared_rows {
            let tag = &catalog.row_tags[row];
            println!("        row={row} label={}", tag.label);
        }
        println!();
    }
}

#[test]
fn host_event_constraints_have_semantic_scope() {
    let vm = WasmVmSpec::default();
    let catalog = vm.constraint_catalog();
    let rows = catalog.host_event_rows();

    assert!(
        !rows.is_empty(),
        "host-event constraints must expose semantic ownership"
    );
    assert_eq!(rows.len(), catalog.count_host_event_rows());
    assert!(rows
        .iter()
        .all(|&row| catalog.row_tags[row].scope == WasmConstraintScope::HostEvent));

    for label in [
        "host event interface",
        "grammar gather binding",
        "host event buffer write",
        "host event perm full round",
        "host event chain update",
    ] {
        assert!(
            rows.iter().any(|&row| catalog.row_tags[row].label == label),
            "missing HostEvent-tagged constraint family `{label}`"
        );
    }
}

#[test]
#[ignore]
fn dump_duplicate_constraint_bodies_by_selector() {
    let vm = WasmVmSpec::default();
    let catalog = vm.constraint_catalog();
    let mut groups = catalog.duplicate_bodies_ignoring_selectors(&SELECTOR_COLS);
    groups.sort_by(|lhs, rhs| {
        rhs.rows
            .len()
            .cmp(&lhs.rows.len())
            .then_with(|| lhs.fingerprint.a_terms.cmp(&rhs.fingerprint.a_terms))
            .then_with(|| lhs.fingerprint.b_terms.cmp(&rhs.fingerprint.b_terms))
            .then_with(|| lhs.fingerprint.c_terms.cmp(&rhs.fingerprint.c_terms))
    });

    println!("total rows: {}", catalog.row_tags.len());
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
            .zip(group.rows.iter().map(|&row| &catalog.row_tags[row]))
        {
            println!(
                "  row={row_idx} label={} scope={:?} selectors_a={selector_a_terms:?} selectors_b={selector_b_terms:?}",
                tag.label, tag.scope
            );
        }
        println!();
    }
}

fn print_label_counts(catalog: &neo_wasm::WasmConstraintCatalog, rows: &[usize], indent: &str) {
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for &row in rows {
        *counts.entry(catalog.row_tags[row].label).or_default() += 1;
    }
    for (label, count) in counts {
        println!("{indent}label={label} count={count}");
    }
}
