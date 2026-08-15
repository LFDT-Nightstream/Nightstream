use std::collections::BTreeSet;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::TERMINAL_R1CS_FAMILY_NAMES;
use nightstream_constraint_exporter::{terminal_verifier_native_guard_names, TerminalColumnLayout};

#[test]
fn terminal_layout_maps_source_columns_into_padded_spartan_order() {
    let layout = TerminalColumnLayout::new(4, 3, 8).expect("valid padded terminal layout");

    assert_eq!(layout.source_public_columns(), 4);
    assert_eq!(layout.source_private_columns(), 3);
    assert_eq!(layout.spartan_private_columns(), 8);
    assert_eq!(layout.source_to_spartan_column(0), Some(8));
    assert_eq!(layout.source_to_spartan_column(1), Some(9));
    assert_eq!(layout.source_to_spartan_column(3), Some(11));
    assert_eq!(layout.source_to_spartan_column(4), Some(0));
    assert_eq!(layout.source_to_spartan_column(6), Some(2));
    assert_eq!(layout.source_to_spartan_column(7), None);
}

#[test]
fn terminal_layout_rejects_missing_public_one_or_private_shrinkage() {
    assert!(TerminalColumnLayout::new(0, 3, 8).is_err());
    assert!(TerminalColumnLayout::new(4, 9, 8).is_err());
}

#[test]
fn terminal_native_guards_are_unique_and_outside_cvc5_families() {
    let guards = terminal_verifier_native_guard_names();
    let unique = guards.iter().copied().collect::<BTreeSet<_>>();

    assert_eq!(guards.len(), 18);
    assert_eq!(unique.len(), guards.len());
    assert!(guards.iter().all(|guard| !guard.is_empty()));
    assert!(TERMINAL_R1CS_FAMILY_NAMES
        .into_iter()
        .all(|family| !unique.contains(family)));
}
