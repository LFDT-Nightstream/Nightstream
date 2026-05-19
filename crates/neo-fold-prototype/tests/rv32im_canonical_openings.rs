//! Focused tests for RV32IM canonical Ajtai opening identities and alias safety.

use neo_fold_prototype::rv32im::kernel::{AjtaiFamilyKind, OpeningAccumulator, SelectedOpeningRef};

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn opening_ref(family: AjtaiFamilyKind, layout_version: u64, logical_index: u64, value_byte: u8) -> SelectedOpeningRef {
    SelectedOpeningRef::from_parts(family, digest(7), layout_version, logical_index, digest(value_byte))
}

#[test]
fn canonical_opening_accumulator_aliases_identical_requests() {
    let reference = opening_ref(AjtaiFamilyKind::Stage2RegisterReads, 1, 3, 11);
    let mut accumulator = OpeningAccumulator::default();
    accumulator.observe(&reference).expect("observe first");
    accumulator
        .observe(&reference)
        .expect("observe identical alias");

    let stats = accumulator.stats();
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.unique_requests, 1);
    assert_eq!(stats.aliased_requests, 1);
}

#[test]
fn canonical_opening_accumulator_rejects_value_mismatch_for_same_id() {
    let reference = opening_ref(AjtaiFamilyKind::Stage2RegisterReads, 1, 3, 11);
    let conflicting = SelectedOpeningRef::new(reference.id.clone(), digest(12));

    let mut accumulator = OpeningAccumulator::default();
    accumulator.observe(&reference).expect("observe first");
    let err = accumulator
        .observe(&conflicting)
        .expect_err("same opening id with different value must fail");
    assert_eq!(err.opening_id_digest, reference.id.digest);
    assert_eq!(err.existing_value_digest, reference.value_digest);
    assert_eq!(err.new_value_digest, conflicting.value_digest);
}

#[test]
fn canonical_opening_accumulator_separates_layout_versions_and_families() {
    let read_v1 = opening_ref(AjtaiFamilyKind::Stage2RegisterReads, 1, 0, 21);
    let read_v2 = opening_ref(AjtaiFamilyKind::Stage2RegisterReads, 2, 0, 21);
    let write_v1 = opening_ref(AjtaiFamilyKind::Stage2RegisterWrites, 1, 0, 21);

    let mut accumulator = OpeningAccumulator::default();
    accumulator.observe(&read_v1).expect("read v1");
    accumulator.observe(&read_v2).expect("read v2");
    accumulator.observe(&write_v1).expect("write v1");

    let stats = accumulator.stats();
    assert_eq!(stats.total_requests, 3);
    assert_eq!(stats.unique_requests, 3);
    assert_eq!(stats.aliased_requests, 0);
}

#[test]
fn canonical_opening_accumulator_is_order_independent() {
    let first = opening_ref(AjtaiFamilyKind::Stage1Rows, 1, 0, 31);
    let second = opening_ref(AjtaiFamilyKind::Stage1Rows, 1, 9, 32);
    let duplicate_first = first.clone();

    let mut left = OpeningAccumulator::default();
    for reference in [&first, &second, &duplicate_first] {
        left.observe(reference).expect("left order");
    }

    let mut right = OpeningAccumulator::default();
    for reference in [&duplicate_first, &second, &first] {
        right.observe(reference).expect("right order");
    }

    assert_eq!(left.stats(), right.stats());
    assert_eq!(left.opening_id_digests(), right.opening_id_digests());
}
