//! SPEC.md section 7.6: private range checks across internal continuity links.
use std::num::NonZeroUsize;

use nightstream::nebula::{Geometry, R1csApplication, RowRelation, SharedColumns, Size};
use nightstream::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityGroup, ContinuityLink, MemoryCatalog,
    MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
use nightstream::{Fq as F, PrimeCharacteristicRing};

const WIDTH: usize = 7;
const BEFORE: usize = 1;
const AFTER: usize = 2;
const INPUT_BITS: [usize; 2] = [3, 4];
const OUTPUT_BITS: [usize; 2] = [5, 6];

fn registry() -> ColumnRegistry {
    ColumnRegistry::new(
        [(0, 3, ColumnWidth::Field), (3, 4, ColumnWidth::Boolean)].map(|(start, len, width)| ColumnFamilySpec {
            region: "test",
            start,
            len,
            name: format!("row_{start}").into(),
            role: "test",
            width,
        }),
    )
    .unwrap()
}

fn relation() -> RowRelation {
    let mut rows = Vec::new();
    for (value, bits) in [(BEFORE, INPUT_BITS), (AFTER, OUTPUT_BITS)] {
        for bit in bits {
            rows.push([vec![(bit, F::ONE)], vec![(bit, F::ONE), (0, -F::ONE)], vec![]]);
        }
        rows.push([
            vec![(bits[0], F::ONE), (bits[1], F::TWO)],
            vec![(0, F::ONE)],
            vec![(value, F::ONE)],
        ]);
    }
    RowRelation {
        rows,
        width: WIDTH,
        one: 0,
    }
}

fn application(relation: RowRelation, rows: usize, shared: SharedColumns, extra_link: bool) -> R1csApplication<u8> {
    let registry = registry();
    let mut links = vec![ContinuityLink {
        previous_step_column: AFTER,
        next_step_column: BEFORE,
    }];
    if extra_link {
        links.push(ContinuityLink {
            previous_step_column: OUTPUT_BITS[0],
            next_step_column: INPUT_BITS[0],
        });
    }
    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "state",
            role: "test",
            links,
        }],
        &registry,
    )
    .unwrap();
    let catalog = MemoryCatalog::<u8>::new([], &registry).unwrap();
    let size = Size {
        geometry: Geometry::new(0, 0, 1, 1, 1).unwrap(),
        rows: NonZeroUsize::new(rows).unwrap(),
        regions: vec![],
        slots: vec![vec![]],
    };
    R1csApplication::with_shared(relation, &continuity, &catalog, size, shared).unwrap()
}

fn witness(before: u32, after: u32) -> Vec<F> {
    vec![
        F::ONE,
        F::from_u32(before),
        F::from_u32(after),
        F::from_u32(before & 1),
        F::from_u32((before >> 1) & 1),
        F::from_u32(after & 1),
        F::from_u32((after >> 1) & 1),
    ]
}

fn batch(shared: &SharedColumns, rows: &[Vec<F>]) -> Vec<F> {
    shared
        .step_witness(WIDTH, 0, &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(), |_| 0)
        .unwrap()
}

#[test]
fn private_bits_can_be_restored_without_changing_values() {
    let app = application(relation(), 3, SharedColumns::NONE, false);
    assert_eq!(app.relation_rows(), 18 + 2 - 6);
    let source = application(relation(), 1, SharedColumns::NONE, false);
    for value in 0..4 {
        let mut rows = vec![witness(0, value), witness(value, value), witness(value, 3)];
        for row in &mut rows[..2] {
            row[OUTPUT_BITS[0]] = F::from_u32(9);
            row[OUTPUT_BITS[1]] = F::from_u32(10);
            assert!(source.check(row).is_err());
        }
        let original = rows.clone();
        assert!(app.check(&batch(&SharedColumns::NONE, &rows)).is_ok());
        for i in 0..2 {
            for (output, input) in OUTPUT_BITS.into_iter().zip(INPUT_BITS) {
                rows[i][output] = rows[i + 1][input];
            }
        }
        for (restored, original) in rows.iter().zip(&original) {
            assert!(source.check(restored).is_ok());
            for column in 0..WIDTH {
                if !OUTPUT_BITS.contains(&column) {
                    assert_eq!(restored[column], original[column]);
                }
            }
        }
    }
}

#[test]
fn boundaries_and_internal_values_remain_checked() {
    for count in [1, 2, 3] {
        let app = application(relation(), count, SharedColumns::NONE, false);
        let rows = vec![witness(1, 1); count];
        assert!(app.check(&batch(&SharedColumns::NONE, &rows)).is_ok());
        for row in 0..count {
            for column in [BEFORE, AFTER] {
                let mut forged = rows.clone();
                forged[row][column] = F::from_u32(4);
                if column == AFTER && row + 1 < count {
                    forged[row + 1][BEFORE] = F::from_u32(4);
                }
                assert!(app.check(&batch(&SharedColumns::NONE, &forged)).is_err());
            }
        }
        let mut forged = rows.clone();
        forged[count - 1][OUTPUT_BITS[0]] = F::ZERO;
        assert!(app.check(&batch(&SharedColumns::NONE, &forged)).is_err());
    }
}

#[test]
fn continuity_mismatch_is_rejected_even_when_both_values_are_bounded() {
    let app = application(relation(), 2, SharedColumns::NONE, false);
    assert!(app
        .check(&batch(&SharedColumns::NONE, &[witness(0, 1), witness(2, 3)]))
        .is_err());
}

#[test]
fn extra_bit_uses_and_mismatched_checks_are_retained() {
    let mut extra_use = relation();
    extra_use
        .rows
        .push([vec![(OUTPUT_BITS[0], F::ONE)], vec![(0, F::ONE)], vec![(0, F::ONE)]]);
    let app = application(extra_use, 2, SharedColumns::NONE, false);
    assert_eq!(app.relation_rows(), 15);
    let mut rows = vec![witness(1, 1); 2];
    rows[0][OUTPUT_BITS[1]] = F::TWO;
    assert!(app.check(&batch(&SharedColumns::NONE, &rows)).is_err());

    let mut mismatch = relation();
    mismatch.rows[2][0].pop();
    assert_eq!(application(mismatch, 2, SharedColumns::NONE, false).relation_rows(), 13);

    let mut wrong_weight = relation();
    wrong_weight.rows[5][0][1].1 = F::from_u32(3);
    assert_eq!(
        application(wrong_weight, 2, SharedColumns::NONE, false).relation_rows(),
        13
    );
}

#[test]
fn shared_and_continuity_bits_are_not_private() {
    let shared = SharedColumns {
        columns: vec![OUTPUT_BITS[0]],
        rows: NonZeroUsize::new(2).unwrap(),
    };
    let app = application(relation(), 2, shared.clone(), false);
    // SPEC.md section 7.6: only the shared bit's duplicate Boolean row is removed.
    assert_eq!(app.relation_rows(), 12);
    let mut rows = vec![witness(1, 1); 2];
    rows[0][OUTPUT_BITS[1]] = F::TWO;
    assert!(app.check(&batch(&shared, &rows)).is_err());
    assert_eq!(
        application(relation(), 2, SharedColumns::NONE, true).relation_rows(),
        14
    );
}

#[test]
fn memory_interface_bits_are_not_private() {
    let registry = registry();
    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "state",
            role: "test",
            links: vec![ContinuityLink {
                previous_step_column: AFTER,
                next_step_column: BEFORE,
            }],
        }],
        &registry,
    )
    .unwrap();
    for position in 0..4 {
        let bit = OUTPUT_BITS[0];
        let catalog = MemoryCatalog::new(
            [MemorySpec {
                id: 0u8,
                kind: MemoryKind::Ram,
                ports: vec![MemoryPortSpec {
                    address_columns: vec![if position == 0 { bit } else { BEFORE }],
                    value_column: if position == 1 { bit } else { AFTER },
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(if position == 2 { bit } else { BEFORE }),
                    },
                    activation: if position == 3 {
                        MemoryPortActivation::When(bit)
                    } else {
                        MemoryPortActivation::Always
                    },
                }],
            }],
            &registry,
        )
        .unwrap();
        let (regions, _) = nightstream::nebula::pack_regions([(0u8, MemoryKind::Ram, vec![2])]).unwrap();
        let size = Size {
            geometry: Geometry::new(2, 2, 2, 1, 1).unwrap(),
            rows: NonZeroUsize::new(2).unwrap(),
            regions,
            slots: nightstream::nebula::batched_slots(&catalog, &[vec![0]], NonZeroUsize::new(2).unwrap()).unwrap(),
        };
        let app = R1csApplication::new(relation(), &continuity, &catalog, size).unwrap();
        assert_eq!(app.relation_rows(), 13);
    }
}
