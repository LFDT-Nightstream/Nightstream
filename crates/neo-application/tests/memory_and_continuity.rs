use std::num::NonZeroUsize;

use neo_application::{
    check_continuity_rows, ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityCatalogError,
    ContinuityCheckError, ContinuityGroup, ContinuityLink, MemoryCatalog, MemoryCatalogError, MemoryKind,
    MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ACTIVE: usize = 0;
const ADDRESS: usize = 1;
const VALUE: usize = 2;
const VALUE_BEFORE: usize = 3;
const STATE_AFTER: usize = 4;
const STATE_BEFORE: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestMemory {
    Program,
    State,
}

fn test_columns() -> ColumnRegistry {
    ColumnRegistry::new([
        ColumnFamilySpec {
            region: "memory_test",
            start: ACTIVE,
            len: 1,
            name: "ACTIVE",
            role: "memory activation",
            width: ColumnWidth::Boolean,
        },
        ColumnFamilySpec {
            region: "memory_test",
            start: ADDRESS,
            len: 1,
            name: "ADDRESS",
            role: "memory address",
            width: ColumnWidth::U32,
        },
        ColumnFamilySpec {
            region: "memory_test",
            start: VALUE,
            len: 1,
            name: "VALUE",
            role: "memory value",
            width: ColumnWidth::U32,
        },
        ColumnFamilySpec {
            region: "memory_test",
            start: VALUE_BEFORE,
            len: 1,
            name: "VALUE_BEFORE",
            role: "memory value before a write",
            width: ColumnWidth::U32,
        },
        ColumnFamilySpec {
            region: "continuity_test",
            start: STATE_AFTER,
            len: 1,
            name: "STATE_AFTER",
            role: "state after this step",
            width: ColumnWidth::Field,
        },
        ColumnFamilySpec {
            region: "continuity_test",
            start: STATE_BEFORE,
            len: 1,
            name: "STATE_BEFORE",
            role: "state before this step",
            width: ColumnWidth::Field,
        },
    ])
    .expect("valid test columns")
}

fn read_port(activation: MemoryPortActivation) -> MemoryPortSpec {
    MemoryPortSpec {
        address_columns: vec![ADDRESS],
        value_column: VALUE,
        kind: MemoryPortKind::Read,
        activation,
    }
}

fn write_port(activation: MemoryPortActivation) -> MemoryPortSpec {
    MemoryPortSpec {
        address_columns: vec![ADDRESS],
        value_column: VALUE,
        kind: MemoryPortKind::Write {
            value_before_column: Some(VALUE_BEFORE),
        },
        activation,
    }
}

/// A ROM with one read, then a RAM with a read and a write.
fn program_and_state() -> MemoryCatalog<TestMemory> {
    MemoryCatalog::new(
        [
            MemorySpec {
                id: TestMemory::Program,
                kind: MemoryKind::Rom,
                ports: vec![read_port(MemoryPortActivation::When(ACTIVE))],
            },
            MemorySpec {
                id: TestMemory::State,
                kind: MemoryKind::Ram,
                ports: vec![
                    read_port(MemoryPortActivation::Unless(ACTIVE)),
                    write_port(MemoryPortActivation::Always),
                ],
            },
        ],
        &test_columns(),
    )
    .expect("valid logical memories")
}

#[test]
fn memory_catalog_preserves_memory_and_port_order() {
    let catalog = program_and_state();
    assert_eq!(catalog.entries()[0].id, TestMemory::Program);
    assert_eq!(catalog.entries()[1].id, TestMemory::State);
    assert_eq!(
        catalog.entries()[1].ports[0].activation,
        MemoryPortActivation::Unless(ACTIVE)
    );
    assert!(matches!(
        catalog.entries()[1].ports[1].kind,
        MemoryPortKind::Write {
            value_before_column: Some(VALUE_BEFORE)
        }
    ));
}

/// Lean `Spec.Nebula.MemoryTrace.batchCatalog`: each memory keeps its id
/// and kind and lists row 0's ports, then row 1's, each with every column
/// moved to its row.
#[test]
fn batched_catalog_lists_memories_then_rows_then_ports() {
    let column = |row: usize, c: usize| 100 + 10 * row + c;
    let batched = program_and_state().batched(NonZeroUsize::new(2).unwrap(), column);
    let read = |row, activation| MemoryPortSpec {
        address_columns: vec![column(row, ADDRESS)],
        value_column: column(row, VALUE),
        kind: MemoryPortKind::Read,
        activation,
    };
    let write = |row| MemoryPortSpec {
        address_columns: vec![column(row, ADDRESS)],
        value_column: column(row, VALUE),
        kind: MemoryPortKind::Write {
            value_before_column: Some(column(row, VALUE_BEFORE)),
        },
        activation: MemoryPortActivation::Always,
    };
    assert_eq!(
        batched.entries(),
        [
            MemorySpec {
                id: TestMemory::Program,
                kind: MemoryKind::Rom,
                ports: vec![
                    read(0, MemoryPortActivation::When(column(0, ACTIVE))),
                    read(1, MemoryPortActivation::When(column(1, ACTIVE))),
                ],
            },
            MemorySpec {
                id: TestMemory::State,
                kind: MemoryKind::Ram,
                ports: vec![
                    read(0, MemoryPortActivation::Unless(column(0, ACTIVE))),
                    write(0),
                    read(1, MemoryPortActivation::Unless(column(1, ACTIVE))),
                    write(1),
                ],
            },
        ]
    );
    assert_eq!(
        program_and_state().batched(NonZeroUsize::MIN, |_, c| c),
        program_and_state(),
        "one row with the identity map is the catalog"
    );
}

#[test]
fn memory_catalog_rejects_invalid_authority_declarations() {
    let columns = test_columns();

    let non_boolean_activation = MemoryCatalog::new(
        [MemorySpec {
            id: TestMemory::State,
            kind: MemoryKind::Ram,
            ports: vec![read_port(MemoryPortActivation::When(ADDRESS))],
        }],
        &columns,
    )
    .expect_err("activation columns must be Boolean");
    assert_eq!(
        non_boolean_activation,
        MemoryCatalogError::ActivationNotBoolean {
            memory: 0,
            port: 0,
            column: ADDRESS,
            family: "ADDRESS",
            width: ColumnWidth::U32,
        }
    );

    let read_only_write = MemoryCatalog::new(
        [MemorySpec {
            id: TestMemory::Program,
            kind: MemoryKind::Rom,
            ports: vec![MemoryPortSpec {
                address_columns: vec![ADDRESS],
                value_column: VALUE,
                kind: MemoryPortKind::Write {
                    value_before_column: None,
                },
                activation: MemoryPortActivation::When(ACTIVE),
            }],
        }],
        &columns,
    )
    .expect_err("read-only memories cannot have write ports");
    assert_eq!(read_only_write, MemoryCatalogError::RomWrite { memory: 0, port: 0 });

    let out_of_range = MemoryCatalog::new(
        [MemorySpec {
            id: TestMemory::State,
            kind: MemoryKind::Ram,
            ports: vec![MemoryPortSpec {
                address_columns: vec![ADDRESS],
                value_column: columns.column_count(),
                kind: MemoryPortKind::Read,
                activation: MemoryPortActivation::When(ACTIVE),
            }],
        }],
        &columns,
    )
    .expect_err("every referenced column must exist");
    assert_eq!(
        out_of_range,
        MemoryCatalogError::ColumnOutOfRange {
            memory: 0,
            port: 0,
            usage: "value",
            column: columns.column_count(),
            column_count: columns.column_count(),
        }
    );

    let duplicate_id = MemoryCatalog::new(
        [
            MemorySpec {
                id: TestMemory::State,
                kind: MemoryKind::Ram,
                ports: vec![read_port(MemoryPortActivation::When(ACTIVE))],
            },
            MemorySpec {
                id: TestMemory::State,
                kind: MemoryKind::Ram,
                ports: vec![read_port(MemoryPortActivation::When(ACTIVE))],
            },
        ],
        &columns,
    )
    .expect_err("logical memory identities must be unique");
    assert_eq!(
        duplicate_id,
        MemoryCatalogError::DuplicateMemoryId {
            first_memory: 0,
            second_memory: 1,
        }
    );
}

#[test]
fn continuity_catalog_preserves_flattened_link_order() {
    let columns = test_columns();
    let catalog = ContinuityCatalog::new(
        [
            ContinuityGroup {
                name: "state",
                role: "carry state across steps",
                links: vec![ContinuityLink {
                    previous_step_column: STATE_AFTER,
                    next_step_column: STATE_BEFORE,
                }],
            },
            ContinuityGroup {
                name: "memory_cursor",
                role: "carry the memory cursor across steps",
                links: vec![
                    ContinuityLink {
                        previous_step_column: VALUE_BEFORE,
                        next_step_column: VALUE,
                    },
                    ContinuityLink {
                        previous_step_column: ADDRESS,
                        next_step_column: ADDRESS,
                    },
                ],
            },
        ],
        &columns,
    )
    .expect("valid continuity declarations");

    assert_eq!(catalog.groups()[0].name, "state");
    assert_eq!(catalog.groups()[1].name, "memory_cursor");
    assert_eq!(catalog.link_count(), 3);
    assert_eq!(
        catalog.links().copied().collect::<Vec<_>>(),
        vec![
            ContinuityLink {
                previous_step_column: STATE_AFTER,
                next_step_column: STATE_BEFORE,
            },
            ContinuityLink {
                previous_step_column: VALUE_BEFORE,
                next_step_column: VALUE,
            },
            ContinuityLink {
                previous_step_column: ADDRESS,
                next_step_column: ADDRESS,
            },
        ]
    );
}

#[test]
fn continuity_checker_replays_adjacent_witness_rows() {
    let columns = test_columns();
    let catalog = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "state",
            role: "carry state across steps",
            links: vec![ContinuityLink {
                previous_step_column: STATE_AFTER,
                next_step_column: STATE_BEFORE,
            }],
        }],
        &columns,
    )
    .expect("valid continuity declarations");

    let mut rows = vec![vec![F::ZERO; columns.column_count()]; 3];
    rows[0][STATE_AFTER] = F::from_u64(7);
    rows[1][STATE_BEFORE] = F::from_u64(7);
    rows[1][STATE_AFTER] = F::from_u64(9);
    rows[2][STATE_BEFORE] = F::from_u64(9);
    check_continuity_rows(&catalog, &rows).expect("adjacent state must be continuous");

    rows[2][STATE_BEFORE] = F::from_u64(10);
    assert_eq!(
        check_continuity_rows(&catalog, &rows).expect_err("mismatched adjacent state must fail"),
        ContinuityCheckError::Mismatch {
            boundary: 1,
            group: 0,
            group_name: "state",
            link: 0,
            previous_step_column: STATE_AFTER,
            next_step_column: STATE_BEFORE,
            previous_value: F::from_u64(9),
            next_value: F::from_u64(10),
        }
    );

    rows[1].truncate(STATE_BEFORE);
    assert_eq!(
        check_continuity_rows(&catalog, &rows).expect_err("missing continuity endpoint must fail cleanly"),
        ContinuityCheckError::MissingColumn {
            row: 1,
            group: 0,
            group_name: "state",
            link: 0,
            column: STATE_BEFORE,
            actual_width: STATE_BEFORE,
        }
    );
}

#[test]
fn continuity_catalog_rejects_reused_and_unknown_columns() {
    let columns = test_columns();
    let link = ContinuityLink {
        previous_step_column: STATE_AFTER,
        next_step_column: STATE_BEFORE,
    };

    let repeated_next = ContinuityCatalog::new(
        [
            ContinuityGroup {
                name: "first",
                role: "first declaration",
                links: vec![link],
            },
            ContinuityGroup {
                name: "second",
                role: "second source for the same destination",
                links: vec![ContinuityLink {
                    previous_step_column: VALUE_BEFORE,
                    next_step_column: STATE_BEFORE,
                }],
            },
        ],
        &columns,
    )
    .expect_err("a next-step column must have exactly one source");
    assert_eq!(
        repeated_next,
        ContinuityCatalogError::RepeatedNextStepColumn {
            column: STATE_BEFORE,
            first_group: 0,
            first_link: 0,
            second_group: 1,
            second_link: 0,
        }
    );

    let repeated_previous = ContinuityCatalog::new(
        [
            ContinuityGroup {
                name: "first",
                role: "first destination",
                links: vec![link],
            },
            ContinuityGroup {
                name: "second",
                role: "second destination for the same source",
                links: vec![ContinuityLink {
                    previous_step_column: STATE_AFTER,
                    next_step_column: VALUE,
                }],
            },
        ],
        &columns,
    )
    .expect_err("a previous-step column must have exactly one destination");
    assert_eq!(
        repeated_previous,
        ContinuityCatalogError::RepeatedPreviousStepColumn {
            column: STATE_AFTER,
            first_group: 0,
            first_link: 0,
            second_group: 1,
            second_link: 0,
        }
    );

    let unknown = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "unknown",
            role: "invalid next-step column",
            links: vec![ContinuityLink {
                previous_step_column: STATE_AFTER,
                next_step_column: columns.column_count(),
            }],
        }],
        &columns,
    )
    .expect_err("every continuity endpoint must exist");
    assert_eq!(
        unknown,
        ContinuityCatalogError::ColumnOutOfRange {
            group: 0,
            link: 0,
            endpoint: "next-step",
            column: columns.column_count(),
            column_count: columns.column_count(),
        }
    );
}
