use neo_application::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityCatalogError, ContinuityGroup,
    ContinuityLink, MemoryCatalog, MemoryCatalogError, MemoryKind, MemoryPortActivation, MemoryPortKind,
    MemoryPortSpec, MemorySpec,
};

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

#[test]
fn memory_catalog_preserves_memory_and_port_order() {
    let columns = test_columns();
    let catalog = MemoryCatalog::new(
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
                    MemoryPortSpec {
                        address_columns: vec![ADDRESS],
                        value_column: VALUE,
                        kind: MemoryPortKind::Write {
                            value_before_column: Some(VALUE_BEFORE),
                        },
                        activation: MemoryPortActivation::Always,
                    },
                ],
            },
        ],
        &columns,
    )
    .expect("valid logical memories");

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
