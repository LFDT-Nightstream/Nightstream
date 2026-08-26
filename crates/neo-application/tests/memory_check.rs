use neo_application::{
    check_memory_rows, ColumnFamilySpec, ColumnRegistry, ColumnWidth, MemoryCatalog, MemoryCheckError,
    MemoryCheckPolicy, MemoryCheckPolicyError, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec,
    MemoryPreload, MemorySpec, RamInitialization,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TestMemory {
    Program,
    State,
}

impl Display for TestMemory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Program => formatter.write_str("program"),
            Self::State => formatter.write_str("state"),
        }
    }
}

fn columns() -> ColumnRegistry {
    let declarations = [
        ("gate", ColumnWidth::Boolean),
        ("ram_address", ColumnWidth::Field),
        ("ram_value", ColumnWidth::Field),
        ("ram_before", ColumnWidth::Field),
        ("rom_address", ColumnWidth::Field),
        ("rom_value", ColumnWidth::Field),
    ];
    ColumnRegistry::new(
        declarations
            .into_iter()
            .enumerate()
            .map(|(start, (name, width))| ColumnFamilySpec {
                region: "test",
                start,
                len: 1,
                name,
                role: "memory checker test",
                width,
            }),
    )
    .unwrap()
}

fn catalog(columns: &ColumnRegistry) -> MemoryCatalog<TestMemory> {
    MemoryCatalog::new(
        [
            MemorySpec {
                id: TestMemory::Program,
                kind: MemoryKind::Rom,
                ports: vec![MemoryPortSpec {
                    address_columns: vec![4],
                    value_column: 5,
                    kind: MemoryPortKind::Read,
                    activation: MemoryPortActivation::Always,
                }],
            },
            MemorySpec {
                id: TestMemory::State,
                kind: MemoryKind::Ram,
                ports: vec![
                    MemoryPortSpec {
                        address_columns: vec![1],
                        value_column: 2,
                        kind: MemoryPortKind::Write {
                            value_before_column: Some(3),
                        },
                        activation: MemoryPortActivation::Unless(0),
                    },
                    MemoryPortSpec {
                        address_columns: vec![1],
                        value_column: 2,
                        kind: MemoryPortKind::Read,
                        activation: MemoryPortActivation::Always,
                    },
                    MemoryPortSpec {
                        address_columns: vec![1],
                        value_column: 2,
                        kind: MemoryPortKind::Read,
                        activation: MemoryPortActivation::When(0),
                    },
                ],
            },
        ],
        columns,
    )
    .unwrap()
}

fn preload() -> MemoryPreload<TestMemory> {
    let mut preload = MemoryPreload::default();
    preload.insert(TestMemory::Program, vec![9], 11);
    preload
}

fn rows() -> Vec<Vec<F>> {
    vec![
        [0, 3, 7, 0, 9, 11].map(F::from_u64).to_vec(),
        [1, 3, 7, 0, 9, 11].map(F::from_u64).to_vec(),
    ]
}

#[test]
fn checker_replays_rom_zero_initialized_ram_and_rmw_ports() {
    let columns = columns();
    let catalog = catalog(&columns);
    let policy = MemoryCheckPolicy::new(&catalog, [(TestMemory::State, RamInitialization::Zero)]).unwrap();

    check_memory_rows(&catalog, &columns, &rows(), &preload(), &policy).unwrap();

    let explicit = MemoryCheckPolicy::new(&catalog, [(TestMemory::State, RamInitialization::Explicit)]).unwrap();
    assert!(matches!(
        check_memory_rows(&catalog, &columns, &rows(), &preload(), &explicit),
        Err(MemoryCheckError::ReadModifyWriteBeforeInitialization { .. })
    ));
}

#[test]
fn checker_rejects_memory_and_activation_mismatches() {
    let columns = columns();
    let catalog = catalog(&columns);
    let policy = MemoryCheckPolicy::new(&catalog, [(TestMemory::State, RamInitialization::Zero)]).unwrap();

    let mut bad_rom = rows();
    bad_rom[0][5] = F::from_u64(12);
    assert!(matches!(
        check_memory_rows(&catalog, &columns, &bad_rom, &preload(), &policy),
        Err(MemoryCheckError::RomMismatch { .. })
    ));

    let mut bad_before = rows();
    bad_before[0][3] = F::ONE;
    assert!(matches!(
        check_memory_rows(&catalog, &columns, &bad_before, &preload(), &policy),
        Err(MemoryCheckError::ZeroReadModifyWriteMismatch { .. })
    ));

    let mut bad_read = rows();
    bad_read[1][2] = F::from_u64(8);
    assert!(matches!(
        check_memory_rows(&catalog, &columns, &bad_read, &preload(), &policy),
        Err(MemoryCheckError::ReadMismatch { .. })
    ));

    let mut bad_gate = rows();
    bad_gate[0][0] = F::from_u64(2);
    assert!(matches!(
        check_memory_rows(&catalog, &columns, &bad_gate, &preload(), &policy),
        Err(MemoryCheckError::NonBooleanGate { .. })
    ));
}

#[test]
fn checker_enforces_u32_memory_words_independently_of_column_width() {
    let columns = columns();
    let catalog = catalog(&columns);
    let policy = MemoryCheckPolicy::new(&catalog, [(TestMemory::State, RamInitialization::Zero)]).unwrap();

    for column in 1..=5 {
        let mut rows = rows();
        rows[0][column] = F::from_u64(u64::from(u32::MAX) + 1);
        assert!(matches!(
            check_memory_rows(&catalog, &columns, &rows, &preload(), &policy),
            Err(MemoryCheckError::ValueNotU32 {
                column: rejected,
                ..
            }) if rejected == column
        ));
    }
}

#[test]
fn ram_initialization_policy_is_exhaustive_and_ram_only() {
    let columns = columns();
    let catalog = catalog(&columns);

    assert_eq!(
        MemoryCheckPolicy::new(&catalog, []),
        Err(MemoryCheckPolicyError::MissingRam { memory: 1 })
    );
    assert_eq!(
        MemoryCheckPolicy::new(&catalog, [(TestMemory::Program, RamInitialization::Explicit)]),
        Err(MemoryCheckPolicyError::RomMemory { entry: 0, memory: 0 })
    );
    assert_eq!(
        MemoryCheckPolicy::new(
            &catalog,
            [
                (TestMemory::State, RamInitialization::Explicit),
                (TestMemory::State, RamInitialization::Zero),
            ],
        ),
        Err(MemoryCheckPolicyError::DuplicateMemory {
            first_entry: 0,
            second_entry: 1,
        })
    );
}
