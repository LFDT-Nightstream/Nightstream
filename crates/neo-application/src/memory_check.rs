//! Witness-driven checking of declared logical memory behavior.
//!
//! This is diagnostic replay, not a substitute for relation or backend
//! enforcement of the memory argument and its initial image.

use crate::{ColumnRegistry, MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemorySpec};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::collections::BTreeMap;
use std::fmt::{Debug, Display};

/// Explicit cells present before the first application row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPreload<Id> {
    cells: BTreeMap<Id, BTreeMap<Vec<u32>, u32>>,
}

impl<Id> Default for MemoryPreload<Id> {
    fn default() -> Self {
        Self { cells: BTreeMap::new() }
    }
}

impl<Id: Copy + Ord> MemoryPreload<Id> {
    pub fn insert(&mut self, memory: Id, address: Vec<u32>, value: u32) {
        self.cells.entry(memory).or_default().insert(address, value);
    }

    pub fn remove(&mut self, memory: Id, address: &[u32]) -> Option<u32> {
        self.cells.get_mut(&memory)?.remove(address)
    }

    pub fn entries(&self) -> Vec<(Id, Vec<u32>, u32)> {
        self.cells
            .iter()
            .flat_map(|(&memory, cells)| {
                cells
                    .iter()
                    .map(move |(address, &value)| (memory, address.clone(), value))
            })
            .collect()
    }
}

/// How the diagnostic checker treats a RAM cell absent from the preload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamInitialization {
    /// A cell cannot be read until it is explicitly preloaded or written.
    Explicit,
    /// An absent cell contains the 32-bit value zero.
    Zero,
}

/// Exhaustive initialization policy for the RAMs in one memory catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCheckPolicy<Id> {
    ram_initialization: BTreeMap<Id, RamInitialization>,
}

impl<Id: Copy + Eq + Ord> MemoryCheckPolicy<Id> {
    pub fn new(
        catalog: &MemoryCatalog<Id>,
        ram_initialization: impl IntoIterator<Item = (Id, RamInitialization)>,
    ) -> Result<Self, MemoryCheckPolicyError> {
        let entries: Vec<_> = ram_initialization.into_iter().collect();
        let mut modes = BTreeMap::new();

        for (entry_index, &(id, mode)) in entries.iter().enumerate() {
            if let Some(first_entry) = entries[..entry_index]
                .iter()
                .position(|(previous, _)| previous == &id)
            {
                return Err(MemoryCheckPolicyError::DuplicateMemory {
                    first_entry,
                    second_entry: entry_index,
                });
            }
            let Some(memory_index) = catalog.entries().iter().position(|memory| memory.id == id) else {
                return Err(MemoryCheckPolicyError::UnknownMemory { entry: entry_index });
            };
            if catalog.entries()[memory_index].kind == MemoryKind::Rom {
                return Err(MemoryCheckPolicyError::RomMemory {
                    entry: entry_index,
                    memory: memory_index,
                });
            }
            modes.insert(id, mode);
        }

        for (memory_index, memory) in catalog.entries().iter().enumerate() {
            if memory.kind == MemoryKind::Ram && !modes.contains_key(&memory.id) {
                return Err(MemoryCheckPolicyError::MissingRam { memory: memory_index });
            }
        }

        Ok(Self {
            ram_initialization: modes,
        })
    }

    fn initialization(&self, memory: &Id) -> RamInitialization {
        self.ram_initialization[memory]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MemoryCheckPolicyError {
    #[error("RAM initialization policy repeats a memory at entries {first_entry} and {second_entry}")]
    DuplicateMemory {
        first_entry: usize,
        second_entry: usize,
    },
    #[error("RAM initialization policy entry {entry} names an unknown memory")]
    UnknownMemory { entry: usize },
    #[error("RAM initialization policy entry {entry} names ROM memory {memory}")]
    RomMemory { entry: usize, memory: usize },
    #[error("RAM memory {memory} has no initialization policy")]
    MissingRam { memory: usize },
}

/// Replay the active memory ports in row and declaration order.
///
/// Each row must have exactly the registry's declared width. Every active
/// address component, value, and prior value must fit the memory backend's
/// 32-bit word representation, independently of its declared column width.
pub fn check_memory_rows<Id>(
    catalog: &MemoryCatalog<Id>,
    columns: &ColumnRegistry,
    witness_rows: &[Vec<F>],
    preload: &MemoryPreload<Id>,
    policy: &MemoryCheckPolicy<Id>,
) -> Result<(), MemoryCheckError<Id>>
where
    Id: Copy + Debug + Eq + Ord + Display,
{
    let mut state = preload.cells.clone();
    for (row_index, witness) in witness_rows.iter().enumerate() {
        if witness.len() != columns.column_count() {
            return Err(MemoryCheckError::WitnessWidth {
                expected: columns.column_count(),
                actual: witness.len(),
                row: row_index,
            });
        }
        for memory in catalog.entries() {
            apply_memory_row(memory, columns, witness, row_index, &mut state, policy)?;
        }
    }
    Ok(())
}

fn apply_memory_row<Id>(
    memory: &MemorySpec<Id>,
    columns: &ColumnRegistry,
    witness: &[F],
    row: usize,
    state: &mut BTreeMap<Id, BTreeMap<Vec<u32>, u32>>,
    policy: &MemoryCheckPolicy<Id>,
) -> Result<(), MemoryCheckError<Id>>
where
    Id: Copy + Debug + Eq + Ord + Display,
{
    let cells = state.entry(memory.id).or_default();
    for port in &memory.ports {
        if !activation_active(port.activation, witness, row, memory.id)? {
            continue;
        }

        let address = port
            .address_columns
            .iter()
            .map(|&column| read_u32_column(columns, witness, column, row, "address", memory.id))
            .collect::<Result<Vec<_>, _>>()?;
        let value = read_u32_column(columns, witness, port.value_column, row, "value", memory.id)?;

        if memory.kind == MemoryKind::Rom {
            match cells.get(&address).copied() {
                Some(expected) if expected != value => {
                    return Err(MemoryCheckError::RomMismatch {
                        memory: memory.id,
                        address,
                        row,
                        expected,
                        actual: value,
                    });
                }
                Some(_) => {}
                None => {
                    return Err(MemoryCheckError::RomReadBeforeInitialization {
                        memory: memory.id,
                        address,
                        row,
                    });
                }
            }
            continue;
        }

        match port.kind {
            MemoryPortKind::Read => match cells.get(&address).copied() {
                Some(expected) if expected != value => {
                    return Err(MemoryCheckError::ReadMismatch {
                        memory: memory.id,
                        address,
                        row,
                        expected,
                        actual: value,
                    });
                }
                Some(_) => {}
                None => match policy.initialization(&memory.id) {
                    RamInitialization::Explicit => {
                        return Err(MemoryCheckError::ReadBeforeInitialization {
                            memory: memory.id,
                            address,
                            row,
                        });
                    }
                    RamInitialization::Zero => {
                        if value != 0 {
                            return Err(MemoryCheckError::ZeroReadMismatch {
                                memory: memory.id,
                                address,
                                row,
                                actual: value,
                            });
                        }
                        cells.insert(address, 0);
                    }
                },
            },
            MemoryPortKind::Write { value_before_column } => {
                // A named prior value is part of the row-local relation, so it
                // must agree with the replayed state before the write lands.
                if let Some(column) = value_before_column {
                    let before = read_u32_column(columns, witness, column, row, "value_before", memory.id)?;
                    match cells.get(&address).copied() {
                        Some(expected) if expected != before => {
                            return Err(MemoryCheckError::ReadModifyWriteMismatch {
                                memory: memory.id,
                                address,
                                row,
                                expected,
                                actual: before,
                            });
                        }
                        Some(_) => {}
                        None => match policy.initialization(&memory.id) {
                            RamInitialization::Explicit => {
                                return Err(MemoryCheckError::ReadModifyWriteBeforeInitialization {
                                    memory: memory.id,
                                    address,
                                    row,
                                });
                            }
                            RamInitialization::Zero => {
                                if before != 0 {
                                    return Err(MemoryCheckError::ZeroReadModifyWriteMismatch {
                                        memory: memory.id,
                                        address,
                                        row,
                                        actual: before,
                                    });
                                }
                                cells.insert(address.clone(), 0);
                            }
                        },
                    }
                }
                cells.insert(address, value);
            }
        }
    }
    Ok(())
}

fn activation_active<Id: Copy + Debug + Display>(
    activation: MemoryPortActivation,
    witness: &[F],
    row: usize,
    memory: Id,
) -> Result<bool, MemoryCheckError<Id>> {
    let (column, negate) = match activation {
        MemoryPortActivation::Always => return Ok(true),
        MemoryPortActivation::When(column) => (column, false),
        MemoryPortActivation::Unless(column) => (column, true),
    };
    let value = witness[column];
    let active = if value == F::ZERO {
        false
    } else if value == F::ONE {
        true
    } else {
        return Err(MemoryCheckError::NonBooleanGate { memory, value, row });
    };
    Ok(if negate { !active } else { active })
}

fn read_u32_column<Id: Copy + Debug + Display>(
    columns: &ColumnRegistry,
    witness: &[F],
    column: usize,
    row: usize,
    role: &'static str,
    memory: Id,
) -> Result<u32, MemoryCheckError<Id>> {
    let value = witness[column].as_canonical_u64();
    u32::try_from(value).map_err(|_| MemoryCheckError::ValueNotU32 {
        memory,
        role,
        column,
        family: columns
            .family_for_column(column)
            .map(|family| family.name)
            .unwrap_or("?"),
        value,
        row,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MemoryCheckError<Id>
where
    Id: Debug + Display,
{
    #[error("memory check expected witness width {expected}, got {actual} on row {row}")]
    WitnessWidth {
        expected: usize,
        actual: usize,
        row: usize,
    },
    #[error(
        "memory `{memory}` {role} column `{family}` (col {column}) carried value {value} that does not fit u32 on row {row}"
    )]
    ValueNotU32 {
        memory: Id,
        role: &'static str,
        column: usize,
        family: &'static str,
        value: u64,
        row: usize,
    },
    #[error("memory `{memory}` has non-boolean gate {value} on row {row}")]
    NonBooleanGate { memory: Id, value: F, row: usize },
    #[error("memory `{memory}` ROM mismatch at {address:?} on row {row}: expected {expected}, got {actual}")]
    RomMismatch {
        memory: Id,
        address: Vec<u32>,
        row: usize,
        expected: u32,
        actual: u32,
    },
    #[error("memory `{memory}` ROM read before initialization at {address:?} on row {row}")]
    RomReadBeforeInitialization {
        memory: Id,
        address: Vec<u32>,
        row: usize,
    },
    #[error("memory `{memory}` read mismatch at {address:?} on row {row}: expected {expected}, got {actual}")]
    ReadMismatch {
        memory: Id,
        address: Vec<u32>,
        row: usize,
        expected: u32,
        actual: u32,
    },
    #[error("memory `{memory}` read before initialization at {address:?} on row {row}")]
    ReadBeforeInitialization {
        memory: Id,
        address: Vec<u32>,
        row: usize,
    },
    #[error("memory `{memory}` expected zero-default read at {address:?} on row {row}, got {actual}")]
    ZeroReadMismatch {
        memory: Id,
        address: Vec<u32>,
        row: usize,
        actual: u32,
    },
    #[error(
        "memory `{memory}` RMW read mismatch at {address:?} on row {row}: prior write was {expected}, witness claims {actual}"
    )]
    ReadModifyWriteMismatch {
        memory: Id,
        address: Vec<u32>,
        row: usize,
        expected: u32,
        actual: u32,
    },
    #[error("memory `{memory}` RMW read before initialization at {address:?} on row {row}")]
    ReadModifyWriteBeforeInitialization {
        memory: Id,
        address: Vec<u32>,
        row: usize,
    },
    #[error("memory `{memory}` expected zero-default RMW read at {address:?} on row {row}, got {actual}")]
    ZeroReadModifyWriteMismatch {
        memory: Id,
        address: Vec<u32>,
        row: usize,
        actual: u32,
    },
}
