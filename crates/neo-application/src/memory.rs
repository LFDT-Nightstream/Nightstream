//! Logical memory declarations over application witness columns.

use crate::{ColumnRegistry, ColumnWidth};

/// Whether a logical memory may be mutated by application rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Rom,
    Ram,
}

/// Declares one logical application memory and its row-local ports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemorySpec<Id> {
    pub id: Id,
    pub kind: MemoryKind,
    pub ports: Vec<MemoryPortSpec>,
}

/// Declares one row-local read or write into a logical memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPortSpec {
    pub address_columns: Vec<usize>,
    pub value_column: usize,
    pub kind: MemoryPortKind,
    pub activation: MemoryPortActivation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPortKind {
    Read,
    Write {
        /// Optional witness column for the value immediately before the write.
        value_before_column: Option<usize>,
    },
}

/// Exact row-local activation rule for a logical memory port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPortActivation {
    /// Active on every application row.
    Always,
    /// Active exactly when the referenced Boolean column is one.
    When(usize),
    /// Active exactly when the referenced Boolean column is zero.
    Unless(usize),
}

/// Validated logical memory declarations in verifier-facing order.
///
/// Memory and port order is preserved because a backend may use it when
/// allocating physical regions and operation slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCatalog<Id> {
    memories: Vec<MemorySpec<Id>>,
}

impl<Id: Eq> MemoryCatalog<Id> {
    pub fn new(
        memories: impl IntoIterator<Item = MemorySpec<Id>>,
        columns: &ColumnRegistry,
    ) -> Result<Self, MemoryCatalogError> {
        let memories: Vec<_> = memories.into_iter().collect();

        for (memory_index, memory) in memories.iter().enumerate() {
            if let Some(first_memory) = memories[..memory_index]
                .iter()
                .position(|previous| previous.id.eq(&memory.id))
            {
                return Err(MemoryCatalogError::DuplicateMemoryId {
                    first_memory,
                    second_memory: memory_index,
                });
            }
            if memory.ports.is_empty() {
                return Err(MemoryCatalogError::EmptyMemory { memory: memory_index });
            }

            for (port_index, port) in memory.ports.iter().enumerate() {
                if port.address_columns.is_empty() {
                    return Err(MemoryCatalogError::EmptyAddress {
                        memory: memory_index,
                        port: port_index,
                    });
                }
                if memory.kind == MemoryKind::Rom && matches!(port.kind, MemoryPortKind::Write { .. }) {
                    return Err(MemoryCatalogError::RomWrite {
                        memory: memory_index,
                        port: port_index,
                    });
                }

                for &column in &port.address_columns {
                    validate_column(columns, memory_index, port_index, "address", column)?;
                }
                validate_column(columns, memory_index, port_index, "value", port.value_column)?;
                if let MemoryPortKind::Write {
                    value_before_column: Some(column),
                } = port.kind
                {
                    validate_column(columns, memory_index, port_index, "value-before", column)?;
                }

                let activation_column = match port.activation {
                    MemoryPortActivation::Always => None,
                    MemoryPortActivation::When(column) | MemoryPortActivation::Unless(column) => Some(column),
                };
                if let Some(column) = activation_column {
                    validate_column(columns, memory_index, port_index, "activation", column)?;
                    let family = columns
                        .family_for_column(column)
                        .expect("the preceding bounds check guarantees a column family");
                    if family.width != ColumnWidth::Boolean {
                        return Err(MemoryCatalogError::ActivationNotBoolean {
                            memory: memory_index,
                            port: port_index,
                            column,
                            family: family.name,
                            width: family.width,
                        });
                    }
                }
            }
        }

        Ok(Self { memories })
    }
}

impl<Id> MemoryCatalog<Id> {
    pub fn memories(&self) -> &[MemorySpec<Id>] {
        &self.memories
    }
}

fn validate_column(
    columns: &ColumnRegistry,
    memory: usize,
    port: usize,
    usage: &'static str,
    column: usize,
) -> Result<(), MemoryCatalogError> {
    if column >= columns.column_count() {
        return Err(MemoryCatalogError::ColumnOutOfRange {
            memory,
            port,
            usage,
            column,
            column_count: columns.column_count(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MemoryCatalogError {
    #[error("logical memory identity is repeated at indices {first_memory} and {second_memory}")]
    DuplicateMemoryId {
        first_memory: usize,
        second_memory: usize,
    },
    #[error("logical memory {memory} has no ports")]
    EmptyMemory { memory: usize },
    #[error("logical memory {memory} port {port} has no address columns")]
    EmptyAddress { memory: usize, port: usize },
    #[error("logical ROM {memory} has a write port at index {port}")]
    RomWrite { memory: usize, port: usize },
    #[error(
        "logical memory {memory} port {port} references {usage} column {column}, but the registry has {column_count} columns"
    )]
    ColumnOutOfRange {
        memory: usize,
        port: usize,
        usage: &'static str,
        column: usize,
        column_count: usize,
    },
    #[error(
        "logical memory {memory} port {port} uses column {column} ({family}) as an activation, but it is declared {width:?} rather than Boolean"
    )]
    ActivationNotBoolean {
        memory: usize,
        port: usize,
        column: usize,
        family: &'static str,
        width: ColumnWidth,
    },
}
