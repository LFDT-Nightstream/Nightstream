//! SPEC.md section 7.6; Lean `MemoryTrace.Binding` (`Spec/Nebula/EventBinding.lean`).

use std::collections::BTreeSet;

use neo_application::{MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPreload};
use neo_math::F;
use p3_field::PrimeField64;

use super::app::Memory;
use super::expr::Expr;
use super::geometry::Geometry;
use super::records::PortAccess;
use super::rows::{bit_row, chunk, View};
use super::Program;

/// A memory's region in its namespace (Lean `Region`): the first address and
/// the bit width of each address part, least significant part first. A RAM
/// memory is in RAM, a ROM memory in ROM. The base is a multiple of the
/// region size `2^Σ widths`, so each address part is a run of the slot's
/// address bits and needs no other range check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub base: u64,
    pub widths: Vec<usize>,
}

impl Region {
    /// Lean `Region.addr`, `Region.Fits`.
    fn addr(&self, parts: &[u64]) -> Option<u64> {
        if parts.len() != self.widths.len() {
            return None;
        }
        let mut offset = 0;
        let mut address = self.base;
        for (part, width) in parts.iter().zip(&self.widths) {
            if part >> width != 0 {
                return None;
            }
            address += part << offset;
            offset += width;
        }
        Some(address)
    }
}

/// One region for each memory, with the widths that `memories` gives. In each
/// space the regions go from the largest to the smallest, so each base is a
/// multiple of its size (SPEC.md section 7.6, rule 3). Returns the regions
/// and the cells that they take in the ROM space and in the RAM space.
/// A region size or combined space size that does not fit a `u64` returns `Region`.
pub fn pack_regions<Id>(
    memories: impl IntoIterator<Item = (Id, MemoryKind, Vec<usize>)>,
) -> Result<(Vec<(Id, Region)>, [u64; 2]), CatalogBindingError> {
    let mut memories: Vec<_> = memories
        .into_iter()
        .map(|(id, kind, widths)| {
            let width = widths
                .iter()
                .try_fold(0usize, |sum, width| sum.checked_add(*width))
                .filter(|width| *width < u64::BITS as usize)
                .ok_or(CatalogBindingError::Region)?;
            Ok((id, usize::from(kind == MemoryKind::Ram), widths, width))
        })
        .collect::<Result<_, CatalogBindingError>>()?;
    memories.sort_by_key(|(_, space, _, width)| (*space, std::cmp::Reverse(*width)));
    let mut used = [0u64; 2];
    let regions = memories
        .into_iter()
        .map(|(id, space, widths, width)| {
            let base = used[space];
            used[space] = base
                .checked_add(1u64 << width)
                .ok_or(CatalogBindingError::Region)?;
            Ok((id, Region { base, widths }))
        })
        .collect::<Result<_, CatalogBindingError>>()?;
    Ok((regions, used))
}

/// Why a catalog, regions, and a slot plan give no binding, or a preload no
/// program.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CatalogBindingError {
    #[error("the geometry breaks SPEC.md section 7.1")]
    Geometry,
    #[error("a region or a preload cell names a memory that the catalog does not have")]
    UnknownMemory,
    #[error("two regions name one memory")]
    DuplicateRegion,
    #[error("a port has another number of address parts than its region has widths")]
    AddressParts,
    #[error("a region is not aligned to its size or leaves its memory space")]
    Region,
    #[error("two regions of one memory space overlap")]
    Overlap,
    #[error("the slot plan does not have B_ops slots")]
    SlotCount,
    #[error("the slot plan does not name each checked port exactly once")]
    Plan,
    #[error("a preload cell has an address that the region of its memory does not hold")]
    Preload,
}

/// Why a row has no ports. An honest row of a good slot plan has none of
/// these; the binding rows reject each one.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CatalogRowError {
    #[error("an activation column is not a bit")]
    Activation,
    #[error("an address part of an active port does not fit its width")]
    Address,
    #[error("a value of an active port does not fit 32 bits")]
    Value,
    #[error("two active ports share a slot, or two dependent active ports leave catalog order")]
    Plan,
    /// Only a host can give it: a step's row has every catalog column.
    #[error("the row has no word at a column of the catalog")]
    Width,
}

/// A port of a built catalog binding: its rank in the batched catalog, the
/// position of its memory among the checked memories, and its step-witness
/// columns.
#[derive(Clone, Debug)]
pub struct BoundPort {
    /// Lean `eventsForRow`.
    pub rank: usize,
    pub memory: usize,
    pub activation: MemoryPortActivation,
    pub kind: MemoryPortKind,
    pub is_ram: bool,
    pub address: Vec<usize>,
    region: Region,
    pub value: usize,
}

fn is_active(activation: MemoryPortActivation, row: &[F]) -> Result<bool, CatalogRowError> {
    let (column, active_value) = match activation {
        MemoryPortActivation::Always => return Ok(true),
        MemoryPortActivation::When(column) => (column, 1),
        MemoryPortActivation::Unless(column) => (column, 0),
    };
    match column_value(row, column)? {
        bit @ (0 | 1) => Ok(bit == active_value),
        _ => Err(CatalogRowError::Activation),
    }
}

fn column_value(row: &[F], column: usize) -> Result<u64, CatalogRowError> {
    row.get(column)
        .map(|value| value.as_canonical_u64())
        .ok_or(CatalogRowError::Width)
}

fn word(row: &[F], column: usize) -> Result<u64, CatalogRowError> {
    let value = column_value(row, column)?;
    if value >> 32 == 0 {
        Ok(value)
    } else {
        Err(CatalogRowError::Value)
    }
}

impl BoundPort {
    fn columns(&self) -> Vec<usize> {
        let activation = match self.activation {
            MemoryPortActivation::Always => None,
            MemoryPortActivation::When(column) | MemoryPortActivation::Unless(column) => Some(column),
        };
        let prior = match self.kind {
            MemoryPortKind::Write {
                value_before_column: Some(prior),
            } => Some(prior),
            _ => None,
        };
        activation
            .into_iter()
            .chain(self.address.iter().copied())
            .chain([self.value])
            .chain(prior)
            .collect()
    }

    /// Lean `Event.Dependent`.
    fn depends_on(&self, other: &BoundPort) -> bool {
        let writes = |port: &BoundPort| matches!(port.kind, MemoryPortKind::Write { .. });
        self.memory == other.memory && (writes(self) || writes(other))
    }

    fn active(&self, columns: &[Expr]) -> Expr {
        match self.activation {
            MemoryPortActivation::Always => Expr::constant(1),
            MemoryPortActivation::When(column) => columns[column].clone(),
            MemoryPortActivation::Unless(column) => Expr::constant(1) - columns[column].clone(),
        }
    }

    fn gate(&self, columns: &[Expr], e: Expr) -> Expr {
        match self.activation {
            MemoryPortActivation::Always => e,
            _ => self.active(columns) * e,
        }
    }

    fn address(&self, row: &[F]) -> Result<u64, CatalogRowError> {
        let parts = self
            .address
            .iter()
            .map(|&column| column_value(row, column))
            .collect::<Result<Vec<u64>, _>>()?;
        self.region.addr(&parts).ok_or(CatalogRowError::Address)
    }
}

fn sum(terms: impl Iterator<Item = Expr>) -> Expr {
    terms.fold(Expr::constant(0), |sum, term| sum + term)
}

#[derive(Clone, Debug)]
pub struct CatalogBinding<Id> {
    geometry: Geometry,
    slots: Vec<Vec<BoundPort>>,
    checked: Vec<Checked<Id>>,
    host_checked: Vec<Id>,
}

#[derive(Clone, Debug)]
struct Checked<Id> {
    id: Id,
    is_ram: bool,
    region: Region,
}

impl<Id: Clone + Eq> CatalogBinding<Id> {
    pub fn new(
        catalog: &MemoryCatalog<Id>,
        geometry: &Geometry,
        regions: impl IntoIterator<Item = (Id, Region)>,
        slots: &[Vec<usize>],
    ) -> Result<Self, CatalogBindingError> {
        if !geometry.is_valid() {
            return Err(CatalogBindingError::Geometry);
        }
        let memories = catalog.entries();
        let mut bound: Vec<Option<Region>> = vec![None; memories.len()];
        for (id, region) in regions {
            let index = memories
                .iter()
                .position(|memory| memory.id == id)
                .ok_or(CatalogBindingError::UnknownMemory)?;
            if bound[index].replace(region).is_some() {
                return Err(CatalogBindingError::DuplicateRegion);
            }
        }
        let mut spans = Vec::new();
        let mut ports = Vec::new();
        let mut checked = Vec::new();
        let mut host_checked = Vec::new();
        for (memory, region) in memories.iter().zip(bound) {
            let Some(region) = region else {
                host_checked.push(memory.id.clone());
                continue;
            };
            let is_ram = memory.kind == MemoryKind::Ram;
            let space = if is_ram {
                geometry.ram_size()
            } else {
                geometry.rom_size()
            } as u64;
            let width = region
                .widths
                .iter()
                .try_fold(0usize, |sum, width| sum.checked_add(*width))
                .filter(|width| *width < 64)
                .ok_or(CatalogBindingError::Region)?;
            let size = 1u64 << width;
            if region.base % size != 0 || region.base.checked_add(size).is_none_or(|end| end > space) {
                return Err(CatalogBindingError::Region);
            }
            spans.push((is_ram, region.base, region.base + size));
            for port in &memory.ports {
                if port.address_columns.len() != region.widths.len() {
                    return Err(CatalogBindingError::AddressParts);
                }
                ports.push(Some(BoundPort {
                    rank: ports.len(),
                    memory: checked.len(),
                    activation: port.activation,
                    kind: port.kind,
                    is_ram,
                    address: port.address_columns.clone(),
                    region: region.clone(),
                    value: port.value_column,
                }));
            }
            checked.push(Checked {
                id: memory.id.clone(),
                is_ram,
                region,
            });
        }
        for (i, (ram, start, end)) in spans.iter().enumerate() {
            if spans[..i]
                .iter()
                .any(|(other_ram, other_start, other_end)| ram == other_ram && start < other_end && other_start < end)
            {
                return Err(CatalogBindingError::Overlap);
            }
        }
        if slots.len() != geometry.b_ops {
            return Err(CatalogBindingError::SlotCount);
        }
        let slots = slots
            .iter()
            .map(|slot| {
                slot.iter()
                    .map(|rank| {
                        ports
                            .get_mut(*rank)
                            .and_then(Option::take)
                            .ok_or(CatalogBindingError::Plan)
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<BoundPort>>, _>>()?;
        if ports.iter().any(Option::is_some) {
            return Err(CatalogBindingError::Plan);
        }
        Ok(Self {
            geometry: *geometry,
            slots,
            checked,
            host_checked,
        })
    }

    pub fn port_count(&self) -> usize {
        self.slots.len()
    }

    pub fn slots(&self) -> &[Vec<BoundPort>] {
        &self.slots
    }

    pub fn regions(&self) -> impl Iterator<Item = (&Id, &Region)> + '_ {
        self.checked
            .iter()
            .map(|checked| (&checked.id, &checked.region))
    }

    pub fn host_checked(&self) -> &[Id] {
        &self.host_checked
    }

    /// Lean `Binding.Initial`.
    pub fn program(&self, preload: &MemoryPreload<Id>) -> Result<Program, CatalogBindingError>
    where
        Id: Copy + Ord,
    {
        let mut rom = vec![0; self.geometry.rom_size()];
        let mut ram = vec![0; self.geometry.ram_size()];
        for (memory, address, value) in preload.entries() {
            let Some(checked) = self.checked.iter().find(|checked| checked.id == memory) else {
                if self.host_checked.contains(&memory) {
                    continue;
                }
                return Err(CatalogBindingError::UnknownMemory);
            };
            let parts: Vec<u64> = address.into_iter().map(u64::from).collect();
            let addr = checked
                .region
                .addr(&parts)
                .ok_or(CatalogBindingError::Preload)?;
            let image = if checked.is_ram { &mut ram } else { &mut rom };
            image[addr as usize] = value;
        }
        Ok(Program::new(self.geometry, rom, ram).expect("the images have the sizes of a valid geometry"))
    }

    /// SPEC.md section 7.6, backend rules 1 to 4; Lean `Reorders`, `active`.
    pub fn rows(&self, view: &View, columns: &[Expr]) -> Result<Vec<Expr>, crate::Error> {
        if *view.geometry() != self.geometry {
            return Err(crate::Error::Circuit("the catalog binding is for another geometry"));
        }
        let needed = self
            .slots
            .iter()
            .flatten()
            .flat_map(BoundPort::columns)
            .max()
            .map_or(0, |column| column + 1);
        if columns.len() < needed {
            return Err(crate::Error::Circuit(
                "the binding has fewer columns than its catalog reads",
            ));
        }
        let activation_columns: BTreeSet<usize> = self
            .slots
            .iter()
            .flatten()
            .filter_map(|port| match port.activation {
                MemoryPortActivation::Always => None,
                MemoryPortActivation::When(column) | MemoryPortActivation::Unless(column) => Some(column),
            })
            .collect();
        let mut rows: Vec<Expr> = activation_columns
            .into_iter()
            .map(|column| bit_row(columns[column].clone()))
            .collect();
        for (j, slot) in self.slots.iter().enumerate() {
            let active_sum = |keep: fn(&BoundPort) -> bool| {
                sum(slot
                    .iter()
                    .filter(|port| keep(port))
                    .map(|port| port.active(columns)))
            };
            rows.push(Expr::constant(1) - view.pad(j) - active_sum(|_| true));
            rows.push(view.is_write(j) - active_sum(|port| matches!(port.kind, MemoryPortKind::Write { .. })));
            rows.push(view.is_ram(j) - active_sum(|port| port.is_ram));
            let bits = view.addr_bits(j);
            for port in slot {
                let mut offset = 0;
                for (column, width) in port.address.iter().zip(&port.region.widths) {
                    let part = chunk(bits[offset..offset + width].to_vec());
                    rows.push(port.gate(columns, columns[*column].clone() - part));
                    offset += width;
                }
                if offset < bits.len() {
                    let high = chunk(bits[offset..].to_vec()) - Expr::constant(port.region.base >> offset);
                    rows.push(port.gate(columns, high));
                }
                let value = columns[port.value].clone();
                match port.kind {
                    MemoryPortKind::Read => rows.push(port.gate(columns, value - view.vr(j))),
                    MemoryPortKind::Write { value_before_column } => {
                        rows.push(port.gate(columns, value - view.vw(j)));
                        if let Some(prior) = value_before_column {
                            rows.push(port.gate(columns, columns[prior].clone() - view.vr(j)));
                        }
                    }
                }
                let mut later = self.slots[..j]
                    .iter()
                    .flatten()
                    .filter(|earlier| earlier.rank > port.rank && earlier.depends_on(port))
                    .peekable();
                if later.peek().is_some() {
                    rows.push(port.gate(columns, sum(later.map(|earlier| earlier.active(columns)))));
                }
            }
        }
        Ok(rows)
    }

    pub fn ports(&self, row: &[F], memory: &Memory) -> Result<Vec<Option<PortAccess>>, CatalogRowError> {
        let mut ports: Vec<Option<PortAccess>> = Vec::with_capacity(self.slots.len());
        let mut shown_ports: Vec<&BoundPort> = Vec::new();
        for slot in &self.slots {
            let mut shown = None;
            for port in slot {
                if is_active(port.activation, row)? && shown.replace(port).is_some() {
                    return Err(CatalogRowError::Plan);
                }
            }
            let Some(port) = shown else {
                ports.push(None);
                continue;
            };
            if shown_ports
                .iter()
                .any(|earlier| earlier.rank > port.rank && earlier.depends_on(port))
            {
                return Err(CatalogRowError::Plan);
            }
            shown_ports.push(port);
            let (is_ram, addr) = (port.is_ram, port.address(row)?);
            let value = word(row, port.value)?;
            let vr = match port.kind {
                MemoryPortKind::Read => value,
                MemoryPortKind::Write {
                    value_before_column: Some(prior),
                } => word(row, prior)?,
                MemoryPortKind::Write {
                    value_before_column: None,
                } => match ports
                    .iter()
                    .flatten()
                    .rfind(|earlier| earlier.is_ram == is_ram && earlier.addr == addr)
                {
                    Some(earlier) => earlier.vw,
                    None => if is_ram { memory.ram(addr) } else { memory.rom(addr) }.ok_or(CatalogRowError::Address)?,
                },
            };
            ports.push(Some(PortAccess {
                is_write: matches!(port.kind, MemoryPortKind::Write { .. }),
                is_ram,
                addr,
                vr,
                vw: value,
            }));
        }
        Ok(ports)
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/catalog.rs"]
mod tests;
