//! The memory backend of catalog applications (spec §10.1). A
//! `CatalogBinding` maps the events of the checked memories of a
//! `MemoryCatalog` to the operation slots of a geometry, as Lean
//! `MemoryTrace.Binding` (`Spec/Nebula/EventBinding.lean`) needs: it gives the
//! binding rows over the application's columns, the ports of one row, and
//! the program images of a preload. Every rule that soundness needs is a row,
//! so a slot plan changes only which honest rows prove and what the rows
//! cost. It does not own the slot plan, the regions, or the application's
//! other rows.

use std::collections::BTreeSet;

use neo_application::{MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPreload};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

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
    /// Lean `Region.addr` of an address that fits the region (Lean
    /// `Region.Fits`): `base + x₀ + 2^w₀ · (x₁ + …)`. `None` for another
    /// number of parts or a part at or above `2^w`.
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

/// Why a catalog, regions, and a slot plan give no binding, or a preload no
/// program.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CatalogBindingError {
    #[error("the geometry breaks spec §4.2")]
    Geometry,
    #[error("a region names a memory that the catalog does not have")]
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

/// One port of a checked memory as its slot sees it.
#[derive(Clone, Debug)]
struct Port {
    /// The position of the port among the checked ports in catalog order:
    /// the order of its events (Lean `eventsForRow`).
    rank: usize,
    /// The position of the port's memory among the checked memories.
    memory: usize,
    activation: MemoryPortActivation,
    kind: MemoryPortKind,
    is_ram: bool,
    address: Vec<usize>,
    region: Region,
    value: usize,
}

/// `Unless`, `When`, or `Always` as a 0/1 value of `row`.
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

impl Port {
    /// Whether the events of the two ports may not swap (Lean
    /// `Event.Dependent`): one memory, and one of them writes. Two memories
    /// have disjoint cells, and two reads keep the memory, so every other
    /// pair may show in either order.
    fn depends_on(&self, other: &Port) -> bool {
        let writes = |port: &Port| matches!(port.kind, MemoryPortKind::Write { .. });
        self.memory == other.memory && (writes(self) || writes(other))
    }

    /// `1` when the port is active, as a 0/1 expression over the columns.
    fn active(&self, columns: &[Expr]) -> Expr {
        match self.activation {
            MemoryPortActivation::Always => Expr::constant(1),
            MemoryPortActivation::When(column) => columns[column].clone(),
            MemoryPortActivation::Unless(column) => Expr::constant(1) - columns[column].clone(),
        }
    }

    /// `e = 0` when the port is active: one rank-one row.
    fn gate(&self, columns: &[Expr], e: Expr) -> Expr {
        match self.activation {
            MemoryPortActivation::Always => e,
            _ => self.active(columns) * e,
        }
    }

    /// The namespace address of the port's address parts in `row`.
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

/// The binding of a catalog's checked memories to the operation slots of a
/// geometry. A memory with a region is checked; every other memory is
/// host-checked and has no slot and no row.
#[derive(Clone, Debug)]
pub struct CatalogBinding<Id> {
    geometry: Geometry,
    slots: Vec<Vec<Port>>,
    checked: Vec<Checked<Id>>,
    host_checked: Vec<Id>,
}

/// A memory with a region, in its namespace.
#[derive(Clone, Debug)]
struct Checked<Id> {
    id: Id,
    is_ram: bool,
    region: Region,
}

impl<Id: Clone + Eq> CatalogBinding<Id> {
    /// Bind the memories that have a region. `slots` has one entry per
    /// operation slot, in slot order: the checked ports that share the slot,
    /// each named by its position among the checked ports in catalog order
    /// (memories, then ports). It must name each checked port exactly once.
    /// The regions are checked here: disjoint in each memory space, aligned,
    /// and inside their space.
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
                ports.push(Some(Port {
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
            .collect::<Result<Vec<Vec<Port>>, _>>()?;
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

    /// The `b_ops` slots of the plan, for `Application::port_count`.
    pub fn port_count(&self) -> usize {
        self.slots.len()
    }

    /// The memories without a region, in catalog order. The host checks them.
    pub fn host_checked(&self) -> &[Id] {
        &self.host_checked
    }

    /// The program whose images hold the preloaded cells of the checked
    /// memories (Lean `Binding.Initial`): the cell at address `a` of memory
    /// `m` is the image word at `base + packParts(a)` of the region of `m`,
    /// in the RAM image for a RAM memory and in the ROM image otherwise. Every
    /// other word is zero. A memory without a region has no cell in the
    /// images. Packing is injective on the addresses that fit a region, and
    /// the regions of one namespace are disjoint, so no two cells share a word.
    pub fn program(&self, preload: &MemoryPreload<Id>) -> Result<Program, CatalogBindingError>
    where
        Id: Copy + Ord,
    {
        let mut rom = vec![0; self.geometry.rom_size()];
        let mut ram = vec![0; self.geometry.ram_size()];
        for (memory, address, value) in preload.entries() {
            let Some(checked) = self.checked.iter().find(|checked| checked.id == memory) else {
                continue;
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

    /// The binding rows over `view` and the expression of each catalog
    /// column (`columns[c]` for column `c`). Each row is zero on an honest
    /// step and is one rank-one constraint, and the rows add no witness word.
    /// When the step's rows hold (spec §10.1 rules I1 to I4):
    ///
    /// - I1: the slot is active exactly when one of its ports is active, and
    ///   then it shows that port: its namespace, mode, address, and values;
    /// - I2: an active port has no dependent active port of a higher rank in
    ///   an earlier slot, so the active slots list the events in an order that
    ///   keeps each dependent pair (Lean `Reorders`);
    /// - I3: each address part of an active port is the run of the slot's
    ///   address bits at its offset, so it fits its width, and the bits above
    ///   the region hold its base;
    /// - I4: a read port's value is `v_r` (and `v_w`, row O3); a write port's
    ///   value is `v_w`, and its prior column, if any, is `v_r`.
    ///
    /// Each activation column is a bit, so `When` and `Unless` agree with
    /// Lean `active`.
    pub fn rows(&self, view: &View, columns: &[Expr]) -> Vec<Expr> {
        assert_eq!(*view.geometry(), self.geometry, "the binding is for another geometry");
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
            let active_sum = |keep: fn(&Port) -> bool| {
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
        rows
    }

    /// The `B_ops` ports of one row in slot order (`None` for an inactive
    /// slot). `memory` holds the cells before the step. A write without a
    /// prior column reads its cell after the earlier slots of the row, as the
    /// runner applies them.
    pub fn ports(&self, row: &[F], memory: &Memory) -> Result<Vec<Option<PortAccess>>, CatalogRowError> {
        let mut ports: Vec<Option<PortAccess>> = Vec::with_capacity(self.slots.len());
        let mut shown_ports: Vec<&Port> = Vec::new();
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
