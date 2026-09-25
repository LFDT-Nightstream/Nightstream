//! Data-driven composition of an application R1CS with Nebula memory ports.
//!
//! The application owns its relation and state-transition plan. This module
//! owns only the verifier-fixed mapping from logical memory ports to physical
//! `S_mem` slots and ROM/RAM ranges.

use std::collections::{BTreeMap, BTreeSet};

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::nebula::circuit::SMemCircuit;
use crate::frontends::nebula::layout::{MemOpRecord, MemSpace, NebulaParams, VAL_BITS};
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::nebula::trace::{Memory, SegmentRun, SegmentTrace, TraceError};
use crate::frontends::r1cs_f_prime::R1csShape;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryRegionKind {
    Rom,
    Ram,
}

impl MemoryRegionKind {
    fn space(self) -> MemSpace {
        match self {
            Self::Rom => MemSpace::Rom,
            Self::Ram => MemSpace::Ram,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRegion {
    name: String,
    kind: MemoryRegionKind,
    base: u64,
    component_bits: Vec<u8>,
}

impl MemoryRegion {
    pub fn new(
        name: impl Into<String>,
        kind: MemoryRegionKind,
        base: u64,
        component_bits: Vec<u8>,
    ) -> Result<Self, ApplicationError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ApplicationError::EmptyRegionName);
        }
        if component_bits.is_empty() || component_bits.iter().any(|&bits| bits == 0 || bits > 32) {
            return Err(ApplicationError::ComponentBits {
                region: name,
                bits: component_bits,
            });
        }
        let total_bits: u32 = component_bits.iter().map(|&bits| u32::from(bits)).sum();
        if total_bits >= 63 {
            return Err(ApplicationError::RegionTooLarge {
                region: name,
                bits: total_bits,
            });
        }
        let cells = 1u64 << total_bits;
        if base.checked_add(cells).is_none() {
            return Err(ApplicationError::RegionAddressOverflow { region: name });
        }
        Ok(Self {
            name,
            kind,
            base,
            component_bits,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> MemoryRegionKind {
        self.kind
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn component_bits(&self) -> &[u8] {
        &self.component_bits
    }

    pub fn cells(&self) -> u64 {
        1u64 << self
            .component_bits
            .iter()
            .map(|&bits| u32::from(bits))
            .sum::<u32>()
    }

    pub fn address(&self, components: &[u64]) -> Result<u64, ApplicationError> {
        self.local_address(components)
    }

    fn local_address(&self, components: &[u64]) -> Result<u64, ApplicationError> {
        if components.len() != self.component_bits.len() {
            return Err(ApplicationError::AddressArity {
                region: self.name.clone(),
                expected: self.component_bits.len(),
                actual: components.len(),
            });
        }
        let mut address = 0u64;
        let mut stride = 1u64;
        for (&component, &bits) in components.iter().zip(&self.component_bits) {
            let bound = 1u64 << bits;
            if component >= bound {
                return Err(ApplicationError::AddressComponentRange {
                    region: self.name.clone(),
                    value: component,
                    bits,
                });
            }
            address += component * stride;
            stride *= bound;
        }
        Ok(self.base + address)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPortActivation {
    Always,
    Column(usize),
    UnlessColumn(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPortKind {
    Read,
    Write { value_before_column: Option<usize> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPort {
    region: usize,
    address_columns: Vec<usize>,
    value_column: usize,
    kind: MemoryPortKind,
    activation: MemoryPortActivation,
}

impl MemoryPort {
    pub fn new(
        region: usize,
        address_columns: Vec<usize>,
        value_column: usize,
        kind: MemoryPortKind,
        activation: MemoryPortActivation,
    ) -> Self {
        Self {
            region,
            address_columns,
            value_column,
            kind,
            activation,
        }
    }

    pub fn region(&self) -> usize {
        self.region
    }

    pub fn address_columns(&self) -> &[usize] {
        &self.address_columns
    }

    pub fn value_column(&self) -> usize {
        self.value_column
    }

    pub fn kind(&self) -> MemoryPortKind {
        self.kind
    }

    pub fn activation(&self) -> MemoryPortActivation {
        self.activation
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryOpSlot {
    candidates: Vec<MemoryPort>,
}

impl MemoryOpSlot {
    pub fn new(candidates: Vec<MemoryPort>) -> Self {
        Self { candidates }
    }

    pub fn candidates(&self) -> &[MemoryPort] {
        &self.candidates
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPortLayout {
    regions: Vec<MemoryRegion>,
    slots: Vec<MemoryOpSlot>,
}

impl MemoryPortLayout {
    pub fn new(regions: Vec<MemoryRegion>, slots: Vec<MemoryOpSlot>) -> Result<Self, ApplicationError> {
        let mut names = BTreeSet::new();
        for region in &regions {
            if !names.insert(region.name.clone()) {
                return Err(ApplicationError::DuplicateRegion(region.name.clone()));
            }
        }
        for (index, left) in regions.iter().enumerate() {
            let left_end = left.base + left.cells();
            for right in &regions[index + 1..] {
                if left.kind != right.kind {
                    continue;
                }
                let right_end = right.base + right.cells();
                if left.base < right_end && right.base < left_end {
                    return Err(ApplicationError::OverlappingRegions {
                        left: left.name.clone(),
                        right: right.name.clone(),
                    });
                }
            }
        }
        for (slot, physical) in slots.iter().enumerate() {
            if physical.candidates.is_empty() {
                return Err(ApplicationError::EmptyMemorySlot { slot });
            }
            for port in &physical.candidates {
                let region = regions
                    .get(port.region)
                    .ok_or(ApplicationError::UnknownRegion {
                        slot,
                        region: port.region,
                    })?;
                if port.address_columns.len() != region.component_bits.len() {
                    return Err(ApplicationError::AddressArity {
                        region: region.name.clone(),
                        expected: region.component_bits.len(),
                        actual: port.address_columns.len(),
                    });
                }
                if region.kind == MemoryRegionKind::Rom && matches!(port.kind, MemoryPortKind::Write { .. }) {
                    return Err(ApplicationError::RomWritePort { slot });
                }
            }
        }
        Ok(Self { regions, slots })
    }

    pub fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    pub fn slots(&self) -> &[MemoryOpSlot] {
        &self.slots
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn logical_port_count(&self) -> usize {
        self.slots.iter().map(|slot| slot.candidates.len()).sum()
    }

    pub fn validate_for(&self, app_columns: usize, params: &NebulaParams) -> Result<(), ApplicationError> {
        if params.num_stacks != 0 {
            return Err(ApplicationError::ApplicationRequiresStacklessNebula);
        }
        if self.slots.len() > params.b_ops {
            return Err(ApplicationError::TooManySlots {
                declared: self.slots.len(),
                available: params.b_ops,
            });
        }
        for region in &self.regions {
            let capacity = match region.kind {
                MemoryRegionKind::Rom => params.rom_cells(),
                MemoryRegionKind::Ram => params.ram_cells(),
            };
            if region.base + region.cells() > capacity {
                return Err(ApplicationError::RegionOutOfBounds {
                    region: region.name.clone(),
                    end: region.base + region.cells(),
                    capacity,
                });
            }
        }
        for (slot, physical) in self.slots.iter().enumerate() {
            for port in &physical.candidates {
                for &column in &port.address_columns {
                    validate_column(slot, column, app_columns)?;
                }
                validate_column(slot, port.value_column, app_columns)?;
                if let MemoryPortKind::Write {
                    value_before_column: Some(column),
                } = port.kind
                {
                    validate_column(slot, column, app_columns)?;
                }
                match port.activation {
                    MemoryPortActivation::Always => {}
                    MemoryPortActivation::Column(column) | MemoryPortActivation::UnlessColumn(column) => {
                        validate_column(slot, column, app_columns)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn execute_assignment(
        &self,
        segment: &mut SegmentRun<'_>,
        assignment: &[F],
    ) -> Result<Vec<Option<MemOpRecord>>, ApplicationError> {
        let mut slots = Vec::with_capacity(self.slots.len());
        for (slot, physical) in self.slots.iter().enumerate() {
            let mut selected = None;
            for (candidate, port) in physical.candidates.iter().enumerate() {
                if !activation_value(assignment, slot, port.activation)? {
                    continue;
                }
                if let Some((first, _)) = selected {
                    return Err(ApplicationError::MemorySlotCollision {
                        slot,
                        first,
                        second: candidate,
                    });
                }
                selected = Some((candidate, port));
            }
            let Some((_, port)) = selected else {
                slots.push(None);
                continue;
            };
            let region = &self.regions[port.region];
            let components = port
                .address_columns
                .iter()
                .map(|&column| assignment_value(assignment, slot, column))
                .collect::<Result<Vec<_>, _>>()?;
            let address = region.local_address(&components)?;
            let value = narrow_u32(assignment_value(assignment, slot, port.value_column)?, slot, "value")?;
            let write = matches!(port.kind, MemoryPortKind::Write { .. }).then_some(value);
            let op = segment.access(region.kind.space(), address, write)?;
            match port.kind {
                MemoryPortKind::Read if op.v_r != value => {
                    return Err(ApplicationError::ReadValueMismatch {
                        slot,
                        region: region.name.clone(),
                        address,
                        expected: op.v_r,
                        actual: value,
                    });
                }
                MemoryPortKind::Write {
                    value_before_column: Some(column),
                } => {
                    let before = narrow_u32(assignment_value(assignment, slot, column)?, slot, "value_before")?;
                    if op.v_r != before {
                        return Err(ApplicationError::WriteBeforeMismatch {
                            slot,
                            region: region.name.clone(),
                            address,
                            expected: op.v_r,
                            actual: before,
                        });
                    }
                }
                _ => {}
            }
            slots.push(Some(op));
        }
        Ok(slots)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaApplication {
    shape: R1csShape,
    recursive_plan: RecursiveStepImagePlan,
    memory: MemoryPortLayout,
}

impl NebulaApplication {
    pub fn new(
        shape: impl Into<R1csShape>,
        recursive_plan: RecursiveStepImagePlan,
        memory: MemoryPortLayout,
    ) -> Result<Self, ApplicationError> {
        let shape = shape.into();
        shape.validate_shape()?;
        crate::frontends::r1cs_f_prime::validate_plan(&recursive_plan, &shape)?;
        Ok(Self {
            shape,
            recursive_plan,
            memory,
        })
    }

    pub fn shape(&self) -> &R1csShape {
        &self.shape
    }

    pub fn recursive_plan(&self) -> &RecursiveStepImagePlan {
        &self.recursive_plan
    }

    pub fn memory(&self) -> &MemoryPortLayout {
        &self.memory
    }

    pub(crate) fn same_relation_profile_as(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.memory == other.memory
            && self
                .recursive_plan
                .same_nebula_relation_profile_as(&other.recursive_plan)
    }

    /// Reuse a prepared application relation while replacing the one
    /// program-specific semantic anchor and its memory-routing instance.
    /// The exact profile check at bind time remains authoritative for every
    /// other field.
    #[doc(hidden)]
    pub fn bind_program_profile(
        &self,
        initial_semantic_state_digest: [u8; 32],
        memory: MemoryPortLayout,
    ) -> Result<Self, ApplicationError> {
        let mut recursive_plan = self.recursive_plan.clone();
        let state = recursive_plan
            .state_x_out
            .as_mut()
            .ok_or(ApplicationError::MissingSemanticStateBinding)?;
        state.initial_semantic_state_digest_anchor = Some(initial_semantic_state_digest);
        Self::new(self.shape.clone(), recursive_plan, memory)
    }

    pub fn validate_for(&self, plan: &NebulaPlan) -> Result<(), ApplicationError> {
        self.memory.validate_for(self.shape.m(), plan.params())
    }

    pub fn trace_segment(
        &self,
        memory: &mut Memory,
        assignments: Vec<Vec<F>>,
    ) -> Result<ApplicationSegmentTrace, ApplicationError> {
        let expected = memory.params().steps_per_segment();
        if assignments.len() != expected {
            return Err(ApplicationError::AssignmentCount {
                actual: assignments.len(),
                expected,
            });
        }
        let mut segment = memory.begin_segment()?;
        let mut slots = Vec::with_capacity(assignments.len());
        for assignment in &assignments {
            self.shape.is_satisfied_by(assignment)?;
            slots.push(self.memory.execute_assignment(&mut segment, assignment)?);
        }
        let memory = segment.finish()?;
        Ok(ApplicationSegmentTrace {
            memory,
            assignments: ApplicationAssignments::from_dense(assignments),
            slots,
        })
    }
}

#[derive(Clone, Debug)]
struct AssignmentDelta {
    columns: Vec<u32>,
    values: Vec<F>,
}

#[derive(Clone, Debug)]
struct ApplicationAssignments {
    initial: Vec<F>,
    deltas: Vec<AssignmentDelta>,
}

impl ApplicationAssignments {
    fn from_dense(assignments: Vec<Vec<F>>) -> Self {
        let mut assignments = assignments.into_iter();
        let initial = assignments.next().expect("application segment is nonempty");
        let mut previous = initial.clone();
        let mut deltas = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            assert_eq!(
                assignment.len(),
                previous.len(),
                "application assignment width changed inside segment"
            );
            let mut columns = Vec::new();
            let mut values = Vec::new();
            for (column, (&before, &after)) in previous.iter().zip(&assignment).enumerate() {
                if before != after {
                    columns.push(u32::try_from(column).expect("application assignment width exceeds u32"));
                    values.push(after);
                }
            }
            deltas.push(AssignmentDelta { columns, values });
            previous = assignment;
        }
        Self { initial, deltas }
    }

    fn cursor(&self) -> ApplicationAssignmentCursor<'_> {
        ApplicationAssignmentCursor {
            assignments: self,
            current: self.initial.clone(),
            next_step: 0,
        }
    }
}

pub(crate) struct ApplicationAssignmentCursor<'a> {
    assignments: &'a ApplicationAssignments,
    current: Vec<F>,
    next_step: usize,
}

impl ApplicationAssignmentCursor<'_> {
    pub(crate) fn next(&mut self) -> Option<&[F]> {
        if self.next_step > self.assignments.deltas.len() {
            return None;
        }
        if self.next_step != 0 {
            let delta = &self.assignments.deltas[self.next_step - 1];
            debug_assert_eq!(delta.columns.len(), delta.values.len());
            for (&column, &value) in delta.columns.iter().zip(&delta.values) {
                self.current[column as usize] = value;
            }
        }
        self.next_step += 1;
        Some(&self.current)
    }
}

#[derive(Clone, Debug)]
pub struct ApplicationSegmentTrace {
    memory: SegmentTrace,
    assignments: ApplicationAssignments,
    slots: Vec<Vec<Option<MemOpRecord>>>,
}

impl ApplicationSegmentTrace {
    pub fn memory(&self) -> &SegmentTrace {
        &self.memory
    }

    pub(crate) fn assignment_cursor(&self) -> ApplicationAssignmentCursor<'_> {
        self.assignments.cursor()
    }

    pub fn slots(&self, step: usize) -> &[Option<MemOpRecord>] {
        &self.slots[step]
    }
}

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error(transparent)]
    App(#[from] crate::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    Plan(#[from] crate::frontends::r1cs_f_prime::Error),
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error("memory region name must not be empty")]
    EmptyRegionName,
    #[error("memory region `{region}` has invalid component widths {bits:?}; each must be in 1..=32")]
    ComponentBits { region: String, bits: Vec<u8> },
    #[error("memory region `{region}` needs {bits} address bits; the maximum is 62")]
    RegionTooLarge { region: String, bits: u32 },
    #[error("memory region `{region}` address range overflows u64")]
    RegionAddressOverflow { region: String },
    #[error("duplicate memory region `{0}`")]
    DuplicateRegion(String),
    #[error("memory regions `{left}` and `{right}` overlap")]
    OverlappingRegions { left: String, right: String },
    #[error("memory slot {slot} references unknown region {region}")]
    UnknownRegion { slot: usize, region: usize },
    #[error("memory region `{region}` expects {expected} address components, got {actual}")]
    AddressArity {
        region: String,
        expected: usize,
        actual: usize,
    },
    #[error("ROM memory slot {slot} contains a write port")]
    RomWritePort { slot: usize },
    #[error("physical memory slot {slot} has no logical port candidates")]
    EmptyMemorySlot { slot: usize },
    #[error("physical memory slot {slot} activated both candidate {first} and candidate {second}")]
    MemorySlotCollision {
        slot: usize,
        first: usize,
        second: usize,
    },
    #[error("application memory requires a stackless Nebula plan")]
    ApplicationRequiresStacklessNebula,
    #[error("prepared application has no semantic-state binding")]
    MissingSemanticStateBinding,
    #[error("application declares {declared} physical memory slots but S_mem has {available}")]
    TooManySlots { declared: usize, available: usize },
    #[error("memory region `{region}` ends at {end}, beyond its namespace capacity {capacity}")]
    RegionOutOfBounds {
        region: String,
        end: u64,
        capacity: u64,
    },
    #[error("memory slot {slot} references application column {column}, but the app has {columns} columns")]
    ColumnOutOfBounds {
        slot: usize,
        column: usize,
        columns: usize,
    },
    #[error("memory slot {slot} activation carried non-Boolean value {value}")]
    NonBooleanActivation { slot: usize, value: u64 },
    #[error("memory slot {slot} reads outside application assignment at column {column}")]
    AssignmentColumn { slot: usize, column: usize },
    #[error("memory slot {slot} {role} value {value} does not fit u32")]
    ValueRange {
        slot: usize,
        role: &'static str,
        value: u64,
    },
    #[error("memory region `{region}` address component {value} does not fit {bits} bits")]
    AddressComponentRange {
        region: String,
        value: u64,
        bits: u8,
    },
    #[error("memory slot {slot} reads `{region}` address {address} as {actual}, but memory contains {expected}")]
    ReadValueMismatch {
        slot: usize,
        region: String,
        address: u64,
        expected: u32,
        actual: u32,
    },
    #[error(
        "memory slot {slot} writes `{region}` address {address} with prior value {actual}, but memory contains {expected}"
    )]
    WriteBeforeMismatch {
        slot: usize,
        region: String,
        address: u64,
        expected: u32,
        actual: u32,
    },
    #[error("memory binding received {actual} S_mem wires, expected {expected}")]
    SMemWireCount { actual: usize, expected: usize },
    #[error("application segment has {actual} assignments, expected exactly {expected}")]
    AssignmentCount { actual: usize, expected: usize },
}

pub(crate) fn enforce_memory_ports(
    builder: &mut R1csBuilder,
    circuit: &SMemCircuit,
    s_mem: &[Var],
    assignment: &[F],
    app_vars: &[Var],
    layout: &MemoryPortLayout,
) -> Result<(), ApplicationError> {
    layout.validate_for(app_vars.len(), circuit.params())?;
    if s_mem.len() != circuit.cols() {
        return Err(ApplicationError::SMemWireCount {
            actual: s_mem.len(),
            expected: circuit.cols(),
        });
    }

    let params = circuit.params();
    let address_offset = 3 + params.num_stacks;
    let value_read_offset = address_offset + params.addr_bits();
    let value_write_offset = value_read_offset + VAL_BITS;
    let mut address_bits = BTreeMap::new();

    let mut logical_port = 0;
    for (slot, physical) in layout.slots.iter().enumerate() {
        let start = circuit.op_slot_column(slot);
        let activations = physical
            .candidates
            .iter()
            .map(|port| activation_lc(builder, assignment, app_vars, slot, port.activation))
            .collect::<Result<Vec<_>, _>>()?;
        let slot_active = activations
            .iter()
            .fold(Lc::zero(), |sum, active| sum.add_scaled(active, F::ONE));
        let slot_is_write = physical
            .candidates
            .iter()
            .zip(&activations)
            .filter(|(port, _)| matches!(port.kind, MemoryPortKind::Write { .. }))
            .fold(Lc::zero(), |sum, (_, active)| sum.add_scaled(active, F::ONE));
        let slot_is_ram = physical
            .candidates
            .iter()
            .zip(&activations)
            .filter(|(port, _)| layout.regions[port.region].kind == MemoryRegionKind::Ram)
            .fold(Lc::zero(), |sum, (_, active)| sum.add_scaled(active, F::ONE));
        let not_active = Lc::from_const(F::ONE).add_scaled(&slot_active, -F::ONE);
        builder.enforce(&slot_active, &not_active, &Lc::zero());
        builder.enforce_eq(&Lc::from_var(s_mem[start]), &not_active);
        builder.enforce_eq(&Lc::from_var(s_mem[start + 1]), &slot_is_write);
        builder.enforce_eq(&Lc::from_var(s_mem[start + 2]), &slot_is_ram);

        for (port, active) in physical.candidates.iter().zip(&activations) {
            let port_start = builder.rows();
            let region = &layout.regions[port.region];
            let slot_address = bits_lc(&s_mem[start + address_offset..start + address_offset + params.addr_bits()]);
            let mut app_address = Lc::from_const(F::from_u64(region.base));
            let mut stride = 1u64;
            for (&column, &bits) in port.address_columns.iter().zip(&region.component_bits) {
                constrain_component(builder, assignment, app_vars, column, bits, active, &mut address_bits);
                app_address.add_term(app_vars[column], F::from_u64(stride));
                stride <<= bits;
            }
            enforce_gated_equality(builder, active, &app_address, &slot_address);

            let slot_read = bits_lc(&s_mem[start + value_read_offset..start + value_read_offset + VAL_BITS]);
            let slot_write = bits_lc(&s_mem[start + value_write_offset..start + value_write_offset + VAL_BITS]);
            let value = Lc::from_var(app_vars[port.value_column]);
            match port.kind {
                MemoryPortKind::Read => {
                    enforce_gated_equality(builder, active, &value, &slot_read);
                    enforce_gated_equality(builder, active, &value, &slot_write);
                }
                MemoryPortKind::Write { value_before_column } => {
                    enforce_gated_equality(builder, active, &value, &slot_write);
                    if let Some(column) = value_before_column {
                        enforce_gated_equality(builder, active, &Lc::from_var(app_vars[column]), &slot_read);
                    }
                }
            }
            builder.record_indexed_row_family("nebula.application.memory_port", logical_port, port_start);
            logical_port += 1;
        }
    }

    for slot in layout.slots.len()..params.b_ops {
        builder.enforce_eq(
            &Lc::from_var(s_mem[circuit.op_slot_column(slot)]),
            &Lc::from_const(F::ONE),
        );
    }
    Ok(())
}

fn enforce_gated_equality(builder: &mut R1csBuilder, active: &Lc, lhs: &Lc, rhs: &Lc) {
    let difference = lhs.clone().add_scaled(rhs, -F::ONE);
    builder.enforce(active, &difference, &Lc::zero());
}

fn activation_value(assignment: &[F], slot: usize, activation: MemoryPortActivation) -> Result<bool, ApplicationError> {
    match activation {
        MemoryPortActivation::Always => Ok(true),
        MemoryPortActivation::Column(column) | MemoryPortActivation::UnlessColumn(column) => {
            let value = assignment_value(assignment, slot, column)?;
            match value {
                0 => Ok(matches!(activation, MemoryPortActivation::UnlessColumn(_))),
                1 => Ok(matches!(activation, MemoryPortActivation::Column(_))),
                value => Err(ApplicationError::NonBooleanActivation { slot, value }),
            }
        }
    }
}

fn activation_lc(
    builder: &mut R1csBuilder,
    assignment: &[F],
    app_vars: &[Var],
    slot: usize,
    activation: MemoryPortActivation,
) -> Result<Lc, ApplicationError> {
    match activation {
        MemoryPortActivation::Always => Ok(Lc::from_const(F::ONE)),
        MemoryPortActivation::Column(column) => {
            let value = assignment_value(assignment, slot, column)?;
            if value > 1 {
                return Err(ApplicationError::NonBooleanActivation { slot, value });
            }
            enforce_bit(builder, app_vars[column]);
            Ok(Lc::from_var(app_vars[column]))
        }
        MemoryPortActivation::UnlessColumn(column) => {
            let value = assignment_value(assignment, slot, column)?;
            if value > 1 {
                return Err(ApplicationError::NonBooleanActivation { slot, value });
            }
            enforce_bit(builder, app_vars[column]);
            let mut active = Lc::from_const(F::ONE);
            active.add_term(app_vars[column], -F::ONE);
            Ok(active)
        }
    }
}

fn bits_lc(bits: &[Var]) -> Lc {
    let mut out = Lc::zero();
    let mut coefficient = F::ONE;
    for &bit in bits {
        out.add_term(bit, coefficient);
        coefficient += coefficient;
    }
    out
}

fn constrain_component(
    builder: &mut R1csBuilder,
    assignment: &[F],
    app_vars: &[Var],
    column: usize,
    bits: u8,
    active: &Lc,
    address_bits: &mut BTreeMap<usize, Vec<Var>>,
) {
    if let std::collections::btree_map::Entry::Vacant(entry) = address_bits.entry(column) {
        let value = assignment[column].as_canonical_u64();
        let wires = (0..32)
            .map(|bit| {
                let wire = builder.alloc(F::from_u64((value >> bit) & 1));
                enforce_bit(builder, wire);
                wire
            })
            .collect::<Vec<_>>();
        builder.enforce_eq(&Lc::from_var(app_vars[column]), &bits_lc(&wires));
        entry.insert(wires);
    }
    for &wire in &address_bits[&column][usize::from(bits)..] {
        builder.enforce(active, &Lc::from_var(wire), &Lc::zero());
    }
}

fn validate_column(slot: usize, column: usize, columns: usize) -> Result<(), ApplicationError> {
    if column >= columns {
        return Err(ApplicationError::ColumnOutOfBounds { slot, column, columns });
    }
    Ok(())
}

fn assignment_value(assignment: &[F], slot: usize, column: usize) -> Result<u64, ApplicationError> {
    assignment
        .get(column)
        .map(|value| value.as_canonical_u64())
        .ok_or(ApplicationError::AssignmentColumn { slot, column })
}

fn narrow_u32(value: u64, slot: usize, role: &'static str) -> Result<u32, ApplicationError> {
    u32::try_from(value).map_err(|_| ApplicationError::ValueRange { slot, role, value })
}

#[cfg(test)]
#[path = "../../../tests/nebula/application_r1cs.rs"]
mod constraint_tests;
