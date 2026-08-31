//! Owns the physical layout of the canonical WASM base witness.
//!
//! Named VM/interface columns form the fixed prefix, followed by host-event advice and
//! reusable range-check bits. Backend-specific advice and batching extend
//! this base layout separately.

use crate::ccs::host_event_chain::{AUX_COLUMN_FAMILIES, AUX_WIDTH};
use crate::layout::{
    ColumnFamilySpec, ColumnWidth, HOST_EVENT_COLUMN_FAMILIES, NAMED_COLUMN_COUNT, WASM_COLUMN_FAMILIES,
};
use neo_application::ColumnRegistry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WitnessRegion {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl WitnessRegion {
    const fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }

    pub(crate) const fn end(self) -> usize {
        self.start + self.len
    }
}

const fn decomposed_bits(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean | ColumnWidth::Field => 0,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
    }
}

pub(crate) const NAMED_COLUMNS: WitnessRegion = WitnessRegion::new(0, NAMED_COLUMN_COUNT);

// Kept standalone so the host-event gadget can use its assigned base without
// depending on the region value that itself contains the gadget's width.
pub(crate) const HOST_EVENT_AUX_START: usize = NAMED_COLUMNS.end();
pub(crate) const HOST_EVENT_AUX: WitnessRegion = WitnessRegion::new(HOST_EVENT_AUX_START, AUX_WIDTH);

const DECLARED_WITNESS_COLUMN_COUNT: usize = HOST_EVENT_AUX.end();
const DECLARED_WITNESS_COLUMN_FAMILY_REGIONS: &[&[ColumnFamilySpec]] =
    &[WASM_COLUMN_FAMILIES, HOST_EVENT_COLUMN_FAMILIES, AUX_COLUMN_FAMILIES];

pub(crate) fn declared_witness_column_families() -> impl Iterator<Item = &'static ColumnFamilySpec> {
    DECLARED_WITNESS_COLUMN_FAMILY_REGIONS
        .iter()
        .flat_map(|families| families.iter())
}

const fn build_range_bit_offsets() -> [usize; DECLARED_WITNESS_COLUMN_COUNT + 1] {
    let mut offsets = [0; DECLARED_WITNESS_COLUMN_COUNT + 1];
    let mut column = 0;
    let mut region_index = 0;
    while region_index < DECLARED_WITNESS_COLUMN_FAMILY_REGIONS.len() {
        let families = DECLARED_WITNESS_COLUMN_FAMILY_REGIONS[region_index];
        let mut family_index = 0;
        while family_index < families.len() {
            let family = &families[family_index];
            assert!(family.start == column);
            let mut member = 0;
            while member < family.len {
                offsets[column + 1] = offsets[column] + decomposed_bits(family.width);
                column += 1;
                member += 1;
            }
            family_index += 1;
        }
        region_index += 1;
    }
    assert!(column == DECLARED_WITNESS_COLUMN_COUNT);
    offsets
}

const RANGE_BIT_OFFSETS: [usize; DECLARED_WITNESS_COLUMN_COUNT + 1] = build_range_bit_offsets();
const RANGE_BIT_COUNT: usize = RANGE_BIT_OFFSETS[DECLARED_WITNESS_COLUMN_COUNT];
pub(crate) const RANGE_BITS: WitnessRegion = WitnessRegion::new(HOST_EVENT_AUX.end(), RANGE_BIT_COUNT);
pub const RANGE_BITS_REGION: &str = "range_bits";
const RANGE_BITS_FAMILY: ColumnFamilySpec = ColumnFamilySpec {
    region: RANGE_BITS_REGION,
    start: RANGE_BITS.start,
    len: RANGE_BIT_COUNT,
    name: "RANGE_BITS",
    role: "Boolean decomposition bits for declared byte and u32 columns",
    width: ColumnWidth::Boolean,
};

/// Width of the WASM witness after named columns, host-event advice, and
/// explicit range-check bit columns have been allocated.
pub const RANGE_CHECKED_WITNESS_WIDTH: usize = RANGE_BITS.end();

pub(crate) fn range_checked_column_registry() -> ColumnRegistry {
    ColumnRegistry::new(
        declared_witness_column_families()
            .copied()
            .chain(core::iter::once(RANGE_BITS_FAMILY)),
    )
    .expect("valid WASM witness column registry")
}

pub(crate) const fn range_bit_region(column: usize) -> Option<WitnessRegion> {
    if column >= DECLARED_WITNESS_COLUMN_COUNT {
        return None;
    }

    let relative_start = RANGE_BIT_OFFSETS[column];
    let relative_end = RANGE_BIT_OFFSETS[column + 1];
    if relative_start == relative_end {
        None
    } else {
        Some(WitnessRegion::new(
            RANGE_BITS.start + relative_start,
            relative_end - relative_start,
        ))
    }
}

/// Declared F' widths of the variables in the range-checked WASM witness.
pub(crate) fn range_checked_variable_widths(columns: &ColumnRegistry) -> Vec<usize> {
    let widths: Vec<_> = columns
        .families()
        .iter()
        .flat_map(|family| core::iter::repeat_n(to_bit_width(family.width), family.len))
        .collect();

    debug_assert_eq!(widths.len(), RANGE_CHECKED_WITNESS_WIDTH);
    widths
}

const fn to_bit_width(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean => 1,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
        // the field doesn't have exactly 64 bits, but we need 64 bits to
        // represent it
        ColumnWidth::Field => 64,
    }
}
