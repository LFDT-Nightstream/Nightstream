//! Owns the physical layout of the canonical WASM base witness.
//!
//! Named VM/interface columns form the fixed prefix, followed by host-event advice and
//! reusable range-check bits. Backend-specific advice and batching extend
//! this base layout separately.

use std::sync::OnceLock;

use crate::ccs::host_event_chain::{AUX_COLUMN_FAMILIES, AUX_WIDTH};
use crate::layout::{ColumnFamilySpec, HOST_EVENT_COLUMN_FAMILIES, NAMED_COLUMN_COUNT, WASM_COLUMN_FAMILIES};
use neo_application::{
    decomposition_bit_count, range_checked_variable_widths as shared_variable_widths, ColumnRegistry,
    RangeCheckBitFamily, RangeCheckLayout,
};

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

// Keep the public witness width const without duplicating the allocation rule:
// the shared `decomposition_bit_count` supplies each size, while these asserts
// prove that the input families tile the same dense prefix validated at runtime.
const fn range_bit_count() -> usize {
    let mut count = 0;
    let mut column = 0;
    let mut region_index = 0;
    while region_index < DECLARED_WITNESS_COLUMN_FAMILY_REGIONS.len() {
        let families = DECLARED_WITNESS_COLUMN_FAMILY_REGIONS[region_index];
        let mut family_index = 0;
        while family_index < families.len() {
            let family = &families[family_index];
            assert!(family.start == column);
            count += decomposition_bit_count(family.width) * family.len;
            column += family.len;
            family_index += 1;
        }
        region_index += 1;
    }
    assert!(column == DECLARED_WITNESS_COLUMN_COUNT);
    count
}

const RANGE_BIT_COUNT: usize = range_bit_count();
pub(crate) const RANGE_BITS: WitnessRegion = WitnessRegion::new(HOST_EVENT_AUX.end(), RANGE_BIT_COUNT);
pub const RANGE_BITS_REGION: &str = "range_bits";
const RANGE_BITS_FAMILY: RangeCheckBitFamily = RangeCheckBitFamily {
    region: RANGE_BITS_REGION,
    name: "RANGE_BITS",
    role: "Boolean decomposition bits for declared bounded columns",
};

/// Width of the WASM witness after named columns, host-event advice, and
/// explicit range-check bit columns have been allocated.
pub const RANGE_CHECKED_WITNESS_WIDTH: usize = RANGE_BITS.end();

pub(crate) fn range_check_layout() -> &'static RangeCheckLayout {
    static LAYOUT: OnceLock<RangeCheckLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let layout = RangeCheckLayout::new(declared_witness_column_families().copied(), RANGE_BITS_FAMILY)
            .expect("valid WASM range-check layout");
        assert_eq!(layout.base_column_count(), DECLARED_WITNESS_COLUMN_COUNT);
        assert_eq!(layout.column_count(), RANGE_CHECKED_WITNESS_WIDTH);
        layout
    })
}

pub(crate) fn range_checked_column_registry() -> ColumnRegistry {
    range_check_layout().columns().clone()
}

/// Declared F' widths of the variables in the range-checked WASM witness.
pub(crate) fn range_checked_variable_widths(columns: &ColumnRegistry) -> Vec<usize> {
    let widths = shared_variable_widths(columns);

    assert_eq!(widths.len(), RANGE_CHECKED_WITNESS_WIDTH);
    widths
}
