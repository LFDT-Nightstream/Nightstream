//! Owns the physical layout of the canonical WASM base witness.
//!
//! Named VM/interface columns form the fixed prefix, followed by host-event advice and
//! reusable range-check bits. Backend-specific advice and batching extend
//! this base layout separately.

use crate::ccs::host_event_chain::{AUX_COLUMN_SPECS, AUX_WIDTH};
use crate::column_registry::expanded_f_prime_widths;
use crate::layout::{ColumnWidth, COLUMN_SPEC_REGIONS, NAMED_COLUMN_COUNT};

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

const fn build_range_bit_offsets() -> [usize; NAMED_COLUMN_COUNT + 1] {
    let mut offsets = [0; NAMED_COLUMN_COUNT + 1];
    let mut column = 0;
    let mut region_index = 0;
    while region_index < COLUMN_SPEC_REGIONS.len() {
        let specs = COLUMN_SPEC_REGIONS[region_index];
        let mut spec_index = 0;
        while spec_index < specs.len() {
            let spec = &specs[spec_index];
            assert!(spec.start == column);
            let mut member = 0;
            while member < spec.len {
                offsets[column + 1] = offsets[column] + decomposed_bits(spec.width);
                column += 1;
                member += 1;
            }
            spec_index += 1;
        }
        region_index += 1;
    }
    assert!(column == NAMED_COLUMN_COUNT);
    offsets
}

const RANGE_BIT_OFFSETS: [usize; NAMED_COLUMN_COUNT + 1] = build_range_bit_offsets();
const RANGE_BIT_COUNT: usize = RANGE_BIT_OFFSETS[NAMED_COLUMN_COUNT];

pub(crate) const NAMED_COLUMNS: WitnessRegion = WitnessRegion::new(0, NAMED_COLUMN_COUNT);

// Kept standalone so the host-event gadget can use its assigned base without
// depending on the region value that itself contains the gadget's width.
pub(crate) const HOST_EVENT_AUX_START: usize = NAMED_COLUMNS.end();
pub(crate) const HOST_EVENT_AUX: WitnessRegion = WitnessRegion::new(HOST_EVENT_AUX_START, AUX_WIDTH);
pub(crate) const RANGE_BITS: WitnessRegion = WitnessRegion::new(HOST_EVENT_AUX.end(), RANGE_BIT_COUNT);

/// Width of the WASM witness after named columns, host-event advice, and
/// explicit range-check bit columns have been allocated.
pub const RANGE_CHECKED_WITNESS_WIDTH: usize = RANGE_BITS.end();

pub(crate) const fn range_bit_region(column: usize) -> Option<WitnessRegion> {
    if column >= NAMED_COLUMN_COUNT {
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
pub(crate) fn range_checked_variable_widths() -> Vec<usize> {
    let mut widths: Vec<usize> = COLUMN_SPEC_REGIONS
        .iter()
        .flat_map(|specs| expanded_f_prime_widths(specs))
        .collect();

    debug_assert_eq!(widths.len(), NAMED_COLUMNS.end());
    widths.extend(expanded_f_prime_widths(AUX_COLUMN_SPECS));
    debug_assert_eq!(widths.len(), HOST_EVENT_AUX.end());
    widths.resize(RANGE_BITS.end(), 1);
    debug_assert_eq!(widths.len(), RANGE_CHECKED_WITNESS_WIDTH);
    widths
}
