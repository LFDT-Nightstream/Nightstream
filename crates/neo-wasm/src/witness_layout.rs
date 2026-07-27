//! Owns the physical layout of the canonical WASM base witness.
//!
//! Named VM columns form the fixed prefix, followed by Poseidon advice and
//! reusable range-check bits. Backend-specific advice and batching extend
//! this base layout separately.

use crate::layout::{ColumnWidth, COLUMN_SPECS, NAMED_COLUMN_COUNT};

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
    let mut i = 0;
    while i < NAMED_COLUMN_COUNT {
        assert!(COLUMN_SPECS[i].index == i);
        offsets[i + 1] = offsets[i] + decomposed_bits(COLUMN_SPECS[i].width);
        i += 1;
    }
    offsets
}

const RANGE_BIT_OFFSETS: [usize; NAMED_COLUMN_COUNT + 1] = build_range_bit_offsets();
const RANGE_BIT_COUNT: usize = RANGE_BIT_OFFSETS[NAMED_COLUMN_COUNT];

pub(crate) const NAMED_COLUMNS: WitnessRegion = WitnessRegion::new(0, NAMED_COLUMN_COUNT);

// Kept as a standalone constant so Poseidon can use its assigned base without
// depending on the `POSEIDON_AUX` value that itself contains Poseidon's width.
pub(crate) const POSEIDON_AUX_START: usize = NAMED_COLUMNS.end();
pub(crate) const POSEIDON_AUX: WitnessRegion =
    WitnessRegion::new(POSEIDON_AUX_START, crate::ccs::host_event_chain::AUX_WIDTH);
pub(crate) const RANGE_BITS: WitnessRegion = WitnessRegion::new(POSEIDON_AUX.end(), RANGE_BIT_COUNT);

/// Width of the WASM witness after named columns, Poseidon advice, and
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
    let mut widths: Vec<usize> = COLUMN_SPECS
        .iter()
        .map(|spec| match spec.width {
            ColumnWidth::Boolean => 1,
            ColumnWidth::Byte => 8,
            ColumnWidth::U32 => 32,
            ColumnWidth::Field => 64,
        })
        .collect();

    debug_assert_eq!(widths.len(), NAMED_COLUMNS.end());
    widths.extend(crate::ccs::host_event_chain::auxiliary_column_widths());
    debug_assert_eq!(widths.len(), POSEIDON_AUX.end());
    widths.resize(RANGE_BITS.end(), 1);
    debug_assert_eq!(widths.len(), RANGE_CHECKED_WITNESS_WIDTH);
    widths
}
