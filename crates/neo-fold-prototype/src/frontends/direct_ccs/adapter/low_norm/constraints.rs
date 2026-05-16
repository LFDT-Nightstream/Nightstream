//! Constraint rows added by low-norm direct R1CS lowering.

use super::*;

pub(super) fn add_bit_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    variable_count: usize,
) {
    for col in 0..variable_count {
        a_trips.push((*row, col, F::ONE));
        b_trips.push((*row, col, F::ONE));
        b_trips.push((*row, 0, -F::ONE));
        *row += 1;
    }
}

pub(super) fn add_canonical_field_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    layout: &DirectR1csLowNormLayout,
    lanes: &[LaneMap],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (col, &kind) in layout.kinds.iter().enumerate() {
        if !kind.needs_canonical_field_check() {
            continue;
        }
        let lane = &lanes[col];
        let aux_start = lane
            .canonical_aux_start_col
            .ok_or_else(|| DirectCcsFPrimeSnarkError::Input("direct low-norm R1CS canonical aux missing".into()))?;
        add_goldilocks_canonical_lane_constraints(a_trips, b_trips, c_trips, row, lane.bits_start_col, aux_start);
    }
    Ok(())
}

fn add_goldilocks_canonical_lane_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    lane_start_col: usize,
    aux_start_col: usize,
) {
    a_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS, F::ONE));
    b_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS + 1, F::ONE));
    c_trips.push((*row, aux_start_col, F::ONE));
    *row += 1;

    for high_index in 2..GOLDILOCKS_HIGH_BITS {
        a_trips.push((*row, aux_start_col + high_index - 2, F::ONE));
        b_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS + high_index, F::ONE));
        c_trips.push((*row, aux_start_col + high_index - 1, F::ONE));
        *row += 1;
    }

    let high_all_ones_col = aux_start_col + GOLDILOCKS_CANONICAL_AUX_BITS - 1;
    for low_index in 0..GOLDILOCKS_LOW_BITS {
        a_trips.push((*row, high_all_ones_col, F::ONE));
        b_trips.push((*row, lane_start_col + low_index, F::ONE));
        *row += 1;
    }
}
