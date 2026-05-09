use super::*;

pub(super) fn add_source_bit_equality_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    lhs_offset: usize,
    rhs_offset: usize,
) {
    for bit_index in 0..CONSTRUCTION2_ENC_INST_BITS {
        a_trips.push((*row, source_start_col + lhs_offset + bit_index, F::ONE));
        a_trips.push((*row, source_start_col + rhs_offset + bit_index, -F::ONE));
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
}

pub(super) fn add_source_u64_constant_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    source_offset: usize,
    expected: u64,
) {
    for bit_index in 0..64 {
        a_trips.push((*row, source_start_col + source_offset + bit_index, F::ONE));
        if ((expected >> bit_index) & 1) != 0 {
            a_trips.push((*row, ONE_COL, -F::ONE));
        }
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
}

pub(super) fn add_source_u64_increment_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    input_offset: usize,
    output_offset: usize,
    carry_start_col: usize,
) {
    for bit_index in 0..U64_BITS {
        a_trips.push((*row, source_start_col + input_offset + bit_index, F::ONE));
        if bit_index == 0 {
            a_trips.push((*row, ONE_COL, F::ONE));
        } else {
            a_trips.push((*row, carry_start_col + bit_index - 1, F::ONE));
        }
        a_trips.push((*row, source_start_col + output_offset + bit_index, -F::ONE));
        if bit_index + 1 < U64_BITS {
            a_trips.push((*row, carry_start_col + bit_index, -F::from_u64(2)));
        }
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
}

pub(super) fn add_source_u64_add_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    lhs_offset: usize,
    rhs_offset: usize,
    output_offset: usize,
    carry_start_col: usize,
) {
    for bit_index in 0..U64_BITS {
        a_trips.push((*row, source_start_col + lhs_offset + bit_index, F::ONE));
        a_trips.push((*row, source_start_col + rhs_offset + bit_index, F::ONE));
        if bit_index > 0 {
            a_trips.push((*row, carry_start_col + bit_index - 1, F::ONE));
        }
        a_trips.push((*row, source_start_col + output_offset + bit_index, -F::ONE));
        if bit_index + 1 < U64_BITS {
            a_trips.push((*row, carry_start_col + bit_index, -F::from_u64(2)));
        }
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
}

pub(super) fn add_source_u64_equality_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    lhs_offset: usize,
    rhs_offset: usize,
) {
    for bit_index in 0..U64_BITS {
        a_trips.push((*row, source_start_col + lhs_offset + bit_index, F::ONE));
        a_trips.push((*row, source_start_col + rhs_offset + bit_index, -F::ONE));
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
}

pub(super) fn add_goldilocks_canonical_lane_constraints(
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

    let high_all_ones_col = aux_start_col + GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE - 1;
    for low_index in 0..GOLDILOCKS_LOW_BITS {
        a_trips.push((*row, high_all_ones_col, F::ONE));
        b_trips.push((*row, lane_start_col + low_index, F::ONE));
        *row += 1;
    }
}

pub(super) fn append_triplets_with_row_offset(
    out: &mut Vec<(usize, usize, F)>,
    input: &[(usize, usize, F)],
    row_offset: usize,
) {
    out.extend(
        input
            .iter()
            .map(|(row, col, value)| (row + row_offset, *col, *value)),
    );
}
