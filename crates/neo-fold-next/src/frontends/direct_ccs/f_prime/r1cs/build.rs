//! Builder for the low-norm direct F' source R1CS shell.

use super::carries::{push_addition_carries, push_increment_carries};
use super::source::validate_low_norm_source_r1cs_inputs;
use super::*;

pub(super) fn build_low_norm_source_r1cs(
    source: &DirectCcsFPrimeLowNormSourceImage,
    public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
    expected_kappa: u64,
    expected_fresh_claims: u64,
    expected_carry_claims: u64,
) -> Result<DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeSnarkError> {
    validate_low_norm_source_r1cs_inputs(source, expected_kappa)?;
    let base_shape = DirectCcsFPrimeLowNormSourceR1csShape::from_source(source);
    let source_start_col = base_shape.public_input_len;
    let carry_start_col = source_start_col + source.len();
    let canonical_aux_start_col = carry_start_col + base_shape.counter_carry_bits;
    let mut carries = Vec::with_capacity(base_shape.counter_carry_bits);
    push_increment_carries(&mut carries, source_u64_at(source, source.chunk_count_in_bit_offset())?);
    push_addition_carries(
        &mut carries,
        source_u64_at(source, source.step_count_in_bit_offset())?,
        source_u64_at(source, source.fresh_claims_bit_offset())?,
    );
    push_addition_carries(
        &mut carries,
        source_u64_at(source, source.incoming_ce_claims_bit_offset())?,
        source_u64_at(source, source.fresh_claims_bit_offset())?,
    );
    debug_assert_eq!(carries.len(), base_shape.counter_carry_bits);
    let canonical_aux = canonical_field_lane_aux_bits(source)?;
    debug_assert_eq!(canonical_aux.len(), base_shape.canonical_field_lane_aux_bits);
    let mut witness = Vec::with_capacity(base_shape.variable_count);
    witness.push(F::ONE);
    witness.extend_from_slice(public_x_out_bits);
    witness.extend_from_slice(source.values());
    witness.extend(carries.iter().copied().map(|bit| F::from_u64(bit as u64)));
    witness.extend(
        canonical_aux
            .iter()
            .copied()
            .map(|bit| F::from_u64(bit as u64)),
    );

    let poseidon_aux_start_col = witness.len();
    let mut poseidon_a_trips = Vec::new();
    let mut poseidon_b_trips = Vec::new();
    let mut poseidon_c_trips = Vec::new();
    let mut poseidon_rows = 0usize;
    add_poseidon_linkage_constraints(
        &mut poseidon_a_trips,
        &mut poseidon_b_trips,
        &mut poseidon_c_trips,
        &mut poseidon_rows,
        &mut witness,
        source,
        source_start_col,
    )?;
    let poseidon_digest_recomputation_aux_bits = witness.len() - poseidon_aux_start_col;
    let mut shape = DirectCcsFPrimeLowNormSourceR1csShape::from_source_metadata(
        source.len(),
        source.field_lane_count(),
        poseidon_digest_recomputation_aux_bits,
        poseidon_rows,
    );
    let link_constraints = shape.shell_constraints() - shape.bit_constraints;
    let mut a_trips = Vec::with_capacity(shape.bit_constraints + link_constraints * 2);
    let mut b_trips = Vec::with_capacity(shape.bit_constraints * 2 + link_constraints);
    let mut c_trips = Vec::new();
    let mut row = 0usize;
    for col in PUBLIC_X_OUT_START_COL..shape.variable_count {
        a_trips.push((row, col, F::ONE));
        b_trips.push((row, col, F::ONE));
        b_trips.push((row, ONE_COL, -F::ONE));
        row += 1;
    }
    for bit_index in 0..CONSTRUCTION2_ENC_INST_BITS {
        a_trips.push((row, PUBLIC_X_OUT_START_COL + bit_index, F::ONE));
        a_trips.push((
            row,
            source_start_col + source.compact_x_out_bit_offset() + bit_index,
            -F::ONE,
        ));
        b_trips.push((row, ONE_COL, F::ONE));
        row += 1;
    }
    add_source_bit_equality_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.compact_x_in_bit_offset(),
        source.construction2_u_in_x_i_bit_offset(),
    );
    add_source_bit_equality_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.compact_construction2_u_in_digest_bit_offset(),
        source.construction2_u_in_fresh_digest_bit_offset(),
    );
    add_source_bit_equality_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.compact_x_out_bit_offset(),
        source.construction2_u_out_x_i_bit_offset(),
    );
    add_source_bit_equality_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.compact_construction2_u_out_digest_bit_offset(),
        source.construction2_u_out_fresh_digest_bit_offset(),
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.construction2_u_in_commitment_d_bit_offset(),
        D as u64,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.construction2_u_in_commitment_kappa_bit_offset(),
        expected_kappa,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.construction2_u_out_commitment_d_bit_offset(),
        D as u64,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.construction2_u_out_commitment_kappa_bit_offset(),
        expected_kappa,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.pc_bit_offset(),
        DIRECT_CCS_TRIVIAL_PC,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.fresh_claims_bit_offset(),
        expected_fresh_claims,
    );
    add_source_u64_constant_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.incoming_ce_claims_bit_offset(),
        expected_carry_claims,
    );
    let mut carry_cursor = carry_start_col;
    add_source_u64_increment_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.chunk_count_in_bit_offset(),
        source.chunk_count_out_bit_offset(),
        carry_cursor,
    );
    carry_cursor += U64_ADD_CARRY_BITS;
    add_source_u64_add_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.step_count_in_bit_offset(),
        source.fresh_claims_bit_offset(),
        source.step_count_out_bit_offset(),
        carry_cursor,
    );
    carry_cursor += U64_ADD_CARRY_BITS;
    add_source_u64_add_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut row,
        source_start_col,
        source.incoming_ce_claims_bit_offset(),
        source.fresh_claims_bit_offset(),
        source.output_ce_claims_bit_offset(),
        carry_cursor,
    );
    carry_cursor += U64_ADD_CARRY_BITS;
    debug_assert_eq!(carry_cursor, carry_start_col + shape.counter_carry_bits);
    for (lhs_offset, rhs_offset) in [
        (
            source.final_ce_claims_bit_offset(),
            source.incoming_ce_claims_bit_offset(),
        ),
        (source.nifs_chunk_index_bit_offset(), source.chunk_count_in_bit_offset()),
        (source.nifs_fresh_claims_bit_offset(), source.fresh_claims_bit_offset()),
        (
            source.nifs_incoming_ce_claims_bit_offset(),
            source.incoming_ce_claims_bit_offset(),
        ),
        (
            source.nifs_pi_ccs_outputs_bit_offset(),
            source.output_ce_claims_bit_offset(),
        ),
        (
            source.nifs_final_ce_claims_bit_offset(),
            source.final_ce_claims_bit_offset(),
        ),
    ] {
        add_source_u64_equality_constraints(
            &mut a_trips,
            &mut b_trips,
            &mut row,
            source_start_col,
            lhs_offset,
            rhs_offset,
        );
    }
    let mut canonical_aux_cursor = canonical_aux_start_col;
    for &offset in source.field_lane_bit_offsets() {
        add_goldilocks_canonical_lane_constraints(
            &mut a_trips,
            &mut b_trips,
            &mut c_trips,
            &mut row,
            source_start_col + offset,
            canonical_aux_cursor,
        );
        canonical_aux_cursor += GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE;
    }
    debug_assert_eq!(
        canonical_aux_cursor,
        canonical_aux_start_col + shape.canonical_field_lane_aux_bits
    );
    append_triplets_with_row_offset(&mut a_trips, &poseidon_a_trips, row);
    append_triplets_with_row_offset(&mut b_trips, &poseidon_b_trips, row);
    append_triplets_with_row_offset(&mut c_trips, &poseidon_c_trips, row);
    row += shape.poseidon_digest_recomputation_constraints;
    debug_assert_eq!(row, shape.constraint_count);
    shape.nonzero_entries = a_trips.len() + b_trips.len() + c_trips.len();

    Ok(DirectCcsFPrimeLowNormSourceR1cs {
        a: CcsMatrix::Csc(CscMat::from_triplets(
            a_trips,
            shape.constraint_count,
            shape.variable_count,
        )),
        b: CcsMatrix::Csc(CscMat::from_triplets(
            b_trips,
            shape.constraint_count,
            shape.variable_count,
        )),
        c: CcsMatrix::Csc(CscMat::from_triplets(
            c_trips,
            shape.constraint_count,
            shape.variable_count,
        )),
        witness,
        shape,
    })
}
