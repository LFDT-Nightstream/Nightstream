//! Builder for the low-norm direct F' source R1CS shell.

use super::carries::{push_addition_carries, push_increment_carries};
use super::nifs_authority::{add_nifs_authority_constraints, DirectCcsFPrimeNifsAuthoritySpec, NIFS_AUTHORITY_ROWS};
use super::source::validate_low_norm_source_r1cs_inputs;
use super::*;

pub(super) fn build_low_norm_source_r1cs(
    source: &DirectCcsFPrimeLowNormSourceImage,
    public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
    expected_kappa: u64,
    expected_fresh_claims: u64,
    expected_carry_claims: u64,
    nifs_authority: Option<&DirectCcsFPrimeNifsAuthoritySpec>,
) -> Result<DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeSnarkError> {
    validate_low_norm_source_r1cs_inputs(source, expected_kappa)?;
    let base_shape = DirectCcsFPrimeLowNormSourceR1csShape::from_source(source);
    let source_start_col = base_shape.public_input_len;
    let carry_start_col = source_start_col + source.len();
    let canonical_aux_start_col = carry_start_col + base_shape.variables.counter_carry_bits;
    let mut witness = build_low_norm_source_witness(source, public_x_out_bits, &base_shape)?;

    let poseidon = build_poseidon_linkage_constraints(source, source_start_col, &mut witness)?;
    let mut shape = DirectCcsFPrimeLowNormSourceR1csShape::from_source_metadata(
        source.len(),
        source.field_lane_count(),
        poseidon.aux_bits,
        poseidon.rows,
        nifs_authority.map_or(0, |_| NIFS_AUTHORITY_ROWS),
    );
    let shell = build_low_norm_source_shell_constraints(
        source,
        &shape,
        source_start_col,
        carry_start_col,
        canonical_aux_start_col,
        expected_kappa,
        expected_fresh_claims,
        expected_carry_claims,
    );
    let mut a_trips = shell.a_trips;
    let mut b_trips = shell.b_trips;
    let mut c_trips = shell.c_trips;
    let mut row = shell.rows;
    append_triplets_with_row_offset(&mut a_trips, &poseidon.a_trips, row);
    append_triplets_with_row_offset(&mut b_trips, &poseidon.b_trips, row);
    append_triplets_with_row_offset(&mut c_trips, &poseidon.c_trips, row);
    row += shape.constraints.poseidon_digest_recomputation;
    if let Some(spec) = nifs_authority {
        let authority_start = row;
        add_nifs_authority_constraints(&mut a_trips, &mut b_trips, &mut row, source_start_col, source, *spec);
        debug_assert_eq!(row - authority_start, shape.constraints.nifs_v_verifier);
    }
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

struct SourceShellConstraints {
    a_trips: Vec<(usize, usize, F)>,
    b_trips: Vec<(usize, usize, F)>,
    c_trips: Vec<(usize, usize, F)>,
    rows: usize,
}

impl SourceShellConstraints {
    fn with_capacity(shape: &DirectCcsFPrimeLowNormSourceR1csShape) -> Self {
        let link_constraints = shape.shell_constraints() - shape.constraints.bitness;
        Self {
            a_trips: Vec::with_capacity(shape.constraints.bitness + link_constraints * 2),
            b_trips: Vec::with_capacity(shape.constraints.bitness * 2 + link_constraints),
            c_trips: Vec::new(),
            rows: 0,
        }
    }

    fn add_private_bitness_constraints(
        &mut self,
        source_start_col: usize,
        shape: &DirectCcsFPrimeLowNormSourceR1csShape,
    ) {
        for col in source_start_col..shape.variable_count {
            self.a_trips.push((self.rows, col, F::ONE));
            self.b_trips.push((self.rows, col, F::ONE));
            self.b_trips.push((self.rows, ONE_COL, -F::ONE));
            self.rows += 1;
        }
    }

    fn add_public_output_link_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
    ) {
        for bit_index in 0..CONSTRUCTION2_ENC_INST_BITS {
            self.a_trips
                .push((self.rows, PUBLIC_X_OUT_START_COL + bit_index, F::ONE));
            self.a_trips.push((
                self.rows,
                source_start_col + source.compact_x_out_bit_offset() + bit_index,
                -F::ONE,
            ));
            self.b_trips.push((self.rows, ONE_COL, F::ONE));
            self.rows += 1;
        }
    }

    fn add_construction2_boundary_link_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
    ) {
        add_source_bit_equality_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.compact_x_in_bit_offset(),
            source.construction2_u_in_x_i_bit_offset(),
        );
        add_source_bit_equality_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.compact_construction2_u_in_digest_bit_offset(),
            source.construction2_u_in_fresh_digest_bit_offset(),
        );
        add_source_bit_equality_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.compact_x_out_bit_offset(),
            source.construction2_u_out_x_i_bit_offset(),
        );
        add_source_bit_equality_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.compact_construction2_u_out_digest_bit_offset(),
            source.construction2_u_out_fresh_digest_bit_offset(),
        );
    }

    fn add_construction2_commitment_shape_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
        expected_kappa: u64,
    ) {
        for (offset, expected) in [
            (source.construction2_u_in_commitment_d_bit_offset(), D as u64),
            (source.construction2_u_in_commitment_kappa_bit_offset(), expected_kappa),
            (source.construction2_u_out_commitment_d_bit_offset(), D as u64),
            (source.construction2_u_out_commitment_kappa_bit_offset(), expected_kappa),
        ] {
            add_source_u64_constant_constraints(
                &mut self.a_trips,
                &mut self.b_trips,
                &mut self.rows,
                source_start_col,
                offset,
                expected,
            );
        }
    }

    fn add_structural_constant_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
        expected_fresh_claims: u64,
        expected_carry_claims: u64,
    ) {
        for (offset, expected) in [
            (source.pc_bit_offset(), DIRECT_CCS_TRIVIAL_PC),
            (source.fresh_claims_bit_offset(), expected_fresh_claims),
            (source.incoming_ce_claims_bit_offset(), expected_carry_claims),
        ] {
            add_source_u64_constant_constraints(
                &mut self.a_trips,
                &mut self.b_trips,
                &mut self.rows,
                source_start_col,
                offset,
                expected,
            );
        }
    }

    fn add_structural_counter_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
        carry_start_col: usize,
        shape: &DirectCcsFPrimeLowNormSourceR1csShape,
    ) {
        let mut carry_cursor = carry_start_col;
        add_source_u64_increment_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.chunk_count_in_bit_offset(),
            source.chunk_count_out_bit_offset(),
            carry_cursor,
        );
        carry_cursor += U64_ADD_CARRY_BITS;
        add_source_u64_add_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.step_count_in_bit_offset(),
            source.fresh_claims_bit_offset(),
            source.step_count_out_bit_offset(),
            carry_cursor,
        );
        carry_cursor += U64_ADD_CARRY_BITS;
        add_source_u64_add_constraints(
            &mut self.a_trips,
            &mut self.b_trips,
            &mut self.rows,
            source_start_col,
            source.incoming_ce_claims_bit_offset(),
            source.fresh_claims_bit_offset(),
            source.output_ce_claims_bit_offset(),
            carry_cursor,
        );
        carry_cursor += U64_ADD_CARRY_BITS;
        debug_assert_eq!(carry_cursor, carry_start_col + shape.variables.counter_carry_bits);
    }

    fn add_nifs_mirror_constraints(&mut self, source: &DirectCcsFPrimeLowNormSourceImage, source_start_col: usize) {
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
                &mut self.a_trips,
                &mut self.b_trips,
                &mut self.rows,
                source_start_col,
                lhs_offset,
                rhs_offset,
            );
        }
    }

    fn add_canonical_field_lane_constraints(
        &mut self,
        source: &DirectCcsFPrimeLowNormSourceImage,
        source_start_col: usize,
        canonical_aux_start_col: usize,
        shape: &DirectCcsFPrimeLowNormSourceR1csShape,
    ) {
        let mut canonical_aux_cursor = canonical_aux_start_col;
        for &offset in source.field_lane_bit_offsets() {
            add_goldilocks_canonical_lane_constraints(
                &mut self.a_trips,
                &mut self.b_trips,
                &mut self.c_trips,
                &mut self.rows,
                source_start_col + offset,
                canonical_aux_cursor,
            );
            canonical_aux_cursor += GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE;
        }
        debug_assert_eq!(
            canonical_aux_cursor,
            canonical_aux_start_col + shape.variables.canonical_field_lane_aux_bits
        );
    }
}

fn build_low_norm_source_shell_constraints(
    source: &DirectCcsFPrimeLowNormSourceImage,
    shape: &DirectCcsFPrimeLowNormSourceR1csShape,
    source_start_col: usize,
    carry_start_col: usize,
    canonical_aux_start_col: usize,
    expected_kappa: u64,
    expected_fresh_claims: u64,
    expected_carry_claims: u64,
) -> SourceShellConstraints {
    let mut shell = SourceShellConstraints::with_capacity(shape);
    shell.add_private_bitness_constraints(source_start_col, shape);
    shell.add_public_output_link_constraints(source, source_start_col);
    shell.add_construction2_boundary_link_constraints(source, source_start_col);
    shell.add_construction2_commitment_shape_constraints(source, source_start_col, expected_kappa);
    shell.add_structural_constant_constraints(source, source_start_col, expected_fresh_claims, expected_carry_claims);
    shell.add_structural_counter_constraints(source, source_start_col, carry_start_col, shape);
    shell.add_nifs_mirror_constraints(source, source_start_col);
    shell.add_canonical_field_lane_constraints(source, source_start_col, canonical_aux_start_col, shape);
    debug_assert_eq!(shell.rows, shape.shell_constraints());
    shell
}

struct PoseidonLinkageConstraints {
    a_trips: Vec<(usize, usize, F)>,
    b_trips: Vec<(usize, usize, F)>,
    c_trips: Vec<(usize, usize, F)>,
    rows: usize,
    aux_bits: usize,
}

fn build_poseidon_linkage_constraints(
    source: &DirectCcsFPrimeLowNormSourceImage,
    source_start_col: usize,
    witness: &mut Vec<F>,
) -> Result<PoseidonLinkageConstraints, DirectCcsFPrimeSnarkError> {
    let poseidon_aux_start_col = witness.len();
    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let mut c_trips = Vec::new();
    let mut rows = 0usize;
    add_poseidon_linkage_constraints(
        &mut a_trips,
        &mut b_trips,
        &mut c_trips,
        &mut rows,
        witness,
        source,
        source_start_col,
    )?;
    Ok(PoseidonLinkageConstraints {
        a_trips,
        b_trips,
        c_trips,
        rows,
        aux_bits: witness.len() - poseidon_aux_start_col,
    })
}

fn build_low_norm_source_witness(
    source: &DirectCcsFPrimeLowNormSourceImage,
    public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
    shape: &DirectCcsFPrimeLowNormSourceR1csShape,
) -> Result<Vec<F>, DirectCcsFPrimeSnarkError> {
    let mut carries = Vec::with_capacity(shape.variables.counter_carry_bits);
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
    debug_assert_eq!(carries.len(), shape.variables.counter_carry_bits);

    let canonical_aux = canonical_field_lane_aux_bits(source)?;
    debug_assert_eq!(canonical_aux.len(), shape.variables.canonical_field_lane_aux_bits);

    let mut witness = Vec::with_capacity(shape.variable_count);
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
    Ok(witness)
}
