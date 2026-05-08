//! Owns the first low-norm R1CS boundary for compact direct F' advice.
//!
//! This is not the full `enc(F')` verifier relation. It proves that the source
//! image is binary low-norm material, links public `x_out`, and recomputes the
//! cheap Construction-2 boundary digests. The NIFS.V verifier body is still
//! required before this can become standalone recursive proof authority.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsMatrix, CscMat};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use super::f_prime::{DirectCcsFPrimeLowNormSourceImage, DirectCcsNativeFPrimeAdvice};
use super::f_prime_r1cs_poseidon::{add_poseidon_linkage_constraints, estimated_poseidon_digest_recomputation_cost};
use super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
use super::public_image::DIRECT_CCS_TRIVIAL_PC;
use super::r1cs::{
    direct_ccs_program_from_sparse_r1cs_with_public_input_len, direct_ccs_step_from_low_norm_full_witness,
};
use crate::construction2::CONSTRUCTION2_ENC_INST_BITS;

const ONE_COL: usize = 0;
const PUBLIC_X_OUT_START_COL: usize = 1;
const U64_BITS: usize = 64;
const U64_ADD_CARRY_BITS: usize = U64_BITS - 1;
const STRUCTURAL_U64_ADDITIONS: usize = 3;
const STRUCTURAL_U64_EQUALITIES: usize = 6;
const STRUCTURAL_U64_FIXED_ARITY_CONSTANTS: usize = 2;
const STRUCTURAL_U64_CONSTANTS: usize = 1 + STRUCTURAL_U64_FIXED_ARITY_CONSTANTS;
const STRUCTURAL_FIXED_ARITY_CONSTRAINTS: usize = STRUCTURAL_U64_FIXED_ARITY_CONSTANTS * U64_BITS;
const STRUCTURAL_COUNTER_CARRY_BITS: usize = STRUCTURAL_U64_ADDITIONS * U64_ADD_CARRY_BITS;
const STRUCTURAL_COUNTER_CONSTRAINTS: usize =
    (STRUCTURAL_U64_ADDITIONS + STRUCTURAL_U64_EQUALITIES + STRUCTURAL_U64_CONSTANTS) * U64_BITS;
const GOLDILOCKS_HIGH_BITS: usize = 32;
const GOLDILOCKS_LOW_BITS: usize = 32;
const GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE: usize = GOLDILOCKS_HIGH_BITS - 1;
const GOLDILOCKS_CANONICAL_CONSTRAINTS_PER_LANE: usize = GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE + GOLDILOCKS_LOW_BITS;

mod constraints;
mod shape;
mod source;

use constraints::*;
pub use shape::DirectCcsFPrimeLowNormSourceR1csShape;
use source::*;

#[derive(Clone, Debug)]
pub struct DirectCcsFPrimeLowNormSourceR1cs {
    pub a: CcsMatrix<F>,
    pub b: CcsMatrix<F>,
    pub c: CcsMatrix<F>,
    pub witness: Vec<F>,
    pub shape: DirectCcsFPrimeLowNormSourceR1csShape,
}

impl DirectCcsFPrimeLowNormSourceR1cs {
    pub fn from_native_advice(
        advice: &DirectCcsNativeFPrimeAdvice,
        expected_kappa: u64,
        expected_fresh_claims: u64,
        expected_carry_claims: u64,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let source = advice.low_norm_source_image()?;
        let image = advice.compact_image();
        if expected_kappa == 0
            || advice.construction2_u_in().commitment_kappa != expected_kappa
            || advice.construction2_u_out().commitment_kappa != expected_kappa
        {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' source R1CS requires Construction-2 commitment kappa to match program params".into(),
            ));
        }
        if image.fresh_claims != expected_fresh_claims
            || image.incoming_ce_claims != expected_carry_claims
            || image.final_ce_claims != expected_carry_claims
        {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' source R1CS requires fixed chunk and carried-CE arity".into(),
            ));
        }
        Self::from_source_image(
            &source,
            &image.x_out.field_image(),
            expected_kappa,
            expected_fresh_claims,
            expected_carry_claims,
        )
    }

    pub fn from_source_image(
        source: &DirectCcsFPrimeLowNormSourceImage,
        public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
        expected_kappa: u64,
        expected_fresh_claims: u64,
        expected_carry_claims: u64,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        if expected_kappa == 0 {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' source R1CS requires nonzero Construction-2 commitment kappa".into(),
            ));
        }
        if source.is_empty() {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' low-norm source R1CS requires a non-empty source image".into(),
            ));
        }
        validate_source_bit_range(source, source.mat_digest_bit_offset(), "matrix digest")?;
        validate_source_bit_range(source, source.vk_fs_digest_bit_offset(), "vk_fs digest")?;
        validate_source_bit_range(source, source.compact_x_in_bit_offset(), "compact x_in")?;
        validate_source_bit_range(source, source.compact_x_out_bit_offset(), "compact x_out")?;
        validate_source_u64_range(source, source.pc_bit_offset(), "pc")?;
        validate_source_u64_range(source, source.chunk_count_in_bit_offset(), "chunk count input")?;
        validate_source_u64_range(source, source.chunk_count_out_bit_offset(), "chunk count output")?;
        validate_source_u64_range(source, source.step_count_in_bit_offset(), "step count input")?;
        validate_source_u64_range(source, source.step_count_out_bit_offset(), "step count output")?;
        validate_source_bit_range(
            source,
            source.initial_boundary_digest_bit_offset(),
            "initial boundary digest",
        )?;
        validate_source_bit_range(
            source,
            source.current_boundary_in_digest_bit_offset(),
            "current boundary input digest",
        )?;
        validate_source_bit_range(
            source,
            source.current_boundary_out_digest_bit_offset(),
            "current boundary output digest",
        )?;
        validate_source_bit_range(
            source,
            source.public_trace_in_digest_bit_offset(),
            "public trace input digest",
        )?;
        validate_source_bit_range(
            source,
            source.public_trace_out_digest_bit_offset(),
            "public trace output digest",
        )?;
        validate_source_bit_range(
            source,
            source.semantic_accumulator_in_digest_bit_offset(),
            "semantic accumulator input digest",
        )?;
        validate_source_bit_range(
            source,
            source.semantic_accumulator_out_digest_bit_offset(),
            "semantic accumulator output digest",
        )?;
        validate_source_bit_range(
            source,
            source.f_prime_accumulator_in_digest_bit_offset(),
            "F' accumulator input digest",
        )?;
        validate_source_bit_range(
            source,
            source.f_prime_accumulator_out_digest_bit_offset(),
            "F' accumulator output digest",
        )?;
        validate_source_u64_range(source, source.fresh_claims_bit_offset(), "fresh claim count")?;
        validate_source_u64_range(
            source,
            source.incoming_ce_claims_bit_offset(),
            "incoming CE claim count",
        )?;
        validate_source_u64_range(source, source.output_ce_claims_bit_offset(), "output CE claim count")?;
        validate_source_u64_range(source, source.final_ce_claims_bit_offset(), "final CE claim count")?;
        for (offset, label) in [
            (source.nifs_chunk_index_bit_offset(), "NIFS chunk index"),
            (source.nifs_fresh_claims_bit_offset(), "NIFS fresh claim count"),
            (
                source.nifs_incoming_ce_claims_bit_offset(),
                "NIFS incoming CE claim count",
            ),
            (source.nifs_pi_ccs_outputs_bit_offset(), "NIFS Pi_CCS output count"),
            (source.nifs_final_ce_claims_bit_offset(), "NIFS final CE claim count"),
            (source.nifs_fe_sumcheck_rounds_bit_offset(), "NIFS FE sumcheck rounds"),
            (
                source.nifs_fe_sumcheck_messages_bit_offset(),
                "NIFS FE sumcheck messages",
            ),
            (source.nifs_nc_sumcheck_rounds_bit_offset(), "NIFS NC sumcheck rounds"),
            (
                source.nifs_nc_sumcheck_messages_bit_offset(),
                "NIFS NC sumcheck messages",
            ),
            (
                source.nifs_transcript_absorbed_in_bit_offset(),
                "NIFS transcript absorbed input",
            ),
            (
                source.nifs_transcript_absorbed_out_bit_offset(),
                "NIFS transcript absorbed output",
            ),
        ] {
            validate_source_u64_range(source, offset, label)?;
        }
        validate_source_bit_range(
            source,
            source.compact_construction2_u_in_digest_bit_offset(),
            "compact Construction-2 input digest",
        )?;
        validate_source_bit_range(
            source,
            source.compact_construction2_u_out_digest_bit_offset(),
            "compact Construction-2 output digest",
        )?;
        validate_source_bit_range(source, source.latest_chunk_digest_bit_offset(), "latest chunk digest")?;
        validate_source_bit_range(source, source.latest_fold_digest_bit_offset(), "latest fold digest")?;
        validate_source_bit_range(
            source,
            source.latest_chunk_relation_digest_bit_offset(),
            "latest chunk relation digest",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_in_fresh_digest_bit_offset(),
            "Construction-2 input boundary fresh digest",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_in_commitment_digest_bit_offset(),
            "Construction-2 input boundary commitment digest",
        )?;
        validate_source_u64_range(
            source,
            source.construction2_u_in_commitment_d_bit_offset(),
            "Construction-2 input boundary commitment d",
        )?;
        validate_source_u64_range(
            source,
            source.construction2_u_in_commitment_kappa_bit_offset(),
            "Construction-2 input boundary commitment kappa",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_in_x_i_bit_offset(),
            "Construction-2 input boundary x_i",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_out_fresh_digest_bit_offset(),
            "Construction-2 output boundary fresh digest",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_out_commitment_digest_bit_offset(),
            "Construction-2 output boundary commitment digest",
        )?;
        validate_source_u64_range(
            source,
            source.construction2_u_out_commitment_d_bit_offset(),
            "Construction-2 output boundary commitment d",
        )?;
        validate_source_u64_range(
            source,
            source.construction2_u_out_commitment_kappa_bit_offset(),
            "Construction-2 output boundary commitment kappa",
        )?;
        validate_source_bit_range(
            source,
            source.construction2_u_out_x_i_bit_offset(),
            "Construction-2 output boundary x_i",
        )?;
        for &offset in source.field_lane_bit_offsets() {
            validate_source_u64_range(source, offset, "canonical field lane")?;
        }
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

        Ok(Self {
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

    pub fn to_direct_ccs_program(&self, params: &NeoParams) -> Result<DirectCcsProgram, DirectCcsFPrimeSnarkError> {
        direct_ccs_program_from_sparse_r1cs_with_public_input_len(
            params,
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.shape.public_input_len,
        )
    }

    pub fn to_direct_ccs_step<L>(
        &self,
        program: &DirectCcsProgram,
        log: &L,
        label: impl Into<String>,
    ) -> Result<DirectCcsStep, DirectCcsFPrimeSnarkError>
    where
        L: SModuleHomomorphism<F, Commitment>,
    {
        direct_ccs_step_from_low_norm_full_witness(program, log, label, &self.witness, self.shape.public_input_len)
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        if self.witness.len() != self.shape.variable_count {
            return Some(0);
        }
        let az = matrix_mul(&self.a, &self.witness);
        let bz = matrix_mul(&self.b, &self.witness);
        let cz = matrix_mul(&self.c, &self.witness);
        if az.len() != self.shape.constraint_count
            || bz.len() != self.shape.constraint_count
            || cz.len() != self.shape.constraint_count
        {
            return Some(0);
        }
        az.iter()
            .zip(bz.iter())
            .zip(cz.iter())
            .position(|((a, b), c)| *a * *b != *c)
    }
}

fn push_increment_carries(carries: &mut Vec<u8>, input: u64) {
    let mut carry = 1u128;
    for bit_index in 0..U64_BITS {
        let sum = ((input >> bit_index) & 1) as u128 + carry;
        carry = sum >> 1;
        if bit_index + 1 < U64_BITS {
            carries.push(carry as u8);
        }
    }
}

fn push_addition_carries(carries: &mut Vec<u8>, lhs: u64, rhs: u64) {
    let mut carry = 0u128;
    for bit_index in 0..U64_BITS {
        let sum = ((lhs >> bit_index) & 1) as u128 + ((rhs >> bit_index) & 1) as u128 + carry;
        carry = sum >> 1;
        if bit_index + 1 < U64_BITS {
            carries.push(carry as u8);
        }
    }
}

fn matrix_mul(matrix: &CcsMatrix<F>, witness: &[F]) -> Vec<F> {
    match matrix {
        CcsMatrix::Identity { n } => witness[..*n].to_vec(),
        CcsMatrix::Csc(csc) => {
            let mut out = vec![F::ZERO; csc.nrows];
            for col in 0..csc.ncols {
                let value = witness.get(col).copied().unwrap_or(F::ZERO);
                if value == F::ZERO {
                    continue;
                }
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    out[csc.row_idx[idx]] += csc.vals[idx] * value;
                }
            }
            out
        }
    }
}
