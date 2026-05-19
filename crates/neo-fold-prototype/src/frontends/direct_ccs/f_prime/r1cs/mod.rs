//! Owns the first low-norm R1CS boundary for compact direct F' advice.
//!
//! This is not the full exact `enc(F')` verifier body. Caller-supplied source
//! images prove only binary low-norm material, public `x_out` linkage, and
//! Poseidon2 boundary digests. Crate-owned native advice additionally installs
//! compact NIFS.V authority rows derived from the latest Direct CCS/F' step.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsMatrix, CscMat};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

mod poseidon;

use super::super::adapter::direct_ccs_program_from_sparse_r1cs_with_public_input_len;
use super::super::public_image::DIRECT_CCS_TRIVIAL_PC;
use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram};
use super::super::step::{direct_ccs_step_from_low_norm_full_witness, DirectCcsStep};
use super::{DirectCcsFPrimeLowNormSourceImage, DirectCcsNativeFPrimeAdvice};
use crate::construction2::CONSTRUCTION2_ENC_INST_BITS;
use poseidon::{add_poseidon_linkage_constraints, estimated_poseidon_digest_recomputation_cost};

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

mod build;
mod carries;
mod constraints;
mod nifs_authority;
mod shape;
mod source;

use constraints::*;
use nifs_authority::DirectCcsFPrimeNifsAuthoritySpec;
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
        Self::from_source_image_with_authority(
            &source,
            &image.x_out.field_image(),
            expected_kappa,
            expected_fresh_claims,
            expected_carry_claims,
            Some(DirectCcsFPrimeNifsAuthoritySpec::from_compact_image(image)?),
        )
    }

    pub fn from_source_image(
        source: &DirectCcsFPrimeLowNormSourceImage,
        public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
        expected_kappa: u64,
        expected_fresh_claims: u64,
        expected_carry_claims: u64,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Self::from_source_image_with_authority(
            source,
            public_x_out_bits,
            expected_kappa,
            expected_fresh_claims,
            expected_carry_claims,
            None,
        )
    }

    fn from_source_image_with_authority(
        source: &DirectCcsFPrimeLowNormSourceImage,
        public_x_out_bits: &[F; CONSTRUCTION2_ENC_INST_BITS],
        expected_kappa: u64,
        expected_fresh_claims: u64,
        expected_carry_claims: u64,
        nifs_authority: Option<DirectCcsFPrimeNifsAuthoritySpec>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        build::build_low_norm_source_r1cs(
            source,
            public_x_out_bits,
            expected_kappa,
            expected_fresh_claims,
            expected_carry_claims,
            nifs_authority.as_ref(),
        )
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
