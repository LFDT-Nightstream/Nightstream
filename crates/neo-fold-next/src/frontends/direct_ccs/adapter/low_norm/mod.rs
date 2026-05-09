//! Low-norm lowering for direct R1CS frontends.
//!
//! Direct SuperNeo CCS steps require the full witness to fit the small
//! coefficient bound. This module rewrites an arbitrary sparse R1CS assignment
//! into an equivalent bit/limb R1CS whose witness is binary.

mod constraints;
mod layout;
mod matrix;
mod witness;

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
use super::r1cs_export::DirectSparseR1csExport;
use constraints::{add_bit_constraints, add_canonical_field_constraints};
use layout::LaneMap;
use matrix::expand_matrix;
use witness::{push_canonical_aux_bits, push_lanes_for_range};

pub use layout::{DirectLowNormLaneKind, DirectR1csLowNormLayout};

const FIELD_BITS: usize = 64;
const U32_BITS: usize = 32;
const GOLDILOCKS_LOW_BITS: usize = 32;
const GOLDILOCKS_HIGH_BITS: usize = 32;
const GOLDILOCKS_CANONICAL_AUX_BITS: usize = GOLDILOCKS_HIGH_BITS - 1;

pub fn lower_sparse_r1cs_export_to_low_norm(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
) -> Result<DirectSparseR1csExport, DirectCcsFPrimeSnarkError> {
    validate_layout(export, layout)?;

    let mut witness = Vec::new();
    let mut lanes = vec![LaneMap::empty(); export.variable_count];
    let public_input_len = push_lanes_for_range(export, layout, 0..export.public_input_len, &mut witness, &mut lanes)?;
    push_lanes_for_range(
        export,
        layout,
        export.public_input_len..export.variable_count,
        &mut witness,
        &mut lanes,
    )?;
    push_canonical_aux_bits(export, layout, &mut witness, &mut lanes)?;

    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let mut c_trips = Vec::new();
    expand_matrix(&export.a, export.constraint_count, &lanes, &mut a_trips)?;
    expand_matrix(&export.b, export.constraint_count, &lanes, &mut b_trips)?;
    expand_matrix(&export.c, export.constraint_count, &lanes, &mut c_trips)?;

    let mut row = export.constraint_count;
    add_bit_constraints(&mut a_trips, &mut b_trips, &mut row, witness.len());
    add_canonical_field_constraints(&mut a_trips, &mut b_trips, &mut c_trips, &mut row, layout, &lanes)?;
    let variable_count = witness.len();

    Ok(DirectSparseR1csExport {
        a: CcsMatrix::Csc(CscMat::from_triplets(a_trips, row, variable_count)),
        b: CcsMatrix::Csc(CscMat::from_triplets(b_trips, row, variable_count)),
        c: CcsMatrix::Csc(CscMat::from_triplets(c_trips, row, variable_count)),
        witness,
        public_input_len,
        constraint_count: row,
        variable_count,
    })
}

/// Lowers an exported sparse R1CS assignment and packages it as one direct CCS
/// step. This is the frontend-neutral bridge for circuits whose native R1CS
/// witness contains full field values instead of SuperNeo small digits.
pub fn lower_sparse_r1cs_export_to_low_norm_program_and_step<L>(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    log: &L,
    label: impl Into<String>,
) -> Result<(DirectSparseR1csExport, DirectCcsProgram, DirectCcsStep), DirectCcsFPrimeSnarkError>
where
    L: SModuleHomomorphism<F, Commitment>,
{
    let lowered = lower_sparse_r1cs_export_to_low_norm(export, layout)?;
    let program = lowered.to_direct_ccs_program()?;
    let step = lowered.clone().into_direct_ccs_step(&program, log, label)?;
    Ok((lowered, program, step))
}

fn validate_layout(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if export.variable_count != export.witness.len() {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS export variable count {} does not match witness len {}",
            export.variable_count,
            export.witness.len()
        )));
    }
    if layout.kinds.len() != export.variable_count {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS layout has {} lane kinds, expected {}",
            layout.kinds.len(),
            export.variable_count
        )));
    }
    if layout.public_input_len != export.public_input_len {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS layout public input len {} does not match export public input len {}",
            layout.public_input_len, export.public_input_len
        )));
    }
    if export.public_input_len == 0
        || export.witness.first().copied() != Some(F::ONE)
        || layout.kinds.first().copied() != Some(DirectLowNormLaneKind::Bit)
    {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct low-norm R1CS lowering requires public column 0 to be the Bellpepper one input".into(),
        ));
    }
    Ok(())
}
