//! Owns the first non-VM frontend adapter: sparse R1CS into Direct CCS program shape.
//!
//! This layer is intentionally thin. It converts an already-built R1CS shape
//! into a generic `DirectCcsProgram`; fresh step construction lives in the
//! Direct CCS step owner because it performs the pre-Pi_CCS
//! witness-to-claim boundary.

use neo_ccs::CcsMatrix;
use neo_math::F;
use neo_params::NeoParams;

use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram};

pub fn direct_ccs_program_from_sparse_r1cs(
    params: &NeoParams,
    a: CcsMatrix<F>,
    b: CcsMatrix<F>,
    c: CcsMatrix<F>,
) -> Result<DirectCcsProgram, DirectCcsFPrimeSnarkError> {
    let structure = direct_ccs_structure_from_sparse_r1cs(a, b, c)?;
    Ok(DirectCcsProgram::new(params, &structure))
}

pub fn direct_ccs_program_from_sparse_r1cs_with_public_input_len(
    params: &NeoParams,
    a: CcsMatrix<F>,
    b: CcsMatrix<F>,
    c: CcsMatrix<F>,
    public_input_len: usize,
) -> Result<DirectCcsProgram, DirectCcsFPrimeSnarkError> {
    let structure = direct_ccs_structure_from_sparse_r1cs(a, b, c)?;
    DirectCcsProgram::new_with_public_input_len(params, &structure, public_input_len)
}

fn direct_ccs_structure_from_sparse_r1cs(
    a: CcsMatrix<F>,
    b: CcsMatrix<F>,
    c: CcsMatrix<F>,
) -> Result<neo_ccs::CcsStructure<F>, DirectCcsFPrimeSnarkError> {
    let structure = neo_ccs::sparse_r1cs_to_ccs(a, b, c)
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("R1CS->CCS conversion failed: {err}")))?;
    Ok(structure)
}
