//! Owns the first non-VM frontend adapter: sparse R1CS into direct CCS.
//!
//! This layer is intentionally thin. It converts an already-built R1CS shape
//! and witness into the generic `DirectCcsProgram`/`DirectCcsStep` objects; it
//! does not own Bellpepper, VM, RAM, or ROM semantics. Inputs must already be
//! representable in SuperNeo's low-norm witness layout; arbitrary field-valued
//! circuits need a frontend bit/limb encoding before they can be folded here.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsMatrix, CcsWitness};
use neo_math::F;
use neo_params::NeoParams;

use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
use crate::proof::StepInput;
use crate::witness_layout::encode_vector_for_full_width;

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

/// Builds a direct CCS step from a full witness that is already SuperNeo
/// low-norm packable.
pub fn direct_ccs_step_from_low_norm_full_witness<L>(
    program: &DirectCcsProgram,
    log: &L,
    label: impl Into<String>,
    witness: &[F],
    public_input_len: usize,
) -> Result<DirectCcsStep, DirectCcsFPrimeSnarkError>
where
    L: SModuleHomomorphism<F, Commitment>,
{
    let structure = program.structure();
    if witness.len() != structure.m {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct R1CS witness has len {}, expected CCS column count {}",
            witness.len(),
            structure.m
        )));
    }
    if public_input_len > witness.len() {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct R1CS public input len {public_input_len} exceeds witness len {}",
            witness.len()
        )));
    }
    let z_mat = encode_vector_for_full_width(program.params(), structure.m, witness).map_err(|err| {
        DirectCcsFPrimeSnarkError::Input(format!("direct R1CS witness is not SuperNeo low-norm packable: {err}"))
    })?;
    Ok(DirectCcsStep::new(StepInput {
        label: label.into(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x: witness[..public_input_len].to_vec(),
            m_in: public_input_len,
        },
        witness: CcsWitness {
            w: witness[public_input_len..].to_vec(),
            Z: z_mat,
        },
    }))
}
