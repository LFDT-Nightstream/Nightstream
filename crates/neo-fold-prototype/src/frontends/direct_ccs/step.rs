//! Fresh Direct CCS step construction.
//!
//! This file owns the pre-Pi_CCS boundary for already-low-norm Direct CCS
//! witnesses: validate the raw witness shape, embed it into the SuperNeo
//! committed-object layout, derive the public projection, commit the embedded
//! witness, and build the fresh `CcsClaim`/`CcsWitness` consumed by folding.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsWitness, Mat};
use neo_math::F;

use super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram};
use crate::proof::StepInput;
use crate::witness_layout::encode_vector_for_full_width;

#[derive(Clone, Debug)]
pub struct DirectCcsStep {
    step: StepInput,
}

impl DirectCcsStep {
    pub fn new(step: StepInput) -> Self {
        Self { step }
    }

    pub fn into_step_input(self) -> StepInput {
        self.step
    }
}

/// Builds a fresh Direct CCS step from a full witness that is already SuperNeo
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
    validate_direct_ccs_step_witness(program, witness, public_input_len)?;
    let embedded_witness = embed_direct_ccs_witness(program, witness)?;
    let public_input = derive_public_input_projection(witness, public_input_len);
    let commitment = commit_embedded_witness(log, &embedded_witness);
    Ok(build_ccs_claim_and_witness(
        label,
        commitment,
        public_input,
        witness,
        public_input_len,
        embedded_witness,
    ))
}

fn validate_direct_ccs_step_witness(
    program: &DirectCcsProgram,
    witness: &[F],
    public_input_len: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
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
    Ok(())
}

fn embed_direct_ccs_witness(program: &DirectCcsProgram, witness: &[F]) -> Result<Mat<F>, DirectCcsFPrimeSnarkError> {
    encode_vector_for_full_width(program.params(), program.structure().m, witness).map_err(|err| {
        DirectCcsFPrimeSnarkError::Input(format!("direct R1CS witness is not SuperNeo low-norm packable: {err}"))
    })
}

fn derive_public_input_projection(witness: &[F], public_input_len: usize) -> Vec<F> {
    witness[..public_input_len].to_vec()
}

fn commit_embedded_witness<L>(log: &L, embedded_witness: &Mat<F>) -> Commitment
where
    L: SModuleHomomorphism<F, Commitment>,
{
    log.commit(embedded_witness)
}

fn build_ccs_claim_and_witness(
    label: impl Into<String>,
    commitment: Commitment,
    public_input: Vec<F>,
    witness: &[F],
    public_input_len: usize,
    embedded_witness: Mat<F>,
) -> DirectCcsStep {
    DirectCcsStep::new(StepInput {
        label: label.into(),
        mcs: CcsClaim {
            c: commitment,
            x: public_input,
            m_in: public_input_len,
        },
        witness: CcsWitness {
            w: witness[public_input_len..].to_vec(),
            Z: embedded_witness,
        },
    })
}
