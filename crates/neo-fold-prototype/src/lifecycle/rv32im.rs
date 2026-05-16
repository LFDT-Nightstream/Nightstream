//! RV32IM one-shot public proof lifecycle.

use crate::frontends::rv32im::{self, Rv32imProof, Rv32imProofInput, SimpleKernelError};

use super::OneShotProofSystem;

pub struct Rv32im;

impl OneShotProofSystem for Rv32im {
    type Input = Rv32imProofInput;
    type Proof = Rv32imProof;
    type Error = SimpleKernelError;

    fn prove(input: &Self::Input) -> Result<Self::Proof, Self::Error> {
        rv32im::prove_rv32im_public_proof(input)
    }

    fn verify(input: &Self::Input, proof: &Self::Proof) -> Result<(), Self::Error> {
        rv32im::audit::audit_rv32im_public_proof_against_input(input, proof)
    }
}

/// Prove the standard RV32IM public proof from a prepared frontend input.
pub fn prove_rv32im(input: &Rv32imProofInput) -> Result<Rv32imProof, SimpleKernelError> {
    Rv32im::prove(input)
}

/// Verify an RV32IM proof against the input it claims to prove.
pub fn verify_rv32im(input: &Rv32imProofInput, proof: &Rv32imProof) -> Result<(), SimpleKernelError> {
    Rv32im::verify(input, proof)
}
