//! Verifier-owned Stage 1 package boundary for `Poseidon2HashChainV1`.
//!
//! Owns loading the one allowlisted Lean package and presenting its exact
//! matrix evaluator and witness program to a generic proof backend. It does
//! not accept a caller-selected relation, application, identity, or key.

use std::ops::Range;

use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, LoadedPerApplicationPackage, LogicalMatrixRow, PackageError,
    PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs, Stage1VerifierBinding, WitnessAssignment,
};

use crate::paper::relations::Structure;

pub use super::ivc::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    PiCcsV1_1PackageBridgeError, PiCcsV1_1ProofInputs,
};

/// The only verifier-owned Stage 1 relation authorized for production.
pub struct Poseidon2HashChainV1Package {
    package: LoadedPerApplicationPackage,
    structure: Structure,
    binding: Stage1VerifierBinding,
}

impl Poseidon2HashChainV1Package {
    /// Load canonical package bytes and recompute every package and key pin.
    pub fn load(bytes: &[u8]) -> Result<Self, PackageError> {
        let package = load_poseidon2_hash_chain_v1_package(bytes)?;
        let structure = package.ccs_structure_header()?;
        let binding = package.production_verifier_binding()?;
        Ok(Self {
            package,
            structure,
            binding,
        })
    }

    pub fn structural_identifier(&self) -> [u64; 4] {
        self.package.structural_identifier()
    }

    pub fn package_identity(&self) -> [u64; 4] {
        self.binding.package_identity()
    }

    pub fn verification_key_digest(&self) -> [u64; 4] {
        self.binding.verification_key_digest()
    }

    /// Matrix-content-free header paired with [`Self::visit_matrix_rows`].
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Visit exact Lean-authored logical matrix rows in ascending order.
    pub fn visit_matrix_rows(
        &self,
        rows: Range<usize>,
        visit: impl FnMut(usize, LogicalMatrixRow) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        self.package.visit_matrix_rows(rows, visit)
    }

    /// Execute the only witness program accepted by this production package.
    pub fn execute_step_witness(
        &self,
        pi_ccs: &PiCcsV1_1PackageInputs,
        pi_dec: &PiDecV1_1PackageInputs,
        application_witness: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        self.package
            .execute_stage1_v1_1_witness(pi_ccs, pi_dec, application_witness)
    }
}
