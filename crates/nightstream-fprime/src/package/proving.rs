//! Direct Spartan lifecycle for the one loaded Lean-emitted relation.
//!
//! The loaded package owns the matrices and identity. This module only owns
//! setup, proof creation, and verification for that exact package.

use wip_spartan::{
    spartan::{RepeatedR1CSSNARK, SpartanProverKey, SpartanVerifierKey, R1CSSNARK},
    SplitR1CSShape,
};

use super::{
    canonical_field, expand_matrices, LoadedPackage, PackageError, ProofRun, SpartanEngine, SpartanField,
    WitnessAssignment,
};

/// Prover key for one verifier-owned package identity.
pub struct PackageProvingKey {
    relation_identifier: [u64; 4],
    key: SpartanProverKey<SpartanEngine>,
    matrix_stats: ProofRun,
}

impl PackageProvingKey {
    /// Sparse-matrix nonzero counts produced during setup.
    pub fn matrix_stats(&self) -> ProofRun {
        self.matrix_stats
    }
}

/// Verifier key for one verifier-owned package identity.
pub struct PackageVerifyingKey {
    relation_identifier: [u64; 4],
    key: SpartanVerifierKey<SpartanEngine>,
}

/// Direct Spartan proof bound to one Lean-emitted package identity.
pub struct PackageProof {
    relation_identifier: [u64; 4],
    proof: RepeatedR1CSSNARK<SpartanEngine>,
}

impl LoadedPackage {
    /// Build direct Spartan keys from the matrices in this loaded package.
    pub fn setup(&self) -> Result<(PackageProvingKey, PackageVerifyingKey), PackageError> {
        let (a, b, c) = expand_matrices(self)?;
        let matrix_stats = ProofRun {
            a_nonzeros: a.nnz(),
            b_nonzeros: b.nnz(),
            c_nonzeros: c.nnz(),
        };
        let shape = SplitR1CSShape::<SpartanEngine>::new(
            2,
            self.layout.row_count,
            0,
            0,
            self.layout.private_column_count,
            self.layout.public_column_count,
            0,
            a,
            b,
            c,
        )
        .map_err(|error| PackageError::Spartan(format!("shape: {error:?}")))?;
        let (prover, verifier) = R1CSSNARK::<SpartanEngine>::setup_direct(shape)
            .map_err(|error| PackageError::Spartan(format!("setup: {error:?}")))?;
        Ok((
            PackageProvingKey {
                relation_identifier: self.relation_identifier,
                key: prover,
                matrix_stats,
            },
            PackageVerifyingKey {
                relation_identifier: self.relation_identifier,
                key: verifier,
            },
        ))
    }

    /// Prove one assignment for this exact loaded package.
    pub fn prove(&self, key: &PackageProvingKey, assignment: &WitnessAssignment) -> Result<PackageProof, PackageError> {
        if key.relation_identifier != self.relation_identifier {
            return Err(PackageError::Invalid("proving key package identity"));
        }
        if assignment.private_values.len() != self.layout.private_column_count
            || assignment.public_values.len() != self.layout.public_column_count
        {
            return Err(PackageError::Invalid("proof assignment shape"));
        }
        let witness = assignment
            .private_values
            .iter()
            .map(|value| SpartanField::from_canonical_u64(*value))
            .collect::<Vec<_>>();
        let public = assignment
            .public_values
            .iter()
            .map(|value| SpartanField::from_canonical_u64(*value))
            .collect::<Vec<_>>();
        let proof = RepeatedR1CSSNARK::<SpartanEngine>::prove_direct(&key.key, &witness, &public, true)
            .map_err(|error| PackageError::Spartan(format!("prove: {error:?}")))?;
        Ok(PackageProof {
            relation_identifier: self.relation_identifier,
            proof,
        })
    }

    /// Verify one proof and its exact expected public values.
    pub fn verify(
        &self,
        key: &PackageVerifyingKey,
        proof: &PackageProof,
        expected_public: &[u64],
    ) -> Result<(), PackageError> {
        if key.relation_identifier != self.relation_identifier || proof.relation_identifier != self.relation_identifier
        {
            return Err(PackageError::Invalid("verification package identity"));
        }
        if expected_public.len() != self.layout.public_column_count {
            return Err(PackageError::Invalid("expected public input length"));
        }
        for value in expected_public {
            canonical_field(*value, "expected public input")?;
        }
        let expected = expected_public
            .iter()
            .map(|value| SpartanField::from_canonical_u64(*value))
            .collect::<Vec<_>>();
        let verified = proof
            .proof
            .verify(&key.key)
            .map_err(|error| PackageError::Spartan(format!("verify: {error:?}")))?;
        if verified != expected {
            return Err(PackageError::Invalid("verified public values"));
        }
        Ok(())
    }
}
