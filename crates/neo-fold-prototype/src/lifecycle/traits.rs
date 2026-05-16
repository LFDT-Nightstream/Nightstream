//! Shared proof lifecycle traits.

/// A frontend whose public proof is created in one call and verified directly.
pub trait OneShotProofSystem {
    type Input;
    type Proof;
    type Error;

    fn prove(input: &Self::Input) -> Result<Self::Proof, Self::Error>;

    fn verify(input: &Self::Input, proof: &Self::Proof) -> Result<(), Self::Error>;
}

/// A frontend whose native proof state can be extended one step at a time.
pub trait IncrementalProofSystem {
    type Preprocessing;
    type Step;
    type Proof;
    type Error;

    fn prove<Steps>(preprocessing: &Self::Preprocessing, steps: Steps) -> Result<Self::Proof, Self::Error>
    where
        Steps: IntoIterator<Item = Self::Step>;

    fn extend(
        preprocessing: &Self::Preprocessing,
        proof: Self::Proof,
        step: Self::Step,
    ) -> Result<Self::Proof, Self::Error>;

    fn verify(preprocessing: &Self::Preprocessing, proof: &Self::Proof) -> Result<(), Self::Error>;
}

/// An incremental frontend whose native proof state can be compressed by Spartan.
pub trait SpartanProofSystem: IncrementalProofSystem {
    type FinishedProof;
    type FinishedVerifierKey;
    type FinishedPublicImage;
    type FinishedProofBundle;

    fn finish_with_spartan(proof: &Self::Proof) -> Result<Self::FinishedProofBundle, Self::Error>;

    fn prove_and_finish_with_spartan<Steps>(
        preprocessing: &Self::Preprocessing,
        steps: Steps,
    ) -> Result<Self::FinishedProofBundle, Self::Error>
    where
        Steps: IntoIterator<Item = Self::Step>,
    {
        let proof = Self::prove(preprocessing, steps)?;
        Self::finish_with_spartan(&proof)
    }

    fn verify_finished_with_spartan(
        verifier_key: &Self::FinishedVerifierKey,
        expected_public_image: &Self::FinishedPublicImage,
        proof: &Self::FinishedProof,
    ) -> Result<(), Self::Error>;
}
