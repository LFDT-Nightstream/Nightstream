use p3_matrix::dense::RowMajorMatrix;

/// A minimal PCS engine interface the Spartan2 bridge calls into.
/// We keep it close to p3-commit::Pcs but isolate the types we need.
pub trait PCSEngineTrait {
    type Val;
    type Challenge;
    type Domain;
    type Commitment;
    type ProverData;
    type OpenedValues;
    type Proof;
    type Challenger;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    fn commit(
        &self,
        evals: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Self::Val>)>,
    ) -> (Self::Commitment, Self::ProverData);

    fn open(
        &self,
        data_and_points: Vec<(&Self::ProverData, Vec<Vec<Self::Challenge>>)>,
        ch: &mut Self::Challenger,
    ) -> (Self::OpenedValues, Self::Proof);

    fn verify(
        &self,
        commits_and_claims: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Self::Challenge, Vec<Self::Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        ch: &mut Self::Challenger,
    ) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time test that our trait is well-formed
    #[allow(dead_code)]
    fn test_trait_bounds<T: PCSEngineTrait>(_pcs: &T) {
        // This function existing and compiling proves the trait is well-formed
        println!("✅ PCSEngineTrait is well-formed");
    }
}
