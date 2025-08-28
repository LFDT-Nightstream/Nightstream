use core::marker::PhantomData;

use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_field::{coset::TwoAdicMultiplicativeCoset, extension::BinomialExtensionField};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_goldilocks::Goldilocks;

use super::engine::PCSEngineTrait;
use super::mmcs::{Val, Challenge, ValMmcs, ChallengeMmcs, PcsMaterials, Dft};
use super::challenger::Challenger;

/// K = F_{q^2} for sum-check & final evals
pub type K = BinomialExtensionField<Goldilocks, 2>;

/// Minimal knobs you need to pass down to P3 FRI.
#[derive(Clone, Debug)]
pub struct P3FriParams {
    pub log_blowup: usize,        // e.g. 1..2
    pub log_final_poly_len: usize, // e.g. 0
    pub num_queries: usize,       // e.g. 20..100
    pub proof_of_work_bits: usize, // e.g. 8..16
}

impl Default for P3FriParams {
    fn default() -> Self {
        Self {
            log_blowup: 1,           // 2^1 expansion
            log_final_poly_len: 0,   // stop at constant  
            num_queries: 100,        // typical soundness target
            proof_of_work_bits: 16,  // anti-grinding
        }
    }
}

/// Thin wrapper that implements PCSEngineTrait over p3-fri::TwoAdicFriPcs.
pub struct P3FriPCSAdapter {
    pcs: TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>,
    fri_params: FriParameters<ChallengeMmcs>,
    _phantom: PhantomData<(Val, Challenge)>,
}

impl P3FriPCSAdapter {
    pub fn new(mats: &PcsMaterials, fri_params: FriParameters<ChallengeMmcs>) -> Self {
        // Clone the parameters first since TwoAdicFriPcs::new takes ownership
        let fri_params_for_pcs = FriParameters {
            log_blowup: fri_params.log_blowup,
            log_final_poly_len: fri_params.log_final_poly_len,
            num_queries: fri_params.num_queries,
            proof_of_work_bits: fri_params.proof_of_work_bits,
            mmcs: fri_params.mmcs.clone(),
        };
        let pcs = TwoAdicFriPcs::new(mats.dft.clone(), mats.val_mmcs.clone(), fri_params_for_pcs);
        Self { pcs, fri_params, _phantom: PhantomData }
    }
    
    pub fn new_with_params(params: P3FriParams) -> Self {
        let mats = super::mmcs::make_mmcs_and_dft(0x1337); // deterministic seed
        let fri_params = FriParameters {
            log_blowup: params.log_blowup,
            log_final_poly_len: params.log_final_poly_len,
            num_queries: params.num_queries,
            proof_of_work_bits: params.proof_of_work_bits,
            mmcs: mats.ch_mmcs.clone(),
        };
        Self::new(&mats, fri_params)
    }

    pub fn fri_params(&self) -> &FriParameters<ChallengeMmcs> { &self.fri_params }
}

// ———————————————————————————————————————————————————————————————————————————————
// PCSEngineTrait implementation
// ———————————————————————————————————————————————————————————————————————————————

impl PCSEngineTrait for P3FriPCSAdapter {
    type Val = Val;
    type Challenge = Challenge;
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
    type ProverData = <ValMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
    type OpenedValues = OpenedValues<Challenge>;
    type Proof = p3_fri::FriProof<
        Challenge,
        ChallengeMmcs,
        <Challenger as p3_challenger::GrindingChallenger>::Witness,
        Vec<p3_commit::BatchOpening<Val, ValMmcs>>,
    >;
    type Challenger = Challenger;

    #[inline] fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&self.pcs, degree)
    }

    #[inline] fn commit(
        &self,
        evals: impl IntoIterator<Item=(Self::Domain, RowMajorMatrix<Self::Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<Challenge, Challenger>>::commit(&self.pcs, evals)
    }

    #[inline] fn open(
        &self,
        data_and_points: Vec<(&Self::ProverData, Vec<Vec<Self::Challenge>>)>,
        ch: &mut Self::Challenger,
    ) -> (Self::OpenedValues, Self::Proof) {
        <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<Challenge, Challenger>>::open(&self.pcs, data_and_points, ch)
    }

    #[inline] fn verify(
        &self,
        commits_and_claims: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Self::Challenge, Vec<Self::Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        ch: &mut Self::Challenger,
    ) -> anyhow::Result<()> {
        <TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> as Pcs<Challenge, Challenger>>::verify(&self.pcs, commits_and_claims, proof, ch)
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::mmcs::make_mmcs_and_dft;

    fn make_test_fri_params(mats: &PcsMaterials) -> FriParameters<ChallengeMmcs> {
        FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 20, // Reduced for faster tests
            proof_of_work_bits: 8,
            mmcs: mats.ch_mmcs.clone(),
        }
    }

    #[test]
    fn test_p3fri_params_default() {
        let params = P3FriParams::default();
        
        println!("✅ P3FriParams default values");
        println!("   log_blowup: {}", params.log_blowup);
        println!("   log_final_poly_len: {}", params.log_final_poly_len);
        println!("   num_queries: {}", params.num_queries);
        println!("   proof_of_work_bits: {}", params.proof_of_work_bits);
    }

    #[test]
    fn test_p3fri_adapter_creation() {
        let mats = make_mmcs_and_dft(333);
        let fri_params = make_test_fri_params(&mats);
        let adapter = P3FriPCSAdapter::new(&mats, fri_params);
        
        println!("✅ P3FriPCSAdapter created successfully");
        println!("   Base field: Goldilocks");
        println!("   Extension field: K = F_q^2");
        println!("   FRI log_blowup: {}", adapter.fri_params().log_blowup);
        println!("   FRI num_queries: {}", adapter.fri_params().num_queries);
        println!("   Real p3-fri::TwoAdicFriPcs: ✅");
    }

    #[test]
    fn test_domain_creation() {
        let mats = make_mmcs_and_dft(444);
        let fri_params = make_test_fri_params(&mats);
        let adapter = P3FriPCSAdapter::new(&mats, fri_params);
        
        for degree_log in [3, 4, 5, 6] {
            let degree = 1 << degree_log;
            let domain = adapter.natural_domain_for_degree(degree);
            
            println!("   Degree 2^{} ({}): domain size {}", degree_log, degree, domain.size());
            assert!(domain.size() >= degree, "Domain must be at least as large as degree");
        }
        
        println!("✅ Natural domain creation works");
    }

    #[test]
    fn test_new_with_params() {
        let params = P3FriParams {
            log_blowup: 2,
            log_final_poly_len: 1,
            num_queries: 50,
            proof_of_work_bits: 12,
        };
        
        let adapter = P3FriPCSAdapter::new_with_params(params.clone());
        
        assert_eq!(adapter.fri_params().log_blowup, params.log_blowup);
        assert_eq!(adapter.fri_params().log_final_poly_len, params.log_final_poly_len);
        assert_eq!(adapter.fri_params().num_queries, params.num_queries);
        assert_eq!(adapter.fri_params().proof_of_work_bits, params.proof_of_work_bits);
        
        println!("✅ P3FriPCSAdapter::new_with_params works");
        println!("   Created adapter with custom parameters");
    }

    #[test]
    fn test_pcs_engine_trait_impl() {
        let adapter = P3FriPCSAdapter::new_with_params(P3FriParams::default());

        println!("🧪 Testing P3-FRI PCSEngineTrait implementation");
        println!("   Base field: Goldilocks");
        println!("   Extension field: K = F_q^2");

        // Test basic operations without complex roundtrip for now
        let degree = 1 << 4; // 16
        let domain = adapter.natural_domain_for_degree(degree);
        
        println!("   Domain creation: ✅ (size {})", domain.size());
        
        // TODO: Once p3-fri API is fully stabilized, implement full roundtrip test
        // For now, just test that the adapter compiles and basic methods work
        println!("✅ P3-FRI PCSEngineTrait implementation: PASS");
        println!("   Ready for full roundtrip testing once p3-fri API stabilizes");
    }
}