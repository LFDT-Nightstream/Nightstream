use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{SeedableRng, rngs::SmallRng};

use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    storage::Buffer,
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        constraints::statement::Statement,
        parameters::WhirConfig,
        prover::Prover,
        verifier::Verifier,
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

#[test]
fn whir_end_to_end_with_mmap_buffers() {
    const DIGEST_ELEMS: usize = 8;

    let old_threshold = std::env::var_os("WHIR_P3_MMAP_THRESHOLD_BYTES");
    unsafe {
        std::env::set_var("WHIR_P3_MMAP_THRESHOLD_BYTES", "1");
    }

    // Ensure we restore the environment even if the test panics.
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.0 {
                    Some(v) => std::env::set_var("WHIR_P3_MMAP_THRESHOLD_BYTES", v),
                    None => std::env::remove_var("WHIR_P3_MMAP_THRESHOLD_BYTES"),
                }
            }
        }
    }
    let _restore = Restore(old_threshold);

    // `num_variables=12` with `ConstantFromSecondRound(4,4)` yields at least one non-final WHIR
    // round, exercising the folded-round commitment path.
    let num_variables = 12;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        univariate_skip: false,
    };

    let params =
        WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

    let polynomial = EvaluationsList::new(vec![F::ONE; 1 << num_variables]);

    let mut statement = Statement::<EF>::initialize(num_variables);
    let point = MultilinearPoint::new((0..num_variables).map(|i| EF::from_u64(i as u64)).collect());
    statement.add_unevaluated_constraint(point, &polynomial);

    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, DIGEST_ELEMS>(&params);
    domainsep.add_whir_proof::<_, _, _, DIGEST_ELEMS>(&params);

    let mut rng = SmallRng::seed_from_u64(1);
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    let committer = CommitmentWriter::new(&params);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit::<DIGEST_ELEMS>(&dft_committer, &mut prover_state, polynomial)
        .unwrap();

    // Validate that the committed matrix is disk-backed (mmap) under a low threshold.
    let mmcs = MerkleTreeMmcs::<
        <F as Field>::Packing,
        <F as Field>::Packing,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >::new(params.merkle_hash.clone(), params.merkle_compress.clone());
    let mats = mmcs.get_matrices(witness.prover_data.as_ref());
    assert_eq!(mats.len(), 1);
    assert!(
        matches!(&mats[0].values, Buffer::Mmap(_)),
        "expected mmap-backed commitment matrix storage"
    );

    let prover = Prover(&params);
    let dft_prover = EvalsDft::<F>::default();
    prover
        .prove::<DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement.clone(), witness)
        .unwrap();

    let checkpoint_prover: EF = prover_state.sample();

    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);
    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);
    let parsed_commitment = commitment_reader
        .parse_commitment::<DIGEST_ELEMS>(&mut verifier_state)
        .unwrap();
    verifier
        .verify::<DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();

    let checkpoint_verifier: EF = verifier_state.sample();
    assert_eq!(checkpoint_prover, checkpoint_verifier);
}
