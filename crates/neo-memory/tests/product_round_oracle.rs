use neo_memory::twist_oracle::ProductRoundOracle;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

#[test]
fn product_round_oracle_empty_factors_is_one() {
    let oracle = ProductRoundOracle::new(Vec::new(), 0);
    assert_eq!(oracle.sum_over_hypercube(), K::ONE);
}

#[test]
fn product_round_oracle_len1_factors_evals_at_product() {
    let f0 = vec![K::from_u64(3)];
    let f1 = vec![K::from_u64(5)];
    let mut oracle = ProductRoundOracle::new(vec![f0, f1], 0);
    assert_eq!(oracle.num_rounds(), 0);

    let evals = oracle.evals_at(&[K::ZERO, K::ONE]);
    assert_eq!(evals, vec![K::from_u64(15), K::from_u64(15)]);
    assert_eq!(oracle.sum_over_hypercube(), K::from_u64(15));
}

