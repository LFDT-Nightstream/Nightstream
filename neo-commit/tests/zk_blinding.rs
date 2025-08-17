use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::F;
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_zk_blinding_different_commitments() {
    let params = TOY_PARAMS;
    assert!(params.sigma > 0.0 && params.beta > 0);
    let committer = AjtaiCommitter::setup_unchecked(params);
    let z: Vec<F> = vec![F::ZERO; params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    let mut rng = StdRng::seed_from_u64(42);
    let (c1, e1, blinded_w1, _) = committer.commit_with_rng(&w, &mut rng).unwrap();
    let (c2, e2, blinded_w2, _) = committer.commit_with_rng(&w, &mut rng).unwrap();

    assert_ne!(c1, c2);
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    // Ensure noise is non-zero and blinding increases witness norm
    assert!(e1.iter().any(|ei| ei.norm_inf() > 0));
    assert!(e2.iter().any(|ei| ei.norm_inf() > 0));
    let orig_norm = w.iter().map(|wi| wi.norm_inf()).max().unwrap();
    let blinded_norm = blinded_w1.iter().map(|wi| wi.norm_inf()).max().unwrap();
    assert!(blinded_norm > orig_norm);
}
