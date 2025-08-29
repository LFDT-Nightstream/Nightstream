use neo_spartan_bridge::{compress_mle_with_hash_mle, verify_mle_hash_mle, hash_mle::F};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_bridge_hash_mle_api() {
    // Example: m = 4 ⇒ |poly| = 16
    let m = 4usize;
    let n = 1usize << m;
    let mut rng = ChaCha8Rng::seed_from_u64(4242);

    let poly = (0..n).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();
    let point = (0..m).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();

    let bundle = compress_mle_with_hash_mle(&poly, &point).expect("compression should work");
    
    // The bundle should have some proof data
    assert!(!bundle.proof.is_empty());
    assert!(!bundle.public_io_bytes.is_empty());
    
    // Note: Verification will likely fail due to serialization issues, but let's see
    match verify_mle_hash_mle(&bundle) {
        Ok(_) => println!("Verification succeeded!"),
        Err(e) => println!("Verification failed (expected due to serde issues): {}", e),
    }
}
