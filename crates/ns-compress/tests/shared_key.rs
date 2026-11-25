use std::sync::{Arc, Barrier};

use neo_ajtai::nightstream_fprime_setup::{
    expanded_key_bytes, ExpandedKey, ELEMENT_OUTPUT_BYTES, PRODUCTION_VERIFIER_ROWS,
};
use ns_compress::ProverKey;

#[test]
fn concurrent_setup_expands_one_live_key() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let start = Barrier::new(workers);
        let setup = || {
            start.wait();
            ProverKey::shared(100).unwrap()
        };
        let before = expanded_key_bytes();
        let (first, second) = pool.install(|| rayon::join(setup, setup));
        assert!(Arc::ptr_eq(&first, &second));
        let blocks = first.verification_key().blocks() as u64;
        assert_eq!(
            expanded_key_bytes() - before,
            blocks * PRODUCTION_VERIFIER_ROWS * ELEMENT_OUTPUT_BYTES as u64
        );
        assert_eq!(ExpandedKey::shared().unwrap().blocks(), blocks);
    }
}
