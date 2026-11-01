use nightstream::nebula::{Circuit, Geometry, Machine, Verifier};
use nightstream::Engine;
use std::{fs, path::PathBuf};

#[test]
fn minimum_110_rejects_64_segment_fingerprint_loss() {
    let geometry = Geometry {
        r: 10,
        mu: 11,
        w_ts: 8,
        b_ops: 2,
        b_scan: 3072,
        n: 1,
        s_max: 64,
    };
    assert!(geometry.is_valid());
    let segment_bits = geometry.memory_security_bits();
    let chain_bits = segment_bits - (geometry.s_max as f64).log2();
    assert!(segment_bits > 110.0 && chain_bits < 110.0);
    let reference = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap();
    let circuit = Circuit::compile(&reference, geometry, Machine).unwrap();
    assert!(
        Verifier::new(&circuit, Engine::Optimized, 110).is_err(),
        "public verifier accepts 110 bits with a {segment_bits:.3}-bit one-segment fingerprint bound and only {chain_bits:.3} bits across 64 allowed segments"
    );
    assert!(Verifier::new(&circuit, Engine::Optimized, 109).is_ok());
}
