//! The compressed proof at the light fold cap (SPEC.md section 6). The light
//! slot's lookup table grows with its depth, and the cap 64
//! (`Schedule.productionLightFoldCap`) gives the largest table. The test holds
//! the compression setup of 10 GB, so it is ignored, and
//! `scripts/ci_rust_tests.sh` runs it alone.

use neo_math::F;
use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::{Circuit, Engine, FinalProof, PrimeCharacteristicRing, State};
use std::time::Instant;

/// Step 1 makes no fold. Step `k` makes the light fold that leaves the light
/// slot at depth `k − 1` until that depth is the cap, so after step 65 the
/// light slot is at depth 64, and step 66 makes a refresh fold
/// (`Schedule.Admissible`).
const AT_CAP: u64 = 65;

/// The proofs of a run of 66 steps verify when compressed after steps 3, 64,
/// 65 and 66, and not for another state. Each one is under 200 KB.
#[test]
#[ignore = "Production-size compression, 30 GB; scripts/ci_rust_tests.sh runs it alone."]
fn hash_chain_compresses_at_the_light_fold_cap() {
    let circuit = Circuit::compile(poseidon2_hash_chain_v1().expect("hash chain")).expect("bounded-depth circuit");
    let verifier = circuit.verifier(Engine::Optimized, 109).expect("verifier");
    let key = verifier.compression_key().expect("compression key");
    let prover = circuit.prover(Engine::Optimized, 109).expect("prover");
    let setup = prover.compression_setup().expect("compression setup");

    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8].map(F::from_u64);
    let mut proof = prover.prove(initial, &message).expect("first step");
    let mut current = poseidon2_hash_chain_step(initial, message);
    for iteration in 2..=AT_CAP + 1 {
        proof = prover.extend(&proof, &message).expect("next step");
        current = poseidon2_hash_chain_step(current, message);
        if iteration != 3 && iteration < AT_CAP - 1 {
            continue;
        }
        let start = Instant::now();
        let compressed = prover.compress(&proof, &setup).expect("compressed proof");
        let elapsed = start.elapsed();
        let bytes = compressed.to_bytes().expect("proof bytes");
        eprintln!("step {iteration}: compress {elapsed:.2?}, {} bytes", bytes.len());
        assert!(bytes.len() < 200_000, "step {iteration}: {} bytes", bytes.len());

        let state = State::new(iteration, initial, current);
        let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
        verifier
            .verify_final(&state, &key, &decoded)
            .unwrap_or_else(|error| panic!("step {iteration}: {error:?}"));
        let mut other = current;
        other[0] += F::ONE;
        assert!(verifier
            .verify_final(&State::new(iteration, initial, other), &key, &decoded)
            .is_err());
    }
}
