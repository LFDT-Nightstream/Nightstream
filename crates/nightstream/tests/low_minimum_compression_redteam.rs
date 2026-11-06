use nightstream::{application::poseidon2_hash_chain_v1, Circuit, Engine};

#[test]
fn a_publicly_derived_compression_key_must_have_a_valid_shrink_config() {
    let circuit = Circuit::compile(poseidon2_hash_chain_v1().unwrap()).unwrap();
    let verifier = circuit.verifier(Engine::Optimized, 1).unwrap();
    let key = verifier.compression_key().unwrap();
    let value = serde_json::to_value(&key).unwrap();
    let bits = value["layer_bits"].as_f64().unwrap();
    let shape = serde_json::from_value(value["shape"].clone()).unwrap();

    assert!(
        neo_spartan::shrink::Shrink::new(shape, bits).is_ok(),
        "the public verifier derives a compression key with no valid shrink configuration at {bits} bits"
    );
}
