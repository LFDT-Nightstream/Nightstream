use nightstream::{application::poseidon2_hash_chain_v1, Circuit, Engine};

#[test]
fn minimum_110_rejects_the_selected_profile() {
    let circuit = Circuit::compile(poseidon2_hash_chain_v1().unwrap()).unwrap();

    // `Lifecycle.RandomOracleKnowledge.statisticalError` and
    // `Lifecycle.Nifs.VerifierErrorBudget.test_error_eq` give at least
    // 74 * 4589 / q^2 with zero adversarial oracle queries.
    let q = 18_446_744_069_414_584_321_f64;
    let one_fold_test_bits = 2.0 * q.log2() - (74.0_f64 * 4589.0).log2();
    assert!(one_fold_test_bits < 110.0);
    assert!(
        circuit.verifier(Engine::Optimized, 110).is_err(),
        "the verifier accepted a 110-bit minimum, but the documented one-fold test bound of {one_fold_test_bits:.3} bits does not certify 110 bits"
    );
}
