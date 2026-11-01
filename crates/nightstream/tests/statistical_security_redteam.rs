//! The public security minimum must cover the selected one-step proof bound.

use std::{fs, path::PathBuf};

use nightstream::{application::poseidon2_hash_chain_v1, Circuit, Engine, Error, Verifier};

#[test]
fn minimum_110_rejects_the_selected_profile() {
    let reference = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap();
    let circuit = Circuit::compile(&reference, poseidon2_hash_chain_v1().unwrap()).unwrap();

    // The current formal bound cannot certify the requested 110-bit minimum.
    // Even with zero adversarial oracle queries, the 74 test challenges alone
    // contribute 74 * 4589 / q^2
    // (`Lifecycle.RandomOracleKnowledge.statisticalError` and
    // `Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`).
    let q = 18_446_744_069_414_584_321_f64;
    let one_step_test_bits = 2.0 * q.log2() - (74.0_f64 * 4589.0).log2();
    assert!(one_step_test_bits < 110.0);
    assert_eq!(circuit.security_bits().unwrap(), 109);
    assert!(
        matches!(
            Verifier::from_package(&circuit, Engine::Optimized, 110),
            Err(Error::InsufficientSecurity {
                required: 110,
                available: 109
            })
        ),
        "the verifier accepts a 110-bit minimum, but the formal one-step test term has only {one_step_test_bits:.3} bits"
    );
    assert!(Verifier::from_package(&circuit, Engine::Optimized, 109).is_ok());
}
