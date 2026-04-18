//! HyperNova Construction 2 §6.3 step 5 parity: for every carried step, the
//! public IO published by the RV64IM main-recursion Spartan circuit must be
//! bit-exact with the `x_out` and `U_{i+1}` digest produced by the native F'.
//!
//! This locks the per-step public surface at exactly two digests
//!   x_out (= hash(vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1}))
//!   folded_accumulator_out_digest (= digest of U_{i+1})
//! and rejects any future drift in either the native or circuit paths.
//!
//! Crucially, this test compares the native step image against the public
//! values extracted from an actual verified Spartan proof, not just against the
//! helper-built published target shell.

use neo_fold_next::rv64im::audit::{
    build_rv64im_main_recursion_step_spartan_published_target, evaluate_rv64im_main_recursion_f_prime_advice,
    prove_rv64im_main_recursion_step_spartan, setup_rv64im_main_recursion_step_spartan_cached,
    verify_rv64im_main_recursion_step_spartan_and_extract_published_target,
};

use super::support::{fast_structural_backend_relations, fast_structural_spartan_shape};

#[test]
fn f_prime_native_equals_circuit_bit_exact() {
    let spartan_shape = fast_structural_spartan_shape();
    let backend_relations = fast_structural_backend_relations();
    assert!(
        !backend_relations.is_empty(),
        "expected at least one recursive-step backend relation for bit-exact parity"
    );
    let keys = setup_rv64im_main_recursion_step_spartan_cached(spartan_shape, &backend_relations[0])
        .expect("setup recursive-step Spartan keys for proof-extracted parity");
    let (pk, vk) = &*keys;

    for (step, backend_relation) in backend_relations.iter().enumerate() {
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(&backend_relation.f_prime_advice)
            .unwrap_or_else(|err| panic!("step {step}: native F' advice must evaluate: {err}"));

        let canonical_target = build_rv64im_main_recursion_step_spartan_published_target(backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: build published target: {err}"));
        let proof = prove_rv64im_main_recursion_step_spartan(pk, spartan_shape, backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: prove recursive-step Spartan proof: {err}"));
        let extracted_target = verify_rv64im_main_recursion_step_spartan_and_extract_published_target(vk, &proof)
            .unwrap_or_else(|err| panic!("step {step}: verify and extract recursive-step public target: {err}"));

        assert_eq!(
            extracted_target,
            canonical_target,
            "step {step}: proof-extracted recursive-step public values drifted from the authoritative native published-target builder"
        );
        assert_eq!(
            extracted_target.x_out,
            *step_image.x_out(),
            "step {step}: HN Construction-2 §6.3 step 5 requires the proof-extracted Spartan recursive-step x_out to equal the native F' x_out bit-exactly"
        );
        assert_eq!(
            extracted_target.folded_accumulator_out_digest,
            step_image.folded_accumulator_digest(),
            "step {step}: proof-extracted Spartan recursive-step folded_accumulator_out_digest must equal the native F' folded U_{{i+1}} digest bit-exactly"
        );
    }
}
