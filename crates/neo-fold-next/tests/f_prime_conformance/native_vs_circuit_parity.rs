//! Goal 1 / Goal 2 parity only: for every carried step, the native F' image,
//! the recursive-step published target shell, the authoritative chunk replay
//! surface, and the in-circuit `x_out` gadget must agree bit-exactly.
//!
//! This deliberately stays off the full Spartan setup/prove/verify path so the
//! acceptance test measures only:
//!   1. native HyperNova Construction-2 F'
//!   2. the recursive-step circuit surface owned by Goal 2
//!
//! It does not exercise the Goal 3 decider or a real Spartan proof. Those
//! heavier proof-path checks belong in separate manual/nightly audits.

use neo_fold_next::rv64im::audit::{
    build_rv64im_main_recursion_step_authoritative_chunk_surface,
    build_rv64im_main_recursion_step_spartan_published_target,
    debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv64im_main_recursion_x_out_gadget_parity, evaluate_rv64im_main_recursion_f_prime_advice,
};

use super::support::fast_structural_backend_relations;

#[test]
fn f_prime_native_equals_circuit_bit_exact() {
    let backend_relations = fast_structural_backend_relations();
    assert!(
        !backend_relations.is_empty(),
        "expected at least one recursive-step backend relation for bit-exact parity"
    );

    for (step, backend_relation) in backend_relations.iter().enumerate() {
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(&backend_relation.f_prime_advice)
            .unwrap_or_else(|err| panic!("step {step}: native F' advice must evaluate: {err}"));

        debug_check_rv64im_main_recursion_x_out_gadget_parity(backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: x_out gadget parity must hold without proving: {err}"));
        debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native(backend_relation)
            .unwrap_or_else(|err| {
                panic!(
                    "step {step}: authoritative recursive-step chunk surface must match the native chunk replay theorem surface: {err}"
                )
            });

        let canonical_target = build_rv64im_main_recursion_step_spartan_published_target(backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: build published target: {err}"));
        let authoritative_surface = build_rv64im_main_recursion_step_authoritative_chunk_surface(backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: build authoritative chunk surface: {err}"));
        assert_eq!(
            canonical_target.x_out,
            *step_image.x_out(),
            "step {step}: Goal 1/2 parity requires the recursive-step published-target x_out to equal the native F' x_out bit-exactly"
        );
        assert_eq!(
            canonical_target.folded_accumulator_out_digest,
            step_image.folded_accumulator_digest(),
            "step {step}: Goal 1/2 parity requires the recursive-step published-target folded_accumulator_out_digest to equal the native F' folded U_{{i+1}} digest bit-exactly"
        );
        assert_eq!(
            authoritative_surface.folded_accumulator_digest,
            step_image.folded_accumulator_digest(),
            "step {step}: authoritative recursive-step chunk surface must carry the same folded accumulator digest as the native F' image"
        );
        assert_eq!(
            authoritative_surface.terminal_handle_digest,
            *step_image.z_next(),
            "step {step}: authoritative recursive-step chunk surface terminal handle must equal the native z_next carried out of F'"
        );
    }
}
