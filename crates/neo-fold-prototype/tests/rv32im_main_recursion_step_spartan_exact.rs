use neo_fold_prototype::rv32im::audit::{
    debug_check_rv32im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv32im_main_recursion_step_spartan_circuit,
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv32im_main_recursion_x_out_gadget_parity,
};
#[path = "support/rv32im_main_recursion_step_spartan_exact.rs"]
mod rv32im_main_recursion_step_spartan_exact_support;
use rv32im_main_recursion_step_spartan_exact_support::{
    assert_backend_relation_exact_surface_contract, single_relation_backend_fixture,
};

#[test]
fn rv32im_main_recursion_step_spartan_fixed_transcript_matches_native_state_out() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let relation = backend_relations.first().expect("first backend relation");

    assert_eq!(
        relation.payload.fixed_transcript_out(),
        &relation.f_prime_advice.fresh_state_out().transcript,
        "fixed recursive-step payload transcript drifted from the carried native state_out transcript"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_single_step_circuit_is_satisfied_with_exact_first_shape() {
    let (exact_shape, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_circuit(&exact_shape, first)
        .expect("single-step recursive-step circuit should synthesize cleanly under its exact first-step shape");
}

#[test]
fn rv32im_main_recursion_step_spartan_state_claims_share_recursive_cover_point() {
    let (exact_shape, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");

    let state_in = &first.payload.state_in_claims[..exact_shape.cover_shape.state_in_claim_count as usize];
    if let Some((head, tail)) = state_in.split_first() {
        for (idx, claim) in tail.iter().enumerate() {
            assert_eq!(claim.r, head.r, "state_in shared r drift at carried slot {}", idx + 1);
            assert_eq!(
                claim.s_col,
                head.s_col,
                "state_in shared s_col drift at carried slot {}",
                idx + 1
            );
        }
    }

    let state_out = &first.payload.state_out_claims[..exact_shape.cover_shape.state_out_claim_count as usize];
    if let Some((head, tail)) = state_out.split_first() {
        for (idx, claim) in tail.iter().enumerate() {
            assert_eq!(claim.r, head.r, "state_out shared r drift at carried slot {}", idx + 1);
            assert_eq!(
                claim.s_col,
                head.s_col,
                "state_out shared s_col drift at carried slot {}",
                idx + 1
            );
        }
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_exact_first_payload_matches_native_chunk_trace() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_chunk_step_recursive_effective_chunk_trace_matches_native(first)
        .expect("exact first-step payload should reconstruct the native chunk replay trace");
}

#[test]
fn rv32im_main_recursion_step_spartan_exact_surface_contract_holds() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    assert_backend_relation_exact_surface_contract(first, "exact first-step");
}

#[test]
fn rv32im_main_recursion_step_spartan_exact_live_claim_me_digest_parity_holds() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity(first)
        .expect("exact first-step live carried claims should hash to the authoritative native ME digests");
}

#[test]
fn rv32im_main_recursion_step_spartan_exact_x_out_gadget_parity_holds() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_x_out_gadget_parity(first)
        .expect("exact first-step x_out gadget should match the canonical native F' image");
}
