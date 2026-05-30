//! R1CS-F' frontend `preprocess` validates the plan's public-input
//! binding + semantic-state binding before any chain compile runs.
//!
//! These tests sit between encoder-time rejection (`r1cs_compiler.rs` /
//! `r1cs_compiler_stateful.rs`) and lifecycle-time rejection
//! (`lifecycle_finalization.rs`): they exercise the verifier-owned-plan
//! boundary where mis-configured chains are caught earliest.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::frontends::r1cs_f_prime;

use support::r1cs_compiler_fixtures::{expect_preprocess_err, make_small_plan, one_product_r1cs};

/// A plan with the wrong `app_public_input_var_indices` would silently
/// miss the public-input binding (the chain's `public_output_digest`
/// would not commit to `x`). `preprocess` must reject this at the
/// verifier-owned-plan boundary, not let a misconfigured chain reach
/// the encoder.
#[test]
fn r1cs_preprocess_rejects_app_public_input_var_indices_mismatch() {
    let r1cs = one_product_r1cs(); // m_in = 1
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    // Corrupt the plan: tell `preprocess` that no app variables are
    // public, even though `r1cs.m_in = 1`. A naive preprocess would
    // accept this and compile silently-wrong chains.
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![];

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C001);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPublicInputMismatch { actual, m_in }
                if actual.is_empty() && *m_in == 1
        ),
        "expected PlanAppPublicInputMismatch with actual=[] m_in=1, got {err:?}"
    );
}

/// Same gate, but the indices are present and span the right *count*
/// while pointing at the wrong variables (`[3]` instead of `[0]`).
/// `state_x_out` would then bind some private witness lane as if it
/// were public — equally fatal — and `preprocess` must reject.
#[test]
fn r1cs_preprocess_rejects_misnamed_public_input_indices() {
    let r1cs = one_product_r1cs(); // m_in = 1, m = neo_math::D
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![3]; // pointing at a private variable

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C002);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPublicInputMismatch { actual, m_in }
                if actual == &vec![3] && *m_in == 1
        ),
        "expected PlanAppPublicInputMismatch with actual=[3] m_in=1, got {err:?}"
    );
}

/// `preprocess` must also reject a plan with `state_x_out = None` —
/// without the state-x_out hash there is nowhere to absorb the
/// app-level public input at all.
#[test]
fn r1cs_preprocess_rejects_plan_without_state_x_out() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.state_x_out = None;

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C003);
    assert!(
        matches!(&err, r1cs_f_prime::Error::PlanMissingStateXOut),
        "expected PlanMissingStateXOut, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_rejects_partial_semantic_state_binding() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![1];
    sxo.semantic_state_out_var_indices = vec![];

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C004);
    assert!(
        matches!(&err, r1cs_f_prime::Error::PlanSemanticStatePartial),
        "expected PlanSemanticStatePartial, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_rejects_semantic_state_index_out_of_range() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![1];
    sxo.semantic_state_out_var_indices = vec![r1cs.m()];

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C005);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanSemanticStateIndexOutOfRange { index, m }
                if *index == r1cs.m() && *m == r1cs.m()
        ),
        "expected PlanSemanticStateIndexOutOfRange, got {err:?}"
    );
}

/// Stateful semantic-state binding (both in + out indices present, in
/// range) but NO verifier-owned `initial_semantic_state_digest_anchor`.
/// The base step's start state would then be prover-chosen rather than
/// pinned to a verifier value, so `preprocess` must reject. Guards the
/// `(has indices) ⟹ (has anchor)` half of the iff equivalence.
#[test]
fn r1cs_preprocess_rejects_stateful_indices_without_anchor() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![1];
    sxo.semantic_state_out_var_indices = vec![0];
    sxo.initial_semantic_state_digest_anchor = None;

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C006);
    assert!(
        matches!(&err, r1cs_f_prime::Error::PlanSemanticStateMissingAnchor),
        "expected PlanSemanticStateMissingAnchor, got {err:?}"
    );
}

/// An `initial_semantic_state_digest_anchor` with NO semantic-state
/// indices to bind it to — a meaningless anchor that desyncs the F'
/// image's base constraint from any claimed start state. Guards the
/// `(has anchor) ⟹ (has indices)` half of the iff equivalence.
#[test]
fn r1cs_preprocess_rejects_anchor_without_stateful_indices() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![];
    sxo.semantic_state_out_var_indices = vec![];
    sxo.initial_semantic_state_digest_anchor = Some([7u8; 32]);

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C007);
    assert!(
        matches!(&err, r1cs_f_prime::Error::PlanSemanticStateAnchorWithoutIndices),
        "expected PlanSemanticStateAnchorWithoutIndices, got {err:?}"
    );
}
