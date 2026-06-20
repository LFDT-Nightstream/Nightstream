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

use neo_ccs::matrix::Mat as NeoMat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::r1cs_f_prime;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use support::r1cs_compiler_fixtures::{expect_preprocess_err, make_small_plan, one_product_r1cs, tiny_params};

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
            r1cs_f_prime::Error::PlanAppPublicInputMismatch { actual, actual_bit, m_in }
                if actual.is_empty() && actual_bit.is_empty() && *m_in == 1
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
            r1cs_f_prime::Error::PlanAppPublicInputMismatch { actual, actual_bit, m_in }
                if actual == &vec![3] && actual_bit.is_empty() && *m_in == 1
        ),
        "expected PlanAppPublicInputMismatch with actual=[3] m_in=1, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_prepared_structure_matches_standard_preprocess() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let params = tiny_params();
    let seed = 0x71C5_C0DE;

    let standard =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, params.clone(), seed).expect("standard preprocess");
    let prepared = r1cs_f_prime::prepare_preprocessing_structure(&r1cs, &plan).expect("prepare structure");
    let prepared_structure_digest = *prepared.structure_digest();
    let from_prepared =
        r1cs_f_prime::preprocess_seeded_prepared_with_params(prepared, params, seed).expect("prepared preprocess");

    assert_eq!(
        standard.prep.structure_digest(),
        from_prepared.prep.structure_digest(),
        "prepared preprocessing must preserve the verifier-owned structure digest"
    );
    assert_eq!(
        *from_prepared.prep.structure_digest(),
        prepared_structure_digest,
        "preprocessing must use the digest from the prepared structure/cache pair"
    );
    assert_eq!(
        standard.prep.vk.digest(),
        from_prepared.prep.vk.digest(),
        "prepared preprocessing must derive the same vk_fs digest"
    );
    assert_eq!(
        standard.prep.semantic_state_mode(),
        from_prepared.prep.semantic_state_mode(),
        "prepared preprocessing must preserve semantic-state mode"
    );
    assert_eq!(
        standard.prep.initial_semantic_state_digest(),
        from_prepared.prep.initial_semantic_state_digest(),
        "prepared preprocessing must preserve the initial semantic anchor"
    );
    from_prepared
        .prep
        .validate_cached_structure()
        .expect("prepared preprocessing cache must validate");
}

#[test]
fn r1cs_preprocess_prepare_rejects_public_input_mismatch() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![3];

    let err = match r1cs_f_prime::prepare_preprocessing_structure(&r1cs, &plan) {
        Ok(_) => panic!("prepared structure must reject a mismatched public-input plan"),
        Err(err) => err,
    };
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPublicInputMismatch { actual, actual_bit, m_in }
                if actual == &vec![3] && actual_bit.is_empty() && *m_in == 1
        ),
        "expected PlanAppPublicInputMismatch with actual=[3] m_in=1, got {err:?}"
    );
}

/// Packed public-input mode is only sound when every packed variable is
/// a proven bit. The honest compiler rejects non-bit assignments, but
/// that is prover-side hygiene; the verifier-owned plan must reject an
/// R1CS shape whose rows do not prove the bitness relation.
#[test]
fn r1cs_preprocess_rejects_packed_public_input_without_boolean_row() {
    let m = neo_math::D;
    let a = NeoMat::zero(1, m, F::default());
    let b = NeoMat::zero(1, m, F::default());
    let c = NeoMat::zero(1, m, F::default());
    let r1cs = R1cs { a, b, c, m_in: 2 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[0] = 1;
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![];
    sxo.app_public_input_bit_var_indices = vec![0, 1];

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C008);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanPackedPublicInputBooleanUnconstrained { index } if *index == 1
        ) || matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPrivateWidthTooNarrow {
                index,
                width: 1,
                proven_width
            } if *index == 1 && *proven_width == POSEIDON2_GOLDILOCKS_BITS
        ),
        "expected fail-closed rejection for packing unconstrained variable 1 as one bit, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_rejects_packed_public_input_without_one_bit_slot() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    let c = NeoMat::zero(1, m, F::default());
    let r1cs = R1cs { a, b, c, m_in: 2 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![];
    sxo.app_public_input_bit_var_indices = vec![0, 1];

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C009);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanPackedPublicInputWidthNotOne { index, width }
                if *index == 0 && *width == POSEIDON2_GOLDILOCKS_BITS
        ),
        "expected PlanPackedPublicInputWidthNotOne for variable 0, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_accepts_packed_public_input_with_boolean_row_and_one_bit_slot() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    let c = NeoMat::zero(1, m, F::default());
    let r1cs = R1cs { a, b, c, m_in: 2 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[0] = 1;
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.app_public_input_var_indices = vec![];
    sxo.app_public_input_bit_var_indices = vec![0, 1];

    r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_C009)
        .expect("packed public bits may be used when the R1CS proves bitness and each packed slot is one bit");
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
fn r1cs_preprocess_rejects_one_bit_slot_without_boolean_row() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C004);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPrivateWidthTooNarrow {
                index,
                width: 1,
                proven_width
            } if *index == 1 && *proven_width == POSEIDON2_GOLDILOCKS_BITS
        ),
        "expected fail-closed rejection for storing unconstrained variable 1 in one bit, got {err:?}"
    );
}

/// The same unconstrained-variable-in-one-bit plan the fail-closed test
/// rejects is accepted when declared widths are range constraints:
/// `app_private_widths_are_range_constraints` opts out of the conservative
/// `PlanAppPrivateWidthTooNarrow` proof obligation.
#[test]
fn r1cs_preprocess_range_constraint_widths_bypass_too_narrow() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_C004)
        .expect("range-constraint widths bypass the PlanAppPrivateWidthTooNarrow check");
}

#[test]
fn r1cs_preprocess_accepts_one_bit_slot_copied_from_boolean_row() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(2, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(1, 1)] = F::ONE;
    let mut b = NeoMat::zero(2, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    b[(1, 0)] = F::ONE;
    let mut c = NeoMat::zero(2, m, F::default());
    c[(1, 2)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.app_private_var_widths[2] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_C005)
        .expect("copy of a Boolean-constrained variable may use a one-bit slot");
}

#[test]
fn r1cs_preprocess_accepts_one_bit_slot_for_product_of_boolean_vars() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(3, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(1, 2)] = F::ONE;
    a[(2, 1)] = F::ONE;
    let mut b = NeoMat::zero(3, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    b[(1, 0)] = F::ONE;
    b[(1, 2)] = F::ZERO - F::ONE;
    b[(2, 2)] = F::ONE;
    let mut c = NeoMat::zero(3, m, F::default());
    c[(2, 3)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.app_private_var_widths[2] = 1;
    plan.app_private_var_widths[3] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_C00A)
        .expect("product of Boolean-constrained variables is itself Boolean");
}

#[test]
fn r1cs_preprocess_accepts_one_bit_slots_for_boolean_affine_or_xor_outputs() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(5, m, F::default());
    let mut b = NeoMat::zero(5, m, F::default());
    let mut c = NeoMat::zero(5, m, F::default());

    // z1 and z2 are direct bits: z * (1 - z) = 0.
    a[(0, 1)] = F::ONE;
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    a[(1, 2)] = F::ONE;
    b[(1, 0)] = F::ONE;
    b[(1, 2)] = F::ZERO - F::ONE;

    // z3 = z1 XOR z2: (2*z1) * z2 = z1 + z2 - z3.
    a[(2, 1)] = F::from_u64(2);
    b[(2, 2)] = F::ONE;
    c[(2, 1)] = F::ONE;
    c[(2, 2)] = F::ONE;
    c[(2, 3)] = F::ZERO - F::ONE;

    // z4 = NOT z1: 1 * (1 - z1) = z4.
    a[(3, 0)] = F::ONE;
    b[(3, 0)] = F::ONE;
    b[(3, 1)] = F::ZERO - F::ONE;
    c[(3, 4)] = F::ONE;

    // z5 = z1 OR z2: z1 * z2 = z1 + z2 - z5.
    a[(4, 1)] = F::ONE;
    b[(4, 2)] = F::ONE;
    c[(4, 1)] = F::ONE;
    c[(4, 2)] = F::ONE;
    c[(4, 5)] = F::ZERO - F::ONE;

    let r1cs = R1cs { a, b, c, m_in: 1 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    for index in 1..=5 {
        plan.app_private_var_widths[index] = 1;
    }
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_C00C)
        .expect("NOT/OR/XOR outputs of proven Boolean variables may use one-bit slots");
}

#[test]
fn r1cs_preprocess_rejects_one_bit_slot_for_scaled_product_output() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(3, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(1, 2)] = F::ONE;
    a[(2, 1)] = F::ONE;
    let mut b = NeoMat::zero(3, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    b[(1, 0)] = F::ONE;
    b[(1, 2)] = F::ZERO - F::ONE;
    b[(2, 2)] = F::ONE;
    let mut c = NeoMat::zero(3, m, F::default());
    c[(2, 3)] = F::from_u64(2);
    let r1cs = R1cs { a, b, c, m_in: 1 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.app_private_var_widths[2] = 1;
    plan.app_private_var_widths[3] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C00B);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanAppPrivateWidthTooNarrow {
                index,
                width: 1,
                proven_width
            } if *index == 3 && *proven_width == POSEIDON2_GOLDILOCKS_BITS
        ),
        "expected fail-closed rejection for storing scaled product output in one bit, got {err:?}"
    );
}

#[test]
fn r1cs_preprocess_rejects_partial_semantic_state_binding() {
    let mut r1cs = one_product_r1cs();
    r1cs.m_in = 0;
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

#[test]
fn r1cs_preprocess_rejects_public_input_not_bound_by_explicit_semantic_state() {
    let mut r1cs = one_product_r1cs();
    r1cs.m_in = 4;
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![1];
    sxo.semantic_state_out_var_indices = vec![0];
    sxo.initial_semantic_state_digest_anchor = Some([0xA5; 32]);

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C008);
    assert!(
        matches!(
            &err,
            r1cs_f_prime::Error::PlanPublicInputNotSemanticBound { index } if *index == 2
        ),
        "expected PlanPublicInputNotSemanticBound for first unbound public variable, got {err:?}"
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

/// Semantic-state digests are carried as 32 bytes at the lifecycle
/// boundary but as four Goldilocks lanes inside the F' image and hash
/// chain. A noncanonical byte limb such as `p` aliases to field zero,
/// so accepting it would let two different public byte digests share
/// the same in-circuit statement.
#[test]
fn r1cs_preprocess_rejects_noncanonical_initial_semantic_anchor() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let sxo = plan.state_x_out.as_mut().expect("plan has state_x_out");
    sxo.semantic_state_in_var_indices = vec![1];
    sxo.semantic_state_out_var_indices = vec![0];
    let mut noncanonical_zero = [0u8; 32];
    noncanonical_zero[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    sxo.initial_semantic_state_digest_anchor = Some(noncanonical_zero);

    let err = expect_preprocess_err(&r1cs, &plan, 0x71C5_C009);
    assert!(
        err.to_string()
            .contains("noncanonical semantic-state digest"),
        "expected noncanonical semantic-state digest rejection, got {err:?}"
    );
}

/// An `initial_semantic_state_digest_anchor` with no semantic-state or
/// app-public binding to connect it to — a meaningless anchor that
/// desyncs the F' image's base constraint from any claimed start state.
#[test]
fn r1cs_preprocess_rejects_anchor_without_stateful_indices() {
    let mut r1cs = one_product_r1cs();
    r1cs.m_in = 0;
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

// ---------------------------------------------------------------------------
// Width-inference pins for the determining-row corner rule.
// ---------------------------------------------------------------------------

/// Builds rows over `m = 12` vars: v0 is the constant lane; v1..=v3 carry
/// explicit Boolean rows `v * (1 - v) = 0`. Extra rows are appended by each
/// test before width derivation.
fn boolean_seeded_rows(extra: usize) -> (NeoMat<F>, NeoMat<F>, NeoMat<F>) {
    let m = 12;
    let n = 3 + extra;
    let mut a = NeoMat::zero(n, m, F::default());
    let mut b = NeoMat::zero(n, m, F::default());
    let c = NeoMat::zero(n, m, F::default());
    for (row, var) in (1..=3usize).enumerate() {
        a[(row, var)] = F::ONE;
        b[(row, 0)] = F::ONE;
        b[(row, var)] = F::ZERO - F::ONE;
    }
    (a, b, c)
}

fn widths_of(r1cs: &R1cs) -> Vec<usize> {
    r1cs_f_prime::R1csShape::from(r1cs).conservative_app_private_var_widths()
}

/// bellpepper's ch/select gadget `(b - c) * a = ch - c` must prove the
/// output Boolean via plain corner enumeration.
#[test]
fn width_inference_corner_rule_proves_ch_output_boolean() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(1);
    a[(3, 2)] = F::ONE;
    a[(3, 3)] = F::ZERO - F::ONE;
    b[(3, 1)] = F::ONE;
    c[(3, 4)] = F::ONE;
    c[(3, 3)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(widths[4], 1, "ch output must be proven Boolean");
}

/// bellpepper's maj pair `b*c = bc`, `(2bc - b - c) * a = bc - maj` needs
/// the definition substitution: `bc` must be computed as `b*c`, not ranged
/// freely, for the output range to stay non-negative.
#[test]
fn width_inference_corner_rule_proves_maj_output_via_definition() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(2);
    // row 3: v2 * v3 = v5
    a[(3, 2)] = F::ONE;
    b[(3, 3)] = F::ONE;
    c[(3, 5)] = F::ONE;
    // row 4: (2*v5 - v2 - v3) * v1 = v5 - v6
    a[(4, 5)] = F::from_u64(2);
    a[(4, 2)] = F::ZERO - F::ONE;
    a[(4, 3)] = F::ZERO - F::ONE;
    b[(4, 1)] = F::ONE;
    c[(4, 5)] = F::ONE;
    c[(4, 6)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(
        widths[6], 1,
        "maj output must be proven Boolean via bc = b*c definition"
    );
}

/// Conservativeness pin: the same maj-shaped row whose helper variable has
/// a Boolean bound but NO determining row must stay at the full lane width.
/// Free enumeration of the helper admits corners (helper = 1, b = c = 0)
/// that drive the local range negative, and a negative range must never be
/// narrowed — doing so would break completeness for satisfying witnesses.
#[test]
fn width_inference_corner_rule_refuses_negative_local_range() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(1);
    // row 3: (2*v1 - v2 - v3) * v1 = v1 - v7, with v1 Boolean-bounded but
    // its only "definition" being the C-empty Boolean row (not usable).
    a[(3, 1)] = F::from_u64(2);
    a[(3, 2)] = F::ZERO - F::ONE;
    a[(3, 3)] = F::ZERO - F::ONE;
    b[(3, 1)] = F::ONE;
    c[(3, 1)] = F::ONE;
    c[(3, 7)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(
        widths[7], POSEIDON2_GOLDILOCKS_BITS,
        "negative local range must not be narrowed"
    );
}

/// Exact-integer-enumeration pin: `t = (3 - a) * a` with `a = b1 + 2*b2`
/// computed from its definition has range {0, 2, 2, 0} over the integer
/// points — max 2, width 2. Endpoint-only enumeration would claim width 1
/// (incomplete: honest `a = 1` gives `t = 2`); naive interval arithmetic
/// would claim width 4 (max 9). Both regressions fail this pin.
#[test]
fn width_inference_interior_extremum_uses_exact_integer_range() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(2);
    // row 3: (v1 + 2*v2) * v0 = v8   — defines a := v8, bound 3
    a[(3, 1)] = F::ONE;
    a[(3, 2)] = F::from_u64(2);
    b[(3, 0)] = F::ONE;
    c[(3, 8)] = F::ONE;
    // row 4: (3*v0 - v8) * v8 = v9   — t := v9 = (3 - a) * a
    a[(4, 0)] = F::from_u64(3);
    a[(4, 8)] = F::ZERO - F::ONE;
    b[(4, 8)] = F::ONE;
    c[(4, 9)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(widths[8], 2, "a = b1 + 2*b2 must prove width 2");
    assert_eq!(widths[9], 2, "(3 - a) * a peaks at 2 on interior integer points");
}

/// Termination + conservativeness pin for cyclic definitions: `v5` and
/// `v6` determine each other; chasing either definition must hit the
/// cycle guard, bail, and leave every involved variable at full width.
#[test]
fn width_inference_definition_cycle_terminates_and_stays_wide() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(3);
    // row 3: v6 * v0 = v5      row 4: v5 * v0 = v6   (cyclic definitions)
    a[(3, 6)] = F::ONE;
    b[(3, 0)] = F::ONE;
    c[(3, 5)] = F::ONE;
    a[(4, 5)] = F::ONE;
    b[(4, 0)] = F::ONE;
    c[(4, 6)] = F::ONE;
    // row 5: v5 * v1 = v7      (target whose support is the cycle)
    a[(5, 5)] = F::ONE;
    b[(5, 1)] = F::ONE;
    c[(5, 7)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(widths[5], POSEIDON2_GOLDILOCKS_BITS);
    assert_eq!(widths[6], POSEIDON2_GOLDILOCKS_BITS);
    assert_eq!(widths[7], POSEIDON2_GOLDILOCKS_BITS);
}

/// A target carried with coefficient 2 is not a ±1 solo output; the row
/// determines `2t`, not `t`, and must not be narrowed.
#[test]
fn width_inference_non_unit_target_coefficient_stays_wide() {
    let (mut a, mut b, mut c) = boolean_seeded_rows(1);
    a[(3, 1)] = F::ONE;
    b[(3, 2)] = F::ONE;
    c[(3, 8)] = F::from_u64(2);
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(widths[8], POSEIDON2_GOLDILOCKS_BITS);
}

/// Combination-cap pin: a mux whose data operand is a 12-bit recomposed
/// word would need 2^14 support assignments — over the cap. The rule must
/// refuse (stay wide) and return quickly instead of blowing up.
#[test]
fn width_inference_wide_support_respects_combination_cap() {
    let m = 24;
    let n_bool = 14; // v1..=v14 Boolean
    let n = n_bool + 2;
    let mut a = NeoMat::zero(n, m, F::default());
    let mut b = NeoMat::zero(n, m, F::default());
    let mut c = NeoMat::zero(n, m, F::default());
    for (row, var) in (1..=n_bool).enumerate() {
        a[(row, var)] = F::ONE;
        b[(row, 0)] = F::ONE;
        b[(row, var)] = F::ZERO - F::ONE;
    }
    // row 14: (sum 2^i * v(i+1) for 12 bits) * v0 = v20  — 12-bit word
    for bit in 0..12 {
        a[(n_bool, bit + 1)] = F::from_u64(1 << bit);
    }
    b[(n_bool, 0)] = F::ONE;
    c[(n_bool, 20)] = F::ONE;
    // row 15: (v13 - v14) * v20 = v21 - v14  — mux selected by the word
    a[(n_bool + 1, 13)] = F::ONE;
    a[(n_bool + 1, 14)] = F::ZERO - F::ONE;
    b[(n_bool + 1, 20)] = F::ONE;
    c[(n_bool + 1, 21)] = F::ONE;
    c[(n_bool + 1, 14)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let widths = widths_of(&r1cs);
    assert_eq!(widths[20], 12, "12-bit word must prove width 12 via the affine rule");
    assert_eq!(
        widths[21], POSEIDON2_GOLDILOCKS_BITS,
        "support beyond the combination cap must refuse to narrow"
    );
}
