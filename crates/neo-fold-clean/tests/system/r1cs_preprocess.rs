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
            r1cs_f_prime::Error::PlanAppPrivateBooleanWidthUnconstrained { index } if *index == 1
        ) || matches!(
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
            r1cs_f_prime::Error::PlanAppPrivateBooleanWidthUnconstrained { index } if *index == 3
        ) || matches!(
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
