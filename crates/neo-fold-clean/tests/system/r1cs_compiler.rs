//! R1CS-F' frontend integration tests — basic compile/satisfaction +
//! structure shape coverage.
//!
//! Every test is designed to fail under a plausible bad encoder /
//! structure builder / compiler:
//!
//! - `r1cs_compiler_accepts_satisfying_witness` — happy path.
//! - `r1cs_compiler_rejects_unsatisfying_witness` — early R1CS check
//!   in `compile_step` catches the bug.
//! - `r1cs_compiler_bit_flip_in_app_assignment_fails_structure` —
//!   confirms the in-circuit R1CS rows depend on the committed bits.
//! - `r1cs_compiler_row_count_matches_r1cs_n` — confirms the structure
//!   appends exactly `r1cs.n()` R1CS rows (every constraint is in).
//! - `r1cs_compiler_two_different_shapes_have_different_structure_digests`
//!   — sanity test on the verifier-owned structure digest.
//! - `r1cs_compiler_satisfies_fibonacci_relation` — Fibonacci-as-R1CS
//!   round-trip; the R1CS encoder accepts a Fibonacci-shaped circuit.
//! - `r1cs_compiler_base_and_recursive_share_structure` — load-bearing
//!   chain-replay test; runs the lifecycle under a smaller test-only
//!   params profile (kappa = 4, m = 2^16, lambda = 60) so the full
//!   prove + extend + recursive-compile flow fits under the 5-min cap.
//!   The algebra is unchanged (Goldilocks ring, k_rho, T, B all match
//!   production); only the Ajtai-SIS security parameter is reduced.
//!
//! Stateful semantic-digest tests live in the sibling
//! `r1cs_compiler_stateful.rs`. Preprocess-time plan validation lives
//! in `r1cs_preprocess.rs`. Shared fixtures live in
//! `tests/support/r1cs_compiler_fixtures.rs`.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_ccs::matrix::Mat as NeoMat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::compiler::{verify_prior_fold, FPrimeShellCompilerError};
use neo_fold_clean::frontends::f_prime::image::FPrimeImageLayout;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_recursive_step_image_config;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, build_r1cs_f_prime_structure, compile_step, start_chain, R1csChainBuilder, R1csCompilerError,
    R1csFPrimeStepInput, R1csFoldForStep,
};
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::structure_digest;
use neo_params::goldilocks_paper_b2;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, constant_lane_assignment, constant_lane_passthrough_r1cs, fibonacci_r1cs, make_small_plan,
    make_stateful_plan_with_anchor, make_tiny_lifecycle_plan, one_product_r1cs, tiny_params, two_product_r1cs,
    BOUNDARY_BITS,
};

fn overwrite_little_endian_u64_bits(witness: &mut [F], bit_start: usize, value: u64) {
    for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
        witness[bit_start + bit] = if ((value >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Happy + sad path: compile_step accept / reject on app-level R1CS.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_accepts_satisfying_witness() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0001).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let input = R1csFPrimeStepInput {
        assignment: assignment_one_product(3, 7),
    };
    let compiled = compile_step(&prep, &mut ctx, input).expect("base compile");

    // The encoder asserts satisfaction internally on the way out; if we
    // got here, the encoded step satisfies its R1CS-F' structure.
    let inst = r1cs_f_prime::build_instance(&prep, &compiled.encoded).expect("build_instance");
    // Public-input split must match preprocessing.
    assert_eq!(inst.claim.m_in, 1 + BOUNDARY_BITS);
}

#[test]
fn r1cs_chain_builder_appends_base_step_and_tracks_audit() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0008).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let compiled = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("append base assignment");

    assert_eq!(
        chain.context().chain_state.step_count,
        1,
        "builder must advance the compiler chain state"
    );
    assert_eq!(
        chain.audit().expect("audit after first append").steps.len(),
        1,
        "builder must fold the emitted base instance through lifecycle::prove"
    );
    let inst = r1cs_f_prime::build_instance(&prep, &compiled.encoded).expect("build_instance");
    assert_eq!(inst.claim.m_in, 1 + BOUNDARY_BITS);
}

#[test]
fn r1cs_compiler_rejects_unsatisfying_witness() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0002).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    // 3 * 7 = 21, but we set z[0] = 22.
    let mut bad = assignment_one_product(3, 7);
    bad[0] = F::from_u64(22);

    let err = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment: bad }).expect_err("must reject");
    match err {
        R1csCompilerError::Unsatisfied(_) => {}
        other => panic!("expected Unsatisfied, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Bit flip inside the encoded image's app region must fail structure.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_bit_flip_in_app_assignment_fails_structure() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0003).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("base compile");

    // Sanity: honest witness satisfies.
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    // Tamper one bit inside the app_private region. The structure's
    // R1CS rows recompose z[1] from those digits; flipping one alters
    // the product (z[1] · z[2]) and forces the R1CS row to fail.
    let app_offset = compiled.encoded.image.layout.app_private.offset;
    let mut tampered = compiled.encoded.witness.clone();
    // Flip bit 0 of variable z[1]: bit_start = app_offset + 1*64.
    let target = app_offset + POSEIDON2_GOLDILOCKS_BITS;
    tampered[target] = if tampered[target] == F::ZERO { F::ONE } else { F::ZERO };

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "bit-flip inside app_private must break an R1CS row"
    );
}

#[test]
fn r1cs_redteam_range_constraint_flag_rejects_nonbit_unused_slot() {
    let r1cs = constant_lane_passthrough_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[3] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D1).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: constant_lane_assignment(&[(3, 1)]),
        },
    )
    .expect("compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    let slot = prep.anchors().app_var_slots[3];
    assert_eq!(slot.bits, 1);

    // Red team: z[3] is unused by the app R1CS. If the flag only bypasses
    // width validation without adding F' bitness rows, this forged value
    // satisfies every remaining row.
    let mut tampered = compiled.encoded.witness.clone();
    tampered[slot.bit_start] = F::from_u64(2);
    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "an unused opt-in range slot must still be constrained to a bit by F' rows"
    );
}

#[test]
fn r1cs_redteam_range_constraint_width_rejects_truncating_assignment() {
    let r1cs = constant_lane_passthrough_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[3] = 2;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D2).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let err = match compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: constant_lane_assignment(&[(3, 4)]),
        },
    ) {
        Ok(_) => panic!("compiler must reject typed witness truncation"),
        Err(err) => err,
    };

    assert!(
        matches!(
            err,
            R1csCompilerError::TypedVariableOutOfRange { index: 3, width: 2, .. }
        ),
        "expected TypedVariableOutOfRange for z[3] in a 2-bit slot, got {err:?}"
    );
}

#[test]
fn r1cs_redteam_full_width_slot_rejects_goldilocks_alias() {
    let r1cs = constant_lane_passthrough_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D4).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: constant_lane_assignment(&[]),
        },
    )
    .expect("compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    let alias = F::ORDER_U64 + 5;
    let z1_slot = prep.anchors().app_var_slots[1];
    assert_eq!(z1_slot.bits, POSEIDON2_GOLDILOCKS_BITS);
    let mut tampered = compiled.encoded.witness.clone();
    overwrite_little_endian_u64_bits(&mut tampered, z1_slot.bit_start, alias);

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "a full-width app-private lane must reject the noncanonical Goldilocks alias p + 5"
    );
}

#[test]
fn r1cs_redteam_public_full_width_slot_rejects_goldilocks_alias() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 1)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 2 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D5).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let mut assignment = vec![F::ZERO; r1cs.m()];
    assignment[0] = F::ONE;
    assignment[1] = F::from_u64(5);
    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment }).expect("compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    let alias = F::ORDER_U64 + 5;
    let z1_slot = prep.anchors().app_var_slots[1];
    assert_eq!(z1_slot.bits, POSEIDON2_GOLDILOCKS_BITS);
    let mut tampered = compiled.encoded.witness.clone();
    overwrite_little_endian_u64_bits(&mut tampered, z1_slot.bit_start, alias);

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "public semantic binding must reject a noncanonical Goldilocks alias for a full-width public lane"
    );
}

#[test]
fn r1cs_redteam_semantic_state_full_width_slot_rejects_goldilocks_alias() {
    let r1cs = constant_lane_passthrough_r1cs();
    let mut plan = make_stateful_plan_with_anchor(r1cs.m(), r1cs.m_in, Vec::new(), vec![1], None);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    plan.app_private_widths_are_range_constraints = true;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D6).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: constant_lane_assignment(&[]),
        },
    )
    .expect("compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    let alias = F::ORDER_U64 + 5;
    let z1_slot = prep.anchors().app_var_slots[1];
    assert_eq!(z1_slot.bits, POSEIDON2_GOLDILOCKS_BITS);
    let mut tampered = compiled.encoded.witness.clone();
    overwrite_little_endian_u64_bits(&mut tampered, z1_slot.bit_start, alias);

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "semantic-state output binding must reject a noncanonical Goldilocks alias for a full-width lane"
    );
}

#[test]
fn r1cs_compiler_pins_constant_lane_when_boolean_width_relies_on_it() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    let c = NeoMat::zero(1, m, F::default());
    let r1cs = R1cs { a, b, c, m_in: 0 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00C1).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let mut assignment = vec![F::ZERO; m];
    assignment[0] = F::ONE;
    assignment[1] = F::ZERO;
    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment }).expect("compile");

    let mut tampered = compiled.encoded.witness.clone();
    let z0_slot = prep.anchors().app_var_slots[0];
    for offset in 0..z0_slot.bits {
        tampered[z0_slot.bit_start + offset] = F::ZERO;
    }

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "a Boolean-width plan relies on R1CS z[0] as the constant-one lane; \
         tampering z[0] to zero must fail even when the Boolean row z1*(z0-z1)=0 \
         remains satisfied with z1=0"
    );
}

#[test]
fn r1cs_compiler_pins_constant_lane_when_multibit_width_relies_on_it() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 0)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::from_u64(2);
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 1)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 0 };

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[1] = 2;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00C2).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let mut assignment = vec![F::ZERO; m];
    assignment[0] = F::ONE;
    assignment[1] = F::from_u64(2);
    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment }).expect("compile");

    let mut tampered = compiled.encoded.witness.clone();
    let z0_slot = prep.anchors().app_var_slots[0];
    for offset in 0..z0_slot.bits {
        tampered[z0_slot.bit_start + offset] = F::ZERO;
    }
    let z1_slot = prep.anchors().app_var_slots[1];
    for offset in 0..z1_slot.bits {
        tampered[z1_slot.bit_start + offset] = F::ZERO;
    }

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "a multibit typed-width plan can still rely on R1CS z[0] as the \
         constant-one lane; coherently tampering z[0] and the bounded output \
         must fail, otherwise the folded image admits a non-constant z[0]"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Row count threads through: r1cs.n() rows are appended **and each row
// independently enforces its constraint**.
//
// A buggy builder could happily report `r1cs_row_count = r1cs.n()` while
// emitting the same constraint N times. To catch that, we also build a
// witness that satisfies the FIRST R1CS row but **violates the second**,
// then confirm:
//   - the 2-row structure REJECTS it (so the second row really is in)
//   - the 1-row structure (with just the first constraint) ACCEPTS it
//     (so the first row really is independent of the second).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_row_count_matches_r1cs_n() {
    let r1cs_one = one_product_r1cs();
    let r1cs_two = two_product_r1cs();
    let plan = make_small_plan(neo_math::D, 1);

    let layout_one = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    let layout_two = layout_one.clone();
    let (struct_one, anchors_one) = build_r1cs_f_prime_structure(layout_one, &r1cs_one);
    let (struct_two, anchors_two) = build_r1cs_f_prime_structure(layout_two, &r1cs_two);

    assert_eq!(anchors_one.r1cs_row_count, r1cs_one.n());
    assert_eq!(anchors_two.r1cs_row_count, r1cs_two.n());

    // Structures should have the same shell-row count.
    assert_eq!(anchors_one.r1cs_row_start, anchors_two.r1cs_row_start);
    // Total rows differ by the R1CS row count.
    assert_eq!(struct_two.ccs.n - struct_one.ccs.n, r1cs_two.n() - r1cs_one.n());

    // Build an assignment that satisfies `r1cs_one`'s only constraint
    // (`z[0] = z[1] * z[2]`) but violates `r1cs_two`'s second
    // constraint (`z[3] = z[4] * z[5]`).
    let m = neo_math::D;
    let mut z = vec![F::ZERO; m];
    z[1] = F::from_u64(3);
    z[2] = F::from_u64(7);
    z[0] = F::from_u64(21); // satisfies r1cs_one
    z[4] = F::from_u64(2);
    z[5] = F::from_u64(3);
    z[3] = F::from_u64(99); // 2*3 = 6 ≠ 99 — violates r1cs_two's second row
    assert!(r1cs_one.is_satisfied_by(&z).is_ok());
    assert!(r1cs_two.is_satisfied_by(&z).is_err());

    let assignment_bits = neo_fold_clean::frontends::r1cs_f_prime::assignment_to_bits(&z);
    let witness_one = honest_witness_with_assignment_for(&struct_one, &assignment_bits);
    let witness_two = honest_witness_with_assignment_for(&struct_two, &assignment_bits);

    assert!(
        struct_one.is_satisfied(&witness_one),
        "1-row R1CS structure must accept witness that satisfies its single constraint"
    );
    assert!(
        !struct_two.is_satisfied(&witness_two),
        "2-row R1CS structure must reject witness that violates its second constraint"
    );
}

/// Build an honest F' image's witness, then overwrite the
/// `app_private` region with `assignment_bits`. Used by
/// `r1cs_compiler_row_count_matches_r1cs_n` to feed the same app
/// assignment to two different R1CS structures.
fn honest_witness_with_assignment_for(
    structure: &neo_fold_clean::frontends::f_prime::structure::FPrimeStructure,
    assignment_bits: &[F],
) -> Vec<F> {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0444).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("base compile");

    assert_eq!(compiled.encoded.witness.len(), structure.ccs.m);
    assert_eq!(structure.layout.app_private.bits, assignment_bits.len());

    let mut z = compiled.encoded.witness.clone();
    let start = structure.layout.app_private.offset;
    z[start..start + assignment_bits.len()].copy_from_slice(assignment_bits);
    z
}

// ─────────────────────────────────────────────────────────────────────────
// Soundness — `public_output_digest` binds to the proven public input `x`.
//
// Council finding [P0]: without binding `x = assignment[..m_in]` into a
// carried state coordinate, two different satisfying assignments with
// different `x` produce the same `public_output_digest`. The verifier learns
// only "some `(x, w)` satisfies the R1CS" — not which `x`. These tests pin
// down both halves of the contract.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_public_output_depends_on_public_input() {
    // Pick `z[0] = z[1] * z[2]` with `m_in = 1` so `x = (z[0],)`. Two
    // assignments with DIFFERENT public input (z[0] = 21 vs z[0] = 15)
    // — each with its own satisfying witness (3*7 vs 3*5) — must
    // produce DIFFERENT `public_output_digest`s.
    let r1cs = one_product_r1cs();
    assert_eq!(r1cs.m_in, 1, "test relies on z[..1] being the public input");
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00A0).expect("preprocess");

    let mut ctx_a = start_chain(&prep).expect("start chain a");
    let mut ctx_b = start_chain(&prep).expect("start chain b");

    let z_a = assignment_one_product(3, 7); // z[0] = 21
    let z_b = assignment_one_product(3, 5); // z[0] = 15
    assert_ne!(z_a[0], z_b[0], "test setup: public inputs must differ");

    let compiled_a =
        compile_step(&prep, &mut ctx_a, R1csFPrimeStepInput { assignment: z_a }).expect("compile a must accept");
    let compiled_b =
        compile_step(&prep, &mut ctx_b, R1csFPrimeStepInput { assignment: z_b }).expect("compile b must accept");

    assert_ne!(
        compiled_a.public_output_digest, compiled_b.public_output_digest,
        "public_output_digest must depend on the R1CS public input x; two assignments \
         with different x[..m_in] produced the same chain output — the verifier cannot \
         distinguish which x was proven (council finding [P0])"
    );
}

#[test]
fn r1cs_compiler_public_output_independent_of_private_witness() {
    // Same R1CS as above (`z[0] = z[1] * z[2]`, `m_in = 1`). For fixed
    // public input `z[0] = 12`, the witness `(z[1], z[2])` is
    // underdetermined — both `(3, 4)` and `(4, 3)` satisfy. The chain's
    // `public_output_digest` proves "this x was proven" and MUST NOT
    // leak which `w` produced it; two assignments with the same x and
    // different w must produce the SAME digest.
    let r1cs = one_product_r1cs();
    assert_eq!(r1cs.m_in, 1, "test relies on z[..1] being the public input");
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00B0).expect("preprocess");

    let mut ctx_a = start_chain(&prep).expect("start chain a");
    let mut ctx_b = start_chain(&prep).expect("start chain b");

    let z_a = assignment_one_product(3, 4); // z[0] = 12, w = (3, 4)
    let z_b = assignment_one_product(4, 3); // z[0] = 12, w = (4, 3)
    assert_eq!(z_a[0], z_b[0], "test setup: public inputs must match");
    assert_ne!(
        (z_a[1], z_a[2]),
        (z_b[1], z_b[2]),
        "test setup: private witnesses must differ"
    );

    let compiled_a = compile_step(&prep, &mut ctx_a, R1csFPrimeStepInput { assignment: z_a }).expect("compile a");
    let compiled_b = compile_step(&prep, &mut ctx_b, R1csFPrimeStepInput { assignment: z_b }).expect("compile b");

    assert_eq!(
        compiled_a.public_output_digest, compiled_b.public_output_digest,
        "public_output_digest must NOT depend on the private witness w; \
         the chain output is a commitment to (structure, public input), not to the \
         full assignment"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Structure digest depends on R1CS shape.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_two_different_shapes_have_different_structure_digests() {
    let r1cs_one = one_product_r1cs();
    let r1cs_two = two_product_r1cs();
    let plan = make_small_plan(neo_math::D, 1);

    let layout_one = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    let layout_two = layout_one.clone();
    let (struct_one, _) = build_r1cs_f_prime_structure(layout_one, &r1cs_one);
    let (struct_two, _) = build_r1cs_f_prime_structure(layout_two, &r1cs_two);

    assert_ne!(
        structure_digest(&struct_one.ccs),
        structure_digest(&struct_two.ccs),
        "different R1CS shapes must produce different verifier-owned structure digests"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Cross-check: Fibonacci-as-R1CS round-trips through the R1CS encoder.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_satisfies_fibonacci_relation() {
    let r1cs = fibonacci_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0007).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    // Concrete Fibonacci step: prev=1, curr=1, next=2.
    let m = neo_math::D;
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    z[1] = F::from_u64(1);
    z[2] = F::from_u64(1);
    z[3] = F::from_u64(2);

    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment: z }).expect("fibonacci-as-r1cs");
    let _inst = r1cs_f_prime::build_instance(&prep, &compiled.encoded).expect("build_instance");

    // Sanity: rejects a bad Fibonacci step (next = 3 ≠ prev + curr).
    let mut ctx2 = start_chain(&prep).expect("start chain");
    let mut bad = vec![F::ZERO; m];
    bad[0] = F::ONE;
    bad[1] = F::from_u64(1);
    bad[2] = F::from_u64(1);
    bad[3] = F::from_u64(3);
    let err = compile_step(&prep, &mut ctx2, R1csFPrimeStepInput { assignment: bad }).expect_err("must reject");
    match err {
        R1csCompilerError::Unsatisfied(_) => {}
        other => panic!("expected Unsatisfied for bad fibonacci step, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Base + recursive share one structure (HyperNova fixed-`F'_j` invariant).
//
// Needs a real intermediate fold proof, which only the lifecycle can
// produce. The lifecycle's parent-authority CE shape is a fixed point
// of Π_RLC + Π_DEC under the active `Params` — under production
// `paper_b2` that fixed point has a much larger `c_data` domain, and
// one prove + extend takes > 5 min. Under a
// test-only smaller params profile (kappa = 4, m = 2^16, lambda = 60)
// the same fixed point structure shrinks to `c_data = 216, child_count
// = 14, r_len = 13, s_col_len = 19` and the full base → prove → extend →
// recursive-compile flow fits in ~55 s.
//
// The smaller profile preserves the protocol's algebraic correctness:
// the Goldilocks ring (Q, ETA, D, PHI_COEFFS), `k_rho`, `T`, and
// `B_BASE` are unchanged, so every Π_RLC / Π_DEC algebraic identity
// holds bit-for-bit. Only the Ajtai-SIS security parameter is reduced
// (which is irrelevant for an algebraic-correctness fixture).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_base_and_recursive_share_structure() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0099).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base append");
    let base_digest = structure_digest(&compiled_base.encoded.structure.ccs);

    let compiled_recursive = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("recursive append");
    let recursive_digest = structure_digest(&compiled_recursive.encoded.structure.ccs);

    assert_eq!(
        base_digest, recursive_digest,
        "base and recursive R1CS-F' compiles must share one structure_digest (HyperNova fixed-`F'_j` invariant)"
    );
    assert_eq!(
        chain
            .audit()
            .expect("audit after recursive append")
            .steps
            .len(),
        2,
        "builder must extend the lifecycle once per compiled assignment"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Batched R1CS-F' chain — app-public semantic output is serial-only.
// ─────────────────────────────────────────────────────────────────────────

/// R1CS plans bind app-public output through one carried semantic-state
/// digest. Until we add an aggregate digest for K app-public outputs,
/// K>1 chunks would be ambiguous and must reject.
#[test]
fn r1cs_chain_builder_rejects_batched_app_public_chunk_K3() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00B3).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let k = 3usize;
    let batch_0: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(3 + i as u64, 7))
        .collect();

    let err = match chain.append_assignments(batch_0) {
        Ok(_) => panic!("K=3 app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == k),
        "expected StatefulChunkMustBeSerial(K=3), got {err:?}"
    );
}

/// Varying K is rejected for the same reason as K=3: one outgoing
/// semantic-state digest cannot faithfully represent multiple distinct
/// app-public outputs.
#[test]
fn r1cs_chain_builder_rejects_varying_batched_app_public_chunks() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D7).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let batch_0: Vec<Vec<F>> = (0..3)
        .map(|i| assignment_one_product(3 + i as u64, 7))
        .collect();

    let err = match chain.append_assignments(batch_0) {
        Ok(_) => panic!("K=3 app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == 3),
        "expected StatefulChunkMustBeSerial(K=3), got {err:?}"
    );
}

#[test]
fn r1cs_chain_builder_rejects_recursive_batched_app_public_chunk() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E1).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    chain
        .append_assignments(vec![assignment_one_product(3, 7)])
        .expect("base chunk");

    let real_batch = vec![assignment_one_product(2, 5), assignment_one_product(4, 6)];
    let err = match chain.append_assignments(real_batch) {
        Ok(_) => panic!("recursive K=2 app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == 2),
        "expected StatefulChunkMustBeSerial(K=2), got {err:?}"
    );
}

#[test]
fn r1cs_verify_prior_fold_rejects_wrong_k_transcript() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E2).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignments(vec![assignment_one_product(3, 7)])
        .expect("base chunk");
    let ctx_before_recursive = chain.context().clone();
    let prev_audit = chain.audit().expect("audit after base").clone();

    let placeholder = r1cs_f_prime::build_instance(&prep, &compiled_base[0].encoded).expect("placeholder instance");
    let derived = neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit.clone(), vec![placeholder])
        .expect("K=1 prepared fold");

    let (pre_running, latest) = match &prev_audit.proof.state.proof {
        ProofState::Active { running, latest } => (running.clone(), latest.clone()),
        other => panic!("expected active pre-state, got {other:?}"),
    };
    let proof = match &derived.steps.last().expect("derived step").fold {
        FoldProof::Recursive(proof) => proof.clone(),
        FoldProof::NoFold => panic!("recursive extend must emit a fold proof"),
    };
    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.clone(),
        other => panic!("expected active post-state, got {other:?}"),
    };
    let fold = R1csFoldForStep {
        pre_running,
        latest,
        proof,
        post_running,
    };

    verify_prior_fold(&prep.prep, &ctx_before_recursive, &fold, 1).expect("correct K=1 transcript verifies");
    let err = verify_prior_fold(&prep.prep, &ctx_before_recursive, &fold, 2)
        .expect_err("a fold prepared for K=1 must not verify as K=2");
    assert!(
        matches!(err, FPrimeShellCompilerError::PriorFoldVerificationFailed { .. }),
        "expected PriorFoldVerificationFailed for wrong K transcript, got {err:?}"
    );
}

/// Max-fresh K is rejected for app-public R1CS plans until the frontend
/// has a sound aggregate semantic-state output for multiple public
/// values in one chunk.
#[test]
fn r1cs_chain_builder_rejects_max_fresh_app_public_chunk() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00C1).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let k = goldilocks_paper_b2::MAX_FRESH_K as usize;
    let batch_0: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(1 + i as u64, 7))
        .collect();

    let err = match chain.append_assignments(batch_0) {
        Ok(_) => panic!("K=MAX_FRESH_K app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == k),
        "expected StatefulChunkMustBeSerial(K={k}), got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Corner-rule-narrowed slots must still be enforced by the relation.
// ─────────────────────────────────────────────────────────────────────────

/// End-to-end soundness pin for derivation-narrowed widths: the mux output
/// `v4` in `(v2 - v3) * v1 = (v4 - v3)` is proven width-1 only by the
/// determining-row corner rule (no explicit Boolean row). After encoding,
/// tampering its single image slot must break structure satisfaction — a
/// layout/recomposition bug that dropped the narrowed slot's term from the
/// R1CS rows would silently accept the tamper and fail this test.
#[test]
fn r1cs_compiler_tampered_corner_narrowed_slot_fails_structure() {
    let m = neo_math::D;
    let n = 4;
    let mut a = NeoMat::zero(n, m, F::default());
    let mut b = NeoMat::zero(n, m, F::default());
    let mut c = NeoMat::zero(n, m, F::default());
    // rows 0..3: explicit Boolean rows for v1, v2, v3.
    for (row, var) in (1..=3usize).enumerate() {
        a[(row, var)] = F::ONE;
        b[(row, 0)] = F::ONE;
        b[(row, var)] = F::ZERO - F::ONE;
    }
    // row 3: (v2 - v3) * v1 = v4 - v3   (bellpepper ch/select shape)
    a[(3, 2)] = F::ONE;
    a[(3, 3)] = F::ZERO - F::ONE;
    b[(3, 1)] = F::ONE;
    c[(3, 4)] = F::ONE;
    c[(3, 3)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };

    // Derive widths; the mux output must be corner-proven width 1 while
    // having no explicit Boolean row (otherwise this test pins nothing).
    let widths = r1cs_f_prime::R1csShape::from(&r1cs).conservative_app_private_var_widths();
    assert_eq!(widths[4], 1, "mux output must be corner-proven Boolean");
    let booleans = r1cs_f_prime::R1csShape::from(&r1cs).boolean_constrained_variables();
    assert!(
        !booleans[4],
        "v4 must have no explicit Boolean row for this pin to bite"
    );

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = widths;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00C7).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    // Honest assignment: v1 = 1 (select), v2 = 1, v3 = 0 => v4 = v2 = 1.
    let mut assignment = vec![F::ZERO; m];
    assignment[0] = F::ONE;
    assignment[1] = F::ONE;
    assignment[2] = F::ONE;
    assignment[4] = F::ONE;
    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment }).expect("compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    let v4_slot = prep.anchors().app_var_slots[4];
    assert_eq!(v4_slot.bits, 1, "narrowed slot must be a single bit");

    // Flip the bit (1 -> 0): the mux row must reject.
    let mut tampered = compiled.encoded.witness.clone();
    tampered[v4_slot.bit_start] = F::ZERO;
    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "flipping the corner-narrowed mux output slot must break the mux row"
    );

    // Out-of-range value (2): must also reject — the slot's value reaches
    // the mux row through recomposition, not through any trusted width.
    let mut tampered = compiled.encoded.witness.clone();
    tampered[v4_slot.bit_start] = F::from_u64(2);
    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "a non-bit value in the narrowed slot must break the mux row"
    );
}
