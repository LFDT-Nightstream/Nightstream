//! R1CS-F' compiler integration and structure checks.
//!
//! Covers witness acceptance, row/shape binding, fixed recursive structure,
//! Fibonacci encoding, and backend-carrier handoff. Stateful tests live in
//! `r1cs_compiler_stateful.rs`; preprocessing checks and shared fixtures live
//! in the neighboring system/support modules.

#![allow(non_snake_case)]

#[path = "r1cs_compiler/backend_carrier.rs"]
mod backend_carrier;
#[path = "../support/mod.rs"]
mod support;

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_ccs::matrix::Mat as NeoMat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_alphabet_sample_5_d;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_dot_product, KVar};
use neo_fold_clean::engine::r1cs_circuit::ring_action::{enforce_beta_ladder, enforce_eval_at_beta};
use neo_fold_clean::engine::r1cs_circuit::transcript::TranscriptGadget;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::compiler::{
    verify_prior_fold, FPrimeFoldPostSummary, FPrimeShellCompilerError,
};
use neo_fold_clean::frontends::f_prime::image::FPrimeImageLayout;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_recursive_step_image_config;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, audit_multi_branch_selective_low_norm_width_with_alignment, build_fixed_shape_low_norm_r1cs,
    build_fixed_shape_low_norm_r1cs_with_shared_private_prefix, build_multi_branch_low_norm_r1cs,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, build_r1cs_f_prime_structure, compile_step,
    lower_field_r1cs, lower_sparse_r1cs_to_low_norm, start_chain, FieldR1csLoweringError, FixedR1csBranch,
    LowNormR1csError, R1csChainBuilder, R1csCompilerError, R1csFPrimeStepInput, R1csFoldForStep,
};
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::paper::f_prime::r1cs::{F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN};
use neo_params::goldilocks_paper_b2;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, fibonacci_r1cs, make_small_plan, make_tiny_lifecycle_plan, one_product_r1cs, tiny_params,
    two_product_r1cs,
};

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
    assert_eq!(inst.claim.m_in, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert!(
        inst.claim.x[F_PRIME_PUBLIC_INPUT_LEN..]
            .iter()
            .all(|value| *value == F::ZERO),
        "compiled fresh carrier padding must be canonical zero"
    );
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
    assert_eq!(inst.claim.m_in, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
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
// = 14, r_len = 24` and the full base → prove → extend →
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

#[test]
fn r1cs_compiler_backend_verified_prior_fold_flag_is_consumed_once() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E3).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base append");
    let mut ctx = chain.context().clone();
    let prev_audit = chain.audit().expect("audit after base").clone();
    let placeholder = r1cs_f_prime::build_instance(&prep, &compiled_base.encoded).expect("placeholder instance");
    let derived =
        neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit.clone(), vec![placeholder]).expect("prepared fold");

    let (pre_running, latest) = match &prev_audit.proof.state.proof {
        ProofState::Active { running, latest } => (
            running.materialize().expect("pre-running materialization"),
            latest.clone(),
        ),
        other => panic!("expected active pre-state, got {other:?}"),
    };
    let proof = match &derived.steps.last().expect("derived step").fold {
        FoldProof::Recursive(proof) => proof
            .materialize()
            .expect("recursive NIFS proof materialization"),
        FoldProof::NoFold => panic!("recursive extend must emit a fold proof"),
    };
    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        other => panic!("expected active post-state, got {other:?}"),
    };
    ctx.fold_for_step = Some(R1csFoldForStep {
        pre_running,
        latest,
        proof,
        post_summary: None,
        post_running,
    });
    ctx.fold_for_step_needs_native_verify = false;

    compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("backend-verified recursive compile");

    assert!(
        ctx.fold_for_step.is_none(),
        "recursive compile must consume fold_for_step"
    );
    assert!(
        ctx.fold_for_step_needs_native_verify,
        "backend-verified skip flag must reset after one compile"
    );
}

#[test]
fn r1cs_compiler_accepts_backend_post_summary_without_full_running_surface() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E4).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base append");
    let mut ctx = chain.context().clone();
    let prev_audit = chain.audit().expect("audit after base").clone();
    let placeholder = r1cs_f_prime::build_instance(&prep, &compiled_base.encoded).expect("placeholder instance");
    let derived =
        neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit.clone(), vec![placeholder]).expect("prepared fold");

    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        other => panic!("expected active post-state, got {other:?}"),
    };
    ctx.fold_for_step = None;
    ctx.fold_summary_for_step = Some(
        FPrimeFoldPostSummary::from_running(&post_running, prep.prep.structure(), ctx.public_input_len)
            .expect("post summary"),
    );
    ctx.fold_for_step_needs_native_verify = false;

    compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("backend-summary recursive compile");

    assert!(
        ctx.fold_summary_for_step.is_none(),
        "recursive compile must consume fold_summary_for_step"
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
        ProofState::Active { running, latest } => (
            running.materialize().expect("pre-running materialization"),
            latest.clone(),
        ),
        other => panic!("expected active pre-state, got {other:?}"),
    };
    let proof = match &derived.steps.last().expect("derived step").fold {
        FoldProof::Recursive(proof) => proof
            .materialize()
            .expect("recursive NIFS proof materialization"),
        FoldProof::NoFold => panic!("recursive extend must emit a fold proof"),
    };
    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        other => panic!("expected active post-state, got {other:?}"),
    };
    let fold = R1csFoldForStep {
        pre_running,
        latest,
        proof,
        post_summary: None,
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

// ─────────────────────────────────────────────────────────────────────────
// Field-native synthesis -> sparse compiler boundary.
// ─────────────────────────────────────────────────────────────────────────

fn two_product_synthesis() -> (R1csBuilder, [Var; 2]) {
    let mut builder = R1csBuilder::new();
    let a = builder.alloc(F::from_u64(3));
    let b = builder.alloc(F::from_u64(7));
    let first = builder.alloc_mul(&Lc::from_var(a), &Lc::from_var(b));
    let c = builder.alloc(F::from_u64(5));
    let d = builder.alloc(F::from_u64(9));
    let second = builder.alloc_mul(&Lc::from_var(c), &Lc::from_var(d));
    (builder, [second, first])
}

#[test]
fn field_r1cs_lowering_preserves_rows_and_public_order() {
    let (builder, public_outputs) = two_product_synthesis();
    let lowered = lower_field_r1cs(builder, &public_outputs).expect("lower synthesized relation");

    assert_eq!(lowered.shape().n, 2, "every synthesized row must survive");
    assert_eq!(lowered.shape().m, 7, "column permutation must not change width");
    assert_eq!(lowered.shape().m_in, 3, "public prefix is [1, second, first]");
    assert_eq!(
        lowered.assignment(),
        &[
            F::ONE,
            F::from_u64(45),
            F::from_u64(21),
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(5),
            F::from_u64(9),
        ]
    );
    lowered
        .shape()
        .is_satisfied_by(lowered.assignment())
        .expect("column-normalized assignment must satisfy the exact sparse relation");

    let mut tampered = lowered.assignment().to_vec();
    tampered[1] += F::ONE;
    assert!(
        lowered.shape().is_satisfied_by(&tampered).is_err(),
        "second product's public output remains constrained"
    );
    let mut tampered = lowered.assignment().to_vec();
    tampered[2] += F::ONE;
    assert!(
        lowered.shape().is_satisfied_by(&tampered).is_err(),
        "first product's public output remains constrained"
    );
}

#[test]
fn field_r1cs_lowering_preserves_unsatisfaction() {
    let (mut builder, public_outputs) = two_product_synthesis();
    let first = public_outputs[1];
    builder.tamper_witness(first.col(), F::from_u64(22));
    assert!(!builder.is_satisfied(), "fixture must be unsatisfied before lowering");

    let lowered = lower_field_r1cs(builder, &public_outputs).expect("lower unsatisfied relation");
    assert!(
        lowered
            .shape()
            .is_satisfied_by(lowered.assignment())
            .is_err(),
        "lowering must preserve rejection instead of repairing the witness"
    );
}

#[test]
fn field_r1cs_lowering_rejects_ambiguous_public_columns() {
    let (builder, public_outputs) = two_product_synthesis();
    let err = lower_field_r1cs(builder, &[public_outputs[0], public_outputs[0]])
        .expect_err("duplicate public output must reject");
    assert!(matches!(
        err,
        FieldR1csLoweringError::DuplicatePublicOutput { col } if col == public_outputs[0].col()
    ));

    let (builder, _) = two_product_synthesis();
    let err = lower_field_r1cs(builder, &[Var::ONE]).expect_err("implicit constant cannot be repeated");
    assert!(matches!(err, FieldR1csLoweringError::ConstantOneIsImplicit));

    let (builder, _) = two_product_synthesis();
    let mut wider_builder = R1csBuilder::new();
    let mut foreign = Var::ONE;
    for _ in 0..8 {
        foreign = wider_builder.alloc(F::ZERO);
    }
    let err = lower_field_r1cs(builder, &[foreign]).expect_err("foreign column must reject");
    assert!(matches!(
        err,
        FieldR1csLoweringError::PublicOutputOutOfRange { col, cols: 7 } if col == foreign.col()
    ));
}

#[test]
fn field_r1cs_lowering_has_a_direct_low_norm_path_without_a_second_f_prime_shell() {
    let (builder, public_outputs) = two_product_synthesis();
    let lowered = lower_field_r1cs(builder, &public_outputs).expect("lower synthesized relation");
    let (shape, assignment) = lowered.into_parts();
    let encoded = lower_sparse_r1cs_to_low_norm(&shape, &assignment).expect("direct low-norm lowering");
    let encoded_width = 1 + encoded.field_widths()[1..].iter().sum::<usize>();
    assert!(
        encoded.is_satisfied(encoded.assignment()),
        "direct low-norm relation must preserve the exact R1CS rows"
    );
    assert_eq!(encoded.structure().m, encoded_width);
    assert_eq!(
        encoded.structure().n,
        encoded_width - 1 + shape.n,
        "only bitness rows plus the exact R1CS rows may be emitted; no F' shell"
    );
    assert_eq!(
        encoded.public_input_len(),
        1 + encoded.field_widths()[1..shape.m_in].iter().sum::<usize>()
    );
    assert!(
        encoded
            .assignment()
            .iter()
            .all(|value| *value == F::ZERO || *value == F::ONE),
        "every directly committed coordinate must be a bit"
    );

    let mut tampered = encoded.assignment().to_vec();
    tampered[1] = F::from_u64(2);
    assert!(
        !encoded.is_satisfied(&tampered),
        "direct lowering must constrain every low-norm coordinate to a bit"
    );
}

#[test]
fn direct_low_norm_lowering_keeps_boolean_public_outputs_one_bit_each() {
    let mut builder = R1csBuilder::new();
    let first = builder.alloc(F::ONE);
    let second = builder.alloc(F::ZERO);
    enforce_bit(&mut builder, first);
    enforce_bit(&mut builder, second);
    let lowered = lower_field_r1cs(builder, &[first, second]).expect("field lowering");
    let (shape, assignment) = lowered.into_parts();
    let encoded = lower_sparse_r1cs_to_low_norm(&shape, &assignment).expect("low-norm lowering");

    assert_eq!(&encoded.field_widths()[..3], &[1, 1, 1]);
    assert_eq!(
        encoded.public_input_len(),
        3,
        "public prefix must be [1, first, second]"
    );
    assert!(encoded.is_satisfied(encoded.assignment()));
}

#[test]
fn direct_low_norm_lowering_rejects_a_missing_public_constant_prefix() {
    let (builder, public_outputs) = two_product_synthesis();
    let lowered = lower_field_r1cs(builder, &public_outputs).expect("field lowering");
    let (mut shape, assignment) = lowered.into_parts();
    shape.m_in = 0;

    let err = lower_sparse_r1cs_to_low_norm(&shape, &assignment)
        .expect_err("the low-norm public prefix must include constant one");
    assert!(matches!(err, LowNormR1csError::MissingPublicConstant));
}

fn boolean_copy_synthesis(value: F, duplicate_copy_row: bool) -> (R1csBuilder, Var) {
    let mut builder = R1csBuilder::new();
    let private = builder.alloc(value);
    let public = builder.alloc(value);
    enforce_bit(&mut builder, private);
    enforce_bit(&mut builder, public);
    builder.enforce_eq(&Lc::from_var(private), &Lc::from_var(public));
    if duplicate_copy_row {
        builder.enforce_eq(&Lc::from_var(private), &Lc::from_var(public));
    }
    (builder, public)
}

fn shared_private_synthesis(value: F, duplicate_copy_row: bool) -> (R1csBuilder, Var) {
    let mut builder = R1csBuilder::new();
    let shared = builder.alloc(value);
    let branch_private = builder.alloc(value);
    let public = builder.alloc(value);
    enforce_bit(&mut builder, shared);
    enforce_bit(&mut builder, branch_private);
    enforce_bit(&mut builder, public);
    builder.enforce_eq(&Lc::from_var(public), &Lc::from_var(shared));
    builder.enforce_eq(&Lc::from_var(public), &Lc::from_var(branch_private));
    if duplicate_copy_row {
        builder.enforce_eq(&Lc::from_var(public), &Lc::from_var(branch_private));
    }
    (builder, public)
}

#[test]
fn fixed_shape_low_norm_relation_selects_base_or_recursive_rows() {
    let (base_builder, base_public) = boolean_copy_synthesis(F::ONE, false);
    let base = lower_field_r1cs(base_builder, &[base_public]).expect("base lowering");
    let (base_shape, base_assignment) = base.into_parts();

    let (recursive_builder, recursive_public) = boolean_copy_synthesis(F::ZERO, true);
    let recursive = lower_field_r1cs(recursive_builder, &[recursive_public]).expect("recursive lowering");
    let (recursive_shape, recursive_assignment) = recursive.into_parts();

    let fixed = build_fixed_shape_low_norm_r1cs(&base_shape, &recursive_shape).expect("fixed-shape relation");
    let base_encoded = fixed
        .encode(FixedR1csBranch::Base, &base_assignment)
        .expect("base encoding");
    let recursive_encoded = fixed
        .encode(FixedR1csBranch::Recursive, &recursive_assignment)
        .expect("recursive encoding");

    assert_eq!(base_encoded.len(), recursive_encoded.len());
    assert_eq!(fixed.public_input_len(), 2, "public prefix is [1, output_bit]");
    assert_eq!(base_encoded[fixed.selector_col()], F::ONE);
    assert_eq!(recursive_encoded[fixed.selector_col()], F::ZERO);
    assert!(fixed.is_satisfied(&base_encoded));
    assert!(fixed.is_satisfied(&recursive_encoded));
    assert_eq!(
        fixed.structure().max_degree(),
        3,
        "branch selection is one cubic CCS gate"
    );
    assert_eq!(
        fixed.structure().n,
        fixed.structure().m - 1 + base_shape.n + recursive_shape.n,
        "fixed relation contains global bitness plus both selector-gated source relations"
    );

    let base_private_col = fixed.selector_col() + 1;
    let recursive_private_col = base_private_col + 1;
    let mut tampered = base_encoded.clone();
    tampered[base_private_col] = F::ZERO;
    assert!(
        !fixed.is_satisfied(&tampered),
        "selected base witness must be load-bearing"
    );

    let mut inactive_tamper = base_encoded.clone();
    inactive_tamper[recursive_private_col] = F::ONE;
    assert!(
        fixed.is_satisfied(&inactive_tamper),
        "inactive recursive semantics must be selector-gated"
    );
    inactive_tamper[recursive_private_col] = F::from_u64(2);
    assert!(
        !fixed.is_satisfied(&inactive_tamper),
        "inactive coordinates remain globally low-norm bits"
    );

    let mut public_tamper = recursive_encoded;
    public_tamper[1] = F::ONE;
    assert!(
        !fixed.is_satisfied(&public_tamper),
        "the selected recursive relation must constrain the shared public output"
    );
}

#[test]
fn fixed_shape_relation_shares_the_application_private_prefix() {
    let (base_builder, base_public) = shared_private_synthesis(F::ONE, false);
    let base = lower_field_r1cs(base_builder, &[base_public]).expect("base lowering");
    let (base_shape, base_assignment) = base.into_parts();

    let (recursive_builder, recursive_public) = shared_private_synthesis(F::ZERO, true);
    let recursive = lower_field_r1cs(recursive_builder, &[recursive_public]).expect("recursive lowering");
    let (recursive_shape, recursive_assignment) = recursive.into_parts();

    let fixed = build_fixed_shape_low_norm_r1cs_with_shared_private_prefix(&base_shape, &recursive_shape, 1)
        .expect("shared-prefix fixed relation");
    let base_shared = fixed
        .field_slot(FixedR1csBranch::Base, base_shape.m_in)
        .expect("base shared slot");
    let recursive_shared = fixed
        .field_slot(FixedR1csBranch::Recursive, recursive_shape.m_in)
        .expect("recursive shared slot");
    assert_eq!(
        base_shared, recursive_shared,
        "the application witness has one fixed slot"
    );
    assert_ne!(
        fixed.field_slot(FixedR1csBranch::Base, base_shape.m_in + 1),
        fixed.field_slot(FixedR1csBranch::Recursive, recursive_shape.m_in + 1),
        "branch-specific verifier advice must remain disjoint"
    );

    let base_encoded = fixed
        .encode(FixedR1csBranch::Base, &base_assignment)
        .expect("base encoding");
    let recursive_encoded = fixed
        .encode(FixedR1csBranch::Recursive, &recursive_assignment)
        .expect("recursive encoding");
    assert!(fixed.is_satisfied(&base_encoded));
    assert!(fixed.is_satisfied(&recursive_encoded));

    let mut tampered = base_encoded;
    tampered[base_shared.0] = F::ZERO;
    assert!(
        !fixed.is_satisfied(&tampered),
        "the shared application slot must remain load-bearing in the selected branch"
    );
}

#[test]
fn multi_branch_relation_one_hot_selects_base_bootstrap_and_steady_rows() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for (value, duplicate) in [(F::ONE, false), (F::ZERO, true), (F::ONE, true)] {
        let (builder, public) = shared_private_synthesis(value, duplicate);
        let lowered = lower_field_r1cs(builder, &[public]).expect("arm lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let fixed = build_multi_branch_low_norm_r1cs(&shapes, 1).expect("three-arm fixed relation");
    assert_eq!(fixed.selector_cols().len(), 3);
    assert_eq!(fixed.structure().t(), 6);
    assert_eq!(fixed.structure().max_degree(), 3);
    let shared_slots: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(arm, shape)| fixed.field_slot(arm, shape.m_in).expect("shared slot"))
        .collect();
    assert!(shared_slots.iter().all(|slot| *slot == shared_slots[0]));
    let branch_slots: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(arm, shape)| {
            fixed
                .field_slot(arm, shape.m_in + 1)
                .expect("branch-local slot")
                .0
        })
        .collect();
    assert!(
        branch_slots.iter().all(|slot| *slot == branch_slots[0]),
        "one-hot arms must overlay branch-local advice instead of paying their summed widths"
    );
    let last_used_col = shapes
        .iter()
        .enumerate()
        .filter_map(|(arm, shape)| fixed.field_slot(arm, shape.m - 1))
        .map(|(start, width)| start + width)
        .chain(fixed.selector_cols().iter().map(|&column| column + 1))
        .max()
        .expect("fixed relation has selectors and branch slots");
    assert_eq!(
        fixed.structure().m,
        last_used_col,
        "overlaid relation must not retain an unused disjoint-arm tail"
    );

    for arm in 0..3 {
        let encoded = fixed.encode(arm, &assignments[arm]).expect("arm encoding");
        assert!(fixed.is_satisfied(&encoded), "arm {arm} must satisfy");
        assert_eq!(
            fixed
                .selector_cols()
                .iter()
                .map(|&col| encoded[col])
                .collect::<Vec<_>>(),
            (0..3)
                .map(|selector| if selector == arm { F::ONE } else { F::ZERO })
                .collect::<Vec<_>>()
        );
    }

    let mut two_hot = fixed.encode(0, &assignments[0]).expect("base encoding");
    two_hot[fixed.selector_cols()[1]] = F::ONE;
    assert!(
        !fixed.is_satisfied(&two_hot),
        "selector sum must enforce exactly one active arm"
    );
}

fn canonical_decomposition_synthesis(value: F, duplicate_row: bool) -> (R1csBuilder, Var) {
    let mut builder = R1csBuilder::new();
    let source = builder.alloc(value);
    let _bits = decompose_var_to_u64_bits(&mut builder, source);
    let public = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    if duplicate_row {
        builder.enforce_eq(&Lc::from_var(source), &Lc::from_var(source));
    }
    (builder, public)
}

#[test]
fn direct_low_norm_lowering_reuses_canonical_decomposition_bits() {
    let (builder, public) = canonical_decomposition_synthesis(F::from_u64(7), false);
    let lowered = lower_field_r1cs(builder, &[public]).expect("field lowering");
    let (shape, assignment) = lowered.into_parts();
    let encoded = lower_sparse_r1cs_to_low_norm(&shape, &assignment).expect("low-norm lowering");

    let source_col = shape.m_in;
    let first_bit_col = source_col + 1;
    let source_slot = encoded.field_slot(source_col).expect("source field slot");
    assert_eq!(source_slot.1, 64);
    for bit in 0..64 {
        assert_eq!(
            encoded.field_slot(first_bit_col + bit),
            Some((source_slot.0 + bit, 1)),
            "decomposition bit {bit} must reuse the source field slot"
        );
    }
    let unaliased_width = 1 + encoded.field_widths()[1..].iter().sum::<usize>();
    assert_eq!(
        encoded.structure().m + 64,
        unaliased_width,
        "one canonical decomposition must remove exactly 64 duplicate committed bits"
    );
    assert!(encoded.is_satisfied(encoded.assignment()));

    let mut inconsistent = assignment;
    inconsistent[first_bit_col] = F::ZERO;
    let error = lower_sparse_r1cs_to_low_norm(&shape, &inconsistent)
        .expect_err("an aliased child cannot disagree with its source bit");
    assert!(matches!(
        error,
        LowNormR1csError::AliasedBitMismatch {
            field_col,
            bit_col,
            bit: 0,
        } if field_col == source_col && bit_col == first_bit_col
    ));
}

#[test]
fn multi_branch_lowering_preserves_canonical_bit_aliases() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for (value, duplicate) in [
        (F::from_u64(7), false),
        (F::from_u64(11), true),
        (F::from_u64(13), false),
    ] {
        let (builder, public) = canonical_decomposition_synthesis(value, duplicate);
        let lowered = lower_field_r1cs(builder, &[public]).expect("arm lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let fixed = build_multi_branch_low_norm_r1cs(&shapes, 0).expect("three-arm fixed relation");
    for arm in 0..3 {
        let source_col = shapes[arm].m_in;
        let source_slot = fixed
            .field_slot(arm, source_col)
            .expect("source field slot");
        assert_eq!(source_slot.1, 64);
        for bit in 0..64 {
            assert_eq!(
                fixed.field_slot(arm, source_col + 1 + bit),
                Some((source_slot.0 + bit, 1)),
                "arm {arm} decomposition bit {bit} must reuse its source slot"
            );
        }
        let encoded = fixed.encode(arm, &assignments[arm]).expect("arm encoding");
        assert!(fixed.is_satisfied(&encoded), "arm {arm} must remain satisfiable");
    }
}

#[test]
fn selective_multi_branch_lowering_preserves_rejection_sampling() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let mut transcript = TranscriptGadget::new(&mut builder, b"selective-product-sum-test");
        let symbols = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, 90 + arm);
        let symbol_bits = decompose_var_to_u64_bits(&mut builder, symbols[0]);
        let lowered = lower_field_r1cs(builder, &[symbol_bits[0]]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, 54, 0).expect("selective relation");
    for arm in 0..3 {
        let encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded), "arm {arm} product-sum relation");
    }
}

#[test]
fn selective_lowering_reuses_surviving_source_bitness_rows_only() {
    fn boolean_arm(record_source_row: bool) -> (r1cs_f_prime::SparseR1cs, Vec<F>) {
        let mut builder = R1csBuilder::new();
        let bit = builder.alloc(F::ONE);
        if record_source_row {
            enforce_bit(&mut builder, bit);
        }
        lower_field_r1cs(builder, &[bit])
            .expect("Boolean arm lowering")
            .into_parts()
    }

    let recorded: [(r1cs_f_prime::SparseR1cs, Vec<F>); 3] = std::array::from_fn(|_| boolean_arm(true));
    let unproved: [(r1cs_f_prime::SparseR1cs, Vec<F>); 3] = std::array::from_fn(|_| boolean_arm(false));
    let recorded_shapes = recorded
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let unproved_shapes = unproved
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let recorded_relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&recorded_shapes, 0, D, 0)
        .expect("recorded Boolean relation");
    let unproved_relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&unproved_shapes, 0, D, 0)
        .expect("unproved Boolean relation");

    assert_eq!(
        recorded_relation.structure().n,
        unproved_relation.structure().n + 2,
        "three selected source rows replace one generated shared-Boolean row without adding a duplicate"
    );

    for arm in 0..3 {
        let slot = recorded_relation
            .field_slot(arm, 1)
            .expect("public Boolean slot");
        let mut encoded = recorded_relation
            .encode(arm, &recorded[arm].1)
            .expect("recorded Boolean encoding");
        assert!(recorded_relation.is_satisfied(&encoded));
        encoded[slot.0] = -F::ONE;
        assert!(
            !recorded_relation.is_satisfied(&encoded),
            "the retained source row must reject -1 after the duplicate row is removed"
        );

        let slot = unproved_relation
            .field_slot(arm, 1)
            .expect("unproved Boolean slot");
        let mut encoded = unproved_relation
            .encode(arm, &unproved[arm].1)
            .expect("unproved Boolean encoding");
        assert!(unproved_relation.is_satisfied(&encoded));
        encoded[slot.0] = -F::ONE;
        assert!(
            !unproved_relation.is_satisfied(&encoded),
            "the generated row must remain when the source relation does not prove bitness"
        );
    }
}

#[test]
fn selective_multi_branch_balanced_ternary_handles_field_edges_and_rejects_non_unit_digits() {
    fn reference_digits(value: F) -> [F; 41] {
        let modulus = F::ORDER_U64 as i128;
        let canonical = value.as_canonical_u64() as i128;
        let mut remaining = if canonical <= modulus / 2 {
            canonical
        } else {
            canonical - modulus
        };
        std::array::from_fn(|_| {
            let residue = remaining.rem_euclid(3);
            let digit = if residue == 2 { -1i128 } else { residue };
            remaining = (remaining - digit) / 3;
            match digit {
                -1 => -F::ONE,
                0 => F::ZERO,
                1 => F::ONE,
                _ => unreachable!("balanced ternary reference digit"),
            }
        })
    }

    let values = [F::ZERO, F::from_u64(F::ORDER_U64 / 2), F::from_u64(F::ORDER_U64 - 1)];
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for value in values {
        let mut builder = R1csBuilder::new();
        let private = builder.alloc(value);
        let square = builder.alloc_mul(&Lc::from_var(private), &Lc::from_var(private));
        builder.enforce_eq(&Lc::from_var(square), &Lc::from_const(value * value));
        let public = builder.alloc(F::ONE);
        enforce_bit(&mut builder, public);
        let lowered = lower_field_r1cs(builder, &[public]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        assert_eq!(assignment[shape.m_in], value, "private edge value moved unexpectedly");
        shapes.push(shape);
        assignments.push(assignment);
    }

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, 54, 0).expect("selective relation");
    for arm in 0..3 {
        let private_slot = relation
            .field_slot(arm, shapes[arm].m_in)
            .expect("retained private field slot");
        assert_eq!(private_slot.1, 41, "non-canonical full fields use balanced ternary");
        let encoded = relation
            .encode(arm, &assignments[arm])
            .expect("balanced encoding");
        assert_eq!(
            &encoded[private_slot.0..private_slot.0 + private_slot.1],
            &reference_digits(values[arm]),
            "optimized balanced ternary changed encoded bytes"
        );
        assert!(
            encoded
                .iter()
                .all(|value| *value == F::ZERO || *value == F::ONE || *value == -F::ONE),
            "every committed coordinate must satisfy the b=2 norm alphabet"
        );
        assert!(relation.is_satisfied(&encoded), "arm {arm} balanced encoding");

        let mut tampered = encoded;
        tampered[private_slot.0] = F::from_u64(2);
        assert!(
            !relation.is_satisfied(&tampered),
            "a balanced digit outside {{-1,0,1}} must fail"
        );
    }
}

#[test]
fn selective_multi_branch_equal_field_alias_rejects_inconsistent_source_assignment() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let source = builder.alloc(F::from_u64(arm + 9));
        let copy = builder.alloc(F::from_u64(arm + 9));
        builder.enforce_eq(&Lc::from_var(copy), &Lc::from_var(source));
        decompose_var_to_u64_bits(&mut builder, source);
        decompose_var_to_u64_bits(&mut builder, copy);
        let product = builder.alloc_mul(&Lc::from_var(copy), &Lc::from_var(source));
        builder.enforce_eq(
            &Lc::from_var(product),
            &Lc::from_const(F::from_u64(arm + 9) * F::from_u64(arm + 9)),
        );
        let public = builder.alloc(F::ONE);
        enforce_bit(&mut builder, public);
        let lowered = lower_field_r1cs(builder, &[public]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, 54, 0).expect("selective relation");
    for arm in 0..3 {
        let source = shapes[arm].m_in;
        let copy = source + 1;
        assert_eq!(relation.field_slot(arm, source), relation.field_slot(arm, copy));
        let encoded = relation
            .encode(arm, &assignments[arm])
            .expect("equal-field encoding");
        assert!(relation.is_satisfied(&encoded));

        let mut inconsistent = assignments[arm].clone();
        inconsistent[copy] += F::ONE;
        assert!(matches!(
            relation.encode(arm, &inconsistent),
            Err(LowNormR1csError::AliasedFieldMismatch {
                field_col,
                source_col,
            }) if field_col == copy && source_col == source
        ));
    }
}

#[test]
fn selective_multi_branch_lowering_preserves_k_dot_product() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    let mut q_sum_cols = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let left: [KVar; 6] = core::array::from_fn(|i| {
            KVar::alloc(
                &mut builder,
                F::from_u64(arm + i as u64 + 2),
                F::from_u64(arm + i as u64 + 3),
            )
        });
        let right: [KVar; 6] = core::array::from_fn(|i| {
            KVar::alloc(
                &mut builder,
                F::from_u64(arm + i as u64 + 5),
                F::from_u64(arm + i as u64 + 7),
            )
        });
        let product_start = builder.cols();
        let output = enforce_k_dot_product(&mut builder, &left, &right);
        // The selected output bit moves to column 1 during field lowering.
        q_sum_cols.push(product_start + 5 * left.len() + 1);
        let output_bits = decompose_var_to_u64_bits(&mut builder, output.c0);
        let lowered = lower_field_r1cs(builder, &[output_bits[0]]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, 54, 0).expect("selective relation");
    for arm in 0..3 {
        let encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded), "arm {arm} K-multiplication relation");
        let q_slot = relation
            .field_slot(arm, q_sum_cols[arm])
            .expect("Q aggregate slot");
        let mut tampered = encoded;
        tampered[q_slot.0] = if tampered[q_slot.0] == F::ZERO { F::ONE } else { F::ZERO };
        assert!(
            !relation.is_satisfied(&tampered),
            "arm {arm} accepted a changed unit Q digit"
        );
    }
}

#[test]
fn selective_multi_branch_telescoping_evaluation_advice_is_load_bearing() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let coefficients: [Var; D] =
            core::array::from_fn(|index| builder.alloc(F::from_u64(arm * 101 + index as u64 + 1)));
        let beta = KVar::alloc(&mut builder, F::from_u64(arm + 3), F::from_u64(arm + 5));
        let powers = enforce_beta_ladder(&mut builder, beta, D);
        let output = enforce_eval_at_beta(&mut builder, &coefficients, &powers);
        let output_bits = decompose_var_to_u64_bits(&mut builder, output.c0);
        let lowered = lower_field_r1cs(builder, &[output_bits[0]]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let audit =
        audit_multi_branch_selective_low_norm_width_with_alignment(&shapes, 0, 54, 0).expect("selective width audit");
    assert!(
        audit.arms.iter().all(|arm| arm.derived_product_sums > 0),
        "a 54-coefficient evaluation must require telescoping advice"
    );

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, 54, 0).expect("selective relation");
    for arm in 0..3 {
        let mut encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded), "arm {arm} honest evaluation");

        let mut source_backed = vec![false; relation.structure().m];
        source_backed[0] = true;
        for &selector in relation.selector_cols() {
            source_backed[selector] = true;
        }
        for column in 1..shapes[arm].m {
            if let Some((start, width)) = relation.field_slot(arm, column) {
                source_backed[start..start + width].fill(true);
            }
        }
        let derived = encoded
            .iter()
            .enumerate()
            .find_map(|(column, value)| (!source_backed[column] && *value != F::ZERO).then_some(column))
            .expect("honest evaluation has nonzero compiler-added advice");
        encoded[derived] += F::ONE;
        assert!(
            !relation.is_satisfied(&encoded),
            "arm {arm} accepted a tampered telescoping accumulator"
        );
    }
}
