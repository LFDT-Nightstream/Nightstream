//! R1CS-F' frontend integration tests.
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

#![allow(non_snake_case)]

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::engine::decider::synthesize_statement_r1cs;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, build_r1cs_f_prime_structure, compile_step, start_chain, R1csChainBuilder, R1csCompilerError,
    R1csFPrimeStepInput,
};
use neo_fold_clean::paper::digest::{digest_fields_as_digest32, structure_digest};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};

/// Number of `c_data` lanes in the test NIFS payload. Small enough to
/// keep structure construction cheap; large enough to exercise the
/// recursive-accumulator hash plumbing.
const TEST_C_DATA_ENTRIES: usize = 2;

/// `preprocess_seeded` wrapper that returns the error directly.
/// `Result::expect_err` would require `R1csFPrimePreprocessing: Debug`,
/// which the type intentionally doesn't carry (the inner `Preprocessing`
/// doesn't either).
fn expect_preprocess_err(r1cs: &R1cs, plan: &RecursiveStepImagePlan, seed: u64) -> r1cs_f_prime::Error {
    match r1cs_f_prime::preprocess_seeded(r1cs, plan, seed) {
        Ok(_) => panic!("preprocess_seeded must reject this plan; it accepted instead"),
        Err(e) => e,
    }
}

/// One R1CS variable occupies 64 bits in `app_private`.
fn app_private_bits_for(m: usize) -> usize {
    m * POSEIDON2_GOLDILOCKS_BITS
}

/// Canonical 4-lane boundary for state_x_out's public digest.
const BOUNDARY_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;

/// Build a small recursive-step plan sized for an R1CS with `m` variables.
/// `m_in` is the public-input variable count; the plan binds variables
/// `[0..m_in)` into `state_x_out` so the chain's `public_output_digest`
/// commits to the proven public input `x`. Uses a 2-entry CE NIFS payload
/// so the structure is small but still hosts the unified-mode accumulator
/// selector + Poseidon transitions.
fn make_small_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    // Sized so app_private holds m * 64 bits. limbs = m*64 + 1 because
    // image::FPrimeImageLayout::new puts app_private at
    // `limbs - 1` bits.
    let limbs = app_private_bits_for(m) + 1;

    let ce_shape = NifsCeClaimShape {
        c_data_entries: TEST_C_DATA_ENTRIES,
        x_rows: 0,
        x_active_cols: 0,
        r_len: 0,
        y_ring_inner_lens: vec![],
        y_zcol_len: 0,
        s_col_len: 0,
    };

    let probe_plan = RecursiveStepImagePlan {
        limbs,
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape.clone())],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: TEST_C_DATA_ENTRIES,
            child_count: 1,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| boundary_start + i * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        // Bind every R1CS public-input variable into state_x_out so the
        // chain's verifier-visible digest commits to the actual `x`.
        app_public_input_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
    });
    plan
}

/// R1CS with one constraint `z[0] = z[1] * z[2]` and `m_in = 1`.
/// Variable order: [z_0 (out), z_1, z_2, ...]. The matrix is padded to
/// `neo_math::D` columns for ergonomics — the bottom `(m - 3)` variables
/// are unconstrained app-private values (they still must be in {0,1} per
/// the F' bit-validity rows, so we set them to zero in the test
/// assignments).
fn one_product_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 2)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 0)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

/// R1CS with two constraints `z[0] = z[1] * z[2]` and `z[3] = z[4] * z[5]`.
fn two_product_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(2, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(1, 4)] = F::ONE;
    let mut b = NeoMat::zero(2, m, F::default());
    b[(0, 2)] = F::ONE;
    b[(1, 5)] = F::ONE;
    let mut c = NeoMat::zero(2, m, F::default());
    c[(0, 0)] = F::ONE;
    c[(1, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

/// Assignment for `one_product_r1cs`: z[1] = a, z[2] = b, z[0] = a*b,
/// rest zero. `a` and `b` must be small enough that `a*b` fits in 64
/// bits unsigned (the encoder writes 64 little-endian bits per variable
/// and the structure recomposes them as `Σ 2^i · bit`; values outside
/// `[0, 2^64)` would silently truncate). For Goldilocks we stay well
/// under that.
fn assignment_one_product(a: u64, b: u64) -> Vec<F> {
    let m = neo_math::D;
    let mut z = vec![F::ZERO; m];
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[0] = F::from_u64(a * b);
    z
}

fn assignment_one_product_with_extras(a: u64, b: u64, extras: &[(usize, u64)]) -> Vec<F> {
    let mut z = assignment_one_product(a, b);
    for &(index, value) in extras {
        z[index] = F::from_u64(value);
    }
    z
}

fn make_stateful_plan(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
) -> RecursiveStepImagePlan {
    let mut plan = make_small_plan(m, m_in);
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("make_small_plan installs state_x_out");
    state_x_out.semantic_state_in_var_indices = semantic_state_in_var_indices;
    state_x_out.semantic_state_out_var_indices = semantic_state_out_var_indices;
    plan
}

fn make_tiny_stateful_lifecycle_plan(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
) -> RecursiveStepImagePlan {
    let mut plan = make_tiny_lifecycle_plan(m, m_in);
    let NifsPayloadShape::CeClaim(shape) = &mut plan.nifs_payload_shapes[0] else {
        panic!("tiny lifecycle plan uses a CE payload");
    };
    // Stateful semantic binding adds two Poseidon2 traces / binding
    // blocks, increasing the fixed-point row-domain length by one bit
    // under the tiny test params.
    shape.r_len = 22;
    shape.s_col_len = 22;
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("make_tiny_lifecycle_plan installs state_x_out");
    state_x_out.semantic_state_in_var_indices = semantic_state_in_var_indices;
    state_x_out.semantic_state_out_var_indices = semantic_state_out_var_indices;
    plan
}

// ─────────────────────────────────────────────────────────────────────────
// Happy + sad path: compile_step accept / reject on app-level R1CS.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_compiler_accepts_satisfying_witness() {
    let r1cs = one_product_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0001).expect("preprocess");
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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0008).expect("preprocess");
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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0002).expect("preprocess");
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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0003).expect("preprocess");
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
    // R1CS rows recompose z[1] from those bits; flipping one alters
    // the product (z[1] · z[2]) and forces either the R1CS row to
    // fail or the bit-validity row to fail (since the flipped bit was
    // either 0→1 or 1→0, but the row enforces `b(b-1)=0`, so flipping
    // 0→1 actually stays valid for the bitness row alone — only the
    // R1CS row catches it).
    let app_offset = compiled.encoded.image.layout.app_private.offset;
    let mut tampered = compiled.encoded.witness.clone();
    // Flip bit 0 of variable z[1]: bit_start = app_offset + 1*64.
    let target = app_offset + POSEIDON2_GOLDILOCKS_BITS;
    tampered[target] = if tampered[target] == F::ZERO { F::ONE } else { F::ZERO };

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "bit-flip inside app_private must break either the bitness row or an R1CS row"
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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0444).expect("preprocess");
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
// Council finding [P0]: without binding `x = assignment[..m_in]` into the
// chain's `state_x_out` Poseidon hash, two different satisfying assignments
// with different `x` produce the same `public_output_digest`. The verifier
// learns only "some `(x, w)` satisfies the R1CS" — not which `x`. These
// tests pin down both halves of the contract.
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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_00A0).expect("preprocess");

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
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_00B0).expect("preprocess");

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
// Stateful semantic lane — app-state digests link serial R1CS steps.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_stateful_semantic_digest_binds_private_state_wires() {
    let r1cs = one_product_r1cs();
    let plan = make_stateful_plan(r1cs.m(), r1cs.m_in, vec![6], vec![7]);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0A01).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product_with_extras(3, 7, &[(6, 42), (7, 43)]),
        },
    )
    .expect("stateful base compile");

    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "honest stateful R1CS-F' image must satisfy its structure"
    );
    assert_ne!(
        compiled.semantic_state_digest_in, compiled.semantic_state_digest_out,
        "test setup uses different state_in/state_out values"
    );
    assert_eq!(
        ctx.chain_state.semantic_state_digest, compiled.semantic_state_digest_out,
        "compiler context must carry the app-state output digest, not the accumulator digest"
    );

    // Variable z[6] is not used by the app R1CS or public-input hash.
    // Flipping it should fail only because the F' shell binds
    // H(z[6]) to state_in.semantic_state_digest_in.
    let mut tampered = compiled.encoded.witness.clone();
    let bit = compiled.encoded.image.layout.app_private.offset + 6 * POSEIDON2_GOLDILOCKS_BITS;
    tampered[bit] = if tampered[bit] == F::ZERO { F::ONE } else { F::ZERO };
    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "semantic-state Poseidon binding must reject tampering with the private app-state wire"
    );
}

#[test]
fn r1cs_stateful_semantic_digest_lane_tamper_rejects() {
    let r1cs = one_product_r1cs();
    let plan = make_stateful_plan(r1cs.m(), r1cs.m_in, vec![6], vec![7]);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0A04).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product_with_extras(3, 7, &[(6, 42), (7, 43)]),
        },
    )
    .expect("stateful base compile");
    assert!(compiled
        .encoded
        .structure
        .is_satisfied(&compiled.encoded.witness));

    // The assignment and semantic-state Poseidon trace remain honest,
    // but the carried state-in digest lane is changed coherently. This
    // isolates the F' shell's trace-digest ↔ state-in binding rows.
    let mut tampered_in = compiled.encoded.witness.clone();
    let semantic_in_lane_0 = compiled.encoded.image.layout.state_in.offset + 16 * POSEIDON2_GOLDILOCKS_BITS;
    tampered_in[semantic_in_lane_0] = if tampered_in[semantic_in_lane_0] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    let row = compiled
        .encoded
        .structure
        .first_unsatisfied_row(&tampered_in)
        .expect("state-in lane tamper must reject");
    let start = compiled
        .encoded
        .structure
        .state_in_digest_binding_row_start();
    let end = start
        + compiled
            .encoded
            .structure
            .state_in_digest_binding_row_count();
    assert!(
        (start..end).contains(&row),
        "semantic-state state-in lane tamper must trip a state-in digest binding row; got {row}, expected {start}..{end}"
    );

    // Same isolation for the outgoing semantic digest lane.
    let mut tampered_out = compiled.encoded.witness.clone();
    let semantic_out_lane_0 = compiled.encoded.image.layout.state_out.offset + 10 * POSEIDON2_GOLDILOCKS_BITS;
    tampered_out[semantic_out_lane_0] = if tampered_out[semantic_out_lane_0] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    let row = compiled
        .encoded
        .structure
        .first_unsatisfied_row(&tampered_out)
        .expect("state-out lane tamper must reject");
    let start = compiled
        .encoded
        .structure
        .state_out_digest_binding_row_start();
    let end = start
        + compiled
            .encoded
            .structure
            .state_out_digest_binding_row_count();
    assert!(
        (start..end).contains(&row),
        "semantic-state state-out lane tamper must trip a state-out digest binding row; got {row}, expected {start}..{end}"
    );
}

#[test]
fn r1cs_stateful_compiler_rejects_disconnected_second_step() {
    let r1cs = one_product_r1cs();
    let plan = make_stateful_plan(r1cs.m(), r1cs.m_in, vec![1], vec![0]);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0A02).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let first = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("base step");
    assert_eq!(ctx.chain_state.semantic_state_digest, first.semantic_state_digest_out);

    // The previous output digest is H(21), but this assignment claims
    // the next input state is H(5). The compiler must reject before it
    // even asks for a recursive fold proof.
    let err = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(5, 2),
        },
    )
    .expect_err("disconnected semantic state must reject");
    match err {
        R1csCompilerError::SemanticStateInputMismatch { expected, got } => {
            assert_eq!(expected, first.semantic_state_digest_out);
            assert_ne!(got, expected);
        }
        other => panic!("expected SemanticStateInputMismatch, got {other:?}"),
    }
}

#[test]
fn r1cs_stateful_chain_builder_rejects_parallel_chunk() {
    let r1cs = one_product_r1cs();
    let plan = make_stateful_plan(r1cs.m(), r1cs.m_in, vec![1], vec![0]);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0A03).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let err = chain
        .append_assignments(vec![assignment_one_product(3, 7), assignment_one_product(4, 5)])
        .expect_err("stateful semantic mode must reject K > 1 chunks");
    match err {
        r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) => {
            assert_eq!(got, 2)
        }
        other => panic!("expected StatefulChunkMustBeSerial, got {other:?}"),
    }
}

/// Stateful Fibonacci transition:
///
/// ```text
/// state_in  = (a, b)
/// state_out = (b, a + b)
/// ```
///
/// Variable layout: `[one, a, b, a_out, b_out, ...]`.
fn fibonacci_transition_stateful_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(2, m, F::default());
    a[(0, 2)] = F::ONE; // b
    a[(1, 1)] = F::ONE; // a
    a[(1, 2)] = F::ONE; // + b
    let mut b = NeoMat::zero(2, m, F::default());
    b[(0, 0)] = F::ONE; // * one
    b[(1, 0)] = F::ONE; // * one
    let mut c = NeoMat::zero(2, m, F::default());
    c[(0, 3)] = F::ONE; // a_out = b
    c[(1, 4)] = F::ONE; // b_out = a + b
    R1cs { a, b, c, m_in: 5 }
}

fn assignment_fibonacci_transition(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; neo_math::D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(b);
    z[4] = F::from_u64(a + b);
    z
}

fn semantic_digest_for_pair(a: u64, b: u64) -> [u8; 32] {
    let fields = [F::from_u64(a), F::from_u64(b)];
    digest_fields_as_digest32(encode_poseidon_trace(&build_semantic_state_preimage_fields(&fields)).digest_native)
}

fn validate_decider_statement(
    prep: &neo_fold_clean::Preprocessing,
    statement: &neo_fold_clean::paper::decider::Statement,
) -> Result<(), neo_fold_clean::paper::decider::Error> {
    neo_fold_clean::paper::decider::validate_witness(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        prep.public_input_len,
        prep.semantic_state_mode,
        statement,
    )
}

struct LinkedFibonacciFixture {
    prep: r1cs_f_prime::R1csFPrimePreprocessing,
    audit: neo_fold_clean::UncompressedAudit,
    step_semantic_out: Vec<[F; 4]>,
}

fn build_linked_fibonacci_fixture() -> LinkedFibonacciFixture {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan(r1cs.m(), r1cs.m_in, vec![1, 2], vec![3, 4]);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0F1B).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");

    let mut step_semantic_out = Vec::new();
    for (a, b) in [(1, 1), (1, 2), (2, 3), (3, 5)] {
        let compiled = chain
            .append_assignment(assignment_fibonacci_transition(a, b))
            .expect("append linked Fibonacci transition");
        step_semantic_out.push(compiled.semantic_state_digest_out);
    }

    let audit = chain
        .finish_with_audit()
        .expect("finish linked Fibonacci chain");
    LinkedFibonacciFixture {
        prep,
        audit,
        step_semantic_out,
    }
}

#[test]
fn r1cs_stateful_linked_fibonacci_chain_verifies_end_to_end() {
    let fixture = build_linked_fibonacci_fixture();

    assert_eq!(
        fixture.audit.proof.state.initial_semantic_state_digest,
        semantic_digest_for_pair(1, 1),
        "initial semantic state must be H(1, 1)"
    );
    assert_eq!(
        fixture.audit.proof.state.semantic_state_digest,
        semantic_digest_for_pair(5, 8),
        "final semantic state after four transitions must be H(5, 8)"
    );

    for (idx, expected) in [(1, 2), (2, 3), (3, 5), (5, 8)].into_iter().enumerate() {
        let expected_digest = semantic_digest_for_pair(expected.0, expected.1);
        assert_eq!(
            fixture.audit.steps[idx].semantic_state_digest, expected_digest,
            "step {idx} proof must carry H({},{})",
            expected.0, expected.1
        );
        assert_eq!(
            digest_fields_as_digest32(fixture.step_semantic_out[idx]),
            expected_digest,
            "compiled step {idx} semantic output must match the proof-carried digest"
        );
    }

    neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &fixture.audit.proof)
        .expect("non-replay verifier accepts linked Fibonacci proof");
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &fixture.audit)
        .expect("audit verifier accepts linked Fibonacci proof");

    let statement = neo_fold_clean::build_decider_statement(&fixture.prep.prep, &fixture.audit);
    validate_decider_statement(&fixture.prep.prep, &statement).expect("decider preflight accepts");
}

#[test]
fn r1cs_stateful_step_proof_semantic_digest_tamper_rejects_audit() {
    let mut fixture = build_linked_fibonacci_fixture();
    fixture.audit.steps[1].semantic_state_digest[0] ^= 0xFF;

    let err = neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &fixture.audit)
        .expect_err("semantic digest tamper in StepProof must reject");
    match err {
        neo_fold_clean::Error::Decider(neo_fold_clean::paper::decider::Error::WalkFailed(reason)) => {
            assert!(
                reason.contains("x_out hash chain mismatch"),
                "expected XOutMismatch in walk failure, got {reason}"
            );
        }
        other => panic!("expected decider walk failure from XOutMismatch, got {other:?}"),
    }
}

#[test]
fn r1cs_stateful_public_image_semantic_digest_tamper_rejects() {
    let fixture = build_linked_fibonacci_fixture();
    let mut statement = neo_fold_clean::build_decider_statement(&fixture.prep.prep, &fixture.audit);
    statement.public.semantic_state_digest[0] ^= 0xFF;

    assert!(
        matches!(
            validate_decider_statement(&fixture.prep.prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::PublicImageMismatch)
        ),
        "decider preflight accepted a tampered final semantic_state_digest"
    );
    assert!(
        matches!(
            synthesize_statement_r1cs(&fixture.prep.prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::PublicImageMismatch)
        ),
        "decider R1CS synthesis accepted a tampered final semantic_state_digest"
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

/// Fibonacci step expressed as a 1-row R1CS:
/// `z[3] = (z[1] + z[2]) · z[0]`  ⟺  next = prev + curr  (`z[0] = 1`).
/// Variable layout: `[1, prev, curr, next, ...zero pads to D]`.
fn fibonacci_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

#[test]
fn r1cs_compiler_satisfies_fibonacci_relation() {
    let r1cs = fibonacci_r1cs();
    let plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let prep = r1cs_f_prime::preprocess_seeded(&r1cs, &plan, 0x71C5_0007).expect("preprocess");
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
// `paper_b2` that fixed point is `c_data = 972, child_count = 14,
// r_len = 23`, and one prove + extend takes > 5 min. Under a
// test-only smaller params profile (kappa = 4, m = 2^16, lambda = 60)
// the same fixed point structure shrinks to `c_data = 216, child_count
// = 14, r_len = 21` and the full base → prove → extend →
// recursive-compile flow fits in ~55 s.
//
// The smaller profile preserves the protocol's algebraic correctness:
// the Goldilocks ring (Q, ETA, D, PHI_COEFFS), `k_rho`, `T`, and
// `B_BASE` are unchanged, so every Π_RLC / Π_DEC algebraic identity
// holds bit-for-bit. Only the Ajtai-SIS security parameter is reduced
// (which is irrelevant for an algebraic-correctness fixture).
// ─────────────────────────────────────────────────────────────────────────

/// Test-only smaller `Params` profile.
///
/// Reuses the production Goldilocks ring + decomposition constants
/// (Q, ETA, D, B_BASE, K_RHO, T, EXTENSION_DEGREE) so every algebraic
/// identity in Π_RLC / Π_DEC holds bit-for-bit. Only the
/// commitment-width `kappa`, constraint count `m`, and security
/// parameter `lambda` are shrunk so the lifecycle fits under the
/// 5-minute test cap.
fn tiny_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 4,
        /* m      */ 1u64 << 16,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 60,
    )
    .expect("tiny NeoParams must satisfy the Π_RLC guard");
    Params::test_only_from_neo_params(inner)
}

/// Plan with the empirically-discovered fixed-point CE shape under
/// [`tiny_params`]. These constants come from running the lifecycle
/// once with a stub plan and reading the actual post-fold parent
/// shape; the recursive-compile path then converges in one iteration
/// because the plan matches the parent.
///
/// If `tiny_params` ever changes, the recursive-compile step will
/// fail with `PostParentShapeMismatch` and surface the new shape in
/// the error message — update these constants from that output.
fn make_tiny_lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    // c_data_entries = kappa * D = 4 * 54 = 216 under tiny_params.
    const TINY_C_DATA_ENTRIES: usize = 216;
    // child_count = K_RHO = 14 (matches production; not params-dependent).
    const TINY_CHILD_COUNT: u64 = 14;
    // r_len / s_col_len = ceil(log2(structure.m)) under the larger
    // (216-entry) NIFS payload region; this is the converged value
    // after one iteration of the probe.
    const TINY_R_LEN: usize = 21;

    let limbs = app_private_bits_for(m) + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries: TINY_C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: TINY_R_LEN,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len: TINY_R_LEN,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: TINY_C_DATA_ENTRIES,
            child_count: TINY_CHILD_COUNT,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| boundary_start + i * POSEIDON2_GOLDILOCKS_BITS);
    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
    });
    plan
}

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
// Batched R1CS-F' chain — `append_assignments` for K > 1.
// ─────────────────────────────────────────────────────────────────────────

/// Quick smoke test for batched R1CS-F' chunks: two `append_assignments`
/// calls of K=3 each. Asserts (a) per-step audit length matches batch
/// count, (b) public_batches lengths are `[3, 3]`, and (c) finish +
/// verify_uncompressed succeed end-to-end.
#[test]
fn r1cs_chain_builder_batched_chunks_K3_finishes_and_verifies() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00B3).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let k = 3usize;
    let batch_0: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(3 + i as u64, 7))
        .collect();
    let batch_1: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(2 + i as u64, 5))
        .collect();

    let compiled_0 = chain.append_assignments(batch_0).expect("K=3 base chunk");
    assert_eq!(compiled_0.len(), k);
    let compiled_1 = chain
        .append_assignments(batch_1)
        .expect("K=3 recursive chunk");
    assert_eq!(compiled_1.len(), k);

    // Chunk-shared trace assembly must NOT share `state_x_out`: each
    // assignment binds its own public input, so distinct assignments
    // (products 21, 28, 35) must produce distinct `public_output_digest`s.
    // If the shared/per-step split accidentally shared `state_x_out`,
    // these would collapse to one value.
    for chunk in [&compiled_0, &compiled_1] {
        let mut outs: Vec<_> = chunk.iter().map(|c| c.public_output_digest).collect();
        outs.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        outs.dedup();
        assert_eq!(
            outs.len(),
            k,
            "each assignment in a chunk must keep its own state_x_out / public_output_digest"
        );
    }

    // Chain advanced once per chunk: chunk_count = 2, step_count = 2K.
    let ctx = chain.context();
    assert_eq!(ctx.chain_state.chunk_count, 2);
    assert_eq!(ctx.chain_state.step_count, 2 * k as u64);

    // Lifecycle audit reflects two extends, each of size K.
    let audit = chain.audit().expect("audit after two batched appends");
    assert_eq!(audit.steps.len(), 2, "two extends, one per K-chunk");
    assert_eq!(audit.public_batches.len(), 2);
    assert_eq!(audit.public_batches[0].len(), k, "first public batch must be K=3");
    assert_eq!(audit.public_batches[1].len(), k, "second public batch must be K=3");

    let finished = chain.finish().expect("finalize batched chain");
    neo_fold_clean::verify_uncompressed(&prep.prep, &finished).expect("verify_uncompressed");
}

/// **Varying** batch sizes across chunks: K=3 then K=2. This is the case
/// the `rows_in_chunk`-vs-`nifs_msg.fresh.len()` separation exists for —
/// at the second chunk the F' transcript / step_count advance must use
/// the *current* chunk size (2), while NIFS folds the *previous* chunk
/// (3). Equal-size tests (`[3,3]`, `[61,61]`) can't distinguish the two,
/// so a regression that conflated them would slip through; this one
/// catches it via (a) `verify_prior_fold`'s K-aware chunk_digest (a
/// wrong K makes the reconstructed transcript diverge and NIFS.V
/// rejects) and (b) the explicit `step_count == 5` assertion (a wrong
/// advance would land on 6).
#[test]
fn r1cs_chain_builder_batched_chunks_varying_k_finishes_and_verifies() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00D7).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let batch_0: Vec<Vec<F>> = (0..3)
        .map(|i| assignment_one_product(3 + i as u64, 7))
        .collect();
    let batch_1: Vec<Vec<F>> = (0..2)
        .map(|i| assignment_one_product(2 + i as u64, 5))
        .collect();

    assert_eq!(chain.append_assignments(batch_0).expect("K=3 base").len(), 3);
    assert_eq!(
        chain
            .append_assignments(batch_1)
            .expect("K=2 recursive")
            .len(),
        2
    );

    // chunk_count advances once per chunk (2); step_count by the per-chunk
    // sizes (3 + 2 = 5), NOT by 2*|previous chunk|.
    let ctx = chain.context();
    assert_eq!(ctx.chain_state.chunk_count, 2);
    assert_eq!(
        ctx.chain_state.step_count, 5,
        "step_count must advance by the *current* chunk size each step (3 then 2 = 5)"
    );

    let audit = chain.audit().expect("audit");
    assert_eq!(audit.public_batches[0].len(), 3);
    assert_eq!(audit.public_batches[1].len(), 2);

    let finished = chain.finish().expect("finish");
    neo_fold_clean::verify_uncompressed(&prep.prep, &finished).expect("verify_uncompressed");
}

/// Full SuperNeo same-shape max-fresh path through the R1CS-F' frontend:
/// two `append_assignments` calls of `K = MAX_FRESH_K = 61` each. This is
/// the load-bearing "61 same-shape R1CS-F' operations at once" surface —
/// finishing folds `K + k_rho = 61 + 14 = 75` claims through Π_RLC under
/// the steady-state bound `75 · 216 = 16200 < 2^14`. (Here the per-op
/// circuit is the trivial `one_product_r1cs`, not SHA — this proves the
/// batching machinery scales to K=61, not any particular app circuit.)
///
/// Marked `#[ignore]` because it runs ~4 minutes (122 R1CS-F' compiles
/// + two heavy NIFS proves), so the default `cargo test` slice stays
/// fast. Verified end-to-end at ~244s under [`tiny_params`] on the
/// project's reference machine. Opt in via
/// `cargo test ... -- --ignored`. The smaller K=3 test
/// [`r1cs_chain_builder_batched_chunks_K3_finishes_and_verifies`] is
/// the fast smoke test that runs by default.
#[test]
#[ignore]
fn r1cs_chain_builder_batched_chunks_K_max_fresh_finishes_and_verifies() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00C1).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let k = goldilocks_paper_b2::MAX_FRESH_K as usize;
    let batch_0: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(1 + i as u64, 7))
        .collect();
    let batch_1: Vec<Vec<F>> = (0..k)
        .map(|i| assignment_one_product(1 + i as u64, 9))
        .collect();

    let compiled_0 = chain
        .append_assignments(batch_0)
        .expect("K=MAX_FRESH_K base chunk");
    assert_eq!(compiled_0.len(), k);
    let compiled_1 = chain
        .append_assignments(batch_1)
        .expect("K=MAX_FRESH_K recursive chunk");
    assert_eq!(compiled_1.len(), k);

    let audit = chain.audit().expect("audit after two batched appends");
    assert_eq!(audit.steps.len(), 2);
    assert_eq!(audit.public_batches[0].len(), k);
    assert_eq!(audit.public_batches[1].len(), k);

    let finished = chain.finish().expect("finalize K=61 R1CS-F' chain");
    neo_fold_clean::verify_uncompressed(&prep.prep, &finished).expect("verify_uncompressed");
}

// ─────────────────────────────────────────────────────────────────────────
// `preprocess` validates the plan's public-input binding.
// ─────────────────────────────────────────────────────────────────────────

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
