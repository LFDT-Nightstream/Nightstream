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
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
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
    assignment_one_product, fibonacci_r1cs, make_small_plan, make_tiny_lifecycle_plan, one_product_r1cs, tiny_params,
    two_product_r1cs, BOUNDARY_BITS,
};

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

#[test]
fn r1cs_chain_builder_placeholder_extend_matches_real_batch_same_shape_same_k() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E1).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    chain
        .append_assignments(vec![assignment_one_product(3, 7)])
        .expect("base chunk");
    let prev_audit = chain.audit().expect("audit after base").clone();

    let real_batch = vec![assignment_one_product(2, 5), assignment_one_product(4, 6)];
    let compiled = chain
        .append_assignments(real_batch)
        .expect("optimized placeholder-swap recursive chunk");
    let optimized = chain.audit().expect("optimized audit").clone();
    let real_instances = compiled
        .iter()
        .map(|step| r1cs_f_prime::build_instance(&prep, &step.encoded))
        .collect::<Result<Vec<_>, _>>()
        .expect("real instances");

    let direct =
        neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit, real_instances).expect("direct real-batch extend");

    assert_eq!(optimized.steps.len(), direct.steps.len());
    assert_eq!(optimized.public_batches.len(), direct.public_batches.len());
    assert_eq!(
        format!("{:?}", optimized.proof.state),
        format!("{:?}", direct.proof.state),
        "placeholder-swap audit must land on the same post-fold state as a direct real-batch extend"
    );
    assert_eq!(
        format!("{:?}", optimized.steps.last().expect("optimized step").fold),
        format!("{:?}", direct.steps.last().expect("direct step").fold),
        "same-shape same-K placeholder and real batch must produce identical fold authority"
    );
    assert_eq!(
        optimized.steps.last().expect("optimized step").x_out,
        direct.steps.last().expect("direct step").x_out,
        "same-shape same-K placeholder and real batch must produce identical x_out"
    );
    assert_eq!(
        format!("{:?}", optimized.public_batches.last()),
        format!("{:?}", direct.public_batches.last()),
        "the optimized audit must swap the real deposited public batch back into the audit trail"
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
