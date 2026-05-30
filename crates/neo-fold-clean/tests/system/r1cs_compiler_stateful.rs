//! R1CS-F' frontend integration tests — stateful semantic-digest path.
//!
//! Covers the app-state lane that links serial R1CS steps via Poseidon
//! binding rows: `state_in.semantic_state_digest_in_lane`,
//! `state_out.semantic_state_digest_out_lane`, the base-step anchor
//! constraint, and the lifecycle's end-to-end Fibonacci-transition
//! chain. Stateless tests live in the sibling `r1cs_compiler.rs`;
//! preprocess-time plan validation lives in `r1cs_preprocess.rs`.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::engine::decider::synthesize_statement_r1cs;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, compile_step, start_chain, R1csChainBuilder, R1csCompilerError, R1csFPrimeStepInput,
};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, assignment_one_product_with_extras, make_stateful_plan, make_stateful_plan_with_anchor,
    make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs, tiny_params,
};

// ─────────────────────────────────────────────────────────────────────────
// Stateful semantic lane — app-state digests link serial R1CS steps.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn r1cs_stateful_semantic_digest_binds_private_state_wires() {
    let r1cs = one_product_r1cs();
    // Anchor matches the first assignment's state_in_app_var (z[6] = 42),
    // so the F' image's base-gated `state_in.semantic_state_digest_in == anchor`
    // constraint is satisfied by the honest witness below.
    let plan = make_stateful_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![6],
        vec![7],
        Some(semantic_digest_for_single(42)),
    );
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
    let plan = make_stateful_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![6],
        vec![7],
        Some(semantic_digest_for_single(42)),
    );
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
    // First assignment is (3, 7) → state_in_app = z[1] = 3.
    let plan = make_stateful_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1],
        vec![0],
        Some(semantic_digest_for_single(3)),
    );
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
fn r1cs_stateful_fibonacci_rejects_rewound_second_step() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A12).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let first = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect("base Fibonacci step");
    assert_eq!(
        digest_fields_as_digest32(first.semantic_state_digest_out),
        semantic_digest_for_pair(1, 2)
    );

    let err = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect_err("rewinding step 2 to the initial Fibonacci state must reject");
    match err {
        R1csCompilerError::SemanticStateInputMismatch { expected, got } => {
            assert_eq!(expected, first.semantic_state_digest_out);
            assert_eq!(digest_fields_as_digest32(got), semantic_digest_for_pair(1, 1));
        }
        other => panic!("expected SemanticStateInputMismatch, got {other:?}"),
    }
}

#[test]
fn r1cs_stateful_fibonacci_rejects_random_second_step_state() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A13).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let first = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect("base Fibonacci step");

    let err = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(10, 10),
        },
    )
    .expect_err("step 2 with a random semantic input must reject");
    match err {
        R1csCompilerError::SemanticStateInputMismatch { expected, got } => {
            assert_eq!(expected, first.semantic_state_digest_out);
            assert_eq!(digest_fields_as_digest32(got), semantic_digest_for_pair(10, 10));
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

fn semantic_digest_for_single(v: u64) -> [u8; 32] {
    let fields = [F::from_u64(v)];
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
        prep.semantic_state_mode(),
        prep.initial_semantic_state_digest(),
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
    // Anchor the verifier-owned initial semantic-state digest to H(1, 1),
    // matching the chain's first assignment `(a, b) = (1, 1)`. The
    // anchor is baked into the F' image's CCS structure via the
    // base-gated `is_base * (state_in.semantic_state_digest_in_lane - anchor) == 0`
    // constraint, so Π_CCS sumcheck rejects any base step that doesn't
    // match the anchor.
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
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

/// Regression for the P1 finding: `initial_semantic_state_digest` is now
/// absorbed into `vk_fs_digest`, so a malicious prover who relabels the
/// public claim after-the-fact fails the cross-check in
/// `validate_witness`. Before the fix, this tamper passed silently —
/// the field was exposed on `PublicImage` but bound to no constraint.
#[test]
fn r1cs_stateful_public_image_initial_semantic_digest_tamper_rejects() {
    let fixture = build_linked_fibonacci_fixture();
    let mut statement = neo_fold_clean::build_decider_statement(&fixture.prep.prep, &fixture.audit);
    // Honest proof verifies cleanly.
    validate_decider_statement(&fixture.prep.prep, &statement).expect("honest preflight accepts");

    // Tamper the claimed initial app-state digest. The
    // verifier-owned `prep.initial_semantic_state_digest()` is
    // anchored to `H(1, 1)` via `with_initial_semantic_state_digest`;
    // any other value MUST reject.
    statement.public.initial_semantic_state_digest[0] ^= 0xFF;

    assert!(
        matches!(
            validate_decider_statement(&fixture.prep.prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::PublicImageMismatch)
        ),
        "decider preflight accepted a tampered initial_semantic_state_digest \
         (the field would otherwise be unbound; this is the P1 fix)"
    );
}

/// Regression for the P1 follow-up: `verify_uncompressed` takes a raw
/// `Uncompressed` proof (not a `Statement`), so the decider-side
/// `validate_witness` anchor check does NOT fire on this path. Without
/// the dedicated `check_initial_semantic_anchor`, a prover could
/// honestly prove a chain, then mutate `proof.state.initial_semantic_state_digest`
/// directly on the proof object and `verify_uncompressed` would
/// accept. With the anchor check in place, this MUST reject.
#[test]
fn r1cs_stateful_verify_uncompressed_rejects_tampered_proof_state_initial_semantic() {
    let fixture = build_linked_fibonacci_fixture();
    let finished = neo_fold_clean::finish_uncompressed(&fixture.prep.prep, fixture.audit.clone())
        .expect("finalize linked Fibonacci");

    // Honest proof verifies cleanly.
    neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &finished).expect("honest verify_uncompressed accepts");

    // Tamper the proof.state field directly (no decider Statement
    // involved). `prep.initial_semantic_state_digest()` is the
    // verifier-owned anchor (H(1, 1)); any other value MUST reject.
    let mut tampered = finished.clone();
    tampered.state.initial_semantic_state_digest[0] ^= 0xFF;

    let err = neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &tampered)
        .expect_err("verify_uncompressed must reject tampered proof.state.initial");
    assert!(
        matches!(err, neo_fold_clean::Error::InitialSemanticStateAnchorMismatch),
        "expected InitialSemanticStateAnchorMismatch, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Base-step initial-anchor attack (other AI's P1 from the latest review).
//
// The R1CS-F' frontend's chain builder couples (state_in_app_vars,
// state_in.semantic_state_digest_in_lane, lifecycle-layer seed) internally,
// so it cannot mount this attack. But `lifecycle::prove::prove_one_with_semantic_state`
// is `pub`, so a hand-crafted prover can:
//
//   1. Encode an F' image for app-state (10, 10) — produces an image whose
//      state_in.semantic_state_digest_in_lane = H(10, 10), state_out lane
//      = H(10, 20). All Poseidon binding rows are satisfied.
//   2. Call `prove_one_with_semantic_state(prep_anchored_to_H(1,1), instance,
//      H(1, 1), H(10, 20))` — pass the verifier's anchor as the State::base
//      seed (the LIE), but submit the instance whose witness lanes encode
//      (10, 10).
//
// The chain is internally consistent: state.initial = H(1, 1) (matches
// `check_initial_semantic_anchor`), state.semantic_state_digest = H(10, 20)
// (matches StepProof and image x_out), the F' image's CCS is satisfied
// (binding rows pass for the witness-chosen lanes). `verify_uncompressed`
// currently accepts.
//
// The base F' R1CS must enforce
//   is_base * (state_in.semantic_state_digest_in_lane[k] - anchor[k]) == 0
// for each digest lane. With that constraint in the F' image's CCS,
// Π_CCS sumcheck (run by the terminal NIFS.V replay inside
// `verify_uncompressed`) rejects the attack at random row α.
// ─────────────────────────────────────────────────────────────────────────

/// Regression for the **base-step initial-anchor attack**.
///
/// Before the structural fix: a hand-crafted prover could submit a
/// base F' image whose `state_in_app_vars = (10, 10)` and
/// `state_in.semantic_state_digest_in_lane = H(10, 10)`. The Poseidon
/// binding row passes (`H(witness app vars) == lane`). The chain coord
/// `state.initial_semantic_state_digest = H(1, 1)` (the verifier's
/// anchor, the LIE). Nothing connected the witness lane to the anchor.
/// `verify_uncompressed` accepted.
///
/// After the fix: the F' image's CCS structure carries
/// `is_base * (state_in.semantic_state_digest_in_lane[k] - anchor[k]) == 0`
/// for each digest lane. The anchor is baked into `structure_digest`
/// via `StateXOutPlanOptions::initial_semantic_state_digest_anchor`. A
/// prover encoding a base image with mismatched app-state cannot
/// satisfy the structure — `encode_r1cs_f_prime_step`'s internal
/// `assert!(structure.is_satisfied)` fires.
///
/// This test wraps the encoder call in `catch_unwind`; the panic IS
/// the rejection signal at the prover's own structural assertion.
/// Π_CCS sumcheck (run by the terminal NIFS.V inside
/// `verify_uncompressed`) would catch a hand-crafted post-encoder
/// tamper too, but the encoder-level rejection means the prover can't
/// even *produce* a malicious witness from a satisfying assignment.
#[test]
fn r1cs_stateful_attack_base_state_in_lane_rejected_at_encode_time() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let verifier_anchor = semantic_digest_for_pair(1, 1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(verifier_anchor),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_AAAA).expect("preprocess");

    // Try to encode a base F' image whose app-state wires encode
    // (10, 10) — different from the verifier's anchor H(1, 1). The
    // structure's new base-gated anchor constraint must trip
    // `structure.is_satisfied` inside `encode_r1cs_f_prime_step`.
    let attack = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ctx = start_chain(&prep).expect("start chain");
        compile_step(
            &prep,
            &mut ctx,
            R1csFPrimeStepInput {
                assignment: assignment_fibonacci_transition(10, 10),
            },
        )
    }));

    assert!(
        attack.is_err(),
        "BASE-ANCHOR SOUNDNESS: compile_step accepted a base image whose state_in_app_vars \
         (10, 10) and lane H(10, 10) disagree with the verifier-owned anchor H(1, 1). The F' \
         image's CCS structure must enforce \
         `is_base * (state_in.semantic_state_digest_in_lane[k] - anchor[k]) == 0` for each \
         digest lane, baked in via `StateXOutPlanOptions::initial_semantic_state_digest_anchor`."
    );
}

/// Companion: even if a malicious prover hand-rolls a CcsInstance that
/// bypasses `encode_r1cs_f_prime_step`'s structural assertion, the
/// chain-replay verifier's Π_CCS sumcheck still catches the lie. We
/// simulate the hand-rolled path by mutating the encoded witness
/// AFTER honest encoding, then submitting via the public lifecycle
/// entry point.
#[test]
fn r1cs_stateful_attack_post_encoder_mutation_rejected_by_verify() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let verifier_anchor = semantic_digest_for_pair(1, 1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(verifier_anchor),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_BBBB).expect("preprocess");

    // Encode an HONEST base step from (1, 1). The structure (with the
    // anchor constraint) is satisfied because H(1,1) == verifier_anchor.
    let mut ctx = start_chain(&prep).expect("start chain");
    let mut compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect("compile honest base step from (1, 1)");

    // Now hand-roll the lie at the witness layer: flip a bit in the
    // state_in.semantic_state_digest_in_lane bits. The Poseidon
    // binding row will now be unsatisfied (lane ≠ H(app_vars)), AND
    // the new base anchor constraint will also fail.
    // Position of state_in.semantic_state_digest_in_lane: state_in.offset
    // + 4 prior digests (vk_fs, structure, z_0, z_i_in) × 4 lanes × 64 bits.
    let lane_bit_offset = compiled.encoded.image.layout.state_in.offset + 4 * 4 * POSEIDON2_GOLDILOCKS_BITS;
    compiled.encoded.witness[lane_bit_offset] = if compiled.encoded.witness[lane_bit_offset] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };

    let instance = r1cs_f_prime::build_instance(&prep, &compiled.encoded).expect("build tampered instance");

    // The attacker still computes a self-consistent x_out at the
    // lifecycle layer. The image's witness was tampered but the
    // structure hash and the prover's claim of step output are
    // arranged to be self-consistent.
    let audit = neo_fold_clean::lifecycle::prove::prove_one_with_semantic_state(
        &prep.prep,
        vec![instance],
        verifier_anchor,
        semantic_digest_for_pair(1, 2),
    );

    // The protocol must reject the tampered witness somewhere along
    // the prove → finalize → verify pipeline. Any rejection counts as
    // success; we just need NONE of these stages to silently accept.
    let rejected_by_prove = audit.is_err();
    let rejected_downstream = audit
        .ok()
        .map(|audit| {
            // Both `finish_uncompressed` (terminal NIFS.P sees the
            // unsatisfied fresh CCS and rejects) and `verify_uncompressed`
            // (Π_CCS sumcheck rejects) are valid rejection paths. Catch
            // either via Result.
            let finalize_result = neo_fold_clean::finish_uncompressed(&prep.prep, audit);
            match finalize_result {
                Err(_) => true,
                Ok(finished) => neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            }
        })
        .unwrap_or(false);
    assert!(
        rejected_by_prove || rejected_downstream,
        "BASE-ANCHOR SOUNDNESS (post-encoder tamper): the protocol accepted a witness whose \
         state_in.semantic_state_digest_in_lane bits were flipped after honest encoding. \
         Π_CCS sumcheck (run by either the prover-side NIFS satisfaction check at finalize \
         time or the verifier's terminal NIFS replay) must reject."
    );
}

#[test]
fn r1cs_stateful_attack_is_base_zero_does_not_make_base_anchor_optional() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let verifier_anchor = semantic_digest_for_pair(1, 1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(verifier_anchor),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_CCCC).expect("preprocess");

    let mut ctx = start_chain(&prep).expect("start chain");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect("compile honest base step");
    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "honest base image must satisfy before tamper"
    );

    let mut tampered = compiled.encoded.witness.clone();
    let is_base_col = compiled.encoded.image.layout.is_base.offset;
    tampered[is_base_col] = F::ZERO;
    let semantic_lane_bit_offset = compiled.encoded.image.layout.state_in.offset + 4 * 4 * POSEIDON2_GOLDILOCKS_BITS;
    tampered[semantic_lane_bit_offset] = if tampered[semantic_lane_bit_offset] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };

    assert!(
        !compiled.encoded.structure.is_satisfied(&tampered),
        "setting is_base=0 must not make a base image with a tampered semantic-state input lane satisfy"
    );
}
