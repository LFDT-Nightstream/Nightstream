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

use std::time::{Duration, Instant};

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::engine::decider::synthesize_statement_r1cs;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::compiler::FPrimeShellCompilerError;
use neo_fold_clean::frontends::f_prime::compiler::{
    assemble_shared_chunk_traces, assemble_step_from_shared, nifs_payload_inputs_for_source_image, perp_nifs_ce_view,
};
use neo_fold_clean::frontends::f_prime::image::NifsPayloadShape;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, assignment_to_bits, compile_step, encode_r1cs_f_prime_step, start_chain, R1csChainBuilder, R1csCompilerError,
    R1csEncoderInput, R1csFPrimeStepInput,
};
use neo_fold_clean::paper::digest::{digest32_as_fields, digest_fields_as_digest32};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_ENC_INST_OFFSET;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, assignment_one_product_with_extras, make_stateful_plan, make_stateful_plan_with_anchor,
    make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs, tiny_params,
};

fn expect_f_prime_non_replay_unsupported(err: neo_fold_clean::Error, chunk_count: u64) {
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::FPrimeNonReplayUnsupported { chunk_count: got } if got == chunk_count
        ),
        "expected FPrimeNonReplayUnsupported({chunk_count}), got {err:?}"
    );
}

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
        vec![0, 7],
        Some(semantic_digest_for_single(42)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A01).expect("preprocess");
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
        vec![0, 7],
        Some(semantic_digest_for_single(42)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A04).expect("preprocess");
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
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A02).expect("preprocess");
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
fn r1cs_stateful_chain_builder_rejects_disconnected_chunk_before_stashing_fold() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A22).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let first = chain
        .append_assignment(assignment_fibonacci_transition(1, 1))
        .expect("base append");

    let err = chain
        .append_assignment(assignment_fibonacci_transition(5, 2))
        .expect_err("disconnected semantic state must reject");
    match err {
        r1cs_f_prime::Error::Compiler(R1csCompilerError::SemanticStateInputMismatch { expected, got }) => {
            assert_eq!(expected, first.semantic_state_digest_out);
            assert_ne!(got, expected);
        }
        other => panic!("expected SemanticStateInputMismatch, got {other:?}"),
    }
    assert!(
        chain.context().fold_for_step.is_none(),
        "semantic mismatch must reject before prepare_next_fold stashes recursive fold authority"
    );

    chain
        .append_assignment(assignment_fibonacci_transition(1, 2))
        .expect("builder remains usable after rejected disconnected chunk");
    let audit = chain.finish_with_audit().expect("finish");
    neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit).expect("repaired multi-step stateful audit verifies");
    let finished = neo_fold_clean::finish_uncompressed(&prep.prep, audit).expect("finish terminal-only proof");
    let err = neo_fold_clean::verify_uncompressed(&prep.prep, &finished)
        .expect_err("multi-chunk F' proof needs audit/decider path");
    expect_f_prime_non_replay_unsupported(err, 2);
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
fn r1cs_stateful_fibonacci_rejects_forged_constant_lane() {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A15).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let mut forged = vec![F::ZERO; neo_math::D];
    forged[0] = F::from_u64(2);
    forged[1] = F::ONE;
    forged[2] = F::ONE;
    forged[3] = F::from_u64(2);
    forged[4] = F::from_u64(4);
    r1cs.is_satisfied_by(&forged)
        .expect("raw R1CS accepts if z[0] is allowed to impersonate the constant-one lane");

    let err = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment: forged })
        .expect_err("F' structure must pin the conventional z[0] constant lane");
    match err {
        R1csCompilerError::ConstantLaneNotOne { got } => assert_eq!(got, F::from_u64(2)),
        other => panic!("expected ConstantLaneNotOne, got {other:?}"),
    }
}

#[test]
fn r1cs_stateful_chain_builder_rejects_parallel_chunk() {
    let r1cs = one_product_r1cs();
    let plan = make_stateful_plan(r1cs.m(), r1cs.m_in, vec![1], vec![0]);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0A03).expect("preprocess");
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

#[test]
fn r1cs_stateful_serial_k2_fibonacci_step_satisfies_and_binds_intermediate_link() {
    let r1cs = fibonacci_two_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![5, 6],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_2F10).expect("preprocess");
    let mut ctx = start_chain(&prep).expect("start chain");

    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_two_transitions(1, 1),
        },
    )
    .expect("base serial-K2 Fibonacci step");

    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "honest serial-K2 Fibonacci image must satisfy its F' structure"
    );
    assert_eq!(
        digest_fields_as_digest32(compiled.semantic_state_digest_out),
        semantic_digest_for_pair(2, 3),
        "one F' step should advance two Fibonacci transitions: (1,1) -> (2,3)"
    );

    // Tamper only the intermediate state a_1. It is not part of the
    // chunk's semantic output `(a_2,b_2)`, so rejection must come from
    // the in-circuit app R1CS rows that enforce
    // `(a_1,b_1) = (b_0,a_0+b_0)`.
    let mut tampered = compiled.encoded.witness.clone();
    let intermediate_a_slot = prep.anchors().app_var_slots[3];
    tampered[intermediate_a_slot.bit_start] = if tampered[intermediate_a_slot.bit_start] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    let row = compiled
        .encoded
        .structure
        .first_unsatisfied_row(&tampered)
        .expect("intermediate-link tamper must reject");
    let r1cs_rows = prep.anchors().r1cs_row_start..prep.anchors().r1cs_row_start + prep.anchors().r1cs_row_count;
    assert!(
        r1cs_rows.contains(&row),
        "intermediate-link tamper should trip an appended app-R1CS row; got {row}, expected {r1cs_rows:?}"
    );
}

#[test]
fn r1cs_stateful_serial_k2_fibonacci_chain_verifies_two_chunks() {
    let r1cs = fibonacci_two_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![5, 6],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_2F11).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");

    let first = chain
        .append_assignment(assignment_fibonacci_two_transitions(1, 1))
        .expect("append first serial-K2 chunk");
    assert_eq!(
        digest_fields_as_digest32(first.semantic_state_digest_out),
        semantic_digest_for_pair(2, 3)
    );

    let second = chain
        .append_assignment(assignment_fibonacci_two_transitions(2, 3))
        .expect("append second serial-K2 chunk");
    assert_eq!(
        digest_fields_as_digest32(second.semantic_state_digest_out),
        semantic_digest_for_pair(5, 8),
        "two F' chunks should cover four Fibonacci transitions"
    );

    let audit = chain.finish_with_audit().expect("finish serial-K2 chain");
    assert_eq!(
        audit.steps.len(),
        2,
        "serial-K2 app circuit should use two F' folds for four app transitions"
    );
    neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit)
        .expect("multi-step serial-K2 stateful audit verifies");
    let finished_audit =
        neo_fold_clean::finish_uncompressed_with_audit(&prep.prep, audit).expect("finish uncompressed");
    let err = neo_fold_clean::verify_uncompressed(&prep.prep, &finished_audit.proof)
        .expect_err("terminal-only verifier must reject multi-chunk F' proof");
    expect_f_prime_non_replay_unsupported(err, 2);
}

#[test]
#[ignore = "stateful Fibonacci perf snapshot; run manually to compare K=1 versus serial-K2 app circuits"]
fn r1cs_stateful_serial_k2_fibonacci_perf_snapshot_compares_four_transitions() {
    let k1 = time_fibonacci_k1_four_transitions();
    let k2 = time_fibonacci_serial_k2_four_transitions();

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  Stateful Fibonacci serial batching perf snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    app transitions                  4");
    eprintln!("    K=1 F' chunks                    {}", k1.folds);
    eprintln!("    serial-K2 F' chunks              {}", k2.folds);
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!("    stage                 K=1 four chunks   serial-K2 two chunks");
    eprintln!("    -------------------   -------------   --------------------");
    eprintln!(
        "    preprocess            {:>10.3}             {:>10.3}",
        ms(k1.preprocess),
        ms(k2.preprocess)
    );
    eprintln!(
        "    append total          {:>10.3}             {:>10.3}",
        ms(k1.append_total),
        ms(k2.append_total)
    );
    eprintln!(
        "    finish                {:>10.3}             {:>10.3}",
        ms(k1.finish),
        ms(k2.finish)
    );
    eprintln!(
        "    verify                {:>10.3}             {:>10.3}",
        ms(k1.verify),
        ms(k2.verify)
    );
    eprintln!(
        "    total                 {:>10.3}             {:>10.3}",
        ms(k1.total),
        ms(k2.total)
    );
    eprintln!();
    eprintln!("  Throughput (app transitions/s)");
    eprintln!("    K=1                  {:>10.2}", 4.0 / k1.total.as_secs_f64());
    eprintln!("    serial-K2            {:>10.2}", 4.0 / k2.total.as_secs_f64());
    eprintln!("======================================================================");

    assert_eq!(
        k1.final_semantic_digest,
        semantic_digest_for_pair(5, 8),
        "K=1 baseline should end at Fibonacci state (5,8)"
    );
    assert_eq!(
        k2.final_semantic_digest, k1.final_semantic_digest,
        "serial-K2 and K=1 should prove the same final Fibonacci state"
    );
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

/// Two serial Fibonacci transitions inside one app R1CS:
///
/// ```text
/// (a_0, b_0) -> (a_1, b_1) -> (a_2, b_2)
/// where
/// a_1 = b_0
/// b_1 = a_0 + b_0
/// a_2 = b_1
/// b_2 = a_1 + b_1
/// ```
///
/// Variable layout: `[one, a_0, b_0, a_1, b_1, a_2, b_2, ...]`.
fn fibonacci_two_transition_stateful_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(4, m, F::default());
    a[(0, 2)] = F::ONE; // b_0
    a[(1, 1)] = F::ONE; // a_0
    a[(1, 2)] = F::ONE; // + b_0
    a[(2, 4)] = F::ONE; // b_1
    a[(3, 3)] = F::ONE; // a_1
    a[(3, 4)] = F::ONE; // + b_1

    let mut b = NeoMat::zero(4, m, F::default());
    for row in 0..4 {
        b[(row, 0)] = F::ONE; // * one
    }

    let mut c = NeoMat::zero(4, m, F::default());
    c[(0, 3)] = F::ONE; // a_1 = b_0
    c[(1, 4)] = F::ONE; // b_1 = a_0 + b_0
    c[(2, 5)] = F::ONE; // a_2 = b_1
    c[(3, 6)] = F::ONE; // b_2 = a_1 + b_1

    R1cs { a, b, c, m_in: 3 }
}

fn assignment_fibonacci_two_transitions(a: u64, b: u64) -> Vec<F> {
    let a1 = b;
    let b1 = a + b;
    let a2 = b1;
    let b2 = a1 + b1;
    let mut z = vec![F::ZERO; neo_math::D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a1);
    z[4] = F::from_u64(b1);
    z[5] = F::from_u64(a2);
    z[6] = F::from_u64(b2);
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

struct FibonacciPerfSnapshot {
    folds: usize,
    preprocess: Duration,
    append_total: Duration,
    finish: Duration,
    verify: Duration,
    total: Duration,
    final_semantic_digest: [u8; 32],
}

fn time_fibonacci_k1_four_transitions() -> FibonacciPerfSnapshot {
    let total_start = Instant::now();
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let start = Instant::now();
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_4B10).expect("preprocess K=1");
    let preprocess = start.elapsed();

    let mut chain = R1csChainBuilder::new(&prep).expect("start K=1 chain");
    let mut append_total = Duration::ZERO;
    let mut final_semantic_digest = semantic_digest_for_pair(1, 1);
    for (a, b) in [(1, 1), (1, 2), (2, 3), (3, 5)] {
        let start = Instant::now();
        let compiled = chain
            .append_assignment(assignment_fibonacci_transition(a, b))
            .expect("append K=1 Fibonacci transition");
        append_total += start.elapsed();
        final_semantic_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }

    let start = Instant::now();
    let audit = chain.finish_with_audit().expect("finish K=1 chain");
    let finish = start.elapsed();

    let start = Instant::now();
    neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit).expect("K=1 stateful audit verifies");
    let verify = start.elapsed();

    FibonacciPerfSnapshot {
        folds: 4,
        preprocess,
        append_total,
        finish,
        verify,
        total: total_start.elapsed(),
        final_semantic_digest,
    }
}

fn time_fibonacci_serial_k2_four_transitions() -> FibonacciPerfSnapshot {
    let total_start = Instant::now();
    let r1cs = fibonacci_two_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![5, 6],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let start = Instant::now();
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_4B20)
        .expect("preprocess serial-K2");
    let preprocess = start.elapsed();

    let mut chain = R1csChainBuilder::new(&prep).expect("start serial-K2 chain");
    let mut append_total = Duration::ZERO;
    let mut final_semantic_digest = semantic_digest_for_pair(1, 1);
    for (a, b) in [(1, 1), (2, 3)] {
        let start = Instant::now();
        let compiled = chain
            .append_assignment(assignment_fibonacci_two_transitions(a, b))
            .expect("append serial-K2 Fibonacci chunk");
        append_total += start.elapsed();
        final_semantic_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }

    let start = Instant::now();
    let audit = chain.finish_with_audit().expect("finish serial-K2 chain");
    let finish = start.elapsed();

    let start = Instant::now();
    neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit).expect("serial-K2 stateful audit verifies");
    let verify = start.elapsed();

    FibonacciPerfSnapshot {
        folds: 2,
        preprocess,
        append_total,
        finish,
        verify,
        total: total_start.elapsed(),
        final_semantic_digest,
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        prep.public_input_len,
        prep.enforces_f_prime_recursive_link(),
        prep.semantic_state_mode(),
        prep.initial_semantic_state_digest(),
        None,
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

fn build_single_fibonacci_fixture() -> LinkedFibonacciFixture {
    let r1cs = fibonacci_transition_stateful_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest_for_pair(1, 1)),
    );
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_0F11).expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled = chain
        .append_assignment(assignment_fibonacci_transition(1, 1))
        .expect("append base Fibonacci transition");
    let audit = chain
        .finish_with_audit()
        .expect("finish single-step Fibonacci chain");
    LinkedFibonacciFixture {
        prep,
        audit,
        step_semantic_out: vec![compiled.semantic_state_digest_out],
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

    let err = neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &fixture.audit.proof)
        .expect_err("terminal-only verifier must reject multi-chunk F' proof");
    expect_f_prime_non_replay_unsupported(err, 4);
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &fixture.audit)
        .expect("multi-step stateful audit verifies");

    let statement = neo_fold_clean::build_decider_statement(&fixture.prep.prep, &fixture.audit);
    validate_decider_statement(&fixture.prep.prep, &statement).expect("decider preflight accepts stateful audit");
}

#[test]
fn r1cs_stateful_audit_rejects_intermediate_public_input_not_linked_to_prior_x_out() {
    let mut fixture = build_linked_fibonacci_fixture();
    fixture.audit.public_batches[1][0].x[F_PRIME_ENC_INST_OFFSET] += F::ONE;

    let err = neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &fixture.audit)
        .expect_err("audit replay must reject an intermediate F' public input not linked to prior x_out");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Decider(
                neo_fold_clean::paper::decider::Error::TerminalLatestPublicInputMismatch { index: 0 }
            )
        ),
        "expected intermediate F' latest-link mismatch, got {err:?}"
    );
}

#[test]
fn r1cs_stateful_terminal_fold_rejects_latest_public_input_not_linked_to_pre_final_x_out() {
    let fixture = build_single_fibonacci_fixture();
    let mut finished =
        neo_fold_clean::finish_uncompressed(&fixture.prep.prep, fixture.audit).expect("finish single-step Fibonacci");
    let final_fold = finished
        .final_fold
        .as_mut()
        .expect("single-step proof carries terminal final_fold");
    final_fold.terminal_inputs.latest.instances[0].claim.x[F_PRIME_ENC_INST_OFFSET] += F::ONE;

    let err = neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &finished)
        .expect_err("terminal verifier must reject F' public input not linked to pre-final x_out");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::TerminalLatestPublicInputMismatch { index: 0 }
        ),
        "expected terminal F' latest-link mismatch, got {err:?}"
    );
}

#[test]
fn r1cs_stateful_terminal_only_rejects_chunk_count_relabel_to_small_counts() {
    let fixture = build_linked_fibonacci_fixture();
    let finished = neo_fold_clean::finish_uncompressed(&fixture.prep.prep, fixture.audit.clone())
        .expect("finalize linked Fibonacci");

    let err = neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &finished)
        .expect_err("honest multi-chunk F' proof must require audit/decider path");
    expect_f_prime_non_replay_unsupported(err, 4);

    // Hacker model: `verify_uncompressed` fails closed for multi-chunk F'
    // proofs by checking `state.chunk_count > 1`. Relabeling the compact
    // state down to one chunk must not make the terminal-only verifier
    // accept; the terminal fold's verifier-derived `x_out` still has to
    // bind the real chain counters.
    let mut one_chunk = finished.clone();
    one_chunk.state.chunk_count = 1;
    neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &one_chunk)
        .expect_err("chunk_count relabel must break terminal-fold state binding, not bypass stateful scope");

    // Relabeling all the way to zero is a distinct branch: it also disables
    // the compact `public_trace == z_i` anchor guarded by `chunk_count > 0`.
    // Terminal-fold state binding must still reject.
    let mut zero_chunk = finished;
    zero_chunk.state.chunk_count = 0;
    neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &zero_chunk)
        .expect_err("chunk_count=0 relabel must not bypass stateful scope or compact public-trace binding");
}

#[test]
fn r1cs_stateful_step_proof_semantic_digest_tamper_rejects_audit() {
    let mut fixture = build_single_fibonacci_fixture();
    fixture.audit.steps[0].semantic_state_digest[0] ^= 0xFF;

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
    let fixture = build_single_fibonacci_fixture();
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
    let fixture = build_single_fibonacci_fixture();
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
    let fixture = build_single_fibonacci_fixture();
    let finished = neo_fold_clean::finish_uncompressed(&fixture.prep.prep, fixture.audit.clone())
        .expect("finalize linked Fibonacci");

    // Honest single-step stateful proofs are still accepted: the base
    // anchor binds the private app-state input to the verifier-owned
    // initial semantic digest, so no recursive state link is needed.
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &fixture.audit)
        .expect("honest audit verifier accepts");
    neo_fold_clean::verify_uncompressed(&fixture.prep.prep, &finished)
        .expect("terminal-only verifier accepts single-step stateful proof");

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

#[test]
fn r1cs_stateful_attack_low_level_base_is_base_zero_forge_rejected_by_verifier() {
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
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_EA5E).expect("preprocess");

    let forged_in = semantic_digest_for_pair(10, 10);
    let forged_out = semantic_digest_for_pair(10, 20);
    let mut ctx = start_chain(&prep).expect("start chain");
    ctx.chain_state.semantic_state_digest = digest32_as_fields(forged_in);

    let assignment = assignment_fibonacci_transition(10, 10);
    let shared = assemble_shared_chunk_traces(&ctx, false, ctx.chain_state.acc_digest, 1);
    let assembly = assemble_step_from_shared(&shared, &ctx, &[], Some(digest32_as_fields(forged_out)));
    let ce_shape = match prep
        .plan()
        .nifs_payload_shapes
        .first()
        .expect("plan must have CE payload")
    {
        NifsPayloadShape::CeClaim(shape) => shape.clone(),
        NifsPayloadShape::CcsClaim(_) => panic!("test expects CE payload shape"),
    };
    let encoded = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_r1cs_f_prime_step(
            R1csEncoderInput {
                plan: prep.plan().clone(),
                boundary_bits: assembly.boundary_bits,
                state_in: assembly.state_in,
                state_out: assembly.state_out,
                chunk_digest: assembly.chunk_digest,
                assignment_bits: assignment_to_bits(&assignment),
                is_base: false,
                nifs_payloads: nifs_payload_inputs_for_source_image(prep.plan(), perp_nifs_ce_view(&ce_shape)),
                kmul_views: vec![],
                ring_action_pairs: vec![],
                one_shot_traces: vec![
                    encode_poseidon_trace(&build_semantic_state_preimage_fields(&[
                        F::from_u64(10),
                        F::from_u64(10),
                    ])),
                    encode_poseidon_trace(&build_semantic_state_preimage_fields(&[
                        F::from_u64(10),
                        F::from_u64(20),
                    ])),
                    assembly.traces.state_x_out,
                ],
                sponge_trace: None,
            },
            std::sync::Arc::clone(prep.structure()),
        )
    })) {
        Ok(encoded) => encoded,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&'static str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("<non-string panic>");
            assert!(
                message.contains("encoded R1CS F' step must satisfy its structure"),
                "unexpected panic while encoding the low-level is_base forge: {message}"
            );
            return;
        }
    };
    let instance = r1cs_f_prime::build_instance(&prep, &encoded).expect("build forged low-level instance");
    let proof = neo_fold_clean::lifecycle::prove::prove_one_with_semantic_state(
        &prep.prep,
        vec![instance],
        verifier_anchor,
        forged_out,
    )
    .expect("prove forged low-level base image");
    let finished = neo_fold_clean::finish_uncompressed(&prep.prep, proof).expect("finish forged low-level image");
    let err = neo_fold_clean::verify_uncompressed(&prep.prep, &finished)
        .expect_err("verifier accepted a base image with is_base=0 and a forged initial app state");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PostStateMismatch
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "expected verifier to reject forged initial app state, got {err:?}"
    );
}

/// Red-team test for the production folded F' image's recursive state binding.
///
/// This bypasses the friendly [`R1csChainBuilder`] state-threading guard
/// and constructs two individually satisfying stateful F' images whose
/// app states do not connect:
///
/// ```text
/// step 0: (1, 1)   -> (1, 2)
/// step 1: (10, 10) -> (10, 20)   // disconnected from step 0
/// ```
///
/// The hostile path generates a recursive NIFS proof under the honest
/// post-step semantic state, then tries to replay it while compiling a
/// disconnected second step whose private `state_in.semantic_state_digest`
/// is self-consistent with `(10, 10)`. F' must reject because the per-step
/// transcript is bound to the state-in semantic lane; otherwise the proof
/// can be replayed under a forged app-state input.
#[test]
fn r1cs_stateful_redteam_folded_f_prime_rejects_disconnected_semantic_input() {
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
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_D15C).expect("preprocess");

    // Honest first step: H(1,1) -> H(1,2).
    let mut honest_ctx = start_chain(&prep).expect("start honest context");
    let first = compile_step(
        &prep,
        &mut honest_ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(1, 1),
        },
    )
    .expect("compile honest first step");
    let first_instance = r1cs_f_prime::build_instance(&prep, &first.encoded).expect("first instance");
    let audit_after_first = neo_fold_clean::lifecycle::prove::prove_one_with_semantic_state(
        &prep.prep,
        vec![first_instance.clone()],
        verifier_anchor,
        semantic_digest_for_pair(1, 2),
    )
    .expect("prove first step");

    // Derive the recursive fold authority for the next chunk, but ask
    // the lifecycle state to advance to the attacker's disconnected
    // semantic output H(10,20). Replaying this proof under H(10,10) as
    // the next step's private state-in must fail because F' binds the
    // NIFS transcript to the state-in semantic lane.
    let disconnected_output = semantic_digest_for_pair(10, 20);
    let pending = neo_fold_clean::lifecycle::prove::extend_with_semantic_state(
        &prep.prep,
        audit_after_first.clone(),
        vec![first_instance.clone()],
        disconnected_output,
    )
    .expect("derive recursive fold authority");

    let pre_state = audit_after_first.proof.state.clone();
    let (pre_running, latest) = match &pre_state.proof {
        neo_fold_clean::paper::construction2::ProofState::Active { running, latest } => {
            (running.clone(), latest.clone())
        }
        _ => panic!("after first step the chain must be Active"),
    };
    let proof = match &pending.steps.last().expect("recursive step appended").fold {
        neo_fold_clean::paper::construction2::FoldProof::Recursive(nifs) => nifs.clone(),
        neo_fold_clean::paper::construction2::FoldProof::NoFold => {
            panic!("second lifecycle step must be recursive")
        }
    };
    let post_running = match &pending.proof.state.proof {
        neo_fold_clean::paper::construction2::ProofState::Active { running, .. } => running.clone(),
        _ => panic!("pending recursive state must be Active"),
    };

    // Forge the compiler context for step 1. All Construction-2 fold
    // coordinates are copied from the honest chain, but the private
    // semantic input lane is rewound to H(10,10), allowing the F' image
    // for the disconnected app assignment to satisfy its local semantic
    // Poseidon rows.
    let mut forged_ctx = start_chain(&prep).expect("start forged context");
    forged_ctx.chain_state = r1cs_f_prime::R1csChainState {
        chunk_count: pre_state.chunk_count,
        step_count: pre_state.step_count,
        z_i: digest32_as_fields(pre_state.z_i),
        semantic_state_digest: digest32_as_fields(semantic_digest_for_pair(10, 10)),
        acc_digest: digest32_as_fields(pre_state.acc_digest),
        public_trace: digest32_as_fields(pre_state.public_trace),
    };
    forged_ctx.fold_for_step = Some(r1cs_f_prime::R1csFoldForStep {
        pre_running,
        latest,
        proof,
        post_running,
    });

    let err = compile_step(
        &prep,
        &mut forged_ctx,
        R1csFPrimeStepInput {
            assignment: assignment_fibonacci_transition(10, 10),
        },
    )
    .expect_err("compile disconnected second step via forged context must reject");
    assert!(
        matches!(
            err,
            R1csCompilerError::Shell(FPrimeShellCompilerError::PriorFoldVerificationFailed { .. })
        ),
        "expected prior-fold verification to reject the semantic-state transcript replay, got {err:?}"
    );
}
