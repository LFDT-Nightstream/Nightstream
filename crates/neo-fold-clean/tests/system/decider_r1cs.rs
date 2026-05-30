//! Direct-CCS full-history audit R1CS synthesis tests.
//!
//! These tests pin the self-contained audit relation for the current
//! direct-CCS interim path. The relation replays every lifecycle/F' step,
//! so it is linear in chain length. It is not the constant-size IVC
//! terminal decider that the future `F'` frontend should enable, and no
//! test here exercises a SNARK prover/verifier.
//!
//! `engine::decider::synthesize_statement_r1cs` builds the full audit
//! relation, with every layer reflected on [`DeciderR1csSynthesis`]:
//!
//!   - **Base F' step** emitted (`base_step_emitted == true`).
//!   - **Base-state seeds pinned** to canonical preprocessing-derived
//!     constants in-circuit (`base_state_pinned == true`).
//!   - **Recursive F' steps** for every `FoldProof::Recursive` in the
//!     witness (`recursive_step_count == N - 1`).
//!   - **Cross-step state-link** between every adjacent pair
//!     (`cross_step_links == N - 1`).
//!   - **CE-claim continuity links** wiring every recursive step's
//!     NIFS children to the next step's (or terminal fold's) running
//!     (`accumulator_claim_links == recursive_step_count`). Goes beyond
//!     commitment-only `acc_digest` continuity by pinning every CE
//!     field wire-for-wire.
//!   - **Terminal NIFS.V** under `FINAL_FOLD_TRANSCRIPT_LABEL`
//!     (`terminal_fold_emitted == true`).
//!   - **Terminal latest link**: `terminal_fresh.x[0] == 1` and
//!     `fresh.x[1..] == last.x_out_bits` (`terminal_latest_link == true`).
//!   - **Public-image pins**: the base-state seed pins
//!     `initial_semantic_state_digest`; `public_image_pins ==
//!     REQUIRED_PUBLIC_IMAGE_PINS` covers the remaining terminal fields.
//!
//! When all of the above hold and the builder is satisfied,
//! [`DeciderR1csSynthesis::is_self_sufficient_relation`] returns
//! `true`. That means the full-history audit relation is complete; it
//! does not mean the production IVC decider has landed.

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::decider::{synthesize_statement_r1cs, REQUIRED_PUBLIC_IMAGE_PINS};
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, State};
use neo_fold_clean::paper::digest::{
    accumulator_digest_from_claims, digest32_as_fields, initial_boundary_digest, public_trace_seed_digest,
    state_x_out_digest, structure_digest,
};
use neo_fold_clean::paper::f_prime::r1cs::{encode_f_prime_public_input, F_PRIME_PUBLIC_INPUT_LEN};
use neo_fold_clean::CcsInstance;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Fixture helpers ─────────────────────────────────────────────────────────

fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_PUBLIC_INPUT_LEN,
    }
}

fn compute_x_out_native(prep: &neo_fold_clean::Preprocessing, state: &State) -> [F; 4] {
    digest32_as_fields(state_x_out_digest(
        prep.vk.digest(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
    ))
}

fn base_state(prep: &neo_fold_clean::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = accumulator_digest_from_claims(prep.params.b(), &[]);
    State::base(z_0, public_trace, acc_digest, acc_digest)
}

fn build_link_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, x_out_target: [F; 4]) -> CcsInstance {
    let mut z = encode_f_prime_public_input(x_out_target);
    z.resize(prep.structure().m, F::ZERO);
    direct_ccs::build_instance(prep, r1cs, &z).expect("recursive-link instance")
}

fn peek_next_state(prep: &neo_fold_clean::Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
    let (next, _) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state.clone(),
        batch.to_vec(),
    )
    .expect("peek");
    next
}

/// Build an F'-honest finished proof of `len` batches: each batch's `x`
/// encodes the `prior_x_out` the F' R1CS will recompute for the next
/// step.
fn build_honest_finished_proof(len: usize) -> (neo_fold_clean::Preprocessing, neo_fold_clean::UncompressedAudit) {
    assert!(len >= 1);
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_inst = || direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy");

    let mut state = base_state(&prep);
    let mut steps = Vec::with_capacity(len);
    let mut public_batches: Vec<Vec<neo_fold_clean::paper::relations::CcsClaim>> = Vec::with_capacity(len);

    for _ in 0..len {
        let predicted = peek_next_state(&prep, &state, &[dummy_inst()]);
        let target_x_out = compute_x_out_native(&prep, &predicted);
        let batch = build_link_instance(&prep, &r1cs, target_x_out);
        let public_batch = vec![batch.claim.clone()];

        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            state,
            vec![batch],
        )
        .expect("step");

        steps.push(step_proof);
        public_batches.push(public_batch);
        state = next_state;
    }

    let in_flight = neo_fold_clean::UncompressedAudit {
        proof: neo_fold_clean::Uncompressed {
            state,
            final_fold: None,
        },
        steps,
        public_batches,
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, in_flight).expect("finish");
    (prep, finished)
}

/// Like [`build_honest_finished_proof`] but with an explicit per-chunk
/// batch size schedule, so the decider replays steps whose
/// `nifs_msg.fresh.len()` (previous chunk) differs from `rows_in_chunk`
/// (current chunk). Every fresh instance in a chunk encodes the same
/// `prior_x_out` (one chunk rooted at one prior Construction-2 state).
fn build_honest_finished_proof_with_sizes(
    sizes: &[usize],
) -> (neo_fold_clean::Preprocessing, neo_fold_clean::UncompressedAudit) {
    assert!(!sizes.is_empty());
    assert!(sizes.iter().all(|&k| k >= 1));
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_batch = |k: usize| -> Vec<CcsInstance> {
        (0..k)
            .map(|_| direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy"))
            .collect()
    };

    let mut state = base_state(&prep);
    let mut steps = Vec::with_capacity(sizes.len());
    let mut public_batches: Vec<Vec<neo_fold_clean::paper::relations::CcsClaim>> = Vec::with_capacity(sizes.len());

    for &k in sizes {
        // Predict the post-step state with a *correctly-sized* dummy batch
        // so chunk_digest and step_count advance match the real fold.
        let predicted = peek_next_state(&prep, &state, &dummy_batch(k));
        let target_x_out = compute_x_out_native(&prep, &predicted);
        // All K fresh in this chunk encode the same prior x_out.
        let batch: Vec<CcsInstance> = (0..k)
            .map(|_| build_link_instance(&prep, &r1cs, target_x_out))
            .collect();
        let public_batch: Vec<_> = batch.iter().map(|i| i.claim.clone()).collect();

        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            state,
            batch,
        )
        .expect("step");

        steps.push(step_proof);
        public_batches.push(public_batch);
        state = next_state;
    }

    let in_flight = neo_fold_clean::UncompressedAudit {
        proof: neo_fold_clean::Uncompressed {
            state,
            final_fold: None,
        },
        steps,
        public_batches,
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, in_flight).expect("finish");
    (prep, finished)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn decider_r1cs_synthesis_accepts_finished_statement() {
    // 2 batches → 1 recursive step. Smallest case that emits F' R1CS at all.
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");
    assert!(synth.recursive_step_count >= 1, "must emit at least one F' R1CS step");
    assert!(
        synth.builder.is_satisfied(),
        "F' R1CS chain rejected an honest finished statement (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );
}

#[test]
fn decider_r1cs_synthesis_rejects_tampered_public_image() {
    let (prep, finished) = build_honest_finished_proof(2);
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Flip a byte of the declared public image without touching the
    // witness. `validate_witness`'s walk will derive the correct public
    // image and the comparison will reject.
    statement.public.acc_digest[0] ^= 0xFF;

    assert!(
        matches!(
            synthesize_statement_r1cs(&prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::PublicImageMismatch)
        ),
        "synthesis accepted a tampered public image"
    );
}

#[test]
fn decider_r1cs_synthesis_rejects_tampered_step_proof() {
    let (prep, finished) = build_honest_finished_proof(3);
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Tamper a recursive step's NIFS proof (mutate one CE child's
    // commitment). The walk inside `validate_witness` runs NIFS.V at
    // this step and rejects.
    let recursive_idx = statement
        .witness
        .steps
        .iter()
        .position(|s| matches!(s.fold, neo_fold_clean::paper::construction2::FoldProof::Recursive(_)))
        .expect("at least one recursive step");
    if let neo_fold_clean::paper::construction2::FoldProof::Recursive(ref mut nifs) =
        statement.witness.steps[recursive_idx].fold
    {
        nifs.pi_dec.children[0].c.data[0] += F::ONE;
    }

    let result = synthesize_statement_r1cs(&prep, &statement);
    let err = match result {
        Ok(_) => panic!("must reject tampered step proof"),
        Err(e) => e,
    };
    assert!(
        matches!(err, neo_fold_clean::paper::decider::Error::WalkFailed(_)),
        "expected WalkFailed for tampered NIFS proof, got {err:?}"
    );
}

#[test]
fn decider_r1cs_synthesis_replays_all_recursive_steps() {
    // 5 batches → step 0 is NoFold (base), steps 1..4 are Recursive. So
    // the synth must emit 1 base step, 4 recursive F' R1CS gates, and 4
    // cross-step links (base→rec1, rec1→rec2, rec2→rec3, rec3→rec4).
    let (prep, finished) = build_honest_finished_proof(5);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");
    assert!(synth.base_step_emitted, "base step must be emitted in-circuit");
    assert!(
        synth.base_state_pinned,
        "base seed wires must be pinned to canonical preprocessing values"
    );
    assert_eq!(
        synth.recursive_step_count, 4,
        "5-batch chain should emit 4 recursive F' R1CS gates; got {}",
        synth.recursive_step_count
    );
    assert_eq!(
        synth.cross_step_links, 4,
        "5-batch chain should emit 4 cross-step state links (one per adjacent pair); got {}",
        synth.cross_step_links
    );
    assert_eq!(
        synth.accumulator_claim_links, synth.recursive_step_count,
        "every recursive step's children must be linked to the next step's (or terminal fold's) running"
    );
    assert!(
        synth.terminal_latest_link,
        "terminal latest must be pinned to last F' step's x_out_bits"
    );
    assert!(synth.terminal_fold_emitted);
    assert_eq!(synth.public_image_pins, REQUIRED_PUBLIC_IMAGE_PINS);
    assert!(
        synth.builder.is_satisfied(),
        "F' R1CS chain rejected a step (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );
}

#[test]
fn decider_r1cs_synthesis_emits_base_step_and_links_terminal_latest() {
    // Positive assertion of the self-sufficient full-history audit relation:
    //   - base_step_emitted == true: F' R1CS base step in-circuit.
    //   - recursive_step_count == N-1 for an N-batch chain.
    //   - cross_step_links == N-1: every adjacent F' step pair is wired
    //     prev.state_out == next.state_in.
    //   - terminal_latest_link == true: terminal fresh.x[0]==1 and
    //     fresh.x[1..]==last.x_out_bits.
    //   - terminal_fold_emitted == true: terminal NIFS.V in-circuit.
    //   - public_image_pins == REQUIRED_PUBLIC_IMAGE_PINS (10): every
    //     terminal field of `statement.public` is bound to chain-derived
    //     wires; `initial_semantic_state_digest` is pinned by base_state_pinned.
    //
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");

    assert!(synth.base_step_emitted, "base F' step must be emitted in-circuit");
    assert!(
        synth.base_state_pinned,
        "base seed wires (vk_fs, structure, z_0, public_trace, …) must be pinned to \
         canonical preprocessing-derived constants in-circuit"
    );
    assert_eq!(synth.recursive_step_count, 1, "2-batch chain has 1 recursive step");
    assert_eq!(
        synth.cross_step_links, 1,
        "2-batch chain has 1 cross-step link (base→rec1)"
    );
    assert_eq!(
        synth.accumulator_claim_links, synth.recursive_step_count,
        "every recursive step's NIFS children must be linked to the next step's (or terminal fold's) running"
    );
    assert!(
        synth.terminal_latest_link,
        "terminal fold's fresh.x must be pinned to last F' step's x_out_bits"
    );
    assert!(
        synth.terminal_fold_emitted,
        "terminal fold must be emitted in-circuit for a self-sufficient audit relation"
    );
    assert_eq!(
        synth.public_image_pins, REQUIRED_PUBLIC_IMAGE_PINS,
        "all {REQUIRED_PUBLIC_IMAGE_PINS} public-image fields must be pinned in-circuit"
    );
    assert_eq!(
        REQUIRED_PUBLIC_IMAGE_PINS, 10,
        "pin_public_image covers 10 terminal PublicImage fields; \
         initial_semantic_state_digest is covered by base_state_pinned"
    );
    assert!(
        synth.builder.is_satisfied(),
        "self-sufficient full-history audit relation rejected an honest statement \
         (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );
}

#[test]
fn decider_r1cs_synthesis_is_self_sufficient_full_history_audit_relation() {
    // Single-call readiness gate: every completeness flag/count is at
    // its full value and the R1CS builder is satisfied. This is a
    // direct-CCS audit marker, not the constant-size IVC decider.
    let (prep, finished) = build_honest_finished_proof(5);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");

    assert!(synth.base_step_emitted);
    assert!(synth.base_state_pinned);
    assert_eq!(synth.recursive_step_count, 4);
    assert_eq!(synth.cross_step_links, synth.recursive_step_count);
    assert_eq!(synth.accumulator_claim_links, synth.recursive_step_count);
    assert!(synth.terminal_latest_link);
    assert!(synth.terminal_fold_emitted);
    assert_eq!(synth.public_image_pins, REQUIRED_PUBLIC_IMAGE_PINS);
    assert!(
        synth.is_self_sufficient_relation(),
        "self-sufficient full-history audit relation gate failed"
    );
}

#[test]
fn decider_r1cs_links_full_ce_accumulator_claims() {
    // Full CE-claim continuity between every adjacent NIFS step. For an
    // N-batch chain (1 base + (N-1) recursive + terminal fold), the
    // expected count is `accumulator_claim_links == recursive_step_count`
    // because:
    //   - the base step has no NIFS.V output, so no link from it;
    //   - each subsequent recursive step links `prev.children ==
    //     next.running`;
    //   - the terminal fold links `last_recursive.children ==
    //     terminal.running`.
    //
    // The structural count is what prevents silently relying on the
    // commitment-only `acc_digest` continuity for chain soundness. Each
    // link enforces wire-for-wire equality across (c_data, x, r, y_ring,
    // y_zcol, s_col, fold_digest_fields) — far beyond what
    // `acc_digest` (commitment-data Poseidon only) covers.
    for batches in [2, 3, 5] {
        let (prep, finished) = build_honest_finished_proof(batches);
        let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
        let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");

        assert_eq!(
            synth.accumulator_claim_links, synth.recursive_step_count,
            "{batches}-batch chain: accumulator_claim_links ({}) must equal \
             recursive_step_count ({})",
            synth.accumulator_claim_links, synth.recursive_step_count,
        );
        assert!(
            synth.builder.is_satisfied(),
            "{batches}-batch CE continuity rejected an honest statement \
             (first bad row: {:?})",
            synth.builder.first_unsatisfied_row()
        );
    }
}

#[test]
fn decider_r1cs_synthesis_accepts_varying_size_batched_chunks() {
    // Batch schedule [2, 3]: the recursive F' R1CS step replays a fold
    // whose `nifs_msg.fresh.len()` (= the *previous* chunk it folds, 2)
    // differs from `rows_in_chunk` (= the *current* chunk's deposit, 3).
    // This is the case the `rows_in_chunk` field exists for: the F' R1CS
    // must advance `step_count` by the current chunk (3), and the decider
    // pins the recomputed `state_out` against the cross-step link and the
    // public image — so a regression that advanced by `fresh.len()` (2)
    // would land on the wrong `step_count` and make the builder
    // unsatisfied. It also exercises `enforce_terminal_latest_link`
    // looping over a 3-wide terminal fresh batch.
    let (prep, finished) = build_honest_finished_proof_with_sizes(&[2, 3]);

    // The chain folds 2 fresh at the recursive step and flushes 3 fresh
    // against the k_rho running at finalize.
    assert_eq!(finished.public_batches.len(), 2);
    assert_eq!(finished.public_batches[0].len(), 2);
    assert_eq!(finished.public_batches[1].len(), 3);

    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");

    assert_eq!(synth.recursive_step_count, 1, "[2,3] chain has 1 recursive step");
    assert!(
        synth.terminal_latest_link,
        "terminal fold must link all 3 terminal fresh public inputs to the last step's x_out_bits"
    );
    assert!(
        synth.builder.is_satisfied(),
        "decider rejected an honest varying-size batched chain (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );
    // Cross-check the public image's step_count: 2 + 3 = 5 ops total.
    assert_eq!(statement.public.step_count, 5, "total ops folded = 2 + 3 = 5");
}

// The previous end-to-end "tamper Z, bypass preflight, expect
// `!is_satisfied`" test has been replaced by the gadget-level
// isolation tests in `tests/system/decider_ce_relation_isolation.rs`.
// Those tests hit the CE-relation gadget directly through the narrow
// `engine::decider::__test_isolation` surface and prove each of the
// five obligations (commit / X / low-norm / y_ring / ct) is load-bearing
// in isolation, without needing to disable the chain-level
// `validate_witness` preflight.
