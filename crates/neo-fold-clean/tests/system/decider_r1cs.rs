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
//!     (`accumulator_claim_links == recursive_step_count`). This is an
//!     explicit wire-for-wire continuity check for the paper-level CE core,
//!     not a substitute for the exact-running `acc_digest` handle. The
//!     optimized `y_zcol` source relation remains open.
//!   - **Parent-authority continuity links** wiring every recursive step's
//!     Π_RLC parent authority to the next step's (or terminal fold's)
//!     running parent (`parent_authority_links == recursive_step_count`).
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

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use neo_ajtai::{setup as setup_ajtai, AjtaiSModule};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_fold_clean::engine::decider::{
    __test_isolation::{
        enforce_base_state_constants_against, enforce_ce_continuity_against_self, enforce_ce_continuity_between,
        enforce_public_image_pins_against, enforce_public_image_pins_against_chain, enforce_state_link_against_self,
        enforce_terminal_fold_against_last_acc_digest, enforce_terminal_fold_ce_closure_against,
        enforce_terminal_fold_children_continuity_against_self, enforce_terminal_fold_parent_authority_against_self,
        enforce_terminal_latest_link_against, CeContinuityProbeWires,
    },
    synthesize_last_step_terminal_r1cs, synthesize_statement_r1cs, REQUIRED_PUBLIC_IMAGE_PINS,
};
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, EncInst, ProofState, State, TRIVIAL_PC};
use neo_fold_clean::paper::decider::{self, PublicImage};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, initial_boundary_digest, public_trace_seed_digest,
    state_x_out_digest_with_mode, structure_digest, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_superneo_public_input, F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::terminal_ce::{TerminalCeProof, TerminalCePublic};
use neo_fold_clean::CcsInstance;
use neo_math::{D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use serde_json::{json, Value};

const FULL_HISTORY_MANIFEST_PATH: &str = "formal/nightstream-lean/assurance/fprime-full-history-program-manifest.json";
const FULL_HISTORY_LEAN_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryManifestData.lean";
const FULL_HISTORY_TOP_LEVEL: &[&str] = &[
    "decider.step.base",
    "decider.step.recursive",
    "decider.state_link",
    "decider.terminal_fold",
    "decider.terminal_continuity",
    "decider.public_pins",
    "decider.terminal_ce",
];

// ── Fixture helpers ─────────────────────────────────────────────────────────

fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    }
}

fn compute_x_out_native(prep: &neo_fold_clean::Preprocessing, state: &State) -> [F; 4] {
    let mode = match prep.semantic_state_mode() {
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    ))
}

fn public_image_pin_fixture(prep: &neo_fold_clean::Preprocessing) -> PublicImage {
    let z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);
    let z_i = [7u8; 32];
    let acc_digest = AccumulatorHandle::empty().digest();
    let mut public = PublicImage {
        vk_fs_digest: prep.vk.digest(),
        chunk_count: 1,
        step_count: 1,
        z_0,
        z_i,
        pc: TRIVIAL_PC,
        initial_semantic_state_digest: prep.initial_semantic_state_digest(),
        semantic_state_digest: acc_digest,
        acc_digest,
        public_trace: z_i,
        x_out: EncInst::from_digest([0u8; 32]),
    };
    refresh_public_image_x_out(prep, &mut public);
    public
}

fn refresh_public_image_x_out(prep: &neo_fold_clean::Preprocessing, public: &mut PublicImage) {
    let mode = match prep.semantic_state_mode() {
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    public.x_out = EncInst::from_digest(state_x_out_digest_with_mode(
        mode,
        public.vk_fs_digest,
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        public.chunk_count,
        public.step_count,
        public.z_0,
        public.z_i,
        public.pc,
        public.semantic_state_digest,
        public.acc_digest,
        public.public_trace,
        None,
    ));
}

fn add_field_modulus_to_digest_limb(digest: &mut [u8; 32], limb: usize) {
    let start = limb * 8;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[start..start + 8]);
    let value = u64::from_le_bytes(bytes);
    let aliased = value
        .checked_add(F::ORDER_U64)
        .expect("test fixture digest limb must be small enough to alias");
    digest[start..start + 8].copy_from_slice(&aliased.to_le_bytes());
}

fn base_state(prep: &neo_fold_clean::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = AccumulatorHandle::empty().digest();
    State::base(z_0, public_trace, acc_digest, acc_digest)
}

fn build_link_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, x_out_target: [F; 4]) -> CcsInstance {
    let z = encode_f_prime_superneo_public_input(x_out_target);
    direct_ccs::build_instance(prep, r1cs, &z).expect("recursive-link instance")
}

fn peek_next_state(prep: &neo_fold_clean::Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
    let (next, _) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
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
    build_honest_finished_proof_with_prep(prep, &r1cs, len)
}

fn build_honest_finished_proof_with_prep(
    prep: neo_fold_clean::Preprocessing,
    r1cs: &R1cs,
    len: usize,
) -> (neo_fold_clean::Preprocessing, neo_fold_clean::UncompressedAudit) {
    assert!(len >= 1);
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_inst = || direct_ccs::build_instance(&prep, r1cs, &placeholder_z).expect("dummy");

    let mut state = base_state(&prep);
    let mut steps = Vec::with_capacity(len);
    let mut public_batches: Vec<Vec<neo_fold_clean::paper::relations::CcsClaim>> = Vec::with_capacity(len);

    for _ in 0..len {
        let predicted = peek_next_state(&prep, &state, &[dummy_inst()]);
        let target_x_out = compute_x_out_native(&prep, &predicted);
        let batch = build_link_instance(&prep, r1cs, target_x_out);
        let public_batch = vec![batch.claim.clone()];

        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
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
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
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
    let unconstrained = synth.builder.unconstrained_columns();
    assert!(
        unconstrained.is_empty(),
        "full-history decider R1CS allocated columns that never appear in any row: {:?}",
        unconstrained
    );
}

#[test]
fn decider_r1cs_honors_explicit_verifier_owned_ajtai_setup() {
    let r1cs = bit_carrier_r1cs();

    // Install the process-global setup that the ordinary direct-CCS helper
    // uses, then deliberately build this verifier context around a distinct,
    // dimension-compatible owned setup. `preprocess_with_test_log` explicitly
    // supports this adversarial-fixture context.
    let canonical = direct_ccs::preprocess_seeded(&r1cs, 42).expect("canonical preprocess");
    let params = canonical.params.clone();
    let structure = canonical.structure().clone();
    let public_input_len = canonical.public_input_len;
    let cols = structure.m.div_ceil(D);
    let mut rng = ChaCha20Rng::from_seed([0x93; 32]);
    let owned_pp = setup_ajtai(&mut rng, D, params.kappa() as usize, cols).expect("owned Ajtai setup");
    let owned_log = AjtaiSModule::new(Arc::new(owned_pp));
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, owned_log, public_input_len)
        .expect("preprocess with verifier-owned log");

    let (prep, finished) = build_honest_finished_proof_with_prep(prep, &r1cs, 1);

    // The audit verifier and the decider's native preflight both accept the
    // honestly generated proof under the setup owned by `prep`.
    neo_fold_clean::verify_uncompressed_audit(&prep, &finished).expect("owned-log audit must pass native verification");
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    decider::validate_witness(
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
        prep.enforces_terminal_induction(),
        prep.semantic_state_mode(),
        prep.initial_semantic_state_digest(),
        prep.nebula(),
        &statement,
    )
    .expect("owned-log decider preflight must pass");

    let ProofState::Active { running, latest } = &statement.witness.final_state.proof else {
        panic!("finished audit must have an active running accumulator");
    };
    assert!(
        latest.instances.is_empty(),
        "finished audit must have no pending latest"
    );
    let running = running.materialize().expect("materialized final running");
    assert!(
        running
            .claims
            .iter()
            .zip(&running.witnesses)
            .all(|(claim, witness)| prep.log.commit(witness) == claim.c),
        "fixture must open every final claim under prep.log"
    );
    let global_log = AjtaiSModule::from_global_for_dims(D, cols).expect("process-global Ajtai setup");
    assert!(
        running
            .claims
            .iter()
            .zip(&running.witnesses)
            .any(|(claim, witness)| global_log.commit(witness) != claim.c),
        "fixture must distinguish the verifier-owned and process-global setups"
    );

    // Regression contract: synthesis must encode the same verifier-owned
    // commitment map that native preflight just accepted. Today the terminal
    // CE gadget silently reloads the process-global PP, making these honest
    // rows unsatisfied.
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize after successful preflight");
    assert!(
        synth.builder.is_satisfied(),
        "NF-RT-093: decider R1CS substituted the process-global Ajtai PP for prep.log \
         (first bad row: {:?})",
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
        let mut proof = nifs
            .materialize()
            .expect("recursive NIFS proof materialization");
        proof.pi_dec.children[0].c.data[0] += F::ONE;
        *nifs = neo_fold_clean::paper::nifs::NifsProofCarrier::materialized(proof);
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
    assert_eq!(
        synth.parent_authority_links, synth.recursive_step_count,
        "every recursive step's parent authority must be linked to the next step's (or terminal fold's) parent"
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
    assert_eq!(
        synth.parent_authority_links, synth.recursive_step_count,
        "every recursive step's NIFS parent must be linked to the next step's (or terminal fold's) parent authority"
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
fn decider_r1cs_synthesis_rejects_missing_terminal_fold() {
    let (prep, finished) = build_honest_finished_proof(2);
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    assert!(
        statement.witness.final_fold.is_some(),
        "test setup must start with a real terminal fold"
    );

    statement.witness.final_fold = None;
    let err = synthesize_statement_r1cs(&prep, &statement)
        .err()
        .expect("decider R1CS synthesis must fail closed without terminal final_fold");
    match err {
        neo_fold_clean::paper::decider::Error::WalkFailed(reason) => {
            assert!(
                reason.contains("terminal final_fold"),
                "unexpected missing-terminal-fold diagnostic: {reason}"
            );
        }
        other => panic!("expected WalkFailed for missing terminal fold, got {other:?}"),
    }
}

#[test]
fn decider_base_state_pins_reject_tampered_seed_wires() {
    let (prep, _finished) = build_honest_finished_proof(1);
    let cases: [(
        &str,
        fn(&neo_fold_clean::engine::decider::__test_isolation::BaseStateProbeWires) -> usize,
    ); 10] = [
        ("vk_fs_digest", |p| p.vk_fs0.col()),
        ("structure_digest", |p| p.structure0.col()),
        ("chunk_count", |p| p.chunk_count.col()),
        ("step_count", |p| p.step_count.col()),
        ("z_0", |p| p.z_0_0.col()),
        ("z_i", |p| p.z_i_0.col()),
        ("pc", |p| p.pc.col()),
        ("initial_semantic_state_digest", |p| p.semantic0.col()),
        ("acc_digest", |p| p.acc0.col()),
        ("public_trace", |p| p.public_trace0.col()),
    ];

    for (name, probe_col) in cases {
        let (mut builder, probes) = enforce_base_state_constants_against(&prep, prep.initial_semantic_state_digest());
        assert!(
            builder.is_satisfied(),
            "honest base-state seed pins must satisfy before {name} tamper (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );

        let target_col = probe_col(&probes);
        builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "base-state pins accepted a {name} wire diverging from the verifier-owned seed"
        );
    }
}

#[test]
fn decider_public_image_pins_reject_coherent_preprocessing_anchor_relabels() {
    let prep = direct_ccs::preprocess_seeded(&bit_carrier_r1cs(), 42).expect("preprocess");
    let honest = public_image_pin_fixture(&prep);
    let builder = enforce_public_image_pins_against(&prep, &honest);
    assert!(
        builder.is_satisfied(),
        "honest public-image pin fixture must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let cases: [(&str, fn(&mut PublicImage)); 3] = [
        ("vk_fs_digest", |p| p.vk_fs_digest[0] ^= 0x01),
        ("z_0", |p| p.z_0[0] ^= 0x02),
        ("initial_semantic_state_digest", |p| {
            p.initial_semantic_state_digest[0] ^= 0x04
        }),
    ];

    for (name, tamper) in cases {
        let mut public = honest.clone();
        tamper(&mut public);
        refresh_public_image_x_out(&prep, &mut public);
        let builder = enforce_public_image_pins_against(&prep, &public);
        assert!(
            !builder.is_satisfied(),
            "public-image pins accepted a coherent relabel of verifier-owned {name}"
        );
    }
}

#[test]
fn decider_public_image_pins_reject_coherent_terminal_relabels() {
    let prep = direct_ccs::preprocess_seeded(&bit_carrier_r1cs(), 42).expect("preprocess");
    let chain = public_image_pin_fixture(&prep);
    let builder = enforce_public_image_pins_against_chain(&prep, &chain, &chain);
    assert!(
        builder.is_satisfied(),
        "honest public-image pin fixture must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let cases: [(&str, fn(&mut PublicImage), bool); 8] = [
        ("chunk_count", |p| p.chunk_count += 1, true),
        ("step_count", |p| p.step_count += 1, true),
        ("z_i", |p| p.z_i[0] ^= 0x08, true),
        ("pc", |p| p.pc += 1, true),
        ("semantic_state_digest", |p| p.semantic_state_digest[0] ^= 0x10, true),
        ("acc_digest", |p| p.acc_digest[0] ^= 0x20, true),
        ("public_trace", |p| p.public_trace[0] ^= 0x40, true),
        ("x_out", |p| p.x_out = EncInst::from_digest([0x80; 32]), false),
    ];

    for (name, tamper, refresh_x_out) in cases {
        let mut public = chain.clone();
        tamper(&mut public);
        if refresh_x_out {
            refresh_public_image_x_out(&prep, &mut public);
        }
        let builder = enforce_public_image_pins_against_chain(&prep, &chain, &public);
        assert!(
            !builder.is_satisfied(),
            "public-image pins accepted a coherent relabel of terminal field {name}"
        );
    }
}

#[test]
fn decider_public_image_pins_reject_field_modulus_u64_aliases() {
    let prep = direct_ccs::preprocess_seeded(&bit_carrier_r1cs(), 42).expect("preprocess");
    let chain = public_image_pin_fixture(&prep);
    let q = F::ORDER_U64;

    let cases: [(&str, fn(&mut PublicImage, u64)); 3] = [
        ("chunk_count", |p, q| p.chunk_count += q),
        ("step_count", |p, q| p.step_count += q),
        ("pc", |p, q| p.pc += q),
    ];

    for (name, tamper) in cases {
        let mut public = chain.clone();
        tamper(&mut public, q);
        // Deliberately keep x_out unchanged. Before this regression was
        // fixed, `pin_u64` compared through `F::from_u64`, so public values
        // offset by the Goldilocks modulus were indistinguishable from the
        // honest chain wire and the whole public-image pin family accepted.
        let builder = enforce_public_image_pins_against_chain(&prep, &chain, &public);
        assert!(
            !builder.is_satisfied(),
            "public-image pins accepted {name} relabeled by the Goldilocks modulus"
        );
    }
}

#[test]
fn decider_public_image_pins_reject_noncanonical_digest_limb_aliases() {
    let prep = direct_ccs::preprocess_seeded(&bit_carrier_r1cs(), 42).expect("preprocess");
    let mut chain = public_image_pin_fixture(&prep);
    let small_digest = digest_fields_as_digest32([F::ONE, F::ZERO, F::ZERO, F::ZERO]);
    chain.z_i = small_digest;
    chain.public_trace = small_digest;
    refresh_public_image_x_out(&prep, &mut chain);

    let mut public = chain.clone();
    add_field_modulus_to_digest_limb(&mut public.z_i, 0);
    // Keep x_out unchanged. Before this regression was fixed, the public
    // z_i bytes decoded to the same field element as the honest chain wire,
    // so the public image could carry a non-canonical digest limb and still
    // satisfy every pin.
    let builder = enforce_public_image_pins_against_chain(&prep, &chain, &public);
    assert!(
        !builder.is_satisfied(),
        "public-image pins accepted a non-canonical digest limb alias"
    );
}

#[test]
fn decider_r1cs_synthesis_is_self_sufficient_full_history_audit_relation() {
    // Single-call readiness gate: every completeness flag/count is at
    // its full value and the R1CS builder is satisfied. This is a
    // direct-CCS audit marker, not the constant-size IVC decider.
    let (prep, finished) = build_honest_finished_proof(5);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    let mut synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize");

    assert!(synth.base_step_emitted);
    assert!(synth.base_state_pinned);
    assert_eq!(synth.recursive_step_count, 4);
    assert_eq!(synth.cross_step_links, synth.recursive_step_count);
    assert_eq!(synth.accumulator_claim_links, synth.recursive_step_count);
    assert_eq!(synth.parent_authority_links, synth.recursive_step_count);
    assert!(synth.terminal_latest_link);
    assert!(synth.terminal_fold_emitted);
    assert!(synth.terminal_ce_direct_relations);
    assert_eq!(synth.public_image_pins, REQUIRED_PUBLIC_IMAGE_PINS);
    assert!(
        synth.is_self_sufficient_relation(),
        "self-sufficient full-history audit relation gate failed"
    );

    synth.terminal_ce_direct_relations = false;
    assert!(
        !synth.is_self_sufficient_relation(),
        "self-sufficient full-history audit relation must require direct terminal CE rows"
    );

    synth.terminal_ce_direct_relations = true;
    synth.parent_authority_links -= 1;
    assert!(
        !synth.is_self_sufficient_relation(),
        "self-sufficient full-history audit relation must require parent-authority continuity links"
    );
}

#[test]
fn decider_r1cs_synthesis_rejects_unsupported_terminal_ce_proof_material() {
    let (prep, finished) = build_honest_finished_proof(2);
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    statement.witness.terminal_ce_proof = Some(TerminalCeProof::new_unchecked([F::ZERO; 4], vec![0xA5, 0xCE]));

    let err = synthesize_statement_r1cs(&prep, &statement)
        .err()
        .expect("compact terminal CE proof bytes must fail closed until a real verifier is wired");
    assert!(
        matches!(err, decider::Error::TerminalCeProofUnsupported),
        "expected TerminalCeProofUnsupported, got {err:?}"
    );
}

#[test]
fn decider_r1cs_synthesis_rejects_matching_terminal_ce_proof_until_backend_exists() {
    let (prep, finished) = build_honest_finished_proof(2);
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let terminal_children = statement
        .witness
        .final_fold
        .as_ref()
        .expect("finished proof carries terminal fold")
        .nifs
        .pi_dec
        .children
        .clone();
    let terminal_public = TerminalCePublic::from_terminal_children(&prep.params, prep.structure(), &terminal_children)
        .expect("honest terminal children form compact terminal CE public statement");
    statement.witness.terminal_ce_proof = Some(TerminalCeProof::new_unchecked(
        terminal_public.digest(),
        vec![0xA5, 0xCE],
    ));

    let err = synthesize_statement_r1cs(&prep, &statement)
        .err()
        .expect("well-bound compact terminal CE proof material must still fail closed");
    assert!(
        matches!(err, decider::Error::TerminalCeProofUnsupported),
        "expected TerminalCeProofUnsupported, got {err:?}"
    );
}

#[test]
fn decider_last_step_terminal_synthesis_requires_terminal_ce_relation_rows() {
    let (prep, finished) = build_honest_finished_proof(2);

    let synth = synthesize_last_step_terminal_r1cs(&prep, &finished).expect("terminal synth");

    assert!(synth.has_final_fold);
    assert_eq!(synth.public_image_pins, REQUIRED_PUBLIC_IMAGE_PINS);
    assert!(
        synth.terminal_ce_direct_relations,
        "last-step terminal synthesis must emit direct terminal CE-relation rows until a compact proof verifier replaces them"
    );
    assert!(
        synth.builder.is_satisfied(),
        "last-step terminal synthesis rejected an honest proof (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );
    let unconstrained = synth.builder.unconstrained_columns();
    assert!(
        unconstrained.is_empty(),
        "last-step terminal decider R1CS allocated columns that never appear in any row: {:?}",
        unconstrained
    );
}

#[test]
fn decider_r1cs_links_full_ce_accumulator_claims() {
    // Full CE-claim continuity between every adjacent NIFS step. For an
    // N-batch chain (1 base + (N-1) recursive + terminal fold), the
    // expected count is `*_links == recursive_step_count` because:
    //   - the base step has no NIFS.V output, so no link from it;
    //   - each subsequent recursive step links `prev.children ==
    //     next.running` and `prev.parent == next.running_parent`;
    //   - the terminal fold links `last_recursive.children ==
    //     terminal.running` and `last_recursive.parent ==
    //     terminal.running_parent`.
    //
    // The structural count prevents silently reducing chain soundness to
    // any single compact handle. Each link enforces wire-for-wire equality
    // across the CE fields consumed by the next step or terminal fold.
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
        assert_eq!(
            synth.parent_authority_links, synth.recursive_step_count,
            "{batches}-batch chain: parent_authority_links ({}) must equal \
             recursive_step_count ({})",
            synth.parent_authority_links, synth.recursive_step_count,
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
fn decider_terminal_fold_rejects_tampered_last_acc_digest() {
    let (prep, finished) = build_honest_finished_proof(3);
    let final_fold = finished
        .proof
        .final_fold
        .as_ref()
        .expect("finished proof has terminal fold");
    let pre_running = &final_fold.terminal_inputs.pre_final_running;
    let trailing_latest = final_fold.terminal_inputs.latest.claims();
    let honest_last_acc_digest = if pre_running.claims.is_empty() {
        AccumulatorHandle::empty().digest()
    } else {
        let parent = pre_running
            .parent_authority
            .as_ref()
            .expect("non-empty pre-final running has parent authority");
        AccumulatorHandle::from_running_parts(&pre_running.claims, Some(parent)).digest()
    };

    let honest = enforce_terminal_fold_against_last_acc_digest(
        &prep,
        pre_running,
        &trailing_latest,
        &final_fold.nifs,
        honest_last_acc_digest,
    )
    .expect("honest terminal-fold isolation emits");
    assert!(
        honest.is_satisfied(),
        "honest terminal fold consumed-handle rows must satisfy (first bad row: {:?})",
        honest.first_unsatisfied_row()
    );

    let mut tampered = honest_last_acc_digest;
    tampered[0] ^= 0xFF;
    let bad =
        enforce_terminal_fold_against_last_acc_digest(&prep, pre_running, &trailing_latest, &final_fold.nifs, tampered)
            .expect("tampered terminal-fold isolation emits");
    assert!(
        !bad.is_satisfied(),
        "terminal fold accepted a last-step accumulator handle that does not match its consumed running"
    );
}

#[test]
fn decider_terminal_ce_rejects_tampered_reattached_child_y_zcol() {
    let (prep, finished) = build_honest_finished_proof(3);
    let final_fold = finished
        .proof
        .final_fold
        .as_ref()
        .expect("finished proof has terminal fold");
    let pre_running = &final_fold.terminal_inputs.pre_final_running;
    let trailing_latest = final_fold.terminal_inputs.latest.claims();
    let final_running = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running");
    let last_acc_digest =
        AccumulatorHandle::from_running_parts(&pre_running.claims, pre_running.parent_authority.as_ref()).digest();

    let honest = enforce_terminal_fold_ce_closure_against(
        &prep,
        pre_running,
        &trailing_latest,
        &final_fold.nifs,
        last_acc_digest,
        &final_running.witnesses,
    )
    .expect("emit honest terminal CE closure");
    assert!(
        honest.is_satisfied(),
        "honest terminal CE closure must satisfy (first bad row: {:?})",
        honest.first_unsatisfied_row()
    );

    let mut tampered_nifs = final_fold.nifs.clone();
    tampered_nifs.pi_dec.children[0].y_zcol[0] += K::ONE;
    let tampered = enforce_terminal_fold_ce_closure_against(
        &prep,
        pre_running,
        &trailing_latest,
        &tampered_nifs,
        last_acc_digest,
        &final_running.witnesses,
    )
    .expect("emit tampered terminal CE closure");
    assert!(
        !tampered.is_satisfied(),
        "terminal CE closure accepted a tampered reattached child y_zcol"
    );
}

#[test]
fn decider_terminal_fold_rejects_tampered_last_parent_authority_wire() {
    let (prep, finished) = build_honest_finished_proof(3);
    let final_fold = finished
        .proof
        .final_fold
        .as_ref()
        .expect("finished proof has terminal fold");
    let pre_running = &final_fold.terminal_inputs.pre_final_running;
    let trailing_latest = final_fold.terminal_inputs.latest.claims();
    let parent = pre_running
        .parent_authority
        .as_ref()
        .expect("pre-final running has parent authority");
    let honest_last_acc_digest = AccumulatorHandle::from_running_parts(&pre_running.claims, Some(parent)).digest();

    let (mut builder, probes) = enforce_terminal_fold_parent_authority_against_self(
        &prep,
        pre_running,
        &trailing_latest,
        &final_fold.nifs,
        honest_last_acc_digest,
    )
    .expect("honest terminal-fold parent-authority isolation emits");
    assert!(
        builder.is_satisfied(),
        "honest terminal parent-authority rows must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let target_col = probes.last_parent_y_ring_c1.col();
    builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "terminal fold accepted a last-step parent-authority y_ring limb that diverged from terminal running"
    );
}

#[test]
fn decider_terminal_fold_rejects_tampered_last_child_wire() {
    let (prep, finished) = build_honest_finished_proof(3);
    let final_fold = finished
        .proof
        .final_fold
        .as_ref()
        .expect("finished proof has terminal fold");
    let pre_running = &final_fold.terminal_inputs.pre_final_running;
    let trailing_latest = final_fold.terminal_inputs.latest.claims();
    let parent = pre_running
        .parent_authority
        .as_ref()
        .expect("pre-final running has parent authority");
    let honest_last_acc_digest = AccumulatorHandle::from_running_parts(&pre_running.claims, Some(parent)).digest();

    let (mut builder, probes) = enforce_terminal_fold_children_continuity_against_self(
        &prep,
        pre_running,
        &trailing_latest,
        &final_fold.nifs,
        honest_last_acc_digest,
    )
    .expect("honest terminal-fold child-continuity isolation emits");
    assert!(
        builder.is_satisfied(),
        "honest terminal child-continuity rows must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let target_col = probes.last_child_y_ring_c1.col();
    builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "terminal fold accepted a last-step child y_ring limb that diverged from terminal running"
    );
}

#[test]
fn decider_ce_continuity_rejects_tampered_shape_metadata_wire() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let cases: [(&str, fn(&CeContinuityProbeWires) -> usize); 5] = [
        ("c_d", |p| p.c_d.col()),
        ("c_kappa", |p| p.c_kappa.col()),
        ("x_rows", |p| p.x_rows.col()),
        ("x_cols", |p| p.x_cols.col()),
        ("m_in", |p| p.m_in.col()),
    ];
    for (name, probe_col) in cases {
        let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
        assert!(
            builder.is_satisfied(),
            "honest CE-continuity isolation must satisfy before {name} tamper (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );

        let target_col = probe_col(&probes);
        builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "CE continuity accepted a running-side {name} metadata wire that diverged from the child"
        );
    }
}

#[test]
fn decider_ce_continuity_rejects_tampered_commitment_and_x_wires() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let cases: [(&str, fn(&CeContinuityProbeWires) -> usize); 2] =
        [("c_data[0]", |p| p.c_data0.col()), ("x[0]", |p| p.x0.col())];
    for (name, probe_col) in cases {
        let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
        assert!(
            builder.is_satisfied(),
            "honest CE-continuity isolation must satisfy before {name} tamper (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );

        let target_col = probe_col(&probes);
        builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "CE continuity accepted a running-side {name} wire that diverged from the child"
        );
    }
}

#[test]
fn decider_ce_continuity_rejects_tampered_point_limbs() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let cases: [(&str, fn(&CeContinuityProbeWires) -> usize); 4] = [
        ("r.c0", |p| p.r_c0.col()),
        ("r.c1", |p| p.r_c1.col()),
        ("s_col.c0", |p| p.s_col_c0.col()),
        ("s_col.c1", |p| p.s_col_c1.col()),
    ];
    for (name, probe_col) in cases {
        let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
        assert!(
            builder.is_satisfied(),
            "honest CE-continuity isolation must satisfy before {name} tamper (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );

        let target_col = probe_col(&probes);
        builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "CE continuity accepted a running-side {name} limb that diverged from the child"
        );
    }
}

#[test]
fn decider_ce_continuity_rejects_tampered_fold_digest_wire() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
    assert!(
        builder.is_satisfied(),
        "honest CE-continuity isolation must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let target_col = probes.fold_digest0.col();
    builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "CE continuity accepted a running-side fold_digest lane that diverged from the child"
    );
}

#[test]
fn decider_ce_continuity_rejects_tampered_ct_c1_limb() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
    assert!(
        builder.is_satisfied(),
        "honest CE-continuity isolation must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let target_col = probes.ct_c1.col();
    builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "CE continuity accepted a running-side ct.c1 limb that diverged from the child"
    );
}

#[test]
fn decider_ce_continuity_rejects_tampered_y_ring_c1_limb() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let (mut builder, probes) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
    assert!(
        builder.is_satisfied(),
        "honest CE-continuity isolation must satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let target_col = probes.y_ring_c1.col();
    builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "CE continuity accepted a running-side y_ring.c1 limb that diverged from the child"
    );
}

#[test]
fn decider_ce_continuity_omits_child_and_running_y_zcol() {
    let (_prep, finished) = build_honest_finished_proof(2);
    let claim = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running")
        .claims
        .first()
        .expect("final running has at least one claim");

    let (baseline_builder, _) = enforce_ce_continuity_against_self(claim).expect("emit continuity rows");
    assert!(
        baseline_builder.is_satisfied(),
        "honest CE-continuity isolation must satisfy (first bad row: {:?})",
        baseline_builder.first_unsatisfied_row()
    );
    let baseline = baseline_builder.snapshot();

    let mut child_mutation = claim.clone();
    child_mutation.y_zcol[0] += K::ONE;
    let (child_builder, _) = enforce_ce_continuity_between(&child_mutation, claim).expect("emit child mutation");
    let child = child_builder.snapshot();
    assert!(baseline.has_same_relation(&child));
    assert_eq!(
        baseline.witness(),
        child.witness(),
        "child y_zcol leaked into continuity"
    );

    let mut running_mutation = claim.clone();
    running_mutation.y_zcol[0] += K::ONE;
    let (running_builder, _) = enforce_ce_continuity_between(claim, &running_mutation).expect("emit running mutation");
    let running = running_builder.snapshot();
    assert!(baseline.has_same_relation(&running));
    assert_eq!(
        baseline.witness(),
        running.witness(),
        "running y_zcol leaked into continuity"
    );
}

#[test]
fn decider_state_link_rejects_tampered_state_field_wires() {
    let cases: [(
        &str,
        fn(&neo_fold_clean::engine::decider::__test_isolation::StateLinkProbeWires) -> usize,
    ); 10] = [
        ("vk_fs_digest", |p| p.vk_fs0.col()),
        ("structure_digest", |p| p.structure0.col()),
        ("chunk_count", |p| p.chunk_count.col()),
        ("step_count", |p| p.step_count.col()),
        ("z_0", |p| p.z_0_0.col()),
        ("z_i", |p| p.z_i_0.col()),
        ("pc", |p| p.pc.col()),
        ("semantic_state_digest", |p| p.semantic0.col()),
        ("acc_digest", |p| p.acc0.col()),
        ("public_trace", |p| p.public_trace0.col()),
    ];

    for (name, probe_col) in cases {
        let (mut builder, probes) = enforce_state_link_against_self();
        assert!(
            builder.is_satisfied(),
            "honest state-link isolation must satisfy before {name} tamper (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );

        let target_col = probe_col(&probes);
        builder.tamper_witness(target_col, builder.witness()[target_col] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "state link accepted a next-state {name} wire diverging from the prior state"
        );
    }
}

#[test]
fn decider_terminal_latest_link_rejects_tampered_second_fresh_bit() {
    let last_bits = vec![F::ZERO; F_PRIME_PUBLIC_INPUT_LEN - 1];
    let honest_fresh = {
        let mut x = Vec::with_capacity(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
        x.push(F::ONE);
        x.extend(last_bits.iter().copied());
        x.resize(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
        x
    };
    let mut fresh_batch = vec![honest_fresh.clone(), honest_fresh.clone(), honest_fresh];

    let honest = enforce_terminal_latest_link_against(&last_bits, &fresh_batch).expect("emit latest-link rows");
    assert!(
        honest.is_satisfied(),
        "honest terminal latest link must satisfy (first bad row: {:?})",
        honest.first_unsatisfied_row()
    );

    // Mutate the *second* fresh instance's first encoded bit while leaving
    // the last F' x_out bits and every other fresh input unchanged. A
    // regression that checks only fresh[0] or only the constant-one slots
    // would accept this.
    fresh_batch[1][F_PRIME_PUBLIC_INPUT_LEN - 1] += F::ONE;
    let bad = enforce_terminal_latest_link_against(&last_bits, &fresh_batch).expect("emit tampered latest-link rows");
    assert!(
        !bad.is_satisfied(),
        "terminal latest link accepted a tampered bit in fresh[1].x[1..]; \
         every terminal fresh instance must encode the last F' x_out"
    );
}

#[test]
fn decider_terminal_latest_link_rejects_tampered_fresh_one_slot() {
    let last_bits = vec![F::ZERO; F_PRIME_PUBLIC_INPUT_LEN - 1];
    let honest_fresh = {
        let mut x = Vec::with_capacity(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
        x.push(F::ONE);
        x.extend(last_bits.iter().copied());
        x.resize(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
        x
    };
    let mut fresh_batch = vec![honest_fresh.clone(), honest_fresh];

    let honest = enforce_terminal_latest_link_against(&last_bits, &fresh_batch).expect("emit latest-link rows");
    assert!(
        honest.is_satisfied(),
        "honest terminal latest one-slot link must satisfy (first bad row: {:?})",
        honest.first_unsatisfied_row()
    );

    // Keep every enc_inst body bit correct and mutate only the CCS
    // constant-one slot. This catches a terminal-link regression that
    // compares `fresh.x[1..]` but forgets `fresh.x[0] == 1`.
    fresh_batch[1][0] = F::ZERO;
    let bad = enforce_terminal_latest_link_against(&last_bits, &fresh_batch).expect("emit tampered latest-link rows");
    assert!(
        !bad.is_satisfied(),
        "terminal latest link accepted a fresh public input with x[0] != 1"
    );
}

#[test]
fn decider_terminal_latest_link_rejects_nonzero_carrier_padding() {
    let last_bits = vec![F::ZERO; F_PRIME_PUBLIC_INPUT_LEN - 1];
    let mut fresh = vec![F::ZERO; F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN];
    fresh[0] = F::ONE;

    let honest = enforce_terminal_latest_link_against(&last_bits, &[fresh.clone()]).expect("emit latest-link rows");
    assert!(honest.is_satisfied(), "zero-padded terminal carrier must satisfy");

    fresh[F_PRIME_PUBLIC_INPUT_LEN] = F::ONE;
    let bad = enforce_terminal_latest_link_against(&last_bits, &[fresh]).expect("emit padding rows");
    assert!(
        !bad.is_satisfied(),
        "terminal latest link accepted a nonzero verifier-fixed carrier coordinate"
    );
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

#[path = "decider_r1cs_manifest.rs"]
mod m4_manifest;
// The previous end-to-end "tamper Z, bypass preflight, expect
// `!is_satisfied`" test has been replaced by the gadget-level
// isolation tests in `tests/system/decider_ce_relation_isolation.rs`.
// Those tests hit the CE-relation gadget directly through the narrow
// `engine::decider::__test_isolation` surface and prove each terminal
// authority obligation (commit / X / low-norm / y_ring / ct, plus the
// optional NC channel when carried) is load-bearing in isolation,
// without needing to disable the chain-level
// `validate_witness` preflight.
