#[path = "../support/mod.rs"]
mod support;

use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{ProofState, RunningInstance};
use neo_fold_clean::paper::nifs::{self, NifsProof, NifsProverAdapter, NifsProverRequest};
use neo_fold_clean::CcsInstance;

/// Toy instance whose complete public ring contains one specified low-norm
/// value, so callers can produce same-shape but distinct-content batches.
fn toy_instance_with_x_value(prep: &neo_fold_clean::Preprocessing, x: F) -> CcsInstance {
    let mut z = vec![F::ZERO; prep.structure().m];
    z[0] = x;
    CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, D)
        .expect("low-norm toy instance with chosen x")
}

/// One-bit relation `z * z = z`. This gives the red-team tests a
/// shape-valid, low-norm-but-unsatisfied assignment: `z = -1`.
fn bitness_r1cs() -> R1cs {
    let mut identity_prefix = Mat::zero(1, D, F::ZERO);
    identity_prefix[(0, 0)] = F::ONE;
    R1cs {
        a: identity_prefix.clone(),
        b: identity_prefix.clone(),
        c: identity_prefix,
        m_in: D,
    }
}

fn valid_bitness_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, z: F) -> CcsInstance {
    let mut assignment = vec![F::ZERO; D];
    assignment[0] = z;
    direct_ccs::build_instance(prep, r1cs, &assignment).expect("valid bitness instance")
}

fn invalid_bitness_instance_with_valid_shape(prep: &neo_fold_clean::Preprocessing) -> CcsInstance {
    let invalid_low_norm = F::ZERO - F::ONE;
    let mut assignment = vec![F::ZERO; D];
    assignment[0] = invalid_low_norm;
    CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, D)
        .expect("shape-valid low-norm instance that intentionally violates z*z=z")
}

fn final_running(proof: &neo_fold_clean::Uncompressed) -> neo_fold_clean::RunningInstance {
    match &proof.state.proof {
        ProofState::Active { running, .. } => running.clone(),
        ProofState::Initial => panic!("finalized proof must be Active"),
    }
}

fn final_running_mut(proof: &mut neo_fold_clean::Uncompressed) -> &mut neo_fold_clean::RunningInstance {
    match &mut proof.state.proof {
        ProofState::Active { running, .. } => running,
        ProofState::Initial => panic!("finalized proof must be Active"),
    }
}

#[derive(Default)]
struct CountingCpuNifsAdapter {
    calls: usize,
}

impl NifsProverAdapter for CountingCpuNifsAdapter {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), nifs::Error> {
        self.calls += 1;
        let NifsProverRequest {
            tr,
            pp,
            s,
            cache,
            log,
            lanes,
            mix_rhos_commits,
            combine_b_pows,
            fresh,
            running,
            ..
        } = request;
        let (running, proof) = nifs::prove(
            tr,
            pp,
            s,
            cache,
            log,
            lanes,
            mix_rhos_commits,
            combine_b_pows,
            fresh,
            running,
        )?;
        Ok((running, proof))
    }
}

#[test]
fn lifecycle_default_cpu_prover_finishes() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(
        &prep,
        [
            vec![support::toy_instance(&prep, 1)],
            vec![support::toy_instance(&prep, 2)],
        ],
    )
    .expect("CPU lifecycle prove");
    neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("CPU backend finalize");
}

#[test]
fn lifecycle_nifs_adapter_covers_recursive_and_terminal_folds() {
    let prep = support::toy_preprocessing();
    let mut adapter = CountingCpuNifsAdapter::default();

    let audit = neo_fold_clean::lifecycle::prove_with_nifs_adapter(
        &prep,
        &mut adapter,
        [
            vec![support::toy_instance(&prep, 5)],
            vec![support::toy_instance(&prep, 6)],
        ],
    )
    .expect("adapter-backed lifecycle prove");
    assert_eq!(adapter.calls, 1, "two lifecycle steps should run one recursive NIFS.P");

    let finalized =
        neo_fold_clean::lifecycle::finish_uncompressed_with_audit_and_nifs_adapter(&prep, &mut adapter, audit)
            .expect("adapter-backed lifecycle finalization");
    assert_eq!(adapter.calls, 2, "finalization should run the terminal NIFS.P");

    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).expect("adapter-backed lifecycle proof verifies");
}

#[test]
fn verify_uncompressed_rejects_unfolded_trailing_latest() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 23)]])
        .expect("one-batch uncompressed proof");

    match &audit.proof.state.proof {
        ProofState::Active { latest, .. } => {
            assert!(
                !latest.instances.is_empty(),
                "test requires the current one-step-lag state to contain a trailing latest"
            );
        }
        ProofState::Initial => panic!("one-batch proof must leave base state"),
    }

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &audit.proof).is_err(),
        "verify_uncompressed accepted a proof whose trailing latest was never folded"
    );
}

#[test]
fn prove_rejects_empty_batch() {
    let prep = support::toy_preprocessing();

    let err = neo_fold_clean::prove(&prep, vec![Vec::<CcsInstance>::new()])
        .expect_err("prover must reject an empty batch before it creates an empty active state");

    assert!(
        matches!(err, neo_fold_clean::Error::EmptyBatch),
        "expected EmptyBatch, got {err:?}"
    );
}

#[test]
fn extend_rejects_noncanonical_chunk_counter() {
    let prep = support::toy_preprocessing();
    let mut audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 31)]])
        .expect("one-batch uncompressed proof");
    audit.proof.state.chunk_count = u64::MAX;

    let err = neo_fold_clean::extend(&prep, audit, vec![support::toy_instance(&prep, 32)])
        .expect_err("extending a noncanonical chunk counter must fail");

    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Construction2(neo_fold_clean::paper::construction2::Error::NoncanonicalCounter {
                counter: "chunk_count"
            })
        ),
        "expected chunk_count NoncanonicalCounter, got {err:?}"
    );
}

#[test]
fn extend_rejects_noncanonical_step_counter() {
    let prep = support::toy_preprocessing();
    let mut audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 33)]])
        .expect("one-batch uncompressed proof");
    audit.proof.state.step_count = u64::MAX;

    let err = neo_fold_clean::extend(&prep, audit, vec![support::toy_instance(&prep, 34)])
        .expect_err("extending a noncanonical step counter must fail");

    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Construction2(neo_fold_clean::paper::construction2::Error::NoncanonicalCounter {
                counter: "step_count"
            })
        ),
        "expected step_count NoncanonicalCounter, got {err:?}"
    );
}

#[test]
fn extend_rejects_chunk_counter_past_the_canonical_field_range() {
    let prep = support::toy_preprocessing();
    let mut audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 35)]])
        .expect("one-batch uncompressed proof");
    audit.proof.state.chunk_count = F::ORDER_U64 - 1;

    let err = neo_fold_clean::extend(&prep, audit, vec![support::toy_instance(&prep, 36)])
        .expect_err("the next chunk counter must remain a canonical field value");

    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Construction2(neo_fold_clean::paper::construction2::Error::CounterOverflow {
                counter: "chunk_count"
            })
        ),
        "expected chunk_count CounterOverflow, got {err:?}"
    );
}

#[test]
fn extend_rejects_step_counter_past_the_canonical_field_range() {
    let prep = support::toy_preprocessing();
    let mut audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 37)]])
        .expect("one-batch uncompressed proof");
    audit.proof.state.step_count = F::ORDER_U64 - 1;

    let err = neo_fold_clean::extend(&prep, audit, vec![support::toy_instance(&prep, 38)])
        .expect_err("the next step counter must remain a canonical field value");

    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Construction2(neo_fold_clean::paper::construction2::Error::CounterOverflow {
                counter: "step_count"
            })
        ),
        "expected step_count CounterOverflow, got {err:?}"
    );
}

#[test]
fn verify_uncompressed_rejects_forged_active_empty_without_terminal_fold() {
    let prep = support::toy_preprocessing();
    let mut forged = neo_fold_clean::prove(&prep, std::iter::empty::<Vec<CcsInstance>>())
        .expect("zero-batch constructor gives us the base state to mutate")
        .proof;

    // Bypass the prover entrypoint and forge the exact shape that used to be
    // accepted: finalized-looking Active state, empty running, empty latest,
    // and no terminal fold. The verifier must reject this artifact itself.
    forged.state.proof = ProofState::active(
        neo_fold_clean::RunningInstance::default(),
        neo_fold_clean::LatestInstance::from_instances(Vec::new()),
    );

    let err = neo_fold_clean::verify_uncompressed(&prep, &forged)
        .expect_err("verify_uncompressed must reject Active+empty+final_fold=None");
    assert!(
        matches!(err, neo_fold_clean::Error::MissingTerminalFoldProof),
        "expected MissingTerminalFoldProof, got {err:?}"
    );
}

#[test]
fn verify_uncompressed_rejects_compact_anchor_tampers() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 24)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    let cases: [(&str, fn(&mut neo_fold_clean::Uncompressed)); 3] = [
        ("z_0", |proof| proof.state.z_0[0] ^= 0xA5),
        ("pc", |proof| proof.state.pc += 1),
        ("public_trace", |proof| proof.state.public_trace[0] ^= 0x5A),
    ];

    for (label, tamper) in cases {
        let mut tampered = finished.clone();
        tamper(&mut tampered);
        let err = neo_fold_clean::verify_uncompressed(&prep, &tampered)
            .expect_err(&format!("verify_uncompressed must reject tampered {label}"));
        assert!(
            matches!(err, neo_fold_clean::Error::PostStateMismatch),
            "tampered {label} should reject through compact-state anchor binding, got {err:?}"
        );
    }
}

#[test]
fn verify_uncompressed_rejects_wrong_preprocessing_structure() {
    let honest_prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&honest_prep, vec![vec![support::toy_instance(&honest_prep, 25)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&honest_prep, proof).expect("finish uncompressed proof");

    let wrong_prep = direct_ccs::preprocess_seeded(&bitness_r1cs(), 0x51_7A_C7).expect("wrong-shape preprocessing");
    assert!(
        neo_fold_clean::verify_uncompressed(&wrong_prep, &finished).is_err(),
        "verify_uncompressed accepted a proof under a different verifier-owned structure"
    );
}

#[test]
fn verify_uncompressed_rejects_terminal_input_relabel_tampers() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 26)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");

    let cases: [(&str, fn(&mut neo_fold_clean::Uncompressed)); 3] = [
        ("terminal latest x", |proof| {
            let latest = &mut proof
                .final_fold
                .as_mut()
                .expect("final fold")
                .terminal_inputs
                .latest
                .instances[0]
                .claim;
            latest.x[0] += F::ONE;
        }),
        ("terminal latest commitment", |proof| {
            let latest = &mut proof
                .final_fold
                .as_mut()
                .expect("final fold")
                .terminal_inputs
                .latest
                .instances[0]
                .claim;
            latest.c.data[0] += F::ONE;
        }),
        ("terminal latest m_in", |proof| {
            let latest = &mut proof
                .final_fold
                .as_mut()
                .expect("final fold")
                .terminal_inputs
                .latest
                .instances[0]
                .claim;
            latest.m_in += 1;
        }),
    ];

    for (label, tamper) in cases {
        let mut tampered = finished.clone();
        tamper(&mut tampered);
        assert!(
            neo_fold_clean::verify_uncompressed(&prep, &tampered).is_err(),
            "verify_uncompressed accepted terminal-input relabel attack on {label}"
        );
    }
}

#[test]
fn finish_uncompressed_folds_trailing_latest_and_verifies() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 29)]])
        .expect("one-batch uncompressed proof");

    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");
    assert!(
        finished.final_fold.is_some(),
        "one trailing latest needs a final fold proof"
    );
    match &finished.state.proof {
        ProofState::Active { latest, .. } => assert!(latest.instances.is_empty()),
        ProofState::Initial => panic!("finished one-batch proof must be active"),
    }

    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("finished proof verifies");
}

// Audit-trail tampers (mutations of `proof.final_fold` or `proof.public_batches`
// while leaving the final running accumulator intact) are no longer in scope for
// `verify_uncompressed` — Phase 1.7 made it a non-replay IVC verifier whose
// authority is the final running accumulator and its `acc_digest`. The chain
// replay catches for those tampers now live exclusively under
// `paper::decider::validate_witness` (covered by `decider_validate_witness_*`
// tests further down in this file).

#[test]
fn verify_uncompressed_rejects_tampered_recorded_acc_digest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 33)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    finished.state.acc_digest = [0xA5; 32];

    // Phase 1.7 (revised): a tampered `state.acc_digest` is caught by the
    // post-state binding step (which compares derived chain coords incl.
    // `acc_digest` against the recorded `state`). Either
    // `PostStateMismatch` (binding step) or `AccDigestMismatch`
    // (downstream final-check) is a correct rejection.
    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed must reject a tampered acc_digest");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PostStateMismatch | neo_fold_clean::Error::AccDigestMismatch
        ),
        "expected PostStateMismatch or AccDigestMismatch, got {err:?}"
    );
}

#[test]
fn verify_uncompressed_rejects_invalid_terminal_fold_even_if_final_accumulator_opens() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![toy_instance_with_x_value(&prep, F::ZERO)],
            vec![toy_instance_with_x_value(&prep, F::ONE)],
        ],
    )
    .expect("two-batch proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish proof");

    // Keep the final running accumulator and its witnesses untouched.
    // Corrupt only the terminal NIFS proof that is supposed to justify how
    // that accumulator was derived from the pre-final running + trailing latest.
    let final_fold = finished
        .final_fold
        .as_mut()
        .expect("two-batch proof must carry a terminal fold");
    final_fold.nifs.pi_dec.children[0].c.data[0] += F::ONE;

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finished).is_err(),
        "verify_uncompressed accepted an internally valid final accumulator whose terminal \
         fold proof was invalid"
    );
}

#[test]
fn verify_uncompressed_rejects_final_accumulator_from_different_fold_history() {
    let prep = support::toy_preprocessing();

    let proof_a = neo_fold_clean::prove(
        &prep,
        vec![
            vec![toy_instance_with_x_value(&prep, F::ZERO)],
            vec![toy_instance_with_x_value(&prep, F::ONE)],
        ],
    )
    .expect("prove a");
    let mut finished_a = neo_fold_clean::finish_uncompressed(&prep, proof_a).expect("finish a");

    let proof_b = neo_fold_clean::prove(
        &prep,
        vec![
            vec![toy_instance_with_x_value(&prep, F::ONE)],
            vec![toy_instance_with_x_value(&prep, F::ZERO)],
        ],
    )
    .expect("prove b");
    let finished_b = neo_fold_clean::finish_uncompressed(&prep, proof_b).expect("finish b");

    // Splice B's final accumulator into A's proof. B's accumulator is
    // CE-valid and opens under the same preprocessing, but it is not the
    // accumulator derived by A's terminal fold transcript.
    finished_a.state.proof = finished_b.state.proof.clone();
    finished_a.state.acc_digest = finished_b.state.acc_digest;

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finished_a).is_err(),
        "verify_uncompressed accepted a valid final accumulator that came from a different \
         fold history"
    );
}

#[test]
fn prove_rejects_shape_valid_invalid_previous_fold_before_it_enters_accumulator() {
    let r1cs = bitness_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x1F17_B17).expect("bitness preprocessing");

    // This instance has the right shape, a valid low-norm witness, a valid
    // commitment opening, and the right public-input split. It is invalid
    // only algebraically: z = -1 does not satisfy z*z = z.
    let invalid_previous = invalid_bitness_instance_with_valid_shape(&prep);
    let valid_current = valid_bitness_instance(&prep, &r1cs, F::ONE);

    // Step 0 would deposit the invalid instance as `latest`; step 1 would
    // be the first point where it can enter the running accumulator. The
    // prover cannot construct that fold transcript: Π_CCS rejects the
    // unsatisfied shape-valid CCS instance before it reaches the
    // accumulator, so there is no honest-looking "correct last fold" to
    // build on top of it.
    assert!(
        neo_fold_clean::prove(&prep, vec![vec![invalid_previous], vec![valid_current]]).is_err(),
        "prove accepted a shape-valid low-norm instance that violates the folded CCS relation"
    );
}

#[test]
fn finish_uncompressed_rejects_inconsistent_already_finalized_proof() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 37)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finish uncompressed proof");

    match &mut finished.proof.state.proof {
        ProofState::Active { latest, .. } => {
            latest.instances.push(support::toy_instance(&prep, 41));
        }
        ProofState::Initial => panic!("finished one-batch proof must be active"),
    }

    assert!(
        matches!(
            neo_fold_clean::finish_uncompressed_with_audit(&prep, finished),
            Err(neo_fold_clean::Error::FinalizedProofInconsistent)
        ),
        "finish_uncompressed_with_audit trusted an already-finalized proof with a non-empty latest"
    );
}

#[test]
fn prove_rejects_public_input_len_mismatch() {
    let prep = support::toy_preprocessing();
    let z = vec![F::ZERO; prep.structure().m];
    let mismatched =
        neo_fold_clean::CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, 0)
            .expect("mismatched public-input split instance");

    assert!(
        matches!(
            neo_fold_clean::prove(&prep, vec![vec![mismatched]]),
            Err(neo_fold_clean::Error::PublicInputLenMismatch { expected: D, got: 0 })
        ),
        "prove accepted an instance whose m_in disagreed with preprocessing"
    );
}

// ── Decider contract preflight (`decider::validate_witness`) ───────────────
//
// `validate_witness` is the non-SNARK preflight the Spartan PR will package.
// Public-vs-witness binding lives in `decider::Statement`; these tests pin
// that contract by running real lifecycle output through `validate_witness`
// and tampering each authority field independently.

/// Thin wrapper: destructure preprocessing into the individual paper-level
/// args `decider::validate_witness` takes.
fn validate(
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
        prep.enforces_terminal_induction(),
        prep.semantic_state_mode(),
        prep.initial_semantic_state_digest(),
        None,
        statement,
    )
}

#[test]
fn decider_validate_witness_accepts_finished_proof() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 61)],
            vec![support::toy_instance(&prep, 62)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    validate(&prep, &statement).expect("validate_witness accepts a finished, untampered statement");
}

#[test]
fn decider_validate_witness_rejects_tampered_public_image_acc_digest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 71)],
            vec![support::toy_instance(&prep, 72)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Flip a byte of the declared public image. The witness-derived
    // acc_digest (recomputed by walking the NIFS chain + final fold) will
    // disagree, and `validate_witness` must surface that mismatch.
    statement.public.acc_digest[0] ^= 0xFF;

    assert!(
        matches!(
            validate(&prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::PublicImageMismatch)
        ),
        "validate_witness accepted a public-image acc_digest that disagreed with the witness"
    );
}

#[test]
fn decider_validate_witness_rejects_tampered_public_batch_x() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 81)],
            vec![support::toy_instance(&prep, 82)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Tamper the first batch's public x in the witness. The verifier's
    // NIFS.V at step 1 sees a fresh claim whose algebraic image no
    // longer matches the proof — the walk must fail.
    statement.witness.public_batches[0][0].x[0] += F::ONE;

    assert!(
        validate(&prep, &statement).is_err(),
        "validate_witness accepted a witness whose public batch x was tampered post-proof"
    );
}

#[test]
fn decider_validate_witness_rejects_missing_final_fold() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 91)],
            vec![support::toy_instance(&prep, 92)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    assert!(
        statement.witness.final_fold.is_some(),
        "test setup: finished proof must carry a final fold"
    );
    // Drop the final fold proof entirely. The terminal-fold walk needs it
    // (trailing latest is non-empty before flush), so `verify_final_fold`
    // must error out and `validate_witness` must surface that.
    statement.witness.final_fold = None;

    let err = validate(&prep, &statement)
        .expect_err("validate_witness accepted a statement whose final fold proof was missing");
    match err {
        neo_fold_clean::paper::decider::Error::WalkFailed(reason) => {
            assert!(
                reason.contains("decider witness must carry a terminal final_fold"),
                "expected decider-level terminal-fold requirement, got {reason}"
            );
        }
        other => panic!("expected decider WalkFailed for missing final_fold, got {other:?}"),
    }
}

#[test]
fn decider_validate_witness_rejects_tampered_final_fold() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 101)],
            vec![support::toy_instance(&prep, 102)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    let final_fold = statement
        .witness
        .final_fold
        .as_mut()
        .expect("final fold present");
    support::mutate_ce_claim(&mut final_fold.nifs.pi_dec.children[0]);

    assert!(
        validate(&prep, &statement).is_err(),
        "validate_witness accepted a tampered final fold proof"
    );
}

// ── Final-state-binding red team ────────────────────────────────────────────
//
// `validate_witness` runs the verifier walk to derive a "walked" state, then
// requires `statement.witness.final_state` to match it: canonical fields,
// running.claims, and Active-with-empty-latest shape. Without these checks,
// a prover could ship witness matrices for an unrelated running accumulator
// that happens to share the public image. These three tests pin each
// component of the binding.

#[test]
fn decider_validate_witness_rejects_final_state_claims_not_matching_walk() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 111)],
            vec![support::toy_instance(&prep, 112)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Mutate `final_state.proof.running.claims[0]` only — the walked state
    // (recomputed inside validate_witness from steps + public_batches +
    // final_fold) still holds the original commitment, so the binding
    // check must reject.
    let final_running = match &mut statement.witness.final_state.proof {
        neo_fold_clean::ProofState::Active { running, .. } => running,
        neo_fold_clean::ProofState::Initial => panic!("test setup: state must be Active"),
    };
    support::mutate_ce_claim(&mut final_running.claims[0]);

    assert!(
        matches!(
            validate(&prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::WitnessShape)
        ),
        "validate_witness accepted a final_state whose running claims diverged from the walked state"
    );
}

#[test]
fn decider_validate_witness_rejects_final_state_public_fields_not_matching_walk() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 121)],
            vec![support::toy_instance(&prep, 122)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Tamper a canonical field of `final_state` without touching
    // `statement.public`. The walked state's `acc_digest` matches
    // `statement.public.acc_digest`; the mutated `final_state.acc_digest`
    // must be caught by the canonical-fields binding check.
    statement.witness.final_state.acc_digest[0] ^= 0xFF;

    assert!(
        matches!(
            validate(&prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::WitnessShape)
        ),
        "validate_witness accepted a final_state whose canonical fields diverged from the walked state"
    );
}

#[test]
fn decider_validate_witness_rejects_final_state_latest_not_empty() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 131)],
            vec![support::toy_instance(&prep, 132)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Push a stray instance into `final_state.proof.latest`. After
    // finalization `latest.instances` must be empty; the shape check
    // rejects this.
    match &mut statement.witness.final_state.proof {
        neo_fold_clean::ProofState::Active { latest, .. } => {
            latest.instances.push(support::toy_instance(&prep, 133));
        }
        neo_fold_clean::ProofState::Initial => panic!("test setup: state must be Active"),
    }

    assert!(
        matches!(
            validate(&prep, &statement),
            Err(neo_fold_clean::paper::decider::Error::WitnessShape)
        ),
        "validate_witness accepted a final_state with a non-empty trailing latest"
    );
}

// ── F'-digest-is-shape-only red team ────────────────────────────────────────
//
// `f_prime_chunk_public_digest` drops `claim.x` and `claim.c.data` to break
// the recursive-link fixed point. That moves chunk-content binding off the
// `z_i` / `public_trace` chain and onto the accumulator path. These tests
// pin the new invariant: tampering a public batch's `x` or commitment after
// proving must still be rejected, and the F' trace coordinates must be
// shape-only across same-shape proofs.

// Audit-trail post-finalize public-batch tampers (`finished.public_batches[*]`)
// are caught by `paper::decider::validate_witness`, not by the non-replay
// `verify_uncompressed`. See `decider_validate_witness_rejects_tampered_public_batch_x`
// above for the `x` tamper, and `decider_validate_witness_rejects_tampered_public_batch_commitment`
// below for the `c.data` tamper.

#[test]
fn decider_validate_witness_rejects_tampered_public_batch_commitment() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 41)],
            vec![support::toy_instance(&prep, 42)],
        ],
    )
    .expect("two-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finish");
    let mut statement = neo_fold_clean::build_decider_statement(&prep, &finished);

    // Tamper the first batch's commitment after proving. The chain replay
    // inside `validate_witness` re-derives NIFS challenges from the absorbed
    // claim and so the algebraic checks at step 1 must fail.
    statement.witness.public_batches[0][0].c.data[0] += F::ONE;

    assert!(
        validate(&prep, &statement).is_err(),
        "validate_witness accepted a statement whose first public batch commitment was \
         tampered post-proof"
    );
}

#[test]
fn same_shape_different_batches_have_same_f_prime_trace_but_different_acc_digest() {
    let prep = support::toy_preprocessing();

    // Two single-batch proofs with same-shape but distinct-content batches.
    // Toy structure has one complete public ring; vary its first element.
    let proof_a = neo_fold_clean::prove(&prep, vec![vec![toy_instance_with_x_value(&prep, F::ZERO)]]).expect("prove a");
    let finished_a = neo_fold_clean::finish_uncompressed(&prep, proof_a).expect("finish a");

    let proof_b = neo_fold_clean::prove(&prep, vec![vec![toy_instance_with_x_value(&prep, F::ONE)]]).expect("prove b");
    let finished_b = neo_fold_clean::finish_uncompressed(&prep, proof_b).expect("finish b");

    // F' trace coordinates are shape-only: same `(d, kappa, m_in,
    // start_index, fresh.len())` ⇒ same chunk digest ⇒ same chained
    // z_i / public_trace.
    assert_eq!(
        finished_a.state.z_i, finished_b.state.z_i,
        "F' chunk digest is shape-only; same-shape chunks must yield identical z_i"
    );
    assert_eq!(
        finished_a.state.public_trace, finished_b.state.public_trace,
        "F' chunk digest is shape-only; same-shape chunks must yield identical public_trace"
    );

    // Acc digest, however, is the content-binding public coordinate after
    // finalization. The two batches' actual contents differ, so the final
    // running accumulator's CE claims differ, and the digest must too.
    assert_ne!(
        finished_a.state.acc_digest, finished_b.state.acc_digest,
        "acc_digest is the content-binding coordinate; distinct batch contents must produce \
         distinct acc_digest values"
    );

    // Sanity: both proofs verify on their own. The non-replay verifier
    // checks each running CE claim directly and then recomputes acc_digest
    // from those claims; both finished proofs satisfy that contract under
    // their own running accumulators.
    neo_fold_clean::verify_uncompressed(&prep, &finished_a).expect("proof a verifies");
    neo_fold_clean::verify_uncompressed(&prep, &finished_b).expect("proof b verifies");
}

// ── H2 regression: stateless semantic invariant ──────────────────────────────
//
// The toy preprocessing is stateless (no `semantic_state_in/out_var_indices`
// in its plan), so the F' image's CCS structure has no Poseidon2 binding
// rows for the `semantic_state_digest` lane. The verifier must therefore
// enforce the protocol invariant `semantic_state_digest == accumulator
// digest carried through finalization` itself; without this check, a
// malicious prover could self-consistently inject arbitrary bytes into
// `PublicImage.semantic_state_digest`.

#[test]
fn verify_uncompressed_rejects_tampered_stateless_semantic_state_digest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 51)]]).expect("one-batch proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finalize");

    // Honest stateless proof must verify cleanly under verify_uncompressed.
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest stateless proof verifies");

    // Tamper just the terminal semantic_state_digest. The accumulator
    // digest stays untouched, so the only invariant that should now fail
    // is the stateless `semantic == acc` carry-through.
    finished.state.semantic_state_digest[0] ^= 0xFF;

    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("stateless verify must reject a tampered semantic_state_digest");
    assert!(
        matches!(err, neo_fold_clean::Error::StatelessSemanticInvariantViolated),
        "expected StatelessSemanticInvariantViolated, got {err:?}"
    );
}

#[test]
fn verify_uncompressed_audit_rejects_tampered_stateless_step_proof_semantic_state_digest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![
            vec![support::toy_instance(&prep, 61)],
            vec![support::toy_instance(&prep, 62)],
        ],
    )
    .expect("two-batch proof");
    let mut audit = neo_fold_clean::finish_uncompressed_with_audit(&prep, proof).expect("finalize with audit");

    // Honest audit verifies cleanly.
    neo_fold_clean::verify_uncompressed_audit(&prep, &audit).expect("honest stateless audit verifies");

    // Tamper the first step's StepProof.semantic_state_digest. Per-step
    // f_prime::verify walks this and must reject with
    // StatelessSemanticInvariantViolated (not XOutMismatch) because the
    // dedicated check fires before x_out comparison.
    audit.steps[0].semantic_state_digest[0] ^= 0xFF;

    let err = neo_fold_clean::verify_uncompressed_audit(&prep, &audit)
        .expect_err("stateless audit must reject a tampered per-step semantic_state_digest");
    // The per-step f_prime::verify surfaces
    // `Construction2(StatelessSemanticInvariantViolated)` wrapped in the
    // decider's `WalkFailed` (which carries the inner error as a string).
    // Match on the message text and confirm it identifies the stateless
    // invariant — not the generic XOutMismatch.
    let msg = format!("{err}");
    assert!(
        msg.contains("stateless chain claimed semantic_state_digest"),
        "expected the per-step verifier to surface the stateless invariant error, got {err}"
    );
}

// `verify_uncompressed`'s witness-authority block (step 5) must enforce
// all five terminal obligations against each `(claim, witness Z)`:
//
//   commit(Z) == claim.c          | Ajtai opening
//   project_x(Z) == claim.X       | public-input projection
//   ||Z||_∞ < b                   | low-norm digit-range
//   y_ring[j] == M_j · Z(r)       | CE-relation closure (this test)
//   ct[j] == const-term(y_ring[j]) | scalar-view closure
// The `acc_digest` chain hash only carries commitment data, so a
// malformed `y_ring` slips past the binding pipeline. The CE-relation
// row is what catches it. Test exercises `validate_final_witness_authority`
// directly (the per-claim helper) so the rejection is unambiguously
// the CE-relation check, not the chain-replay binding step.

#[test]
fn final_witness_authority_rejects_y_ring_inconsistent_with_m_z_at_r() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 71)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    let honest_running = final_running(&finished);
    // Sanity: honest running passes the full witness-authority block,
    // including the new CE-relation check.
    neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &honest_running)
        .expect("honest running passes the full witness-authority gate");

    // Tamper: mutate y_ring[0][0] on the first claim. Witness Z stays
    // honest, so commit/X/low-norm still pass. Only the CE-relation
    // row can fail.
    let mut tampered_running = honest_running;
    assert!(
        !tampered_running.claims.is_empty()
            && !tampered_running.claims[0].y_ring.is_empty()
            && !tampered_running.claims[0].y_ring[0].is_empty(),
        "test setup requires a non-empty y_ring"
    );
    let original = tampered_running.claims[0].y_ring[0][0];
    tampered_running.claims[0].y_ring[0][0] = original + neo_math::K::ONE;
    assert_ne!(
        tampered_running.claims[0].y_ring[0][0], original,
        "mutation must actually change y_ring[0][0]"
    );

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &tampered_running)
        .expect_err("verify_uncompressed's witness-authority gate must reject y_ring inconsistent with M·Z at r");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }),
        "expected FinalAccumulatorCeRelationViolation on y_ring tamper, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_extra_claim_without_witness() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 76)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    running.claims.push(running.claims[0].clone());

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("extra claim without matching witness must reject");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorWitnessShapeMismatch),
        "expected FinalAccumulatorWitnessShapeMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_extra_witness_without_claim() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 77)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    running.witnesses.push(running.witnesses[0].clone());

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("extra witness without matching claim must reject");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorWitnessShapeMismatch),
        "expected FinalAccumulatorWitnessShapeMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_zero_commitment_with_wrong_ajtai_shape() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 78)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "toy fixture must exercise the zero-witness fast path"
    );

    // Keep the zero commitment internally self-consistent, but make it a
    // commitment to the wrong Ajtai target dimension. Before the shape check,
    // the zero-witness fast path compared against `Commitment::zeros` using
    // the prover-supplied shape and accepted this forged opening.
    let claim = &mut running.claims[0];
    claim.c.kappa += 1;
    claim
        .c
        .data
        .extend(std::iter::repeat(F::ZERO).take(claim.c.d));
    assert_eq!(
        claim.c.data.len(),
        claim.c.d * claim.c.kappa,
        "test mutation keeps the forged zero commitment internally consistent"
    );

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("zero-witness fast path must reject wrong Ajtai commitment shape");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::FinalAccumulatorWitnessCommitmentMismatch { .. }
        ),
        "expected FinalAccumulatorWitnessCommitmentMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_zero_witness_with_wrong_packed_shape() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 79)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "toy fixture must exercise the zero-witness fast path"
    );
    let wrong_cols = running.witnesses[0].cols() + 1;
    running.witnesses[0] = Mat::zero(neo_math::D, wrong_cols, F::ZERO);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("zero-witness fast path must reject wrong packed witness shape");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorWitnessShapeMismatch),
        "expected FinalAccumulatorWitnessShapeMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_zero_witness_m_in_exceeds_structure_m() {
    let prep = support::toy_preprocessing_unfixed_public_input_len();
    assert!(
        prep.public_input_len.is_none(),
        "this test intentionally bypasses the program public_input_len guard so it reaches the structure-width guard"
    );
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 80)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "toy fixture must exercise the zero-witness fast path"
    );

    // Keep the claim locally self-consistent for a zero witness, but set
    // `m_in` beyond the verifier-owned CCS structure width. The nonzero
    // path rejects this via `project_x_from_witness_mat`; the zero path
    // must not skip that same public-input projection shape check.
    let too_wide_m_in = prep.structure().m + 1;
    running.claims[0].m_in = too_wide_m_in;
    running.claims[0].X = Mat::zero(neo_math::D, too_wide_m_in, F::ZERO);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("zero-witness path must reject m_in beyond the structure width");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorPublicInputMismatch { .. }),
        "expected FinalAccumulatorPublicInputMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_zero_witness_partial_ring_public_input() {
    let prep = support::toy_preprocessing_unfixed_public_input_len();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 801)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "toy fixture must exercise the zero-witness fast path"
    );
    running.claims[0].m_in = 1;
    running.claims[0].X = Mat::zero(neo_math::D, 1, F::ZERO);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("zero-witness authority accepted a partial-ring public input");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorPublicInputMismatch { .. }),
        "expected FinalAccumulatorPublicInputMismatch, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_m_in_relabel_below_program_public_input_len() {
    let prep = support::toy_preprocessing();
    assert_eq!(
        prep.public_input_len,
        Some(D),
        "toy preprocessing fixes one complete public ring"
    );
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 81)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "toy fixture must exercise a self-consistent zero-witness relabel"
    );

    // Hacker model: relabel the final CE claim under a smaller public
    // projection. With a zero witness, commit(Z), X=L_in(Z), low-norm,
    // y_ring, and ct all remain locally self-consistent if m_in is
    // allowed to shrink to zero. The verifier must still reject because
    // `public_input_len` is a verifier-owned program-shape boundary.
    running.claims[0].m_in = 0;
    running.claims[0].X = Mat::zero(neo_math::D, 0, F::ZERO);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("terminal CE authority must reject m_in relabelled below program public_input_len");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PublicInputLenMismatch { expected: D, got: 0 }
        ),
        "expected PublicInputLenMismatch {{ expected: D, got: 0 }}, got {err:?}"
    );
}

#[test]
fn final_witness_authority_rejects_nonzero_witness_m_in_relabel_below_program_public_input_len() {
    let prep = support::toy_preprocessing();
    assert_eq!(
        prep.public_input_len,
        Some(D),
        "toy preprocessing fixes one complete public ring"
    );
    let proof = neo_fold_clean::prove(&prep, vec![vec![toy_instance_with_x_value(&prep, F::ONE)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .any(|&entry| entry != F::ZERO),
        "test setup must carry a non-zero terminal witness so this is not a zero-fast-path check"
    );

    // Hacker model: keep c, y_ring, ct, and Z untouched, but relabel the
    // terminal CE claim as having no public input and make X empty. If
    // `m_in` were treated as prover-owned, the local CE equations would
    // still be self-consistent: commit(Z) still opens, low-norm still
    // holds, and y_ring/ct are independent of L_in's projection width.
    // The verifier must reject because the program fixed `m_in = D`.
    running.claims[0].m_in = 0;
    running.claims[0].X = Mat::zero(neo_math::D, 0, F::ZERO);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("terminal CE authority must reject non-zero witness m_in relabel");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PublicInputLenMismatch { expected: D, got: 0 }
        ),
        "expected PublicInputLenMismatch {{ expected: D, got: 0 }}, got {err:?}"
    );
}

/// End-to-end version of the y_ring tamper attack: build a real
/// finished proof, mutate `proof.state.proof.running.claims[0].y_ring[0][0]`
/// on the recorded final accumulator, leave the NIFS proof outputs and
/// witness `Z` untouched, then call `verify_uncompressed`.
///
/// **Expected rejection path: `PostStateMismatch`** — NOT
/// `FinalAccumulatorCeRelationViolation`. Reason: the verifier-derived
/// running comes from `nifs::verify(proof.final_fold.nifs, ...)` which
/// returns the NIFS proof's *honest* outputs. The binding step
/// (`derived.running.claims == recorded.running.claims`) fires first
/// and catches the mismatch before reaching the witness-authority block.
///
/// The dedicated CE-relation isolation test
/// [`final_witness_authority_rejects_y_ring_inconsistent_with_m_z_at_r`]
/// hits the witness-authority block directly via
/// `validate_final_witness_authority`, so it sees the precise
/// `FinalAccumulatorCeRelationViolation`. Both tests together prove
/// y_ring is bound end-to-end:
///
///   - Helper-direct test: CE-relation rows are load-bearing.
///   - This end-to-end test: ANY y_ring tamper in the recorded state
///     is rejected by *some* gate in the verifier pipeline.
#[test]
fn verify_uncompressed_rejects_recorded_y_ring_tamper_via_binding_step() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 72)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest finished proof verifies");

    let running = final_running_mut(&mut finished);
    assert!(
        !running.claims.is_empty() && !running.claims[0].y_ring.is_empty() && !running.claims[0].y_ring[0].is_empty(),
        "test setup requires non-empty y_ring on the recorded running"
    );
    running.claims[0].y_ring[0][0] = running.claims[0].y_ring[0][0] + neo_math::K::ONE;

    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed must reject a recorded-y_ring tamper");
    assert!(
        matches!(err, neo_fold_clean::Error::PostStateMismatch),
        "expected PostStateMismatch (binding step fires first); see this test's docstring for why \
         the CE-relation variant isn't reached via this attack vector. Got: {err:?}"
    );
}

/// `claim.y_ring.len() != structure.t()` — outer-dimension shape
/// mismatch. The CE-relation gadget cannot compute expected values for
/// the missing matrix and rejects up-front.
#[test]
fn final_witness_authority_rejects_y_ring_outer_length_mismatch() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 73)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    running.claims[0].y_ring.pop();

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("y_ring outer-length mismatch must reject");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }),
        "expected FinalAccumulatorCeRelationViolation, got {err:?}"
    );
}

/// `claim.ct[j] != constant_term(y_ring[j])` — the scalar view must
/// agree with the lane it summarises. `ct` enters downstream
/// consistency checks, so leaving it unbound would let a prover lie
/// about it independently of `y_ring`.
#[test]
fn final_witness_authority_rejects_ct_inconsistent_with_y_ring() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 74)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        !running.claims[0].ct.is_empty(),
        "test setup requires a non-empty ct vector"
    );
    let original = running.claims[0].ct[0];
    running.claims[0].ct[0] = original + neo_math::K::ONE;
    assert_ne!(running.claims[0].ct[0], original, "mutation must change ct[0]");

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("ct inconsistent with y_ring must reject");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorCtMismatch { .. }),
        "expected FinalAccumulatorCtMismatch, got {err:?}"
    );
}

/// `claim.r.len() != log2(next_pow2(structure.n).max(2))` — the multilinear
/// evaluation point is the wrong shape. `compute_y_from_Z_and_r` clamps
/// its eval domain to `min(n, 2^|r|)`, so a short `r` would silently
/// truncate the closure and a long `r` would over-extend it. The
/// CE-relation check must reject an off-shape `r` before evaluating
/// `M · Z(r)`. Lengthening the honest `r` by one element is the minimal
/// off-shape mutation that holds for any `structure.n`.
#[test]
fn final_witness_authority_rejects_r_length_mismatch() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 75)]])
        .expect("one-batch uncompressed proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    // Honest `r` is correctly shaped; pushing one extra element makes it
    // disagree with `log2(next_pow2(structure.n).max(2))` for any `n`.
    running.claims[0].r.push(neo_math::K::ONE);

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("off-shape r must reject");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::FinalAccumulatorEvaluationPointShapeMismatch { .. }
        ),
        "expected FinalAccumulatorEvaluationPointShapeMismatch, got {err:?}"
    );
}
