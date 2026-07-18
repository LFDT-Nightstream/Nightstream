//! Steady-state terminal-decider synthesis for the final F' step.

use super::*;

// ──────────────────────────────────────────────────────────────────────
// Last-step terminal decider — steady-state O(1) in chain length
// ───────────────────────────────────────────────────────────────────────

/// Output of the steady-state terminal-decider synthesis.
///
/// Emits the R1CS for **three** things:
///   (a) the **last** encoded F' step's recursive verifier
///       ([`emit_recursive_step_r1cs`] / [`emit_base_step_r1cs`]),
///   (b) the terminal NIFS.V fold ([`emit_terminal_fold`]), and
///   (c) the public-image pins ([`pin_public_image`]).
///
/// This is NOT a pure accumulator-only terminal decider: the last
/// step's full F' shell still lands in the builder. The asymptotic win
/// is that the per-step `for` loop over `proof.steps[]` (which
/// [`synthesize_statement_r1cs`] uses to replay every historical step)
/// is gone, so `builder.rows()` is constant in the steady-state where
/// every recursive last step folds a `k_rho`-sized running into
/// `k_rho`-sized children.
///
/// Soundness for prior steps is carried inductively inside the encoded
/// F' instance the prover folded at the last step — each `enc(F'_i)`
/// image embeds the in-circuit NIFS.V trace of the previous fold
/// (NIFS payloads, ring-action traces, Poseidon traces). Verifying the
/// latest binds the chain transitively.
///
/// **Important scope limit.** This helper is still an audit/row-shape
/// synthesis path, not a deployable compressed verifier. It receives an
/// [`crate::lifecycle::UncompressedAudit`], runs the native chain replay to
/// recover the last step's `state_in`, and only then emits the last-step
/// rows. A real terminal verifier must not rely on that native replay as
/// authority; it must either prove the full audit relation, or prove a
/// compact relation whose public/proof inputs bind the last state and
/// terminal CE statement in-circuit. Until that proof layer exists,
/// `crate::lifecycle::compress` remains fail-closed.
///
/// A future "pure accumulator-only" terminal decider would emit just
/// (b) + (c) and pin the latest F' relation's correctness via in-circuit
/// verification of a compact proof for the running accumulator. That is out
/// of scope for this milestone.
pub struct LastStepTerminalSynthesis {
    pub builder: R1csBuilder,
    /// Count of CE claims carried by the final running accumulator
    /// post-final-fold (typically `k_rho`).
    pub running_claim_count: usize,
    /// `true` once the terminal fold's NIFS.V has been emitted in-circuit.
    pub has_final_fold: bool,
    /// Count of `statement.public` fields pinned in-circuit (must equal
    /// [`REQUIRED_PUBLIC_IMAGE_PINS`] for the relation to be
    /// self-sufficient).
    pub public_image_pins: usize,
    /// Direct terminal CE-relation rows emitted against NIFS-output children.
    ///
    /// The future compact verifier must use a separate marker once it really
    /// verifies proof bytes; this field means the current direct rows exist.
    pub terminal_ce_direct_relations: bool,
}

/// Synthesize the steady-state O(1) "last F' step + terminal fold +
/// public-image pins" decider relation for a finalized `Uncompressed`
/// proof.
///
/// This is not the pure accumulator-only terminal decider HyperNova
/// Construction 2 ultimately targets — see [`LastStepTerminalSynthesis`]
/// for the scope. What it *does* deliver is: the per-step `for`-loop
/// emission that grows with chain length is gone. The native walk over
/// `proof.steps` is still O(N) (it has to derive the last step's
/// state_in), and that walk is **not** a substitute for a proof checked by
/// the final verifier. Only one F' step's R1CS lands in the builder, so
/// `builder.rows()` is constant in the steady-state (last step folding
/// `k_rho` → `k_rho`), but this helper is not the final compressed
/// verifier contract.
///
/// Use [`synthesize_statement_r1cs`] for the audit-replay path that
/// emits one F' shell per historical step.
///
/// # Errors
///
/// Returns [`decider::Error`] when:
/// - The proof has not been finalized (`final_fold = None`).
/// - The native preflight `decider::validate_witness` fails.
/// - Emission of the last step or terminal fold fails.
pub fn synthesize_last_step_terminal_r1cs(
    prep: &Preprocessing,
    audit: &crate::lifecycle::UncompressedAudit,
) -> Result<LastStepTerminalSynthesis, decider::Error> {
    if audit.proof.final_fold.is_none() {
        return Err(decider::Error::WalkFailed(
            "terminal decider requires a finalized proof (run `finish_uncompressed_with_audit` first)".into(),
        ));
    }
    if audit.steps.is_empty() {
        return Err(decider::Error::WalkFailed(
            "terminal decider requires at least one F' step".into(),
        ));
    }

    let statement = crate::lifecycle::build_decider_statement(prep, audit);

    // 1. Native preflight on the full statement (O(N) work, zero R1CS rows).
    decider::validate_witness(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        prep.public_input_len,
        prep.enforces_f_prime_recursive_link(),
        prep.enforces_terminal_induction(),
        prep.semantic_state_mode,
        prep.initial_semantic_state_digest(),
        prep.nebula(),
        &statement,
    )?;

    // 2. Walk natively to compute the last step's state_in / state_out.
    //    No R1CS rows are emitted here — `verify_step` runs out-of-circuit.
    let structure_digest_v = *prep.structure_digest();
    let z_0 = initial_boundary_digest(&structure_digest_v, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure_digest_v);
    let acc_digest = AccumulatorHandle::empty().digest();
    let mut state = State::base(
        z_0,
        public_trace,
        acc_digest,
        statement.public.initial_semantic_state_digest,
    );

    let last_idx = audit.steps.len() - 1;
    let mut last_state_in: Option<State> = None;
    let mut running_pre_final_fold = crate::paper::construction2::RunningInstance::default();

    for (idx, (public_batch, step_proof)) in statement
        .witness
        .public_batches
        .iter()
        .zip(&statement.witness.steps)
        .enumerate()
    {
        if idx == last_idx {
            last_state_in = Some(state.clone());
        }
        let nebula_advance = replay_nebula_advance(prep, &state, step_proof, public_batch)?;
        state = construction2::verify_step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            state,
            public_batch,
            step_proof,
            prep.semantic_state_mode,
            nebula_advance,
        )
        .map_err(|e| decider::Error::WalkFailed(format!("native walk step {idx}: {e}")))?;
        if idx == last_idx {
            if let ProofState::Active { running, .. } = &state.proof {
                running_pre_final_fold = running.materialize().map_err(|e| {
                    decider::Error::WalkFailed(format!("materialize last running before final fold: {e}"))
                })?;
            }
        }
    }
    let last_state_in = last_state_in.expect("non-empty proof has a last step");
    let last_state_out = state;
    let last_step_proof = &statement.witness.steps[last_idx];
    let last_public_batch = &statement.witness.public_batches[last_idx];

    // 3. Emit ONLY the last F' step. Constant in N.
    let mut builder = R1csBuilder::new();
    let last_output = match &last_step_proof.fold {
        FoldProof::NoFold => {
            let out = emit_base_step_r1cs(&mut builder, prep, &last_state_in, &last_state_out, last_public_batch)
                .map_err(|e| decider::Error::WalkFailed(format!("emit last (base) step: {e}")))?;
            enforce_base_state_constants(&mut builder, prep, &statement.public, &out);
            out
        }
        FoldProof::Recursive(nifs) => {
            let nifs = nifs
                .materialize()
                .map_err(|e| decider::Error::WalkFailed(format!("materialize last recursive step: {e}")))?;
            emit_recursive_step_r1cs(
                &mut builder,
                prep,
                &last_state_in,
                &last_state_out,
                last_public_batch,
                &nifs,
            )
            .map_err(|e| decider::Error::WalkFailed(format!("emit last (recursive) step: {e}")))?
        }
    };

    // 4. Emit terminal fold NIFS.V + terminal latest link.
    let final_fold = statement
        .witness
        .final_fold
        .as_ref()
        .expect("proof.final_fold checked above");
    let (
        _terminal_fold_emitted,
        _terminal_latest_link,
        _terminal_parent_authority_link,
        final_acc_digest,
        terminal_running,
        terminal_children,
    ) = emit_terminal_fold(
        &mut builder,
        prep,
        &last_output,
        &running_pre_final_fold,
        last_public_batch,
        &final_fold.nifs,
    )?;

    // 4b. CE-claim continuity: the last recursive F' step's Π_DEC
    //     children must equal the terminal fold's Π_CCS running input
    //     wire-for-wire across every carried CE-core field (c_data, X, r,
    //     s_col, y_ring, ct, fold_digest_fields). Mirrors
    //     the analogous check in
    //     `synthesize_statement_r1cs_inner` (full-history audit). Base
    //     last-step has no nifs_children, so this is guarded by `if let Some`.
    if let Some(prev_children) = last_output.nifs_children.as_ref() {
        enforce_child_core_equal_running(&mut builder, prev_children, &terminal_running)
            .map_err(|e| decider::Error::WalkFailed(format!("CE continuity terminal fold (last-step): {e}")))?;
    }

    // 5. Public-image pins. Anchors the relation to a SNARK-verifiable
    //    public statement.
    let public_image_pins = pin_public_image(&mut builder, &statement.public, prep, &last_output, &final_acc_digest);

    // 6. Terminal CE-relation closure — SOUND DIRECT PATH, NOT COMPACT.
    //    Same contract as `synthesize_statement_r1cs_inner` step 8: these
    //    rows directly enforce the CE relation against the opened terminal
    //    witnesses and are the current soundness contract — do not remove.
    //    A future compact terminal-CE proof could replace them with
    //    in-circuit verifier rows. See `paper::decider_ce_relation` docs.
    let ProofState::Active {
        running: final_running,
        latest: _final_latest,
    } = &statement.witness.final_state.proof
    else {
        return Err(decider::Error::WalkFailed(
            "statement.witness.final_state must be Active after finalization".into(),
        ));
    };
    let final_running = final_running
        .materialize()
        .map_err(|e| decider::Error::WalkFailed(format!("materialize final running: {e}")))?;
    crate::paper::decider_ce_relation::enforce_final_ce_relations(
        &mut builder,
        prep,
        &terminal_children,
        &final_running.witnesses,
    )
    .map_err(|e| decider::Error::WalkFailed(format!("terminal CE relation: {e}")))?;
    let terminal_ce_direct_relations = true;

    Ok(LastStepTerminalSynthesis {
        builder,
        running_claim_count: terminal_running.len(),
        has_final_fold: true,
        public_image_pins,
        terminal_ce_direct_relations,
    })
}
