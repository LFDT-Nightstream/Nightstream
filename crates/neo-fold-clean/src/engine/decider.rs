//! Direct-CCS full-history audit R1CS synthesis.
//!
//! ## Scope
//!
//! This module packages a validated `decider::Statement` into a
//! self-contained R1CS that **replays every lifecycle/F' step** and the
//! terminal fold. It is useful for auditing the direct-CCS interim path,
//! but it is not the asymptotic IVC decider from HyperNova Construction 2:
//! its size is linear in the number of historical steps because the
//! lifecycle is currently folding application CCS instances, not encoded
//! `F'` instances.
//!
//! This module does **not** create, verify, or serialize a compact proof.
//! Do not use this full-history builder for production compression sizing.
//! The constant-size terminal decider belongs to the future `F'` frontend
//! path, where each online step folds `enc(F')` and the final SNARK proves
//! only the terminal folded accumulator.
//!
//! The R1CS itself, not native preflight, enforces canonical base-state
//! pins, every base/recursive F' step, adjacent state links, full CE
//! continuity, terminal NIFS.V, terminal latest-link rows, public-image
//! pins, and terminal CE rows against NIFS-output children. Completeness
//! is summarized by [`DeciderR1csSynthesis`]; it is an audit marker, not a
//! production-compression claim.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::finalization::FINAL_FOLD_TRANSCRIPT_LABEL;
use crate::paper::construction2::{self, FoldProof, ProofState, SemanticStateMode, State, StepProof, TRIVIAL_PC};
use crate::paper::decider::{self, PublicImage, Statement};
use crate::paper::digest::{
    digest32_as_fields, f_prime_chunk_public_digest, initial_boundary_digest, public_trace_seed_digest,
    state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use crate::paper::f_prime::digest_circuit::{enforce_state_x_out_digest_circuit, StateXOutDigestInputs};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::nebula_lane_circuit::{
    delayed_nebula_public_suffix_len, enforce_nebula_lane_equality_circuit,
};
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs,
    FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStateWires, FPrimeStepConfig,
    FPrimeStepOutput, F_PRIME_ENC_INST_BITS, F_PRIME_ENC_INST_OFFSET, F_PRIME_PUBLIC_INPUT_LEN,
    F_PRIME_PUBLIC_ONE_OFFSET,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::nifs::circuit::{enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::nifs::NifsProof;
use crate::paper::reductions::accumulator_digest_circuit::enforce_accumulator_digest_from_parent_circuit;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_ce_claim_digest, CeClaimDigestInputs, SplitNcPiCcsOutputWires, SplitNcPiCcsVConfig,
};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;
use crate::paper::relations::product_commitment_circuit::enforce_adv_equality;
use crate::paper::relations::CcsClaim;

/// Full-history audit R1CS output plus completeness tracking.
pub struct DeciderR1csSynthesis {
    pub builder: R1csBuilder,
    /// Exact step wire layouts retained for artifact exporters. This metadata
    /// is non-authoritative; emitted rows remain the circuit authority.
    #[doc(hidden)]
    pub step_wire_audits: Vec<FPrimeStepWireAudit>,
    /// Base F' step emitted in-circuit.
    pub base_step_emitted: bool,
    /// Base seed wires pinned to preprocessing-derived constants.
    pub base_state_pinned: bool,
    /// Number of `FoldProof::Recursive` steps emitted.
    pub recursive_step_count: usize,
    /// Adjacent F' state links emitted.
    pub cross_step_links: usize,
    /// Terminal latest `fresh.x == enc(last.x_out)` link emitted.
    pub terminal_latest_link: bool,
    /// Terminal `final_fold` NIFS.V emitted in-circuit.
    pub terminal_fold_emitted: bool,
    /// Terminal `statement.public` field pins emitted by `pin_public_image`.
    pub public_image_pins: usize,
    /// Wire-for-wire CE continuity links between NIFS.V step boundaries.
    pub accumulator_claim_links: usize,
    /// Wire-for-wire Π_RLC parent-authority continuity links between NIFS.V
    /// step boundaries.
    pub parent_authority_links: usize,
    /// Direct terminal CE-relation rows emitted against NIFS-output children.
    ///
    /// Until a real compact terminal-CE proof verifier exists, the readiness
    /// gate must require these direct rows specifically.
    pub terminal_ce_direct_relations: bool,
}

/// Read-only wire layout for one step inside the composed audit builder.
#[doc(hidden)]
pub struct FPrimeStepWireAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub is_base: bool,
    pub state_in_columns: Vec<usize>,
    pub state_out_columns: Vec<usize>,
    pub x_out_columns: [usize; 4],
    pub x_out_bit_columns: Vec<usize>,
    pub prior_link_digest_columns: Option<[usize; 4]>,
    pub prior_link_bit_columns: Vec<usize>,
    pub prior_fresh_public_columns: Vec<Vec<usize>>,
    pub prior_link_row_range: Option<(usize, usize)>,
    pub prior_link_first_allocated_column: Option<usize>,
}

fn f_prime_state_audit_columns(state: &FPrimeStateWires) -> Vec<usize> {
    let mut columns = Vec::new();
    columns.extend(state.vk_fs_digest.map(Var::col));
    columns.extend(state.pi_ccs_header_bundle.map(Var::col));
    columns.push(state.chunk_count.col());
    columns.push(state.step_count.col());
    columns.extend(state.z_0.map(Var::col));
    columns.extend(state.z_i.map(Var::col));
    columns.push(state.pc.col());
    columns.extend(state.semantic_state_digest.map(Var::col));
    columns.extend(state.acc_digest.map(Var::col));
    columns.extend(state.public_trace.map(Var::col));
    if let Some(nebula) = state.nebula {
        columns.extend([
            nebula.open.col(),
            nebula.seg_idx.col(),
            nebula.idx.col(),
            nebula.ts.col(),
        ]);
        for value in nebula.gamma.into_iter().chain(nebula.h) {
            columns.extend([value.c0.col(), value.c1.col()]);
        }
        columns.extend(nebula.sp.map(Var::col));
        for digest in nebula.d_pre.into_iter().chain(nebula.d_seen) {
            columns.extend(digest.map(Var::col));
        }
        columns.extend(nebula.d_mem.map(Var::col));
    }
    columns
}

/// `pin_public_image` pins every public field except the initial semantic
/// seed, which is part of the base-state pin.
pub const REQUIRED_PUBLIC_IMAGE_PINS: usize = 10;

impl DeciderR1csSynthesis {
    /// Readiness gate for the self-contained full-history audit relation.
    pub fn is_self_sufficient_relation(&self) -> bool {
        self.base_step_emitted
            && self.base_state_pinned
            && self.cross_step_links == self.recursive_step_count
            && self.accumulator_claim_links == self.recursive_step_count
            && self.parent_authority_links == self.recursive_step_count
            && self.terminal_latest_link
            && self.terminal_fold_emitted
            && self.public_image_pins == REQUIRED_PUBLIC_IMAGE_PINS
            && self.terminal_ce_direct_relations
            && self.builder.is_satisfied()
    }
}

/// Run the non-SNARK preflight on `statement`, then synthesize the
/// full-history audit R1CS for the direct-CCS interim path. This module
/// stops here: no compact proof is created, verified, or serialized.
///
/// Errors propagate from [`crate::paper::decider::validate_witness`] and
/// from in-circuit emission (wrapped in [`decider::Error::WalkFailed`]).
/// This relation grows linearly with the number of steps and should not
/// be used to size the future constant-size IVC terminal decider.
pub fn synthesize_statement_r1cs(
    prep: &Preprocessing,
    statement: &Statement,
) -> Result<DeciderR1csSynthesis, decider::Error> {
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
        statement,
    )?;
    synthesize_statement_r1cs_inner(prep, statement)
}

/// **TEST-ONLY GADGET HARNESS.** Narrow integration-test access to
/// individual decider row families. Production code MUST NOT touch this.
#[doc(hidden)]
#[path = "decider_test_isolation.rs"]
pub mod __test_isolation;

fn synthesize_statement_r1cs_inner(
    prep: &Preprocessing,
    statement: &Statement,
) -> Result<DeciderR1csSynthesis, decider::Error> {
    // 1-3. F' chain (base + recursive steps + cross-step links).
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

    let mut builder = R1csBuilder::new();
    let full_history_start = builder.rows();
    let mut base_step_emitted = false;
    let mut base_state_pinned = false;
    let mut recursive_step_count = 0;
    let mut cross_step_links = 0;
    let mut accumulator_claim_links = 0;
    let mut parent_authority_links = 0;
    let mut step_wire_audits = Vec::new();
    let mut last_output: Option<FPrimeStepOutput> = None;
    let mut previous_children: Option<Vec<CeClaimWires>> = None;
    let mut previous_parent: Option<CeClaimWires> = None;
    let mut running_pre_final_fold = crate::paper::construction2::RunningInstance::default();

    let zipped = statement
        .witness
        .public_batches
        .iter()
        .zip(&statement.witness.steps);
    for (idx, (public_batch, step_proof)) in zipped.enumerate() {
        let state_in = state.clone();
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
        .map_err(|e| decider::Error::WalkFailed(format!("step {idx}: {e}")))?;

        let step_start = builder.rows();
        let (output, step_family, is_base) = match &step_proof.fold {
            FoldProof::NoFold => {
                base_step_emitted = true;
                let out = emit_base_step_r1cs(&mut builder, prep, &state_in, &state, public_batch)
                    .map_err(|e| decider::Error::WalkFailed(format!("emit F' base step {idx}: {e}")))?;
                enforce_base_state_constants(&mut builder, prep, &statement.public, &out);
                base_state_pinned = true;
                (out, "decider.step.base", true)
            }
            FoldProof::Recursive(nifs) => {
                recursive_step_count += 1;
                let out = emit_recursive_step_r1cs(&mut builder, prep, &state_in, &state, public_batch, nifs)
                    .map_err(|e| decider::Error::WalkFailed(format!("emit F' recursive step {idx}: {e}")))?;
                (out, "decider.step.recursive", false)
            }
        };
        builder.record_row_family(step_family, step_start);
        step_wire_audits.push(FPrimeStepWireAudit {
            row_start: step_start,
            row_end: builder.rows(),
            is_base,
            state_in_columns: f_prime_state_audit_columns(&output.state_in),
            state_out_columns: f_prime_state_audit_columns(&output.state_out),
            x_out_columns: output.x_out.map(Var::col),
            x_out_bit_columns: output.x_out_bits.iter().map(|wire| wire.col()).collect(),
            prior_link_digest_columns: output
                .prior_link
                .as_ref()
                .map(|link| link.digest.map(Var::col)),
            prior_link_bit_columns: output
                .prior_link
                .as_ref()
                .map(|link| link.encoded_bits.iter().map(|wire| wire.col()).collect())
                .unwrap_or_default(),
            prior_fresh_public_columns: output
                .prior_link
                .as_ref()
                .map(|link| {
                    link.fresh_public_inputs
                        .iter()
                        .map(|input| input.iter().map(|wire| wire.col()).collect())
                        .collect()
                })
                .unwrap_or_default(),
            prior_link_row_range: output
                .prior_link
                .as_ref()
                .map(|link| (link.row_start, link.row_end)),
            prior_link_first_allocated_column: output
                .prior_link
                .as_ref()
                .map(|link| link.first_allocated_column),
        });

        if let Some(prev) = last_output.as_ref() {
            let state_link_start = builder.rows();
            enforce_state_link(&mut builder, &prev.state_out, &output.state_in);
            builder.record_row_family("decider.state_link", state_link_start);
            cross_step_links += 1;
        }

        // CE-claim continuity: previous step's NIFS children must equal
        // this step's NIFS running, wire-for-wire (not just by digest).
        // Skipped if either side has no NIFS.V (base step).
        if let (Some(prev_children), Some(curr_running)) = (previous_children.as_ref(), output.nifs_running.as_ref()) {
            let continuity_start = builder.rows();
            enforce_children_equal_running(&mut builder, prev_children, curr_running)
                .map_err(|e| decider::Error::WalkFailed(format!("CE continuity step {idx}: {e}")))?;
            builder.record_row_family("decider.ce_continuity", continuity_start);
            accumulator_claim_links += 1;
        }
        if let (Some(prev_parent), Some(curr_parent)) =
            (previous_parent.as_ref(), output.nifs_running_parent_authority.as_ref())
        {
            let parent_continuity_start = builder.rows();
            enforce_children_equal_running(
                &mut builder,
                std::slice::from_ref(prev_parent),
                std::slice::from_ref(curr_parent),
            )
            .map_err(|e| decider::Error::WalkFailed(format!("parent-authority continuity step {idx}: {e}")))?;
            builder.record_row_family("decider.parent_continuity", parent_continuity_start);
            parent_authority_links += 1;
        }
        previous_children = output.nifs_children.clone();
        previous_parent = output.nifs_parent.clone();

        // Snapshot the post-step running so the terminal fold can use it.
        // (Empty after a NoFold step, K-claim vector after Recursive.)
        if let ProofState::Active { running, .. } = &state.proof {
            running_pre_final_fold = running.clone();
        }
        last_output = Some(output);
    }

    let last = last_output
        .ok_or_else(|| decider::Error::WalkFailed("full-history audit relation needs at least one step".into()))?;

    // 5-6. Terminal fold + terminal latest link.
    let final_fold = statement
        .witness
        .final_fold
        .as_ref()
        .ok_or_else(|| decider::Error::WalkFailed("decider R1CS requires a terminal final_fold".into()))?;
    let trailing_latest = statement
        .witness
        .public_batches
        .last()
        .ok_or_else(|| decider::Error::WalkFailed("final_fold present but public_batches empty".into()))?;

    let terminal_fold_start = builder.rows();
    let (
        terminal_fold_emitted,
        terminal_latest_link,
        terminal_parent_authority_link,
        final_acc_digest,
        terminal_running,
        terminal_children,
    ) = emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        &running_pre_final_fold,
        trailing_latest,
        &final_fold.nifs,
    )?;
    builder.record_row_family("decider.terminal_fold", terminal_fold_start);

    // Final CE-claim continuity link: terminal fold's running must equal
    // the last recursive F' step's children.
    if let Some(prev_children) = previous_children.as_ref() {
        let terminal_continuity_start = builder.rows();
        enforce_children_equal_running(&mut builder, prev_children, &terminal_running)
            .map_err(|e| decider::Error::WalkFailed(format!("CE continuity terminal fold: {e}")))?;
        builder.record_row_family("decider.terminal_continuity", terminal_continuity_start);
        accumulator_claim_links += 1;
    }
    if terminal_parent_authority_link {
        parent_authority_links += 1;
    }

    // 7. Public-image pins.
    let public_pins_start = builder.rows();
    let public_image_pins = pin_public_image(&mut builder, &statement.public, prep, &last, &final_acc_digest);
    builder.record_row_family("decider.public_pins", public_pins_start);

    // 8. Terminal CE-relation closure — SOUND DIRECT PATH, NOT COMPACT.
    //    These rows ARE the decider R1CS's current soundness contract:
    //    they allocate Z and directly enforce commit(Z) == c, X ==
    //    L_in(Z), low-norm, y_ring == M·Z(r), and ct == const-term(y_ring)
    //    against the opened terminal witnesses. Do not remove them — the
    //    F'-chain `acc_digest` commits to the terminal CE claims, but it
    //    does not prove that the opened witness Z satisfies those claims.
    //    Without this closure a SNARK consumer could accept a final Z that
    //    is not a real opening of the terminal accumulator.
    //
    //    Not the eventual COMPACT shape: a future off-circuit terminal-CE
    //    proof could replace these direct rows with in-circuit verifier
    //    rows for that proof, keeping Z out-of-circuit at real F'-image
    //    sizes (n, m ~ 10⁶+). No compact backend is wired here; see
    //    `paper::decider_ce_relation` module docs.
    let ProofState::Active {
        running: final_running,
        latest: _final_latest,
    } = &statement.witness.final_state.proof
    else {
        return Err(decider::Error::WalkFailed(
            "statement.witness.final_state must be Active after finalization".into(),
        ));
    };
    let terminal_ce_start = builder.rows();
    crate::paper::decider_ce_relation::enforce_final_ce_relations(
        &mut builder,
        prep,
        &terminal_children,
        &final_running.witnesses,
    )
    .map_err(|e| decider::Error::WalkFailed(format!("terminal CE relation: {e}")))?;
    builder.record_row_family("decider.terminal_ce", terminal_ce_start);
    builder.record_row_family("decider.full_history", full_history_start);
    let terminal_ce_direct_relations = true;

    Ok(DeciderR1csSynthesis {
        builder,
        step_wire_audits,
        base_step_emitted,
        base_state_pinned,
        recursive_step_count,
        cross_step_links,
        terminal_latest_link,
        terminal_fold_emitted,
        public_image_pins,
        accumulator_claim_links,
        parent_authority_links,
        terminal_ce_direct_relations,
    })
}

// ───────────────────────────────────────────────────────────────────────
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
                running_pre_final_fold = running.clone();
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
        FoldProof::Recursive(nifs) => emit_recursive_step_r1cs(
            &mut builder,
            prep,
            &last_state_in,
            &last_state_out,
            last_public_batch,
            nifs,
        )
        .map_err(|e| decider::Error::WalkFailed(format!("emit last (recursive) step: {e}")))?,
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
    //     wire-for-wire, across every carried CE field (c_data, X, r,
    //     s_col, y_ring, ct, y_zcol, fold_digest_fields). The accumulator
    //     digest omits non-authority sidecars such as y_zcol, so this
    //     direct equality is the terminal-boundary continuity gate. Mirrors
    //     the analogous check in
    //     `synthesize_statement_r1cs_inner` (full-history audit). Base
    //     last-step has no nifs_children, so this is guarded by `if let Some`.
    if let Some(prev_children) = last_output.nifs_children.as_ref() {
        enforce_children_equal_running(&mut builder, prev_children, &terminal_running)
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

fn replay_nebula_advance(
    prep: &Preprocessing,
    state: &State,
    step_proof: &StepProof,
    public_batch: &[CcsClaim],
) -> Result<Option<construction2::NebulaAdvance>, decider::Error> {
    match (prep.nebula(), &state.nebula) {
        (Some(cfg), Some(lane)) => {
            let mut lane_out = lane.clone();
            if prep.enforces_terminal_induction() {
                if step_proof.nebula_open.is_some() {
                    return Err(decider::Error::WalkFailed(
                        "folded F' carries Nebula open data in the delayed claim suffix".into(),
                    ));
                }
                if let ProofState::Active { latest, .. } = &state.proof {
                    lane_out
                        .advance_for_delayed_claims(
                            cfg,
                            prep.vk.digest(),
                            state.z_i,
                            state.acc_digest,
                            F_PRIME_PUBLIC_INPUT_LEN,
                            &latest.claims(),
                        )
                        .map_err(|e| decider::Error::WalkFailed(format!("nebula lane: {e}")))?;
                }
                Ok(Some(construction2::NebulaAdvance { lane_out, open: None }))
            } else {
                lane_out
                    .advance_for_batch(
                        cfg,
                        prep.vk.digest(),
                        state.z_i,
                        state.acc_digest,
                        step_proof.nebula_open,
                        public_batch,
                    )
                    .map_err(|e| decider::Error::WalkFailed(format!("nebula lane: {e}")))?;
                Ok(Some(construction2::NebulaAdvance {
                    lane_out,
                    open: step_proof.nebula_open,
                }))
            }
        }
        (None, None) => Ok(None),
        _ => Err(decider::Error::WalkFailed(
            "nebula config/lane presence mismatch between preprocessing and chain state".into(),
        )),
    }
}

/// Emit one base F' step. Used for `FoldProof::NoFold` (always the first
/// lifecycle step). Constrains state-in to be a base state (counters=0,
/// z_i_in==z_0, acc_digest_in==empty_acc) and advances to a state-out
/// with chunk_count'=1, step_count'=|batch|.
fn emit_base_step_r1cs(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    state_in: &State,
    state_out: &State,
    public_batch: &[CcsClaim],
) -> Result<FPrimeStepOutput, String> {
    let chunk_digest = f_prime_chunk_public_digest(state_in.step_count, public_batch);
    let post_step_x_out = state_x_out_lanes(prep, state_out);

    let f_state = FPrimeStateIn {
        vk_fs_digest: digest32_as_fields(prep.vk.digest()),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: state_in.chunk_count,
        step_count_in: state_in.step_count,
        z_0: digest32_as_fields(state_in.z_0),
        z_i_in: digest32_as_fields(state_in.z_i),
        pc: state_in.pc,
        semantic_state_digest_in: digest32_as_fields(state_in.semantic_state_digest),
        acc_digest_in: digest32_as_fields(state_in.acc_digest),
        public_trace_in: digest32_as_fields(state_in.public_trace),
        nebula: state_in.nebula.clone(),
    };

    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(f_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(f_state.step_count_in);
    let pc_word = image.push_u64_le(f_state.pc);
    let public_x_out_bits = image.push_enc_inst(post_step_x_out);

    let cfg = FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep)?,
        },
        b: prep.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: f_prime_public_input_layout(prep),
        nebula: prep.nebula(),
        state_x_out_digest_mode: state_x_out_digest_mode(prep),
    };
    let rows_in_chunk = public_batch.len() as u64;
    let inputs = FPrimeBaseInputs {
        state: f_state,
        chunk_digest,
        semantic_state_digest_out: digest32_as_fields(state_out.semantic_state_digest),
        rows_in_chunk,
        source_image: &image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    };

    enforce_f_prime_base_step_circuit(builder, &cfg, &inputs).map_err(|e| format!("F' base step emission: {e}"))
}

/// Emit one recursive F' step. Used for `FoldProof::Recursive`.
fn emit_recursive_step_r1cs(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    state_in: &State,
    state_out: &State,
    public_batch: &[CcsClaim],
    nifs: &NifsProof,
) -> Result<FPrimeStepOutput, String> {
    let (running_claims, running_parent_authority, fresh) = match &state_in.proof {
        ProofState::Active { running, latest } => (
            running.claims.as_slice(),
            running.parent_authority.as_ref(),
            latest.claims(),
        ),
        ProofState::Initial => return Err("recursive step requires Active state-in".into()),
    };

    let chunk_digest = f_prime_chunk_public_digest(state_in.step_count, public_batch);
    let prior_x_out = state_x_out_lanes(prep, state_in);
    let post_step_x_out = state_x_out_lanes(prep, state_out);

    let f_state = FPrimeStateIn {
        vk_fs_digest: digest32_as_fields(prep.vk.digest()),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: state_in.chunk_count,
        step_count_in: state_in.step_count,
        z_0: digest32_as_fields(state_in.z_0),
        z_i_in: digest32_as_fields(state_in.z_i),
        pc: state_in.pc,
        semantic_state_digest_in: digest32_as_fields(state_in.semantic_state_digest),
        acc_digest_in: digest32_as_fields(state_in.acc_digest),
        public_trace_in: digest32_as_fields(state_in.public_trace),
        nebula: state_in.nebula.clone(),
    };

    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(f_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(f_state.step_count_in);
    let pc_word = image.push_u64_le(f_state.pc);
    let prior_public = image.push_f_prime_public_input(prior_x_out);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(post_step_x_out);

    let cfg = FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep)?,
        },
        b: prep.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: f_prime_public_input_layout(prep),
        nebula: prep.nebula(),
        state_x_out_digest_mode: state_x_out_digest_mode(prep),
    };
    let inputs = FPrimeRecursiveInputs {
        state: f_state,
        chunk_digest,
        semantic_state_digest_out: digest32_as_fields(state_out.semantic_state_digest),
        acc_digest_out: digest32_as_fields(state_out.acc_digest),
        nifs_msg: NifsVCircuitMessages {
            fresh: &fresh,
            running: running_claims,
            running_parent_authority,
            pi_ccs: &nifs.pi_ccs,
            combined: &nifs.pi_rlc.combined,
            children: &nifs.pi_dec.children,
        },
        // The current-step chunk size = number of fresh claims deposited
        // at this step (= `public_batch.len()`, mirroring native
        // `advance_state(..., fresh_count, ...)`). This may differ from
        // `nifs_msg.fresh.len()` (the previous step's batch being folded
        // by NIFS.V) when batch sizes vary across steps.
        rows_in_chunk: public_batch.len() as u64,
        source_image: &image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };

    enforce_f_prime_recursive_step_circuit(builder, &prep.params, &cfg, &inputs)
        .map_err(|e| format!("F' recursive step emission: {e}"))
}

/// Pin the base step's state-in wires to the canonical preprocessing-derived
/// seed values: `vk_fs_digest`, `pi_ccs_header_bundle`, `z_0`, `z_i = z_0`,
/// `public_trace_seed`, empty `acc_digest`, zero counters, `pc == TRIVIAL_PC`.
///
/// Without this pin, a SNARK verifier would have no way to reject a
/// statement whose base seeds disagree with preprocessing — the F' base
/// step itself only enforces shape (counters=0, z_i==z_0, acc==empty), not
/// the seed values. Native `validate_witness` catches it, but the R1CS
/// must stand alone.
fn enforce_base_state_constants(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    public: &PublicImage,
    base: &FPrimeStepOutput,
) {
    let structure_lanes = *prep.structure_digest();
    let z_0_bytes = initial_boundary_digest(&structure_lanes, prep.public_input_len);
    let public_trace_bytes = public_trace_seed_digest(&structure_lanes);
    let empty_acc_bytes = AccumulatorHandle::empty().digest();

    pin_digest32(builder, &base.state_in.vk_fs_digest, prep.vk.digest());
    pin_digest_fields(
        builder,
        &base.state_in.pi_ccs_header_bundle,
        prep.pi_ccs_header_bundle(),
    );
    pin_digest32(builder, &base.state_in.z_0, z_0_bytes);
    // Base step also enforces z_i == z_0 in-circuit, but pinning here
    // gives the SNARK verifier a direct constant to compare against.
    pin_digest32(builder, &base.state_in.z_i, z_0_bytes);
    pin_digest32(
        builder,
        &base.state_in.semantic_state_digest,
        public.initial_semantic_state_digest,
    );
    pin_digest32(builder, &base.state_in.public_trace, public_trace_bytes);
    pin_digest32(builder, &base.state_in.acc_digest, empty_acc_bytes);
    pin_u64(builder, base.state_in.chunk_count, 0);
    pin_u64(builder, base.state_in.step_count, 0);
    pin_u64(builder, base.state_in.pc, TRIVIAL_PC);
}

fn pin_digest_fields(builder: &mut R1csBuilder, wires: &[Var; 4], expected: [F; 4]) {
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_var(wires[k]), &Lc::from_const(expected[k]));
    }
}

/// Wire-to-wire equality on every field of two [`FPrimeStateWires`]. Used
/// to chain `prev.state_out` into `next.state_in`.
fn enforce_state_link(builder: &mut R1csBuilder, a: &FPrimeStateWires, b: &FPrimeStateWires) {
    enforce_digest_eq(builder, &a.vk_fs_digest, &b.vk_fs_digest);
    enforce_digest_eq(builder, &a.pi_ccs_header_bundle, &b.pi_ccs_header_bundle);
    builder.enforce_eq(&Lc::from_var(a.chunk_count), &Lc::from_var(b.chunk_count));
    builder.enforce_eq(&Lc::from_var(a.step_count), &Lc::from_var(b.step_count));
    enforce_digest_eq(builder, &a.z_0, &b.z_0);
    enforce_digest_eq(builder, &a.z_i, &b.z_i);
    builder.enforce_eq(&Lc::from_var(a.pc), &Lc::from_var(b.pc));
    enforce_digest_eq(builder, &a.semantic_state_digest, &b.semantic_state_digest);
    enforce_digest_eq(builder, &a.acc_digest, &b.acc_digest);
    enforce_digest_eq(builder, &a.public_trace, &b.public_trace);
    match (a.nebula.as_ref(), b.nebula.as_ref()) {
        (None, None) => {}
        (Some(a), Some(b)) => enforce_nebula_lane_equality_circuit(builder, a, b),
        _ => builder.enforce_eq(&Lc::zero(), &Lc::from_const(F::ONE)),
    }
}

fn enforce_digest_eq(builder: &mut R1csBuilder, a: &[Var; 4], b: &[Var; 4]) {
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_var(a[k]), &Lc::from_var(b[k]));
    }
}

/// Full CE-claim continuity: pin every wire of `children[i]` equal to the
/// corresponding wire of `running[i]`, for all `i`. Returns an error if
/// the shapes don't line up (length mismatch, or per-claim shape constants
/// disagree).
///
/// `children` is the Π_DEC output (next-running) view; `running` is the
/// Π_CCS verifier's input-running view. They carry the same data in two
/// representations:
///   - both share Vec<Var> for c_data / x.
///   - both share Vec<KVar> for r / s_col / ct.
///   - children's y_ring / y_zcol are flattened K_LIMBS=2 base-field
///     limbs (Vec<Var>); running's are Vec<KVar>. The helper expands
///     each KVar back into (c0, c1) and pins limb-by-limb.
fn enforce_children_equal_running(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
    running: &[SplitNcPiCcsOutputWires],
) -> Result<(), String> {
    if children.len() != running.len() {
        return Err(format!(
            "CE continuity length mismatch: children={} running={}",
            children.len(),
            running.len()
        ));
    }
    for (idx, (child, run)) in children.iter().zip(running.iter()).enumerate() {
        if child.c_d != run.c_d
            || child.c_kappa != run.c_kappa
            || child.m_in != run.m_in
            || child.x_rows != run.x_rows
            || child.x_cols != run.x_cols
        {
            return Err(format!(
                "CE continuity shape mismatch at index {idx}: child=(d={}, kappa={}, m_in={}, x_rows={}, x_cols={}) \
                 run=(d={}, kappa={}, m_in={}, x_rows={}, x_cols={})",
                child.c_d,
                child.c_kappa,
                child.m_in,
                child.x_rows,
                child.x_cols,
                run.c_d,
                run.c_kappa,
                run.m_in,
                run.x_rows,
                run.x_cols,
            ));
        }
        // c_data + x are Vec<Var> in both representations.
        enforce_vec_var_eq(builder, &child.c_data, &run.c_data, "c_data", idx)?;
        enforce_adv_equality(
            builder,
            child.adv.as_ref(),
            run.adv.as_ref(),
            &format!("CE continuity[{idx}]"),
        )?;
        enforce_vec_var_eq(builder, &child.x, &run.x, "x", idx)?;
        // Shape metadata is also represented as scalar wires inside the
        // verifier circuit. Pin it here so CE continuity is genuinely
        // wire-for-wire, not only a Rust-side shape precheck.
        builder.enforce_eq(&Lc::from_var(child.c_d_var), &Lc::from_var(run.c_d_var));
        builder.enforce_eq(&Lc::from_var(child.c_kappa_var), &Lc::from_var(run.c_kappa_var));
        builder.enforce_eq(&Lc::from_var(child.x_rows_var), &Lc::from_var(run.x_rows_var));
        builder.enforce_eq(&Lc::from_var(child.x_cols_var), &Lc::from_var(run.x_cols_var));
        builder.enforce_eq(&Lc::from_var(child.m_in_var), &Lc::from_var(run.m_in_var));
        // r, s_col, ct are Vec<KVar> in both — KVar exposes c0/c1 directly.
        enforce_vec_kvar_eq(builder, &child.r, &run.r, "r", idx)?;
        enforce_vec_kvar_eq(builder, &child.s_col, &run.s_col, "s_col", idx)?;
        enforce_vec_kvar_eq(builder, &child.ct, &run.ct, "ct", idx)?;
        // y_ring representation differs: child has [j][lane*2 + limb]
        // base-field wires; run has [j][lane] KVars.
        if child.y_ring.len() != run.y_ring.len() {
            return Err(format!(
                "CE continuity y_ring outer-dim mismatch at {idx}: child={} run={}",
                child.y_ring.len(),
                run.y_ring.len()
            ));
        }
        for (j, (child_row, run_row)) in child.y_ring.iter().zip(run.y_ring.iter()).enumerate() {
            enforce_flat_limbs_vs_kvar_row(builder, child_row, run_row, "y_ring", idx, Some(j))?;
        }
        // y_zcol similar but single-row.
        enforce_flat_limbs_vs_kvar_row(builder, &child.y_zcol, &run.y_zcol, "y_zcol", idx, None)?;
        // fold_digest_fields: [Var; 4] in both.
        for k in 0..4 {
            builder.enforce_eq(
                &Lc::from_var(child.fold_digest_fields[k]),
                &Lc::from_var(run.fold_digest_fields[k]),
            );
        }
    }
    Ok(())
}

fn enforce_vec_var_eq(
    builder: &mut R1csBuilder,
    a: &[Var],
    b: &[Var],
    name: &'static str,
    idx: usize,
) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "CE continuity {name} length mismatch at {idx}: child={} run={}",
            a.len(),
            b.len()
        ));
    }
    for (va, vb) in a.iter().zip(b.iter()) {
        builder.enforce_eq(&Lc::from_var(*va), &Lc::from_var(*vb));
    }
    Ok(())
}

fn enforce_vec_kvar_eq(
    builder: &mut R1csBuilder,
    a: &[KVar],
    b: &[KVar],
    name: &'static str,
    idx: usize,
) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "CE continuity {name} length mismatch at {idx}: child={} run={}",
            a.len(),
            b.len()
        ));
    }
    for (ka, kb) in a.iter().zip(b.iter()) {
        builder.enforce_eq(&Lc::from_var(ka.c0), &Lc::from_var(kb.c0));
        builder.enforce_eq(&Lc::from_var(ka.c1), &Lc::from_var(kb.c1));
    }
    Ok(())
}

/// Pin a flattened base-field-limb representation (`lanes * K_LIMBS`
/// `Var`s, indexed `lane * 2 + limb`) equal to a KVar-typed lane vector.
/// `K_LIMBS = 2` for the K extension used throughout.
fn enforce_flat_limbs_vs_kvar_row(
    builder: &mut R1csBuilder,
    flat: &[Var],
    kvars: &[KVar],
    name: &'static str,
    idx: usize,
    sub: Option<usize>,
) -> Result<(), String> {
    const K_LIMBS: usize = 2;
    if flat.len() != kvars.len() * K_LIMBS {
        return Err(format!(
            "CE continuity {name} flat-vs-KVar length mismatch at {idx}{}: flat={} kvars={} (expected {} flat for {} kvars)",
            sub.map(|j| format!("[{j}]")).unwrap_or_default(),
            flat.len(),
            kvars.len(),
            kvars.len() * K_LIMBS,
            kvars.len(),
        ));
    }
    for (lane, k) in kvars.iter().enumerate() {
        builder.enforce_eq(&Lc::from_var(flat[lane * K_LIMBS]), &Lc::from_var(k.c0));
        builder.enforce_eq(&Lc::from_var(flat[lane * K_LIMBS + 1]), &Lc::from_var(k.c1));
    }
    Ok(())
}

fn flat_kvars(flat: &[Var], lanes: usize) -> Result<Vec<KVar>, String> {
    const K_LIMBS: usize = 2;
    if flat.len() != lanes * K_LIMBS {
        return Err(format!(
            "flat limb length {} does not match lanes {} × K_LIMBS {}",
            flat.len(),
            lanes,
            K_LIMBS
        ));
    }
    Ok(flat
        .chunks_exact(K_LIMBS)
        .map(|chunk| KVar::new(chunk[0], chunk[1]))
        .collect())
}

/// Emit the terminal final-fold NIFS.V inside the builder + enforce the
/// terminal latest recursive link. Returns
/// `(terminal_fold_emitted, terminal_latest_link,
/// terminal_parent_authority_link, post_fold_acc_digest,
/// terminal_running_wires, terminal_children)`. The decider uses the
/// running wires for the final CE-claim continuity link (terminal fold's
/// running == last recursive F' step's children), and the children wires
/// as the terminal CE-relation closure's claim inputs — the NIFS-output
/// CE claims that `enforce_final_ce_relations` binds to the opened
/// witnesses.
fn emit_terminal_fold(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    last: &FPrimeStepOutput,
    running_pre_final_fold: &crate::paper::construction2::RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
) -> Result<
    (
        bool,
        bool,
        bool,
        [Var; 4],
        Vec<SplitNcPiCcsOutputWires>,
        Vec<crate::paper::reductions::pi_dec_circuit::CeClaimWires>,
    ),
    decider::Error,
> {
    let terminal_start = builder.rows();
    let nifs_config = NifsVCircuitConfig {
        pi_ccs: split_nc_config(prep).map_err(|e| decider::Error::WalkFailed(format!("split_nc_config: {e}")))?,
    };

    let mut transcript = TranscriptGadget::new(builder, FINAL_FOLD_TRANSCRIPT_LABEL);
    let nifs_msg = NifsVCircuitMessages {
        fresh: trailing_latest,
        running: &running_pre_final_fold.claims,
        running_parent_authority: running_pre_final_fold.parent_authority.as_ref(),
        pi_ccs: &final_fold_nifs.pi_ccs,
        combined: &final_fold_nifs.pi_rlc.combined,
        children: &final_fold_nifs.pi_dec.children,
    };
    let nifs_outputs =
        enforce_nifs_v_circuit_with_transcript(builder, &prep.params, &nifs_config, &mut transcript, &nifs_msg)
            .map_err(|e| decider::Error::WalkFailed(format!("terminal fold NIFS.V emission: {e}")))?;
    builder.record_row_family("terminal.nifs", terminal_start);

    // Pin terminal fold's input running to last F' step's acc_digest.
    let running_link_start = builder.rows();
    enforce_digest_eq(builder, &nifs_outputs.running_acc_digest, &last.state_out.acc_digest);
    builder.record_row_family("terminal.running_link", running_link_start);
    let mut terminal_parent_authority_link = false;
    if let (Some(prev_parent), Some(curr_parent)) = (
        last.nifs_parent.as_ref(),
        nifs_outputs.running_parent_authority.as_ref(),
    ) {
        let parent_link_start = builder.rows();
        enforce_children_equal_running(
            builder,
            std::slice::from_ref(prev_parent),
            std::slice::from_ref(curr_parent),
        )
        .map_err(|e| decider::Error::WalkFailed(format!("terminal parent-authority continuity: {e}")))?;
        builder.record_row_family("terminal.parent_link", parent_link_start);
        terminal_parent_authority_link = true;
    }

    // Terminal latest recursive link: fresh.x[0]==1 and
    // fresh.x[1..]==last.x_out_bits.
    let latest_link_start = builder.rows();
    enforce_terminal_latest_link(
        builder,
        f_prime_public_input_layout(prep),
        &nifs_outputs.fresh_x,
        &last.x_out_bits,
    )?;
    builder.record_row_family("terminal.latest_link", latest_link_start);

    // NIFS.V has just enforced strict Pi_DEC(parent, children), so the
    // parent CE digest is the post-fold accumulator authority.
    let accumulator_start = builder.rows();
    let parent_digest = enforce_dec_ce_claim_digest(builder, &nifs_outputs.parent)?;
    let post_fold_acc_digest =
        enforce_accumulator_digest_from_parent_circuit(builder, nifs_outputs.children.len(), Some(parent_digest));
    builder.record_row_family("terminal.accumulator", accumulator_start);
    builder.record_row_family("terminal.total", terminal_start);

    Ok((
        true,
        true,
        terminal_parent_authority_link,
        post_fold_acc_digest,
        nifs_outputs.running,
        nifs_outputs.children,
    ))
}

fn enforce_dec_ce_claim_digest(builder: &mut R1csBuilder, claim: &CeClaimWires) -> Result<[Var; 4], decider::Error> {
    let y_ring = dec_y_ring_kvars(claim)
        .map_err(|e| decider::Error::WalkFailed(format!("terminal accumulator parent CE digest y_ring: {e}")))?;
    enforce_ce_claim_digest(
        builder,
        &CeClaimDigestInputs {
            c_d: claim.c_d,
            c_kappa: claim.c_kappa,
            c_data: &claim.c_data,
            x_rows: claim.x_rows,
            x_cols: claim.x_cols,
            x_flat_row_major: &claim.x,
            r: &claim.r,
            y_ring: &y_ring,
            m_in: claim.m_in,
            fold_digest_fields: claim.fold_digest_fields,
            adv: claim.adv.as_ref(),
        },
    )
    .map_err(|e| decider::Error::WalkFailed(format!("terminal accumulator parent CE digest: {e}")))
}

fn dec_y_ring_kvars(claim: &CeClaimWires) -> Result<Vec<Vec<KVar>>, String> {
    let mut rows = Vec::with_capacity(claim.y_ring.len());
    for (j, row) in claim.y_ring.iter().enumerate() {
        rows.push(flat_kvars(row, claim.y_ring_lanes).map_err(|e| format!("y_ring[{j}] {e}"))?);
    }
    Ok(rows)
}

/// Constrain every terminal-fold fresh CCS instance's public input to
/// encode the last F' step's `x_out`. The trailing latest is one
/// SuperNeo chunk rooted at the last step's `x_out`, so every fresh in
/// the batch shares the same recursive link. Specifically, for every
/// `fresh_x[i]`:
///   - `fresh_x[i].len()` equals the verifier-owned F' public layout.
///   - `fresh_x[i][0] == 1` (CCS constant-one slot).
///   - the canonical `fresh_x[i][1..257]` prefix equals
///     `last_x_out_bits` (bit-by-bit); an application suffix is preserved
///     for its own delayed consumer.
fn enforce_terminal_latest_link(
    builder: &mut R1csBuilder,
    layout: FPrimePublicInputLayout,
    fresh_x: &[Vec<Var>],
    last_x_out_bits: &[Var],
) -> Result<(), decider::Error> {
    if fresh_x.is_empty() {
        return Err(decider::Error::WalkFailed(
            "terminal fold must have a non-empty fresh batch".into(),
        ));
    }
    if last_x_out_bits.len() != F_PRIME_ENC_INST_BITS {
        return Err(decider::Error::WalkFailed(format!(
            "last step's x_out_bits length {} != F_PRIME_ENC_INST_BITS ({F_PRIME_ENC_INST_BITS})",
            last_x_out_bits.len()
        )));
    }
    for (idx, x) in fresh_x.iter().enumerate() {
        if x.len() != layout.total_len() {
            return Err(decider::Error::WalkFailed(format!(
                "terminal fresh[{idx}] public input has length {}, expected {}",
                x.len(),
                layout.total_len(),
            )));
        }
        builder.enforce_eq(&Lc::from_var(x[F_PRIME_PUBLIC_ONE_OFFSET]), &Lc::from_const(F::ONE));
        let link_end = F_PRIME_ENC_INST_OFFSET + F_PRIME_ENC_INST_BITS;
        for (fresh_bit, out_bit) in x[F_PRIME_ENC_INST_OFFSET..link_end]
            .iter()
            .zip(last_x_out_bits)
        {
            builder.enforce_eq(&Lc::from_var(*fresh_bit), &Lc::from_var(*out_bit));
        }
    }
    Ok(())
}

/// Pin every terminal field of `statement.public` to chain-derived
/// wires. The initial semantic-state digest is pinned in
/// `enforce_base_state_constants`, because it is part of the base seed.
/// Returns [`REQUIRED_PUBLIC_IMAGE_PINS`].
fn pin_public_image(
    builder: &mut R1csBuilder,
    public: &PublicImage,
    prep: &Preprocessing,
    last: &FPrimeStepOutput,
    final_acc_digest: &[Var; 4],
) -> usize {
    let so = &last.state_out;
    enforce_public_preprocessing_anchors(builder, prep, public);
    // 1. vk_fs_digest.
    pin_digest32(builder, &so.vk_fs_digest, public.vk_fs_digest);
    // 2. chunk_count — final-fold does not advance counters.
    pin_u64(builder, so.chunk_count, public.chunk_count);
    // 3. step_count — same.
    pin_u64(builder, so.step_count, public.step_count);
    // 4. z_0.
    pin_digest32(builder, &so.z_0, public.z_0);
    // 5. z_i — last F' step's value; final-fold does not update it.
    pin_digest32(builder, &so.z_i, public.z_i);
    // 6. pc.
    pin_u64(builder, so.pc, public.pc);
    // 7. semantic_state_digest — final app / VM state.
    pin_digest32(builder, &so.semantic_state_digest, public.semantic_state_digest);
    if matches!(prep.semantic_state_mode(), SemanticStateMode::Stateless) {
        // Stateless finalization does not advance the semantic coordinate:
        // it remains the pre-terminal accumulator digest carried by the last
        // F' step. The stateless `state_x_out` preimage omits the duplicate
        // semantic lanes, so this equality is load-bearing for public-image
        // binding.
        enforce_digest_eq(builder, &so.semantic_state_digest, &so.acc_digest);
    }
    // 8. acc_digest — post-final-fold Construction-2 accumulator.
    pin_digest32(builder, final_acc_digest, public.acc_digest);
    // 9. public_trace — last F' step's value.
    pin_digest32(builder, &so.public_trace, public.public_trace);
    // 10. x_out — recomputed in-circuit from the post-fold state.
    let terminal_x_out_inputs = StateXOutDigestInputs {
        mode: state_x_out_digest_mode(prep),
        vk_fs_digest: so.vk_fs_digest,
        pi_ccs_header_bundle: so.pi_ccs_header_bundle,
        structure_digest: so.pi_ccs_header_bundle,
        chunk_count: so.chunk_count,
        step_count: so.step_count,
        initial_boundary: so.z_0,
        current_boundary: so.z_i,
        pc: so.pc,
        semantic_acc: so.semantic_state_digest,
        construction2_acc: *final_acc_digest,
        public_trace: so.public_trace,
    };
    let terminal_x_out = enforce_state_x_out_digest_circuit(builder, &terminal_x_out_inputs);
    pin_digest32(builder, &terminal_x_out, public.x_out.digest_bytes);
    // Belt-and-braces: pin the SplitNc header to preprocessing.
    let header_bundle = prep.pi_ccs_header_bundle();
    for k in 0..4 {
        builder.enforce_eq(
            &Lc::from_var(so.pi_ccs_header_bundle[k]),
            &Lc::from_const(header_bundle[k]),
        );
    }
    REQUIRED_PUBLIC_IMAGE_PINS
}

fn enforce_public_preprocessing_anchors(builder: &mut R1csBuilder, prep: &Preprocessing, public: &PublicImage) {
    let structure_lanes = *prep.structure_digest();
    let expected_z_0 = initial_boundary_digest(&structure_lanes, prep.public_input_len);

    enforce_digest32_const_eq(builder, public.vk_fs_digest, prep.vk.digest());
    enforce_digest32_const_eq(builder, public.z_0, expected_z_0);
    enforce_digest32_const_eq(
        builder,
        public.initial_semantic_state_digest,
        prep.initial_semantic_state_digest(),
    );
}

fn enforce_digest32_const_eq(builder: &mut R1csBuilder, actual: [u8; 32], expected: [u8; 32]) {
    let Some(actual) = canonical_digest32_fields(actual) else {
        enforce_unsat(builder);
        return;
    };
    let Some(expected) = canonical_digest32_fields(expected) else {
        enforce_unsat(builder);
        return;
    };
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_const(actual[k]), &Lc::from_const(expected[k]));
    }
}

fn pin_digest32(builder: &mut R1csBuilder, wires: &[Var; 4], expected: [u8; 32]) {
    let Some(expected_lanes) = canonical_digest32_fields(expected) else {
        enforce_unsat(builder);
        return;
    };
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_var(wires[k]), &Lc::from_const(expected_lanes[k]));
    }
}

fn pin_u64(builder: &mut R1csBuilder, wire: Var, expected: u64) {
    // Public u64s are pinned to single Goldilocks state wires. Values
    // outside the canonical field range would alias through F::from_u64.
    if expected >= F::ORDER_U64 {
        enforce_unsat(builder);
        return;
    }
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_u64(expected)));
}

fn canonical_digest32_fields(bytes: [u8; 32]) -> Option<[F; 4]> {
    let mut fields = [F::ZERO; 4];
    for (lane, out) in fields.iter_mut().enumerate() {
        let start = lane * 8;
        let value = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("8-byte digest limb"),
        );
        if value >= F::ORDER_U64 {
            return None;
        }
        *out = F::from_u64(value);
    }
    Some(fields)
}

fn enforce_unsat(builder: &mut R1csBuilder) {
    builder.enforce_eq(&Lc::zero(), &Lc::from_const(F::ONE));
}

fn state_x_out_lanes(prep: &Preprocessing, state: &State) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        state_x_out_digest_mode(prep),
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        state.nebula.as_ref().map(|lane| lane.digest()),
    ))
}

fn state_x_out_digest_mode(prep: &Preprocessing) -> StateXOutDigestMode {
    match prep.semantic_state_mode() {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    }
}

fn f_prime_public_input_layout(prep: &Preprocessing) -> FPrimePublicInputLayout {
    match prep.nebula() {
        None => FPrimePublicInputLayout::plain(),
        Some(config) => FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(config.stacks)),
    }
}

fn split_nc_config(prep: &Preprocessing) -> Result<SplitNcPiCcsVConfig<'_>, String> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        crate::config::MIN_EFFECTIVE_LAMBDA,
        crate::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .map_err(|e| format!("raw params: {e}"))?;
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure())
        .map_err(|e| format!("dims: {e}"))?;
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &mat_digest,
    )
    .map_err(|e| format!("header bundle: {e}"))?;
    Ok(SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    })
}
