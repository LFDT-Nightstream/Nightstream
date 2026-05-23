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
//! This module does **not** create, verify, or serialize a Spartan proof.
//! Do not use this full-history builder for production compression sizing.
//! The constant-size terminal decider belongs to the future `F'` frontend
//! path, where each online step folds `enc(F')` and the final SNARK proves
//! only the terminal folded accumulator.
//!
//! ## What the relation enforces (in-circuit only)
//!
//! 1. **Native preflight** ([`crate::paper::decider::validate_witness`]):
//!    sanity check that errors out early; **not part of the SNARK
//!    statement**. The R1CS below stands on its own.
//! 2. **Base F' step + canonical base-state pins**: the first lifecycle
//!    step (`FoldProof::NoFold`) emits
//!    [`enforce_f_prime_base_step_circuit`], then
//!    `enforce_base_state_constants` pins every base-state seed wire
//!    (`vk_fs_digest`, `structure_digest`, `z_0`, `z_i = z_0`,
//!    `public_trace_seed`, empty `acc_digest`, counters = 0, pc =
//!    `TRIVIAL_PC`) to the canonical preprocessing-derived constants.
//!    This anchors the start of the chain to preprocessing in-circuit;
//!    a SNARK verifier can reject a statement whose base seeds disagree
//!    with `prep.vk.digest()` etc. without trusting native preflight.
//! 3. **Recursive F' steps**: every `FoldProof::Recursive` emits
//!    [`enforce_f_prime_recursive_step_circuit`]. The source-image
//!    `prior_x_out_bits` chain implicitly carries each step's `x_out`
//!    into the next step's `fresh.x` under Poseidon collision-resistance.
//! 4. **Cross-step state-link constraints**: for every adjacent pair of
//!    F' steps `(prev, next)`, `enforce_state_link(prev.state_out,
//!    next.state_in)` pins every state field wire-to-wire. The chain is
//!    one continuous in-circuit object, not a series of islands.
//! 5. **CE-claim continuity links**: between every adjacent NIFS.V step
//!    (recursive→recursive and last-recursive→terminal-fold),
//!    `enforce_children_equal_running` pins `prev.children ==
//!    next.running` wire-for-wire across every CE field — `c_data`,
//!    `x`, `r`, `s_col`, `y_ring`, `y_zcol`, `fold_digest_fields`, plus
//!    shape constants. This goes beyond the commitment-only continuity
//!    that `state_out.acc_digest` chaining provides.
//! 6. **Terminal final-fold NIFS.V**: the witness's `final_fold.nifs` is
//!    replayed under [`FINAL_FOLD_TRANSCRIPT_LABEL`]. The terminal
//!    fold's input running is pinned to the last F' step's `acc_digest`
//!    via [`NifsVOutputs::running_acc_digest`] *and* to the last
//!    recursive step's NIFS children via the CE-continuity link.
//! 7. **Terminal latest link**: the terminal fold's `fresh.x[0]` is
//!    pinned to `1` (CCS constant-one slot) and `fresh.x[1..]` is pinned
//!    to the last F' step's `x_out_bits`. The trailing latest is bound
//!    to the actual chain output, not an attacker-chosen value.
//! 8. **Nine public-image pins**: every field of
//!    [`crate::paper::decider::PublicImage`] is bound to chain-derived
//!    wires. The terminal `x_out` is recomputed in-circuit from the
//!    post-fold state and pinned to `statement.public.x_out`.
//!
//! Each audit layer's completeness is reported on [`DeciderR1csSynthesis`].
//! [`DeciderR1csSynthesis::is_self_sufficient_relation`] returns `true`
//! iff every flag/count is at its full value and the builder is
//! satisfied — that is a full-history audit readiness marker, nothing
//! more.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget, Var};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::finalization::FINAL_FOLD_TRANSCRIPT_LABEL;
use crate::paper::construction2::{self, FoldProof, ProofState, State, TRIVIAL_PC};
use crate::paper::decider::{self, PublicImage, Statement};
use crate::paper::digest::{
    accumulator_digest_from_claims, digest32_as_fields, f_prime_chunk_public_digest, initial_boundary_digest,
    public_trace_seed_digest, state_x_out_digest,
};
use crate::paper::f_prime::digest_circuit::{enforce_state_x_out_digest_circuit, StateXOutDigestInputs};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs, FPrimeRecursiveInputs,
    FPrimeStateIn, FPrimeStateWires, FPrimeStepConfig, FPrimeStepOutput, F_PRIME_ENC_INST_BITS,
    F_PRIME_ENC_INST_OFFSET, F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_PUBLIC_ONE_OFFSET,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::nifs::circuit::{enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::nifs::NifsProof;
use crate::paper::reductions::accumulator_digest_circuit::enforce_accumulator_digest_from_parent_circuit;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{SplitNcPiCcsOutputWires, SplitNcPiCcsVConfig};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;
use crate::paper::relations::CcsClaim;

/// Full-history audit R1CS output plus relation-completeness tracking.
///
/// This builder replays the whole direct-CCS interim transcript. It is
/// intentionally self-contained, but it is not the constant-size IVC
/// terminal decider. `is_self_sufficient_relation()` means the audit
/// relation is complete and satisfied:
///
/// - `base_step_emitted == true`
/// - `base_state_pinned == true` (base seed values pinned to canonical
///   preprocessing-derived constants in-circuit)
/// - `recursive_step_count == N - 1` for an N-batch chain
/// - `cross_step_links == N - 1` (one link per adjacent pair)
/// - `accumulator_claim_links == recursive_step_count` (full CE
///   continuity between every adjacent NIFS.V step)
/// - `terminal_latest_link == true`
/// - `terminal_fold_emitted == true`
/// - `public_image_pins == REQUIRED_PUBLIC_IMAGE_PINS`
pub struct DeciderR1csSynthesis {
    pub builder: R1csBuilder,
    /// `true` once the base F' step has been emitted in-circuit.
    pub base_step_emitted: bool,
    /// `true` once the base step's seed wires (vk_fs_digest,
    /// structure_digest, z_0, z_i, public_trace, empty acc_digest, plus
    /// the zero counters and `pc == TRIVIAL_PC`) are pinned to their
    /// canonical preprocessing-derived values in-circuit.
    pub base_state_pinned: bool,
    /// Number of `FoldProof::Recursive` steps emitted.
    pub recursive_step_count: usize,
    /// Count of `enforce_state_link` invocations between adjacent F' steps.
    /// For an N-batch chain there are `N - 1` links (base→rec, rec→rec, ...).
    pub cross_step_links: usize,
    /// `true` once `terminal_fresh.x[1..] == last.x_out_bits` is enforced
    /// (with `x[0] == 1`).
    pub terminal_latest_link: bool,
    /// `true` once the terminal `final_fold` NIFS.V is re-emitted
    /// in-circuit under [`FINAL_FOLD_TRANSCRIPT_LABEL`].
    pub terminal_fold_emitted: bool,
    /// Count of in-circuit `statement.public` field pins (out of
    /// [`REQUIRED_PUBLIC_IMAGE_PINS`]).
    pub public_image_pins: usize,
    /// Number of CE-claim continuity links emitted between adjacent
    /// NIFS.V steps. For an N-batch chain (1 base + (N-1) recursive +
    /// terminal fold) this equals `recursive_step_count`: each
    /// recursive step's `nifs_children` is pinned wire-for-wire to the
    /// next step's `nifs_running` (or to the terminal fold's running
    /// for the last recursive step). This goes beyond the
    /// commitment-only continuity that `state_out.acc_digest` chaining
    /// provides — it binds `(c_data, x, r, y_ring, y_zcol, s_col,
    /// fold_digest)` across every step boundary.
    pub accumulator_claim_links: usize,
}

/// Number of fields in `decider::PublicImage` that must be pinned
/// in-circuit before the R1CS is a self-sufficient SNARK relation.
pub const REQUIRED_PUBLIC_IMAGE_PINS: usize = 9;

impl DeciderR1csSynthesis {
    /// Single-call readiness gate. Returns `true` exactly when every
    /// completeness flag/count is at its full value **and** the builder
    /// is satisfied — i.e. the synthesized full-history audit R1CS is
    /// self-contained.
    ///
    /// This is not a production compression gate. The production IVC
    /// terminal decider should be constant-size in the number of steps
    /// and must be built around folded `enc(F')` instances.
    pub fn is_self_sufficient_relation(&self) -> bool {
        self.base_step_emitted
            && self.base_state_pinned
            && self.cross_step_links == self.recursive_step_count
            && self.accumulator_claim_links == self.recursive_step_count
            && self.terminal_latest_link
            && self.terminal_fold_emitted
            && self.public_image_pins == REQUIRED_PUBLIC_IMAGE_PINS
            && self.builder.is_satisfied()
    }
}

/// Run the non-SNARK preflight on `statement`, then synthesize the
/// full-history audit R1CS for the direct-CCS interim path. This module
/// stops here: no Spartan proof is created, verified, or serialized.
///
/// Errors propagate from [`crate::paper::decider::validate_witness`] and
/// from in-circuit emission (wrapped in [`decider::Error::WalkFailed`]).
/// This relation grows linearly with the number of steps and should not
/// be used to size the future constant-size IVC terminal decider.
pub fn synthesize_statement_r1cs(
    prep: &Preprocessing,
    statement: &Statement,
) -> Result<DeciderR1csSynthesis, decider::Error> {
    // 1. Preflight (sanity, not part of SNARK).
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
        statement,
    )?;

    // 2-4. F' chain (base + recursive steps + cross-step links).
    let structure_digest_v = *prep.structure_digest();
    let z_0 = initial_boundary_digest(&structure_digest_v, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure_digest_v);
    let acc_digest = accumulator_digest_from_claims(prep.params.b(), &[]);
    let mut state = State::base(z_0, public_trace, acc_digest);

    let mut builder = R1csBuilder::new();
    let mut base_step_emitted = false;
    let mut base_state_pinned = false;
    let mut recursive_step_count = 0;
    let mut cross_step_links = 0;
    let mut accumulator_claim_links = 0;
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
        )
        .map_err(|e| decider::Error::WalkFailed(format!("step {idx}: {e}")))?;

        let output = match &step_proof.fold {
            FoldProof::NoFold => {
                base_step_emitted = true;
                let out = emit_base_step_r1cs(&mut builder, prep, &state_in, &state, public_batch)
                    .map_err(|e| decider::Error::WalkFailed(format!("emit F' base step {idx}: {e}")))?;
                enforce_base_state_constants(&mut builder, prep, &out);
                base_state_pinned = true;
                out
            }
            FoldProof::Recursive(nifs) => {
                recursive_step_count += 1;
                emit_recursive_step_r1cs(&mut builder, prep, &state_in, &state, public_batch, nifs)
                    .map_err(|e| decider::Error::WalkFailed(format!("emit F' recursive step {idx}: {e}")))?
            }
        };

        if let Some(prev) = last_output.as_ref() {
            enforce_state_link(&mut builder, &prev.state_out, &output.state_in);
            cross_step_links += 1;
        }

        // CE-claim continuity: previous step's NIFS children must equal
        // this step's NIFS running, wire-for-wire (not just by digest).
        // Skipped if either side has no NIFS.V (base step).
        if let (Some(prev_children), Some(curr_running)) = (previous_children.as_ref(), output.nifs_running.as_ref()) {
            enforce_children_equal_running(&mut builder, prev_children, curr_running)
                .map_err(|e| decider::Error::WalkFailed(format!("CE continuity step {idx}: {e}")))?;
            accumulator_claim_links += 1;
        }
        if let (Some(prev_parent), Some(curr_parent)) =
            (previous_parent.as_ref(), output.nifs_running_parent_authority.as_ref())
        {
            enforce_children_equal_running(
                &mut builder,
                std::slice::from_ref(prev_parent),
                std::slice::from_ref(curr_parent),
            )
            .map_err(|e| decider::Error::WalkFailed(format!("parent-authority continuity step {idx}: {e}")))?;
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

    let (terminal_fold_emitted, terminal_latest_link, final_acc_digest, terminal_running) = emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        &running_pre_final_fold,
        trailing_latest,
        &final_fold.nifs,
    )?;

    // Final CE-claim continuity link: terminal fold's running must equal
    // the last recursive F' step's children.
    if let Some(prev_children) = previous_children.as_ref() {
        enforce_children_equal_running(&mut builder, prev_children, &terminal_running)
            .map_err(|e| decider::Error::WalkFailed(format!("CE continuity terminal fold: {e}")))?;
        accumulator_claim_links += 1;
    }

    // 7. Public-image pins.
    let public_image_pins = pin_public_image(&mut builder, &statement.public, prep, &last, &final_acc_digest);

    Ok(DeciderR1csSynthesis {
        builder,
        base_step_emitted,
        base_state_pinned,
        recursive_step_count,
        cross_step_links,
        terminal_latest_link,
        terminal_fold_emitted,
        public_image_pins,
        accumulator_claim_links,
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
/// A future "pure accumulator-only" terminal decider would emit just
/// (b) + (c) and pin the latest F' relation's correctness via an
/// in-circuit Spartan verification of the running accumulator. That
/// is out of scope for this milestone.
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
/// state_in), but only one F' step's R1CS lands in the builder, so
/// `builder.rows()` is constant in the steady-state (last step folding
/// `k_rho` → `k_rho`).
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
        &statement,
    )?;

    // 2. Walk natively to compute the last step's state_in / state_out.
    //    No R1CS rows are emitted here — `verify_step` runs out-of-circuit.
    let structure_digest_v = *prep.structure_digest();
    let z_0 = initial_boundary_digest(&structure_digest_v, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure_digest_v);
    let acc_digest = accumulator_digest_from_claims(prep.params.b(), &[]);
    let mut state = State::base(z_0, public_trace, acc_digest);

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
            enforce_base_state_constants(&mut builder, prep, &out);
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
    let (_terminal_fold_emitted, _terminal_latest_link, final_acc_digest, terminal_running) = emit_terminal_fold(
        &mut builder,
        prep,
        &last_output,
        &running_pre_final_fold,
        last_public_batch,
        &final_fold.nifs,
    )?;

    // 5. Public-image pins. Anchors the relation to a SNARK-verifiable
    //    public statement.
    let public_image_pins = pin_public_image(&mut builder, &statement.public, prep, &last_output, &final_acc_digest);

    Ok(LastStepTerminalSynthesis {
        builder,
        running_claim_count: terminal_running.len(),
        has_final_fold: true,
        public_image_pins,
    })
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
        structure_digest: *prep.structure_digest(),
        chunk_count_in: state_in.chunk_count,
        step_count_in: state_in.step_count,
        z_0: digest32_as_fields(state_in.z_0),
        z_i_in: digest32_as_fields(state_in.z_i),
        pc: state_in.pc,
        acc_digest_in: digest32_as_fields(state_in.acc_digest),
        public_trace_in: digest32_as_fields(state_in.public_trace),
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
    };
    let rows_in_chunk = public_batch.len() as u64;
    let inputs = FPrimeBaseInputs {
        state: f_state,
        chunk_digest,
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
        structure_digest: *prep.structure_digest(),
        chunk_count_in: state_in.chunk_count,
        step_count_in: state_in.step_count,
        z_0: digest32_as_fields(state_in.z_0),
        z_i_in: digest32_as_fields(state_in.z_i),
        pc: state_in.pc,
        acc_digest_in: digest32_as_fields(state_in.acc_digest),
        public_trace_in: digest32_as_fields(state_in.public_trace),
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
    };
    let inputs = FPrimeRecursiveInputs {
        state: f_state,
        chunk_digest,
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
/// seed values: `vk_fs_digest`, `structure_digest`, `z_0`, `z_i = z_0`,
/// `public_trace_seed`, empty `acc_digest`, zero counters, `pc == TRIVIAL_PC`.
///
/// Without this pin, a SNARK verifier would have no way to reject a
/// statement whose base seeds disagree with preprocessing — the F' base
/// step itself only enforces shape (counters=0, z_i==z_0, acc==empty), not
/// the seed values. Native `validate_witness` catches it, but the R1CS
/// must stand alone.
fn enforce_base_state_constants(builder: &mut R1csBuilder, prep: &Preprocessing, base: &FPrimeStepOutput) {
    let structure_lanes = *prep.structure_digest();
    let z_0_bytes = initial_boundary_digest(&structure_lanes, prep.public_input_len);
    let public_trace_bytes = public_trace_seed_digest(&structure_lanes);
    let empty_acc_bytes = accumulator_digest_from_claims(prep.params.b(), &[]);

    pin_digest32(builder, &base.state_in.vk_fs_digest, prep.vk.digest());
    pin_digest_fields(builder, &base.state_in.structure_digest, structure_lanes);
    pin_digest32(builder, &base.state_in.z_0, z_0_bytes);
    // Base step also enforces z_i == z_0 in-circuit, but pinning here
    // gives the SNARK verifier a direct constant to compare against.
    pin_digest32(builder, &base.state_in.z_i, z_0_bytes);
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
    enforce_digest_eq(builder, &a.structure_digest, &b.structure_digest);
    builder.enforce_eq(&Lc::from_var(a.chunk_count), &Lc::from_var(b.chunk_count));
    builder.enforce_eq(&Lc::from_var(a.step_count), &Lc::from_var(b.step_count));
    enforce_digest_eq(builder, &a.z_0, &b.z_0);
    enforce_digest_eq(builder, &a.z_i, &b.z_i);
    builder.enforce_eq(&Lc::from_var(a.pc), &Lc::from_var(b.pc));
    enforce_digest_eq(builder, &a.acc_digest, &b.acc_digest);
    enforce_digest_eq(builder, &a.public_trace, &b.public_trace);
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
///   - both share Vec<KVar> for r / s_col.
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
        enforce_vec_var_eq(builder, &child.x, &run.x, "x", idx)?;
        // r, s_col are Vec<KVar> in both — KVar exposes c0/c1 directly.
        enforce_vec_kvar_eq(builder, &child.r, &run.r, "r", idx)?;
        enforce_vec_kvar_eq(builder, &child.s_col, &run.s_col, "s_col", idx)?;
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

/// Emit the terminal final-fold NIFS.V inside the builder + enforce the
/// terminal latest recursive link. Returns
/// `(terminal_fold_emitted, terminal_latest_link, post_fold_acc_digest,
/// terminal_running_wires)`. The decider uses the running wires for the
/// final CE-claim continuity link (terminal fold's running == last
/// recursive F' step's children).
fn emit_terminal_fold(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    last: &FPrimeStepOutput,
    running_pre_final_fold: &crate::paper::construction2::RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
) -> Result<(bool, bool, [Var; 4], Vec<SplitNcPiCcsOutputWires>), decider::Error> {
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

    // Pin terminal fold's input running to last F' step's acc_digest.
    enforce_digest_eq(builder, &nifs_outputs.running_acc_digest, &last.state_out.acc_digest);
    if let (Some(prev_parent), Some(curr_parent)) = (
        last.nifs_parent.as_ref(),
        nifs_outputs.running_parent_authority.as_ref(),
    ) {
        enforce_children_equal_running(
            builder,
            std::slice::from_ref(prev_parent),
            std::slice::from_ref(curr_parent),
        )
        .map_err(|e| decider::Error::WalkFailed(format!("terminal parent-authority continuity: {e}")))?;
    }

    // Terminal latest recursive link: fresh.x[0]==1 and
    // fresh.x[1..]==last.x_out_bits.
    enforce_terminal_latest_link(builder, &nifs_outputs.fresh_x, &last.x_out_bits)?;

    // Compute post-fold accumulator digest from NIFS.V's parent.
    let k_carry = final_fold_nifs.pi_dec.children.len();
    let post_fold_acc_digest =
        enforce_accumulator_digest_from_parent_circuit(builder, k_carry, &nifs_outputs.parent_c_data);

    Ok((true, true, post_fold_acc_digest, nifs_outputs.running))
}

/// Constrain every terminal-fold fresh CCS instance's public input to
/// encode the last F' step's `x_out`. The trailing latest is one
/// SuperNeo chunk rooted at the last step's `x_out`, so every fresh in
/// the batch shares the same recursive link. Specifically, for every
/// `fresh_x[i]`:
///   - `fresh_x[i].len() == F_PRIME_PUBLIC_INPUT_LEN`.
///   - `fresh_x[i][0] == 1` (CCS constant-one slot).
///   - `fresh_x[i][1..] == last_x_out_bits` (bit-by-bit).
fn enforce_terminal_latest_link(
    builder: &mut R1csBuilder,
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
        if x.len() != F_PRIME_PUBLIC_INPUT_LEN {
            return Err(decider::Error::WalkFailed(format!(
                "terminal fresh[{idx}] public input has length {}, expected {F_PRIME_PUBLIC_INPUT_LEN}",
                x.len()
            )));
        }
        builder.enforce_eq(&Lc::from_var(x[F_PRIME_PUBLIC_ONE_OFFSET]), &Lc::from_const(F::ONE));
        for (fresh_bit, out_bit) in x[F_PRIME_ENC_INST_OFFSET..].iter().zip(last_x_out_bits) {
            builder.enforce_eq(&Lc::from_var(*fresh_bit), &Lc::from_var(*out_bit));
        }
    }
    Ok(())
}

/// Pin every field of `statement.public` to chain-derived wires. Returns
/// [`REQUIRED_PUBLIC_IMAGE_PINS`].
fn pin_public_image(
    builder: &mut R1csBuilder,
    public: &PublicImage,
    prep: &Preprocessing,
    last: &FPrimeStepOutput,
    final_acc_digest: &[Var; 4],
) -> usize {
    let so = &last.state_out;
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
    // 7. acc_digest — post-final-fold value.
    pin_digest32(builder, final_acc_digest, public.acc_digest);
    // 8. public_trace — last F' step's value.
    pin_digest32(builder, &so.public_trace, public.public_trace);
    // 9. x_out — recomputed in-circuit from the post-fold state.
    let terminal_x_out_inputs = StateXOutDigestInputs {
        vk_fs_digest: so.vk_fs_digest,
        structure_digest: so.structure_digest,
        chunk_count: so.chunk_count,
        step_count: so.step_count,
        initial_boundary: so.z_0,
        current_boundary: so.z_i,
        pc: so.pc,
        semantic_acc: *final_acc_digest,
        construction2_acc: *final_acc_digest,
        public_trace: so.public_trace,
    };
    let terminal_x_out = enforce_state_x_out_digest_circuit(builder, &terminal_x_out_inputs);
    pin_digest32(builder, &terminal_x_out, public.x_out.digest_bytes);
    // Belt-and-braces: pin `structure_digest` to the canonical
    // verifier-derived value (not a `PublicImage` field).
    let structure_lanes = *prep.structure_digest();
    for k in 0..4 {
        builder.enforce_eq(
            &Lc::from_var(so.structure_digest[k]),
            &Lc::from_const(structure_lanes[k]),
        );
    }
    REQUIRED_PUBLIC_IMAGE_PINS
}

fn pin_digest32(builder: &mut R1csBuilder, wires: &[Var; 4], expected: [u8; 32]) {
    let expected_lanes = digest32_as_fields(expected);
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_var(wires[k]), &Lc::from_const(expected_lanes[k]));
    }
}

fn pin_u64(builder: &mut R1csBuilder, wire: Var, expected: u64) {
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_u64(expected)));
}

fn state_x_out_lanes(prep: &Preprocessing, state: &State) -> [F; 4] {
    digest32_as_fields(state_x_out_digest(
        prep.vk.digest(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.acc_digest,
        state.acc_digest,
        state.public_trace,
    ))
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
