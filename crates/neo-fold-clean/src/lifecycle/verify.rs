//! Verifier-side lifecycle: `verify_uncompressed` + `verify_uncompressed_audit`.
//!
//! ## Two verifiers, two input types
//!
//! - [`verify_uncompressed`] **(terminal-only IVC verifier)** consumes
//!   the terminal-only [`Uncompressed`]. Constant verifier work in chain
//!   length. Authenticates the terminal fold, never iterating per-step
//!   proofs. Multi-chunk histories require preprocessing that certifies the
//!   authoritative fixed F' relation; historical image-only relations remain
//!   audit-only.
//! - [`verify_uncompressed_audit`] **(chain-replay / audit verifier)**
//!   consumes the audit-bearing [`crate::lifecycle::UncompressedAudit`].
//!   Linear in chain length. Replays every `extend`'s NIFS.V to catch
//!   audit-trail tampers (`steps`, `public_batches`) the IVC verifier
//!   intentionally ignores.
//!
//! The checked decider statement is built from
//! [`crate::lifecycle::UncompressedAudit`] for the same reason: the
//! audit trail binds the public image to a verifiable history. See
//! [`crate::lifecycle::build_decider_statement`].
//!
//! ## Contract — non-replay IVC verifier
//!
//! [`verify_uncompressed`] is the compact verifier: its work is constant in
//! chain length. For plain authoritative F' it checks HyperNova's running
//! accumulator and latest fresh relation separately. Nebula additionally
//! authenticates its terminal NIFS fold to consume the delayed memory claim.
//! The walk over per-step F' proofs lives in
//! [`verify_uncompressed_audit`] (and the decider's `validate_witness`);
//! audit-trail tampers are caught there, not here.
//!
//! This distinction is load-bearing. The compact artifact has dropped
//! the intermediate step witnesses and NIFS.V messages. For the authoritative
//! fixed relation, those checks were constraints of every folded F' instance,
//! so the running accumulator plus latest relation authenticate them
//! inductively. Other frontends do not own that relation and remain rejected.
//!
//! The plain HyperNova branch checks the opened running CE accumulator and the
//! latest CCS relation directly. For Nebula or other terminal-fold inputs,
//! the verifier additionally follows this path:
//! the prover stored in `final_fold.terminal_inputs`, the verifier:
//!
//! 1. Reconstructs the pre-final-fold `State` from
//!    `proof.state`'s chain coordinates + `terminal_inputs.pre_final_running`
//!    + `terminal_inputs.latest`.
//! 2. Calls [`construction2::verify_final_fold`], which:
//!    - runs Π_CCS / Π_RLC / Π_DEC on `(pre_final_running, latest)` with a
//!      verifier-driven transcript → derives `post_running` whose
//!      sumcheck point `r` is verifier-bound (not prover-supplied);
//!    - asserts the resulting `state_after.x_out` equals
//!      `final_fold.x_out` (chain-coordinate binding).
//! 3. Binds the verifier-derived `post_running.claims` to the prover's
//!    recorded `proof.state.running.claims`.
//! 4. Re-derives `acc_digest` from `proof.state.running.claims` and
//!    asserts it matches `proof.state.acc_digest`.
//! 5. Discharges every terminal witness-authority obligation against
//!    each `(claim, witness Z)` in `proof.state.running`:
//!    - `claim.m_in == prep.public_input_len` when the program fixed
//!      a public-input length,
//!    - `commit(Z) == claim.c` (Ajtai opening),
//!    - `project_x(Z) == claim.X` (public-input projection),
//!    - `||Z||_∞ < b` (low-norm),
//!    - identity-first `claim.y_ring` matches the padded identity and every
//!      CCS matrix evaluation (CE-relation closure),
//!    - `claim.ct[j] == constant_term(claim.y_ring[j])` (the SuperNeo
//!      scalar view of the same ring evaluations).
//!
//! ## What this is and isn't
//!
//! `verify_uncompressed` **executes the SuperNeo verifier equations
//! directly over the folded CCS/CE circuit relation.** Rust is the
//! executor; SuperNeo is the source of soundness. A consumer that
//! runs this function gets either HyperNova's running/latest checks or the
//! Nebula terminal-fold check, plus relation closure against opened witnesses.
//!
//! It is NOT the soundness contract for a consumer that verifies a
//! *compressed* artifact (the decider R1CS + a SNARK over it). For
//! that consumer, the CE-relation obligation has to live as
//! constraint rows in the decider R1CS — that's a separate parallel
//! obligation tracked by `paper::decider_ce_relation` (reference
//! gadget). The check in step 5 of THIS verifier does not substitute
//! for the in-circuit version.
//!
//! The load-bearing soundness step in §1–4 is the Π_CCS sumcheck inside
//! step 2: at random row `α` it implies CCS satisfaction for the latest
//! and correct CE evaluation for `pre_final_running` at its
//! (prover-supplied but circuit-bound) `r`. For a certified fixed-relation
//! chain, recursive F' satisfaction supplies the cross-step induction. For
//! every other multi-chunk relation this verifier fails closed instead of
//! relying on a digest-only story.
//! `pc` is pinned/linked as a state field and absorbed directly into the
//! per-step `state_x_out` preimage. In this single-`F'_j` build it is
//! always `TRIVIAL_PC`, but the binding remains explicit for the
//! HyperNova recursive-link shape.
//! Step 5's CE-relation check is what binds those transcript-derived
//! `y_j` values back to the *opened* witness `Z`.

use neo_ccs::utils::tensor_point_parallel;
use neo_ccs::{check_ccs_rowwise_zero, traits::SModuleHomomorphism};
use neo_math::balanced::within_nc_bound;
use neo_math::{superneo_bar_block, Fq, KExtensions, Rq, D, F, K};
use neo_reductions::common::{
    decode_superneo_coeffs_from_witness_mat, project_x_from_witness_mat, validate_superneo_witness_mat,
};
use neo_reductions::superneo_eval::{
    eval_ring_linear_forms_real_z_blocks, SuperneoEvalCache, SuperneoRingLinearForm, SuperneoZBlocks,
};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
#[cfg(feature = "perf-timers")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::lifecycle::{Error, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{
    self, FinalFoldProof, LaneCommitmentMode, LatestInstance, ProofState, RunningInstance, State, TerminalFoldInputs,
};
use crate::paper::decider;
use crate::paper::digest::{
    digest_fields_as_digest32, initial_boundary_digest, public_trace_seed_digest, AccumulatorHandle,
};
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    f_prime_public_input_link_matches, FPrimePublicInputLayout, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use crate::paper::relations::{CeClaim, WitnessMat};
use neo_ajtai::Commitment;

use super::final_openings::{check_claim_openings, validate_opening_shape, FinalWitnessOpeningBackend};

#[cfg(feature = "perf-timers")]
struct WitnessAuthorityPerf {
    forms_ns: AtomicU64,
    commit_ns: AtomicU64,
    project_ns: AtomicU64,
    norm_ns: AtomicU64,
    ce_ns: AtomicU64,
}

#[cfg(feature = "perf-timers")]
impl WitnessAuthorityPerf {
    fn new() -> Self {
        Self {
            forms_ns: AtomicU64::new(0),
            commit_ns: AtomicU64::new(0),
            project_ns: AtomicU64::new(0),
            norm_ns: AtomicU64::new(0),
            ce_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    fn add_forms(&self, elapsed: Duration) {
        self.forms_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    fn add_commit(&self, elapsed: Duration) {
        self.commit_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    fn add_project(&self, elapsed: Duration) {
        self.project_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    fn add_norm(&self, elapsed: Duration) {
        self.norm_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    fn add_ce(&self, elapsed: Duration) {
        self.ce_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    fn print(&self, mode: &str, claims: usize) {
        let ns_to_s = |ns: u64| ns as f64 / 1_000_000_000.0;
        eprintln!(
            "[verify] witness authority ({mode}, claims={claims}): forms {:>6.2}s commit {:>6.2}s project {:>6.2}s norm {:>6.2}s ce {:>6.2}s",
            ns_to_s(self.forms_ns.load(Ordering::Relaxed)),
            ns_to_s(self.commit_ns.load(Ordering::Relaxed)),
            ns_to_s(self.project_ns.load(Ordering::Relaxed)),
            ns_to_s(self.norm_ns.load(Ordering::Relaxed)),
            ns_to_s(self.ce_ns.load(Ordering::Relaxed)),
        );
    }
}

#[cfg(not(feature = "perf-timers"))]
struct WitnessAuthorityPerf;

#[cfg(not(feature = "perf-timers"))]
impl WitnessAuthorityPerf {
    #[inline]
    fn new() -> Self {
        Self
    }

    #[inline]
    fn add_forms(&self, _elapsed: Duration) {}

    #[inline]
    fn add_commit(&self, _elapsed: Duration) {}

    #[inline]
    fn add_project(&self, _elapsed: Duration) {}

    #[inline]
    fn add_norm(&self, _elapsed: Duration) {}

    #[inline]
    fn add_ce(&self, _elapsed: Duration) {}

    #[inline]
    fn print(&self, _mode: &str, _claims: usize) {}
}

/// Verify an uncompressed proof in O(1) verifier work (constant in chain length).
///
/// Authority comes from the certified F' induction: check the running CE
/// accumulator and latest CCS instance separately. Nebula additionally
/// re-runs its terminal NIFS fold to close the delayed memory lane.
pub fn verify_uncompressed(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    verify_uncompressed_inner(prep, proof, None)
}

/// Verify with a trusted caller-owned backend for final witness openings.
/// This profiling hook expands the verifier TCB; untrusted callers use [`verify_uncompressed`].
#[doc(hidden)]
pub fn verify_uncompressed_with_opening_backend(
    prep: &Preprocessing,
    proof: &Uncompressed,
    backend: &mut dyn FinalWitnessOpeningBackend,
) -> Result<(), Error> {
    verify_uncompressed_inner(prep, proof, Some(backend))
}

fn verify_uncompressed_inner(
    prep: &Preprocessing,
    proof: &Uncompressed,
    opening_backend: Option<&mut dyn FinalWitnessOpeningBackend>,
) -> Result<(), Error> {
    prep.validate_verifier_key_binding()?;
    if matches!(&proof.state.proof, ProofState::Initial) {
        return verify_base_case(prep, proof);
    }

    let (recorded_running, recorded_latest) = require_active_state(&proof.state.proof)?;
    let hypernova_terminal =
        prep.enforces_terminal_induction() && prep.nebula().is_none() && proof.final_fold.is_none();
    if !hypernova_terminal && !recorded_latest.instances.is_empty() {
        return Err(Error::NotFinalized);
    }
    if hypernova_terminal && recorded_latest.instances.is_empty() {
        return Err(Error::NotFinalized);
    }
    check_running_shape(&recorded_running)?;
    if let Some(final_fold) = &proof.final_fold {
        check_terminal_fold_claim_shapes(prep, &final_fold.terminal_inputs)?;
    }

    // (0a) Initial-semantic-state anchor. The decider preflight catches
    // a tampered `statement.public.initial_semantic_state_digest` via the
    // anchor cross-check in `validate_witness`, but `verify_uncompressed`
    // takes a raw `Uncompressed` proof — `state.initial_semantic_state_digest`
    // is prover-supplied and never cross-checked elsewhere in this verifier
    // path. `vk_fs_digest` absorbs the verifier-owned anchor, so a chain
    // whose actual initial differs from `prep.initial_semantic_state_digest()`
    // also breaks `XOutMismatch`; this dedicated check exists so the prover
    // sees the precise invariant they violated.
    check_initial_semantic_anchor(prep, proof)?;

    // (0b) Compact-state anchors. These coordinates are intentionally
    // omitted from the hot `state_x_out` preimage, so the non-replay
    // verifier pins them directly to the verifier-owned lifecycle shape.
    check_compact_state_anchors(prep, proof)?;

    // (0c) Stateless semantic invariant — checked next so a tampered
    // `semantic_state_digest` produces a precise
    // `StatelessSemanticInvariantViolated` rather than an opaque
    // `XOutMismatch` from the terminal-fold re-run. For stateless plans
    // the F' image's CCS structure has no Poseidon2 binding rows for
    // the `semantic_state_digest` lane, so a malicious prover could
    // otherwise self-consistently inject arbitrary bytes there.
    // Stateful plans skip this — terminal Π_CCS sumcheck authenticates
    // the field inductively via the binding rows.
    check_stateless_semantic_invariant(prep, proof)?;

    // (0d) Only an authoritative fixed relation may carry the HyperNova
    // induction after the intermediate messages are dropped. Public-link-only
    // image relations remain fail-closed for multi-chunk proofs.
    check_f_prime_non_replay_scope(prep, proof)?;

    // (0e) An externally accepted Nebula proof must end at a closed segment.
    // A trailing open segment has
    // folded op rows whose product equation and D_seen == D_pre binding
    // were never checked. Mid-segment State is prover resume material
    // only.
    check_nebula_terminal_state(prep, &proof.state)?;

    // HyperNova verifies the running accumulator and newest F' instance as
    // two relations. Legacy and Nebula proofs retain the older terminal-fold
    // finalization shape.
    if hypernova_terminal {
        verify_hypernova_terminal_case(prep, proof, recorded_latest)?;
    } else {
        match &proof.final_fold {
            None => verify_no_terminal_fold_case(prep, proof, &recorded_running)?,
            Some(final_fold) => verify_terminal_fold_case(prep, proof, &recorded_running, final_fold)?,
        }
    }

    // (5) Witness-side authority: each prover-stored witness must
    // satisfy ALL five terminal CE obligations against its claim —
    // commit / X / low-norm / `y_ring == M_j · Z(r)` / `ct ==
    // constant-term(y_ring)`. These are the SuperNeo verifier
    // equations on the folded CE relation; this Rust function
    // executes them directly. See module docs for the layering
    // boundary with the decider R1CS path.
    check_running_witnesses_authority(prep, &recorded_running, opening_backend)?;

    // (4) acc_digest is recomputed from the just-authenticated claims.
    check_recorded_acc_digest(prep, &recorded_running, &proof.state.acc_digest)?;
    Ok(())
}

/// Verify HyperNova's exact zero-step state without inventing a running or
/// latest instance. Every accepted coordinate is verifier-derived.
fn verify_base_case(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if proof.final_fold.is_some() {
        return Err(construction2::Error::UnexpectedFinalFoldProof.into());
    }

    construction2::state_base_case_check(&proof.state)?;

    let expected_z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);
    if proof.state.z_0 != expected_z_0 {
        return Err(Error::PostStateMismatch);
    }

    construction2::enforce_pc_in_range(&proof.state)?;

    check_initial_semantic_anchor(prep, proof)?;
    if proof.state.semantic_state_digest != prep.initial_semantic_state_digest() {
        return Err(Error::PostStateMismatch);
    }

    if proof.state.acc_digest != AccumulatorHandle::empty().digest() {
        return Err(Error::AccDigestMismatch);
    }

    if proof.state.public_trace != public_trace_seed_digest(prep.structure_digest()) {
        return Err(Error::PostStateMismatch);
    }

    match (prep.nebula(), &proof.state.nebula) {
        (None, None) => {}
        (Some(config), Some(lane)) if lane == &construction2::NebulaLane::base(config) => {}
        (Some(_), None) | (None, Some(_)) => return Err(Error::NebulaLanePresenceMismatch),
        (Some(_), Some(_)) => return Err(Error::PostStateMismatch),
    }

    Ok(())
}

fn check_initial_semantic_anchor(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if proof.state.initial_semantic_state_digest != prep.initial_semantic_state_digest() {
        return Err(Error::InitialSemanticStateAnchorMismatch);
    }
    Ok(())
}

fn check_compact_state_anchors(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    let expected_z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);
    if proof.state.z_0 != expected_z_0 {
        return Err(Error::PostStateMismatch);
    }
    if proof.state.pc != construction2::TRIVIAL_PC {
        return Err(Error::PostStateMismatch);
    }
    if proof.state.chunk_count > 0 && proof.state.public_trace != proof.state.z_i {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

fn check_stateless_semantic_invariant(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if !matches!(
        prep.semantic_state_mode,
        crate::paper::construction2::SemanticStateMode::Stateless
    ) {
        return Ok(());
    }
    let expected = match &proof.final_fold {
        None => {
            // No terminal fold ran (Initial or empty-latest path). The
            // current acc_digest IS the pre-terminal acc_digest.
            proof.state.acc_digest
        }
        Some(final_fold) => pre_fold_acc_digest(&final_fold.terminal_inputs.pre_final_running, prep.structure())?,
    };
    if proof.state.semantic_state_digest != expected {
        return Err(Error::StatelessSemanticInvariantViolated);
    }
    Ok(())
}

fn check_f_prime_non_replay_scope(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if prep.enforces_f_prime_recursive_link() && !prep.enforces_terminal_induction() && proof.state.chunk_count > 1 {
        return Err(Error::FPrimeNonReplayUnsupported {
            chunk_count: proof.state.chunk_count,
        });
    }
    Ok(())
}

// ── State-shape gates ─────────────────────────────────────────────────────

fn require_active_state(state: &ProofState) -> Result<(RunningInstance, &LatestInstance), Error> {
    match state {
        ProofState::Initial => Err(Error::NotFinalized),
        ProofState::Active { running, latest } => Ok((running.clone(), latest)),
    }
}

fn check_running_shape(running: &RunningInstance) -> Result<(), Error> {
    if !running.shape_ok() {
        return Err(Error::FinalAccumulatorWitnessShapeMismatch);
    }
    Ok(())
}

fn check_terminal_fold_claim_shapes(
    prep: &Preprocessing,
    inputs: &construction2::TerminalFoldInputs,
) -> Result<(), Error> {
    for (index, claim) in inputs.pre_final_running.claims.iter().enumerate() {
        check_claim_commitment_shape(prep, index, claim)?;
        check_claim_public_input_len(prep, claim)?;
        if claim.m_in > prep.structure().m
            || claim.m_in % neo_math::D != 0
            || claim.X.rows() != neo_math::D
            || claim.X.cols() != crate::paper::relations::superneo_public_x_cols(claim.m_in)
        {
            return Err(Error::FinalAccumulatorPublicInputMismatch { index });
        }
    }

    let (expected_d, _) = prep.log.dims();
    let expected_kappa = prep.params.kappa() as usize;
    for (index, instance) in inputs.latest.instances.iter().enumerate() {
        let claim = &instance.claim;
        let fail = |reason: String| Error::TerminalLatestAuthority { index, reason };
        if claim.m_in > prep.structure().m || claim.m_in % neo_math::D != 0 || claim.x.len() != claim.m_in {
            return Err(fail(format!(
                "public input shape is m_in={}, x.len()={}, structure.m={}",
                claim.m_in,
                claim.x.len(),
                prep.structure().m
            )));
        }
        if let Some(expected) = prep.public_input_len {
            if claim.m_in != expected {
                return Err(fail(format!(
                    "public input length is {}, expected {expected}",
                    claim.m_in
                )));
            }
        }
        if claim.c.d != expected_d
            || claim.c.kappa != expected_kappa
            || claim.c.data.len() != expected_d * expected_kappa
        {
            return Err(fail("commitment shape does not match verifier preprocessing".into()));
        }
    }
    Ok(())
}

// ── HyperNova terminal path: running accumulator + latest fresh F' ───────

fn verify_hypernova_terminal_case(
    prep: &Preprocessing,
    proof: &Uncompressed,
    latest: &LatestInstance,
) -> Result<(), Error> {
    let latest_count = latest.instances.len();
    if latest_count != 1 {
        return Err(Error::TerminalInductionArity { got: latest_count });
    }
    if proof.state.chunk_count == 0 || proof.state.step_count == 0 {
        return Err(Error::PostStateMismatch);
    }

    let start_index = proof
        .state
        .step_count
        .checked_sub(latest_count as u64)
        .ok_or(Error::PostStateMismatch)?;
    let expected_chunk_digest = construction2::f_prime_chunk_public_digest_from_claims(start_index, &latest.claims());
    let expected_boundary = digest_fields_as_digest32(expected_chunk_digest);
    if proof.state.z_i != expected_boundary || proof.state.public_trace != expected_boundary {
        return Err(Error::PostStateMismatch);
    }

    check_terminal_latest_link(prep, &proof.state, latest)?;
    check_latest_instances_authority(prep, latest)
}

fn check_latest_instances_authority(prep: &Preprocessing, latest: &LatestInstance) -> Result<(), Error> {
    let expected_m_in = prep
        .public_input_len
        .unwrap_or(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    for (index, instance) in latest.instances.iter().enumerate() {
        let fail = |reason: String| Error::TerminalLatestAuthority { index, reason };
        let claim = &instance.claim;
        let witness = &instance.witness;
        if claim.m_in != expected_m_in || claim.x.len() != expected_m_in {
            return Err(fail(format!(
                "public input shape is m_in={}, x.len()={}, expected {expected_m_in}",
                claim.m_in,
                claim.x.len()
            )));
        }
        if claim.adv.is_some() {
            return Err(fail(
                "plain F' latest claim carries an unsupported product-commitment sidecar".into(),
            ));
        }
        let private = witness
            .private_values(claim.x.len(), prep.structure().m)
            .ok_or_else(|| {
                fail(format!(
                    "packed/private witness does not complete public length {} to expected width {}",
                    claim.x.len(),
                    prep.structure().m
                ))
            })?;
        if claim.x.len() + private.len() != prep.structure().m {
            return Err(fail(format!(
                "x||w has length {}, expected {}",
                claim.x.len() + private.len(),
                prep.structure().m
            )));
        }
        validate_superneo_witness_mat(&witness.Z, prep.structure().m)
            .map_err(|error| fail(format!("packed witness shape: {error}")))?;

        let mut z = claim.x.clone();
        z.extend_from_slice(&private);
        let decoded = decode_superneo_coeffs_from_witness_mat(&witness.Z, prep.structure().m)
            .map_err(|error| fail(format!("packed witness decoding: {error}")))?;
        for (column, value) in z.iter().enumerate() {
            if decoded[column] != K::from(*value) {
                return Err(fail(format!("packed witness disagrees with x||w at column {column}")));
            }
        }
        if decoded[z.len()..].iter().any(|value| *value != K::ZERO) {
            return Err(fail("packed witness has a nonzero padded tail".into()));
        }
        for row in 0..witness.Z.rows() {
            for col in 0..witness.Z.cols() {
                if !within_nc_bound(witness.Z[(row, col)], prep.params.b()) {
                    return Err(fail(format!(
                        "low-norm bound violated at packed row {row}, column {col}"
                    )));
                }
            }
        }
        if prep.log.commit(&witness.Z) != claim.c {
            return Err(fail("Ajtai commitment does not open to the supplied witness".into()));
        }
        check_ccs_rowwise_zero(prep.structure(), &claim.x, &private)
            .map_err(|error| fail(format!("CCS relation: {error}")))?;
    }
    Ok(())
}

// ── Terminal-fold path: re-run NIFS.V and bind the result ─────────────────

fn verify_terminal_fold_case(
    prep: &Preprocessing,
    proof: &Uncompressed,
    recorded_running: &RunningInstance,
    final_fold: &FinalFoldProof,
) -> Result<(), Error> {
    check_terminal_boundary_from_latest(prep, &proof.state, final_fold)?;
    let pre_state = build_pre_final_state(prep, &proof.state, &final_fold.terminal_inputs)?;
    check_terminal_latest_link(prep, &pre_state, &final_fold.terminal_inputs.latest)?;
    let derived_state = construction2::verify_final_fold(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        prep.enforces_terminal_induction()
            .then(|| prep.nebula())
            .flatten(),
        pre_state,
        Some(final_fold),
        prep.semantic_state_mode,
    )?;
    bind_derived_state_to_recorded(&derived_state, &proof.state)?;
    // Cross-check the derived running matches the recorded one. The
    // step above already binds chain coordinates + x_out + acc_digest;
    // this asserts the same on the claim-level data (commitments, X,
    // r, y_j).
    let derived_running = derived_state
        .proof
        .running_for_binding()
        .map_err(|_| Error::PostStateMismatch)?;
    if derived_running.claims != recorded_running.claims
        || derived_running.parent_authority != recorded_running.parent_authority
    {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

fn check_terminal_boundary_from_latest(
    prep: &Preprocessing,
    post: &State,
    final_fold: &FinalFoldProof,
) -> Result<(), Error> {
    if !terminal_fold_inputs_are_public(&final_fold.terminal_inputs) {
        return Err(Error::PostStateMismatch);
    }
    let latest_count = final_fold.terminal_inputs.latest.instances.len() as u64;
    if latest_count == 0 || latest_count > post.step_count {
        return Err(Error::PostStateMismatch);
    }
    if prep.enforces_terminal_induction() && latest_count != 1 {
        return Err(Error::TerminalInductionArity {
            got: latest_count as usize,
        });
    }
    let max_fresh = prep.params.max_fresh_count();
    if latest_count as usize > max_fresh {
        return Err(Error::BatchTooLarge {
            got: latest_count as usize,
            max: max_fresh,
        });
    }
    if !prep.enforces_terminal_induction() {
        let pre_running = &final_fold.terminal_inputs.pre_final_running;
        let fresh_m_in = final_fold
            .terminal_inputs
            .latest
            .instances
            .first()
            .ok_or(Error::PostStateMismatch)?
            .claim
            .m_in;
        let default_running = RunningInstance::canonical_zero(
            &prep.params,
            prep.structure(),
            fresh_m_in,
            LaneCommitmentMode::from_nebula(prep.nebula().is_some()),
        )
        .map_err(construction2::Error::from)?
        .claims_only();
        if pre_running.claims != default_running.claims
            || pre_running.parent_authority != default_running.parent_authority
        {
            return Err(Error::TerminalOnlyMultiChunkUnsupported {
                chunk_count: post.chunk_count,
            });
        }
    }
    if post.chunk_count == 0 {
        return Err(Error::PostStateMismatch);
    }
    if !prep.enforces_terminal_induction() && (post.chunk_count != 1 || post.step_count != latest_count) {
        return Err(Error::PostStateMismatch);
    }
    let start_index = post.step_count - latest_count;
    let latest_claims = final_fold.terminal_inputs.latest.claims();
    let expected_chunk_digest = construction2::f_prime_chunk_public_digest_from_claims(start_index, &latest_claims);
    let expected_boundary = digest_fields_as_digest32(expected_chunk_digest);
    if post.z_i != expected_boundary || post.public_trace != expected_boundary {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

/// A terminal-fold snapshot is a public statement. It must not carry any
/// prover-only witness matrix, even if the verifier can otherwise ignore it.
fn terminal_fold_inputs_are_public(inputs: &construction2::TerminalFoldInputs) -> bool {
    inputs.pre_final_running.witnesses.is_empty()
        && inputs.latest.instances.iter().all(|instance| {
            instance.witness.w.is_empty() && instance.witness.Z.rows() == 0 && instance.witness.Z.cols() == 0
        })
}

fn check_terminal_latest_link(prep: &Preprocessing, pre_state: &State, latest: &LatestInstance) -> Result<(), Error> {
    if !prep.enforces_f_prime_recursive_link() {
        return Ok(());
    }

    check_required_f_prime_latest_link(prep, pre_state, latest)
}

fn check_required_f_prime_latest_link(
    prep: &Preprocessing,
    pre_state: &State,
    latest: &LatestInstance,
) -> Result<(), Error> {
    let expected = construction2::compute_x_out(
        &prep.vk,
        &prep.params,
        prep.structure_digest(),
        pre_state,
        prep.semantic_state_mode,
    );
    let layout = match prep.nebula() {
        None => FPrimePublicInputLayout::plain(),
        Some(config) => FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(config.stacks)),
    };
    let expected_public_input_len = prep.public_input_len.unwrap_or(layout.total_len());
    for (index, instance) in latest.instances.iter().enumerate() {
        let claim = &instance.claim;
        if !f_prime_public_input_link_matches(layout, &expected, expected_public_input_len, claim.m_in, &claim.x) {
            return Err(Error::TerminalLatestPublicInputMismatch { index });
        }
    }
    Ok(())
}

fn verify_no_terminal_fold_case(
    _prep: &Preprocessing,
    _proof: &Uncompressed,
    _recorded_running: &RunningInstance,
) -> Result<(), Error> {
    // `verify_uncompressed` is the external verifier for a finalized proof.
    // The exact `ProofState::Initial` base is handled before active-state
    // materialization. For an Active state, no terminal NIFS proof means
    // there is no verifier-driven fold tying the recorded terminal state to
    // a real pre-final `(running, latest)`. Even an empty recorded running
    // accumulator is not authority: it can be forged by directly
    // constructing `Active { running: empty, latest: empty }`.
    Err(Error::MissingTerminalFoldProof)
}

/// Construct the pre-final-fold `State` from chain coords (which are
/// unchanged by finalization) + the snapshotted pre-fold inputs.
fn build_pre_final_state(prep: &Preprocessing, post: &State, terminal: &TerminalFoldInputs) -> Result<State, Error> {
    let pre_acc_digest = pre_fold_acc_digest(&terminal.pre_final_running, prep.structure())?;
    Ok(State {
        chunk_count: post.chunk_count,
        step_count: post.step_count,
        z_0: post.z_0,
        z_i: post.z_i,
        pc: post.pc,
        initial_semantic_state_digest: post.initial_semantic_state_digest,
        semantic_state_digest: post.semantic_state_digest,
        acc_digest: pre_acc_digest,
        public_trace: post.public_trace,
        proof: ProofState::active(terminal.pre_final_running.clone(), terminal.latest.clone()),
        nebula: terminal.pre_nebula.clone(),
    })
}

/// `acc_digest` of a pre-finalization running. Mirrors the formula in
/// [`construction2::prove_final_fold`] / [`construction2::verify_final_fold`].
fn pre_fold_acc_digest(
    pre_running: &RunningInstance,
    structure: &crate::paper::relations::Structure,
) -> Result<[u8; 32], Error> {
    if !pre_running.claims.is_empty() && pre_running.parent_authority.is_none() {
        Err(Error::PreFinalAccumulatorMissingParentAuthority)
    } else {
        pre_running
            .accumulator_digest(structure)
            .map_err(|_| Error::AccDigestMismatch)
    }
}

fn bind_derived_state_to_recorded(derived: &State, recorded: &State) -> Result<(), Error> {
    if derived.chunk_count != recorded.chunk_count
        || derived.step_count != recorded.step_count
        || derived.z_0 != recorded.z_0
        || derived.z_i != recorded.z_i
        || derived.pc != recorded.pc
        || derived.initial_semantic_state_digest != recorded.initial_semantic_state_digest
        || derived.semantic_state_digest != recorded.semantic_state_digest
        || derived.public_trace != recorded.public_trace
        || derived.acc_digest != recorded.acc_digest
        || derived.nebula != recorded.nebula
    {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

// ── Witness-side authority ────────────────────────────────────────────────

/// Step (5) of [`verify_uncompressed`]: every running witness must
/// satisfy the **SuperNeo terminal CE relation** against its claim.
///
/// Paper-level CE relation (SuperNeo Theorem 5, §5):
///
/// 1. `commit_Ajtai(Z) == claim.c`
/// 2. `claim.m_in` matches the verifier-owned `public_input_len`, then
///    `project_x(Z) == claim.X`
/// 3. every entry of `Z` is low-norm: `|z| < b`
/// 4. `claim.y_ring[0]` is the padded identity evaluation, and
///    `claim.y_ring[j + 1] == multilinear_eval(M_j · Z, claim.r)` for
///    every CCS matrix `M_j`
///
/// Implementation-consistency obligation (the SuperNeo paper's
/// `ct(y_j) = M̄_j z(r)` identity, made checkable from cached state):
///
/// 5. `claim.ct[j] == constant_term(claim.y_ring[j])` — the lane-0
///    K-element of `y_ring[j]`. `ct` is the scalar/constant-term view
///    of `y_ring`; if `y_ring` matches `M_j · Z(r)` and `ct` is the
///    constant term of `y_ring`, then `ct == M_j z(r)` transitively.
///
/// (4) and (5) close the selected CE relation against the opened witness.
/// Without them, the F'-chain `acc_digest` compact handle would bind
/// only the recorded CE claims, not prove that the opened terminal `Z`
/// realizes their `y_ring`/`ct` values. The Rust code below faithfully
/// executes the SuperNeo verifier equations; it does not invent a new
/// check.
///
/// **Layering note.** This makes `verify_uncompressed` sound for any
/// consumer that runs it. It does NOT substitute for the parallel
/// obligation in the decider R1CS; that lives
/// in `paper::decider_ce_relation` (reference gadget for the
/// in-circuit version).
///
/// Exposed `pub` via [`validate_final_witness_authority`] so isolation
/// tests can exercise the authority obligations against a hand-crafted
/// `(claim, witness)` pair without driving the full
/// `verify_uncompressed` binding pipeline.
fn check_running_witnesses_authority(
    prep: &Preprocessing,
    running: &RunningInstance,
    mut opening_backend: Option<&mut dyn FinalWitnessOpeningBackend>,
) -> Result<(), Error> {
    check_running_shape(running)?;

    // Commitment dimensions are proof-controlled. Validate them before the
    // zero-witness fast path can allocate a zero commitment of that shape.
    for (index, claim) in running.claims.iter().enumerate() {
        check_claim_commitment_shape(prep, index, claim)?;
        check_claim_public_input_len(prep, claim)?;
    }

    let b = prep.params.b();
    let ell_d = ell_d_for_ce_check();
    let expected_r_len = expected_row_point_len(prep);
    let perf = WitnessAuthorityPerf::new();
    let t_norm = std::time::Instant::now();
    let witness_nonzero = scan_terminal_witnesses(prep, running, b)?;
    perf.add_norm(t_norm.elapsed());
    check_terminal_r_shapes(running, expected_r_len)?;
    let superneo_cache = prep.optimized_cache().superneo();
    let opened_commitments = commit_running_witnesses(prep, running, &witness_nonzero, &perf);

    if running.claims.len() > 1 && rayon::current_thread_index().is_none() {
        let first_r = running
            .claims
            .first()
            .expect("check_running_shape accepted a non-empty running instance")
            .r
            .as_slice();
        if running
            .claims
            .iter()
            .all(|claim| claim.r.as_slice() == first_r)
        {
            if let Some(backend) = opening_backend.as_deref_mut() {
                let t_forms = std::time::Instant::now();
                if let Some(openings) = backend
                    .final_witness_openings(prep.optimized_cache(), &running.witnesses, first_r, prep.structure().m)
                    .map_err(|reason| Error::FinalAccumulatorOpeningBackend { reason })?
                {
                    let expected_matrix_count = prep.structure().t() + 1;
                    validate_opening_shape(&openings, running.claims.len(), expected_matrix_count)?;
                    perf.add_forms(t_forms.elapsed());
                    let results: Vec<Result<(), Error>> = running
                        .claims
                        .par_iter()
                        .zip(running.witnesses.par_iter())
                        .zip(opened_commitments.par_iter())
                        .zip(witness_nonzero.par_iter())
                        .zip(openings.par_iter())
                        .enumerate()
                        .map(|(index, ((((claim, witness), opened), &nonzero), rows))| {
                            check_running_claim_authority(
                                prep,
                                index,
                                claim,
                                witness,
                                opened,
                                nonzero,
                                expected_r_len,
                                ell_d,
                                None,
                                Some(rows),
                                expected_matrix_count,
                                &perf,
                            )
                        })
                        .collect();
                    for result in results {
                        result?;
                    }
                    perf.print("shared-r backend", running.claims.len());
                    return Ok(());
                }
                perf.add_forms(t_forms.elapsed());
            }
            let t_forms = std::time::Instant::now();
            let forms = build_ring_linear_forms_for_r(prep, superneo_cache, first_r);
            perf.add_forms(t_forms.elapsed());
            let results: Vec<Result<(), Error>> = running
                .claims
                .par_iter()
                .zip(running.witnesses.par_iter())
                .zip(opened_commitments.par_iter())
                .zip(witness_nonzero.par_iter())
                .enumerate()
                .map(|(index, (((claim, witness), opened), &nonzero))| {
                    check_running_claim_authority(
                        prep,
                        index,
                        claim,
                        witness,
                        opened,
                        nonzero,
                        expected_r_len,
                        ell_d,
                        Some(&forms),
                        None,
                        forms.len() + 1,
                        &perf,
                    )
                })
                .collect();
            for result in results {
                result?;
            }
            perf.print("shared-r parallel", running.claims.len());
            return Ok(());
        }
    }

    let mut cached_forms: Option<(Vec<K>, Vec<SuperneoRingLinearForm>)> = None;
    for (index, (((claim, witness), opened), &nonzero)) in running
        .claims
        .iter()
        .zip(&running.witnesses)
        .zip(opened_commitments.iter())
        .zip(witness_nonzero.iter())
        .enumerate()
    {
        let t_forms = std::time::Instant::now();
        let forms = ring_linear_forms_for_claim_r(prep, superneo_cache, &mut cached_forms, &claim.r);
        perf.add_forms(t_forms.elapsed());
        check_running_claim_authority(
            prep,
            index,
            claim,
            witness,
            opened,
            nonzero,
            expected_r_len,
            ell_d,
            Some(forms),
            None,
            forms.len() + 1,
            &perf,
        )?;
    }
    perf.print("sequential", running.claims.len());
    Ok(())
}

fn scan_terminal_witnesses(prep: &Preprocessing, running: &RunningInstance, b: u32) -> Result<Vec<bool>, Error> {
    running
        .witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            validate_superneo_witness_mat(witness, prep.structure().m)
                .map_err(|_| Error::FinalAccumulatorWitnessShapeMismatch)?;
            if witness
                .virtual_constant_value()
                .is_some_and(|value| *value == F::ZERO)
            {
                return Ok(false);
            }
            if b > 1 {
                if let Some(nonzero) = witness.packed_signed_unit_nonzero_count() {
                    return Ok(nonzero != 0);
                }
            }
            let mut nonzero = false;
            for row in 0..witness.rows() {
                for col in 0..witness.cols() {
                    let entry = witness[(row, col)];
                    if !within_nc_bound(entry, b) {
                        return Err(Error::FinalAccumulatorLowNormViolation { index, row, col });
                    }
                    nonzero |= entry != F::ZERO;
                }
            }
            Ok(nonzero)
        })
        .collect()
}

fn check_terminal_r_shapes(running: &RunningInstance, expected_r_len: usize) -> Result<(), Error> {
    for (index, claim) in running.claims.iter().enumerate() {
        if claim.r.len() != expected_r_len {
            return Err(Error::FinalAccumulatorEvaluationPointShapeMismatch {
                index,
                expected: expected_r_len,
                got: claim.r.len(),
            });
        }
    }
    Ok(())
}

fn commit_running_witnesses(
    prep: &Preprocessing,
    running: &RunningInstance,
    witness_nonzero: &[bool],
    perf: &WitnessAuthorityPerf,
) -> Vec<Commitment> {
    debug_assert_eq!(running.witnesses.len(), witness_nonzero.len());
    let t_commit = std::time::Instant::now();
    let witness_refs: Vec<&WitnessMat> = running
        .witnesses
        .iter()
        .zip(witness_nonzero.iter())
        .filter_map(|(witness, &nonzero)| nonzero.then_some(witness))
        .collect();
    let nonzero_opened = prep.log.commit_many(&witness_refs);
    let mut nonzero_iter = nonzero_opened.into_iter();
    let opened = running
        .claims
        .iter()
        .zip(witness_nonzero.iter())
        .map(|(claim, &nonzero)| {
            if nonzero {
                nonzero_iter
                    .next()
                    .expect("verify_uncompressed: nonzero witness commitment count mismatch")
            } else {
                Commitment::zeros(claim.c.d, claim.c.kappa)
            }
        })
        .collect();
    debug_assert!(
        nonzero_iter.next().is_none(),
        "verify_uncompressed: unused nonzero witness commitment"
    );
    perf.add_commit(t_commit.elapsed());
    opened
}

#[allow(clippy::too_many_arguments)]
fn check_running_claim_authority(
    prep: &Preprocessing,
    index: usize,
    claim: &CeClaim,
    witness: &WitnessMat,
    opened: &Commitment,
    witness_nonzero: bool,
    expected_r_len: usize,
    ell_d: usize,
    ring_linear_forms: Option<&[SuperneoRingLinearForm]>,
    precomputed_openings: Option<&[[K; D]]>,
    expected_matrix_count: usize,
    perf: &WitnessAuthorityPerf,
) -> Result<(), Error> {
    if opened != &claim.c {
        return Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index });
    }
    // Each published Nebula lane
    // commitment must open against its lane slice of the *same* witness
    // the full-`z` commitment just opened — the check that pins the
    // mirrored fold algebra to lane content.
    match (prep.nebula(), &claim.adv) {
        (Some(cfg), Some(adv)) => {
            let opens = cfg
                .scheme
                .open_matches(adv, witness)
                .map_err(|_| Error::NebulaSliceOpeningFailed)?;
            if !opens {
                return Err(Error::NebulaSliceOpeningFailed);
            }
        }
        (None, None) => {}
        _ => return Err(Error::NebulaAdvPresenceMismatch),
    }
    if !witness_nonzero {
        let t_project = std::time::Instant::now();
        let result = check_zero_public_projection(prep, index, claim);
        perf.add_project(t_project.elapsed());
        result?;
        if claim.r.len() != expected_r_len {
            return Err(Error::FinalAccumulatorEvaluationPointShapeMismatch {
                index,
                expected: expected_r_len,
                got: claim.r.len(),
            });
        }
        let t_ce = std::time::Instant::now();
        let result = check_zero_ce_relation(index, claim, expected_matrix_count, 1usize << ell_d);
        perf.add_ce(t_ce.elapsed());
        return result;
    }
    let t_project = std::time::Instant::now();
    let projected = project_x_from_witness_mat(witness, prep.structure().m, claim.m_in)
        .map_err(|_| Error::FinalAccumulatorPublicInputMismatch { index })?;
    perf.add_project(t_project.elapsed());
    if projected != claim.X {
        return Err(Error::FinalAccumulatorPublicInputMismatch { index });
    }
    if claim.r.len() != expected_r_len {
        return Err(Error::FinalAccumulatorEvaluationPointShapeMismatch {
            index,
            expected: expected_r_len,
            got: claim.r.len(),
        });
    }
    let t_ce = std::time::Instant::now();
    let result = match (ring_linear_forms, precomputed_openings) {
        (Some(forms), None) => check_ce_relation(prep, index, claim, witness, ell_d, forms),
        (None, Some(openings)) => check_claim_openings(index, claim, ell_d, openings),
        _ => Err(Error::FinalAccumulatorOpeningBackend {
            reason: "verifier received an invalid opening source".to_string(),
        }),
    };
    perf.add_ce(t_ce.elapsed());
    result
}

fn check_claim_commitment_shape(prep: &Preprocessing, index: usize, claim: &CeClaim) -> Result<(), Error> {
    let (expected_d, _) = prep.log.dims();
    let expected_kappa = prep.params.kappa() as usize;
    if claim.c.d != expected_d || claim.c.kappa != expected_kappa || claim.c.data.len() != expected_d * expected_kappa {
        return Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index });
    }
    Ok(())
}

fn check_claim_public_input_len(prep: &Preprocessing, claim: &CeClaim) -> Result<(), Error> {
    if let Some(expected) = prep.public_input_len {
        if claim.m_in != expected {
            return Err(Error::PublicInputLenMismatch {
                expected,
                got: claim.m_in,
            });
        }
    }
    Ok(())
}

/// Public entry that runs the witness-authority block
/// from [`check_running_witnesses_authority`] against a caller-provided
/// `RunningInstance`. Used by tests that want to isolate the CE-relation
/// obligation without first passing the chain-replay + binding steps
/// `verify_uncompressed` does up-front.
pub fn validate_final_witness_authority(prep: &Preprocessing, running: &RunningInstance) -> Result<(), Error> {
    check_running_witnesses_authority(prep, running, None)
}

/// Isolate the exact recursive terminal-link check used by
/// [`verify_uncompressed`] for conformance and mutation tests.
pub fn validate_terminal_latest_link(
    prep: &Preprocessing,
    state: &State,
    latest: &LatestInstance,
) -> Result<(), Error> {
    check_terminal_latest_link(prep, state, latest)
}

/// Check the F' latest-link equation without consulting the preprocessing
/// capability flag. Conformance tests use this entry to isolate the equation
/// from the full terminal wrapper.
pub fn validate_required_f_prime_latest_link(
    prep: &Preprocessing,
    state: &State,
    latest: &LatestInstance,
) -> Result<(), Error> {
    check_required_f_prime_latest_link(prep, state, latest)
}

/// Isolate the exact latest-CCS relation check used by
/// [`verify_uncompressed`] for conformance and mutation tests.
pub fn validate_latest_witness_authority(prep: &Preprocessing, latest: &LatestInstance) -> Result<(), Error> {
    check_latest_instances_authority(prep, latest)
}

/// `ell_d = log2(next_power_of_two(D))`, matching the prover's
/// `compute_y_from_Z_and_r` padding so the verifier's expected
/// `y_ring` lengths align with the proof's.
#[inline]
fn ell_d_for_ce_check() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn expected_row_point_len(prep: &Preprocessing) -> usize {
    prep.structure()
        .n
        .max(neo_reductions::common::superneo_carrier_width(prep.structure().m))
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize
}

fn ring_linear_forms_for_claim_r<'a>(
    prep: &Preprocessing,
    superneo_cache: &SuperneoEvalCache,
    cached_forms: &'a mut Option<(Vec<K>, Vec<SuperneoRingLinearForm>)>,
    r: &[K],
) -> &'a [SuperneoRingLinearForm] {
    let needs_rebuild = cached_forms
        .as_ref()
        .is_none_or(|(cached_r, _)| cached_r.as_slice() != r);
    if needs_rebuild {
        *cached_forms = Some((r.to_vec(), build_ring_linear_forms_for_r(prep, superneo_cache, r)));
    }
    cached_forms
        .as_ref()
        .expect("ring-linear forms must be cached")
        .1
        .as_slice()
}

fn build_ring_linear_forms_for_r(
    prep: &Preprocessing,
    superneo_cache: &SuperneoEvalCache,
    r: &[K],
) -> Vec<SuperneoRingLinearForm> {
    let rb = tensor_point_parallel::<K>(r);
    let n_eff = core::cmp::min(prep.structure().n, rb.len());
    superneo_cache.build_ring_linear_forms(&rb, n_eff)
}

/// Verify the identity-first selected CE relation against the opened witness.
fn check_ce_relation(
    prep: &Preprocessing,
    index: usize,
    claim: &CeClaim,
    witness: &WitnessMat,
    ell_d: usize,
    ring_linear_forms: &[SuperneoRingLinearForm],
) -> Result<(), Error> {
    let expected_count = ring_linear_forms.len() + 1;
    if claim.y_ring.len() != expected_count {
        return Err(Error::FinalAccumulatorCeRelationViolation {
            index,
            matrix_index: expected_count.min(claim.y_ring.len()),
        });
    }

    let d_pad = 1usize << ell_d;
    let identity = identity_ring_mle(witness, prep.structure().m, &claim.r);
    if !padded_ring_row_matches(&claim.y_ring[0], &identity, d_pad) {
        return Err(Error::FinalAccumulatorCeRelationViolation { index, matrix_index: 0 });
    }

    let z_blocks = SuperneoZBlocks::from_witness_mat(witness, prep.structure().m)
        .expect("check_ce_relation: witness shape was validated before CE closure");
    let evaluated = eval_ring_linear_forms_real_z_blocks(ring_linear_forms, &z_blocks);
    let mut expected_ct = Vec::with_capacity(expected_count);
    expected_ct.push(identity[0]);
    for (application, (coeffs, recorded)) in evaluated.iter().zip(&claim.y_ring[1..]).enumerate() {
        if !padded_ring_row_matches(recorded, coeffs, d_pad) {
            return Err(Error::FinalAccumulatorCeRelationViolation {
                index,
                matrix_index: application + 1,
            });
        }
        expected_ct.push(coeffs[0]);
    }

    if expected_ct.len() != claim.ct.len() {
        return Err(Error::FinalAccumulatorCtMismatch {
            index,
            matrix_index: expected_ct.len().min(claim.ct.len()),
        });
    }
    for (matrix_index, (expected, recorded)) in expected_ct.iter().zip(&claim.ct).enumerate() {
        if expected != recorded {
            return Err(Error::FinalAccumulatorCtMismatch { index, matrix_index });
        }
    }
    Ok(())
}

fn padded_ring_row_matches(recorded: &[K], expected: &[K; D], d_pad: usize) -> bool {
    recorded.len() == d_pad
        && recorded
            .iter()
            .take(D)
            .zip(expected)
            .all(|(left, right)| left == right)
        && recorded.iter().skip(D).all(|&value| value == K::ZERO)
}

fn identity_ring_mle(witness: &WitnessMat, expected_m: usize, point: &[K]) -> [K; D] {
    let assignment = decode_superneo_coeffs_from_witness_mat(witness, expected_m)
        .expect("identity_ring_mle: witness shape was validated before CE closure");
    let weights = tensor_point_parallel::<K>(point);
    let mut output = [K::ZERO; D];

    for (row, &weight) in weights.iter().take(assignment.len()).enumerate() {
        let block = row / D;
        let mut basis = [Fq::ZERO; D];
        basis[row % D] = Fq::ONE;
        let transformed = Rq(superneo_bar_block(basis));

        let mut real = [Fq::ZERO; D];
        let mut imaginary = [Fq::ZERO; D];
        for lane in 0..D {
            let [low, high] = assignment[block * D + lane].as_coeffs();
            real[lane] = low;
            imaginary[lane] = high;
        }

        let real_product = transformed.mul(&Rq(real));
        let imaginary_product = transformed.mul(&Rq(imaginary));
        for coefficient in 0..D {
            output[coefficient] +=
                weight * K::from_coeffs([real_product.0[coefficient], imaginary_product.0[coefficient]]);
        }
    }
    output
}
fn check_zero_public_projection(prep: &Preprocessing, index: usize, claim: &CeClaim) -> Result<(), Error> {
    if claim.m_in > prep.structure().m
        || claim.m_in % neo_math::D != 0
        || claim.X.rows() != neo_math::D
        || claim.X.cols() != crate::paper::relations::superneo_public_x_cols(claim.m_in)
    {
        return Err(Error::FinalAccumulatorPublicInputMismatch { index });
    }
    if claim.X.nnz() != 0 {
        return Err(Error::FinalAccumulatorPublicInputMismatch { index });
    }
    Ok(())
}

fn check_zero_ce_relation(
    index: usize,
    claim: &CeClaim,
    matrix_count: usize,
    expected_y_len: usize,
) -> Result<(), Error> {
    if claim.y_ring.len() != matrix_count {
        return Err(Error::FinalAccumulatorCeRelationViolation {
            index,
            matrix_index: matrix_count.min(claim.y_ring.len()),
        });
    }
    for (matrix_index, recorded) in claim.y_ring.iter().enumerate() {
        if recorded.len() != expected_y_len || recorded.iter().any(|&value| value != K::ZERO) {
            return Err(Error::FinalAccumulatorCeRelationViolation { index, matrix_index });
        }
    }

    if claim.ct.len() != matrix_count {
        return Err(Error::FinalAccumulatorCtMismatch {
            index,
            matrix_index: matrix_count.min(claim.ct.len()),
        });
    }
    for (matrix_index, &recorded) in claim.ct.iter().enumerate() {
        if recorded != K::ZERO {
            return Err(Error::FinalAccumulatorCtMismatch { index, matrix_index });
        }
    }
    Ok(())
}

// ── acc_digest consistency ────────────────────────────────────────────────

fn check_recorded_acc_digest(
    prep: &Preprocessing,
    running: &RunningInstance,
    recorded: &[u8; 32],
) -> Result<(), Error> {
    let recomputed = running
        .accumulator_digest(prep.structure())
        .map_err(|_| Error::AccDigestMismatch)?;
    if recomputed != *recorded {
        return Err(Error::AccDigestMismatch);
    }
    Ok(())
}

// ── Helper trait: avoids cloning the whole ProofState to read its claims ──

trait ProofStateBinding {
    fn running_for_binding(&self) -> Result<RunningInstance, ()>;
}

impl ProofStateBinding for ProofState {
    fn running_for_binding(&self) -> Result<RunningInstance, ()> {
        match self {
            ProofState::Initial => Err(()),
            ProofState::Active { running, latest } => {
                if !latest.instances.is_empty() {
                    return Err(());
                }
                Ok(running.claims_only())
            }
        }
    }
}

// ── Chain-replay / audit verifier ─────────────────────────────────────────

/// Diagnostic / chain-replay verifier — replays every step's NIFS.V, checks
/// the terminal fold, and discharges the final running accumulator's
/// SuperNeo CE witness-authority obligations.
///
/// **Not the production IVC verifier.** It does the same work as
/// `paper::decider::validate_witness`, walking `audit.steps` and
/// `audit.public_batches` step by step, so its cost is **linear in
/// chain length**. Production callers want the constant-cost
/// [`verify_uncompressed`] instead.
///
/// Reach for `verify_uncompressed_audit` only when you need to detect
/// tampers on the per-step audit trail (`steps`, `public_batches`,
/// `final_fold.nifs`) that an attacker might attempt while leaving the
/// final running accumulator self-consistent, or when you need the linear-cost
/// verifier for multi-chunk audit artifacts. Concretely that means red-team
/// tests for the audit trail, the Spartan compressed-decider preflight, and
/// chain-replay debugging.
pub fn verify_uncompressed_audit(prep: &Preprocessing, audit: &UncompressedAudit) -> Result<(), Error> {
    if let Some(final_fold) = &audit.proof.final_fold {
        check_terminal_fold_claim_shapes(prep, &final_fold.terminal_inputs)?;
    }
    let statement = super::build_decider_statement(prep, audit);
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
    )
    .map_err(Error::from)?;

    let ProofState::Active { running, latest } = &audit.proof.state.proof else {
        return Err(Error::PostStateMismatch);
    };
    if !latest.instances.is_empty() {
        return Err(Error::PostStateMismatch);
    }
    check_nebula_terminal_state(prep, &audit.proof.state)?;
    check_running_witnesses_authority(prep, running, None)
}

/// Enforce Nebula finalization and lane/config presence coherence.
fn check_nebula_terminal_state(prep: &Preprocessing, state: &State) -> Result<(), Error> {
    match (prep.nebula(), &state.nebula) {
        (Some(_), Some(lane)) => {
            if !lane.is_closed() {
                return Err(Error::NebulaSegmentOpenAtTerminal);
            }
            Ok(())
        }
        (None, None) => Ok(()),
        _ => Err(Error::NebulaLanePresenceMismatch),
    }
}
