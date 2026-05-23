//! Shared F' compiler state and helpers.
//!
//! This module owns app-agnostic compiler state and the protocol-generic
//! checks every concrete frontend (Fibonacci, R1CS, …) repeats: chain
//! coordinates, prior-fold authority, NIFS payload views, and
//! transcript-bound prior-fold verification. App frontends keep only
//! their app-specific witness checks, encoder dispatch, and output
//! shape.

use neo_ajtai::Commitment;
use neo_ccs::matrix::Mat;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::f_prime::image::{NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut};
use crate::frontends::f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_boundary_update_preimage_fields,
    build_public_trace_update_preimage_fields, build_state_x_out_preimage_fields_with_app_x, RecursiveStepImagePlan,
};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::{LatestInstance, ProofState, RunningInstance, State as PaperState};
use crate::paper::digest::{
    accumulator_digest_from_claims, digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest,
    initial_boundary_digest, public_trace_seed_digest,
};
use crate::paper::f_prime::native::f_prime_step_transcript;
use crate::paper::f_prime::poseidon_trace::{encode_poseidon_trace, PoseidonTraceImage};
use crate::paper::nifs::NifsProof;
use crate::paper::relations::{CcsClaim, CeClaim};

// ─────────────────────────────────────────────────────────────────────────
// Shared compiler state.
// ─────────────────────────────────────────────────────────────────────────

/// Per-chain compiler context shared by every F' app frontend.
///
/// The chain header (`vk_fs_digest`, `structure_digest`, `z_0`, `pc`,
/// `public_input_len`, commitment / boundary / limb shape) is constant
/// across steps; [`FPrimeChainState`] is updated each step;
/// [`FPrimeFoldForStep`] is the optional per-step fold authority the
/// caller writes between successive recursive compiles.
#[derive(Clone, Debug)]
pub struct FPrimeCompilerContext {
    // Chain header — constant across steps.
    pub vk_fs_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub z_0: [F; 4],
    pub pc: u64,
    pub public_input_len: usize,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub boundary_bits: usize,
    pub limbs: usize,
    // Threaded F' chain state — compiler updates each step.
    pub chain_state: FPrimeChainState,
    // Prior fold authority boundary — caller writes between steps.
    pub fold_for_step: Option<FPrimeFoldForStep>,
}

/// F'-level chain state threaded between successive compile calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FPrimeChainState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_i: [F; 4],
    pub acc_digest: [F; 4],
    pub public_trace: [F; 4],
}

/// Authority boundary of the prior fold the compiler embeds into this
/// step's encoded image.
///
/// All four components matter:
/// - `pre_running` and `latest` are the **inputs** to the prior fold,
/// - `proof` is the NIFS witness that authorises the transition,
/// - `post_running` is the **output**, and is what this F' step
///   actually commits to via its NIFS payload + accumulator hash.
///
/// The recursive plan is derived from `post_running.parent_authority`'s
/// real CE shape; the NIFS payload view is filled from the same
/// post-fold parent.
#[derive(Clone, Debug)]
pub struct FPrimeFoldForStep {
    pub pre_running: RunningInstance,
    pub latest: LatestInstance,
    pub proof: NifsProof,
    pub post_running: RunningInstance,
}

/// The five Poseidon traces every unified F' step emits.
pub struct UnifiedStepPoseidonTraces {
    pub boundary: PoseidonTraceImage,
    pub public_trace: PoseidonTraceImage,
    pub base_accumulator: PoseidonTraceImage,
    pub recursive_accumulator: PoseidonTraceImage,
    pub state_x_out: PoseidonTraceImage,
}

/// App-agnostic shell data assembled for one encoded F' step.
///
/// App frontends consume this to build their own encoder input. The
/// shell owns the lifecycle / state / Poseidon parts; the app frontend
/// still owns app-private bits, app-specific output, and encoder
/// dispatch.
pub struct UnifiedStepTraceAssembly {
    pub state_in: StateIn,
    pub state_out: StateOut,
    pub chunk_digest: [F; 4],
    pub public_output_digest: [F; 4],
    pub boundary_bits: Vec<F>,
    pub next_chain_state: FPrimeChainState,
    pub traces: UnifiedStepPoseidonTraces,
}

// ─────────────────────────────────────────────────────────────────────────
// Shell-level error.
// ─────────────────────────────────────────────────────────────────────────

/// Protocol-generic compiler errors raised by the shared shell. App
/// frontends wrap these via `#[from]` so their public errors stay
/// app-named while delegating shell semantics.
#[derive(Debug, Error)]
pub enum FPrimeShellCompilerError {
    #[error("F' shell compiler: preprocessing.public_input_len must be Some(..) for the compiler to derive z_0")]
    PreprocessingMissingPublicInputLen,

    #[error(
        "F' shell compiler: public_input_len = {got} doesn't match the canonical boundary shape (expected {expected})"
    )]
    UnsupportedPublicInputShape { got: usize, expected: usize },

    #[error("F' shell compiler: base step (chunk_count == 0) must not carry a prior fold")]
    BaseStepUnexpectedPriorFold,

    #[error("F' shell compiler: chunk must contain at least one assignment (got empty)")]
    EmptyChunk,

    #[error(
        "F' shell compiler: recursive step at chunk_count = {chunk_count} requires `ctx.fold_for_step = Some(..)`"
    )]
    PriorFoldMissingForRecursiveStep { chunk_count: u64 },

    #[error("F' shell compiler: post-fold running has no parent_authority; expected a non-empty post-fold running with Π_RLC parent")]
    PostRunningMissingParentAuthority,

    #[error("F' shell compiler: prior fold's NIFS proof failed to verify: {reason}")]
    PriorFoldVerificationFailed { reason: String },

    #[error("F' shell compiler: derived post-fold running does not match caller-supplied `fold.post_running` (claims or parent_authority differ)")]
    PriorFoldPostRunningMismatch,

    #[error("F' shell compiler: prep.plan must carry `accumulator = Some(..)`; got `None`")]
    CanonicalPlanMissingAccumulator,

    #[error("F' shell compiler: prep.plan.accumulator.unified must be true; got false (legacy plan)")]
    CanonicalPlanNotUnified,

    #[error("F' shell compiler: prep.plan.nifs_payload_shapes[acc.ce_claim_payload_index] must be `CeClaim`")]
    CanonicalPlanPayloadNotCeClaim,

    #[error("F' shell compiler: post-fold parent CE shape does not match canonical (canonical={canonical:?}, actual={actual:?})")]
    PostParentShapeMismatch {
        canonical: NifsCeClaimShape,
        actual: NifsCeClaimShape,
    },

    #[error("F' shell compiler: post-fold running.claims.len() = {actual} does not match canonical child_count = {canonical}")]
    PostRunningClaimsCountMismatch { canonical: u64, actual: u64 },
}

// ─────────────────────────────────────────────────────────────────────────
// Shared compiler entrypoints.
// ─────────────────────────────────────────────────────────────────────────

/// Initialise an [`FPrimeCompilerContext`] from `prep`.
///
/// Derives the chain header (vk_fs_digest, structure_digest, z_0,
/// public-trace seed, empty-accumulator digest) and seeds the chain
/// state to the base case. `pc` is the verifier-pinned program counter
/// for this chain; `limbs` is the app-private bit width (Fibonacci uses
/// 3; R1CS sizes from its own `plan.limbs`).
pub fn start_f_prime_chain_context(
    prep: &Preprocessing,
    pc: u64,
    limbs: usize,
) -> Result<FPrimeCompilerContext, FPrimeShellCompilerError> {
    let structure_digest = *prep.structure_digest();
    let public_input_len = prep
        .public_input_len
        .ok_or(FPrimeShellCompilerError::PreprocessingMissingPublicInputLen)?;

    // Boundary bits = 4 lanes × POSEIDON2_GOLDILOCKS_BITS = 256.
    let boundary_bits = 4 * POSEIDON2_GOLDILOCKS_BITS;
    if public_input_len != 1 + boundary_bits {
        return Err(FPrimeShellCompilerError::UnsupportedPublicInputShape {
            got: public_input_len,
            expected: 1 + boundary_bits,
        });
    }

    let z_0 = digest32_as_fields(initial_boundary_digest(&structure_digest, Some(public_input_len)));
    let public_trace = digest32_as_fields(public_trace_seed_digest(&structure_digest));
    let acc_digest = digest32_as_fields(accumulator_digest_from_claims(prep.params.b(), &[]));
    let vk_fs_digest = digest32_as_fields(prep.vk.digest());

    Ok(FPrimeCompilerContext {
        vk_fs_digest,
        structure_digest,
        z_0,
        pc,
        public_input_len,
        commitment_d: neo_math::D,
        commitment_kappa: prep.params.kappa() as usize,
        boundary_bits,
        limbs,
        chain_state: FPrimeChainState {
            chunk_count: 0,
            step_count: 0,
            z_i: z_0,
            acc_digest,
            public_trace,
        },
        fold_for_step: None,
    })
}

/// Re-derive `post_running` from `(pre_running, latest, proof)` via
/// `nifs::verify`. Reject on (a) NIFS rejection or (b) derived ≠
/// caller-supplied `post_running`.
///
/// The transcript is the per-step F' transcript
/// (`F_PRIME_STEP_TRANSCRIPT_LABEL` plus the six F'-step context
/// absorbs over `ctx.chain_state` + this step's `chunk_digest`), so it
/// matches what `paper::f_prime::native::prove` initialised for the
/// same fold. Callers must therefore source `fold.proof` from a
/// per-step `StepProof::Recursive` (e.g. `audit.steps[i].fold`) —
/// terminal-fold proofs from `finish_uncompressed_with_audit` use a
/// different transcript label and will be rejected here.
pub fn verify_prior_fold(
    prep: &Preprocessing,
    ctx: &FPrimeCompilerContext,
    fold: &FPrimeFoldForStep,
    rows_in_chunk: usize,
) -> Result<(), FPrimeShellCompilerError> {
    // Reconstruct the per-step F' transcript prefix. The proof variant
    // is irrelevant here — `f_prime_step_transcript` only reads digest
    // fields (z_0, z_i, …); `ProofState::Initial` is a sentinel.
    let state_in = PaperState {
        chunk_count: ctx.chain_state.chunk_count,
        step_count: ctx.chain_state.step_count,
        z_0: digest_fields_as_digest32(ctx.z_0),
        z_i: digest_fields_as_digest32(ctx.chain_state.z_i),
        pc: ctx.pc,
        acc_digest: digest_fields_as_digest32(ctx.chain_state.acc_digest),
        public_trace: digest_fields_as_digest32(ctx.chain_state.public_trace),
        proof: ProofState::Initial,
    };
    // `rows_in_chunk` is the size of the **current** batch being deposited
    // at this step (= `next_latest.len()` in native). The native fold
    // transcript used `f_prime_chunk_public_digest(state.step_count,
    // &next_latest)`, which absorbs K = `next_latest.len()` in the
    // preimage. Reconstructing K-aware shape digest replays the same
    // transcript bit-for-bit.
    let chunk_digest = chunk_digest_for_shape_count(
        ctx.chain_state.step_count,
        rows_in_chunk,
        ctx.commitment_d,
        ctx.commitment_kappa,
        ctx.public_input_len,
    );
    let mut tr = f_prime_step_transcript(&prep.vk, prep.structure_digest(), &state_in, chunk_digest);

    let derived = crate::paper::nifs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &fold.latest.claims(),
        &fold.pre_running,
        &fold.proof,
    )
    .map_err(|e| FPrimeShellCompilerError::PriorFoldVerificationFailed { reason: e.to_string() })?;

    // NIFS.V returns claims + parent_authority; witnesses are
    // prover-side only. Compare the authority-bearing pair.
    if derived.claims != fold.post_running.claims || derived.parent_authority != fold.post_running.parent_authority {
        return Err(FPrimeShellCompilerError::PriorFoldPostRunningMismatch);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// NIFS CE-claim view helpers.
// ─────────────────────────────────────────────────────────────────────────

/// Build a perp (inactive, deterministic-filler) NIFS CE claim view
/// matching `shape`.
///
/// **Soundness contract.** This view is *not* a real `CeClaim`. The
/// unified-mode F' structure binds the NIFS payload region bit-by-bit,
/// so the perp payload's bits must be deterministic — but the
/// payload's *authority* (its `c_data`, commitment, fold_digest, etc.)
/// must not enter the chain's accumulator. The selector achieves this:
/// when `is_base = 1` the structure forces
/// `state_out.new_acc_digest = base_trace.digest`, which depends only
/// on the constant preimage `(tag, 0)`. The recursive accumulator
/// trace's preimage still sources its `c_data` lanes from this
/// payload, but its digest is *discarded* by the selector.
///
/// Audit hooks: any future change that weakens the selector binding
/// (e.g. allowing both branches' digests to enter authority through a
/// different code path) must keep the perp payload's `c_data` out of
/// the new authority path, or this filler becomes a soundness bug.
pub fn perp_nifs_ce_view(shape: &NifsCeClaimShape) -> NifsCeClaimView {
    let y_ring: Vec<Vec<[F; 2]>> = shape
        .y_ring_inner_lens
        .iter()
        .map(|&len| vec![[F::ZERO; 2]; len])
        .collect();
    NifsCeClaimView {
        d: 0,
        kappa: 0,
        c_data: vec![F::ZERO; shape.c_data_entries],
        x_rows: shape.x_rows as u64,
        // The encoder fills `x_active_cols` × `x_rows` entries; the
        // legacy `x_cols` header is informational. We mirror
        // `x_active_cols` so the deterministic-filler view stays
        // internally consistent.
        x_cols: shape.x_active_cols as u64,
        x_active_cols: shape.x_active_cols as u64,
        x_active_flat: vec![F::ZERO; shape.x_rows * shape.x_active_cols],
        r: vec![[F::ZERO; 2]; shape.r_len],
        y_ring,
        y_zcol: vec![[F::ZERO; 2]; shape.y_zcol_len],
        s_col: vec![[F::ZERO; 2]; shape.s_col_len],
        m_in: 0,
        fold_digest_fields: [F::ZERO; 4],
    }
}

/// Build the full NIFS CE-claim view from the real post-fold parent
/// authority. Every field is derived honestly from `post_parent`; no
/// zero placeholders for authority-bearing data.
pub fn nifs_ce_view_from_claim(post_parent: &CeClaim, _public_input_len: usize) -> NifsCeClaimView {
    let active_cols = crate::paper::relations::superneo_public_x_cols(post_parent.m_in);
    let x_active_flat: Vec<F> = collect_active_x(&post_parent.X, active_cols);
    let r_pairs: Vec<[F; 2]> = post_parent.r.iter().map(k_to_pair).collect();
    let y_ring_pairs: Vec<Vec<[F; 2]>> = post_parent
        .y_ring
        .iter()
        .map(|row| row.iter().map(k_to_pair).collect())
        .collect();
    let y_zcol_pairs: Vec<[F; 2]> = post_parent.y_zcol.iter().map(k_to_pair).collect();
    let s_col_pairs: Vec<[F; 2]> = post_parent.s_col.iter().map(k_to_pair).collect();

    NifsCeClaimView {
        d: post_parent.c.d as u64,
        kappa: post_parent.c.kappa as u64,
        c_data: post_parent.c.data.clone(),
        x_rows: post_parent.X.rows() as u64,
        x_cols: post_parent.X.cols() as u64,
        x_active_cols: active_cols as u64,
        x_active_flat,
        r: r_pairs,
        y_ring: y_ring_pairs,
        y_zcol: y_zcol_pairs,
        s_col: s_col_pairs,
        m_in: post_parent.m_in as u64,
        fold_digest_fields: digest32_as_fields(post_parent.fold_digest),
    }
}

/// Flatten the active-columns prefix of `X` row-major.
fn collect_active_x(x: &Mat<F>, active_cols: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(x.rows() * active_cols);
    for row in 0..x.rows() {
        for col in 0..active_cols {
            out.push(x[(row, col)]);
        }
    }
    out
}

/// Convert a `K` (Goldilocks degree-2 extension) into the `[F; 2]`
/// shape `NifsCeClaimView` expects.
fn k_to_pair(k: &K) -> [F; 2] {
    let coeffs = neo_math::field::KExtensions::as_coeffs(k);
    [coeffs[0], coeffs[1]]
}

// ─────────────────────────────────────────────────────────────────────────
// Shape / boundary digest helpers.
// ─────────────────────────────────────────────────────────────────────────

/// `chunk_digest` for the canonical F' shape, with explicit batch
/// size `fresh_count`. Mirrors a same-shape SuperNeo chunk of K =
/// `fresh_count` fresh CCS instances at this step. Shape-only: no
/// circularity with the step's own commitment data.
///
/// Synthesises one minimal [`CcsClaim`] with the right `(d, kappa,
/// m_in)` and feeds it `fresh_count` times to
/// [`f_prime_chunk_public_digest`], which reads only those three
/// per-claim fields and `(start_index, fresh.len())`. Two batches of
/// identical shape and different size therefore produce *different*
/// digests — the K-bind comes from `fresh.len()` in the absorbed
/// preimage.
pub fn chunk_digest_for_shape_count(
    start_index: u64,
    fresh_count: usize,
    d: usize,
    kappa: usize,
    m_in: usize,
) -> [F; 4] {
    assert!(
        fresh_count >= 1,
        "chunk_digest_for_shape_count: SuperNeo K \u{2265} 1 (got 0)"
    );
    let shape_claim = CcsClaim {
        c: Commitment {
            d,
            kappa,
            data: Vec::new(),
        },
        x: Vec::new(),
        m_in,
    };
    let claims = vec![shape_claim; fresh_count];
    f_prime_chunk_public_digest(start_index, &claims)
}

/// `chunk_digest` for the canonical F' shape with K = 1. Convenience
/// wrapper around [`chunk_digest_for_shape_count`].
pub fn chunk_digest_for_shape(start_index: u64, d: usize, kappa: usize, m_in: usize) -> [F; 4] {
    chunk_digest_for_shape_count(start_index, 1, d, kappa, m_in)
}

/// Decompose the 4-lane Goldilocks digest into `boundary_bits` boolean
/// field elements, little-endian per lane.
pub fn boundary_bits_from_digest(digest: [F; 4], boundary_bits: usize) -> Vec<F> {
    let mut bits = vec![F::ZERO; boundary_bits];
    for (m, lane) in digest.iter().enumerate() {
        let value = lane.as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            let pos = m * POSEIDON2_GOLDILOCKS_BITS + j;
            if pos >= boundary_bits {
                break;
            }
            bits[pos] = if (value >> j) & 1 == 1 { F::ONE } else { F::ZERO };
        }
    }
    bits
}

// ─────────────────────────────────────────────────────────────────────────
// Unified-plan trace assembly.
// ─────────────────────────────────────────────────────────────────────────

/// Read the canonical CE shape + child count out of a unified F' plan.
///
/// Rejects configurations the unified-mode compiler doesn't support:
/// no accumulator, legacy non-unified plan, or non-CE-claim payload.
pub fn canonical_ce_shape_and_child_count(
    plan: &RecursiveStepImagePlan,
) -> Result<(NifsCeClaimShape, u64), FPrimeShellCompilerError> {
    let acc = plan
        .accumulator
        .as_ref()
        .ok_or(FPrimeShellCompilerError::CanonicalPlanMissingAccumulator)?;
    if !acc.unified {
        return Err(FPrimeShellCompilerError::CanonicalPlanNotUnified);
    }
    let ce_shape = match &plan.nifs_payload_shapes[acc.ce_claim_payload_index] {
        NifsPayloadShape::CeClaim(s) => s.clone(),
        _ => return Err(FPrimeShellCompilerError::CanonicalPlanPayloadNotCeClaim),
    };
    Ok((ce_shape, acc.child_count))
}

/// Assemble the app-agnostic shell traces for one unified F' step.
///
/// Produces the five Poseidon traces every unified step emits
/// (`boundary_update`, `public_trace_update`, base / recursive
/// accumulator, `state_x_out`), the matching `StateIn` / `StateOut`,
/// the `boundary_bits` to splice into the image, and the chain state
/// the caller should advance to after the encoder consumes the
/// assembly.
///
/// `app_public_input` is the app-level public input appended to the
/// `state_x_out` preimage. Fibonacci passes `&[]` (the boundary digest
/// alone commits to the chain's verifier-visible output); R1CS passes
/// the satisfying assignment's public prefix so the verifier learns
/// "this `x` was proven," not just "some assignment satisfies the
/// shape."
///
/// `rows_in_chunk` is the SuperNeo same-shape batch size of *this*
/// step's deposit (= `next_latest.len()` in native). It drives
/// `step_count` advance (`+= rows_in_chunk`) and is absorbed into the
/// shape `chunk_digest`. For a K=1 step, pass `1`; the produced trace
/// matches the legacy K=1 path bit-for-bit.
pub fn assemble_unified_step_traces(
    ctx: &FPrimeCompilerContext,
    is_base: bool,
    recursive_c_data: &[F],
    child_count: u64,
    app_public_input: &[F],
    rows_in_chunk: usize,
) -> UnifiedStepTraceAssembly {
    let shared = assemble_shared_chunk_traces(ctx, is_base, recursive_c_data, child_count, rows_in_chunk);
    assemble_step_from_shared(&shared, ctx, app_public_input)
}

/// The chunk-shared portion of [`assemble_unified_step_traces`]:
/// everything that does **not** depend on a step's `app_public_input`.
///
/// For a K-sized SuperNeo chunk all assignments share the same pre-step
/// `ctx`, so they share the `chunk_digest`, the four Poseidon traces
/// (boundary, public_trace, base/recursive accumulator), and the
/// post-step chain-coordinate advance. Computing this once per chunk and
/// reusing it (via [`assemble_step_from_shared`]) avoids recomputing the
/// four Poseidon traces K times — that recomputation was the dominant
/// per-assignment compile cost. The only per-assignment work left is the
/// `state_x_out` trace (which absorbs the app public input) and the
/// `boundary_bits` derived from it.
pub struct SharedChunkTraces {
    pub state_in: StateIn,
    pub state_out: StateOut,
    pub chunk_digest: [F; 4],
    pub next_chain_state: FPrimeChainState,
    boundary: PoseidonTraceImage,
    public_trace: PoseidonTraceImage,
    base_accumulator: PoseidonTraceImage,
    recursive_accumulator: PoseidonTraceImage,
}

/// Compute the [`SharedChunkTraces`] once for a SuperNeo chunk. See that
/// type's docs. `rows_in_chunk` is the chunk's batch size `K`.
pub fn assemble_shared_chunk_traces(
    ctx: &FPrimeCompilerContext,
    is_base: bool,
    recursive_c_data: &[F],
    child_count: u64,
    rows_in_chunk: usize,
) -> SharedChunkTraces {
    assert!(
        rows_in_chunk >= 1,
        "assemble_shared_chunk_traces: SuperNeo K \u{2265} 1 (got 0)"
    );
    // SuperNeo same-shape chunk of K = `rows_in_chunk` fresh CCS
    // instances. The chunk_digest absorbs K alongside the per-claim
    // shape digest, matching native
    // `f_prime_chunk_public_digest_for_step(start_index, next_latest)`.
    let chunk_digest = chunk_digest_for_shape_count(
        ctx.chain_state.step_count,
        rows_in_chunk,
        ctx.commitment_d,
        ctx.commitment_kappa,
        ctx.public_input_len,
    );

    let state_in = StateIn {
        vk_fs_digest: ctx.vk_fs_digest,
        structure_digest: ctx.structure_digest,
        z_0: ctx.z_0,
        z_i_in: ctx.chain_state.z_i,
        acc_digest_in: ctx.chain_state.acc_digest,
        public_trace_in: ctx.chain_state.public_trace,
    };

    let boundary = encode_poseidon_trace(&build_boundary_update_preimage_fields(state_in.z_i_in, chunk_digest));
    let public_trace = encode_poseidon_trace(&build_public_trace_update_preimage_fields(
        state_in.public_trace_in,
        chunk_digest,
    ));
    let base_accumulator = encode_poseidon_trace(&build_accumulator_preimage_fields(0, &[]));
    let recursive_accumulator =
        encode_poseidon_trace(&build_accumulator_preimage_fields(child_count, recursive_c_data));

    let new_acc_digest = if is_base {
        base_accumulator.digest_native
    } else {
        recursive_accumulator.digest_native
    };
    let new_z_i = boundary.digest_native;
    let new_public_trace = public_trace.digest_native;
    let new_chunk_count = ctx.chain_state.chunk_count + 1;
    // Native `advance_state(prev, _, fresh_count, _)` increments
    // `step_count` by `fresh_count == next_latest.len()` per step. For
    // a K-deposit step, `rows_in_chunk = K`.
    let new_step_count = ctx.chain_state.step_count + rows_in_chunk as u64;

    let state_out = StateOut {
        new_chunk_count,
        new_step_count,
        new_z_i,
        new_public_trace,
        new_acc_digest,
    };
    let next_chain_state = FPrimeChainState {
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        z_i: new_z_i,
        acc_digest: new_acc_digest,
        public_trace: new_public_trace,
    };

    SharedChunkTraces {
        state_in,
        state_out,
        chunk_digest,
        next_chain_state,
        boundary,
        public_trace,
        base_accumulator,
        recursive_accumulator,
    }
}

/// Build one step's [`UnifiedStepTraceAssembly`] from the chunk-shared
/// traces plus this step's `app_public_input`. Only the `state_x_out`
/// trace and the `boundary_bits` it produces are computed here; the four
/// shared Poseidon traces are cloned from `shared` (a memcpy, far cheaper
/// than recomputing the permutations). The result is byte-for-byte
/// identical to calling [`assemble_unified_step_traces`] directly.
pub fn assemble_step_from_shared(
    shared: &SharedChunkTraces,
    ctx: &FPrimeCompilerContext,
    app_public_input: &[F],
) -> UnifiedStepTraceAssembly {
    let state_x_out = encode_poseidon_trace(&build_state_x_out_preimage_fields_with_app_x(
        ctx.vk_fs_digest,
        ctx.structure_digest,
        shared.state_out.new_chunk_count,
        shared.state_out.new_step_count,
        ctx.z_0,
        shared.state_out.new_z_i,
        ctx.pc,
        shared.state_out.new_acc_digest,
        shared.state_out.new_public_trace,
        app_public_input,
    ));

    let public_output_digest = state_x_out.digest_native;
    let boundary_bits = boundary_bits_from_digest(public_output_digest, ctx.boundary_bits);

    UnifiedStepTraceAssembly {
        state_in: shared.state_in,
        state_out: shared.state_out,
        chunk_digest: shared.chunk_digest,
        public_output_digest,
        boundary_bits,
        next_chain_state: shared.next_chain_state,
        traces: UnifiedStepPoseidonTraces {
            boundary: shared.boundary.clone(),
            public_trace: shared.public_trace.clone(),
            base_accumulator: shared.base_accumulator.clone(),
            recursive_accumulator: shared.recursive_accumulator.clone(),
            state_x_out,
        },
    }
}
