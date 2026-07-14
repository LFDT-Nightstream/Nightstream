//! Shared F' compiler state and helpers.
//!
//! This module owns app-agnostic compiler state and the protocol-generic
//! checks every concrete frontend (Fibonacci, R1CS, …) repeats: chain
//! coordinates, prior-fold authority, optional source-image NIFS payload
//! views, and
//! transcript-bound prior-fold verification. App frontends keep only
//! their app-specific witness checks, encoder dispatch, and output
//! shape.

use neo_ajtai::Commitment;
use neo_ccs::matrix::Mat;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::f_prime::encoder::NifsPayloadInput;
use crate::frontends::f_prime::image::{NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut};
use crate::frontends::f_prime::recursive_plan::{
    build_state_x_out_preimage_fields_with_app_x, source_image_emits_nifs_payloads, RecursiveStepImagePlan,
};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::{LatestInstance, ProofState, RunningInstance, State as PaperState};
use crate::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, initial_boundary_digest,
    public_trace_seed_digest, AccumulatorHandle, StateXOutDigestMode,
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
/// The chain header (`vk_fs_digest`, `pi_ccs_header_bundle`, `z_0`, `pc`,
/// `public_input_len`, commitment / boundary / limb shape) is constant
/// across steps. `pc` is pinned as the single-`F'_j` state selector and
/// absorbed into `state_x_out`. [`FPrimeChainState`] is updated each step;
/// [`FPrimeFoldForStep`] is the optional full per-step fold authority the
/// caller writes between successive recursive compiles. Accelerated provers
/// that just produced and validated the fold may instead provide only
/// [`FPrimeFoldPostSummary`] through `fold_summary_for_step`.
#[derive(Clone, Debug)]
pub struct FPrimeCompilerContext {
    // Chain header — constant across steps.
    pub vk_fs_digest: [F; 4],
    pub pi_ccs_header_bundle: [F; 4],
    pub z_0: [F; 4],
    pub pc: u64,
    pub public_input_len: usize,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub boundary_bits: usize,
    pub limbs: usize,
    pub state_x_out_digest_mode: StateXOutDigestMode,
    // Threaded F' chain state — compiler updates each step.
    pub chain_state: FPrimeChainState,
    // Prior fold authority boundary — caller writes between steps.
    pub fold_for_step: Option<FPrimeFoldForStep>,
    /// Compile-facing surface for a backend-validated prior fold.
    ///
    /// This is mutually exclusive with `fold_for_step`. It deliberately does
    /// not carry verifier authority: final proof/audit verification still
    /// materializes and checks the ordinary NIFS proof.
    pub fold_summary_for_step: Option<FPrimeFoldPostSummary>,
    /// If true, recursive compile replays `NIFS.V` before consuming
    /// `fold_for_step`. CUDA-backed sessions can set this false for the
    /// next step after the backend has just produced the fold; final
    /// proof/audit verification still checks the emitted NIFS proof.
    pub fold_for_step_needs_native_verify: bool,
}

/// F'-level chain state threaded between successive compile calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FPrimeChainState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_i: [F; 4],
    pub semantic_state_digest: [F; 4],
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
///   commits to via the outgoing accumulator handle.
///
/// The recursive plan is derived from `post_running.parent_authority`'s
/// real CE shape. Legacy non-unified plans also fill a source-image
/// NIFS payload view from that post-fold parent; unified plans keep the
/// shape metadata for prior-fold verification but elide the payload
/// columns from the low-norm source image.
#[derive(Clone, Debug)]
pub struct FPrimeFoldForStep {
    pub pre_running: RunningInstance,
    pub latest: LatestInstance,
    pub proof: NifsProof,
    pub post_running: RunningInstance,
    pub post_summary: Option<FPrimeFoldPostSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FPrimeFoldPostSummary {
    pub parent_shape: NifsCeClaimShape,
    pub child_count: u64,
    pub acc_digest: [F; 4],
}

impl FPrimeFoldPostSummary {
    pub fn from_running(running: &RunningInstance, public_input_len: usize) -> Result<Self, FPrimeShellCompilerError> {
        let parent = running
            .parent_authority
            .as_ref()
            .ok_or(FPrimeShellCompilerError::PostRunningMissingParentAuthority)?;
        Ok(Self {
            parent_shape: nifs_ce_shape_from_claim(parent, public_input_len),
            child_count: running.claims.len() as u64,
            acc_digest: AccumulatorHandle::from_running_parts(&running.claims, Some(parent)).digest_fields(),
        })
    }
}

/// The Poseidon trace every canonical unified F' step emits.
///
/// The accumulator handle is carried in `state_out` and checked when the
/// next recursive step or terminal fold consumes it. The local chunk
/// coordinate `new_z_i` mirrors `chunk_digest` linearly.
pub struct UnifiedStepPoseidonTraces {
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

    #[error("F' shell compiler: recursive step supplied both a full prior fold and a backend post-fold summary")]
    ConflictingPriorFoldInputs,

    #[error(
        "F' shell compiler: backend post-fold summary requires the prior fold to have been validated by the backend"
    )]
    UnverifiedPriorFoldSummary,

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
/// Derives the chain header (vk_fs_digest, pi_ccs_header_bundle, z_0,
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
    let acc_digest = AccumulatorHandle::empty().digest_fields();
    let vk_fs_digest = digest32_as_fields(prep.vk.digest());

    Ok(FPrimeCompilerContext {
        vk_fs_digest,
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        z_0,
        pc,
        public_input_len,
        commitment_d: neo_math::D,
        commitment_kappa: prep.params.kappa() as usize,
        boundary_bits,
        limbs,
        state_x_out_digest_mode: match prep.semantic_state_mode() {
            crate::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
            crate::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
        },
        chain_state: FPrimeChainState {
            chunk_count: 0,
            step_count: 0,
            z_i: z_0,
            semantic_state_digest: acc_digest,
            acc_digest,
            public_trace,
        },
        fold_for_step: None,
        fold_summary_for_step: None,
        fold_for_step_needs_native_verify: true,
    })
}

/// Re-derive `post_running` from `(pre_running, latest, proof)` via
/// `nifs::verify`. Reject on (a) NIFS rejection or (b) derived ≠
/// caller-supplied `post_running`.
///
/// The transcript is the per-step F' transcript
/// (`F_PRIME_STEP_TRANSCRIPT_LABEL` plus the state-bound F'-step context
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
        initial_semantic_state_digest: prep.initial_semantic_state_digest(),
        semantic_state_digest: digest_fields_as_digest32(ctx.chain_state.semantic_state_digest),
        acc_digest: digest_fields_as_digest32(ctx.chain_state.acc_digest),
        public_trace: digest_fields_as_digest32(ctx.chain_state.public_trace),
        proof: ProofState::Initial,
        nebula: None,
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
/// must not enter the base step's accumulator. The compiler achieves
/// this by carrying the constant empty accumulator handle on base
/// steps; recursive steps carry a handle computed from the full
/// verified post-fold running accumulator.
///
/// Audit hook: any future change must keep the base-step perp payload's
/// `c_data` out of authority, or this filler becomes a soundness bug.
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

pub fn nifs_ce_shape_from_claim(post_parent: &CeClaim, _public_input_len: usize) -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: post_parent.c.data.len(),
        x_rows: post_parent.X.rows(),
        x_active_cols: crate::paper::relations::superneo_public_x_cols(post_parent.m_in),
        r_len: post_parent.r.len(),
        y_ring_inner_lens: post_parent.y_ring.iter().map(|row| row.len()).collect(),
        y_zcol_len: post_parent.y_zcol.len(),
        s_col_len: post_parent.s_col.len(),
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
        adv: None,
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
/// Produces the Poseidon trace every unified step emits
/// (`state_x_out`), the matching `StateIn` / `StateOut`,
/// the `boundary_bits` to splice into the image, and the chain state
/// the caller should advance to after the encoder consumes the
/// assembly.
///
/// `app_public_input` is retained only for older call sites. New
/// R1CS-F' code binds app-public data through the outgoing semantic-state
/// digest before calling this shared shell, so native `compute_x_out`
/// and the F' CCS agree on the `state_x_out` preimage.
///
/// `rows_in_chunk` is the SuperNeo same-shape batch size of *this*
/// step's deposit (= `next_latest.len()` in native). It drives
/// `step_count` advance (`+= rows_in_chunk`) and is absorbed into the
/// shape `chunk_digest`. For a K=1 step, pass `1`; the produced trace
/// matches the legacy K=1 path bit-for-bit.
pub fn assemble_unified_step_traces(
    ctx: &FPrimeCompilerContext,
    is_base: bool,
    new_acc_digest: [F; 4],
    app_public_input: &[F],
    rows_in_chunk: usize,
) -> UnifiedStepTraceAssembly {
    let shared = assemble_shared_chunk_traces(ctx, is_base, new_acc_digest, rows_in_chunk);
    assemble_step_from_shared(&shared, ctx, app_public_input, None)
}

pub fn nifs_payload_inputs_for_source_image(
    plan: &RecursiveStepImagePlan,
    ce_view: NifsCeClaimView,
) -> Vec<NifsPayloadInput> {
    if source_image_emits_nifs_payloads(plan) {
        vec![NifsPayloadInput::Ce(ce_view)]
    } else {
        Vec::new()
    }
}

/// The chunk-shared portion of [`assemble_unified_step_traces`]:
/// everything that does **not** depend on a step's app-public semantic
/// output.
///
/// For a K-sized SuperNeo chunk all assignments share the same pre-step
/// `ctx`, so they share the `chunk_digest` and the post-step
/// chain-coordinate advance. Any app-public binding is carried through
/// the semantic-state digest before `state_x_out`, so this shared shell
/// has no hidden per-assignment public-input trailer.
pub struct SharedChunkTraces {
    pub state_in: StateIn,
    pub state_out: StateOut,
    pub chunk_digest: [F; 4],
    pub next_chain_state: FPrimeChainState,
}

/// Compute the [`SharedChunkTraces`] once for a SuperNeo chunk. See that
/// type's docs. `rows_in_chunk` is the chunk's batch size `K`.
pub fn assemble_shared_chunk_traces(
    ctx: &FPrimeCompilerContext,
    is_base: bool,
    new_acc_digest: [F; 4],
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
        structure_digest: ctx.pi_ccs_header_bundle,
        z_0: ctx.z_0,
        z_i_in: ctx.chain_state.z_i,
        semantic_state_digest_in: ctx.chain_state.semantic_state_digest,
        acc_digest_in: ctx.chain_state.acc_digest,
        public_trace_in: ctx.chain_state.public_trace,
    };

    let _ = is_base;
    let new_z_i = chunk_digest;
    let new_public_trace = new_z_i;
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
        new_semantic_state_digest: new_acc_digest,
        new_acc_digest,
    };
    let next_chain_state = FPrimeChainState {
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        z_i: new_z_i,
        semantic_state_digest: new_acc_digest,
        acc_digest: new_acc_digest,
        public_trace: new_public_trace,
    };

    SharedChunkTraces {
        state_in,
        state_out,
        chunk_digest,
        next_chain_state,
    }
}

/// Build one step's [`UnifiedStepTraceAssembly`] from the chunk-shared
/// traces. App public input is already represented by
/// `semantic_state_digest_out`; `state_x_out` only hashes Construction-2
/// state coordinates.
pub fn assemble_step_from_shared(
    shared: &SharedChunkTraces,
    ctx: &FPrimeCompilerContext,
    app_public_input: &[F],
    semantic_state_digest_out: Option<[F; 4]>,
) -> UnifiedStepTraceAssembly {
    let mut state_out = shared.state_out;
    if let Some(semantic_state_digest) = semantic_state_digest_out {
        state_out.new_semantic_state_digest = semantic_state_digest;
    }
    #[cfg(feature = "perf-timers")]
    let t_preimage = std::time::Instant::now();
    let preimage = build_state_x_out_preimage_fields_with_app_x(
        ctx.state_x_out_digest_mode,
        ctx.vk_fs_digest,
        ctx.pi_ccs_header_bundle,
        state_out.new_chunk_count,
        state_out.new_step_count,
        ctx.z_0,
        state_out.new_z_i,
        ctx.pc,
        state_out.new_semantic_state_digest,
        state_out.new_acc_digest,
        state_out.new_public_trace,
        app_public_input,
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[f-prime] state_x_out preimage ({} fields)   {:>7.2}s",
        preimage.len(),
        t_preimage.elapsed().as_secs_f64()
    );
    #[cfg(feature = "perf-timers")]
    let t_trace = std::time::Instant::now();
    let state_x_out = encode_poseidon_trace(&preimage);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[f-prime] state_x_out poseidon trace         {:>7.2}s",
        t_trace.elapsed().as_secs_f64()
    );

    let public_output_digest = state_x_out.digest_native;
    let boundary_bits = boundary_bits_from_digest(public_output_digest, ctx.boundary_bits);

    UnifiedStepTraceAssembly {
        state_in: shared.state_in,
        state_out,
        chunk_digest: shared.chunk_digest,
        public_output_digest,
        boundary_bits,
        next_chain_state: FPrimeChainState {
            semantic_state_digest: state_out.new_semantic_state_digest,
            ..shared.next_chain_state
        },
        traces: UnifiedStepPoseidonTraces { state_x_out },
    }
}
