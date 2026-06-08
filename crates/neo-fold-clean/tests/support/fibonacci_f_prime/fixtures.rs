//! Test-support helpers for folding **encoded F'** instances through the
//! existing lifecycle.
//!
//! This is a deliberate, narrow path for tests that want a "real F'
//! shape" fixture without touching `direct_ccs` (which still folds raw
//! application CCS for legacy callers). Production callers do **not**
//! go through this module: the encoder-driven IVC entry point is a
//! later milestone with a proper app-to-F' compiler contract.
//!
//! This module owns **only fixtures**: canonical Phase 1.5a step input
//! constants, the threaded-chain builder, and the threaded base state.
//! Preprocessing / instance construction / lifecycle wiring is the
//! production [`fibonacci_f_prime`] frontend's job and
//! tests call it directly.
//!
//! Exposes:
//! - [`build_honest_step_input`] — the Phase 1.5a fixture builder.
//! - [`honest_state_threaded_encoded_f_prime_records`] /
//!   [`honest_state_threaded_encoded_f_prime_steps`] — Phase 1.6a:
//!   build N encoded F' steps whose `state_out` of step i threads into
//!   `state_in` of step i+1.

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::compiler::nifs_payload_inputs_for_source_image;
use neo_fold_clean::frontends::f_prime::encoder::{encode_f_prime_step, EncodedFPrimeStep, FPrimeStepInput};
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImageLayout, NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut,
};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, build_state_x_out_preimage_fields, AccumulatorPlanOptions,
    RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::paper::digest::{AccumulatorHandle, StateXOutDigestMode};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Canonical CE shape pinned by the fixture, **a fixed point** of the
/// lifecycle's Π_RLC + Π_DEC fold under the SuperNeo Goldilocks
/// `paper_b2` params:
///
/// 1. Set `canonical_threaded_plan` to these dimensions.
/// 2. Preprocess → derive the canonical structure (whose `m`, `n`
///    depend on this shape).
/// 3. Fold one perp-fill step through `prove_encoded_steps` +
///    `finish_uncompressed_with_audit`.
/// 4. Inspect `running.parent_authority`. It comes out with exactly
///    these dimensions — i.e. `f(plan) = plan_shape`. (Verified by a
///    one-off probe). `r_len` tracks the F' row domain and is smaller
///    than `s_col_len`, which tracks the still-large column domain. The
///    values below are re-derived after the stateless `state_x_out`
///    digest omits the duplicate semantic accumulator lanes.
///
/// Because this is a fixed point, the compiler can use `prep.plan`
/// for both the base and recursive paths and `nifs_ce_view_from_claim`
/// always fits inside `prep.plan`'s NIFS payload region — see
/// `compiler::compile_recursive_step`'s shape-validation guard.
const C_DATA_ENTRIES: usize = 972;
const X_ROWS: usize = 54;
const X_ACTIVE_COLS: usize = 5;
const R_LEN: usize = 12;
const Y_RING_OUTER: usize = 8;
const Y_RING_INNER: usize = 64;
const Y_ZCOL_LEN: usize = 64;
const S_COL_LEN: usize = 18;
const CHILD_COUNT: u64 = 14;
const PC: u64 = 1;
const NEW_CHUNK_COUNT: u64 = 7;
const NEW_STEP_COUNT: u64 = 13;
const PUBLIC_X_OUT_LANE_COUNT: usize = 4;
/// Public-x_out boundary bit count (4 lanes × 64 bits = 256). Tests can
/// reference this when they need the canonical `m_in = 1 + BOUNDARY_BITS`
/// directly.
pub const BOUNDARY_BITS: usize = PUBLIC_X_OUT_LANE_COUNT * POSEIDON2_GOLDILOCKS_BITS;

fn canonical_ce_shape() -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: C_DATA_ENTRIES,
        x_rows: X_ROWS,
        x_active_cols: X_ACTIVE_COLS,
        r_len: R_LEN,
        y_ring_inner_lens: vec![Y_RING_INNER; Y_RING_OUTER],
        y_zcol_len: Y_ZCOL_LEN,
        s_col_len: S_COL_LEN,
    }
}

fn make_plan_without_state_x_out() -> RecursiveStepImagePlan {
    RecursiveStepImagePlan {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(canonical_ce_shape())],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: C_DATA_ENTRIES,
            child_count: CHILD_COUNT,
            // Canonical Fibonacci chains are unified-mode end-to-end:
            // the producer carries the post-fold accumulator handle in
            // state_out and the next recursive step / terminal fold checks
            // it against NIFS.V output. There is no producer-side
            // accumulator Poseidon trace in the F' image.
            unified: true,
        }),
        state_x_out: None,
    }
}

/// Perp/zero CE view matching `canonical_ce_shape()` — used as the
/// fixture's deterministic-filler NIFS payload. The recursive
/// accumulator handle sources from this payload's `c_data`, so each
/// fixture step's accumulator digest is the deterministic
/// `H(tag, CHILD_COUNT, c_data_entries, 0, 0, ..., 0)`.
fn perp_canonical_ce_view() -> NifsCeClaimView {
    let shape = canonical_ce_shape();
    NifsCeClaimView {
        d: 0,
        kappa: 0,
        c_data: vec![F::ZERO; shape.c_data_entries],
        x_rows: shape.x_rows as u64,
        x_cols: shape.x_active_cols as u64,
        x_active_cols: shape.x_active_cols as u64,
        x_active_flat: vec![F::ZERO; shape.x_rows * shape.x_active_cols],
        r: vec![[F::ZERO; 2]; shape.r_len],
        y_ring: shape
            .y_ring_inner_lens
            .iter()
            .map(|&len| vec![[F::ZERO; 2]; len])
            .collect(),
        y_zcol: vec![[F::ZERO; 2]; shape.y_zcol_len],
        s_col: vec![[F::ZERO; 2]; shape.s_col_len],
        m_in: 0,
        fold_digest_fields: [F::ZERO; 4],
    }
}

/// Build the three Poseidon traces, post-step state-out digests, boundary
/// public-x_out bits, and the matching `FPrimeStepInput`. The
/// caller hands the returned input directly to
/// `encode_f_prime_step`.
///
/// Returns `(input, state_x_out_digest)` so callers can cross-check the
/// public-x_out boundary bits against `encode_x_out_public_bits`.
pub fn build_honest_step_input() -> (FPrimeStepInput, [F; 4]) {
    // Probe-build to learn boundary's offset, then rebuild the plan with
    // concrete public-x_out lane bit starts.
    let probe_plan = make_plan_without_state_x_out();
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|m| boundary_start + m * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = make_plan_without_state_x_out();
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: PC,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });

    let vk_fs_digest: [F; 4] = [
        F::from_u64(0x101),
        F::from_u64(0x202),
        F::from_u64(0x303),
        F::from_u64(0x404),
    ];
    let structure_digest: [F; 4] = [
        F::from_u64(0x505),
        F::from_u64(0x606),
        F::from_u64(0x707),
        F::from_u64(0x808),
    ];
    let z_0: [F; 4] = [
        F::from_u64(0x900),
        F::from_u64(0xa00),
        F::from_u64(0xb00),
        F::from_u64(0xc00),
    ];
    let z_i_in: [F; 4] = [
        F::from_u64(0x111),
        F::from_u64(0x222),
        F::from_u64(0x333),
        F::from_u64(0x444),
    ];
    let public_trace_in: [F; 4] = [
        F::from_u64(0xaaa),
        F::from_u64(0xbbb),
        F::from_u64(0xccc),
        F::from_u64(0xddd),
    ];
    let chunk_digest: [F; 4] = [
        F::from_u64(0x10001),
        F::from_u64(0x20002),
        F::from_u64(0x30003),
        F::from_u64(0x40004),
    ];

    let perp_view = perp_canonical_ce_view();

    let state_in = StateIn {
        vk_fs_digest,
        structure_digest,
        z_0,
        z_i_in,
        semantic_state_digest_in: [F::ZERO; 4],
        acc_digest_in: [F::ZERO; 4],
        public_trace_in,
    };

    // Encode the state-output Poseidon trace. The local chunk-shape
    // coordinate mirrors `chunk_digest` linearly; no producer-side
    // boundary-update hash trace is spliced into this F' image.
    let new_z_i = chunk_digest;
    let new_public_trace = new_z_i;
    let new_acc_digest = AccumulatorHandle::empty().digest_fields();

    let state_out = StateOut {
        new_chunk_count: NEW_CHUNK_COUNT,
        new_step_count: NEW_STEP_COUNT,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest: new_acc_digest,
        new_acc_digest,
    };

    let state_x_out_preimage = build_state_x_out_preimage_fields(
        StateXOutDigestMode::Stateless,
        vk_fs_digest,
        structure_digest,
        NEW_CHUNK_COUNT,
        NEW_STEP_COUNT,
        z_0,
        new_z_i,
        PC,
        new_acc_digest,
        new_acc_digest,
        new_public_trace,
    );
    let state_x_out_trace = encode_poseidon_trace(&state_x_out_preimage);
    let state_x_out_digest = state_x_out_trace.digest_native;

    // Pack boundary public-x_out bits little-endian, matching what
    // `encode_x_out_public_bits` produces.
    let mut boundary_bits = vec![F::ZERO; BOUNDARY_BITS];
    for (m, digest_lane) in state_x_out_digest.iter().enumerate() {
        let value = digest_lane.as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            boundary_bits[m * POSEIDON2_GOLDILOCKS_BITS + j] = if ((value >> j) & 1) == 1 { F::ONE } else { F::ZERO };
        }
    }

    let nifs_payloads = nifs_payload_inputs_for_source_image(&plan, perp_view);
    let input = FPrimeStepInput {
        plan,
        boundary_bits,
        state_in,
        state_out,
        chunk_digest,
        // limbs = 3 → 2 app-private carry bits.
        app_private_carries: vec![F::ZERO, F::ZERO],
        is_base: false,
        nifs_payloads,
        kmul_views: vec![],
        ring_action_pairs: vec![],
        one_shot_traces: vec![state_x_out_trace],
        sponge_trace: None,
    };

    (input, state_x_out_digest)
}

// ───────────────────────────────────────────────────────────────────────
// Phase 1.6a — State-threaded encoded F' fixtures
// ───────────────────────────────────────────────────────────────────────
//
// The plain `honest_encoded_f_prime_step(s)` fixture above produces
// *same-shape synthetic repeats*: each call builds an independent
// encoded F' step that doesn't depend on any prior step's outputs. That
// proves the encoder/lifecycle mechanism works, but not that the F'
// state machine threads realistically.
//
// The helpers below build a sequence of encoded F' steps where each
// step's input state is the previous step's output state. Specifically:
//
//   - `state_out.new_z_i` (= chunk digest)
//      → next step's `state_in.z_i_in`
//   - `state_out.new_public_trace` (= `state_out.new_z_i` in this
//     compact shape) → next step's `state_in.public_trace_in`
//   - `state_out.new_acc_digest` (= this fixture's empty accumulator handle)
//      → next step's `state_in.acc_digest_in`
//   - `chunk_count` and `step_count` increment by 1 per step.
//   - `vk_fs`, `structure`, `z_0`, `pc` stay constant across the chain
//      (per Construction 2's preprocessing-derived header).

/// A snapshot of the F' state machine — every field the encoder's
/// `StateIn` / `StateOut` populate, kept in one place so we can compare
/// `state_out` of step i to `state_in` of step i+1 trivially.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadedFPrimeState {
    pub vk_fs_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [F; 4],
    pub z_i: [F; 4],
    pub semantic_state_digest: [F; 4],
    pub acc_digest: [F; 4],
    pub public_trace: [F; 4],
    pub pc: u64,
}

/// One encoded F' step plus the surrounding (state_in, state_out)
/// snapshot, so tests can cross-check the chain's threading without
/// re-decoding the encoded image.
pub struct ThreadedEncodedFPrimeRecord {
    pub state_in: ThreadedFPrimeState,
    pub state_out: ThreadedFPrimeState,
    pub encoded: EncodedFPrimeStep,
}

/// The base-case state for the threaded fixture: counters at 0,
/// `z_i = z_0`, empty `acc_digest`, public_trace at its seed value.
/// Headers (`vk_fs_digest`, `structure_digest`, `z_0`, `pc`) are
/// arbitrary constants — the test fixture is not derived from
/// preprocessing, but the threading semantics are what matters here.
fn threaded_base_state() -> ThreadedFPrimeState {
    let z_0 = [
        F::from_u64(0x900),
        F::from_u64(0xa00),
        F::from_u64(0xb00),
        F::from_u64(0xc00),
    ];
    ThreadedFPrimeState {
        vk_fs_digest: [
            F::from_u64(0x101),
            F::from_u64(0x202),
            F::from_u64(0x303),
            F::from_u64(0x404),
        ],
        structure_digest: [
            F::from_u64(0x505),
            F::from_u64(0x606),
            F::from_u64(0x707),
            F::from_u64(0x808),
        ],
        chunk_count: 0,
        step_count: 0,
        z_0,
        z_i: z_0,
        semantic_state_digest: [F::ZERO; 4],
        acc_digest: [F::ZERO; 4],
        public_trace: [
            F::from_u64(0xaaa),
            F::from_u64(0xbbb),
            F::from_u64(0xccc),
            F::from_u64(0xddd),
        ],
        pc: PC,
    }
}

/// Canonical [`RecursiveStepImagePlan`] for the threaded fixture.
///
/// Verifier-side helper: tests hand this to
/// `frontends::fibonacci_f_prime::preprocess_seeded` so the verifier
/// derives the canonical CCS structure from a known plan (instead of
/// reading it off a prover-supplied first step). The prover-side
/// fixture (`build_threaded_step_record`) uses the same plan, so
/// `build_instance`'s `structure_digest` cross-check passes on honest
/// inputs. The fixture's `pc` is the canonical-constant `PC`; pure
/// app-step / pc threading is out of scope for this fixture.
pub fn canonical_threaded_plan() -> RecursiveStepImagePlan {
    let probe_plan = make_plan_without_state_x_out();
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|m| boundary_start + m * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = make_plan_without_state_x_out();
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: PC,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

/// Build one encoded F' step starting from `state` and produce the
/// resulting `(state_in, state_out, encoded)` record. The `step_idx`
/// is mixed into `chunk_digest` so consecutive steps produce distinct
/// boundary/public-trace digests (otherwise the chain would be a
/// degenerate fixed point).
///
/// The NIFS payload is a perp/zero view matching `canonical_ce_shape`;
/// the fixture chain doesn't model real per-step authority. Its
/// `state_out.new_acc_digest` therefore uses the empty accumulator
/// handle rather than deriving authority from the filler payload.
fn build_threaded_step_record(state: &ThreadedFPrimeState, step_idx: u64) -> ThreadedEncodedFPrimeRecord {
    debug_assert_eq!(state.pc, PC, "threaded fixture pins pc = PC across the chain");
    let plan = canonical_threaded_plan();

    let chunk_digest: [F; 4] = [
        F::from_u64(0x10001 + step_idx),
        F::from_u64(0x20002 + step_idx),
        F::from_u64(0x30003 + step_idx),
        F::from_u64(0x40004 + step_idx),
    ];

    let perp_view = perp_canonical_ce_view();
    let new_z_i = chunk_digest;
    let new_public_trace = new_z_i;
    let new_acc_digest = AccumulatorHandle::empty().digest_fields();
    let new_chunk_count = state.chunk_count + 1;
    let new_step_count = state.step_count + 1;

    let state_x_out_trace = encode_poseidon_trace(&build_state_x_out_preimage_fields(
        StateXOutDigestMode::Stateless,
        state.vk_fs_digest,
        state.structure_digest,
        new_chunk_count,
        new_step_count,
        state.z_0,
        new_z_i,
        state.pc,
        new_acc_digest,
        new_acc_digest,
        new_public_trace,
    ));

    let mut boundary_bits = vec![F::ZERO; BOUNDARY_BITS];
    for (m, lane) in state_x_out_trace.digest_native.iter().enumerate() {
        let value = lane.as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            boundary_bits[m * POSEIDON2_GOLDILOCKS_BITS + j] = if ((value >> j) & 1) == 1 { F::ONE } else { F::ZERO };
        }
    }

    let nifs_payloads = nifs_payload_inputs_for_source_image(&plan, perp_view);
    let input = FPrimeStepInput {
        plan,
        boundary_bits,
        state_in: StateIn {
            vk_fs_digest: state.vk_fs_digest,
            structure_digest: state.structure_digest,
            z_0: state.z_0,
            z_i_in: state.z_i,
            semantic_state_digest_in: state.semantic_state_digest,
            acc_digest_in: state.acc_digest,
            public_trace_in: state.public_trace,
        },
        state_out: StateOut {
            new_chunk_count,
            new_step_count,
            new_z_i,
            new_public_trace,
            new_semantic_state_digest: new_acc_digest,
            new_acc_digest,
        },
        chunk_digest,
        // limbs = 3 → 2 app-private carry bits.
        app_private_carries: vec![F::ZERO, F::ZERO],
        is_base: false,
        nifs_payloads,
        kmul_views: vec![],
        ring_action_pairs: vec![],
        one_shot_traces: vec![state_x_out_trace],
        sponge_trace: None,
    };

    let state_out = ThreadedFPrimeState {
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        z_i: new_z_i,
        semantic_state_digest: new_acc_digest,
        acc_digest: new_acc_digest,
        public_trace: new_public_trace,
        ..state.clone()
    };

    ThreadedEncodedFPrimeRecord {
        state_in: state.clone(),
        state_out,
        encoded: encode_f_prime_step(input),
    }
}

/// Build `n` encoded F' steps where each step's output state threads
/// into the next step's input state. The starting state is
/// [`threaded_base_state`].
pub fn honest_state_threaded_encoded_f_prime_records(n: usize) -> Vec<ThreadedEncodedFPrimeRecord> {
    let mut state = threaded_base_state();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let record = build_threaded_step_record(&state, i as u64);
        state = record.state_out.clone();
        out.push(record);
    }
    out
}

/// Convenience: same as [`honest_state_threaded_encoded_f_prime_records`]
/// but returns only the encoded steps. Use this when only the encoded
/// step is needed (e.g., to drive the lifecycle).
pub fn honest_state_threaded_encoded_f_prime_steps(n: usize) -> Vec<EncodedFPrimeStep> {
    honest_state_threaded_encoded_f_prime_records(n)
        .into_iter()
        .map(|r| r.encoded)
        .collect()
}
