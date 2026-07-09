//! Shared R1CS-F' compiler test fixtures.
//!
//! Re-used by the three sibling test binaries:
//! - `tests/system/r1cs_compiler.rs` — basic compile/satisfaction + shape.
//! - `tests/system/r1cs_compiler_stateful.rs` — stateful semantic-digest path.
//! - `tests/system/r1cs_preprocess.rs` — preprocess-time plan validation.

#![allow(non_snake_case, dead_code, unused_imports)]

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};

/// Number of `c_data` lanes in the test NIFS payload. Small enough to
/// keep structure construction cheap; large enough to exercise the
/// recursive-accumulator hash plumbing.
pub const TEST_C_DATA_ENTRIES: usize = 2;

/// Canonical 4-lane boundary for state_x_out's public digest.
pub const BOUNDARY_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;

/// `preprocess_seeded` wrapper that returns the error directly.
/// `Result::expect_err` would require `R1csFPrimePreprocessing: Debug`,
/// which the type intentionally doesn't carry (the inner `Preprocessing`
/// doesn't either).
pub fn expect_preprocess_err(r1cs: &R1cs, plan: &RecursiveStepImagePlan, seed: u64) -> r1cs_f_prime::Error {
    match r1cs_f_prime::preprocess_seeded(r1cs, plan, seed) {
        Ok(_) => panic!("preprocess_seeded must reject this plan; it accepted instead"),
        Err(e) => e,
    }
}

/// One R1CS variable occupies 64 bits in `app_private`.
pub fn app_private_bits_for(m: usize) -> usize {
    m * POSEIDON2_GOLDILOCKS_BITS
}

/// Build a small recursive-step plan sized for an R1CS with `m` variables.
/// `m_in` is the public-input variable count; the plan binds variables
/// `[0..m_in)` into the carried semantic-state digest, which is then
/// absorbed by `state_x_out`. Uses a 2-entry CE NIFS payload so the
/// structure is small but still hosts the unified-mode accumulator
/// selector + Poseidon transitions.
pub fn make_small_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    // Sized so app_private holds m * 64 bits. limbs = m*64 + 1 because
    // image::FPrimeImageLayout::new puts app_private at
    // `limbs - 1` bits.
    let limbs = app_private_bits_for(m) + 1;

    let ce_shape = NifsCeClaimShape {
        c_data_entries: TEST_C_DATA_ENTRIES,
        x_rows: 0,
        x_active_cols: 0,
        r_len: 0,
        y_ring_inner_lens: vec![],
        y_zcol_len: 0,
        s_col_len: 0,
    };

    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths: Vec::new(),
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_pair_count: 0,
        projection_identity_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape.clone())],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: TEST_C_DATA_ENTRIES,
            child_count: 1,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| boundary_start + i * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        // Bind every R1CS public-input variable into the carried
        // semantic-state digest, which `state_x_out` then absorbs.
        app_public_input_var_indices: (0..m_in).collect(),
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

/// R1CS with one constraint `z[0] = z[1] * z[2]` and `m_in = 1`.
/// Variable order: [z_0 (out), z_1, z_2, ...]. The matrix is padded to
/// `neo_math::D` columns for ergonomics — the bottom `(m - 3)` variables
/// are unconstrained app-private values. We set them to zero in the
/// test assignments so the low-norm F' image stays canonical.
pub fn one_product_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 2)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 0)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

/// R1CS with two constraints `z[0] = z[1] * z[2]` and `z[3] = z[4] * z[5]`.
pub fn two_product_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(2, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(1, 4)] = F::ONE;
    let mut b = NeoMat::zero(2, m, F::default());
    b[(0, 2)] = F::ONE;
    b[(1, 5)] = F::ONE;
    let mut c = NeoMat::zero(2, m, F::default());
    c[(0, 0)] = F::ONE;
    c[(1, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

/// Fibonacci step expressed as a 1-row R1CS:
/// `z[3] = (z[1] + z[2]) · z[0]`  ⟺  next = prev + curr  (`z[0] = 1`).
/// Variable layout: `[1, prev, curr, next, ...zero pads to D]`.
pub fn fibonacci_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

/// Assignment for `one_product_r1cs`: z[1] = a, z[2] = b, z[0] = a*b,
/// rest zero. `a` and `b` must be small enough that `a*b` fits in 64
/// bits unsigned (the encoder writes 64 little-endian bits per variable
/// and the structure recomposes them as `Σ 2^i · bit`; values outside
/// `[0, 2^64)` would silently truncate). For Goldilocks we stay well
/// under that.
pub fn assignment_one_product(a: u64, b: u64) -> Vec<F> {
    let m = neo_math::D;
    let mut z = vec![F::ZERO; m];
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[0] = F::from_u64(a * b);
    z
}

pub fn assignment_one_product_with_extras(a: u64, b: u64, extras: &[(usize, u64)]) -> Vec<F> {
    let mut z = assignment_one_product(a, b);
    for &(index, value) in extras {
        z[index] = F::from_u64(value);
    }
    z
}

pub fn make_stateful_plan(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
) -> RecursiveStepImagePlan {
    // Tests that don't care about the specific anchor value still need
    // SOME anchor — the new validation requires `(has indices) iff
    // (has anchor)`. Use an arbitrary placeholder; tests that DO care
    // about the anchor value call `make_stateful_plan_with_anchor`
    // directly with their chosen value.
    let default_anchor = [0x42u8; 32];
    make_stateful_plan_with_anchor(
        m,
        m_in,
        semantic_state_in_var_indices,
        semantic_state_out_var_indices,
        Some(default_anchor),
    )
}

pub fn make_stateful_plan_with_anchor(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
    initial_semantic_state_digest_anchor: Option<[u8; 32]>,
) -> RecursiveStepImagePlan {
    let mut plan = make_small_plan(m, m_in);
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("make_small_plan installs state_x_out");
    state_x_out.semantic_state_in_var_indices = semantic_state_in_var_indices;
    state_x_out.semantic_state_out_var_indices = semantic_state_out_var_indices;
    state_x_out.initial_semantic_state_digest_anchor = initial_semantic_state_digest_anchor;
    plan
}

pub fn make_tiny_stateful_lifecycle_plan_with_anchor(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
    initial_semantic_state_digest_anchor: Option<[u8; 32]>,
) -> RecursiveStepImagePlan {
    let mut plan = make_tiny_lifecycle_plan(m, m_in);
    let NifsPayloadShape::CeClaim(shape) = &mut plan.nifs_payload_shapes[0] else {
        panic!("tiny lifecycle plan uses a CE payload");
    };
    // Stateful semantic binding adds two Poseidon2 traces / binding
    // blocks. Under the tiny test params this converges to a slightly
    // larger fixed point than the stateless tiny lifecycle shape.
    shape.r_len = 13;
    shape.s_col_len = 19;
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("make_tiny_lifecycle_plan installs state_x_out");
    state_x_out.semantic_state_in_var_indices = semantic_state_in_var_indices;
    state_x_out.semantic_state_out_var_indices = semantic_state_out_var_indices;
    state_x_out.initial_semantic_state_digest_anchor = initial_semantic_state_digest_anchor;
    plan
}

/// Test-only smaller `Params` profile.
///
/// Reuses the production Goldilocks ring + decomposition constants
/// (Q, ETA, D, B_BASE, K_RHO, T, EXTENSION_DEGREE) so every algebraic
/// identity in Π_RLC / Π_DEC holds bit-for-bit. Only the
/// commitment-width `kappa`, constraint count `m`, and security
/// parameter `lambda` are shrunk so the lifecycle fits under the
/// 5-minute test cap.
pub fn tiny_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 4,
        /* m      */ 1u64 << 16,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 60,
    )
    .expect("tiny NeoParams must satisfy the Π_RLC guard");
    Params::test_only_from_neo_params(inner)
}

/// Plan with the empirically-discovered fixed-point CE shape under
/// [`tiny_params`]. These constants come from running the lifecycle
/// once with a stub plan and reading the actual post-fold parent
/// shape; the recursive-compile path then converges in one iteration
/// because the plan matches the parent.
///
/// If `tiny_params` ever changes, the recursive-compile step will
/// fail with `PostParentShapeMismatch` and surface the new shape in
/// the error message — update these constants from that output.
pub fn make_tiny_lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    // c_data_entries = kappa * D = 4 * 54 = 216 under tiny_params.
    const TINY_C_DATA_ENTRIES: usize = 216;
    // child_count = K_RHO = 14 (matches production; not params-dependent).
    const TINY_CHILD_COUNT: u64 = 14;
    // r_len tracks the row domain, while s_col_len tracks the column
    // domain under the larger (216-entry) NIFS payload region. These
    // are the converged values after one iteration of the probe.
    const TINY_R_LEN: usize = 12;
    const TINY_S_COL_LEN: usize = 18;

    let limbs = app_private_bits_for(m) + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries: TINY_C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: TINY_R_LEN,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len: TINY_S_COL_LEN,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths: Vec::new(),
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_pair_count: 0,
        projection_identity_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: TINY_C_DATA_ENTRIES,
            child_count: TINY_CHILD_COUNT,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| boundary_start + i * POSEIDON2_GOLDILOCKS_BITS);
    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: (0..m_in).collect(),
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}
