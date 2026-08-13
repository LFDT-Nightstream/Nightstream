//! Small application and plan fixtures for the authoritative R1CS IVC tests.

#![allow(dead_code)]

use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::{RecursiveStepImagePlan, StateXOutPlanOptions};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use p3_field::PrimeCharacteristicRing;

pub fn one_product_r1cs() -> R1cs {
    let mut a = Mat::zero(1, neo_math::D, F::ZERO);
    let mut b = Mat::zero(1, neo_math::D, F::ZERO);
    let mut c = Mat::zero(1, neo_math::D, F::ZERO);
    a[(0, 1)] = F::ONE;
    b[(0, 2)] = F::ONE;
    c[(0, 0)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

pub fn assignment_one_product(a: u64, b: u64) -> Vec<F> {
    let mut assignment = vec![F::ZERO; neo_math::D];
    assignment[0] = F::from_u64(a * b);
    assignment[1] = F::from_u64(a);
    assignment[2] = F::from_u64(b);
    assignment
}

pub fn make_tiny_lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    plan(m, m_in, Vec::new(), Vec::new(), None)
}

pub fn make_tiny_stateful_lifecycle_plan_with_anchor(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
    initial_semantic_state_digest_anchor: Option<[u8; 32]>,
) -> RecursiveStepImagePlan {
    plan(
        m,
        m_in,
        semantic_state_in_var_indices,
        semantic_state_out_var_indices,
        initial_semantic_state_digest_anchor,
    )
}

fn plan(
    m: usize,
    m_in: usize,
    semantic_state_in_var_indices: Vec<usize>,
    semantic_state_out_var_indices: Vec<usize>,
    initial_semantic_state_digest_anchor: Option<[u8; 32]>,
) -> RecursiveStepImagePlan {
    RecursiveStepImagePlan {
        limbs: m * POSEIDON2_GOLDILOCKS_BITS + 1,
        app_private_var_widths: Vec::new(),
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_batches: Vec::new(),
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: Vec::new(),
        accumulator: None,
        state_x_out: Some(StateXOutPlanOptions {
            pc: 1,
            public_x_out_lane_bit_starts: std::array::from_fn(|lane| lane * POSEIDON2_GOLDILOCKS_BITS),
            app_public_input_var_indices: (0..m_in).collect(),
            app_public_input_bit_var_indices: Vec::new(),
            semantic_state_in_var_indices,
            semantic_state_out_var_indices,
            initial_semantic_state_digest_anchor,
        }),
    }
}

pub fn tiny_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        4,
        1u64 << 24,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        60,
    )
    .expect("test parameters satisfy the reduction guard");
    Params::test_only_from_neo_params(inner)
}

/// Smallest Goldilocks algebraic profile that can run the selected IVC path.
///
/// With `b = 2` and the minimum nonzero `T = 1`, `k_rho = 2` is the first
/// exponent for which `(k_rho + 1) T (b - 1) < b^k_rho`. Rank and lambda use
/// their constructor minima. The production ring and extension stay fixed.
pub fn minimal_ivc_test_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        1,
        goldilocks_paper_b2::M,
        goldilocks_paper_b2::B_BASE,
        2,
        1,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("minimal IVC test parameters satisfy the exact RLC guard");
    Params::test_only_from_neo_params(inner)
}
