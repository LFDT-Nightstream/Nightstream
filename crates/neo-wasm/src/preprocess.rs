//! Build the wasm VM's `r1cs_f_prime` preprocessing.
//!
//! The wasm CCS is R1CS-derived (its tagged builder runs `r1cs_to_ccs(A, B, C)`
//! internally). Routing through `r1cs_f_prime` rather than the bare `direct_ccs`
//! frontend gives us the bit-decomposition shell for free: each R1CS variable
//! `z_j` is committed as its 64 LE bits, the F' structure adds bit-validity
//! rows + R1CS-product rows that algebraically reconstruct each variable as
//! `Σ 2^i · bit_i` and enforce `(A_i·z)(B_i·z) = (C_i·z)`. The committed
//! witness entries are 0/1, so `‖z‖_∞ < b = 2` holds soundly.
//!
//! The R1CS-F' plan constants below were tuned by running the recursive
//! compile and reading the `PostParentShapeMismatch` error — same fixed-point
//! discovery used by the SHA-256 R1CS-F' plan.

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::f_prime::NifsCeClaimShape;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csFPrimePreprocessing, SparseR1cs};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};

use crate::ccs::WasmVmSpec;
use crate::layout::{ColumnWidth, COLUMN_SPECS};

/// Test/demo Ajtai SRS seed. The Ajtai PP is shape-keyed in the global
/// registry, so any consistent value across prover + verifier in the same
/// test session is fine.
const WASM_AJTAI_SEED: u64 = 0xa55ec_a11ed_15ea;

#[derive(Debug, thiserror::Error)]
pub enum WasmPreprocessError {
    #[error(transparent)]
    Params(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Frontend(#[from] neo_fold_clean::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    R1csFPrime(#[from] neo_fold_clean::frontends::r1cs_f_prime::Error),
}

/// Build preprocessing for the wasm VM using a process-global Ajtai SRS
/// installed deterministically from the canonical wasm seed. Tests that
/// reuse the same `(D, cols)` shape share one PP.
pub fn preprocess_seeded(vm: &WasmVmSpec) -> Result<R1csFPrimePreprocessing, WasmPreprocessError> {
    let core = vm.core_ccs_spec();
    let sparse_r1cs = SparseR1cs::new(
        core.structure.matrices[0].clone(),
        core.structure.matrices[1].clone(),
        core.structure.matrices[2].clone(),
        core.structure.n,
        core.structure.m,
        core.m_in,
    )?;
    let plan = wasm_recursive_plan(core.structure.m, core.m_in);
    let params = wasm_tiny_params();
    Ok(r1cs_f_prime::preprocess_sparse_seeded_with_params(
        &sparse_r1cs,
        &plan,
        Params::test_only_from_neo_params(params),
        WASM_AJTAI_SEED,
    )?)
}

/// Test-only `NeoParams` profile, mirroring `sha256_tiny_neo_params`.
/// Production Goldilocks ring (Q, ETA, D, B_BASE, K_RHO, T) is preserved;
/// only `kappa`, `m`, `lambda` are shrunk so the lifecycle fits under the
/// 5-minute test cap. Π_RLC / Π_DEC algebraic identities hold bit-for-bit;
/// only the Ajtai-SIS security parameter is reduced.
fn wasm_tiny_params() -> NeoParams {
    NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 2,
        /* m      */ 1u64 << 15,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 40,
    )
    .expect("wasm tiny NeoParams must satisfy the Π_RLC guard")
}

/// `RecursiveStepImagePlan` for the wasm R1CS shape.
///
/// `m` here is the wasm R1CS variable count (= `WasmCoreCcs::structure.m`);
/// `m_in` is the public-input split (= `WasmCoreCcs::m_in`).
fn wasm_recursive_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    // kappa * D for `wasm_tiny_params` = 2 * 54 = 108.
    const C_DATA_ENTRIES: usize = 108;
    // = K_RHO.
    const CHILD_COUNT: u64 = 14;
    // Tuned by running the recursive compile and reading
    // `PostParentShapeMismatch`. ceil(log2(post-F' structure.m)) for the
    // wasm R1CS shape under the tiny params profile.
    const R_LEN: usize = 21;

    let app_private_var_widths = wasm_app_private_var_widths(m);
    let limbs = app_private_var_widths.iter().sum::<usize>() + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries: C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: R_LEN,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len: R_LEN,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths,
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
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
            c_data_entries: C_DATA_ENTRIES,
            child_count: CHILD_COUNT,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| probe_layout.boundary.offset + i * POSEIDON2_GOLDILOCKS_BITS);
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

fn wasm_app_private_var_widths(m: usize) -> Vec<usize> {
    assert_eq!(
        m,
        COLUMN_SPECS.len(),
        "wasm R1CS variable count must match COLUMN_SPECS"
    );
    COLUMN_SPECS
        .iter()
        .map(|spec| match spec.width {
            ColumnWidth::Boolean => 1,
            ColumnWidth::Byte => 8,
            ColumnWidth::U32 => 32,
            ColumnWidth::Field => POSEIDON2_GOLDILOCKS_BITS,
        })
        .collect()
}
