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
use neo_fold_clean::frontends::f_prime::structure::FPrimeStructure;
use neo_fold_clean::frontends::f_prime::NifsCeClaimShape;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, build_r1cs_f_prime_structure, R1csFPrimePreprocessing, SparseR1cs,
};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};

use crate::batch::{self, BatchError};
use crate::ccs::WasmVmSpec;

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
    #[error(transparent)]
    Batch(#[from] BatchError),
}

/// Canonical structural inputs for the wasm R1CS-F' frontend.
///
/// This deliberately stops before lifecycle/Ajtai preprocessing. It is the
/// cheap verifier-side shape surface: the wasm R1CS, recursive image plan,
/// and resulting F' CCS structure.
pub struct WasmCanonicalFPrimeShape {
    pub sparse_r1cs: SparseR1cs,
    pub plan: RecursiveStepImagePlan,
    pub structure: FPrimeStructure,
}

/// Canonical wasm F' shape at `batch_size = 1`.
///
/// Thin wrapper over [`canonical_wasm_f_prime_shape_batched`] for callers
/// that only care about the single-step shape. Internally, single-step
/// and batched share the same construction path — at `batch_size = 1`
/// the batched matrices reduce to the single-step matrices with no
/// linking rows.
pub fn canonical_wasm_f_prime_shape(vm: &WasmVmSpec) -> Result<WasmCanonicalFPrimeShape, WasmPreprocessError> {
    canonical_wasm_f_prime_shape_batched(vm, 1)
}

/// Canonical wasm F' shape for a given batch size.
///
/// Block-diagonalises the wasm R1CS `batch_size` times and adds the
/// cross-step linking rows from
/// [`WasmLookupBindingLayout::cross_step_links`]. See
/// [`crate::batch`] for the construction details.
pub fn canonical_wasm_f_prime_shape_batched(
    _vm: &WasmVmSpec,
    batch_size: usize,
) -> Result<WasmCanonicalFPrimeShape, WasmPreprocessError> {
    let batched = batch::build_batched_wasm_ccs(batch_size)?;
    let (plan, structure) =
        wasm_recursive_plan_and_structure(&batched.sparse_r1cs, &batched.widths, batched.sparse_r1cs.m_in);
    Ok(WasmCanonicalFPrimeShape {
        sparse_r1cs: batched.sparse_r1cs,
        plan,
        structure,
    })
}

/// Build preprocessing for the wasm VM at `batch_size = 1`. Thin wrapper
/// over [`preprocess_seeded_batched`].
pub fn preprocess_seeded(vm: &WasmVmSpec) -> Result<R1csFPrimePreprocessing, WasmPreprocessError> {
    preprocess_seeded_batched(vm, 1)
}

/// Build preprocessing for the wasm VM at an arbitrary `batch_size`.
///
/// Tests can pick any `batch_size >= 1`; the resulting prep folds N
/// wasm steps per F'-shell fold, amortising the per-fold cost (Poseidon
/// traces, ring-action products, NIFS prover work) across the batch.
pub fn preprocess_seeded_batched(
    vm: &WasmVmSpec,
    batch_size: usize,
) -> Result<R1csFPrimePreprocessing, WasmPreprocessError> {
    let WasmCanonicalFPrimeShape {
        sparse_r1cs,
        plan,
        structure: _,
    } = canonical_wasm_f_prime_shape_batched(vm, batch_size)?;
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

/// Build the recursive `RecursiveStepImagePlan` for the wasm R1CS shape
/// at the requested batch size, together with the F' structure that
/// matches it.
///
/// The post-parent CE claim has `r ∈ K^{ell_n}` (row-domain sumcheck
/// challenge) and `s_col ∈ K^{ell_m}` (column-domain point for the NC
/// check), where
///
///   ell_n = ceil_log2(next_pow2(F' structure.n))
///   ell_m = ceil_log2(next_pow2(F' structure.m))
///
/// Downstream validation in `compiler.rs:181` reads the actual NIFS
/// proof's `r.len()` / `s_col.len()` and demands exact equality with
/// the canonical shape, so tracking them as separate lengths matters
/// even when they coincide for the current shape.
///
/// `r_len` / `s_col_len` feed back into the F' structure (each adds
/// `len · NIFS_K_LIMB_BITS` bits to the image), so the two lengths and
/// the structure they index are mutually constrained. Iterate to the
/// fixed point: seed both, build the structure, recompute the required
/// lengths, repeat until stable. The dependency is logarithmic in both
/// directions, so convergence is 1-2 iterations.
fn wasm_recursive_plan_and_structure(
    sparse_r1cs: &SparseR1cs,
    app_private_var_widths: &[usize],
    m_in: usize,
) -> (RecursiveStepImagePlan, FPrimeStructure) {
    // kappa * D for `wasm_tiny_params` = 2 * 54 = 108.
    const C_DATA_ENTRIES: usize = 108;
    // = K_RHO.
    const CHILD_COUNT: u64 = 14;
    // Safety bound: each sumcheck length contributes linearly to F'
    // structure rows, so log2(rows) grows by at most ~1 per +1 of either
    // length. Eight rounds is far more than needed; the bound just guards
    // against an unexpected non-monotone iteration.
    const MAX_ITERATIONS: usize = 8;

    let limbs = app_private_var_widths.iter().sum::<usize>() + 1;
    let mut r_len = 8usize;
    let mut s_col_len = 8usize;

    for _ in 0..MAX_ITERATIONS {
        let ce_shape = NifsCeClaimShape {
            c_data_entries: C_DATA_ENTRIES,
            x_rows: 54,
            x_active_cols: 5,
            r_len,
            y_ring_inner_lens: vec![64; 8],
            y_zcol_len: 64,
            s_col_len,
        };
        let probe_plan = RecursiveStepImagePlan {
            limbs,
            app_private_var_widths: app_private_var_widths.to_vec(),
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

        let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
        let (structure, _) = build_r1cs_f_prime_structure(layout, sparse_r1cs);
        let required_r = ceil_log2(structure.ccs.n.max(2));
        let required_s = ceil_log2(structure.ccs.m.max(2));
        if required_r == r_len && required_s == s_col_len {
            return (plan, structure);
        }
        r_len = required_r;
        s_col_len = required_s;
    }

    panic!(
        "wasm_recursive_plan_and_structure did not converge within {MAX_ITERATIONS} iterations \
         (last r_len = {r_len}, s_col_len = {s_col_len}); the dependency should be logarithmic, \
         so non-convergence indicates a deeper protocol mismatch"
    );
}

fn ceil_log2(n: usize) -> usize {
    assert!(n > 0, "ceil_log2 requires n >= 1");
    (usize::BITS - (n - 1).leading_zeros()) as usize
}
