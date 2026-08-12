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

use crate::adapters::wasmtime::WasmProgramTables;
use crate::batch::{self, BatchError};
use crate::comm_chain::CommChainState;
use crate::ir::{
    WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmHostEventState, WasmOutputState, WasmStepState,
};
use crate::layout::Column;
use crate::layout::{
    COL_CALL_STACK_DEPTH_BEFORE, COL_COMM_CHAIN_BEFORE, COL_EVBUF_BEFORE, COL_GRAMMAR_ARGS_BASE_BEFORE,
    COL_GRAMMAR_EVIDX_BEFORE, COL_GRAMMAR_EVREM_BEFORE, COL_GRAMMAR_SLOT_CURSOR_BEFORE, COL_HALTED_BEFORE,
    COL_HOST_CALLEE_FREF_BEFORE, COL_LOCALS_FBP_BEFORE, COL_MAX_MEMORY_PAGES_BEFORE, COL_MEMORY_PAGES_BEFORE,
    COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_PARAM_INIT_ACTIVE_BEFORE,
    COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_BEFORE, COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_BEFORE,
    COL_PERM_STATE_BEFORE, COL_SP_BEFORE, COL_STACK_FRAME_BASE_BEFORE, COL_TAIL_CALL_PENDING_BEFORE,
    COL_TRAPPED_BEFORE, COL_TURN_EXPORT_FREF_BEFORE,
};
use crate::lookup_circuit::{extend_relation, LookupCircuitError};
use crate::relation_layout::build_wasm_relation_layout;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, build_semantic_state_preimage_fields, AccumulatorPlanOptions,
    RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::f_prime::structure::FPrimeStructure;
use neo_fold_clean::frontends::f_prime::NifsCeClaimShape;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, build_r1cs_f_prime_structure, R1csFPrimePreprocessing, SparseR1cs,
};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use p3_field::PrimeCharacteristicRing;

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
    #[error(transparent)]
    Lookup(#[from] LookupCircuitError),
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

pub(crate) struct WasmNebulaCanonicalShape {
    pub(crate) sparse_r1cs: SparseR1cs,
    pub(crate) plan: RecursiveStepImagePlan,
    pub(crate) lookup_auxiliary_columns_per_instruction: usize,
    pub(crate) lookup_auxiliary_columns_total: usize,
    pub(crate) single_step_columns: usize,
}

pub fn canonical_wasm_f_prime_shape_batched_with_initial_state_digest(
    batch_size: usize,
    initial_semantic_state_digest: [u8; 32],
) -> Result<WasmCanonicalFPrimeShape, WasmPreprocessError> {
    let batched = batch::build_batched_wasm_ccs(batch_size)?;
    let mut sparse_r1cs = batched.sparse_r1cs;
    sparse_r1cs.m_in = 1;
    let (plan, structure) = wasm_recursive_plan_and_structure(
        &sparse_r1cs,
        &batched.widths,
        batched.batch_size,
        sparse_r1cs.m_in,
        initial_semantic_state_digest,
    );
    Ok(WasmCanonicalFPrimeShape {
        sparse_r1cs,
        plan,
        structure,
    })
}

pub(crate) fn canonical_wasm_nebula_shape_batched_with_initial_state_digest(
    batch_size: usize,
    initial_semantic_state_digest: [u8; 32],
) -> Result<WasmNebulaCanonicalShape, WasmPreprocessError> {
    let mut single = batch::build_batched_wasm_ccs(1)?;
    single.sparse_r1cs.m_in = 1;
    let compact = extend_relation(&single.sparse_r1cs, single.widths)?;
    let single_step_columns = compact.relation.m;
    let lookup_auxiliary_columns_per_instruction = compact.auxiliary_column_count;
    let batched = batch::batch_wasm_relation(&compact.relation, &compact.widths, batch_size)?;
    let (plan, _) = wasm_recursive_plan_and_structure(
        &batched.sparse_r1cs,
        &batched.widths,
        batch_size,
        batched.sparse_r1cs.m_in,
        initial_semantic_state_digest,
    );
    Ok(WasmNebulaCanonicalShape {
        sparse_r1cs: batched.sparse_r1cs,
        plan,
        lookup_auxiliary_columns_per_instruction,
        lookup_auxiliary_columns_total: lookup_auxiliary_columns_per_instruction * batch_size,
        single_step_columns,
    })
}

/// Build preprocessing with cross-batch VM-state continuity enabled.
///
/// The carried columns are derived from
/// `WasmRelationLayout::auxiliary.ivc_state_links`: the first block's
/// `*_before` columns are hashed as semantic input, and the last block's
/// `*_after` columns are hashed as semantic output. `initial_state_digest`
/// is verifier-owned: callers must derive or otherwise agree on it from
/// authoritative initial VM state, not from prover-supplied proof material.
pub fn preprocess_seeded_batched(
    batch_size: usize,
    initial_state_digest: [u8; 32],
) -> Result<R1csFPrimePreprocessing, WasmPreprocessError> {
    let WasmCanonicalFPrimeShape {
        sparse_r1cs,
        plan,
        structure: _,
    } = canonical_wasm_f_prime_shape_batched_with_initial_state_digest(batch_size, initial_state_digest)?;
    let params = wasm_tiny_params();
    Ok(r1cs_f_prime::preprocess_sparse_seeded_with_params(
        &sparse_r1cs,
        &plan,
        Params::test_only_from_neo_params(params),
        WASM_AJTAI_SEED,
    )?)
}

/// Top-level VM state before executing an exported wasm function of an
/// import-free program: [`host_event_top_level_initial_state`] specialized to
/// the canonical import-free bindings (empty boundary template for the
/// invoked export, zero commitment chain).
///
/// The entry PC is an explicit verifier input: callers should resolve it from
/// the export they intend to prove.
pub fn top_level_initial_state(tables: &WasmProgramTables, entry_pc: u64) -> WasmStepState {
    let export_fref = export_fref_for_entry_pc(tables, entry_pc);
    host_event_top_level_initial_state(
        tables,
        entry_pc,
        &crate::host_event_bindings::HostEventBindings::import_free(export_fref),
        export_fref,
        CommChainState::default(),
    )
    .expect("canonical import-free bindings contain the selected export")
}

/// The function ref whose body starts at `entry_pc`; the verifier-side
/// counterpart of the normalizer reading the entered export off the trace.
pub(crate) fn export_fref_for_entry_pc(tables: &WasmProgramTables, entry_pc: u64) -> u32 {
    tables
        .function_entries
        .iter()
        .find(|&&(_, pc)| pc == entry_pc)
        .map(|&(fref, _)| u32::try_from(fref).expect("function refs fit in u32"))
        .unwrap_or_else(|| panic!("entry pc {entry_pc} is not a function entry"))
}

/// Hash a carried VM state into the IVC semantic-state digest: the
/// verifier-owned initial anchor expected by [`preprocess_seeded_batched`],
/// and the final-state claim checked by [`crate::verify`].
///
/// `halted` is carried explicitly, so the terminal claim cannot be changed
/// independently of the folded semantic-state digest.
pub fn semantic_state_digest(state: WasmStepState) -> [u8; 32] {
    let layout = build_wasm_relation_layout();
    let fields = layout
        .auxiliary
        .ivc_state_links
        .iter()
        .flat_map(|link| link.column_pairs.iter())
        .map(|pair| carried_state_field(state, pair.next_before))
        .collect::<Vec<_>>();
    digest_fields_as_digest32(encode_poseidon_trace(&build_semantic_state_preimage_fields(&fields)).digest_native)
}

/// Host-event initial state: seeds the commitment chain and loads the invoked
/// export's entry schedule. Event values remain
/// bound by the final commitment rather than this per-program anchor.
pub fn host_event_top_level_initial_state(
    tables: &WasmProgramTables,
    entry_pc: u64,
    bindings: &crate::host_event_bindings::HostEventBindings,
    export_fref: u32,
    initial_comm_chain: CommChainState,
) -> Result<WasmStepState, WasmBuildError> {
    let entry_fref = tables
        .function_entries
        .iter()
        .find(|&&(_, pc)| pc == entry_pc)
        .map(|&(fref, _)| u32::try_from(fref).expect("function refs fit in u32"))
        .ok_or_else(|| WasmBuildError::Trace(format!("entry pc {entry_pc} is not a function entry")))?;
    if entry_fref != export_fref {
        return Err(WasmBuildError::Trace(format!(
            "export fref {export_fref} enters at a different pc than selected entry pc {entry_pc} (which belongs to fref {entry_fref})"
        )));
    }
    let template = bindings.exports.get(&export_fref).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "host-event bindings have no export template for selected export fref {export_fref}"
        ))
    })?;
    let mut state = WasmStepState {
        pc: entry_pc,
        sp: 0,
        stack_frame_base: 0,
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: tables.initial_memory_pages,
        max_memory_pages: tables.max_memory_pages,
        locals_fbp: 0,
        halted: false,
        trapped: false,
        param_init: WasmCountdownState::ZERO,
        tail_call_pending: false,
        host_callee_fref: 0,
        comm_chain: initial_comm_chain.canonical_u64(),
        event_absorb: WasmEventAbsorbState::ZERO,
        host_events: WasmHostEventState::ZERO,
    };
    state.host_callee_fref = export_fref;
    state.host_events.turn_export_fref = export_fref;
    state.host_events.events_remaining = template.entry.len() as u32;
    Ok(state)
}

/// Convenience wrapper for the common top-level export-entry boundary.
pub fn top_level_initial_state_digest(tables: &WasmProgramTables, entry_pc: u64) -> [u8; 32] {
    semantic_state_digest(top_level_initial_state(tables, entry_pc))
}

/// [`top_level_initial_state_digest`] for an event-bound program.
pub fn host_event_top_level_initial_state_digest(
    tables: &WasmProgramTables,
    entry_pc: u64,
    bindings: &crate::host_event_bindings::HostEventBindings,
    export_fref: u32,
    initial_comm_chain: CommChainState,
) -> Result<[u8; 32], WasmBuildError> {
    Ok(semantic_state_digest(host_event_top_level_initial_state(
        tables,
        entry_pc,
        bindings,
        export_fref,
        initial_comm_chain,
    )?))
}

fn carried_state_field(state: WasmStepState, column: Column) -> F {
    if let Some(limb) = COL_COMM_CHAIN_BEFORE
        .iter()
        .position(|&candidate| candidate == column.0)
    {
        return F::from_u64(state.comm_chain[limb]);
    }
    if let Some(word) = COL_EVBUF_BEFORE
        .iter()
        .position(|&candidate| candidate == column.0)
    {
        return F::from_u64(state.event_absorb.evbuf[word]);
    }
    if let Some(lane) = COL_PERM_STATE_BEFORE
        .iter()
        .position(|&candidate| candidate == column.0)
    {
        return F::from_u64(state.event_absorb.perm_state[lane]);
    }

    match column.0 {
        COL_PC_BEFORE => F::from_u64(state.pc),
        COL_SP_BEFORE => F::from_u64(state.sp),
        COL_STACK_FRAME_BASE_BEFORE => F::from_u64(state.stack_frame_base),
        COL_HALTED_BEFORE => bool_field(state.halted),
        COL_OUTPUT_ENABLED_BEFORE => bool_field(state.output.enabled),
        COL_OUTPUT_VALUE_LO_BEFORE => F::from_u64(u64::from(state.output.value_lo)),
        COL_OUTPUT_VALUE_HI_BEFORE => F::from_u64(u64::from(state.output.value_hi)),
        COL_CALL_STACK_DEPTH_BEFORE => F::from_u64(state.call_stack_depth),
        COL_MEMORY_PAGES_BEFORE => F::from_u64(u64::from(state.memory_pages.unwrap_or(0))),
        COL_MAX_MEMORY_PAGES_BEFORE => F::from_u64(u64::from(state.max_memory_pages.unwrap_or(0))),
        COL_LOCALS_FBP_BEFORE => F::from_u64(state.locals_fbp),
        COL_PARAM_INIT_ACTIVE_BEFORE => bool_field(state.param_init.active),
        COL_PARAM_INIT_REMAINING_BEFORE => F::from_u64(u64::from(state.param_init.remaining)),
        COL_TAIL_CALL_PENDING_BEFORE => bool_field(state.tail_call_pending),
        COL_HOST_CALLEE_FREF_BEFORE => F::from_u64(u64::from(state.host_callee_fref)),
        COL_TURN_EXPORT_FREF_BEFORE => F::from_u64(u64::from(state.host_events.turn_export_fref)),
        COL_GRAMMAR_EVREM_BEFORE => F::from_u64(u64::from(state.host_events.events_remaining)),
        COL_GRAMMAR_EVIDX_BEFORE => F::from_u64(u64::from(state.host_events.event_index)),
        COL_GRAMMAR_ARGS_BASE_BEFORE => F::from_u64(state.host_events.args_base),
        COL_GRAMMAR_SLOT_CURSOR_BEFORE => F::from_u64(u64::from(state.host_events.slot_cursor)),
        COL_PERM_PENDING_BEFORE => bool_field(state.event_absorb.perm_pending),
        COL_PERM_ROUND_BEFORE => F::from_u64(u64::from(state.event_absorb.perm_round)),
        COL_TRAPPED_BEFORE => bool_field(state.trapped),
        other => panic!("unsupported initial semantic-state column {other}"),
    }
}

fn bool_field(value: bool) -> F {
    if value {
        F::ONE
    } else {
        F::ZERO
    }
}

/// Test-only `NeoParams` profile, mirroring `sha256_tiny_neo_params`.
/// Production Goldilocks ring (Q, ETA, D, B_BASE, K_RHO, T) is preserved;
/// only `kappa`, `m`, `lambda` are shrunk so the lifecycle fits under the
/// 5-minute test cap. Π_RLC / Π_DEC algebraic identities hold bit-for-bit;
/// only the Ajtai-SIS security parameter is reduced.
pub(crate) fn wasm_tiny_params() -> NeoParams {
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
pub(crate) fn wasm_recursive_plan_and_structure(
    sparse_r1cs: &SparseR1cs,
    app_private_var_widths: &[usize],
    batch_size: usize,
    m_in: usize,
    initial_semantic_state_digest: [u8; 32],
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
    assert_eq!(
        sparse_r1cs.m % batch_size,
        0,
        "batched R1CS width must be a multiple of batch_size"
    );
    let single_width = sparse_r1cs.m / batch_size;
    let semantic_state_indices = wasm_batch_semantic_state_indices(batch_size, single_width);

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
            projection_batches: Vec::new(),
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
        let (semantic_state_in_var_indices, semantic_state_out_var_indices) = semantic_state_indices.clone();
        plan.state_x_out = Some(StateXOutPlanOptions {
            pc: 1,
            public_x_out_lane_bit_starts,
            app_public_input_var_indices: (0..m_in).collect(),
            app_public_input_bit_var_indices: Vec::new(),
            semantic_state_in_var_indices,
            semantic_state_out_var_indices,
            initial_semantic_state_digest_anchor: Some(initial_semantic_state_digest),
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

pub(crate) fn wasm_batch_semantic_state_indices(batch_size: usize, single_width: usize) -> (Vec<usize>, Vec<usize>) {
    assert!(batch_size >= 1, "batch_size must be at least 1");
    let layout = build_wasm_relation_layout();
    let mut input = Vec::new();
    let mut output = Vec::new();
    let last_block_offset = (batch_size - 1) * single_width;
    for pair in layout
        .auxiliary
        .ivc_state_links
        .iter()
        .flat_map(|link| link.column_pairs.iter())
    {
        input.push(pair.next_before.0);
        output.push(last_block_offset + pair.prev_after.0);
    }
    (input, output)
}
