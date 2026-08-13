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
//! Recursive image dimensions are derived to a fixed point from the compiled
//! relation and selected protocol parameters.

use crate::adapters::wasmtime::WasmProgramTables;
use crate::batch::{self, BatchError};
use crate::comm_chain::CommChainState;
use crate::ir::{WasmCountdownState, WasmEventAbsorbState, WasmGrammarState, WasmOutputState, WasmStepState};
use crate::layout::Column;
use crate::layout::{
    COL_CALL_STACK_DEPTH_BEFORE, COL_COMM_CHAIN0_BEFORE, COL_COMM_CHAIN1_BEFORE, COL_COMM_CHAIN2_BEFORE,
    COL_COMM_CHAIN3_BEFORE, COL_EVBUF0_BEFORE, COL_EVBUF1_BEFORE, COL_EVBUF2_BEFORE, COL_EVBUF3_BEFORE,
    COL_EVBUF4_BEFORE, COL_EVBUF5_BEFORE, COL_EVBUF6_BEFORE, COL_EVBUF7_BEFORE, COL_EVBUF_SLOT0_BEFORE,
    COL_EVBUF_SLOT1_BEFORE, COL_EVBUF_SLOT2_BEFORE, COL_EVBUF_SLOT3_BEFORE, COL_GRAMMAR_ARGS_BASE_BEFORE,
    COL_GRAMMAR_EVIDX_BEFORE, COL_GRAMMAR_EVREM_BEFORE, COL_GRAMMAR_MODE_BEFORE, COL_GRAMMAR_SLOT_CURSOR_BEFORE,
    COL_HALTED_BEFORE, COL_HOST_ARGS_ACTIVE_BEFORE, COL_HOST_ARGS_REMAINING_BEFORE, COL_HOST_CALLEE_FREF_BEFORE,
    COL_HOST_RESULT_PENDING_BEFORE, COL_LOCALS_FBP_BEFORE, COL_MAX_MEMORY_PAGES_BEFORE, COL_MEMORY_PAGES_BEFORE,
    COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_PARAM_INIT_ACTIVE_BEFORE,
    COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_BEFORE, COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_BEFORE,
    COL_PERM_STATE0_BEFORE, COL_PERM_STATE10_BEFORE, COL_PERM_STATE11_BEFORE, COL_PERM_STATE1_BEFORE,
    COL_PERM_STATE2_BEFORE, COL_PERM_STATE3_BEFORE, COL_PERM_STATE4_BEFORE, COL_PERM_STATE5_BEFORE,
    COL_PERM_STATE6_BEFORE, COL_PERM_STATE7_BEFORE, COL_PERM_STATE8_BEFORE, COL_PERM_STATE9_BEFORE, COL_SP_BEFORE,
    COL_STACK_FRAME_BASE_BEFORE, COL_TAIL_CALL_PENDING_BEFORE, COL_TRAPPED_BEFORE, COL_TURN_EXPORT_FREF_BEFORE,
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
use neo_fold_clean::frontends::r1cs_f_prime::{build_r1cs_f_prime_structure, SparseR1cs};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[derive(Debug, thiserror::Error)]
pub enum WasmPreprocessError {
    #[error(transparent)]
    Frontend(#[from] neo_fold_clean::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    Batch(#[from] BatchError),
    #[error(transparent)]
    Lookup(#[from] LookupCircuitError),
}

pub(crate) struct WasmNebulaCanonicalShape {
    pub(crate) sparse_r1cs: SparseR1cs,
    pub(crate) plan: RecursiveStepImagePlan,
    pub(crate) lookup_auxiliary_columns_per_instruction: usize,
    pub(crate) lookup_auxiliary_columns_total: usize,
    pub(crate) single_step_columns: usize,
}

pub(crate) fn canonical_wasm_nebula_shape_batched_with_initial_state_digest(
    params: &Params,
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
        params,
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

/// Top-level VM state before executing an exported wasm function.
///
/// The entry PC is an explicit verifier claim: callers should resolve it from
/// the export they intend to prove. The remaining state is the canonical empty
/// top-level call boundary plus the module's static initial memory page count.
pub fn top_level_initial_state(tables: &WasmProgramTables, entry_pc: u64) -> WasmStepState {
    WasmStepState {
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
        host_args: WasmCountdownState::ZERO,
        host_result_pending: false,
        host_callee_fref: 0,
        comm_chain: [0; 4],
        event_absorb: WasmEventAbsorbState::ZERO,
        grammar_mode: false,
        grammar: WasmGrammarState::ZERO,
    }
}

/// Hash a carried VM state into the IVC semantic-state digest: the
/// verifier-owned initial anchor used by Nebula preprocessing and the
/// final-state claim checked by [`crate::verify`].
///
/// `halted` is carried explicitly, so the terminal claim cannot be changed
/// independently of the folded semantic-state digest.
pub fn semantic_state_digest(state: WasmStepState) -> [u8; 32] {
    let layout = build_wasm_relation_layout();
    let fields = core::iter::once(F::ONE)
        .chain(
            layout
                .auxiliary
                .ivc_state_links
                .iter()
                .flat_map(|link| link.column_pairs.iter())
                .map(|pair| carried_state_field(state, pair.next_before)),
        )
        .collect::<Vec<_>>();
    digest_fields_as_digest32(encode_poseidon_trace(&build_semantic_state_preimage_fields(&fields)).digest_native)
}

/// Grammar-mode initial state: enables the grammar, seeds the commitment
/// chain, and loads the invoked export's entry schedule. Event values remain
/// bound by the final commitment rather than this per-program anchor.
pub fn grammar_top_level_initial_state(
    tables: &WasmProgramTables,
    entry_pc: u64,
    grammar: &crate::event_grammar::HostEventGrammar,
    export_fref: u32,
    initial_comm_chain: CommChainState,
) -> WasmStepState {
    let mut state = top_level_initial_state(tables, entry_pc);
    state.grammar_mode = true;
    state.comm_chain = initial_comm_chain.canonical_u64();
    if let Some(template) = grammar.exports.get(&export_fref) {
        state.host_callee_fref = export_fref;
        state.grammar.turn_export_fref = export_fref;
        state.grammar.events_remaining = template.entry.len() as u32;
    }
    state
}

/// Convenience wrapper for the common top-level export-entry boundary.
pub fn top_level_initial_state_digest(tables: &WasmProgramTables, entry_pc: u64) -> [u8; 32] {
    semantic_state_digest(top_level_initial_state(tables, entry_pc))
}

/// [`top_level_initial_state_digest`] for a grammar-mode program.
pub fn grammar_top_level_initial_state_digest(
    tables: &WasmProgramTables,
    entry_pc: u64,
    grammar: &crate::event_grammar::HostEventGrammar,
    export_fref: u32,
    initial_comm_chain: CommChainState,
) -> [u8; 32] {
    semantic_state_digest(grammar_top_level_initial_state(
        tables,
        entry_pc,
        grammar,
        export_fref,
        initial_comm_chain,
    ))
}

fn carried_state_field(state: WasmStepState, column: Column) -> F {
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
        COL_HOST_ARGS_ACTIVE_BEFORE => bool_field(state.host_args.active),
        COL_HOST_ARGS_REMAINING_BEFORE => F::from_u64(u64::from(state.host_args.remaining)),
        COL_HOST_RESULT_PENDING_BEFORE => bool_field(state.host_result_pending),
        COL_HOST_CALLEE_FREF_BEFORE => F::from_u64(u64::from(state.host_callee_fref)),
        COL_TURN_EXPORT_FREF_BEFORE => F::from_u64(u64::from(state.grammar.turn_export_fref)),
        COL_COMM_CHAIN0_BEFORE => F::from_u64(state.comm_chain[0]),
        COL_COMM_CHAIN1_BEFORE => F::from_u64(state.comm_chain[1]),
        COL_COMM_CHAIN2_BEFORE => F::from_u64(state.comm_chain[2]),
        COL_COMM_CHAIN3_BEFORE => F::from_u64(state.comm_chain[3]),
        COL_EVBUF0_BEFORE => F::from_u64(state.event_absorb.evbuf[0]),
        COL_EVBUF1_BEFORE => F::from_u64(state.event_absorb.evbuf[1]),
        COL_EVBUF2_BEFORE => F::from_u64(state.event_absorb.evbuf[2]),
        COL_EVBUF3_BEFORE => F::from_u64(state.event_absorb.evbuf[3]),
        COL_EVBUF4_BEFORE => F::from_u64(state.event_absorb.evbuf[4]),
        COL_EVBUF5_BEFORE => F::from_u64(state.event_absorb.evbuf[5]),
        COL_EVBUF6_BEFORE => F::from_u64(state.event_absorb.evbuf[6]),
        COL_EVBUF7_BEFORE => F::from_u64(state.event_absorb.evbuf[7]),
        COL_EVBUF_SLOT0_BEFORE => bool_field(state.event_absorb.evbuf_slot == 0),
        COL_EVBUF_SLOT1_BEFORE => bool_field(state.event_absorb.evbuf_slot == 1),
        COL_EVBUF_SLOT2_BEFORE => bool_field(state.event_absorb.evbuf_slot == 2),
        COL_EVBUF_SLOT3_BEFORE => bool_field(state.event_absorb.evbuf_slot == 3),
        COL_GRAMMAR_MODE_BEFORE => bool_field(state.grammar_mode),
        COL_GRAMMAR_EVREM_BEFORE => F::from_u64(u64::from(state.grammar.events_remaining)),
        COL_GRAMMAR_EVIDX_BEFORE => F::from_u64(u64::from(state.grammar.event_index)),
        COL_GRAMMAR_ARGS_BASE_BEFORE => F::from_u64(state.grammar.args_base),
        COL_GRAMMAR_SLOT_CURSOR_BEFORE => F::from_u64(u64::from(state.grammar.slot_cursor)),
        COL_PERM_PENDING_BEFORE => bool_field(state.event_absorb.perm_pending),
        COL_PERM_ROUND_BEFORE => F::from_u64(u64::from(state.event_absorb.perm_round)),
        COL_PERM_STATE0_BEFORE => F::from_u64(state.event_absorb.perm_state[0]),
        COL_PERM_STATE1_BEFORE => F::from_u64(state.event_absorb.perm_state[1]),
        COL_PERM_STATE2_BEFORE => F::from_u64(state.event_absorb.perm_state[2]),
        COL_PERM_STATE3_BEFORE => F::from_u64(state.event_absorb.perm_state[3]),
        COL_PERM_STATE4_BEFORE => F::from_u64(state.event_absorb.perm_state[4]),
        COL_PERM_STATE5_BEFORE => F::from_u64(state.event_absorb.perm_state[5]),
        COL_PERM_STATE6_BEFORE => F::from_u64(state.event_absorb.perm_state[6]),
        COL_PERM_STATE7_BEFORE => F::from_u64(state.event_absorb.perm_state[7]),
        COL_PERM_STATE8_BEFORE => F::from_u64(state.event_absorb.perm_state[8]),
        COL_PERM_STATE9_BEFORE => F::from_u64(state.event_absorb.perm_state[9]),
        COL_PERM_STATE10_BEFORE => F::from_u64(state.event_absorb.perm_state[10]),
        COL_PERM_STATE11_BEFORE => F::from_u64(state.event_absorb.perm_state[11]),
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

/// Build the recursive `RecursiveStepImagePlan` for the wasm R1CS shape
/// at the requested batch size, together with the F' structure that
/// matches it.
///
/// The post-parent CE claim has one evaluation point over the padded row
/// cube. Its length is `ceil_log2(max(structure.n, structure.m))`.
///
/// `r_len` feeds back into the F' structure, so the point length and the
/// structure are mutually constrained. Iterate until the value is stable.
pub(crate) fn wasm_recursive_plan_and_structure(
    params: &Params,
    sparse_r1cs: &SparseR1cs,
    app_private_var_widths: &[usize],
    batch_size: usize,
    m_in: usize,
    initial_semantic_state_digest: [u8; 32],
) -> (RecursiveStepImagePlan, FPrimeStructure) {
    let c_data_entries = params.kappa() as usize * neo_math::D;
    let child_count = u64::from(params.k_rho());

    let limbs = app_private_var_widths.iter().sum::<usize>() + 1;
    let mut r_len = 8usize;
    let mut y_ring_count = 1usize;
    assert_eq!(
        sparse_r1cs.m % batch_size,
        0,
        "batched R1CS width must be a multiple of batch_size"
    );
    let single_width = sparse_r1cs.m / batch_size;
    let semantic_state_indices = wasm_batch_semantic_state_indices(batch_size, single_width);

    let mut seen = Vec::new();
    loop {
        if seen.contains(&(r_len, y_ring_count)) {
            panic!(
                "wasm_recursive_plan_and_structure entered a shape cycle at r_len={r_len}, y_ring_count={y_ring_count}"
            );
        }
        seen.push((r_len, y_ring_count));
        let ce_shape = NifsCeClaimShape {
            c_data_entries,
            x_rows: 54,
            x_active_cols: 5,
            r_len,
            y_ring_inner_lens: vec![64; y_ring_count],
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
                c_data_entries,
                child_count,
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
        let required_r = ceil_log2(structure.ccs.n.max(structure.ccs.m).max(2));
        let required_y_ring_count = structure.ccs.t() + 1;
        if required_r == r_len && required_y_ring_count == y_ring_count {
            return (plan, structure);
        }
        r_len = required_r;
        y_ring_count = required_y_ring_count;
    }
}

fn ceil_log2(n: usize) -> usize {
    assert!(n > 0, "ceil_log2 requires n >= 1");
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

pub(crate) fn wasm_batch_semantic_state_indices(batch_size: usize, single_width: usize) -> (Vec<usize>, Vec<usize>) {
    assert!(batch_size >= 1, "batch_size must be at least 1");
    let layout = build_wasm_relation_layout();
    // `z[0]` is the public constant-one lane. Keep it in both carried
    // preimages so the explicit semantic hashes bind the complete public
    // tuple and use the same domain on both sides of a transition.
    let mut input = vec![0];
    let mut output = vec![0];
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
