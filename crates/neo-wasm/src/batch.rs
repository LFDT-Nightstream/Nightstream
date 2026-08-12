//! Batched wasm CCS construction: amortizes F'-shell fold cost across
//! `batch_size` consecutive wasm steps.
//!
//! The single-step wasm CCS proves *one* step satisfies the row constraints
//! but does not, on its own, enforce that consecutive steps form a coherent
//! execution — e.g., that `step[i].state_after.pc == step[i+1].state_before.pc`. The
//! cross-step links live in `WasmRelationLayout::auxiliary.ivc_state_links`
//! as metadata; this module compiles them into actual R1CS rows by:
//!
//! 1. **Block-diagonalising** the single-step matrices `A`, `B`, `C`: each
//!    of `batch_size` step blocks gets its own copy, with no cross-block
//!    entries on the diagonal portion of the matrix — except `COL_ONE`
//!    references, which all blocks share as the global `z[0]`.
//! 2. **Adding state-continuity link rows** between adjacent blocks: for
//!    each `(prev_after, next_before)` pair in the spec, an equality row
//!    across adjacent blocks.
//!
//! Witness shape: `m_batch = batch_size * m_single`. Step `s`'s columns
//! live at `z[s * m_single .. (s+1) * m_single]`. The first step's public
//! prefix remains the batched public prefix; later step copies are linked by
//! explicit continuity rows where needed.
//!
//! Padding: when a trace isn't a multiple of `batch_size`, we extend it
//! with synthetic `WasmAuxOpcode::Padding` rows. Each padding row has
//! `_after == _before` for every state column the cross-step links touch
//! (pc, sp, memory_pages, locals_fbp, param_init), enforced by the
//! `padding_active` selector in the wasm CCS. So a padding row is a true
//! fixed point: cross-step links between padding rows are trivially
//! satisfied, and the boundary `last_real -> first_padding` requires the
//! padding row's `_before` to equal the last real row's `_after` — which
//! [`padding_step_after`] handles by construction.
//!
//! Scope: this module emits only the equality rows *within* one batch.
//! Cross-batch continuity is carried by preprocessing built with
//! [`crate::preprocess::preprocess_seeded_batched`],
//! which uses spec-derived state columns and a verifier-owned initial
//! semantic-state digest.

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::frontends::direct_ccs::FrontendError;
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::ccs::WasmVmSpec;
use crate::ir::{WasmAuxOpcode, WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep};
use crate::isa::{opcode_info_from_code, WasmOpcode};
use crate::layout::COL_ONE;
use crate::relation_layout::build_wasm_relation_layout;
use crate::witness_builder::build_witness_vector;

/// Block-diagonal R1CS shape for a batch of `batch_size` consecutive wasm steps.
pub struct BatchedWasmCcs {
    pub batch_size: usize,
    pub sparse_r1cs: SparseR1cs,
    pub widths: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error("batch_size must be at least 1")]
    BatchSizeZero,
    #[error("wasm batch relation has {actual} width declarations for {expected} columns")]
    WidthCount { actual: usize, expected: usize },
    #[error("wasm batching requires ordinary R1CS matrices; compact seeded Phi81 blocks are unsupported")]
    CompactSeededMatrixUnsupported,
    #[error(transparent)]
    Frontend(#[from] FrontendError),
}

/// Build the batched wasm R1CS for the requested batch size.
///
/// Sources the single-step matrices from [`WasmVmSpec::default`] and the
/// cross-step link spec from [`build_wasm_relation_layout`]. Every
/// `(prev_after, next_before)` column pair in every spec link is emitted
/// as a linking row; the `flat_map` over `column_pairs` happens to also
/// be a no-op for any future link whose invariant can't be expressed as
/// column equalities (none today).
pub fn build_batched_wasm_ccs(batch_size: usize) -> Result<BatchedWasmCcs, BatchError> {
    let vm = WasmVmSpec::default();
    let core = vm.core_ccs_spec();
    let m_single = core.structure.m;
    assert_eq!(m_single, core.witness_width);
    let single = SparseR1cs::new(
        core.structure.matrices[0].clone(),
        core.structure.matrices[1].clone(),
        core.structure.matrices[2].clone(),
        core.structure.n,
        core.structure.m,
        core.m_in,
    )?;
    let widths = crate::witness_layout::range_checked_variable_widths();
    batch_wasm_relation(&single, &widths, batch_size)
}

/// Replicate one authoritative single-step WASM relation into an ordered
/// batch. This is also used after compact lookup closure, so every block owns
/// the same arithmetic, lookup, and range-check rows as the single-step path.
pub(crate) fn batch_wasm_relation(
    single: &SparseR1cs,
    single_widths: &[usize],
    batch_size: usize,
) -> Result<BatchedWasmCcs, BatchError> {
    if batch_size == 0 {
        return Err(BatchError::BatchSizeZero);
    }
    if single_widths.len() != single.m {
        return Err(BatchError::WidthCount {
            actual: single_widths.len(),
            expected: single.m,
        });
    }

    let m_single = single.m;
    let n_single = single.n;

    let layout = build_wasm_relation_layout();
    let link_pairs: Vec<(usize, usize)> = layout
        .auxiliary
        .ivc_state_links
        .iter()
        .flat_map(|link| link.column_pairs.iter())
        .map(|pair| (pair.prev_after.0, pair.next_before.0))
        .collect();

    // Per boundary: one row per state-continuity pair.
    let n_link_per_boundary = link_pairs.len();
    let n_link = batch_size.saturating_sub(1) * n_link_per_boundary;

    let m_batch = batch_size * m_single;
    let n_batch = batch_size * n_single + n_link;

    let single_a = matrix_triplets(&single.a)?;
    let single_b = matrix_triplets(&single.b)?;
    let single_c = matrix_triplets(&single.c)?;

    let mut a_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_a.len());
    let mut b_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_b.len());
    let mut c_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_c.len());

    // Block-diagonal replication. Sharing `COL_ONE` as global `z[0]` keeps
    // constant-anchored rows in the shape expected by F' width inference.
    // Block-local `COL_ONE` slots after block 0 remain unreferenced.
    for step in 0..batch_size {
        let row_offset = step * n_single;
        let col_offset = step * m_single;
        let map_col = |c: usize| if c == COL_ONE { COL_ONE } else { col_offset + c };
        for &(r, c, v) in &single_a {
            a_triplets.push((row_offset + r, map_col(c), v));
        }
        for &(r, c, v) in &single_b {
            b_triplets.push((row_offset + r, map_col(c), v));
        }
        for &(r, c, v) in &single_c {
            c_triplets.push((row_offset + r, map_col(c), v));
        }
    }

    // Linking rows. Each row enforces `(A_row · z) * (B_row · z) = C_row · z`
    // with `A_row = z[lhs] - z[rhs]`, `B_row = z[COL_ONE]`, `C_row = 0` —
    // i.e. `z[lhs] - z[rhs] = 0`.
    let mut link_row = batch_size * n_single;
    for boundary in 0..batch_size.saturating_sub(1) {
        let curr_offset = boundary * m_single;
        let next_offset = (boundary + 1) * m_single;

        // Spec-driven state continuity links.
        for &(prev_col, next_col) in &link_pairs {
            a_triplets.push((link_row, curr_offset + prev_col, F::ONE));
            a_triplets.push((link_row, next_offset + next_col, -F::ONE));
            b_triplets.push((link_row, COL_ONE, F::ONE));
            link_row += 1;
        }
    }
    debug_assert_eq!(link_row, n_batch, "linking row count mismatch");

    let a = CcsMatrix::Csc(CscMat::from_triplets(a_triplets, n_batch, m_batch));
    let b = CcsMatrix::Csc(CscMat::from_triplets(b_triplets, n_batch, m_batch));
    let c = CcsMatrix::Csc(CscMat::from_triplets(c_triplets, n_batch, m_batch));

    let sparse_r1cs = SparseR1cs::new(a, b, c, n_batch, m_batch, single.m_in)?;

    let mut all_widths = Vec::with_capacity(m_batch);
    for _ in 0..batch_size {
        all_widths.extend(single_widths.iter().copied());
    }

    Ok(BatchedWasmCcs {
        batch_size,
        sparse_r1cs,
        widths: all_widths,
    })
}

/// Build the witness vector for one batch.
///
/// Each batch covers `batch_size` consecutive step witnesses, concatenated
/// in trace order. If `traces[batch_idx * batch_size..]` is shorter than
/// `batch_size`, the tail is padded with synthetic state-preserving
/// padding rows (see [`padding_step_after`]).
pub fn build_batched_witness(traces: &[WasmVmStep], batch_size: usize, batch_idx: usize) -> Vec<F> {
    let single_width = crate::RANGE_CHECKED_WITNESS_WIDTH;
    assert!(batch_size >= 1, "batch_size must be at least 1");
    let start = batch_idx * batch_size;
    assert!(
        start < traces.len(),
        "batch_idx {batch_idx} out of range for trace len {}",
        traces.len()
    );

    let real_end = ((batch_idx + 1) * batch_size).min(traces.len());
    let real_witnesses: Vec<Vec<F>> = traces[start..real_end]
        .iter()
        .map(build_witness_vector)
        .collect();

    let mut witness = Vec::with_capacity(batch_size * single_width);
    for w in &real_witnesses {
        witness.extend_from_slice(w);
    }
    if real_witnesses.len() < batch_size {
        let last_real = &traces[real_end - 1];
        let mut padding = padding_step_after(last_real);
        let pad_count = batch_size - real_witnesses.len();
        for _ in 0..pad_count {
            witness.extend_from_slice(&build_witness_vector(&padding));
            // Each subsequent padding row starts from the previous one,
            // which is state-preserving — so `_after` is the same as the
            // first padding row's `_after`. Reuse `padding` directly.
            padding = padding_step_after(&padding);
        }
    }
    debug_assert_eq!(witness.len(), batch_size * single_width);
    witness
}

/// How many batches a trace of length `n` produces at `batch_size`,
/// counting the partial-final batch (padded by [`build_batched_witness`])
/// as one full batch.
pub fn batch_count(trace_len: usize, batch_size: usize) -> usize {
    assert!(batch_size >= 1, "batch_size must be at least 1");
    assert!(trace_len > 0, "trace must be non-empty");
    trace_len.div_ceil(batch_size)
}

/// Build a synthetic state-preserving `WasmVmStep` whose `_before`
/// matches `prev`'s `_after`. Used to pad a trace up to a multiple of
/// `batch_size` without breaking cross-step continuity.
///
/// The wasm CCS enforces state preservation for `padding_active = 1`
/// rows (see the `non-program row shape` and `padding row state
/// preservation` constraint groups in `ccs/call.rs`); the witness we
/// build here is just the values that satisfy those rows.
pub fn padding_step_after(prev: &WasmVmStep) -> WasmVmStep {
    let pages = prev.state_after.memory_pages;
    let max_pages = prev.state_after.max_memory_pages;
    let fbp = prev.state_after.locals_fbp;
    let pc = prev.state_after.pc;
    let sp = prev.state_after.sp;
    let stack_frame_base = prev.state_after.stack_frame_base;
    let call_stack_depth = prev.state_after.call_stack_depth;
    let param_init = prev.state_after.param_init;
    let tail_call_pending = prev.state_after.tail_call_pending;
    debug_assert!(
        !param_init.active,
        "padding inside a param-init aux sequence is unsupported"
    );
    debug_assert!(!tail_call_pending, "padding before a tail-enter aux row is unsupported");
    let host_callee_fref = prev.state_after.host_callee_fref;
    let comm_chain = prev.state_after.comm_chain;
    let event_absorb = prev.state_after.event_absorb;
    let host_events = prev.state_after.host_events;
    debug_assert!(
        !event_absorb.perm_pending && event_absorb.perm_round == 0,
        "padding inside a host-event perm group is unsupported"
    );
    WasmVmStep {
        cycle: prev.cycle + 1,
        row_kind: WasmRowKind::Aux(WasmAuxOpcode::Padding),
        state_before: WasmStepState {
            pc,
            sp,
            stack_frame_base,
            output: prev.state_after.output,
            call_stack_depth,
            memory_pages: pages,
            max_memory_pages: max_pages,
            locals_fbp: fbp,
            halted: prev.state_after.halted,
            trapped: prev.state_after.trapped,
            param_init,
            tail_call_pending,
            host_callee_fref,
            comm_chain,
            event_absorb,
            host_events,
        },
        state_after: WasmStepState {
            pc,
            sp,
            stack_frame_base,
            output: prev.state_after.output,
            call_stack_depth,
            memory_pages: pages,
            max_memory_pages: max_pages,
            locals_fbp: fbp,
            halted: prev.state_after.halted,
            trapped: prev.state_after.trapped,
            param_init,
            tail_call_pending,
            host_callee_fref,
            comm_chain,
            event_absorb,
            host_events,
        },
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        wide_values_enabled: false,
        // opcode_code = 0: no opcode selector fires on this row (the
        // selector one-hot demands `sum(selectors) = is_program_row`,
        // which is 0 for padding).
        opcode: WasmOpcode::Nop,
        info: opcode_info_from_code(0),
        stack_reads_override: Some(0),
        stack_writes_override: Some(0),
        output_captured: false,
        current_function_ref: prev.current_function_ref,
        current_function_num_locals: prev.current_function_num_locals,
        stack_read0: None,
        stack_read1: None,
        stack_read2: None,
        stack_write0: None,
        linear_memory: None,
        linear_memory_offset: 0,
        local_index: None,
        local_read_value: None,
        local_read_value_hi: None,
        local_write_value: None,
        local_write_value_hi: None,
        global_index: None,
        global_read_value: None,
        global_read_value_hi: None,
        global_write_value: None,
        global_write_value_hi: None,
        table_id: None,
        table_index: None,
        table_value: None,
        function_type_id: None,
        call_indirect_type_index: None,
        expected_type_id: None,
        table_size: None,
        function_ref: None,
        target_function_is_guest: false,
        call_param_count: None,
        call_result_count: None,
        call_stack_push: None,
        call_stack_pop: None,
        host_event_rom_slot: None,
        host_event_initial_schedule_count: None,
        host_event_exit_schedule_count: None,
    }
}

fn matrix_triplets(m: &CcsMatrix<F>) -> Result<Vec<(usize, usize, F)>, BatchError> {
    let triplets = match m {
        CcsMatrix::Identity { n } => (0..*n).map(|i| (i, i, F::ONE)).collect(),
        CcsMatrix::Csc(csc) => {
            let mut out = Vec::with_capacity(csc.vals.len());
            for c in 0..csc.ncols {
                for k in csc.column_range(c) {
                    out.push((csc.row_index(k), c, csc.vals[k]));
                }
            }
            out
        }
        CcsMatrix::CscWithSeededPhi81 { .. } => return Err(BatchError::CompactSeededMatrixUnsupported),
    };
    Ok(triplets)
}
