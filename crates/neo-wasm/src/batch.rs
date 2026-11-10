//! Batched wasm CCS construction: amortizes F'-shell fold cost across
//! `batch_size` consecutive wasm steps.
//!
//! The single-step wasm CCS proves *one* step satisfies the row constraints
//! but does not, on its own, enforce that consecutive steps form a coherent
//! execution — e.g., that `step[i].state_after.pc == step[i+1].state_before.pc`. The
//! cross-step links live in `WasmRelationLayout::auxiliary.continuity`
//! as metadata; this module compiles them into actual R1CS rows by:
//!
//! 1. **Block-diagonalising** the single-step matrices `A`, `B`, `C`: each
//!    of `batch_size` step blocks gets its own copy, with no cross-block
//!    entries on the diagonal portion of the matrix — except `COL_ONE`
//!    references, which all blocks share as the global `z[0]`.
//! 2. **Adding state-continuity link rows** between adjacent blocks: for
//!    each `(previous_step_column, next_step_column)` pair, an equality row
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
//! (pc, sp, memory_pages, locals_fbp, frame_move), enforced by the
//! `padding_active` selector in the wasm CCS. So a padding row is a true
//! fixed point: cross-step links between padding rows are trivially
//! satisfied, and the boundary `last_real -> first_padding` requires the
//! padding row's `_before` to equal the last real row's `_after` — which
//! [`padding_step_after`] handles by construction.
//!
//! Scope: this module emits only the equality rows *within* one batch.
//! Cross-batch continuity belongs to the prover that folds the batches.

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::ir::{WasmAuxOpcode, WasmBuildError, WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep};
use crate::isa::{opcode_info_from_code, WasmOpcode};
use crate::layout::COL_ONE;
use crate::operations::r1cs::{self, OperationCircuitError};
use crate::relation_layout::build_wasm_relation_layout;
use crate::sparse_r1cs::{FrontendError, SparseR1cs};
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
    #[error("trace must be non-empty")]
    EmptyTrace,
    #[error("batch dimensions or index are out of range")]
    Dimensions,
    #[error("cannot allocate the batch witness: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    #[error(transparent)]
    Trace(#[from] WasmBuildError),
    #[error("wasm batch relation has {actual} width declarations for {expected} columns")]
    WidthCount { actual: usize, expected: usize },
    #[error(transparent)]
    Operation(#[from] OperationCircuitError),
    #[error(transparent)]
    Frontend(#[from] FrontendError),
}

/// Build the batched wasm R1CS for the requested batch size.
///
/// Extends [`crate::build_wasm_relation`] with operation constraints before
/// batching. Sources cross-step links from [`build_wasm_relation_layout`]. Every
/// declared continuity link is emitted as a linking row.
pub fn build_batched_wasm_ccs(batch_size: usize) -> Result<BatchedWasmCcs, BatchError> {
    let operation = single_wasm_relation()?;
    batch_wasm_relation(&operation.relation, &operation.widths, batch_size)
}

/// The single-step WASM relation with its operation constraints, which
/// `build_batched_wasm_ccs` replicates.
pub(crate) fn single_wasm_relation() -> Result<r1cs::OperationShape, BatchError> {
    Ok(r1cs::single_relation()?)
}

/// Replicate one authoritative single-step WASM relation into an ordered
/// batch. This is also used after operation construction, so every block owns
/// the same arithmetic, operation, and range-check rows as the single-step path.
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
        .continuity
        .links()
        .map(|link| (link.previous_step_column, link.next_step_column))
        .collect();

    // Per boundary: one row per state-continuity pair.
    let n_link_per_boundary = link_pairs.len();
    let n_link = batch_size.saturating_sub(1) * n_link_per_boundary;

    let m_batch = batch_size * m_single;
    let n_batch = batch_size * n_single + n_link;

    let single_a = matrix_triplets(&single.a);
    let single_b = matrix_triplets(&single.b);
    let single_c = matrix_triplets(&single.c);

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
/// Each step includes the VM columns, range bits, and operation advice.
/// Each batch covers `batch_size` consecutive step witnesses, concatenated
/// in trace order. If `traces[batch_idx * batch_size..]` is shorter than
/// `batch_size`, the tail is padded with synthetic state-preserving
/// padding rows (see [`padding_step_after`]).
pub fn build_batched_witness(traces: &[WasmVmStep], batch_size: usize, batch_idx: usize) -> Result<Vec<F>, BatchError> {
    if batch_size == 0 {
        return Err(BatchError::BatchSizeZero);
    }
    let start = batch_idx
        .checked_mul(batch_size)
        .filter(|&start| start < traces.len())
        .ok_or(BatchError::Dimensions)?;
    let real_end = start
        .checked_add(batch_size)
        .ok_or(BatchError::Dimensions)?
        .min(traces.len());
    let padding = (real_end - start < batch_size)
        .then(|| padding_step_after(&traces[real_end - 1]))
        .transpose()?;
    let real_witnesses: Vec<Vec<F>> = traces[start..real_end]
        .iter()
        .map(build_step_witness)
        .collect::<Result<_, _>>()?;

    let single_width = real_witnesses[0].len();
    let length = batch_size
        .checked_mul(single_width)
        .ok_or(BatchError::Dimensions)?;
    let mut witness = Vec::new();
    witness.try_reserve_exact(length)?;
    for w in &real_witnesses {
        witness.extend_from_slice(w);
    }
    if let Some(padding) = padding {
        let padding = build_step_witness(&padding)?;
        let pad_count = batch_size - real_witnesses.len();
        for _ in 0..pad_count {
            witness.extend_from_slice(&padding);
        }
    }
    debug_assert_eq!(witness.len(), length);
    Ok(witness)
}

fn build_step_witness(step: &WasmVmStep) -> Result<Vec<F>, BatchError> {
    Ok(r1cs::extend_witness(build_witness_vector(step)?)?)
}

/// How many batches a trace of length `n` produces at `batch_size`,
/// counting the partial-final batch (padded by [`build_batched_witness`])
/// as one full batch.
pub fn batch_count(trace_len: usize, batch_size: usize) -> Result<usize, BatchError> {
    if batch_size == 0 {
        return Err(BatchError::BatchSizeZero);
    }
    if trace_len == 0 {
        return Err(BatchError::EmptyTrace);
    }
    Ok(trace_len.div_ceil(batch_size))
}

/// Build a synthetic state-preserving `WasmVmStep` whose `_before`
/// matches `prev`'s `_after`. Used to pad a trace up to a multiple of
/// `batch_size` without breaking cross-step continuity.
///
/// The wasm CCS enforces state preservation for `padding_active = 1`
/// rows (see the `non-program row shape` and `padding row state
/// preservation` constraint groups in `ccs/call.rs`); the witness we
/// build here is just the values that satisfy those rows.
pub fn padding_step_after(prev: &WasmVmStep) -> Result<WasmVmStep, BatchError> {
    let state = prev.state_after;
    if state.frame_move.active || state.local_zero.active || state.host_events.suspend_pending {
        return Err(WasmBuildError::Trace("cannot pad an unfinished auxiliary group".into()).into());
    }
    let pages = prev.state_after.memory_pages;
    let max_pages = prev.state_after.max_memory_pages;
    let fbp = prev.state_after.locals_fbp;
    let pc = prev.state_after.pc;
    let sp = prev.state_after.sp;
    let stack_frame_base = prev.state_after.stack_frame_base;
    let call_stack_depth = prev.state_after.call_stack_depth;
    let frame_move = prev.state_after.frame_move;
    let host_callee_fref = prev.state_after.host_callee_fref;
    let comm_chain = prev.state_after.comm_chain;
    let event_absorb = prev.state_after.event_absorb;
    let host_events = prev.state_after.host_events;
    Ok(WasmVmStep {
        cycle: prev.cycle.checked_add(1).ok_or(BatchError::Dimensions)?,
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
            activation_finished: prev.state_after.activation_finished,
            trapped: prev.state_after.trapped,
            frame_move,
            local_zero: prev.state_after.local_zero,
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
            activation_finished: prev.state_after.activation_finished,
            trapped: prev.state_after.trapped,
            frame_move,
            local_zero: prev.state_after.local_zero,
            host_callee_fref,
            comm_chain,
            event_absorb,
            host_events,
        },
        control_choice: 0,
        br_table_len: 0,
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
        expected_type_id: None,
        table_size: None,
        function_ref: None,
        target_function_is_guest: false,
        call_param_count: None,
        call_result_count: None,
        call_stack_push: None,
        call_stack_pop: None,
        host_context: None,
        host_event_return_schedule_count: None,
        host_event_rom_slot: None,
        host_event_word: None,
        host_event_initial_schedule_count: None,
        host_event_exit_schedule_count: None,
    })
}

fn matrix_triplets(m: &CcsMatrix<F>) -> Vec<(usize, usize, F)> {
    match m {
        CcsMatrix::Identity { n } => (0..*n).map(|i| (i, i, F::ONE)).collect(),
        CcsMatrix::Csc(csc) => {
            let mut out = Vec::with_capacity(csc.vals().len());
            for c in 0..csc.ncols() {
                for k in csc.column_range(c) {
                    out.push((csc.row_index(k), c, csc.vals()[k]));
                }
            }
            out
        }
        _ => unreachable!("the WASM relation builds plain CSC matrices"),
    }
}
