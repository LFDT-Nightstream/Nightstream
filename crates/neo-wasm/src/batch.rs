//! Batched wasm CCS construction: amortizes F'-shell fold cost across
//! `batch_size` consecutive wasm steps.
//!
//! The single-step wasm CCS proves *one* step satisfies the row constraints
//! but does not, on its own, enforce that consecutive steps form a coherent
//! execution — e.g., that `step[i].pc_after == step[i+1].pc_before`. The
//! cross-step links live in [`WasmLookupBindingLayout::cross_step_links`]
//! as metadata; this module compiles them into actual R1CS rows by:
//!
//! 1. **Block-diagonalising** the single-step matrices `A`, `B`, `C`: each
//!    of `batch_size` step blocks gets its own copy, with no cross-block
//!    entries on the diagonal portion of the matrix.
//! 2. **Adding link rows** between adjacent blocks. Two families:
//!    - **Local-constant link**: every step has its own copy of `COL_ONE`;
//!      link rows force them all equal to the global `z[0]`.
//!    - **State continuity**: for each `(prev_after, next_before)` pair in
//!      the spec, an equality row across adjacent blocks.
//!
//! Witness shape: `m_batch = batch_size * m_single`. Step `s`'s columns
//! live at `z[s * m_single .. (s+1) * m_single]`. `m_in = 1` (the single
//! global constant slot at `z[0]`).
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
//! Scope: links only fire *within* a batch. The boundary between batch
//! `i`'s last block and batch `i+1`'s first block is not currently
//! enforced — see [`crate::prove::verify`] for the soundness
//! implication. Cross-batch linking is open follow-up work.

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::frontends::direct_ccs::FrontendError;
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::builder::build_witness_vector;
use crate::ccs::WasmVmSpec;
use crate::ir::{WasmAuxOpcode, WasmParamInitState, WasmPcEdgeKind, WasmRowKind, WasmStepTrace};
use crate::isa::{opcode_info_from_code, WasmOpcode};
use crate::layout::{ColumnWidth, COLUMN_SPECS, COL_ONE, WITNESS_WIDTH};
use crate::lookup_binding_builder::build_wasm_lookup_binding_layout;

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
    #[error(transparent)]
    Frontend(#[from] FrontendError),
}

/// Build the batched wasm R1CS for the requested batch size.
///
/// Sources the single-step matrices from [`WasmVmSpec::default`] and the
/// cross-step link spec from [`build_wasm_lookup_binding_layout`]. Every
/// `(prev_after, next_before)` column pair in every spec link is emitted
/// as a linking row; the `flat_map` over `column_pairs` happens to also
/// be a no-op for any future link whose invariant can't be expressed as
/// column equalities (none today).
pub fn build_batched_wasm_ccs(batch_size: usize) -> Result<BatchedWasmCcs, BatchError> {
    if batch_size == 0 {
        return Err(BatchError::BatchSizeZero);
    }

    let vm = WasmVmSpec::default();
    let core = vm.core_ccs_spec();
    let m_single = core.structure.m;
    let n_single = core.structure.n;
    assert_eq!(m_single, WITNESS_WIDTH);
    assert_eq!(core.m_in, 1, "wasm m_in must be 1 (just the constant slot)");

    let layout = build_wasm_lookup_binding_layout();
    let link_pairs: Vec<(usize, usize)> = layout
        .cross_step_links
        .iter()
        .flat_map(|link| link.column_pairs.iter())
        .map(|pair| (pair.prev_after.0, pair.next_before.0))
        .collect();

    // Per boundary: 1 local-constant link + one row per state-continuity pair.
    let n_link_per_boundary = 1 + link_pairs.len();
    let n_link = batch_size.saturating_sub(1) * n_link_per_boundary;

    let m_batch = batch_size * m_single;
    let n_batch = batch_size * n_single + n_link;

    let single_a = matrix_triplets(&core.structure.matrices[0]);
    let single_b = matrix_triplets(&core.structure.matrices[1]);
    let single_c = matrix_triplets(&core.structure.matrices[2]);

    let mut a_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_a.len());
    let mut b_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_b.len());
    let mut c_triplets: Vec<(usize, usize, F)> = Vec::with_capacity(batch_size * single_c.len());

    // Block-diagonal replication.
    for step in 0..batch_size {
        let row_offset = step * n_single;
        let col_offset = step * m_single;
        for &(r, c, v) in &single_a {
            a_triplets.push((row_offset + r, col_offset + c, v));
        }
        for &(r, c, v) in &single_b {
            b_triplets.push((row_offset + r, col_offset + c, v));
        }
        for &(r, c, v) in &single_c {
            c_triplets.push((row_offset + r, col_offset + c, v));
        }
    }

    // Linking rows. Each row enforces `(A_row · z) * (B_row · z) = C_row · z`
    // with `A_row = z[lhs] - z[rhs]`, `B_row = z[COL_ONE]`, `C_row = 0` —
    // i.e. `z[lhs] - z[rhs] = 0`. Step 0's `COL_ONE` IS the global `z[0]`
    // (col_offset = 0), so we only link steps 1..batch_size.
    let mut link_row = batch_size * n_single;
    for boundary in 0..batch_size.saturating_sub(1) {
        let curr_offset = boundary * m_single;
        let next_offset = (boundary + 1) * m_single;

        // Local-constant link: z[next_offset + COL_ONE] = z[COL_ONE].
        a_triplets.push((link_row, next_offset + COL_ONE, F::ONE));
        a_triplets.push((link_row, COL_ONE, -F::ONE));
        b_triplets.push((link_row, COL_ONE, F::ONE));
        link_row += 1;

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

    let sparse_r1cs = SparseR1cs::new(a, b, c, n_batch, m_batch, 1)?;

    let widths_single = wasm_app_private_var_widths();
    let mut all_widths = Vec::with_capacity(m_batch);
    for _ in 0..batch_size {
        all_widths.extend(widths_single.iter().copied());
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
pub fn build_batched_witness(traces: &[WasmStepTrace], batch_size: usize, batch_idx: usize) -> Vec<F> {
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

    let mut witness = Vec::with_capacity(batch_size * WITNESS_WIDTH);
    for w in &real_witnesses {
        witness.extend(w);
    }
    if real_witnesses.len() < batch_size {
        let last_real = &traces[real_end - 1];
        let mut anchor = padding_step_after(last_real);
        let pad_count = batch_size - real_witnesses.len();
        for _ in 0..pad_count {
            witness.extend(build_witness_vector(&anchor));
            // Each subsequent padding row anchors on the previous one,
            // which is state-preserving — so `_after` is the same as the
            // first padding row's `_after`. Reuse `anchor` directly.
            anchor = padding_step_after(&anchor);
        }
    }
    debug_assert_eq!(witness.len(), batch_size * WITNESS_WIDTH);
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

/// Build a synthetic state-preserving `WasmStepTrace` whose `_before`
/// matches `prev`'s `_after`. Used to pad a trace up to a multiple of
/// `batch_size` without breaking cross-step continuity.
///
/// The wasm CCS enforces state preservation for `padding_active = 1`
/// rows (see the `non-program row shape` and `padding row state
/// preservation` constraint groups in `ccs/call.rs`); the witness we
/// build here is just the values that satisfy those rows.
pub fn padding_step_after(prev: &WasmStepTrace) -> WasmStepTrace {
    let pages = prev.memory_pages_after;
    let fbp = prev.locals_fbp_after;
    let pc = prev.pc_after;
    let sp = prev.sp_after;
    let param_init = prev.param_init_after;
    debug_assert!(
        !param_init.active,
        "padding inside a param-init aux sequence is unsupported"
    );
    WasmStepTrace {
        cycle: prev.cycle + 1,
        row_kind: WasmRowKind::Aux(WasmAuxOpcode::Padding),
        pc_before: pc,
        pc_after: pc,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        param_init_before: WasmParamInitState::ZERO,
        param_init_after: WasmParamInitState::ZERO,
        wide_values_enabled: false,
        // opcode_code = 0: no opcode selector fires on this row (the
        // selector one-hot demands `sum(selectors) = is_program_row`,
        // which is 0 for padding).
        opcode_code: 0,
        opcode: WasmOpcode::Nop,
        info: opcode_info_from_code(0),
        stack_reads_override: Some(0),
        stack_writes_override: Some(0),
        sp_before: sp,
        sp_after: sp,
        current_function_ref: prev.current_function_ref,
        current_function_num_locals: prev.current_function_num_locals,
        stack_read0: None,
        stack_read0_hi: None,
        stack_read1: None,
        stack_read1_hi: None,
        stack_read2: None,
        stack_read2_hi: None,
        stack_write0: None,
        stack_write0_hi: None,
        linear_memory: None,
        linear_memory_offset: 0,
        memory_pages_before: pages,
        memory_pages_after: pages,
        halted: false,
        locals_fbp: fbp,
        locals_fbp_after: fbp,
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
    }
}

fn matrix_triplets(m: &CcsMatrix<F>) -> Vec<(usize, usize, F)> {
    match m {
        CcsMatrix::Identity { n } => (0..*n).map(|i| (i, i, F::ONE)).collect(),
        CcsMatrix::Csc(csc) => {
            let mut out = Vec::with_capacity(csc.vals.len());
            for c in 0..csc.ncols {
                for k in csc.col_ptr[c]..csc.col_ptr[c + 1] {
                    out.push((csc.row_idx[k], c, csc.vals[k]));
                }
            }
            out
        }
    }
}

fn wasm_app_private_var_widths() -> Vec<usize> {
    COLUMN_SPECS
        .iter()
        .map(|spec| match spec.width {
            ColumnWidth::Boolean => 1,
            ColumnWidth::Byte => 8,
            ColumnWidth::U32 => 32,
            ColumnWidth::Field => 64,
        })
        .collect()
}
