//! Read-only compact Poseidon2 audit views.
//!
//! Owns the public audit data and its projection from builder traces. The
//! emitted R1CS rows remain authoritative.

use super::{Lc, R1csBuilder};

/// Compact assurance view of one exact production Poseidon2 invocation.
///
/// The isolated artifact numbers its eight inputs as columns 1..8 and its
/// fresh columns from 9 onward. A call site is therefore identified by its
/// eight input columns and first fresh column; the remaining renaming is
/// affine. Row hashes remain the drift authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2PermutationAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub input_cols: [usize; 8],
    pub first_allocated_col: usize,
    pub allocated_col_count: usize,
    pub output_cols: [usize; 8],
}

/// Exact compact input and output of one Poseidon2 `x^7` site.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2CompactSboxAudit {
    pub input: Lc,
    pub output_col: usize,
}

/// Exact degree-seven trace used by selective Poseidon2 lowering.
///
/// This view omits the ordinary R1CS materialization columns. Artifact
/// exporters use it to prove that the compact relation has the same round
/// schedule and final state as the independent Lean model.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2CompactPermutationAudit {
    pub input_cols: [usize; 8],
    pub sboxes: Vec<Poseidon2CompactSboxAudit>,
    pub output_cols: [usize; 8],
    pub output_linear_forms: [Lc; 8],
}

impl R1csBuilder {
    /// Exact column-renaming certificates for all emitted Poseidon2 calls.
    #[doc(hidden)]
    pub fn poseidon2_permutation_audits(&self) -> Vec<Poseidon2PermutationAudit> {
        self.poseidon2_traces
            .iter()
            .map(|trace| {
                let first_allocated_col = trace
                    .allocated_columns
                    .first()
                    .copied()
                    .expect("Poseidon2 permutation allocates fresh columns");
                assert!(
                    trace
                        .allocated_columns
                        .iter()
                        .copied()
                        .eq(first_allocated_col..first_allocated_col + trace.allocated_columns.len()),
                    "Poseidon2 fresh columns must remain contiguous",
                );
                Poseidon2PermutationAudit {
                    row_start: trace.row_start,
                    row_end: trace.row_end,
                    input_cols: trace.input_cols,
                    first_allocated_col,
                    allocated_col_count: trace.allocated_columns.len(),
                    output_cols: trace.output_cols,
                }
            })
            .collect()
    }

    /// Exact compact traces consumed by selective Poseidon2 lowering.
    #[doc(hidden)]
    pub fn poseidon2_compact_permutation_audits(&self) -> Vec<Poseidon2CompactPermutationAudit> {
        self.poseidon2_traces
            .iter()
            .map(|trace| Poseidon2CompactPermutationAudit {
                input_cols: trace.input_cols,
                sboxes: trace
                    .sboxes
                    .iter()
                    .map(|sbox| Poseidon2CompactSboxAudit {
                        input: sbox.input.clone(),
                        output_col: sbox.output_col,
                    })
                    .collect(),
                output_cols: trace.output_cols,
                output_linear_forms: trace.output_linear_forms.clone(),
            })
            .collect()
    }
}
