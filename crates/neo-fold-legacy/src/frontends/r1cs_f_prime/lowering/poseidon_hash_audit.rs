//! Column remapping for compact Poseidon2 sponge audits.
//!
//! Owns: translating retained hash-level column provenance through the exact
//! field-R1CS column permutation.
//!
//! Does not own: Poseidon2 equations, transcript semantics, row ownership, or
//! semantic authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: this is a bijective coordinate rename of an emitted
//! builder trace. Consumers must still match the renamed trace to an
//! independently specified hash schedule and to the retained permutation rows.
//!
//! | Child | Preserved fact |
//! |---|---|
//! | hash inputs | exact ordered sponge preimage columns |
//! | rounds | absorb/pad kind, state columns, and defining rows |
//! | hash outputs | exact four digest columns |

use crate::engine::r1cs_circuit::builder::{Poseidon2HashAudit, Poseidon2HashRoundAudit, Poseidon2HashRoundAuditKind};

pub(super) fn remap(audits: &[Poseidon2HashAudit], old_to_new: &[usize]) -> Vec<Poseidon2HashAudit> {
    let column = |old: usize| {
        *old_to_new
            .get(old)
            .unwrap_or_else(|| panic!("Poseidon2 hash audit column {old} escaped synthesized width"))
    };
    let columns = |values: &[usize]| values.iter().map(|&old| column(old)).collect();

    audits
        .iter()
        .map(|audit| Poseidon2HashAudit {
            row_start: audit.row_start,
            row_end: audit.row_end,
            input_cols: columns(&audit.input_cols),
            zero_col: column(audit.zero_col),
            zero_row: audit.zero_row,
            rounds: audit
                .rounds
                .iter()
                .map(|round| Poseidon2HashRoundAudit {
                    kind: match &round.kind {
                        Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => Poseidon2HashRoundAuditKind::Absorb {
                            chunk_cols: columns(chunk_cols),
                        },
                        Poseidon2HashRoundAuditKind::Pad => Poseidon2HashRoundAuditKind::Pad,
                    },
                    state_before_cols: round.state_before_cols.map(column),
                    permutation_input_cols: round.permutation_input_cols.map(column),
                    defining_rows: round.defining_rows.clone(),
                    first_allocated_column: column(round.first_allocated_column),
                    permutation_output_cols: round.permutation_output_cols.map(column),
                })
                .collect(),
            output_cols: audit.output_cols.map(column),
        })
        .collect()
}
