//! Shared prepared-layout audits for selective compiler and decoder data.

use super::{
    prepare_selective_layout_core, projected_decoder, trace_error, LowNormR1csError, SelectiveEncoding,
    SelectiveLayout, SelectiveLayoutCore, SelectiveProjectedDecoderRunProvenance, SelectiveRowMappingAudit, SparseR1cs,
};

/// Small exact projection of the prepared selective layout used by replay
/// call-placement certificates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveCompactLayoutAudit {
    rows: SelectiveRowMappingAudit,
    selector_columns: Vec<usize>,
    final_columns: usize,
}

impl SelectiveCompactLayoutAudit {
    pub fn rows(&self) -> &SelectiveRowMappingAudit {
        &self.rows
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    pub fn final_columns(&self) -> usize {
        self.final_columns
    }
}

/// Return the compact row and coordinate ledger plus requested decoder runs
/// from one exact selective-layout-core preparation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    norm_base: u32,
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<(SelectiveCompactLayoutAudit, Vec<SelectiveProjectedDecoderRunProvenance>), LowNormR1csError> {
    let layout = prepare_selective_layout_core(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        SelectiveEncoding::for_norm_base(norm_base)?,
    )?;
    let decoders = decoder_runs_from_core(arms, &layout, requests)?;
    let audit = SelectiveCompactLayoutAudit {
        rows: layout.prepared_rows.audit(),
        selector_columns: layout.selector_cols.clone(),
        final_columns: layout.summary().columns,
    };
    Ok((audit, decoders))
}

pub(super) fn decoder_runs_from_layout(
    arms: &[SparseR1cs],
    layout: &SelectiveLayout,
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<Vec<SelectiveProjectedDecoderRunProvenance>, LowNormR1csError> {
    requests
        .iter()
        .map(|(arm, source_range)| {
            let source_arm = arms
                .get(*arm)
                .ok_or_else(|| trace_error("complete decoder arm is out of range"))?;
            projected_decoder::decoder_run_provenance(layout, *arm, source_range.clone(), source_arm)
        })
        .collect()
}

fn decoder_runs_from_core(
    arms: &[SparseR1cs],
    layout: &SelectiveLayoutCore,
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<Vec<SelectiveProjectedDecoderRunProvenance>, LowNormR1csError> {
    requests
        .iter()
        .map(|(arm, source_range)| {
            let source_arm = arms
                .get(*arm)
                .ok_or_else(|| trace_error("complete decoder arm is out of range"))?;
            projected_decoder::decoder_run_provenance_from_core(layout, *arm, source_range.clone(), source_arm)
        })
        .collect()
}
