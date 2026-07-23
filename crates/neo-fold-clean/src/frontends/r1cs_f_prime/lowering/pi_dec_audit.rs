use crate::engine::r1cs_circuit::builder::{PiDecAdvAudit, PiDecClaimAudit, PiDecCommitmentAudit, PiDecStrictAudit};

fn commitment(audit: &PiDecCommitmentAudit, old_to_new: &[usize]) -> PiDecCommitmentAudit {
    PiDecCommitmentAudit {
        d_col: old_to_new[audit.d_col],
        kappa_col: old_to_new[audit.kappa_col],
        data_cols: audit.data_cols.iter().map(|&col| old_to_new[col]).collect(),
    }
}

fn adv(audit: &PiDecAdvAudit, old_to_new: &[usize]) -> PiDecAdvAudit {
    PiDecAdvAudit {
        ops: commitment(&audit.ops, old_to_new),
        is: commitment(&audit.is, old_to_new),
        fs: commitment(&audit.fs, old_to_new),
    }
}

fn claim(audit: &PiDecClaimAudit, old_to_new: &[usize]) -> PiDecClaimAudit {
    let remap_pair = |cols: [usize; 2]| cols.map(|col| old_to_new[col]);
    PiDecClaimAudit {
        commitment: commitment(&audit.commitment, old_to_new),
        adv: audit.adv.as_ref().map(|audit| adv(audit, old_to_new)),
        x_cols: audit.x_cols.iter().map(|&col| old_to_new[col]).collect(),
        x_rows: audit.x_rows,
        x_width: audit.x_width,
        x_rows_col: old_to_new[audit.x_rows_col],
        x_width_col: old_to_new[audit.x_width_col],
        m_in: audit.m_in,
        m_in_col: old_to_new[audit.m_in_col],
        y_ring_cols: audit
            .y_ring_cols
            .iter()
            .map(|row| row.iter().map(|&col| old_to_new[col]).collect())
            .collect(),
        ct_cols: audit.ct_cols.iter().copied().map(remap_pair).collect(),
        r_cols: audit.r_cols.iter().copied().map(remap_pair).collect(),
        s_col_cols: audit.s_col_cols.iter().copied().map(remap_pair).collect(),
        fold_digest_cols: audit.fold_digest_cols.map(|col| old_to_new[col]),
    }
}

pub(super) fn remap(audits: &[PiDecStrictAudit], old_to_new: &[usize]) -> Vec<PiDecStrictAudit> {
    let remap_pair = |cols: [usize; 2]| cols.map(|col| old_to_new[col]);
    audits
        .iter()
        .map(|audit| PiDecStrictAudit {
            row_start: audit.row_start,
            row_end: audit.row_end,
            x_recomposition_rows: audit.x_recomposition_rows.clone(),
            x_canonicality_rows: audit.x_canonicality_rows.clone(),
            first_allocated_column: old_to_new[audit.first_allocated_column],
            radix: audit.radix,
            parent: claim(&audit.parent, old_to_new),
            children: audit
                .children
                .iter()
                .map(|child| claim(child, old_to_new))
                .collect(),
            x_sign_traces: audit
                .x_sign_traces
                .iter()
                .copied()
                .map(remap_pair)
                .collect(),
        })
        .collect()
}
