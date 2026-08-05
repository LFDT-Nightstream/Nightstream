//! Read-only row/column schedules for the strict and terminal decider paths.
//!
//! These records carry no acceptance bit. Exporters must compare every named
//! row with the synthesized sparse matrices before assigning semantic meaning
//! to any recorded column.

/// Exact wire schedule for one commitment coordinate consumed by strict
/// PiDEC.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecCommitmentAudit {
    pub d_col: usize,
    pub kappa_col: usize,
    pub data_cols: Vec<usize>,
}

/// The optional three-coordinate Nebula commitment carried by a CE claim.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecAdvAudit {
    pub ops: PiDecCommitmentAudit,
    pub is: PiDecCommitmentAudit,
    pub fs: PiDecCommitmentAudit,
}

/// Exact input-wire layout for one CE claim consumed by strict PiDEC.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecClaimAudit {
    pub commitment: PiDecCommitmentAudit,
    pub adv: Option<PiDecAdvAudit>,
    pub x_cols: Vec<usize>,
    pub x_rows: usize,
    pub x_width: usize,
    pub x_rows_col: usize,
    pub x_width_col: usize,
    pub m_in: usize,
    pub m_in_col: usize,
    pub y_ring_cols: Vec<Vec<usize>>,
    pub ct_cols: Vec<[usize; 2]>,
    pub r_cols: Vec<[usize; 2]>,
    pub fold_digest_cols: [usize; 4],
}

/// Complete strict-PiDEC input schedule for one emitted verifier invocation.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecStrictAudit {
    pub row_start: usize,
    pub row_end: usize,
    /// Exact retained parent-X radix-recomposition source rows.
    pub x_recomposition_rows: std::ops::Range<usize>,
    /// Exact uniform-sign/digit canonicality source rows.
    pub x_canonicality_rows: std::ops::Range<usize>,
    pub first_allocated_column: usize,
    pub radix: u32,
    pub parent: PiDecClaimAudit,
    pub children: Vec<PiDecClaimAudit>,
    /// `[sign, centered-product]` columns, row-major over active X.
    pub x_sign_traces: Vec<[usize; 2]>,
}

/// Exact input-wire ownership for one direct terminal-CE claim program.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCeClaimAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub norm_bound: u32,
    pub expected_public_width: Option<usize>,
    pub structure_rows: usize,
    pub structure_columns: usize,
    pub witness_rows: usize,
    pub witness_columns: usize,
    pub witness_cols: Vec<usize>,
    pub norm_first_allocated_column: usize,
    pub commitment_cols: Vec<usize>,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub public_cols: Vec<usize>,
    pub public_rows: usize,
    pub public_width: usize,
    pub public_input_len: usize,
    pub point_cols: Vec<[usize; 2]>,
    pub evaluation_cols: Vec<Vec<usize>>,
    pub constant_term_cols: Vec<[usize; 2]>,
}
