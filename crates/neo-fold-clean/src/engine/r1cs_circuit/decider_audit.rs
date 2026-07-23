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
    pub s_col_cols: Vec<[usize; 2]>,
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
    pub nc_point_cols: Vec<[usize; 2]>,
    pub nc_evaluation_cols: Vec<usize>,
    pub nc_evaluation_lanes: usize,
}

/// Exact terminal delayed-projection program over authoritative raw witnesses.
///
/// Each child witness range is the same allocation subsequently opened by the
/// terminal Ajtai relation.  No child `CeClaim.y_zcol`, digest, or sidecar is
/// part of this schedule.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalPendingProjectionAudit {
    /// Stable schema identifier for the single `PendingProjectionWires`
    /// value consumed by this terminal projection program.
    pub pending_projection_join_id: usize,
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub plan: super::RawOldBlockProjectionPlan,
    /// The exact indexed owner/A/B/C compiler consumed by the emitter.
    pub program: super::RawOldBlockProjectionProgram,
    /// Internally constructed canonical-to-physical map consumed by that
    /// compiler. It is never supplied by the prover or artifact caller.
    pub column_map: super::RawOldBlockProjectionColumnMap,
    pub radix: u32,
    pub tensor_rows: std::ops::Range<usize>,
    pub tensor_first_allocated_column: usize,
    pub projection_product_rows: std::ops::Range<usize>,
    pub projection_product_first_allocated_column: usize,
    pub final_scale_rows: std::ops::Range<usize>,
    pub final_scale_first_allocated_column: usize,
    pub terminal_rows: std::ops::Range<usize>,
    pub pending_old_block_cols: Vec<[usize; 2]>,
    pub parent_y_zcol_cols: Vec<[usize; 2]>,
    /// Absolute first column of each ordered raw witness allocation as seen
    /// by the projection program.
    pub projection_child_witness_first_columns: Vec<usize>,
    /// The same absolute columns as seen by the subsequent terminal
    /// CE/Ajtai consumer.  This is filled only after all fourteen claims have
    /// consumed the exact `FinalWitnessWires` allocations.
    pub ajtai_child_witness_first_columns: Vec<usize>,
}

impl super::R1csBuilder {
    pub(crate) fn record_terminal_pending_projection_ajtai_join(
        &mut self,
        pending_projection_join_id: usize,
        child_witness_first_columns: Vec<usize>,
    ) {
        if !self.records_structure() {
            return;
        }
        let audit = self
            .terminal_pending_projection_audits
            .last_mut()
            .expect("terminal CE/Ajtai join requires a preceding raw projection audit");
        assert_eq!(
            audit.pending_projection_join_id, pending_projection_join_id,
            "terminal pending-projection join identifier drift"
        );
        assert_eq!(
            audit.projection_child_witness_first_columns, child_witness_first_columns,
            "terminal projection and CE/Ajtai must consume the same FinalWitnessWires allocations"
        );
        audit.ajtai_child_witness_first_columns = child_witness_first_columns;
    }
}
