use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::RiscvExecTable;
use crate::riscv::trace::{Rv64TraceLayout, Rv64TraceWitness};

use super::constraint_builder::{build_r1cs_ccs, Constraint};

#[derive(Clone, Debug)]
pub struct Rv64TraceCcsLayout {
    pub t: usize,
    pub m_in: usize,
    pub m: usize,

    pub const_one: usize,
    pub pc0: usize,
    pub pc_final: usize,
    pub halted_in: usize,
    pub halted_out: usize,

    pub trace_base: usize,
    pub trace: Rv64TraceLayout,
}

impl Rv64TraceCcsLayout {
    pub fn new(t: usize) -> Result<Self, String> {
        Self::new_internal(t)
    }

    pub fn new_uniform(t: usize) -> Result<Self, String> {
        Self::new_internal(t)
    }

    fn new_internal(t: usize) -> Result<Self, String> {
        if t == 0 {
            return Err("Rv64TraceCcsLayout: t must be >= 1".into());
        }

        let const_one = 0usize;
        let pc0 = 1usize;
        let pc_final = 2usize;
        let halted_in = 3usize;
        let halted_out = 4usize;
        let m_in = 5usize;

        let trace = Rv64TraceLayout::new();
        let trace_base = m_in;
        let m = trace_base
            .checked_add(trace.cols)
            .ok_or_else(|| "Rv64TraceCcsLayout: m overflow".to_string())?;

        Ok(Self {
            t,
            m_in,
            m,
            const_one,
            pc0,
            pc_final,
            halted_in,
            halted_out,
            trace_base,
            trace,
        })
    }

    #[inline]
    pub fn cell(&self, trace_col: usize, row: usize) -> usize {
        debug_assert!(trace_col < self.trace.cols);
        debug_assert!(row < self.t);
        assert!(
            row == 0,
            "uniform kernel physical cell indexing must not use per-row offsets"
        );
        self.trace_base + trace_col
    }

    #[inline]
    pub fn is_uniform_kernel(&self) -> bool {
        true
    }
}

pub fn rv64_trace_ccs_witness_from_exec_table(
    layout: &Rv64TraceCcsLayout,
    exec: &RiscvExecTable,
) -> Result<(Vec<F>, Vec<F>), String> {
    let wit = Rv64TraceWitness::from_exec_table(&layout.trace, exec)?;
    rv64_trace_ccs_witness_from_trace_witness(layout, &wit)
}

pub fn rv64_trace_ccs_witness_from_trace_witness(
    layout: &Rv64TraceCcsLayout,
    wit: &Rv64TraceWitness,
) -> Result<(Vec<F>, Vec<F>), String> {
    if wit.t != layout.t {
        return Err(format!(
            "RV64 trace CCS witness: t mismatch (wit.t={} layout.t={})",
            wit.t, layout.t
        ));
    }
    if wit.cols.len() != layout.trace.cols {
        return Err(format!(
            "RV64 trace CCS witness: width mismatch (wit.cols={} trace.cols={})",
            wit.cols.len(),
            layout.trace.cols
        ));
    }

    let mut x = vec![F::ZERO; layout.m_in];
    x[layout.const_one] = F::ONE;
    x[layout.pc0] = wit.cols[layout.trace.pc_before][0];
    x[layout.pc_final] = wit.cols[layout.trace.pc_after][layout.t - 1];
    x[layout.halted_in] = wit.cols[layout.trace.halted][0];
    x[layout.halted_out] = wit.cols[layout.trace.halted][layout.t - 1];

    let mut w = vec![F::ZERO; layout.m - layout.m_in];
    for trace_col in 0..layout.trace.cols {
        w[trace_col] = wit.cols[trace_col][0];
    }

    Ok((x, w))
}

pub fn build_rv64_trace_wiring_ccs(layout: &Rv64TraceCcsLayout) -> Result<CcsStructure<F>, String> {
    build_rv64_trace_wiring_ccs_with_reserved_rows(layout, 0)
}

pub fn build_rv64_trace_wiring_ccs_with_reserved_rows(
    layout: &Rv64TraceCcsLayout,
    reserved_rows: usize,
) -> Result<CcsStructure<F>, String> {
    let one = layout.const_one;
    let tr = |c: usize| -> usize { layout.cell(c, 0) };
    let l = &layout.trace;
    let lo32_base = F::from_u64(1u64 << 32);

    let mut cons: Vec<Constraint<F>> = vec![
        Constraint::terms(one, false, vec![(tr(l.one), F::ONE), (one, -F::ONE)]),
        Constraint::terms(one, false, vec![(layout.pc0, F::ONE), (tr(l.pc_before), -F::ONE)]),
        Constraint::terms(one, false, vec![(layout.halted_in, F::ONE), (tr(l.halted), -F::ONE)]),
        Constraint::terms(one, false, vec![(tr(l.active), F::ONE), (one, -F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_val), F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_lhs), F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_rhs), F::ONE)]),
    ];

    for &(scalar, lo, hi) in &[
        (l.rs1_val, l.rs1_val_lo32, l.rs1_val_hi32),
        (l.rs2_val, l.rs2_val_lo32, l.rs2_val_hi32),
        (l.rd_val, l.rd_val_lo32, l.rd_val_hi32),
        (l.ram_rv, l.ram_rv_lo32, l.ram_rv_hi32),
        (l.ram_wv, l.ram_wv_lo32, l.ram_wv_hi32),
        (l.shout_val, l.shout_val_lo32, l.shout_val_hi32),
        (l.shout_lhs, l.shout_lhs_lo32, l.shout_lhs_hi32),
        (l.shout_rhs, l.shout_rhs_lo32, l.shout_rhs_hi32),
        (l.shout_add_sub_key, l.shout_add_sub_key_lo32, l.shout_add_sub_key_hi32),
    ] {
        cons.push(Constraint::terms(
            one,
            false,
            vec![(tr(scalar), F::ONE), (tr(lo), -F::ONE), (tr(hi), -lo32_base)],
        ));
    }

    let time_rows = layout
        .m_in
        .checked_add(layout.t)
        .ok_or_else(|| "RV64 trace CCS: m_in + t overflow".to_string())?;
    let n = time_rows
        .max(cons.len())
        .checked_add(reserved_rows)
        .ok_or_else(|| "RV64 trace CCS: n overflow".to_string())?;

    build_r1cs_ccs(&cons, n, layout.m, layout.const_one)
}
