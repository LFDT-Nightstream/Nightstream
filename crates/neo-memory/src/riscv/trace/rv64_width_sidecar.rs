use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;

pub const RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE: u32 = 0x5256_6800;
pub const RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID: u32 = RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE;
pub const RV64_TRACE_WIDTH_ADDR_GROUP_BASE: u32 = 0x5256_6A00;
const RV64_TRACE_WIDTH_LOOKUP_GROUPED: bool = true;

#[derive(Clone, Debug)]
pub struct Rv64WidthSidecarLayout {
    pub cols: usize,
    pub ram_rv_low_bit: [usize; 32],
    pub rs2_low_bit: [usize; 32],
}

impl Rv64WidthSidecarLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };

        let mut ram_rv_low_bit = [0usize; 32];
        let mut rs2_low_bit = [0usize; 32];
        for bit_col in &mut ram_rv_low_bit {
            *bit_col = take();
        }
        for bit_col in &mut rs2_low_bit {
            *bit_col = take();
        }

        debug_assert_eq!(next, 64);
        Self {
            cols: next,
            ram_rv_low_bit,
            rs2_low_bit,
        }
    }
}

#[inline]
pub fn rv64_width_lookup_backed_cols(layout: &Rv64WidthSidecarLayout) -> Vec<usize> {
    (0..layout.cols).collect()
}

#[inline]
pub const fn rv64_width_lookup_table_id_for_col(col: usize) -> u32 {
    if RV64_TRACE_WIDTH_LOOKUP_GROUPED {
        RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID
    } else {
        RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE + col as u32
    }
}

#[inline]
pub const fn rv64_is_width_lookup_table_id(table_id: u32) -> bool {
    (table_id >= RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE && table_id < RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE + 64)
        || table_id == RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID
}

#[inline]
pub const fn rv64_is_width_lookup_grouped_table_id(table_id: u32) -> bool {
    RV64_TRACE_WIDTH_LOOKUP_GROUPED && table_id == RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID
}

#[inline]
pub fn rv64_width_lookup_transport_n_vals() -> usize {
    let layout = Rv64WidthSidecarLayout::new();
    rv64_width_lookup_backed_cols(&layout).len().max(1)
}

#[inline]
pub fn rv64_width_lookup_val_slot_for_col(col: usize) -> Option<usize> {
    let layout = Rv64WidthSidecarLayout::new();
    let cols = rv64_width_lookup_backed_cols(&layout);
    cols.iter().position(|&c| c == col)
}

#[inline]
pub fn rv64_width_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    if !rv64_is_width_lookup_table_id(table_id) {
        return None;
    }
    Some(RV64_TRACE_WIDTH_ADDR_GROUP_BASE)
}

#[derive(Clone, Debug)]
pub struct Rv64WidthSidecarWitness {
    pub t: usize,
    pub cols: Vec<Vec<F>>,
}

impl Rv64WidthSidecarWitness {
    pub fn new_zero(layout: &Rv64WidthSidecarLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }
}

pub fn rv64_width_sidecar_witness_from_exec_table(
    layout: &Rv64WidthSidecarLayout,
    exec: &Rv32ExecTable,
) -> Rv64WidthSidecarWitness {
    let cols = exec.to_columns();
    let t = cols.len();
    let mut wit = Rv64WidthSidecarWitness::new_zero(layout, t);

    for i in 0..t {
        if !cols.active[i] {
            continue;
        }
        let rs2 = cols.rs2_val[i];
        for (bit, &col_id) in layout.rs2_low_bit.iter().enumerate() {
            wit.cols[col_id][i] = F::from_u64((rs2 >> bit) & 1);
        }
    }

    for (i, row) in exec.rows.iter().enumerate() {
        if !row.active {
            continue;
        }
        let mut read_value = None;
        for event in &row.ram_events {
            if event.kind == neo_vm_trace::TwistOpKind::Read {
                read_value = Some(event.value);
                break;
            }
        }
        if let Some(rv) = read_value {
            for (bit, &col_id) in layout.ram_rv_low_bit.iter().enumerate() {
                wit.cols[col_id][i] = F::from_u64((rv >> bit) & 1);
            }
        }
    }

    wit
}
