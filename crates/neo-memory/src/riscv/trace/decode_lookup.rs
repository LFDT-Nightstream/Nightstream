use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

/// Base lookup table id for decode-column lookup families in shared-bus mode.
///
/// Table id for decode column `c` is `RV32_TRACE_DECODE_LOOKUP_TABLE_BASE + c`.
pub const RV32_TRACE_DECODE_LOOKUP_TABLE_BASE: u32 = 0x5256_4400;
/// Grouped decode lookup table id used in shared-bus trace mode.
///
/// In grouped mode, all decode transport columns share one `table_id` and use
/// value slots (`n_vals`) to carry per-column values.
pub const RV32_TRACE_DECODE_LOOKUP_GROUPED_TABLE_ID: u32 = RV32_TRACE_DECODE_LOOKUP_TABLE_BASE;
/// Base address-group id for decode lookup lanes.
pub const RV32_TRACE_DECODE_ADDR_GROUP_BASE: u32 = 0x5256_4A00;
const RV32_TRACE_DECODE_LOOKUP_GROUPED: bool = true;

#[derive(Clone, Debug)]
pub struct Rv32DecodeSidecarLayout {
    pub cols: usize,
    pub opcode: usize,
    pub funct3: usize,
    pub funct7: usize,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rd_has_write: usize,
    pub ram_has_read: usize,
    pub ram_has_write: usize,
    pub shout_table_id: usize,
    pub op_lui: usize,
    pub op_auipc: usize,
    pub op_jal: usize,
    pub op_jalr: usize,
    pub op_branch: usize,
    pub op_load: usize,
    pub op_store: usize,
    pub op_alu_imm: usize,
    pub op_alu_reg: usize,
    pub op_misc_mem: usize,
    pub op_system: usize,
    pub op_amo: usize,
    pub op_custom: usize,
    pub op_lui_write: usize,
    pub op_auipc_write: usize,
    pub op_jal_write: usize,
    pub op_jalr_write: usize,
    pub op_alu_imm_write: usize,
    pub op_alu_reg_write: usize,
    pub is_lb_write: usize,
    pub is_lbu_write: usize,
    pub is_lh_write: usize,
    pub is_lhu_write: usize,
    pub is_lw_write: usize,
    pub funct3_is: [usize; 8],
    pub alu_reg_table_delta: usize,
    pub alu_imm_table_delta: usize,
    pub alu_imm_shift_rhs_delta: usize,
    pub imm_i: usize,
    pub imm_s: usize,
    pub imm_b: usize,
    pub imm_j: usize,
    pub rd_bit: [usize; 5],
    pub funct3_bit: [usize; 3],
    pub rs1_bit: [usize; 5],
    pub rs2_bit: [usize; 5],
    pub funct7_bit: [usize; 7],
    pub rd_is_zero_01: usize,
    pub rd_is_zero_012: usize,
    pub rd_is_zero_0123: usize,
    pub rd_is_zero: usize,
}

impl Rv32DecodeSidecarLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };
        let opcode = take();
        let funct3 = take();
        let funct7 = take();
        let rd = take();
        let rs1 = take();
        let rs2 = take();
        let rd_has_write = take();
        let ram_has_read = take();
        let ram_has_write = take();
        let shout_table_id = take();
        let op_lui = take();
        let op_auipc = take();
        let op_jal = take();
        let op_jalr = take();
        let op_branch = take();
        let op_load = take();
        let op_store = take();
        let op_alu_imm = take();
        let op_alu_reg = take();
        let op_misc_mem = take();
        let op_system = take();
        let op_amo = take();
        let op_custom = take();
        let op_lui_write = take();
        let op_auipc_write = take();
        let op_jal_write = take();
        let op_jalr_write = take();
        let op_alu_imm_write = take();
        let op_alu_reg_write = take();
        let is_lb_write = take();
        let is_lbu_write = take();
        let is_lh_write = take();
        let is_lhu_write = take();
        let is_lw_write = take();
        let funct3_is_0 = take();
        let funct3_is_1 = take();
        let funct3_is_2 = take();
        let funct3_is_3 = take();
        let funct3_is_4 = take();
        let funct3_is_5 = take();
        let funct3_is_6 = take();
        let funct3_is_7 = take();
        let alu_reg_table_delta = take();
        let alu_imm_table_delta = take();
        let alu_imm_shift_rhs_delta = take();
        let imm_i = take();
        let imm_s = take();
        let imm_b = take();
        let imm_j = take();
        let rd_b0 = take();
        let rd_b1 = take();
        let rd_b2 = take();
        let rd_b3 = take();
        let rd_b4 = take();
        let funct3_b0 = take();
        let funct3_b1 = take();
        let funct3_b2 = take();
        let rs1_b0 = take();
        let rs1_b1 = take();
        let rs1_b2 = take();
        let rs1_b3 = take();
        let rs1_b4 = take();
        let rs2_b0 = take();
        let rs2_b1 = take();
        let rs2_b2 = take();
        let rs2_b3 = take();
        let rs2_b4 = take();
        let funct7_b0 = take();
        let funct7_b1 = take();
        let funct7_b2 = take();
        let funct7_b3 = take();
        let funct7_b4 = take();
        let funct7_b5 = take();
        let funct7_b6 = take();
        let rd_is_zero_01 = take();
        let rd_is_zero_012 = take();
        let rd_is_zero_0123 = take();
        let rd_is_zero = take();
        debug_assert_eq!(next, 78);
        Self {
            cols: next,
            opcode,
            funct3,
            funct7,
            rd,
            rs1,
            rs2,
            rd_has_write,
            ram_has_read,
            ram_has_write,
            shout_table_id,
            op_lui,
            op_auipc,
            op_jal,
            op_jalr,
            op_branch,
            op_load,
            op_store,
            op_alu_imm,
            op_alu_reg,
            op_misc_mem,
            op_system,
            op_amo,
            op_custom,
            op_lui_write,
            op_auipc_write,
            op_jal_write,
            op_jalr_write,
            op_alu_imm_write,
            op_alu_reg_write,
            is_lb_write,
            is_lbu_write,
            is_lh_write,
            is_lhu_write,
            is_lw_write,
            funct3_is: [
                funct3_is_0,
                funct3_is_1,
                funct3_is_2,
                funct3_is_3,
                funct3_is_4,
                funct3_is_5,
                funct3_is_6,
                funct3_is_7,
            ],
            alu_reg_table_delta,
            alu_imm_table_delta,
            alu_imm_shift_rhs_delta,
            imm_i,
            imm_s,
            imm_b,
            imm_j,
            rd_bit: [rd_b0, rd_b1, rd_b2, rd_b3, rd_b4],
            funct3_bit: [funct3_b0, funct3_b1, funct3_b2],
            rs1_bit: [rs1_b0, rs1_b1, rs1_b2, rs1_b3, rs1_b4],
            rs2_bit: [rs2_b0, rs2_b1, rs2_b2, rs2_b3, rs2_b4],
            funct7_bit: [
                funct7_b0, funct7_b1, funct7_b2, funct7_b3, funct7_b4, funct7_b5, funct7_b6,
            ],
            rd_is_zero_01,
            rd_is_zero_012,
            rd_is_zero_0123,
            rd_is_zero,
        }
    }
}

#[inline]
pub fn rv32_decode_lookup_backed_cols(layout: &Rv32DecodeSidecarLayout) -> Vec<usize> {
    let mut out = Vec::with_capacity(57);
    out.push(layout.opcode);
    out.push(layout.rs2);
    out.push(layout.rd_has_write);
    out.push(layout.ram_has_read);
    out.push(layout.ram_has_write);
    out.push(layout.shout_table_id);
    out.extend_from_slice(&[
        layout.op_lui,
        layout.op_auipc,
        layout.op_jal,
        layout.op_jalr,
        layout.op_branch,
        layout.op_load,
        layout.op_store,
        layout.op_alu_imm,
        layout.op_alu_reg,
        layout.op_misc_mem,
        layout.op_system,
        layout.op_amo,
        layout.op_custom,
    ]);
    out.extend_from_slice(&layout.funct3_is);
    out.extend_from_slice(&[layout.imm_i, layout.imm_s, layout.imm_b, layout.imm_j]);
    out.extend_from_slice(&layout.rd_bit);
    out.extend_from_slice(&layout.funct3_bit);
    out.extend_from_slice(&layout.rs1_bit);
    out.extend_from_slice(&layout.rs2_bit);
    out.extend_from_slice(&layout.funct7_bit);
    out.push(layout.rd_is_zero);
    out
}

/// Decode lookup columns transported on the shared Shout bus in trace mode.
///
/// This currently aliases the full decode-backed set. The dedicated entrypoint keeps
/// call sites explicit and allows future reduction work without touching all users.
#[inline]
pub fn rv32_decode_lookup_transport_cols(layout: &Rv32DecodeSidecarLayout) -> Vec<usize> {
    rv32_decode_lookup_backed_cols(layout)
}

#[inline]
pub const fn rv32_decode_lookup_table_id_for_col(col: usize) -> u32 {
    if RV32_TRACE_DECODE_LOOKUP_GROUPED {
        RV32_TRACE_DECODE_LOOKUP_GROUPED_TABLE_ID
    } else {
        RV32_TRACE_DECODE_LOOKUP_TABLE_BASE + col as u32
    }
}

#[inline]
pub const fn rv32_is_decode_lookup_table_id(table_id: u32) -> bool {
    (table_id >= RV32_TRACE_DECODE_LOOKUP_TABLE_BASE && table_id < RV32_TRACE_DECODE_LOOKUP_TABLE_BASE + 77)
        || table_id == RV32_TRACE_DECODE_LOOKUP_GROUPED_TABLE_ID
}

#[inline]
pub const fn rv32_is_decode_lookup_grouped_table_id(table_id: u32) -> bool {
    RV32_TRACE_DECODE_LOOKUP_GROUPED && table_id == RV32_TRACE_DECODE_LOOKUP_GROUPED_TABLE_ID
}

#[inline]
pub fn rv32_decode_lookup_transport_n_vals() -> usize {
    let layout = Rv32DecodeSidecarLayout::new();
    rv32_decode_lookup_transport_cols(&layout).len().max(1)
}

#[inline]
pub fn rv32_decode_lookup_val_slot_for_col(col: usize) -> Option<usize> {
    let layout = Rv32DecodeSidecarLayout::new();
    let cols = rv32_decode_lookup_transport_cols(&layout);
    cols.iter().position(|&c| c == col)
}

#[inline]
pub fn rv32_decode_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    if !rv32_is_decode_lookup_table_id(table_id) {
        return None;
    }
    let col_id = (table_id - RV32_TRACE_DECODE_LOOKUP_TABLE_BASE) as usize;
    let layout = Rv32DecodeSidecarLayout::new();
    let backed_cols = rv32_decode_lookup_backed_cols(&layout);
    backed_cols
        .iter()
        .any(|&c| c == col_id)
        .then_some(RV32_TRACE_DECODE_ADDR_GROUP_BASE)
}

#[inline]
fn sign_extend_to_u32(value: u32, bits: u32) -> u32 {
    debug_assert!(bits > 0 && bits <= 32);
    let shift = 32 - bits;
    (((value << shift) as i32) >> shift) as u32
}

#[inline]
fn imm_i_from_word(instr_word: u32) -> u32 {
    sign_extend_to_u32((instr_word >> 20) & 0x0fff, 12)
}

#[inline]
fn imm_s_from_word(instr_word: u32) -> u32 {
    let imm = ((instr_word >> 7) & 0x1f) | (((instr_word >> 25) & 0x7f) << 5);
    sign_extend_to_u32(imm, 12)
}

#[inline]
fn imm_b_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 12)
        | (((instr_word >> 7) & 0x1) << 11)
        | (((instr_word >> 25) & 0x3f) << 5)
        | (((instr_word >> 8) & 0xf) << 1);
    sign_extend_to_u32(imm, 13)
}

#[inline]
fn imm_j_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 20)
        | (((instr_word >> 12) & 0xff) << 12)
        | (((instr_word >> 20) & 0x1) << 11)
        | (((instr_word >> 21) & 0x3ff) << 1);
    sign_extend_to_u32(imm, 21)
}

#[inline]
fn opcode_writes_rd(opcode_u64: u64) -> bool {
    matches!(opcode_u64, 0x37 | 0x17 | 0x6F | 0x67 | 0x03 | 0x13 | 0x33)
}

pub fn rv32_decode_lookup_backed_row_from_instr_word(
    layout: &Rv32DecodeSidecarLayout,
    instr_word: u32,
    active: bool,
) -> Vec<F> {
    let mut row = vec![F::ZERO; layout.cols];
    let opcode_u64 = (instr_word & 0x7f) as u64;
    let funct3_u64 = ((instr_word >> 12) & 0x7) as u64;
    let funct7_u64 = ((instr_word >> 25) & 0x7f) as u64;
    let rd_u64 = ((instr_word >> 7) & 0x1f) as u64;
    let rs1_u64 = ((instr_word >> 15) & 0x1f) as u64;
    let rs2_u64 = ((instr_word >> 20) & 0x1f) as u64;

    row[layout.opcode] = F::from_u64(opcode_u64);
    row[layout.funct3] = F::from_u64(funct3_u64);
    row[layout.funct7] = F::from_u64(funct7_u64);
    row[layout.rd] = F::from_u64(rd_u64);
    row[layout.rs1] = F::from_u64(rs1_u64);
    row[layout.rs2] = F::from_u64(rs2_u64);
    row[layout.imm_i] = F::from_u64(imm_i_from_word(instr_word) as u64);
    row[layout.imm_s] = F::from_u64(imm_s_from_word(instr_word) as u64);
    row[layout.imm_b] = F::from_u64(imm_b_from_word(instr_word) as u64);
    row[layout.imm_j] = F::from_u64(imm_j_from_word(instr_word) as u64);
    for (k, &bit_col) in layout.rd_bit.iter().enumerate() {
        row[bit_col] = F::from_u64((rd_u64 >> k) & 1);
    }
    for (k, &bit_col) in layout.funct3_bit.iter().enumerate() {
        row[bit_col] = F::from_u64((funct3_u64 >> k) & 1);
    }
    for (k, &bit_col) in layout.rs1_bit.iter().enumerate() {
        row[bit_col] = F::from_u64((rs1_u64 >> k) & 1);
    }
    for (k, &bit_col) in layout.rs2_bit.iter().enumerate() {
        row[bit_col] = F::from_u64((rs2_u64 >> k) & 1);
    }
    for (k, &bit_col) in layout.funct7_bit.iter().enumerate() {
        row[bit_col] = F::from_u64((funct7_u64 >> k) & 1);
    }
    let one_minus_b0 = F::ONE - row[layout.rd_bit[0]];
    let one_minus_b1 = F::ONE - row[layout.rd_bit[1]];
    let one_minus_b2 = F::ONE - row[layout.rd_bit[2]];
    let one_minus_b3 = F::ONE - row[layout.rd_bit[3]];
    let one_minus_b4 = F::ONE - row[layout.rd_bit[4]];
    row[layout.rd_is_zero_01] = one_minus_b0 * one_minus_b1;
    row[layout.rd_is_zero_012] = row[layout.rd_is_zero_01] * one_minus_b2;
    row[layout.rd_is_zero_0123] = row[layout.rd_is_zero_012] * one_minus_b3;
    row[layout.rd_is_zero] = row[layout.rd_is_zero_0123] * one_minus_b4;

    let is = |op: u64| if opcode_u64 == op { F::ONE } else { F::ZERO };
    row[layout.op_lui] = is(0x37);
    row[layout.op_auipc] = is(0x17);
    row[layout.op_jal] = is(0x6F);
    row[layout.op_jalr] = is(0x67);
    row[layout.op_branch] = is(0x63);
    row[layout.op_load] = is(0x03);
    row[layout.op_store] = is(0x23);
    row[layout.op_alu_imm] = is(0x13);
    row[layout.op_alu_reg] = is(0x33);
    row[layout.op_misc_mem] = is(0x0F);
    row[layout.op_system] = is(0x73);
    row[layout.op_amo] = is(0x2F);
    row[layout.op_custom] = is(0x0B);

    let custom_squeeze_writes_rd = opcode_u64 == 0x0B && funct7_u64 == 0x02;
    let rd_has_write_f = if (opcode_writes_rd(opcode_u64) || custom_squeeze_writes_rd) && rd_u64 != 0 {
        F::ONE
    } else {
        F::ZERO
    };
    row[layout.rd_has_write] = rd_has_write_f;
    row[layout.op_lui_write] = row[layout.op_lui] * rd_has_write_f;
    row[layout.op_auipc_write] = row[layout.op_auipc] * rd_has_write_f;
    row[layout.op_jal_write] = row[layout.op_jal] * rd_has_write_f;
    row[layout.op_jalr_write] = row[layout.op_jalr] * rd_has_write_f;
    row[layout.op_alu_imm_write] = row[layout.op_alu_imm] * rd_has_write_f;
    row[layout.op_alu_reg_write] = row[layout.op_alu_reg] * rd_has_write_f;

    let is_load = opcode_u64 == 0x03;
    let is_lb = is_load && funct3_u64 == 0b000;
    let is_lh = is_load && funct3_u64 == 0b001;
    let is_lw = is_load && funct3_u64 == 0b010;
    let is_lbu = is_load && funct3_u64 == 0b100;
    let is_lhu = is_load && funct3_u64 == 0b101;
    let flag = |on: bool| if on { F::ONE } else { F::ZERO };
    row[layout.is_lb_write] = flag(is_lb) * rd_has_write_f;
    row[layout.is_lbu_write] = flag(is_lbu) * rd_has_write_f;
    row[layout.is_lh_write] = flag(is_lh) * rd_has_write_f;
    row[layout.is_lhu_write] = flag(is_lhu) * rd_has_write_f;
    row[layout.is_lw_write] = flag(is_lw) * rd_has_write_f;
    let is_store = opcode_u64 == 0x23;
    let is_sb = is_store && funct3_u64 == 0b000;
    let is_sh = is_store && funct3_u64 == 0b001;
    row[layout.ram_has_read] = if is_load || is_sb || is_sh { F::ONE } else { F::ZERO };
    row[layout.ram_has_write] = if is_store { F::ONE } else { F::ZERO };

    for (k, &f3_col) in layout.funct3_is.iter().enumerate() {
        row[f3_col] = if active && funct3_u64 == k as u64 {
            F::ONE
        } else {
            F::ZERO
        };
    }

    let funct7_b5 = (funct7_u64 >> 5) & 1;
    let is_rv32m = (funct7_u64 & 1) == 1;
    let f3_is_0 = if active && funct3_u64 == 0 { 1 } else { 0 };
    let f3_is_5 = if active && funct3_u64 == 5 { 1 } else { 0 };
    let alu_table_base: u64 = match funct3_u64 {
        0 => 3,
        1 => 7,
        2 => 5,
        3 => 6,
        4 => 1,
        5 => 8,
        6 => 2,
        _ => 0,
    };
    let alu_rv32m_table: u64 = match funct3_u64 {
        0 => 12, // MUL
        1 => 13, // MULH
        2 => 15, // MULHSU
        3 => 14, // MULHU
        4 => 16, // DIV
        5 => 17, // DIVU
        6 => 18, // REM
        7 => 19, // REMU
        _ => 0,
    };
    let alu_reg_table_id = if is_rv32m {
        alu_rv32m_table
    } else {
        alu_table_base + (funct7_b5 * (f3_is_0 + f3_is_5))
    };
    let branch_table_expected: u64 =
        10 - 5 * ((funct3_u64 >> 2) & 1) + (((funct3_u64 >> 1) & 1) * ((funct3_u64 >> 2) & 1));
    row[layout.shout_table_id] = if opcode_u64 == 0x33 {
        F::from_u64(alu_reg_table_id)
    } else if opcode_u64 == 0x13 {
        F::from_u64(alu_table_base + (funct7_b5 * f3_is_5))
    } else if opcode_u64 == 0x63 {
        F::from_u64(branch_table_expected)
    } else if matches!(opcode_u64, 0x03 | 0x23 | 0x67 | 0x17) {
        // LOAD/STORE/JALR/AUIPC use ADD shout semantics in the current trace runner.
        F::from_u64(3)
    } else {
        F::ZERO
    };
    row[layout.alu_reg_table_delta] = F::from_u64(alu_reg_table_id.wrapping_sub(alu_table_base));
    row[layout.alu_imm_table_delta] = F::from_u64(funct7_b5 * f3_is_5);

    let shift_f3_sel = row[layout.funct3_is[1]] + row[layout.funct3_is[5]];
    row[layout.alu_imm_shift_rhs_delta] = shift_f3_sel * (F::from_u64(rs2_u64) - row[layout.imm_i]);

    row
}
