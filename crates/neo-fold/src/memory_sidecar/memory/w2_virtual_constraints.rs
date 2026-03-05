use super::*;
use neo_memory::riscv::lookups::{RiscvOpcode, RiscvShoutTables};
use std::sync::OnceLock;

const W2_SELECTOR_RESIDUAL_COUNT: usize = 8;
const W2_BITNESS_RESIDUAL_COUNT: usize = 20;
const W2_ALU_BRANCH_RESIDUAL_COUNT: usize =
    W2_FIELDS_RESIDUAL_COUNT - W2_SELECTOR_RESIDUAL_COUNT - W2_BITNESS_RESIDUAL_COUNT;

#[derive(Clone, Copy, Debug)]
struct W2VirtualTableIds {
    add: u64,
    xor: u64,
    sub: u64,
    sltu: u64,
    eq: u64,
    sra: u64,
    mul: u64,
    mulh: u64,
    mulhu: u64,
    div: u64,
    divu: u64,
}

#[inline]
fn w2_virtual_table_ids() -> &'static W2VirtualTableIds {
    static IDS: OnceLock<W2VirtualTableIds> = OnceLock::new();
    IDS.get_or_init(|| {
        let tables = RiscvShoutTables::new(32);
        let id = |op| tables.opcode_to_id(op).0 as u64;
        W2VirtualTableIds {
            add: id(RiscvOpcode::Add),
            xor: id(RiscvOpcode::Xor),
            sub: id(RiscvOpcode::Sub),
            sltu: id(RiscvOpcode::Sltu),
            eq: id(RiscvOpcode::Eq),
            sra: id(RiscvOpcode::Sra),
            mul: id(RiscvOpcode::Mul),
            mulh: id(RiscvOpcode::Mulh),
            mulhu: id(RiscvOpcode::Mulhu),
            div: id(RiscvOpcode::Div),
            divu: id(RiscvOpcode::Divu),
        }
    })
}

#[derive(Clone, Copy, Debug)]
struct W2VirtualConstantsK {
    alu_table_weights: [K; 7],
    branch_base_10: K,
    branch_sub_5: K,
    movsign_rhs: K,
    v0: K,
    v1: K,
    v2: K,
    two_pow_32: K,
    rv32_all_ones: K,
    add_table_id: K,
    xor_table_id: K,
    sub_table_id: K,
    sltu_table_id: K,
    eq_table_id: K,
    sra_table_id: K,
    mul_table_id: K,
    mulh_table_id: K,
    mulhu_table_id: K,
    div_table_id: K,
    divu_table_id: K,
}

#[inline]
fn k_u64(v: u64) -> K {
    K::from(F::from_u64(v))
}

#[inline]
fn w2_virtual_constants_k() -> &'static W2VirtualConstantsK {
    static CONSTS: OnceLock<W2VirtualConstantsK> = OnceLock::new();
    CONSTS.get_or_init(|| {
        let table_ids = w2_virtual_table_ids();
        let two_pow_32 = k_u64(1u64 << 32);
        W2VirtualConstantsK {
            alu_table_weights: [k_u64(3), k_u64(7), k_u64(5), k_u64(6), k_u64(1), k_u64(8), k_u64(2)],
            branch_base_10: k_u64(10),
            branch_sub_5: k_u64(5),
            movsign_rhs: k_u64(31),
            v0: k_u64(32),
            v1: k_u64(33),
            v2: k_u64(34),
            two_pow_32,
            rv32_all_ones: two_pow_32 - K::ONE,
            add_table_id: k_u64(table_ids.add),
            xor_table_id: k_u64(table_ids.xor),
            sub_table_id: k_u64(table_ids.sub),
            sltu_table_id: k_u64(table_ids.sltu),
            eq_table_id: k_u64(table_ids.eq),
            sra_table_id: k_u64(table_ids.sra),
            mul_table_id: k_u64(table_ids.mul),
            mulh_table_id: k_u64(table_ids.mulh),
            mulhu_table_id: k_u64(table_ids.mulhu),
            div_table_id: k_u64(table_ids.div),
            divu_table_id: k_u64(table_ids.divu),
        }
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct W2DecodeFieldsOpenings {
    pub active: K,
    pub halted: K,
    pub is_virtual: K,
    pub virtual_sequence_remaining: K,
    pub trace_rs1_addr: K,
    pub trace_rs2_addr: K,
    pub trace_rd_addr: K,
    pub rs1_val: K,
    pub rs2_val: K,
    pub rd_val: K,
    pub trace_rd_has_write: K,
    pub ram_addr: K,
    pub shout_has_lookup: K,
    pub shout_table_id: K,
    pub shout_val: K,
    pub shout_lhs: K,
    pub shout_rhs: K,
    pub shout_add_sub_key: K,
    pub decode_opcode: K,
    pub decode_rs1_addr: K,
    pub decode_rs2_addr: K,
    pub decode_rd_addr: K,
    pub rd_is_zero: K,
    pub decode_rd_has_write: K,
    pub ram_has_read: K,
    pub ram_has_write: K,
    pub opcode_flags: [K; 12],
    pub op_custom: K,
    pub funct3_is: [K; 8],
    pub funct3_bits: [K; 3],
    pub funct7_bits: [K; 7],
    pub imm_i: K,
    pub imm_s: K,
}

#[inline]
pub(crate) fn w2_decode_fields_weighted_residual(openings: &W2DecodeFieldsOpenings, fields_weights: &[K]) -> K {
    let mut alu_branch_residuals = Vec::with_capacity(W2_ALU_BRANCH_RESIDUAL_COUNT);
    w2_decode_fields_weighted_residual_with_scratch(openings, fields_weights, &mut alu_branch_residuals)
}

#[inline]
pub(crate) fn w2_decode_fields_weighted_residual_with_scratch(
    openings: &W2DecodeFieldsOpenings,
    fields_weights: &[K],
    alu_branch_residuals: &mut Vec<K>,
) -> K {
    debug_assert_eq!(
        fields_weights.len(),
        W2_FIELDS_RESIDUAL_COUNT,
        "decode/fields weight length mismatch: expected {}, got {}",
        W2_FIELDS_RESIDUAL_COUNT,
        fields_weights.len()
    );

    let rd_keep = K::ONE - openings.rd_is_zero;
    let op_write_flags = [
        openings.opcode_flags[0] * rd_keep,
        openings.opcode_flags[1] * rd_keep,
        openings.opcode_flags[2] * rd_keep,
        openings.opcode_flags[3] * rd_keep,
        openings.opcode_flags[7] * rd_keep,
        openings.opcode_flags[8] * rd_keep,
    ];
    let alu_reg_table_delta = w2_alu_reg_table_delta_from_bits(openings.funct7_bits, openings.funct3_is);
    let alu_imm_table_delta = openings.funct7_bits[5] * openings.funct3_is[5];
    let alu_imm_shift_rhs_delta =
        (openings.funct3_is[1] + openings.funct3_is[5]) * (openings.decode_rs2_addr - openings.imm_i);

    let selector_residuals = w2_decode_selector_residuals(
        openings.active,
        openings.decode_opcode,
        openings.opcode_flags,
        openings.op_custom,
        openings.funct3_is,
        openings.funct3_bits,
        openings.opcode_flags[11],
    );
    let bitness_residuals = w2_decode_bitness_residuals(openings.opcode_flags, openings.funct3_is);
    w2_alu_branch_lookup_residuals_into(
        openings.active,
        openings.is_virtual,
        openings.virtual_sequence_remaining,
        openings.halted,
        openings.shout_has_lookup,
        openings.shout_lhs,
        openings.shout_rhs,
        openings.shout_add_sub_key,
        openings.shout_table_id,
        openings.trace_rs1_addr,
        openings.trace_rs2_addr,
        openings.trace_rd_addr,
        openings.decode_rs1_addr,
        openings.decode_rs2_addr,
        openings.decode_rd_addr,
        openings.rs1_val,
        openings.rs2_val,
        openings.trace_rd_has_write,
        openings.decode_rd_has_write,
        openings.rd_is_zero,
        openings.rd_val,
        openings.ram_has_read,
        openings.ram_has_write,
        openings.ram_addr,
        openings.shout_val,
        openings.funct3_bits,
        openings.funct7_bits,
        openings.opcode_flags,
        op_write_flags,
        openings.funct3_is,
        alu_reg_table_delta,
        alu_imm_table_delta,
        alu_imm_shift_rhs_delta,
        openings.decode_rs2_addr,
        openings.imm_i,
        openings.imm_s,
        alu_branch_residuals,
    );
    debug_assert!(
        alu_branch_residuals.len() <= W2_ALU_BRANCH_RESIDUAL_COUNT,
        "decode/fields alu_branch residual count overflow: expected <= {}, got {}",
        W2_ALU_BRANCH_RESIDUAL_COUNT,
        alu_branch_residuals.len()
    );

    let mut weighted = K::ZERO;
    let mut w_idx = 0usize;
    for r in selector_residuals {
        weighted += fields_weights[w_idx] * r;
        w_idx += 1;
    }
    for r in bitness_residuals {
        weighted += fields_weights[w_idx] * r;
        w_idx += 1;
    }
    let alu_branch_len = alu_branch_residuals.len();
    for &r in alu_branch_residuals.iter() {
        weighted += fields_weights[w_idx] * r;
        w_idx += 1;
    }
    if w_idx < fields_weights.len() {
        w_idx += W2_ALU_BRANCH_RESIDUAL_COUNT.saturating_sub(alu_branch_len);
    }
    debug_assert_eq!(
        w_idx,
        fields_weights.len(),
        "decode/fields residual packing mismatch: consumed {}, weights {}",
        w_idx,
        fields_weights.len()
    );
    weighted
}

const W2_STAGE_GATE_TABLE_CAP: usize = 21; // supports max_remaining up to 19 (plus sentinel)

#[inline]
fn w2_build_stage_gate_table(remaining: K, max_remaining: usize, gates: &mut [K; W2_STAGE_GATE_TABLE_CAP]) -> K {
    debug_assert!(max_remaining + 1 < W2_STAGE_GATE_TABLE_CAP);

    let mut prefix = [K::ONE; W2_STAGE_GATE_TABLE_CAP];
    let mut suffix = [K::ONE; W2_STAGE_GATE_TABLE_CAP];

    for r in 1..=max_remaining {
        prefix[r] = prefix[r - 1] * (remaining - K::from(F::from_u64(r as u64)));
    }
    suffix[max_remaining + 1] = K::ONE;
    for r in (1..=max_remaining).rev() {
        suffix[r] = suffix[r + 1] * (remaining - K::from(F::from_u64(r as u64)));
    }
    for r in 1..=max_remaining {
        gates[r] = prefix[r - 1] * suffix[r + 1];
    }

    prefix[max_remaining]
}

#[derive(Clone, Copy, Debug)]
struct VirtualStageRow {
    remaining: u64,
    rs1: K,
    rs2: K,
    rd_has_write: K,
    rd_addr: K,
    has_lookup: K,
    table_id: K,
    lhs: K,
    rhs: K,
    rd_val: K,
    extra: Option<K>,
}

#[inline]
fn push_virtual_stage_row(residuals: &mut Vec<K>, gate: K, row: VirtualStageRow) {
    residuals.push(gate * row.rs1);
    residuals.push(gate * row.rs2);
    residuals.push(gate * row.rd_has_write);
    residuals.push(gate * row.rd_addr);
    residuals.push(gate * row.has_lookup);
    residuals.push(gate * row.table_id);
    residuals.push(gate * row.lhs);
    residuals.push(gate * row.rhs);
    residuals.push(gate * row.rd_val);
    if let Some(extra) = row.extra {
        residuals.push(gate * extra);
    }
}

#[derive(Clone, Copy, Debug)]
struct VirtualStageSparseRow {
    remaining: u64,
    rs1: K,
    rs2: K,
    rd_has_write: K,
    rd_addr: Option<K>,
    has_lookup: K,
    table_id: Option<K>,
    lhs: Option<K>,
    rhs: Option<K>,
    rd_val: Option<K>,
    extra: Option<K>,
}

#[inline]
fn push_virtual_stage_sparse_row(residuals: &mut Vec<K>, gate: K, row: VirtualStageSparseRow) {
    residuals.push(gate * row.rs1);
    residuals.push(gate * row.rs2);
    residuals.push(gate * row.rd_has_write);
    if let Some(rd_addr) = row.rd_addr {
        residuals.push(gate * rd_addr);
    }
    residuals.push(gate * row.has_lookup);
    if let Some(table_id) = row.table_id {
        residuals.push(gate * table_id);
    }
    if let Some(lhs) = row.lhs {
        residuals.push(gate * lhs);
    }
    if let Some(rhs) = row.rhs {
        residuals.push(gate * rhs);
    }
    if let Some(rd_val) = row.rd_val {
        residuals.push(gate * rd_val);
    }
    if let Some(extra) = row.extra {
        residuals.push(gate * extra);
    }
}

#[inline]
#[cfg(debug_assertions)]
pub(crate) fn w2_alu_branch_lookup_residuals(
    active: K,
    is_virtual: K,
    virtual_sequence_remaining: K,
    halted: K,
    shout_has_lookup: K,
    shout_lhs: K,
    shout_rhs: K,
    shout_add_sub_key: K,
    shout_table_id: K,
    trace_rs1_addr: K,
    trace_rs2_addr: K,
    trace_rd_addr: K,
    decode_rs1_addr: K,
    decode_rs2_addr: K,
    decode_rd_addr: K,
    rs1_val: K,
    rs2_val: K,
    trace_rd_has_write: K,
    decode_rd_has_write: K,
    rd_is_zero: K,
    rd_val: K,
    ram_has_read: K,
    ram_has_write: K,
    ram_addr: K,
    shout_val: K,
    funct3_bits: [K; 3],
    funct7_bits: [K; 7],
    opcode_flags: [K; 12],
    op_write_flags: [K; 6],
    funct3_is: [K; 8],
    alu_reg_table_delta: K,
    alu_imm_table_delta: K,
    alu_imm_shift_rhs_delta: K,
    rs2_decode_addr: K,
    imm_i: K,
    imm_s: K,
) -> Vec<K> {
    let mut residuals = Vec::with_capacity(W2_ALU_BRANCH_RESIDUAL_COUNT);
    w2_alu_branch_lookup_residuals_into(
        active,
        is_virtual,
        virtual_sequence_remaining,
        halted,
        shout_has_lookup,
        shout_lhs,
        shout_rhs,
        shout_add_sub_key,
        shout_table_id,
        trace_rs1_addr,
        trace_rs2_addr,
        trace_rd_addr,
        decode_rs1_addr,
        decode_rs2_addr,
        decode_rd_addr,
        rs1_val,
        rs2_val,
        trace_rd_has_write,
        decode_rd_has_write,
        rd_is_zero,
        rd_val,
        ram_has_read,
        ram_has_write,
        ram_addr,
        shout_val,
        funct3_bits,
        funct7_bits,
        opcode_flags,
        op_write_flags,
        funct3_is,
        alu_reg_table_delta,
        alu_imm_table_delta,
        alu_imm_shift_rhs_delta,
        rs2_decode_addr,
        imm_i,
        imm_s,
        &mut residuals,
    );
    residuals
}

#[inline]
pub(crate) fn w2_alu_branch_lookup_residuals_into(
    active: K,
    is_virtual: K,
    virtual_sequence_remaining: K,
    halted: K,
    shout_has_lookup: K,
    shout_lhs: K,
    shout_rhs: K,
    shout_add_sub_key: K,
    shout_table_id: K,
    trace_rs1_addr: K,
    trace_rs2_addr: K,
    trace_rd_addr: K,
    decode_rs1_addr: K,
    decode_rs2_addr: K,
    decode_rd_addr: K,
    rs1_val: K,
    rs2_val: K,
    trace_rd_has_write: K,
    decode_rd_has_write: K,
    rd_is_zero: K,
    rd_val: K,
    ram_has_read: K,
    ram_has_write: K,
    ram_addr: K,
    shout_val: K,
    funct3_bits: [K; 3],
    funct7_bits: [K; 7],
    opcode_flags: [K; 12],
    op_write_flags: [K; 6],
    funct3_is: [K; 8],
    alu_reg_table_delta: K,
    alu_imm_table_delta: K,
    alu_imm_shift_rhs_delta: K,
    rs2_decode_addr: K,
    imm_i: K,
    imm_s: K,
    residuals: &mut Vec<K>,
) {
    let op_lui = opcode_flags[0];
    let op_auipc = opcode_flags[1];
    let op_jal = opcode_flags[2];
    let op_jalr = opcode_flags[3];
    let op_branch = opcode_flags[4];
    let op_load = opcode_flags[5];
    let op_store = opcode_flags[6];
    let op_alu_imm = opcode_flags[7];
    let op_alu_reg = opcode_flags[8];
    let op_misc_mem = opcode_flags[9];
    let op_system = opcode_flags[10];

    let op_lui_write = op_write_flags[0];
    let op_auipc_write = op_write_flags[1];
    let op_jal_write = op_write_flags[2];
    let op_jalr_write = op_write_flags[3];
    let op_alu_imm_write = op_write_flags[4];
    let op_alu_reg_write = op_write_flags[5];

    let non_mem_ops =
        op_lui + op_auipc + op_jal + op_jalr + op_branch + op_alu_imm + op_alu_reg + op_misc_mem + op_system;
    let mem_lookup_ops = op_load + op_store;
    let add_lookup_ops = op_load + op_store + op_jalr;
    let k_consts = w2_virtual_constants_k();
    let add_table_id = k_consts.add_table_id;

    let alu_table_base = k_consts.alu_table_weights[0] * funct3_is[0]
        + k_consts.alu_table_weights[1] * funct3_is[1]
        + k_consts.alu_table_weights[2] * funct3_is[2]
        + k_consts.alu_table_weights[3] * funct3_is[3]
        + k_consts.alu_table_weights[4] * funct3_is[4]
        + k_consts.alu_table_weights[5] * funct3_is[5]
        + k_consts.alu_table_weights[6] * funct3_is[6];
    let branch_table_expected =
        k_consts.branch_base_10 - k_consts.branch_sub_5 * funct3_bits[2] + (funct3_bits[1] * funct3_bits[2]);
    let shift_selector = funct3_is[1] + funct3_is[5];
    let funct7_m_tail =
        funct7_bits[1] + funct7_bits[2] + funct7_bits[3] + funct7_bits[4] + funct7_bits[5] + funct7_bits[6];
    let alu_reg_table_delta_expected = w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is);

    let op_add_imm = op_alu_imm * funct3_is[0];
    let op_add_reg = op_alu_reg * funct3_is[0] * (K::ONE - funct7_bits[0]) * (K::ONE - funct7_bits[5]);
    let op_sub_reg = op_alu_reg * funct3_is[0] * (K::ONE - funct7_bits[0]) * funct7_bits[5];
    let op_mul_reg = op_alu_reg * funct3_is[0] * funct7_bits[0];
    let op_mulhu_reg = op_alu_reg * funct3_is[3] * funct7_bits[0];
    let op_add_total = add_lookup_ops + op_add_imm + op_add_reg;
    let two_pow_32 = k_consts.two_pow_32;
    let add_key_delta = shout_lhs + shout_rhs - shout_add_sub_key;
    let sub_key_delta = shout_lhs - shout_rhs - shout_add_sub_key;
    let mul_key_delta = shout_lhs * shout_rhs - shout_add_sub_key;
    let add_sub_combined_key_mode = if neo_memory::riscv::instruction::opcode_uses_combined_lookup_key(RiscvOpcode::Add)
    {
        K::ONE
    } else {
        K::ZERO
    };
    let mul_combined_key_mode = if neo_memory::riscv::instruction::opcode_uses_combined_lookup_key(RiscvOpcode::Mul) {
        K::ONE
    } else {
        K::ZERO
    };

    let raw = [
        (op_alu_imm + op_load + op_jalr) * (shout_has_lookup - K::ONE),
        (op_alu_reg + op_store) * (shout_has_lookup - K::ONE),
        op_branch * (shout_has_lookup - K::ONE),
        (K::ONE - shout_has_lookup) * shout_table_id,
        (op_alu_imm + op_alu_reg + op_branch + mem_lookup_ops + op_jalr) * (shout_lhs - rs1_val),
        alu_imm_shift_rhs_delta - shift_selector * (rs2_decode_addr - imm_i),
        op_alu_imm * (shout_rhs - imm_i - alu_imm_shift_rhs_delta) + (op_load + op_jalr) * (shout_rhs - imm_i),
        op_alu_reg * (shout_rhs - rs2_val) + op_store * (shout_rhs - imm_s),
        op_branch * (shout_rhs - rs2_val),
        op_alu_imm_write * (rd_val - shout_val),
        op_alu_reg_write * (rd_val - shout_val),
        op_alu_reg * (shout_table_id - alu_table_base - alu_reg_table_delta)
            + op_store * (shout_table_id - add_table_id),
        op_alu_imm * (shout_table_id - alu_table_base - alu_imm_table_delta)
            + add_lookup_ops * (shout_table_id - add_table_id),
        op_branch * (shout_table_id - branch_table_expected),
        op_alu_reg * funct7_bits[0] * funct7_m_tail,
        alu_reg_table_delta - alu_reg_table_delta_expected,
        alu_imm_table_delta - funct7_bits[5] * funct3_is[5],
        add_sub_combined_key_mode * op_add_total * add_key_delta * (add_key_delta - two_pow_32),
        add_sub_combined_key_mode * op_sub_reg * sub_key_delta * (sub_key_delta + two_pow_32),
        mul_combined_key_mode * (op_mul_reg + op_mulhu_reg) * mul_key_delta,
        trace_rs1_addr - decode_rs1_addr,
        trace_rs2_addr - decode_rs2_addr,
        // `rd` field bits are not semantically an architectural destination on opcodes
        // without a register write (e.g. branch/store). Only link rd_addr when decode
        // indicates a real destination write.
        decode_rd_has_write * (trace_rd_addr - decode_rd_addr),
        trace_rd_has_write - decode_rd_has_write,
        op_lui * decode_rd_has_write - op_lui_write,
        op_auipc * decode_rd_has_write - op_auipc_write,
        op_jal * decode_rd_has_write - op_jal_write,
        op_jalr * decode_rd_has_write - op_jalr_write,
        op_alu_imm * decode_rd_has_write - op_alu_imm_write,
        op_alu_reg * decode_rd_has_write - op_alu_reg_write,
        op_lui * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_auipc * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_jal * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_jalr * (decode_rd_has_write + rd_is_zero - K::ONE),
        opcode_flags[5] * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_alu_imm * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_alu_reg * (decode_rd_has_write + rd_is_zero - K::ONE),
        op_branch * decode_rd_has_write,
        opcode_flags[6] * decode_rd_has_write,
        op_misc_mem * decode_rd_has_write,
        op_system * decode_rd_has_write,
        active * (halted - op_system),
        opcode_flags[5] * (ram_has_read - K::ONE),
        opcode_flags[6] * (ram_has_write - K::ONE),
        non_mem_ops * ram_has_read,
        non_mem_ops * ram_has_write,
        non_mem_ops * ram_addr,
        // RV32 effective addresses are modular (u32 wraparound). We therefore bind RAM addr to
        // ADD lookup output (`shout_val`) instead of raw field addition `rs1 + imm`.
        op_load * (ram_addr - shout_val),
        op_store * (ram_addr - shout_val),
    ];
    let non_virtual = K::ONE - is_virtual;
    residuals.clear();
    if residuals.capacity() < W2_ALU_BRANCH_RESIDUAL_COUNT {
        residuals.reserve(W2_ALU_BRANCH_RESIDUAL_COUNT - residuals.capacity());
    }
    for r in raw {
        residuals.push(non_virtual * r);
    }

    // Virtual-stage shape + semantic constraints for signed multiply decomposition paths.
    let is_rv32m = op_alu_reg * funct7_bits[0];
    let op_mul = is_rv32m * funct3_is[0];
    let op_mulh = is_rv32m * funct3_is[1];
    let op_mulhsu = is_rv32m * funct3_is[2];
    let op_mulhu = is_rv32m * funct3_is[3];
    let op_div = is_rv32m * funct3_is[4];
    let op_divu = is_rv32m * funct3_is[5];
    let op_rem = is_rv32m * funct3_is[6];
    let op_remu = is_rv32m * funct3_is[7];
    let op_virtual_decomp = op_mul + op_mulh + op_mulhu + op_mulhsu + op_div + op_divu + op_rem + op_remu;
    let rem = virtual_sequence_remaining;
    let v0 = k_consts.v0;
    let v1 = k_consts.v1;
    let v2 = k_consts.v2;
    let movsign_rhs = k_consts.movsign_rhs;
    let sra_table_id = k_consts.sra_table_id;
    let mul_table_id = k_consts.mul_table_id;
    let mulh_table_id = k_consts.mulh_table_id;
    let mulhu_table_id = k_consts.mulhu_table_id;
    let xor_table_id = k_consts.xor_table_id;
    let sub_table_id = k_consts.sub_table_id;
    let sltu_table_id = k_consts.sltu_table_id;
    let eq_table_id = k_consts.eq_table_id;
    let div_table_id = k_consts.div_table_id;
    let divu_table_id = k_consts.divu_table_id;
    let rv32_all_ones = k_consts.rv32_all_ones;

    let virtual_mulh = is_virtual * op_mulh;
    let virtual_mulhsu = is_virtual * op_mulhsu;
    let virtual_div = is_virtual * op_div;
    let virtual_divu = is_virtual * op_divu;
    let virtual_rem = is_virtual * op_rem;
    let virtual_remu = is_virtual * op_remu;
    let has_virtual_stage = virtual_mulh != K::ZERO
        || virtual_mulhsu != K::ZERO
        || virtual_div != K::ZERO
        || virtual_divu != K::ZERO
        || virtual_rem != K::ZERO
        || virtual_remu != K::ZERO;

    residuals.push(is_virtual * (K::ONE - op_virtual_decomp));
    if !has_virtual_stage {
        return;
    }

    let mut stage_gate_7 = [K::ZERO; W2_STAGE_GATE_TABLE_CAP];
    let mut stage_gate_8 = [K::ZERO; W2_STAGE_GATE_TABLE_CAP];
    let mut stage_gate_11 = [K::ZERO; W2_STAGE_GATE_TABLE_CAP];
    let mut stage_gate_18 = [K::ZERO; W2_STAGE_GATE_TABLE_CAP];
    let mut stage_gate_19 = [K::ZERO; W2_STAGE_GATE_TABLE_CAP];
    let rem_poly_7 = w2_build_stage_gate_table(rem, 7, &mut stage_gate_7);
    let rem_poly_8 = w2_build_stage_gate_table(rem, 8, &mut stage_gate_8);
    let rem_poly_11 = w2_build_stage_gate_table(rem, 11, &mut stage_gate_11);
    let rem_poly_18 = w2_build_stage_gate_table(rem, 18, &mut stage_gate_18);
    let rem_poly_19 = w2_build_stage_gate_table(rem, 19, &mut stage_gate_19);

    residuals.push(virtual_mulh * rem_poly_7);
    residuals.push(virtual_mulhsu * rem_poly_11);

    let add_stage_key = add_sub_combined_key_mode * add_key_delta * (add_key_delta - two_pow_32);
    let sub_stage_key = add_sub_combined_key_mode * sub_key_delta * (sub_key_delta + two_pow_32);
    let mul_stage_key = mul_combined_key_mode * mul_key_delta;

    // MULH virtual rows (remaining = 7..1)
    let mulh_rows = [
        VirtualStageRow {
            remaining: 7,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sra_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - movsign_rhs,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 6,
            rs1: trace_rs1_addr - decode_rs2_addr,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v1,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sra_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - movsign_rhs,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 5,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mul_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 4,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - decode_rs1_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v1,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mul_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 3,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mulhu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 2,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - add_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(add_stage_key),
        },
        VirtualStageRow {
            remaining: 1,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v1,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - add_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(add_stage_key),
        },
    ];
    for row in mulh_rows {
        let gate = virtual_mulh * stage_gate_7[row.remaining as usize];
        push_virtual_stage_row(residuals, gate, row);
    }

    // MULHSU virtual rows (remaining = 11..1), Jolt-equivalent expanded sequence.
    // v0=sign/sum/carry, v1=one-bit mask, v2=abs/product temp, v3=high/final accumulator.
    let v3_mulhsu = K::from(F::from_u64(35));
    let mulhsu_rows = [
        VirtualStageRow {
            remaining: 11,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sra_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - movsign_rhs,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        // ANDI(v0, 1) as SUB(x0, v0): for v0 in {0, 0xFFFF_FFFF}, this yields {0,1}.
        VirtualStageRow {
            remaining: 10,
            rs1: trace_rs1_addr,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v1,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sub_table_id,
            lhs: shout_lhs,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(sub_stage_key),
        },
        VirtualStageRow {
            remaining: 9,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - xor_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 8,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v1,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - add_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(add_stage_key),
        },
        VirtualStageRow {
            remaining: 7,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v3_mulhsu,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mulhu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 6,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mul_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 5,
            rs1: trace_rs1_addr - v3_mulhsu,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v3_mulhsu,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - xor_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 4,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - xor_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 3,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v1,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - add_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(add_stage_key),
        },
        VirtualStageRow {
            remaining: 2,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - v2,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sltu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 1,
            rs1: trace_rs1_addr - v3_mulhsu,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v3_mulhsu,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - add_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(add_stage_key),
        },
    ];
    for row in mulhsu_rows {
        let gate = virtual_mulhsu * stage_gate_11[row.remaining as usize];
        push_virtual_stage_row(residuals, gate, row);
    }

    // DIV/DIVU/REM/REMU virtual-stage shape + semantic constraints.
    let v3 = K::from(F::from_u64(35));
    let v4 = K::from(F::from_u64(36));
    let v5 = K::from(F::from_u64(37));
    let v6 = K::from(F::from_u64(38));
    let v7 = K::from(F::from_u64(39));

    residuals.push(virtual_div * rem_poly_18);
    residuals.push(virtual_divu * rem_poly_8);
    residuals.push(virtual_rem * rem_poly_19);
    residuals.push(virtual_remu * rem_poly_7);

    // DIV (remaining = 18..1), Jolt-style signed path:
    // v0=q, v1=|r|, v2=adj_div, v3=mulhi, v4=prod/sum, v5=tmp_sign, v6=r_signed, v7=|adj_div|.
    let div_rows = [
        VirtualStageSparseRow {
            remaining: 18,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v0),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - div_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 17,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v1),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: None,
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 16,
            rs1: trace_rs1_addr - decode_rs2_addr,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs),
            rd_val: Some(shout_val * (rs2_val - rv32_all_ones)),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 15,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v2),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: Some((rd_val - rs2_val) * (rs2_val - rv32_all_ones)),
            extra: Some((rd_val - rs2_val) * (rd_val - K::ONE)),
        },
        VirtualStageSparseRow {
            remaining: 14,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - v2,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v3),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - mulh_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 13,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - v2,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v4),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - mul_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(mul_combined_key_mode * mul_key_delta),
        },
        VirtualStageSparseRow {
            remaining: 12,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 11,
            rs1: trace_rs1_addr - v3,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(shout_val - K::ONE),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 10,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 9,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - xor_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 8,
            rs1: trace_rs1_addr - v6,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sub_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * sub_key_delta * (sub_key_delta + two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 7,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr - v6,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v4),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - add_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * add_key_delta * (add_key_delta - two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 6,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr - decode_rs1_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(shout_val - K::ONE),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 5,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 4,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v7),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - xor_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 3,
            rs1: trace_rs1_addr - v7,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v7),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sub_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * sub_key_delta * (sub_key_delta + two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 2,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - v7,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sltu_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rs2_val * (K::ONE - shout_val)),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 1,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v0),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: Some(rd_val - rs1_val),
            extra: None,
        },
    ];
    for row in div_rows {
        let gate = virtual_div * stage_gate_18[row.remaining as usize];
        push_virtual_stage_sparse_row(residuals, gate, row);
    }

    // REM (remaining = 19..1), Jolt-style signed path with final virtual self-moves:
    // v0=q, v1=|r|, v2=adj_div, v3=mulhi, v4=prod/sum, v5=tmp_sign, v6=r_signed, v7=|adj_div|.
    let rem_rows = [
        VirtualStageSparseRow {
            remaining: 19,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v0),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - div_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 18,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v1),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: None,
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 17,
            rs1: trace_rs1_addr - decode_rs2_addr,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs),
            rd_val: Some(shout_val * (rs2_val - rv32_all_ones)),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 16,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v2),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: Some((rd_val - rs2_val) * (rs2_val - rv32_all_ones)),
            extra: Some((rd_val - rs2_val) * (rd_val - K::ONE)),
        },
        VirtualStageSparseRow {
            remaining: 15,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - v2,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v3),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - mulh_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 14,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - v2,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v4),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - mul_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(mul_combined_key_mode * mul_key_delta),
        },
        VirtualStageSparseRow {
            remaining: 13,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 12,
            rs1: trace_rs1_addr - v3,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(shout_val - K::ONE),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 11,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 10,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - xor_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 9,
            rs1: trace_rs1_addr - v6,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sub_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * sub_key_delta * (sub_key_delta + two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 8,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr - v6,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v4),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - add_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * add_key_delta * (add_key_delta - two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 7,
            rs1: trace_rs1_addr - v4,
            rs2: trace_rs2_addr - decode_rs1_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - eq_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(shout_val - K::ONE),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 6,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v5),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sra_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - movsign_rhs),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 5,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v7),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - xor_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 4,
            rs1: trace_rs1_addr - v7,
            rs2: trace_rs2_addr - v5,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v7),
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sub_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rd_val - shout_val),
            extra: Some(add_sub_combined_key_mode * sub_key_delta * (sub_key_delta + two_pow_32)),
        },
        VirtualStageSparseRow {
            remaining: 3,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - v7,
            rd_has_write: trace_rd_has_write,
            rd_addr: None,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: Some(shout_table_id - sltu_table_id),
            lhs: Some(shout_lhs - rs1_val),
            rhs: Some(shout_rhs - rs2_val),
            rd_val: Some(rs2_val * (K::ONE - shout_val)),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 2,
            rs1: trace_rs1_addr - v6,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: Some(rd_val - rs1_val),
            extra: None,
        },
        VirtualStageSparseRow {
            remaining: 1,
            rs1: trace_rs1_addr - v6,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: Some(trace_rd_addr - v6),
            has_lookup: shout_has_lookup,
            table_id: None,
            lhs: None,
            rhs: None,
            rd_val: Some(rd_val - rs1_val),
            extra: None,
        },
    ];
    for row in rem_rows {
        let gate = virtual_rem * stage_gate_19[row.remaining as usize];
        push_virtual_stage_sparse_row(residuals, gate, row);
    }

    // DIVU (remaining = 8..1), v_q=v0..v_rem=v2.
    let divu_rows = [
        VirtualStageRow {
            remaining: 8,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - divu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 7,
            rs1: trace_rs1_addr - decode_rs2_addr,
            rs2: trace_rs2_addr - v0,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - eq_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs,
            rd_val: shout_val * (rs2_val - rv32_all_ones),
            extra: None,
        },
        VirtualStageRow {
            remaining: 6,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mulhu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 5,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v1,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mul_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 4,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - decode_rs1_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sltu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: (rs2_val - rs1_val) * (K::ONE - shout_val),
            extra: None,
        },
        VirtualStageRow {
            remaining: 3,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - v1,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sub_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(sub_stage_key),
        },
        VirtualStageRow {
            remaining: 2,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sltu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rs2_val * (K::ONE - shout_val),
            extra: None,
        },
        VirtualStageRow {
            remaining: 1,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup,
            table_id: K::ZERO,
            lhs: K::ZERO,
            rhs: K::ZERO,
            rd_val: rd_val - rs1_val,
            extra: None,
        },
    ];
    for row in divu_rows {
        let gate = virtual_divu * stage_gate_8[row.remaining as usize];
        push_virtual_stage_row(residuals, gate, row);
    }

    // REMU (remaining = 7..1), v_q=v0..v_rem=v2.
    let remu_rows = [
        VirtualStageRow {
            remaining: 7,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v0,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - divu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: None,
        },
        VirtualStageRow {
            remaining: 6,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mulhu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 5,
            rs1: trace_rs1_addr - v0,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v1,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - mul_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(mul_stage_key),
        },
        VirtualStageRow {
            remaining: 4,
            rs1: trace_rs1_addr - v1,
            rs2: trace_rs2_addr - decode_rs1_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sltu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: (rs2_val - rs1_val) * (K::ONE - shout_val),
            extra: None,
        },
        VirtualStageRow {
            remaining: 3,
            rs1: trace_rs1_addr - decode_rs1_addr,
            rs2: trace_rs2_addr - v1,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sub_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rd_val - shout_val,
            extra: Some(sub_stage_key),
        },
        VirtualStageRow {
            remaining: 2,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr - decode_rs2_addr,
            rd_has_write: trace_rd_has_write,
            rd_addr: K::ZERO,
            has_lookup: shout_has_lookup - K::ONE,
            table_id: shout_table_id - sltu_table_id,
            lhs: shout_lhs - rs1_val,
            rhs: shout_rhs - rs2_val,
            rd_val: rs2_val * (K::ONE - shout_val),
            extra: None,
        },
        VirtualStageRow {
            remaining: 1,
            rs1: trace_rs1_addr - v2,
            rs2: trace_rs2_addr,
            rd_has_write: trace_rd_has_write - K::ONE,
            rd_addr: trace_rd_addr - v2,
            has_lookup: shout_has_lookup,
            table_id: K::ZERO,
            lhs: K::ZERO,
            rhs: K::ZERO,
            rd_val: rd_val - rs1_val,
            extra: None,
        },
    ];
    for row in remu_rows {
        let gate = virtual_remu * stage_gate_7[row.remaining as usize];
        push_virtual_stage_row(residuals, gate, row);
    }

    debug_assert_eq!(
        residuals.len(),
        W2_ALU_BRANCH_RESIDUAL_COUNT,
        "decode/fields alu_branch residual count mismatch: expected {}, got {}",
        W2_ALU_BRANCH_RESIDUAL_COUNT,
        residuals.len()
    );
}
