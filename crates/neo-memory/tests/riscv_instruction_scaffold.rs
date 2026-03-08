use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::instruction::{
    encode_lookup_key_with_mode, opcode_operand_mode, opcode_uses_combined_lookup_key, try_decode_lookup_operands,
    DecomposedOp, OperandMode, VirtualRegisterAllocator, VIRTUAL_REG_BASE,
};
use neo_memory::riscv::lookups::{
    compute_op, decode_program, encode_program, interleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID, REG_ID,
};
use neo_vm_trace::{trace_program, Twist, TwistOpKind};
use std::collections::HashMap;

fn simulate_decomposition_sequence(seq: &[DecomposedOp], regs: &mut [u64; 32], xlen: usize) -> HashMap<u64, u64> {
    let mut vregs = HashMap::<u64, u64>::new();
    let mask = if xlen == 32 { u32::MAX as u64 } else { u64::MAX };

    let read = |addr: u64, regs: &[u64; 32], vregs: &HashMap<u64, u64>| -> u64 {
        if addr < 32 {
            regs[addr as usize]
        } else {
            *vregs.get(&addr).unwrap_or(&0)
        }
    };
    let write = |addr: u64, value: u64, regs: &mut [u64; 32], vregs: &mut HashMap<u64, u64>| {
        let value = value & mask;
        if addr == 0 {
            return;
        }
        if addr < 32 {
            regs[addr as usize] = value;
        } else {
            vregs.insert(addr, value);
        }
    };

    for op in seq {
        match *op {
            DecomposedOp::MovSignWord { dst, src } => {
                let x = read(src, regs, &vregs) as u32 as u64;
                let sign_mask = if ((x >> 31) & 1) == 1 { 0xFFFF_FFFF } else { 0 };
                write(dst, sign_mask, regs, &mut vregs);
            }
            DecomposedOp::MovSign { dst, src } => {
                let x = read(src, regs, &vregs) & mask;
                let sign_set = if xlen == 32 {
                    ((x >> 31) & 1) == 1
                } else {
                    ((x >> 63) & 1) == 1
                };
                let sign_mask = if sign_set { mask } else { 0 };
                write(dst, sign_mask, regs, &mut vregs);
            }
            DecomposedOp::ComposeU64FromLoHi32 { dst, lo_src, hi_src } => {
                let lo = read(lo_src, regs, &vregs) as u32 as u64;
                let hi = read(hi_src, regs, &vregs) as u32 as u64;
                write(dst, lo | (hi << 32), regs, &mut vregs);
            }
            DecomposedOp::Move { dst, src } => {
                let x = read(src, regs, &vregs);
                write(dst, x, regs, &mut vregs);
            }
            DecomposedOp::Add { dst, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                write(dst, x.wrapping_add(y), regs, &mut vregs);
            }
            DecomposedOp::Sub { dst, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                write(dst, x.wrapping_sub(y), regs, &mut vregs);
            }
            DecomposedOp::Xor { dst, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                write(dst, x ^ y, regs, &mut vregs);
            }
            DecomposedOp::Mul { dst, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                write(dst, x.wrapping_mul(y), regs, &mut vregs);
            }
            DecomposedOp::Mulhu { dst, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                let hi = (((x as u128) * (y as u128)) >> xlen) as u64;
                write(dst, hi, regs, &mut vregs);
            }
            DecomposedOp::AdviceQuotient { dst, op, lhs, rhs } => {
                let x = read(lhs, regs, &vregs) & mask;
                let y = read(rhs, regs, &vregs) & mask;
                write(dst, compute_op(op, x, y, xlen), regs, &mut vregs);
            }
            DecomposedOp::Advice { .. }
            | DecomposedOp::AdviceRemainderAbs { .. }
            | DecomposedOp::AssertEq { .. }
            | DecomposedOp::AssertLtu { .. }
            | DecomposedOp::AssertLte { .. }
            | DecomposedOp::AssertLtAbs { .. }
            | DecomposedOp::AssertEqSigns { .. }
            | DecomposedOp::AssertValidDiv0 { .. }
            | DecomposedOp::ChangeDivisor { .. }
            | DecomposedOp::AssertMulUNoOverflow { .. }
            | DecomposedOp::AssertValidUnsignedRemainder { .. } => {
                panic!("unsupported op in decomposition sequence simulator test")
            }
        }
    }

    vregs
}

#[test]
fn opcode_operand_mode_scaffold_mapping_matches_expectations() {
    assert_eq!(opcode_operand_mode(RiscvOpcode::Add), OperandMode::AddOperands);
    assert_eq!(opcode_operand_mode(RiscvOpcode::Sub), OperandMode::SubtractOperands);
    assert_eq!(opcode_operand_mode(RiscvOpcode::Mul), OperandMode::MultiplyOperands);
    assert_eq!(opcode_operand_mode(RiscvOpcode::Div), OperandMode::MultiplyOperands);
    assert_eq!(opcode_operand_mode(RiscvOpcode::And), OperandMode::Interleaved);
}

#[test]
fn operand_mode_key_helpers_preserve_compat_and_define_rollout_behavior() {
    let lhs = 7u64;
    let rhs = 5u64;

    // Compatibility mode keeps canonical interleaved keys for all opcodes.
    let add_key_compat =
        encode_lookup_key_with_mode(RiscvOpcode::Add, lhs, rhs, 32, /*use_operand_mode_keys=*/ false);
    assert_eq!(add_key_compat, interleave_bits(lhs, rhs));
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Add,
            add_key_compat as u128,
            /*use_operand_mode_keys=*/ false,
            /*xlen=*/ 32
        ),
        Some((lhs, rhs))
    );

    // Rollout mode: ADD/SUB move to combined keys and are no longer key-decodable.
    let add_key_rollout =
        encode_lookup_key_with_mode(RiscvOpcode::Add, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);
    assert_eq!(add_key_rollout, lhs.wrapping_add(rhs) as u128);
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Add,
            add_key_rollout as u128,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 32
        ),
        None
    );

    let sub_key_rollout =
        encode_lookup_key_with_mode(RiscvOpcode::Sub, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);
    assert_eq!(sub_key_rollout, lhs.wrapping_sub(rhs) as u128);
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Sub,
            sub_key_rollout as u128,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 32
        ),
        None
    );

    // Interleaved-mode opcodes stay decodable in rollout mode.
    let xor_key_rollout =
        encode_lookup_key_with_mode(RiscvOpcode::Xor, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);
    assert_eq!(xor_key_rollout, interleave_bits(lhs, rhs));
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Xor,
            xor_key_rollout as u128,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 32
        ),
        Some((lhs, rhs))
    );

    // MUL/MULHU stay interleaved in rollout mode.
    let mul_key_rollout =
        encode_lookup_key_with_mode(RiscvOpcode::Mul, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);
    assert_eq!(mul_key_rollout, interleave_bits(lhs, rhs));
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Mul,
            mul_key_rollout as u128,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 32
        ),
        Some((lhs, rhs))
    );

    let mulhu_key_rollout =
        encode_lookup_key_with_mode(RiscvOpcode::Mulhu, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);
    assert_eq!(mulhu_key_rollout, interleave_bits(lhs, rhs));
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Mulhu,
            mulhu_key_rollout as u128,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 32
        ),
        Some((lhs, rhs))
    );
}

#[test]
fn rv64_virtual_word_helper_keys_are_normalized_to_low32_operands() {
    let lhs = 0xFFFF_FFFF_8000_0001u64;
    let rhs = 0x1234_5678_FFFF_FFFEu64;
    let lhs32 = lhs as u32 as u64;
    let rhs32 = rhs as u32 as u64;

    for op in [
        RiscvOpcode::VirtualMulWord,
        RiscvOpcode::VirtualDivuWord,
        RiscvOpcode::VirtualRemuWord,
        RiscvOpcode::VirtualDivWord,
        RiscvOpcode::VirtualRemWord,
        RiscvOpcode::VirtualMovsignWord,
    ] {
        let key = encode_lookup_key_with_mode(op, lhs, rhs, 64, /*use_operand_mode_keys=*/ true);
        assert_eq!(
            key,
            interleave_bits(lhs32, rhs32),
            "helper op {op:?} must key on low-32 operands only"
        );
        assert_eq!(
            try_decode_lookup_operands(op, key as u128, /*use_operand_mode_keys=*/ true, /*xlen=*/ 64),
            Some((lhs32, rhs32)),
            "helper op {op:?} must decode low-32 operands from its key"
        );
    }
}

#[test]
fn rv64_direct_64bit_lookup_keys_are_exact_under_u128_transport() {
    let lhs = 1u64;
    let rhs = 0x1_0000_0001u64;
    let key1 = encode_lookup_key_with_mode(RiscvOpcode::Eq, lhs, lhs, 64, /*use_operand_mode_keys=*/ true);
    let key2 = encode_lookup_key_with_mode(RiscvOpcode::Eq, lhs, rhs, 64, /*use_operand_mode_keys=*/ true);
    assert_ne!(
        key1, key2,
        "u128 transport must distinguish full-width RV64 operand pairs"
    );
    assert_eq!(
        try_decode_lookup_operands(
            RiscvOpcode::Eq,
            key2,
            /*use_operand_mode_keys=*/ true,
            /*xlen=*/ 64
        ),
        Some((lhs, rhs)),
        "RV64 direct lookup key must round-trip exact operands under u128 transport"
    );
}

#[test]
#[should_panic(expected = "use u128 shout keys")]
fn rv64_legacy_u64_shout_transport_rejects_full_width_direct_keys() {
    let mut shout = RiscvShoutTables::new(64);
    let key = encode_lookup_key_with_mode(
        RiscvOpcode::Eq,
        1,
        0x1_0000_0001,
        64,
        /*use_operand_mode_keys=*/ true,
    );
    let key_u64 = key as u64;
    let eq_id = shout.opcode_to_id(RiscvOpcode::Eq);
    let _ = neo_vm_trace::Shout::lookup(&mut shout, eq_id, key_u64);
}

#[test]
fn rv64_helper_owned_w_ops_have_no_direct_shout_ids() {
    let shout = RiscvShoutTables::new(64);
    for op in [
        RiscvOpcode::Mulw,
        RiscvOpcode::Divw,
        RiscvOpcode::Divuw,
        RiscvOpcode::Remw,
        RiscvOpcode::Remuw,
    ] {
        assert_eq!(
            shout.try_opcode_to_id(op),
            None,
            "helper-owned RV64 W op {op:?} must not expose a direct Shout table id"
        );
    }
}

#[test]
fn rv64_virtual_word_helper_lookup_results_ignore_high64_signfill() {
    let lhs = 0xFFFF_FFFF_8000_0000u64;
    let rhs = 0xFFFF_FFFF_FFFF_FFFDu64;
    let lhs32 = lhs as u32 as u64;
    let rhs32 = rhs as u32 as u64;

    let mut shout = RiscvShoutTables::new(64);
    for op in [
        RiscvOpcode::VirtualMulWord,
        RiscvOpcode::VirtualDivuWord,
        RiscvOpcode::VirtualRemuWord,
        RiscvOpcode::VirtualDivWord,
        RiscvOpcode::VirtualRemWord,
    ] {
        let shout_id = shout.opcode_to_id(op);
        let key = encode_lookup_key_with_mode(op, lhs, rhs, 64, /*use_operand_mode_keys=*/ true);
        let out = neo_vm_trace::Shout::lookup(&mut shout, shout_id, key);
        assert_eq!(
            out,
            compute_op(op, lhs32, rhs32, 64),
            "helper op {op:?} must evaluate from low-32 normalized operands"
        );
    }

    let movsign_id = shout.opcode_to_id(RiscvOpcode::VirtualMovsignWord);
    let movsign_key = encode_lookup_key_with_mode(
        RiscvOpcode::VirtualMovsignWord,
        lhs,
        rhs,
        64,
        /*use_operand_mode_keys=*/ true,
    );
    let movsign_out = neo_vm_trace::Shout::lookup(&mut shout, movsign_id, movsign_key);
    let expected_movsign = if ((lhs32 >> 31) & 1) == 1 { 0xFFFF_FFFF } else { 0 };
    assert_eq!(
        movsign_out, expected_movsign,
        "VMOVSIGNW must emit the low-word sign mask"
    );
}

#[test]
fn mul_lookup_semantics_match_expected_words_in_rollout_mode() {
    let lhs = 0xFFFF_FFFFu64;
    let rhs = 0xFFFF_FFFDu64;
    let key = encode_lookup_key_with_mode(RiscvOpcode::Mul, lhs, rhs, 32, /*use_operand_mode_keys=*/ true);

    let mut shout = RiscvShoutTables::new(32);
    let mul_id = shout.opcode_to_id(RiscvOpcode::Mul);
    let mulhu_id = shout.opcode_to_id(RiscvOpcode::Mulhu);

    let mul_out = neo_vm_trace::Shout::lookup(&mut shout, mul_id, key);
    let mulhu_out = neo_vm_trace::Shout::lookup(&mut shout, mulhu_id, key);

    assert_eq!(mul_out, compute_op(RiscvOpcode::Mul, lhs, rhs, 32));
    assert_eq!(mulhu_out, compute_op(RiscvOpcode::Mulhu, lhs, rhs, 32));
}

#[test]
fn combined_lookup_key_opcode_set_is_width_safe() {
    assert!(opcode_uses_combined_lookup_key(RiscvOpcode::Add));
    assert!(opcode_uses_combined_lookup_key(RiscvOpcode::Sub));

    for op in [
        RiscvOpcode::Mul,
        RiscvOpcode::Mulh,
        RiscvOpcode::Mulhu,
        RiscvOpcode::Mulhsu,
        RiscvOpcode::Mulw,
        RiscvOpcode::Div,
        RiscvOpcode::Divu,
        RiscvOpcode::Rem,
        RiscvOpcode::Remu,
        RiscvOpcode::Divw,
        RiscvOpcode::Divuw,
        RiscvOpcode::Remw,
        RiscvOpcode::Remuw,
        RiscvOpcode::Addw,
        RiscvOpcode::Subw,
    ] {
        assert!(
            !opcode_uses_combined_lookup_key(op),
            "unexpected combined key opcode: {op:?}"
        );
    }
}

#[test]
fn rv32_exec_table_virtual_metadata_defaults_to_inert_values() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let table = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    for row in &table.rows {
        assert!(!row.is_virtual);
        assert_eq!(row.virtual_sequence_remaining, None);
    }

    let cols = table.to_columns();
    assert_eq!(cols.is_virtual.len(), cols.len());
    assert_eq!(cols.virtual_sequence_remaining.len(), cols.len());
    assert!(cols.is_virtual.iter().all(|v| !*v));
    assert!(cols.virtual_sequence_remaining.iter().all(|v| *v == 0));
}

#[test]
fn mul_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::mul::decomposition_sequence(
        /*rd=*/ 5, /*rs1=*/ 3, /*rs2=*/ 4, &mut alloc,
    );
    assert_eq!(seq.len(), 2);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Mul,
            lhs: 3,
            rhs: 4
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::Move {
            dst: 5,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn mulhu_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::mulhu::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 2);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Mulhu,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn mulw_decomposition_sequence_uses_helper_mask_then_local_compose() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::mulw::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 4);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::VirtualMulWord,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::MovSignWord {
            dst: VIRTUAL_REG_BASE + 1,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: VIRTUAL_REG_BASE,
            lo_src: VIRTUAL_REG_BASE,
            hi_src: VIRTUAL_REG_BASE + 1
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn divuw_decomposition_sequence_uses_helper_mask_then_local_compose() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::divuw::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 4);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::VirtualDivuWord,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::MovSignWord {
            dst: VIRTUAL_REG_BASE + 1,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: VIRTUAL_REG_BASE,
            lo_src: VIRTUAL_REG_BASE,
            hi_src: VIRTUAL_REG_BASE + 1
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn divw_decomposition_sequence_uses_helper_mask_then_local_compose() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::divw::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 4);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::VirtualDivWord,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::MovSignWord {
            dst: VIRTUAL_REG_BASE + 1,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: VIRTUAL_REG_BASE,
            lo_src: VIRTUAL_REG_BASE,
            hi_src: VIRTUAL_REG_BASE + 1
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn remuw_decomposition_sequence_uses_helper_mask_then_local_compose() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::remuw::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 4);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::VirtualRemuWord,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::MovSignWord {
            dst: VIRTUAL_REG_BASE + 1,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: VIRTUAL_REG_BASE,
            lo_src: VIRTUAL_REG_BASE,
            hi_src: VIRTUAL_REG_BASE + 1
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn remw_decomposition_sequence_uses_helper_mask_then_local_compose() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::remw::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 4);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::VirtualRemWord,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::MovSignWord {
            dst: VIRTUAL_REG_BASE + 1,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: VIRTUAL_REG_BASE,
            lo_src: VIRTUAL_REG_BASE,
            hi_src: VIRTUAL_REG_BASE + 1
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Move {
            dst: 7,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn mulh_decomposition_sequence_matches_jolt_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::mulh::decomposition_sequence(
        /*rd=*/ 5, /*rs1=*/ 3, /*rs2=*/ 4, &mut alloc,
    );
    assert_eq!(seq.len(), 8);

    let v0 = VIRTUAL_REG_BASE;
    let v1 = VIRTUAL_REG_BASE + 1;
    let v2 = VIRTUAL_REG_BASE + 2;
    assert_eq!(seq[0], DecomposedOp::MovSign { dst: v0, src: 3 });
    assert_eq!(seq[1], DecomposedOp::MovSign { dst: v1, src: 4 });
    assert_eq!(
        seq[2],
        DecomposedOp::Mul {
            dst: v0,
            lhs: v0,
            rhs: 4
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Mul {
            dst: v1,
            lhs: v1,
            rhs: 3
        }
    );
    assert_eq!(
        seq[4],
        DecomposedOp::Mulhu {
            dst: v2,
            lhs: 3,
            rhs: 4
        }
    );
    assert_eq!(
        seq[5],
        DecomposedOp::Add {
            dst: v2,
            lhs: v2,
            rhs: v0
        }
    );
    assert_eq!(
        seq[6],
        DecomposedOp::Add {
            dst: v2,
            lhs: v2,
            rhs: v1
        }
    );
    assert_eq!(seq[7], DecomposedOp::Move { dst: 5, src: v2 });
}

#[test]
fn mulhsu_decomposition_sequence_matches_jolt_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::mulhsu::decomposition_sequence(
        /*rd=*/ 7, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 12);

    let v0 = VIRTUAL_REG_BASE;
    let v1 = VIRTUAL_REG_BASE + 1;
    let v2 = VIRTUAL_REG_BASE + 2;
    let v3 = VIRTUAL_REG_BASE + 3;
    assert_eq!(seq[0], DecomposedOp::MovSign { dst: v0, src: 1 });
    assert_eq!(
        seq[1],
        DecomposedOp::Sub {
            dst: v1,
            lhs: 0,
            rhs: v0
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::Xor {
            dst: v2,
            lhs: 1,
            rhs: v0
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::Add {
            dst: v2,
            lhs: v2,
            rhs: v1
        }
    );
    assert_eq!(
        seq[4],
        DecomposedOp::Mulhu {
            dst: v3,
            lhs: v2,
            rhs: 2
        }
    );
    assert_eq!(
        seq[5],
        DecomposedOp::Mul {
            dst: v2,
            lhs: v2,
            rhs: 2
        }
    );
    assert_eq!(
        seq[6],
        DecomposedOp::Xor {
            dst: v3,
            lhs: v3,
            rhs: v0
        }
    );
    assert_eq!(
        seq[7],
        DecomposedOp::Xor {
            dst: v2,
            lhs: v2,
            rhs: v0
        }
    );
    assert_eq!(
        seq[8],
        DecomposedOp::Add {
            dst: v0,
            lhs: v2,
            rhs: v1
        }
    );
    assert_eq!(
        seq[9],
        DecomposedOp::AdviceQuotient {
            dst: v0,
            op: RiscvOpcode::Sltu,
            lhs: v0,
            rhs: v2
        }
    );
    assert_eq!(
        seq[10],
        DecomposedOp::Add {
            dst: v3,
            lhs: v3,
            rhs: v0
        }
    );
    assert_eq!(seq[11], DecomposedOp::Move { dst: 7, src: v3 });
}

#[test]
fn divu_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::divu::decomposition_sequence(
        /*rd=*/ 9, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 9);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Divu,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::AssertValidDiv0 {
            divisor: 2,
            quotient: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[seq.len() - 2],
        DecomposedOp::Move {
            dst: VIRTUAL_REG_BASE,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[seq.len() - 1],
        DecomposedOp::Move {
            dst: 9,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn remu_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::remu::decomposition_sequence(
        /*rd=*/ 10, /*rs1=*/ 1, /*rs2=*/ 2, &mut alloc,
    );
    assert_eq!(seq.len(), 8);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Divu,
            lhs: 1,
            rhs: 2
        }
    );
    assert_eq!(
        seq[seq.len() - 2],
        DecomposedOp::Move {
            dst: VIRTUAL_REG_BASE + 2,
            src: VIRTUAL_REG_BASE + 2
        }
    );
    assert_eq!(
        seq[seq.len() - 1],
        DecomposedOp::Move {
            dst: 10,
            src: VIRTUAL_REG_BASE + 2
        }
    );
}

#[test]
fn div_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::div::decomposition_sequence(
        /*rd=*/ 11, /*rs1=*/ 3, /*rs2=*/ 4, &mut alloc,
    );
    assert_eq!(seq.len(), 19);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Div,
            lhs: 3,
            rhs: 4
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::AdviceRemainderAbs {
            dst: VIRTUAL_REG_BASE + 1,
            dividend: 3,
            divisor: 4
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::AssertValidDiv0 {
            divisor: 4,
            quotient: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::ChangeDivisor {
            dst: VIRTUAL_REG_BASE + 2,
            dividend: 3,
            divisor: 4
        }
    );
    assert_eq!(
        seq[seq.len() - 2],
        DecomposedOp::Move {
            dst: VIRTUAL_REG_BASE,
            src: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[seq.len() - 1],
        DecomposedOp::Move {
            dst: 11,
            src: VIRTUAL_REG_BASE
        }
    );
}

#[test]
fn rem_decomposition_sequence_matches_shape() {
    let mut alloc = VirtualRegisterAllocator::new();
    let seq = neo_memory::riscv::instruction::rem::decomposition_sequence(
        /*rd=*/ 12, /*rs1=*/ 3, /*rs2=*/ 4, &mut alloc,
    );
    assert_eq!(seq.len(), 20);
    assert_eq!(
        seq[0],
        DecomposedOp::AdviceQuotient {
            dst: VIRTUAL_REG_BASE,
            op: RiscvOpcode::Div,
            lhs: 3,
            rhs: 4
        }
    );
    assert_eq!(
        seq[1],
        DecomposedOp::AdviceRemainderAbs {
            dst: VIRTUAL_REG_BASE + 1,
            dividend: 3,
            divisor: 4
        }
    );
    assert_eq!(
        seq[2],
        DecomposedOp::AssertValidDiv0 {
            divisor: 4,
            quotient: VIRTUAL_REG_BASE
        }
    );
    assert_eq!(
        seq[3],
        DecomposedOp::ChangeDivisor {
            dst: VIRTUAL_REG_BASE + 2,
            dividend: 3,
            divisor: 4
        }
    );
    assert_eq!(
        seq[seq.len() - 1],
        DecomposedOp::Move {
            dst: 12,
            src: VIRTUAL_REG_BASE + 6
        }
    );
}

#[test]
fn mul_decomposition_sequence_matches_mul_semantics_on_samples() {
    let cases = [
        (0u32, 0u32),
        (1u32, 1u32),
        (u32::MAX, 2u32),
        (0x8000_0000u32, 0xFFFF_FFFFu32),
        (0x7FFF_FFFFu32, 0x8000_0000u32),
        (0x1234_5678u32, 0x9ABC_DEF0u32),
    ];

    for (lhs, rhs) in cases {
        let mut regs = [0u64; 32];
        let rd = 5u8;
        let rs1 = 3u8;
        let rs2 = 4u8;
        regs[rs1 as usize] = lhs as u64;
        regs[rs2 as usize] = rhs as u64;

        let mut alloc = VirtualRegisterAllocator::new();
        let seq = neo_memory::riscv::instruction::mul::decomposition_sequence(rd, rs1, rs2, &mut alloc);
        let _ = simulate_decomposition_sequence(&seq, &mut regs, /*xlen=*/ 32);

        let expected = compute_op(RiscvOpcode::Mul, lhs as u64, rhs as u64, /*xlen=*/ 32);
        assert_eq!(
            regs[rd as usize], expected,
            "mul mismatch for lhs={lhs:#x}, rhs={rhs:#x}"
        );
    }
}

#[test]
fn mulhu_decomposition_sequence_matches_mulhu_semantics_on_samples() {
    let cases = [
        (0u32, 0u32),
        (1u32, 1u32),
        (u32::MAX, 2u32),
        (0x8000_0000u32, 0xFFFF_FFFFu32),
        (0x7FFF_FFFFu32, 0x8000_0000u32),
        (0x1234_5678u32, 0x9ABC_DEF0u32),
    ];

    for (lhs, rhs) in cases {
        let mut regs = [0u64; 32];
        let rd = 6u8;
        let rs1 = 1u8;
        let rs2 = 2u8;
        regs[rs1 as usize] = lhs as u64;
        regs[rs2 as usize] = rhs as u64;

        let mut alloc = VirtualRegisterAllocator::new();
        let seq = neo_memory::riscv::instruction::mulhu::decomposition_sequence(rd, rs1, rs2, &mut alloc);
        let _ = simulate_decomposition_sequence(&seq, &mut regs, /*xlen=*/ 32);

        let expected = compute_op(RiscvOpcode::Mulhu, lhs as u64, rhs as u64, /*xlen=*/ 32);
        assert_eq!(
            regs[rd as usize], expected,
            "mulhu mismatch for lhs={lhs:#x}, rhs={rhs:#x}"
        );
    }
}

#[test]
fn mulh_decomposition_sequence_matches_mulh_semantics_on_samples() {
    let cases = [
        (0u32, 0u32),
        (1u32, 1u32),
        (u32::MAX, 2u32),
        (0x8000_0000u32, 0xFFFF_FFFFu32),
        (0x7FFF_FFFFu32, 0x8000_0000u32),
        (0x1234_5678u32, 0x9ABC_DEF0u32),
    ];

    for (lhs, rhs) in cases {
        let mut regs = [0u64; 32];
        let rd = 5u8;
        let rs1 = 3u8;
        let rs2 = 4u8;
        regs[rs1 as usize] = lhs as u64;
        regs[rs2 as usize] = rhs as u64;

        let mut alloc = VirtualRegisterAllocator::new();
        let seq = neo_memory::riscv::instruction::mulh::decomposition_sequence(rd, rs1, rs2, &mut alloc);
        let _ = simulate_decomposition_sequence(&seq, &mut regs, /*xlen=*/ 32);

        let expected = compute_op(RiscvOpcode::Mulh, lhs as u64, rhs as u64, /*xlen=*/ 32);
        assert_eq!(
            regs[rd as usize], expected,
            "mulh mismatch for lhs={lhs:#x}, rhs={rhs:#x}"
        );
    }
}

#[test]
fn mulhsu_decomposition_sequence_matches_mulhsu_semantics_on_samples() {
    let cases = [
        (0u32, 0u32),
        (1u32, 1u32),
        (u32::MAX, 2u32),
        (0x8000_0000u32, 0xFFFF_FFFFu32),
        (0x7FFF_FFFFu32, 0x8000_0000u32),
        (0x1234_5678u32, 0x9ABC_DEF0u32),
    ];

    for (lhs, rhs) in cases {
        let mut regs = [0u64; 32];
        let rd = 7u8;
        let rs1 = 1u8;
        let rs2 = 2u8;
        regs[rs1 as usize] = lhs as u64;
        regs[rs2 as usize] = rhs as u64;

        let mut alloc = VirtualRegisterAllocator::new();
        let seq = neo_memory::riscv::instruction::mulhsu::decomposition_sequence(rd, rs1, rs2, &mut alloc);
        let _ = simulate_decomposition_sequence(&seq, &mut regs, /*xlen=*/ 32);

        let expected = compute_op(RiscvOpcode::Mulhsu, lhs as u64, rhs as u64, /*xlen=*/ 32);
        assert_eq!(
            regs[rd as usize], expected,
            "mulhsu mismatch for lhs={lhs:#x}, rhs={rhs:#x}"
        );
    }
}

#[test]
fn mul_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0xFFFF_FFFFu64;
    let rhs = 0x0000_0003u64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");
    assert_eq!(trace.steps.len(), 3);

    let virtual_row = &trace.steps[0];
    assert!(virtual_row.is_virtual);
    assert_eq!(virtual_row.pc_before, 0);
    assert_eq!(virtual_row.pc_after, 0);
    assert_eq!(virtual_row.virtual_sequence_remaining, Some(1));

    let commit_row = &trace.steps[1];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);
    assert_eq!(commit_row.virtual_sequence_remaining, None);

    let halt_row = trace.steps.last().expect("halt row");
    assert!(!halt_row.is_virtual);
    let expected = compute_op(RiscvOpcode::Mul, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[4], expected);
}

#[test]
fn mulhu_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0xFFFF_FFFFu64;
    let rhs = 0x0000_0003u64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");
    assert_eq!(trace.steps.len(), 3);

    let virtual_row = &trace.steps[0];
    assert!(virtual_row.is_virtual);
    assert_eq!(virtual_row.pc_before, 0);
    assert_eq!(virtual_row.pc_after, 0);
    assert_eq!(virtual_row.virtual_sequence_remaining, Some(1));

    let commit_row = &trace.steps[1];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);
    assert_eq!(commit_row.virtual_sequence_remaining, None);

    let halt_row = trace.steps.last().expect("halt row");
    assert!(!halt_row.is_virtual);
    let expected = compute_op(RiscvOpcode::Mulhu, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[5], expected);
}

#[test]
fn mulh_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0x8000_0000u64;
    let rhs = 0xFFFF_FFFFu64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");
    assert_eq!(trace.steps.len(), 9);

    for (idx, step) in trace.steps.iter().take(7).enumerate() {
        assert!(step.is_virtual, "row {idx} should be virtual");
        assert_eq!(step.pc_before, 0);
        assert_eq!(step.pc_after, 0);
        assert_eq!(step.virtual_sequence_remaining, Some((7 - idx) as u32));
    }

    let commit_row = &trace.steps[7];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);
    assert_eq!(commit_row.virtual_sequence_remaining, None);

    let halt_row = trace.steps.last().expect("halt row");
    assert!(!halt_row.is_virtual);
    let expected = compute_op(RiscvOpcode::Mulh, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[5], expected);
}

#[test]
fn mulhsu_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0x8000_0000u64;
    let rhs = 0xFFFF_FFFFu64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");
    assert_eq!(trace.steps.len(), 13);

    for (idx, step) in trace.steps.iter().take(11).enumerate() {
        assert!(step.is_virtual, "row {idx} should be virtual");
        assert_eq!(step.pc_before, 0);
        assert_eq!(step.pc_after, 0);
        assert_eq!(step.virtual_sequence_remaining, Some((11 - idx) as u32));
    }

    let commit_row = &trace.steps[11];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);
    assert_eq!(commit_row.virtual_sequence_remaining, None);

    let halt_row = trace.steps.last().expect("halt row");
    assert!(!halt_row.is_virtual);
    let expected = compute_op(RiscvOpcode::Mulhsu, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[6], expected);
}

#[test]
fn divu_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0x1234_5678u64;
    let rhs = 0x0000_0111u64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert_eq!(trace.steps.len(), 10);
    assert!(trace.steps.iter().take(8).all(|s| s.is_virtual));
    let commit_row = &trace.steps[8];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);

    let halt_row = trace.steps.last().expect("halt row");
    let expected = compute_op(RiscvOpcode::Divu, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[8], expected);
}

#[test]
fn remu_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0x1234_5678u64;
    let rhs = 0x0000_0111u64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert_eq!(trace.steps.len(), 9);
    assert!(trace.steps.iter().take(7).all(|s| s.is_virtual));
    let commit_row = &trace.steps[7];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);

    let halt_row = trace.steps.last().expect("halt row");
    let expected = compute_op(RiscvOpcode::Remu, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[9], expected);
}

#[test]
fn div_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0x8000_0000u64;
    let rhs = 0xFFFF_FFFFu64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert_eq!(trace.steps.len(), 20);
    assert!(trace.steps.iter().take(18).all(|s| s.is_virtual));
    let commit_row = &trace.steps[18];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);

    let halt_row = trace.steps.last().expect("halt row");
    let expected = compute_op(RiscvOpcode::Div, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[10], expected);
}

#[test]
fn div_runtime_trace_signed_non_edge_matches_semantics() {
    let lhs = 0xFFFF_FFF9u64; // -7 in RV32
    let rhs = 0x0000_0003u64; // +3
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 12,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let halt_row = trace.steps.last().expect("halt row");
    let expected = compute_op(RiscvOpcode::Div, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[12], expected);
}

#[test]
fn rem_runtime_trace_decomposes_into_virtual_rows() {
    let lhs = 0xFFFF_FFF5u64;
    let rhs = 0x0000_0003u64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 11,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert_eq!(trace.steps.len(), 21);
    assert!(trace.steps.iter().take(19).all(|s| s.is_virtual));
    let commit_row = &trace.steps[19];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);

    let halt_row = trace.steps.last().expect("halt row");
    let expected = compute_op(RiscvOpcode::Rem, lhs, rhs, /*xlen=*/ 32);
    assert_eq!(halt_row.regs_after[11], expected);
}

#[test]
fn decomposition_runtime_trace_enforces_virtual_vs_arch_reg_write_domains() {
    let cases = [
        (RiscvOpcode::Mul, 4u8, 0xFFFF_FFFFu64, 0x0000_0003u64),
        (RiscvOpcode::Mulhu, 5u8, 0xFFFF_FFFFu64, 0x0000_0003u64),
        (RiscvOpcode::Mulh, 5u8, 0x8000_0000u64, 0xFFFF_FFFFu64),
        (RiscvOpcode::Mulhsu, 6u8, 0x8000_0000u64, 0xFFFF_FFFFu64),
        (RiscvOpcode::Divu, 8u8, 0x1234_5678u64, 0x0000_0111u64),
        (RiscvOpcode::Remu, 9u8, 0x1234_5678u64, 0x0000_0111u64),
        (RiscvOpcode::Div, 10u8, 0xFFFF_FFF9u64, 0x0000_0003u64),
        (RiscvOpcode::Rem, 11u8, 0xFFFF_FFF5u64, 0x0000_0003u64),
    ];

    for (op, rd, lhs, rhs) in cases {
        let program = vec![
            RiscvInstruction::RAlu { op, rd, rs1: 1, rs2: 2 },
            RiscvInstruction::Halt,
        ];
        let program_bytes = encode_program(&program);
        let decoded_program = decode_program(&program_bytes).expect("decode_program");

        let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
        cpu.load_program(/*base=*/ 0, decoded_program);
        cpu.set_runtime_decomposition_enabled(true);
        let mut twist =
            RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
        twist.store(REG_ID, 1, lhs);
        twist.store(REG_ID, 2, rhs);
        let shout = RiscvShoutTables::new(/*xlen=*/ 32);

        let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 128).expect("trace_program");
        for (row_idx, step) in trace.steps.iter().enumerate() {
            for event in step.twist_events.iter().filter(|ev| ev.twist_id == REG_ID) {
                match event.kind {
                    TwistOpKind::Read => {
                        if !step.is_virtual {
                            assert!(
                                event.addr < 32,
                                "non-virtual row {row_idx} for {op:?} read virtual reg addr={} (>=32)",
                                event.addr
                            );
                        }
                    }
                    TwistOpKind::Write => {
                        if step.is_virtual {
                            assert!(
                                event.addr >= 32,
                                "virtual row {row_idx} for {op:?} wrote architectural reg addr={} (<32)",
                                event.addr
                            );
                        } else {
                            assert!(
                                event.addr < 32,
                                "non-virtual row {row_idx} for {op:?} wrote virtual reg addr={} (>=32)",
                                event.addr
                            );
                        }
                    }
                }
            }
        }
    }
}
