#![cfg(feature = "poseidon-precompile")]

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash as poseidon2_hash_ref;
use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32PoseidonSidecarTable};
use neo_memory::riscv::lookups::{
    decode_instruction, decode_program, encode_instruction, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory,
    RiscvOpcode, RiscvShoutTables, POSEIDON2_CUSTOM_OPCODE, PROG_ID,
};
use neo_vm_trace::trace_program;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

fn run_exec(program: Vec<RiscvInstruction>) -> Result<Rv32ExecTable, String> {
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes)?;

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(32, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 1 << 16)?;
    Rv32ExecTable::from_trace(&trace)
}

fn build_poseidon_program(input: &[u64]) -> Vec<RiscvInstruction> {
    build_poseidon_program_with_squeeze_rd(input, 5)
}

fn build_poseidon_program_with_squeeze_rd(input: &[u64], squeeze_rd: u8) -> Vec<RiscvInstruction> {
    let mut program = Vec::new();
    for &elem in input {
        let lo = (elem & 0xffff_ffff) as i32;
        let hi = ((elem >> 32) & 0xffff_ffff) as i32;
        program.push(RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: lo,
        });
        program.push(RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 11,
            rs1: 0,
            imm: hi,
        });
        program.push(RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 });
    }
    program.push(RiscvInstruction::Poseidon2Finalize);
    for idx in 0..8u8 {
        program.push(RiscvInstruction::Poseidon2SqueezeWord { rd: squeeze_rd, idx });
    }
    program.push(RiscvInstruction::Halt);
    program
}

#[test]
fn poseidon2_custom_decode_roundtrip() {
    let absorb = RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 };
    let enc_absorb = encode_instruction(&absorb);
    assert_eq!(enc_absorb & 0x7f, POSEIDON2_CUSTOM_OPCODE);
    match decode_instruction(enc_absorb).expect("decode absorb") {
        RiscvInstruction::Poseidon2AbsorbElem { rs1, rs2 } => {
            assert_eq!(rs1, 10);
            assert_eq!(rs2, 11);
        }
        other => panic!("unexpected decode for absorb: {other:?}"),
    }

    let finalize = RiscvInstruction::Poseidon2Finalize;
    let enc_finalize = encode_instruction(&finalize);
    assert_eq!(enc_finalize & 0x7f, POSEIDON2_CUSTOM_OPCODE);
    assert!(matches!(
        decode_instruction(enc_finalize).expect("decode finalize"),
        RiscvInstruction::Poseidon2Finalize
    ));

    let squeeze = RiscvInstruction::Poseidon2SqueezeWord { rd: 5, idx: 7 };
    let enc_squeeze = encode_instruction(&squeeze);
    assert_eq!(enc_squeeze & 0x7f, POSEIDON2_CUSTOM_OPCODE);
    match decode_instruction(enc_squeeze).expect("decode squeeze") {
        RiscvInstruction::Poseidon2SqueezeWord { rd, idx } => {
            assert_eq!(rd, 5);
            assert_eq!(idx, 7);
        }
        other => panic!("unexpected decode for squeeze: {other:?}"),
    }
}

#[test]
fn poseidon2_custom_decode_rejects_noncanonical_absorb_and_finalize() {
    // Absorb with rd != x0 must be rejected.
    let absorb = encode_instruction(&RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 });
    let absorb_bad_rd = absorb | (1u32 << 7);
    assert!(decode_instruction(absorb_bad_rd).is_err());

    // Absorb with funct3 != 0 must be rejected.
    let absorb_bad_funct3 = absorb | (1u32 << 12);
    assert!(decode_instruction(absorb_bad_funct3).is_err());

    // Finalize with rd != x0 must be rejected.
    let finalize = encode_instruction(&RiscvInstruction::Poseidon2Finalize);
    let finalize_bad_rd = finalize | (3u32 << 7);
    assert!(decode_instruction(finalize_bad_rd).is_err());

    // Finalize with funct3 != 0 must be rejected.
    let finalize_bad_funct3 = finalize | (2u32 << 12);
    assert!(decode_instruction(finalize_bad_funct3).is_err());

    // Finalize with rs1 != x0 must be rejected.
    let finalize_bad_rs1 = finalize | (1u32 << 15);
    assert!(decode_instruction(finalize_bad_rs1).is_err());

    // Finalize with rs2 != x0 must be rejected.
    let finalize_bad_rs2 = finalize | (1u32 << 20);
    assert!(decode_instruction(finalize_bad_rs2).is_err());

    // Squeeze with rs1/rs2 != x0 must be rejected.
    let squeeze = encode_instruction(&RiscvInstruction::Poseidon2SqueezeWord { rd: 5, idx: 3 });
    let squeeze_bad_rs1 = squeeze | (2u32 << 15);
    assert!(decode_instruction(squeeze_bad_rs1).is_err());
    let squeeze_bad_rs2 = squeeze | (4u32 << 20);
    assert!(decode_instruction(squeeze_bad_rs2).is_err());
}

#[test]
#[should_panic(expected = "Poseidon2SqueezeWord idx must be in 0..=7")]
fn poseidon2_encode_rejects_out_of_range_squeeze_idx() {
    let _ = encode_instruction(&RiscvInstruction::Poseidon2SqueezeWord { rd: 5, idx: 8 });
}

#[test]
fn poseidon2_vm_digest_matches_reference_hash() {
    // Input elements as u64 values (mapped to Goldilocks via from_u64).
    let inputs = [5u64, 7u64, 11u64, 13u64, 17u64];
    let program = build_poseidon_program(&inputs);
    let exec = run_exec(program).expect("trace run");

    let mut squeezed_words = Vec::new();
    for row in exec.rows.iter() {
        if let Some(RiscvInstruction::Poseidon2SqueezeWord { .. }) = row.decoded.as_ref() {
            let write = row.reg_write_lane0.as_ref().expect("squeeze must write rd");
            squeezed_words.push(write.value as u32);
        }
    }
    assert_eq!(squeezed_words.len(), 8);

    let input_field: Vec<Goldilocks> = inputs.iter().copied().map(Goldilocks::from_u64).collect();
    let digest = poseidon2_hash_ref(&input_field);
    let mut expected_words = Vec::with_capacity(8);
    for d in digest.iter() {
        let v = d.as_canonical_u64();
        expected_words.push(v as u32);
        expected_words.push((v >> 32) as u32);
    }

    assert_eq!(squeezed_words, expected_words);
}

#[test]
fn poseidon2_vm_digest_matches_reference_for_lengths_0_to_8() {
    for n in 0..=8usize {
        let inputs: Vec<u64> = (0..n).map(|i| (i as u64) * 17 + 3).collect();
        let program = build_poseidon_program(&inputs);
        let exec = run_exec(program).expect("trace run");
        let sidecar = Rv32PoseidonSidecarTable::from_exec_table(&exec).expect("poseidon sidecar extraction");

        let mut squeezed_words = Vec::new();
        for row in exec.rows.iter() {
            if let Some(RiscvInstruction::Poseidon2SqueezeWord { .. }) = row.decoded.as_ref() {
                let write = row.reg_write_lane0.as_ref().expect("squeeze must write rd");
                squeezed_words.push(write.value as u32);
            }
        }
        assert_eq!(squeezed_words.len(), 8, "n={n}");

        let input_field: Vec<Goldilocks> = inputs.iter().copied().map(Goldilocks::from_u64).collect();
        let digest = poseidon2_hash_ref(&input_field);
        let mut expected_words = Vec::with_capacity(8);
        for d in digest.iter() {
            let v = d.as_canonical_u64();
            expected_words.push(v as u32);
            expected_words.push((v >> 32) as u32);
        }
        assert_eq!(squeezed_words, expected_words, "n={n}");

        // Canonicality aux: digest elements are canonical, so carry-out c1 must be zero.
        for row in sidecar.cycle_rows.iter().filter(|r| r.op_squeeze) {
            assert_eq!(row.canonical_c1, 0, "n={} cycle={}", n, row.cycle);
        }
    }
}

#[test]
fn poseidon2_finalize_partial_block_uses_two_slots() {
    // n=1 => finalize must run partial-block permute (slot0) and padded final permute (slot1).
    let program = build_poseidon_program(&[42u64]);
    let exec = run_exec(program).expect("trace run");
    let sidecar = Rv32PoseidonSidecarTable::from_exec_table(&exec).expect("poseidon sidecar extraction");

    let finalize_row = sidecar
        .cycle_rows
        .iter()
        .find(|r| r.op_finalize)
        .expect("missing finalize row");
    assert!(finalize_row.do_perm_slot0);
    assert!(finalize_row.do_perm_slot1);

    let perms_at_finalize: Vec<_> = sidecar
        .perm_rows
        .iter()
        .filter(|r| r.cycle == finalize_row.cycle)
        .collect();
    assert_eq!(perms_at_finalize.len(), 2);
    assert_eq!(perms_at_finalize[0].slot, 0);
    assert_eq!(perms_at_finalize[1].slot, 1);
}

#[test]
fn poseidon2_finalize_full_block_uses_only_slot1() {
    // n=4 => last absorb already permutes at block boundary; finalize runs only padded final permute.
    let program = build_poseidon_program(&[1u64, 2u64, 3u64, 4u64]);
    let exec = run_exec(program).expect("trace run");
    let sidecar = Rv32PoseidonSidecarTable::from_exec_table(&exec).expect("poseidon sidecar extraction");

    let finalize_row = sidecar
        .cycle_rows
        .iter()
        .find(|r| r.op_finalize)
        .expect("missing finalize row");
    assert!(!finalize_row.do_perm_slot0);
    assert!(finalize_row.do_perm_slot1);

    let perms_at_finalize: Vec<_> = sidecar
        .perm_rows
        .iter()
        .filter(|r| r.cycle == finalize_row.cycle)
        .collect();
    assert_eq!(perms_at_finalize.len(), 1);
    assert_eq!(perms_at_finalize[0].slot, 1);
}

#[test]
fn poseidon2_vm_rejects_squeeze_before_finalize() {
    let program = vec![
        RiscvInstruction::Poseidon2SqueezeWord { rd: 5, idx: 0 },
        RiscvInstruction::Halt,
    ];
    let err = run_exec(program).expect_err("squeeze-before-finalize must fail");
    assert!(err.contains("squeeze called before finalize"), "unexpected err: {err}");
}

#[test]
fn poseidon2_vm_rejects_double_finalize_without_restart() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 11,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 },
        RiscvInstruction::Poseidon2Finalize,
        RiscvInstruction::Poseidon2Finalize,
        RiscvInstruction::Halt,
    ];
    let err = run_exec(program).expect_err("double-finalize must fail");
    assert!(
        err.contains("finalize called in Finalized mode"),
        "unexpected err: {err}"
    );
}

#[test]
fn poseidon2_sidecar_allows_squeeze_to_x0() {
    let inputs = [9u64, 10u64, 11u64];
    let program = build_poseidon_program_with_squeeze_rd(&inputs, 0);
    let exec = run_exec(program).expect("trace run");
    let sidecar = Rv32PoseidonSidecarTable::from_exec_table(&exec).expect("poseidon sidecar extraction");

    let squeeze_cycles: Vec<u64> = sidecar
        .cycle_rows
        .iter()
        .filter(|r| r.op_squeeze)
        .map(|r| r.cycle)
        .collect();
    assert_eq!(squeeze_cycles.len(), 8);
    for cycle in squeeze_cycles {
        let row = exec
            .rows
            .iter()
            .find(|r| r.cycle == cycle)
            .expect("missing exec row for squeeze cycle");
        assert!(
            row.reg_write_lane0.is_none(),
            "squeeze to x0 should not produce reg_write lane (cycle={cycle})"
        );
    }
}
