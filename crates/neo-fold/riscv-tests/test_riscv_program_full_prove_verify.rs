//! End-to-end prove+verify for a small RV32 program under the B1 shared-bus step circuit.
//!
//! This exercises:
//! - B1 instruction fetch via `PROG_ID` Twist reads
//! - shared CPU bus tail wiring (Twist + Shout)
//! - implicit Shout table spec (`LutTableSpec::RiscvOpcode`)
//! - the RV32 B1 step CCS glue constraints

#![allow(non_snake_case)]

use std::collections::HashMap;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::matrix::Mat;
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_shard::fold_shard_verify_rv32_b1_with_statement_mem_init;
use neo_fold::shard::{fold_shard_prove, CommitMixers};
use neo_math::{F, K};
use neo_memory::builder::build_shard_witness_shared_cpu_bus;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{build_rv32_b1_step_ccs, rv32_b1_chunk_to_witness, rv32_b1_shared_cpu_bus_config};
use neo_memory::riscv::lookups::{
    encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::rom_init::prog_init_words;
use neo_memory::riscv::shard::extract_boundary_state;
use neo_memory::witness::{LutTableSpec, StepInstanceBundle};
use neo_memory::R1csCpu;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(neo_math::D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(neo_math::D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn pow2_ceil_k(min_k: usize) -> (usize, usize) {
    let k = min_k.next_power_of_two().max(2);
    let d = k.trailing_zeros() as usize;
    (k, d)
}

fn add_only_table_specs(xlen: usize) -> HashMap<u32, LutTableSpec> {
    HashMap::from([(
        3u32,
        LutTableSpec::RiscvOpcode {
            opcode: RiscvOpcode::Add,
            xlen,
        },
    )])
}

#[test]
fn test_riscv_program_full_prove_verify() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 1 }, // x1 = 0x1000
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        }, // x1 = 0x1005
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // x2 = 7
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 0x100c
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0x100,
        }, // mem[0x100] = x3
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 4,
            rs1: 0,
            imm: 0x100,
        }, // x4 = mem[0x100]
        RiscvInstruction::Auipc { rd: 5, imm: 0 }, // x5 = pc
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();

    let program_bytes = encode_program(&program);
    let mut vm = RiscvCpu::new(xlen);
    vm.load_program(0, program);
    let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);

    // Keep k small to reduce bus tail width and proof work.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Build CCS + shared-bus CPU arithmetization.
    // Keep the Shout bus lean: this program only needs ADD (for ADD/ADDI and effective address calculation).
    let shout_table_ids: Vec<u32> = vec![3u32];
    let add_idx = shout_table_ids
        .iter()
        .position(|&id| id == 3u32)
        .expect("ADD table id present");
    let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

    let table_specs = add_only_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs_base,
        params.clone(),
        DummyCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
            .expect("cfg"),
        1,
    )
    .expect("shared bus inject");

    // Build shared-bus step bundles (includes CPU MCS + metadata-only mem/lut instances).
    let lut_tables = HashMap::new();
    let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
        vm,
        twist,
        shout,
        /*max_steps=*/ max_steps,
        /*chunk_size=*/ 1,
        &mem_layouts,
        &lut_tables,
        &table_specs,
        &initial_mem,
        &cpu,
    )
    .expect("build shard witness");

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mixers = default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-full");
    // PaperExact is intentionally slow (brute-force oracle) and can make this end-to-end
    // test take minutes. Use the optimized engine here and keep PaperExact covered by
    // smaller unit tests.
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &cpu.ccs,
        &steps,
        &[],
        &[],
        &DummyCommit::default(),
        mixers,
    )
    .expect("prove");

    // Ensure the Shout addr-pre proof uses the fixed-count profile (no active_mask / no skipping).
    //
    // This program uses only the ADD table. LUI and HALT use no Shout lookups, so their per-step
    // Shout addr-pre claimed sums should be all zero.
    let mut saw_no_shout = false;
    let mut saw_add_only = false;
    for step in &proof.steps {
        let pre = &step.mem.shout_addr_pre;
        assert_eq!(
            pre.claimed_sums.len(),
            shout_table_ids.len(),
            "Shout addr-pre claimed_sums must be fixed-count"
        );
        assert_eq!(
            pre.round_polys.len(),
            shout_table_ids.len(),
            "Shout addr-pre round_polys must be fixed-count"
        );

        let mut any_nonzero = false;
        for (i, &claim) in pre.claimed_sums.iter().enumerate() {
            if claim != K::ZERO {
                any_nonzero = true;
                assert_eq!(i, add_idx, "only the ADD table should be non-zero");
            }
        }
        if any_nonzero {
            saw_add_only = true;
        } else {
            saw_no_shout = true;
        }
    }
    assert!(saw_no_shout, "expected at least one no-Shout step");
    assert!(saw_add_only, "expected at least one ADD-lookup step (mask=ADD)");

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-full");
    let _ = fold_shard_verify_rv32_b1_with_statement_mem_init(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &cpu.ccs,
        &mem_layouts,
        &initial_mem,
        &steps_public,
        &[],
        &proof,
        mixers,
        &layout,
    )
    .expect("verify");

    let mut bad_steps = steps_public.clone();
    bad_steps[1].mcs_inst.x[layout.pc0] += F::ONE;
    let mut tr_bad = Poseidon2Transcript::new(b"riscv-b1-full");
    assert!(
        fold_shard_verify_rv32_b1_with_statement_mem_init(
            FoldingMode::Optimized,
            &mut tr_bad,
            &params,
            &cpu.ccs,
            &mem_layouts,
            &initial_mem,
            &bad_steps,
            &[],
            &proof,
            mixers,
            &layout,
        )
        .is_err(),
        "expected step linking failure"
    );

    // Tamper: flip Shout addr-pre claimed sum; verification must fail.
    let mut bad_proof = proof.clone();
    let first = bad_proof.steps.first_mut().expect("non-empty");
    first.mem.shout_addr_pre.claimed_sums[add_idx] += K::ONE;
    let mut tr_bad_mask = Poseidon2Transcript::new(b"riscv-b1-full");
    assert!(
        fold_shard_verify_rv32_b1_with_statement_mem_init(
            FoldingMode::Optimized,
            &mut tr_bad_mask,
            &params,
            &cpu.ccs,
            &mem_layouts,
            &initial_mem,
            &steps_public,
            &[],
            &bad_proof,
            mixers,
            &layout,
        )
        .is_err(),
        "expected Shout addr-pre claimed sum mismatch failure"
    );
}

#[test]
fn test_riscv_statement_mem_init_mismatch_fails() {
    let xlen = 32usize;
    let program = vec![RiscvInstruction::Halt];
    let max_steps = program.len();

    let program_bytes = encode_program(&program);

    // Keep k small to reduce bus tail width and proof work.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Keep the Shout bus lean: this program uses no Shout lookups, but include ADD to keep the bus schema stable.
    let table_specs = add_only_table_specs(xlen);
    let shout_table_ids: Vec<u32> = vec![3u32];

    let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

    let cpu = R1csCpu::new(
        ccs_base,
        params.clone(),
        DummyCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone()).expect("cfg"),
        1,
    )
    .expect("shared bus inject");

    let mut vm = RiscvCpu::new(xlen);
    vm.load_program(0, program);
    let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);

    let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
        vm,
        twist,
        shout,
        max_steps,
        1,
        &mem_layouts,
        &HashMap::new(),
        &table_specs,
        &initial_mem,
        &cpu,
    )
    .expect("build shard witness");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mixers = default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-stmt-mem-init");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &cpu.ccs,
        &steps,
        &[],
        &[],
        &DummyCommit::default(),
        mixers,
    )
    .expect("prove");

    // Sanity: correct statement must verify.
    let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-stmt-mem-init");
    let _ = fold_shard_verify_rv32_b1_with_statement_mem_init(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &cpu.ccs,
        &mem_layouts,
        &initial_mem,
        &steps_public,
        &[],
        &proof,
        mixers,
        &layout,
    )
    .expect("verify");

    // Mismatch the *statement* initial memory (RAM starts non-zero) while keeping the proof fixed.
    // Verification must fail at the chunk0 init check.
    let mut bad_statement_initial_mem = initial_mem.clone();
    bad_statement_initial_mem.insert((0u32, 0u64), F::ONE);

    let mut tr_bad = Poseidon2Transcript::new(b"riscv-b1-stmt-mem-init");
    assert!(
        fold_shard_verify_rv32_b1_with_statement_mem_init(
            FoldingMode::Optimized,
            &mut tr_bad,
            &params,
            &cpu.ccs,
            &mem_layouts,
            &bad_statement_initial_mem,
            &steps_public,
            &[],
            &proof,
            mixers,
            &layout,
        )
        .is_err(),
        "expected statement init mismatch failure"
    );
}

#[test]
#[ignore = "manual benchmark sweep; run with --ignored --nocapture"]
fn perf_rv32_b1_chunk_size_sweep() {
    use std::time::Instant;

    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::Branch {
            cond: neo_memory::riscv::lookups::BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // not taken
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        }, // mem[0] = x1
        RiscvInstruction::Jal { rd: 5, imm: 8 }, // jump over the next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 123,
        }, // skipped
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 3,
            rs1: 0,
            imm: 0,
        }, // x3 = mem[0]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let max_steps = 64usize;

    // Keep k small to reduce bus tail width and proof work.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    fn table_specs_from_ids(ids: &[u32], xlen: usize) -> HashMap<u32, LutTableSpec> {
        ids.iter()
            .copied()
            .map(|id| {
                let opcode = match id {
                    0 => RiscvOpcode::And,
                    1 => RiscvOpcode::Xor,
                    2 => RiscvOpcode::Or,
                    3 => RiscvOpcode::Add,
                    4 => RiscvOpcode::Sub,
                    5 => RiscvOpcode::Slt,
                    6 => RiscvOpcode::Sltu,
                    7 => RiscvOpcode::Sll,
                    8 => RiscvOpcode::Srl,
                    9 => RiscvOpcode::Sra,
                    10 => RiscvOpcode::Eq,
                    11 => RiscvOpcode::Neq,
                    12 => RiscvOpcode::Mul,
                    13 => RiscvOpcode::Mulh,
                    14 => RiscvOpcode::Mulhu,
                    15 => RiscvOpcode::Mulhsu,
                    16 => RiscvOpcode::Div,
                    17 => RiscvOpcode::Divu,
                    18 => RiscvOpcode::Rem,
                    19 => RiscvOpcode::Remu,
                    _ => panic!("unsupported RV32 B1 table_id={id}"),
                };
                (
                    id,
                    LutTableSpec::RiscvOpcode {
                        opcode,
                        xlen,
                    },
                )
            })
            .collect()
    }

    let profiles: &[(&str, &[u32])] = &[
        ("min3", neo_memory::riscv::ccs::RV32_B1_SHOUT_PROFILE_MIN3),
        ("full12", neo_memory::riscv::ccs::RV32_B1_SHOUT_PROFILE_FULL12),
    ];

    let mixers = default_mixers();

    for (profile_name, shout_table_ids) in profiles {
        let table_specs = table_specs_from_ids(shout_table_ids, xlen);
        println!("\n== profile={profile_name} shout_tables={} ==", shout_table_ids.len());

        for chunk_size in [1usize, 2, 4, 8, 16] {
            let mut vm = RiscvCpu::new(xlen);
            vm.load_program(0, program.clone());
            let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
            let shout = RiscvShoutTables::new(xlen);

            let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, shout_table_ids, chunk_size).expect("ccs");
            let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

            let cpu = R1csCpu::new(
                ccs_base,
                params.clone(),
                DummyCommit::default(),
                layout.m_in,
                &HashMap::new(),
                &table_specs,
                rv32_b1_chunk_to_witness(layout.clone()),
            )
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                    .expect("cfg"),
                chunk_size,
            )
            .expect("shared bus inject");

            let t_build = Instant::now();
            let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
                vm,
                twist,
                shout,
                max_steps,
                chunk_size,
                &mem_layouts,
                &HashMap::new(),
                &table_specs,
                &initial_mem,
                &cpu,
            )
            .expect("build shard witness");
            let build_dur = t_build.elapsed();

            let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
                steps.iter().map(StepInstanceBundle::from).collect();

            let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-chunk-sweep");
            let t_prove = Instant::now();
            let proof = fold_shard_prove(
                FoldingMode::Optimized,
                &mut tr_prove,
                &params,
                &cpu.ccs,
                &steps,
                &[],
                &[],
                &DummyCommit::default(),
                mixers,
            )
            .expect("prove");
            let prove_dur = t_prove.elapsed();

            let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-chunk-sweep");
            let t_verify = Instant::now();
            let _ = fold_shard_verify_rv32_b1_with_statement_mem_init(
                FoldingMode::Optimized,
                &mut tr_verify,
                &params,
                &cpu.ccs,
                &mem_layouts,
                &initial_mem,
                &steps_public,
                &[],
                &proof,
                mixers,
                &layout,
            )
            .expect("verify");
            let verify_dur = t_verify.elapsed();

            println!(
                "chunk_size={chunk_size:<2} chunks={:<3} build={:?} prove={:?} verify={:?}",
                steps_public.len(),
                build_dur,
                prove_dur,
                verify_dur
            );
        }
    }
}

#[test]
fn test_riscv_program_chunk_size_equivalence() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        }, // mem[0] = x1
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0,
        }, // x2 = mem[0]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let max_steps = program.len() + 2; // include padding steps after HALT under fixed-length semantics

    // Keep k small to reduce bus tail width and proof work.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Keep the Shout bus lean: this program only needs ADD (for ADDI and effective address calculation).
    let table_specs = add_only_table_specs(xlen);
    let shout_table_ids: Vec<u32> = vec![3u32];

    let mixers = default_mixers();

    let run = |chunk_size: usize| -> (neo_memory::riscv::ccs::Rv32B1Layout, Vec<StepInstanceBundle<Cmt, F, K>>) {
        let mut vm = RiscvCpu::new(xlen);
        vm.load_program(0, program.clone());
        let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
        let shout = RiscvShoutTables::new(xlen);

        let (ccs_base, layout) =
            build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
        let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

        let cpu = R1csCpu::new(
            ccs_base,
            params.clone(),
            DummyCommit::default(),
            layout.m_in,
            &HashMap::new(),
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        )
        .with_shared_cpu_bus(
            rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                .expect("cfg"),
            chunk_size,
        )
        .expect("shared bus inject");

        let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
            vm,
            twist,
            shout,
            max_steps,
            chunk_size,
            &mem_layouts,
            &HashMap::new(),
            &table_specs,
            &initial_mem,
            &cpu,
        )
        .expect("build shard witness");
        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

        let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-chunk-eq");
        let proof = fold_shard_prove(
            FoldingMode::Optimized,
            &mut tr_prove,
            &params,
            &cpu.ccs,
            &steps,
            &[],
            &[],
            &DummyCommit::default(),
            mixers,
        )
        .expect("prove");

        let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-chunk-eq");
        let _ = fold_shard_verify_rv32_b1_with_statement_mem_init(
            FoldingMode::Optimized,
            &mut tr_verify,
            &params,
            &cpu.ccs,
            &mem_layouts,
            &initial_mem,
            &steps_public,
            &[],
            &proof,
            mixers,
            &layout,
        )
        .expect("verify");

        // Tamper boundary chaining for chunk_size>1.
        if chunk_size > 1 && steps_public.len() > 1 {
            let mut bad_steps = steps_public.clone();
            bad_steps[1].mcs_inst.x[layout.pc0] += F::ONE;
            let mut tr_bad = Poseidon2Transcript::new(b"riscv-b1-chunk-eq");
            assert!(
                fold_shard_verify_rv32_b1_with_statement_mem_init(
                    FoldingMode::Optimized,
                    &mut tr_bad,
                    &params,
                    &cpu.ccs,
                    &mem_layouts,
                    &initial_mem,
                    &bad_steps,
                    &[],
                    &proof,
                    mixers,
                    &layout,
                )
                .is_err(),
                "expected step linking failure for chunk_size={chunk_size}"
            );

            // With max_steps > program_len, the final chunk is padding and must be chained via halted_out -> halted_in.
            let last = steps_public.len() - 1;
            assert_eq!(
                steps_public[last].mcs_inst.x[layout.halted_in],
                F::ONE,
                "final chunk must have halted_in=1"
            );
            assert_eq!(
                steps_public[last - 1].mcs_inst.x[layout.halted_out],
                F::ONE,
                "chunk before final must have halted_out=1"
            );

            let mut bad_steps = steps_public.clone();
            bad_steps[last].mcs_inst.x[layout.halted_in] = F::ZERO;
            let mut tr_bad_halt = Poseidon2Transcript::new(b"riscv-b1-chunk-eq");
            assert!(
                fold_shard_verify_rv32_b1_with_statement_mem_init(
                    FoldingMode::Optimized,
                    &mut tr_bad_halt,
                    &params,
                    &cpu.ccs,
                    &mem_layouts,
                    &initial_mem,
                    &bad_steps,
                    &[],
                    &proof,
                    mixers,
                    &layout,
                )
                .is_err(),
                "expected halted_in/out step linking failure for chunk_size={chunk_size}"
            );
        }

        (layout, steps_public)
    };

    let (layout_1, steps_1) = run(1);
    let (layout_2, steps_2) = run(2);

    let start_1 = extract_boundary_state(&layout_1, &steps_1[0].mcs_inst.x).expect("boundary");
    let start_2 = extract_boundary_state(&layout_2, &steps_2[0].mcs_inst.x).expect("boundary");
    assert_eq!(start_1.pc0, start_2.pc0, "pc0 must be chunk-size invariant");
    assert_eq!(start_1.regs0, start_2.regs0, "regs0 must be chunk-size invariant");

    let end_1 = extract_boundary_state(&layout_1, &steps_1.last().expect("non-empty").mcs_inst.x).expect("boundary");
    let end_2 = extract_boundary_state(&layout_2, &steps_2.last().expect("non-empty").mcs_inst.x).expect("boundary");
    assert_eq!(end_1.pc_final, end_2.pc_final, "pc_final must be chunk-size invariant");
    assert_eq!(end_1.regs_final, end_2.regs_final, "regs_final must be chunk-size invariant");

    // Stronger equivalence: each chunk boundary in chunk_size=2 corresponds to the same boundary
    // after the same number of steps in chunk_size=1.
    let n = steps_1.len();
    let k = 2usize;
    assert_eq!(n, max_steps, "chunk_size=1 should produce one chunk per step");
    assert_eq!(steps_2.len(), n.div_ceil(k), "unexpected chunk count for chunk_size=2");

    for c in 0..steps_2.len() {
        let s = c * k;
        let e = ((c + 1) * k).min(n) - 1;
        let st_k = extract_boundary_state(&layout_2, &steps_2[c].mcs_inst.x).expect("boundary");
        let st_1s = extract_boundary_state(&layout_1, &steps_1[s].mcs_inst.x).expect("boundary");
        let st_1e = extract_boundary_state(&layout_1, &steps_1[e].mcs_inst.x).expect("boundary");

        assert_eq!(st_k.pc0, st_1s.pc0, "pc0 mismatch at chunk {c}");
        assert_eq!(st_k.regs0, st_1s.regs0, "regs0 mismatch at chunk {c}");
        assert_eq!(st_k.halted_in, st_1s.halted_in, "halted_in mismatch at chunk {c}");

        assert_eq!(st_k.pc_final, st_1e.pc_final, "pc_final mismatch at chunk {c}");
        assert_eq!(st_k.regs_final, st_1e.regs_final, "regs_final mismatch at chunk {c}");
        assert_eq!(st_k.halted_out, st_1e.halted_out, "halted_out mismatch at chunk {c}");
    }
}

#[test]
#[ignore = "RV32M end-to-end requires implicit Shout table MLE support for MUL/DIV (M5)."]
fn test_riscv_program_rv32m_full_prove_verify() {
    let xlen = 32usize;
    // This test requires implicit Shout table MLE support for RV32M (M5). When that support is
    // missing, running `cargo test -- --ignored` without a name filter should not fail.
    for opcode in [RiscvOpcode::Mul, RiscvOpcode::Div] {
        if let Err(err) = neo_memory::riscv::shout_oracle::RiscvAddressLookupOracleSparse::validate_spec(opcode, xlen) {
            eprintln!("skipping RV32M full prove+verify: {err:?}");
            return;
        }
    }
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -6,
        }, // x1 = -6
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();

    let program_bytes = encode_program(&program);
    let mut vm = RiscvCpu::new(xlen);
    vm.load_program(0, program);
    let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Minimal table set: ADD + MUL + DIV.
    let shout_table_ids: Vec<u32> = vec![3, 12, 16];
    let table_specs = HashMap::from([
        (
            3u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            12u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Mul,
                xlen,
            },
        ),
        (
            16u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Div,
                xlen,
            },
        ),
    ]);

    let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

    let cpu = R1csCpu::new(
        ccs_base,
        params.clone(),
        DummyCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
            .expect("cfg"),
        1,
    )
    .expect("shared bus inject");

    let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
        vm,
        twist,
        shout,
        /*max_steps=*/ max_steps,
        /*chunk_size=*/ 1,
        &mem_layouts,
        &HashMap::new(),
        &table_specs,
        &initial_mem,
        &cpu,
    )
    .expect("build shard witness");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mixers = default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-rv32m-full");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &cpu.ccs,
        &steps,
        &[],
        &[],
        &DummyCommit::default(),
        mixers,
    )
    .expect("prove");

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-rv32m-full");
    let _ = fold_shard_verify_rv32_b1_with_statement_mem_init(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &cpu.ccs,
        &mem_layouts,
        &initial_mem,
        &steps_public,
        &[],
        &proof,
        mixers,
        &layout,
    )
    .expect("verify");
}
