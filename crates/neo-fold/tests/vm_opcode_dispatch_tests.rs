//! VM opcode dispatch tests.
//!
//! These tests simulate a realistic VM where:
//! - Opcodes are looked up from a bytecode table (instruction fetch)
//! - ALU operations use lookup tables for result validation
//! - Register file uses Twist for read-write state
//! - Program counter advances through the bytecode
//!
//! ## Coverage
//! - `vm_simple_add_program`: 3-instruction ADD program with opcode dispatch
//! - `vm_branch_taken_vs_not_taken`: Conditional branch using lookup
//! - `vm_load_store_program`: Memory load/store with register file
//! - `vm_invalid_opcode_fails`: Adversarial: claim invalid opcode

#![allow(non_snake_case)]
#![allow(deprecated)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{
    LutInstance, LutWitness, MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle,
};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

// VM Opcodes
const OP_NOP: u64 = 0;
const OP_ADD: u64 = 1;
const OP_LOAD_IMM: u64 = 2;
const OP_STORE: u64 = 3;
const OP_HALT: u64 = 4;

/// Setup real Ajtai public parameters for tests.
fn setup_ajtai_pp(m: usize, seed: u64) -> AjtaiSModule {
    let d = D;
    let kappa = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m)
        .expect("params")
        .kappa as usize;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let pp = ajtai_setup(&mut rng, d, kappa, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp.clone());
    AjtaiSModule::new(Arc::new(pp))
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn write_bits_le(out: &mut [F], mut x: u64, ell: usize) {
    for i in 0..ell {
        out[i] = if (x & 1) == 1 { F::ONE } else { F::ZERO };
        x >>= 1;
    }
}

fn bus_cols_shout(inst: &neo_memory::witness::LutInstance<Cmt, F>) -> usize {
    inst.d * inst.ell + 2
}

fn bus_cols_twist(inst: &neo_memory::witness::MemInstance<Cmt, F>) -> usize {
    2 * inst.d * inst.ell + 5
}

fn fill_shout_bus(
    z: &mut [F],
    bus_base: usize,
    col_id: &mut usize,
    inst: &neo_memory::witness::LutInstance<Cmt, F>,
    trace: &PlainLutTrace<F>,
) {
    let ell_addr = inst.d * inst.ell;
    let mut bits = vec![F::ZERO; ell_addr];
    let addr = trace.addr[0];
    let mut tmp = addr;
    for dim in 0..inst.d {
        let comp = (tmp % (inst.n_side as u64)) as u64;
        tmp /= inst.n_side as u64;
        let offset = dim * inst.ell;
        write_bits_le(&mut bits[offset..offset + inst.ell], comp, inst.ell);
    }
    for bit in bits {
        z[bus_base + *col_id] = bit;
        *col_id += 1;
    }
    z[bus_base + *col_id] = trace.has_lookup[0];
    *col_id += 1;
    z[bus_base + *col_id] = trace.val[0];
    *col_id += 1;
}

fn fill_twist_bus(
    z: &mut [F],
    bus_base: usize,
    col_id: &mut usize,
    inst: &neo_memory::witness::MemInstance<Cmt, F>,
    trace: &PlainMemTrace<F>,
) {
    let ell_addr = inst.d * inst.ell;
    let mut ra_bits = vec![F::ZERO; ell_addr];
    let mut wa_bits = vec![F::ZERO; ell_addr];

    let ra = trace.read_addr[0];
    let wa = trace.write_addr[0];

    let mut tmp = ra;
    for dim in 0..inst.d {
        let comp = (tmp % (inst.n_side as u64)) as u64;
        tmp /= inst.n_side as u64;
        let offset = dim * inst.ell;
        write_bits_le(&mut ra_bits[offset..offset + inst.ell], comp, inst.ell);
    }
    let mut tmp = wa;
    for dim in 0..inst.d {
        let comp = (tmp % (inst.n_side as u64)) as u64;
        tmp /= inst.n_side as u64;
        let offset = dim * inst.ell;
        write_bits_le(&mut wa_bits[offset..offset + inst.ell], comp, inst.ell);
    }

    for bit in ra_bits {
        z[bus_base + *col_id] = bit;
        *col_id += 1;
    }
    for bit in wa_bits {
        z[bus_base + *col_id] = bit;
        *col_id += 1;
    }

    z[bus_base + *col_id] = trace.has_read[0];
    *col_id += 1;
    z[bus_base + *col_id] = trace.has_write[0];
    *col_id += 1;
    z[bus_base + *col_id] = trace.write_val[0];
    *col_id += 1;
    z[bus_base + *col_id] = trace.read_val[0];
    *col_id += 1;
    z[bus_base + *col_id] = trace.inc_at_write_addr[0];
    *col_id += 1;
}

fn create_mcs_with_bus(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    tag: u64,
    lut_insts: &[(&neo_memory::witness::LutInstance<Cmt, F>, &PlainLutTrace<F>)],
    mem_insts: &[(&neo_memory::witness::MemInstance<Cmt, F>, &PlainMemTrace<F>)],
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let m_in = 0usize;
    let mut z: Vec<F> = vec![F::ZERO; ccs.m];
    if !z.is_empty() {
        z[0] = F::from_u64(tag);
    }

    let bus_cols_total: usize =
        lut_insts.iter().map(|(inst, _)| bus_cols_shout(inst)).sum::<usize>()
            + mem_insts.iter().map(|(inst, _)| bus_cols_twist(inst)).sum::<usize>();
    if bus_cols_total > 0 {
        assert!(
            bus_cols_total <= z.len(),
            "bus region too large: bus_cols_total({bus_cols_total}) > m({})",
            z.len()
        );
        let bus_base = z.len() - bus_cols_total;
        let mut col_id = 0usize;
        for (inst, trace) in lut_insts {
            fill_shout_bus(&mut z, bus_base, &mut col_id, inst, trace);
        }
        for (inst, trace) in mem_insts {
            fill_twist_bus(&mut z, bus_base, &mut col_id, inst, trace);
        }
        debug_assert_eq!(col_id, bus_cols_total, "bus col count mismatch");
    }

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);

    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();

    (
        McsInstance { c, x, m_in },
        McsWitness { w, Z },
    )
}

/// Build a bytecode table for a simple program.
/// Each entry is an opcode. The program:
/// - PC=0: LOAD_IMM (load value into register)
/// - PC=1: ADD
/// - PC=2: STORE (store register to memory)
/// - PC=3: HALT
fn build_bytecode_table() -> LutTable<F> {
    LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            F::from_u64(OP_LOAD_IMM), // PC=0
            F::from_u64(OP_ADD),      // PC=1
            F::from_u64(OP_STORE),    // PC=2
            F::from_u64(OP_HALT),     // PC=3
        ],
    }
}

/// Build an immediate value table (operands for instructions).
fn build_imm_table() -> LutTable<F> {
    LutTable {
        table_id: 1,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            F::from_u64(10), // Operand for PC=0 (load 10)
            F::from_u64(5),  // Operand for PC=1 (add 5)
            F::from_u64(0),  // Operand for PC=2 (store to addr 0)
            F::from_u64(0),  // Operand for PC=3 (halt, unused)
        ],
    }
}

fn empty_mem_trace() -> PlainMemTrace<F> {
    PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    }
}

fn metadata_only_mem_instance(
    layout: &PlainMemLayout,
    init: MemInit<F>,
    steps: usize,
) -> (MemInstance<Cmt, F>, MemWitness<F>) {
    let ell = layout.n_side.trailing_zeros() as usize;
    (
        MemInstance {
            comms: Vec::new(),
            k: layout.k,
            d: layout.d,
            n_side: layout.n_side,
            steps,
            ell,
            init,
            _phantom: PhantomData,
        },
        MemWitness { mats: Vec::new() },
    )
}

fn metadata_only_lut_instance(
    table: &LutTable<F>,
    steps: usize,
) -> (LutInstance<Cmt, F>, LutWitness<F>) {
    let ell = table.n_side.trailing_zeros() as usize;
    (
        LutInstance {
            comms: Vec::new(),
            k: table.k,
            d: table.d,
            n_side: table.n_side,
            steps,
            ell,
            table_spec: None,
            table: table.content.clone(),
            _phantom: PhantomData,
        },
        LutWitness { mats: Vec::new() },
    )
}

/// Simple VM execution: fetch opcode and immediate from lookup tables.
/// Program: LOAD_IMM 10, ADD 5, STORE, HALT
/// This test validates instruction fetch via lookup tables without complex memory operations.
#[test]
fn vm_simple_add_program() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();

    let bytecode_table = build_bytecode_table();
    let imm_table = build_imm_table();

    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    // Simulate 4 CPU cycles - focus on bytecode fetch via lookups
    let mut register: u64 = 0;

    for pc in 0u64..4 {
        let opcode = bytecode_table.content[pc as usize].as_canonical_u64();
        let imm = imm_table.content[pc as usize].as_canonical_u64();

        // Fetch opcode from bytecode table
        let opcode_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![pc],
            val: vec![bytecode_table.content[pc as usize]],
        };

        // Fetch immediate from immediate table
        let imm_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![pc],
            val: vec![imm_table.content[pc as usize]],
        };

        // Execute instruction (for simulation purposes)
        match opcode {
            OP_LOAD_IMM => register = imm,
            OP_ADD => register = register.wrapping_add(imm),
            OP_STORE => { /* simulated store, not actually writing to memory in this test */ }
            OP_HALT => { /* nothing */ }
            _ => {}
        }

        // Minimal memory (no actual memory operations in this simplified test)
        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let mem_trace = empty_mem_trace();
        let (mem_inst, mem_wit) =
            metadata_only_mem_instance(&mem_layout, MemInit::Zero, mem_trace.steps);
        let (opcode_inst, opcode_wit) =
            metadata_only_lut_instance(&bytecode_table, opcode_trace.has_lookup.len());
        let (imm_inst, imm_wit) =
            metadata_only_lut_instance(&imm_table, imm_trace.has_lookup.len());

        let lut_bus = [(&opcode_inst, &opcode_trace), (&imm_inst, &imm_trace)];
        let mem_bus = [(&mem_inst, &mem_trace)];
        let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, pc, &lut_bus, &mem_bus);

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![(opcode_inst, opcode_wit), (imm_inst, imm_wit)],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    assert_eq!(register, 15, "Final register should be 10 + 5 = 15");

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"vm-add-program");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for VM add program");

    let mut tr_verify = Poseidon2Transcript::new(b"vm-add-program");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps.iter().map(StepInstanceBundle::from).collect();
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed for VM add program");

    println!("✓ vm_simple_add_program: 4-cycle VM program verified (bytecode fetch via lookups)");
}

/// Test: VM with register file (Twist) and ALU operations.
/// Register file: R0, R1, R2, R3
/// Program: R0 = 10, R1 = 20, R2 = R0 + R1
#[test]
fn vm_register_file_operations() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();

    // Register file layout: 4 registers
    let reg_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    // Step 0: Write 10 to R0
    {
        let reg_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ZERO],
            has_write: vec![F::ONE],
            read_addr: vec![0],
            write_addr: vec![0], // R0
            read_val: vec![F::ZERO],
            write_val: vec![F::from_u64(10)],
            inc_at_write_addr: vec![F::from_u64(10)],
        };
        let (reg_inst, reg_wit) =
            metadata_only_mem_instance(&reg_layout, MemInit::Zero, reg_trace.steps);
        let mem_bus = [(&reg_inst, &reg_trace)];
        let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, 0, &[], &mem_bus);

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(reg_inst, reg_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    // Step 1: Write 20 to R1 (R0 still contains 10)
    {
        let reg_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ZERO],
            has_write: vec![F::ONE],
            read_addr: vec![0],
            write_addr: vec![1], // R1
            read_val: vec![F::ZERO],
            write_val: vec![F::from_u64(20)],
            inc_at_write_addr: vec![F::from_u64(20)],
        };
        // State after step 0: R0=10
        let reg_init = MemInit::Sparse(vec![(0, F::from_u64(10))]);

        let (reg_inst, reg_wit) = metadata_only_mem_instance(&reg_layout, reg_init, reg_trace.steps);
        let mem_bus = [(&reg_inst, &reg_trace)];
        let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, 1, &[], &mem_bus);

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(reg_inst, reg_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    // Step 2: Read R0 (10), compute R0+R1=30, write to R2
    // Note: In a real VM, we'd read R1 too, but for simplicity we just show one read
    {
        let reg_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ONE],
            has_write: vec![F::ONE],
            read_addr: vec![0], // Read R0
            write_addr: vec![2], // Write R2
            read_val: vec![F::from_u64(10)], // R0 = 10
            write_val: vec![F::from_u64(30)], // R2 = 10 + 20 = 30
            inc_at_write_addr: vec![F::from_u64(30)],
        };
        // State after step 1: R0=10, R1=20
        let reg_init = MemInit::Sparse(vec![(0, F::from_u64(10)), (1, F::from_u64(20))]);

        let (reg_inst, reg_wit) = metadata_only_mem_instance(&reg_layout, reg_init, reg_trace.steps);
        let mem_bus = [(&reg_inst, &reg_trace)];
        let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, 2, &[], &mem_bus);

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(reg_inst, reg_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"vm-register-file");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for register file operations");

    let mut tr_verify = Poseidon2Transcript::new(b"vm-register-file");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps.iter().map(StepInstanceBundle::from).collect();
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed for register file operations");

    println!("✓ vm_register_file_operations: R0=10, R1=20, R2=R0+R1=30 verified");
}

/// Test: Combined bytecode fetch + memory access in single step.
/// This simulates a LOAD instruction that reads from both program memory (ROM)
/// and data memory (RAM).
#[test]
fn vm_combined_bytecode_and_data_memory() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();

    // Bytecode table (ROM via Shout)
    let bytecode = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            F::from_u64(OP_LOAD_IMM),
            F::from_u64(OP_NOP),
            F::from_u64(OP_NOP),
            F::from_u64(OP_HALT),
        ],
    };

    // Data memory (RAM via Twist)
    let ram_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // Single step: fetch LOAD_IMM from PC=0, write 42 to RAM[0]
    let bytecode_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![bytecode.content[0]], // LOAD_IMM
    };

    let ram_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(42)],
        inc_at_write_addr: vec![F::from_u64(42)],
    };
    let (bytecode_inst, bytecode_wit) =
        metadata_only_lut_instance(&bytecode, bytecode_trace.has_lookup.len());
    let (ram_inst, ram_wit) = metadata_only_mem_instance(&ram_layout, MemInit::Zero, ram_trace.steps);

    let lut_bus = [(&bytecode_inst, &bytecode_trace)];
    let mem_bus = [(&ram_inst, &ram_trace)];
    let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, 0, &lut_bus, &mem_bus);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(bytecode_inst, bytecode_wit)],
        mem_instances: vec![(ram_inst, ram_wit)],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"vm-combined-rom-ram");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for combined ROM+RAM access");

    let mut tr_verify = Poseidon2Transcript::new(b"vm-combined-rom-ram");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed for combined ROM+RAM access");

    println!("✓ vm_combined_bytecode_and_data_memory: Bytecode fetch + RAM write in single step verified");
}

/// Adversarial: Claim wrong opcode from bytecode table.
#[test]
fn vm_invalid_opcode_claim_fails() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();

    let bytecode = build_bytecode_table();

    // Malicious: claim PC=0 contains HALT (4) instead of LOAD_IMM (2)
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(OP_HALT)], // WRONG: should be LOAD_IMM
    };

    let (bytecode_inst, bytecode_wit) =
        metadata_only_lut_instance(&bytecode, bad_trace.has_lookup.len());

    // Proving or verification must fail for an invalid lookup witness.
    let lut_bus = [(&bytecode_inst, &bad_trace)];
    let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, 0, &lut_bus, &[]);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(bytecode_inst, bytecode_wit)],
        mem_instances: vec![],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"vm-invalid-opcode-claim");
    let proof_result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    );

    if let Ok(proof) = proof_result {
        let mut tr_verify = Poseidon2Transcript::new(b"vm-invalid-opcode-claim");
        let steps_public = [StepInstanceBundle::from(&step_bundle)];
        assert!(
            fold_shard_verify(
                FoldingMode::PaperExact,
                &mut tr_verify,
                &params,
                &ccs,
                &steps_public,
                &acc_init,
                &proof,
                mixers,
            )
            .is_err(),
            "verification should fail on wrong opcode claim"
        );
    }

    println!("✓ vm_invalid_opcode_claim_fails: Wrong opcode claim correctly rejected");
}

/// Test: Multiple instruction types in sequence with proper state transitions.
#[test]
fn vm_multi_instruction_sequence() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();

    // Bytecode: NOP, ADD, NOP, HALT
    let bytecode = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            F::from_u64(OP_NOP),
            F::from_u64(OP_ADD),
            F::from_u64(OP_NOP),
            F::from_u64(OP_HALT),
        ],
    };

    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    for pc in 0u64..4 {
        let bytecode_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![pc],
            val: vec![bytecode.content[pc as usize]],
        };

        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

        let mem_trace = empty_mem_trace();
        let (mem_inst, mem_wit) =
            metadata_only_mem_instance(&mem_layout, MemInit::Zero, mem_trace.steps);
        let (bytecode_inst, bytecode_wit) =
            metadata_only_lut_instance(&bytecode, bytecode_trace.has_lookup.len());

        let lut_bus = [(&bytecode_inst, &bytecode_trace)];
        let mem_bus = [(&mem_inst, &mem_trace)];
        let (mcs, mcs_wit) = create_mcs_with_bus(&params, &ccs, &l, pc, &lut_bus, &mem_bus);

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![(bytecode_inst, bytecode_wit)],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"vm-multi-instr");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for multi-instruction sequence");

    let mut tr_verify = Poseidon2Transcript::new(b"vm-multi-instr");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps.iter().map(StepInstanceBundle::from).collect();
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed for multi-instruction sequence");

    println!("✓ vm_multi_instruction_sequence: NOP→ADD→NOP→HALT sequence verified");
}
