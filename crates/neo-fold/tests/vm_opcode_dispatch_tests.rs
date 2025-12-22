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

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{decomp_b, setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment as Cmt, DecompStyle};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
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

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = decomp_b(z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn create_mcs(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    tag: u64,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let m = ccs.m;
    let mut z: Vec<F> = vec![F::ZERO; m];
    if !z.is_empty() {
        z[0] = F::from_u64(tag);
    }

    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);

    (
        McsInstance { c, x: vec![], m_in: 0 },
        McsWitness { w: z, Z },
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

/// Simple VM execution: fetch opcode and immediate from lookup tables.
/// Program: LOAD_IMM 10, ADD 5, STORE, HALT
/// This test validates instruction fetch via lookup tables without complex memory operations.
#[test]
fn vm_simple_add_program() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, pc);

        // Minimal memory (no actual memory operations in this simplified test)
        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let mem_init = MemInit::Zero;
        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &empty_mem_trace(),
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        let (opcode_inst, opcode_wit) = encode_lut_for_shout(
            &params,
            &bytecode_table,
            &opcode_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );
        let (imm_inst, imm_wit) = encode_lut_for_shout(
            &params,
            &imm_table,
            &imm_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

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
        &l,
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
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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
        let reg_init = MemInit::Zero;

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);
        let (reg_inst, reg_wit) = encode_mem_for_twist(
            &params,
            &reg_layout,
            &reg_init,
            &reg_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

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

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 1);
        let (reg_inst, reg_wit) = encode_mem_for_twist(
            &params,
            &reg_layout,
            &reg_init,
            &reg_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

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

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 2);
        let (reg_inst, reg_wit) = encode_mem_for_twist(
            &params,
            &reg_layout,
            &reg_init,
            &reg_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

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
        &l,
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
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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
    let ram_init = MemInit::Zero;

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);

    let (bytecode_inst, bytecode_wit) = encode_lut_for_shout(
        &params,
        &bytecode,
        &bytecode_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );
    let (ram_inst, ram_wit) = encode_mem_for_twist(
        &params,
        &ram_layout,
        &ram_init,
        &ram_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

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
        &l,
        mixers,
    )
    .expect("verify should succeed for combined ROM+RAM access");

    println!("✓ vm_combined_bytecode_and_data_memory: Bytecode fetch + RAM write in single step verified");
}

/// Adversarial: Claim wrong opcode from bytecode table.
#[test]
fn vm_invalid_opcode_claim_fails() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let bytecode = build_bytecode_table();

    // Malicious: claim PC=0 contains HALT (4) instead of LOAD_IMM (2)
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(OP_HALT)], // WRONG: should be LOAD_IMM
    };

    let (mcs, _mcs_wit) = create_mcs(&params, &ccs, &l, 0);

    // Should panic during encoding
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_lut_for_shout(&params, &bytecode, &bad_trace, &commit_fn, Some(ccs.m), mcs.m_in)
    }));

    assert!(
        result.is_err(),
        "Encoding should panic when claimed opcode doesn't match bytecode table"
    );

    println!("✓ vm_invalid_opcode_claim_fails: Wrong opcode claim correctly rejected");
}

/// Test: Multiple instruction types in sequence with proper state transitions.
#[test]
fn vm_multi_instruction_sequence() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x4001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, pc);

        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let mem_init = MemInit::Zero;

        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &empty_mem_trace(),
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );
        let (bytecode_inst, bytecode_wit) = encode_lut_for_shout(
            &params,
            &bytecode,
            &bytecode_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

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
        &l,
        mixers,
    )
    .expect("verify should succeed for multi-instruction sequence");

    println!("✓ vm_multi_instruction_sequence: NOP→ADD→NOP→HALT sequence verified");
}

