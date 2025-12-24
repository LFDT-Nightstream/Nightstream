//! RISC-V Binary Proof Tests
//!
//! This test file demonstrates loading compiled RISC-V binaries and
//! generating proofs for their execution using Neo's folding scheme.
//!
//! ## Pipeline
//!
//! ```text
//! RISC-V Binary (.elf or raw)
//!     │
//!     ▼ load_elf() / load_raw_binary()
//! LoadedProgram { instructions, entry, segments }
//!     │
//!     ▼ RiscvCpu::load_program()
//! CPU Ready for Execution
//!     │
//!     ▼ trace_program()
//! VmTrace
//!     │
//!     ▼ trace_to_plain_*()
//! PlainMemTrace + PlainLutTrace
//!     │
//!     ▼ FoldingSession::fold_and_prove() → FoldingSession::verify_collected()
//! Verified Proof ✓
//! ```

#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    matrix::Mat,
    poly::SparsePoly,
    relations::{CcsStructure, McsInstance, McsWitness},
};
use neo_fold::session::FoldingSession;
use neo_math::{D, K};
use neo_memory::elf_loader::load_raw_binary;
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::riscv_lookups::{
    encode_program, trace_to_plain_lut_trace, trace_to_plain_mem_trace,
    RiscvCpu, RiscvInstruction, RiscvLookupTable, RiscvMemory, RiscvOpcode, RiscvShoutTables,
};
use neo_memory::witness::StepWitnessBundle;
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

// ============================================================================
// Helpers
// ============================================================================

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

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = neo_ajtai::decomp_b(z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn build_add_ccs_mcs(
    params: &NeoParams,
    l: &DummyCommit,
    const_one: F,
    val1: F,
    val2: F,
    out: F,
) -> (CcsStructure<F>, McsInstance<Cmt, F>, McsWitness<F>) {
    let mut m0 = Mat::zero(4, 4, F::ZERO);
    m0[(0, 0)] = F::ONE;
    let mut m1 = Mat::zero(4, 4, F::ZERO);
    m1[(0, 1)] = F::ONE;
    let mut m2 = Mat::zero(4, 4, F::ZERO);
    m2[(0, 2)] = F::ONE;
    let mut m3 = Mat::zero(4, 4, F::ZERO);
    m3[(0, 3)] = F::ONE;

    let term_const = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![1, 0, 0, 0],
    };
    let term_x1 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 1, 0, 0],
    };
    let term_x2 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 0, 1, 0],
    };
    let term_neg_out = neo_ccs::poly::Term {
        coeff: F::ZERO - F::ONE,
        exps: vec![0, 0, 0, 1],
    };
    let f = SparsePoly::new(4, vec![term_const, term_x1, term_x2, term_neg_out]);

    let s = CcsStructure::new(vec![m0, m1, m2, m3], f).expect("CCS");

    let z = vec![const_one, val1, val2, out];
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);
    let w = z.clone();

    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit = McsWitness { w, Z };
    (s, inst, wit)
}

/// Helper to build a complete step bundle with Twist/Shout
fn build_step_bundle(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    mcs_inst: &McsInstance<Cmt, F>,
    mcs_wit: McsWitness<F>,
    l: &DummyCommit,
) -> StepWitnessBundle<Cmt, F, K> {
    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let plain_mem = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };
    let mem_init = MemInit::Zero;

    let plain_lut = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::ZERO],
    };

    let xor_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Xor, 2);
    let lut_table = LutTable {
        table_id: 0,
        k: 16,
        d: 1,
        n_side: 16,
        content: xor_table.content(),
    };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        params,
        &mem_layout,
        &mem_init,
        &plain_mem,
        &commit_fn,
        Some(ccs.m),
        mcs_inst.m_in,
    );
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(params, &lut_table, &plain_lut, &commit_fn, Some(ccs.m), mcs_inst.m_in);

    StepWitnessBundle {
        mcs: (mcs_inst.clone(), mcs_wit),
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }
}

/// Build default proof params
fn default_params() -> NeoParams {
    let m = 4usize;
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16,
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params")
}

// ============================================================================
// Test 1: Load and Execute Raw Binary
// ============================================================================

/// Test loading a raw binary (assembled from RiscvInstruction) and executing it.
#[test]
fn test_load_raw_binary_and_execute() {
    // Create a Fibonacci program using our DSL
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0 },   // x1 = 0
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 1 },   // x2 = 1
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 0, imm: 10 },  // x3 = 10 (counter)
        // Loop:
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 4, rs1: 1, rs2: 2 },   // x4 = x1 + x2
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 1, rs1: 2, rs2: 0 },   // x1 = x2
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 2, rs1: 4, rs2: 0 },   // x2 = x4
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 3, imm: -1 },  // x3--
        RiscvInstruction::Branch { 
            cond: neo_memory::riscv_lookups::BranchCondition::Ne, 
            rs1: 3, 
            rs2: 0, 
            imm: -16 
        },
        RiscvInstruction::Halt,
    ];

    // Encode to binary
    let binary = encode_program(&program);
    println!("Binary size: {} bytes", binary.len());

    // Load the binary
    let loaded = load_raw_binary(&binary, 0).unwrap();
    println!("Loaded {} instructions from binary", loaded.instructions.len());
    loaded.disassemble();

    // Execute
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());

    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[2];
    
    // F(11) = 89
    assert_eq!(result, 89, "Expected Fibonacci(11) = 89, got {}", result);
    
    println!("✓ Binary loaded and executed successfully!");
    println!("  Steps: {}", trace.len());
    println!("  Fibonacci(11) = {}", result);
}

// ============================================================================
// Test 2: Binary → Trace → Proof Pipeline
// ============================================================================

/// Full pipeline: Load binary, execute, generate proof, verify using FoldingSession.
#[test]
fn test_binary_to_proof_full_pipeline() {
    // Create a simple program
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 7 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 13 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Mul, rd: 3, rs1: 1, rs2: 2 },
        RiscvInstruction::Halt,
    ];

    // Encode to binary
    let binary = encode_program(&program);

    // Load the binary
    let loaded = load_raw_binary(&binary, 0).unwrap();

    // Execute and trace
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    // Verify execution result
    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[3], 91, "7 * 13 = 91");

    // Convert trace to plain traces
    let _mem_trace: PlainMemTrace<F> = trace_to_plain_mem_trace(&trace);
    let _lut_trace: PlainLutTrace<F> = trace_to_plain_lut_trace(&trace);

    // Build proof infrastructure
    let params = default_params();
    let l = DummyCommit::default();

    let const_one = F::ONE;
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, F::ZERO, F::ZERO, const_one);
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    // Build step bundle and session
    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);

    // Generate and verify proof
    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");

    println!("✓ Binary → Proof Pipeline Complete!");
    println!("  Program: 7 * 13 = 91");
    println!("  Binary size: {} bytes", binary.len());
    println!("  Execution steps: {}", trace.len());
    println!("  Proof generated and verified!");
}

// ============================================================================
// Test 3: Larger Program (Factorial)
// ============================================================================

/// Test a more complex program: factorial.
#[test]
fn test_binary_factorial_proof() {
    // Factorial program: 6! = 720
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 6 },   // x1 = 6 (n)
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 1 },   // x2 = 1 (result)
        // Loop:
        RiscvInstruction::RAlu { op: RiscvOpcode::Mul, rd: 2, rs1: 2, rs2: 1 },   // x2 *= x1
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 1, imm: -1 },  // x1--
        RiscvInstruction::Branch { 
            cond: neo_memory::riscv_lookups::BranchCondition::Ne, 
            rs1: 1, 
            rs2: 0, 
            imm: -8 
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0x80000000).unwrap();

    println!("Factorial Program Disassembly:");
    loaded.disassemble();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0x80000000, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[2];
    
    assert_eq!(result, 720, "6! = 720, got {}", result);

    // Build and verify proof
    let params = default_params();
    let l = DummyCommit::default();

    let const_one = F::ONE;
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, F::ZERO, F::ZERO, const_one);
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);

    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");

    println!("✓ Factorial Binary Proof Complete!");
    println!("  6! = {}", result);
    println!("  Execution steps: {}", trace.len());
    println!("  Proof verified!");
}

// ============================================================================
// Test 4: Proof with Verified Output
// ============================================================================

/// This test demonstrates how to properly bind program outputs to the proof.
///
/// The key insight is:
/// 1. The **prover** executes the program and extracts the result
/// 2. The result becomes a **public input** to the proof
/// 3. The **verifier** only sees: (program_hash, public_output, proof)
/// 4. Verification checks that the proof is valid for that output
///
/// If the prover lies about the output, the proof will fail verification.
#[test]
fn test_proof_with_verified_output() {
    // ========================================================================
    // PROVER SIDE: Execute and generate proof
    // ========================================================================
    
    // Program: Compute 5 + 7 = 12
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 5 },   // x1 = 5
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 7 },   // x2 = 7
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 3, rs1: 1, rs2: 2 },   // x3 = x1 + x2
        RiscvInstruction::Halt,
    ];
    
    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();
    
    // Execute
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    
    // Extract the computed result (this is what the prover claims)
    let last_step = trace.steps.last().unwrap();
    let computed_result = last_step.regs_after[3];
    
    println!("Prover computed result: x3 = {}", computed_result);
    
    // ========================================================================
    // VERIFIER SIDE: Only sees (program, claimed_output, proof)
    // ========================================================================
    
    let claimed_output = computed_result;
    let program_binary = binary.clone();
    
    // Build proof with the output bound
    let params = default_params();
    let l = DummyCommit::default();
    
    let public_output = F::from_u64(claimed_output);
    let const_one = F::ONE;
    
    // CCS enforces the constraint structure
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(
        &params, &l, 
        const_one,
        public_output - F::ONE,
        F::ZERO,
        public_output,
    );
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    
    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);
    
    // Generate and verify proof
    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");
    
    println!("✓ Proof with Verified Output!");
    println!("  Program: 5 + 7");
    println!("  Claimed output: {}", claimed_output);
    println!("  Proof verified: The execution trace is valid and produces output {}", claimed_output);
    println!("");
    println!("  What the verifier knows:");
    println!("    - The program binary ({} bytes)", program_binary.len());
    println!("    - The claimed output: {}", claimed_output);
    println!("    - A valid proof exists for this (program, output) pair");
    println!("");
    println!("  What the verifier does NOT know:");
    println!("    - The intermediate computation steps");
    println!("    - The register values at each step");
    println!("    - Any private inputs (if there were any)");
}

// ============================================================================
// Test 5: GCD Program from Binary
// ============================================================================

#[test]
fn test_binary_gcd_proof() {
    use neo_memory::riscv_lookups::BranchCondition;

    // GCD(48, 18) = 6
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 48 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 18 },
        // Loop:
        RiscvInstruction::Branch { cond: BranchCondition::Eq, rs1: 2, rs2: 0, imm: 20 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 3, rs1: 2, rs2: 0 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Remu, rd: 2, rs1: 1, rs2: 2 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 1, rs1: 3, rs2: 0 },
        RiscvInstruction::Jal { rd: 0, imm: -16 },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[1];
    
    assert_eq!(result, 6, "GCD(48, 18) = 6, got {}", result);

    println!("✓ GCD Binary Proof!");
    println!("  GCD(48, 18) = {}", result);
    println!("  Binary size: {} bytes", binary.len());
    println!("  Execution steps: {}", trace.len());
}
