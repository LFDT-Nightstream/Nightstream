//! Tests for RISC-V CCS constraint synthesis.

use neo_memory::riscv::ccs::*;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn test_witness_layout_indices() {
    let layout = RiscvWitnessLayout::new();

    // Check that indices don't overlap
    assert_eq!(RiscvWitnessLayout::CONST_1, 0);
    assert_eq!(RiscvWitnessLayout::PC_IN, 1);
    assert_eq!(RiscvWitnessLayout::PC_OUT, 2);
    assert_eq!(RiscvWitnessLayout::OUTPUT_VAL, 3);

    // Check register indices
    assert_eq!(RiscvWitnessLayout::reg_in(0), 4);
    assert_eq!(RiscvWitnessLayout::reg_in(31), 35);
    assert_eq!(RiscvWitnessLayout::reg_out(0), 36);
    assert_eq!(RiscvWitnessLayout::reg_out(31), 67);

    // Verify m_in (public inputs)
    assert_eq!(layout.m_in, 4);
}

#[test]
fn test_witness_creation() {
    let layout = RiscvWitnessLayout::new();
    let mut witness = RiscvStepWitness::new(layout);

    // Check const_1 is set
    assert_eq!(witness.values[RiscvWitnessLayout::CONST_1], F::ONE);

    // Set and verify PC
    witness.set_pc(0x1000, 0x1004);
    assert_eq!(witness.values[RiscvWitnessLayout::PC_IN], F::from_u64(0x1000));
    assert_eq!(witness.values[RiscvWitnessLayout::PC_OUT], F::from_u64(0x1004));

    // Set and verify registers
    witness.set_reg_in(1, 42);
    witness.set_reg_out(1, 100);
    assert_eq!(witness.values[RiscvWitnessLayout::reg_in(1)], F::from_u64(42));
    assert_eq!(witness.values[RiscvWitnessLayout::reg_out(1)], F::from_u64(100));

    // x0 should remain 0 even if we try to set it
    witness.set_reg_in(0, 999);
    witness.set_reg_out(0, 999);
    assert_eq!(witness.values[RiscvWitnessLayout::reg_in(0)], F::ZERO);
    assert_eq!(witness.values[RiscvWitnessLayout::reg_out(0)], F::ZERO);
}

#[test]
fn test_add_constraint_ccs_satisfied() {
    // Build a simple CCS: z[4] = z[2] + z[3]
    let n = 8;
    let m = 8;
    let ccs = build_add_constraint_ccs(n, m, 2, 3, 4);

    // Valid witness: z[2]=5, z[3]=7, z[4]=12
    let mut witness = vec![F::ZERO; m];
    witness[0] = F::ONE; // const_1
    witness[2] = F::from_u64(5); // in1
    witness[3] = F::from_u64(7); // in2
    witness[4] = F::from_u64(12); // out = in1 + in2

    assert!(
        check_ccs_satisfaction(&ccs, &witness),
        "Valid witness should satisfy CCS"
    );
}

#[test]
fn test_add_constraint_ccs_unsatisfied() {
    // Build a simple CCS: z[4] = z[2] + z[3]
    let n = 8;
    let m = 8;
    let ccs = build_add_constraint_ccs(n, m, 2, 3, 4);

    // Invalid witness: z[2]=5, z[3]=7, z[4]=10 (should be 12)
    let mut witness = vec![F::ZERO; m];
    witness[0] = F::ONE;
    witness[2] = F::from_u64(5);
    witness[3] = F::from_u64(7);
    witness[4] = F::from_u64(10); // Wrong!

    assert!(
        !check_ccs_satisfaction(&ccs, &witness),
        "Invalid witness should not satisfy CCS"
    );
}

#[test]
fn test_pc_increment_ccs_satisfied() {
    let layout = RiscvWitnessLayout::new();
    let ccs = build_pc_increment_ccs(layout.m, layout.m, &layout);

    // Build a valid witness
    let mut witness = RiscvStepWitness::new(layout);
    witness.set_pc(0x1000, 0x1004); // PC advances by 4

    assert!(
        check_ccs_satisfaction(&ccs, witness.as_slice()),
        "PC increment by 4 should satisfy CCS"
    );
}

#[test]
fn test_pc_increment_ccs_unsatisfied() {
    let layout = RiscvWitnessLayout::new();
    let ccs = build_pc_increment_ccs(layout.m, layout.m, &layout);

    // Build an invalid witness
    let mut witness = RiscvStepWitness::new(layout);
    witness.set_pc(0x1000, 0x1008); // PC advances by 8 (should be 4)

    assert!(
        !check_ccs_satisfaction(&ccs, witness.as_slice()),
        "PC increment by 8 should not satisfy CCS"
    );
}

#[test]
fn test_riscv_alu_step_ccs() {
    let layout = RiscvWitnessLayout::new();
    let ccs = build_riscv_alu_step_ccs(&layout);

    // Build a valid witness with PC increment
    let mut witness = RiscvStepWitness::new(layout);
    witness.set_pc(0x100, 0x104);
    witness.set_reg_in(1, 10);
    witness.set_reg_out(1, 10); // Unchanged register
    witness.set_output(42);

    assert!(
        check_ccs_satisfaction(&ccs, witness.as_slice()),
        "Valid ALU step should satisfy CCS"
    );
}

#[test]
fn test_public_inputs_extraction() {
    let layout = RiscvWitnessLayout::new();
    let mut witness = RiscvStepWitness::new(layout.clone());

    witness.set_pc(0x1000, 0x1004);
    witness.set_output(999);

    let public = witness.public_inputs();
    assert_eq!(public.len(), 4);
    assert_eq!(public[0], F::ONE); // const_1
    assert_eq!(public[1], F::from_u64(0x1000)); // pc_in
    assert_eq!(public[2], F::from_u64(0x1004)); // pc_out
    assert_eq!(public[3], F::from_u64(999)); // output_val
}

#[test]
fn test_witness_from_trace() {
    use neo_vm_trace::{StepTrace, VmTrace};

    // Create a synthetic trace manually
    let mut trace = VmTrace::new();

    // Step 1: x1 = 42 (ADDI x1, x0, 42)
    let mut regs_before: Vec<u64> = vec![0; 32];
    let mut regs_after: Vec<u64> = vec![0; 32];
    regs_after[1] = 42;

    trace.steps.push(StepTrace {
        cycle: 0,
        pc_before: 0u64,
        pc_after: 4u64,
        opcode: 0x13, // ADDI
        regs_before: regs_before.clone(),
        regs_after: regs_after.clone(),
        twist_events: vec![],
        shout_events: vec![],
        halted: false,
    });

    // Step 2: x2 = 50 (ADDI x2, x1, 8)
    regs_before = regs_after.clone();
    regs_after[2] = 50;

    trace.steps.push(StepTrace {
        cycle: 1,
        pc_before: 4,
        pc_after: 8,
        opcode: 0x13,
        regs_before,
        regs_after,
        twist_events: vec![],
        shout_events: vec![],
        halted: false,
    });

    // Generate witnesses
    let layout = RiscvWitnessLayout::new();
    let witnesses = witnesses_from_trace(&trace, &layout);

    assert_eq!(witnesses.len(), 2, "Should generate 2 witnesses");

    // Check first witness
    let w0 = &witnesses[0];
    assert_eq!(w0.values[RiscvWitnessLayout::PC_IN], F::ZERO);
    assert_eq!(w0.values[RiscvWitnessLayout::PC_OUT], F::from_u64(4));

    // After first step, x1 should be 42
    assert_eq!(w0.values[RiscvWitnessLayout::reg_out(1)], F::from_u64(42));

    // Check second witness
    let w1 = &witnesses[1];
    assert_eq!(w1.values[RiscvWitnessLayout::PC_IN], F::from_u64(4));
    assert_eq!(w1.values[RiscvWitnessLayout::PC_OUT], F::from_u64(8));
    assert_eq!(w1.values[RiscvWitnessLayout::reg_in(1)], F::from_u64(42));
    assert_eq!(w1.values[RiscvWitnessLayout::reg_out(2)], F::from_u64(50));
}
