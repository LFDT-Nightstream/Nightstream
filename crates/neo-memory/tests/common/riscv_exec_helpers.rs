use neo_memory::riscv::lookups::*;
use neo_vm_trace::{trace_program, Twist, TwistId, TwistOpKind};

pub fn run_program(instructions: Vec<RiscvInstruction>, xlen: usize) -> Vec<u64> {
    let program_bytes = encode_program(&instructions);
    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, instructions);
    let memory = RiscvMemory::with_program_in_twist(xlen, TwistId(1), 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);

    let trace = trace_program(cpu, memory, shout, 1000).expect("execution failed");
    assert!(trace.did_halt(), "program should halt");

    trace.steps.last().unwrap().regs_after.clone()
}

#[allow(dead_code)]
pub fn run_program_with_memory(
    instructions: Vec<RiscvInstruction>,
    xlen: usize,
    initial_memory: Vec<(u64, u64)>,
) -> (Vec<u64>, RiscvMemory) {
    let program_bytes = encode_program(&instructions);
    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, instructions);
    let mut memory = RiscvMemory::with_program_in_twist(xlen, TwistId(1), 0, &program_bytes);

    for (addr, val) in initial_memory {
        memory.store(neo_vm_trace::TwistId(0), addr, val);
    }

    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu, memory, shout, 1000).expect("execution failed");

    let final_regs = trace.steps.last().unwrap().regs_after.clone();

    let mut final_memory = RiscvMemory::new(xlen);
    for step in &trace.steps {
        for event in &step.twist_events {
            if matches!(event.kind, TwistOpKind::Write) {
                final_memory.store(event.twist_id, event.addr, event.value);
            }
        }
    }

    (final_regs, final_memory)
}
