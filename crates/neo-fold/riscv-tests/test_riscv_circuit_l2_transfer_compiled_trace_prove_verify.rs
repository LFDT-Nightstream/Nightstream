//! Compiled ROM coverage for the circuit_l2_transfer guest.
//!
//! Runtime prove/verify currently exercises the missing Poseidon2 precompile path.

#[path = "binaries/circuit_l2_transfer_rom.rs"]
mod circuit_l2_transfer_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvMemory, RiscvShoutTables, PROG_ID};
use neo_vm_trace::{trace_program, TwistOpKind};
use std::time::Instant;

fn parse_row_idx(err: &str) -> Option<usize> {
    let marker = "row=";
    let start = err.find(marker)? + marker.len();
    let tail = &err[start..];
    let end = tail.find(',').unwrap_or(tail.len());
    tail[..end].trim().parse::<usize>().ok()
}

fn dump_exec_row_context(program_base: u64, program_bytes: &[u8], max_steps: usize, center_row: usize) {
    let decoded_program = match decode_program(program_bytes) {
        Ok(p) => p,
        Err(e) => {
            println!("debug_trace_error=decode_program failed: {e}");
            return;
        }
    };
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(program_base, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(32, PROG_ID, program_base, program_bytes);
    let shout = RiscvShoutTables::new(32);
    let trace = match trace_program(cpu, twist, shout, max_steps) {
        Ok(t) => t,
        Err(e) => {
            println!("debug_trace_error=trace_program failed: {e}");
            return;
        }
    };
    let exec = match Rv32ExecTable::from_trace_padded(&trace, trace.steps.len()) {
        Ok(e) => e,
        Err(e) => {
            println!("debug_trace_error=Rv32ExecTable::from_trace_padded failed: {e}");
            return;
        }
    };

    let start = center_row.saturating_sub(2);
    let end = (center_row + 3).min(exec.rows.len());
    println!("debug_exec_rows_window=[{start}..{end}) total_rows={}", exec.rows.len());
    for i in start..end {
        let row = &exec.rows[i];
        let rs1 = row.reg_read_lane0.as_ref().map(|v| (v.addr, v.value));
        let rs2 = row.reg_read_lane1.as_ref().map(|v| (v.addr, v.value));
        let rdw = row.reg_write_lane0.as_ref().map(|v| (v.addr, v.value));
        println!(
            "debug_row idx={i} cycle={} pc_before={:#x} pc_after={:#x} instr_word={:#010x} decoded={:?}",
            row.cycle, row.pc_before, row.pc_after, row.instr_word, row.decoded
        );
        println!(
            "debug_row_io idx={i} active={} halted={} rs1={:?} rs2={:?} rd_write={:?}",
            row.active, row.halted, rs1, rs2, rdw
        );
        if row.ram_events.is_empty() {
            println!("debug_row_ram idx={i} ram_events=[]");
        } else {
            for (eidx, ev) in row.ram_events.iter().enumerate() {
                let kind = match ev.kind {
                    TwistOpKind::Read => "read",
                    TwistOpKind::Write => "write",
                };
                println!(
                    "debug_row_ram idx={i} ev={} kind={} mem_id={} addr={:#x} value={:#x} lane={:?}",
                    eidx, kind, ev.twist_id.0, ev.addr, ev.value, ev.lane
                );
            }
        }
        if row.shout_events.is_empty() {
            println!("debug_row_shout idx={i} shout_events=[]");
        } else {
            for (eidx, ev) in row.shout_events.iter().enumerate() {
                println!(
                    "debug_row_shout idx={i} ev={} shout_id={} key={:#x} value={:#x}",
                    eidx, ev.shout_id.0, ev.key, ev.value
                );
            }
        }
    }
}

#[test]
#[ignore = "slow full-trace prove/verify benchmark"]
fn test_riscv_circuit_l2_transfer_compiled_trace_prove_verify_with_metrics() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let static_instruction_words = program_bytes.len() / 4;

    let setup_wall_start = Instant::now();
    let decoded_program = decode_program(program_bytes).expect("decode circuit_l2_transfer ROM");
    let mut sim_cpu = RiscvCpu::new(32);
    sim_cpu.load_program(program_base, decoded_program);
    let sim_twist = RiscvMemory::with_program_in_twist(32, PROG_ID, program_base, program_bytes);
    let sim_shout = RiscvShoutTables::new(32);
    let sim_trace = trace_program(sim_cpu, sim_twist, sim_shout, static_instruction_words)
        .expect("trace circuit_l2_transfer ROM for pre-prove metrics");
    println!(
        "trace_sim_steps={} trace_sim_did_halt={} trace_sim_total_twist_events={} trace_sim_total_shout_events={}",
        sim_trace.len(),
        sim_trace.did_halt(),
        sim_trace.total_twist_events(),
        sim_trace.total_shout_events()
    );
    let executed_steps = sim_trace.len();
    let min_trace_len = executed_steps;
    let max_steps = executed_steps;
    // Poseidon local lane split requires t_len <= (ccs_m - m_in).
    // For this circuit/profile that cap is 510, so pick the largest safe chunk automatically.
    const MAX_SAFE_CHUNK_ROWS: usize = 510;
    let chunk_rows = executed_steps.min(MAX_SAFE_CHUNK_ROWS);

    let wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(min_trace_len)
        .chunk_rows(chunk_rows)
        .max_steps(max_steps)
        .shout_auto_minimal();
    let setup_wall = setup_wall_start.elapsed();

    let prove_wall_start = Instant::now();
    println!("prove_start");
    let prove_result = wiring.prove();
    let prove_wall = prove_wall_start.elapsed();

    println!("==== circuit_l2_transfer metrics ====");
    println!(
        "program_base={} rom_bytes={} static_instruction_words={} min_trace_len={} chunk_rows={} max_steps={}",
        program_base,
        program_bytes.len(),
        static_instruction_words,
        min_trace_len,
        chunk_rows,
        max_steps
    );
    println!("setup_time_wall={:?}", setup_wall);
    println!("prove_wall_time={:?}", prove_wall);

    let mut run = match prove_result {
        Ok(run) => run,
        Err(err) => {
            let err_s = err.to_string();
            println!("trace_instructions_active_rows=N/A (prove failed)");
            println!("fold_steps=N/A (prove failed)");
            println!("ccs_constraints=N/A (prove failed)");
            println!("ccs_variables=N/A (prove failed)");
            println!("layout_t=N/A (prove failed)");
            println!("layout_m_in=N/A (prove failed)");
            println!("layout_m=N/A (prove failed)");
            println!("used_memory_ids=N/A (prove failed)");
            println!("used_shout_table_ids=N/A (prove failed)");
            println!("setup_time=N/A (prove failed)");
            println!("chunk_build_commit_time=N/A (prove failed)");
            println!("fold_and_prove_time=N/A (prove failed)");
            println!("prove_time_total=N/A (prove failed)");
            println!("verify_time=N/A (prove failed)");
            println!("verify_wall_time=N/A (prove failed)");
            println!("prove_error={err_s}");
            if let Some(row_idx) = parse_row_idx(&err_s) {
                println!("debug_failure_row_idx={row_idx}");
                dump_exec_row_context(program_base, program_bytes, 1 << 20, row_idx);
            } else {
                println!("debug_failure_row_idx=unavailable");
            }
            println!("=====================================");
            panic!("prove circuit_l2_transfer failed: {err}");
        }
    };

    let phase = run.prove_phase_durations();
    let layout = run.layout();
    println!("trace_instructions_active_rows={}", run.trace_len());
    println!("fold_steps={}", run.fold_count());
    println!("trace_hit_max_steps_cap={}", run.trace_len() == max_steps);
    println!(
        "ccs_constraints={} ccs_variables={}",
        run.ccs_num_constraints(),
        run.ccs_num_variables()
    );
    println!(
        "layout_t={} layout_m_in={} layout_m={}",
        layout.t, layout.m_in, layout.m
    );
    println!("used_memory_ids={:?}", run.used_memory_ids());
    println!("used_shout_table_ids={:?}", run.used_shout_table_ids());
    println!("setup_time={:?}", phase.setup);
    println!("chunk_build_commit_time={:?}", phase.chunk_build_commit);
    println!("fold_and_prove_time={:?}", phase.fold_and_prove);
    println!("prove_time_total={:?}", run.prove_duration());

    let verify_wall_start = Instant::now();
    println!("verify_start");
    let verify_result = run.verify();
    let verify_wall = verify_wall_start.elapsed();
    println!("verify_time={:?}", run.verify_duration().unwrap_or(verify_wall));
    println!("verify_wall_time={:?}", verify_wall);
    if let Err(err) = verify_result {
        println!("verify_error={err}");
        println!("=====================================");
        panic!("verify circuit_l2_transfer failed: {err}");
    }
    println!("=====================================");
}
