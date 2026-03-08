//! Canonical real-ELF RV64IM note repro tests.
//!
//! These are the maintained end-to-end note-circuit prove/verify perf repros.
//! The older RV32 compiled-ROM note tests remain in-tree as legacy reference
//! paths, but new note-circuit validation and perf work should target this file.
#![cfg(feature = "poseidon-precompile")]

use neo_fold::pi_ccs::FoldingMode;
use neo_fold::rv64_trace_shard::{Rv64TraceWiring, Rv64TraceWiringRun};
use neo_math::F;
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_vm_trace::TwistOpKind;
use p3_field::PrimeCharacteristicRing;
use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

#[path = "support/note_deposit_fixture.rs"]
mod note_deposit_fixture;
#[path = "support/note_spend_fixture.rs"]
mod note_spend_fixture;
#[path = "support/rv64_guest.rs"]
mod rv64_guest;

fn simulate_exec_with_witness_rv64(
    elf: &[u8],
    ram_pairs: &[(u64, u32)],
    max_steps: usize,
) -> Result<Rv32ExecTable, String> {
    let mut wiring = Rv64TraceWiring::from_elf(elf)
        .map_err(|e| format!("from_elf failed: {e}"))?
        .mode(FoldingMode::Optimized)
        .max_steps(max_steps);
    for &(addr, val) in ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    let trace = wiring
        .simulate()
        .map_err(|e| format!("simulate failed: {e}"))?;
    if !trace.did_halt() {
        return Err(format!("RV64 note guest did not halt within {max_steps} steps"));
    }
    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), 64)
        .map_err(|e| format!("exec table build failed: {e}"))
}

fn derive_output_claims_for_addresses(
    exec: &Rv32ExecTable,
    ram_pairs: &[(u64, u32)],
    output_addrs: &[u64],
) -> Vec<(u64, u32)> {
    let output_addr_set: BTreeSet<u64> = output_addrs.iter().copied().collect();
    let mut final_output_values: HashMap<u64, u32> = output_addr_set.iter().map(|&addr| (addr, 0u32)).collect();
    for &(addr, value) in ram_pairs {
        if output_addr_set.contains(&addr) {
            final_output_values.insert(addr, value);
        }
    }
    for row in exec.rows.iter().filter(|r| r.active) {
        for ev in &row.ram_events {
            if ev.kind == TwistOpKind::Write && output_addr_set.contains(&ev.addr) {
                final_output_values.insert(ev.addr, ev.value as u32);
            }
        }
    }
    output_addrs
        .iter()
        .map(|addr| (*addr, *final_output_values.get(addr).unwrap_or(&0u32)))
        .collect()
}

fn parse_usize_after(msg: &str, key: &str) -> Option<usize> {
    let start = msg.find(key)? + key.len();
    let rest = &msg[start..];
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    rest[..digits_end].parse::<usize>().ok()
}

fn next_chunk_rows_from_poseidon_split(msg: &str, current_chunk_rows: usize) -> Option<usize> {
    if !msg.contains("poseidon split") {
        return None;
    }
    let ccs_m = parse_usize_after(msg, "ccs_m=")?;
    let m_in = parse_usize_after(msg, "m_in=")?;
    let t_len = parse_usize_after(msg, "t_len=")?;
    let t_cap = ccs_m.saturating_sub(m_in).max(1);
    if t_len <= t_cap {
        return None;
    }
    let mut next = current_chunk_rows
        .saturating_mul(t_cap)
        .checked_div(t_len)
        .unwrap_or(0)
        .max(1);
    if next >= current_chunk_rows {
        next = current_chunk_rows.saturating_sub(1).max(1);
    }
    (next < current_chunk_rows).then_some(next)
}

fn summarize_error_for_log(msg: &str) -> String {
    const MAX_CHARS: usize = 180;
    let compact = msg.replace('\n', " ");
    let mut out = compact.chars().take(MAX_CHARS).collect::<String>();
    if compact.chars().count() > MAX_CHARS {
        out.push_str("...");
    }
    out
}

#[derive(Debug, Clone)]
struct ProveAttemptMetrics {
    attempt: usize,
    chunk_rows: usize,
    prove_wall: Duration,
    succeeded: bool,
    next_chunk_rows: Option<usize>,
    error_summary: Option<String>,
}

struct ProveWithRetryResult {
    run: Rv64TraceWiringRun,
    final_chunk_rows: usize,
    attempts: Vec<ProveAttemptMetrics>,
}

struct NoteCaseMetrics {
    simulated_steps: usize,
    simulated_twist_events: usize,
    simulated_shout_events: usize,
    output_claim_words: usize,
    witness_ram_words: usize,
}

fn prove_with_poseidon_retry_rv64(
    elf: &[u8],
    ram_pairs: &[(u64, u32)],
    output_claims: &[(u64, u32)],
    executed_steps: usize,
    initial_chunk_rows: usize,
    max_retries: usize,
) -> Result<ProveWithRetryResult, String> {
    let mut chunk_rows = executed_steps.max(1);
    let configured_initial = initial_chunk_rows.max(1);
    if configured_initial < chunk_rows {
        chunk_rows = configured_initial;
    }
    let mut retry_idx = 0usize;
    let mut attempts = Vec::new();
    loop {
        let mut wiring = Rv64TraceWiring::from_elf(elf)
            .map_err(|e| format!("from_elf failed: {e}"))?
            .mode(FoldingMode::Optimized)
            .chunk_rows(chunk_rows)
            .max_steps(executed_steps);
        for &(addr, val) in ram_pairs {
            wiring = wiring.ram_init_u32(addr, val);
        }
        for &(addr, val) in output_claims {
            wiring = wiring.output_claim(addr, F::from_u64(val as u64));
        }

        let attempt_start = Instant::now();
        match wiring.prove() {
            Ok(run) => {
                attempts.push(ProveAttemptMetrics {
                    attempt: retry_idx + 1,
                    chunk_rows,
                    prove_wall: attempt_start.elapsed(),
                    succeeded: true,
                    next_chunk_rows: None,
                    error_summary: None,
                });
                return Ok(ProveWithRetryResult {
                    run,
                    final_chunk_rows: chunk_rows,
                    attempts,
                });
            }
            Err(err) => {
                let msg = err.to_string();
                let next_chunk_rows = if retry_idx < max_retries {
                    next_chunk_rows_from_poseidon_split(&msg, chunk_rows)
                } else {
                    None
                };
                attempts.push(ProveAttemptMetrics {
                    attempt: retry_idx + 1,
                    chunk_rows,
                    prove_wall: attempt_start.elapsed(),
                    succeeded: false,
                    next_chunk_rows,
                    error_summary: Some(summarize_error_for_log(&msg)),
                });
                if let Some(next_chunk_rows) = next_chunk_rows {
                    println!(
                        "rv64_poseidon_split_retry: chunk_rows {} -> {} (attempt {}/{})",
                        chunk_rows,
                        next_chunk_rows,
                        retry_idx + 1,
                        max_retries
                    );
                    chunk_rows = next_chunk_rows;
                    retry_idx += 1;
                    continue;
                }
                return Err(format!("prove failed at chunk_rows={chunk_rows}: {err}"));
            }
        }
    }
}

fn run_rv64_note_case(
    label: &str,
    elf: &[u8],
    witness_ram_pairs: &[(u64, u32)],
    output_layout_words: &[(u64, u32)],
    max_steps: usize,
) {
    let metrics = NoteCaseMetrics {
        witness_ram_words: witness_ram_pairs.len(),
        simulated_steps: 0,
        simulated_twist_events: 0,
        simulated_shout_events: 0,
        output_claim_words: output_layout_words.len(),
    };
    println!(
        "{label}: witness_ram_words={} output_claim_words={}",
        metrics.witness_ram_words, metrics.output_claim_words
    );

    let exec = simulate_exec_with_witness_rv64(elf, witness_ram_pairs, max_steps)
        .unwrap_or_else(|e| panic!("{label}: simulate witness failed: {e}"));
    let output_addrs: Vec<u64> = output_layout_words.iter().map(|(addr, _)| *addr).collect();
    let output_claims = derive_output_claims_for_addresses(&exec, witness_ram_pairs, &output_addrs);
    let metrics = NoteCaseMetrics {
        witness_ram_words: witness_ram_pairs.len(),
        simulated_steps: exec.rows.len(),
        simulated_twist_events: exec.rows.iter().map(|row| row.ram_events.len()).sum(),
        simulated_shout_events: exec.rows.iter().map(|row| row.shout_events.len()).sum(),
        output_claim_words: output_claims.len(),
    };
    println!(
        "{label}: simulated_steps={} simulated_twist_events={} simulated_shout_events={} output_claim_words={}",
        metrics.simulated_steps,
        metrics.simulated_twist_events,
        metrics.simulated_shout_events,
        metrics.output_claim_words
    );

    let prove_start = Instant::now();
    let prove_result = prove_with_poseidon_retry_rv64(
        elf,
        witness_ram_pairs,
        &output_claims,
        metrics.simulated_steps,
        metrics.simulated_steps,
        8,
    )
    .unwrap_or_else(|e| panic!("{label}: prove failed: {e}"));
    let prove_wall = prove_start.elapsed();
    let mut run = prove_result.run;
    let layout = run.layout().clone();
    let phases = run.prove_phase_durations();

    println!(
        "{label}: attempts={} final_chunk_rows={} prove_wall_ms={:.1}",
        prove_result.attempts.len(),
        prove_result.final_chunk_rows,
        prove_wall.as_secs_f64() * 1000.0
    );
    for attempt in &prove_result.attempts {
        println!(
            "{label}: attempt={} chunk_rows={} ok={} prove_wall_ms={:.1} next_chunk_rows={:?} err={}",
            attempt.attempt,
            attempt.chunk_rows,
            attempt.succeeded,
            attempt.prove_wall.as_secs_f64() * 1000.0,
            attempt.next_chunk_rows,
            attempt.error_summary.as_deref().unwrap_or("-")
        );
    }
    println!(
        "{label}: trace_len={} fold_count={} ccs_constraints={} ccs_variables={}",
        run.trace_len(),
        run.fold_count(),
        run.ccs_num_constraints(),
        run.ccs_num_variables()
    );
    println!(
        "{label}: layout_t={} layout_m_in={} layout_m={} used_memory_ids={:?} used_shout_table_ids={:?}",
        layout.t,
        layout.m_in,
        layout.m,
        run.used_memory_ids(),
        run.used_shout_table_ids()
    );
    println!(
        "{label}: setup_ms={:.1} chunk_build_commit_ms={:.1} fold_and_prove_ms={:.1}",
        phases.setup.as_secs_f64() * 1000.0,
        phases.chunk_build_commit.as_secs_f64() * 1000.0,
        phases.fold_and_prove.as_secs_f64() * 1000.0
    );

    let verify_start = Instant::now();
    run.verify()
        .unwrap_or_else(|e| panic!("{label}: verify failed: {e}"));
    let verify_wall = verify_start.elapsed();
    println!("{label}: verify_wall_ms={:.1}", verify_wall.as_secs_f64() * 1000.0);
}

#[test]
#[ignore = "slow RV64IM note-spend ELF perf repro"]
fn test_rv64_note_spend_from_elf_perf_repro() {
    let elf = rv64_guest::build_circuit_l2_transfer_rv64im_elf().expect("build RV64IM note guest ELF");
    let witness = note_spend_fixture::build_note_spend_fixture_witness();
    run_rv64_note_case(
        "rv64_note_spend_elf",
        &elf,
        &witness.ram_pairs,
        &witness.output_layout_words,
        400_000,
    );
}

#[test]
#[ignore = "slow RV64IM note-deposit ELF perf repro"]
fn test_rv64_note_deposit_from_elf_perf_repro() {
    let elf = rv64_guest::build_note_deposit_rv64im_elf().expect("build RV64IM note deposit guest ELF");
    let witness = note_deposit_fixture::build_note_deposit_witness();
    run_rv64_note_case(
        "rv64_note_deposit_elf",
        &elf,
        &witness.ram_pairs,
        &witness.output_layout_words,
        200_000,
    );
}
