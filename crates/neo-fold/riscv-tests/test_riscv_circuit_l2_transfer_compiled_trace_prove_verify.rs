//! Compiled ROM coverage for the circuit_l2_transfer guest.
//!
//! Contains a realistic note-spend transfer test that constructs a valid
//! 1-input / 1-output self-transfer witness (including Merkle path, nullifier,
//! enforce-product, and blacklist non-membership proof), writes it to RAM, then
//! proves + verifies and emits detailed metrics.
#![cfg(feature = "poseidon-precompile")]

#[path = "binaries/circuit_l2_transfer_rom.rs"]
mod circuit_l2_transfer_rom;
#[path = "binaries/sovereign_note_deposit_rom.rs"]
mod sovereign_note_deposit_rom;
#[path = "binaries/sovereign_note_spend_rom.rs"]
mod sovereign_note_spend_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
#[cfg(feature = "protocol-metrics")]
use neo_fold::shard::{MemOrLutProof, ShardProof, StepProof};
use neo_math::F;
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvMemory, RiscvShoutTables, PROG_ID, RAM_ID};
use neo_vm_trace::{trace_program, Twist, TwistOpKind};
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

#[cfg(feature = "protocol-metrics")]
#[derive(Clone, Copy, Debug, Default)]
struct RouteAClaimBreakdown {
    shout: usize,
    twist: usize,
    wb_wp: usize,
    decode: usize,
    width: usize,
    control: usize,
    poseidon: usize,
    output: usize,
    other: usize,
}

#[cfg(feature = "protocol-metrics")]
impl RouteAClaimBreakdown {
    fn add_label(&mut self, label: &[u8]) {
        if label.starts_with(b"shout/") {
            self.shout += 1;
        } else if label.starts_with(b"twist/") {
            self.twist += 1;
        } else if label.starts_with(b"wb/") || label.starts_with(b"wp/") {
            self.wb_wp += 1;
        } else if label.starts_with(b"decode/") {
            self.decode += 1;
        } else if label.starts_with(b"width/") {
            self.width += 1;
        } else if label.starts_with(b"control/") {
            self.control += 1;
        } else if label.starts_with(b"poseidon/") {
            self.poseidon += 1;
        } else if label.starts_with(b"output/") {
            self.output += 1;
        } else {
            self.other += 1;
        }
    }
}

#[cfg(feature = "poseidon-precompile")]
fn simulate_exec_with_witness(
    program_base: u64,
    program_bytes: &[u8],
    ram_pairs: &[(u64, u32)],
    max_steps: usize,
) -> Result<Rv32ExecTable, String> {
    let decoded = decode_program(program_bytes).map_err(|e| format!("decode ROM failed: {e}"))?;
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(program_base, decoded);
    let mut twist = RiscvMemory::with_program_in_twist(32, PROG_ID, program_base, program_bytes);
    for &(addr, val) in ram_pairs {
        twist.store(RAM_ID, addr, val as u64);
    }
    let shout = RiscvShoutTables::new(32);
    let sim = trace_program(cpu, twist, shout, max_steps).map_err(|e| format!("simulation trace failed: {e}"))?;
    if !sim.did_halt() {
        return Err(format!("circuit did not halt within {max_steps} simulation steps"));
    }
    Rv32ExecTable::from_trace_padded(&sim, sim.steps.len()).map_err(|e| format!("exec table build failed: {e}"))
}

#[cfg(feature = "poseidon-precompile")]
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

#[cfg(feature = "poseidon-precompile")]
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

#[cfg(feature = "poseidon-precompile")]
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

#[cfg(feature = "poseidon-precompile")]
fn prove_with_poseidon_retry(
    program_base: u64,
    program_bytes: &[u8],
    ram_pairs: &[(u64, u32)],
    output_claims: &[(u64, u32)],
    executed_steps: usize,
    initial_chunk_rows: usize,
    max_retries: usize,
) -> Result<(neo_fold::riscv_trace_shard::Rv32TraceWiringRun, usize), String> {
    let mut chunk_rows = initial_chunk_rows.max(1);
    let mut retry_idx = 0usize;
    loop {
        let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
            .xlen(32)
            .min_trace_len(executed_steps)
            .chunk_rows(chunk_rows)
            .max_steps(executed_steps)
            .shout_auto_minimal();
        for &(addr, val) in ram_pairs {
            wiring = wiring.ram_init_u32(addr, val);
        }
        for &(addr, val) in output_claims {
            wiring = wiring.output_claim(addr, F::from_u64(val as u64));
        }

        match wiring.prove() {
            Ok(run) => return Ok((run, chunk_rows)),
            Err(err) => {
                let msg = err.to_string();
                if retry_idx < max_retries {
                    if let Some(next_chunk_rows) = next_chunk_rows_from_poseidon_split(&msg, chunk_rows) {
                        println!(
                            "poseidon_split_retry: chunk_rows {} -> {} (attempt {}/{})",
                            chunk_rows,
                            next_chunk_rows,
                            retry_idx + 1,
                            max_retries
                        );
                        chunk_rows = next_chunk_rows;
                        retry_idx += 1;
                        continue;
                    }
                }
                return Err(format!("prove failed at chunk_rows={chunk_rows}: {err}"));
            }
        }
    }
}

#[cfg(feature = "protocol-metrics")]
#[derive(Debug)]
struct StepProtocolMetrics {
    step_id: String,
    compressed_substeps: usize,
    ccs_variant: &'static str,
    ccs_fe_rounds: usize,
    ccs_fe_coeffs: usize,
    ccs_nc_rounds: usize,
    ccs_nc_coeffs: usize,
    cpu_sumcheck_rounds: usize,
    cpu_sumcheck_coeffs: usize,
    shift_sumcheck_rounds: usize,
    shift_sumcheck_coeffs: usize,
    route_a_claims: usize,
    route_a_rounds: usize,
    route_a_coeffs: usize,
    route_a_degree_max: usize,
    route_a_breakdown: RouteAClaimBreakdown,
    poseidon_local_claims: usize,
    poseidon_local_rounds: usize,
    poseidon_local_coeffs: usize,
    ccs_out_claims: usize,
    rlc_rhos: usize,
    dec_children: usize,
    val_me_claims: usize,
    wb_me_claims: usize,
    wp_me_claims: usize,
    poseidon_cycle_me_claims: usize,
    poseidon_local_me_claims: usize,
    time_cpu_commitments: usize,
    time_mem_commitments: usize,
    stage8_joint_commitments: usize,
    shout_proofs: usize,
    twist_proofs: usize,
    opening_points: usize,
    opening_cols: usize,
    opening_bound_proofs: usize,
    opening_bound_cols: usize,
    opening_reduction_groups: usize,
    opening_unification_rounds: usize,
    opening_unification_coeffs: usize,
    opening_joint_groups: usize,
    opening_joint_unified: usize,
    val_fold_lanes: usize,
    wb_fold_lanes: usize,
    wp_fold_lanes: usize,
    poseidon_cycle_fold_lanes: usize,
    poseidon_local_fold_lanes: usize,
    stage8_fold_lanes: usize,
    fold_lane_dec_children: usize,
    time_t: usize,
}

#[cfg(feature = "protocol-metrics")]
#[derive(Debug, Default)]
struct ProtocolFlowTotals {
    route_a_breakdown: RouteAClaimBreakdown,
    ccs_fe_rounds: usize,
    ccs_fe_coeffs: usize,
    ccs_nc_rounds: usize,
    ccs_nc_coeffs: usize,
    cpu_sumcheck_rounds: usize,
    cpu_sumcheck_coeffs: usize,
    shift_sumcheck_rounds: usize,
    shift_sumcheck_coeffs: usize,
    route_a_claims: usize,
    route_a_rounds: usize,
    route_a_coeffs: usize,
    route_a_degree_max: usize,
    poseidon_local_claims: usize,
    poseidon_local_rounds: usize,
    poseidon_local_coeffs: usize,
    ccs_out_claims: usize,
    rlc_rhos: usize,
    dec_children: usize,
    val_me_claims: usize,
    wb_me_claims: usize,
    wp_me_claims: usize,
    poseidon_cycle_me_claims: usize,
    poseidon_local_me_claims: usize,
    time_cpu_commitments: usize,
    time_mem_commitments: usize,
    stage8_joint_commitments: usize,
    shout_proofs: usize,
    twist_proofs: usize,
    opening_points: usize,
    opening_cols: usize,
    opening_bound_proofs: usize,
    opening_bound_cols: usize,
    opening_reduction_groups: usize,
    opening_unification_rounds: usize,
    opening_unification_coeffs: usize,
    opening_joint_groups: usize,
    opening_joint_unified: usize,
    val_fold_lanes: usize,
    wb_fold_lanes: usize,
    wp_fold_lanes: usize,
    poseidon_cycle_fold_lanes: usize,
    poseidon_local_fold_lanes: usize,
    stage8_fold_lanes: usize,
    fold_lane_dec_children: usize,
    max_time_t: usize,
}

#[cfg(feature = "protocol-metrics")]
fn coeffs_2d<T>(rounds: &[Vec<T>]) -> usize {
    rounds.iter().map(Vec::len).sum()
}

#[cfg(feature = "protocol-metrics")]
fn coeffs_3d<T>(claims: &[Vec<Vec<T>>]) -> usize {
    claims.iter().map(|claim| coeffs_2d(claim)).sum()
}

#[cfg(feature = "protocol-metrics")]
fn fold_dec_children_total(step: &StepProof) -> usize {
    let mut total = step.fold.dec_children.len();
    for lane in &step.val_fold {
        total += lane.dec_children.len();
    }
    for lane in &step.wb_fold {
        total += lane.dec_children.len();
    }
    for lane in &step.wp_fold {
        total += lane.dec_children.len();
    }
    for lane in &step.poseidon_cycle_fold {
        total += lane.dec_children.len();
    }
    for lane in &step.poseidon_local_fold {
        total += lane.dec_children.len();
    }
    for lane in &step.stage8_fold {
        total += lane.dec_children.len();
    }
    total
}

#[cfg(feature = "protocol-metrics")]
fn build_step_protocol_metrics(step_id: String, step: &StepProof) -> StepProtocolMetrics {
    let mut route_a_breakdown = RouteAClaimBreakdown::default();
    for label in &step.batched_time.labels {
        route_a_breakdown.add_label(label);
    }

    let mut shout_proofs = 0usize;
    let mut twist_proofs = 0usize;
    for proof in &step.mem.proofs {
        match proof {
            MemOrLutProof::Shout(_) => shout_proofs += 1,
            MemOrLutProof::Twist(_) => twist_proofs += 1,
        }
    }

    let stage8_joint_commitments =
        step.fold.joint_opening_lane.groups.len() + usize::from(step.fold.joint_opening_lane.unified_fold.is_some());
    let opening_cols = step
        .fold
        .openings
        .iter()
        .map(|entry| entry.col_ids.len())
        .sum();
    let opening_bound_cols = step
        .fold
        .opening_proofs
        .iter()
        .map(|entry| entry.col_ids.len())
        .sum();
    let route_a_rounds = step.batched_time.round_polys.iter().map(Vec::len).sum();
    let route_a_degree_max = step
        .batched_time
        .degree_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let poseidon_local_claims = step
        .poseidon_local_time
        .as_ref()
        .map(|proof| proof.labels.len())
        .unwrap_or(0);
    let poseidon_local_rounds = step
        .poseidon_local_time
        .as_ref()
        .map(|proof| proof.round_polys.iter().map(Vec::len).sum())
        .unwrap_or(0);
    let poseidon_local_coeffs = step
        .poseidon_local_time
        .as_ref()
        .map(|proof| coeffs_3d(&proof.round_polys))
        .unwrap_or(0);

    StepProtocolMetrics {
        step_id,
        compressed_substeps: step.compressed_substeps.as_ref().map_or(0, Vec::len),
        ccs_variant: match step.fold.ccs_proof.variant {
            neo_fold::optimized_engine::PiCcsProofVariant::SplitNcV1 => "SplitNcV1",
        },
        ccs_fe_rounds: step.fold.ccs_proof.sumcheck_rounds.len(),
        ccs_fe_coeffs: coeffs_2d(&step.fold.ccs_proof.sumcheck_rounds),
        ccs_nc_rounds: step.fold.ccs_proof.sumcheck_rounds_nc.len(),
        ccs_nc_coeffs: coeffs_2d(&step.fold.ccs_proof.sumcheck_rounds_nc),
        cpu_sumcheck_rounds: step.fold.cpu_sumcheck.round_polys.len(),
        cpu_sumcheck_coeffs: coeffs_2d(&step.fold.cpu_sumcheck.round_polys),
        shift_sumcheck_rounds: step.fold.shift_sumcheck.round_polys.len(),
        shift_sumcheck_coeffs: coeffs_2d(&step.fold.shift_sumcheck.round_polys),
        route_a_claims: step.batched_time.labels.len(),
        route_a_rounds,
        route_a_coeffs: coeffs_3d(&step.batched_time.round_polys),
        route_a_degree_max,
        route_a_breakdown,
        poseidon_local_claims,
        poseidon_local_rounds,
        poseidon_local_coeffs,
        ccs_out_claims: step.fold.ccs_out.len(),
        rlc_rhos: step.fold.rlc_rhos.len(),
        dec_children: step.fold.dec_children.len(),
        val_me_claims: step.mem.val_me_claims.len(),
        wb_me_claims: step.mem.wb_me_claims.len(),
        wp_me_claims: step.mem.wp_me_claims.len(),
        poseidon_cycle_me_claims: step.mem.poseidon_cycle_me_claims.len(),
        poseidon_local_me_claims: step.mem.poseidon_local_me_claims.len(),
        time_cpu_commitments: step.fold.time_cpu_commitments.len(),
        time_mem_commitments: step.fold.time_mem_commitments.len(),
        stage8_joint_commitments,
        shout_proofs,
        twist_proofs,
        opening_points: step.fold.openings.len(),
        opening_cols,
        opening_bound_proofs: step.fold.opening_proofs.len(),
        opening_bound_cols,
        opening_reduction_groups: step.fold.opening_reduction.groups.len(),
        opening_unification_rounds: step.fold.opening_unification.round_polys.len(),
        opening_unification_coeffs: coeffs_2d(&step.fold.opening_unification.round_polys),
        opening_joint_groups: step.fold.joint_opening_lane.groups.len(),
        opening_joint_unified: usize::from(step.fold.joint_opening_lane.unified_fold.is_some()),
        val_fold_lanes: step.val_fold.len(),
        wb_fold_lanes: step.wb_fold.len(),
        wp_fold_lanes: step.wp_fold.len(),
        poseidon_cycle_fold_lanes: step.poseidon_cycle_fold.len(),
        poseidon_local_fold_lanes: step.poseidon_local_fold.len(),
        stage8_fold_lanes: step.stage8_fold.len(),
        fold_lane_dec_children: fold_dec_children_total(step),
        time_t: step.fold.time_t,
    }
}

#[cfg(feature = "protocol-metrics")]
fn collect_step_protocol_metrics(step_id: String, step: &StepProof, out: &mut Vec<StepProtocolMetrics>) {
    out.push(build_step_protocol_metrics(step_id.clone(), step));
    if let Some(substeps) = &step.compressed_substeps {
        for (idx, inner) in substeps.iter().enumerate() {
            collect_step_protocol_metrics(format!("{step_id}.{idx}"), inner, out);
        }
    }
}

#[cfg(feature = "protocol-metrics")]
fn accumulate_totals(rows: &[StepProtocolMetrics]) -> ProtocolFlowTotals {
    let mut totals = ProtocolFlowTotals::default();
    for row in rows {
        totals.route_a_breakdown.shout += row.route_a_breakdown.shout;
        totals.route_a_breakdown.twist += row.route_a_breakdown.twist;
        totals.route_a_breakdown.wb_wp += row.route_a_breakdown.wb_wp;
        totals.route_a_breakdown.decode += row.route_a_breakdown.decode;
        totals.route_a_breakdown.width += row.route_a_breakdown.width;
        totals.route_a_breakdown.control += row.route_a_breakdown.control;
        totals.route_a_breakdown.poseidon += row.route_a_breakdown.poseidon;
        totals.route_a_breakdown.output += row.route_a_breakdown.output;
        totals.route_a_breakdown.other += row.route_a_breakdown.other;
        totals.ccs_fe_rounds += row.ccs_fe_rounds;
        totals.ccs_fe_coeffs += row.ccs_fe_coeffs;
        totals.ccs_nc_rounds += row.ccs_nc_rounds;
        totals.ccs_nc_coeffs += row.ccs_nc_coeffs;
        totals.cpu_sumcheck_rounds += row.cpu_sumcheck_rounds;
        totals.cpu_sumcheck_coeffs += row.cpu_sumcheck_coeffs;
        totals.shift_sumcheck_rounds += row.shift_sumcheck_rounds;
        totals.shift_sumcheck_coeffs += row.shift_sumcheck_coeffs;
        totals.route_a_claims += row.route_a_claims;
        totals.route_a_rounds += row.route_a_rounds;
        totals.route_a_coeffs += row.route_a_coeffs;
        totals.route_a_degree_max = totals.route_a_degree_max.max(row.route_a_degree_max);
        totals.poseidon_local_claims += row.poseidon_local_claims;
        totals.poseidon_local_rounds += row.poseidon_local_rounds;
        totals.poseidon_local_coeffs += row.poseidon_local_coeffs;
        totals.ccs_out_claims += row.ccs_out_claims;
        totals.rlc_rhos += row.rlc_rhos;
        totals.dec_children += row.dec_children;
        totals.val_me_claims += row.val_me_claims;
        totals.wb_me_claims += row.wb_me_claims;
        totals.wp_me_claims += row.wp_me_claims;
        totals.poseidon_cycle_me_claims += row.poseidon_cycle_me_claims;
        totals.poseidon_local_me_claims += row.poseidon_local_me_claims;
        totals.time_cpu_commitments += row.time_cpu_commitments;
        totals.time_mem_commitments += row.time_mem_commitments;
        totals.stage8_joint_commitments += row.stage8_joint_commitments;
        totals.shout_proofs += row.shout_proofs;
        totals.twist_proofs += row.twist_proofs;
        totals.opening_points += row.opening_points;
        totals.opening_cols += row.opening_cols;
        totals.opening_bound_proofs += row.opening_bound_proofs;
        totals.opening_bound_cols += row.opening_bound_cols;
        totals.opening_reduction_groups += row.opening_reduction_groups;
        totals.opening_unification_rounds += row.opening_unification_rounds;
        totals.opening_unification_coeffs += row.opening_unification_coeffs;
        totals.opening_joint_groups += row.opening_joint_groups;
        totals.opening_joint_unified += row.opening_joint_unified;
        totals.val_fold_lanes += row.val_fold_lanes;
        totals.wb_fold_lanes += row.wb_fold_lanes;
        totals.wp_fold_lanes += row.wp_fold_lanes;
        totals.poseidon_cycle_fold_lanes += row.poseidon_cycle_fold_lanes;
        totals.poseidon_local_fold_lanes += row.poseidon_local_fold_lanes;
        totals.stage8_fold_lanes += row.stage8_fold_lanes;
        totals.fold_lane_dec_children += row.fold_lane_dec_children;
        totals.max_time_t = totals.max_time_t.max(row.time_t);
    }
    totals
}

#[cfg(feature = "protocol-metrics")]
fn percent_of(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (total as f64)
    }
}

#[cfg(feature = "protocol-metrics")]
fn env_flag_true(var_name: &str) -> bool {
    match std::env::var(var_name) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

#[cfg(feature = "protocol-metrics")]
fn print_superneo_protocol_metrics(
    proof: &ShardProof,
    trace_len: usize,
    fold_count: usize,
    setup_duration: std::time::Duration,
    chunk_commit_duration: std::time::Duration,
    fold_and_prove_duration: std::time::Duration,
    prove_wall: std::time::Duration,
    verify_wall: std::time::Duration,
    verify_run: Option<std::time::Duration>,
) {
    let mut rows = Vec::new();
    for (idx, step) in proof.steps.iter().enumerate() {
        collect_step_protocol_metrics(idx.to_string(), step, &mut rows);
    }
    let totals = accumulate_totals(&rows);
    let public_steps = proof
        .segment_meta
        .as_ref()
        .map(|meta| meta.iter().map(|entry| entry.public_steps).sum())
        .unwrap_or(rows.len());
    let proof_steps = proof.steps.len();
    let compressed_containers = rows
        .iter()
        .filter(|row| row.compressed_substeps > 0)
        .count();
    let instructions = trace_len.max(1) as f64;
    let public_steps_safe = public_steps.max(1) as f64;
    let fold_steps_safe = fold_count.max(1) as f64;
    let verify_run_duration = verify_run.unwrap_or(verify_wall);
    let verbose = env_flag_true("NS_PROTOCOL_METRICS_VERBOSE");
    let route_claim_total = totals.route_a_claims.max(1);
    let pi_ccs_coeffs = totals.ccs_fe_coeffs + totals.ccs_nc_coeffs;
    let route_a_coeffs = totals.route_a_coeffs + totals.cpu_sumcheck_coeffs + totals.shift_sumcheck_coeffs;
    let stage8_coeffs = totals.opening_unification_coeffs;
    let poseidon_local_coeffs = totals.poseidon_local_coeffs;
    let coeff_proxy_total = (pi_ccs_coeffs + route_a_coeffs + stage8_coeffs + poseidon_local_coeffs).max(1);

    if !env_flag_true("NS_PROTOCOL_METRICS_LEGACY") {
        println!("  +-------------------------------------------------------------------------+");
        println!("  | Protocol Metrics (SuperNEO Flow)                                        |");
        println!("  +-------------------------------------------------------------------------+");
        println!("  | Feature     | enabled (neo-fold/protocol-metrics)                       |");
        println!("  | Reference   | SuperNEO Section 7 + Route-A/Stage-8                      |");
        println!(
            "  | Shape       | public={} proof={} expanded={} compressed={} folds={}           |",
            public_steps,
            proof_steps,
            rows.len(),
            compressed_containers,
            fold_count
        );
        println!(
            "  | CCS variant | {:<58} |",
            rows.first().map(|row| row.ccs_variant).unwrap_or("n/a")
        );
        println!("  | Max time_t  | {:<58} |", totals.max_time_t);
        println!("  | Out binding | {:<58} |", proof.output_proof.is_some());
        println!("  +-------------------------------------------------------------------------+");

        println!("  +-----------------------------+--------+--------+--------+----------+");
        println!("  | Stage                       | Claims | Rounds | Coeffs | Extra    |");
        println!("  +-----------------------------+--------+--------+--------+----------+");
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "pi_ccs_fe", "-", totals.ccs_fe_rounds, totals.ccs_fe_coeffs, "-"
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "pi_ccs_nc", "-", totals.ccs_nc_rounds, totals.ccs_nc_coeffs, "-"
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "route_a_batched_time",
            totals.route_a_claims,
            totals.route_a_rounds,
            totals.route_a_coeffs,
            totals.route_a_degree_max
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "route_a_cpu_shift",
            "-",
            totals.cpu_sumcheck_rounds + totals.shift_sumcheck_rounds,
            totals.cpu_sumcheck_coeffs + totals.shift_sumcheck_coeffs,
            "-"
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "poseidon_local_time",
            totals.poseidon_local_claims,
            totals.poseidon_local_rounds,
            totals.poseidon_local_coeffs,
            "-"
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "stage8_unification",
            "-",
            totals.opening_unification_rounds,
            totals.opening_unification_coeffs,
            totals.opening_reduction_groups
        );
        println!(
            "  | {:<27} | {:>6} | {:>6} | {:>6} | {:>8} |",
            "pi_rlc_dec_lanes", "-", "-", "-", totals.rlc_rhos
        );
        println!("  +-----------------------------+--------+--------+--------+----------+");

        println!("  +----------------+--------+--------+");
        println!("  | Route-A family | Claims | Share% |");
        println!("  +----------------+--------+--------+");
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "shout",
            totals.route_a_breakdown.shout,
            percent_of(totals.route_a_breakdown.shout, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "twist",
            totals.route_a_breakdown.twist,
            percent_of(totals.route_a_breakdown.twist, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "wb/wp",
            totals.route_a_breakdown.wb_wp,
            percent_of(totals.route_a_breakdown.wb_wp, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "decode",
            totals.route_a_breakdown.decode,
            percent_of(totals.route_a_breakdown.decode, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "width",
            totals.route_a_breakdown.width,
            percent_of(totals.route_a_breakdown.width, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "control",
            totals.route_a_breakdown.control,
            percent_of(totals.route_a_breakdown.control, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "poseidon",
            totals.route_a_breakdown.poseidon,
            percent_of(totals.route_a_breakdown.poseidon, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "output",
            totals.route_a_breakdown.output,
            percent_of(totals.route_a_breakdown.output, route_claim_total)
        );
        println!(
            "  | {:<14} | {:>6} | {:>6.1} |",
            "other",
            totals.route_a_breakdown.other,
            percent_of(totals.route_a_breakdown.other, route_claim_total)
        );
        println!("  +----------------+--------+--------+");

        println!(
            "  lane_claims: ccs_out={} dec_children={} fold_dec_children_total={}",
            totals.ccs_out_claims, totals.dec_children, totals.fold_lane_dec_children
        );
        println!(
            "  me_claims: val={} wb={} wp={} poseidon_cycle={} poseidon_local={}",
            totals.val_me_claims,
            totals.wb_me_claims,
            totals.wp_me_claims,
            totals.poseidon_cycle_me_claims,
            totals.poseidon_local_me_claims
        );
        println!(
            "  commitments/openings: cpu={} mem={} stage8_joint={} open_points={} open_cols={} open_proofs={}",
            totals.time_cpu_commitments,
            totals.time_mem_commitments,
            totals.stage8_joint_commitments,
            totals.opening_points,
            totals.opening_cols,
            totals.opening_bound_proofs
        );
        println!(
            "  folds: val={} wb={} wp={} poseidon_cycle={} poseidon_local={} stage8={}",
            totals.val_fold_lanes,
            totals.wb_fold_lanes,
            totals.wp_fold_lanes,
            totals.poseidon_cycle_fold_lanes,
            totals.poseidon_local_fold_lanes,
            totals.stage8_fold_lanes
        );
        println!(
            "  proof_payloads: shout={} twist={}",
            totals.shout_proofs, totals.twist_proofs
        );

        println!("  +-------------------------+--------------+");
        println!("  | Timing metric           | Value (ms)   |");
        println!("  +-------------------------+--------------+");
        println!(
            "  | {:<23} | {:>12.1} |",
            "setup",
            setup_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.1} |",
            "chunk+commit",
            chunk_commit_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.1} |",
            "fold+prove",
            fold_and_prove_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.1} |",
            "prove_wall",
            prove_wall.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.1} |",
            "verify_run",
            verify_run_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.1} |",
            "verify_wall",
            verify_wall.as_secs_f64() * 1000.0
        );
        println!(
            "  | {:<23} | {:>12.3} |",
            "prove per instruction",
            prove_wall.as_secs_f64() * 1000.0 / instructions
        );
        println!(
            "  | {:<23} | {:>12.3} |",
            "fold+prove per instr",
            fold_and_prove_duration.as_secs_f64() * 1000.0 / instructions
        );
        println!(
            "  | {:<23} | {:>12.3} |",
            "verify per instruction",
            verify_wall.as_secs_f64() * 1000.0 / instructions
        );
        println!(
            "  | {:<23} | {:>12.3} |",
            "fold+prove per public",
            fold_and_prove_duration.as_secs_f64() * 1000.0 / public_steps_safe
        );
        println!(
            "  | {:<23} | {:>12.3} |",
            "fold+prove per fold",
            fold_and_prove_duration.as_secs_f64() * 1000.0 / fold_steps_safe
        );
        println!("  +-------------------------+--------------+");

        println!("  +-------------------------+------------+--------+");
        println!("  | Coeff proxy component   | Count      | Share% |");
        println!("  +-------------------------+------------+--------+");
        println!(
            "  | {:<23} | {:>10} | {:>6.1} |",
            "route_a",
            route_a_coeffs,
            percent_of(route_a_coeffs, coeff_proxy_total)
        );
        println!(
            "  | {:<23} | {:>10} | {:>6.1} |",
            "pi_ccs",
            pi_ccs_coeffs,
            percent_of(pi_ccs_coeffs, coeff_proxy_total)
        );
        println!(
            "  | {:<23} | {:>10} | {:>6.1} |",
            "poseidon_local",
            poseidon_local_coeffs,
            percent_of(poseidon_local_coeffs, coeff_proxy_total)
        );
        println!(
            "  | {:<23} | {:>10} | {:>6.1} |",
            "stage8_unification",
            stage8_coeffs,
            percent_of(stage8_coeffs, coeff_proxy_total)
        );
        println!("  +-------------------------+------------+--------+");

        if verbose {
            println!("  per_step_flow (NS_PROTOCOL_METRICS_VERBOSE=1):");
            for row in rows {
                println!(
                    "    step={} substeps={} ccs_rounds(fe/nc={} / {}) route_a(claims/rounds={} / {}) rlc_rhos={} dec_children={} commits(cpu/mem/joint={} / {} / {}) folds(val/wb/wp/p2c/p2l/s8={}/{}/{}/{}/{}/{})",
                    row.step_id,
                    row.compressed_substeps,
                    row.ccs_fe_rounds,
                    row.ccs_nc_rounds,
                    row.route_a_claims,
                    row.route_a_rounds,
                    row.rlc_rhos,
                    row.dec_children,
                    row.time_cpu_commitments,
                    row.time_mem_commitments,
                    row.stage8_joint_commitments,
                    row.val_fold_lanes,
                    row.wb_fold_lanes,
                    row.wp_fold_lanes,
                    row.poseidon_cycle_fold_lanes,
                    row.poseidon_local_fold_lanes,
                    row.stage8_fold_lanes
                );
            }
        } else {
            println!("  per_step_flow=hidden (set NS_PROTOCOL_METRICS_VERBOSE=1 for details)");
        }
        return;
    }

    println!("  protocol_metrics_feature=enabled (neo-fold/protocol-metrics)");
    println!("  protocol_flow_reference=SuperNEO Section 7 + Route-A/Stage-8");
    println!(
        "  protocol_shape: public_steps={} proof_steps={} expanded_steps={} compressed_containers={} fold_steps={}",
        public_steps,
        proof_steps,
        rows.len(),
        compressed_containers,
        fold_count
    );
    println!(
        "  ccs_variant={} max_time_t={} output_binding_present={}",
        rows.first().map(|row| row.ccs_variant).unwrap_or("n/a"),
        totals.max_time_t,
        proof.output_proof.is_some()
    );

    println!("  stage_summary:");
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "stage", "claims", "rounds", "coeffs", "extra"
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "pi_ccs_fe", "-", totals.ccs_fe_rounds, totals.ccs_fe_coeffs, "-"
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "pi_ccs_nc", "-", totals.ccs_nc_rounds, totals.ccs_nc_coeffs, "-"
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "route_a_batched_time",
        totals.route_a_claims,
        totals.route_a_rounds,
        totals.route_a_coeffs,
        totals.route_a_degree_max
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "route_a_cpu_shift",
        "-",
        totals.cpu_sumcheck_rounds + totals.shift_sumcheck_rounds,
        totals.cpu_sumcheck_coeffs + totals.shift_sumcheck_coeffs,
        "-"
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "poseidon_local_time",
        totals.poseidon_local_claims,
        totals.poseidon_local_rounds,
        totals.poseidon_local_coeffs,
        "-"
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "stage8_unification",
        "-",
        totals.opening_unification_rounds,
        totals.opening_unification_coeffs,
        totals.opening_reduction_groups
    );
    println!(
        "    {:<28} {:>8} {:>8} {:>8} {:>10}",
        "pi_rlc_dec_lanes", "-", "-", "-", totals.rlc_rhos
    );

    println!("  route_a_claim_families:");
    println!(
        "    shout={:>3} ({:>5.1}%)  twist={:>3} ({:>5.1}%)  wb/wp={:>3} ({:>5.1}%)",
        totals.route_a_breakdown.shout,
        percent_of(totals.route_a_breakdown.shout, route_claim_total),
        totals.route_a_breakdown.twist,
        percent_of(totals.route_a_breakdown.twist, route_claim_total),
        totals.route_a_breakdown.wb_wp,
        percent_of(totals.route_a_breakdown.wb_wp, route_claim_total),
    );
    println!(
        "    decode={:>3} ({:>5.1}%)  width={:>3} ({:>5.1}%)  control={:>3} ({:>5.1}%)",
        totals.route_a_breakdown.decode,
        percent_of(totals.route_a_breakdown.decode, route_claim_total),
        totals.route_a_breakdown.width,
        percent_of(totals.route_a_breakdown.width, route_claim_total),
        totals.route_a_breakdown.control,
        percent_of(totals.route_a_breakdown.control, route_claim_total),
    );
    println!(
        "    poseidon={:>3} ({:>5.1}%)  output={:>3} ({:>5.1}%)  other={:>3} ({:>5.1}%)",
        totals.route_a_breakdown.poseidon,
        percent_of(totals.route_a_breakdown.poseidon, route_claim_total),
        totals.route_a_breakdown.output,
        percent_of(totals.route_a_breakdown.output, route_claim_total),
        totals.route_a_breakdown.other,
        percent_of(totals.route_a_breakdown.other, route_claim_total),
    );

    println!(
        "  lane_claims: ccs_out={} dec_children={} fold_dec_children_total={}",
        totals.ccs_out_claims, totals.dec_children, totals.fold_lane_dec_children
    );
    println!(
        "  me_claims: val={} wb={} wp={} poseidon_cycle={} poseidon_local={}",
        totals.val_me_claims,
        totals.wb_me_claims,
        totals.wp_me_claims,
        totals.poseidon_cycle_me_claims,
        totals.poseidon_local_me_claims
    );
    println!(
        "  commitments_openings: time_cpu={} time_mem={} stage8_joint={} opening_points={} opening_cols={} opening_proofs={} opening_proof_cols={}",
        totals.time_cpu_commitments,
        totals.time_mem_commitments,
        totals.stage8_joint_commitments,
        totals.opening_points,
        totals.opening_cols,
        totals.opening_bound_proofs,
        totals.opening_bound_cols
    );
    println!(
        "  folds: val={} wb={} wp={} poseidon_cycle={} poseidon_local={} stage8={}",
        totals.val_fold_lanes,
        totals.wb_fold_lanes,
        totals.wp_fold_lanes,
        totals.poseidon_cycle_fold_lanes,
        totals.poseidon_local_fold_lanes,
        totals.stage8_fold_lanes
    );
    println!(
        "  proof_payloads: shout_proofs={} twist_proofs={}",
        totals.shout_proofs, totals.twist_proofs
    );

    println!(
        "  timings_ms: setup={:.1} chunk+commit={:.1} fold+prove={:.1} prove_wall={:.1} verify_run={:.1} verify_wall={:.1}",
        setup_duration.as_secs_f64() * 1000.0,
        chunk_commit_duration.as_secs_f64() * 1000.0,
        fold_and_prove_duration.as_secs_f64() * 1000.0,
        prove_wall.as_secs_f64() * 1000.0,
        verify_run_duration.as_secs_f64() * 1000.0,
        verify_wall.as_secs_f64() * 1000.0
    );
    println!(
        "  per_instruction_ms: prove={:.3} fold+prove={:.3} verify={:.3}",
        prove_wall.as_secs_f64() * 1000.0 / instructions,
        fold_and_prove_duration.as_secs_f64() * 1000.0 / instructions,
        verify_wall.as_secs_f64() * 1000.0 / instructions
    );
    println!(
        "  per_step_ms: public={:.3} fold={:.3}",
        fold_and_prove_duration.as_secs_f64() * 1000.0 / public_steps_safe,
        fold_and_prove_duration.as_secs_f64() * 1000.0 / fold_steps_safe
    );
    println!("  coeff_proxy_share (higher means heavier):");
    println!(
        "    route_a={:>8} ({:>5.1}%)  pi_ccs={:>8} ({:>5.1}%)  poseidon_local={:>8} ({:>5.1}%)  stage8={:>8} ({:>5.1}%)",
        route_a_coeffs,
        percent_of(route_a_coeffs, coeff_proxy_total),
        pi_ccs_coeffs,
        percent_of(pi_ccs_coeffs, coeff_proxy_total),
        poseidon_local_coeffs,
        percent_of(poseidon_local_coeffs, coeff_proxy_total),
        stage8_coeffs,
        percent_of(stage8_coeffs, coeff_proxy_total),
    );

    if verbose {
        println!("  per_step_flow (NS_PROTOCOL_METRICS_VERBOSE=1):");
        for row in rows {
            println!(
                "    step={} substeps={} ccs_rounds(fe/nc={} / {}) route_a(claims/rounds={} / {}) rlc_rhos={} dec_children={} commits(cpu/mem/joint={} / {} / {}) folds(val/wb/wp/p2c/p2l/s8={}/{}/{}/{}/{}/{})",
                row.step_id,
                row.compressed_substeps,
                row.ccs_fe_rounds,
                row.ccs_nc_rounds,
                row.route_a_claims,
                row.route_a_rounds,
                row.rlc_rhos,
                row.dec_children,
                row.time_cpu_commitments,
                row.time_mem_commitments,
                row.stage8_joint_commitments,
                row.val_fold_lanes,
                row.wb_fold_lanes,
                row.wp_fold_lanes,
                row.poseidon_cycle_fold_lanes,
                row.poseidon_local_fold_lanes,
                row.stage8_fold_lanes
            );
        }
    } else {
        println!("  per_step_flow=hidden (set NS_PROTOCOL_METRICS_VERBOSE=1 for details)");
    }
}

// ---------------------------------------------------------------------------
// Real note-spend transfer: 1-input, 1-output self-transfer with valid witness
// ---------------------------------------------------------------------------

/// Helper that builds a vector of (address, u32_value) pairs in the exact RAM
/// layout the circuit guest expects, mirroring `write_note_spend_witness` from
/// the sov-nightstream-adapter host.
mod witness_builder {
    use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;

    pub type GlDigest = [Goldilocks; 4];
    pub const ZERO_DIGEST: GlDigest = [Goldilocks::ZERO; 4];

    pub const TAG_MT_NODE: u64 = 1;
    pub const TAG_NOTE: u64 = 2;
    pub const TAG_PRF_NF: u64 = 3;
    pub const TAG_PK: u64 = 4;
    pub const TAG_ADDR: u64 = 5;
    pub const TAG_NFKEY: u64 = 6;
    pub const TAG_BL_BUCKET: u64 = 7;

    pub const BL_DEPTH: u32 = 16;
    pub const BL_BUCKET_SIZE: usize = 12;
    pub const INPUT_ADDR: u64 = 0x104;
    pub const OUTPUT_ADDR: u64 = 0x100;

    pub struct TransferWitness {
        pub ram_pairs: Vec<(u64, u32)>,
        pub output_claims: Vec<(u64, u32)>,
    }

    pub fn gl(v: u64) -> Goldilocks {
        Goldilocks::from_u64(v)
    }

    pub fn gl_digest_to_bytes(d: &GlDigest) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, elem) in d.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
        }
        out
    }

    pub fn h(input: &[Goldilocks]) -> GlDigest {
        poseidon2_hash(input)
    }

    /// Writes witness data into RAM as (addr, u32) pairs.
    pub struct RamWriter {
        addr: u64,
        pub pairs: Vec<(u64, u32)>,
    }

    impl RamWriter {
        pub fn new(start: u64) -> Self {
            Self {
                addr: start,
                pairs: Vec::new(),
            }
        }

        pub fn write_u32(&mut self, val: u32) {
            self.pairs.push((self.addr, val));
            self.addr += 4;
        }

        pub fn write_u64(&mut self, val: u64) {
            self.write_u32(val as u32);
            self.write_u32((val >> 32) as u32);
        }

        pub fn write_digest_bytes(&mut self, hash32: &[u8; 32]) {
            for i in 0..4 {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&hash32[i * 8..(i + 1) * 8]);
                self.write_u64(u64::from_le_bytes(buf));
            }
        }

        pub fn write_gl_digest(&mut self, d: &GlDigest) {
            self.write_digest_bytes(&gl_digest_to_bytes(d));
        }
    }

    /// Compute the default empty blacklist root (matches `default_blacklist_root`
    /// from the sovereign adapter).
    pub fn default_blacklist_root() -> (GlDigest, Vec<GlDigest>) {
        let empty_leaf = {
            let mut input = [Goldilocks::ZERO; 1 + BL_BUCKET_SIZE * 4];
            input[0] = gl(TAG_BL_BUCKET);
            h(&input)
        };

        let mut nodes: Vec<GlDigest> = Vec::with_capacity(BL_DEPTH as usize + 1);
        nodes.push(empty_leaf);
        for lvl in 0..BL_DEPTH {
            let prev = nodes[lvl as usize];
            let mut mt_input = [Goldilocks::ZERO; 10];
            mt_input[0] = gl(TAG_MT_NODE);
            mt_input[1] = gl(lvl as u64);
            mt_input[2..6].copy_from_slice(&prev);
            mt_input[6..10].copy_from_slice(&prev);
            nodes.push(h(&mt_input));
        }
        let root = nodes[BL_DEPTH as usize];
        (root, nodes)
    }

    /// Compute the bucket inverse witness for blacklist non-membership.
    pub fn compute_bucket_inv(id: &GlDigest, bucket_entries: &[[Goldilocks; 4]; BL_BUCKET_SIZE]) -> Goldilocks {
        let mut prod = Goldilocks::ONE;
        for entry in bucket_entries {
            for i in 0..4 {
                prod *= id[i] - entry[i];
            }
        }
        prod.inverse()
    }

    /// Build the complete RAM witness for a 1-input, 1-output self-transfer.
    pub fn build_1in_1out_transfer() -> TransferWitness {
        build_1in_1out_transfer_with_depth(2)
    }

    /// Build the complete RAM witness for a 1-input, 1-output self-transfer
    /// at an arbitrary Merkle depth for input inclusion.
    pub fn build_1in_1out_transfer_with_depth(depth: u32) -> TransferWitness {
        build_1in_1out_transfer_with_depth_and_addrs(depth, INPUT_ADDR, OUTPUT_ADDR)
    }

    /// Build the complete RAM witness for a 1-input, 1-output self-transfer
    /// with explicit input/output memory addresses (used by Sovereign ROMs).
    pub fn build_1in_1out_transfer_with_depth_and_addrs(
        depth: u32,
        input_addr: u64,
        output_addr: u64,
    ) -> TransferWitness {
        let mut ram = RamWriter::new(input_addr);

        // --- Deterministic keys ---
        let domain_gl: GlDigest = [gl(1), gl(1), gl(1), gl(1)];
        let spend_sk_gl: GlDigest = [gl(42), gl(43), gl(44), gl(45)];
        let pk_ivk_gl: GlDigest = [gl(100), gl(101), gl(102), gl(103)];

        // pk_spend = H(TAG_PK, spend_sk)
        let mut pk_input = [Goldilocks::ZERO; 5];
        pk_input[0] = gl(TAG_PK);
        pk_input[1..5].copy_from_slice(&spend_sk_gl);
        let pk_spend_gl = h(&pk_input);

        // nf_key = H(TAG_NFKEY, domain, spend_sk)
        let mut nfk_input = [Goldilocks::ZERO; 9];
        nfk_input[0] = gl(TAG_NFKEY);
        nfk_input[1..5].copy_from_slice(&domain_gl);
        nfk_input[5..9].copy_from_slice(&spend_sk_gl);
        let nf_key_gl = h(&nfk_input);

        // recipient = H(TAG_ADDR, domain, pk_spend, pk_ivk)
        let mut addr_input = [Goldilocks::ZERO; 13];
        addr_input[0] = gl(TAG_ADDR);
        addr_input[1..5].copy_from_slice(&domain_gl);
        addr_input[5..9].copy_from_slice(&pk_spend_gl);
        addr_input[9..13].copy_from_slice(&pk_ivk_gl);
        let recipient_gl = h(&addr_input);
        let sender_id_gl = recipient_gl;

        // --- Input note ---
        let value: u64 = 1000;
        let rho_gl: GlDigest = [gl(200), gl(201), gl(202), gl(203)];

        // cm = H(TAG_NOTE, domain, value, rho, recipient, sender_id)
        let mut cm_input = [Goldilocks::ZERO; 18];
        cm_input[0] = gl(TAG_NOTE);
        cm_input[1..5].copy_from_slice(&domain_gl);
        cm_input[5] = gl(value);
        cm_input[6..10].copy_from_slice(&rho_gl);
        cm_input[10..14].copy_from_slice(&recipient_gl);
        cm_input[14..18].copy_from_slice(&sender_id_gl);
        let cm_gl = h(&cm_input);

        // --- Merkle tree (leaf at position 0 with deterministic default-empty siblings) ---
        assert!(depth > 0, "depth must be > 0");
        let mut siblings: Vec<GlDigest> = Vec::with_capacity(depth as usize);
        siblings.push(ZERO_DIGEST);
        for lvl in 1..depth {
            let prev = siblings[(lvl - 1) as usize];
            let mut inp = [Goldilocks::ZERO; 10];
            inp[0] = gl(TAG_MT_NODE);
            inp[1] = gl((lvl - 1) as u64);
            inp[2..6].copy_from_slice(&prev);
            inp[6..10].copy_from_slice(&prev);
            siblings.push(h(&inp));
        }
        let anchor_gl = {
            let mut cur = cm_gl;
            for (lvl, sib) in siblings.iter().enumerate() {
                let mut inp = [Goldilocks::ZERO; 10];
                inp[0] = gl(TAG_MT_NODE);
                inp[1] = gl(lvl as u64);
                inp[2..6].copy_from_slice(&cur);
                inp[6..10].copy_from_slice(sib);
                cur = h(&inp);
            }
            cur
        };

        // nullifier = H(TAG_PRF_NF, domain, nf_key, rho)
        let mut nf_input = [Goldilocks::ZERO; 13];
        nf_input[0] = gl(TAG_PRF_NF);
        nf_input[1..5].copy_from_slice(&domain_gl);
        nf_input[5..9].copy_from_slice(&nf_key_gl);
        nf_input[9..13].copy_from_slice(&rho_gl);
        let nullifier_gl = h(&nf_input);

        // --- Output note (same value, self-transfer) ---
        let out_rho_gl: GlDigest = [gl(300), gl(301), gl(302), gl(303)];
        // Self-transfer: same pk_spend, pk_ivk -> same recipient
        let out_cm_gl = {
            let mut inp = [Goldilocks::ZERO; 18];
            inp[0] = gl(TAG_NOTE);
            inp[1..5].copy_from_slice(&domain_gl);
            inp[5] = gl(value);
            inp[6..10].copy_from_slice(&out_rho_gl);
            inp[10..14].copy_from_slice(&recipient_gl);
            inp[14..18].copy_from_slice(&sender_id_gl);
            h(&inp)
        };

        // --- Enforce product: prod(values) * prod(rho_diffs) != 0 ---
        let mut enforce_prod = gl(value) * gl(value);
        for i in 0..4 {
            enforce_prod *= out_rho_gl[i] - rho_gl[i];
        }
        let inv_enforce = enforce_prod.inverse();

        // --- Blacklist ---
        let (bl_root_gl, bl_nodes) = default_blacklist_root();
        let empty_bucket = [ZERO_DIGEST; BL_BUCKET_SIZE];
        let sender_bl_inv = compute_bucket_inv(&sender_id_gl, &empty_bucket);
        let recipient_bl_inv = compute_bucket_inv(&recipient_gl, &empty_bucket);

        // === Write witness to RAM ===
        // Header
        ram.write_gl_digest(&domain_gl);
        ram.write_gl_digest(&spend_sk_gl);
        ram.write_gl_digest(&pk_ivk_gl);
        ram.write_u32(depth);
        ram.write_gl_digest(&anchor_gl);
        ram.write_u32(1); // n_in

        // Input 0: value, rho, sender_id, position, siblings
        ram.write_u64(value);
        ram.write_gl_digest(&rho_gl);
        ram.write_gl_digest(&sender_id_gl);
        ram.write_u32(0); // position
        for sib in &siblings {
            ram.write_gl_digest(sib);
        }

        // Nullifier (separate loop in circuit)
        ram.write_gl_digest(&nullifier_gl);

        // Withdraw binding (pure transfer: amount=0, to=zero)
        ram.write_u64(0); // withdraw_amount
        ram.write_gl_digest(&ZERO_DIGEST); // withdraw_to
        ram.write_u32(1); // n_out

        // Output 0: value, rho, pk_spend, pk_ivk
        ram.write_u64(value);
        ram.write_gl_digest(&out_rho_gl);
        ram.write_gl_digest(&pk_spend_gl);
        ram.write_gl_digest(&pk_ivk_gl);

        // Output commitment public (separate loop)
        ram.write_gl_digest(&out_cm_gl);

        // Enforce product inverse
        ram.write_u64(inv_enforce.as_canonical_u64());

        // Blacklist root
        ram.write_gl_digest(&bl_root_gl);

        // Blacklist proof for sender (transfer -> 2 proofs: sender + pay recipient)
        // Proof 1: sender
        for _ in 0..BL_BUCKET_SIZE {
            ram.write_gl_digest(&ZERO_DIGEST);
        }
        ram.write_u64(sender_bl_inv.as_canonical_u64());
        for node in bl_nodes.iter().take(BL_DEPTH as usize) {
            ram.write_gl_digest(node);
        }

        // Proof 2: pay recipient (for transfers, withdraw_amount == 0)
        for _ in 0..BL_BUCKET_SIZE {
            ram.write_gl_digest(&ZERO_DIGEST);
        }
        ram.write_u64(recipient_bl_inv.as_canonical_u64());
        for node in bl_nodes.iter().take(BL_DEPTH as usize) {
            ram.write_gl_digest(node);
        }

        // Viewers: n_viewers = 0
        ram.write_u32(0);

        // === Expected public outputs at OUTPUT_ADDR ===
        let mut out = RamWriter::new(output_addr);
        out.write_gl_digest(&anchor_gl);
        out.write_u32(1); // n_in
        out.write_gl_digest(&nullifier_gl); // one input
        out.write_u64(0); // withdraw_amount
        out.write_gl_digest(&ZERO_DIGEST); // withdraw_to
        out.write_u32(1); // n_out
        out.write_gl_digest(&out_cm_gl); // one output commitment
        out.write_gl_digest(&bl_root_gl);
        out.write_u32(0); // n_viewers

        TransferWitness {
            ram_pairs: ram.pairs,
            output_claims: out.pairs,
        }
    }

    /// Build a note-deposit witness in the exact RAM/output layout expected by
    /// the Sovereign note_deposit circuit.
    pub fn build_note_deposit_with_addrs(input_addr: u64, output_addr: u64) -> TransferWitness {
        let mut ram = RamWriter::new(input_addr);

        let domain_gl: GlDigest = [gl(1), gl(1), gl(1), gl(1)];
        let value: u64 = 777;
        let rho_gl: GlDigest = [gl(200), gl(201), gl(202), gl(203)];
        let pk_spend_gl: GlDigest = [gl(42), gl(43), gl(44), gl(45)];
        let pk_ivk_gl: GlDigest = [gl(100), gl(101), gl(102), gl(103)];

        // recipient = H(TAG_ADDR, domain, pk_spend, pk_ivk)
        let recipient_gl = {
            let mut inp = [Goldilocks::ZERO; 13];
            inp[0] = gl(TAG_ADDR);
            inp[1..5].copy_from_slice(&domain_gl);
            inp[5..9].copy_from_slice(&pk_spend_gl);
            inp[9..13].copy_from_slice(&pk_ivk_gl);
            h(&inp)
        };

        // Deposit notes bind sender_id := recipient.
        let cm_out_gl = {
            let mut inp = [Goldilocks::ZERO; 18];
            inp[0] = gl(TAG_NOTE);
            inp[1..5].copy_from_slice(&domain_gl);
            inp[5] = gl(value);
            inp[6..10].copy_from_slice(&rho_gl);
            inp[10..14].copy_from_slice(&recipient_gl);
            inp[14..18].copy_from_slice(&recipient_gl);
            h(&inp)
        };

        let (bl_root_gl, bl_nodes) = default_blacklist_root();
        let empty_bucket = [ZERO_DIGEST; BL_BUCKET_SIZE];
        let recipient_bl_inv = compute_bucket_inv(&recipient_gl, &empty_bucket);

        // Input layout: domain, value, rho, pk_spend, pk_ivk, cm_out, blacklist_root,
        // then bucket entries, bucket_inv, siblings.
        ram.write_gl_digest(&domain_gl);
        ram.write_u64(value);
        ram.write_gl_digest(&rho_gl);
        ram.write_gl_digest(&pk_spend_gl);
        ram.write_gl_digest(&pk_ivk_gl);
        ram.write_gl_digest(&cm_out_gl);
        ram.write_gl_digest(&bl_root_gl);
        for _ in 0..BL_BUCKET_SIZE {
            ram.write_gl_digest(&ZERO_DIGEST);
        }
        ram.write_u64(recipient_bl_inv.as_canonical_u64());
        for node in bl_nodes.iter().take(BL_DEPTH as usize) {
            ram.write_gl_digest(node);
        }

        // Output layout: domain, value, recipient, cm_out, blacklist_root.
        let mut out = RamWriter::new(output_addr);
        out.write_gl_digest(&domain_gl);
        out.write_u64(value);
        out.write_gl_digest(&recipient_gl);
        out.write_gl_digest(&cm_out_gl);
        out.write_gl_digest(&bl_root_gl);

        TransferWitness {
            ram_pairs: ram.pairs,
            output_claims: out.pairs,
        }
    }
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureNoteSpendWitness {
    domain: [u8; 32],
    spend_sk: [u8; 32],
    pk_ivk_owner: [u8; 32],
    depth: u32,
    anchor: [u8; 32],
    inputs: Vec<FixtureInput>,
    withdraw_amount: u64,
    withdraw_to: [u8; 32],
    outputs: Vec<FixtureOutput>,
    inv_enforce: [u8; 32],
    blacklist_root: [u8; 32],
    blacklist_proofs: Vec<FixtureBlacklistProof>,
    viewers: Vec<FixtureViewer>,
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureInput {
    value: u64,
    rho: [u8; 32],
    sender_id: [u8; 32],
    position: u32,
    siblings: Vec<[u8; 32]>,
    nullifier: [u8; 32],
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureOutput {
    value: u64,
    rho: [u8; 32],
    pk_spend: [u8; 32],
    pk_ivk: [u8; 32],
    cm: [u8; 32],
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureBlacklistProof {
    bucket_entries: Vec<[u8; 32]>,
    bucket_inv: [u8; 32],
    siblings: Vec<[u8; 32]>,
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureViewer {
    fvk_commitment: [u8; 32],
    fvk: [u8; 32],
    per_output: Vec<FixtureViewerOutput>,
}

#[cfg(feature = "poseidon-precompile")]
#[derive(Clone, Deserialize)]
struct FixtureViewerOutput {
    ct_hash: [u8; 32],
    mac: [u8; 32],
}

#[cfg(feature = "poseidon-precompile")]
fn build_transfer_witness_from_fixture(f: &FixtureNoteSpendWitness) -> witness_builder::TransferWitness {
    build_transfer_witness_from_fixture_with_addrs(f, 0x104, 0x100)
}

#[cfg(feature = "poseidon-precompile")]
fn build_transfer_witness_from_fixture_with_addrs(
    f: &FixtureNoteSpendWitness,
    input_addr: u64,
    output_addr: u64,
) -> witness_builder::TransferWitness {
    struct RamWordWriter {
        addr: u64,
        pairs: Vec<(u64, u32)>,
    }
    impl RamWordWriter {
        fn new(start: u64) -> Self {
            Self {
                addr: start,
                pairs: Vec::new(),
            }
        }
        fn write_u32(&mut self, val: u32) {
            self.pairs.push((self.addr, val));
            self.addr += 4;
        }
        fn write_u64(&mut self, val: u64) {
            self.write_u32(val as u32);
            self.write_u32((val >> 32) as u32);
        }
        fn write_hash32(&mut self, d: &[u8; 32]) {
            for i in 0..4 {
                let mut word = [0u8; 8];
                word.copy_from_slice(&d[i * 8..(i + 1) * 8]);
                self.write_u64(u64::from_le_bytes(word));
            }
        }
    }

    let mut in_w = RamWordWriter::new(input_addr);
    in_w.write_hash32(&f.domain);
    in_w.write_hash32(&f.spend_sk);
    in_w.write_hash32(&f.pk_ivk_owner);
    in_w.write_u32(f.depth);
    in_w.write_hash32(&f.anchor);
    in_w.write_u32(f.inputs.len() as u32);

    for input in &f.inputs {
        in_w.write_u64(input.value);
        in_w.write_hash32(&input.rho);
        in_w.write_hash32(&input.sender_id);
        in_w.write_u32(input.position);
        for sib in &input.siblings {
            in_w.write_hash32(sib);
        }
    }
    for input in &f.inputs {
        in_w.write_hash32(&input.nullifier);
    }

    in_w.write_u64(f.withdraw_amount);
    in_w.write_hash32(&f.withdraw_to);
    in_w.write_u32(f.outputs.len() as u32);

    for output in &f.outputs {
        in_w.write_u64(output.value);
        in_w.write_hash32(&output.rho);
        in_w.write_hash32(&output.pk_spend);
        in_w.write_hash32(&output.pk_ivk);
    }
    for output in &f.outputs {
        in_w.write_hash32(&output.cm);
    }

    let mut inv_enforce_lo = [0u8; 8];
    inv_enforce_lo.copy_from_slice(&f.inv_enforce[..8]);
    in_w.write_u64(u64::from_le_bytes(inv_enforce_lo));

    in_w.write_hash32(&f.blacklist_root);
    for proof in &f.blacklist_proofs {
        for entry in &proof.bucket_entries {
            in_w.write_hash32(entry);
        }
        let mut inv_lo = [0u8; 8];
        inv_lo.copy_from_slice(&proof.bucket_inv[..8]);
        in_w.write_u64(u64::from_le_bytes(inv_lo));
        for sib in &proof.siblings {
            in_w.write_hash32(sib);
        }
    }

    in_w.write_u32(f.viewers.len() as u32);
    for viewer in &f.viewers {
        in_w.write_hash32(&viewer.fvk_commitment);
        in_w.write_hash32(&viewer.fvk);
        for out_w in &viewer.per_output {
            in_w.write_hash32(&out_w.ct_hash);
            in_w.write_hash32(&out_w.mac);
        }
    }

    let mut out_w = RamWordWriter::new(output_addr);
    out_w.write_hash32(&f.anchor);
    out_w.write_u32(f.inputs.len() as u32);
    for input in &f.inputs {
        out_w.write_hash32(&input.nullifier);
    }
    out_w.write_u64(f.withdraw_amount);
    out_w.write_hash32(&f.withdraw_to);
    out_w.write_u32(f.outputs.len() as u32);
    for output in &f.outputs {
        out_w.write_hash32(&output.cm);
    }
    out_w.write_hash32(&f.blacklist_root);
    out_w.write_u32(f.viewers.len() as u32);
    for viewer in &f.viewers {
        assert_eq!(
            viewer.per_output.len(),
            f.outputs.len(),
            "fixture viewer/per_output length must match n_out"
        );
        for (out_witness, output) in viewer.per_output.iter().zip(&f.outputs) {
            out_w.write_hash32(&output.cm);
            out_w.write_hash32(&viewer.fvk_commitment);
            out_w.write_hash32(&out_witness.ct_hash);
            out_w.write_hash32(&out_witness.mac);
        }
    }

    witness_builder::TransferWitness {
        ram_pairs: in_w.pairs,
        output_claims: out_w.pairs,
    }
}

/// Full prove+verify cycle for the note-spend circuit with a real 1-input,
/// 1-output self-transfer witness.
///
/// This mirrors the `test_note_spend_prove_verify_with_witness` integration
/// test from `sov-nightstream-adapter`, but runs directly against Nightstream's
/// `Rv32TraceWiring` API without the sovereign adapter layer.
#[test]
#[ignore = "requires poseidon-precompile feature and slow full prove/verify"]
fn test_note_spend_1in_1out_transfer_prove_verify() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let static_instruction_words = program_bytes.len() / 4;

    let mut witness = witness_builder::build_1in_1out_transfer();
    println!("witness_ram_words={}", witness.ram_pairs.len());
    println!("output_claim_words={}", witness.output_claims.len());

    // This ROM export currently includes only code bytes, so there is no separate
    // rodata blob to preload here.
    println!("total_ram_words={} (witness only)", witness.ram_pairs.len());

    // --- Simulation pass: measure step count and verify circuit halts ---
    let (executed_steps, simulated_output_claims, output_claim_mismatches) = {
        let decoded = decode_program(program_bytes).expect("decode ROM");
        let mut cpu = RiscvCpu::new(32);
        cpu.load_program(program_base, decoded);
        let mut twist = RiscvMemory::with_program_in_twist(32, PROG_ID, program_base, program_bytes);
        for &(addr, val) in &witness.ram_pairs {
            twist.store(RAM_ID, addr, val as u64);
        }
        let shout = RiscvShoutTables::new(32);
        let sim = trace_program(cpu, twist, shout, 200_000).expect("simulation trace");
        println!(
            "trace_sim_steps={} trace_sim_did_halt={} trace_sim_total_twist_events={} trace_sim_total_shout_events={}",
            sim.len(),
            sim.did_halt(),
            sim.total_twist_events(),
            sim.total_shout_events()
        );
        if !sim.did_halt() {
            panic!("circuit did not halt within 200K simulation steps");
        }
        let exec = Rv32ExecTable::from_trace_padded(&sim, sim.steps.len()).expect("simulation exec table");
        let output_addrs: Vec<u64> = witness
            .output_claims
            .iter()
            .map(|(addr, _)| *addr)
            .collect();
        let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
        let simulated_map: HashMap<u64, u32> = simulated_claims.iter().copied().collect();
        let mismatches = witness
            .output_claims
            .iter()
            .filter_map(|(addr, claimed)| {
                let actual = *simulated_map.get(addr).unwrap_or(&0u32);
                if actual == *claimed {
                    None
                } else {
                    Some((*addr, *claimed, actual))
                }
            })
            .collect::<Vec<_>>();
        (sim.steps.len(), simulated_claims, mismatches)
    };
    witness.output_claims = simulated_output_claims;
    if !output_claim_mismatches.is_empty() {
        println!(
            "output_claim_mismatches_detected={} (using simulated final RAM outputs for binding)",
            output_claim_mismatches.len()
        );
    }

    // Poseidon lane split requires t_len <= (ccs_m - m_in).
    // For this circuit/profile that cap is 510, so pick the largest safe chunk automatically.
    const MAX_SAFE_CHUNK_ROWS: usize = 510;
    let chunk_rows = executed_steps.min(MAX_SAFE_CHUNK_ROWS);
    let max_steps = executed_steps;
    let min_trace_len = executed_steps;

    let setup_wall_start = Instant::now();
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(min_trace_len)
        .chunk_rows(chunk_rows)
        .max_steps(max_steps)
        .shout_auto_minimal();

    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }
    let setup_wall = setup_wall_start.elapsed();

    println!("prove_start");
    let prove_start = Instant::now();
    let mut run = match wiring.prove() {
        Ok(run) => run,
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("poseidon-precompile") || msg.contains("PoseidonPrecompile") {
                println!("Skipping: poseidon-precompile feature is not enabled");
                return;
            }
            panic!("prove failed: {msg}");
        }
    };
    let prove_ms = prove_start.elapsed().as_millis();
    let prove_wall = prove_start.elapsed();

    let phases = run.prove_phase_durations();
    let layout = run.layout();
    println!("\n=============================================");
    println!("  Note-Spend 1-in/1-out Transfer (Nightstream)");
    println!("=============================================");
    println!(
        "  program_base={} rom_bytes={} static_instruction_words={}",
        program_base,
        program_bytes.len(),
        static_instruction_words
    );
    println!(
        "  min_trace_len={} chunk_rows={} max_steps={}",
        min_trace_len, chunk_rows, max_steps
    );
    println!("  setup_time_wall={:?}", setup_wall);
    println!("  prove_wall_time={:?}", prove_wall);
    println!("  RISC-V steps:     {}", run.trace_len());
    println!("  trace_hit_max_steps_cap={}", run.trace_len() == max_steps);
    println!("  CCS constraints:  {}", run.ccs_num_constraints());
    println!("  CCS variables:    {}", run.ccs_num_variables());
    println!(
        "  layout_t={} layout_m_in={} layout_m={}",
        layout.t, layout.m_in, layout.m
    );
    println!("  used_memory_ids={:?}", run.used_memory_ids());
    println!("  used_shout_table_ids={:?}", run.used_shout_table_ids());
    println!("  Folding steps:    {}", run.fold_count());
    println!("  Chunk rows:       {}", chunk_rows);
    println!("  Prove time:       {} ms", prove_ms);
    println!("    setup:          {:.1} ms", phases.setup.as_secs_f64() * 1000.0);
    println!(
        "    chunk+commit:   {:.1} ms",
        phases.chunk_build_commit.as_secs_f64() * 1000.0
    );
    println!(
        "    fold+prove:     {:.1} ms",
        phases.fold_and_prove.as_secs_f64() * 1000.0
    );

    let verify_start = Instant::now();
    println!("verify_start");
    let verify_result = run.verify();
    let verify_ms = verify_start.elapsed().as_millis();
    let verify_wall = verify_start.elapsed();
    println!("  Verify time:      {} ms", verify_ms);
    println!(
        "    verify_run:     {:.1} ms",
        run.verify_duration().unwrap_or(verify_wall).as_secs_f64() * 1000.0
    );
    println!("    verify_wall:    {:.1} ms", verify_wall.as_secs_f64() * 1000.0);
    #[cfg(feature = "protocol-metrics")]
    print_superneo_protocol_metrics(
        run.proof(),
        run.trace_len(),
        run.fold_count(),
        phases.setup,
        phases.chunk_build_commit,
        phases.fold_and_prove,
        prove_wall,
        verify_wall,
        run.verify_duration(),
    );
    #[cfg(not(feature = "protocol-metrics"))]
    println!("  protocol_metrics_feature=disabled (enable with --features protocol-metrics)");
    println!("=============================================\n");

    if let Err(err) = verify_result {
        panic!("verification failed: {err}");
    }
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_output_claim_template_mismatch_is_detected() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let witness = witness_builder::build_1in_1out_transfer();
    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 200_000)
        .expect("simulate note-spend witness");
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    let simulated_map: HashMap<u64, u32> = simulated_claims.iter().copied().collect();
    let mismatches = witness
        .output_claims
        .iter()
        .filter(|(addr, claimed)| simulated_map.get(addr).copied().unwrap_or(0) != *claimed)
        .count();
    assert!(
        mismatches > 0,
        "expected this test to detect template-vs-sim output mismatches; got {mismatches}"
    );
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_depth16_poseidon_geometry_regression_is_reproducible() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let mut witness = witness_builder::build_1in_1out_transfer_with_depth(16);

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 300_000)
        .expect("simulate depth16 note-spend witness");
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    witness.output_claims = simulated_claims;

    let chunk_rows = std::env::var("NS_DEPTH16_CHUNK_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(42);

    let executed_steps = exec.rows.len();
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(chunk_rows)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let mut run = wiring
        .prove()
        .expect("depth16 poseidon geometry regression should prove after carry fix");
    run.verify()
        .expect("depth16 poseidon geometry regression should verify after carry fix");
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_depth16_poseidon_geometry_repro_prove_verify() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let mut witness = witness_builder::build_1in_1out_transfer_with_depth(16);

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 300_000)
        .expect("simulate depth16 note-spend witness");
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    witness.output_claims = simulated_claims;

    let chunk_rows = std::env::var("NS_DEPTH16_CHUNK_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(42);

    let executed_steps = exec.rows.len();
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(chunk_rows)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let mut run = wiring
        .prove()
        .expect("depth16 poseidon geometry repro should prove after carry fix");
    run.verify()
        .expect("depth16 poseidon geometry repro should verify after carry fix");
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_stale_output_claims_repro_fails_at_verify() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let witness = witness_builder::build_1in_1out_transfer();

    // Keep template output claims as-is (do NOT reconcile against simulation).
    let executed_steps = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 200_000)
        .expect("simulate note-spend witness")
        .rows
        .len();

    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(510)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let mut run = wiring
        .prove()
        .expect("stale output-claim repro should still prove");
    let err = run
        .verify()
        .expect_err("stale output-claim repro must fail at verify()");
    let msg = err.to_string();
    assert!(
        msg.contains("output sparse final check failed")
            || msg.contains("output binding final check failed")
            || msg.contains("output sumcheck failed"),
        "unexpected stale-output verify error: {msg}"
    );
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_depth16_poseidon_chunk17_repro_prove_verify() {
    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let mut witness = witness_builder::build_1in_1out_transfer_with_depth(16);

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 300_000)
        .expect("simulate depth16 note-spend witness");
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    witness.output_claims = simulated_claims;

    // Fixed geometry profile that currently fails in this branch's poseidon path.
    let chunk_rows = 17usize;
    let executed_steps = exec.rows.len();
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(chunk_rows)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let mut run = wiring
        .prove()
        .expect("depth16 chunk-17 poseidon repro should prove after carry fix");
    run.verify()
        .expect("depth16 chunk-17 poseidon repro should verify after carry fix");
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_note_spend_sovereign_fixture_repro_prove_verify() {
    let fixture: FixtureNoteSpendWitness =
        serde_json::from_str(include_str!("fixtures/nightstream_note_spend_poseidon_fail.json"))
            .expect("parse fixture JSON");

    let program_base = circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM_BASE;
    let program_bytes: &[u8] = &circuit_l2_transfer_rom::CIRCUIT_L2_TRANSFER_ROM;
    let mut witness = build_transfer_witness_from_fixture(&fixture);

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 400_000)
        .expect("simulate fixture witness");
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let simulated_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    witness.output_claims = simulated_claims;

    let chunk_rows = std::env::var("NS_FIXTURE_CHUNK_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(42);

    let executed_steps = exec.rows.len();
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(chunk_rows)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let mut run = wiring
        .prove()
        .expect("Sovereign fixture repro should prove after carry fix");
    run.verify()
        .expect("Sovereign fixture repro should verify after carry fix");
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_sovereign_note_spend_rom_fixture_repro_fails_with_poseidon_split_tlen_798() {
    let fixture: FixtureNoteSpendWitness = serde_json::from_str(include_str!(
        "fixtures/sovereign_note_spend_poseidon_split_tlen798.json"
    ))
    .expect("parse sovereign note-spend poseidon-split fixture JSON");

    let program_base = sovereign_note_spend_rom::NOTE_SPEND_ROM_BASE;
    let program_bytes: &[u8] = &sovereign_note_spend_rom::NOTE_SPEND_ROM;
    let witness = build_transfer_witness_from_fixture_with_addrs(&fixture, 0x4104, 0x4100);

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 400_000)
        .expect("simulate sovereign note-spend fixture witness");

    let executed_steps = exec.rows.len();

    let chunk_rows = 42usize;
    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(executed_steps)
        .chunk_rows(chunk_rows)
        .max_steps(executed_steps)
        .shout_auto_minimal();
    for &(addr, val) in &witness.ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }
    for &(addr, val) in &witness.output_claims {
        wiring = wiring.output_claim(addr, F::from_u64(val as u64));
    }

    let err = match wiring.prove() {
        Ok(_) => panic!("sovereign note-spend fixture must reproduce poseidon split failure"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("poseidon split: ccs_m too small for one time-column after m_in offset")
            && msg.contains("ccs_m=518")
            && msg.contains("m_in=5")
            && msg.contains("t_len=798"),
        "unexpected prove error for sovereign poseidon-split repro: {msg}"
    );
}

#[cfg(feature = "poseidon-precompile")]
#[test]
#[ignore = "slow sovereign-like note_spend proving repro"]
fn test_sovereign_note_spend_slow_proving_repro() {
    let program_base = sovereign_note_spend_rom::NOTE_SPEND_ROM_BASE;
    let program_bytes: &[u8] = &sovereign_note_spend_rom::NOTE_SPEND_ROM;

    let witness = witness_builder::build_1in_1out_transfer_with_depth_and_addrs(2, 0x4104, 0x4100);
    println!(
        "sovereign_note_spend_slow: witness_ram_words={} output_claim_words={}",
        witness.ram_pairs.len(),
        witness.output_claims.len()
    );

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 400_000)
        .expect("simulate sovereign note-spend witness");
    let executed_steps = exec.rows.len();
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let output_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    let mismatch_count = witness
        .output_claims
        .iter()
        .zip(output_claims.iter())
        .filter(|((_, expected), (_, actual))| expected != actual)
        .count();
    println!(
        "sovereign_note_spend_slow: simulated_steps={} output_claim_mismatches={}",
        executed_steps, mismatch_count
    );

    let t_prove = Instant::now();
    let (mut run, final_chunk_rows) = prove_with_poseidon_retry(
        program_base,
        program_bytes,
        &witness.ram_pairs,
        &output_claims,
        executed_steps,
        510,
        8,
    )
    .expect("sovereign note-spend slow repro should prove");
    let prove_ms = t_prove.elapsed().as_millis();

    let t_verify = Instant::now();
    run.verify()
        .expect("sovereign note-spend slow repro should verify");
    let verify_ms = t_verify.elapsed().as_millis();

    println!(
        "sovereign_note_spend_slow: prove_ms={} verify_ms={} steps={} folds={} chunk_rows={}",
        prove_ms,
        verify_ms,
        run.trace_len(),
        run.fold_count(),
        final_chunk_rows
    );
}

#[cfg(feature = "poseidon-precompile")]
#[test]
#[ignore = "slow sovereign-like note_deposit proving repro"]
fn test_sovereign_note_deposit_slow_proving_repro() {
    let program_base = sovereign_note_deposit_rom::NOTE_DEPOSIT_ROM_BASE;
    let program_bytes: &[u8] = &sovereign_note_deposit_rom::NOTE_DEPOSIT_ROM;

    let witness = witness_builder::build_note_deposit_with_addrs(0x4104, 0x4100);
    println!(
        "sovereign_note_deposit_slow: witness_ram_words={} output_claim_words={}",
        witness.ram_pairs.len(),
        witness.output_claims.len()
    );

    let exec = simulate_exec_with_witness(program_base, program_bytes, &witness.ram_pairs, 400_000)
        .expect("simulate sovereign note-deposit witness");
    let executed_steps = exec.rows.len();
    let output_addrs: Vec<u64> = witness
        .output_claims
        .iter()
        .map(|(addr, _)| *addr)
        .collect();
    let output_claims = derive_output_claims_for_addresses(&exec, &witness.ram_pairs, &output_addrs);
    let mismatch_count = witness
        .output_claims
        .iter()
        .zip(output_claims.iter())
        .filter(|((_, expected), (_, actual))| expected != actual)
        .count();
    println!(
        "sovereign_note_deposit_slow: simulated_steps={} output_claim_mismatches={}",
        executed_steps, mismatch_count
    );

    let t_prove = Instant::now();
    let (mut run, final_chunk_rows) = prove_with_poseidon_retry(
        program_base,
        program_bytes,
        &witness.ram_pairs,
        &output_claims,
        executed_steps,
        510,
        8,
    )
    .expect("sovereign note-deposit slow repro should prove");
    let prove_ms = t_prove.elapsed().as_millis();

    let t_verify = Instant::now();
    run.verify()
        .expect("sovereign note-deposit slow repro should verify");
    let verify_ms = t_verify.elapsed().as_millis();

    println!(
        "sovereign_note_deposit_slow: prove_ms={} verify_ms={} steps={} folds={} chunk_rows={}",
        prove_ms,
        verify_ms,
        run.trace_len(),
        run.fold_count(),
        final_chunk_rows
    );
}
