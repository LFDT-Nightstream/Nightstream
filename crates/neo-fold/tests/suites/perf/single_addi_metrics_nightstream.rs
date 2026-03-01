use std::time::{Duration, Instant};

use neo_ccs::CeClaim;
use neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_fold::shard::ShardProof;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, Rv32TraceCcsLayout};
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test perf -- --ignored --nocapture compare_single_mixed_metrics_nightstream_only`"]
fn compare_single_mixed_metrics_nightstream_only() {
    let instruction_label = "Mixed sequence (ADD/AND/OR/XOR/SLT/SLTU/SLL/SRL/SRA/BNE)";

    let ns_program = mixed_instruction_sequence();
    let ns_program_bytes = encode_program(&ns_program);
    let ns_chunk_rows = ns_program.len();
    let ns_max_steps = ns_program.len();

    let ns_total_start = Instant::now();
    let mut ns_run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &ns_program_bytes)
        .chunk_rows(ns_chunk_rows)
        .max_steps(ns_max_steps)
        .prove()
        .expect("Nightstream prove");

    let ns_constraints = ns_run.ccs_num_constraints();
    let ns_witness_cols_physical = ns_run.ccs_num_variables();
    let ns_witness_cols_uniform = ns_run.uniform_ccs_num_variables();
    let ns_constraints_padded_pow2 = ns_constraints.next_power_of_two();
    let ns_witness_cols_padded_pow2 = ns_witness_cols_physical.next_power_of_two();
    let ns_fold_count = ns_run.fold_count();
    let ns_trace_len = ns_run.trace_len();
    let ns_shout_tables = ns_run.used_shout_table_ids().len();
    let ns_step0 = ns_run
        .steps_public()
        .first()
        .cloned()
        .expect("Nightstream collected steps");
    let ns_m_in = ns_step0.mcs_inst.m_in;
    let ns_witness_private = ns_witness_cols_physical.saturating_sub(ns_m_in);
    let ns_lut_instances = ns_step0.lut_insts.len();
    let ns_mem_instances = ns_step0.mem_insts.len();

    ns_run.verify().expect("Nightstream verify");
    let ns_prove_time = ns_run.prove_duration();
    let ns_verify_time = ns_run
        .verify_duration()
        .expect("Nightstream verify duration");
    let ns_total_duration = ns_total_start.elapsed();

    println!();
    println!("Instruction under test: {instruction_label}");
    println!();
    println!("**Nightstream (RV32 Trace)**");
    println!(
        "- CCS: n={} constraints (padded_pow2_n={}), m={} cols (padded_pow2_m={}) (m_in={} public, w={} private)",
        ns_constraints,
        ns_constraints_padded_pow2,
        ns_witness_cols_physical,
        ns_witness_cols_padded_pow2,
        ns_m_in,
        ns_witness_private
    );
    println!(
        "- Trace: executed_steps={} (max_steps={}), fold_chunks={} (chunk_rows={})",
        ns_trace_len, ns_max_steps, ns_fold_count, ns_chunk_rows
    );
    println!(
        "- Sidecars: lut_instances={} mem_instances={} shout_tables_used={}",
        ns_lut_instances, ns_mem_instances, ns_shout_tables
    );
    println!(
        "- Time: prove={} verify={} total_end_to_end={}",
        fmt_duration(ns_prove_time),
        fmt_duration(ns_verify_time),
        fmt_duration(ns_total_duration)
    );
    println!();

    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Metric", "Nightstream");
    println!("{:<40} {:>18}", "", "(RV32 Trace)");
    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Rows per step (raw)", ns_constraints);
    println!(
        "{:<40} {:>18}",
        "Rows per step (padded pow2)", ns_constraints_padded_pow2
    );
    println!(
        "{:<40} {:>18}",
        "Total rows in proof (padded)",
        ns_constraints_padded_pow2.saturating_mul(ns_fold_count)
    );
    println!(
        "{:<40} {:>18}",
        "Total rows (estimate, unpadded)",
        ns_constraints.saturating_mul(ns_trace_len)
    );
    println!(
        "{:<40} {:>18}",
        "Cols / vars (raw, physical ccs.m)", ns_witness_cols_physical
    );
    println!("{:<40} {:>18}", "Cols / vars (uniform width)", ns_witness_cols_uniform);
    println!(
        "{:<40} {:>18}",
        "Cols / vars (padded pow2)", ns_witness_cols_padded_pow2
    );
    println!("{:<40} {:>18}", "Public inputs (m_in)", ns_m_in);
    println!(
        "{:<40} {:>18}",
        "Trace len (unpadded)",
        format!("{} steps", ns_trace_len)
    );
    println!("{:<40} {:>18}", "Lookup tables", format!("{} Shout", ns_lut_instances));
    println!("{:<40} {:>18}", "Shout tables used", ns_shout_tables);
    println!("{:<40} {:>18}", "Prove time", fmt_duration(ns_prove_time));
    println!("{:<40} {:>18}", "Verify time", fmt_duration(ns_verify_time));
    println!("{:-<80}", "");
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs_f64() < 1.0 {
        format!("{:.3}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OpeningSurfaceBuckets {
    core_ccs: usize,
    sidecars: usize,
    claim_reduction_linkage: usize,
    pcs_open: usize,
}

impl OpeningSurfaceBuckets {
    fn total(self) -> usize {
        self.core_ccs + self.sidecars + self.claim_reduction_linkage + self.pcs_open
    }
}

fn sum_y_scalars<C, FF, KK>(claims: &[CeClaim<C, FF, KK>]) -> usize {
    claims.iter().map(|me| me.ct.len()).sum()
}

fn opening_surface_from_shard_proof(proof: &ShardProof) -> OpeningSurfaceBuckets {
    let mut buckets = OpeningSurfaceBuckets::default();
    for step in &proof.steps {
        buckets.core_ccs += sum_y_scalars(&step.fold.ccs_out);

        buckets.sidecars += sum_y_scalars(&step.mem.val_me_claims);

        buckets.claim_reduction_linkage += sum_y_scalars(&step.mem.wb_me_claims);
        buckets.claim_reduction_linkage += sum_y_scalars(&step.mem.wp_me_claims);
        buckets.claim_reduction_linkage += step.batched_time.claimed_sums.len();

        buckets.pcs_open += step.fold.dec_children.len();
        buckets.pcs_open += step
            .val_fold
            .iter()
            .map(|p| p.dec_children.len())
            .sum::<usize>();
        buckets.pcs_open += step
            .wb_fold
            .iter()
            .map(|p| p.dec_children.len())
            .sum::<usize>();
        buckets.pcs_open += step
            .wp_fold
            .iter()
            .map(|p| p.dec_children.len())
            .sum::<usize>();
    }
    buckets
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn mixed_instruction_sequence() -> Vec<RiscvInstruction> {
    vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ]
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_single_n_mixed_ops"]
fn debug_trace_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);

    assert!(chunk_rows > 0);

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove()
        .expect("trace prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace verify");
    let verify_time = run.verify_duration().expect("trace verify duration");
    let total_time = total_start.elapsed();
    let phases = run.prove_phase_durations();

    println!(
        "TRACE n={} chunk_rows={} ccs_n={} ccs_m_physical={} ccs_m_uniform={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={} phases(setup={}, chunk_commit={}, fold={})",
        n,
        chunk_rows,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.uniform_ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        run.trace_len(),
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        fmt_duration(phases.setup),
        fmt_duration(phases.chunk_build_commit),
        fmt_duration(phases.fold_and_prove),
    );
    let openings = opening_surface_from_shard_proof(run.proof());
    println!(
        "TRACE_OPENINGS core_ccs={} sidecars={} claim_reduction_linkage={} pcs_open={} total={}",
        openings.core_ccs,
        openings.sidecars,
        openings.claim_reduction_linkage,
        openings.pcs_open,
        openings.total()
    );
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_chunked_single_n_mixed_ops"]
fn debug_chunked_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .chunk_rows(steps)
        .max_steps(steps)
        .prove()
        .expect("trace single-chunk prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace single-chunk verify");
    let verify_time = run
        .verify_duration()
        .expect("trace single-chunk verify duration");
    let total_time = total_start.elapsed();
    let trace_len = run.trace_len();
    let phases = run.prove_phase_durations();

    println!(
        "TRACE_SINGLE_CHUNK n={} ccs_n={} ccs_m_physical={} ccs_m_uniform={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={} phases(setup={}, chunk_commit={}, fold={})",
        n,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.uniform_ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        trace_len,
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        fmt_duration(phases.setup),
        fmt_duration(phases.chunk_build_commit),
        fmt_duration(phases.fold_and_prove),
    );
    let openings = opening_surface_from_shard_proof(run.proof());
    println!(
        "TRACE_SINGLE_CHUNK_OPENINGS core_ccs={} sidecars={} claim_reduction_linkage={} pcs_open={} total={}",
        openings.core_ccs,
        openings.sidecars,
        openings.claim_reduction_linkage,
        openings.pcs_open,
        openings.total()
    );
}

#[test]
#[ignore = "perf-style report hook: cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_core_rows_per_cycle_equiv"]
fn debug_trace_core_rows_per_cycle_equiv() {
    let t = env_usize("NS_DEBUG_T", 257);
    let layout = Rv32TraceCcsLayout::new(t).expect("trace layout");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace core ccs");
    println!(
        "TRACE_CORE t={} trace_width={} core_ccs_n={} rows_per_cycle={:.3}",
        t,
        layout.trace.cols,
        ccs.n,
        ccs.n as f64 / t as f64
    );
}

#[test]
#[ignore = "W0 snapshot: NS_DEBUG_N=10 cargo test -p neo-fold --release --test perf -- --ignored --nocapture report_track_a_w0_w1_snapshot"]
fn report_track_a_w0_w1_snapshot() {
    let n = env_usize("NS_DEBUG_N", 256);

    let chunk_rows = n + 1;

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove()
        .expect("trace prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace verify");
    let verify_time = run.verify_duration().expect("trace verify duration");
    let total_time = total_start.elapsed();
    let openings = opening_surface_from_shard_proof(run.proof());

    let layout = run.layout().clone();
    let core_ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace core ccs");
    let rows_per_cycle = core_ccs.n as f64 / steps as f64;
    let cpu_trace_cols = layout.trace.cols;
    let public_input_cols = layout.m_in;

    let sep = "=".repeat(80);
    let thin_sep = "-".repeat(80);

    println!("\n{sep}");
    println!("  TRACK A CONSTRAINT ARCHITECTURE REPORT (n={steps} steps)");
    println!("{sep}\n");

    // ── 1. Main CCS Layer ──
    println!("1. MAIN CCS LAYER (core glue constraints)");
    println!("{thin_sep}");
    println!("  Trace columns:           {cpu_trace_cols}");
    println!("  Core CCS rows (n):       {}", core_ccs.n);
    println!("  Uniform witness cols (m):{}", core_ccs.m);
    println!("  Rows per cycle:          {:.3}", rows_per_cycle);
    println!("  Public inputs (m_in):    {public_input_cols}");
    println!();

    let col_names = [
        "one",
        "active",
        "halted",
        "cycle",
        "pc_before",
        "pc_after",
        "instr_word",
        "rs1_addr",
        "rs1_val",
        "rs2_addr",
        "rs2_val",
        "rd_addr",
        "rd_val",
        "ram_addr",
        "ram_rv",
        "ram_wv",
        "shout_has_lookup",
        "shout_val",
        "shout_lhs",
        "shout_rhs",
        "jalr_drop_bit",
    ];
    println!("  Trace columns ({}):", col_names.len());
    for (i, name) in col_names.iter().enumerate() {
        println!("    [{i:>2}] {name}");
    }
    println!();

    // ── 2. Shared CPU Bus (Sidecar) Layer ──
    println!("2. SHARED CPU BUS LAYER (Shout + Twist named columns)");
    println!("{thin_sep}");
    let total_ccs_m_physical = run.ccs_num_variables();
    let total_ccs_m_uniform = run.uniform_ccs_num_variables();
    let total_ccs_n = run.ccs_num_constraints();
    let cpu_base_m = public_input_cols.saturating_add(cpu_trace_cols);
    let committed_mem_bus_cols = total_ccs_m_uniform.saturating_sub(cpu_base_m);
    let bus_tail_cols_physical = total_ccs_m_physical.saturating_sub(total_ccs_m_uniform);
    println!("  Uniform CCS width proxy: {total_ccs_m_uniform}");
    println!("  Physical CCS m:          {total_ccs_m_physical}");
    println!("  Total CCS n (with bus):  {total_ccs_n}");
    println!("  CPU base cols (m_in+trace): {cpu_base_m}");
    println!("  Committed mem/bus cols:  {committed_mem_bus_cols}");
    println!("  Bus-tail columns (legacy physical): {bus_tail_cols_physical}");
    let bus_reserved_rows = total_ccs_n.saturating_sub(core_ccs.n);
    println!(
        "  Bus reserved rows:       {bus_reserved_rows} (total_n={total_ccs_n} - core_n={})",
        core_ccs.n
    );
    assert_eq!(
        total_ccs_m_physical, total_ccs_m_uniform,
        "route-a width gate: physical ccs.m must equal uniform width proxy"
    );
    assert_eq!(
        bus_tail_cols_physical, 0,
        "route-a width gate: legacy bus tail must be 0"
    );
    assert_eq!(bus_reserved_rows, 0, "route-a rows gate: reserved bus rows must be 0");
    println!();

    let step0 = run
        .steps_public()
        .into_iter()
        .next()
        .expect("at least one step");
    let n_lut = step0.lut_insts.len();
    let n_mem = step0.mem_insts.len();
    println!("  Shout instances (LUT):   {n_lut}");
    for inst in &step0.lut_insts {
        let ell_addr = inst.d * inst.ell;
        let bus_cols_per_lane = ell_addr + 2;
        println!(
            "    - table_id={:<10} d={} n_side={} ell={} lanes={} bus_cols={}",
            inst.table_id,
            inst.d,
            inst.n_side,
            inst.ell,
            inst.lanes,
            bus_cols_per_lane * inst.lanes
        );
    }
    println!("  Twist instances (MEM):   {n_mem}");
    for inst in &step0.mem_insts {
        let ell_addr = inst.d * inst.ell;
        let bus_cols_per_lane = 2 * ell_addr + 5;
        println!(
            "    - mem_id={:<10} d={} n_side={} ell={} lanes={} bus_cols={}",
            inst.mem_id,
            inst.d,
            inst.n_side,
            inst.ell,
            inst.lanes,
            bus_cols_per_lane * inst.lanes
        );
    }
    let shout_gamma_groups = RouteATimeClaimPlan::derive_shout_gamma_groups_for_instances(step0.lut_insts.iter());
    if !shout_gamma_groups.is_empty() {
        let mut old_value_width = 0usize;
        let mut old_adapter_width = 0usize;
        let mut old_bitness_width = 0usize;
        let mut new_value_width = 0usize;
        let mut new_adapter_width = 0usize;
        let mut new_bitness_width = 0usize;
        for g in shout_gamma_groups.iter() {
            let lane_count = g.lanes.len();
            old_value_width = old_value_width.saturating_add(2usize.saturating_mul(lane_count));
            old_adapter_width = old_adapter_width.saturating_add((1usize + g.ell_addr).saturating_mul(lane_count));
            old_bitness_width = old_bitness_width.saturating_add((1usize + g.ell_addr).saturating_mul(lane_count));

            let mut selector_mark: Option<Option<u64>> = None;
            let mut selector_consistent = true;
            for lane in g.lanes.iter() {
                let sel = step0
                    .lut_insts
                    .get(lane.inst_idx)
                    .map(|inst| inst.selector_group)
                    .unwrap_or(None);
                if let Some(prev) = selector_mark {
                    if prev != sel {
                        selector_consistent = false;
                        break;
                    }
                } else {
                    selector_mark = Some(sel);
                }
            }
            let has_shared_selector = selector_consistent && selector_mark.flatten().is_some();
            let has_arity = if has_shared_selector { 1usize } else { lane_count };
            new_value_width = new_value_width.saturating_add(has_arity.saturating_add(lane_count));
            new_adapter_width = new_adapter_width.saturating_add(g.ell_addr.saturating_add(has_arity));
            new_bitness_width = new_bitness_width.saturating_add(g.ell_addr.saturating_add(has_arity));
        }
        println!("  Shout grouped fan-in (legacy reference-only): value={old_value_width} adapter={old_adapter_width} bitness={old_bitness_width} total={}",
            old_value_width.saturating_add(old_adapter_width).saturating_add(old_bitness_width));
        println!("  Shout grouped fan-in (shared-aware, active): value={new_value_width} adapter={new_adapter_width} bitness={new_bitness_width} total={}",
            new_value_width.saturating_add(new_adapter_width).saturating_add(new_bitness_width));
    }
    println!();

    // ── 3. Route-A Claims ──
    println!("3. ROUTE-A BATCHED TIME CLAIMS");
    println!("{thin_sep}");
    let proof = run.proof();
    let step_proof = &proof.steps[0];
    let bt = &step_proof.batched_time;
    println!("  Total batched claims:    {}", bt.claimed_sums.len());
    println!();

    // Group claims by category.
    let mut ccs_claims = Vec::new();
    let mut shout_claims = Vec::new();
    let mut twist_claims = Vec::new();
    let mut wb_wp_claims = Vec::new();
    let mut decode_claims = Vec::new();
    let mut width_claims = Vec::new();
    let mut control_claims = Vec::new();
    let mut other_claims = Vec::new();

    for i in 0..bt.labels.len() {
        let label = std::str::from_utf8(&bt.labels[i]).unwrap_or("<invalid>");
        let deg = bt.degree_bounds[i];
        let entry = (label.to_string(), deg);
        if label.starts_with("ccs/") {
            ccs_claims.push(entry);
        } else if label.starts_with("shout/") {
            shout_claims.push(entry);
        } else if label.starts_with("twist/") {
            twist_claims.push(entry);
        } else if label.starts_with("wb/") || label.starts_with("wp/") {
            wb_wp_claims.push(entry);
        } else if label.starts_with("decode/") {
            decode_claims.push(entry);
        } else if label.starts_with("width/") {
            width_claims.push(entry);
        } else if label.starts_with("control/") {
            control_claims.push(entry);
        } else {
            other_claims.push(entry);
        }
    }

    let print_group = |name: &str, claims: &[(String, usize)], aggregate: bool| {
        if claims.is_empty() {
            return;
        }
        println!("  {name} ({} claims):", claims.len());
        if aggregate {
            // Aggregate by label, show count and degree range.
            let mut label_counts: Vec<(String, usize, usize, usize)> = Vec::new();
            for (label, deg) in claims {
                if let Some(entry) = label_counts.iter_mut().find(|(l, _, _, _)| l == label) {
                    entry.1 += 1;
                    entry.2 = entry.2.min(*deg);
                    entry.3 = entry.3.max(*deg);
                } else {
                    label_counts.push((label.clone(), 1, *deg, *deg));
                }
            }
            for (label, count, deg_min, deg_max) in &label_counts {
                if deg_min == deg_max {
                    println!("    - {label:<40} x{count:<4} degree_bound={deg_min}");
                } else {
                    println!("    - {label:<40} x{count:<4} degree_bound={deg_min}..{deg_max}");
                }
            }
        } else {
            for (label, deg) in claims {
                println!("    - {label:<40} degree_bound={deg}");
            }
        }
    };

    print_group("CCS (main constraint satisfaction)", &ccs_claims, false);
    print_group("Shout (lookup argument)", &shout_claims, true);
    print_group("Twist (memory argument)", &twist_claims, true);
    print_group("WB/WP (booleanity + quiescence)", &wb_wp_claims, false);
    print_group("Decode stage (lookup-backed decode)", &decode_claims, false);
    print_group("Width stage (lookup-backed width)", &width_claims, false);
    print_group("Control stage (branch/jump/writeback)", &control_claims, false);
    print_group("Other", &other_claims, false);
    println!();

    // ── 4. Opening Surface ──
    println!("4. OPENING SURFACE");
    println!("{thin_sep}");
    println!("  Core CCS:                {}", openings.core_ccs);
    println!("  Sidecars:                {}", openings.sidecars);
    println!("  Claim reduction/linkage: {}", openings.claim_reduction_linkage);
    println!("  PCS open:                {}", openings.pcs_open);
    println!("  Total:                   {}", openings.total());
    println!();

    // ── 5. Fold Lanes ──
    println!("5. FOLD LANES");
    println!("{thin_sep}");
    println!("  Main fold (ccs_out):     {} ME claims", step_proof.fold.ccs_out.len());
    println!(
        "  Main fold (dec children):{} DEC children",
        step_proof.fold.dec_children.len()
    );
    let val_count: usize = step_proof
        .val_fold
        .iter()
        .map(|v| v.dec_children.len())
        .sum();
    println!(
        "  Val fold lanes:          {} (dec children={})",
        step_proof.val_fold.len(),
        val_count
    );
    let wb_count: usize = step_proof
        .wb_fold
        .iter()
        .map(|w| w.dec_children.len())
        .sum();
    println!(
        "  WB fold lanes:           {} (dec children={})",
        step_proof.wb_fold.len(),
        wb_count
    );
    let wp_count: usize = step_proof
        .wp_fold
        .iter()
        .map(|w| w.dec_children.len())
        .sum();
    println!(
        "  WP fold lanes:           {} (dec children={})",
        step_proof.wp_fold.len(),
        wp_count
    );
    let stage8_count: usize = step_proof
        .stage8_fold
        .iter()
        .map(|w| w.dec_children.len())
        .sum();
    println!(
        "  Stage-8 fold lanes:      {} (dec children={})",
        step_proof.stage8_fold.len(),
        stage8_count
    );
    println!();

    // ── 6. ME Claims (Sidecar Proofs) ──
    println!("6. MEMORY SIDECAR ME CLAIMS");
    println!("{thin_sep}");
    let mem = &step_proof.mem;
    println!("  Val ME @ r_val:          {} claims", mem.val_me_claims.len());
    println!("  WB ME claims:            {} claims", mem.wb_me_claims.len());
    println!("  WP ME claims:            {} claims", mem.wp_me_claims.len());
    let (open_src_derived, open_src_ccs_tail, open_src_wb_tail, open_src_wp_tail, open_src_committed, open_src_virtual) =
        step_proof.fold.openings.iter().fold(
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize),
            |(d, c, wb, wp, co, vr), opening| match format!("{:?}", opening.source).as_str() {
                "TimeColumnsDerived" => (d + 1, c, wb, wp, co, vr),
                "CcsTail" => (d, c + 1, wb, wp, co, vr),
                "WbMeTail" => (d, c, wb + 1, wp, co, vr),
                "WpMeTail" => (d, c, wb, wp + 1, co, vr),
                "CommittedOpening" => (d, c, wb, wp, co + 1, vr),
                "VirtualReducedOpening" => (d, c, wb, wp, co, vr + 1),
                _ => (d, c, wb, wp, co, vr),
            },
        );
    println!(
        "  Time opening sources:    derived={} ccs_tail={} wb_tail={} wp_tail={} committed={} virtual_reduced={}",
        open_src_derived, open_src_ccs_tail, open_src_wb_tail, open_src_wp_tail, open_src_committed, open_src_virtual
    );
    println!();

    // ── 7. Used Sets ──
    println!("7. USED SETS (dynamic instantiation)");
    println!("{thin_sep}");
    println!("  Memory IDs (S_memory):   {:?}", run.used_memory_ids());
    println!("  Shout table IDs (S_lookup): {:?}", run.used_shout_table_ids());
    println!();

    // ── 8. Timing ──
    println!("8. TIMING");
    println!("{thin_sep}");
    println!("  Prove:                   {}", fmt_duration(prove_time));
    println!("  Verify:                  {}", fmt_duration(verify_time));
    println!("  Total end-to-end:        {}", fmt_duration(total_time));
    let phases = run.prove_phase_durations();
    println!("  Phase: setup             {}", fmt_duration(phases.setup));
    println!("  Phase: chunk commit      {}", fmt_duration(phases.chunk_build_commit));
    println!("  Phase: fold+prove        {}", fmt_duration(phases.fold_and_prove));
    println!();

    // ── 9. Summary ──
    let instr_count = n as f64;
    let prove_ms_per_instr = prove_time.as_secs_f64() * 1000.0 / instr_count;
    let verify_ms_per_instr = verify_time.as_secs_f64() * 1000.0 / instr_count;
    let fold_ms_per_instr = phases.fold_and_prove.as_secs_f64() * 1000.0 / instr_count;
    let total_ms_per_instr = total_time.as_secs_f64() * 1000.0 / instr_count;
    println!("9. SUMMARY");
    println!("{sep}");
    println!("  {:<36} {:>10}", "Instruction count (non-halt)", n);
    println!("  {:<36} {:>10}", "Trace steps (incl halt)", steps);
    println!("  {:<36} {:>9.3} ms", "Prove time / instruction", prove_ms_per_instr);
    println!(
        "  {:<36} {:>9.3} ms",
        "Fold+prove time / instruction", fold_ms_per_instr
    );
    println!("  {:<36} {:>9.3} ms", "Verify time / instruction", verify_ms_per_instr);
    println!("  {:<36} {:>9.3} ms", "Total time / instruction", total_ms_per_instr);
    println!("  {:<36} {:>10}", "Main trace columns", layout.trace.cols);
    println!("  {:<36} {:>10}", "CPU base cols (m_in+trace)", cpu_base_m);
    println!("  {:<36} {:>10}", "Committed mem/bus cols", committed_mem_bus_cols);
    println!("  {:<36} {:>10}", "Uniform CCS width proxy", total_ccs_m_uniform);
    println!(
        "  {:<36} {:>10}",
        "Bus-tail columns (legacy physical)", bus_tail_cols_physical
    );
    println!("  {:<36} {:>10}", "Core CCS rows", core_ccs.n);
    println!("  {:<36} {:>10}", "Bus reserved rows", bus_reserved_rows);
    println!("  {:<36} {:>10}", "Total CCS rows (n)", total_ccs_n);
    println!("  {:<36} {:>10}", "Total CCS cols (m, physical)", total_ccs_m_physical);
    println!("  {:<36} {:>10}", "Route-A batched claims", bt.claimed_sums.len());
    println!("  {:<36} {:>10}", "  of which: CCS", ccs_claims.len());
    println!("  {:<36} {:>10}", "  of which: Shout", shout_claims.len());
    println!("  {:<36} {:>10}", "  of which: Twist", twist_claims.len());
    println!("  {:<36} {:>10}", "  of which: WB/WP", wb_wp_claims.len());
    println!("  {:<36} {:>10}", "  of which: Decode", decode_claims.len());
    println!("  {:<36} {:>10}", "  of which: Width", width_claims.len());
    println!("  {:<36} {:>10}", "  of which: Control", control_claims.len());
    println!("  {:<36} {:>10}", "Commit lanes", 1);
    println!("  {:<36} {:>10}", "Stage-8 fold lanes", step_proof.stage8_fold.len());
    let committed_sidecars = step_proof
        .fold
        .openings
        .iter()
        .filter(|o| format!("{:?}", o.source).as_str() == "CommittedOpening")
        .count();
    println!("  {:<36} {:>10}", "Committed sidecars", committed_sidecars);
    println!("{sep}");
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_vs_chunked_single_n_mixed_ops"]
fn debug_trace_vs_chunked_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);

    assert!(chunk_rows > 0);
    let base = mixed_instruction_sequence();
    assert_eq!(base.len(), 10);

    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let chunk_total_start = Instant::now();
    let mut chunk_run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .chunk_rows(steps)
        .max_steps(steps)
        .prove()
        .expect("trace single-chunk prove (mixed)");
    let chunk_prove = chunk_run.prove_duration();
    let chunk_phases = chunk_run.prove_phase_durations();
    chunk_run
        .verify()
        .expect("trace single-chunk verify (mixed)");
    let chunk_verify = chunk_run
        .verify_duration()
        .expect("trace single-chunk verify duration");
    let chunk_total = chunk_total_start.elapsed();

    let trace_total_start = Instant::now();
    let trace_res = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove();
    match trace_res {
        Ok(mut trace_run) => {
            let trace_prove = trace_run.prove_duration();
            trace_run.verify().expect("trace verify (mixed)");
            let trace_verify = trace_run.verify_duration().expect("trace verify duration");
            let trace_total = trace_total_start.elapsed();
            let trace_phases = trace_run.prove_phase_durations();
            println!(
                "MIXED n={} TRACE(prove={}, verify={}, total={}, n_p2={}, m_p2={}, phases: setup={}, chunk_commit={}, fold={}) TRACE_SINGLE_CHUNK(prove={}, verify={}, total={}, n_p2={}, m_p2={}, phases: setup={}, chunk_commit={}, fold={}) ratio_prove={:.2}x",
                n,
                fmt_duration(trace_prove),
                fmt_duration(trace_verify),
                fmt_duration(trace_total),
                trace_run.ccs_num_constraints().next_power_of_two(),
                trace_run.ccs_num_variables().next_power_of_two(),
                fmt_duration(trace_phases.setup),
                fmt_duration(trace_phases.chunk_build_commit),
                fmt_duration(trace_phases.fold_and_prove),
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
                fmt_duration(chunk_phases.setup),
                fmt_duration(chunk_phases.chunk_build_commit),
                fmt_duration(chunk_phases.fold_and_prove),
                trace_prove.as_secs_f64() / chunk_prove.as_secs_f64(),
            );
        }
        Err(e) => {
            println!(
                "MIXED n={} TRACE(prove=ERROR:{}) TRACE_SINGLE_CHUNK(prove={}, verify={}, total={}, n_p2={}, m_p2={})",
                n,
                e,
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PerfSample {
    end_to_end: Duration,
    prove: Duration,
    verify: Duration,
    setup: Duration,
    build_commit: Duration,
    fold: Duration,
}

fn median_duration(values: &[Duration]) -> Duration {
    let mut nanos: Vec<u128> = values.iter().map(|d| d.as_nanos()).collect();
    nanos.sort_unstable();
    Duration::from_nanos(nanos[nanos.len() / 2] as u64)
}

fn spread_pct(values: &[Duration], median: Duration) -> f64 {
    if values.is_empty() || median.is_zero() {
        return 0.0;
    }
    let med = median.as_secs_f64();
    let max_abs = values
        .iter()
        .map(|v| (v.as_secs_f64() - med).abs())
        .fold(0.0f64, f64::max);
    (max_abs / med) * 100.0
}

fn build_mixed_program(n: usize) -> Vec<RiscvInstruction> {
    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    program
}

fn run_trace_sample(program: &[RiscvInstruction]) -> PerfSample {
    let steps = program.len();
    let program_bytes = encode_program(program);
    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(steps)
        .prove()
        .expect("trace prove");
    let prove = run.prove_duration();
    let phases = run.prove_phase_durations();
    run.verify().expect("trace verify");
    let verify = run.verify_duration().expect("trace verify duration");
    PerfSample {
        end_to_end: total_start.elapsed(),
        prove,
        verify,
        setup: phases.setup,
        build_commit: phases.chunk_build_commit,
        fold: phases.fold_and_prove,
    }
}

fn run_single_chunk_trace_sample(program: &[RiscvInstruction]) -> PerfSample {
    let steps = program.len();
    let program_bytes = encode_program(program);
    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .chunk_rows(steps)
        .max_steps(steps)
        .prove()
        .expect("trace single-chunk prove");
    let prove = run.prove_duration();
    let phases = run.prove_phase_durations();
    run.verify().expect("trace single-chunk verify");
    let verify = run
        .verify_duration()
        .expect("trace single-chunk verify duration");
    PerfSample {
        end_to_end: total_start.elapsed(),
        prove,
        verify,
        setup: phases.setup,
        build_commit: phases.chunk_build_commit,
        fold: phases.fold_and_prove,
    }
}

fn report_samples(label: &str, samples: &[PerfSample]) {
    let end_vals: Vec<Duration> = samples.iter().map(|s| s.end_to_end).collect();
    let prove_vals: Vec<Duration> = samples.iter().map(|s| s.prove).collect();
    let verify_vals: Vec<Duration> = samples.iter().map(|s| s.verify).collect();
    let setup_vals: Vec<Duration> = samples.iter().map(|s| s.setup).collect();
    let build_vals: Vec<Duration> = samples.iter().map(|s| s.build_commit).collect();
    let fold_vals: Vec<Duration> = samples.iter().map(|s| s.fold).collect();
    let prove_window_vals: Vec<Duration> = samples
        .iter()
        .map(|s| s.setup + s.build_commit + s.fold)
        .collect();

    let end_med = median_duration(&end_vals);
    let prove_med = median_duration(&prove_vals);
    let verify_med = median_duration(&verify_vals);
    let setup_med = median_duration(&setup_vals);
    let build_med = median_duration(&build_vals);
    let fold_med = median_duration(&fold_vals);
    let prove_window_med = median_duration(&prove_window_vals);

    println!(
        "{}: median(end={}, prove_api={}, prove_window={}, verify={}, setup={}, build_commit={}, fold={}) spread(end={:.2}%, prove_window={:.2}%, fold={:.2}%)",
        label,
        fmt_duration(end_med),
        fmt_duration(prove_med),
        fmt_duration(prove_window_med),
        fmt_duration(verify_med),
        fmt_duration(setup_med),
        fmt_duration(build_med),
        fmt_duration(fold_med),
        spread_pct(&end_vals, end_med),
        spread_pct(&prove_window_vals, prove_window_med),
        spread_pct(&fold_vals, fold_med),
    );
}

#[test]
#[ignore = "perf baseline report: cargo test -p neo-fold --release --test perf -- --ignored --nocapture report_trace_vs_chunked_medians"]
fn report_trace_vs_chunked_medians() {
    const RUNS: usize = 5;
    let cases = [
        ("mixed", 10usize, build_mixed_program(10)),
        ("mixed", 256usize, build_mixed_program(256)),
    ];

    for (kind, n, program) in cases {
        let mut trace_samples = Vec::with_capacity(RUNS);
        let mut chunked_samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            trace_samples.push(run_trace_sample(&program));
            chunked_samples.push(run_single_chunk_trace_sample(&program));
        }
        println!("CASE kind={} n={} runs={}", kind, n, RUNS);
        report_samples("TRACE", &trace_samples);
        report_samples("TRACE_SINGLE_CHUNK", &chunked_samples);
    }
}

#[test]
fn acceptance_uniform_ccs_width_independent_of_chunk_rows() {
    let program = build_mixed_program(64);
    let steps = program.len();
    let bytes = encode_program(&program);

    let mut run_small_chunks = Rv32TraceWiring::from_rom(0, &bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(8)
        .prove()
        .expect("prove with chunk_rows=8");
    run_small_chunks.verify().expect("verify with chunk_rows=8");

    let mut run_large_chunks = Rv32TraceWiring::from_rom(0, &bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(32)
        .prove()
        .expect("prove with chunk_rows=32");
    run_large_chunks
        .verify()
        .expect("verify with chunk_rows=32");

    assert_eq!(
        run_small_chunks.uniform_ccs_num_variables(),
        run_large_chunks.uniform_ccs_num_variables(),
        "uniform CCS acceptance: witness width should be independent of shard chunk size"
    );
    assert_eq!(
        run_small_chunks.ccs_num_variables(),
        run_small_chunks.uniform_ccs_num_variables(),
        "uniform CCS acceptance: physical ccs.m must equal uniform width proxy (chunk_rows=8)"
    );
    assert_eq!(
        run_large_chunks.ccs_num_variables(),
        run_large_chunks.uniform_ccs_num_variables(),
        "uniform CCS acceptance: physical ccs.m must equal uniform width proxy (chunk_rows=32)"
    );
}
