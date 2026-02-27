//! Compiled ROM coverage for the circuit_l2_transfer guest.
//!
//! Contains:
//! - A structural metrics benchmark (`#[ignore]`) that traces and proves without witness data.
//! - A real note-spend transfer test that constructs a valid 1-input / 1-output
//!   self-transfer witness (including Merkle path, nullifier, enforce-product,
//!   and blacklist non-membership proof), writes it to RAM, then proves + verifies.

#[path = "binaries/circuit_l2_transfer_rom.rs"]
mod circuit_l2_transfer_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvMemory, RiscvShoutTables, PROG_ID, RAM_ID};
use neo_vm_trace::{trace_program, Twist, TwistOpKind};
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
    // Poseidon lane split requires t_len <= (ccs_m - m_in).
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
    pub fn build_1in_1out_transfer() -> Vec<(u64, u32)> {
        let mut ram = RamWriter::new(INPUT_ADDR);

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

        // --- Merkle tree (depth=2, leaf at position 0) ---
        let depth: u32 = 2;
        let sib0 = ZERO_DIGEST;
        let node_left = {
            let mut inp = [Goldilocks::ZERO; 10];
            inp[0] = gl(TAG_MT_NODE);
            inp[1] = gl(0);
            inp[2..6].copy_from_slice(&cm_gl);
            inp[6..10].copy_from_slice(&sib0);
            h(&inp)
        };
        let sib1 = {
            let mut inp = [Goldilocks::ZERO; 10];
            inp[0] = gl(TAG_MT_NODE);
            inp[1] = gl(0);
            inp[2..6].copy_from_slice(&ZERO_DIGEST);
            inp[6..10].copy_from_slice(&ZERO_DIGEST);
            h(&inp)
        };
        let anchor_gl = {
            let mut inp = [Goldilocks::ZERO; 10];
            inp[0] = gl(TAG_MT_NODE);
            inp[1] = gl(1);
            inp[2..6].copy_from_slice(&node_left);
            inp[6..10].copy_from_slice(&sib1);
            h(&inp)
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
        ram.write_gl_digest(&sib0); // sibling level 0
        ram.write_gl_digest(&sib1); // sibling level 1

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

        ram.pairs
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

    let ram_pairs = witness_builder::build_1in_1out_transfer();
    println!("witness_ram_words={}", ram_pairs.len());

    // This ROM export currently includes only code bytes, so there is no separate
    // rodata blob to preload here.
    println!("total_ram_words={} (witness only)", ram_pairs.len());

    // --- Simulation pass: measure step count and verify circuit halts ---
    let executed_steps = {
        let decoded = decode_program(program_bytes).expect("decode ROM");
        let mut cpu = RiscvCpu::new(32);
        cpu.load_program(program_base, decoded);
        let mut twist = RiscvMemory::with_program_in_twist(32, PROG_ID, program_base, program_bytes);
        for &(addr, val) in &ram_pairs {
            twist.store(RAM_ID, addr, val as u64);
        }
        let shout = RiscvShoutTables::new(32);
        let sim = trace_program(cpu, twist, shout, 200_000).expect("simulation trace");
        println!("sim_steps={} sim_halted={}", sim.steps.len(), sim.did_halt());
        if !sim.did_halt() {
            panic!("circuit did not halt within 200K simulation steps");
        }
        sim.steps.len()
    };

    // Poseidon lane split requires t_len <= (ccs_m - m_in).
    // For this circuit/profile that cap is 510, so pick the largest safe chunk automatically.
    const MAX_SAFE_CHUNK_ROWS: usize = 510;
    let chunk_rows = executed_steps.min(MAX_SAFE_CHUNK_ROWS);
    let max_steps = executed_steps;
    let min_trace_len = executed_steps;

    let mut wiring = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .min_trace_len(min_trace_len)
        .chunk_rows(chunk_rows)
        .max_steps(max_steps)
        .shout_auto_minimal();

    for &(addr, val) in &ram_pairs {
        wiring = wiring.ram_init_u32(addr, val);
    }

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

    let phases = run.prove_phase_durations();
    println!("\n=============================================");
    println!("  Note-Spend 1-in/1-out Transfer (Nightstream)");
    println!("=============================================");
    println!("  RISC-V steps:     {}", run.trace_len());
    println!("  CCS constraints:  {}", run.ccs_num_constraints());
    println!("  CCS variables:    {}", run.ccs_num_variables());
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
    let verify_result = run.verify();
    let verify_ms = verify_start.elapsed().as_millis();
    println!("  Verify time:      {} ms", verify_ms);
    println!("=============================================\n");

    if let Err(err) = verify_result {
        panic!("verification failed: {err}");
    }
}
