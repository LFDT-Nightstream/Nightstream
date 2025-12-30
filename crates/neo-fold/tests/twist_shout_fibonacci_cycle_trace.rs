#![allow(non_snake_case)]

//! # Twist+Shout Fibonacci “cycle trace” integration test
//!
//! This test is meant to be a **small, readable, end-to-end proving run** that still exercises the
//! full Twist-and-Shout Route-A integration inside Neo’s shard prover/verifier.
//!
//! ## What it proves
//! Per VM step, but *packed into a large folding chunk* (`CHUNK_SIZE` lanes per folding step):
//! - **Fibonacci transition**: `(f_curr, f_next) -> (f_next, f_curr + f_next)`.
//! - **Shout is active** every step: we do a lookup into a tiny public table `[0, 1]` at key `1`.
//!   The CCS uses `shout_val` multiplicatively in the Fibonacci constraint so the lookup is not a
//!   dead artifact; the Route-A Shout proof enforces the value is consistent with the table.
//! - **Twist is active** every step: we read `mem[0]` (expected to equal `f_next`) and then write
//!   the new `f_next` back to `mem[0]`. Route-A Twist proves address/bitness/time constraints and
//!   (via its val-eval lane) produces `ME(...)` claims that get folded in the shard.
//!
//! ## Why this answers the “show me the whole proving cycle” inquiry
//! The goal is to have a test where you can see, per cycle:
//! - what’s *stored* (CPU witness + shared-bus tail),
//! - what’s *computed* (CCS sumcheck, Route-A batched time sumcheck, memory sidecar subproofs),
//! - how many **ME / MLE-related claims** are produced and folded.
//!
//! Concretely:
//! - We call `fold_shard_prove_with_witnesses(...)` so the test has access to the final
//!   `ShardFoldOutputs` (obligations) and the corresponding witnesses, not just the proof.
//! - When `NEO_FIB_TRACE=1` is set, we print a per-step breakdown that includes:
//!   - `step_proof.fold.ccs_out.len()` (how many CCS `ME(...)` instances were output that step),
//!   - `step_proof.batched_time.*` (how many Route-A time/oracle claims were batched + rounds),
//!   - `step_proof.mem.*` (Twist/Shout proof metadata; CPU ME-at-`r_val` claims for Twist),
//!   - whether a val-lane fold happened (`step_proof.val_fold`).
//!
//! Run:
//! - Default (Ajtai commitment + real mixers; **production-like**):
//!   `NEO_FIB_TRACE=1 cargo test -p neo-fold --release twist_shout_fibonacci_cycle_trace -- --nocapture`
//! - Scale the workload (more fold steps) + print coarse timings:
//!   `NEO_FIB_CHUNKS=8 NEO_FIB_TIME=1 cargo test -p neo-fold --release twist_shout_fibonacci_cycle_trace -- --nocapture`
//!
//! ## Output binding
//! This test also demonstrates **output binding**: we claim that the program output is the final
//! value stored in Twist memory `mem[0]`, and we attach an output-binding proof so the verifier
//! checks that claim (not just that the execution is internally consistent).

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::matrix::Mat;
use neo_ccs::relations::CcsStructure;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove_with_output_binding, fold_shard_verify_with_output_binding, MemOrLutProof,
};
use neo_fold::shard::CommitMixers;
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::builder::build_shard_witness_shared_cpu_bus;
use neo_memory::cpu::{R1csCpu, SharedCpuBusConfig, ShoutCpuBinding, TwistCpuBinding};
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::witness::{LutTableSpec, StepInstanceBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_vm_trace::{Shout, ShoutId, StepMeta, Twist, TwistId, VmCpu};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use neo_fold::output_binding::OutputBindingConfig;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

const M_IN: usize = 1; // public constant-one column required for shared-bus guardrails
// We intentionally use a *large* chunk size here to better reflect “do more before folding” and to
// avoid the worst-case overheads of `chunk_size=1`.
const CHUNK_SIZE: usize = 32;
const DEFAULT_CHUNKS: usize = 2;

// Minimal geometry: 1 Shout + 1 Twist, both with `d=1, n_side=2 => ell=1 => ell_addr=1`.
const SHOUT_ELL_ADDR: usize = 1;
const TWIST_ELL_ADDR: usize = 1;

const BUS_COLS_SHOUT: usize = SHOUT_ELL_ADDR + 2;
const BUS_COLS_TWIST: usize = 2 * TWIST_ELL_ADDR + 5;
const BUS_COLS: usize = BUS_COLS_SHOUT + BUS_COLS_TWIST;
const BUS_REGION_LEN: usize = BUS_COLS * CHUNK_SIZE;

// Injected shared-bus constraints (exact counts from `neo_memory::cpu::constraints` when `inc=None`):
// - Shout: `5 + 2*ell_addr` constraints per lane
// - Twist: `11 + 4*ell_addr` constraints per lane
const BUS_CONSTRAINTS_PER_LANE: usize = (5 + 2 * SHOUT_ELL_ADDR) + (11 + 4 * TWIST_ELL_ADDR);
const BUS_CONSTRAINTS: usize = BUS_CONSTRAINTS_PER_LANE * CHUNK_SIZE;

// CPU constraints (this file): 4 per lane + 2 continuity constraints per link (lane j -> j+1).
const CPU_CONSTRAINTS: usize = 4 * CHUNK_SIZE + 2 * (CHUNK_SIZE - 1);
const CONSTRAINT_ROWS: usize = BUS_CONSTRAINTS + CPU_CONSTRAINTS;

// CPU witness columns (this file): 13 per-lane signals + 1 global const-one column.
const CPU_GROUPS: usize = 13;
const CPU_USED_COLS: usize = 1 + CPU_GROUPS * CHUNK_SIZE;

const USED_COLS: usize = CPU_USED_COLS + BUS_REGION_LEN;

const fn max_usize(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

// Square CCS is required for identity-first Route-A shared-bus semantics, so set `m=n` as the max
// of (witness width) and (row count needed by CPU + injected bus constraints).
const M: usize = max_usize(USED_COLS, CONSTRAINT_ROWS);
const BUS_BASE: usize = M - BUS_REGION_LEN;

// Witness column layout (prefix of z; the shared-bus tail is appended automatically by `R1csCpu`).
//
// Think of this prefix as the “CPU-local” part of the witness. The shared CPU-bus constraints
// then *bind* some of these columns to the canonical Twist/Shout bus columns (which Route A
// consumes from the CPU commitment).
const COL_CONST_ONE: usize = 0;
const COL_F_CURR_BEFORE_BASE: usize = 1;
const COL_F_NEXT_BEFORE_BASE: usize = COL_F_CURR_BEFORE_BASE + CHUNK_SIZE;
const COL_F_CURR_AFTER_BASE: usize = COL_F_NEXT_BEFORE_BASE + CHUNK_SIZE;
const COL_F_NEXT_AFTER_BASE: usize = COL_F_CURR_AFTER_BASE + CHUNK_SIZE;

const COL_MEM_HAS_READ_BASE: usize = COL_F_NEXT_AFTER_BASE + CHUNK_SIZE;
const COL_MEM_HAS_WRITE_BASE: usize = COL_MEM_HAS_READ_BASE + CHUNK_SIZE;
const COL_MEM_READ_ADDR_BASE: usize = COL_MEM_HAS_WRITE_BASE + CHUNK_SIZE;
const COL_MEM_WRITE_ADDR_BASE: usize = COL_MEM_READ_ADDR_BASE + CHUNK_SIZE;
const COL_MEM_RV_BASE: usize = COL_MEM_WRITE_ADDR_BASE + CHUNK_SIZE;
const COL_MEM_WV_BASE: usize = COL_MEM_RV_BASE + CHUNK_SIZE;

const COL_SHOUT_HAS_LOOKUP_BASE: usize = COL_MEM_WV_BASE + CHUNK_SIZE;
const COL_SHOUT_ADDR_BASE: usize = COL_SHOUT_HAS_LOOKUP_BASE + CHUNK_SIZE;
const COL_SHOUT_VAL_BASE: usize = COL_SHOUT_ADDR_BASE + CHUNK_SIZE;

const CPU_USED_END: usize = COL_SHOUT_VAL_BASE + CHUNK_SIZE;

#[derive(Clone, Default)]
struct MapTwist {
    mem: HashMap<(TwistId, u64), u64>,
}

impl Twist<u64, u64> for MapTwist {
    fn load(&mut self, id: TwistId, addr: u64) -> u64 {
        *self.mem.get(&(id, addr)).unwrap_or(&0)
    }

    fn store(&mut self, id: TwistId, addr: u64, val: u64) {
        self.mem.insert((id, addr), val);
    }
}

#[derive(Clone)]
struct MapShout {
    table: Vec<u64>,
}

impl Shout<u64> for MapShout {
    fn lookup(&mut self, id: ShoutId, key: u64) -> u64 {
        assert_eq!(id.0, 0, "this test only supports shout_id=0");
        self.table.get(key as usize).copied().unwrap_or(0)
    }
}

/// Fibonacci VM that (a) updates `(f_curr, f_next)` and (b) drives Twist+Shout every step.
///
/// - Shout: lookup table 0 at key=1 (value=1) to bind Shout into the CCS.
/// - Twist: RW at (twist_id=0, addr=0): read current `f_next`, then write `f_curr + f_next`.
struct FibTwistShoutVm {
    f_curr: u64,
    f_next: u64,
    pc: u64,
    step: u64,
    max_steps: u64,
    halted: bool,
    q: u64,
}

impl FibTwistShoutVm {
    fn new(max_steps: u64, q: u64) -> Self {
        Self {
            f_curr: 0,
            f_next: 1,
            pc: 0,
            step: 0,
            max_steps,
            halted: false,
            q,
        }
    }
}

impl VmCpu<u64, u64> for FibTwistShoutVm {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.f_curr, self.f_next]
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<TW, SH>(&mut self, twist: &mut TW, shout: &mut SH) -> Result<StepMeta<u64>, Self::Error>
    where
        TW: Twist<u64, u64>,
        SH: Shout<u64>,
    {
        let _one = shout.lookup(ShoutId(0), 1);

        let mem_id = TwistId(0);
        let mem_next = twist.load(mem_id, 0);
        if mem_next != self.f_next {
            return Err(format!(
                "memory/state mismatch before step {}: f_next(reg)={} vs mem[0]={}",
                self.step, self.f_next, mem_next
            ));
        }

        // IMPORTANT: keep the VM's arithmetic consistent with Goldilocks field arithmetic by
        // computing Fibonacci mod q (not mod 2^64). This allows arbitrarily many steps without
        // overflow mismatches between u64 "register" values and field elements in the CCS.
        let f_new = add_mod_q(self.f_curr, self.f_next, self.q);
        self.f_curr = self.f_next;
        self.f_next = f_new;
        twist.store(mem_id, 0, self.f_next);

        self.step += 1;
        self.pc += 4;
        if self.step >= self.max_steps {
            self.halted = true;
        }

        Ok(StepMeta {
            pc_after: self.pc,
            opcode: 0xF1B0, // arbitrary "FIB" opcode for trace readability
        })
    }
}

fn add_mod_q(a: u64, b: u64, q: u64) -> u64 {
    debug_assert!(a < q);
    debug_assert!(b < q);
    let sum = (a as u128) + (b as u128);
    let q128 = q as u128;
    let reduced = if sum >= q128 { sum - q128 } else { sum };
    // a,b<q => sum<2q, so one subtraction is enough.
    reduced as u64
}

fn fib_mod_q_u64(n: usize, q: u64) -> u64 {
    // F_0=0, F_1=1, computed mod q.
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let c = add_mod_q(a, b, q);
        a = b;
        b = c;
    }
    a
}

fn build_fib_ccs(m: usize) -> CcsStructure<F> {
    let n = m;
    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    // We write per-lane constraints in the first `CPU_CONSTRAINTS` rows, and leave the remainder
    // of the matrix rows empty for the injected shared-bus constraints.
    let mut row = 0usize;

    // Per-lane semantics.
    for j in 0..CHUNK_SIZE {
        let f_curr_before = COL_F_CURR_BEFORE_BASE + j;
        let f_next_before = COL_F_NEXT_BEFORE_BASE + j;
        let f_curr_after = COL_F_CURR_AFTER_BASE + j;
        let f_next_after = COL_F_NEXT_AFTER_BASE + j;
        let mem_rv = COL_MEM_RV_BASE + j;
        let mem_wv = COL_MEM_WV_BASE + j;
        let shout_val = COL_SHOUT_VAL_BASE + j;

        // (f_curr_before + f_next_before) * shout_val = f_next_after
        //
        // We intentionally multiply by `shout_val` so Shout is a *required* part of satisfying the
        // CPU relation. In this test `shout_val == 1` always (table=[0,1], key=1).
        A[(row, f_curr_before)] = F::ONE;
        A[(row, f_next_before)] = F::ONE;
        B[(row, shout_val)] = F::ONE;
        C[(row, f_next_after)] = F::ONE;
        row += 1;

        // (f_curr_after - f_next_before) * 1 = 0
        A[(row, f_curr_after)] = F::ONE;
        A[(row, f_next_before)] = F::ZERO - F::ONE;
        B[(row, COL_CONST_ONE)] = F::ONE;
        row += 1;

        // (f_next_before - mem_rv) * 1 = 0
        A[(row, f_next_before)] = F::ONE;
        A[(row, mem_rv)] = F::ZERO - F::ONE;
        B[(row, COL_CONST_ONE)] = F::ONE;
        row += 1;

        // (f_next_after - mem_wv) * 1 = 0
        A[(row, f_next_after)] = F::ONE;
        A[(row, mem_wv)] = F::ZERO - F::ONE;
        B[(row, COL_CONST_ONE)] = F::ONE;
        row += 1;
    }

    // Intra-chunk continuity: the VM state after lane j becomes the "before" state for lane j+1.
    for j in 0..(CHUNK_SIZE - 1) {
        let f_curr_after = COL_F_CURR_AFTER_BASE + j;
        let f_next_after = COL_F_NEXT_AFTER_BASE + j;
        let f_curr_before_next = COL_F_CURR_BEFORE_BASE + (j + 1);
        let f_next_before_next = COL_F_NEXT_BEFORE_BASE + (j + 1);

        // (f_curr_before[j+1] - f_curr_after[j]) * 1 = 0
        A[(row, f_curr_before_next)] = F::ONE;
        A[(row, f_curr_after)] = F::ZERO - F::ONE;
        B[(row, COL_CONST_ONE)] = F::ONE;
        row += 1;

        // (f_next_before[j+1] - f_next_after[j]) * 1 = 0
        A[(row, f_next_before_next)] = F::ONE;
        A[(row, f_next_after)] = F::ZERO - F::ONE;
        B[(row, COL_CONST_ONE)] = F::ONE;
        row += 1;
    }

    assert_eq!(
        row, CPU_CONSTRAINTS,
        "CPU constraint row count mismatch: wrote {}, expected {}",
        row, CPU_CONSTRAINTS
    );

    neo_ccs::r1cs_to_ccs(A, B, C)
}

fn dump_enabled() -> bool {
    // Run with:
    //   NEO_FIB_TRACE=1 cargo test -p neo-fold --release twist_shout_fibonacci_cycle_trace -- --nocapture
    std::env::var("NEO_FIB_TRACE").is_ok()
}

fn dump_full_lanes() -> bool {
    matches!(
        std::env::var("NEO_FIB_TRACE").as_deref(),
        Ok("full") | Ok("FULL")
    )
}

fn timing_enabled() -> bool {
    std::env::var("NEO_FIB_TIME").is_ok() || dump_enabled()
}

fn read_usize_env(key: &str) -> Option<usize> {
    std::env::var(key).ok().map(|v| {
        v.parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid {key}={v:?} (expected usize)"))
    })
}

fn setup_ajtai_committer(m: usize, kappa: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, kappa, m).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn ajtai_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    Mixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

#[test]
fn twist_shout_fibonacci_cycle_trace() {
    // Scaling knobs (keep defaults tiny so CI is fast):
    // - `NEO_FIB_CHUNKS=<n>` controls how many folding steps we run (each is `CHUNK_SIZE` lanes).
    // - `NEO_FIB_STEPS=<t>` overrides chunks and directly sets VM steps (must be multiple of CHUNK_SIZE).
    let n_steps = read_usize_env("NEO_FIB_STEPS").unwrap_or_else(|| {
        let n_chunks = read_usize_env("NEO_FIB_CHUNKS").unwrap_or(DEFAULT_CHUNKS);
        n_chunks * CHUNK_SIZE
    });
    assert_eq!(
        n_steps % CHUNK_SIZE,
        0,
        "NEO_FIB_STEPS must be a multiple of CHUNK_SIZE={CHUNK_SIZE} (got {n_steps})"
    );
    let max_steps = n_steps;

    assert!(
        CPU_USED_END <= BUS_BASE,
        "CPU witness layout overlaps bus tail: cpu_end={} bus_base={}",
        CPU_USED_END,
        BUS_BASE
    );

    // Params:
    // - bump k_rho for comfortable Π_RLC norm bound margin in tests
    // - use b=4 so Ajtai digit encoding can represent full Goldilocks values (b^d >> q),
    //   which matters once you run many chunks/steps (values quickly leave the tiny b=2^54 range).
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(M).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        4,  // b
        16, // k_rho
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");

    // One Twist memory (id=0), one Shout table (id=0).
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let mut initial_mem = HashMap::new();
    initial_mem.insert((0u32, 0u64), F::ONE);

    let mut lut_tables = HashMap::new();
    lut_tables.insert(
        0u32,
        LutTable {
            table_id: 0,
            k: 2,
            d: 1,
            n_side: 2,
            content: vec![F::ZERO, F::ONE],
        },
    );
    let lut_table_specs: HashMap<u32, LutTableSpec> = HashMap::new();

    let shout_cpu = HashMap::from([(
        0u32,
        ShoutCpuBinding {
            has_lookup: COL_SHOUT_HAS_LOOKUP_BASE,
            addr: COL_SHOUT_ADDR_BASE,
            val: COL_SHOUT_VAL_BASE,
        },
    )]);
    let twist_cpu = HashMap::from([(
        0u32,
        TwistCpuBinding {
            has_read: COL_MEM_HAS_READ_BASE,
            has_write: COL_MEM_HAS_WRITE_BASE,
            read_addr: COL_MEM_READ_ADDR_BASE,
            write_addr: COL_MEM_WRITE_ADDR_BASE,
            rv: COL_MEM_RV_BASE,
            wv: COL_MEM_WV_BASE,
            inc: None, // bus computes inc; CPU doesn't need to carry it for this demo
        },
    )]);

    let shared_bus_cfg = SharedCpuBusConfig {
        mem_layouts: mem_layouts.clone(),
        initial_mem: initial_mem.clone(),
        const_one_col: COL_CONST_ONE,
        shout_cpu,
        twist_cpu,
    };

    let committer = setup_ajtai_committer(M, params.kappa as usize);
    let mixers = ajtai_mixers();

    let cpu_ccs = build_fib_ccs(M);
    let cpu_arith = R1csCpu::<F, Cmt, _>::new(
        cpu_ccs,
        params,
        committer,
        M_IN,
        &lut_tables,
        &lut_table_specs,
        Box::new(|chunk| {
            assert_eq!(
                chunk.len(),
                CHUNK_SIZE,
                "this test assumes full chunks (max_steps must be a multiple of chunk_size)"
            );

            let mut z = vec![F::ZERO; CPU_USED_END];
            z[COL_CONST_ONE] = F::ONE;

            for (j, step) in chunk.iter().enumerate() {
                let regs_before = step.regs_before.as_slice();
                let regs_after = step.regs_after.as_slice();
                assert_eq!(regs_before.len(), 2, "expected 2 regs for Fibonacci VM");
                assert_eq!(regs_after.len(), 2, "expected 2 regs for Fibonacci VM");

                let mut read: Option<(u64, u64)> = None;
                let mut write: Option<(u64, u64)> = None;
                for ev in &step.twist_events {
                    if ev.twist_id.0 != 0 {
                        continue;
                    }
                    match ev.kind {
                        neo_vm_trace::TwistOpKind::Read => read = Some((ev.addr, ev.value)),
                        neo_vm_trace::TwistOpKind::Write => write = Some((ev.addr, ev.value)),
                    }
                }
                let (read_addr, read_val) = read.expect("missing Twist read event");
                let (write_addr, write_val) = write.expect("missing Twist write event");

                let shout = step
                    .shout_events
                    .iter()
                    .find(|ev| ev.shout_id.0 == 0)
                    .cloned()
                    .expect("missing Shout event");

                z[COL_F_CURR_BEFORE_BASE + j] = F::from_u64(regs_before[0]);
                z[COL_F_NEXT_BEFORE_BASE + j] = F::from_u64(regs_before[1]);
                z[COL_F_CURR_AFTER_BASE + j] = F::from_u64(regs_after[0]);
                z[COL_F_NEXT_AFTER_BASE + j] = F::from_u64(regs_after[1]);

                z[COL_MEM_HAS_READ_BASE + j] = F::ONE;
                z[COL_MEM_HAS_WRITE_BASE + j] = F::ONE;
                z[COL_MEM_READ_ADDR_BASE + j] = F::from_u64(read_addr);
                z[COL_MEM_WRITE_ADDR_BASE + j] = F::from_u64(write_addr);
                z[COL_MEM_RV_BASE + j] = F::from_u64(read_val);
                z[COL_MEM_WV_BASE + j] = F::from_u64(write_val);

                z[COL_SHOUT_HAS_LOOKUP_BASE + j] = F::ONE;
                z[COL_SHOUT_ADDR_BASE + j] = F::from_u64(shout.key);
                z[COL_SHOUT_VAL_BASE + j] = F::from_u64(shout.value);
            }

            z
        }),
    )
    .with_shared_cpu_bus(shared_bus_cfg, CHUNK_SIZE)
    .expect("R1csCpu shared-bus config should succeed");

    // Run VM + build shard witnesses.
    let mut twist = MapTwist::default();
    twist.store(TwistId(0), 0, 1);
    let shout = MapShout { table: vec![0, 1] };

    let t_witness = Instant::now();
    let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
        FibTwistShoutVm::new(max_steps as u64, base_params.q),
        twist,
        shout,
        max_steps,
        CHUNK_SIZE,
        &mem_layouts,
        &lut_tables,
        &lut_table_specs,
        &initial_mem,
        &cpu_arith,
    )
    .expect("build_shard_witness_shared_cpu_bus should succeed");
    let witness_dur = t_witness.elapsed();
    assert_eq!(steps.len(), n_steps / CHUNK_SIZE);

    // Sanity: final Fibonacci value.
    //
    // NOTE: We sanity-check two ways:
    //  1) simulate the VM locally (u64 arithmetic mod q),
    //  2) decode the final committed witness and check the last lane's `(f_next_after)`.
    //
    // If (2) fails but (1) passes, the bug is in witness layout / encoding, not in the VM.
    let expected_next = fib_mod_q_u64(n_steps + 1, base_params.q);
    {
        let mut vm = FibTwistShoutVm::new(max_steps as u64, base_params.q);
        let mut twist = MapTwist::default();
        twist.store(TwistId(0), 0, 1);
        let mut shout = MapShout { table: vec![0, 1] };
        for _ in 0..max_steps {
            vm.step(&mut twist, &mut shout).expect("VM step should succeed");
        }
        assert_eq!(vm.f_next, expected_next, "VM simulation mismatch");
    }
    let last_z = neo_memory::ajtai::decode_vector(&params, &steps.last().expect("non-empty").mcs.1.Z);
    assert_eq!(
        last_z[COL_F_NEXT_AFTER_BASE + (CHUNK_SIZE - 1)].as_canonical_u64(),
        expected_next,
        "final f_next should equal F_{}",
        n_steps + 1
    );

    // Prove+verify.
    let acc_init = Vec::new();
    let acc_wit_init = Vec::new();

    let t_prove = Instant::now();
    let mut tr_prove = Poseidon2Transcript::new(b"fib/twist_shout/prove");
    let output_addr = 0u64;
    let output_val = F::from_u64(expected_next);
    let ob_cfg = OutputBindingConfig::new(1, ProgramIO::new().with_output(output_addr, output_val));
    let final_memory_state = vec![output_val, F::ZERO]; // k=2, addr ∈ {0,1}

    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &cpu_arith.ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &cpu_arith.committer,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )
    .expect("prove should succeed");
    let prove_dur = t_prove.elapsed();
    assert!(proof.output_proof.is_some(), "expected output binding proof to be attached");

    let outputs = proof.compute_fold_outputs(&acc_init);

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();
    let t_verify = Instant::now();
    let mut tr_verify = Poseidon2Transcript::new(b"fib/twist_shout/prove");
    let outputs_v = fold_shard_verify_with_output_binding(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &cpu_arith.ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )
    .expect("verify should succeed");
    let verify_dur = t_verify.elapsed();
    assert_eq!(outputs_v.obligations.all_len(), outputs.obligations.all_len());
    assert!(
        proof.steps.len() >= 2,
        "this test is meant to demonstrate at least one fold (need >=2 folding steps)"
    );

    if timing_enabled() {
        println!(
            "\n[timing] chunks={} steps={} chunk_size={} witness_build={:?} prove={:?} verify={:?}",
            steps.len(),
            n_steps,
            CHUNK_SIZE,
            witness_dur,
            prove_dur,
            verify_dur
        );
    }

    // Optional: dump a detailed per-step proving trace.
    if dump_enabled() {
        // The bus layout is deterministic given:
        // - `m` (CCS width),
        // - `m_in` (public prefix length),
        // - `chunk_size`,
        // - per-instance address bit-lengths (`ell_addr = d * ell`).
        //
        // Here we have exactly one Shout instance and one Twist instance, both with `ell_addr=1`.
        let bus_layout = neo_memory::cpu::build_bus_layout_for_instances(
            M,
            M_IN,
            CHUNK_SIZE,
            core::iter::once(1usize), // shout ell_addr = d*ell = 1*1
            core::iter::once(1usize), // twist ell_addr = d*ell = 1*1
        )
        .expect("bus layout");

        println!("\n=== Twist+Shout Fibonacci proving trace ===");
        println!(
            "vm_steps={n_steps}  fold_steps={}  chunk_size={CHUNK_SIZE}  m={M}  m_in={M_IN}",
            steps.len()
        );
        println!(
            "commit: backend=AjtaiSModule  kappa={}",
            cpu_arith.committer.pp.kappa
        );
        println!(
            "bus: base={}  cols={}  region_len={}",
            bus_layout.bus_base,
            bus_layout.bus_cols,
            bus_layout.bus_region_len()
        );
        println!(
            "final: F_{}={}  main_obligations={}  val_obligations={}",
            n_steps + 1,
            expected_next,
            outputs.obligations.main.len(),
            outputs.obligations.val.len()
        );
        println!(
            "output_binding: addr={} expected={} num_bits={}",
            output_addr, output_val, ob_cfg.num_bits
        );

        for (i, (step_wit, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
            let z = neo_memory::ajtai::decode_vector(&params, &step_wit.mcs.1.Z);
            let f0 = z[COL_F_CURR_BEFORE_BASE].as_canonical_u64();
            let f1 = z[COL_F_NEXT_BEFORE_BASE].as_canonical_u64();
            let f0n = z[COL_F_CURR_AFTER_BASE + (CHUNK_SIZE - 1)].as_canonical_u64();
            let f1n = z[COL_F_NEXT_AFTER_BASE + (CHUNK_SIZE - 1)].as_canonical_u64();

            let mem_init = &step_wit.mem_instances[0].0.init;
            let mem_rv = z[COL_MEM_RV_BASE].as_canonical_u64();
            let mem_wv = z[COL_MEM_WV_BASE].as_canonical_u64();
            let shout_key = z[COL_SHOUT_ADDR_BASE].as_canonical_u64();
            let shout_val = z[COL_SHOUT_VAL_BASE].as_canonical_u64();

            println!("\n-- step {i} --");
            println!("state: ({f0}, {f1}) -> ({f0n}, {f1n})");
            println!("twist: init={mem_init:?}  lane0_read={mem_rv}  lane0_write={mem_wv}");
            println!("shout: lane0_key={shout_key}  lane0_val={shout_val}");

            let full = dump_full_lanes();
            let prefix = 4usize.min(CHUNK_SIZE);
            let suffix = if full { 0 } else { 2usize.min(CHUNK_SIZE.saturating_sub(prefix)) };
            if full {
                println!("lanes:");
            } else {
                println!("lanes (prefix {} + suffix {}):", prefix, suffix);
            }
            for j in 0..CHUNK_SIZE {
                let in_prefix = j < prefix;
                let in_suffix = j >= CHUNK_SIZE.saturating_sub(suffix);
                if !full && !in_prefix && !in_suffix {
                    continue;
                }
                let f0 = z[COL_F_CURR_BEFORE_BASE + j].as_canonical_u64();
                let f1 = z[COL_F_NEXT_BEFORE_BASE + j].as_canonical_u64();
                let f0n = z[COL_F_CURR_AFTER_BASE + j].as_canonical_u64();
                let f1n = z[COL_F_NEXT_AFTER_BASE + j].as_canonical_u64();
                let mem_rv = z[COL_MEM_RV_BASE + j].as_canonical_u64();
                let mem_wv = z[COL_MEM_WV_BASE + j].as_canonical_u64();
                let shout_key = z[COL_SHOUT_ADDR_BASE + j].as_canonical_u64();
                let shout_val = z[COL_SHOUT_VAL_BASE + j].as_canonical_u64();
                println!(
                    "  lane {j:>2}: ({f0:>8}, {f1:>8}) -> ({f0n:>8}, {f1n:>8}) | mem: {mem_rv:>8} -> {mem_wv:>8} | shout: ({shout_key} -> {shout_val})"
                );
            }

            println!(
                "ccs: out_me={}  ccs_sumcheck_rounds={}",
                step_proof.fold.ccs_out.len(),
                step_proof.fold.ccs_proof.sumcheck_rounds.len()
            );
            println!(
                "time_batch: claims={}  rounds_per_claim={}..{}",
                step_proof.batched_time.claimed_sums.len(),
                step_proof
                    .batched_time
                    .round_polys
                    .iter()
                    .map(|r| r.len())
                    .min()
                    .unwrap_or(0),
                step_proof
                    .batched_time
                    .round_polys
                    .iter()
                    .map(|r| r.len())
                    .max()
                    .unwrap_or(0)
            );

            println!(
                "mem_sidecar: cpu_me_claims_val={}  proofs={}",
                step_proof.mem.cpu_me_claims_val.len(),
                step_proof.mem.proofs.len()
            );
            
            println!(
                "shout_addr_pre: claimed_sums={} active_mask=0x{:x} rounds={} r_addr_len={}",
                step_proof.mem.shout_addr_pre.claimed_sums.len(),
                step_proof.mem.shout_addr_pre.active_mask,
                step_proof.mem.shout_addr_pre.round_polys.len(),
                step_proof.mem.shout_addr_pre.r_addr.len()
            );

            for (idx, p) in step_proof.mem.proofs.iter().enumerate() {
                match p {
                    MemOrLutProof::Shout(shout_pf) => {
                        println!(
                            "  proof[{idx}] Shout: addr_pre_claims={} (expected 0; batched in shout_addr_pre)",
                            shout_pf.addr_pre.claimed_sums.len(),
                        );
                    }
                    MemOrLutProof::Twist(twist_pf) => {
                        println!(
                            "  proof[{idx}] Twist: addr_pre_claims={} addr_rounds={} r_addr_len={}",
                            twist_pf.addr_pre.claimed_sums.len(),
                            twist_pf.addr_pre.round_polys.first().map(|v| v.len()).unwrap_or(0),
                            twist_pf.addr_pre.r_addr.len()
                        );
                        if let Some(val) = &twist_pf.val_eval {
                            println!(
                                "    val_eval: lt_rounds={} total_rounds={} prev_total_rounds={}",
                                val.rounds_lt.len(),
                                val.rounds_total.len(),
                                val.rounds_prev_total.as_ref().map(|v| v.len()).unwrap_or(0)
                            );
                        }
                    }
                }
            }

            if let Some(val_fold) = &step_proof.val_fold {
                println!(
                    "val_lane: rlc_rhos={} dec_children={}",
                    val_fold.rlc_rhos.len(),
                    val_fold.dec_children.len()
                );
            } else {
                println!("val_lane: <none>");
            }
        }
    }
}
