//! Integration tests for Twist/Shout memory argument pipeline.
//!
//! These tests validate the end-to-end flow from VM tracing through to
//! cryptographic verification. They are marked `#[ignore]` until the
//! full implementation is complete.
//!
//! ## Test 1: twist_shout_trace_to_witness_smoke
//!
//! End-to-end test: trace → plain traces → encoding → semantic checks.
//! Uses a 4-step program with Shout (read-only table) and Twist (RW memory).
//!
//! ## Test 2: idx2oh_matches_naive_onehot_eval
//!
//! Tests that the index-bits approach matches naive one-hot MLE evaluation.
//! This validates the foundation for the IDX→OneHot virtual protocol.

#![allow(non_snake_case)]

use std::collections::HashMap;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, poly::SparsePoly, relations::CcsStructure, traits::SModuleHomomorphism};
use neo_math::K as KElem;
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::mle::build_chi_table;
use neo_memory::plain::{build_plain_lut_traces, build_plain_mem_traces, LutTable, PlainMemLayout};
use neo_memory::{shout, twist};
use neo_params::NeoParams;
use neo_reductions::api;
use neo_reductions::api::FoldingMode;
use neo_vm_trace::{trace_program, Shout, ShoutId, StepMeta, Twist, TwistId, VmCpu, VmTrace};

// ============================================================================
// Test 1: End-to-end trace → plain traces → encoding → semantic checks
// ============================================================================

/// A scripted CPU that executes a specific 4-step program for testing.
///
/// Step script:
/// - j=0: read mem[1] -> 0, lookup table[2] -> 30, write mem[1] = 30
/// - j=1: read mem[1] -> 30
/// - j=2: lookup table[3] -> 40, write mem[1] = 40
/// - j=3: read mem[1] -> 40, lookup table[0] -> 10, write mem[0] = 10
struct ScriptCpu {
    pc: u64,
    step: usize,
    halted: bool,
}

impl ScriptCpu {
    fn new() -> Self {
        Self {
            pc: 0,
            step: 0,
            halted: false,
        }
    }
}

impl VmCpu<u64, u64> for ScriptCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.step as u64]
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<TW, SH>(&mut self, twist_mem: &mut TW, shout_tbl: &mut SH) -> Result<StepMeta<u64>, Self::Error>
    where
        TW: Twist<u64, u64>,
        SH: Shout<u64>,
    {
        let ram = TwistId(1);
        let tbl = ShoutId(0);

        match self.step {
            0 => {
                let _ = twist_mem.load(ram, 1);
                let v = shout_tbl.lookup(tbl, 2);
                twist_mem.store(ram, 1, v);
            }
            1 => {
                let _ = twist_mem.load(ram, 1);
            }
            2 => {
                let v = shout_tbl.lookup(tbl, 3);
                twist_mem.store(ram, 1, v);
            }
            3 => {
                let _ = twist_mem.load(ram, 1);
                let v = shout_tbl.lookup(tbl, 0);
                twist_mem.store(ram, 0, v);
            }
            _ => {}
        }

        self.step += 1;
        self.pc += 4;
        if self.step >= 4 {
            self.halted = true;
        }

        Ok(StepMeta {
            pc_after: self.pc,
            opcode: 0xAA,
        })
    }
}

// ----------------  Minimal Twist implementation ----------------

#[derive(Default)]
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

// ----------------  Minimal Shout implementation ----------------

struct VecShout {
    tables: HashMap<ShoutId, Vec<u64>>,
}

impl Shout<u64> for VecShout {
    fn lookup(&mut self, id: ShoutId, key: u64) -> u64 {
        self.tables
            .get(&id)
            .and_then(|t| t.get(key as usize).copied())
            .unwrap_or(0)
    }
}

/// Create test NeoParams suitable for the small test geometry.
fn create_test_params() -> NeoParams {
    NeoParams::goldilocks_127()
}

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

fn build_dummy_ccs(m: usize) -> CcsStructure<F> {
    let mat = Mat::zero(1, m, F::ZERO);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("dummy CCS")
}

/// Test: end-to-end trace → plain traces → encoding → semantic checks
///
/// This test validates the full Twist+Shout pipeline:
/// 1. Run a 4-step VM program with traced memory and lookups
/// 2. Build plain traces from the VM trace
/// 3. Encode traces for Twist (RW memory) and Shout (RO lookup)
/// 4. Verify semantic correctness without cryptographic proofs
///
/// Geometry:
/// - d=2, n_side=2 → address space size n_side^d = 4 cells (scalar addrs 0..3)
/// - tracked memory k=4
/// - steps T=4
///
/// RO table (ShoutId=0): [10, 20, 30, 40] at addresses 0..3
/// RW memory (TwistId=1): starts all zeros
#[test]
fn twist_shout_trace_to_witness_smoke() {
    // 1) Build RO table
    let mut shout_impl = VecShout { tables: HashMap::new() };
    shout_impl.tables.insert(ShoutId(0), vec![10, 20, 30, 40]);

    // 2) Run VM trace
    let trace: VmTrace<u64, u64> =
        trace_program(ScriptCpu::new(), MapTwist::default(), shout_impl, 16).expect("trace_program should succeed");
    assert_eq!(trace.len(), 4, "Expected 4 steps in trace");

    // 3) Build plain traces
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(1u32, PlainMemLayout { k: 4, d: 2, n_side: 2 });

    let initial_mem: HashMap<(u32, u64), F> = HashMap::new();
    let plain_mem = build_plain_mem_traces::<F>(&trace, &mem_layouts, &initial_mem);
    let t = &plain_mem[&1u32];

    // Verify Twist (TwistId=1) plain trace
    // Expected from the step script:
    // j=0: read mem[1] -> 0, write mem[1] = 30
    // j=1: read mem[1] -> 30
    // j=2: write mem[1] = 40
    // j=3: read mem[1] -> 40, write mem[0] = 10
    assert_eq!(t.has_read, vec![F::ONE, F::ONE, F::ZERO, F::ONE], "has_read mismatch");
    assert_eq!(t.has_write, vec![F::ONE, F::ZERO, F::ONE, F::ONE], "has_write mismatch");
    assert_eq!(t.read_addr, vec![1, 1, 0, 1], "read_addr mismatch");
    assert_eq!(t.write_addr, vec![1, 0, 1, 0], "write_addr mismatch");

    // read_val: [0, 30, 0, 40] (reads observe pre-state)
    assert_eq!(
        t.read_val,
        vec![F::ZERO, F::from_u64(30), F::ZERO, F::from_u64(40)],
        "read_val mismatch"
    );

    // write_val: [30, 0, 40, 10]
    assert_eq!(
        t.write_val,
        vec![F::from_u64(30), F::ZERO, F::from_u64(40), F::from_u64(10)],
        "write_val mismatch"
    );

    // inc_at_write_addr (step-major): delta applied at the write address in that step.
    assert_eq!(
        t.inc_at_write_addr,
        vec![F::from_u64(30), F::ZERO, F::from_u64(10), F::from_u64(10)],
        "inc_at_write_addr mismatch"
    );

    // Build Shout (ShoutId=0) plain trace
    let mut table_sizes = HashMap::new();
    table_sizes.insert(0u32, (4usize, 2usize)); // (size, d) - table has 4 entries, decomposed in d=2
    let plain_lut = build_plain_lut_traces::<F>(&trace, &table_sizes);
    let l = &plain_lut[&0u32];

    // Expected Shout trace:
    // j=0: lookup table[2] -> 30
    // j=1: no lookup
    // j=2: lookup table[3] -> 40
    // j=3: lookup table[0] -> 10
    assert_eq!(
        l.has_lookup,
        vec![F::ONE, F::ZERO, F::ONE, F::ONE],
        "has_lookup mismatch"
    );
    assert_eq!(
        l.addr,
        vec![2, 0, 3, 0],
        "Shout addr mismatch (default 0 where no-lookup)"
    );
    assert_eq!(
        l.val,
        vec![F::from_u64(30), F::ZERO, F::from_u64(40), F::from_u64(10)],
        "Shout val mismatch"
    );

    // 4) Encode for Twist and Shout
    let params = create_test_params();
    let dummy = DummyCommit::default();
    let commit = |m: &neo_ccs::matrix::Mat<F>| dummy.commit(m);

    // Encode memory for Twist
    let mem_init = neo_memory::MemInit::Zero;
    let (mem_inst, mem_wit) =
        encode_mem_for_twist(&params, &mem_layouts[&1u32], &mem_init, t, &commit, t.steps, 0);

    // Create the LUT table struct
    let table = LutTable {
        table_id: 0,
        k: 4,
        d: 2,
        n_side: 2,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };

    // Encode lookup for Shout
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &table, l, &commit, l.has_lookup.len(), 0);

    // Sanity: matrix counts
    // Twist: 2*d*ell + 5 matrices (Route A layout)
    // (ra_bits[d*ell], wa_bits[d*ell], has_read, has_write, wv, rv, inc_at_write_addr)
    // Note: The d here is the address decomposition dimension from PlainMemLayout, not params.d
    let expected_mem_mats = 2 * mem_layouts[&1u32].d * mem_inst.ell + 5;
    assert_eq!(
        mem_wit.mats.len(),
        expected_mem_mats,
        "Twist witness should have {} matrices, got {}",
        expected_mem_mats,
        mem_wit.mats.len()
    );

    // Shout: d*ell + 2 matrices (address bits + has_lookup + val)
    // Note: table_at_addr is NOT committed in address-domain architecture
    let expected_lut_mats = table.d * lut_inst.ell + 2;
    assert_eq!(
        lut_wit.mats.len(),
        expected_lut_mats,
        "Shout witness should have {} matrices, got {}",
        expected_lut_mats,
        lut_wit.mats.len()
    );

    // 5) Semantic checks (the real "integration guardrails")
    twist::check_twist_semantics(&params, &mem_inst, &mem_wit).expect("Twist semantic check should pass");

    shout::check_shout_semantics(&params, &lut_inst, &lut_wit, &l.val).expect("Shout semantic check should pass");

    println!("✓ twist_shout_trace_to_witness_smoke passed all checks");
}

// ============================================================================
// Test 2: IDX→OneHot adapter correctness
// ============================================================================

/// Compute the bits of an integer in little-endian order.
fn bits_of(x: usize, ell: usize) -> Vec<KElem> {
    (0..ell)
        .map(|k| if ((x >> k) & 1) == 1 { KElem::ONE } else { KElem::ZERO })
        .collect()
}

/// Compute χ_idx(u) = ∏_k (b_k·u_k + (1-b_k)·(1-u_k))
///
/// This is the "virtual one-hot" evaluation using index bits.
fn chi_idx(u: &[KElem], bits: &[KElem]) -> KElem {
    bits.iter()
        .zip(u.iter())
        .fold(KElem::ONE, |acc, (&b, &uk)| {
            acc * (b * uk + (KElem::ONE - b) * (KElem::ONE - uk))
        })
}

/// Test: IDX→OneHot adapter correctness
///
/// This test validates that the index-bits approach matches naive one-hot
/// MLE evaluation. This is the foundation for the Π_IDX2OH protocol.
///
/// Given:
/// - address domain m=8 (so ℓ_addr=3)
/// - cycle domain T=4 (ℓ_cycle=2)
/// - indices idx = [5, 0, 5, 3]
/// - mask = [1, 0, 1, 1] (for Shout-like masking)
///
/// We verify:
/// Σ_j χ_j(r)·mask[j]·χ_idx[j](u) == Σ_{j,a} χ_j(r)·χ_a(u)·A[j,a]
///
/// where A[j,a] = mask[j] iff a == idx[j], else 0.
#[test]
fn idx2oh_matches_naive_onehot_eval() {
    // Test parameters
    let idx = [5usize, 0, 5, 3];
    let mask = [KElem::ONE, KElem::ZERO, KElem::ONE, KElem::ONE];
    let ell_cycle = 2usize; // T = 4 cycles
    let ell_addr = 3usize; // m = 8 addresses

    // Fixed evaluation points (don't need to be random for regression testing)
    let r = [KElem::from_u64(7), KElem::from_u64(11)];
    let u = [KElem::from_u64(3), KElem::from_u64(5), KElem::from_u64(9)];

    // Build χ tables
    let chi_cycle = build_chi_table(&r); // len 4
    let chi_addr = build_chi_table(&u); // len 8

    assert_eq!(chi_cycle.len(), 1 << ell_cycle, "chi_cycle length mismatch");
    assert_eq!(chi_addr.len(), 1 << ell_addr, "chi_addr length mismatch");

    // "Naive one-hot MLE" (conceptual A[j,a])
    // A[j,a] = mask[j] if a == idx[j], else 0
    // Sum = Σ_j χ_j(r) · χ_{idx[j]}(u) · mask[j]
    let mut naive = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        if mask[j] == KElem::ZERO {
            continue;
        }
        naive += chi_cycle[j] * chi_addr[idx[j]];
    }

    // "Index-bits virtual one-hot" (what Π_IDX2OH must justify)
    // Sum = Σ_j χ_j(r) · χ_idx[j](u) · mask[j]
    // where χ_idx(u) = ∏_k (b_k·u_k + (1-b_k)·(1-u_k))
    let mut virt = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        if mask[j] == KElem::ZERO {
            continue;
        }
        let bits = bits_of(idx[j], ell_addr);
        virt += chi_cycle[j] * chi_idx(&u, &bits);
    }

    assert_eq!(
        naive, virt,
        "Naive one-hot and virtual one-hot should evaluate to the same value"
    );

    // Additional test: verify without masking (all entries)
    let mut naive_unmasked = KElem::ZERO;
    let mut virt_unmasked = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        naive_unmasked += chi_cycle[j] * chi_addr[idx[j]];
        let bits = bits_of(idx[j], ell_addr);
        virt_unmasked += chi_cycle[j] * chi_idx(&u, &bits);
    }
    assert_eq!(naive_unmasked, virt_unmasked, "Unmasked naive and virtual should match");

    // Test edge case: single active entry
    let single_mask = [KElem::ZERO, KElem::ZERO, KElem::ONE, KElem::ZERO];
    let mut naive_single = KElem::ZERO;
    let mut virt_single = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        if single_mask[j] == KElem::ZERO {
            continue;
        }
        naive_single += chi_cycle[j] * chi_addr[idx[j]];
        let bits = bits_of(idx[j], ell_addr);
        virt_single += chi_cycle[j] * chi_idx(&u, &bits);
    }
    assert_eq!(naive_single, virt_single, "Single-entry case should match");

    println!("✓ idx2oh_matches_naive_onehot_eval passed all checks");
}

/// Regression: ensure memory sidecar outputs pad correctly for merge RLC/DEC.
#[test]
fn twist_shout_sidecar_shapes_and_rlc() {
    // Reuse the scripted program to build witnesses
    let params = create_test_params();
    let dummy = DummyCommit::default();
    let commit = |m: &neo_ccs::matrix::Mat<F>| dummy.commit(m);

    // VM trace
    let mut shout_impl = VecShout { tables: HashMap::new() };
    shout_impl.tables.insert(ShoutId(0), vec![10, 20, 30, 40]);
    let trace: VmTrace<u64, u64> =
        trace_program(ScriptCpu::new(), MapTwist::default(), shout_impl, 16).expect("trace_program should succeed");

    // Plain traces and encodings
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(1u32, PlainMemLayout { k: 4, d: 2, n_side: 2 });
    let initial_mem: HashMap<(u32, u64), F> = HashMap::new();
    let plain_mem = build_plain_mem_traces::<F>(&trace, &mem_layouts, &initial_mem);
    let t = &plain_mem[&1u32];
    let mem_init = neo_memory::MemInit::Zero;
    let (_mem_inst, mem_wit) =
        encode_mem_for_twist(&params, &mem_layouts[&1u32], &mem_init, t, &commit, t.steps, 0);

    let mut table_sizes = HashMap::new();
    table_sizes.insert(0u32, (4usize, 2usize));
    let plain_lut = build_plain_lut_traces::<F>(&trace, &table_sizes);
    let l = &plain_lut[&0u32];
    let table = LutTable {
        table_id: 0,
        k: 4,
        d: 2,
        n_side: 2,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };
    let (_lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &table, l, &commit, l.has_lookup.len(), 0);

    // CCS structure used only for dimensions/padding
    let target_cols = mem_wit.mats.iter().map(|m| m.cols()).max().unwrap_or(1);
    let s_me = build_dummy_ccs(target_cols);
    // Manual dims (skip extension policy for tiny dummy CCS)
    let ell_d = params.d.next_power_of_two().trailing_zeros() as usize;
    let ell_n = s_me.n.next_power_of_two().max(2).trailing_zeros() as usize;

    // Build padded witnesses and dummy ME claims (shape-only check; bypass protocol)
    let pad_mat = |mat: &Mat<F>| -> Mat<F> {
        if mat.cols() >= s_me.m {
            return mat.clone();
        }
        let mut out = Mat::zero(mat.rows(), s_me.m, F::ZERO);
        for r in 0..mat.rows() {
            for c in 0..mat.cols() {
                out[(r, c)] = mat[(r, c)];
            }
        }
        out
    };

    let y_pad = 1usize << ell_d;
    let mut me_witnesses: Vec<Mat<F>> = Vec::new();
    let mut me_claims: Vec<neo_ccs::relations::MeInstance<Cmt, F, KElem>> = Vec::new();

    for mat in lut_wit.mats.iter().chain(mem_wit.mats.iter()) {
        let padded = pad_mat(mat);
        me_witnesses.push(padded.clone());

        let mut y_rows = Vec::new();
        for _ in 0..s_me.t() {
            y_rows.push(vec![KElem::ZERO; y_pad]);
        }
        let y_scalars = vec![KElem::ZERO; s_me.t()];

        me_claims.push(neo_ccs::relations::MeInstance {
            c: dummy.commit(&padded),
            X: Mat::from_row_major(params.d as usize, 0, vec![]),
            r: vec![KElem::ZERO; ell_n],
            y: y_rows,
            y_scalars,
            m_in: 0,
            fold_digest: [0u8; 32],
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
        });
    }

    assert_eq!(me_claims.len(), me_witnesses.len(), "ME claims/witnesses mismatch");
    assert!(!me_claims.is_empty(), "sidecar should produce ME claims");

    // Padding checks
    for me in &me_claims {
        assert_eq!(me.r.len(), ell_n, "r should be padded to ell_n");
        assert_eq!(me.y.len(), s_me.t(), "y rows should match t");
        assert!(
            me.y.iter().all(|row| row.len() == y_pad),
            "y rows should be padded to 2^ell_d"
        );
    }
    for w in &me_witnesses {
        assert_eq!(w.rows(), params.d as usize, "witness rows should equal params.d");
        assert_eq!(w.cols(), s_me.m, "witness cols should be padded to s_me.m");
    }

    // RLC sanity: combine ME claims/witnesses with identity rhos
    let rhos: Vec<Mat<F>> = (0..me_claims.len())
        .map(|_| Mat::identity(params.d as usize))
        .collect();
    let (parent, Z_mix) = api::rlc_with_commit(
        FoldingMode::Optimized,
        &s_me,
        &params,
        &rhos,
        &me_claims,
        &me_witnesses,
        ell_d,
        |_rhos, _cs| Cmt::zeros(params.d as usize, 1),
    );
    assert_eq!(parent.X.cols(), 0, "X should be empty for m_in=0");
    assert_eq!(Z_mix.rows(), params.d as usize, "Z_mix rows");
    assert_eq!(Z_mix.cols(), s_me.m, "Z_mix cols");

    println!("✓ twist_shout_sidecar_shapes_and_rlc passed");
}

/// Extended test: verify IDX→OneHot for Twist's read/write masks
///
/// Same principle as above, but tests both read and write address encodings
/// that would be used in the Twist protocol.
#[test]
fn idx2oh_twist_read_write_masks() {
    // Simulate a trace with:
    // - 4 steps
    // - read at steps 0, 2 from addresses 1, 3
    // - write at steps 1, 3 to addresses 2, 0

    let ell_cycle = 2usize;
    let ell_addr = 2usize; // m = 4 addresses

    // Read indices and mask
    let read_idx = [1usize, 0, 3, 0]; // default 0 where no read
    let has_read = [KElem::ONE, KElem::ZERO, KElem::ONE, KElem::ZERO];

    // Write indices and mask
    let write_idx = [0usize, 2, 0, 0]; // default 0 where no write
    let has_write = [KElem::ZERO, KElem::ONE, KElem::ZERO, KElem::ONE];

    // Fixed evaluation points
    let r = [KElem::from_u64(13), KElem::from_u64(17)];
    let u = [KElem::from_u64(19), KElem::from_u64(23)];

    let chi_cycle = build_chi_table(&r);
    let chi_addr = build_chi_table(&u);

    // Test read addresses
    let mut read_naive = KElem::ZERO;
    let mut read_virt = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        if has_read[j] == KElem::ZERO {
            continue;
        }
        read_naive += chi_cycle[j] * chi_addr[read_idx[j]];
        let bits = bits_of(read_idx[j], ell_addr);
        read_virt += chi_cycle[j] * chi_idx(&u, &bits);
    }
    assert_eq!(read_naive, read_virt, "Read address IDX→OneHot should match");

    // Test write addresses
    let mut write_naive = KElem::ZERO;
    let mut write_virt = KElem::ZERO;
    for j in 0..(1 << ell_cycle) {
        if has_write[j] == KElem::ZERO {
            continue;
        }
        write_naive += chi_cycle[j] * chi_addr[write_idx[j]];
        let bits = bits_of(write_idx[j], ell_addr);
        write_virt += chi_cycle[j] * chi_idx(&u, &bits);
    }
    assert_eq!(write_naive, write_virt, "Write address IDX→OneHot should match");

    println!("✓ idx2oh_twist_read_write_masks passed all checks");
}
