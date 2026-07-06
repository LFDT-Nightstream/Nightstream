//! Perf snapshot — spec §13 step 8: record the §10 cost-budget actuals.
//! All `--ignored`; run with `--nocapture` to read the table. Off-by-2×
//! on any §10 line reopens the spec (v1's D3 discipline).
//!
//! ```text
//! cargo test -p neo-fold-clean --release --test perf_nebula -- \
//!     --ignored --nocapture
//! ```

use std::time::Instant;

use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::prove_segment;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::lifecycle::{self, preprocess, verify_uncompressed_audit};

#[path = "../support/mod.rs"]
mod support;

const LANE_KAPPA: usize = 18;

/// §10 structural actuals at the v3 targets (`r = 12, μ = 16, B = 64,
/// N = 1,088`): `S_mem` rows / columns / nnz, plan compile time (includes
/// the full `D_init` sweep of 1,088 lane commitments), and the evaluated
/// §9 budget. Structure-only — no folding.
#[test]
#[ignore]
fn nebula_v3_targets_structure_snapshot() {
    let params = NebulaParams::v3_targets();
    let t = Instant::now();
    let rom = vec![0u32; params.rom_cells() as usize];
    let plan = NebulaPlan::new(params, rom, [0xBE; 32], LANE_KAPPA).expect("v3 plan");
    let compile = t.elapsed();

    let c = plan.circuit();
    let budget = plan.error_budget();
    // §10 "per-op amortized rows": one step's rows divided by its op
    // slots (the scan cost is inside the step, so the ratio is per-step).
    let ops_amortized = c.rows() as f64 / params.b_ops as f64;
    println!("== Nebula §10 actuals, v3 targets ==");
    println!("S_mem rows                {:>10}   (§10 budget ≈ 58k)", c.rows());
    println!(
        "S_mem witness columns     {:>10}   (committed coordinates per step)",
        c.cols()
    );
    println!("S_mem nnz                 {:>10}", c.nnz());
    println!("steps per segment N       {:>10}", params.steps_per_segment());
    println!(
        "rows per op (amortized)   {:>10.1}   (§10 budget ≈ 875 at N_ops = R+M)",
        ops_amortized
    );
    println!(
        "§9 budget per FS attempt  m_seg = {} → 2^{:.1}",
        budget.m_seg, budget.log2_bound_per_attempt
    );
    println!("plan compile (incl. D_init sweep) {:>8.2?}", compile);

    // §10 discipline: off-by-2× on the headline row count reopens the spec.
    assert!(
        c.rows() < 2 * 58_000,
        "S_mem rows {} exceed 2× the §10 budget",
        c.rows()
    );
}

/// End-to-end timing at the spec §2 test profile (`r = 4, μ = 8,
/// B_ops = B_scan = 8, N = 34`): one full segment — native pass, lane
/// commits, γ, 34 witnesses, 34 folds — plus finalization and audit
/// verification.
#[test]
#[ignore]
fn nebula_test_profile_segment_snapshot() {
    let params = NebulaParams::test_profile();
    let rom: Vec<u32> = (0..params.rom_cells() as u32).map(|i| i * 3 + 1).collect();
    let plan = NebulaPlan::new(params, rom, [0xBF; 32], LANE_KAPPA).expect("test-profile plan");

    let structure = plan.circuit().structure().clone();
    let engine_params = config::r1cs_params(structure.n, structure.m).expect("engine params");
    support::install_ajtai_module(&engine_params, &structure);
    let t = Instant::now();
    let prep = preprocess(engine_params, structure, Some(plan.circuit().m_in()))
        .expect("preprocessing")
        .with_nebula(plan.config());
    let t_preprocess = t.elapsed();

    let mut memory = Memory::new(params, plan.rom_image()).expect("memory");
    let mut run = memory.begin_segment().expect("segment");
    for i in 0..(params.steps_per_segment() * params.b_ops) as u64 / 2 {
        let addr = i % params.ram_cells();
        run.write(true, addr, (i + 1) as u32).expect("write");
        assert_eq!(run.read(true, addr).expect("read"), (i + 1) as u32);
    }
    let trace = run.finish();

    let t = Instant::now();
    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace).expect("segment");
    let t_prove = t.elapsed();

    let t = Instant::now();
    let audit = lifecycle::finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    let t_finalize = t.elapsed();

    let t = Instant::now();
    verify_uncompressed_audit(&prep, &audit).expect("verify");
    let t_verify = t.elapsed();

    let n = params.steps_per_segment();
    println!("== Nebula segment timing, §2 test profile (N = {n}) ==");
    println!(
        "F' steps (chunks)        {:>8}   (batched folding: ⌈N / max_fresh⌉ — the recursion-overhead amortization of Nebula §5)",
        audit.steps.len()
    );
    println!(
        "S_mem rows / cols        {:>8} / {:>8}",
        plan.circuit().rows(),
        plan.circuit().cols()
    );
    println!("preprocess               {:>10.2?}", t_preprocess);
    println!(
        "prove segment            {:>10.2?}   ({:.2?} per step incl. fold)",
        t_prove,
        t_prove / n as u32
    );
    println!("finalize                 {:>10.2?}", t_finalize);
    println!("audit verify             {:>10.2?}", t_verify);
}
