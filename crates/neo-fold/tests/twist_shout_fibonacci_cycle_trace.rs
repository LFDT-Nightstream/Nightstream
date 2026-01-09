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

#[path = "common/fib_twist_shout_vm.rs"]
mod fib_twist_shout_vm;

use std::collections::HashMap;
use std::sync::Arc;

use fib_twist_shout_vm::{fib_mod_q_u64, FibTwistShoutVm, MapShout, MapTwist};
use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{CcsBuilder, FoldingSession, NeoCircuit, ShoutPort, SharedBusResources, TwistPort};
use neo_fold::session::{Lane, Public, Scalar};
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout};
use neo_fold::shard::MemOrLutProof;
use neo_fold::shard::StepLinkingConfig;
use neo_math::{D, F};
use neo_memory::plain::PlainMemLayout;
use neo_params::NeoParams;
use neo_vm_trace::{Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use neo_fold::output_binding::simple_output_config;

// We intentionally use a *large* chunk size here to better reflect “do more before folding” and to
// avoid the worst-case overheads of `chunk_size=1`.
const CHUNK_SIZE: usize = 32;
const DEFAULT_CHUNKS: usize = 2;

witness_layout! {
    #[derive(Clone, Debug)]
    pub FibCols<const N: usize> {
        pub one: Public<Scalar>,

        pub f_curr_before: Lane<N>,
        pub f_next_before: Lane<N>,
        pub f_curr_after: Lane<N>,
        pub f_next_after: Lane<N>,

        pub twist0: TwistPort<N>,
        pub shout0: ShoutPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct FibCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for FibCircuit<N> {
    type Layout = FibCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        resources
            .twist(0)
            .layout(PlainMemLayout { k: 2, d: 1, n_side: 2 , lanes: 1})
            .init_cell(0, F::ONE);
        resources.set_binary_table(0, vec![F::ZERO, F::ONE]);
    }

    fn cpu_bindings(
        &self,
        layout: &Self::Layout,
    ) -> Result<
        (
            HashMap<u32, Vec<neo_memory::cpu::ShoutCpuBinding>>,
            HashMap<u32, Vec<neo_memory::cpu::TwistCpuBinding>>,
        ),
        String,
    > {
        Ok((
            HashMap::from([(0u32, vec![layout.shout0.cpu_binding()])]),
            HashMap::from([(0u32, vec![layout.twist0.cpu_binding()])]),
        ))
    }

    fn define_cpu_constraints(&self, cs: &mut CcsBuilder<F>, layout: &Self::Layout) -> Result<(), String> {
        for j in 0..N {
            // (f_curr_before + f_next_before) * shout_val = f_next_after
            cs.r1cs_terms(
                [
                    (layout.f_curr_before.at(j), F::ONE),
                    (layout.f_next_before.at(j), F::ONE),
                ],
                [(layout.shout0.val.at(j), F::ONE)],
                [(layout.f_next_after.at(j), F::ONE)],
            );

            // f_curr_after == f_next_before
            cs.eq(layout.f_curr_after.at(j), layout.f_next_before.at(j));

            // f_next_before == twist_rv
            cs.eq(layout.f_next_before.at(j), layout.twist0.rv.at(j));

            // f_next_after == twist_wv
            cs.eq(layout.f_next_after.at(j), layout.twist0.wv.at(j));
        }

        cs.lane_continuity(layout.f_curr_before, layout.f_curr_after);
        cs.lane_continuity(layout.f_next_before, layout.f_next_after);
        Ok(())
    }

    fn build_witness_prefix(&self, layout: &Self::Layout, chunk: &[neo_vm_trace::StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
        if chunk.len() != N {
            return Err(format!(
                "FibCircuit witness builder expects full chunks (len {} != N {})",
                chunk.len(),
                N
            ));
        }

        let mut z = <Self::Layout as neo_fold::session::WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        for (j, step) in chunk.iter().enumerate() {
            if step.regs_before.len() != 2 || step.regs_after.len() != 2 {
                return Err(format!("expected 2 regs for Fibonacci VM (lane {j})"));
            }
        }

        layout
            .f_curr_before
            .set_from_iter(&mut z, chunk.iter().map(|step| F::from_u64(step.regs_before[0])))?;
        layout
            .f_next_before
            .set_from_iter(&mut z, chunk.iter().map(|step| F::from_u64(step.regs_before[1])))?;
        layout
            .f_curr_after
            .set_from_iter(&mut z, chunk.iter().map(|step| F::from_u64(step.regs_after[0])))?;
        layout
            .f_next_after
            .set_from_iter(&mut z, chunk.iter().map(|step| F::from_u64(step.regs_after[1])))?;

        layout.twist0.fill_from_trace(chunk, 0, &mut z)?;
        layout.shout0.fill_from_trace(chunk, 0, &mut z)?;
        Ok(z)
    }
}

fn dump_enabled() -> bool {
    // Run with:
    //   NEO_FIB_TRACE=1 cargo test -p neo-fold --release twist_shout_fibonacci_cycle_trace -- --nocapture
    std::env::var("NEO_FIB_TRACE").is_ok()
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
    let circuit = Arc::new(FibCircuit::<CHUNK_SIZE>::default());
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    // Params:
    // - bump k_rho for comfortable Π_RLC norm bound margin in tests
    // - use b=4 so Ajtai digit encoding can represent full Goldilocks values (b^d >> q),
    //   which matters once you run many chunks/steps (values quickly leave the tiny b=2^54 range).
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
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

    let committer = setup_ajtai_committer(m, params.kappa as usize);
    let prover = pre
        .into_prover(params.clone(), committer.clone())
        .expect("into_prover (R1csCpu shared-bus config)");

    // Execute VM and build shared-bus step bundles into the session.
    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), committer);
    let mut twist = MapTwist::default();
    twist.store(TwistId(0), 0, 1);
    let shout = MapShout { table: vec![0, 1] };

    let t_witness = Instant::now();
    prover
        .execute_into_session(
            &mut session,
            FibTwistShoutVm::new(max_steps as u64, base_params.q),
            twist,
            shout,
            max_steps,
        )
        .expect("execute_into_session should succeed");
    let witness_dur = t_witness.elapsed();

    // Sanity: final Fibonacci value.
    //
    // NOTE: We sanity-check two ways:
    //  1) simulate the VM locally (u64 arithmetic mod q),
    //  2) verify the shard proof with output binding against the claimed output.
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

    // Prove + verify (output binding auto-derives final memory state from the witness build).
    let output_addr = 0u64;
    let output_val = F::from_u64(expected_next);
    let ob_cfg = simple_output_config(1, output_addr, output_val);

    let t_prove = Instant::now();
    let run = session
        .fold_and_prove_with_output_binding_auto_simple(prover.ccs(), &ob_cfg)
        .expect("prove should succeed");
    let prove_dur = t_prove.elapsed();
    assert!(run.output_proof.is_some(), "expected output binding proof to be attached");

    let outputs = run.compute_fold_outputs(&[]);

    let t_verify = Instant::now();
    session.set_step_linking(StepLinkingConfig::new(vec![(0, 0)]));
    let ok = session
        .verify_with_output_binding_collected_simple(prover.ccs(), &run, &ob_cfg)
        .expect("verify should run");
    let verify_dur = t_verify.elapsed();
    assert!(ok, "verification should pass");
    assert!(
        run.steps.len() >= 2,
        "this test is meant to demonstrate at least one fold (need >=2 folding steps)"
    );

    if timing_enabled() {
        println!(
            "\n[timing] chunks={} steps={} chunk_size={} witness_build={:?} prove={:?} verify={:?}",
            max_steps / CHUNK_SIZE,
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
            prover.ccs().m,
            <FibCols<CHUNK_SIZE> as neo_fold::session::WitnessLayout>::M_IN,
            CHUNK_SIZE,
            core::iter::once(1usize), // shout ell_addr = d*ell = 1*1
            core::iter::once(1usize), // twist ell_addr = d*ell = 1*1
        )
        .expect("bus layout");

        println!("\n=== Twist+Shout Fibonacci proving trace ===");
        let m_in = <FibCols<CHUNK_SIZE> as neo_fold::session::WitnessLayout>::M_IN;
        let m = prover.ccs().m;
        println!(
            "vm_steps={n_steps}  fold_steps={}  chunk_size={CHUNK_SIZE}  m={m}  m_in={m_in}",
            max_steps / CHUNK_SIZE
        );
        println!("commit: backend=AjtaiSModule  kappa={}", prover.committer.kappa());
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
        for (i, step_proof) in run.steps.iter().enumerate() {
            println!("\n-- step {i} --");

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
                "shout_addr_pre: claimed_sums={} active_lanes={} rounds={} r_addr_len={}",
                step_proof.mem.shout_addr_pre.claimed_sums.len(),
                step_proof.mem.shout_addr_pre.active_lanes.len(),
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
