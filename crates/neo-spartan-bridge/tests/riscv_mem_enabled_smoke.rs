#![allow(non_snake_case)]

//! Phase-1 smoke test (mem-enabled): compress an RV32 B1 shard proof (Twist/Shout present) into a Spartan2 proof.
//!
//! This exercises:
//! - `absorb_step_memory` transcript framing,
//! - Shout/Twist addr-pre transcript + batched sumcheck plumbing,
//! - Route-A batched time multi-claim verification,
//! - Route-A memory terminal checks (Twist/Shout time + val-eval terminals),
//! - Twist val-eval batch verification + val-lane RLC/DEC verification.
//!
//! Output binding is intentionally disabled in this test.
//!
//! Note: this test is `#[ignore]` because Spartan2 proving for RV32 B1 is very slow.

use neo_fold::riscv_shard::Rv32B1;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{compute_vm_digest_v1, prove_fold_run, setup_fold_run, verify_fold_run};
use p3_field::PrimeCharacteristicRing;

#[test]
#[ignore]
fn test_riscv_rv32_b1_mem_enabled_spartan_phase1_smoke() {
    // Minimal RV32 program: one `addi x0, x0, 0` (NOP).
    //
    // Encoding: 0x00000013 (little-endian).
    let program_base = 0u64;
    let program_bytes: &[u8] = &[0x13, 0x00, 0x00, 0x00];
    let vm_digest = compute_vm_digest_v1(program_bytes);

    // Keep this test tiny: one chunk/step is enough to exercise the mem-enabled transcript plumbing.
    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(4)
        .chunk_size(1)
        .max_steps(1)
        .shout_auto_minimal()
        .prove()
        .expect("prove");

    run.verify().expect("native verify");

    let steps_public = run.steps_public();
    let initial_accumulator = Vec::new(); // seed run
    let witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public.clone(),
        initial_accumulator,
        vm_digest,
        None,
    );

    let (pk, vk) = setup_fold_run(run.params(), run.ccs(), &witness).expect("setup_fold_run");
    let spartan = prove_fold_run(&pk, run.params(), run.ccs(), witness).expect("prove_fold_run");

    assert!(
        verify_fold_run(&vk, run.params(), run.ccs(), &vm_digest, &steps_public, None, &[], &spartan)
            .expect("verify_fold_run"),
        "Spartan proof should verify"
    );
}

#[test]
#[ignore]
fn test_riscv_rv32_b1_mem_enabled_spartan_rejects_tampered_bus_opening() {
    // Minimal RV32 program: one `addi x0, x0, 0` (NOP).
    let program_base = 0u64;
    let program_bytes: &[u8] = &[0x13, 0x00, 0x00, 0x00];
    let vm_digest = compute_vm_digest_v1(program_bytes);

    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(4)
        .chunk_size(1)
        .max_steps(1)
        .shout_auto_minimal()
        .prove()
        .expect("prove");
    run.verify().expect("native verify");

    let steps_public = run.steps_public();
    let initial_accumulator = Vec::new();
    let good_witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public.clone(),
        initial_accumulator.clone(),
        vm_digest,
        None,
    );

    let (pk, _vk) = setup_fold_run(run.params(), run.ccs(), &good_witness).expect("setup_fold_run");

    // Tamper a Twist time-lane opening (has_write) in ccs_out[0].y at a bus row.
    let mut bad_proof = run.proof().clone();
    let step0 = steps_public.first().expect("step0 public");
    let shout_ell_addrs = step0.lut_insts.iter().map(|inst| inst.d * inst.ell);
    let twist_ell_addrs = step0.mem_insts.iter().map(|inst| inst.d * inst.ell);
    let bus = neo_memory::cpu::build_bus_layout_for_instances(
        run.ccs().m,
        step0.mcs_inst.m_in,
        /*chunk_size=*/ 1,
        shout_ell_addrs,
        twist_ell_addrs,
    )
    .expect("bus layout");
    assert!(!bus.twist_cols.is_empty(), "expected >=1 Twist instance");

    let ccs_t = run.ccs().t();
    let has_write_col = bus.twist_cols[0].has_write;
    let y_idx = ccs_t + has_write_col;
    bad_proof.steps[0].fold.ccs_out[0].y[y_idx][0] += neo_math::K::ONE;

    let bad_witness = FoldRunWitness::new(bad_proof, steps_public, initial_accumulator, vm_digest, None);
    assert!(
        prove_fold_run(&pk, run.params(), run.ccs(), bad_witness).is_err(),
        "tampered witness should not satisfy the circuit"
    );
}
