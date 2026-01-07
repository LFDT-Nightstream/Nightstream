#![allow(non_snake_case)]

//! Phase-2 smoke test (output binding): compress an RV32 B1 shard proof (Twist/Shout + output binding)
//! into a Spartan2 proof.
//!
//! Note: this test is `#[ignore]` because Spartan2 proving for RV32 B1 is very slow.

use neo_fold::output_binding::simple_output_config;
use neo_fold::riscv_shard::Rv32B1;
use neo_math::{F, K};
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{compute_vm_digest_v1, prove_fold_run, setup_fold_run, verify_fold_run};
use p3_field::PrimeCharacteristicRing;

#[test]
#[ignore]
fn test_riscv_rv32_b1_output_binding_spartan_smoke() {
    // Minimal RV32 program: one `addi x0, x0, 0` (NOP).
    //
    // Encoding: 0x00000013 (little-endian).
    let program_base = 0u64;
    let program_bytes: &[u8] = &[0x13, 0x00, 0x00, 0x00];
    let vm_digest = compute_vm_digest_v1(program_bytes);

    // Bind a single output claim against RAM addr 0. This should remain 0 for a NOP-only trace.
    let output_addr = 0u64;
    let expected_output = F::ZERO;

    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(4)
        .chunk_size(1)
        .max_steps(1)
        .shout_auto_minimal()
        .output(output_addr, expected_output)
        .prove()
        .expect("prove");
    run.verify().expect("native verify (with output binding)");

    let steps_public = run.steps_public();
    let num_bits = steps_public
        .first()
        .expect("step0 public")
        .mem_insts
        .first()
        .expect("expected RAM Twist instance")
        .d;
    let ob_cfg = simple_output_config(num_bits, output_addr, expected_output);

    let initial_accumulator = Vec::new();
    let witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public.clone(),
        initial_accumulator,
        vm_digest,
        Some(ob_cfg.clone()),
    );

    let (pk, vk) = setup_fold_run(run.params(), run.ccs(), &witness).expect("setup_fold_run");
    let spartan = prove_fold_run(&pk, run.params(), run.ccs(), witness).expect("prove_fold_run");

    assert!(
        verify_fold_run(&vk, run.params(), run.ccs(), &vm_digest, &steps_public, Some(&ob_cfg), &[], &spartan)
            .expect("verify_fold_run"),
        "Spartan proof should verify"
    );
}

#[test]
#[ignore]
fn test_riscv_rv32_b1_output_binding_rejects_tampered_output_sumcheck() {
    let program_base = 0u64;
    let program_bytes: &[u8] = &[0x13, 0x00, 0x00, 0x00];
    let vm_digest = compute_vm_digest_v1(program_bytes);
    let output_addr = 0u64;
    let expected_output = F::ZERO;

    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(4)
        .chunk_size(1)
        .max_steps(1)
        .shout_auto_minimal()
        .output(output_addr, expected_output)
        .prove()
        .expect("prove");
    run.verify().expect("native verify");

    let steps_public = run.steps_public();
    let num_bits = steps_public
        .first()
        .expect("step0 public")
        .mem_insts
        .first()
        .expect("expected RAM Twist instance")
        .d;
    let ob_cfg = simple_output_config(num_bits, output_addr, expected_output);

    let initial_accumulator = Vec::new();
    let good_witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public.clone(),
        initial_accumulator.clone(),
        vm_digest,
        Some(ob_cfg.clone()),
    );
    let (pk, _vk) = setup_fold_run(run.params(), run.ccs(), &good_witness).expect("setup_fold_run");

    // Tamper one output sumcheck coefficient.
    let mut bad_proof = run.proof().clone();
    let out = bad_proof.output_proof.as_mut().expect("output_proof");
    out.output_sc.round_polys[0][0] += K::ONE;

    let bad_witness = FoldRunWitness::new(
        bad_proof,
        steps_public,
        initial_accumulator,
        vm_digest,
        Some(ob_cfg),
    );
    assert!(
        prove_fold_run(&pk, run.params(), run.ccs(), bad_witness).is_err(),
        "tampered output sumcheck should not satisfy the circuit"
    );
}

#[test]
#[ignore]
fn test_riscv_rv32_b1_output_binding_rejects_tampered_program_io() {
    let program_base = 0u64;
    let program_bytes: &[u8] = &[0x13, 0x00, 0x00, 0x00];
    let vm_digest = compute_vm_digest_v1(program_bytes);
    let output_addr = 0u64;
    let expected_output = F::ZERO;

    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(4)
        .chunk_size(1)
        .max_steps(1)
        .shout_auto_minimal()
        .output(output_addr, expected_output)
        .prove()
        .expect("prove");
    run.verify().expect("native verify");

    let steps_public = run.steps_public();
    let num_bits = steps_public
        .first()
        .expect("step0 public")
        .mem_insts
        .first()
        .expect("expected RAM Twist instance")
        .d;
    let ob_cfg_good = simple_output_config(num_bits, output_addr, expected_output);

    let initial_accumulator = Vec::new();
    let good_witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public.clone(),
        initial_accumulator.clone(),
        vm_digest,
        Some(ob_cfg_good.clone()),
    );
    let (pk, _vk) = setup_fold_run(run.params(), run.ccs(), &good_witness).expect("setup_fold_run");

    // Keep the native proof but change the claimed output value.
    let ob_cfg_bad = simple_output_config(num_bits, output_addr, F::ONE);
    let bad_witness = FoldRunWitness::new(
        run.proof().clone(),
        steps_public,
        initial_accumulator,
        vm_digest,
        Some(ob_cfg_bad),
    );
    assert!(
        prove_fold_run(&pk, run.params(), run.ccs(), bad_witness).is_err(),
        "tampered ProgramIO should not satisfy the circuit"
    );
}
