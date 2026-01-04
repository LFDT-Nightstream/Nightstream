//! End-to-end prove+verify for a small *compiled* RV32 guest program under the B1 shared-bus step circuit.
//!
//! The guest is authored in Rust under `riscv-tests/guests/rv32-smoke/` and its ROM bytes are committed in
//! `riscv-tests/binaries/rv32_smoke_rom.rs` so this test doesn't need to cross-compile at runtime.

#![allow(non_snake_case)]

#[path = "binaries/rv32_smoke_rom.rs"]
mod rv32_smoke_rom;

use neo_fold::riscv_shard::Rv32B1;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_program_compiled_full_prove_verify() {
    let program_base = rv32_smoke_rom::RV32_SMOKE_ROM_BASE;
    let program_bytes: &[u8] = &rv32_smoke_rom::RV32_SMOKE_ROM;
    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(0x200)
        .chunk_size(4)
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(0x100c))
        .prove()
        .expect("prove");

    println!("Prove duration: {:?}", run.prove_duration());
    run.verify().expect("verify");
    println!("Verify duration: {:?}", run.verify_duration().expect("verify duration"));

    let end = run.final_boundary_state().expect("boundary");
    assert_eq!(end.regs_final[4], F::from_u64(0x100c));

    assert!(
        matches!(
            run.verify_output_claim(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(0x100d)),
            Ok(false) | Err(_)
        ),
        "wrong output claim must not verify"
    );
}
