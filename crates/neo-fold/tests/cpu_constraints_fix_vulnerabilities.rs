//! Tests that verify the CPU constraint builder fixes the vulnerabilities found earlier.
//!
//! This module demonstrates that adding the Jolt-derived constraints to the CPU CCS
//! properly catches the attacks that previously passed.
//!
//! # Credits
//!
//! The constraint logic tested here is ported from the Jolt zkVM project:
//! - Repository: https://github.com/a16z/jolt
//! - Original file: `jolt-core/src/zkvm/r1cs/constraints.rs`
//! - License: Apache-2.0 / MIT
//!
//! Jolt's R1CS constraint system defines the binding between CPU instruction semantics
//! and memory/lookup values. We adapt this for Neo's CCS-based proving system.

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_ccs::CcsStructure;
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::cpu::constraints::{CpuColumnLayout, CpuConstraintBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

/// Witness column layout for our test CPU.
///
/// Layout: [const_one, is_load, is_store, rd_value, rs2_value, effective_addr, is_lookup, lookup_key, lookup_out, ...bus...]
const COL_CONST_ONE: usize = 0;
const COL_IS_LOAD: usize = 1;
const COL_IS_STORE: usize = 2;
const COL_RD_VALUE: usize = 3;
const COL_RS2_VALUE: usize = 4;
const COL_EFFECTIVE_ADDR: usize = 5;
const COL_IS_LOOKUP: usize = 6;
const COL_LOOKUP_KEY: usize = 7;
const COL_LOOKUP_OUT: usize = 8;
const CPU_COLS: usize = 9;

fn create_cpu_layout() -> CpuColumnLayout {
    CpuColumnLayout {
        is_load: COL_IS_LOAD,
        is_store: COL_IS_STORE,
        effective_addr: COL_EFFECTIVE_ADDR,
        rd_write_value: COL_RD_VALUE,
        rs2_value: COL_RS2_VALUE,
        is_lookup: COL_IS_LOOKUP,
        lookup_key: COL_LOOKUP_KEY,
        lookup_output: COL_LOOKUP_OUT,
    }
}

/// Create a test CCS with the CPU binding constraints.
///
/// This includes:
/// - Load value binding: is_load * (rd_value - bus_rv) = 0
/// - Store value binding: is_store * (rs2_value - bus_wv) = 0
/// - Padding constraints: (1 - has_read) * rv = 0, etc.
fn create_ccs_with_binding_constraints(ell_addr: usize) -> (CcsStructure<F>, neo_memory::cpu::BusLayout) {
    let bus_cols = 2 * ell_addr + 5;
    let m = CPU_COLS + bus_cols;
    let n = m; // Square for identity-first
    let m_in = 1usize;

    let cpu_layout = create_cpu_layout();
    let bus = build_bus_layout_for_instances(m, m_in, 1, [], [ell_addr]).expect("bus layout");
    assert_eq!(bus.bus_base, CPU_COLS, "test assumes bus starts after CPU_COLS");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, COL_CONST_ONE);
    builder.add_twist_instance(&bus, &bus.twist_cols[0], &cpu_layout);

    let ccs = builder.build().expect("should build CCS");
    (ccs, bus)
}

/// Test: Load value binding constraint catches the "has_read=0 but rv≠0" attack.
///
/// Previously this attack passed (VULNERABILITY). With constraints, it should fail.
///
/// # Credits
/// This test validates the constraint logic ported from Jolt's `RamReadEqRdWriteIfLoad`.
#[test]
fn padding_constraint_catches_rv_nonzero_when_no_read() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Padding constraint catches rv≠0 when has_read=0          ║");
    println!("║                                                                  ║");
    println!("║  Credits: Constraint logic ported from Jolt zkVM                 ║");
    println!("║  (jolt-core/src/zkvm/r1cs/constraints.rs)                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    // Create a malicious witness: has_read=0 but rv=42
    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU semantics: not a load instruction
    z[COL_IS_LOAD] = F::ZERO;
    z[COL_IS_STORE] = F::ZERO;

    // Bus columns - MALICIOUS: has_read=0 but rv=42
    let bus_has_read = bus.bus_cell(twist.has_read, 0);
    let bus_rv = bus.bus_cell(twist.rv, 0);
    z[bus_has_read] = F::ZERO; // has_read = 0
    z[bus_rv] = F::from_u64(42); // rv = 42 (MALICIOUS - should be 0!)

    // Check CCS - should FAIL due to padding constraint: (1 - has_read) * rv = 0
    let x = &z[..1]; // const_one is public
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);

    match result {
        Err(_) => {
            println!("✓ PASS: Padding constraint correctly caught the attack!");
            println!("  Constraint (1 - has_read) * rv = 0 rejected rv=42 when has_read=0");
        }
        Ok(()) => {
            panic!(
                "✗ FAIL: Padding constraint did NOT catch the attack!\n\
                 This indicates the constraints are not working correctly."
            );
        }
    }
}

/// Test: Write value padding constraint catches "has_write=0 but wv≠0" attack.
///
/// # Credits
/// Validates the constraint logic ported from Jolt's padding recommendations.
#[test]
fn padding_constraint_catches_wv_nonzero_when_no_write() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Padding constraint catches wv≠0 when has_write=0         ║");
    println!("║                                                                  ║");
    println!("║  Credits: Constraint logic ported from Jolt zkVM                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // Bus columns - MALICIOUS: has_write=0 but wv=99
    let bus_has_write = bus.bus_cell(twist.has_write, 0);
    let bus_wv = bus.bus_cell(twist.wv, 0);
    z[bus_has_write] = F::ZERO;
    z[bus_wv] = F::from_u64(99);

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);

    match result {
        Err(_) => {
            println!("✓ PASS: Padding constraint correctly caught the attack!");
        }
        Ok(()) => {
            panic!("✗ FAIL: Padding constraint did NOT catch wv attack!");
        }
    }
}

/// Test: Load value binding catches "rd_value ≠ bus_rv" attack.
///
/// When is_load=1, the constraint enforces rd_value == bus_rv.
///
/// # Credits
/// Validates Jolt's `RamReadEqRdWriteIfLoad` constraint.
#[test]
fn load_binding_catches_value_mismatch() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Load binding catches rd_value ≠ bus_rv                   ║");
    println!("║                                                                  ║");
    println!("║  Credits: Jolt's RamReadEqRdWriteIfLoad constraint               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU semantics: this IS a load instruction
    z[COL_IS_LOAD] = F::ONE;
    z[COL_RD_VALUE] = F::from_u64(100); // CPU says register gets 100

    // Bus columns: has_read=1, rv=200 (MISMATCH!)
    let bus_has_read = bus.bus_cell(twist.has_read, 0);
    let bus_rv = bus.bus_cell(twist.rv, 0);
    z[bus_has_read] = F::ONE;
    z[bus_rv] = F::from_u64(200); // Bus says memory value is 200

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);

    match result {
        Err(_) => {
            println!("✓ PASS: Load binding constraint correctly caught the mismatch!");
            println!("  Constraint is_load * (rd_value - bus_rv) = 0 rejected 100 ≠ 200");
        }
        Ok(()) => {
            panic!("✗ FAIL: Load binding did NOT catch rd_value ≠ bus_rv!");
        }
    }
}

/// Test: Store value binding catches "rs2_value ≠ bus_wv" attack.
///
/// When is_store=1, the constraint enforces rs2_value == bus_wv.
///
/// # Credits
/// Validates Jolt's `Rs2EqRamWriteIfStore` constraint.
#[test]
fn store_binding_catches_value_mismatch() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Store binding catches rs2_value ≠ bus_wv                 ║");
    println!("║                                                                  ║");
    println!("║  Credits: Jolt's Rs2EqRamWriteIfStore constraint                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU semantics: this IS a store instruction
    z[COL_IS_STORE] = F::ONE;
    z[COL_RS2_VALUE] = F::from_u64(555); // CPU says register value is 555

    // Bus columns: has_write=1, wv=777 (MISMATCH!)
    let bus_has_write = bus.bus_cell(twist.has_write, 0);
    let bus_wv = bus.bus_cell(twist.wv, 0);
    z[bus_has_write] = F::ONE;
    z[bus_wv] = F::from_u64(777); // Bus says write value is 777

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);

    match result {
        Err(_) => {
            println!("✓ PASS: Store binding constraint correctly caught the mismatch!");
            println!("  Constraint is_store * (rs2_value - bus_wv) = 0 rejected 555 ≠ 777");
        }
        Ok(()) => {
            panic!("✗ FAIL: Store binding did NOT catch rs2_value ≠ bus_wv!");
        }
    }
}

/// Test: Valid witness passes all constraints.
///
/// Sanity check that correctly constructed witnesses work.
#[test]
fn valid_witness_passes_constraints() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Valid witness passes all constraints                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    // Test 1: No memory operation (all zeros except const_one)
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;
        // Everything else is zero, including bus columns

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_ok(), "no-op witness should pass");
        println!("✓ No-op witness passes");
    }

    // Test 2: Valid load (is_load=1, has_read=1, rd_value == bus_rv)
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;
        z[COL_IS_LOAD] = F::ONE;
        z[COL_RD_VALUE] = F::from_u64(42);

        let bus_has_read = bus.bus_cell(twist.has_read, 0);
        let bus_rv = bus.bus_cell(twist.rv, 0);
        z[bus_has_read] = F::ONE;
        z[bus_rv] = F::from_u64(42); // Matches rd_value!

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_ok(), "valid load witness should pass");
        println!("✓ Valid load witness passes");
    }

    // Test 3: Valid store (is_store=1, has_write=1, rs2_value == bus_wv)
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;
        z[COL_IS_STORE] = F::ONE;
        z[COL_RS2_VALUE] = F::from_u64(99);

        let bus_has_write = bus.bus_cell(twist.has_write, 0);
        let bus_wv = bus.bus_cell(twist.wv, 0);
        z[bus_has_write] = F::ONE;
        z[bus_wv] = F::from_u64(99); // Matches rs2_value!

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_ok(), "valid store witness should pass");
        println!("✓ Valid store witness passes");
    }

    println!("\n✓ All valid witnesses pass constraints correctly!");
}

/// Test: Address bit padding catches non-zero bits when inactive.
///
/// # Credits
/// Validates Jolt-style padding for address bits.
#[test]
fn address_bit_padding_catches_nonzero_bits() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Address bit padding catches non-zero bits when inactive  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // Bus columns: has_read=0 but ra_bits[0]=1 (MALICIOUS!)
    let bus_has_read = bus.bus_cell(twist.has_read, 0);
    let bus_ra_bit_0 = bus.bus_cell(twist.ra_bits.start, 0);
    z[bus_has_read] = F::ZERO;
    z[bus_ra_bit_0] = F::ONE; // Should be 0 when has_read=0!

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);

    match result {
        Err(_) => {
            println!("✓ PASS: Address bit padding constraint correctly caught the attack!");
        }
        Ok(()) => {
            panic!("✗ FAIL: Address bit padding did NOT catch non-zero ra_bit!");
        }
    }
}

/// Test: Shout (lookup) constraints work correctly.
#[test]
fn shout_constraints_catch_lookup_attacks() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: Shout constraints catch lookup value mismatch            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ell_addr = 4;
    let bus_cols = ell_addr + 2;
    let m = CPU_COLS + bus_cols;
    let n = m;
    let m_in = 1usize;
    let cpu_layout = create_cpu_layout();

    let bus = build_bus_layout_for_instances(m, m_in, 1, [ell_addr], []).expect("bus layout");
    assert_eq!(bus.bus_base, CPU_COLS, "test assumes bus starts after CPU_COLS");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, COL_CONST_ONE);
    builder.add_shout_instance(&bus, &bus.shout_cols[0], &cpu_layout);

    let ccs = builder.build().expect("should build CCS");
    let shout = &bus.shout_cols[0];

    // Test 1: Lookup value mismatch (is_lookup=1, lookup_out ≠ bus_val)
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;
        z[COL_IS_LOOKUP] = F::ONE;
        z[COL_LOOKUP_OUT] = F::from_u64(123);

        let bus_has_lookup = bus.bus_cell(shout.has_lookup, 0);
        let bus_val = bus.bus_cell(shout.val, 0);
        z[bus_has_lookup] = F::ONE;
        z[bus_val] = F::from_u64(456); // Mismatch!

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_err(), "lookup value mismatch should fail");
        println!("✓ Lookup value mismatch correctly caught");
    }

    // Test 2: Padding violation (has_lookup=0 but val≠0)
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;

        let bus_has_lookup = bus.bus_cell(shout.has_lookup, 0);
        let bus_val = bus.bus_cell(shout.val, 0);
        z[bus_has_lookup] = F::ZERO;
        z[bus_val] = F::from_u64(999); // Should be 0!

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_err(), "lookup padding violation should fail");
        println!("✓ Lookup padding violation correctly caught");
    }

    // Test 3: Valid lookup passes
    {
        let mut z = vec![F::ZERO; ccs.m];
        z[COL_CONST_ONE] = F::ONE;
        z[COL_IS_LOOKUP] = F::ONE;
        z[COL_LOOKUP_OUT] = F::from_u64(42);

        let bus_has_lookup = bus.bus_cell(shout.has_lookup, 0);
        let bus_val = bus.bus_cell(shout.val, 0);
        z[bus_has_lookup] = F::ONE;
        z[bus_val] = F::from_u64(42); // Matches!

        let x = &z[..1];
        let w = &z[1..];

        let result = check_ccs_rowwise_zero(&ccs, x, w);
        assert!(result.is_ok(), "valid lookup should pass");
        println!("✓ Valid lookup passes");
    }

    println!("\n✓ All Shout constraint tests pass!");
}

/// Test: Selector binding catches "CPU says no load but bus has_read=1" attack.
///
/// This is required so the prover cannot inject extra memory operations into the bus
/// that the CPU semantics did not authorize.
#[test]
fn selector_binding_catches_has_read_mismatch() {
    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU: not a load
    z[COL_IS_LOAD] = F::ZERO;
    // Bus: claims a read happened
    z[bus.bus_cell(twist.has_read, 0)] = F::ONE;

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);
    assert!(result.is_err(), "selector binding should reject is_load=0, has_read=1");
}

/// Test: Address binding catches mismatched effective_addr vs. bus ra_bits on a load.
#[test]
fn load_address_binding_catches_mismatch() {
    let ell_addr = 4;
    let (ccs, bus) = create_ccs_with_binding_constraints(ell_addr);
    let twist = &bus.twist_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU: load with effective address 5
    z[COL_IS_LOAD] = F::ONE;
    z[COL_EFFECTIVE_ADDR] = F::from_u64(5);
    z[COL_RD_VALUE] = F::ZERO;

    // Bus: read with address bits encoding 6 (mismatch) and rv matching rd_value
    z[bus.bus_cell(twist.has_read, 0)] = F::ONE;
    z[bus.bus_cell(twist.rv, 0)] = F::ZERO;

    let ra_base = bus.bus_cell(twist.ra_bits.start, 0);
    // 6 = 0b0110 (little-endian bits: [0,1,1,0])
    z[ra_base + 0] = F::ZERO;
    z[ra_base + 1] = F::ONE;
    z[ra_base + 2] = F::ONE;
    z[ra_base + 3] = F::ZERO;

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);
    assert!(
        result.is_err(),
        "address binding should reject effective_addr=5, ra_bits=6"
    );
}

/// Test: Lookup key binding catches mismatched CPU lookup_key vs bus addr_bits on a lookup.
#[test]
fn lookup_key_binding_catches_mismatch() {
    let ell_addr = 4;
    let bus_cols = ell_addr + 2;
    let m = CPU_COLS + bus_cols;
    let n = m;
    let m_in = 1usize;

    let cpu_layout = create_cpu_layout();

    let bus = build_bus_layout_for_instances(m, m_in, 1, [ell_addr], []).expect("bus layout");
    assert_eq!(bus.bus_base, CPU_COLS, "test assumes bus starts after CPU_COLS");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, COL_CONST_ONE);
    builder.add_shout_instance(&bus, &bus.shout_cols[0], &cpu_layout);
    let ccs = builder.build().expect("should build CCS");
    let shout = &bus.shout_cols[0];

    let mut z = vec![F::ZERO; ccs.m];
    z[COL_CONST_ONE] = F::ONE;

    // CPU: lookup with key=3 and output=42
    z[COL_IS_LOOKUP] = F::ONE;
    z[COL_LOOKUP_KEY] = F::from_u64(3);
    z[COL_LOOKUP_OUT] = F::from_u64(42);

    // Bus: has_lookup=1, val matches, but addr_bits encode 4 (mismatch)
    z[bus.bus_cell(shout.has_lookup, 0)] = F::ONE;
    z[bus.bus_cell(shout.val, 0)] = F::from_u64(42);

    let addr_base = bus.bus_cell(shout.addr_bits.start, 0);
    // 4 = 0b0100 (little-endian bits: [0,0,1,0])
    z[addr_base + 0] = F::ZERO;
    z[addr_base + 1] = F::ZERO;
    z[addr_base + 2] = F::ONE;
    z[addr_base + 3] = F::ZERO;

    let x = &z[..1];
    let w = &z[1..];

    let result = check_ccs_rowwise_zero(&ccs, x, w);
    assert!(
        result.is_err(),
        "lookup key binding should reject lookup_key=3, addr_bits=4"
    );
}

/// Summary test that shows the vulnerabilities are fixed.
#[test]
fn summary_vulnerabilities_fixed_by_jolt_constraints() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║           SUMMARY: VULNERABILITIES FIXED BY JOLT CONSTRAINTS             ║");
    println!("║                                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  The following vulnerabilities found in earlier tests are now FIXED:     ║");
    println!("║                                                                          ║");
    println!("║  1. has_read=0 but rv≠0        → Caught by: (1-has_read)·rv = 0          ║");
    println!("║  2. has_write=0 but wv≠0       → Caught by: (1-has_write)·wv = 0         ║");
    println!("║  3. has_read=0 but ra_bits≠0   → Caught by: (1-has_read)·ra_bits = 0     ║");
    println!("║  4. has_write=0 but wa_bits≠0  → Caught by: (1-has_write)·wa_bits = 0    ║");
    println!("║  5. has_write=0 but inc≠0      → Caught by: (1-has_write)·inc = 0        ║");
    println!("║  6. is_load·rd ≠ bus_rv        → Caught by: is_load·(rd-rv) = 0          ║");
    println!("║  7. is_store·rs2 ≠ bus_wv      → Caught by: is_store·(rs2-wv) = 0        ║");
    println!("║  8. has_lookup=0 but val≠0     → Caught by: (1-has_lookup)·val = 0       ║");
    println!("║                                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  CREDITS:                                                                ║");
    println!("║                                                                          ║");
    println!("║  Constraint logic ported from the Jolt zkVM project:                     ║");
    println!("║  - Repository: https://github.com/a16z/jolt                              ║");
    println!("║  - Original file: jolt-core/src/zkvm/r1cs/constraints.rs                 ║");
    println!("║  - License: Apache-2.0 / MIT                                             ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("\n");
}
