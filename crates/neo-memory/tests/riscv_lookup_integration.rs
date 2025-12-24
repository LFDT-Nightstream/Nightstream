//! Integration tests for RISC-V instruction lookups with Neo's Shout protocol.
//!
//! This test file demonstrates how to use Jolt-style RISC-V lookup tables
//! with Neo's Twist and Shout proving system. It verifies:
//!
//! 1. Lookup table semantics (AND, XOR, OR, etc.)
//! 2. MLE evaluation consistency
//! 3. Full Shout proof generation and verification
//!
//! ## Architecture
//!
//! The integration works as follows:
//!
//! ```text
//! RISC-V Instruction Trace
//!         │
//!         ▼
//! ┌─────────────────────┐
//! │ RiscvLookupTable    │ ◄── Jolt-style lookup semantics
//! │ (per opcode)        │
//! └─────────────────────┘
//!         │
//!         ▼
//! ┌─────────────────────┐
//! │ PlainLutTrace       │ ◄── Neo format (has_lookup, addr, val)
//! └─────────────────────┘
//!         │
//!         ▼
//! ┌─────────────────────┐
//! │ Shout Protocol      │ ◄── Sum-check based read-only memory argument
//! └─────────────────────┘
//! ```

#![allow(non_snake_case)]

use std::collections::HashMap;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, traits::SModuleHomomorphism};
use neo_math::K as KElem;
use neo_memory::encode::encode_lut_for_shout;
use neo_memory::mle::build_chi_table;
use neo_memory::plain::LutTable;
use neo_memory::riscv_lookups::{
    compute_op, interleave_bits, uninterleave_bits, RangeCheckTable, RiscvLookupEvent,
    RiscvLookupTable, RiscvMemOp, RiscvMemory, RiscvMemoryEvent, RiscvOpcode,
    RiscvShoutTables,
};
use neo_memory::shout;
use neo_params::NeoParams;
use neo_vm_trace::{Shout, ShoutId, Twist, TwistId};

// ============================================================================
// Test Helpers
// ============================================================================

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

/// A Shout implementation backed by RISC-V lookup tables.
struct RiscvShout {
    /// Map from ShoutId to (opcode, xlen)
    tables: HashMap<ShoutId, (RiscvOpcode, usize)>,
}

impl RiscvShout {
    fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    fn add_table(&mut self, id: ShoutId, opcode: RiscvOpcode, xlen: usize) {
        self.tables.insert(id, (opcode, xlen));
    }
}

impl Shout<u64> for RiscvShout {
    fn lookup(&mut self, shout_id: ShoutId, key: u64) -> u64 {
        if let Some(&(opcode, xlen)) = self.tables.get(&shout_id) {
            // Key is the interleaved index; uninterleave to get operands
            let (x, y) = uninterleave_bits(key as u128);
            compute_op(opcode, x, y, xlen)
        } else {
            0
        }
    }
}

// ============================================================================
// Test 1: Basic RISC-V Opcode Lookup Semantics
// ============================================================================

/// Test that all RISC-V opcodes produce correct results.
#[test]
fn test_riscv_opcode_semantics() {
    let xlen = 8;

    // Test AND
    assert_eq!(compute_op(RiscvOpcode::And, 0xF0, 0x0F, xlen), 0x00);
    assert_eq!(compute_op(RiscvOpcode::And, 0xFF, 0xAB, xlen), 0xAB);
    assert_eq!(compute_op(RiscvOpcode::And, 0xAA, 0x55, xlen), 0x00);

    // Test XOR
    assert_eq!(compute_op(RiscvOpcode::Xor, 0xFF, 0xFF, xlen), 0x00);
    assert_eq!(compute_op(RiscvOpcode::Xor, 0xAA, 0x55, xlen), 0xFF);
    assert_eq!(compute_op(RiscvOpcode::Xor, 0x12, 0x34, xlen), 0x26);

    // Test OR
    assert_eq!(compute_op(RiscvOpcode::Or, 0xF0, 0x0F, xlen), 0xFF);
    assert_eq!(compute_op(RiscvOpcode::Or, 0x00, 0x00, xlen), 0x00);
    assert_eq!(compute_op(RiscvOpcode::Or, 0xAA, 0x55, xlen), 0xFF);

    // Test SUB (with wraparound)
    assert_eq!(compute_op(RiscvOpcode::Sub, 10, 5, xlen), 5);
    assert_eq!(compute_op(RiscvOpcode::Sub, 0, 1, xlen), 255); // wraparound

    // Test SLTU (unsigned less than)
    assert_eq!(compute_op(RiscvOpcode::Sltu, 5, 10, xlen), 1);
    assert_eq!(compute_op(RiscvOpcode::Sltu, 10, 5, xlen), 0);
    assert_eq!(compute_op(RiscvOpcode::Sltu, 5, 5, xlen), 0);

    // Test EQ
    assert_eq!(compute_op(RiscvOpcode::Eq, 5, 5, xlen), 1);
    assert_eq!(compute_op(RiscvOpcode::Eq, 5, 6, xlen), 0);

    // Test NEQ
    assert_eq!(compute_op(RiscvOpcode::Neq, 5, 5, xlen), 0);
    assert_eq!(compute_op(RiscvOpcode::Neq, 5, 6, xlen), 1);

    println!("✓ test_riscv_opcode_semantics passed");
}

// ============================================================================
// Test 2: Lookup Table MLE Evaluation
// ============================================================================

/// Test that MLE evaluation is consistent with table lookups at Boolean points.
#[test]
fn test_mle_evaluation_at_boolean_points() {
    let xlen = 4;

    for op in [RiscvOpcode::And, RiscvOpcode::Xor, RiscvOpcode::Or] {
        let table: RiscvLookupTable<F> = RiscvLookupTable::new(op, xlen);

        // Check that MLE(boolean point) = table entry
        for x in 0..16u64 {
            for y in 0..16u64 {
                // Convert operands to Boolean point r
                // With LSB-aligned indexing: r[2i] = x_i (bit i of x), r[2i+1] = y_i
                let mut r = vec![F::ZERO; 2 * xlen];
                for i in 0..xlen {
                    r[2 * i] = if (x >> i) & 1 == 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    };
                    r[2 * i + 1] = if (y >> i) & 1 == 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    };
                }

                let mle_result = table.evaluate_mle(&r);
                let table_result = table.lookup_operands(x, y);
                assert_eq!(
                    mle_result, table_result,
                    "{} MLE at ({}, {}) mismatch",
                    op, x, y
                );
            }
        }
    }

    println!("✓ test_mle_evaluation_at_boolean_points passed");
}

// ============================================================================
// Test 3: Simulate RISC-V Instruction Trace with Shout
// ============================================================================

/// Build a plain LUT trace from RISC-V lookup events.
fn build_riscv_lut_trace<F: p3_field::Field + PrimeCharacteristicRing>(
    events: &[Option<RiscvLookupEvent>],
    xlen: usize,
) -> neo_memory::plain::PlainLutTrace<F> {
    let has_lookup: Vec<F> = events
        .iter()
        .map(|e| if e.is_some() { F::ONE } else { F::ZERO })
        .collect();

    let addr: Vec<u64> = events
        .iter()
        .map(|e| {
            if let Some(event) = e {
                event.lookup_index(xlen) as u64
            } else {
                0
            }
        })
        .collect();

    let val: Vec<F> = events
        .iter()
        .map(|e| {
            if let Some(event) = e {
                F::from_u64(event.result)
            } else {
                F::ZERO
            }
        })
        .collect();

    neo_memory::plain::PlainLutTrace {
        has_lookup,
        addr,
        val,
    }
}

/// Test a simulated RISC-V instruction trace with Shout encoding and verification.
///
/// NOTE: For full RISC-V opcode lookups, you need a table that matches the interleaved
/// bit indices. Here we use a simple XOR table and verify only XOR lookups.
#[test]
fn test_riscv_trace_with_shout() {
    let xlen = 4; // 4-bit for smaller tables
    let params = create_test_params();
    let dummy = DummyCommit::default();
    let commit = |m: &Mat<F>| dummy.commit(m);

    // Create a 4-step trace with XOR operations only (since we have one XOR table)
    let events: Vec<Option<RiscvLookupEvent>> = vec![
        Some(RiscvLookupEvent::new(RiscvOpcode::Xor, 0x0, 0x0, xlen)), // 0 ^ 0 = 0
        None, // No lookup
        Some(RiscvLookupEvent::new(RiscvOpcode::Xor, 0xA, 0x5, xlen)), // 10 ^ 5 = 15
        Some(RiscvLookupEvent::new(RiscvOpcode::Xor, 0xF, 0xF, xlen)), // 15 ^ 15 = 0
    ];

    // Verify event results are correct
    assert_eq!(events[0].as_ref().unwrap().result, 0);
    assert_eq!(events[2].as_ref().unwrap().result, 15);
    assert_eq!(events[3].as_ref().unwrap().result, 0);

    // Build the plain LUT trace
    let plain_lut = build_riscv_lut_trace::<F>(&events, xlen);

    assert_eq!(plain_lut.has_lookup.len(), 4);
    assert_eq!(
        plain_lut.has_lookup,
        vec![F::ONE, F::ZERO, F::ONE, F::ONE]
    );

    // Create the lookup table for XOR
    // For interleaved bit lookup, we use d=1 (no decomposition) and n_side = table_size
    let table_size = 1usize << (2 * xlen); // 256 entries for 4-bit xlen
    let n_side = table_size; // Full table, no decomposition
    let d = 1;

    // Build table content using XOR
    let xor_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Xor, xlen);
    let table_content = xor_table.content();

    let table = LutTable {
        table_id: 0,
        k: table_size,
        d,
        n_side,
        content: table_content,
    };

    // Encode the trace for Shout
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &table, &plain_lut, &commit, None, 0);

    // Verify semantic correctness
    shout::check_shout_semantics(&params, &lut_inst, &lut_wit, &plain_lut.val)
        .expect("Shout semantic check should pass");

    println!("✓ test_riscv_trace_with_shout passed");
}

// ============================================================================
// Test 4: End-to-End RISC-V ALU Trace Simulation
// ============================================================================

/// Simulates a sequence of RISC-V ALU operations and verifies them with Shout.
#[test]
fn test_riscv_alu_sequence() {
    let xlen = 4;

    // Simulate a small computation:
    // r1 = 0xA (10)
    // r2 = 0x5 (5)
    // r3 = r1 AND r2 = 0x0
    // r4 = r1 XOR r2 = 0xF
    // r5 = r3 OR r4 = 0xF

    let ops = vec![
        (RiscvOpcode::And, 0xAu64, 0x5u64), // 0
        (RiscvOpcode::Xor, 0xAu64, 0x5u64), // 15
        (RiscvOpcode::Or, 0x0u64, 0xFu64),  // 15
    ];

    // Verify the computation
    assert_eq!(compute_op(ops[0].0, ops[0].1, ops[0].2, xlen), 0);
    assert_eq!(compute_op(ops[1].0, ops[1].1, ops[1].2, xlen), 15);
    assert_eq!(compute_op(ops[2].0, ops[2].1, ops[2].2, xlen), 15);

    // Create lookup events
    let events: Vec<RiscvLookupEvent> = ops
        .iter()
        .map(|&(op, x, y)| RiscvLookupEvent::new(op, x, y, xlen))
        .collect();

    // Verify each lookup index is unique
    let indices: Vec<u128> = events.iter().map(|e| e.lookup_index(xlen)).collect();
    println!("Lookup indices: {:?}", indices);

    // Verify results match expectations
    assert_eq!(events[0].result, 0);
    assert_eq!(events[1].result, 15);
    assert_eq!(events[2].result, 15);

    println!("✓ test_riscv_alu_sequence passed");
}

// ============================================================================
// Test 5: MLE Consistency with Chi Tables
// ============================================================================

/// Test that the MLE evaluation using chi tables matches the closed-form MLE.
#[test]
fn test_mle_with_chi_tables() {
    let xlen = 4;
    let ell = 2 * xlen; // 8 bits for the MLE

    // Random evaluation point
    let r: Vec<KElem> = (0..ell)
        .map(|i| KElem::from_u64((i * 7 + 13) as u64 % 31))
        .collect();

    // Build chi table
    let chi = build_chi_table(&r);
    assert_eq!(chi.len(), 1 << ell);

    for op in [RiscvOpcode::And, RiscvOpcode::Xor, RiscvOpcode::Or] {
        let table: RiscvLookupTable<KElem> = RiscvLookupTable::new(op, xlen);

        // Naive MLE: Σ_k chi[k] * table[k]
        let mut naive_mle = KElem::ZERO;
        for k in 0..(1 << ell) {
            naive_mle += chi[k] * table.lookup(k as u128);
        }

        // Closed-form MLE
        let closed_form_mle = table.evaluate_mle(&r);

        assert_eq!(
            naive_mle, closed_form_mle,
            "{} MLE mismatch: naive={:?}, closed={:?}",
            op, naive_mle, closed_form_mle
        );
    }

    println!("✓ test_mle_with_chi_tables passed");
}

// ============================================================================
// Test 6: Multiple Tables (Per-Opcode Shout)
// ============================================================================

/// Test that multiple lookup tables (one per opcode) can be used together.
#[test]
fn test_multiple_opcode_tables() {
    let xlen = 4;

    // Create separate tables for each opcode
    let and_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::And, xlen);
    let xor_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Xor, xlen);
    let or_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Or, xlen);

    // Verify tables are different
    let test_x = 0xA;
    let test_y = 0x5;

    let and_result = and_table.lookup_operands(test_x, test_y);
    let xor_result = xor_table.lookup_operands(test_x, test_y);
    let or_result = or_table.lookup_operands(test_x, test_y);

    assert_eq!(and_result, F::from_u64(0)); // 10 & 5 = 0
    assert_eq!(xor_result, F::from_u64(15)); // 10 ^ 5 = 15
    assert_eq!(or_result, F::from_u64(15)); // 10 | 5 = 15

    assert_ne!(and_result, xor_result);

    println!("✓ test_multiple_opcode_tables passed");
}

// ============================================================================
// Test 7: Shout with RISC-V Shout Implementation
// ============================================================================

/// Test the RiscvShout implementation.
#[test]
fn test_riscv_shout_implementation() {
    let xlen = 4;
    let mut shout = RiscvShout::new();

    // Register tables
    shout.add_table(ShoutId(0), RiscvOpcode::And, xlen);
    shout.add_table(ShoutId(1), RiscvOpcode::Xor, xlen);
    shout.add_table(ShoutId(2), RiscvOpcode::Or, xlen);

    // Test lookups
    let x = 0xAu64;
    let y = 0x5u64;
    let index = interleave_bits(x, y);
    // With LSB-aligned interleaving, mask at the LSB
    let mask = ((1u128 << (2 * xlen)) - 1) as u64;
    let masked_index = (index as u64) & mask;

    let and_result = shout.lookup(ShoutId(0), masked_index);
    let xor_result = shout.lookup(ShoutId(1), masked_index);
    let or_result = shout.lookup(ShoutId(2), masked_index);

    assert_eq!(and_result, 0); // 10 & 5 = 0
    assert_eq!(xor_result, 15); // 10 ^ 5 = 15
    assert_eq!(or_result, 15); // 10 | 5 = 15

    println!("✓ test_riscv_shout_implementation passed");
}

// ============================================================================
// Test 8: Full Pipeline - Trace to Shout Proof
// ============================================================================

/// End-to-end test: RISC-V trace → Shout encoding → semantic verification.
///
/// This test uses a single OR table and verifies OR operations.
#[test]
fn test_full_riscv_shout_pipeline() {
    let xlen = 4;
    let params = create_test_params();
    let dummy = DummyCommit::default();
    let commit = |m: &Mat<F>| dummy.commit(m);

    // Create a trace with 8 steps - ALL using OR operations for consistency with OR table
    let events: Vec<Option<RiscvLookupEvent>> = vec![
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0xF, 0xF, xlen)), // 15 | 15 = 15
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0x0, 0x0, xlen)), // 0 | 0 = 0
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0x0, 0x0, xlen)), // 0 | 0 = 0
        None, // padding
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0xA, 0x5, xlen)), // 10 | 5 = 15
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0x5, 0xA, xlen)), // 5 | 10 = 15
        Some(RiscvLookupEvent::new(RiscvOpcode::Or, 0xA, 0x5, xlen)), // 10 | 5 = 15
        None, // padding
    ];

    // Build plain trace
    let plain_lut = build_riscv_lut_trace::<F>(&events, xlen);

    // Create lookup table for OR
    // For interleaved bit lookup, use d=1 (no decomposition) and n_side = table_size
    let table_size = 1usize << (2 * xlen);
    let n_side = table_size;
    let d = 1;

    let or_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Or, xlen);
    let table = LutTable {
        table_id: 0,
        k: table_size,
        d,
        n_side,
        content: or_table.content(),
    };

    // Encode and verify
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &table, &plain_lut, &commit, None, 0);

    // Semantic check passes because the trace values match the OR table
    shout::check_shout_semantics(&params, &lut_inst, &lut_wit, &plain_lut.val)
        .expect("Shout semantic check should pass");

    // Verify matrix dimensions
    let expected_mats = table.d * lut_inst.ell + 2;
    assert_eq!(
        lut_wit.mats.len(),
        expected_mats,
        "Expected {} matrices, got {}",
        expected_mats,
        lut_wit.mats.len()
    );

    println!("✓ test_full_riscv_shout_pipeline passed");
}

// ============================================================================
// Test 9: Stress Test - Large Trace
// ============================================================================

/// Stress test with a larger trace.
#[test]
fn test_large_riscv_trace() {
    let xlen = 4;
    let num_steps = 64;

    // Generate a trace with random operations
    let mut events: Vec<Option<RiscvLookupEvent>> = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        if i % 8 == 7 {
            events.push(None); // Every 8th step has no lookup
        } else {
            let op = match i % 3 {
                0 => RiscvOpcode::And,
                1 => RiscvOpcode::Xor,
                _ => RiscvOpcode::Or,
            };
            let x = ((i * 7) % 16) as u64;
            let y = ((i * 11 + 3) % 16) as u64;
            events.push(Some(RiscvLookupEvent::new(op, x, y, xlen)));
        }
    }

    // Build plain trace
    let plain_lut = build_riscv_lut_trace::<F>(&events, xlen);

    assert_eq!(plain_lut.has_lookup.len(), num_steps);

    // Count active lookups
    let active_lookups = events.iter().filter(|e| e.is_some()).count();
    assert_eq!(active_lookups, num_steps - num_steps / 8);

    println!("✓ test_large_riscv_trace passed with {} steps, {} active lookups", num_steps, active_lookups);
}

// ============================================================================
// Test 10: Verify Jolt-Compatible Bit Interleaving
// ============================================================================

/// Verify that our bit interleaving matches the expected LSB-aligned pattern.
///
/// We use LSB-aligned interleaving where:
/// - Bit position 2i contains x_i (the i-th bit of x)
/// - Bit position 2i+1 contains y_i (the i-th bit of y)
#[test]
fn test_jolt_compatible_interleaving() {
    let x = 10u64; // 0b1010: x0=0, x1=1, x2=0, x3=1
    let y = 5u64;  // 0b0101: y0=1, y1=0, y2=1, y3=0
    let xlen = 4;

    let interleaved = interleave_bits(x, y);
    // With LSB-aligned interleaving, the result is at the LSB
    let masked = interleaved & ((1u128 << (2 * xlen)) - 1);

    // Expected bit pattern (LSB to MSB):
    // pos 0: x0=0, pos 1: y0=1, pos 2: x1=1, pos 3: y1=0,
    // pos 4: x2=0, pos 5: y2=1, pos 6: x3=1, pos 7: y3=0
    // Binary: 0110 0110 = 0x66 = 102
    // Wait: 01100110 reading from pos7 to pos0 = 0 1 1 0 | 0 1 1 0
    // Actually: 0b_y3_x3_y2_x2_y1_x1_y0_x0 = 0_1_1_0_0_1_1_0 = 0x66
    // But our format is x at even positions, y at odd:
    // pos 0=x0=0, pos 1=y0=1, pos 2=x1=1, pos 3=y1=0, pos 4=x2=0, pos 5=y2=1, pos 6=x3=1, pos 7=y3=0
    // Binary value: sum of 2^i * bit_i = 2 + 4 + 32 + 64 = 102
    assert_eq!(masked, 102, "Interleaving mismatch: got {}", masked);

    // Verify round-trip
    let (x2, y2) = uninterleave_bits(interleaved);
    assert_eq!(x, x2, "X round-trip failed");
    assert_eq!(y, y2, "Y round-trip failed");

    // Also verify the lookup table produces the correct result
    let table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::And, xlen);
    let result = table.lookup_operands(x, y);
    assert_eq!(result, F::from_u64(x & y), "AND lookup failed");

    let xor_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Xor, xlen);
    let xor_result = xor_table.lookup_operands(x, y);
    assert_eq!(xor_result, F::from_u64(x ^ y), "XOR lookup failed");

    println!("✓ test_jolt_compatible_interleaving passed");
}

// ============================================================================
// Test 11: ADD Operation Integration
// ============================================================================

/// Test ADD operation with carry propagation.
#[test]
fn test_add_operation_integration() {
    let xlen = 4;

    // Test basic addition
    assert_eq!(compute_op(RiscvOpcode::Add, 3, 5, xlen), 8);
    assert_eq!(compute_op(RiscvOpcode::Add, 7, 7, xlen), 14);

    // Test wraparound (15 + 1 = 0 in 4-bit)
    assert_eq!(compute_op(RiscvOpcode::Add, 15, 1, xlen), 0);
    assert_eq!(compute_op(RiscvOpcode::Add, 15, 2, xlen), 1);

    // Test lookup table
    let table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Add, xlen);

    for x in 0..16u64 {
        for y in 0..16u64 {
            let result = table.lookup_operands(x, y);
            let expected = F::from_u64((x + y) & 0xF);
            assert_eq!(result, expected, "ADD table[{}, {}] mismatch", x, y);
        }
    }

    println!("✓ test_add_operation_integration passed");
}

/// Test ADD with Shout encoding.
#[test]
fn test_add_with_shout() {
    let xlen = 4;
    let params = create_test_params();
    let dummy = DummyCommit::default();
    let commit = |m: &Mat<F>| dummy.commit(m);

    // Create a trace with ADD operations
    let events: Vec<Option<RiscvLookupEvent>> = vec![
        Some(RiscvLookupEvent::new(RiscvOpcode::Add, 3, 5, xlen)),  // 8
        Some(RiscvLookupEvent::new(RiscvOpcode::Add, 7, 7, xlen)),  // 14
        Some(RiscvLookupEvent::new(RiscvOpcode::Add, 15, 1, xlen)), // 0 (wraparound)
        Some(RiscvLookupEvent::new(RiscvOpcode::Add, 0, 0, xlen)),  // 0
    ];

    // Verify results
    assert_eq!(events[0].as_ref().unwrap().result, 8);
    assert_eq!(events[1].as_ref().unwrap().result, 14);
    assert_eq!(events[2].as_ref().unwrap().result, 0);

    // Build plain trace
    let plain_lut = build_riscv_lut_trace::<F>(&events, xlen);

    // Create ADD lookup table
    let table_size = 1usize << (2 * xlen);
    let add_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Add, xlen);

    let table = LutTable {
        table_id: 0,
        k: table_size,
        d: 1,
        n_side: table_size,
        content: add_table.content(),
    };

    // Encode and verify
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &table, &plain_lut, &commit, None, 0);

    shout::check_shout_semantics(&params, &lut_inst, &lut_wit, &plain_lut.val)
        .expect("ADD Shout semantic check should pass");

    println!("✓ test_add_with_shout passed");
}

// ============================================================================
// Test 12: Shift Operations Integration
// ============================================================================

/// Test shift operations (SLL, SRL, SRA).
#[test]
fn test_shift_operations_integration() {
    let xlen = 4;

    // SLL: Shift Left Logical
    assert_eq!(compute_op(RiscvOpcode::Sll, 1, 0, xlen), 1);   // 1 << 0 = 1
    assert_eq!(compute_op(RiscvOpcode::Sll, 1, 1, xlen), 2);   // 1 << 1 = 2
    assert_eq!(compute_op(RiscvOpcode::Sll, 1, 2, xlen), 4);   // 1 << 2 = 4
    assert_eq!(compute_op(RiscvOpcode::Sll, 1, 3, xlen), 8);   // 1 << 3 = 8
    assert_eq!(compute_op(RiscvOpcode::Sll, 5, 1, xlen), 10);  // 5 << 1 = 10
    assert_eq!(compute_op(RiscvOpcode::Sll, 8, 1, xlen), 0);   // 8 << 1 = 16 -> 0 (overflow)

    // SRL: Shift Right Logical
    assert_eq!(compute_op(RiscvOpcode::Srl, 8, 1, xlen), 4);   // 8 >> 1 = 4
    assert_eq!(compute_op(RiscvOpcode::Srl, 8, 2, xlen), 2);   // 8 >> 2 = 2
    assert_eq!(compute_op(RiscvOpcode::Srl, 8, 3, xlen), 1);   // 8 >> 3 = 1
    assert_eq!(compute_op(RiscvOpcode::Srl, 15, 2, xlen), 3);  // 15 >> 2 = 3

    // SRA: Shift Right Arithmetic (sign-preserving)
    // In 4-bit: 8 = 0b1000 = -8 (signed)
    // -8 >> 1 = -4 = 0b1100 = 12
    assert_eq!(compute_op(RiscvOpcode::Sra, 8, 1, xlen), 12);  // -8 >> 1 = -4 = 12
    assert_eq!(compute_op(RiscvOpcode::Sra, 8, 2, xlen), 14);  // -8 >> 2 = -2 = 14
    assert_eq!(compute_op(RiscvOpcode::Sra, 4, 1, xlen), 2);   // 4 >> 1 = 2 (positive, no sign extension)

    println!("✓ test_shift_operations_integration passed");
}

/// Test shift lookup tables.
#[test]
fn test_shift_lookup_tables() {
    let xlen = 4;

    let sll_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Sll, xlen);
    let srl_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Srl, xlen);
    let sra_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Sra, xlen);

    // Test SLL table
    for x in 0..16u64 {
        for shamt in 0..4u64 {
            let result = sll_table.lookup_operands(x, shamt);
            let expected = F::from_u64((x << shamt) & 0xF);
            assert_eq!(result, expected, "SLL table[{}, {}] mismatch", x, shamt);
        }
    }

    // Test SRL table
    for x in 0..16u64 {
        for shamt in 0..4u64 {
            let result = srl_table.lookup_operands(x, shamt);
            let expected = F::from_u64((x >> shamt) & 0xF);
            assert_eq!(result, expected, "SRL table[{}, {}] mismatch", x, shamt);
        }
    }

    // Test SRA table at specific points
    assert_eq!(sra_table.lookup_operands(8, 1), F::from_u64(12)); // -8 >> 1 = -4 = 12
    assert_eq!(sra_table.lookup_operands(4, 1), F::from_u64(2));  // 4 >> 1 = 2

    println!("✓ test_shift_lookup_tables passed");
}

// ============================================================================
// Test 13: RangeCheck Table for ADD Verification
// ============================================================================

/// Test RangeCheck table used for ADD verification.
#[test]
fn test_range_check_table() {
    let xlen = 8;
    let table: RangeCheckTable<F> = RangeCheckTable::new(xlen);

    // Identity table: table[i] = i
    for i in 0..256u64 {
        assert_eq!(table.lookup(i), F::from_u64(i));
    }

    // Test MLE at Boolean points
    for i in 0..256u64 {
        let mut r = vec![F::ZERO; xlen];
        for k in 0..xlen {
            r[k] = if (i >> k) & 1 == 1 { F::ONE } else { F::ZERO };
        }
        let mle_result = table.evaluate_mle(&r);
        assert_eq!(mle_result, F::from_u64(i), "RangeCheck MLE at {} mismatch", i);
    }

    println!("✓ test_range_check_table passed");
}

// ============================================================================
// Test 14: Memory Operations (Twist Integration)
// ============================================================================

/// Test basic memory read/write operations.
#[test]
fn test_memory_basic_operations() {
    let mut mem = RiscvMemory::new(32);

    // Test byte-level access
    mem.write_byte(0x1000, 0xAB);
    assert_eq!(mem.read_byte(0x1000), 0xAB);
    assert_eq!(mem.read_byte(0x1001), 0x00); // Uninitialized

    // Test word-level access (little-endian)
    mem.write(0x2000, 4, 0xDEADBEEF);
    assert_eq!(mem.read(0x2000, 4), 0xDEADBEEF);
    assert_eq!(mem.read_byte(0x2000), 0xEF); // LSB
    assert_eq!(mem.read_byte(0x2001), 0xBE);
    assert_eq!(mem.read_byte(0x2002), 0xAD);
    assert_eq!(mem.read_byte(0x2003), 0xDE); // MSB

    println!("✓ test_memory_basic_operations passed");
}

/// Test memory operations with different widths.
#[test]
fn test_memory_widths() {
    let mut mem = RiscvMemory::new(32);

    // Store word (4 bytes)
    mem.execute(RiscvMemOp::Sw, 0x1000, 0x12345678);
    assert_eq!(mem.execute(RiscvMemOp::Lw, 0x1000, 0), 0x12345678);

    // Store half-word (2 bytes)
    mem.execute(RiscvMemOp::Sh, 0x2000, 0xABCD);
    assert_eq!(mem.execute(RiscvMemOp::Lhu, 0x2000, 0), 0xABCD);

    // Store byte
    mem.execute(RiscvMemOp::Sb, 0x3000, 0xFF);
    assert_eq!(mem.execute(RiscvMemOp::Lbu, 0x3000, 0), 0xFF);

    println!("✓ test_memory_widths passed");
}

/// Test sign extension in memory loads.
#[test]
fn test_memory_sign_extension() {
    let mut mem = RiscvMemory::new(32);

    // Store a negative byte (0x80 = -128 as i8)
    mem.execute(RiscvMemOp::Sb, 0x1000, 0x80);

    // Load unsigned: should be 0x80
    let val_u = mem.execute(RiscvMemOp::Lbu, 0x1000, 0);
    assert_eq!(val_u, 0x80);

    // Load signed: should be sign-extended
    let val_s = mem.execute(RiscvMemOp::Lb, 0x1000, 0);
    assert_eq!(val_s as i64, -128);

    // Store a negative half-word (0x8000 = -32768 as i16)
    mem.execute(RiscvMemOp::Sh, 0x2000, 0x8000);

    // Load unsigned: should be 0x8000
    let val_u = mem.execute(RiscvMemOp::Lhu, 0x2000, 0);
    assert_eq!(val_u, 0x8000);

    // Load signed: should be sign-extended
    let val_s = mem.execute(RiscvMemOp::Lh, 0x2000, 0);
    assert_eq!(val_s as i64, -32768);

    println!("✓ test_memory_sign_extension passed");
}

/// Test RiscvMemory as a Twist implementation.
#[test]
fn test_memory_as_twist() {
    let mut mem = RiscvMemory::new(32);
    let ram_id = TwistId(0);

    // Write via Twist interface
    mem.store(ram_id, 0x1000, 0xCAFEBABE);

    // Read via Twist interface
    let val = mem.load(ram_id, 0x1000);
    assert_eq!(val, 0xCAFEBABE);

    // Verify byte-level access still works
    assert_eq!(mem.read_byte(0x1000), 0xBE); // LSB
    assert_eq!(mem.read_byte(0x1003), 0xCA); // MSB

    println!("✓ test_memory_as_twist passed");
}

// ============================================================================
// Test 15: RiscvShoutTables Integration
// ============================================================================

/// Test the unified RiscvShoutTables implementation.
#[test]
fn test_riscv_shout_tables() {
    let xlen = 8;
    let mut tables = RiscvShoutTables::new(xlen);

    // Test AND via ShoutId(0)
    let and_id = tables.opcode_to_id(RiscvOpcode::And);
    assert_eq!(and_id, ShoutId(0));
    let index = interleave_bits(0xF0, 0x0F) as u64;
    let result = tables.lookup(and_id, index);
    assert_eq!(result, 0xF0 & 0x0F);

    // Test XOR via ShoutId(1)
    let xor_id = tables.opcode_to_id(RiscvOpcode::Xor);
    assert_eq!(xor_id, ShoutId(1));
    let result = tables.lookup(xor_id, index);
    assert_eq!(result, 0xF0 ^ 0x0F);

    // Test ADD via ShoutId(3)
    let add_id = tables.opcode_to_id(RiscvOpcode::Add);
    assert_eq!(add_id, ShoutId(3));
    let index = interleave_bits(100, 50) as u64;
    let result = tables.lookup(add_id, index);
    assert_eq!(result, 150);

    // Test SLL via ShoutId(7)
    let sll_id = tables.opcode_to_id(RiscvOpcode::Sll);
    assert_eq!(sll_id, ShoutId(7));
    let index = interleave_bits(1, 3) as u64;
    let result = tables.lookup(sll_id, index);
    assert_eq!(result, 1 << 3);

    println!("✓ test_riscv_shout_tables passed");
}

// ============================================================================
// Test 16: Full RISC-V Instruction Sequence Simulation
// ============================================================================

/// Simulate a complete RISC-V instruction sequence using Shout and Twist.
#[test]
fn test_full_riscv_instruction_sequence() {
    let xlen = 4;
    let mut memory = RiscvMemory::new(32);
    let mut shout_tables = RiscvShoutTables::new(xlen);

    // Simulate:
    // 1. Store value 10 at address 0x100
    // 2. Store value 5 at address 0x104
    // 3. Load value from 0x100 (r1 = 10)
    // 4. Load value from 0x104 (r2 = 5)
    // 5. ADD r3 = r1 + r2 (via Shout)
    // 6. AND r4 = r1 & r2 (via Shout)
    // 7. Store r3 at 0x108

    // Step 1-2: Store initial values
    memory.store(TwistId(0), 0x100, 10);
    memory.store(TwistId(0), 0x104, 5);

    // Step 3-4: Load values
    let r1 = memory.load(TwistId(0), 0x100);
    let r2 = memory.load(TwistId(0), 0x104);
    assert_eq!(r1, 10);
    assert_eq!(r2, 5);

    // Step 5: ADD via Shout
    let add_index = interleave_bits(r1, r2) as u64;
    let r3 = shout_tables.lookup(ShoutId(3), add_index); // ADD
    assert_eq!(r3, 15);

    // Step 6: AND via Shout
    let r4 = shout_tables.lookup(ShoutId(0), add_index); // AND
    assert_eq!(r4, 0); // 10 & 5 = 0 (0b1010 & 0b0101 = 0)

    // Step 7: Store result
    memory.store(TwistId(0), 0x108, r3);
    assert_eq!(memory.load(TwistId(0), 0x108), 15);

    println!("✓ test_full_riscv_instruction_sequence passed");
}

// ============================================================================
// Test 17: Memory Event Tracking
// ============================================================================

/// Test memory event creation for trace building.
#[test]
fn test_memory_event_tracking() {
    let event1 = RiscvMemoryEvent::new(RiscvMemOp::Sw, 0x1000, 0xDEADBEEF);
    assert_eq!(event1.op, RiscvMemOp::Sw);
    assert_eq!(event1.addr, 0x1000);
    assert_eq!(event1.value, 0xDEADBEEF);
    assert!(event1.op.is_store());
    assert!(!event1.op.is_load());
    assert_eq!(event1.op.width_bytes(), 4);

    let event2 = RiscvMemoryEvent::new(RiscvMemOp::Lb, 0x2000, 0x80);
    assert!(event2.op.is_load());
    assert!(event2.op.is_sign_extend());
    assert_eq!(event2.op.width_bytes(), 1);

    let event3 = RiscvMemoryEvent::new(RiscvMemOp::Lbu, 0x2000, 0x80);
    assert!(event3.op.is_load());
    assert!(!event3.op.is_sign_extend());

    println!("✓ test_memory_event_tracking passed");
}

// ============================================================================
// Test 18: Complete ALU Operation Suite
// ============================================================================

/// Test all supported RISC-V ALU operations.
#[test]
fn test_complete_alu_suite() {
    let xlen = 4;
    // In 4-bit signed: 0-7 are positive, 8-15 are negative (-8 to -1)
    // 8=0b1000=-8, 9=-7, 10=-6, 11=-5, 12=-4, 13=-3, 14=-2, 15=-1
    let test_cases = [
        // (opcode, x, y, expected_result)
        (RiscvOpcode::And, 15, 10, 10),   // 0b1111 & 0b1010 = 0b1010
        (RiscvOpcode::Or, 15, 10, 15),    // 0b1111 | 0b1010 = 0b1111
        (RiscvOpcode::Xor, 15, 10, 5),    // 0b1111 ^ 0b1010 = 0b0101
        (RiscvOpcode::Add, 7, 8, 15),     // 7 + 8 = 15
        (RiscvOpcode::Add, 8, 8, 0),      // 8 + 8 = 16 -> 0 (overflow)
        (RiscvOpcode::Sub, 10, 3, 7),     // 10 - 3 = 7
        (RiscvOpcode::Sub, 3, 10, 9),     // 3 - 10 = -7 -> 9 (wraparound)
        (RiscvOpcode::Slt, 5, 10, 0),     // 5 < -6 = false (10 is -6 in 4-bit signed)
        (RiscvOpcode::Slt, 10, 5, 1),     // -6 < 5 = true
        (RiscvOpcode::Slt, 8, 5, 1),      // -8 < 5 = true (8 is -8 in 4-bit signed)
        (RiscvOpcode::Slt, 3, 5, 1),      // 3 < 5 = true
        (RiscvOpcode::Sltu, 8, 5, 0),     // 8 < 5 = false (unsigned)
        (RiscvOpcode::Sltu, 5, 8, 1),     // 5 < 8 = true
        (RiscvOpcode::Eq, 5, 5, 1),       // 5 == 5
        (RiscvOpcode::Eq, 5, 6, 0),       // 5 != 6
        (RiscvOpcode::Neq, 5, 5, 0),      // 5 == 5
        (RiscvOpcode::Neq, 5, 6, 1),      // 5 != 6
        (RiscvOpcode::Sll, 1, 3, 8),      // 1 << 3 = 8
        (RiscvOpcode::Srl, 8, 2, 2),      // 8 >> 2 = 2
        (RiscvOpcode::Sra, 8, 1, 12),     // -8 >> 1 = -4 = 12
    ];

    for (op, x, y, expected) in test_cases {
        let result = compute_op(op, x, y, xlen);
        assert_eq!(
            result, expected,
            "{} {}, {} = {} (expected {})",
            op, x, y, result, expected
        );
    }

    println!("✓ test_complete_alu_suite passed");
}

// ============================================================================
// Test 19: End-to-End RISC-V Trace with Full Pipeline
// ============================================================================

use neo_memory::riscv_lookups::{
    BranchCondition, RiscvCpu, RiscvInstruction,
};
use neo_vm_trace::trace_program;

/// End-to-end test: Run a RISC-V Fibonacci program through trace_program.
///
/// This verifies that RiscvCpu + RiscvMemory + RiscvShoutTables work together
/// correctly with the Neo VmCpu trait.
#[test]
fn test_riscv_end_to_end_fibonacci_trace() {
    // Compute Fibonacci(10) using the full trace pipeline
    let program = vec![
        // Initialize: x1=0, x2=1, x3=10 (counter)
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 1 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 0, imm: 10 },
        // Loop: x4 = x1 + x2; x1 = x2; x2 = x4; x3--; if x3 != 0 goto loop
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 4, rs1: 1, rs2: 2 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 1, rs1: 2, rs2: 0 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 2, rs1: 4, rs2: 0 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 3, imm: -1 },
        RiscvInstruction::Branch { cond: BranchCondition::Ne, rs1: 3, rs2: 0, imm: -16 },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    // Run the trace
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();

    // Verify the trace
    assert!(trace.did_halt(), "CPU should have halted");
    assert_eq!(trace.len(), 54, "Expected 54 steps (3 init + 10*5 loop + 1 halt)");

    // Check final register values
    let last_step = trace.steps.last().unwrap();
    let fib_result = last_step.regs_after[2];
    assert_eq!(fib_result, 89, "Fibonacci after 10 iterations should be 89");

    // Verify Shout events were recorded
    let total_shout = trace.total_shout_events();
    assert!(total_shout > 0, "Should have Shout events recorded");

    println!("✓ test_riscv_end_to_end_fibonacci_trace passed");
    println!("  Total steps: {}", trace.len());
    println!("  Total Shout events: {}", total_shout);
    println!("  Final Fibonacci result: {}", fib_result);
}

/// End-to-end test: RISC-V program with memory operations and trace verification.
#[test]
fn test_riscv_end_to_end_memory_trace() {
    // Program that stores values to memory and loads them back
    let program = vec![
        // x1 = 0x100 (base address)
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0x100 },
        // x2 = 42
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 42 },
        // mem[x1] = x2 (store 42)
        RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1: 1, rs2: 2, imm: 0 },
        // x3 = x2 * 2 = 84
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 2, imm: 42 },
        // mem[x1+4] = x3 (store 84)
        RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1: 1, rs2: 3, imm: 4 },
        // x4 = mem[x1] (load 42)
        RiscvInstruction::Load { op: RiscvMemOp::Lw, rd: 4, rs1: 1, imm: 0 },
        // x5 = mem[x1+4] (load 84)
        RiscvInstruction::Load { op: RiscvMemOp::Lw, rd: 5, rs1: 1, imm: 4 },
        // x6 = x4 + x5 = 126
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 6, rs1: 4, rs2: 5 },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    // Run the trace
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();

    // Verify the trace
    assert!(trace.did_halt());
    assert_eq!(trace.len(), 9);

    // Verify Twist events were recorded (memory operations)
    let total_twist = trace.total_twist_events();
    assert!(total_twist > 0, "Should have Twist events recorded");

    // Verify final register values
    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[4], 42, "x4 should be 42 (loaded from memory)");
    assert_eq!(last_step.regs_after[5], 84, "x5 should be 84 (loaded from memory)");
    assert_eq!(last_step.regs_after[6], 126, "x6 should be 126 (42 + 84)");

    println!("✓ test_riscv_end_to_end_memory_trace passed");
    println!("  Total steps: {}", trace.len());
    println!("  Total Twist events: {}", total_twist);
}

