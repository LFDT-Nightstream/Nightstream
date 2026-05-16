//! u64 arithmetic + conditional-select gadget tests.
//!
//! These are the F' R1CS state-advance primitives:
//!   - `chunk_count_out = chunk_count_in + 1`
//!   - `step_count_out  = step_count_in + K`
//!   - `mux(base_case_bit, u_⊥, NIFS.V_output)` for the base-case branch.

use neo_fold_clean::engine::r1cs_circuit::{
    alloc_u64_bits, enforce_mux_var, enforce_mux_vec, enforce_u64_add, enforce_u64_constant, enforce_u64_equality,
    enforce_u64_increment, Lc, R1csBuilder, Var,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── u64 bitness / equality / constant ─────────────────────────────────────

#[test]
fn alloc_u64_bits_enforces_bitness() {
    let mut b = R1csBuilder::new();
    let _bits = alloc_u64_bits(&mut b, 0x1234_5678_9ABC_DEF0);
    assert!(b.is_satisfied(), "alloc_u64_bits must accept a valid u64");
}

#[test]
fn enforce_u64_constant_accepts_matching_value() {
    let value = 0xDEADBEEF_CAFEBABE_u64;
    let mut b = R1csBuilder::new();
    let bits = alloc_u64_bits(&mut b, value);
    enforce_u64_constant(&mut b, &bits, value);
    assert!(b.is_satisfied());
}

#[test]
fn enforce_u64_constant_rejects_wrong_value() {
    let mut b = R1csBuilder::new();
    let bits = alloc_u64_bits(&mut b, 0x1234);
    enforce_u64_constant(&mut b, &bits, 0x5678);
    assert!(!b.is_satisfied(), "constant check should reject mismatch");
}

#[test]
fn enforce_u64_equality_accepts_matching_bits() {
    let value = 0xFEDC_BA98_7654_3210_u64;
    let mut b = R1csBuilder::new();
    let bits_a = alloc_u64_bits(&mut b, value);
    let bits_b = alloc_u64_bits(&mut b, value);
    enforce_u64_equality(&mut b, &bits_a, &bits_b);
    assert!(b.is_satisfied());
}

#[test]
fn enforce_u64_equality_rejects_mismatch() {
    let mut b = R1csBuilder::new();
    let bits_a = alloc_u64_bits(&mut b, 0x1234);
    let bits_b = alloc_u64_bits(&mut b, 0x1235);
    enforce_u64_equality(&mut b, &bits_a, &bits_b);
    assert!(!b.is_satisfied());
}

// ── u64 increment ────────────────────────────────────────────────────────

#[test]
fn enforce_u64_increment_basic_cases() {
    // Note: the gadget rejects wraparound at bit 63 (no top carry slot).
    // For F' counters, overflow is unreachable in practice (we'd run out of
    // memory before reaching 2^63 steps), so this is the right behavior.
    for (input, output) in [
        (0u64, 1u64),
        (1, 2),
        (0xFF, 0x100),
        (0xFFFF_FFFF, 0x1_0000_0000),
        (1u64 << 62, (1u64 << 62) + 1),
    ] {
        let mut b = R1csBuilder::new();
        let in_bits = alloc_u64_bits(&mut b, input);
        let out_bits = alloc_u64_bits(&mut b, output);
        enforce_u64_increment(&mut b, &in_bits, &out_bits);
        assert!(
            b.is_satisfied(),
            "increment {input} → {output} must be accepted (first bad row: {:?})",
            b.first_unsatisfied_row()
        );
    }
}

#[test]
fn enforce_u64_increment_rejects_wrong_output() {
    let mut b = R1csBuilder::new();
    let in_bits = alloc_u64_bits(&mut b, 42);
    let out_bits = alloc_u64_bits(&mut b, 100); // wrong: should be 43
    enforce_u64_increment(&mut b, &in_bits, &out_bits);
    assert!(!b.is_satisfied(), "increment must reject wrong output");
}

// ── u64 add ──────────────────────────────────────────────────────────────

#[test]
fn enforce_u64_add_basic_cases() {
    // Same overflow-rejecting semantics as increment.
    for (a, b_val, sum) in [
        (0u64, 0u64, 0u64),
        (1, 1, 2),
        (123, 456, 579),
        (0xFFFF_FFFF, 1, 0x1_0000_0000),
        (0xDEAD_BEEF, 0xCAFE_BABE, 0xDEAD_BEEF_u64.wrapping_add(0xCAFE_BABE)),
    ] {
        let mut bd = R1csBuilder::new();
        let a_bits = alloc_u64_bits(&mut bd, a);
        let b_bits = alloc_u64_bits(&mut bd, b_val);
        let sum_bits = alloc_u64_bits(&mut bd, sum);
        enforce_u64_add(&mut bd, &a_bits, &b_bits, &sum_bits);
        assert!(
            bd.is_satisfied(),
            "{a} + {b_val} = {sum} must be accepted (first bad row: {:?})",
            bd.first_unsatisfied_row()
        );
    }
}

#[test]
fn enforce_u64_add_rejects_wrong_sum() {
    let mut bd = R1csBuilder::new();
    let a_bits = alloc_u64_bits(&mut bd, 100);
    let b_bits = alloc_u64_bits(&mut bd, 200);
    let sum_bits = alloc_u64_bits(&mut bd, 999); // wrong
    enforce_u64_add(&mut bd, &a_bits, &b_bits, &sum_bits);
    assert!(!bd.is_satisfied());
}

// ── mux ──────────────────────────────────────────────────────────────────

#[test]
fn mux_var_selects_a_when_s_is_1() {
    let mut b = R1csBuilder::new();
    let s = b.alloc(F::ONE);
    let a = b.alloc(F::from_u64(42));
    let b_val = b.alloc(F::from_u64(99));
    let out = enforce_mux_var(&mut b, s, a, b_val);
    assert!(b.is_satisfied());
    assert_eq!(b.witness()[out.col()], F::from_u64(42));
}

#[test]
fn mux_var_selects_b_when_s_is_0() {
    let mut b = R1csBuilder::new();
    let s = b.alloc(F::ZERO);
    let a = b.alloc(F::from_u64(42));
    let b_val = b.alloc(F::from_u64(99));
    let out = enforce_mux_var(&mut b, s, a, b_val);
    assert!(b.is_satisfied());
    assert_eq!(b.witness()[out.col()], F::from_u64(99));
}

#[test]
fn mux_var_rejects_tampered_output() {
    let mut b = R1csBuilder::new();
    let s = b.alloc(F::ONE);
    let a = b.alloc(F::from_u64(42));
    let b_val = b.alloc(F::from_u64(99));
    let out = enforce_mux_var(&mut b, s, a, b_val);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[out.col()] + F::ONE;
    b.tamper_witness(out.col(), tampered);
    assert!(!b.is_satisfied(), "mux must reject tampered output");
}

#[test]
fn mux_vec_routes_all_lanes() {
    let mut bd = R1csBuilder::new();
    let s = bd.alloc(F::ONE); // select a
    let a: Vec<Var> = (0..5)
        .map(|i| bd.alloc(F::from_u64(10 + i as u64)))
        .collect();
    let b: Vec<Var> = (0..5)
        .map(|i| bd.alloc(F::from_u64(100 + i as u64)))
        .collect();
    let out = enforce_mux_vec(&mut bd, s, &a, &b);
    assert!(bd.is_satisfied());
    for (i, ov) in out.iter().enumerate() {
        assert_eq!(bd.witness()[ov.col()], F::from_u64(10 + i as u64));
    }
}

#[test]
fn mux_vec_routes_b_when_s_zero() {
    let mut bd = R1csBuilder::new();
    let s = bd.alloc(F::ZERO);
    let a: Vec<Var> = (0..3)
        .map(|i| bd.alloc(F::from_u64(10 + i as u64)))
        .collect();
    let b: Vec<Var> = (0..3)
        .map(|i| bd.alloc(F::from_u64(100 + i as u64)))
        .collect();
    let out = enforce_mux_vec(&mut bd, s, &a, &b);
    assert!(bd.is_satisfied());
    for (i, ov) in out.iter().enumerate() {
        assert_eq!(bd.witness()[ov.col()], F::from_u64(100 + i as u64));
    }
}

// Silence unused
fn _unused(_: Lc) {}
