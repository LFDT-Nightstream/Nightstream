//! Trace-level coverage for the i64 ALU op-table opcodes (comparisons,
//! shifts/rotates, div/rem, clz/ctz/popcnt) plus i32.popcnt. Each test runs
//! a wat program through the full checked pipeline: wasmtime trace →
//! normalize → lookup/memory sanity checkers → CCS row satisfaction.

mod common;

#[test]
fn i64_comparisons_cover_all_orderings() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -2
                i64.const 1
                i64.lt_s            ;; 1: -2 < 1 signed
                i64.const -2
                i64.const 1
                i64.lt_u            ;; 0: 2^64-2 is huge unsigned
                i32.add
                i64.const 0x100000000
                i64.const 7
                i64.gt_s            ;; 1
                i32.add
                i64.const 7
                i64.const 0x100000000
                i64.gt_u            ;; 0
                i32.add
                i64.const -5
                i64.const -5
                i64.le_s            ;; 1
                i32.add
                i64.const 6
                i64.const 5
                i64.le_u            ;; 0
                i32.add
                i64.const -1
                i64.const 0
                i64.ge_s            ;; 0: -1 < 0 signed
                i32.add
                i64.const -1
                i64.const 0
                i64.ge_u            ;; 1: max u64 unsigned
                i32.add))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["4".to_string()]);
}

#[test]
fn i64_shifts_and_rotates() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 1
                i64.const 40
                i64.shl             ;; 2^40
                i64.const 36
                i64.shr_u           ;; 16
                i64.const -64
                i64.const 3
                i64.shr_s           ;; -8 (sign-preserving)
                i64.add             ;; 8
                i64.const 1
                i64.const 63
                i64.rotl            ;; 2^63
                i64.const 62
                i64.rotr            ;; 2
                i64.add             ;; 10
                i32.wrap_i64))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["10".to_string()]);
}

#[test]
fn i64_shift_count_is_masked_mod_64() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 1
                i64.const 68        ;; masked to 4
                i64.shl             ;; 16
                i32.wrap_i64))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["16".to_string()]);
}

#[test]
fn i64_select_preserves_selected_high_limb() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 0x200000007
                i64.const 0x300000009
                i32.const 1
                select
                i64.const 32
                i64.shr_u
                i32.wrap_i64))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["2".to_string()]);
}

#[test]
fn i64_div_and_rem_signed_unsigned() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -7
                i64.const 2
                i64.div_s           ;; -3 (truncated toward zero)
                i64.const 7
                i64.const 2
                i64.div_u           ;; 3
                i64.add             ;; 0
                i64.const -7
                i64.const 3
                i64.rem_s           ;; -1 (sign of dividend)
                i64.add             ;; -1
                i64.const -1
                i64.const 10
                i64.rem_u           ;; (2^64-1) % 10 = 5
                i64.add             ;; 4
                i64.const 100
                i64.const 10
                i64.div_u           ;; 10
                i64.add             ;; 14
                i32.wrap_i64))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["14".to_string()]);
}

#[test]
fn i64_and_i32_bit_counting() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 1
                i64.clz             ;; 63
                i64.const 0x10
                i64.ctz             ;; 4
                i64.add             ;; 67
                i64.const -1
                i64.popcnt          ;; 64
                i64.add             ;; 131
                i32.wrap_i64
                i32.const 0xF0F0
                i32.popcnt          ;; 8
                i32.add))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["139".to_string()]);
}
