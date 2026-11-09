//! Differential check of the compact operation table: honest rows pass,
//! forged outputs fail, and every value fits its declared width.

mod common;

use neo_math::F;
use neo_wasm::layout::{COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::{audit_compact_lookup_auxiliary_load_bearing, audit_compact_lookup_witness, write_range_check_bits};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::collections::HashSet;

const P: u64 = 0xffff_ffff_0000_0001;

const BINARY32: &[&str] = &[
    "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u", "and", "or", "xor", "mul", "shl", "shr_u", "shr_s",
    "rotl", "rotr", "div_u", "div_s", "rem_u", "rem_s",
];
const UNARY: &[&str] = &["clz", "ctz", "popcnt"];

fn values() -> Vec<u64> {
    vec![
        0,
        1,
        7,
        33,
        0x8000_0000,
        0xffff_ffff,
        0x8000_0000_0000_0000,
        u64::MAX,
        u64::MAX - 6,
        0x0123_4567_89ab_cdef,
    ]
}

fn program(wide: bool) -> String {
    let ty = if wide { "i64" } else { "i32" };
    let cast = |value: u64| {
        if wide {
            value as i64 as i128
        } else {
            value as u32 as i32 as i128
        }
    };
    let mut body = String::new();
    for &left in &values() {
        for op in UNARY {
            body += &format!("{ty}.const {} {ty}.{op} drop\n", cast(left));
        }
        for &right in &values() {
            for op in BINARY32 {
                let divides = op.starts_with("div") || op.starts_with("rem");
                let r = if wide { right } else { right as u32 as u64 };
                let l_min = if wide { left == 1 << 63 } else { left as u32 == 1 << 31 };
                let r_minus_one = if wide { r == u64::MAX } else { r == 0xffff_ffff };
                if divides && (r == 0 || (*op == "div_s" && l_min && r_minus_one)) {
                    continue;
                }
                body += &format!("{ty}.const {} {ty}.const {} {ty}.{op} drop\n", cast(left), cast(right));
            }
        }
    }
    format!("(module (func (export \"main\") (result i32)\n{body} i32.const 0))")
}

fn check(wide: bool) {
    let checked = common::checked_main(&program(wide));
    let relation = neo_wasm::batch::build_batched_wasm_ccs(1).expect("relation");
    let mut representatives = HashSet::new();
    let mut rows = 0;
    for row in checked.trace.iter().filter(|row| row.info.uses_op_table) {
        rows += 1;
        let assignment =
            neo_wasm::batch::build_batched_witness(std::slice::from_ref(row), 1, 0).expect("batched witness");
        for (column, (value, &width)) in assignment.iter().zip(&relation.widths).enumerate() {
            assert!(
                width >= 64 || value.as_canonical_u64() >> width == 0,
                "{:?}: column {column} holds {} but declares {width} bits",
                row.opcode,
                value.as_canonical_u64()
            );
        }
        let witness = neo_wasm::build_witness_vector(row).expect("witness");
        audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("rejected honest {:?}: {error}", row.opcode));
        if representatives.insert(row.opcode) {
            audit_compact_lookup_auxiliary_load_bearing(&witness)
                .unwrap_or_else(|error| panic!("{:?}: {error}", row.opcode));
        }
        let low = witness[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64();
        let high = witness[COL_STACK_WRITE0_VALUE_HI].as_canonical_u64();
        let honest = low | (high << 32);
        let bits = if wide { 64 } else { 32 };
        let mut forgeries = (0..bits)
            .map(|bit| honest ^ (1u64 << bit))
            .collect::<Vec<_>>();
        if wide {
            if let Some(wrapped) = honest.checked_add(P) {
                forgeries.push(wrapped);
            }
        }
        for forged in forgeries {
            let mut tampered = witness.clone();
            tampered[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(forged & 0xffff_ffff);
            tampered[COL_STACK_WRITE0_VALUE_HI] = F::from_u64(forged >> 32);
            write_range_check_bits(&mut tampered);
            assert!(
                audit_compact_lookup_witness(&tampered).is_err(),
                "accepted forged {:?} output {forged:#x} (honest {honest:#x})",
                row.opcode
            );
        }
    }
    eprintln!("wide={wide}: {rows} op-table rows, {} opcodes", representatives.len());
    assert_eq!(representatives.len(), 24);
}

#[test]
fn differential_narrow() {
    check(false);
}

#[test]
fn differential_wide() {
    check(true);
}
