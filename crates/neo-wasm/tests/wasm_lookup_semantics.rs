mod common;

use neo_math::F;
use neo_wasm::layout::{COL_STACK_READ_VALUE_HI, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::WasmOpTable;
use neo_wasm::{build_wasm_relation_layout, sanity_check_lookup_row, traces_from_wasmtime_wasm_bytes, WasmOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::collections::HashSet;

fn trace_rows() -> Vec<neo_wasm::WasmVmStep> {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 0
                i32.const 9
                i32.store
                i32.const 6
                i32.const 7
                i32.mul
                drop
                i64.const 3
                i64.const 5
                i64.mul
                drop
                i64.const 0x100000001
                i64.clz
                drop
                i64.const -2
                i64.const 1
                i64.lt_s
                drop
                i32.const 0
                i32.load)
        )"#,
    )
    .expect("wat");
    traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("trace")
}

#[test]
fn lookup_semantics_accept_real_wasm_rows() {
    let layout = build_wasm_relation_layout();
    for row in trace_rows() {
        let witness = build_witness_vector(&row);
        sanity_check_lookup_row(&layout.auxiliary, &witness)
            .unwrap_or_else(|err| panic!("expected lookup semantics to accept {:?}: {err}", row.opcode));
    }
}

#[test]
fn lookup_semantics_reject_tampered_shout_output() {
    let layout = build_wasm_relation_layout();
    let row = trace_rows()
        .into_iter()
        .find(|row| row.opcode == WasmOpcode::I32Mul)
        .expect("i32.mul row");
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(1234);
    let err = sanity_check_lookup_row(&layout.auxiliary, &witness).expect_err("tampered op_table output should fail");
    assert!(err.contains("i32_mul"));
}

#[test]
fn lookup_semantics_reject_tampered_i64_unary_input_hi() {
    let layout = build_wasm_relation_layout();
    let row = trace_rows()
        .into_iter()
        .find(|row| row.opcode == WasmOpcode::I64Clz)
        .expect("i64.clz row");
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_READ_VALUE_HI[0]] = F::from_u64(2);
    let err = sanity_check_lookup_row(&layout.auxiliary, &witness).expect_err("tampered i64 unary input should fail");
    assert!(err.contains("i64_clz"));
}

#[test]
fn lookup_semantics_reject_tampered_i64_binary_input_hi() {
    let layout = build_wasm_relation_layout();
    let row = trace_rows()
        .into_iter()
        .find(|row| row.opcode == WasmOpcode::I64LtS)
        .expect("i64.lt_s row");
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_READ_VALUE_HI[0]] = F::ZERO;
    let err = sanity_check_lookup_row(&layout.auxiliary, &witness).expect_err("tampered i64 binary input should fail");
    assert!(err.contains("i64_lt_s"));
}

#[test]
fn compact_lookup_relation_covers_and_rejects_all_families() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 305419896 i32.clz drop
                i32.const 305419896 i32.ctz drop
                i32.const -123 i32.const 7 i32.lt_s drop
                i32.const 123 i32.const 7 i32.lt_u drop
                i32.const -123 i32.const 7 i32.gt_s drop
                i32.const 123 i32.const 7 i32.gt_u drop
                i32.const -123 i32.const 7 i32.le_s drop
                i32.const 123 i32.const 7 i32.le_u drop
                i32.const -123 i32.const 7 i32.ge_s drop
                i32.const 123 i32.const 7 i32.ge_u drop
                i32.const 305419896 i32.const 252645135 i32.and drop
                i32.const 305419896 i32.const 252645135 i32.or drop
                i32.const 305419896 i32.const 252645135 i32.xor drop
                i32.const 305419896 i32.const 7 i32.mul drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.and drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.or drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.xor drop
                i64.const 81985529216486895 i64.const 7 i64.mul drop
                i32.const 305419896 i32.const 5 i32.shl drop
                i32.const 305419896 i32.const 5 i32.shr_u drop
                i32.const -305419896 i32.const 5 i32.shr_s drop
                i32.const 305419896 i32.const 5 i32.rotl drop
                i32.const 305419896 i32.const 5 i32.rotr drop
                i32.const 305419896 i32.const 7 i32.div_u drop
                i32.const -305419896 i32.const 7 i32.div_s drop
                i32.const 305419896 i32.const 7 i32.rem_u drop
                i32.const -305419896 i32.const 7 i32.rem_s drop
                i32.const 305419896 i32.popcnt drop
                i64.const -81985529216486895 i64.const 7 i64.lt_s drop
                i64.const 81985529216486895 i64.const 7 i64.lt_u drop
                i64.const -81985529216486895 i64.const 7 i64.gt_s drop
                i64.const 81985529216486895 i64.const 7 i64.gt_u drop
                i64.const -81985529216486895 i64.const 7 i64.le_s drop
                i64.const 81985529216486895 i64.const 7 i64.le_u drop
                i64.const -81985529216486895 i64.const 7 i64.ge_s drop
                i64.const 81985529216486895 i64.const 7 i64.ge_u drop
                i64.const 81985529216486895 i64.const 13 i64.shl drop
                i64.const -81985529216486895 i64.const 13 i64.shr_s drop
                i64.const 81985529216486895 i64.const 13 i64.shr_u drop
                i64.const 81985529216486895 i64.const 13 i64.rotl drop
                i64.const 81985529216486895 i64.const 13 i64.rotr drop
                i64.const -81985529216486895 i64.const 7 i64.div_s drop
                i64.const 81985529216486895 i64.const 7 i64.div_u drop
                i64.const -81985529216486895 i64.const 7 i64.rem_s drop
                i64.const 81985529216486895 i64.const 7 i64.rem_u drop
                i64.const 81985529216486895 i64.clz drop
                i64.const 81985529216486895 i64.ctz drop
                i64.const 81985529216486895 i64.popcnt drop
                i32.const 0))"#,
    );
    let full_relation = neo_wasm::batch::build_batched_wasm_ccs(1).expect("lookup relation");
    let mut seen = HashSet::new();
    for row in checked.trace.iter().filter(|row| row.info.uses_op_table) {
        seen.insert(row.opcode);
        let assignment = neo_wasm::batch::build_batched_witness(std::slice::from_ref(row), 1, 0);
        full_relation
            .sparse_r1cs
            .is_satisfied_by(&assignment)
            .unwrap_or_else(|error| panic!("full relation rejected honest {:?}: {error}", row.opcode));
        let witness = neo_wasm::build_witness_vector(row);
        neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("compact lookup relation rejected honest {:?}: {error}", row.opcode));

        let mut tampered = witness;
        let value = tampered[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64() ^ 1;
        tampered[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::from_u64(value);
        neo_wasm::write_range_check_bits(&mut tampered);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&tampered).is_err(),
            "compact lookup relation accepted a forged {:?} output",
            row.opcode
        );
    }
    let expected = WasmOpTable::all()
        .into_iter()
        .map(WasmOpTable::opcode)
        .collect::<HashSet<_>>();
    assert_eq!(
        seen, expected,
        "fixture must execute every lookup family exactly at least once"
    );
}

#[test]
fn compact_lookup_signed_division_edges_and_advice_are_bound() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -1 i32.const 7 i32.div_s drop
                i32.const 1 i32.const -7 i32.div_s drop
                i32.const -13 i32.const -5 i32.div_s drop
                i32.const -13 i32.const -5 i32.rem_s drop
                i64.const -1 i64.const 7 i64.div_s drop
                i64.const 1 i64.const -7 i64.div_s drop
                i64.const -13 i64.const -5 i64.div_s drop
                i64.const -13 i64.const -5 i64.rem_s drop
                i32.const 0))"#,
    );
    let rows = checked
        .trace
        .iter()
        .filter(|row| {
            matches!(
                row.opcode,
                WasmOpcode::I32DivS | WasmOpcode::I32RemS | WasmOpcode::I64DivS | WasmOpcode::I64RemS
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 8);
    for row in &rows {
        let witness = neo_wasm::build_witness_vector(row);
        neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("signed division edge {:?} failed: {error}", row.opcode));
        let mut tampered = witness;
        let value = tampered[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64() ^ 1;
        tampered[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::from_u64(value);
        neo_wasm::write_range_check_bits(&mut tampered);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&tampered).is_err(),
            "signed division edge accepted a forged {:?} result",
            row.opcode
        );
    }

    let representative = neo_wasm::build_witness_vector(rows.last().expect("signed division row"));
    let auxiliary_columns =
        neo_wasm::audit_compact_lookup_witness(&representative).expect("honest signed division witness");
    assert_eq!(
        neo_wasm::audit_compact_lookup_auxiliary_load_bearing(&representative)
            .expect("every compact lookup auxiliary must be load-bearing"),
        auxiliary_columns,
    );
}

#[test]
fn compact_lookup_accepts_nontrapping_signed_remainder_overflow() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -2147483648 i32.const -1 i32.rem_s drop
                i64.const -9223372036854775808 i64.const -1 i64.rem_s drop
                i32.const 0))"#,
    );
    let rows = checked
        .trace
        .iter()
        .filter(|row| matches!(row.opcode, WasmOpcode::I32RemS | WasmOpcode::I64RemS))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert!(!row.state_after.trapped, "signed remainder overflow does not trap");
        let mut witness = neo_wasm::build_witness_vector(row);
        neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("compact lookup rejected honest {:?}: {error}", row.opcode));
        witness[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::ONE;
        neo_wasm::write_range_check_bits(&mut witness);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&witness).is_err(),
            "signed remainder overflow accepted a nonzero {:?} result",
            row.opcode
        );
    }
}
