mod common;

use neo_fold_clean::frontends::r1cs_f_prime;
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_wasm::preprocess::preprocess_seeded;
use neo_wasm::{prove, verify, WasmStepTrace, WasmVmSpec};

/// Compile a WAT module, run it through the wasmtime adapter, exercise the
/// witness-derived sanity checks, and return the trace + ROMs.
fn compile_and_trace(
    wat_src: &str,
) -> (
    Vec<u8>,
    Vec<WasmStepTrace>,
    Vec<(u64, u64, u64)>,
    Vec<(u64, u64)>,
    Vec<(u64, u64)>,
) {
    compile_and_trace_with(wat_src, "main", &[])
}

fn compile_and_trace_with(
    wat_src: &str,
    export: &str,
    params: &[i32],
) -> (
    Vec<u8>,
    Vec<WasmStepTrace>,
    Vec<(u64, u64, u64)>,
    Vec<(u64, u64)>,
    Vec<(u64, u64)>,
) {
    let checked = common::checked_wasm_run(wat_src, export, params);
    let pc_rom = checked.run.pc_rom.clone();
    let pc_edge_kinds = checked.run.pc_edge_kinds.clone();
    let function_entries = checked.run.function_entries.clone();
    (checked.wasm, checked.trace, pc_rom, pc_edge_kinds, function_entries)
}

#[test]
fn wasm_trace_run() {
    compile_and_trace(
        r#"(module (func (export "main") (result i32)
             i32.const 7
             i32.const 9
             i32.eq))"#,
    );
}

#[test]
fn wasm_trace_run_with_linear_memory() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 42
               i32.store
               i32.const 0
               i32.load))"#,
    );
    assert!(trace.iter().any(|row| row.linear_memory.is_some()));
}

#[test]
fn wasm_trace_run_with_memory_size_and_grow() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1 3)
             (func (export "main") (result i32)
               memory.size
               drop
               i32.const 1
               memory.grow
               drop
               memory.size))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::MemorySize)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::MemoryGrow)));
}

#[test]
fn wasm_trace_run_with_linear_memory_offset() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 42
               i32.const 4
               i32.store offset=8
               i32.const 4
               i32.load offset=8))"#,
    );
    assert!(trace.iter().any(|row| row.linear_memory_offset != 0));
}

#[test]
fn wasm_trace_run_with_byte_linear_memory() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 511
               i32.store8
               i32.const 0
               i32.load8_u))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Store8)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Load8U)));
}

/// Exercises i64.store8 / i64.store16 / i64.store32 end-to-end: each
/// truncates the i64 input to N bytes, writes them, and the corresponding
/// i32.loadN_u reads them back. `common::checked_wasm_run` runs the lookup
/// + memory + CCS sanity checks on every row, so passing here proves the
/// full constraint chain holds for the new opcodes.
#[test]
fn wasm_trace_run_with_i64_subword_stores() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i64.const 0xAABBCCDD11223344
               i64.store8
               i32.const 8
               i64.const 0xAABBCCDD11223344
               i64.store16
               i32.const 16
               i64.const 0xAABBCCDD11223344
               i64.store32
               i32.const 0
               i32.load8_u
               i32.const 8
               i32.load16_u
               i32.add
               i32.const 16
               i32.load
               i32.add))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I64Store8)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I64Store16)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I64Store32)));
}

/// i64.store32 at every sub-word alignment (offsets 1, 2, 3 cross into
/// lane1). Exercises the full-width lane-usage / byte-routing constraints
/// for the new opcode under unaligned addresses.
#[test]
fn wasm_trace_run_with_unaligned_i64_store32() {
    for addr in [1, 2, 3] {
        let (_, trace, ..) = compile_and_trace(&format!(
            r#"(module
                 (memory 1)
                 (func (export "main") (result i32)
                   i32.const {addr}
                   i64.const 0xAABBCCDD11223344
                   i64.store32
                   i32.const {addr}
                   i32.load))"#,
        ));
        assert!(
            trace
                .iter()
                .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I64Store32)),
            "addr {addr}: expected an i64.store32 row"
        );
    }
}

/// i64.load8_u / load16_u / load32_u round-trips: store a known pattern,
/// then load it back zero-extended. The load-correctness check is the
/// memory-semantics layer inside `checked_wasm_run` (`sanity_check_memory_rows`
/// verifies each load returns the last-stored value), and `ccs_check_trace`
/// confirms every row's witness satisfies the CCS. The opcode-presence
/// asserts below just guarantee the new ops are actually exercised.
#[test]
fn wasm_trace_run_with_i64_unsigned_loads() {
    // Store byte 0xDD at addr 0, halfword 0xCCDD at addr 8, word 0xAABBCCDD
    // at addr 16; load each back (zero-extended) and sum.
    let checked = common::checked_main(
        r#"(module
             (memory 1)
             (func (export "main") (result i64)
               i32.const 0
               i64.const 0xDD
               i64.store8
               i32.const 8
               i64.const 0xCCDD
               i64.store16
               i32.const 16
               i64.const 0xAABBCCDD
               i64.store32
               i32.const 0
               i64.load8_u
               i32.const 8
               i64.load16_u
               i64.add
               i32.const 16
               i64.load32_u
               i64.add))"#,
    );
    for op in [
        neo_wasm::WasmOpcode::I64Load8U,
        neo_wasm::WasmOpcode::I64Load16U,
        neo_wasm::WasmOpcode::I64Load32U,
    ] {
        assert!(
            checked.trace.iter().any(|row| row.opcode == op),
            "expected a {op:?} row"
        );
    }
}

/// i64.loadN_u must zero-extend: the output hi limb is pinned to 0. Since
/// these ops set `wide_values_enabled = 1` (the input side is i64-shaped),
/// the generic "narrow high limbs zero" rule does not cover `write0_value_hi`
/// here — the dedicated `i64 unsigned load high zero` gate does. Forging a
/// nonzero hi limb must be rejected.
#[test]
fn i64_load8_u_zero_extension_is_enforced() {
    use neo_ccs::check_ccs_rowwise_zero;
    use neo_math::F;
    use neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI;
    use neo_wasm::witness_builder::build_witness_vector;
    use p3_field::PrimeCharacteristicRing;

    let (_, trace, ..) = compile_and_trace(
        r#"(module (memory 1) (func (export "main") (result i64)
             i32.const 0 i64.const 0xDD i64.store8 i32.const 0 i64.load8_u))"#,
    );
    let load = trace
        .iter()
        .find(|r| matches!(r.opcode, neo_wasm::WasmOpcode::I64Load8U))
        .expect("i64.load8_u row");
    let mut wit = build_witness_vector(load);
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).expect("honest i64.load8_u row should satisfy the CCS");

    wit[COL_STACK_WRITE0_VALUE_HI] = F::ONE;
    assert!(
        check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).is_err(),
        "a nonzero output hi limb must be rejected (zero-extension)"
    );
}

/// i64.load8_s / load16_s / load32_s round-trips, covering both signs at
/// each width. Asserts on the load row's *witness* (the columns
/// `ccs_check_trace` validated inside `checked_main`): the lo limb carries the
/// 32-bit value and the hi limb is the replicated sign bit — 0 for a clear
/// top bit, 0xFFFF_FFFF for a set one. `sanity_check_memory_rows` separately
/// ties the loaded bytes to what was stored.
#[test]
fn wasm_trace_run_with_i64_signed_loads() {
    use neo_wasm::layout::{COL_STACK_WRITE0_VALUE, COL_STACK_WRITE0_VALUE_HI};
    use neo_wasm::witness_builder::build_witness_vector;
    use p3_field::PrimeField64;

    // (store opcode, stored value, load opcode, expected lo limb, expected hi limb)
    let cases: &[(&str, &str, &str, u64, u64)] = &[
        ("i64.store8", "0x7F", "i64.load8_s", 0x7F, 0),
        ("i64.store8", "0x80", "i64.load8_s", 0xFFFF_FF80, 0xFFFF_FFFF),
        ("i64.store16", "0x7FFF", "i64.load16_s", 0x7FFF, 0),
        ("i64.store16", "0x8000", "i64.load16_s", 0xFFFF_8000, 0xFFFF_FFFF),
        ("i64.store32", "0x7FFFFFFF", "i64.load32_s", 0x7FFF_FFFF, 0),
        ("i64.store32", "0x80000000", "i64.load32_s", 0x8000_0000, 0xFFFF_FFFF),
    ];
    for &(store_op, value, load_op, expected_lo, expected_hi) in cases {
        let checked = common::checked_main(&format!(
            r#"(module
                 (memory 1)
                 (func (export "main") (result i64)
                   i32.const 0
                   i64.const {value}
                   {store_op}
                   i32.const 0
                   {load_op}))"#,
        ));
        let load = checked
            .trace
            .iter()
            .find(|r| r.opcode.name() == load_op.replace('.', "_"))
            .unwrap_or_else(|| panic!("expected a {load_op} row"));
        let wit = build_witness_vector(load);
        assert_eq!(
            wit[COL_STACK_WRITE0_VALUE].as_canonical_u64(),
            expected_lo,
            "{load_op}({value}) lo limb"
        );
        assert_eq!(
            wit[COL_STACK_WRITE0_VALUE_HI].as_canonical_u64(),
            expected_hi,
            "{load_op}({value}) hi limb (sign extension)"
        );
    }
}

/// i64.loadN_s sign extension is load-bearing: on a negative load the hi limb
/// must be 0xFFFF_FFFF, and the `i64 signed load high fill` gate
/// (`write0_value_hi = sign_ext_bit · 0xFFFF_FFFF`) rejects a forged hi limb.
#[test]
fn i64_load8_s_sign_extension_is_enforced() {
    use neo_ccs::check_ccs_rowwise_zero;
    use neo_math::F;
    use neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI;
    use neo_wasm::witness_builder::build_witness_vector;
    use p3_field::PrimeCharacteristicRing;

    // 0x80 is negative -> hi limb must be 0xFFFF_FFFF.
    let (_, trace, ..) = compile_and_trace(
        r#"(module (memory 1) (func (export "main") (result i64)
             i32.const 0 i64.const 0x80 i64.store8 i32.const 0 i64.load8_s))"#,
    );
    let load = trace
        .iter()
        .find(|r| matches!(r.opcode, neo_wasm::WasmOpcode::I64Load8S))
        .expect("i64.load8_s row");
    let mut wit = build_witness_vector(load);
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).expect("honest negative i64.load8_s row should satisfy the CCS");
    assert_eq!(
        wit[COL_STACK_WRITE0_VALUE_HI],
        F::from_u64(0xFFFF_FFFF),
        "negative load must fill the hi limb with all ones"
    );

    // Forge the hi limb to 0 (as if it zero-extended); the sign-fill gate rejects it.
    wit[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
    assert!(
        check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).is_err(),
        "a zeroed hi limb on a negative signed load must be rejected"
    );
}

/// Full-width i64.load32_{u,s} at every sub-word alignment (offsets 1/2/3
/// cross into lane1). Exercises the full-width byte-routing + use_lane1
/// constraints for the new load opcodes under unaligned addresses.
#[test]
fn wasm_trace_run_with_unaligned_i64_load32() {
    for addr in [1, 2, 3] {
        for load_op in ["i64.load32_u", "i64.load32_s"] {
            let (_, trace, ..) = compile_and_trace(&format!(
                r#"(module
                     (memory 1)
                     (func (export "main") (result i64)
                       i32.const {addr}
                       i64.const 0x11223344
                       i64.store32
                       i32.const {addr}
                       {load_op}))"#,
            ));
            let want = load_op.replace('.', "_");
            assert!(
                trace.iter().any(|row| row.opcode.name() == want),
                "addr {addr}: expected a {load_op} row"
            );
        }
    }
}

/// Full-width loads that cross into lane1 (offset 1/2/3) must force
/// `use_lane1 = 1`. Otherwise a malicious witness could satisfy the
/// cross-lane byte shuffle from unconstrained lane1 bytes without activating
/// the lane1 memory access. The full-width `use_lane1` row now covers
/// i64.load32_{u,s}; zeroing `use_lane1` on an offset-1 load must be rejected.
#[test]
fn i64_load32_u_unaligned_requires_use_lane1() {
    use neo_ccs::check_ccs_rowwise_zero;
    use neo_math::F;
    use neo_wasm::layout::COL_LINEAR_MEM_USE_LANE1;
    use neo_wasm::witness_builder::build_witness_vector;
    use p3_field::PrimeCharacteristicRing;

    let (_, trace, ..) = compile_and_trace(
        r#"(module (memory 1) (func (export "main") (result i64)
             i32.const 1 i64.const 0x11223344 i64.store32 i32.const 1 i64.load32_u))"#,
    );
    let load = trace
        .iter()
        .find(|r| matches!(r.opcode, neo_wasm::WasmOpcode::I64Load32U))
        .expect("i64.load32_u row");
    let mut wit = build_witness_vector(load);
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..])
        .expect("honest unaligned i64.load32_u row should satisfy the CCS");
    assert_eq!(
        wit[COL_LINEAR_MEM_USE_LANE1],
        F::ONE,
        "an offset-1 full-width load must activate lane1"
    );

    wit[COL_LINEAR_MEM_USE_LANE1] = F::ZERO;
    assert!(
        check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).is_err(),
        "deactivating lane1 on an unaligned full-width load must be rejected"
    );
}

/// The width-family flags (`is_byte_width` etc.) must be pinned per opcode:
/// zeroing the byte-width family on an i64.store8 row would otherwise
/// vacuously satisfy the byte-routing gates and decouple the stored byte
/// from the memory lane. The `linear memory width opcode binding` rows
/// reject it. (Same pin now protects the i32 subword ops.)
#[test]
fn i64_store8_width_family_pin_is_enforced() {
    use neo_ccs::check_ccs_rowwise_zero;
    use neo_math::F;
    use neo_wasm::layout::{
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_IS_BYTE_WIDTH,
    };
    use neo_wasm::witness_builder::build_witness_vector;
    use p3_field::PrimeCharacteristicRing;

    let (_, trace, ..) = compile_and_trace(
        r#"(module (memory 1) (func (export "main") (result i32)
             i32.const 0 i64.const 0xAB i64.store8 i32.const 0 i32.load8_u))"#,
    );
    let store = trace
        .iter()
        .find(|r| matches!(r.opcode, neo_wasm::WasmOpcode::I64Store8))
        .expect("i64.store8 row");
    let mut wit = build_witness_vector(store);
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).expect("honest i64.store8 row should satisfy the CCS");

    wit[COL_LINEAR_MEM_IS_BYTE_WIDTH] = F::ZERO;
    wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0] = F::ZERO;
    wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1] = F::ZERO;
    wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2] = F::ZERO;
    wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3] = F::ZERO;
    assert!(
        check_ccs_rowwise_zero(ccs, &wit[..1], &wit[1..]).is_err(),
        "zeroing the byte-width family must be rejected"
    );
}

#[test]
fn wasm_trace_run_with_globals() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (global (mut i32) (i32.const 7))
             (func (export "main") (result i32)
               global.get 0
               drop
               i32.const 9
               global.set 0
               global.get 0))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::GlobalGet)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::GlobalSet)));
    assert!(trace.iter().any(|row| row.global_index.is_some()));
}

#[test]
fn wasm_trace_run_with_shift_div_rem() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               i32.const 3
               i32.const 4
               i32.shl
               drop
               i32.const 128
               i32.const 3
               i32.shr_u
               drop
               i32.const -128
               i32.const 3
               i32.shr_s
               drop
               i32.const 22
               i32.const 5
               i32.div_u
               drop
               i32.const -22
               i32.const 5
               i32.div_s
               drop
               i32.const 22
               i32.const 5
               i32.rem_u
               drop
               i32.const -22
               i32.const 5
               i32.rem_s
               drop
               i32.const 123))"#,
    );
    for op in [
        neo_wasm::WasmOpcode::I32Shl,
        neo_wasm::WasmOpcode::I32ShrU,
        neo_wasm::WasmOpcode::I32ShrS,
        neo_wasm::WasmOpcode::I32DivU,
        neo_wasm::WasmOpcode::I32DivS,
        neo_wasm::WasmOpcode::I32RemU,
        neo_wasm::WasmOpcode::I32RemS,
    ] {
        assert!(trace.iter().any(|row| row.opcode == op), "missing {op:?}");
    }
}

#[test]
fn wasm_trace_run_with_compare_unary_and_rotate() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               i32.const 16
               i32.clz
               drop
               i32.const 24
               i32.ctz
               drop
               i32.const 9
               i32.const 3
               i32.gt_s
               drop
               i32.const 9
               i32.const 3
               i32.gt_u
               drop
               i32.const 3
               i32.const 9
               i32.le_s
               drop
               i32.const 3
               i32.const 9
               i32.le_u
               drop
               i32.const 9
               i32.const 3
               i32.ge_s
               drop
               i32.const 9
               i32.const 3
               i32.ge_u
               drop
               i32.const 305419896
               i32.const 8
               i32.rotl
               drop
               i32.const 305419896
               i32.const 8
               i32.rotr
               drop
               i32.const 123))"#,
    );
    for op in [
        neo_wasm::WasmOpcode::I32Clz,
        neo_wasm::WasmOpcode::I32Ctz,
        neo_wasm::WasmOpcode::I32GtS,
        neo_wasm::WasmOpcode::I32GtU,
        neo_wasm::WasmOpcode::I32LeS,
        neo_wasm::WasmOpcode::I32LeU,
        neo_wasm::WasmOpcode::I32GeS,
        neo_wasm::WasmOpcode::I32GeU,
        neo_wasm::WasmOpcode::I32Rotl,
        neo_wasm::WasmOpcode::I32Rotr,
    ] {
        assert!(trace.iter().any(|row| row.opcode == op), "missing {op:?}");
    }
}

#[test]
fn wasm_trace_run_with_br_table() {
    let (_, trace, pc_rom, ..) = compile_and_trace_with(
        r#"(module
             (func (export "main") (param i32) (result i32)
               (block $default
                 (block $case1
                   (block $case0
                     local.get 0
                     br_table $case0 $case1 $default
                   )
                   i32.const 10
                   return
                 )
                 i32.const 20
                 return
               )
               i32.const 30))"#,
        "main",
        &[5],
    );
    let row = trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::BrTable)
        .expect("br_table row");
    assert_eq!(row.control_choice, 0);
    assert_eq!(
        pc_rom
            .iter()
            .filter(|(pc, _, _)| *pc == row.pc_before)
            .count(),
        3
    );
}

#[test]
fn wasm_trace_run_with_table_size() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (table 4 funcref)
             (func (export "main") (result i32)
               table.size 0))"#,
    );
    let row = trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::TableSize)
        .expect("table.size row");
    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_size, Some(4));
}

#[test]
fn wasm_trace_run_with_funcref_tables() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (type (func))
             (func $f)
             (elem declare funcref (ref.func $f))
             (table 1 funcref)
             (func (export "main") (result i32)
               i32.const 0
               ref.func $f
               table.set 0
               i32.const 0
               table.get 0
               drop
               i32.const 1))"#,
    );
    let table_get = trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::TableGet)
        .expect("table.get row");
    assert_eq!(table_get.table_id, Some(0));
    assert_eq!(table_get.table_index, Some(0));
    assert_eq!(table_get.table_value, Some(1));
    assert_eq!(table_get.function_type_id, Some(1));
}

#[test]
fn wasm_trace_run_with_call_indirect() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
            (type $t (func (result i32)))
            (func $f (type $t) (result i32)
                i32.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $f)
            (func (export "main") (result i32)
                i32.const 0
                call_indirect (type $t))
        )"#,
    );
    let row = trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_index, Some(0));
    assert_eq!(row.table_value, Some(1));
    assert_eq!(row.function_type_id, Some(1));
    assert_eq!(row.call_indirect_type_index, Some(0));
    assert_eq!(row.expected_type_id, Some(1));
}

#[test]
fn wasm_trace_run_with_basic_i64_ops() {
    let (_, trace, ..) = compile_and_trace_with(
        r#"(module
            (func (export "run") (result i32)
                i64.const 4294967295
                i64.const 1
                i64.add
                drop
                i64.const 6
                i64.const 7
                i64.mul
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.and
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.or
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.xor
                drop
                i64.const 4294967296
                i64.const 4294967296
                i64.sub
                i64.eqz)
        )"#,
        "run",
        &[],
    );
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I64Add));
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I64Mul));
}

#[test]
fn wasm_trace_run_with_aligned_i64_linear_memory() {
    let (_, trace, ..) = compile_and_trace_with(
        r#"(module
            (memory 1)
            (data (i32.const 8) "\88\77\66\55\44\33\22\11")
            (func (export "run") (result i32)
                i32.const 8
                i64.load
                i64.const 0x1122334455667788
                i64.sub
                i64.eqz)
        )"#,
        "run",
        &[],
    );
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I64Load));
}

#[test]
fn wasm_trace_run_with_unaligned_i64_linear_memory() {
    let (_, trace, ..) = compile_and_trace_with(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 9
                i64.const 0x1122334455667788
                i64.store
                i32.const 9
                i64.load
                i64.const 0x1122334455667788
                i64.sub
                i64.eqz)
        )"#,
        "run",
        &[],
    );
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I64Load));
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I64Store));
}

#[test]
fn wasm_trace_run_with_halfword_linear_memory() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 3
               i32.const 4660
               i32.store16
               i32.const 3
               i32.load16_u))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Store16)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Load16U)));
}

#[test]
fn wasm_trace_run_with_signed_subword_loads() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 129
               i32.store8
               i32.const 0
               i32.load8_s
               drop
               i32.const 3
               i32.const 32769
               i32.store16
               i32.const 3
               i32.load16_s))"#,
    );
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Load8S)));
    assert!(trace
        .iter()
        .any(|row| matches!(row.opcode, neo_wasm::WasmOpcode::I32Load16S)));
}

#[test]
fn wasm_trace_run_with_drop() {
    compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               i32.const 7
               drop
               i32.const 9))"#,
    );
}

#[test]
fn wasm_trace_run_with_structured_control_rows() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               block
                 i32.const 1
                 if
                   i32.const 7
                   drop
                 else
                   i32.const 9
                   drop
                 end
               end
               i32.const 5))"#,
    );
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            neo_wasm::WasmOpcode::Block,
            neo_wasm::WasmOpcode::I32Const,
            neo_wasm::WasmOpcode::If,
            neo_wasm::WasmOpcode::I32Const,
            neo_wasm::WasmOpcode::Drop,
            neo_wasm::WasmOpcode::Else,
            neo_wasm::WasmOpcode::End,
            neo_wasm::WasmOpcode::I32Const,
            neo_wasm::WasmOpcode::End,
        ]
    );
}

#[test]
fn wasm_trace_run_with_nop_and_br() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               block
                 nop
                 br 0
                 i32.const 9
                 drop
               end
               i32.const 5))"#,
    );
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            neo_wasm::WasmOpcode::Block,
            neo_wasm::WasmOpcode::Nop,
            neo_wasm::WasmOpcode::Br,
            neo_wasm::WasmOpcode::I32Const,
            neo_wasm::WasmOpcode::End,
        ]
    );
}

#[test]
fn wasm_trace_run_with_i32_wrap_i64() {
    let checked = common::checked_wasm_run(
        r#"(module
             (func (export "main") (result i32)
               i64.const 0x1234567889abcdef
               i32.wrap_i64))"#,
        "main",
        &[],
    );
    assert_eq!(checked.run.results.as_slice(), &["-1985229329"]);
    let row = checked
        .trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::I32WrapI64)
        .expect("i32.wrap_i64 row");
    assert!(row.wide_values_enabled, "i32.wrap_i64 should read an i64 input");
    assert_eq!(row.stack_write0.map(|lane| lane.value), Some(0x89ab_cdef));
    assert_eq!(row.stack_write0_hi, Some(0));
}

#[test]
fn wasm_trace_run_with_i64_extend_i32() {
    let checked = common::checked_wasm_run(
        r#"(module
             (func (export "main") (result i32)
               i32.const -1985229329
               i64.extend_i32_u
               i64.const 0x0000000089abcdef
               i64.eq
               i32.const -1985229329
               i64.extend_i32_s
               i64.const -1985229329
               i64.eq
               i32.add))"#,
        "main",
        &[],
    );
    assert_eq!(checked.run.results.as_slice(), &["2"]);
    let unsigned = checked
        .trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::I64ExtendI32U)
        .expect("i64.extend_i32_u row");
    assert!(unsigned.wide_values_enabled, "i64.extend_i32_u should write an i64");
    assert_eq!(unsigned.stack_write0.map(|lane| lane.value), Some(0x89ab_cdef));
    assert_eq!(unsigned.stack_write0_hi, Some(0));

    let signed = checked
        .trace
        .iter()
        .find(|row| row.opcode == neo_wasm::WasmOpcode::I64ExtendI32S)
        .expect("i64.extend_i32_s row");
    assert!(signed.wide_values_enabled, "i64.extend_i32_s should write an i64");
    assert_eq!(signed.stack_write0.map(|lane| lane.value), Some(0x89ab_cdef));
    assert_eq!(signed.stack_write0_hi, Some(0xffff_ffff));
}

#[test]
fn wasm_trace_run_with_integer_sign_extensions() {
    let checked = common::checked_wasm_run(
        r#"(module
             (func (export "main") (result i32)
               i32.const 128
               i32.extend8_s
               i32.const -128
               i32.eq

               i32.const 32768
               i32.extend16_s
               i32.const -32768
               i32.eq
               i32.add

               i64.const 128
               i64.extend8_s
               i64.const -128
               i64.eq
               i32.add

               i64.const 32768
               i64.extend16_s
               i64.const -32768
               i64.eq
               i32.add

               i64.const 2147483648
               i64.extend32_s
               i64.const -2147483648
               i64.eq
               i32.add))"#,
        "main",
        &[],
    );
    assert_eq!(checked.run.results.as_slice(), &["5"]);

    for opcode in [
        neo_wasm::WasmOpcode::I32Extend8S,
        neo_wasm::WasmOpcode::I32Extend16S,
        neo_wasm::WasmOpcode::I64Extend8S,
        neo_wasm::WasmOpcode::I64Extend16S,
        neo_wasm::WasmOpcode::I64Extend32S,
    ] {
        let row = checked
            .trace
            .iter()
            .find(|row| row.opcode == opcode)
            .unwrap_or_else(|| panic!("missing {opcode:?} row"));
        assert_eq!(
            row.wide_values_enabled,
            matches!(
                opcode,
                neo_wasm::WasmOpcode::I64Extend8S
                    | neo_wasm::WasmOpcode::I64Extend16S
                    | neo_wasm::WasmOpcode::I64Extend32S
            ),
            "{opcode:?} wide value flag"
        );
    }
}

#[test]
fn wasm_trace_run_folding_proof() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module (func (export "main") (result i32)
             i32.const 7
             i32.const 9
             i32.add))"#,
    );
    let prep = preprocess_seeded(&WasmVmSpec::default()).expect("prep");
    let proof = prove(&prep, &trace).expect("prove kernel run");
    verify(&prep, &proof).expect("verify kernel run");
}

#[test]
fn wasm_verify_rejects_preprocessing_with_wrong_widths() {
    let (_, trace, ..) = compile_and_trace(
        r#"(module (func (export "main") (result i32)
             i32.const 7
             i32.const 9
             i32.add))"#,
    );
    let prep = preprocess_seeded(&WasmVmSpec::default()).expect("prep");
    let proof = prove(&prep, &trace).expect("prove with canonical prep");

    let mut canonical = neo_wasm::preprocess::canonical_wasm_f_prime_shape(&WasmVmSpec::default()).expect("shape");
    canonical.plan.app_private_var_widths = vec![64; canonical.plan.app_private_var_widths.len()];
    canonical.plan.limbs = canonical.plan.app_private_var_widths.iter().sum::<usize>() + 1;
    let verifier_prep = r1cs_f_prime::preprocess_sparse_seeded_with_params(
        &canonical.sparse_r1cs,
        &canonical.plan,
        Params::test_only_from_neo_params(wasm_tiny_params()),
        0xa55ec_a11ed_15ea,
    )
    .expect("wrong-width prep");

    let err = match verify(&verifier_prep, &proof) {
        Ok(_) => panic!("verify must reject a wasm preprocessing with non-canonical widths"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("preprocessing widths do not match"),
        "unexpected error: {err}"
    );
}

fn wasm_tiny_params() -> NeoParams {
    NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 2,
        /* m      */ 1u64 << 15,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 40,
    )
    .expect("wasm tiny NeoParams must satisfy the Pi_RLC guard")
}
