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
    use neo_wasm::builder::build_witness_vector;
    use neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI;
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

/// The width-family flags (`is_byte_width` etc.) must be pinned per opcode:
/// zeroing the byte-width family on an i64.store8 row would otherwise
/// vacuously satisfy the byte-routing gates and decouple the stored byte
/// from the memory lane. The `linear memory width opcode binding` rows
/// reject it. (Same pin now protects the i32 subword ops.)
#[test]
fn i64_store8_width_family_pin_is_enforced() {
    use neo_ccs::check_ccs_rowwise_zero;
    use neo_math::F;
    use neo_wasm::builder::build_witness_vector;
    use neo_wasm::layout::{
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_IS_BYTE_WIDTH,
    };
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
