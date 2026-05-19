use neo_wasm::preprocess::preprocess_seeded;
use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, prove,
    sanity_check_lookup_row, sanity_check_memory_rows, traces_from_wasmtime_steps, verify, WasmStepTrace, WasmVmSpec,
    WasmtimeTraceRun,
};

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
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, export, params).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    sanity_check_witnesses(&trace, &run);
    let pc_rom = run.pc_rom.clone();
    let pc_edge_kinds = run.pc_edge_kinds.clone();
    let function_entries = run.function_entries.clone();
    (wasm, trace, pc_rom, pc_edge_kinds, function_entries)
}

fn sanity_check_witnesses(trace: &[WasmStepTrace], run: &WasmtimeTraceRun) {
    let layout = build_wasm_lookup_binding_layout();
    let mut witnesses = Vec::with_capacity(trace.len());
    for row in trace {
        let witness = neo_wasm::builder::build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
        witnesses.push(witness);
    }
    let preload = preload_from_wasmtime_run(run, &run.initial_locals);
    sanity_check_memory_rows(layout, &witnesses, &preload)
        .unwrap_or_else(|err| panic!("memory semantics rejected trace: {err}"));
}

#[test]
fn wasm_kernel_roundtrip() {
    compile_and_trace(
        r#"(module (func (export "main") (result i32)
             i32.const 7
             i32.const 9
             i32.eq))"#,
    );
}

#[test]
fn wasm_kernel_roundtrip_with_linear_memory() {
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
fn wasm_kernel_roundtrip_with_memory_size_and_grow() {
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
fn wasm_kernel_roundtrip_with_linear_memory_offset() {
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
fn wasm_kernel_roundtrip_with_byte_linear_memory() {
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

#[test]
fn wasm_kernel_roundtrip_with_globals() {
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
fn wasm_kernel_roundtrip_with_shift_div_rem() {
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
fn wasm_kernel_roundtrip_with_compare_unary_and_rotate() {
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
fn wasm_kernel_roundtrip_with_br_table() {
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
fn wasm_kernel_roundtrip_with_table_size() {
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
fn wasm_kernel_roundtrip_with_funcref_tables() {
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
fn wasm_kernel_roundtrip_with_call_indirect() {
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
fn wasm_kernel_roundtrip_with_basic_i64_ops() {
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
fn wasm_kernel_roundtrip_with_aligned_i64_linear_memory() {
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
fn wasm_kernel_roundtrip_with_unaligned_i64_linear_memory() {
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
fn wasm_kernel_roundtrip_with_halfword_linear_memory() {
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
fn wasm_kernel_roundtrip_with_signed_subword_loads() {
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
fn wasm_kernel_roundtrip_with_drop() {
    compile_and_trace(
        r#"(module
             (func (export "main") (result i32)
               i32.const 7
               drop
               i32.const 9))"#,
    );
}

#[test]
fn wasm_kernel_roundtrip_with_structured_control_rows() {
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
fn wasm_kernel_roundtrip_with_nop_and_br() {
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
fn wasm_kernel_run_roundtrip() {
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
