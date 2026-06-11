mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{
    COL_CALL_PARAM_COUNT, COL_CALL_STACK_POP_CALLER_FBP, COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_POP_RETURN_PC,
    COL_CALL_STACK_PUSH_PRESENT, COL_CURRENT_FUNCTION_NUM_LOCALS, COL_DIV_TRAP, COL_LOCALS_FBP_AFTER,
    COL_LOCALS_FBP_BEFORE, COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_REMAINING_AFTER,
    COL_PARAM_INIT_REMAINING_AFTER_INV, COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PC_ROM_ACTIVE,
    COL_STACK_READ0_ACTIVE, COL_STACK_READ1_ACTIVE, COL_STACK_READ2_ACTIVE, COL_STACK_READS, COL_STACK_WRITE0_ACTIVE,
    COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::WasmRowKind;
use neo_wasm::{
    collect_wasmtime_steps, opcode_code, opcode_info_from_code, traces_from_rwasm_instr_states,
    traces_from_wasmtime_steps, traces_from_wasmtime_wasm_bytes, LinearMemoryAccess, StackValueAccess, WasmAuxOpcode,
    WasmOpcode, WasmOutputState, WasmParamInitState, WasmPcEdgeKind, WasmStepState, WasmStepTrace,
};
use p3_field::PrimeCharacteristicRing;
use rwasm::mem::{MemoryAccessRecord, MemoryReadRecord, MemoryRecordEnum, MemoryWriteRecord};
use rwasm::{Opcode as ConcreteOpcode, TracerInstrState};

fn step(
    cycle: u64,
    pc_before: u64,
    opcode_code: u16,
    sp_before: u64,
    sp_after: u64,
    stack_read0: Option<StackValueAccess>,
    stack_read1: Option<StackValueAccess>,
    stack_read2: Option<StackValueAccess>,
    stack_write0: Option<StackValueAccess>,
    linear_memory: Option<LinearMemoryAccess>,
    linear_memory_offset: u64,
    halted: bool,
) -> WasmStepTrace {
    fn state(pc: u64, sp: u64, halted: bool) -> WasmStepState {
        WasmStepState {
            pc,
            sp,
            output: WasmOutputState::ZERO,
            call_stack_depth: 0,
            memory_pages: None,
            locals_fbp: 0,
            halted,
            trapped: false,
            param_init: WasmParamInitState::ZERO,
        }
    }

    fn physical(access: Option<StackValueAccess>) -> Option<StackValueAccess> {
        access.map(|lane| StackValueAccess::new(lane.addr_lo * 2, lane.value_lo))
    }

    let opcode = opcode_info_from_code(opcode_code).opcode;
    let div_zero_trap = opcode.traps_on_zero_divisor()
        && stack_read1.is_some_and(|lane| lane.value_lo == 0 && lane.value_hi.unwrap_or(0) == 0);
    WasmStepTrace {
        cycle,
        row_kind: WasmRowKind::Program,
        state_before: state(pc_before, sp_before, false),
        state_after: WasmStepState {
            trapped: matches!(opcode, WasmOpcode::Unreachable) || div_zero_trap,
            ..state(pc_before + 1, sp_after, halted)
        },
        control_choice: 0,
        pc_edge_kind: match opcode {
            WasmOpcode::Return | WasmOpcode::End => WasmPcEdgeKind::ReturnLike,
            WasmOpcode::CallIndirect => WasmPcEdgeKind::DynamicCallIndirect,
            WasmOpcode::Unreachable => WasmPcEdgeKind::Terminal,
            _ => WasmPcEdgeKind::Static,
        },
        wide_values_enabled: opcode_info_from_code(opcode_code).opcode.uses_wide_values(),
        opcode: opcode_info_from_code(opcode_code).opcode,
        info: opcode_info_from_code(opcode_code),
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: physical(stack_read0),
        stack_read1: physical(stack_read1),
        stack_read2: physical(stack_read2),
        stack_write0: physical(stack_write0),
        linear_memory,
        linear_memory_offset,
        local_index: None,
        local_read_value: None,
        local_read_value_hi: None,
        local_write_value: None,
        local_write_value_hi: None,
        global_index: None,
        global_read_value: None,
        global_read_value_hi: None,
        global_write_value: None,
        global_write_value_hi: None,
        table_id: None,
        table_index: None,
        table_value: None,
        function_ref: None,
        target_function_is_guest: false,
        function_type_id: None,
        call_indirect_type_index: None,
        expected_type_id: None,
        table_size: None,
        call_param_count: None,
        call_result_count: None,
        call_stack_push: None,
        call_stack_pop: None,
    }
}

fn trace_from_wat(wat_src: &str) -> Vec<WasmStepTrace> {
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    traces_from_wasmtime_steps(&run.steps).expect("normalize trace")
}

#[test]
fn normalization_tracks_stack_pointer_for_binary_op() {
    let rows = vec![
        TracerInstrState {
            program_counter: 0,
            opcode: ConcreteOpcode::I32Const(7u32.into()),
            value: 7,
            memory_changes: vec![],
            table_changes: vec![],
            table_size_changes: vec![],
            next_table_idx: None,
            call_id: 0,
            memory_access: MemoryAccessRecord::default(),
        },
        TracerInstrState {
            program_counter: 1,
            opcode: ConcreteOpcode::I32Const(9u32.into()),
            value: 9,
            memory_changes: vec![],
            table_changes: vec![],
            table_size_changes: vec![],
            next_table_idx: None,
            call_id: 0,
            memory_access: MemoryAccessRecord::default(),
        },
        TracerInstrState {
            program_counter: 2,
            opcode: ConcreteOpcode::I32Add,
            value: 0,
            memory_changes: vec![],
            table_changes: vec![],
            table_size_changes: vec![],
            next_table_idx: None,
            call_id: 0,
            memory_access: MemoryAccessRecord {
                a: Some(MemoryRecordEnum::Read(MemoryReadRecord::new(7, 1, 1, 0, 0))),
                b: Some(MemoryRecordEnum::Read(MemoryReadRecord::new(9, 1, 2, 0, 0))),
                c: Some(MemoryRecordEnum::Write(MemoryWriteRecord::new(16, 1, 3, 0, 0, 0))),
                memory: None,
            },
        },
    ];

    let trace = traces_from_rwasm_instr_states(&rows, 0).expect("normalize");
    assert_eq!(trace[0].state_before.sp, 0);
    assert_eq!(trace[0].state_after.sp, 1);
    assert_eq!(trace[1].state_before.sp, 1);
    assert_eq!(trace[1].state_after.sp, 2);
    assert_eq!(trace[2].state_before.sp, 2);
    assert_eq!(trace[2].state_after.sp, 1);
    assert_eq!(trace[2].stack_read0.expect("lhs").addr_lo, 0);
    assert_eq!(trace[2].stack_read1.expect("rhs").addr_lo, 2);
    assert_eq!(trace[2].stack_write0.expect("out").addr_lo, 0);
    assert_eq!(trace[2].opcode, WasmOpcode::I32Add);
}

#[test]
fn direct_rows_satisfy_real_wasm_ccs() {
    let rows = vec![
        build_witness_vector(&step(
            0,
            0,
            opcode_code(WasmOpcode::I32Const),
            0,
            1,
            None,
            None,
            None,
            Some(StackValueAccess::new(0, 7)),
            None,
            0,
            false,
        )),
        build_witness_vector(&step(
            1,
            1,
            opcode_code(WasmOpcode::I32Const),
            1,
            2,
            None,
            None,
            None,
            Some(StackValueAccess::new(1, 9)),
            None,
            0,
            false,
        )),
        build_witness_vector(&step(
            2,
            2,
            opcode_code(WasmOpcode::I32Add),
            2,
            1,
            Some(StackValueAccess::new(0, 7)),
            Some(StackValueAccess::new(1, 9)),
            None,
            Some(StackValueAccess::new(0, 16)),
            None,
            0,
            false,
        )),
        build_witness_vector(&step(
            3,
            3,
            opcode_code(WasmOpcode::I32Sub),
            2,
            1,
            Some(StackValueAccess::new(0, 20)),
            Some(StackValueAccess::new(1, 5)),
            None,
            Some(StackValueAccess::new(0, 15)),
            None,
            0,
            false,
        )),
        build_witness_vector(&step(
            4,
            4,
            opcode_code(WasmOpcode::Select),
            3,
            1,
            Some(StackValueAccess::new(0, 11)),
            Some(StackValueAccess::new(1, 22)),
            Some(StackValueAccess::new(2, 1)),
            Some(StackValueAccess::new(0, 11)),
            None,
            0,
            false,
        )),
        build_witness_vector(&step(
            5,
            5,
            opcode_code(WasmOpcode::Return),
            1,
            1,
            None,
            None,
            None,
            None,
            None,
            0,
            true,
        )),
    ];

    for (idx, row) in rows.iter().enumerate() {
        assert_satisfied(row, &format!("row {idx}"));
    }
}

#[test]
fn add_row_rejects_tampered_output_value() {
    let mut row = build_witness_vector(&step(
        0,
        2,
        opcode_code(WasmOpcode::I32Add),
        2,
        0,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 9)),
        None,
        Some(StackValueAccess::new(0, 16)),
        None,
        0,
        false,
    ));
    row[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(17);
    assert_rejected(&row, "tampered i32.add output");
}

#[test]
fn add_row_rejects_tampered_static_stack_arity() {
    let mut row = build_witness_vector(&step(
        0,
        2,
        opcode_code(WasmOpcode::I32Add),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 9)),
        None,
        Some(StackValueAccess::new(0, 16)),
        None,
        0,
        false,
    ));

    row[COL_STACK_READS] = F::ONE;
    row[COL_STACK_WRITES] = F::ZERO;
    row[COL_STACK_READ0_ACTIVE] = F::ONE;
    row[COL_STACK_READ1_ACTIVE] = F::ZERO;
    row[COL_STACK_READ2_ACTIVE] = F::ZERO;
    row[COL_STACK_WRITE0_ACTIVE] = F::ZERO;

    assert_rejected(&row, "tampered i32.add stack arity");
}

#[test]
fn i32_add_wraps_on_overflow() {
    // 0xFFFF_FFFF + 0x0000_0001 = 0x1_0000_0000 → wasm result is 0, carry = 1.
    let row = build_witness_vector(&step(
        0,
        2,
        opcode_code(WasmOpcode::I32Add),
        2,
        1,
        Some(StackValueAccess::new(0, 0xFFFF_FFFF)),
        Some(StackValueAccess::new(1, 1)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "wrapping i32.add row");
}

#[test]
fn i32_sub_wraps_on_underflow() {
    // 0x0000_0000 - 0x0000_0001 = 0xFFFF_FFFF (mod 2^32), borrow = 1.
    let row = build_witness_vector(&step(
        0,
        2,
        opcode_code(WasmOpcode::I32Sub),
        2,
        1,
        Some(StackValueAccess::new(0, 0)),
        Some(StackValueAccess::new(1, 1)),
        None,
        Some(StackValueAccess::new(0, 0xFFFF_FFFF)),
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "wrapping i32.sub row");
}

#[test]
fn selector_opcode_mismatch_is_rejected() {
    let mut row = build_witness_vector(&step(
        0,
        2,
        opcode_code(WasmOpcode::I32Add),
        2,
        0,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 9)),
        None,
        Some(StackValueAccess::new(0, 16)),
        None,
        0,
        false,
    ));
    row[1] = F::from_u64(u64::from(opcode_code(WasmOpcode::I32Sub)));
    assert_rejected(&row, "opcode byte does not match active selector");
}

#[test]
fn i32_load_row_is_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 77
               i32.store
               i32.const 0
               i32.load))"#,
    );
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("load row");
    let row = build_witness_vector(load);
    assert_satisfied(&row, "i32.load row");
}

#[test]
fn i32_store_row_is_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 9
               i32.store
               i32.const 0))"#,
    );
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Store)
        .expect("store row");
    let row = build_witness_vector(store);
    assert_satisfied(&row, "i32.store row");
}

#[test]
fn i32_load8_u_and_store8_rows_are_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 511
               i32.store8
               i32.const 0
               i32.load8_u))"#,
    );
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Store8)
        .expect("store8 row");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load8U)
        .expect("load8_u row");
    assert_satisfied(&build_witness_vector(store), "i32.store8 row");
    assert_satisfied(&build_witness_vector(load), "i32.load8_u row");
}

/// Row-local guard: subword stores must reject changed bytes outside the
/// store's write window unless the claimed prior state changes too.
#[test]
fn i32_load16_u_and_store16_rows_are_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 3
               i32.const 4660
               i32.store16
               i32.const 3
               i32.load16_u))"#,
    );
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Store16)
        .expect("store16 row");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load16U)
        .expect("load16_u row");
    assert_satisfied(&build_witness_vector(store), "i32.store16 row");
    assert_satisfied(&build_witness_vector(load), "i32.load16_u row");
}

#[test]
fn i32_load8_s_and_load16_s_rows_are_accepted() {
    let trace = trace_from_wat(
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
    let load8 = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load8S)
        .expect("load8_s row");
    let load16 = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load16S)
        .expect("load16_s row");
    assert_satisfied(&build_witness_vector(load8), "i32.load8_s row");
    assert_satisfied(&build_witness_vector(load16), "i32.load16_s row");
}

#[test]
fn global_get_and_set_rows_are_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (global (mut i32) (i32.const 7))
             (func (export "main") (result i32)
               global.get 0
               drop
               i32.const 9
               global.set 0
               global.get 0))"#,
    );
    let get_rows = trace
        .iter()
        .filter(|row| row.opcode == WasmOpcode::GlobalGet)
        .collect::<Vec<_>>();
    let set_row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::GlobalSet)
        .expect("global.set row");
    assert_eq!(get_rows.len(), 2, "expected two global.get rows");
    for (idx, row) in get_rows.into_iter().enumerate() {
        assert_satisfied(&build_witness_vector(row), &format!("global.get row {idx}"));
    }
    assert_satisfied(&build_witness_vector(set_row), "global.set row");
}

#[test]
fn memory_size_and_grow_rows_are_accepted() {
    let trace = trace_from_wat(
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
    let size_rows = trace
        .iter()
        .filter(|row| row.opcode == WasmOpcode::MemorySize)
        .collect::<Vec<_>>();
    let grow_row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::MemoryGrow)
        .expect("memory.grow row");
    assert_eq!(size_rows.len(), 2, "expected two memory.size rows");
    for (idx, row) in size_rows.into_iter().enumerate() {
        assert_satisfied(&build_witness_vector(row), &format!("memory.size row {idx}"));
    }
    assert_satisfied(&build_witness_vector(grow_row), "memory.grow row");
}

#[test]
fn table_size_row_is_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (table 4 funcref)
             (func (export "main") (result i32)
               table.size 0))"#,
    );
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::TableSize)
        .expect("table.size row");
    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_size, Some(4));
    assert_satisfied(&build_witness_vector(row), "table.size row");
}

#[test]
fn ref_func_and_table_rows_are_accepted() {
    let trace = trace_from_wat(
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
    let ref_func = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::RefFunc)
        .expect("ref.func row");
    let table_set = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::TableSet)
        .expect("table.set row");
    let table_get = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::TableGet)
        .expect("table.get row");
    assert_eq!(ref_func.stack_write0.expect("ref.func write").value_lo, 1);
    assert_eq!(ref_func.function_type_id, Some(1));
    assert_eq!(table_set.table_id, Some(0));
    assert_eq!(table_set.table_index, Some(0));
    assert_eq!(table_set.table_value, Some(1));
    assert_eq!(table_set.function_type_id, Some(1));
    assert_eq!(table_get.table_id, Some(0));
    assert_eq!(table_get.table_index, Some(0));
    assert_eq!(table_get.table_value, Some(1));
    assert_eq!(table_get.function_type_id, Some(1));
    assert_satisfied(&build_witness_vector(ref_func), "ref.func row");
    assert_satisfied(&build_witness_vector(table_set), "table.set row");
    assert_satisfied(&build_witness_vector(table_get), "table.get row");
}

#[test]
fn call_indirect_row_is_accepted() {
    let wasm = wat::parse_str(
        r#"(module
            (type $t (func (result i32)))
            (func $f (type $t) (result i32)
                i32.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $f)
            (func (export "run") (result i32)
                i32.const 0
                call_indirect (type $t))
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_index, Some(0));
    assert_eq!(row.table_value, Some(1));
    assert_eq!(row.function_type_id, Some(1));
    assert_eq!(row.call_indirect_type_index, Some(0));
    assert_eq!(row.expected_type_id, Some(1));
    assert_eq!(row.stack_reads_override, Some(1));
    assert_satisfied(&build_witness_vector(row), "call_indirect row");
}

#[test]
fn call_indirect_row_rejects_tampered_dynamic_stack_arity() {
    let wasm = wat::parse_str(
        r#"(module
            (type $t (func (result i32)))
            (func $f (type $t) (result i32)
                i32.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $f)
            (func (export "run") (result i32)
                i32.const 0
                call_indirect (type $t))
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    let mut witness = build_witness_vector(row);
    witness[COL_CALL_PARAM_COUNT] = F::ONE;

    assert_rejected(&witness, "call_indirect row with tampered param count");
}

#[test]
fn i64_rows_are_accepted() {
    let wasm = wat::parse_str(
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
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    for opcode in [
        WasmOpcode::I64Const,
        WasmOpcode::I64Add,
        WasmOpcode::I64Mul,
        WasmOpcode::I64And,
        WasmOpcode::I64Or,
        WasmOpcode::I64Xor,
        WasmOpcode::I64Sub,
        WasmOpcode::I64Eqz,
    ] {
        let row = trace
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("i64 row");
        assert!(row.wide_values_enabled, "{opcode:?} should enable wide values");
        assert_satisfied(&build_witness_vector(row), "i64 row");
    }
}

#[test]
fn i64_linear_memory_rows_are_accepted() {
    let wasm = wat::parse_str(
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
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Load)
        .expect("i64.load row");
    assert!(load.wide_values_enabled, "i64.load should enable wide values");
    assert_satisfied(&build_witness_vector(load), "i64.load row");
}

#[test]
fn i32_wrap_i64_row_projects_low_limb() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I32WrapI64),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_read0 = row
        .stack_read0
        .map(|lane| lane.with_optional_hi(Some(0x1234_5678)));
    row.stack_write0 = row.stack_write0.map(|lane| lane.with_optional_hi(Some(0)));

    assert_satisfied(&build_witness_vector(&row), "i32.wrap_i64 row");
}

#[test]
fn i32_wrap_i64_row_rejects_tampered_output() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I32WrapI64),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdee)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_read0 = row
        .stack_read0
        .map(|lane| lane.with_optional_hi(Some(0x1234_5678)));
    row.stack_write0 = row.stack_write0.map(|lane| lane.with_optional_hi(Some(0)));

    assert_rejected(&build_witness_vector(&row), "i32.wrap_i64 row with tampered output");
}

#[test]
fn i32_wrap_i64_row_rejects_tampered_high_output() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I32WrapI64),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_read0 = row
        .stack_read0
        .map(|lane| lane.with_optional_hi(Some(0x1234_5678)));
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_WRITE0_VALUE_HI] = F::ONE;

    assert_rejected(&witness, "i32.wrap_i64 row with tampered high output");
}

#[test]
fn i64_extend_i32_u_row_zero_extends() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I64ExtendI32U),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_write0 = row.stack_write0.map(|lane| lane.with_optional_hi(Some(0)));

    assert_satisfied(&build_witness_vector(&row), "i64.extend_i32_u row");
}

#[test]
fn i64_extend_i32_u_row_rejects_nonzero_high_output() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I64ExtendI32U),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_WRITE0_VALUE_HI] = F::ONE;

    assert_rejected(&witness, "i64.extend_i32_u row with nonzero high output");
}

#[test]
fn i64_extend_i32_s_row_sign_extends_negative_value() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I64ExtendI32S),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_write0 = row
        .stack_write0
        .map(|lane| lane.with_optional_hi(Some(0xffff_ffff)));

    assert_satisfied(&build_witness_vector(&row), "i64.extend_i32_s negative row");
}

#[test]
fn i64_extend_i32_s_row_sign_extends_positive_value() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I64ExtendI32S),
        1,
        1,
        Some(StackValueAccess::new(0, 0x09ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x09ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    row.stack_write0 = row.stack_write0.map(|lane| lane.with_optional_hi(Some(0)));

    assert_satisfied(&build_witness_vector(&row), "i64.extend_i32_s positive row");
}

#[test]
fn i64_extend_i32_s_row_rejects_tampered_high_output() {
    let mut row = step(
        0,
        0,
        opcode_code(WasmOpcode::I64ExtendI32S),
        1,
        1,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        None,
        Some(StackValueAccess::new(0, 0x89ab_cdef)),
        None,
        0,
        false,
    );
    row.wide_values_enabled = true;
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;

    assert_rejected(&witness, "i64.extend_i32_s row with tampered high output");
}

#[test]
fn i64_store_row_is_accepted() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 8
                i64.const 0x1122334455667788
                i64.store
                i32.const 0)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    assert!(store.wide_values_enabled, "i64.store should enable wide values");
    assert_satisfied(&build_witness_vector(store), "i64.store row");
}

#[test]
fn i64_unaligned_linear_memory_rows_are_accepted() {
    let wasm = wat::parse_str(
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
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Load)
        .expect("i64.load row");
    assert_satisfied(&build_witness_vector(store), "unaligned i64.store row");
    assert_satisfied(&build_witness_vector(load), "unaligned i64.load row");
}

#[test]
fn i32_load_row_accepts_nonzero_offset() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 77
               i32.const 4
               i32.store offset=8
               i32.const 4
               i32.load offset=8))"#,
    );
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("load row");
    let row = build_witness_vector(load);
    assert_satisfied(&row, "i32.load row with nonzero offset");
}

#[test]
fn drop_row_is_accepted() {
    let row = build_witness_vector(&step(
        0,
        11,
        opcode_code(WasmOpcode::Drop),
        1,
        0,
        Some(StackValueAccess::new(0, 77)),
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "drop row");
}

#[test]
fn drop_i64_row_zeroes_unused_high_limb_in_witness() {
    let trace = traces_from_wasmtime_wasm_bytes(
        &wat::parse_str(
            r#"(module
               (func (export "run") (result i32)
                 i64.const 4294967296
                 drop
                 i32.const 1))"#,
        )
        .expect("wat"),
        "run",
    )
    .expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Drop)
        .expect("drop row");
    assert_eq!(row.stack_read0.and_then(|lane| lane.value_hi), Some(1));
    assert_satisfied(&build_witness_vector(row), "i64 drop row");
}

#[test]
fn nop_row_is_accepted() {
    let row = build_witness_vector(&step(
        0,
        12,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "nop row");
}

#[test]
fn br_row_is_accepted() {
    let row = build_witness_vector(&step(
        0,
        13,
        opcode_code(WasmOpcode::Br),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "br row");
}

#[test]
fn if_row_is_accepted() {
    let row = build_witness_vector(&step(
        0,
        12,
        opcode_code(WasmOpcode::If),
        1,
        0,
        Some(StackValueAccess::new(0, 1)),
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "if row");
}

#[test]
fn structured_end_row_is_accepted() {
    let trace = trace_from_wat(
        r#"(module
             (func (export "main") (result i32)
               block
                 i32.const 1
                 drop
               end
               i32.const 5))"#,
    );
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::End && !row.state_after.halted)
        .expect("structured end row");
    assert_eq!(row.pc_edge_kind, WasmPcEdgeKind::Static);
    assert_satisfied(&build_witness_vector(row), "structured end row");
}

#[test]
fn unreachable_row_requires_halted_boundary() {
    let accepted = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ));
    assert_satisfied(&accepted, "unreachable row");

    let rejected = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_rejected(&rejected, "unreachable row with halted=0");
}

#[test]
fn unreachable_row_rejects_clean_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ));
    assert_satisfied(&witness, "unreachable row with trapped_after=1");
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    assert_rejected(&witness, "unreachable row claiming a clean (non-trapped) exit");
}

#[test]
fn non_trap_row_rejects_fake_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&witness, "nop row");
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "nop row claiming a trap");
}

#[test]
fn program_row_rejects_execution_after_trap() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    // trapped_before = trapped_after = 1 satisfies the transition row, so
    // the rejection must come from `is_program_row · trapped_before = 0`.
    witness[COL_TRAPPED_BEFORE] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "program row executing after a trap");
}

#[test]
fn div_by_zero_row_rejects_clean_claim() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 0)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        true,
    ));
    assert_satisfied(&witness, "i32.div_u row trapping on a zero divisor");
    // Hiding the trap: div_trap = 0 contradicts (Σ div sel) · divisor_is_zero.
    witness[COL_DIV_TRAP] = F::ZERO;
    assert_rejected(&witness, "div-by-zero row claiming no trap");
}

#[test]
fn div_row_rejects_fake_trap_on_nonzero_divisor() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 2)),
        None,
        Some(StackValueAccess::new(0, 3)),
        None,
        0,
        false,
    ));
    assert_satisfied(&witness, "i32.div_u row with a nonzero divisor");
    // Faking the trap: divisor_is_zero is pinned to 0 by the zero test, so
    // div_trap = (Σ div sel) · 0 must be 0.
    witness[COL_DIV_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "div row faking a trap on a nonzero divisor");
}

#[test]
fn return_row_requires_halted_boundary() {
    let row = build_witness_vector(&step(
        0,
        9,
        opcode_code(WasmOpcode::Return),
        1,
        1,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_rejected(&row, "return row with halted=0");
}

#[test]
fn non_final_return_row_is_accepted() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.call_stack_pop.is_some())
        .expect("non-final return row");
    assert_satisfied(&build_witness_vector(row), "non-final return row");
}

#[test]
fn non_final_return_row_rejects_tampered_return_pc() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.call_stack_pop.is_some())
        .expect("non-final return row");
    let mut witness = build_witness_vector(row);
    witness[COL_CALL_STACK_POP_RETURN_PC] += F::ONE;
    assert_rejected(&witness, "non-final return row with tampered return pc");
}

#[test]
fn non_final_return_row_rejects_tampered_caller_fbp() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.call_stack_pop.is_some())
        .expect("non-final return row");
    let mut witness = build_witness_vector(row);
    witness[COL_CALL_STACK_POP_CALLER_FBP] += F::ONE;
    assert_rejected(&witness, "non-final return row with tampered caller fbp");
}

#[test]
fn call_row_rejects_tampered_locals_fbp_after() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Call)
        .expect("call row");
    let mut witness = build_witness_vector(row);
    witness[COL_LOCALS_FBP_AFTER] += F::ONE;
    assert_rejected(&witness, "call row with tampered locals fbp after");
}

#[test]
fn call_row_rejects_tampered_current_function_num_locals() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (param i32) (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = collect_wasmtime_steps(&wasm, "run", &[9])
        .and_then(|run| traces_from_wasmtime_steps(&run.steps))
        .expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Call)
        .expect("call row");
    let mut witness = build_witness_vector(row);
    witness[COL_CURRENT_FUNCTION_NUM_LOCALS] += F::ONE;
    assert_rejected(&witness, "call row with tampered current function local count");
}

#[test]
fn static_program_row_rejects_disabled_pc_rom_active() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 5))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.row_kind.is_program() && row.pc_edge_kind == WasmPcEdgeKind::Static)
        .expect("static program row");
    let mut witness = build_witness_vector(row);
    witness[COL_PC_ROM_ACTIVE] = F::ZERO;
    assert_rejected(&witness, "static program row with disabled pc_rom_active");
}

#[test]
fn guest_call_row_rejects_suppressed_call_stack_push() {
    let wasm = wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Call)
        .expect("call row");
    assert!(row.target_function_is_guest);
    let mut witness = build_witness_vector(row);
    witness[COL_CALL_STACK_PUSH_PRESENT] = F::ZERO;
    witness[COL_PARAM_INIT_ACTIVE_AFTER] = F::ZERO;
    witness[COL_PARAM_INIT_REMAINING_AFTER] = F::ZERO;
    witness[COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO] = F::ONE;
    witness[COL_PARAM_INIT_REMAINING_AFTER_INV] = F::ZERO;
    witness[COL_LOCALS_FBP_AFTER] = witness[COL_LOCALS_FBP_BEFORE];
    witness[COL_CALL_STACK_POP_PRESENT] = F::ZERO;
    assert_rejected(&witness, "guest call row with suppressed call-stack push");
}

#[test]
fn i64_call_param_init_aux_row_is_wide_and_accepted() {
    let wasm = wat::parse_str(
        r#"(module
            (func $take64 (param i64)
                local.get 0
                drop)
            (func (export "run") (result i32)
                i64.const 4294967296
                call $take64
                i32.const 1))
        "#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let aux = trace
        .iter()
        .find(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::CallParamInit))
        .expect("i64 param-init aux row");
    assert!(aux.wide_values_enabled, "i64 param aux row must allow high limbs");
    assert_eq!(aux.stack_read0.and_then(|lane| lane.value_hi), Some(1));
    assert_eq!(aux.local_write_value_hi, Some(1));
    let witness = build_witness_vector(aux);
    assert_satisfied(&witness, "i64 call param init aux row");
}

#[test]
fn select_row_accepts_nonzero_condition() {
    let row = build_witness_vector(&step(
        0,
        4,
        opcode_code(WasmOpcode::Select),
        3,
        1,
        Some(StackValueAccess::new(0, 11)),
        Some(StackValueAccess::new(1, 22)),
        Some(StackValueAccess::new(2, 2)),
        Some(StackValueAccess::new(0, 11)),
        None,
        0,
        false,
    ));
    assert_satisfied(&row, "select row with nonzero condition");
}

#[test]
fn select_row_rejects_nonzero_condition_with_rhs_output() {
    let row = build_witness_vector(&step(
        0,
        4,
        opcode_code(WasmOpcode::Select),
        3,
        1,
        Some(StackValueAccess::new(0, 11)),
        Some(StackValueAccess::new(1, 22)),
        Some(StackValueAccess::new(2, 2)),
        Some(StackValueAccess::new(0, 22)),
        None,
        0,
        false,
    ));
    assert_rejected(&row, "select row with nonzero condition and rhs output");
}

#[test]
fn i64_select_row_rejects_tampered_high_output() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i64)
                i64.const 0x20000000b
                i64.const 0x300000016
                i32.const 2
                select))"#,
    )
    .expect("wat");
    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let select = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Select)
        .expect("i64 select row");
    assert_eq!(select.stack_write0.and_then(|lane| lane.value_hi), Some(2));
    let mut witness = build_witness_vector(select);
    assert_satisfied(&witness, "i64 select row with nonzero condition");
    witness[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
    assert_rejected(&witness, "i64 select row with tampered high output");
}
