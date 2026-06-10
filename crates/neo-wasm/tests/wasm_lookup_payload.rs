use neo_wasm::{
    lookup_payload, opcode_code, opcode_info_from_code, StackValueAccess, WasmOpTable, WasmOpcode, WasmOutputState,
    WasmParamInitState, WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmStepTrace,
};

fn step(opcode: WasmOpcode, lhs: u32, rhs: Option<u32>, out: u32) -> WasmStepTrace {
    let code = opcode_code(opcode);
    let info = opcode_info_from_code(code);
    let state_before = WasmStepState {
        pc: 0,
        sp: u64::from(info.stack_reads),
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: None,
        locals_fbp: 0,
        halted: false,
        param_init: WasmParamInitState::ZERO,
    };
    let state_after = WasmStepState {
        pc: 1,
        sp: u64::from(info.stack_writes),
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: None,
        locals_fbp: 0,
        halted: false,
        param_init: WasmParamInitState::ZERO,
    };
    WasmStepTrace {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        state_before,
        state_after,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        wide_values_enabled: false,
        opcode,
        info,
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: (info.stack_reads > 0).then_some(StackValueAccess::new(0, lhs)),
        stack_read1: rhs.map(|value| StackValueAccess::new(1, value)),
        stack_read2: None,
        stack_write0: (info.stack_writes > 0).then_some(StackValueAccess::new(0, out)),
        linear_memory: None,
        linear_memory_offset: 0,
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

#[test]
fn binary_lookup_payload_is_emitted_for_i32_xor() {
    let trace = step(WasmOpcode::I32Xor, 0x55aa, Some(0x0ff0), 0x5a5a);
    let payload = lookup_payload(&trace).expect("payload");
    assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
    assert_eq!(payload.op_table_id, WasmOpTable::I32Xor.op_table_id());
    assert_eq!(payload.inputs, vec![0x55aa, 0x0ff0]);
    assert_eq!(payload.outputs, vec![0x5a5a]);
}

#[test]
fn binary_lookup_payload_is_emitted_for_new_i32_lookup_family() {
    let cases = [
        (WasmOpcode::I32Shl, WasmOpTable::I32Shl, 3, 4, 48),
        (WasmOpcode::I32ShrU, WasmOpTable::I32ShrU, 128, 3, 16),
        (WasmOpcode::I32ShrS, WasmOpTable::I32ShrS, 0xffff_ff80, 3, 0xffff_fff0),
        (WasmOpcode::I32DivU, WasmOpTable::I32DivU, 22, 5, 4),
        (WasmOpcode::I32DivS, WasmOpTable::I32DivS, 0xffff_ffea, 5, 0xffff_fffc),
        (WasmOpcode::I32RemU, WasmOpTable::I32RemU, 22, 5, 2),
        (WasmOpcode::I32RemS, WasmOpTable::I32RemS, 0xffff_ffea, 5, 0xffff_fffe),
    ];
    for (opcode, op_table, lhs, rhs, out) in cases {
        let trace = step(opcode, lhs, Some(rhs), out);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
        assert_eq!(payload.op_table_id, op_table.op_table_id());
        assert_eq!(payload.inputs, vec![lhs, rhs]);
        assert_eq!(payload.outputs, vec![out]);
    }
}

#[test]
fn lookup_payload_is_emitted_for_compare_unary_and_rotate_family() {
    let unary_cases = [
        (WasmOpcode::I32Clz, WasmOpTable::I32Clz, 16, 27),
        (WasmOpcode::I32Ctz, WasmOpTable::I32Ctz, 24, 3),
    ];
    for (opcode, op_table, input, output) in unary_cases {
        let trace = step(opcode, input, None, output);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Unary);
        assert_eq!(payload.op_table_id, op_table.op_table_id());
        assert_eq!(payload.inputs, vec![input]);
        assert_eq!(payload.outputs, vec![output]);
    }

    let binary_cases = [
        (WasmOpcode::I32GtS, WasmOpTable::I32GtS, 9, 3, 1),
        (WasmOpcode::I32GtU, WasmOpTable::I32GtU, 9, 3, 1),
        (WasmOpcode::I32LeS, WasmOpTable::I32LeS, 3, 9, 1),
        (WasmOpcode::I32LeU, WasmOpTable::I32LeU, 3, 9, 1),
        (WasmOpcode::I32GeS, WasmOpTable::I32GeS, 9, 3, 1),
        (WasmOpcode::I32GeU, WasmOpTable::I32GeU, 9, 3, 1),
        (WasmOpcode::I32Rotl, WasmOpTable::I32Rotl, 0x1234_5678, 8, 0x3456_7812),
        (WasmOpcode::I32Rotr, WasmOpTable::I32Rotr, 0x1234_5678, 8, 0x7812_3456),
    ];
    for (opcode, op_table, lhs, rhs, output) in binary_cases {
        let trace = step(opcode, lhs, Some(rhs), output);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
        assert_eq!(payload.op_table_id, op_table.op_table_id());
        assert_eq!(payload.inputs, vec![lhs, rhs]);
        assert_eq!(payload.outputs, vec![output]);
    }
}

#[test]
fn i64_binary_lookup_payload_uses_four_inputs_and_two_outputs() {
    let code = opcode_code(WasmOpcode::I64Mul);
    let info = opcode_info_from_code(code);
    let trace = WasmStepTrace {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        state_before: WasmStepState {
            pc: 0,
            sp: 2,
            output: WasmOutputState::ZERO,
            call_stack_depth: 0,
            memory_pages: None,
            locals_fbp: 0,
            halted: false,
            param_init: WasmParamInitState::ZERO,
        },
        state_after: WasmStepState {
            pc: 1,
            sp: 1,
            output: WasmOutputState::ZERO,
            call_stack_depth: 0,
            memory_pages: None,
            locals_fbp: 0,
            halted: false,
            param_init: WasmParamInitState::ZERO,
        },
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        wide_values_enabled: true,
        opcode: WasmOpcode::I64Mul,
        info,
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: Some(StackValueAccess::with_hi(0, 3, 1)),
        stack_read1: Some(StackValueAccess::with_hi(1, 5, 2)),
        stack_read2: None,
        stack_write0: Some(StackValueAccess::with_hi(0, 15, 7)),
        linear_memory: None,
        linear_memory_offset: 0,
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
    };
    let payload = lookup_payload(&trace).expect("payload");
    assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Tuple(4));
    assert_eq!(payload.op_table_id, WasmOpTable::I64Mul.op_table_id());
    assert_eq!(payload.inputs, vec![3, 1, 5, 2]);
    assert_eq!(payload.outputs, vec![15, 7]);
}

#[test]
fn i64_unary_lookup_payload_uses_two_inputs_and_two_outputs() {
    // Distinct lo/hi limb values so a swapped limb order cannot pass.
    let cases = [
        // clz(0x1_0000_1234) = 31
        (WasmOpcode::I64Clz, WasmOpTable::I64Clz, (0x1234, 1), (31, 0)),
        // ctz(0x10_0000_0000) = 36
        (WasmOpcode::I64Ctz, WasmOpTable::I64Ctz, (0, 0x10), (36, 0)),
        // popcnt(0xF_0000_F0F0) = 12
        (WasmOpcode::I64Popcnt, WasmOpTable::I64Popcnt, (0xF0F0, 0xF), (12, 0)),
    ];
    for (opcode, op_table, (in_lo, in_hi), (out_lo, out_hi)) in cases {
        let mut trace = step(opcode, in_lo, None, out_lo);
        trace.wide_values_enabled = true;
        trace.stack_read0 = Some(StackValueAccess::with_hi(0, in_lo, in_hi));
        trace.stack_write0 = Some(StackValueAccess::with_hi(0, out_lo, out_hi));
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Tuple(2));
        assert_eq!(payload.op_table_id, op_table.op_table_id());
        assert_eq!(payload.inputs, vec![in_lo, in_hi]);
        assert_eq!(payload.outputs, vec![out_lo, out_hi]);
    }
}
