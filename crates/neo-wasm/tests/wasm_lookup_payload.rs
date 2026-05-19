use neo_wasm::{
    lookup_payload, opcode_code, opcode_info_from_code, StackLaneAccess, WasmOpcode, WasmParamInitState,
    WasmPcEdgeKind, WasmRowKind, WasmShoutOpcode, WasmStepTrace, WasmTraceBuilder, WasmVmSpec,
};

fn step(opcode: WasmOpcode, lhs: u32, rhs: Option<u32>, out: u32) -> WasmStepTrace {
    let code = opcode_code(opcode);
    let info = opcode_info_from_code(code);
    WasmStepTrace {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        pc_before: 0,
        pc_after: 1,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        param_init_before: WasmParamInitState::ZERO,
        param_init_after: WasmParamInitState::ZERO,
        wide_values_enabled: false,
        opcode_code: code,
        opcode,
        info,
        stack_reads_override: None,
        stack_writes_override: None,
        sp_before: u64::from(info.stack_reads),
        sp_after: u64::from(info.stack_writes),
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: (info.stack_reads > 0).then_some(StackLaneAccess { addr: 0, value: lhs }),
        stack_read0_hi: None,
        stack_read1: rhs.map(|value| StackLaneAccess { addr: 1, value }),
        stack_read1_hi: None,
        stack_read2: None,
        stack_read2_hi: None,
        stack_write0: (info.stack_writes > 0).then_some(StackLaneAccess { addr: 0, value: out }),
        stack_write0_hi: None,
        linear_memory: None,
        linear_memory_offset: 0,
        memory_pages_before: None,
        memory_pages_after: None,
        halted: false,
        locals_fbp: 0,
        locals_fbp_after: 0,
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
fn unary_lookup_payload_is_emitted_for_i32_eqz() {
    let trace = step(WasmOpcode::I32Eqz, 11, None, 0);
    let payload = lookup_payload(&trace).expect("payload");
    assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Unary);
    assert_eq!(payload.shout_id, WasmShoutOpcode::I32Eqz.to_shout_id());
    assert_eq!(payload.inputs, vec![11]);
    assert_eq!(payload.outputs, vec![0]);
}

#[test]
fn binary_lookup_payload_is_emitted_for_i32_xor() {
    let trace = step(WasmOpcode::I32Xor, 0x55aa, Some(0x0ff0), 0x5a5a);
    let payload = lookup_payload(&trace).expect("payload");
    assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
    assert_eq!(payload.shout_id, WasmShoutOpcode::I32Xor.to_shout_id());
    assert_eq!(payload.inputs, vec![0x55aa, 0x0ff0]);
    assert_eq!(payload.outputs, vec![0x5a5a]);
}

#[test]
fn binary_lookup_payload_is_emitted_for_new_i32_lookup_family() {
    let cases = [
        (WasmOpcode::I32Shl, WasmShoutOpcode::I32Shl, 3, 4, 48),
        (WasmOpcode::I32ShrU, WasmShoutOpcode::I32ShrU, 128, 3, 16),
        (
            WasmOpcode::I32ShrS,
            WasmShoutOpcode::I32ShrS,
            0xffff_ff80,
            3,
            0xffff_fff0,
        ),
        (WasmOpcode::I32DivU, WasmShoutOpcode::I32DivU, 22, 5, 4),
        (
            WasmOpcode::I32DivS,
            WasmShoutOpcode::I32DivS,
            0xffff_ffea,
            5,
            0xffff_fffc,
        ),
        (WasmOpcode::I32RemU, WasmShoutOpcode::I32RemU, 22, 5, 2),
        (
            WasmOpcode::I32RemS,
            WasmShoutOpcode::I32RemS,
            0xffff_ffea,
            5,
            0xffff_fffe,
        ),
    ];
    for (opcode, shout_opcode, lhs, rhs, out) in cases {
        let trace = step(opcode, lhs, Some(rhs), out);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
        assert_eq!(payload.shout_id, shout_opcode.to_shout_id());
        assert_eq!(payload.inputs, vec![lhs, rhs]);
        assert_eq!(payload.outputs, vec![out]);
    }
}

#[test]
fn lookup_payload_is_emitted_for_compare_unary_and_rotate_family() {
    let unary_cases = [
        (WasmOpcode::I32Clz, WasmShoutOpcode::I32Clz, 16, 27),
        (WasmOpcode::I32Ctz, WasmShoutOpcode::I32Ctz, 24, 3),
    ];
    for (opcode, shout_opcode, input, output) in unary_cases {
        let trace = step(opcode, input, None, output);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Unary);
        assert_eq!(payload.shout_id, shout_opcode.to_shout_id());
        assert_eq!(payload.inputs, vec![input]);
        assert_eq!(payload.outputs, vec![output]);
    }

    let binary_cases = [
        (WasmOpcode::I32GtS, WasmShoutOpcode::I32GtS, 9, 3, 1),
        (WasmOpcode::I32GtU, WasmShoutOpcode::I32GtU, 9, 3, 1),
        (WasmOpcode::I32LeS, WasmShoutOpcode::I32LeS, 3, 9, 1),
        (WasmOpcode::I32LeU, WasmShoutOpcode::I32LeU, 3, 9, 1),
        (WasmOpcode::I32GeS, WasmShoutOpcode::I32GeS, 9, 3, 1),
        (WasmOpcode::I32GeU, WasmShoutOpcode::I32GeU, 9, 3, 1),
        (
            WasmOpcode::I32Rotl,
            WasmShoutOpcode::I32Rotl,
            0x1234_5678,
            8,
            0x3456_7812,
        ),
        (
            WasmOpcode::I32Rotr,
            WasmShoutOpcode::I32Rotr,
            0x1234_5678,
            8,
            0x7812_3456,
        ),
    ];
    for (opcode, shout_opcode, lhs, rhs, output) in binary_cases {
        let trace = step(opcode, lhs, Some(rhs), output);
        let payload = lookup_payload(&trace).expect("payload");
        assert_eq!(payload.arity, neo_wasm::WasmLookupArity::Binary);
        assert_eq!(payload.shout_id, shout_opcode.to_shout_id());
        assert_eq!(payload.inputs, vec![lhs, rhs]);
        assert_eq!(payload.outputs, vec![output]);
    }
}

#[test]
fn trace_builder_attaches_lookup_payload_to_extension_data() {
    let vm = WasmVmSpec::new().expect("vm");

    let builder = WasmTraceBuilder::new();
    let trace = step(WasmOpcode::I32Mul, 7, Some(9), 63);

    let built = builder.build_steps(&vm, &[trace]).expect("build");
    let payload = built[0]
        .extension_data
        .shout_lookup
        .clone()
        .expect("lookup payload");

    assert_eq!(payload.shout_id, WasmShoutOpcode::I32Mul.to_shout_id());
    assert_eq!(payload.inputs, vec![7, 9]);
    assert_eq!(payload.outputs, vec![63]);
}

#[test]
fn i64_binary_lookup_payload_uses_four_inputs_and_two_outputs() {
    let code = opcode_code(WasmOpcode::I64Mul);
    let info = opcode_info_from_code(code);
    let trace = WasmStepTrace {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        pc_before: 0,
        pc_after: 1,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        param_init_before: WasmParamInitState::ZERO,
        param_init_after: WasmParamInitState::ZERO,
        wide_values_enabled: true,
        opcode_code: code,
        opcode: WasmOpcode::I64Mul,
        info,
        stack_reads_override: None,
        stack_writes_override: None,
        sp_before: 2,
        sp_after: 1,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: Some(StackLaneAccess { addr: 0, value: 3 }),
        stack_read0_hi: Some(1),
        stack_read1: Some(StackLaneAccess { addr: 1, value: 5 }),
        stack_read1_hi: Some(2),
        stack_read2: None,
        stack_read2_hi: None,
        stack_write0: Some(StackLaneAccess { addr: 0, value: 15 }),
        stack_write0_hi: Some(7),
        linear_memory: None,
        linear_memory_offset: 0,
        memory_pages_before: None,
        memory_pages_after: None,
        halted: false,
        locals_fbp: 0,
        locals_fbp_after: 0,
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
    assert_eq!(payload.shout_id, WasmShoutOpcode::I64Mul.to_shout_id());
    assert_eq!(payload.inputs, vec![3, 1, 5, 2]);
    assert_eq!(payload.outputs, vec![15, 7]);
}
