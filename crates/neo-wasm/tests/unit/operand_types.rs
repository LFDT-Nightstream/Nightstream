use super::*;

#[test]
fn a_sum_stores_at_most_one_operand_lane_per_instruction() {
    let terms = 512;
    let instructions = format!("{}{}", "i32.const 1 ".repeat(terms), "i32.add ".repeat(terms - 1));
    let wasm = wat::parse_str(format!("(module (func (result i32) {instructions}))")).unwrap();
    let artifacts = crate::extract_wasm_program_artifacts(&wasm).unwrap();
    let stored = artifacts.trace.operand_types.nodes.len();
    let operators = artifacts.trace.opcode_map.len();
    assert!(stored <= operators, "{stored} lanes for {operators} instructions");
}

#[test]
fn snapshots_match_validation_across_control_frames() {
    let wasm = wat::parse_str(
        "(module
            (func $pair (param i32 i64) (result i32 i64) local.get 0 local.get 1)
            (func (param i32) (result i64)
                ref.null func i64.const 42
                i32.const 7 i64.const 8
                local.get 0
                if (param i32 i64) (result i32 i64)
                    call $pair
                else
                    block (param i32 i64) (result i32 i64)
                        local.get 0 br_if 0
                        unreachable select drop
                    end
                end
                drop drop return))",
    )
    .unwrap();
    let (stacks, types) = operator_stacks_by_offset(&wasm).unwrap();
    let mut validator = Validator::new();
    for payload in Parser::new(0).parse_all(&wasm) {
        if let ValidPayload::Func(function, body) = validator.payload(&payload.unwrap()).unwrap() {
            let mut function = function.into_validator(Default::default());
            function.read_locals(&mut body.get_binary_reader()).unwrap();
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                let offset = operators.original_position();
                let stack = &stacks[&(offset as u32)];
                let height = function.operand_stack_height();
                assert_eq!(stack.height, height);
                let count = height as usize - function.get_control_frame(0).unwrap().height;
                let expected: Vec<_> = (0..count)
                    .rev()
                    .map(|depth| {
                        function
                            .get_operand_type(depth)
                            .unwrap()
                            .map(OperandLane::of_type)
                    })
                    .collect();
                assert_eq!(types.lanes(stack.frame_operands), expected, "at byte {offset}");
                function.op(offset, &operators.read().unwrap()).unwrap();
            }
        }
    }
}
