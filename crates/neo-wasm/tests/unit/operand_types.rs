use super::*;

#[test]
fn a_sum_stores_at_most_one_operand_lane_per_instruction() {
    let terms = 512;
    let instructions = format!("{}{}", "i32.const 1 ".repeat(terms), "i32.add ".repeat(terms - 1));
    let wasm = wat::parse_str(format!("(module (func (result i32) {instructions}))")).unwrap();
    let stacks = operator_stacks_by_offset(&wasm).unwrap();
    let stored: usize = stacks
        .values()
        .map(|stack| stack.frame_operands.len())
        .sum();
    assert!(
        stored <= stacks.len(),
        "{stored} lanes for {} instructions",
        stacks.len()
    );
}
