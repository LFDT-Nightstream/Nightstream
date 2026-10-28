mod common;

fn check(source: &str) {
    let wasm = wat::parse_str(source).unwrap();
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).unwrap();
    let mut edges = std::collections::BTreeMap::new();
    for &(pc, choice, target) in &artifacts.tables.pc_rom {
        assert!(
            edges.insert((pc, choice), target).is_none(),
            "duplicate edge for pc {pc}, choice {choice}"
        );
    }
    common::checked_wasm_run(source, "main");
}

#[test]
fn conditional_block_exit_preserves_taken_and_fallthrough_edges() {
    for condition in [0, 1] {
        check(&format!(
            r#"(module
          (func (export "main") (result i32) (local $value i32)
            block $exit
              i32.const {condition}
              br_if $exit
              i32.const 11
              local.set $value
            end
            local.get $value))"#
        ));
    }
}

#[test]
fn table_block_exits_preserve_each_arm_and_default() {
    for selector in [0, 1, 5] {
        check(&format!(
            r#"(module
          (func (export "main") (result i32)
            block $default
              block $case1
                block $case0
                  i32.const {selector}
                  br_table $case0 $case1 $default
                end
                i32.const 10
                return
              end
              i32.const 20
              return
            end
            i32.const 30))"#
        ));
    }
}

fn reject_wrong_edge(source: &str, opcode: neo_wasm::WasmOpcode, choice: u32) {
    use neo_wasm::{build_wasm_relation_layout, preload_from_program_artifacts, sanity_check_memory_rows};
    let run = common::checked_main(source);
    let index = run
        .trace
        .iter()
        .position(|row| row.opcode == opcode)
        .unwrap();
    let mut forged = run.trace[index].clone();
    forged.control_choice = choice;
    assert_ne!(forged.control_choice, run.trace[index].control_choice);
    forged.state_after.pc = run
        .artifacts
        .tables
        .pc_rom
        .iter()
        .find(|&&(pc, choice, _)| pc == forged.state_before.pc && choice == u64::from(forged.control_choice))
        .expect("both branch edges belong to the program")
        .2;
    let witness = neo_wasm::witness_builder::build_witness_vector(&forged);
    let mut prefix = run.witnesses[..=index].to_vec();
    prefix[index] = witness.clone();
    // ROM membership and memory replay alone do not bind the choice to the condition.
    sanity_check_memory_rows(
        build_wasm_relation_layout(),
        &prefix,
        &preload_from_program_artifacts(&run.artifacts),
    )
    .expect("the alternate edge is in the ROM");
    common::assert_rejected(&witness, "conditional branch must follow its stack condition");
}

#[test]
fn br_if_rejects_the_wrong_valid_edge() {
    for condition in [0, 1, 7, -1] {
        reject_wrong_edge(
            &format!(
                r#"(module (func (export "main") (result i32)
            block i32.const {condition} br_if 0 nop end i32.const 42))"#
            ),
            neo_wasm::WasmOpcode::BrIf,
            u32::from(condition == 0),
        );
    }
}

#[test]
fn if_rejects_the_wrong_valid_edge() {
    for condition in [0, 1, 7, -1] {
        reject_wrong_edge(
            &format!(
                r#"(module (func (export "main") (result i32)
            i32.const {condition} if nop else nop end i32.const 42))"#
            ),
            neo_wasm::WasmOpcode::If,
            u32::from(condition == 0),
        );
    }
}

fn table_program(index: i64) -> String {
    format!(
        r#"(module (func (export "main") (result i32)
        block $default
          block $case1
            block $case0
              i32.const {index} br_table $case0 $case1 $default
            end
            i32.const 10 return
          end
          i32.const 20 return
        end
        i32.const 30))"#
    )
}

#[test]
fn br_table_rejects_wrong_arms_and_default() {
    for index in [0, 1, 2, 5, -1] {
        let expected = if (index as u32) < 2 { index as u32 + 1 } else { 0 };
        for choice in 0..=2 {
            if choice != expected {
                reject_wrong_edge(&table_program(index), neo_wasm::WasmOpcode::BrTable, choice);
            }
        }
    }
}

#[test]
fn br_table_length_is_bound_to_the_program() {
    use neo_wasm::{build_wasm_relation_layout, preload_from_program_artifacts, sanity_check_memory_rows, WasmOpcode};
    let run = common::checked_main(&table_program(0));
    let index = run
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::BrTable)
        .unwrap();
    let mut forged = run.trace[index].clone();
    assert_eq!(forged.br_table_len, 2);
    // Lie about the arm count to make an in-range index take the default.
    forged.br_table_len = 0;
    forged.control_choice = 0;
    forged.state_after.pc = run
        .artifacts
        .tables
        .pc_rom
        .iter()
        .find(|&&(pc, choice, _)| pc == forged.state_before.pc && choice == 0)
        .unwrap()
        .2;
    let witness = neo_wasm::witness_builder::build_witness_vector(&forged);
    common::assert_satisfied(&witness, "choice is consistent with the forged length");
    let mut prefix = run.witnesses[..=index].to_vec();
    prefix[index] = witness;
    let error = sanity_check_memory_rows(
        build_wasm_relation_layout(),
        &prefix,
        &preload_from_program_artifacts(&run.artifacts),
    )
    .expect_err("length must match the program ROM");
    assert!(error.contains("program_br_table_len"), "{error}");
}

#[test]
fn empty_br_table_always_uses_default() {
    for index in [0, 1, -1] {
        check(&format!(
            r#"(module (func (export "main") (result i32)
            block i32.const {index} br_table 0 end i32.const 42))"#
        ));
    }
}

/// Only br_if, if and br_table choose an edge. Elsewhere the control choice is pinned to
/// zero by the row itself, so a dense ROM's padded keys are never read.
#[test]
fn rows_without_a_branch_cannot_choose_an_edge() {
    let run = common::checked_main(
        r#"(module
          (func $one (result i32) i32.const 1)
          (func (export "main") (result i32) (local i32)
            local.get 0 drop
            call $one))"#,
    );
    for opcode in [neo_wasm::WasmOpcode::LocalGet, neo_wasm::WasmOpcode::Call] {
        let row = run
            .trace
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("row");
        let mut forged = row.clone();
        forged.control_choice = 7;
        common::assert_rejected(
            &neo_wasm::witness_builder::build_witness_vector(&forged),
            "a non-branch row choosing an edge",
        );
    }
}
