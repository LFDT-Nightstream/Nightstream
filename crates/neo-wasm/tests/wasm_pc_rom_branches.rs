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
