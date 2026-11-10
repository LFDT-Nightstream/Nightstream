//! Export real traces' rows to Rocq, so the refinement theorem's premises can be
//! checked against actual runs (`formal/wasm-rocq/tests/*Witness.v`).
//! Regenerate with `cargo test -p neo-wasm --release --test wasm_rocq_witness_export -- --ignored`.

use neo_wasm::host_event_bindings::{EventSequenceBuilder, EventSources, HostEventBindings, HostEventBindingsBuilder};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    collect_wasmtime_steps, extract_wasm_program_artifacts, traces_from_wasmtime_steps,
    traces_from_wasmtime_steps_with_host_events, CommChainState, WasmAuxOpcode, WasmRowKind,
};
use p3_field::PrimeField64;
use std::fmt::Write;

/// `tests/BlockCalls.v`'s program: a call from inside two blocks.
const BLOCK_CALLS: &str = r#"(module
  (func $readZero (result i32) (local i32) local.get 0)
  (func (export "main") (result i32) (local i32)
    block block call $readZero local.set 0 end end local.get 0))"#;

/// A taken br_if that leaves two blocks, dropping the operand under its condition
/// and the outer block's operand below the inner block.
const UNWINDING_BRANCH: &str = r#"(module
  (func (export "main") (param i32) (result i32) (local i32)
    block
      local.get 0
      block local.get 1 local.get 0 br_if 1 local.set 1 end
      local.set 1
    end
    local.get 1))"#;

/// The export (function 0, the only one with parameters here) takes each parameter
/// from a turn input; its exit outputs the result.
fn entry_bindings(wasm: &[u8], params: usize) -> HostEventBindings {
    if params == 0 {
        return HostEventBindings::default();
    }
    let artifacts = extract_wasm_program_artifacts(wasm).expect("artifacts");
    let mut entry = EventSequenceBuilder::op(1);
    for local in 0..params as u8 {
        entry = entry
            .opaque([1, 2, 3, 4], EventSources::new().input_local_i32(local, local))
            .expect("entry input");
    }
    let exit = EventSequenceBuilder::absorbing()
        .output_i32()
        .expect("exit")
        .finish()
        .expect("exit");
    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    builder
        .export(1, entry.finish().expect("entry"), exit)
        .expect("export");
    builder.finish().expect("bindings")
}

/// A return from inside a block that drops the operand below its result.
const EARLY_RETURN: &str = r#"(module
  (func (export "main") (param i32) (result i32) local.get 0 call 1)
  (func (param i32) (result i32) (local i32)
    block local.get 1 local.get 0 return end
    local.get 1))"#;

/// A taken br_if out of a result block: it keeps the block's result and drops the
/// operand below it, so the result moves down; the outer block's END carries its
/// result out.
const RESULT_BLOCKS: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32)
    block (result i32)
      block (result i32) local.get 0 local.get 1 local.get 0 br_if 0 i32.add end
      local.get 1
      i32.sub
    end))"#;

/// A br out of two blocks to a result block: it keeps the block's result and drops
/// the outer block's operand below it, so the result moves down.
const BR_OUT: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32)
    block (result i32)
      local.get 0
      block local.get 1 br 1 end
    end))"#;

/// Two br_tables: index 1 takes an arm out of a zero-result block; index 7 is past
/// the arms and takes the default out of a result block, which keeps its result and
/// drops the operand below it, so the result moves down.
const BR_TABLES: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32)
    block (result i32)
      local.get 0
      block local.get 1 br_table 0 0 0 end
      local.get 1
      local.get 0
      br_table 0 0
    end))"#;

/// Three ifs: a zero condition takes the else-body, a nonzero one the then-body,
/// whose ELSE jumps past the else-body; a zero condition skips an if with no
/// else-body.
const IFS: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    if (result i32) local.get 1 else local.get 0 end
    i32.add
    local.get 0
    if (result i32) local.get 0 else local.get 1 end
    i32.add
    local.get 1
    if local.get 0 local.set 1 end))"#;

/// A loop summing n, n - 1, ..., 1 for main(n, 1): each taken br_if drops the
/// running sum it pushed and runs the loop again; the last iteration falls out of
/// the loop with the sum as its result.
const LOOPS: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32) (local i32)
    loop (result i32)
      local.get 2
      local.get 0
      i32.add
      local.set 2
      local.get 0
      local.get 1
      i32.sub
      local.set 0
      local.get 2
      local.get 0
      br_if 0
    end))"#;

/// Nops at the top level and inside a block, around an operand.
const NOPS: &str = r#"(module
  (func (export "main") (param i32) (result i32)
    nop
    local.get 0
    block nop end
    nop))"#;

/// A br to the callee's own label from inside a block: a return of 9 to main,
/// dropping the 7 below it.
const FUNCTION_BR: &str = r#"(module
  (func (export "main") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    call 1)
  (func (param i32 i32) (result i32)
    block (result i32)
      local.get 0
      local.get 1
      br 1
    end))"#;

/// An indirect call through a table holding $double, $main (another type) and
/// null. The index comes from a parameter: the proved fragment has no constants.
const INDIRECT_CALL: &str = r#"(module
  (type $unary (func (param i32) (result i32)))
  (table 3 funcref)
  (elem (i32.const 0) $double $main)
  (func $main (export "main") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    call_indirect (type $unary))
  (func $double (type $unary)
    local.get 0
    local.get 0
    i32.add))"#;

/// The final run of program, zero and move rows, optionally including its terminal row.
fn export_witness(wat: &str, params: &[i32], name: &str, file: &str, include_terminal: bool) {
    export_witness_format(wat, params, name, file, include_terminal, false);
}

fn export_witness_format(wat: &str, params: &[i32], name: &str, file: &str, include_terminal: bool, sparse: bool) {
    let wasm = wat::parse_str(wat).expect("wat");
    let run = collect_wasmtime_steps(&wasm, &entry_bindings(&wasm, params.len()), "main", params).expect("trace");
    let trace = if params.is_empty() {
        traces_from_wasmtime_steps(&run.steps)
    } else {
        traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), CommChainState::default())
    }
    .expect("normalize");
    let end = trace
        .iter()
        .position(|row| row.state_after.activation_finished)
        .expect("the final return");
    let in_run = |row: &neo_wasm::WasmVmStep| {
        row.row_kind.is_program()
            || matches!(
                row.row_kind,
                WasmRowKind::Aux(WasmAuxOpcode::LocalZero | WasmAuxOpcode::FrameMove)
            )
    };
    let start = trace[..end]
        .iter()
        .rposition(|row| !in_run(row))
        .map_or(0, |index| index + 1);
    let mut out = String::from(
        "(* AUTO-GENERATED by tests/wasm_rocq_witness_export.rs. DO NOT EDIT. *)\n\
         From Stdlib Require Import List ZArith.\nImport ListNotations.\nOpen Scope Z_scope.\n\n",
    );
    if sparse {
        out.push_str("Require Import Columns tests.SparseWitness.\n\n");
        write!(out, "Definition {name}Sparse : list (list (nat * Z)) :=\n  [").unwrap();
    } else {
        write!(
            out,
            "(* Each row's witness, column by column, in trace order. *)\n\
            Definition {name} : list (list Z) :=\n  ["
        )
        .unwrap();
    }
    for (index, row) in trace[start..end + usize::from(include_terminal)]
        .iter()
        .enumerate()
    {
        if index > 0 {
            out.push_str(";\n   ");
        }
        let values = build_witness_vector(row).unwrap();
        if sparse {
            let entries = values
                .iter()
                .enumerate()
                .filter_map(|(column, value)| {
                    let value = value.as_canonical_u64();
                    if value == 0 {
                        return None;
                    }
                    let name = match neo_wasm::named_column_family(column) {
                        Some(family) if family.len == 1 => family.name.to_string(),
                        Some(family) => format!("(nth {}%nat {} 0%nat)", column - family.start, family.name),
                        None => format!("{column}%nat"),
                    };
                    Some(format!("({name}, {value})"))
                })
                .collect::<Vec<_>>()
                .join(";\n    ");
            write!(out, "[{entries}]").unwrap();
        } else {
            let entries = values
                .iter()
                .map(|value| value.as_canonical_u64().to_string())
                .collect::<Vec<_>>()
                .join("; ");
            write!(out, "[{entries}]").unwrap();
        }
    }
    out.push_str("].\n");
    if sparse {
        write!(
            out,
            "\n(* Materialize dense lists once, not on every witness lookup. *)\n\
            Definition {name} : list (list Z) :=\n  Eval vm_compute in expandRows columnCount {name}Sparse.\n\n\
            Example {name}_expansion :\n  map (expandWitness columnCount) {name}Sparse = map (@Some (list Z)) {name}.\n\
            Proof. vm_compute. reflexivity. Qed.\n"
        )
        .unwrap();
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/wasm-rocq/tests")
        .join(file);
    std::fs::write(&path, out).expect("write");
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_witnesses() {
    export_witness(BLOCK_CALLS, &[], "blockCallRows", "BlockCallTrace.v", false);
    export_witness(UNWINDING_BRANCH, &[7], "unwindingRows", "UnwindingTrace.v", false);
    export_witness(EARLY_RETURN, &[7], "earlyReturnRows", "EarlyReturnTrace.v", false);
    export_witness(RESULT_BLOCKS, &[7, 9], "resultBlockRows", "ResultBlockTrace.v", false);
    export_witness(BR_OUT, &[7, 9], "brRows", "BrTrace.v", false);
    export_witness(BR_TABLES, &[7, 1], "brTableRows", "BrTableTrace.v", false);
    export_witness(IFS, &[7, 0], "ifRows", "IfTrace.v", false);
    export_witness(LOOPS, &[3, 1], "loopRows", "LoopTrace.v", false);
    export_witness(NOPS, &[7], "nopRows", "NopTrace.v", false);
    export_witness(FUNCTION_BR, &[7, 9], "functionBrRows", "FunctionBrTrace.v", false);
    export_witness(INDIRECT_CALL, &[7, 0], "indirectRows", "IndirectTrace.v", false);
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_integer_local_witnesses() {
    // Both limbs are nonzero, and the unsigned value exceeds Goldilocks.
    export_witness_format(
        r#"(module (func (export "main") (local i64)
            i64.const -1 local.set 0 local.get 0 drop))"#,
        &[],
        "integerLocalRows",
        "IntegerLocalTrace.v",
        false,
        true,
    );
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_integer_stack_witnesses() {
    // The i64 prefix must survive arithmetic and local transfers, including its high limb.
    export_witness_format(
        r#"(module (func (export "main") (result i64) (local i64)
            i64.const 4294967305
            i32.const 7 i32.const 9 i32.add drop
            i64.const -1 local.set 0 local.get 0 drop))"#,
        &[],
        "integerStackRows",
        "IntegerStackTrace.v",
        false,
        true,
    );
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_integer_call_witnesses() {
    export_witness_format(
        r#"(module
          (func (export "main") (result i64) (local i32 i64)
            i64.const -1 call 1)
          (func (param i64) (result i64) (local i32 i64)
            local.get 0))"#,
        &[],
        "integerCallRows",
        "IntegerCallTrace.v",
        false,
        true,
    );
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_terminal_witnesses() {
    export_witness(NOPS, &[7], "nopTerminalRows", "NopTerminalTrace.v", true);
    export_witness(INDIRECT_CALL, &[7, 5], "indirectTrapRows", "IndirectTrapTrace.v", true);
    export_witness(
        INDIRECT_CALL,
        &[7, 1],
        "indirectMismatchRows",
        "IndirectMismatchTrace.v",
        true,
    );
    export_witness(INDIRECT_CALL, &[7, 2], "indirectNullRows", "IndirectNullTrace.v", true);
    export_witness(
        r#"(module
          (func (export "main") block call 1 end)
          (func block unreachable end))"#,
        &[],
        "trapRows",
        "TrapTrace.v",
        true,
    );
    export_witness(
        r#"(module (func (export "main") (param i32) (result i32)
          block (result i32) local.get 0 return end))"#,
        &[7],
        "baseReturnRows",
        "BaseReturnTrace.v",
        true,
    );
    export_witness(
        r#"(module (func (export "main") nop))"#,
        &[],
        "voidReturnRows",
        "VoidReturnTrace.v",
        true,
    );
}
