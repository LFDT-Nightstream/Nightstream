//! Program ROMs for Rocq `formal/wasm-rocq/tests/ProgramRom.v`.

use neo_wasm::{build_wasm_relation_layout, extract_wasm_program_artifacts, preload_from_program_artifacts};
use std::fmt::Write;
use wasmparser::{BlockType, Operator, Parser, Payload, ValType};

/// Modules inside Rocq `Fragment.v`.
const CORPUS: &[(&str, &str)] = &[
    (
        "blocks",
        r#"(module
  (func (export "main") (param i32 i32) (result i32) (local i32)
    block (result i32)
      local.get 0
      block
        local.get 1
        br_if 0
        local.get 0
        local.get 1
        i32.sub
        local.set 2
      end
      local.get 2
      i32.add
    end
    loop
      local.get 2
      local.get 0
      i32.sub
      local.set 2
      local.get 2
      br_if 0
    end
    block
      br 0
      br 0
    end))"#,
    ),
    (
        "ifs",
        r#"(module
  (func (export "main") (param i32) (result i32) (local i32)
    local.get 0
    if (result i32)
      local.get 0
      local.get 0
      i32.add
    else
      local.get 0
      if
        local.get 1
        br 2
      end
      local.get 0
      if
        local.get 0
        local.set 1
      end
      local.get 1
    end))"#,
    ),
    (
        "brTables",
        r#"(module
  (func (export "main") (param i32) (result i32) (local i32)
    block
      block
        block
          local.get 0
          br_table 0 1 2 1
        end
        local.get 0
        local.set 1
      end
      block (result i32)
        local.get 1
        local.get 0
        br_table 0
      end
      local.set 1
    end
    loop (result i32)
      local.get 1
    end))"#,
    ),
    (
        "calls",
        r#"(module
  (func $pick (param i64 i32) (result i64) (local i64)
    local.get 1
    if
      local.get 0
      return
    end
    local.get 2)
  (func $sum (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)
  (func $trap
    unreachable)
  (func $empty)
  (func (export "main") (param i32 i64) (result i32) (local i64 i32)
    local.get 1
    local.get 0
    call $pick
    local.set 2
    call $empty
    local.get 0
    local.get 0
    call $sum
    nop))"#,
    ),
    (
        "indirect",
        r#"(module
  (type $narrow (func (param i32) (result i32)))
  (type $wide (func (param i64) (result i64)))
  (type $narrowAgain (func (param i32) (result i32)))
  (type $main (func (param i32 i32) (result i32)))
  (table 3 funcref)
  (elem (i32.const 0) $double $same $negate)
  (func $double (type $narrow)
    local.get 0
    local.get 0
    i32.add)
  (func $same (type $wide)
    local.get 0)
  (func $negate (type $narrowAgain)
    local.get 0
    local.get 0
    i32.sub
    local.get 0
    i32.sub)
  (func (export "main") (type $main) (local i64)
    local.get 2
    local.get 1
    call_indirect (type $wide)
    local.set 2
    local.get 0
    local.get 1
    call_indirect (type $narrowAgain)))"#,
    ),
];

fn value_type(ty: ValType) -> &'static str {
    match ty {
        ValType::I32 => "valtype_I32",
        ValType::I64 => "valtype_I64",
        other => panic!("value type {other:?} is outside the fragment"),
    }
}

fn list(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join("; "))
}

fn index(value: u32) -> String {
    format!("(mk_uN {value})")
}

fn block_type(ty: BlockType) -> String {
    match ty {
        BlockType::Empty => "(_RESULT None)".into(),
        BlockType::Type(ty) => format!("(_RESULT (Some {}))", value_type(ty)),
        BlockType::FuncType(_) => panic!("a block type by index is outside the fragment"),
    }
}

struct Open {
    head: &'static str,
    ty: String,
    body: Vec<String>,
    then_body: Option<Vec<String>>,
}

fn structured_body(operators: wasmparser::OperatorsReader<'_>) -> Vec<String> {
    let mut open: Vec<Open> = Vec::new();
    let mut body = Vec::new();
    for operator in operators {
        let operator = operator.expect("operator");
        let instruction = match operator {
            Operator::LocalGet { local_index } => format!("LOCAL_GET {}", index(local_index)),
            Operator::LocalSet { local_index } => format!("LOCAL_SET {}", index(local_index)),
            Operator::I32Add => "BINOP I32 (mk_binop__0 Inn_I32 ADD)".into(),
            Operator::I32Sub => "BINOP I32 (mk_binop__0 Inn_I32 SUB)".into(),
            Operator::Nop => "NOP".into(),
            Operator::Unreachable => "UNREACHABLE".into(),
            Operator::Return => "RETURN".into(),
            Operator::Br { relative_depth } => format!("BR {}", index(relative_depth)),
            Operator::BrIf { relative_depth } => format!("BR_IF {}", index(relative_depth)),
            Operator::BrTable { targets } => {
                let arms = targets
                    .targets()
                    .map(|target| index(target.expect("target")));
                format!("BR_TABLE {} {}", list(arms), index(targets.default()))
            }
            Operator::Call { function_index } => format!("CALL {}", index(function_index)),
            Operator::CallIndirect {
                type_index,
                table_index,
            } => {
                format!("CALL_INDIRECT {} {}", index(table_index), index(type_index))
            }
            Operator::Block { blockty } | Operator::Loop { blockty } | Operator::If { blockty } => {
                let head = match operator {
                    Operator::Block { .. } => "BLOCK",
                    Operator::Loop { .. } => "LOOP",
                    _ => "IFELSE",
                };
                open.push(Open {
                    head,
                    ty: block_type(blockty),
                    body: Vec::new(),
                    then_body: None,
                });
                continue;
            }
            Operator::Else => {
                let frame = open.last_mut().expect("else inside an if");
                frame.then_body = Some(std::mem::take(&mut frame.body));
                continue;
            }
            Operator::End => {
                let Some(frame) = open.pop() else { continue };
                match frame.then_body {
                    Some(then_body) => format!("IFELSE {} {} {}", frame.ty, list(then_body), list(frame.body)),
                    None if frame.head == "IFELSE" => format!("IFELSE {} {} []", frame.ty, list(frame.body)),
                    None => format!("{} {} {}", frame.head, frame.ty, list(frame.body)),
                }
            }
            other => panic!("operator {other:?} is outside the fragment"),
        };
        match open.last_mut() {
            Some(frame) => frame.body.push(instruction),
            None => body.push(instruction),
        }
    }
    assert!(open.is_empty(), "every block ends");
    body
}

fn rocq_program(name: &str, wasm: &[u8]) -> String {
    let mut types = Vec::new();
    let mut function_types = Vec::new();
    let mut tables = 0;
    let mut functions = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.expect("payload") {
            Payload::TypeSection(reader) => {
                for ty in reader.into_iter_err_on_gc_types() {
                    let ty = ty.expect("function type");
                    let params = ty
                        .params()
                        .iter()
                        .map(|t| value_type(*t).to_string())
                        .collect::<Vec<_>>();
                    let results = ty
                        .results()
                        .iter()
                        .map(|t| value_type(*t).to_string())
                        .collect::<Vec<_>>();
                    types.push((params, results));
                }
            }
            Payload::ImportSection(reader) => assert_eq!(reader.count(), 0, "imports are outside the fragment"),
            Payload::FunctionSection(reader) => {
                function_types = reader
                    .into_iter()
                    .map(|ty| ty.expect("type index"))
                    .collect();
            }
            Payload::TableSection(reader) => tables = reader.count(),
            Payload::CodeSectionEntry(body) => {
                let mut locals = Vec::new();
                for group in body.get_locals_reader().expect("locals") {
                    let (count, ty) = group.expect("local group");
                    locals.extend((0..count).map(|_| value_type(ty).to_string()));
                }
                functions.push((locals, structured_body(body.get_operators_reader().expect("operators"))));
            }
            _ => {}
        }
    }
    assert_eq!(function_types.len(), functions.len());
    let module = format!("{name}Module");
    let mut out = String::new();
    writeln!(
        out,
        "Definition {module} := MKmoduleinst\n  {}\n  {} [] {} [] [] [] [].",
        list(types.iter().map(|(params, results)| {
            format!(
                "mk_functype (mk_list _ {}) (mk_list _ {})",
                list(params.clone()),
                list(results.clone())
            )
        })),
        list((0..functions.len()).map(|f| format!("{f}%N"))),
        list((0..tables).map(|t| format!("{t}%N"))),
    )
    .unwrap();
    let functions = functions
        .into_iter()
        .zip(&function_types)
        .map(|((locals, body), &ty)| {
            let (params, results) = &types[ty as usize];
            format!(
                "typedFunction {module} {} {} {} {}\n    {}",
                index(ty),
                list(params.clone()),
                list(results.clone()),
                list(locals),
                list(body)
            )
        });
    writeln!(out, "Definition {name}Program : Program :=\n  {}.", list(functions)).unwrap();
    out
}

fn rocq_cells(name: &str, wasm: &[u8]) -> String {
    let memories = &build_wasm_relation_layout().auxiliary.logical_memory;
    let artifacts = extract_wasm_program_artifacts(wasm).expect("artifacts");
    let cells = preload_from_program_artifacts(&artifacts)
        .entries()
        .into_iter()
        .map(|(id, key, value)| {
            let id = memories
                .iter()
                .position(|memory| memory.id == id)
                .expect("cataloged memory");
            format!("({id}%nat, {}, {value})", list(key.iter().map(u32::to_string)))
        });
    format!(
        "Definition {name}Cells : list (nat * list Z * Z) :=\n  {}.\n",
        list(cells)
    )
}

#[test]
#[ignore = "regenerates Rocq files"]
fn export_program_roms() {
    let mut out = String::from(
        "(* Generated by crates/neo-wasm/tests/wasm_rocq_program_rom_export.rs. *)\n\
         From Stdlib Require Import List ZArith.\n\
         Require Import Wasm2 SpecTecCalls ProgramSpec.\n\
         Import ListNotations.\n\
         Open Scope Z_scope.\n\n",
    );
    for (name, wat) in CORPUS {
        let wasm = wat::parse_str(wat).expect("wat");
        out.push_str(&rocq_program(name, &wasm));
        out.push_str(&rocq_cells(name, &wasm));
        out.push('\n');
    }
    writeln!(
        out,
        "Definition corpus : list (Program * list (nat * list Z * Z)) :=\n  {}.",
        list(
            CORPUS
                .iter()
                .map(|(name, _)| format!("({name}Program, {name}Cells)"))
        )
    )
    .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../formal/wasm-rocq/tests/ProgramRomTrace.v");
    // Keep the mtime of an unchanged file, so `make` does not recompile it.
    if std::fs::read(&path).map_or(true, |old| old != out.as_bytes()) {
        std::fs::write(&path, out).expect("write");
    }
}
