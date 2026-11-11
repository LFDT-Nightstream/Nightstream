//! The programs of the register-row study and a wasmtime reference run.
//! `corpus/` holds compiled Rust programs from the zkEngine repository
//! (`zk_*`), compiled Starstream contracts from the Starstream compiler's
//! checked-in snapshots (`starstream_*`), and the Fibonacci program of
//! `tests/wasm_fibonacci.rs`. Imports are stubs that return zero.

use std::path::{Path, PathBuf};

use wasmtime::{Config, Engine, ExternType, Linker, Module, Store, Val, ValType};

pub fn fib_wat(n: u32) -> String {
    format!(
        r#"(module (func (export "main") (result i32) (local i32 i32 i32)
    i32.const {n} local.set 0 i32.const 1 local.set 1 i32.const 0 local.set 2
    (loop $L local.get 1 local.get 2 i32.add local.get 2 local.set 1 local.set 2
      local.get 0 i32.const 1 i32.sub local.tee 0 br_if $L)
    local.get 2))"#
    )
}

pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/register/corpus")
}

/// Every program of the corpus, by name, as WASM bytes.
pub fn programs() -> Vec<(String, Vec<u8>)> {
    let mut out = vec![("neo_fibonacci".to_string(), wat::parse_str(fib_wat(127)).unwrap())];
    let mut paths: Vec<_> = std::fs::read_dir(dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).unwrap();
        let bytes = if path.extension().is_some_and(|e| e == "wat") {
            match wat::parse_bytes(&bytes) {
                Ok(b) => b.into_owned(),
                Err(e) => {
                    eprintln!("skip {name}: {e}");
                    continue;
                }
            }
        } else {
            bytes
        };
        out.push((name, bytes));
    }
    out
}

/// The reference result of `export(args)` under wasmtime, with every import a
/// stub that returns zero. `None` on a trap, a missing export, or no fuel.
pub fn reference(bytes: &[u8], export: &str, args: &[u64]) -> Option<Result<Vec<u64>, String>> {
    let mut config = Config::new();
    config.consume_fuel(true);
    let engine = Engine::new(&config).ok()?;
    let module = match Module::new(&engine, bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("wasmtime: {e}");
            return None;
        }
    };
    let mut linker: Linker<()> = Linker::new(&engine);
    for import in module.imports() {
        if let ExternType::Func(ty) = import.ty() {
            let results: Vec<ValType> = ty.results().collect();
            linker
                .func_new(import.module(), import.name(), ty.clone(), move |_, _, out| {
                    for (slot, t) in out.iter_mut().zip(&results) {
                        *slot = match t {
                            ValType::I64 => Val::I64(0),
                            _ => Val::I32(0),
                        };
                    }
                    Ok(())
                })
                .ok()?;
        }
    }
    let mut store = Store::new(&engine, ());
    store.set_fuel(200_000_000).ok()?;
    let instance = match linker.instantiate(&mut store, &module) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("wasmtime instantiate: {e}");
            return None;
        }
    };
    let func = instance.get_func(&mut store, export)?;
    let ty = func.ty(&store);
    let params: Vec<Val> = ty
        .params()
        .zip(args)
        .map(|(t, &a)| match t {
            ValType::I64 => Val::I64(a as i64),
            _ => Val::I32(a as i32),
        })
        .collect();
    let mut results: Vec<Val> = ty.results().map(|_| Val::I32(0)).collect();
    Some(match func.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(results
            .iter()
            .map(|v| match v {
                Val::I32(x) => *x as u32 as u64,
                Val::I64(x) => *x as u64,
                _ => 0,
            })
            .collect()),
        Err(e) => Err(format!("{e}")),
    })
}
