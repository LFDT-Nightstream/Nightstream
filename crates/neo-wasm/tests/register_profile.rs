//! Milestone 4 of the register-row study (scratch, not for review): general
//! rows per original WASM instruction on every export of every program,
//! with the rows split by the register-form row that made them. Every run
//! must agree with wasmtime. Run with `--nocapture`.

mod register;

use std::collections::BTreeMap;

use register::alloc::allocate;
use register::corpus::{programs, reference};
use register::cpu::Cpu;
use register::lower::{lower, origins, ALLOC_K};
use register::machine::Machine;
use register::translate::parse;

struct Export {
    program: String,
    name: String,
    instr: u64,
    rows: u64,
    origins: BTreeMap<String, u64>,
}

fn masked(results: &[wasmparser::ValType], v: &[u64]) -> Vec<u64> {
    v.iter()
        .zip(results)
        .map(|(x, t)| {
            if *t == wasmparser::ValType::I32 {
                x & 0xffff_ffff
            } else {
                *x
            }
        })
        .collect()
}

/// Every export with at most 4 integer parameters that runs to a result
/// under wasmtime, the cell machine, and the general-row `Cpu`.
fn exports() -> Vec<Export> {
    let mut out = Vec::new();
    for (program, bytes) in programs() {
        let Ok(module) = parse(&bytes) else { continue };
        let Ok(allocated) = allocate(&module.program, ALLOC_K) else {
            continue;
        };
        let mut names: Vec<_> = module
            .exports
            .iter()
            .map(|(n, f)| (n.clone(), *f))
            .collect();
        names.sort();
        if let Some((_, f)) = names.iter().find(|(_, f)| *f as usize >= module.imports) {
            let low = lower(&allocated, *f as usize);
            println!(
                "SIZE|{program}|program rows={}|table words={}|globals={}|data bytes={}",
                low.rows.len(),
                low.table.len(),
                module.program.globals.len(),
                module
                    .program
                    .data
                    .iter()
                    .map(|(o, b)| *o as usize + b.len())
                    .max()
                    .unwrap_or(0)
            );
        }
        for (name, f) in names {
            let (params, results) = &module.types[module.func_types[f as usize] as usize];
            if (f as usize) < module.imports
                || params.len() > 4
                || !params
                    .iter()
                    .all(|t| matches!(t, wasmparser::ValType::I32 | wasmparser::ValType::I64))
            {
                continue;
            }
            let args: Vec<u64> = (0..params.len()).map(|i| [10, 7, 3, 2][i]).collect();
            let Some(Ok(expected)) = reference(&bytes, &name, &args) else {
                continue;
            };
            let mut cell = Machine::new(&module.program, None);
            let Ok(got) = cell.run(f as usize, &args, 50_000_000) else {
                continue;
            };
            assert_eq!(
                masked(results, &expected),
                masked(results, &got),
                "{program}::{name} cell form"
            );
            let low = lower(&allocated, f as usize);
            let words = (module.program.memory_pages as usize * 16384).max(1);
            let mut cpu = Cpu::new(&low, words, 1 << 20);
            let regs = cpu
                .run(&args, 400_000_000)
                .unwrap_or_else(|e| panic!("{program}::{name}: {e:?}"));
            assert_eq!(
                masked(results, &expected),
                masked(results, &regs[..results.len()]),
                "{program}::{name} general rows"
            );
            let extra: u64 = cpu.estimated.values().map(|e| e.1 - e.0).sum();
            let names = origins().lock().unwrap();
            let origins = cpu
                .origins
                .iter()
                .map(|(k, v)| (names[*k as usize].clone(), v.1))
                .collect();
            out.push(Export {
                program: program.clone(),
                name,
                instr: cell.marks,
                rows: cpu.rows + extra,
                origins,
            });
        }
    }
    out
}

fn report(label: &str, runs: &[&Export]) {
    let instr: u64 = runs.iter().map(|r| r.instr).sum();
    let rows: u64 = runs.iter().map(|r| r.rows).sum();
    let mean = runs
        .iter()
        .map(|r| r.rows as f64 / r.instr as f64)
        .sum::<f64>()
        / runs.len() as f64;
    println!(
        "TOTAL|{label}|exports={}|instr={instr}|rows={rows}|weighted={:.3}|mean={mean:.3}",
        runs.len(),
        rows as f64 / instr as f64
    );
    let mut origin: BTreeMap<&str, u64> = BTreeMap::new();
    for r in runs {
        for (k, v) in &r.origins {
            *origin.entry(k).or_default() += v;
        }
    }
    let mut origin: Vec<_> = origin.into_iter().collect();
    origin.sort_by_key(|o| std::cmp::Reverse(o.1));
    for (k, v) in origin.iter().take(16) {
        println!("ORIGIN|{label}|{k}|rows={v}|per instr={:.3}", *v as f64 / instr as f64);
    }
}

#[test]
fn general_rows_per_instruction_on_every_export() {
    let all = exports();
    for r in &all {
        println!(
            "EXPORT|{}::{}|instr={}|rows={}|ratio={:.3}",
            r.program,
            r.name,
            r.instr,
            r.rows,
            r.rows as f64 / r.instr as f64
        );
    }
    let starstream: Vec<&Export> = all
        .iter()
        .filter(|r| r.program.starts_with("starstream"))
        .collect();
    let compute: Vec<&Export> = all
        .iter()
        .filter(|r| !r.program.starts_with("starstream") && r.instr >= 100)
        .collect();
    report("starstream", &starstream);
    report("compute", &compute);
}
