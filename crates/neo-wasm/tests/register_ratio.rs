//! Milestone 1 of the register-row study (scratch, not for review): the
//! ratio of register rows to original WASM instructions on real programs,
//! statically and on runs that wasmtime checks. Run with `--nocapture`.

mod register;

use std::collections::BTreeMap;

use register::alloc::allocate;
use register::corpus::{programs, reference};
use register::isa::{Item, Program, Tag};
use register::machine::Machine;
use register::translate::{parse, Module};

fn static_counts(program: &Program) -> (usize, usize, BTreeMap<Tag, usize>) {
    let (mut marks, mut rows) = (0, 0);
    let mut tags = BTreeMap::new();
    for f in &program.functions {
        for item in &f.items {
            match item {
                Item::Mark(_) => marks += 1,
                Item::Row(row) => {
                    rows += 1;
                    *tags.entry(row.tag).or_default() += 1;
                }
                Item::Label(_) => {}
            }
        }
    }
    (marks, rows, tags)
}

const REGISTERS: [usize; 4] = [8, 12, 16, 32];

/// Entry points with integer arguments: each export with at most 4 params.
fn entries(module: &Module) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = module
        .exports
        .iter()
        .filter(|(_, &f)| (f as usize) >= module.imports)
        .filter(|(_, &f)| {
            let (p, _) = &module.types[module.func_types[f as usize] as usize];
            p.len() <= 4
                && p.iter()
                    .all(|t| matches!(t, wasmparser::ValType::I32 | wasmparser::ValType::I64))
        })
        .map(|(n, &f)| (n.clone(), f))
        .collect();
    out.sort();
    out
}

struct RunRecord {
    name: String,
    instr: u64,
    cell: u64,
    regs: BTreeMap<usize, (u64, BTreeMap<Tag, u64>)>,
    mix: BTreeMap<&'static str, u64>,
    rows: BTreeMap<String, u64>,
    unaligned: u64,
    depth: usize,
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

fn agree(
    results: &[wasmparser::ValType],
    e: &Result<Vec<u64>, String>,
    g: &Result<Vec<u64>, register::machine::Trap>,
) -> bool {
    match (e, g) {
        (Ok(e), Ok(g)) => masked(results, e) == masked(results, g),
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

#[test]
fn register_rows_per_wasm_instruction_on_the_corpus() {
    let mut best: Vec<RunRecord> = Vec::new();
    let mut disagreements = 0;
    for (name, bytes) in programs() {
        let module = match parse(&bytes) {
            Ok(m) => m,
            Err(e) => {
                println!("SKIP|{name}|{e}");
                continue;
            }
        };
        let (marks, rows, _) = static_counts(&module.program);
        let mut line = format!(
            "STATIC|{name}|instr={marks}|cell rows={rows}|ratio={:.3}",
            rows as f64 / marks as f64
        );
        for k in REGISTERS {
            match allocate(&module.program, k) {
                Ok(p) => line += &format!("|K{k}={:.3}", static_counts(&p).1 as f64 / marks as f64),
                Err(e) => line += &format!("|K{k}=err {e}"),
            }
        }
        println!("{line}");
        let mut chosen: Option<RunRecord> = None;
        for (export, f) in entries(&module) {
            let (params, results) = &module.types[module.func_types[f as usize] as usize];
            let args: Vec<u64> = (0..params.len()).map(|i| [10, 7, 3, 2][i]).collect();
            let Some(expected) = reference(&bytes, &export, &args) else {
                println!("NOREF|{name}::{export}");
                continue;
            };
            let mut cell = Machine::new(&module.program, None);
            let got = cell.run(f as usize, &args, 50_000_000);
            if !agree(results, &expected, &got) {
                disagreements += 1;
                println!("DISAGREE|{name}::{export}|expected={expected:?} got={got:?}");
                continue;
            }
            let mut record = RunRecord {
                name: format!("{name}::{export}"),
                instr: cell.marks,
                cell: cell.rows,
                regs: BTreeMap::new(),
                mix: cell.mark_mix.clone(),
                rows: cell.row_mix.clone(),
                unaligned: cell.unaligned,
                depth: cell.max_depth,
            };
            for k in REGISTERS {
                let Ok(p) = allocate(&module.program, k) else { continue };
                let mut m = Machine::new(&p, Some(k));
                let got = m.run(f as usize, &args, 200_000_000);
                assert!(
                    agree(results, &expected, &got),
                    "{name}::{export} K{k}: {expected:?} {got:?}"
                );
                assert_eq!(
                    m.marks, cell.marks,
                    "{name}::{export} K{k}: the same original instructions"
                );
                record.regs.insert(k, (m.rows, m.tag_mix.clone()));
            }
            if chosen.as_ref().is_none_or(|c| c.instr < record.instr) {
                chosen = Some(record);
            }
        }
        if let Some(record) = chosen {
            best.push(record);
        }
    }
    println!("PROGRAM|run|instr|cell ratio|K8|K12|K16|K32|unaligned|call depth|K16 glue");
    let mut sums: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut mix_share: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut row_share: BTreeMap<String, f64> = BTreeMap::new();
    let counted: Vec<&RunRecord> = best.iter().filter(|r| r.instr >= 100).collect();
    for r in &best {
        let ratio = |rows: u64| rows as f64 / r.instr as f64;
        let mut line = format!("PROGRAM|{}|{}|{:.3}", r.name, r.instr, ratio(r.cell));
        for k in REGISTERS {
            line += &r
                .regs
                .get(&k)
                .map_or("|-".into(), |(rows, _)| format!("|{:.3}", ratio(*rows)));
        }
        let glue = r.regs.get(&16).map_or(String::new(), |(_, tags)| {
            tags.iter()
                .filter(|(t, _)| **t != Tag::Core)
                .map(|(t, c)| format!("{t:?}={c}"))
                .collect::<Vec<_>>()
                .join(",")
        });
        line += &format!("|{}|{}|{glue}", r.unaligned, r.depth);
        println!("{line}");
    }
    for r in &counted {
        sums.entry("cell".into())
            .or_default()
            .push(r.cell as f64 / r.instr as f64);
        for k in REGISTERS {
            if let Some((rows, _)) = r.regs.get(&k) {
                sums.entry(format!("K{k:02}"))
                    .or_default()
                    .push(*rows as f64 / r.instr as f64);
            }
        }
        for (op, c) in &r.mix {
            *mix_share.entry(op).or_default() += *c as f64 / r.instr as f64 / counted.len() as f64;
        }
        let rows: u64 = r.rows.values().sum();
        for (op, c) in &r.rows {
            *row_share.entry(op.clone()).or_default() += *c as f64 / rows as f64 / counted.len() as f64;
        }
    }
    for (form, ratios) in &sums {
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let min = ratios.iter().cloned().fold(f64::MAX, f64::min);
        let max = ratios.iter().cloned().fold(0.0, f64::max);
        println!(
            "SUMMARY|{form}|programs={}|mean={mean:.3}|min={min:.3}|max={max:.3}",
            ratios.len()
        );
    }
    let mut mix: Vec<_> = mix_share.into_iter().collect();
    mix.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("MIX (equal weight per program, share of original instructions)");
    for (op, share) in mix.iter().take(30) {
        println!("MIX|{op}|{:.2}%", 100.0 * share);
    }
    let mut rows: Vec<_> = row_share.into_iter().collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("ROWMIX (equal weight per program, share of cell-form rows)");
    for (op, share) in rows.iter().take(30) {
        println!("ROWMIX|{op}|{:.2}%", 100.0 * share);
    }
    assert_eq!(disagreements, 0, "every run agrees with wasmtime");
}
