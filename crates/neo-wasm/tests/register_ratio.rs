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
    /// General-row rows (run by `Cpu`), rows that `native` rows stand for,
    /// and the reason a run stopped early.
    general: Option<(u64, u64, String)>,
    families: BTreeMap<String, u64>,
    origins: BTreeMap<String, (u64, u64)>,
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
                general: None,
                families: BTreeMap::new(),
                origins: BTreeMap::new(),
            };
            if let Ok(p) = allocate(&module.program, register::lower::ALLOC_K) {
                let low = register::lower::lower(&p, f as usize);
                let words = (module.program.memory_pages as usize * 16384).max(1);
                let mut cpu = register::cpu::Cpu::new(&low, words, 1 << 20);
                let got = cpu.run(&args, 400_000_000);
                let natives: u64 = cpu.estimated.values().map(|e| e.0).sum();
                let extra: u64 = cpu.estimated.values().map(|e| e.1).sum();
                let note = match &got {
                    Ok(regs) => {
                        let ok =
                            matches!(&expected, Ok(e) if masked(results, e) == masked(results, &regs[..results.len()]));
                        if ok {
                            "ok".to_string()
                        } else {
                            format!("WRONG {expected:?} {:?}", &regs[..results.len()])
                        }
                    }
                    Err(stop) => format!("{stop:?}"),
                };
                let natives_note: Vec<String> = cpu
                    .estimated
                    .iter()
                    .map(|(k, v)| format!("{k}={}", v.0))
                    .collect();
                record.general = Some((cpu.rows - natives, extra, format!("{note} {}", natives_note.join(","))));
                record.families = cpu
                    .families
                    .iter()
                    .map(|(k, v)| (format!("{k:?}"), *v))
                    .collect();
                let names = register::lower::origins().lock().unwrap();
                record.origins = cpu
                    .origins
                    .iter()
                    .map(|(k, v)| (names[*k as usize].clone(), *v))
                    .collect();
            }
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
        if let Some((rows, extra, note)) = &r.general {
            line += &format!(
                "|general={:.3} (+{:.3} estimated) {note}",
                *rows as f64 / r.instr as f64,
                *extra as f64 / r.instr as f64
            );
        }
        println!("{line}");
    }
    let mut fam_share: BTreeMap<String, f64> = BTreeMap::new();
    let mut origin_share: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for r in &counted {
        if let Some((rows, extra, note)) = &r.general {
            if note.starts_with("ok") {
                sums.entry("general".into())
                    .or_default()
                    .push((*rows + *extra) as f64 / r.instr as f64);
                let total: u64 = r.families.values().sum();
                for (k, v) in &r.families {
                    *fam_share.entry(k.clone()).or_default() += *v as f64 / total as f64;
                }
                for (k, (first, all)) in &r.origins {
                    let e = origin_share.entry(k.clone()).or_default();
                    e.0 += *first as f64 / r.instr as f64;
                    e.1 += *all as f64 / r.instr as f64;
                }
            }
        }
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
    let programs = sums.get("general").map_or(1, |v| v.len()) as f64;
    for (k, v) in &fam_share {
        println!("FAMILY|{k}|{:.2}%", 100.0 * v / programs);
    }
    let mut origins: Vec<_> = origin_share.into_iter().collect();
    origins.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (k, (first, all)) in origins.iter().take(30) {
        println!(
            "ORIGIN|{k}|register rows/instr={:.4}|general rows/instr={:.4}|x{:.2}",
            first / programs,
            all / programs,
            all / first
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
