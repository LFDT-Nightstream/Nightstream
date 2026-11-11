//! Milestone 2 of the register-row study (scratch, not for review): the
//! general register row on real programs, with the real Nebula memory
//! check, the real step packaging, adversarial tests for each group of
//! rows, and full proofs (ignored; run each alone).

mod register;

use nightstream::nebula::{size_parts, Circuit, R1csStepError, Run, SegmentError};
use nightstream::Goldilocks as F;
use nightstream::PrimeCharacteristicRing;

use register::alloc::allocate;
use register::corpus::{fib_wat, programs};
use register::cpu::{Cpu, RowInput};
use register::general::{application, geometry, preload, shape, state, witnesses, ROWS, STEPS};
use register::lower::{lower, Fam, Lowered, ALLOC_K};
use register::row::{synthesize_tampered, Cs, Private, LIN_WIDTH};
use register::translate::parse;

/// A program run ready for the circuit: its lowering, the trace padded to
/// whole segments, and the instructions it ran.
struct Prepared {
    name: String,
    low: Lowered,
    trace: Vec<RowInput>,
    rows: usize,
    instructions: u64,
    result: Vec<u64>,
}

fn prepare(name: &str, bytes: &[u8], export: &str, args: &[u64]) -> Result<Prepared, String> {
    let module = parse(bytes)?;
    let f = *module.exports.get(export).ok_or("export")? as usize;
    let mut cell = register::machine::Machine::new(&module.program, None);
    cell.run(f, args, 10_000_000)
        .map_err(|e| format!("{e:?}"))?;
    let p = allocate(&module.program, ALLOC_K)?;
    let low = lower(&p, f);
    if low.rows.len() > 1 << register::general::CODE_WIDTH {
        return Err(format!("{} rows of program", low.rows.len()));
    }
    let mut cpu = Cpu::new(&low, 1 << LIN_WIDTH, 1 << register::general::FRAME_WIDTH);
    if module
        .program
        .data
        .iter()
        .any(|(o, b)| *o as usize + b.len() > 4 << LIN_WIDTH)
    {
        return Err("data outside the region".into());
    }
    cpu.strict = true;
    cpu.trace = Some(Vec::new());
    let regs = cpu.run(args, 1 << 16).map_err(|e| format!("{e:?}"))?;
    let rows = cpu.rows as usize;
    let segment = ROWS * STEPS;
    let target = rows.div_ceil(segment).max(1) * segment;
    while cpu.trace.as_ref().unwrap().len() < target {
        cpu.step().map_err(|e| format!("{e:?}"))?;
    }
    let results = module.types[module.func_types[f] as usize].1.len();
    Ok(Prepared {
        name: name.to_string(),
        trace: cpu.trace.take().unwrap(),
        low,
        rows,
        instructions: cell.marks,
        result: regs[..results].to_vec(),
    })
}

fn steps(p: &Prepared) -> Vec<Vec<F>> {
    witnesses(&p.trace)
        .chunks(ROWS)
        .map(|rows| rows.concat())
        .collect()
}

/// The provable runs of the corpus: those that fit the Small regions.
fn provable() -> Vec<Prepared> {
    let mut out = Vec::new();
    for (name, bytes) in programs() {
        let Ok(module) = parse(&bytes) else { continue };
        let mut exports: Vec<_> = module
            .exports
            .iter()
            .map(|(n, f)| (n.clone(), *f))
            .collect();
        exports.sort();
        for (export, f) in exports {
            let (params, _) = &module.types[module.func_types[f as usize] as usize];
            if params.len() > 4 || (f as usize) < module.imports {
                continue;
            }
            let args: Vec<u64> = (0..params.len()).map(|i| [10, 7, 3, 2][i]).collect();
            match prepare(&format!("{name}::{export}"), &bytes, &export, &args) {
                Ok(p) if p.instructions >= 20 => out.push(p),
                Ok(_) => {}
                Err(e) => println!("NOT PROVABLE|{name}::{export}|{e}"),
            }
        }
    }
    out
}

#[test]
fn honest_runs_of_real_programs_hold_every_row_and_the_memory_check() {
    let (app, _) = application();
    let mut total = (0u64, 0usize);
    for p in provable() {
        let program = app.program(&preload(&p.low)).unwrap();
        let inputs = steps(&p);
        for step in &inputs {
            app.check(step)
                .unwrap_or_else(|e| panic!("{}: {e}", p.name));
        }
        let mut run = Run::new(app.clone(), &program, state(&p.trace[0]));
        for segment in inputs.chunks(STEPS) {
            run.segment(segment)
                .unwrap_or_else(|e| panic!("{}: {e:?}", p.name));
        }
        assert_eq!(
            &run.state()[3..3 + 2 * p.result.len()]
                .chunks(2)
                .map(|c| c[0] | c[1] << 32)
                .collect::<Vec<_>>(),
            &p.result
        );
        println!(
            "PROVABLE|{}|instructions={}|rows={}|rows/instr={:.3}|segments={}",
            p.name,
            p.instructions,
            p.rows,
            p.rows as f64 / p.instructions as f64,
            inputs.len() / STEPS
        );
        total.0 += p.instructions;
        total.1 += p.rows;
    }
    println!("PROVABLE TOTAL|instructions={}|rows={}", total.0, total.1);
}

#[test]
fn census_of_the_general_row() {
    let (cs, l) = shape();
    let (app, _) = application();
    let terms: usize = cs
        .rows
        .iter()
        .map(|r| r[0].len() + r[1].len() + r[2].len())
        .sum();
    println!(
        "ROW|rank-one rows={}|matrix terms={terms}|columns={}|links={}",
        cs.rows.len(),
        l.width,
        register::general::links(&l).len()
    );
    let mut tags = cs.tags.clone();
    tags.push(("end", cs.rows.len()));
    for w in tags.windows(2) {
        println!("GROUP|{}|{}", w[0].0, w[1].1 - w[0].1);
    }
    println!(
        "STEP|relation rows={}|per row={:.1}",
        app.relation_rows(),
        app.relation_rows() as f64 / ROWS as f64
    );
    let g = geometry();
    println!("GEOMETRY|{g:?}");
    for part in size_parts(&g, &app).unwrap() {
        println!(
            "PART|{}|rows={}|local={}|witness={}",
            part.name, part.size.rows, part.size.local_words, part.size.witness_words
        );
    }
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_general_row_proof_of_fibonacci_verifies() {
    prove("neo_fibonacci", &wat::parse_str(fib_wat(127)).unwrap(), "main", &[]);
}

fn corpus(name: &str) -> Vec<u8> {
    programs().into_iter().find(|(n, _)| n == name).unwrap().1
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_general_row_proof_of_recursive_factorial_verifies() {
    prove(
        "zk_factorial_rec",
        &corpus("zk_factorial_rec"),
        "recursive_factorial",
        &[10],
    );
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_general_row_proof_of_a_starstream_contract_verifies() {
    prove(
        "starstream_utxo_public",
        &corpus("starstream_utxo_public"),
        "counter#get-storage",
        &[10],
    );
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_general_row_proof_of_memory_sum_verifies() {
    prove("zk_memory-sum", &corpus("zk_memory-sum"), "sum_bytes", &[10]);
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_general_row_proof_of_every_family_verifies() {
    prove("mix_families", &corpus("mix_families"), "main", &[4]);
}

fn prove(name: &str, bytes: &[u8], export: &str, args: &[u64]) {
    use std::time::Instant;
    let started = Instant::now();
    let p = prepare(name, bytes, export, args).unwrap();
    let (app, _) = application();
    let program = app.program(&preload(&p.low)).unwrap();
    let inputs = steps(&p);
    let initial = state(&p.trace[0]);
    let circuit = Circuit::compile(geometry(), app.clone()).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    let engine = nightstream::Engine::Optimized;
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let proving = Instant::now();
    let mut run = prover.start(&program, initial.clone()).unwrap();
    for segment in inputs.chunks(STEPS) {
        run.segment(segment).unwrap();
    }
    let last = run.state().clone();
    let proof = run.finish().unwrap();
    eprintln!(
        "{name}: prove {} steps ({} rows, {} instructions) elapsed={:?}",
        inputs.len(),
        p.rows,
        p.instructions,
        proving.elapsed()
    );
    let bytes = prover.encode_proof(&proof).unwrap();
    let proof = verifier.decode_proof(&bytes).unwrap();
    let verified = verifier.verify(&program, &initial, &last, &proof).unwrap();
    assert_eq!(verified.steps, inputs.len() as u64);
    let mut other = last.clone();
    other[3] ^= 1;
    assert!(verifier.verify(&program, &initial, &other, &proof).is_err());
    eprintln!("proof bytes={} total elapsed={:?}", bytes.len(), started.elapsed());
}

fn fib_prepared() -> Prepared {
    prepare("fib", &wat::parse_str(fib_wat(40)).unwrap(), "main", &[]).unwrap()
}

/// The first row of family `fam` in the trace, and its index.
fn row_of(p: &Prepared, fam: Fam) -> usize {
    p.trace
        .iter()
        .position(|r| Fam::from((r.word & 31) as u8) == fam)
        .unwrap()
}

fn relation_fails(z: &[F]) -> bool {
    let (app, _) = application();
    matches!(app.check(z), Err(R1csStepError::Relation { .. }))
}

#[test]
fn a_result_other_than_the_sum_is_rejected() {
    let p = fib_prepared();
    let (_, l) = shape();
    let at = row_of(&p, Fam::Add);
    let mut inputs = steps(&p);
    let (step, row) = (at / ROWS, at % ROWS);
    let col = |c: usize| row * l.width + c;
    inputs[step][col(l.res[0])] += F::ONE;
    assert!(relation_fails(&inputs[step]), "a wrong result fails its family gate");
    let mut inputs = steps(&p);
    for &x in &l.x[..32] {
        inputs[step][col(x)] = F::ZERO;
    }
    inputs[step][col(l.carry[0])] = F::ONE - inputs[step][col(l.carry[0])];
    assert!(
        relation_fails(&inputs[step]),
        "a sum with another carry has no 32 result bits"
    );
}

#[test]
fn a_program_word_other_than_the_image_fails_the_memory_check() {
    let p = fib_prepared();
    let (app, l) = application();
    let program = app.program(&preload(&p.low)).unwrap();
    let mut inputs = steps(&p);
    let at = row_of(&p, Fam::Add);
    inputs[at / ROWS][(at % ROWS) * l.width + l.word] += F::ONE;
    let mut run = Run::new(app, &program, state(&p.trace[0]));
    let result = run.segment(&inputs[..STEPS]);
    assert!(result.is_err(), "the word is the image word at pc");
    let _ = SegmentError::Access;
}

#[test]
fn a_branch_to_another_pc_is_rejected() {
    let p = fib_prepared();
    let (_, l) = shape();
    let at = row_of(&p, Fam::Br);
    let mut inputs = steps(&p);
    inputs[at / ROWS][(at % ROWS) * l.width + l.pc1] += F::ONE;
    assert!(
        relation_fails(&inputs[at / ROWS]),
        "the next pc is the taken target or pc + 1"
    );
}

#[test]
fn a_write_to_a_register_other_than_the_destination_is_rejected() {
    let p = fib_prepared();
    let (_, l) = shape();
    let at = row_of(&p, Fam::Add);
    let d = (p.trace[at].word >> 19 & 15) as usize;
    let other = (d + 1) % 16;
    let mut inputs = steps(&p);
    inputs[at / ROWS][(at % ROWS) * l.width + l.regs1[other][0]] += F::ONE;
    assert!(relation_fails(&inputs[at / ROWS]), "only register D changes");
}

/// Every provable run, once.
fn runs() -> &'static [Prepared] {
    static RUNS: std::sync::OnceLock<Vec<Prepared>> = std::sync::OnceLock::new();
    RUNS.get_or_init(provable)
}

/// The first honest row of a provable run that `pick` accepts.
fn find(pick: impl Fn(&RowInput) -> bool) -> RowInput {
    runs()
        .iter()
        .flat_map(|p| p.trace.iter())
        .find(|r| pick(r))
        .expect("a provable run has such a row")
        .clone()
}

fn fam_of(r: &RowInput) -> Fam {
    Fam::from((r.word & 31) as u8)
}

/// The groups of the rows of the one-row relation that `z` breaks, in row
/// order.
fn broken(z: &[F]) -> Vec<&'static str> {
    let (cs, _) = shape();
    let eval = |t: &Vec<(usize, F)>| t.iter().fold(F::ZERO, |s, (c, x)| s + *x * z[*c]);
    let mut groups: Vec<&'static str> = cs
        .rows
        .iter()
        .enumerate()
        .filter(|(_, [a, b, c])| eval(a) * eval(b) != eval(c))
        .map(|(i, _)| {
            cs.tags
                .iter()
                .rev()
                .find(|(_, start)| *start <= i)
                .unwrap()
                .0
        })
        .collect();
    groups.dedup();
    groups
}

fn tampered(input: &RowInput, tamper: &dyn Fn(&mut Private)) -> Vec<F> {
    let mut cs = Cs::new(false);
    synthesize_tampered(&mut cs, input, tamper);
    cs.z
}

fn honest(input: &RowInput) -> Vec<F> {
    tampered(input, &|_| {})
}

const Q: u64 = 0xffff_ffff_0000_0001;

#[test]
fn a_product_written_as_its_alias_plus_q_is_rejected_only_by_the_alias_rows() {
    let row = find(|r| {
        let x = register::row::private(r).x;
        fam_of(r) == Fam::Mul && x > 0 && x < (1 << 32) - 1
    });
    assert!(broken(&honest(&row)).is_empty());
    let z = tampered(&row, &|p| {
        assert!(p.x < (1 << 32) - 1);
        p.x = p.x.wrapping_add(Q);
    });
    assert_eq!(
        broken(&z),
        ["multiplier"],
        "X + q satisfies the product row, so only the alias rows see it"
    );
}

#[test]
fn a_wrong_sign_bit_of_a_compare_is_rejected_by_the_compare_rows() {
    let row = find(|r| matches!(fam_of(r), Fam::Br | Fam::Cmp));
    let z = tampered(&row, &|p| p.y ^= 1 << 31);
    assert_eq!(broken(&z)[0], "compare");
}

#[test]
fn a_wrong_bitwise_operand_is_rejected_by_the_bitwise_rows() {
    let row = find(|r| matches!(fam_of(r), Fam::And | Fam::Or | Fam::Xor));
    let z = tampered(&row, &|p| p.y ^= 1);
    assert_eq!(broken(&z), ["bitwise"]);
}

#[test]
fn a_wrong_shift_amount_is_rejected_by_the_multiplier_rows() {
    let row = find(|r| fam_of(r) == Fam::Mul && (r.word >> 7 & 7) >= 3);
    let z = tampered(&row, &|p| p.s ^= 1);
    assert_eq!(broken(&z)[0], "multiplier");
}

#[test]
fn an_operand_other_than_register_a_is_rejected_by_the_selectors() {
    let row = find(|r| fam_of(r) == Fam::Add);
    let (_, l) = shape();
    let mut z = honest(&row);
    z[l.av[0]] += F::ONE;
    assert_eq!(broken(&z)[0], "operands");
}

#[test]
fn two_families_at_once_are_rejected_by_the_one_hot_rows() {
    let row = find(|r| fam_of(r) == Fam::Add);
    let (_, l) = shape();
    let mut z = honest(&row);
    z[l.fam[Fam::Mul as usize]] = F::ONE;
    assert_eq!(broken(&z)[0], "family");
}

#[test]
fn an_i32_result_with_a_high_limb_is_rejected_by_the_result_rows() {
    let row = find(|r| fam_of(r) == Fam::Add && r.word >> 5 & 1 == 0);
    let (_, l) = shape();
    let mut z = honest(&row);
    z[l.res[1]] = F::ONE;
    assert_eq!(broken(&z)[0], "result");
}

#[test]
fn a_store_or_a_call_that_writes_another_word_is_rejected_by_the_memory_rows() {
    let (_, l) = shape();
    for fam in [Fam::Store, Fam::Call] {
        let row = find(|r| fam_of(r) == fam);
        let mut z = honest(&row);
        z[l.new] += F::ONE;
        assert_eq!(broken(&z), ["memory"], "{fam:?}");
    }
}

#[test]
fn a_loaded_word_other_than_the_memory_cell_fails_the_memory_check() {
    let p = runs()
        .iter()
        .find(|p| p.name == "starstream_utxo_public::counter#get-storage")
        .unwrap();
    let (app, _) = application();
    let mut changed = Lowered {
        rows: p.low.rows.clone(),
        entry: p.low.entry.clone(),
        table: p.low.table.clone(),
        program: p.low.program.clone(),
    };
    changed.program.globals[2] ^= 1;
    let program = app.program(&preload(&changed)).unwrap();
    let mut run = Run::new(app.clone(), &program, state(&p.trace[0]));
    assert!(
        run.segment(&steps(p)[..STEPS]).is_err(),
        "the witness loads the old global, the memory holds another"
    );
    let program = app.program(&preload(&p.low)).unwrap();
    let mut run = Run::new(app, &program, state(&p.trace[0]));
    assert!(run.segment(&steps(p)[..STEPS]).is_ok());
}
