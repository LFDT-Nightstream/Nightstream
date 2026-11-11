//! Prototype of a register-machine row (scratch branch, not for review): one
//! row is one register operation, read as ONE program word at `pc` from a
//! ROM through the real Nebula memory check, with four 32-bit registers
//! carried in the row's state. The program word holds the operation flags,
//! one-hot register selectors, an 8-bit immediate and an 8-bit branch target,
//! so no flag is a checked witness column apart from its bit. A row runs
//! `dst = src1 ± src2` (or `± imm`), and with `BRNZ` jumps to `target` when
//! the result is not zero; `HALT` keeps `pc`. Local reads and writes,
//! constants, moves of the stack, and `br_if` are not rows of their own.
//!
//! `FIB` is the register form of `tests/wasm_fibonacci.rs` in `neo-wasm`:
//! 3 setup rows, then 4 rows for each 11-instruction round.

use std::num::NonZeroUsize;

use nightstream::nebula::{batched_slots, pack_regions, Circuit, Geometry, R1csApplication, RowRelation, Size};
use nightstream::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ConstraintTag, ContinuityCatalog, ContinuityGroup, ContinuityLink,
    MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemoryPreload, MemorySpec,
    R1csBuilder, R1csRelation,
};
use nightstream::{Goldilocks as F, PrimeCharacteristicRing};
use p3_field::Field;

const REGISTERS: usize = 4;
const ONE: usize = 0;
const PC: usize = 1;
const PC_AFTER: usize = 2;
const REG: usize = 3;
const REG_AFTER: usize = REG + REGISTERS;
const WORD: usize = REG_AFTER + REGISTERS;
const BIT: usize = WORD + 1;
const P1: usize = BIT + 32;
const P2: usize = P1 + REGISTERS;
const IMM_TERM: usize = P2 + REGISTERS;
const SIGNED: usize = IMM_TERM + 1;
const CARRY: usize = SIGNED + 1;
const SIGNED_CARRY: usize = CARRY + 1;
const RESULT: usize = SIGNED_CARRY + 1;
const RESULT_BIT: usize = RESULT + 1;
const INVERSE: usize = RESULT_BIT + 32;
const IS_ZERO: usize = INVERSE + 1;
const TAKEN: usize = IS_ZERO + 1;
const WIDTH: usize = TAKEN + 1;

/// Fields of the program word, as bit offsets.
const ADD: usize = 0;
const SUB: usize = 1;
const BRNZ: usize = 2;
const HALT: usize = 3;
const SRC1: usize = 4;
const SRC2: usize = 8;
const DST: usize = 12;
const IMM: usize = 16;
const TARGET: usize = 24;

const ROWS: usize = 64;
const STEPS: usize = 8;

fn f(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        -F::from_u64(value.unsigned_abs())
    }
}

fn bit(field: usize) -> usize {
    BIT + field
}

fn field_terms(offset: usize, width: usize) -> impl Iterator<Item = (usize, F)> {
    (0..width).map(move |i| (bit(offset + i), F::from_u64(1 << i)))
}

fn registry() -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "register row",
        start: 0,
        len: WIDTH,
        name: "words".into(),
        role: "prototype",
        width: ColumnWidth::Field,
    }])
    .unwrap()
}

/// The rows of one register operation.
fn relation() -> R1csRelation<&'static str> {
    let mut builder = R1csBuilder::new(1, ONE).unwrap();
    let sign = [(bit(ADD), F::ONE), (bit(SUB), -F::ONE)];
    builder.with_tag(ConstraintTag::new("program word", "row"), |rows| {
        for i in 0..32 {
            rows.push_boolean(bit(i));
        }
        rows.push_linear_zero(field_terms(0, 32).chain([(WORD, -F::ONE)]));
    });
    builder.with_tag(ConstraintTag::new("operands", "row"), |rows| {
        for k in 0..REGISTERS {
            rows.push_row([(bit(SRC1 + k), F::ONE)], [(REG + k, F::ONE)], [(P1 + k, F::ONE)]);
            rows.push_row([(bit(SRC2 + k), F::ONE)], [(REG + k, F::ONE)], [(P2 + k, F::ONE)]);
        }
        let no_src2 = std::iter::once((ONE, F::ONE)).chain((0..REGISTERS).map(|k| (bit(SRC2 + k), -F::ONE)));
        rows.push_row(no_src2, field_terms(IMM, 8), [(IMM_TERM, F::ONE)]);
    });
    builder.with_tag(ConstraintTag::new("add or subtract", "row"), |rows| {
        let b = (0..REGISTERS)
            .map(|k| (P2 + k, F::ONE))
            .chain([(IMM_TERM, F::ONE)]);
        rows.push_row(sign, b, [(SIGNED, F::ONE)]);
        rows.push_boolean(CARRY);
        rows.push_row(sign, [(CARRY, F::ONE)], [(SIGNED_CARRY, F::ONE)]);
        rows.push_linear_zero((0..REGISTERS).map(|k| (P1 + k, F::ONE)).chain([
            (SIGNED, F::ONE),
            (SIGNED_CARRY, -f(1 << 32)),
            (RESULT, -F::ONE),
        ]));
        for i in 0..32 {
            rows.push_boolean(RESULT_BIT + i);
        }
        rows.push_linear_zero(
            (0..32)
                .map(|i| (RESULT_BIT + i, F::from_u64(1 << i)))
                .chain([(RESULT, -F::ONE)]),
        );
    });
    builder.with_tag(ConstraintTag::new("write", "row"), |rows| {
        for k in 0..REGISTERS {
            rows.push_row(
                [(bit(DST + k), F::ONE)],
                [(RESULT, F::ONE), (REG + k, -F::ONE)],
                [(REG_AFTER + k, F::ONE), (REG + k, -F::ONE)],
            );
        }
    });
    builder.with_tag(ConstraintTag::new("next pc", "row"), |rows| {
        rows.push_row(
            [(RESULT, F::ONE)],
            [(INVERSE, F::ONE)],
            [(ONE, F::ONE), (IS_ZERO, -F::ONE)],
        );
        rows.push_row([(RESULT, F::ONE)], [(IS_ZERO, F::ONE)], []);
        rows.push_row(
            [(bit(BRNZ), F::ONE)],
            [(ONE, F::ONE), (IS_ZERO, -F::ONE)],
            [(TAKEN, F::ONE)],
        );
        rows.push_row(
            [(TAKEN, F::ONE)],
            field_terms(TARGET, 8).chain([(PC, -F::ONE), (ONE, -F::ONE)]),
            [(PC_AFTER, F::ONE), (PC, -F::ONE), (ONE, -F::ONE), (bit(HALT), F::ONE)],
        );
    });
    builder.build(&registry()).unwrap()
}

fn catalog() -> MemoryCatalog<u32> {
    MemoryCatalog::new(
        [MemorySpec {
            id: 0,
            kind: MemoryKind::Rom,
            ports: vec![MemoryPortSpec {
                address_columns: vec![PC],
                value_column: WORD,
                kind: MemoryPortKind::Read,
                activation: MemoryPortActivation::Always,
            }],
        }],
        &registry(),
    )
    .unwrap()
}

fn continuity() -> ContinuityCatalog {
    let link = |after, before| ContinuityLink {
        previous_step_column: after,
        next_step_column: before,
    };
    ContinuityCatalog::new(
        [ContinuityGroup {
            name: "pc and registers",
            role: "prototype",
            links: std::iter::once(link(PC_AFTER, PC))
                .chain((0..REGISTERS).map(|k| link(REG_AFTER + k, REG + k)))
                .collect(),
        }],
        &registry(),
    )
    .unwrap()
}

/// The geometry of `WasmSize::Small` (r 14, μ 10, 8 steps a segment, 32
/// segments) with one slot for each of the 64 rows of a step.
pub(crate) fn geometry() -> Geometry {
    Geometry::new(14, 10, ROWS, STEPS, 32).unwrap()
}

pub(crate) fn application() -> R1csApplication<u32> {
    let catalog = catalog();
    let (regions, _) = pack_regions([(0u32, MemoryKind::Rom, vec![4])]).unwrap();
    let rows = NonZeroUsize::new(ROWS).unwrap();
    let size = Size {
        geometry: geometry(),
        rows,
        regions,
        slots: batched_slots(&catalog, &[vec![0]], rows).unwrap(),
    };
    R1csApplication::new(RowRelation::of(&relation()).unwrap(), &continuity(), &catalog, size).unwrap()
}

/// One instruction: flags, sources, destination, immediate, target.
struct Op {
    flags: &'static [usize],
    src1: Option<usize>,
    src2: Option<usize>,
    dst: Option<usize>,
    imm: u32,
    target: u32,
}

impl Op {
    fn word(&self) -> u32 {
        let mut word = self.flags.iter().map(|flag| 1u32 << flag).sum::<u32>();
        word |= self.src1.map_or(0, |k| 1 << (SRC1 + k));
        word |= self.src2.map_or(0, |k| 1 << (SRC2 + k));
        word |= self.dst.map_or(0, |k| 1 << (DST + k));
        word | self.imm << IMM | self.target << TARGET
    }
}

/// `r0 = n`, `(r1, r2) = (1, 0)`, then `n` rounds of `r3 = r1 + r2`,
/// `r1 = r2`, `r2 = r3`, `r0 -= 1` with a branch while `r0 ≠ 0`.
fn fib(n: u32) -> Vec<Op> {
    let op = |flags, src1, src2, dst, imm, target| Op {
        flags,
        src1,
        src2,
        dst,
        imm,
        target,
    };
    vec![
        op(&[ADD], None, None, Some(0), n, 0),
        op(&[ADD], None, None, Some(1), 1, 0),
        op(&[ADD], None, None, Some(2), 0, 0),
        op(&[ADD], Some(1), Some(2), Some(3), 0, 0),
        op(&[ADD], Some(2), None, Some(1), 0, 0),
        op(&[ADD], Some(3), None, Some(2), 0, 0),
        op(&[SUB, BRNZ], Some(0), None, Some(0), 1, 3),
        op(&[HALT], None, None, None, 0, 0),
    ]
}

fn preload(program: &[Op]) -> MemoryPreload<u32> {
    let mut preload = MemoryPreload::default();
    for (pc, op) in (0u32..).zip(program) {
        preload.insert(0, vec![pc], op.word());
    }
    preload
}

/// The honest one-row witnesses of `rows` rows from `pc = 0` and zero
/// registers, and the final registers.
fn run(program: &[Op], rows: usize) -> (Vec<Vec<F>>, [u32; REGISTERS]) {
    let mut pc = 0usize;
    let mut reg = [0u32; REGISTERS];
    let mut out = Vec::with_capacity(rows);
    for _ in 0..rows {
        let op = &program[pc];
        let word = op.word();
        let mut z = vec![F::ZERO; WIDTH];
        z[ONE] = F::ONE;
        z[PC] = F::from_u64(pc as u64);
        z[WORD] = F::from_u32(word);
        for i in 0..32 {
            z[bit(i)] = F::from_u32((word >> i) & 1);
        }
        let has = |flag: usize| op.flags.contains(&flag);
        let a = op.src1.map_or(0, |k| reg[k]);
        let b = op.src2.map_or(op.imm, |k| reg[k]);
        for k in 0..REGISTERS {
            z[REG + k] = F::from_u32(reg[k]);
            z[P1 + k] = F::from_u32(if op.src1 == Some(k) { reg[k] } else { 0 });
            z[P2 + k] = F::from_u32(if op.src2 == Some(k) { reg[k] } else { 0 });
        }
        z[IMM_TERM] = F::from_u32(if op.src2.is_none() { op.imm } else { 0 });
        let (result, carry) = if has(ADD) {
            a.overflowing_add(b)
        } else if has(SUB) {
            a.overflowing_sub(b)
        } else {
            (a, false)
        };
        let sign = i64::from(has(ADD)) - i64::from(has(SUB));
        z[SIGNED] = f(sign * i64::from(b));
        z[CARRY] = F::from_bool(carry);
        z[SIGNED_CARRY] = f(sign * i64::from(carry));
        z[RESULT] = F::from_u32(result);
        for i in 0..32 {
            z[RESULT_BIT + i] = F::from_u32((result >> i) & 1);
        }
        z[INVERSE] = F::from_u32(result).try_inverse().unwrap_or(F::ZERO);
        z[IS_ZERO] = F::from_bool(result == 0);
        let taken = has(BRNZ) && result != 0;
        z[TAKEN] = F::from_bool(taken);
        if let Some(k) = op.dst {
            reg[k] = result;
        }
        pc = if taken {
            op.target as usize
        } else if has(HALT) {
            pc
        } else {
            pc + 1
        };
        z[PC_AFTER] = F::from_u64(pc as u64);
        for k in 0..REGISTERS {
            z[REG_AFTER + k] = F::from_u32(reg[k]);
        }
        out.push(z);
    }
    (out, reg)
}

/// Fibonacci with 127 rounds fills one segment: 3 + 4 · 127 = 511 rows and
/// one `HALT` row. The same run in WASM is 7 + 11 · 127 = 1,404 instructions.
const ROUNDS: u32 = 127;

fn fib_inputs() -> (Vec<Vec<F>>, Vec<u64>, MemoryPreload<u32>) {
    let program = fib(ROUNDS);
    let (rows, reg) = run(&program, ROWS * STEPS);
    let mut last = vec![7u64];
    last.extend(reg.iter().map(|&value| u64::from(value)));
    let inputs = rows.chunks(ROWS).map(<[Vec<F>]>::concat).collect();
    (inputs, last, preload(&program))
}

#[test]
fn honest_fibonacci_steps_hold_and_compute_fib() {
    let app = application();
    let (inputs, last, _) = fib_inputs();
    for step in &inputs {
        app.check(step).unwrap();
    }
    let (a, b) = (1..ROUNDS).fold((0u32, 1u32), |(a, b), _| (b, a.wrapping_add(b)));
    assert_eq!(last[3], u64::from(b), "r2 holds fib(127) mod 2^32");
    assert_eq!(last[2], u64::from(a));
    let one_row = relation();
    let rows = one_row.catalog().rows();
    let terms: usize = rows
        .iter()
        .map(|row| row.row().a_terms().len() + row.row().b_terms().len() + row.row().c_terms().len())
        .sum();
    eprintln!(
        "register row: {} rows and {terms} matrix terms per row, {} relation rows per step of {ROWS} rows, {WIDTH} columns",
        rows.len(),
        app.relation_rows(),
    );
}

#[test]
fn a_wrong_sum_or_a_sum_past_32_bits_is_rejected() {
    let app = application();
    let (inputs, _, _) = fib_inputs();
    let add_row = 3 + 4;
    let column = |name: usize| add_row * WIDTH + name;
    let mut wrong = inputs[0].clone();
    wrong[column(RESULT)] += F::ONE;
    assert!(app.check(&wrong).is_err(), "a result other than the sum fails");
    let mut wide = inputs[0].clone();
    wide[column(RESULT)] += f(1 << 32);
    for i in 0..32 {
        wide[column(RESULT_BIT + i)] = F::ZERO;
    }
    assert!(app.check(&wide).is_err(), "a result of 2^32 or more has no 32 bits");
}

#[test]
fn a_program_word_other_than_the_image_fails_the_memory_check() {
    use nightstream::nebula::{Run, SegmentError};
    let app = application();
    let (mut inputs, _, preload) = fib_inputs();
    let program = app.program(&preload).unwrap();
    let mut run = Run::new(app.clone(), &program, vec![0; 1 + REGISTERS]);
    run.segment(&inputs).expect("the honest run");
    inputs[0][3 * WIDTH + bit(DST + 3)] = F::ZERO;
    inputs[0][3 * WIDTH + WORD] -= F::from_u64(1 << (DST + 3));
    let mut run = Run::new(app, &program, vec![0; 1 + REGISTERS]);
    assert!(matches!(run.segment(&inputs), Err(SegmentError::Access)));
}

#[test]
#[ignore = "Full proof at the Small geometry; run it alone."]
fn a_register_row_proof_of_fibonacci_verifies() {
    use std::time::Instant;
    let started = Instant::now();
    let app = application();
    let (inputs, last, preload) = fib_inputs();
    let program = app.program(&preload).unwrap();
    let circuit = Circuit::compile(geometry(), app).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    let engine = nightstream::Engine::Optimized;
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let proving = Instant::now();
    let initial = vec![0; 1 + REGISTERS];
    let mut run = prover.start(&program, initial.clone()).unwrap();
    run.segment(&inputs).unwrap();
    assert_eq!(*run.state(), last);
    let proof = run.finish().unwrap();
    eprintln!("prove {STEPS} steps elapsed={:?}", proving.elapsed());
    let bytes = prover.encode_proof(&proof).unwrap();
    let proof = verifier.decode_proof(&bytes).unwrap();
    let verified = verifier.verify(&program, &initial, &last, &proof).unwrap();
    assert_eq!(verified.steps, STEPS as u64);
    let mut other = last.clone();
    other[3] ^= 1;
    assert!(verifier.verify(&program, &initial, &other, &proof).is_err());
    eprintln!("proof bytes={} total elapsed={:?}", bytes.len(), started.elapsed());
}
