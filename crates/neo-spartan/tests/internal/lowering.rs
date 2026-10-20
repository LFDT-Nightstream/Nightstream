//! The Lean lowering (`formal/compression-lean`, `Circuit/Record.lean`)
//! against `record.rs` on real program runs.
//!
//! `Log` forwards every `Backend` call to a `Recorder<Trace>` and logs it as
//! one instruction over value handles (`Circuit/Ir.lean`). Handle `v` is the
//! `v`-th value of the run; each instruction appends its outputs in order.
//! `add`, `sub` and `scale` log `linear`. `select`, the extension sums and
//! scales and the default assertions log the calls that `record.rs` makes
//! for them. The test writes each program's statement, IR and trace to
//! `target/lean-lowering/<program>.txt`, runs `lowerCheck` on the file, and
//! requires the lowered run to equal the trace: every row with its seven
//! sides in order, the glue cells, the blocks and the statement words.
//!
//! The programs are in `lowering_stages.rs`. For every program but `Edges`,
//! `lowerCheck` also writes the program with the Lean builder and requires
//! all of the logged instructions. The verifier's relation and WHIR keys go
//! to `target/lean-lowering/layer1.json`, and `Linear`'s, `Honest`'s,
//! `Late`'s and `Openings`' inputs to `linear.json`, `honest.json`,
//! `late.json` and `openings.json`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use p3_sumcheck_v08::layout::plan_stacked_layout;
use p3_whir::transcript::WhirShape;

use super::layer1::{relation_of, setup_toy, transcript};
use super::lowering_stages::{Early, Edges, Honest, LaneSplit, Late, Linear, Openings, Shared};
use super::replay::relation_json;
use super::shrink::Layer1;
use crate::circuit::block::Kind;
use crate::circuit::r1cs::R1cs;
use crate::circuit::record::{Form, Recorder, Trace, Wire};
use crate::circuit::{Backend, Native};
use crate::device::Accelerator;
use crate::field::Gl;
use crate::hash::seed;
use crate::pcs::Pcs;
use crate::shrink::Program;
use crate::{prove, Error, Relation};

/// A recorded value and its handle.
#[derive(Clone, Copy)]
struct Value {
    form: Form,
    handle: u32,
}

/// `Recorder<Trace>` with every call logged as an instruction.
struct Log {
    recorder: Recorder<Trace>,
    ir: String,
    values: u32,
}

impl Log {
    fn new() -> Self {
        Self {
            recorder: Recorder::new(Trace::default()),
            ir: String::new(),
            values: 0,
        }
    }

    /// Log one instruction and hand out the handles of its outputs.
    fn emit(&mut self, instruction: String, forms: &[Form]) -> Vec<Value> {
        self.ir.push_str(&instruction);
        self.ir.push('\n');
        forms
            .iter()
            .map(|&form| {
                self.values += 1;
                Value {
                    form,
                    handle: self.values - 1,
                }
            })
            .collect()
    }

    fn one(&mut self, instruction: String, form: Form) -> Value {
        self.emit(instruction, &[form])[0]
    }

    fn linear(&mut self, parts: &[(Value, Gl)], form: Form) -> Value {
        let mut line = format!("l {}", parts.len());
        for (value, weight) in parts {
            write!(line, " {} {}", value.handle, weight.as_canonical_u64()).unwrap();
        }
        self.one(line, form)
    }
}

fn handles(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| format!(" {}", value.handle))
        .collect()
}

fn three(values: Vec<Value>) -> [Value; 3] {
    [values[0], values[1], values[2]]
}

impl Backend for Log {
    type F = Value;
    type E = [Value; 3];

    fn constant(&mut self, value: u64) -> Value {
        let form = self.recorder.constant(value);
        self.one(format!("c {}", Gl::from_u64(value).as_canonical_u64()), form)
    }

    fn private(&mut self, value: u64) -> Value {
        let form = self.recorder.private(value);
        self.one("p".to_string(), form)
    }

    fn public(&mut self, value: u64) -> Value {
        let form = self.recorder.public(value);
        self.one("u".to_string(), form)
    }

    fn add(&mut self, a: Value, b: Value) -> Value {
        let form = self.recorder.add(a.form, b.form);
        self.linear(&[(a, Gl::ONE), (b, Gl::ONE)], form)
    }

    fn sub(&mut self, a: Value, b: Value) -> Value {
        let form = self.recorder.sub(a.form, b.form);
        self.linear(&[(a, Gl::ONE), (b, -Gl::ONE)], form)
    }

    fn scale(&mut self, a: Value, by: u64) -> Value {
        let form = self.recorder.scale(a.form, by);
        self.linear(&[(a, Gl::from_u64(by))], form)
    }

    fn mul(&mut self, a: Value, b: Value) -> Value {
        let form = self.recorder.mul(a.form, b.form);
        self.one(format!("m {} {}", a.handle, b.handle), form)
    }

    fn ext(&mut self, coordinates: [Value; 3]) -> [Value; 3] {
        coordinates
    }

    fn coordinates(&mut self, a: [Value; 3]) -> [Value; 3] {
        a
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> [Value; 3] {
        value.map(|word| self.constant(word))
    }

    fn ext_add(&mut self, a: [Value; 3], b: [Value; 3]) -> [Value; 3] {
        std::array::from_fn(|i| self.add(a[i], b[i]))
    }

    fn ext_sub(&mut self, a: [Value; 3], b: [Value; 3]) -> [Value; 3] {
        std::array::from_fn(|i| self.sub(a[i], b[i]))
    }

    fn ext_mul(&mut self, a: [Value; 3], b: [Value; 3]) -> [Value; 3] {
        let product = self
            .recorder
            .ext_mul(a.map(|value| value.form), b.map(|value| value.form));
        three(self.emit(format!("x{}{}", handles(&a), handles(&b)), &product))
    }

    fn ext_scale(&mut self, a: [Value; 3], by: Value) -> [Value; 3] {
        std::array::from_fn(|i| self.mul(a[i], by))
    }

    fn ext_inverse(&mut self, a: [Value; 3], what: &'static str) -> Result<[Value; 3], Error> {
        let inverse = self.recorder.ext_inverse(a.map(|value| value.form), what)?;
        Ok(three(self.emit(format!("i{}", handles(&a)), &inverse)))
    }

    fn assert_zero(&mut self, a: Value, what: &'static str) -> Result<(), Error> {
        self.recorder.assert_zero(a.form, what)?;
        self.emit(format!("z {}", a.handle), &[]);
        Ok(())
    }

    fn bits(&mut self, a: Value, count: usize, what: &'static str) -> Result<Vec<Value>, Error> {
        let bits = self.recorder.bits(a.form, count, what)?;
        Ok(self.emit(format!("b {} {count}", a.handle), &bits))
    }

    fn select(&mut self, bit: Value, zero: Value, one: Value) -> Value {
        let difference = self.sub(one, zero);
        let chosen = self.mul(bit, difference);
        self.add(zero, chosen)
    }

    fn permute(&mut self, state: [Value; WIDTH]) -> [Value; WIDTH] {
        let outputs = self.recorder.permute(state.map(|value| value.form));
        let outputs = self.emit(format!("q{}", handles(&state)), &outputs);
        std::array::from_fn(|lane| outputs[lane])
    }

    fn hint(&mut self, inputs: &[Value], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Value> {
        let inputs: Vec<Form> = inputs.iter().map(|value| value.form).collect();
        let outputs = self.recorder.hint(&inputs, len, compute);
        self.emit(format!("h {len}"), &outputs)
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Value], what: &'static str) -> Result<(), Error> {
        let forms: Vec<Form> = columns.iter().map(|value| value.form).collect();
        self.recorder.r1cs(system, &forms, what)?;
        let mut line = format!("r {} {}{}", system.rows(), columns.len(), handles(columns));
        for row in 0..system.rows() {
            for (columns, coefficients) in system.sides(row) {
                write!(line, " {}", columns.len()).unwrap();
                for (column, coefficient) in columns.iter().zip(coefficients) {
                    write!(line, " {column} {}", coefficient.as_canonical_u64()).unwrap();
                }
            }
        }
        self.emit(line, &[]);
        Ok(())
    }
}

/// The index bits and folding factor of each query set of `pcs`
/// (`OpeningRows` in `Circuit/Views.lean`).
fn opening_rows(pcs: &Pcs) -> String {
    let config = pcs.config();
    let shape = WhirShape::new(config, pcs.protocol().num_openings());
    let set = |index_bits: usize, folding: usize| format!("{{\"indexBits\":{index_bits},\"foldingFactor\":{folding}}}");
    let rounds: Vec<String> = shape
        .rounds
        .iter()
        .enumerate()
        .map(|(r, round)| set(round.index_bits, config.round_folding_factor(r)))
        .collect();
    let last = set(shape.final_index_bits, config.final_round_config().folding_factor);
    format!("{{\"rounds\":[{}],\"final\":{last}}}", rounds.join(","))
}

/// What `whir::verify` reads of `pcs` beyond the replay's `WhirShape`
/// (`WhirKey` in `Circuit/Whir.lean`).
fn whir_key(pcs: &Pcs) -> String {
    let (config, protocol) = (pcs.config(), pcs.protocol());
    let tables = protocol.table_shapes();
    let (k, placements) = plan_stacked_layout(&tables);
    let join = |items: Vec<String>| items.join(",");
    let numbers = |values: &[usize]| join(values.iter().map(usize::to_string).collect());
    let arities: Vec<usize> = tables.iter().map(|table| table.num_variables()).collect();
    let batches = protocol
        .iter_openings()
        .map(|(table, batch)| format!("{{\"table\":{table},\"columns\":[{}]}}", numbers(batch.current())))
        .collect();
    let placements = placements
        .iter()
        .map(|placement| {
            let selectors = placement
                .selectors()
                .iter()
                .map(|s| format!("{{\"numVariables\":{},\"index\":{}}}", s.num_variables(), s.index()))
                .collect();
            format!("{{\"table\":{},\"selectors\":[{}]}}", placement.idx(), join(selectors))
        })
        .collect();
    let rounds = config
        .round_parameters()
        .iter()
        .map(|p| {
            format!(
                "{{\"numVariables\":{},\"logFoldedDomain\":{}}}",
                p.num_variables, p.log_folded_domain_size
            )
        })
        .collect();
    format!(
        "{{\"numVariables\":{k},\"arities\":[{}],\"batches\":[{}],\"placements\":[{}],\"rounds\":[{}],\
         \"finalLogFoldedDomain\":{},\"rows\":{}}}",
        numbers(&arities),
        join(batches),
        join(placements),
        join(rounds),
        config.final_round_config().log_folded_domain_size,
        opening_rows(pcs)
    )
}

/// What `layer1` in `Circuit/Check.lean` reads: the replay's relation and the
/// `WhirKey` of each opening.
fn layer1_input(relation: &Relation<'_>) -> String {
    format!(
        "{{\"relation\":{},\"p1Key\":{},\"p0Key\":{}}}",
        relation_json(relation),
        whir_key(&relation.p1),
        whir_key(&relation.p0)
    )
}

fn wire(out: &mut String, wire: Wire) {
    match wire {
        Wire::Glue(cell) => write!(out, " 0 {cell}"),
        Wire::Block(_, block, cell) => write!(out, " 1 {block} {cell}"),
        Wire::Public(index) => write!(out, " 2 {index}"),
    }
    .unwrap();
}

/// The statement, the IR, then the trace: its counts, its statement words and
/// every row as seven sides of `(wire, coefficient)` terms.
fn export_text(statement: &[u64], ir: &str, trace: &Trace) -> String {
    let mut out = format!("statement {}", statement.len());
    for word in statement {
        write!(out, " {word}").unwrap();
    }
    writeln!(out).unwrap();
    out.push_str(ir);
    writeln!(
        out,
        "end\ncells {} blocks {} publics {}",
        trace.glue.len(),
        trace.count(Kind::Permutation),
        trace.public.len()
    )
    .unwrap();
    for word in &trace.public {
        write!(out, " {}", word.as_canonical_u64()).unwrap();
    }
    writeln!(out, "\nrows {}", trace.rows.len()).unwrap();
    for offsets in &trace.rows {
        for side in 0..7 {
            let terms = &trace.entries[offsets[side] as usize..offsets[side + 1] as usize];
            write!(out, " {}", terms.len()).unwrap();
            for &(name, coefficient) in terms {
                wire(&mut out, name);
                write!(out, " {}", coefficient.as_canonical_u64()).unwrap();
            }
        }
        writeln!(out).unwrap();
    }
    out
}

/// Record `program` with and without the log, require one trace, and write
/// the export.
fn export(name: &str, program: &impl Program) -> PathBuf {
    let mut plain = Recorder::new(Trace::default());
    let plain_ok = program.run(&mut plain).is_ok();
    let mut log = Log::new();
    let logged_ok = program.run(&mut log).is_ok();
    let (trace, failure) = log.recorder.finish();
    let (plain, plain_failure) = plain.finish();
    assert_eq!(logged_ok, plain_ok, "{name}: one verdict");
    assert_eq!(failure, plain_failure, "{name}: one failed check");
    assert!(trace == plain, "{name}: the log leaves the trace unchanged");
    let text = export_text(&program.statement(), &log.ir, &trace);
    let dir = workspace().join("target/lean-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.txt"));
    std::fs::write(&path, &text).unwrap();
    eprintln!(
        "{name}: {} values, {} glue cells, {} blocks, {} rows, {} entries, {} bytes, first failed check {failure:?}",
        log.values,
        trace.glue.len(),
        trace.count(Kind::Permutation),
        trace.rows.len(),
        trace.entries.len(),
        text.len()
    );
    path
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "needs the Lean check: lake build lowerCheck in formal/compression-lean"]
fn lowering_matches_lean() {
    let (toy, setup, _scratch) = setup_toy(8, 4, 6);
    let relation = relation_of(&toy, setup.key());
    let proof = prove(
        &relation,
        &setup,
        &Accelerator::cpu(),
        transcript(1),
        &toy.claim,
        &toy.witness,
    )
    .unwrap();
    let layer1 = Layer1::new(&relation, seed(transcript(1)), &toy.claim, Some(&proof));
    let linear = Linear::new(&relation, seed(transcript(1)), &toy.claim, &proof);
    linear
        .run(&mut Native)
        .expect("the toy proof passes the linear claim");
    let honest = Honest::new(&relation, seed(transcript(1)), &toy.claim, &proof);
    honest
        .run(&mut Native)
        .expect("the toy proof passes the honest fold");
    let late = Late::new(&relation, seed(transcript(1)), &toy.claim, &proof);
    late.run(&mut Native)
        .expect("the toy proof passes the late checks");
    let openings = Openings::new(&relation, seed(transcript(1)), &toy.claim, &proof);
    openings
        .run(&mut Native)
        .expect("the toy proof passes the openings");
    let input = workspace().join("target/lean-lowering/layer1.json");
    let linear_input = workspace().join("target/lean-lowering/linear.json");
    let honest_input = workspace().join("target/lean-lowering/honest.json");
    let late_input = workspace().join("target/lean-lowering/late.json");
    let openings_input = workspace().join("target/lean-lowering/openings.json");
    let exports = [
        ("edges", export("edges", &Edges), vec![]),
        ("shared", export("shared", &Shared), vec!["shared".into()]),
        ("early", export("early", &Early), vec!["early".into()]),
        (
            "linear",
            export("linear", &linear),
            vec!["linear".into(), linear_input.clone()],
        ),
        (
            "honest",
            export("honest", &honest),
            vec!["honest".into(), honest_input.clone()],
        ),
        ("late", export("late", &late), vec!["late".into(), late_input.clone()]),
        (
            "lane_split",
            export("lane_split", &LaneSplit),
            vec!["lane_split".into()],
        ),
        (
            "openings",
            export("openings", &openings),
            vec!["openings".into(), openings_input.clone()],
        ),
        (
            "layer1",
            export("layer1", &layer1),
            vec!["layer1".into(), input.clone()],
        ),
    ];
    std::fs::write(&input, layer1_input(&relation)).unwrap();
    std::fs::write(
        &linear_input,
        format!(
            "{{\"relation\":{},\"pending\":{},\"output\":{}}}",
            relation_json(&relation),
            linear.input.len(),
            linear.output
        ),
    )
    .unwrap();
    std::fs::write(
        &honest_input,
        format!(
            "{{\"relation\":{},\"pending\":{},\"output\":{}}}",
            relation_json(&relation),
            honest.input.len(),
            honest.output
        ),
    )
    .unwrap();
    std::fs::write(
        &late_input,
        format!(
            "{{\"relation\":{},\"pending\":{},\"output\":{}}}",
            relation_json(&relation),
            late.input.len(),
            late.output
        ),
    )
    .unwrap();
    std::fs::write(
        &openings_input,
        format!(
            "{{\"relation\":{},\"p1Key\":{},\"p0Key\":{},\"pending\":{},\"output\":{}}}",
            relation_json(&relation),
            whir_key(&relation.p1),
            whir_key(&relation.p0),
            openings.input.len(),
            openings.output
        ),
    )
    .unwrap();
    let check = workspace().join("formal/compression-lean/.lake/build/bin/lowerCheck");
    assert!(
        check.exists(),
        "wrote target/lean-lowering; build the check with `lake build lowerCheck` in formal/compression-lean"
    );
    for (name, path, program) in exports {
        let output = Command::new(&check)
            .arg(&path)
            .args(program)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("{name}: {}", stdout.trim_end());
        assert!(
            output.status.success(),
            "{name}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
