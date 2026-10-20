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
//! For `Shared`, `Early` and the layer-1 verifier, `lowerCheck` also writes
//! the program with the Lean builder and requires the logged instructions:
//! all of `Shared`'s and `Early`'s, and the verifier's first ones, up to
//! `verifier.rs:168`.
//! The verifier's relation goes to `target/lean-lowering/layer1.json`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use p3_whir::transcript::WhirShape;

use super::layer1::{relation_of, setup_toy, transcript};
use super::replay::relation_json;
use super::shrink::Layer1;
use crate::circuit::algebra::{self, Kx, K};
use crate::circuit::block::Kind;
use crate::circuit::fold::FoldTranscript;
use crate::circuit::hash::{compress, hash_leaf, merkle_root, Duplex};
use crate::circuit::r1cs::R1cs;
use crate::circuit::record::{Form, Recorder, Trace, Wire};
use crate::circuit::Backend;
use crate::device::Accelerator;
use crate::field::{Ext, Gl};
use crate::gkr::{self, GkrView, TreeShape};
use crate::hash::seed;
use crate::matrix::{EarlyChallenges, EarlyValues, Structure};
use crate::pcs::Pcs;
use crate::shrink::Program;
use crate::{norm, prove, sumcheck, Error, Relation};

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

/// Every branch of `record.rs`: merged, cancelled, zero-weight and long
/// forms; constant sides of products; extension products by and from
/// constants; a folded, a recorded and a failing constant inverse; a skipped
/// and a failing constant assertion; short and 64-bit splits; a selection; a
/// hint; a fixed R1CS system over forms with constants; and a permutation of
/// mixed lanes. Some checks fail: the rows do not depend on that.
struct Edges;

impl Program for Edges {
    fn statement(&self) -> Vec<u64> {
        vec![7, 11]
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let t = b.public(7);
        let t2 = b.public(11);
        let s = b.private(5);
        let u = b.private(3);
        let w = b.private(9);
        let zero = b.constant(0);
        let two = b.constant(2);
        let double = b.add(s, s);
        let sum = b.add(s, u);
        let sum = b.add(sum, t);
        let four = b.add(sum, t2);
        let shifted = b.add(four, two);
        let five = b.add(shifted, w);
        let cancelled = b.sub(four, s);
        let scaled = b.scale(cancelled, 3);
        let vanished = b.scale(double, 0);
        let left = b.mul(two, s);
        let right = b.mul(scaled, two);
        let constant = b.mul(two, two);
        let product = b.mul(shifted, five);
        let e = b.ext([s, shifted, product]);
        let c = b.ext_constant([3, 0, 1]);
        let by_constant = b.ext_mul(c, e);
        let from_constant = b.ext_mul(e, c);
        let both = b.ext_mul(e, by_constant);
        let folded = b.ext_inverse(c, "constant inverse")?;
        let probe = b.ext_mul(folded, e);
        b.assert_ext_zero(probe, "probe")?;
        let inverse = b.ext_inverse(e, "inverse")?;
        let sum3 = b.ext_add(e, inverse);
        let difference3 = b.ext_sub(sum3, from_constant);
        let scaled3 = b.ext_scale(difference3, u);
        let zero3 = b.ext_constant([0, 0, 0]);
        b.ext_inverse(zero3, "zero inverse")?;
        b.assert_zero(zero, "zero constant")?;
        b.assert_zero(two, "nonzero constant")?;
        b.assert_zero(vanished, "vanished")?;
        b.assert_equal(left, right, "difference")?;
        b.assert_equal(constant, two, "constants")?;
        let low = b.bits(s, 8, "short split")?;
        let all = b.bits(five, 64, "full split")?;
        let chosen = b.select(low[0], u, w);
        let hinted = b.hint(&[s], 2, &|values| vec![values[0] + 1, 9]);
        let mut system = R1cs::new(4);
        system.push(&[(0, 1)], &[(0, 1)], &[(1, 1)]);
        system.push(&[(2, 3), (3, 1), (2, 1)], &[(2, 1)], &[(3, 5)]);
        b.r1cs(&system, &[s, product, two, shifted], "system")?;
        let [_, _, scaled3] = b.coordinates(scaled3);
        let [_, both, _] = b.coordinates(both);
        let mut state = [zero; WIDTH];
        let lanes = [chosen, two, shifted, hinted[0], all[63], scaled3, both, five];
        state[..lanes.len()].copy_from_slice(&lanes);
        let digest = b.permute(state);
        b.assert_equal(digest[0], t, "digest")?;
        Ok(())
    }
}

/// Every helper that the layer-1 verifier shares, over proof words: the
/// algebra (`circuit/algebra.rs`), the hashes and the duplex
/// (`circuit/hash.rs`), the fold transcript and its handoff
/// (`circuit/fold.rs`) and `sumcheck::replay`. The duplex absorbs a full and
/// a partial chunk, reads its output, and duplexes once with no input.
/// `shared` in `Circuit/Check.lean` writes the same program.
struct Shared;

impl Program for Shared {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let words: Vec<B::F> = (0..14).map(|word| b.private(word)).collect();
        let x: Vec<B::E> = (0..4).map(|_| algebra::private_ext(b, Ext::ZERO)).collect();
        let scaled = algebra::scale(b, words[0], Gl::from_u64(3));
        let lifted = algebra::lift(b, scaled);
        let weighted = algebra::ext_scale_constant(b, lifted, Gl::from_u64(9));
        let dot = algebra::dot(b, &x[..2], &x[2..]);
        let eq = algebra::eq_eval(b, &x[..2], &x[2..]);
        let table = algebra::eq_table(b, &x[..2]);
        let at = algebra::eq_at(b, &x[1..], 5);
        let value = algebra::interpolate(b, &table, weighted);
        let horner = algebra::horner(b, &x, dot);
        let k = Kx::from_k(b, [scaled, words[1]]);
        let one = Kx::one(b);
        let zero = Kx::zero(b);
        let k = k.mul(b, Kx { re: eq, im: at });
        let k = k.add(b, one).scale(b, horner).one_minus(b).add(b, zero);
        let bits = b.bits(words[2], 5, "bits")?;
        let power = algebra::power_from_bits(b, Gl::from_u64(7), &bits);
        let leaf = hash_leaf(b, &words);
        let short = hash_leaf(b, &words[..3]);
        let node = compress(b, leaf, short);
        let root = merkle_root(b, leaf, &bits[..2], &[short, node]);
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &words);
        duplex.observe_ext(b, k.re);
        let first = duplex.sample(b);
        let drawn = duplex.sample_ext(b);
        let index = duplex.sample_bits(b, 10)?;
        let more: Vec<B::E> = (0..3).map(|_| duplex.sample_ext(b)).collect();
        duplex.observe(b, power);
        let rounds = [vec![x[0], x[1]], vec![x[2], x[3], value]];
        let (point, last) = sumcheck::replay(b, &mut duplex, &rounds, drawn);
        let mut fold = FoldTranscript::new(b);
        fold.absorb(b, &words);
        fold.absorb_constants(b, &[1, 2, 3]);
        let digest = fold.squeeze_digest(b);
        let seed = FoldTranscript::from_state(std::array::from_fn(|lane| words[lane % words.len()])).handoff(b);
        b.assert_ext_equal(last, point[1], "replay")?;
        b.assert_ext_equal(k.im, more[2], "draws")?;
        b.assert_equal(first, index[3], "index")?;
        for lane in 0..4 {
            b.assert_equal(root[lane], seed[lane], "root")?;
            b.assert_equal(digest[lane], fold.prefix()[lane], "digest")?;
        }
        Ok(())
    }
}

/// The early stage's helpers in `verify`'s order (`verifier.rs:154-167`), on
/// a structure that the toy relation does not have: three matrices, an
/// empty run range, a run count of `2^run_variables`, a GKR tree of depth
/// one and a row point shorter than `r`. The program then reads every
/// early opening. `early` in `Circuit/Check.lean` writes the same program.
struct Early;

impl Program for Early {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let structure = Structure {
            matrices: 3,
            row_variables: 1,
            run_variables: 3,
            slot_variables: 1,
            run_bounds: vec![0, 2, 2, 8],
            classes: Vec::new(),
            segments: Vec::new(),
            slots: 2,
        };
        let mut shapes = vec![TreeShape { depth: 2, factors: 0 }];
        shapes.extend(structure.early_shapes());
        let histogram: Vec<B::F> = (0..5).map(|count| b.private(count)).collect();
        let r: Vec<K<B>> = (0..3).map(|_| [b.private(1), b.private(2)]).collect();
        let view = GkrView::read(b, None, &shapes)?;
        let values = EarlyValues {
            runs: std::array::from_fn(|_| algebra::private_ext(b, Ext::ZERO)),
            pairs: (0..structure.matrices)
                .map(|_| std::array::from_fn(|_| algebra::private_ext(b, Ext::ZERO)))
                .collect(),
            block: algebra::private_ext(b, Ext::ZERO),
        };
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &histogram);
        let beta = duplex.sample_ext(b);
        let challenges = EarlyChallenges {
            lookup: [duplex.sample_ext(b), duplex.sample_ext(b)],
            scatter: [duplex.sample_ext(b), duplex.sample_ext(b)],
        };
        let total = norm::table_sum(b, &histogram, 2, beta)?;
        let claims = gkr::verify(b, &view, &shapes, &mut duplex)?;
        let early = structure.check_early(b, &r, challenges, &claims[1..], &values)?;
        for &x in early
            .run_point
            .iter()
            .chain(&early.pair_point)
            .chain(&early.row_point)
        {
            duplex.observe_ext(b, x);
        }
        let [row, slot, coefficient] = early.setup;
        b.assert_ext_equal(slot, total, "slot")?;
        b.assert_ext_equal(row, coefficient, "setup")?;
        b.assert_ext_equal(early.mult, claims[0].point[1], "mult")
    }
}

/// What `layer1` in `Circuit/Check.lean` reads: the replay's relation, and
/// the index bits and folding factor of each query set of the two openings.
fn layer1_input(relation: &Relation<'_>) -> String {
    let rows = |pcs: &Pcs| {
        let config = pcs.config();
        let shape = WhirShape::new(config, pcs.protocol().num_openings());
        let set =
            |index_bits: usize, folding: usize| format!("{{\"indexBits\":{index_bits},\"foldingFactor\":{folding}}}");
        let rounds: Vec<String> = shape
            .rounds
            .iter()
            .enumerate()
            .map(|(r, round)| set(round.index_bits, config.round_folding_factor(r)))
            .collect();
        let last = set(shape.final_index_bits, config.final_round_config().folding_factor);
        format!("{{\"rounds\":[{}],\"final\":{last}}}", rounds.join(","))
    };
    format!(
        "{{\"relation\":{},\"p1Rows\":{},\"p0Rows\":{}}}",
        relation_json(relation),
        rows(&relation.p1),
        rows(&relation.p0)
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
    let input = workspace().join("target/lean-lowering/layer1.json");
    let exports = [
        ("edges", export("edges", &Edges), vec![]),
        ("shared", export("shared", &Shared), vec!["shared".into()]),
        ("early", export("early", &Early), vec!["early".into()]),
        (
            "layer1",
            export("layer1", &layer1),
            vec!["layer1".into(), input.clone()],
        ),
    ];
    std::fs::write(&input, layer1_input(&relation)).unwrap();
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
