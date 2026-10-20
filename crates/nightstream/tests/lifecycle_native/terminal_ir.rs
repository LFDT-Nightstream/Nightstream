//! The final program up to layer 1 against the Lean builder
//! (`formal/compression-lean`, `Circuit/Terminal.lean`).
//!
//! `Log` reads `FinalProgram::run` as one instruction per `Backend` call over
//! value handles, in the format of `neo-spartan`'s
//! `tests/internal/lowering.rs`. The layer-1 relation is a toy whose public
//! width the terminal claim does not fit, so `verify_with` returns at its
//! shape check, before any backend call: the log ends where layer 1 starts.
//! A second log replays the wrapper and calls `Package::apply` itself to read
//! the claim and the state that it hands to layer 1; its instructions must be
//! those of the run. The test writes the context, the IR and those handles to
//! `target/lean-lowering/terminal.txt` and requires `terminalCheck` to accept.
//!
//! `final_program_matches_lean` logs the whole shape run on the production
//! compression key and requires `terminalCheck` with Lean's layer-1 verifier
//! for that key (`production_layer1.json`, from `neo-spartan`'s
//! `production_layer1_matches_lean`).
//!
//! `terminal_digest_is_pinned` (CI) and `final_program_digest_is_pinned`
//! record the same two runs with `neo_spartan::shrink::run_digest` and
//! require the digests that `terminalCheck` prints, so a change to the
//! wrapper, `load` or `apply` that changes a row fails without Lean.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{CachedMatrixRows, SuperneoEvalCacheBuilder};
use neo_spartan::shrink::{run_digest, Program};
use neo_spartan::{Backend, Error, Key, R1cs, Relation};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::finish::CompressionKey;
use super::super::terminal::{package, FinalProgram};
use super::Stage1State;

/// Each call as one instruction; value `v` is the `v`-th output of the run.
#[derive(Default)]
struct Log {
    ir: String,
    values: u32,
}

impl Log {
    fn emit(&mut self, instruction: &str, outputs: usize) -> Vec<u32> {
        self.ir.push_str(instruction);
        self.ir.push('\n');
        let first = self.values;
        self.values += outputs as u32;
        (first..self.values).collect()
    }

    fn one(&mut self, instruction: &str) -> u32 {
        self.emit(instruction, 1)[0]
    }
}

fn canonical(value: u64) -> u64 {
    F::from_u64(value).as_canonical_u64()
}

fn handles(values: &[u32]) -> String {
    values.iter().map(|value| format!(" {value}")).collect()
}

impl Backend for Log {
    type F = u32;
    type E = [u32; 3];

    fn constant(&mut self, value: u64) -> u32 {
        self.one(&format!("c {}", canonical(value)))
    }

    fn private(&mut self, _value: u64) -> u32 {
        self.one("p")
    }

    fn public(&mut self, _value: u64) -> u32 {
        self.one("u")
    }

    fn add(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("l 2 {a} 1 {b} 1"))
    }

    fn sub(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("l 2 {a} 1 {b} {}", (-F::ONE).as_canonical_u64()))
    }

    fn scale(&mut self, a: u32, by: u64) -> u32 {
        self.one(&format!("l 1 {a} {}", canonical(by)))
    }

    fn mul(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("m {a} {b}"))
    }

    fn ext(&mut self, coordinates: [u32; 3]) -> [u32; 3] {
        coordinates
    }

    fn coordinates(&mut self, a: [u32; 3]) -> [u32; 3] {
        a
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> [u32; 3] {
        value.map(|word| self.constant(word))
    }

    fn ext_add(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        std::array::from_fn(|i| self.add(a[i], b[i]))
    }

    fn ext_sub(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        std::array::from_fn(|i| self.sub(a[i], b[i]))
    }

    fn ext_mul(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        let product = self.emit(&format!("x{}{}", handles(&a), handles(&b)), 3);
        [product[0], product[1], product[2]]
    }

    fn ext_scale(&mut self, a: [u32; 3], by: u32) -> [u32; 3] {
        std::array::from_fn(|i| self.mul(a[i], by))
    }

    fn ext_inverse(&mut self, a: [u32; 3], _what: &'static str) -> Result<[u32; 3], Error> {
        let inverse = self.emit(&format!("i{}", handles(&a)), 3);
        Ok([inverse[0], inverse[1], inverse[2]])
    }

    fn assert_zero(&mut self, a: u32, _what: &'static str) -> Result<(), Error> {
        self.emit(&format!("z {a}"), 0);
        Ok(())
    }

    fn bits(&mut self, a: u32, count: usize, _what: &'static str) -> Result<Vec<u32>, Error> {
        Ok(self.emit(&format!("b {a} {count}"), count))
    }

    fn select(&mut self, bit: u32, zero: u32, one: u32) -> u32 {
        let difference = self.sub(one, zero);
        let chosen = self.mul(bit, difference);
        self.add(zero, chosen)
    }

    fn permute(&mut self, state: [u32; WIDTH]) -> [u32; WIDTH] {
        let outputs = self.emit(&format!("q{}", handles(&state)), WIDTH);
        std::array::from_fn(|lane| outputs[lane])
    }

    fn hint(&mut self, _inputs: &[u32], len: usize, _compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<u32> {
        self.emit(&format!("h {len}"), len)
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[u32], _what: &'static str) -> Result<(), Error> {
        let mut line = format!("r {} {}{}", system.rows(), columns.len(), handles(columns));
        for row in 0..system.rows() {
            for side in system.row(row) {
                write!(line, " {}", side.len()).unwrap();
                for (column, coefficient) in side {
                    write!(line, " {column} {coefficient}").unwrap();
                }
            }
        }
        self.emit(&line, 0);
        Ok(())
    }
}

/// A key of four blocks, six rows and four matrices.
fn toy_key() -> Key {
    let mut builder = SuperneoEvalCacheBuilder::new(6, 4 * D, 4).unwrap();
    for row in 0..6 {
        for matrix in 0..4 {
            builder
                .push_row(matrix, row, [(row * 31 + matrix, F::ONE)])
                .unwrap();
        }
    }
    Key::derive(&CachedMatrixRows::new(&builder.finish().unwrap()).unwrap()).unwrap()
}

fn list(out: &mut String, name: &str, values: impl IntoIterator<Item = u32>) {
    let values: Vec<u32> = values.into_iter().collect();
    writeln!(out, "{name} {}{}", values.len(), handles(&values)).unwrap();
}

/// One public block: the terminal claim has five, so `verify_with` returns at
/// its shape check, before any backend call.
fn toy_relation(key: &Key) -> Relation<'_> {
    Relation::new(key, 1, 9, 3_672, 100.0).unwrap()
}

const CONTEXT: [u64; 4] = [101, 103, 107, 109];

fn state() -> Stage1State {
    Stage1State::new(3, [5, 7, 11, 13].map(F::from_u64), [17, 19, 23, 29].map(F::from_u64))
}

fn production_key() -> CompressionKey {
    let bytes = std::fs::read(workspace().join("target/compression-setup/key.bin")).unwrap();
    CompressionKey::from_bytes(&bytes).unwrap()
}

/// `FinalProgram::run` on `toy_relation`, which stops at layer 1's shape
/// check: the run up to layer 1.
struct Terminal<'a>(FinalProgram<'a>);

impl Program for Terminal<'_> {
    fn statement(&self) -> Vec<u64> {
        self.0.statement()
    }

    fn key(&self) -> Vec<u64> {
        self.0.key()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        match self.0.run(b) {
            Err(Error::Shape("claim words")) => Ok(()),
            other => other,
        }
    }
}

/// The run digest (`run_digest`) of the final program as `terminalCheck`
/// computes it from Lean's `finalProgram` (`digest` on its `match` line):
/// up to layer 1 on `toy_relation` and `CONTEXT`, and whole on the production
/// key. Update a pin only after its Lean check passes, from that line:
/// `terminal_program_matches_lean` or `final_program_matches_lean`.
const TERMINAL_DIGEST: [u64; 4] = [
    16410863328979593198,
    7181375196460713867,
    6085068597532652256,
    6171033847868749347,
];
const FINAL_PROGRAM_DIGEST: [u64; 4] = [
    5659430957240615970,
    6226718047465598965,
    5404693510076364812,
    7916513672709227231,
];

#[test]
fn terminal_digest_is_pinned() {
    let key = toy_key();
    let relation = toy_relation(&key);
    let state = state();
    let program = FinalProgram {
        context: CONTEXT,
        package: package().unwrap(),
        relation: &relation,
        state: &state,
        proof: None,
    };
    assert_eq!(run_digest(&Terminal(program)).unwrap(), TERMINAL_DIGEST);
}

#[test]
#[ignore = "needs the production compression key, target/compression-setup/key.bin"]
fn final_program_digest_is_pinned() {
    let key = production_key();
    let relation = key.relation().unwrap();
    let state = state();
    let program = key.program(&relation, package().unwrap(), &state, None);
    assert_eq!(run_digest(&program).unwrap(), FINAL_PROGRAM_DIGEST);
}

#[test]
#[ignore = "needs the Lean check: lake build terminalCheck in formal/compression-lean"]
fn terminal_program_matches_lean() {
    let package = package().unwrap();
    let key = toy_key();
    let relation = toy_relation(&key);
    let state = state();
    let context = CONTEXT;
    let program = FinalProgram {
        context,
        package,
        relation: &relation,
        state: &state,
        proof: None,
    };
    let mut run = Log::default();
    let stopped = program.run(&mut run);
    assert!(
        matches!(stopped, Err(Error::Shape("claim words"))),
        "verify_with returns at its shape check: {stopped:?}"
    );

    let mut replay = Log::default();
    let statement: Vec<u32> = program
        .statement()
        .into_iter()
        .map(|word| replay.public(word))
        .collect();
    let inverse = replay.hint(&[statement[0]], 1, &|_| Vec::new())[0];
    let product = replay.mul(statement[0], inverse);
    let one = replay.constant(1);
    replay
        .assert_equal(product, one, "positive iteration")
        .unwrap();
    let (claim, transcript) = package
        .apply(&mut replay, &statement, context, None)
        .unwrap();
    assert!(replay.ir == run.ir, "the replay writes the instructions of the run");

    let mut out = format!("context 4 {}\n", context.map(|word| word.to_string()).join(" "));
    out.push_str(&run.ir);
    out.push_str("end\n");
    list(&mut out, "commitment", claim.commitment.iter().flatten().copied());
    list(&mut out, "public", claim.public.iter().flatten().copied());
    list(&mut out, "point", claim.point.iter().flatten().copied());
    list(&mut out, "eval_k", claim.eval_k.iter().flatten().copied());
    list(&mut out, "eval_a", claim.eval_a.iter().flatten().flatten().copied());
    list(&mut out, "state", transcript.state());
    eprintln!("terminal: {} values, {} bytes", run.values, out.len());
    terminal_check("terminal.txt", &out, &[]);
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Write `out` to `target/lean-lowering/<name>` and require `terminalCheck`
/// to accept it with `args`.
fn terminal_check(name: &str, out: &str, args: &[PathBuf]) {
    let dir = workspace().join("target/lean-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, out).unwrap();
    let check = workspace().join("formal/compression-lean/.lake/build/bin/terminalCheck");
    assert!(
        check.exists(),
        "wrote {}; build the check with `lake build terminalCheck` in formal/compression-lean",
        path.display()
    );
    let output = Command::new(&check).arg(&path).args(args).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("terminalCheck: {}", stdout.trim_end());
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The final program on the production key and package: the shape run of
/// `FinalProgram::run`, terminal and layer 1, against Lean's `finalProgram`
/// with `verifyWith`.
#[test]
#[ignore = "needs target/compression-setup/key.bin, the production_layer1.json of neo-spartan's \
            production_layer1_matches_lean, and lake build terminalCheck in formal/compression-lean"]
fn final_program_matches_lean() {
    let key = production_key();
    let relation = key.relation().unwrap();
    let state = state();
    let program = key.program(&relation, package().unwrap(), &state, None);
    let mut run = Log::default();
    program.run(&mut run).unwrap();

    let context = program.context.map(|word| word.to_string()).join(" ");
    let out = format!("context 4 {context}\n{}end\n", run.ir);
    eprintln!("final program: {} values, {} bytes", run.values, out.len());
    let layer1 = workspace().join("target/lean-lowering/production_layer1.json");
    assert!(
        layer1.exists(),
        "run neo-spartan's production_layer1_matches_lean first"
    );
    terminal_check("final.txt", &out, &[layer1]);
}
