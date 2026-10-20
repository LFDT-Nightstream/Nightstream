//! The Lean replay of the layer-1 verifier (`formal/compression-lean`,
//! `replayLayer1`) against the Rust verifier.
//!
//! The test proves a toy CE(B) claim, runs the Rust verifier on the honest
//! proof and on tampered variants with a backend that records every draw,
//! writes the variants to `target/lean-replay/layer1.json`, runs the Lean
//! executable on that file, and requires the same draws and the same verdict
//! for every variant. Rust's draws and verdicts go to `layer1.rust.txt` and
//! the replay's to `layer1.lean.txt` beside it; the executable never reads
//! Rust's. The replay takes WHIR openings as the Lean model does: it replays
//! their transcript but not their checks, so a variant that Rust rejects
//! inside WHIR must be rejected later by the replay, after every draw Rust
//! made.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use p3_sumcheck_v08::layout::plan_stacked_layout;
use p3_whir::transcript::WhirShape;

use super::layer1::{claim_mutations, proof_mutations, relation_of, setup_toy, transcript};
use super::verifier::statement_input;
use crate::circuit::r1cs::R1cs;
use crate::circuit::{Backend, Native};
use crate::device::Accelerator;
use crate::field::{ext_words, Ext, Gl};
use crate::gkr::GkrProof;
use crate::pcs::{Commitment, Opening, Pcs};
use crate::verifier::{verify, ProofView};
use crate::whir::seeds;
use crate::{prove, Claim, ClaimWords, Error, Proof, Relation};

/// A word and whether it is a lane of a permutation output.
#[derive(Clone, Copy)]
struct Word {
    value: Gl,
    sample: bool,
}

impl Word {
    fn plain(value: Gl) -> Self {
        Self { value, sample: false }
    }
}

/// One challenge as the verifier draws it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Draw {
    /// `sample_ext`: three permutation lanes made one extension value.
    Ext([u64; 3]),
    /// `sample_bits` or a WHIR query index: one lane split into bits.
    Sample(u64),
}

/// `Native` that records every draw. Only `Duplex` builds an extension value
/// from three permutation lanes or splits a lane into bits, so these calls
/// are exactly its draws.
#[derive(Default)]
struct Tap {
    draws: Vec<Draw>,
}

impl Backend for Tap {
    type F = Word;
    type E = Ext;

    fn constant(&mut self, value: u64) -> Word {
        Word::plain(Native.constant(value))
    }

    fn private(&mut self, value: u64) -> Word {
        Word::plain(Native.private(value))
    }

    fn public(&mut self, value: u64) -> Word {
        Word::plain(Native.public(value))
    }

    fn add(&mut self, a: Word, b: Word) -> Word {
        Word::plain(a.value + b.value)
    }

    fn sub(&mut self, a: Word, b: Word) -> Word {
        Word::plain(a.value - b.value)
    }

    fn scale(&mut self, a: Word, by: u64) -> Word {
        Word::plain(Native.scale(a.value, by))
    }

    fn mul(&mut self, a: Word, b: Word) -> Word {
        Word::plain(a.value * b.value)
    }

    fn ext(&mut self, coordinates: [Word; 3]) -> Ext {
        let value = Native.ext(coordinates.map(|word| word.value));
        if coordinates.iter().all(|word| word.sample) {
            self.draws
                .push(Draw::Ext(ext_words(value).map(|word| word.as_canonical_u64())));
        }
        value
    }

    fn coordinates(&mut self, a: Ext) -> [Word; 3] {
        Native.coordinates(a).map(Word::plain)
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> Ext {
        Native.ext_constant(value)
    }

    fn ext_add(&mut self, a: Ext, b: Ext) -> Ext {
        a + b
    }

    fn ext_sub(&mut self, a: Ext, b: Ext) -> Ext {
        a - b
    }

    fn ext_mul(&mut self, a: Ext, b: Ext) -> Ext {
        a * b
    }

    fn ext_scale(&mut self, a: Ext, by: Word) -> Ext {
        Native.ext_scale(a, by.value)
    }

    fn ext_inverse(&mut self, a: Ext, what: &'static str) -> Result<Ext, Error> {
        Native.ext_inverse(a, what)
    }

    fn assert_zero(&mut self, a: Word, what: &'static str) -> Result<(), Error> {
        Native.assert_zero(a.value, what)
    }

    fn bits(&mut self, a: Word, count: usize, what: &'static str) -> Result<Vec<Word>, Error> {
        if a.sample {
            self.draws.push(Draw::Sample(a.value.as_canonical_u64()));
        }
        Ok(Native
            .bits(a.value, count, what)?
            .into_iter()
            .map(Word::plain)
            .collect())
    }

    fn select(&mut self, bit: Word, zero: Word, one: Word) -> Word {
        if bit.value == Gl::ZERO {
            zero
        } else {
            one
        }
    }

    fn permute(&mut self, state: [Word; WIDTH]) -> [Word; WIDTH] {
        Native
            .permute(state.map(|word| word.value))
            .map(|value| Word { value, sample: true })
    }

    fn hint(&mut self, inputs: &[Word], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Word> {
        let inputs: Vec<Gl> = inputs.iter().map(|word| word.value).collect();
        Native
            .hint(&inputs, len, compute)
            .into_iter()
            .map(Word::plain)
            .collect()
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Word], what: &'static str) -> Result<(), Error> {
        let columns: Vec<Gl> = columns.iter().map(|word| word.value).collect();
        Native.r1cs(system, &columns, what)
    }
}

/// The name of a rejection.
fn label(error: Error) -> String {
    match error {
        Error::Rejected(what) => what.to_string(),
        other => other.to_string(),
    }
}

/// One variant's draws and verdict.
#[derive(Debug)]
struct Run {
    name: String,
    draws: Vec<Draw>,
    verdict: Result<(), String>,
}

/// The Rust verifier on `proof` after `seed`, with every draw recorded.
fn tap(relation: &Relation<'_>, seed: [Gl; 4], claim: &Claim, proof: &Proof) -> (Vec<Draw>, Result<(), String>) {
    let statement = ClaimWords::new(&relation.shape, claim).unwrap();
    let mut tap = Tap::default();
    let b = &mut tap;
    let seed = seed.map(|word| b.public(word.as_canonical_u64()));
    let statement = statement_input(b, &statement);
    let verdict =
        ProofView::read(b, relation, Some(proof)).and_then(|view| verify(b, relation, seed, &statement, &view));
    (tap.draws, verdict.map_err(label))
}

fn list(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}

fn object(fields: &[(&str, String)]) -> String {
    let fields: Vec<String> = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect();
    format!("{{{}}}", fields.join(","))
}

fn words(values: impl IntoIterator<Item = Gl>) -> String {
    list(
        values
            .into_iter()
            .map(|word| word.as_canonical_u64().to_string()),
    )
}

fn ext(value: Ext) -> String {
    words(ext_words(value))
}

fn exts(values: &[Ext]) -> String {
    list(values.iter().map(|&value| ext(value)))
}

fn pairs(values: &[[Ext; 2]]) -> String {
    list(values.iter().map(|pair| list(pair.map(ext))))
}

fn root(commitment: &Commitment) -> String {
    let [root] = commitment.roots() else {
        panic!("a layer-1 commitment has one root")
    };
    words(*root)
}

/// One WHIR opening's transcript constants: its separators and its counts.
fn whir_shape(pcs: &Pcs) -> String {
    let (config, protocol) = (pcs.config(), pcs.protocol());
    let tables = protocol.table_shapes();
    let (k, _) = plan_stacked_layout(&tables);
    let shape = WhirShape::new(config, protocol.num_openings());
    let batches: Vec<(usize, Vec<usize>)> = protocol
        .iter_openings()
        .map(|(table, batch)| (table, batch.current().to_vec()))
        .collect();
    let claims = batches.iter().map(|(_, columns)| columns.len()).sum();
    let sumcheck = |rounds: usize| {
        let seed = if rounds == 0 {
            "[]".to_string()
        } else {
            words(seeds::sumcheck(rounds))
        };
        object(&[("rounds", rounds.to_string()), ("seed", seed)])
    };
    let batches = batches.iter().map(|(table, columns)| {
        object(&[
            ("seed", words(seeds::opening(k, &tables, *table, columns))),
            ("size", columns.len().to_string()),
        ])
    });
    let rounds = shape.rounds.iter().map(|round| {
        object(&[
            ("oodSamples", round.ood_samples.to_string()),
            ("queryDraws", round.query_draws.to_string()),
            ("sumcheck", sumcheck(round.sumcheck.rounds)),
        ])
    });
    object(&[
        ("virtualSeed", words(seeds::virtual_claim(k, &tables))),
        ("oodSamples", shape.commitment_ood_samples.to_string()),
        ("batches", list(batches)),
        ("whirSeed", words(seeds::whir(config, protocol.num_openings()))),
        (
            "batchingSeed",
            words(seeds::batching(k, &tables, claims, shape.commitment_ood_samples)),
        ),
        ("initialSumcheck", sumcheck(shape.initial_sumcheck.rounds)),
        ("rounds", list(rounds)),
        ("finalPolyLen", shape.final_poly_len.to_string()),
        ("finalQueryDraws", shape.final_query_draws.to_string()),
        ("finalSumcheck", sumcheck(shape.final_sumcheck.rounds)),
    ])
}

fn relation_json(relation: &Relation<'_>) -> String {
    let (key, shape) = (relation.key, &relation.shape);
    let s = &key.structure;
    let classes = s
        .classes
        .iter()
        .map(|class| object(&[("len", class.len.to_string()), ("ratio", class.ratio.to_string())]));
    let segments = s.segments.iter().map(|segment| {
        object(&[
            ("first", segment.first.to_string()),
            ("classIndex", segment.class.to_string()),
            ("lane", segment.lane.to_string()),
        ])
    });
    let layout = object(&[
        ("matrices", s.matrices.to_string()),
        ("rowVariables", s.row_variables.to_string()),
        ("runVariables", s.run_variables.to_string()),
        ("slotVariables", s.slot_variables.to_string()),
        ("runBounds", list(s.run_bounds.iter().map(u64::to_string))),
        ("classes", list(classes)),
        ("segments", list(segments)),
        ("slots", s.slots.to_string()),
    ]);
    let key = object(&[
        ("root", words(key.root())),
        ("blocks", key.blocks.to_string()),
        ("rows", key.rows.to_string()),
        ("layout", layout),
    ]);
    object(&[
        ("key", key),
        ("publicBlocks", shape.public_blocks.to_string()),
        ("pointVariables", shape.point_variables.to_string()),
        ("normBound", shape.norm_bound.to_string()),
        ("kappa", shape.kappa.to_string()),
        ("queries", relation.queries.to_string()),
        ("p0Profile", words(relation.p0.profile_words())),
        ("p1Profile", words(relation.p1.profile_words())),
        ("commitmentSeed", words(seeds::commitment())),
        ("p0Whir", whir_shape(&relation.p0)),
        ("p1Whir", whir_shape(&relation.p1)),
    ])
}

fn statement_json(statement: &ClaimWords<Gl>) -> String {
    let rings = |values: &[[Gl; neo_math::D]]| list(values.iter().map(|ring| words(*ring)));
    let k = |values: &[[Gl; 2]]| list(values.iter().map(|value| words(*value)));
    object(&[
        ("commitment", rings(&statement.commitment)),
        ("publicInput", rings(&statement.public)),
        ("point", k(&statement.point)),
        ("evalK", k(&statement.eval_k)),
        ("evalA", list(statement.eval_a.iter().map(|values| k(values)))),
    ])
}

fn gkr_json(proof: &GkrProof) -> String {
    let layers = proof.layers.iter().map(|layer| {
        object(&[
            ("rounds", list(layer.rounds.iter().map(|message| exts(message)))),
            ("children", list(layer.children.iter().map(|values| exts(values)))),
        ])
    });
    object(&[("roots", pairs(&proof.roots)), ("layers", list(layers))])
}

/// The parts of a WHIR opening its transcript absorbs; the query rows and
/// paths stay out, since the replay does not run WHIR's checks.
fn opening_json(opening: &Opening) -> String {
    let whir = &opening.whir;
    let rounds = whir.rounds.iter().map(|round| {
        let cap = round.commitment.as_ref().expect("a WHIR round commits");
        object(&[
            ("root", words(cap.roots()[0])),
            ("ood", exts(&round.ood_answers)),
            ("sumcheck", pairs(&round.sumcheck.polynomial_evaluations)),
        ])
    });
    let final_sumcheck = whir
        .final_sumcheck
        .as_ref()
        .map_or(&[][..], |data| data.polynomial_evaluations.as_slice());
    object(&[
        ("initialOod", exts(&whir.initial_ood_answers)),
        ("evals", list(opening.evals.iter().map(|batch| exts(batch.current())))),
        ("initialSumcheck", pairs(&whir.initial_sumcheck.polynomial_evaluations)),
        ("rounds", list(rounds)),
        (
            "finalPoly",
            exts(
                whir.final_poly
                    .as_ref()
                    .expect("a final polynomial")
                    .as_slice(),
            ),
        ),
        ("finalSumcheck", pairs(final_sumcheck)),
    ])
}

fn proof_json(proof: &Proof) -> String {
    let [e_re, e_im, row] = proof.early_values.runs;
    let early_values = object(&[
        ("eRe", ext(e_re)),
        ("eIm", ext(e_im)),
        ("row", ext(row)),
        ("pairs", pairs(&proof.early_values.pairs)),
        ("block", ext(proof.early_values.block)),
    ]);
    let leaves = proof.leaves.iter().map(|leaf| {
        object(&[
            ("values", words(leaf.values.iter().copied())),
            ("path", list(leaf.path.iter().map(|digest| words(*digest)))),
        ])
    });
    object(&[
        ("p0", root(&proof.p0)),
        ("histogram", list(proof.histogram.iter().map(u32::to_string))),
        ("early", gkr_json(&proof.early)),
        ("earlyValues", early_values),
        ("quotient", exts(&proof.quotient)),
        ("linear", list(proof.linear.iter().map(|message| exts(message)))),
        ("finals", exts(&proof.finals)),
        ("partials", list(proof.partials.iter().map(|values| exts(values)))),
        ("p1", root(&proof.p1)),
        ("leaves", list(leaves)),
        ("late", gkr_json(&proof.late)),
        ("lateValues", pairs(&proof.late_values)),
        ("p1Opening", opening_json(&proof.p1_opening)),
        ("p0Opening", opening_json(&proof.p0_opening)),
    ])
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs in the replay's output format, so that the two compare with `diff`.
fn render(runs: &[Run]) -> String {
    let mut text = String::new();
    for run in runs {
        writeln!(text, "variant {}", run.name).unwrap();
        for draw in &run.draws {
            match draw {
                Draw::Ext([c0, c1, c2]) => writeln!(text, "e {c0} {c1} {c2}"),
                Draw::Sample(word) => writeln!(text, "s {word}"),
            }
            .unwrap();
        }
        match &run.verdict {
            Ok(()) => writeln!(text, "accept"),
            Err(what) => writeln!(text, "reject {what}"),
        }
        .unwrap();
    }
    text
}

/// The replay's output: per variant, its draws and its verdict.
fn parse(output: &str) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for line in output.lines() {
        let words: Vec<u64> = line
            .split(' ')
            .skip(1)
            .filter_map(|word| word.parse().ok())
            .collect();
        if let Some(name) = line.strip_prefix("variant ") {
            runs.push(Run {
                name: name.to_string(),
                draws: Vec::new(),
                verdict: Err("no verdict".to_string()),
            });
            continue;
        }
        let run = runs.last_mut().expect("a variant line comes first");
        match line.split(' ').next() {
            Some("e") => run.draws.push(Draw::Ext([words[0], words[1], words[2]])),
            Some("s") => run.draws.push(Draw::Sample(words[0])),
            Some("accept") => run.verdict = Ok(()),
            _ => {
                run.verdict = Err(line
                    .strip_prefix("reject ")
                    .expect("a draw or a verdict")
                    .to_string())
            }
        }
    }
    runs
}

#[test]
#[ignore = "needs the Lean replay: lake build replayLayer1 in formal/compression-lean"]
fn lean_replay_matches_the_rust_verifier() {
    let (toy, setup, _scratch) = setup_toy(2, 4, 6);
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
    let seed = crate::hash::seed(transcript(1));
    let mut variants = vec![("honest".to_string(), toy.claim.clone(), proof.clone())];
    for (name, changed) in proof_mutations(&proof) {
        variants.push((name.to_string(), toy.claim.clone(), changed));
    }
    for (name, claim) in claim_mutations(&toy.claim) {
        variants.push((format!("statement {name}"), claim, proof.clone()));
    }

    let mut expected = Vec::new();
    let mut json = Vec::new();
    for (name, claim, proof) in &variants {
        let (draws, verdict) = tap(&relation, seed, claim, proof);
        let native = crate::verify(&relation, transcript(1), claim, proof).map_err(label);
        assert_eq!(verdict, native, "{name}: the recording run is the verifier");
        assert_eq!(verdict.is_ok(), name == "honest", "{name}: Rust's verdict");
        let statement = ClaimWords::new(&relation.shape, claim).unwrap();
        json.push(object(&[
            ("name", format!("\"{name}\"")),
            ("seed", words(seed)),
            ("statement", statement_json(&statement)),
            ("proof", proof_json(proof)),
        ]));
        expected.push(Run {
            name: name.clone(),
            draws,
            verdict,
        });
    }
    let dir = workspace().join("target/lean-replay");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer1.json");
    std::fs::write(
        &path,
        object(&[("relation", relation_json(&relation)), ("variants", list(json))]),
    )
    .unwrap();
    std::fs::write(dir.join("layer1.rust.txt"), render(&expected)).unwrap();

    let replay = workspace().join("formal/compression-lean/.lake/build/bin/replayLayer1");
    assert!(
        replay.exists(),
        "wrote {}; build the replay with `lake build replayLayer1` in formal/compression-lean",
        path.display()
    );
    let output = Command::new(&replay).arg(&path).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let output = String::from_utf8(output.stdout).unwrap();
    std::fs::write(dir.join("layer1.lean.txt"), &output).unwrap();
    let replayed = parse(&output);
    assert_eq!(replayed.len(), expected.len());
    for (rust, lean) in expected.iter().zip(&replayed) {
        assert_eq!(lean.name, rust.name);
        let whir = rust
            .verdict
            .as_ref()
            .is_err_and(|what| what.starts_with("WHIR "));
        if whir {
            assert!(lean.verdict.is_err(), "{}: the replay rejects", rust.name);
            assert!(lean.draws.starts_with(&rust.draws), "{}: draws", rust.name);
        } else {
            assert_eq!(lean.verdict, rust.verdict, "{}: verdict", rust.name);
            assert_eq!(lean.draws, rust.draws, "{}: draws", rust.name);
        }
        eprintln!(
            "{:<28} Rust {:<38} Lean {:<30} {} of {} draws",
            rust.name,
            format!("{:?}", rust.verdict),
            format!("{:?}", lean.verdict),
            rust.draws.len(),
            lean.draws.len()
        );
    }
}
