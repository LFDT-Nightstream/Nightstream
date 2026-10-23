//! The layer-1 verifier read by `Recorder`: satisfied rows for a true claim,
//! failing rows for every tampered input and every moved hint word, and rows
//! that do not depend on the proof.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::layer1::{claim_mutations, proof_mutations, relation_of, setup_toy, transcript};
use crate::circuit::block::Kind;
use crate::circuit::r1cs::R1cs;
use crate::circuit::record::{Recorder, Trace};
use crate::circuit::{Backend, Native};
use crate::device::Accelerator;
use crate::field::Gl;
use crate::hash::seed;
use crate::verifier::{verify, ProofView};
use crate::{prove, Claim, ClaimWords, Error, Proof, Relation};
use neo_math::D;

/// `statement` as statement words, in `ClaimWords::words` order.
pub(super) fn statement_input<B: Backend>(b: &mut B, statement: &ClaimWords<Gl>) -> ClaimWords<B::F> {
    let mut rings = |values: &[[Gl; D]]| -> Vec<[B::F; D]> {
        values
            .iter()
            .map(|ring| ring.map(|word| b.public(word.as_canonical_u64())))
            .collect()
    };
    let commitment = rings(&statement.commitment);
    let public = rings(&statement.public);
    let mut pairs = |values: &[[Gl; 2]]| -> Vec<[B::F; 2]> {
        values
            .iter()
            .map(|pair| pair.map(|word| b.public(word.as_canonical_u64())))
            .collect()
    };
    let point = pairs(&statement.point);
    let array = |words: Vec<[B::F; 2]>| -> [[B::F; 2]; D] { std::array::from_fn(|lane| words[lane]) };
    let eval_k = array(pairs(&statement.eval_k));
    let eval_a = statement
        .eval_a
        .iter()
        .map(|values| array(pairs(values)))
        .collect();
    ClaimWords {
        commitment,
        public,
        point,
        eval_k,
        eval_a,
    }
}

/// Run the verifier on `b` with the seed and the statement as statement words.
fn run<B: Backend>(
    b: &mut B,
    relation: &Relation<'_>,
    seed: [Gl; 4],
    claim: &Claim,
    proof: Option<&Proof>,
) -> Result<(), Error> {
    let statement = ClaimWords::new(&relation.shape, claim).unwrap();
    let seed = seed.map(|word| b.public(word.as_canonical_u64()));
    let statement = statement_input(b, &statement);
    let view = ProofView::read(b, relation, proof)?;
    verify(b, relation, seed, &statement, &view).map(drop)
}

fn record(
    relation: &Relation<'_>,
    seed: [Gl; 4],
    claim: &Claim,
    proof: Option<&Proof>,
) -> (Trace, Option<&'static str>) {
    let mut recorder = Recorder::new(Trace::default());
    run(&mut recorder, relation, seed, claim, proof).unwrap();
    recorder.finish()
}

/// Run the verifier on `inner` with hint word `target` moved; also return
/// the length of each hint call.
fn run_moved<B: Backend>(
    inner: B,
    target: usize,
    relation: &Relation<'_>,
    seed: [Gl; 4],
    claim: &Claim,
    proof: &Proof,
) -> (B, Vec<usize>, Result<(), Error>) {
    let mut b = MovedHint {
        inner,
        target,
        lengths: Vec::new(),
    };
    let result = run(&mut b, relation, seed, claim, Some(proof));
    (b.inner, b.lengths, result)
}

/// The recorder notes a failed check, and the rows themselves reject: a
/// recorded row or block fails on the recorded witness.
fn assert_rejected((trace, failure): (Trace, Option<&'static str>), name: &str) {
    assert!(failure.is_some(), "no noted failure: {name}");
    assert!(
        (0..trace.rows.len()).any(|r| trace.row_value(r) != Gl::ZERO) || trace.failing_block().is_some(),
        "every recorded row holds: {name}"
    );
}

/// `inner` with hint output word `target`, counted over all hints in call
/// order, moved by one. A shrink prover chooses every hint word.
struct MovedHint<B> {
    inner: B,
    target: usize,
    lengths: Vec<usize>,
}

impl<B: Backend> Backend for MovedHint<B> {
    type F = B::F;
    type E = B::E;

    fn constant(&mut self, value: u64) -> B::F {
        self.inner.constant(value)
    }

    fn private(&mut self, value: u64) -> B::F {
        self.inner.private(value)
    }

    fn public(&mut self, value: u64) -> B::F {
        self.inner.public(value)
    }

    fn add(&mut self, a: B::F, b: B::F) -> B::F {
        self.inner.add(a, b)
    }

    fn sub(&mut self, a: B::F, b: B::F) -> B::F {
        self.inner.sub(a, b)
    }

    fn scale(&mut self, a: B::F, by: u64) -> B::F {
        self.inner.scale(a, by)
    }

    fn mul(&mut self, a: B::F, b: B::F) -> B::F {
        self.inner.mul(a, b)
    }

    fn ext(&mut self, coordinates: [B::F; 3]) -> B::E {
        self.inner.ext(coordinates)
    }

    fn coordinates(&mut self, a: B::E) -> [B::F; 3] {
        self.inner.coordinates(a)
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> B::E {
        self.inner.ext_constant(value)
    }

    fn ext_add(&mut self, a: B::E, b: B::E) -> B::E {
        self.inner.ext_add(a, b)
    }

    fn ext_sub(&mut self, a: B::E, b: B::E) -> B::E {
        self.inner.ext_sub(a, b)
    }

    fn ext_mul(&mut self, a: B::E, b: B::E) -> B::E {
        self.inner.ext_mul(a, b)
    }

    fn ext_scale(&mut self, a: B::E, by: B::F) -> B::E {
        self.inner.ext_scale(a, by)
    }

    fn ext_inverse(&mut self, a: B::E, what: &'static str) -> Result<B::E, Error> {
        self.inner.ext_inverse(a, what)
    }

    fn assert_zero(&mut self, a: B::F, what: &'static str) -> Result<(), Error> {
        self.inner.assert_zero(a, what)
    }

    fn bits(&mut self, a: B::F, count: usize, what: &'static str) -> Result<Vec<B::F>, Error> {
        self.inner.bits(a, count, what)
    }

    fn select(&mut self, bit: B::F, zero: B::F, one: B::F) -> B::F {
        self.inner.select(bit, zero, one)
    }

    fn permute(&mut self, state: [B::F; WIDTH]) -> [B::F; WIDTH] {
        self.inner.permute(state)
    }

    fn hint(&mut self, inputs: &[B::F], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<B::F> {
        let start: usize = self.lengths.iter().sum();
        let moved = self.target.checked_sub(start).filter(|&k| k < len);
        self.lengths.push(len);
        self.inner.hint(inputs, len, &|values| {
            let mut words = compute(values);
            if let Some(k) = moved {
                words[k] = (Gl::from_u64(words[k]) + Gl::ONE).as_canonical_u64();
            }
            words
        })
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[B::F], what: &'static str) -> Result<(), Error> {
        self.inner.r1cs(system, columns, what)
    }
}

#[test]
fn recorded_layer1_verifier_holds_rejects_and_keeps_its_shape() {
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
    let honest_seed = seed(transcript(1));

    let (honest, failure) = record(&relation, honest_seed, &toy.claim, Some(&proof));
    assert_eq!(failure, None);
    for r in 0..honest.rows.len() {
        assert_eq!(honest.row_value(r), Gl::ZERO, "row {r}");
    }
    assert_eq!(honest.failing_block(), None);
    eprintln!(
        "layer-1 verifier: {} rows, {} glue cells, {} permutations",
        honest.rows.len(),
        honest.glue.len(),
        honest.count(Kind::Permutation)
    );

    let (shape, _) = record(&relation, honest_seed, &toy.claim, None);
    assert_eq!(honest.rows, shape.rows);
    assert_eq!(honest.entries, shape.entries);
    assert_eq!(honest.glue.len(), shape.glue.len());
    assert_eq!(
        honest.blocks.each_ref().map(Vec::len),
        shape.blocks.each_ref().map(Vec::len)
    );
    assert_eq!(honest.public.len(), shape.public.len());

    assert_rejected(
        record(&relation, seed(transcript(2)), &toy.claim, Some(&proof)),
        "fold transcript",
    );
    for (name, claim) in claim_mutations(&toy.claim) {
        assert_rejected(record(&relation, honest_seed, &claim, Some(&proof)), name);
    }
    for (name, changed) in proof_mutations(&proof) {
        assert_rejected(record(&relation, honest_seed, &toy.claim, Some(&changed)), name);
    }
}

/// `Native` rejects each moved hint word, so the shared verifier code binds
/// every word. `Recorder` reads a hint the same way at every word, so its
/// rows are checked at the first and last word of each hint call.
#[test]
fn every_hint_word_of_the_layer1_verifier_is_bound() {
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
    let honest_seed = seed(transcript(1));
    let moved = |inner, target| run_moved(inner, target, &relation, honest_seed, &toy.claim, &proof);

    let (_, lengths, honest) = moved(Native, usize::MAX);
    assert!(honest.is_ok());
    let words: usize = lengths.iter().sum();
    assert!(words > 0, "the verifier reads hint words");
    eprintln!("layer-1 verifier: {words} hint words in {} calls", lengths.len());
    (0..words).into_par_iter().for_each(|target| {
        assert!(
            moved(Native, target).2.is_err(),
            "Native accepts moved hint word {target}"
        );
    });

    let mut start = 0;
    for len in lengths {
        if len > 0 {
            for target in [start, start + len - 1] {
                let (recorder, _, result) = run_moved(
                    Recorder::new(Trace::default()),
                    target,
                    &relation,
                    honest_seed,
                    &toy.claim,
                    &proof,
                );
                result.unwrap();
                assert_rejected(recorder.finish(), &format!("hint word {target}"));
            }
        }
        start += len;
    }
}
