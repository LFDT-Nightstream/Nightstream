//! The layer-1 verifier read by `Recorder`: satisfied rows for a true claim,
//! a failing row at the first rejected check for every tampered input and
//! for the first and last moved word of each hint call, and rows that do not
//! depend on the proof. `Fast` rejects every moved hint word.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_verified::circuit::Kind;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::layer1::{claim_mutations, proof_mutations, relation_of, setup_toy, transcript, whir_mutations};
use super::shrink::Layer1;
use crate::circuit::record::{Form, Recorder, Trace};
use crate::circuit::{Backend, Fast, Hint, Rejected};
use crate::device::Accelerator;
use crate::field::{from_verified, Ext, Gl};
use crate::hash::seed;
use crate::shrink::Program;
use crate::R1cs;
use crate::{prove, ClaimWords, Error};
use neo_math::D;
use neo_verified::native::hint_words;

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

/// Record every row of `layer1`.
fn record(layer1: &Layer1<'_>) -> (Trace, Option<&'static str>) {
    let mut recorder = Recorder::new(Trace::default());
    layer1.run(&mut recorder).unwrap();
    recorder.finish()
}

/// Run `layer1` on `inner` with hint word `moved` moved by one.
fn probe<B: Values>(inner: B, layer1: &Layer1<'_>, moved: Option<usize>) -> (Probe<B>, Result<(), Error>) {
    let mut b = Probe {
        inner,
        moved,
        lengths: Vec::new(),
    };
    let result = layer1.run(&mut b);
    (b, result)
}

/// Record `layer1` with hint word `moved` moved. `Fast` rejects the run,
/// the recorder notes a failure, and a recorded row fails on the recorded
/// witness. The run stops at the first check that `Fast` rejects, so a
/// check that notes a failure but writes no row cannot pass on a later row.
fn assert_rejected(layer1: &Layer1<'_>, moved: Option<usize>, name: &str) {
    let (probe, result) = probe(Recorder::new(Trace::default()), layer1, moved);
    let (trace, failure) = probe.inner.finish();
    assert!(result.is_err(), "Fast accepts: {name}");
    assert!(failure.is_some(), "no noted failure: {name}");
    assert!(
        (0..trace.rows.len()).any(|r| trace.row_value(r) != Gl::ZERO),
        "every recorded row holds: {name}"
    );
}

/// The value of a word, so that a run can stop where `Fast` stops.
trait Values: Backend {
    fn value(a: Self::F) -> Gl;
    fn ext_value(a: Self::E) -> Ext;
}

impl Values for Fast {
    fn value(a: Gl) -> Gl {
        a
    }

    fn ext_value(a: Ext) -> Ext {
        a
    }
}

impl Values for Recorder<Trace> {
    fn value(a: Form) -> Gl {
        from_verified(a.value())
    }

    fn ext_value(a: [Form; 3]) -> Ext {
        Ext::from_basis_coefficients_fn(|i| from_verified(a[i].value()))
    }
}

/// `inner` with hint output word `moved`, counted over all hints in call
/// order, moved by one, and a stop at the first check that `Fast` rejects.
/// A shrink prover chooses every hint word.
struct Probe<B> {
    inner: B,
    moved: Option<usize>,
    lengths: Vec<usize>,
}

impl<B: Values> Backend for Probe<B> {
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

    fn ext_inverse(&mut self, a: B::E, what: &'static str) -> Result<B::E, Rejected> {
        let inverse = self.inner.ext_inverse(a, what)?;
        Fast.ext_inverse(B::ext_value(a), what).map(|_| inverse)
    }

    fn assert_zero(&mut self, a: B::F, what: &'static str) -> Result<(), Rejected> {
        self.inner.assert_zero(a, what)?;
        Fast.assert_zero(B::value(a), what)
    }

    fn bits(&mut self, a: B::F, count: usize, what: &'static str) -> Result<Vec<B::F>, Rejected> {
        let bits = self.inner.bits(a, count, what)?;
        Fast.bits(B::value(a), count, what).map(|_| bits)
    }

    fn select(&mut self, bit: B::F, zero: B::F, one: B::F) -> B::F {
        self.inner.select(bit, zero, one)
    }

    fn permute(&mut self, state: [B::F; WIDTH]) -> [B::F; WIDTH] {
        self.inner.permute(state)
    }

    fn hint(&mut self, inputs: &[B::F], len: usize, hint: &Hint) -> Vec<B::F> {
        let start: usize = self.lengths.iter().sum();
        let moved = self
            .moved
            .and_then(|target| target.checked_sub(start))
            .filter(|&k| k < len);
        self.lengths.push(len);
        let values: Vec<u64> = inputs
            .iter()
            .map(|&input| B::value(input).as_canonical_u64())
            .collect();
        let mut words = hint_words(hint, &values);
        words.resize(len, 0);
        if let Some(k) = moved {
            words[k] = (Gl::from_u64(words[k]) + Gl::ONE).as_canonical_u64();
        }
        self.inner.hint(inputs, len, &Hint::Words(words))
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[B::F], what: &'static str) -> Result<(), Rejected> {
        self.inner.r1cs(system, columns, what)?;
        let values: Vec<Gl> = columns.iter().map(|&column| B::value(column)).collect();
        Fast.r1cs(system, &values, what)
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

    let (honest, failure) = record(&Layer1::new(&relation, honest_seed, &toy.claim, Some(&proof)));
    assert_eq!(failure, None);
    for r in 0..honest.rows.len() {
        assert_eq!(honest.row_value(r), Gl::ZERO, "row {r}");
    }
    assert!(honest.failing_block().is_none());
    eprintln!(
        "layer-1 verifier: {} rows, {} glue cells, {} permutations",
        honest.rows.len(),
        honest.glue.len(),
        honest.count(Kind::Permutation)
    );

    let (shape, _) = record(&Layer1::new(&relation, honest_seed, &toy.claim, None));
    assert_eq!(honest.rows, shape.rows);
    assert!(honest.entries == shape.entries);
    assert_eq!(honest.glue.len(), shape.glue.len());
    assert_eq!(
        honest.blocks.each_ref().map(Vec::len),
        shape.blocks.each_ref().map(Vec::len)
    );
    assert_eq!(honest.public.len(), shape.public.len());

    assert_rejected(
        &Layer1::new(&relation, seed(transcript(2)), &toy.claim, Some(&proof)),
        None,
        "fold transcript",
    );
    for (name, claim) in claim_mutations(&toy.claim) {
        assert_rejected(&Layer1::new(&relation, honest_seed, &claim, Some(&proof)), None, name);
    }
    for (name, changed) in proof_mutations(&proof)
        .into_iter()
        .chain(whir_mutations(&proof))
    {
        assert_rejected(
            &Layer1::new(&relation, honest_seed, &toy.claim, Some(&changed)),
            None,
            name,
        );
    }
}

/// `Fast` rejects each moved hint word, so the shared verifier code binds
/// every word. The recorder's rows are checked only at the first and last
/// word of each hint call: the one hint site, `whir::stir`, sends every word
/// of a call through the same kind of `merkle_root` rows (select, compress,
/// root equality).
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
    let honest = Layer1::new(&relation, seed(transcript(1)), &toy.claim, Some(&proof));

    let (native, result) = probe(Fast, &honest, None);
    assert!(result.is_ok());
    let words: usize = native.lengths.iter().sum();
    assert!(words > 0, "the verifier reads hint words");
    eprintln!("layer-1 verifier: {words} hint words in {} calls", native.lengths.len());
    (0..words).into_par_iter().for_each(|target| {
        assert!(
            probe(Fast, &honest, Some(target)).1.is_err(),
            "Fast accepts moved hint word {target}"
        );
    });

    let mut start = 0;
    for len in native.lengths {
        if len > 0 {
            for target in [start, start + len - 1] {
                assert_rejected(&honest, Some(target), &format!("hint word {target}"));
            }
        }
        start += len;
    }
}
