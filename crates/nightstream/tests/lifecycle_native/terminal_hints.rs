//! The hint words of the terminal wrapper: the iteration's inverse in
//! `FinalProgram::run` and the computed columns in `Package::apply`. The
//! shrink prover chooses every hint word. A word must be bound by a later
//! check, or be a package column that layer 1 does not read: a move of such
//! a column changes no output, and the rows, whose meaning Lean proves,
//! still decide the run.

use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::F;
use neo_spartan::shrink::Program;
use neo_spartan::{Backend, Error, Native, R1cs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{package, Package, Source};
use crate::lifecycle::{CompressionKey, Stage1State};

fn manifest(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// The check that rejects `result`, or `None` when it holds.
fn rejection(result: Result<(), Error>) -> Option<&'static str> {
    match result {
        Ok(()) => None,
        Err(Error::Rejected(what)) => Some(what),
        Err(error) => panic!("not a rejection: {error:?}"),
    }
}

/// The hint words, with their columns, that no row fixes by its terms
/// alone. A row fixes a word when the word's column has a nonzero net
/// coefficient on C and is absent from A and B, or has one on A or B while
/// the other product side is a nonzero constant and the column is absent
/// from the rest. A move by one then changes that row by a nonzero constant,
/// so the row fails on every witness that the rows accept.
fn unfixed_words(package: &Package) -> Vec<(usize, usize)> {
    let constant = |side: &[(usize, u64)]| {
        !side.is_empty()
            && side
                .iter()
                .all(|&(column, _)| matches!(package.sources[column], Source::One))
            && side.iter().map(|&(_, k)| F::from_u64(k)).sum::<F>() != F::ZERO
    };
    let mut fixed = vec![false; package.sources.len()];
    for row in 0..package.system.rows() {
        let [a, b, c] = package.system.row(row);
        let mut fix = |side: &[(usize, u64)], others: [&[(usize, u64)]; 2]| {
            for column in alone(side, others) {
                fixed[column] = true;
            }
        };
        fix(&c, [&a, &b]);
        if constant(&b) {
            fix(&a, [&b, &c]);
        }
        if constant(&a) {
            fix(&b, [&a, &c]);
        }
    }
    package
        .sources
        .iter()
        .enumerate()
        .filter_map(|(column, source)| match *source {
            Source::Computed(word) if !fixed[column] => Some((word, column)),
            _ => None,
        })
        .collect()
}

/// The columns of `side` with a nonzero net coefficient that `others` do
/// not name.
fn alone(side: &[(usize, u64)], others: [&[(usize, u64)]; 2]) -> Vec<usize> {
    let mut terms = side.to_vec();
    terms.sort_unstable();
    terms
        .chunk_by(|x, y| x.0 == y.0)
        .filter(|run| run.iter().map(|&(_, k)| F::from_u64(k)).sum::<F>() != F::ZERO)
        .map(|run| run[0].0)
        .filter(|&column| {
            others
                .iter()
                .all(|other| other.iter().all(|&(c, _)| c != column))
        })
        .collect()
}

/// `Native` with word `word` of hint call `call` moved by one. A shrink
/// prover chooses every hint word.
struct Moved {
    call: usize,
    word: usize,
    calls: usize,
}

impl Moved {
    fn new(call: usize, word: usize) -> Self {
        Self { call, word, calls: 0 }
    }
}

impl Backend for Moved {
    type F = <Native as Backend>::F;
    type E = <Native as Backend>::E;

    fn constant(&mut self, value: u64) -> Self::F {
        Native.constant(value)
    }

    fn private(&mut self, value: u64) -> Self::F {
        Native.private(value)
    }

    fn public(&mut self, value: u64) -> Self::F {
        Native.public(value)
    }

    fn add(&mut self, a: Self::F, b: Self::F) -> Self::F {
        Native.add(a, b)
    }

    fn sub(&mut self, a: Self::F, b: Self::F) -> Self::F {
        Native.sub(a, b)
    }

    fn scale(&mut self, a: Self::F, by: u64) -> Self::F {
        Native.scale(a, by)
    }

    fn mul(&mut self, a: Self::F, b: Self::F) -> Self::F {
        Native.mul(a, b)
    }

    fn ext(&mut self, coordinates: [Self::F; 3]) -> Self::E {
        Native.ext(coordinates)
    }

    fn coordinates(&mut self, a: Self::E) -> [Self::F; 3] {
        Native.coordinates(a)
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> Self::E {
        Native.ext_constant(value)
    }

    fn ext_add(&mut self, a: Self::E, b: Self::E) -> Self::E {
        Native.ext_add(a, b)
    }

    fn ext_sub(&mut self, a: Self::E, b: Self::E) -> Self::E {
        Native.ext_sub(a, b)
    }

    fn ext_mul(&mut self, a: Self::E, b: Self::E) -> Self::E {
        Native.ext_mul(a, b)
    }

    fn ext_scale(&mut self, a: Self::E, by: Self::F) -> Self::E {
        Native.ext_scale(a, by)
    }

    fn ext_inverse(&mut self, a: Self::E, what: &'static str) -> Result<Self::E, Error> {
        Native.ext_inverse(a, what)
    }

    fn assert_zero(&mut self, a: Self::F, what: &'static str) -> Result<(), Error> {
        Native.assert_zero(a, what)
    }

    fn bits(&mut self, a: Self::F, count: usize, what: &'static str) -> Result<Vec<Self::F>, Error> {
        Native.bits(a, count, what)
    }

    fn select(&mut self, bit: Self::F, zero: Self::F, one: Self::F) -> Self::F {
        Native.select(bit, zero, one)
    }

    fn permute(&mut self, state: [Self::F; WIDTH]) -> [Self::F; WIDTH] {
        Native.permute(state)
    }

    fn hint(&mut self, inputs: &[Self::F], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Self::F> {
        let moved = (self.calls == self.call).then_some(self.word);
        self.calls += 1;
        Native.hint(inputs, len, &|values| {
            let mut words = compute(values);
            if let Some(k) = moved {
                words[k] = (F::from_u64(words[k]) + F::ONE).as_canonical_u64();
            }
            words
        })
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Self::F], what: &'static str) -> Result<(), Error> {
        Native.r1cs(system, columns, what)
    }
}

/// The iteration's inverse, the one word of the wrapper's first hint call,
/// is bound by `positive iteration`: the honest run passes that check (a
/// shape run then fails at the package rows), and a moved inverse fails it.
#[test]
fn the_iteration_inverse_is_bound() {
    let key =
        CompressionKey::from_bytes(&std::fs::read(manifest("tests/fixtures/compression/key.bin")).unwrap()).unwrap();
    let relation = key.relation().unwrap();
    let state = Stage1State::new(3, [5, 7, 11, 13].map(F::from_u64), [17, 19, 23, 29].map(F::from_u64));
    let program = key.program(&relation, package().unwrap(), &state, None);
    assert_ne!(rejection(program.run(&mut Native)), Some("positive iteration"));
    assert_eq!(
        rejection(program.run(&mut Moved::new(0, 0))),
        Some("positive iteration")
    );
}

/// Every package hint word that layer 1 reads is fixed by its rows' terms
/// alone, so a move of it fails on every witness that the rows accept. The
/// other words that need values to be fixed are not outputs. A move of one
/// changes only its own column, so `apply` hands layer 1 the same claim and
/// transcript, and `Native` can accept the move: such a word is free on that
/// witness.
#[test]
fn every_output_hint_word_is_fixed_by_its_rows() {
    let package = package().unwrap();
    let outputs = &package.outputs;
    let read: Vec<usize> = [
        &outputs.commitment,
        &outputs.public_input,
        &outputs.point,
        &outputs.eval_k,
        &outputs.eval_a,
        &outputs.state,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect();
    let unfixed = unfixed_words(package);
    eprintln!(
        "package hint: {} words, {} not fixed by their rows' terms alone",
        package.computed.len(),
        unfixed.len()
    );
    for (word, column) in unfixed {
        assert!(
            !read.contains(&column),
            "output hint word {word} is not fixed by its rows"
        );
    }
}
