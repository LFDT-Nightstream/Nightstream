//! The circuit core: `Native` equals Plonky3, recorded rows hold for honest
//! values, break under mutation, and do not depend on the values.

use p3_challenger_v08::{CanObserve, CanSample, CanSampleBits, FieldChallenger};
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric_v08::Permutation;

use neo_math::{cf_inv, D, F};
use p3_field::PrimeCharacteristicRing as _;
use p3_field::PrimeField64 as _;

use super::word;
use crate::circuit::block::{Kind, KINDS};
use crate::circuit::fold::FoldTranscript;
use crate::circuit::hash::{compress, hash_leaf, merkle_root, Duplex};
use crate::circuit::poseidon2::{self, OUTPUT};
use crate::circuit::record::{Recorder, Trace};
use crate::circuit::ring_mul;
use crate::circuit::{algebra, Backend, Native};
use crate::field::{Ext, Gl};
use crate::hash::permutation;
use crate::Error;

fn gl(seed: u64, index: u64) -> Gl {
    Gl::from_u64(word(seed, index))
}

#[test]
fn permutation_block_matches_p3() {
    for seed in 0..16 {
        let input: [Gl; 16] = std::array::from_fn(|i| gl(seed, i as u64));
        let cells = poseidon2::trace(&input);
        assert_eq!(cells[OUTPUT..], permutation().permute(input)[..], "input {seed}");
    }
    let trace = Trace {
        blocks: [poseidon2::trace(&std::array::from_fn(|i| gl(3, i as u64))), Vec::new()],
        ..Default::default()
    };
    assert_eq!(trace.block_failure(Kind::Permutation, 0), None);
    for cell in [0, 16, 100, 165, 170] {
        let mut mutated = Trace {
            blocks: trace.blocks.clone(),
            ..Default::default()
        };
        mutated.blocks[0][cell] += Gl::ONE;
        assert!(mutated.block_failure(Kind::Permutation, 0).is_some(), "cell {cell}");
    }
}

#[test]
fn ring_product_block_matches_the_ring() {
    for seed in 0..4 {
        let a: [Gl; D] = std::array::from_fn(|i| gl(40 + seed, i as u64));
        let b: [Gl; D] = std::array::from_fn(|i| gl(50 + seed, i as u64));
        let to_f = |values: &[Gl; D]| values.map(|v| F::from_u64(v.as_canonical_u64()));
        let expected = cf_inv(to_f(&a)).mul(&cf_inv(to_f(&b))).0;
        let product = ring_mul::multiply(&a, &b);
        assert_eq!(
            product.map(|v| v.as_canonical_u64()),
            expected.map(|v| v.as_canonical_u64()),
            "seed {seed}"
        );
        let cells = Kind::RingProduct.trace(&[a.as_slice(), b.as_slice()].concat());
        assert_eq!(cells[Kind::RingProduct.outputs()], product[..]);
        assert_eq!(Kind::RingProduct.failure(&cells), None);
        for cell in [0, 60, 130, 220, 268] {
            let mut mutated = cells.clone();
            mutated[cell] += Gl::ONE;
            assert!(Kind::RingProduct.failure(&mutated).is_some(), "cell {cell}");
        }
    }
    // The recorded product equals the native one and its rows hold.
    let a: [Gl; D] = std::array::from_fn(|i| gl(60, i as u64));
    let b: [Gl; D] = std::array::from_fn(|i| gl(61, i as u64));
    let mut recorder = Recorder::new(Trace::default());
    let (x, y) = (
        a.map(|v| algebra::private(&mut recorder, v)),
        b.map(|v| algebra::private(&mut recorder, v)),
    );
    let outputs = recorder.ring_mul(&x, &y);
    let (trace, failure) = recorder.finish();
    assert_eq!(failure, None);
    assert_eq!(outputs.map(|form| form.value()), Native.ring_mul(&a, &b));
    assert_eq!(trace.failing_block(), None);
    assert!((0..trace.rows.len()).all(|r| trace.row_value(r) == Gl::ZERO));
}

#[test]
fn fold_transcript_matches_neo_transcript() {
    let b = &mut Native;
    let mut theirs = neo_transcript::Poseidon2Transcript::new_v1_1();
    let mut ours = FoldTranscript::new(b);
    for (step, length) in [0usize, 1, 11, 12, 13, 30].into_iter().enumerate() {
        let words: Vec<u64> = (0..length as u64)
            .map(|i| word(70 + step as u64, i) % F::ORDER_U64)
            .collect();
        theirs.absorb_v1_1(&words.iter().map(|&w| F::from_u64(w)).collect::<Vec<_>>());
        ours.absorb_constants(b, &words);
        for pair in 0..6 {
            let expected = theirs.read_pair_v1_1(pair).map(|v| v.as_canonical_u64());
            assert_eq!(
                ours.read_pair(pair).map(|v| v.as_canonical_u64()),
                expected,
                "step {step} pair {pair}"
            );
        }
        if step % 2 == 1 {
            let expected = theirs.squeeze_digest_v1_1().map(|v| v.as_canonical_u64());
            assert_eq!(
                ours.squeeze_digest(b).map(|v| v.as_canonical_u64()),
                expected,
                "step {step}"
            );
        }
    }
    assert_eq!(
        ours.handoff(b).map(|v| v.as_canonical_u64()),
        crate::hash::seed(theirs).map(|v| v.as_canonical_u64())
    );
}

#[test]
fn native_duplex_matches_p3_challenger() {
    let mut ours = Duplex::new(&mut Native);
    let mut fresh = crate::hash::Challenger::new(permutation().clone());
    for step in 0..200u64 {
        match word(5, step) % 5 {
            0 | 1 => {
                let value = gl(6, step);
                ours.observe(&mut Native, value);
                fresh.observe(value);
            }
            2 => assert_eq!(
                ours.sample(&mut Native),
                CanSample::<Gl>::sample(&mut fresh),
                "step {step}"
            ),
            3 => {
                let ext: Ext = fresh.sample_algebra_element();
                assert_eq!(ours.sample_ext(&mut Native), ext, "step {step}");
            }
            _ => {
                let bits = 1 + (word(7, step) % 30) as usize;
                let expected = fresh.sample_bits(bits);
                let got = ours.sample_bits(&mut Native, bits).unwrap();
                let value = got
                    .iter()
                    .rev()
                    .fold(0usize, |acc, bit| 2 * acc + (*bit == Gl::ONE) as usize);
                assert_eq!(value, expected, "step {step}");
            }
        }
    }
}

/// A program that exercises every primitive. Returns its outputs.
fn program<B: Backend>(b: &mut B, words: &[Gl]) -> Result<Vec<B::F>, Error> {
    let inputs: Vec<B::F> = words.iter().map(|&w| algebra::private(b, w)).collect();
    let statement = b.public(77);
    let mut duplex = Duplex::new(b);
    duplex.observe_slice(b, &inputs[..10]);
    duplex.observe(b, statement);
    let x = duplex.sample_ext(b);
    let y = b.ext(std::array::from_fn(|i| inputs[10 + i]));
    let scaled = b.ext_scale(y, inputs[46]);
    let sum = b.ext_add(x, scaled);
    duplex.observe_ext(b, sum);
    let product = b.ext_mul(sum, y);
    let inverse = b.ext_inverse(product, "inverse")?;
    let one = b.ext_mul(product, inverse);
    let unit = algebra::ext_one(b);
    b.assert_ext_equal(one, unit, "inverse times value")?;
    let bits = duplex.sample_bits(b, 12)?;
    let leaf = hash_leaf(b, &inputs[13..30]);
    let siblings: Vec<[B::F; 4]> = (0..12)
        .map(|level| std::array::from_fn(|i| inputs[30 + 4 * (level % 4) + i]))
        .collect();
    let root = merkle_root(b, leaf, &bits, &siblings);
    let square = b.mul(inputs[0], inputs[1]);
    let hinted = b.hint(&[square], 1, &|values| {
        let value = Gl::from_u64(values[0]);
        vec![(value * value).as_canonical_u64()]
    });
    let check = b.mul(square, square);
    b.assert_equal(hinted[0], check, "hint")?;
    let mut outputs = b.coordinates(one).to_vec();
    outputs.extend(root);
    outputs.extend(b.coordinates(inverse));
    outputs.push(hinted[0]);
    Ok(outputs)
}

/// Exactly the words `program` uses, so every input cell is constrained.
fn words(seed: u64) -> Vec<Gl> {
    (0..47).map(|i| gl(seed, i)).collect()
}

#[test]
fn recorded_program_is_satisfied_and_matches_native() {
    let expected = program(&mut Native, &words(1)).unwrap();
    let mut recorder = Recorder::new(Trace::default());
    let outputs = program(&mut recorder, &words(1)).unwrap();
    let (trace, failure) = recorder.finish();
    assert_eq!(failure, None);
    let values: Vec<Gl> = outputs.iter().map(|form| form.value()).collect();
    assert_eq!(values, expected);
    for r in 0..trace.rows.len() {
        assert_eq!(trace.row_value(r), Gl::ZERO, "row {r}");
    }
    assert_eq!(trace.failing_block(), None);
}

#[test]
fn rows_do_not_depend_on_values() {
    let record = |words: &[Gl]| {
        let mut recorder = Recorder::new(Trace::default());
        let _ = program(&mut recorder, words);
        recorder.finish().0
    };
    let honest = record(&words(2));
    let zero = record(&vec![Gl::ZERO; 47]);
    assert_eq!(honest.rows, zero.rows);
    assert_eq!(honest.entries, zero.entries);
    assert_eq!(honest.glue.len(), zero.glue.len());
    assert_eq!(KINDS.map(|kind| honest.count(kind)), KINDS.map(|kind| zero.count(kind)));
}

#[test]
fn every_mutated_cell_breaks_a_row() {
    let mut recorder = Recorder::new(Trace::default());
    program(&mut recorder, &words(3)).unwrap();
    let (trace, _) = recorder.finish();
    let broken = |trace: &Trace| {
        (0..trace.rows.len()).any(|r| trace.row_value(r) != Gl::ZERO) || trace.failing_block().is_some()
    };
    assert!(!broken(&trace));
    for cell in 0..trace.glue.len() {
        let mut mutated = Trace {
            glue: trace.glue.clone(),
            blocks: trace.blocks.clone(),
            public: trace.public.clone(),
            entries: trace.entries.clone(),
            rows: trace.rows.clone(),
        };
        mutated.glue[cell] += Gl::ONE;
        assert!(broken(&mutated), "glue cell {cell}");
    }
    for cell in (0..trace.blocks[0].len()).step_by(37) {
        let mut mutated = Trace {
            glue: trace.glue.clone(),
            blocks: trace.blocks.clone(),
            public: trace.public.clone(),
            entries: trace.entries.clone(),
            rows: trace.rows.clone(),
        };
        mutated.blocks[0][cell] += Gl::ONE;
        assert!(broken(&mutated), "block cell {cell}");
    }
}

#[test]
fn a_non_canonical_bit_split_breaks_a_row() {
    // 5 has a second 64-bit representation, 5 + p. Record the honest split,
    // then replace the bits by those of 5 + p.
    let mut recorder = Recorder::new(Trace::default());
    let five = recorder.private(5);
    recorder.bits(five, 64, "bits").unwrap();
    let (mut trace, failure) = recorder.finish();
    assert_eq!(failure, None);
    let forged = 5u64 + 0xFFFF_FFFF_0000_0001;
    for i in 0..64 {
        trace.glue[1 + i] = Gl::from_u64((forged >> i) & 1);
    }
    let failing: Vec<usize> = (0..trace.rows.len())
        .filter(|&r| trace.row_value(r) != Gl::ZERO)
        .collect();
    assert!(!failing.is_empty());
    // The recomposition alone holds (5 + p = 5 mod p): the canonical row fails.
    assert!(failing.iter().all(|&r| r > 64));
}

#[test]
fn recorder_notes_failed_checks_and_keeps_going() {
    let mut recorder = Recorder::new(Trace::default());
    let a = recorder.private(3);
    recorder.assert_zero(a, "first").unwrap();
    let x = recorder.ext([a, a, a]);
    let zero = recorder.ext_sub(x, x);
    recorder.ext_inverse(zero, "second").unwrap();
    recorder.mul(a, a);
    let (trace, failure) = recorder.finish();
    assert_eq!(failure, Some("first"));
    assert_eq!(trace.rows.len(), 2, "rows continue after a failed check");
    assert!(Native.assert_zero(Gl::from_u64(3), "first").is_err());
    assert!(Native.ext_inverse(Ext::ZERO, "zero").is_err());
}

#[test]
fn compress_and_leaf_hash_match_the_layer_one_hash() {
    let values: Vec<Gl> = (0..29).map(|i| gl(9, i)).collect();
    assert_eq!(hash_leaf(&mut Native, &values), crate::hash::hash_leaf(&values));
    let left = std::array::from_fn(|i| gl(10, i as u64));
    let right = std::array::from_fn(|i| gl(11, i as u64));
    assert_eq!(compress(&mut Native, left, right), crate::hash::compress(left, right));
    let ext = Ext::from_basis_coefficients_fn(|i| gl(12, i as u64));
    assert_eq!(Native.ext_inverse(ext, "x").unwrap(), ext.inverse());
}
