//! Lean `Lifecycle.BoundedDepth.ApplicationIdentity` and
//! `Export.EncodingParity.builtApplication`; SPEC.md section 4.4.

use neo_ccs::crypto::poseidon2::poseidon2_hash;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::Value;

use super::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder, ApplicationCircuit};

const TAG: usize = 61;
const FRONT: usize = TAG + 4;

fn built_application() -> ApplicationCircuit {
    let mut builder = ApplicationBuilder::new(2).expect("builder");
    let [x0, x1, x2, x3] = builder.input_state().map(Affine::from);
    let [w0, w1] = [0, 1].map(|word| Affine::from(builder.private_inputs()[word]));
    let constant = |value| Affine::constant(Fq::from_u64(value));
    let product = builder
        .multiply(x0 + w0 * Fq::from_u64(3), w1 + constant(5))
        .expect("product");
    let sum = builder
        .affine(Affine::from(product) + x1.clone())
        .expect("sum");
    builder.assert_equal(x2.clone(), x3).expect("equality");
    builder
        .finish([sum.into(), x1, x2 + constant(7), constant(9)])
        .expect("application")
}

fn identity_words(application: &ApplicationCircuit) -> Vec<u64> {
    crate::artifacts::embedded()
        .packages
        .application_identity_words(
            application.records(),
            application.private_input_count(),
            application.shape(),
        )
        .expect("identity words")
}

fn counted(words: &[u64], entry: impl Fn(&[u64]) -> &[u64]) -> &[u64] {
    (0..words[0]).fold(&words[1..], |rest, _| entry(rest))
}

fn skip<const WORDS: usize>(words: &[u64]) -> &[u64] {
    &words[WORDS..]
}

/// Lean `Linear.words`.
fn linear(words: &[u64]) -> &[u64] {
    counted(&words[1..], skip::<2>)
}

/// Lean `Row.words`.
fn row(words: &[u64]) -> &[u64] {
    linear(linear(linear(words)))
}

/// Lean `indexedRow`.
fn indexed(words: &[u64]) -> &[u64] {
    row(&words[1..])
}

/// Lean `Layout.words`.
fn layout(words: &[u64]) -> &[u64] {
    let fusion = counted(words, |group| {
        row(counted(counted(counted(group, skip::<2>), indexed), indexed))
    });
    let packing = counted(fusion, |group| {
        counted(linear(counted(counted(group, indexed), indexed)), skip::<4>)
    });
    let removed = counted(counted(&packing[1..], skip::<1>), skip::<1>);
    counted(counted(counted(removed, skip::<4>), skip::<5>), skip::<3>)
}

/// Lean `RecordRow.words`.
fn record_row(words: &[u64]) -> &[u64] {
    &words[6 + 2 * (words[1] + words[3] + words[5]) as usize..]
}

/// Lean `Block.words`.
fn block(words: &[u64]) -> &[u64] {
    (0..16).fold(&words[1..], |rest, _| linear(rest))
}

/// Lean `Application.words`.
fn after_application(words: &[u64]) -> &[u64] {
    let rows = counted(&words[FRONT + 1..], record_row);
    layout(layout(counted(counted(rows, skip::<1>), block)))
}

fn hash(words: &[u64]) -> [u64; 4] {
    let input = words.iter().map(|&word| Fq::from_u64(word));
    poseidon2_hash(&input.collect::<Vec<_>>()).map(|word| word.as_canonical_u64())
}

#[test]
fn a_built_application_has_the_lean_words_and_their_hash() {
    let file: Value = serde_json::from_str(include_str!(
        "../../../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json"
    ))
    .expect("Lean encoding file");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    let sample = file[4]
        .as_array()
        .and_then(|samples| samples.last())
        .expect("Lean's built application");
    let lean = sample[1].as_array().expect("words").iter();
    let lean = lean
        .map(|word| word.as_u64().expect("u64"))
        .collect::<Vec<_>>();
    let records = lean.len() - 16;

    let application = built_application();
    let words = identity_words(&application);
    let packages = &crate::artifacts::embedded().packages;
    assert_eq!(words[..TAG], lean[..TAG]);
    assert_eq!(words[TAG..FRONT], packages.identity());
    assert_eq!(words[FRONT..records], lean[FRONT..records]);
    assert!(layout(layout(&words[records..])).is_empty());
    assert!(after_application(&words).is_empty());

    let circuit = crate::Circuit::compile(application).expect("circuit");
    assert_eq!(circuit.identity(), hash(&words));
}

#[test]
fn the_words_of_an_application_with_blocks_have_the_lean_form() {
    let application = poseidon2_hash_chain_v1().expect("hash chain");
    assert!(!application.shape().blocks.is_empty());
    assert!(after_application(&identity_words(&application)).is_empty());
}
