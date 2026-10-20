//! The Lean terminal package (`formal/compression-lean`) against the native
//! last fold: its witness program accepts the honest inputs of a production
//! fold, so every terminal row holds, and its output columns hold the native
//! PiRLC parent and the transcript state after PiRLC.

use super::*;
use crate::application::poseidon2_hash_chain_v1;
use crate::lifecycle::TERMINAL_PACKAGE_IDENTIFIER;
use neo_math::KExtensions;
use nightstream_fprime::{load_terminal_package, PackageError};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::{fs, path::PathBuf, time::Instant};

fn read(path: &str) -> Vec<u8> {
    fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()
}

/// The embedded package loads: the artifact has the pinned identifier, its
/// first caller inputs are the prior state preimage, and its rows build.
#[test]
fn terminal_package_loads() {
    crate::lifecycle::terminal_package().unwrap();
}

fn load_error(bytes: &[u8], identifier: [u64; 4]) -> PackageError {
    load_terminal_package(bytes, identifier).err().unwrap()
}

/// A space after the first bracket leaves the JSON value the same, so only
/// the canonical-bytes check rejects it.
#[test]
fn terminal_package_rejects_a_non_canonical_byte() {
    let mut bytes = read("artifacts/compression-terminal-v1.json");
    bytes.insert(1, b' ');
    assert!(matches!(
        load_error(&bytes, TERMINAL_PACKAGE_IDENTIFIER),
        PackageError::NonCanonicalBytes
    ));
}

/// Another last digit keeps the bytes canonical and changes one number.
#[test]
fn terminal_package_rejects_a_changed_number() {
    let mut bytes = read("artifacts/compression-terminal-v1.json");
    let digit = bytes.iter().rposition(u8::is_ascii_digit).unwrap();
    bytes[digit] = if bytes[digit] == b'9' { b'0' } else { bytes[digit] + 1 };
    match load_error(&bytes, TERMINAL_PACKAGE_IDENTIFIER) {
        PackageError::ExpectedIdentityMismatch { expected, computed } => {
            assert_eq!(expected, TERMINAL_PACKAGE_IDENTIFIER);
            assert_ne!(computed, TERMINAL_PACKAGE_IDENTIFIER);
        }
        error => panic!("unexpected error: {error}"),
    }
}

#[test]
fn terminal_package_rejects_another_identifier() {
    let mut expected = TERMINAL_PACKAGE_IDENTIFIER;
    expected[3] ^= 1;
    match load_error(&read("artifacts/compression-terminal-v1.json"), expected) {
        PackageError::ExpectedIdentityMismatch {
            expected: reported,
            computed,
        } => {
            assert_eq!(reported, expected);
            assert_eq!(computed, TERMINAL_PACKAGE_IDENTIFIER);
        }
        error => panic!("unexpected error: {error}"),
    }
}

#[test]
#[ignore = "Production base step, fold and terminal package; run separately under the 300-second cap."]
fn terminal_package_accepts_the_last_fold() {
    let circuit = Circuit::compile(
        &read("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
        poseidon2_hash_chain_v1().unwrap(),
    )
    .unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);
    let proof = prover
        .extend(&prover.prove(initial, &message).unwrap(), &message)
        .unwrap();
    let last = prover.lifecycle.last_fold(&proof).unwrap();
    let inputs = prover.lifecycle.terminal_inputs(&last).unwrap();

    let started = Instant::now();
    let (package, outputs) = load_terminal_package(
        &read("artifacts/compression-terminal-v1.json"),
        TERMINAL_PACKAGE_IDENTIFIER,
    )
    .unwrap();
    eprintln!("terminal package load elapsed={:?}", started.elapsed());
    let encoded = package.encode_terminal_inputs(&inputs).unwrap();
    let started = Instant::now();
    let assignment = package
        .execute_witness(encoded.private_values(), encoded.public_values())
        .unwrap();
    eprintln!("terminal package witness elapsed={:?}", started.elapsed());

    let private = assignment.private_values();
    let public = assignment.public_values();
    let words = |columns: &[usize]| -> Vec<u64> {
        columns
            .iter()
            .map(|&column| {
                if column < private.len() {
                    private[column]
                } else {
                    public[column - private.len() - 1]
                }
            })
            .collect()
    };
    let canonical = |values: &[F]| -> Vec<u64> {
        values
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect()
    };
    let extension = |values: &[neo_math::K]| -> Vec<u64> {
        values
            .iter()
            .flat_map(|value| value.as_coeffs().map(|word| word.as_canonical_u64()))
            .collect()
    };
    let parent = &last.parent.claim;
    assert_eq!(words(&outputs.commitment), canonical(&parent.c.data), "commitment");
    let x: Vec<F> = (0..outputs.public_input.len())
        .map(|column| parent.X[(column % neo_math::D, column / neo_math::D)])
        .collect();
    assert_eq!(words(&outputs.public_input), canonical(&x), "public input");
    assert_eq!(words(&outputs.point), extension(&parent.r), "point");
    assert_eq!(
        words(&outputs.eval_k),
        extension(&parent.eval_k[..neo_math::D]),
        "Eval_K"
    );
    let eval_a: Vec<neo_math::K> = parent
        .eval_a
        .iter()
        .flat_map(|matrix| matrix[..neo_math::D].iter().copied())
        .collect();
    assert_eq!(words(&outputs.eval_a), extension(&eval_a), "Eval_A");
    let transcript = last.transcript.into_inner();
    assert_eq!(transcript.absorbed(), 0);
    assert_eq!(
        words(&outputs.state),
        canonical(&transcript.state()),
        "transcript state"
    );
}
