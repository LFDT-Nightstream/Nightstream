//! Complete standalone pilot parity against the Lean-emitted nonzero fixture.

use std::{fs, path::PathBuf};

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use nightstream_fprime::PI_CCS_V1_1_ROUND_COUNT;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
const STATE_PREIMAGE_WORDS: usize = 45_933;
const PRIOR_PUBLIC_WORDS: usize = 270;
const DIGEST_WORDS: usize = 4;
const PUBLIC_WORDS: usize = PRIOR_PUBLIC_WORDS + DIGEST_WORDS;
const RUNNING_COUNT: usize = 16;
const MATRIX_COUNT: usize = 14;
const RUNNING_GROUP_WORDS: usize = 2_865;
const RUNNING_POINT_WORDS: usize = 2 * PI_CCS_V1_1_ROUND_COUNT;

#[derive(Clone, Deserialize)]
struct RawInput(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>);

#[derive(Deserialize)]
struct RawResult(Vec<u64>, Vec<u64>, Vec<u64>, Vec<[u64; 3]>, Vec<u64>);

#[derive(Deserialize)]
struct RawParity(u64, RawInput, RawResult);

#[derive(Debug, PartialEq, Eq)]
struct PilotResult {
    prior_digest: [u64; DIGEST_WORDS],
    output_digest: [u64; DIGEST_WORDS],
    public_values: Vec<u64>,
    public_segments: Vec<[u64; 3]>,
    assurance_flags: Vec<u64>,
}

fn parity_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-pilot-parity-v1.json")
}

fn parity() -> RawParity {
    serde_json::from_slice(&fs::read(parity_path()).expect("Lean pilot parity bytes")).expect("Lean pilot parity JSON")
}

fn changed_word(word: u64) -> u64 {
    if word + 1 == GOLDILOCKS_MODULUS {
        0
    } else {
        word + 1
    }
}

fn canonical_words(words: &[u64]) -> bool {
    words.iter().all(|word| *word < GOLDILOCKS_MODULUS)
}

fn hash_preimage(words: &[u64]) -> Result<[u64; DIGEST_WORDS], &'static str> {
    if words.len() != STATE_PREIMAGE_WORDS {
        return Err("state preimage length");
    }
    if !canonical_words(words) {
        return Err("noncanonical state preimage word");
    }

    let fields: Vec<Goldilocks> = words.iter().copied().map(Goldilocks::from_u64).collect();
    Ok(poseidon2_hash(&fields).map(|word| word.as_canonical_u64()))
}

fn verify_pilot(input: &RawInput) -> Result<PilotResult, &'static str> {
    let RawInput(prior_preimage, prior_public, output_preimage, claimed_output_digest) = input;
    if prior_public.len() != PRIOR_PUBLIC_WORDS {
        return Err("prior public-input length");
    }
    if claimed_output_digest.len() != DIGEST_WORDS {
        return Err("output digest length");
    }
    if !canonical_words(prior_public) || !canonical_words(claimed_output_digest) {
        return Err("noncanonical public word");
    }

    let prior_digest = hash_preimage(prior_preimage)?;
    let output_digest = hash_preimage(output_preimage)?;

    let mut expected_prior_public = vec![0; PRIOR_PUBLIC_WORDS];
    expected_prior_public[0] = 1;
    for (word, value) in prior_digest.iter().copied().enumerate() {
        for bit in 0..64 {
            expected_prior_public[1 + word * 64 + bit] = (value >> bit) & 1;
        }
    }
    if *prior_public != expected_prior_public {
        return Err("prior public-input binding");
    }
    if claimed_output_digest.as_slice() != output_digest {
        return Err("output digest binding");
    }

    let mut public_values = prior_public.clone();
    public_values.extend_from_slice(claimed_output_digest);
    Ok(PilotResult {
        prior_digest,
        output_digest,
        public_values,
        public_segments: vec![[4, 0, 270], [5, 270, 4]],
        assurance_flags: vec![
            u64::from(prior_digest.len() == DIGEST_WORDS),
            u64::from(output_digest.len() == DIGEST_WORDS),
            u64::from(prior_digest != output_digest),
            u64::from(PUBLIC_WORDS == 274),
        ],
    })
}

fn lean_result(raw: RawResult) -> PilotResult {
    PilotResult {
        prior_digest: raw.0.try_into().expect("four-word Lean prior digest"),
        output_digest: raw.1.try_into().expect("four-word Lean output digest"),
        public_values: raw.2,
        public_segments: raw.3,
        assurance_flags: raw.4,
    }
}

fn expect_prefix(words: &[u64], cursor: &mut usize, expected: usize, label: &str) {
    assert_eq!(words[*cursor], expected as u64, "{label} length prefix");
    *cursor += 1;
}

fn fixture_mutation_indices(words: &[u64]) -> Vec<(String, usize)> {
    assert_eq!(words.len(), STATE_PREIMAGE_WORDS);
    let domain_tag: Vec<u64> = b"HyperNova/NIVC/state/v1"
        .iter()
        .copied()
        .map(u64::from)
        .collect();
    assert_eq!(&words[..domain_tag.len()], domain_tag);

    let mut indices = Vec::new();
    indices.extend((0..domain_tag.len()).map(|index| (format!("domain tag word {index}"), index)));

    let mut cursor = domain_tag.len();
    indices.push(("verifier-key length prefix".into(), cursor));
    expect_prefix(words, &mut cursor, 4, "verifier key");
    for lane in 0..4 {
        indices.push((format!("verifier-key lane {lane}"), cursor + lane));
    }
    cursor += 4;

    indices.push(("iteration".into(), cursor));
    cursor += 1;

    indices.push(("initial-state length prefix".into(), cursor));
    expect_prefix(words, &mut cursor, 4, "initial state");
    for lane in 0..4 {
        indices.push((format!("initial-state lane {lane}"), cursor + lane));
    }
    cursor += 4;

    indices.push(("current-state length prefix".into(), cursor));
    expect_prefix(words, &mut cursor, 4, "current state");
    for lane in 0..4 {
        indices.push((format!("current-state lane {lane}"), cursor + lane));
    }
    cursor += 4;

    assert_eq!(cursor, 39, "running-state start");
    indices.push(("running-point length prefix".into(), cursor));
    expect_prefix(words, &mut cursor, RUNNING_POINT_WORDS, "running point");
    for component in 0..RUNNING_POINT_WORDS {
        indices.push((format!("running-point component {component}"), cursor + component));
    }
    cursor += RUNNING_POINT_WORDS;

    for source in 0..RUNNING_COUNT {
        let group_start = cursor;

        indices.push((format!("source {source} commitment length prefix"), cursor));
        expect_prefix(words, &mut cursor, 972, "running commitment");
        indices.push((format!("source {source} commitment first word"), cursor));
        indices.push((format!("source {source} commitment last word"), cursor + 971));
        cursor += 972;

        indices.push((format!("source {source} public-input length prefix"), cursor));
        expect_prefix(words, &mut cursor, 270, "running public input");
        indices.push((format!("source {source} public-input first word"), cursor));
        indices.push((format!("source {source} public-input last word"), cursor + 269));
        cursor += 270;

        indices.push((format!("source {source} evaluation length prefix"), cursor));
        expect_prefix(words, &mut cursor, 1_620, "running evaluation");
        indices.push((format!("source {source} Eval_K first word"), cursor));
        indices.push((format!("source {source} Eval_K last word"), cursor + 107));
        cursor += 108;
        for matrix in 0..MATRIX_COUNT {
            indices.push((format!("source {source} Eval_A matrix {matrix}"), cursor + matrix * 108));
        }
        cursor += MATRIX_COUNT * 108;

        assert_eq!(cursor - group_start, RUNNING_GROUP_WORDS);
    }

    indices.push(("program counter".into(), cursor));
    cursor += 1;
    assert_eq!(cursor, STATE_PREIMAGE_WORDS, "complete preimage parse");
    indices
}

#[test]
fn rust_recomputation_matches_complete_lean_pilot_result() {
    let RawParity(schema, input, expected) = parity();
    assert_eq!(schema, 1, "pilot parity schema");

    let actual = verify_pilot(&input).expect("valid nonzero pilot fixture");
    assert_eq!(actual, lean_result(expected));
    assert_ne!(actual.prior_digest, actual.output_digest);
    assert_eq!(actual.public_values.len(), PUBLIC_WORDS);
    assert_eq!(actual.public_values[0], 1);
    for (word, value) in actual.prior_digest.iter().copied().enumerate() {
        for bit in 0..64 {
            assert_eq!(actual.public_values[1 + word * 64 + bit], (value >> bit) & 1);
        }
    }
    assert!(actual.public_values[257..270].iter().all(|word| *word == 0));
    assert_eq!(&actual.public_values[270..274], actual.output_digest);
}

#[test]
fn pilot_rejects_every_authoritative_preimage_family_mutation() {
    let RawParity(_, input, _) = parity();
    for output_preimage in [false, true] {
        let words = if output_preimage { &input.2 } else { &input.0 };
        for (family, index) in fixture_mutation_indices(words) {
            let mut mutated = input.clone();
            let target = if output_preimage {
                &mut mutated.2
            } else {
                &mut mutated.0
            };
            target[index] = changed_word(target[index]);
            assert!(
                verify_pilot(&mutated).is_err(),
                "{} preimage mutation must reject: {family}",
                if output_preimage { "output" } else { "prior" },
            );
        }
    }
}

#[test]
fn pilot_rejects_every_public_value_and_malformed_encoding() {
    let RawParity(_, input, _) = parity();

    for column in 0..PRIOR_PUBLIC_WORDS {
        let mut mutated = input.clone();
        mutated.1[column] = changed_word(mutated.1[column]);
        assert!(
            verify_pilot(&mutated).is_err(),
            "prior public column {column} mutation must reject",
        );
    }
    for lane in 0..DIGEST_WORDS {
        let mut mutated = input.clone();
        mutated.3[lane] = changed_word(mutated.3[lane]);
        assert!(
            verify_pilot(&mutated).is_err(),
            "output digest lane {lane} mutation must reject",
        );
    }

    for output_preimage in [false, true] {
        let mut truncated = input.clone();
        if output_preimage {
            truncated.2.pop();
        } else {
            truncated.0.pop();
        }
        assert!(verify_pilot(&truncated).is_err());

        let mut extended = input.clone();
        if output_preimage {
            extended.2.push(0);
        } else {
            extended.0.push(0);
        }
        assert!(verify_pilot(&extended).is_err());

        let mut noncanonical = input.clone();
        if output_preimage {
            noncanonical.2[0] = GOLDILOCKS_MODULUS;
        } else {
            noncanonical.0[0] = GOLDILOCKS_MODULUS;
        }
        assert!(verify_pilot(&noncanonical).is_err());
    }

    let mut short_public = input.clone();
    short_public.1.pop();
    assert!(verify_pilot(&short_public).is_err());

    let mut long_digest = input.clone();
    long_digest.3.push(0);
    assert!(verify_pilot(&long_digest).is_err());

    let mut noncanonical_public = input;
    noncanonical_public.1[0] = GOLDILOCKS_MODULUS;
    assert!(verify_pilot(&noncanonical_public).is_err());
}
