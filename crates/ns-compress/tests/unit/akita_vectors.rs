//! Test vectors of Akita openings for a Lean replay. `the_vector_files_are_those_of_new_proofs`
//! reads the files and never writes. `write_the_vector_files` writes them:
//! `cargo test --release -p ns-compress --lib akita_vectors -- --ignored`.

#[path = "akita_vectors/replay.rs"]
mod replay;
#[path = "akita_vectors/trace.rs"]
mod trace;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use akita_types::{sample_akita_setup_seed, CommittedGroup, SetupPrefixVerifierRegistry};
use neo_math::{from_complex, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};

use self::replay::{replay, Statement};
use self::trace::{base_words, extension_words, hex, Step, MODULUS};
use crate::akita::prover::Prover;
use crate::akita::verifier::{Opening, Verifier};
use crate::akita::{base, commitment, extension, setup_prefixes, Base, Polynomial, Schedules, Shape, SCHEDULES};
use crate::key;
use crate::layout::LOOKUP_VARS;
use crate::support::{instances, key as prover_key, layout as layout_of, statements};
use crate::transcript::{absorb_bytes, absorb_extension, challenges as draw, session};
use crate::{
    absorb_statements, challenges, claimed_sum, commit, inverse_point, lookup_challenges, sumcheck, verify, weights,
    Error, SESSION,
};

/// The checks of `akita-verifier` that need its private code. The replay does not evaluate them.
const UNCHECKED: [&str; 4] = [
    "fused.end",
    "stage3.end",
    "terminal.a_rows",
    "terminal_t_last_block_digest",
];

/// The first seed of `direct` at `key20.aks` whose opening has a response nonce that is not zero.
const RETRY_SEED: u64 = 1;

struct Case {
    schedules: &'static Schedules,
    file: &'static str,
    shape: Shape,
    commitments: [Vec<u8>; 3],
    points: [Vec<K>; 3],
    values: [K; 3],
    session: Vec<u8>,
    proof: Vec<u8>,
    registry: SetupPrefixVerifierRegistry<Base>,
    verifier: &'static Verifier,
    origin: Value,
}

impl Case {
    fn accepts(&self, proof: &[u8]) -> bool {
        self.verdict(&self.commitments, &self.points, &self.values, proof)
            .is_ok()
    }

    fn verdict(
        &self,
        commitments: &[Vec<u8>; 3],
        points: &[Vec<K>; 3],
        values: &[K; 3],
        proof: &[u8],
    ) -> Result<(), Error> {
        let openings = [0, 1, 2].map(|group| Opening {
            commitment: &commitments[group],
            point: &points[group],
            value: values[group],
        });
        self.verifier.verify(openings, proof, &self.session)
    }

    fn groups(&self) -> Vec<CommittedGroup<Base>> {
        self.commitments
            .iter()
            .map(|bytes| commitment(bytes).expect("commitment"))
            .collect()
    }

    fn replay(&self, proof: &[u8]) -> Step<Value> {
        let points: Vec<Vec<_>> = self
            .points
            .iter()
            .map(|point| point.iter().copied().map(extension).collect())
            .collect();
        let values: Vec<_> = self.values.iter().copied().map(extension).collect();
        replay(
            &Statement {
                schedules: self.schedules,
                shape: self.shape,
                commitments: &self.groups(),
                points: &points,
                values: &values,
                registry: &self.registry,
                session: &self.session,
            },
            proof,
        )
    }
}

/// `crate::verifier::verify`, to the statement of the Akita opening.
fn compressed() -> Case {
    let instances = instances();
    let statements = statements(&instances);
    let layout = layout_of(&instances);
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let proof = commit(prover_key(), layout.clone(), &witnesses)
        .expect("commitment")
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements)
        .expect("proof")
        .0;
    let verification = prover_key().verification_key();
    verify(&mut Poseidon2Transcript::new_v1_2(), verification, &statements, &proof).expect("the proof verifies");

    let transcript = &mut Poseidon2Transcript::new_v1_2();
    absorb_statements(transcript, verification, &statements);
    absorb_bytes(transcript, &proof.witness_commitment);
    let taus = lookup_challenges(transcript);
    absorb_bytes(transcript, &proof.inverse_commitment);
    let challenges = challenges(transcript, &layout, taus);
    let (row_point, _) = key::row_point(challenges.omega).expect("row point");
    let (rho, _) = sumcheck::verify(transcript, &proof.rounds, claimed_sum(&statements, &challenges)).expect("rounds");
    absorb_extension(transcript, &[proof.witness_value]);
    absorb_extension(transcript, &proof.inverse_values);
    absorb_extension(transcript, &[proof.key_part]);
    let column = draw(transcript, LOOKUP_VARS);
    let (mut key_point, _) = sumcheck::verify(transcript, &proof.key_rounds, proof.key_part).expect("key rounds");
    absorb_extension(transcript, &[proof.key_polynomial_value]);
    key_point.extend(row_point);
    let inverse_value = weights::equality(&column, proof.inverse_values.len())
        .iter()
        .zip(&proof.inverse_values)
        .map(|(&weight, &value)| weight * value)
        .sum();
    let shape = SCHEDULES[0].first();
    Case {
        schedules: &SCHEDULES[0],
        file: "key20-opening.json",
        shape,
        commitments: [
            verification.commitment().to_vec(),
            proof.witness_commitment.clone(),
            proof.inverse_commitment.clone(),
        ],
        points: [key_point, rho.clone(), inverse_point(&layout, &rho, &column)],
        values: [proof.key_polynomial_value, proof.witness_value, inverse_value],
        session: session(transcript, SESSION),
        proof: proof.opening,
        registry: SetupPrefixVerifierRegistry::new(sample_akita_setup_seed()),
        verifier: verification.akita(shape).expect("verifier"),
        origin: json!("the opening of a proof of `ns_compress::commit` for the claims of `tests/support`"),
    }
}

/// `point[i]` binds bit `i` of the cell index.
fn value(cells: &[F], point: &[K]) -> K {
    let mut table: Vec<K> = cells.iter().map(|&cell| K::from(cell)).collect();
    table.resize(cells.len().next_power_of_two(), K::ZERO);
    let low = table.len().trailing_zeros() as usize;
    for &coordinate in &point[..low] {
        table = table
            .chunks(2)
            .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
            .collect();
    }
    point[low..]
        .iter()
        .fold(table[0], |value, &coordinate| value * (K::ONE - coordinate))
}

fn direct(schedules: &'static Schedules, row: usize, file: &'static str, cells: usize, seed: u64) -> Case {
    let (witness, inverse) = schedules.rows[row];
    let shape = Shape {
        key: schedules.key_vars,
        witness,
        inverse,
    };
    let vars = [shape.key, shape.witness, shape.inverse];
    let rng = &mut ChaCha8Rng::seed_from_u64(seed);
    let prover = Prover::new(schedules).expect("prover");
    let order = [Polynomial::Key, Polynomial::Witness, Polynomial::Inverse];
    let polynomials: [Vec<F>; 3] = std::array::from_fn(|_| (0..cells).map(|_| F::new(rng.next_u64())).collect());
    let points: [Vec<K>; 3] = std::array::from_fn(|index| {
        (0..vars[index])
            .map(|_| from_complex(F::new(rng.next_u64()), F::new(rng.next_u64())))
            .collect()
    });
    let values: [K; 3] = std::array::from_fn(|index| value(&polynomials[index], &points[index]));
    let committed: [_; 3] = std::array::from_fn(|index| {
        prover
            .commit(
                shape,
                order[index],
                polynomials[index].iter().copied().map(base).collect(),
            )
            .expect("commitment")
    });
    let commitments = std::array::from_fn(|index| committed[index].bytes.clone());
    let session = format!("ns-compress/tests/akita_vectors/{file}").into_bytes();
    let [first, second, third] = committed;
    let proof = prover
        .open(
            [
                (first, &points[0], values[0]),
                (second, &points[1], values[1]),
                (third, &points[2], values[2]),
            ],
            &session,
        )
        .expect("opening");
    let prefixes = prover.setup_prefixes().expect("prefixes");
    Case {
        schedules,
        file,
        shape,
        commitments,
        points,
        values,
        session,
        proof,
        registry: setup_prefixes(&prefixes).expect("registry"),
        verifier: Box::leak(Box::new(Verifier::new(shape, &prefixes).expect("verifier"))),
        origin: json!({"prover": "akita", "random_cells": cells, "seed": seed}),
    }
}

fn vector(case: &Case) -> Value {
    assert!(
        case.accepts(&case.proof),
        "{}: the verifier accepts the proof",
        case.file
    );
    let mut vector = case
        .replay(&case.proof)
        .unwrap_or_else(|failure| panic!("{}: the replay fails: {failure:?}", case.file));
    let groups: Value = case
        .groups()
        .iter()
        .enumerate()
        .map(|(group, committed)| {
            let point: Vec<_> = case.points[group].iter().copied().map(extension).collect();
            json!({
                "commitment": hex(&case.commitments[group]),
                "commitment_digest": hex(committed.digest().as_bytes()),
                "commitment_rows": base_words(committed.rows().coeffs()),
                "point": extension_words(&point),
                "value": extension_words(&[extension(case.values[group])]),
            })
        })
        .collect();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for event in vector["events"].as_array().expect("events") {
        *counts
            .entry(event["op"].as_str().expect("op").to_string())
            .or_default() += 1;
    }
    vector["format"] = json!("nightstream-akita-opening-v2");
    vector["schedule"] = json!(format!("crates/ns-compress/artifacts/key{}.aks", case.shape.key));
    vector["shape"] = json!({"key": case.shape.key, "witness": case.shape.witness, "inverse": case.shape.inverse});
    vector["modulus"] = json!(MODULUS.to_string());
    vector["origin"] = case.origin.clone();
    vector["session"] = json!(hex(&case.session));
    vector["groups"] = groups;
    vector["proof"] = json!(hex(&case.proof));
    vector["counts"] = json!(counts);
    vector["unchecked"] = json!(UNCHECKED);
    vector["result"] = json!("accept");
    vector
}

fn change(case: &Case, mut record: Value, proof: &[u8]) -> Value {
    let accepted = case.accepts(proof);
    record["result"] = json!(if accepted { "accept" } else { "reject" });
    match case.replay(proof) {
        Err(failure) => {
            assert!(
                !accepted,
                "the replay fails on a proof that the verifier accepts: {failure:?}"
            );
            record["stage"] = json!(failure.stage);
            record["failed"] = json!({"level": failure.level, "site": failure.site});
        }
        Ok(_) if !accepted => record["stage"] = json!("unchecked"),
        Ok(_) => {}
    }
    record
}

fn negative(case: &Case, vector: &Value, kind: impl Fn(&str) -> bool) -> Value {
    let events = vector["events"].as_array().expect("events");
    let mut kinds: BTreeMap<(u64, String), Vec<&Value>> = BTreeMap::new();
    for event in events {
        let op = event["op"].as_str().expect("op");
        if ["message", "work", "fold_nonce"].contains(&op) {
            let site = if op == "message" {
                event["site"].as_str().expect("site")
            } else {
                op
            };
            if !kind(site) {
                continue;
            }
            kinds
                .entry((event["level"].as_u64().expect("level"), site.to_string()))
                .or_default()
                .push(event);
        }
    }
    let mut changed = BTreeSet::new();
    let mut changes = Vec::new();
    for ((level, site), occurrences) in &kinds {
        for (occurrence, event) in [
            (0, occurrences[0]),
            (occurrences.len() - 1, occurrences[occurrences.len() - 1]),
        ] {
            let offset = event["offset"].as_u64().expect("offset") as usize;
            let length = event["bytes"].as_u64().expect("bytes") as usize;
            for at in [offset, offset + length - length.min(8)] {
                assert!(
                    (offset..offset + length).contains(&at),
                    "byte {at} is not in {site} of level {level}"
                );
                if !changed.insert(at) {
                    continue;
                }
                let mut proof = case.proof.clone();
                proof[at] ^= 1;
                changes.push(change(
                    case,
                    json!({"site": site, "level": level, "occurrence": occurrence, "offset": at, "xor": "01"}),
                    &proof,
                ));
            }
        }
    }
    let mut longer = case.proof.clone();
    longer.push(0);
    changes.push(change(case, json!({"site": "end", "append": "00"}), &longer));
    changes.push(change(
        case,
        json!({"site": "end", "truncate": 1}),
        &case.proof[..case.proof.len() - 1],
    ));
    let mut stages: BTreeMap<String, usize> = BTreeMap::new();
    for change in &changes {
        assert_eq!(change["result"], "reject", "a changed proof is accepted: {change}");
        *stages
            .entry(change["stage"].as_str().expect("stage").to_string())
            .or_default() += 1;
    }
    json!({
        "format": "nightstream-akita-opening-negative-v2", "of": case.file, "stages": stages, "changes": changes,
    })
}

/// Changes of the statement with the proof of `case`. `Verifier::verify` takes three openings
/// with one value each, so a statement has no other count of groups or of values.
fn statement_negative(case: &Case) -> Value {
    let mut changes = Vec::new();
    let mut record =
        |kind: &str, group: usize, commitment: Option<Vec<u8>>, point: Option<Vec<K>>, value: Option<K>| {
            let mut change = json!({"change": kind, "group": group});
            let (mut commitments, mut points, mut values) =
                (case.commitments.clone(), case.points.clone(), case.values);
            if let Some(commitment) = commitment {
                change["commitment"] = json!(hex(&commitment));
                commitments[group] = commitment;
            }
            if let Some(point) = point {
                let words: Vec<_> = point.iter().copied().map(extension).collect();
                change["point"] = extension_words(&words);
                points[group] = point;
            }
            if let Some(value) = value {
                change["value"] = extension_words(&[extension(value)]);
                values[group] = value;
            }
            match case.verdict(&commitments, &points, &values, &case.proof) {
                Ok(()) => change["result"] = json!("accept"),
                Err(error) => {
                    change["result"] = json!("reject");
                    change["error"] = json!(error.to_string());
                }
            }
            assert_eq!(
                change["result"] == "accept",
                kind == "none",
                "the result of the statement change {kind} of group {group}"
            );
            changes.push(change);
        };
    record("none", 0, None, None, None);
    for (group, committed) in case.groups().iter().enumerate() {
        let commitment = &case.commitments[group];
        let point = &case.points[group];
        let header = commitment.len() - 8 * committed.rows().coeffs().len();
        let with = |at: usize, bytes: &[u8]| {
            let mut changed = commitment.clone();
            changed[at..at + bytes.len()].copy_from_slice(bytes);
            Some(changed)
        };
        let flipped = |at: usize| with(at, &[commitment[at] ^ 1]);
        record(
            "point.shorter",
            group,
            None,
            Some(point[..point.len() - 1].to_vec()),
            None,
        );
        record(
            "point.longer",
            group,
            None,
            Some([&point[..], &[K::ZERO]].concat()),
            None,
        );
        record(
            "point.coordinate",
            group,
            None,
            Some([&[point[0] + K::ONE], &point[1..]].concat()),
            None,
        );
        record("value", group, None, None, Some(case.values[group] + K::ONE));
        record(
            "commitment.shorter",
            group,
            Some(commitment[..commitment.len() - 1].to_vec()),
            None,
            None,
        );
        record(
            "commitment.longer",
            group,
            Some([&commitment[..], &[0]].concat()),
            None,
            None,
        );
        for at in [
            0,
            1,
            9,
            17,
            41,
            49,
            61,
            63,
            64,
            95,
            96,
            100,
            116,
            132,
            144,
            211,
            header - 1,
        ] {
            record(&format!("header.{at}"), group, flipped(at), None, None);
        }
        for at in [header, commitment.len() - 8] {
            record("row.modulus", group, with(at, &MODULUS.to_le_bytes()), None, None);
            record("row.maximum", group, with(at, &u64::MAX.to_le_bytes()), None, None);
            let changed = u64::from_le_bytes(commitment[at..at + 8].try_into().expect("8 bytes")) ^ 1;
            assert!(changed < MODULUS, "the changed coefficient is canonical");
            record("row.coefficient", group, with(at, &changed.to_le_bytes()), None, None);
        }
    }
    json!({"format": "nightstream-akita-statement-negative-v1", "of": case.file, "changes": changes})
}

fn folder() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/tests/vectors/akita")
}

fn nonces(vector: &Value) -> Vec<u64> {
    vector["events"]
        .as_array()
        .expect("events")
        .iter()
        .filter(|event| event["op"] == "fold_nonce")
        .map(|event| event["nonce"].as_u64().expect("nonce"))
        .collect()
}

fn files() -> Vec<(&'static str, String)> {
    let text = |value: &Value| serde_json::to_string(value).expect("JSON") + "\n";
    let case = compressed();
    let positive = vector(&case);
    let mut files = vec![
        (
            "key20-opening-negative.json",
            text(&negative(&case, &positive, |_| true)),
        ),
        ("key20-statement-negative.json", text(&statement_negative(&case))),
        (case.file, text(&positive)),
    ];

    let retried = vector(&direct(&SCHEDULES[0], 0, "key20-opening-retry.json", 5, RETRY_SEED));
    assert!(
        nonces(&retried).iter().any(|&nonce| nonce != 0),
        "seed {RETRY_SEED} gives no response nonce that is not zero"
    );
    files.push(("key20-opening-retry.json", text(&retried)));

    for (row, file) in [(0, "key30-opening-26-28.json"), (1, "key30-opening-27-28.json")] {
        let case = direct(&SCHEDULES[1], row, file, 5, row as u64);
        let positive = vector(&case);
        if row == 0 {
            files.push((
                "key30-opening-26-28-negative.json",
                text(&negative(&case, &positive, |site| site.starts_with("stage3."))),
            ));
        }
        files.push((file, text(&positive)));
    }
    files
}

#[test]
fn the_vector_files_are_those_of_new_proofs() {
    for (name, text) in files() {
        let file = fs::read_to_string(folder().join(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let at = file
            .bytes()
            .zip(text.bytes())
            .position(|(left, right)| left != right)
            .unwrap_or(file.len().min(text.len()));
        assert!(
            file == text,
            "{name} differs from the vector of a new proof at byte {at}; `write_the_vector_files` writes it"
        );
    }
}

#[test]
#[ignore = "writes the vector files"]
fn write_the_vector_files() {
    fs::create_dir_all(folder()).expect("folder");
    for (name, text) in files() {
        fs::write(folder().join(name), text).expect("vector file");
    }
}
