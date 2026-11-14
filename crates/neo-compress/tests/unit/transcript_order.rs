//! The transcript order of `verify` against the Lean schedule
//! (`Spec.Compression.Rust.schedule`, whose labels follow the order of the
//! verifier game, `Spec.Compression.Rust.game_follows`). A real proof is
//! replayed on a fresh Poseidon2 transcript in the Lean order. The replay
//! must end in the sponge state of the verifier, and each challenge that it
//! draws must be the one that the verifier used: both sum-checks, their
//! claimed sums, their final checks and the Akita opening hold with the
//! replayed challenges and session.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use neo_math::{KExtensions, D, F, K};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::PrimeCharacteristicRing;
use serde_json::Value;

use super::commit;
use crate::akita::verifier::Opening;
use crate::key::{row_point, weight_at};
use crate::layout::{Layout, LOOKUPS};
use crate::support::{instances, key, layout, statements};
use crate::transcript::{byte_words, challenges};
use crate::weights::{self, Challenges, Uses};
use crate::{claimed_sum, inverse_point, verify, Norm, Proof, Statement, DOMAIN, SESSION};

fn lean_schedule() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-compression-parity-v1.json");
    let lean: Value = serde_json::from_slice(&fs::read(path).expect("Lean compression parity file")).expect("JSON");
    lean["transcript"].clone()
}

fn number(value: &Value) -> usize {
    value.as_u64().expect("a natural number") as usize
}

fn words(values: &[K]) -> Vec<F> {
    values.iter().flat_map(|value| value.as_coeffs()).collect()
}

/// The words of a claim in SPEC.md section 5.4, row 6.
fn claim_words(statement: &Statement<'_>) -> Vec<F> {
    let claim = statement.claim;
    let bound = match statement.norm {
        Norm::Table { bound } => bound + 1,
        Norm::Caller => 0,
    };
    let sizes = [
        (statement.rows.shape().columns / D) as u64,
        bound,
        claim.m_in as u64,
        claim.r.len() as u64,
        claim.eval_a.len() as u64,
    ];
    let mut words: Vec<F> = sizes.map(F::from_u64).to_vec();
    words.extend_from_slice(claim.c.data());
    words.extend(claim.X.to_dense_vec());
    let extension = claim
        .r
        .iter()
        .chain(&claim.eval_k[..D])
        .chain(claim.eval_a.iter().flat_map(|family| &family[..D]));
    words.extend(extension.flat_map(|value| value.as_coeffs()));
    words
}

/// The messages of each Lean label, in order.
fn messages(statements: &[Statement<'_>], proof: &Proof) -> HashMap<&'static str, Vec<Vec<F>>> {
    let mut statement = vec![
        domain_chunk_v1_2(DOMAIN).to_vec(),
        byte_words(&key().verification_key().to_bytes()),
    ];
    statement.extend(statements.iter().map(claim_words));
    HashMap::from([
        ("statement", statement),
        ("witness", vec![byte_words(&proof.witness_commitment)]),
        ("inverse", vec![byte_words(&proof.inverse_commitment)]),
        ("round", proof.rounds.iter().map(|round| words(round)).collect()),
        (
            "values",
            vec![
                words(&[proof.witness_value]),
                words(&proof.inverse_values),
                words(&[proof.key_weight]),
            ],
        ),
        ("key_round", proof.key_rounds.iter().map(|round| words(round)).collect()),
        ("key_value", vec![words(&[proof.key_value])]),
    ])
}

/// The transcript after `steps` and the challenges of each draw label.
fn replay(steps: &[Value], mut messages: HashMap<&'static str, Vec<Vec<F>>>) -> (Poseidon2Transcript, Draws) {
    for queue in messages.values_mut() {
        queue.reverse();
    }
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let mut draws = Draws::default();
    for step in steps {
        let label = step["label"].as_str().expect("label");
        for call in step["calls"].as_array().expect("calls") {
            let mut message = || {
                messages
                    .get_mut(label)
                    .and_then(Vec::pop)
                    .unwrap_or_else(|| panic!("a message for {label}"))
            };
            match call[0].as_str().expect("operation") {
                "absorb" => {
                    let message = message();
                    assert_eq!(message.len(), number(&call[1]), "the words of {label}");
                    transcript.absorb_v1_2(&message);
                }
                "bytes" => transcript.absorb_v1_2(&message()),
                "draw" => draws
                    .0
                    .entry(label.to_owned())
                    .or_default()
                    .extend(challenges(&mut transcript, number(&call[1]))),
                other => panic!("operation {other}"),
            }
        }
    }
    assert!(messages.values().all(Vec::is_empty), "each message is absorbed");
    (transcript, draws)
}

#[derive(Default)]
struct Draws(HashMap<String, Vec<K>>);

impl Draws {
    fn of(&self, label: &str) -> &[K] {
        &self.0[label]
    }

    fn challenges(&self, layout: &Layout) -> Challenges {
        let coordinates: Vec<F> = self
            .of("tau")
            .iter()
            .flat_map(|coin| coin.as_coeffs())
            .collect();
        let mut drawn = self.of("challenges").iter().copied();
        let mut next = |count: usize| -> Vec<K> { drawn.by_ref().take(count).collect() };
        let tables = layout.tables().count();
        Challenges {
            taus: std::array::from_fn(|k| coordinates[k]),
            omega: next(1)[0],
            delta: next(1)[0],
            gamma: next(1)[0],
            mu: next(1)[0],
            psi: next(LOOKUPS).try_into().expect("one per lookup"),
            xi: next(tables),
            lambda: next(tables),
            beta: next(layout.span_vars),
        }
    }
}

/// A round polynomial at `x`, from its coefficients of degree 0 up.
fn at(coefficients: &[K], x: K) -> K {
    coefficients
        .iter()
        .rev()
        .fold(K::ZERO, |sum, &coefficient| sum * x + coefficient)
}

/// The multilinear extension of `cells` at `point`, bit `i` of a cell with
/// coordinate `i`.
fn extension(cells: &[F], point: &[K]) -> K {
    let mut table: Vec<K> = cells.iter().map(|&cell| K::from(cell)).collect();
    table.resize(1 << point.len(), K::ZERO);
    for &coordinate in point {
        table = table
            .chunks(2)
            .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
            .collect();
    }
    table[0]
}

/// Each check of `verify` with the replayed challenges and session.
fn verifies_with(
    statements: &[Statement<'_>],
    layout: &Layout,
    witness: &[F],
    proof: &Proof,
    draws: &Draws,
    session: &[u8],
) -> bool {
    let challenges = draws.challenges(layout);
    let rho = draws.of("rho");
    let mut claim = claimed_sum(statements, &challenges);
    for (round, &coordinate) in proof.rounds.iter().zip(rho) {
        if at(round, K::ZERO) + at(round, K::ONE) != claim {
            return false;
        }
        claim = at(round, coordinate);
    }
    if extension(witness, rho) != proof.witness_value {
        return false;
    }
    let zeta = draws.of("zeta");
    let mut key_claim = proof.key_weight;
    for (round, &coordinate) in proof.key_rounds.iter().zip(zeta) {
        if at(round, K::ZERO) + at(round, K::ONE) != key_claim {
            return false;
        }
        key_claim = at(round, coordinate);
    }
    let Ok((row, scale)) = row_point(challenges.omega) else {
        return false;
    };
    let uses = Uses::new(layout, &challenges, rho);
    if key_claim != scale * proof.key_value * weight_at(&uses, zeta) {
        return false;
    }
    let low: K = rho[layout.span_vars..]
        .iter()
        .map(|&coordinate| K::ONE - coordinate)
        .product();
    let combine = |weights: [K; LOOKUPS]| -> K {
        low * weights
            .iter()
            .zip(&proof.inverse_values)
            .map(|(&weight, &value)| weight * value)
            .sum::<K>()
    };
    let inverse = combine(challenges.psi);
    let scaled = combine(std::array::from_fn(|k| challenges.psi[k] * challenges.taus[k]));
    let (mut check, mut sum) = (K::ZERO, K::ZERO);
    for (table, (_, slice, _, _)) in layout.tables().enumerate() {
        let selector = slice.selector(rho);
        let equality: K = challenges.beta[..slice.vars]
            .iter()
            .zip(rho)
            .map(|(&beta, &rho)| beta * rho + (K::ONE - beta) * (K::ONE - rho))
            .product();
        check += challenges.xi[table] * equality * selector;
        sum += challenges.lambda[table] * selector;
    }
    let closed = weights::closed_part(statements, layout, &challenges, &uses, rho).expect("closed part");
    let matrices = weights::matrix_part(statements, &challenges, &uses).expect("matrix part");
    let weights = proof.key_weight + matrices + closed;
    if claim != proof.witness_value * (weights - check * inverse) + check * scaled + sum * inverse {
        return false;
    }
    let column = draws.of("column");
    let inverse_value = weights::equality(column, proof.inverse_values.len())
        .iter()
        .zip(&proof.inverse_values)
        .map(|(&weight, &value)| weight * value)
        .sum();
    let mut key_point = zeta.to_vec();
    key_point.extend(row);
    let verification = key().verification_key();
    verification
        .akita(layout.shape())
        .and_then(|akita| {
            akita.verify(
                [
                    Opening {
                        commitment: verification.commitment(),
                        point: &key_point,
                        value: proof.key_value,
                    },
                    Opening {
                        commitment: &proof.witness_commitment,
                        point: rho,
                        value: proof.witness_value,
                    },
                    Opening {
                        commitment: &proof.inverse_commitment,
                        point: &inverse_point(layout, rho, column),
                        value: inverse_value,
                    },
                ],
                &proof.opening,
                session,
            )
        })
        .is_ok()
}

#[test]
fn a_replay_in_the_lean_order_draws_the_challenges_of_verify() {
    let lean = lean_schedule();
    let instances = instances();
    let statements = statements(&instances);
    let shape: Vec<(usize, Option<u64>, usize, usize, usize)> = statements
        .iter()
        .map(|statement| {
            let bound = match statement.norm {
                Norm::Table { bound } => Some(bound),
                Norm::Caller => None,
            };
            let claim = statement.claim;
            (statement.blocks(), bound, claim.m_in, claim.r.len(), claim.eval_a.len())
        })
        .collect();
    let lean_shape: Vec<(usize, Option<u64>, usize, usize, usize)> = lean["claims"]
        .as_array()
        .expect("claims")
        .iter()
        .map(|claim| {
            let bound = claim[1].as_u64();
            (
                number(&claim[0]),
                bound,
                number(&claim[2]),
                number(&claim[3]),
                number(&claim[4]),
            )
        })
        .collect();
    assert_eq!(shape, lean_shape, "the Lean schedule is for the claims of this proof");
    assert_eq!(number(&lean["key_blocks"]), key().verification_key().blocks());

    let layout = layout(&instances);
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let committed = commit(key(), layout.clone(), &witnesses).expect("commitment");
    let witness = committed.witness.clone();
    let (proof, _) = committed
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements)
        .expect("proof");
    let mut verifier = Poseidon2Transcript::new_v1_2();
    verify(&mut verifier, key().verification_key(), &statements, &proof).expect("the proof verifies");

    let steps = lean["steps"].as_array().expect("steps");
    let (transcript, draws) = replay(steps, messages(&statements, &proof));
    assert_eq!(
        transcript.state_prefix_v1_2(),
        verifier.state_prefix_v1_2(),
        "the replay ends in the verifier's sponge state"
    );
    let session = [SESSION, &transcript.state_prefix_v1_2()].concat();
    assert!(
        verifies_with(&statements, &layout, &witness, &proof, &draws, &session),
        "each replayed challenge is the verifier's"
    );

    // `τ` drawn after the commitment of `V`: the same sponge state, other
    // challenges.
    let mut moved = steps.clone();
    let tau = moved
        .iter()
        .position(|step| step["label"] == "tau")
        .expect("tau");
    moved.swap(tau, tau + 1);
    let (transcript, draws) = replay(&moved, messages(&statements, &proof));
    assert_eq!(transcript.state_prefix_v1_2(), verifier.state_prefix_v1_2());
    assert!(
        !verifies_with(&statements, &layout, &witness, &proof, &draws, &session),
        "a draw in another place gives other challenges"
    );
}
