//! The events of a replay and the place of each prover message in the proof bytes.

use std::collections::HashMap;

use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_params::GrindingSite;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckShape;
use jolt_field::Ring;
use serde_json::{json, Value};

use crate::akita::{Base, Extension};

pub(super) const MODULUS: u64 = 18_446_744_073_709_551_557;

pub(super) const STATEMENT: &str = "statement";
pub(super) const OBSERVED: &str = "observed";
const REREAD: &str = "reread";
pub(super) const RECONSTRUCTED: &str = "reconstructed";

/// The first step of the replay that fails. The public error of Akita is `InvalidProof` at each
/// step, so the stage is the kind of the public function that returned it.
#[derive(Debug)]
pub(super) struct Failure {
    pub(super) stage: &'static str,
    pub(super) level: u32,
    pub(super) site: String,
}

pub(super) type Step<T> = Result<T, Failure>;

pub(super) fn failure(stage: &'static str, level: u32, site: &str) -> Failure {
    Failure {
        stage,
        level,
        site: site.to_string(),
    }
}

pub(super) fn parse(level: u32, site: &str) -> impl FnOnce(AkitaError) -> Failure + '_ {
    move |_| failure("parse", level, site)
}

pub(super) fn work(level: u32, site: &str) -> impl FnOnce(AkitaError) -> Failure + '_ {
    move |_| failure("proof_of_work", level, site)
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn bytes_of(value: &impl AkitaSerialize) -> Vec<u8> {
    let mut bytes = Vec::new();
    value.serialize_compressed(&mut bytes).expect("bytes");
    bytes
}

fn word(value: &Base) -> u64 {
    u64::from_le_bytes(bytes_of(value).try_into().expect("8 bytes"))
}

pub(super) fn base_words(values: &[Base]) -> Value {
    values.iter().map(|value| word(value).to_string()).collect()
}

fn pair(value: &Extension) -> [u64; 2] {
    let bytes = bytes_of(value);
    assert_eq!(bytes.len(), 16);
    [0, 8].map(|at| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes")))
}

pub(super) fn extension_words(values: &[Extension]) -> Value {
    values
        .iter()
        .map(|value| pair(value).map(|word| word.to_string()))
        .collect()
}

fn extension_bytes(values: &[Extension]) -> Vec<u8> {
    values.iter().flat_map(bytes_of).collect()
}

pub(super) fn sparse(challenges: &Challenges) -> Value {
    challenges
        .as_slice()
        .iter()
        .map(|challenge| {
            challenge
                .positions
                .iter()
                .zip(challenge.coeffs.iter())
                .map(|(position, coefficient)| json!([position, coefficient]))
                .collect::<Value>()
        })
        .collect()
}

/// The place of each prover message in the proof bytes. The public interface does not return
/// the nonces of `jolt_transcript::VerifierTranscript::check_grind` and the round messages of a
/// sum-check, so the trace reads them from the proof at the cursor.
pub(super) struct Trace<'a> {
    pub(super) proof: &'a [u8],
    pub(super) cursor: usize,
    pub(super) work: HashMap<GrindingSite, u8>,
    pub(super) events: Vec<Value>,
}

impl Trace<'_> {
    pub(super) fn absorb(&mut self, level: u32, site: &str, source: &str, words: Value) {
        self.events
            .push(json!({"op": "absorb", "level": level, "site": site, "source": source, "words": words}));
    }

    pub(super) fn challenge(&mut self, level: u32, site: &str, words: Value) {
        self.events
            .push(json!({"op": "challenge", "level": level, "site": site, "source": OBSERVED, "words": words}));
    }

    pub(super) fn value(&mut self, level: u32, site: &str, source: &str, words: Value) {
        self.events
            .push(json!({"op": "value", "level": level, "site": site, "source": source, "words": words}));
    }

    pub(super) fn check(&mut self, level: u32, site: &str, holds: bool, confirms: &[&str]) -> Step<()> {
        if !holds {
            return Err(failure("equation", level, site));
        }
        self.events
            .push(json!({"op": "check", "level": level, "site": site, "confirms": confirms}));
        Ok(())
    }

    pub(super) fn message(&mut self, level: u32, site: &str, bytes: &[u8], words: Value) {
        if bytes.is_empty() {
            return;
        }
        let end = self.cursor + bytes.len();
        assert_eq!(
            self.proof.get(self.cursor..end),
            Some(bytes),
            "the message {site} of level {level}"
        );
        self.events.push(json!({
            "op": "message", "level": level, "site": site, "source": OBSERVED, "offset": self.cursor,
            "bytes": bytes.len(), "words": words,
        }));
        self.cursor = end;
    }

    pub(super) fn canonical(&self, from: usize, elements: usize) -> bool {
        self.proof
            .get(from..from + 8 * elements)
            .is_some_and(|bytes| {
                bytes
                    .chunks(8)
                    .all(|word| u64::from_le_bytes(word.try_into().expect("8 bytes")) < MODULUS)
            })
    }

    pub(super) fn read_then_worked<T>(
        &self,
        result: Result<T, AkitaError>,
        level: u32,
        site: &str,
        elements: usize,
    ) -> Step<T> {
        result.map_err(|_| {
            let stage = if self.canonical(self.cursor, elements) {
                "proof_of_work"
            } else {
                "parse"
            };
            failure(stage, level, site)
        })
    }

    /// LEB128, `akita-transcript/src/proof_stream/nonce.rs` and `jolt-transcript/src/pow.rs`.
    pub(super) fn nonce(&mut self) -> u64 {
        let mut value = 0;
        for index in 0.. {
            let byte = self.proof[self.cursor];
            self.cursor += 1;
            value |= u64::from(byte & 0x7f) << (7 * index);
            if byte & 0x80 == 0 {
                break;
            }
        }
        value
    }

    pub(super) fn bits(&self, site: GrindingSite) -> u8 {
        self.work.get(&site).copied().unwrap_or(0)
    }

    pub(super) fn work(&mut self, level: u32, site: GrindingSite) {
        let bits = self.bits(site);
        if bits > 0 {
            let offset = self.cursor;
            let nonce = self.nonce();
            self.events.push(json!({
                "op": "work", "level": level, "site": format!("{site:?}"), "source": REREAD, "bits": bits,
                "offset": offset, "bytes": self.cursor - offset, "nonce": nonce,
            }));
        }
    }

    pub(super) fn fold_nonce(&mut self, level: u32, nonce: u32) {
        let offset = self.cursor;
        assert_eq!(self.nonce(), u64::from(nonce), "the response nonce of level {level}");
        self.events.push(json!({
            "op": "fold_nonce", "level": level, "source": OBSERVED, "offset": offset, "bytes": self.cursor - offset,
            "nonce": nonce,
        }));
        self.work(level, GrindingSite::FoldChallengeWork { level });
    }

    pub(super) fn round(&mut self, level: u32, site: &str, round: usize, elements: usize) -> Vec<Extension> {
        let bytes = &self.proof[self.cursor..self.cursor + 16 * elements];
        let words: Vec<[u64; 2]> = bytes
            .chunks(16)
            .map(|pair| [0, 8].map(|at| u64::from_le_bytes(pair[at..at + 8].try_into().expect("8 bytes"))))
            .collect();
        self.events.push(json!({
            "op": "message", "level": level, "site": site, "source": REREAD, "round": round, "offset": self.cursor,
            "bytes": bytes.len(),
            "words": words.iter().map(|pair| pair.map(|word| word.to_string())).collect::<Value>(),
        }));
        self.cursor += bytes.len();
        words
            .iter()
            .map(|&[real, imaginary]| Extension::new(Base::from_u64(real), Base::from_u64(imaginary)))
            .collect()
    }

    /// `verify_sumcheck_rounds` fails at a round message that does not parse or at the proof of
    /// work of a round.
    pub(super) fn rounds_failure(
        &self,
        level: u32,
        name: &str,
        shape: SumcheckShape,
        site: impl Fn(u32) -> GrindingSite,
    ) -> Failure {
        let mut at = self.cursor;
        for round in 0..shape.num_rounds() {
            if !self.canonical(at, 2 * shape.degree_bound()) {
                return failure("parse", level, &format!("{name}.round"));
            }
            at += 16 * shape.degree_bound();
            if self.bits(site(round as u32)) > 0 {
                while self.proof.get(at).is_some_and(|byte| byte & 0x80 != 0) {
                    at += 1;
                }
                at += 1;
            }
        }
        failure("proof_of_work", level, &format!("{name}.round"))
    }

    pub(super) fn base_message(&mut self, level: u32, site: &str, values: &[Base]) {
        let bytes: Vec<u8> = values.iter().flat_map(bytes_of).collect();
        self.message(level, site, &bytes, base_words(values));
    }

    pub(super) fn extension_message(&mut self, level: u32, site: &str, values: &[Extension]) {
        self.message(level, site, &extension_bytes(values), extension_words(values));
    }

    pub(super) fn events_with_their_checks(mut self) -> Vec<Value> {
        let mut checked: HashMap<(u64, String), String> = HashMap::new();
        for event in &self.events {
            if event["op"] == "check" {
                for site in event["confirms"].as_array().expect("sites") {
                    checked.insert(
                        (
                            event["level"].as_u64().expect("level"),
                            site.as_str().expect("site").to_string(),
                        ),
                        event["site"].as_str().expect("site").to_string(),
                    );
                }
            }
        }
        for event in &mut self.events {
            let key = (
                event["level"].as_u64().expect("level"),
                event["site"]
                    .as_str()
                    .unwrap_or(event["op"].as_str().expect("op"))
                    .to_string(),
            );
            if event["op"] != "check" {
                if let Some(check) = checked.get(&key) {
                    event["checked"] = json!(check);
                }
            }
        }
        self.events
    }
}
