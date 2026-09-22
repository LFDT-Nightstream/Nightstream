//! Poseidon2 identity binding for the canonical Lean-emitted package.

use neo_ajtai::nightstream_fprime_setup::{authority_words, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS};
use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use crate::package::{PackageError, PI_CCS_V1_1_ROUND_COUNT};

mod native;
pub(crate) use native::{
    native_application_identity, native_application_word_count, native_relation_identifier,
    visit_native_application_words,
};

const IDENTITY_DOMAIN: [u64; 29] = [
    78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 70, 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97, 103,
    101, 47, 118, 50,
];

const VERIFIER_CONTEXT_PROFILE: [u64; 14] = [
    4_294_967_295,
    1,
    2,
    16,
    65_536,
    1,
    16,
    17,
    16,
    14,
    PI_CCS_V1_1_ROUND_COUNT as u64,
    9,
    54,
    22,
];
const VERIFIER_CONTEXT_SCHEDULE: [u64; 10] = [1, 1, 1, PI_CCS_V1_1_ROUND_COUNT as u64, 10, 17, 14, 54, 16, 64];
const VERIFIER_CONTEXT_COMPONENT_DOMAIN: &[u8] = b"Nightstream/FPrime/context/v1_1";
const VERIFIER_CONTEXT_DOMAIN: &[u8] = b"Nightstream/FPrime/verifier-context/v1_1";
const NIFS_KEY_DOMAIN: &[u8] = b"Nightstream/FPrime/nifs-key/v1_1";
const PACKAGE_IDENTITY_DOMAIN: &[u8] = b"Nightstream/FPrime/sealed-package/v2";
const VERIFICATION_KEY_DOMAIN: &[u8] = b"Nightstream/FPrime/verifier-key/v1";

pub const POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER: [u64; 4] = [
    10_842_744_081_364_099_826,
    8_947_654_861_443_655_402,
    4_184_000_853_237_937_223,
    4_234_949_493_831_660_790,
];
pub const POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY: [u64; 4] = [
    11_780_343_655_336_100_175,
    3_739_837_894_403_928_952,
    5_393_028_243_801_154_131,
    3_026_325_569_224_679_930,
];
pub const POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST: [u64; 4] = [
    6_990_396_241_798_352_215,
    3_177_540_295_219_691_657,
    4_868_921_110_330_287_131,
    17_048_300_498_170_007_748,
];

/// Verifier-owned context derived from one identity-checked package and the
/// canonical serialization of its commitment setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1VerifierContext {
    package_identity: [u64; 4],
    relation_words: Vec<u64>,
    application: ApplicationIdentity,
    nifs_key_words: Vec<u64>,
    commitment_key_words: Vec<u64>,
    descriptor_words: Vec<u64>,
    digest: [u64; 4],
}

impl PiCcsV1_1VerifierContext {
    pub fn digest(&self) -> [u64; 4] {
        self.digest
    }

    pub fn relation_words(&self) -> &[u64] {
        &self.relation_words
    }

    pub fn application_word_count(&self) -> usize {
        self.application.word_count
    }

    pub fn application_digest(&self) -> [u64; 4] {
        self.application.digest
    }

    pub fn nifs_key_words(&self) -> &[u64] {
        &self.nifs_key_words
    }

    pub fn commitment_key_words(&self) -> &[u64] {
        &self.commitment_key_words
    }

    pub fn descriptor_words(&self) -> &[u64] {
        &self.descriptor_words
    }

    pub(crate) fn structural_identifier(&self) -> [u64; 4] {
        self.package_identity
    }
}

/// Application digest and exact preimage length, computed during compilation
/// or retained as non-authoritative prepared-package metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ApplicationIdentity {
    digest: [u64; 4],
    word_count: usize,
}

impl ApplicationIdentity {
    pub(crate) fn from_words(words: &[u64]) -> Result<Self, PackageError> {
        validate_context_words(words)?;
        Ok(Self {
            digest: component_digest(2, words)?,
            word_count: words.len(),
        })
    }

    pub(crate) fn from_cached(digest: [u64; 4], word_count: usize) -> Result<Self, PackageError> {
        validate_context_words(&digest)?;
        Ok(Self { digest, word_count })
    }

    pub(crate) fn cached_parts(&self) -> ([u64; 4], usize) {
        (self.digest, self.word_count)
    }
}

/// Complete Stage 1 binding; circuit authority belongs to the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1VerifierBinding {
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    verifier_context: PiCcsV1_1VerifierContext,
    verification_key_words: Vec<u64>,
    verification_key_digest: [u64; 4],
}

impl Stage1VerifierBinding {
    pub fn structural_identifier(&self) -> [u64; 4] {
        self.structural_identifier
    }

    pub fn package_identity(&self) -> [u64; 4] {
        self.package_identity
    }

    pub fn verifier_context(&self) -> &PiCcsV1_1VerifierContext {
        &self.verifier_context
    }

    pub fn verification_key_words(&self) -> &[u64] {
        &self.verification_key_words
    }

    pub fn verification_key_digest(&self) -> [u64; 4] {
        self.verification_key_digest
    }
}

pub(super) fn relation_identifier(package: &Value) -> Result<[u64; 4], PackageError> {
    let mut input = poseidon2::Poseidon2Hasher::default();
    input.update(&IDENTITY_DOMAIN.map(Goldilocks::from_u64));
    append_value_preimage(package, &mut input)?;
    Ok(input.finalize().map(|value| value.as_canonical_u64()))
}

pub(super) fn pi_ccs_v1_1_verifier_context(
    package_identity: [u64; 4],
    commitment_key_words: &[u64],
) -> Result<PiCcsV1_1VerifierContext, PackageError> {
    validate_context_words(commitment_key_words)?;
    let relation_words = package_identity.to_vec();
    let application = ApplicationIdentity::from_words(&package_identity)?;
    let commitment_digest = component_digest(4, commitment_key_words)?;

    let mut nifs_key_words = bytes_as_words(NIFS_KEY_DOMAIN);
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut nifs_key_words, &relation_words)?;
    append_framed(&mut nifs_key_words, &commitment_digest)?;

    let relation = component_digest(1, &relation_words)?;
    let nifs_key = component_digest(3, &nifs_key_words)?;

    let mut descriptor = bytes_as_words(VERIFIER_CONTEXT_DOMAIN);
    append_framed(&mut descriptor, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut descriptor, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut descriptor, &relation)?;
    append_framed(&mut descriptor, &application.digest)?;
    append_framed(&mut descriptor, &nifs_key)?;
    append_framed(&mut descriptor, &commitment_digest)?;
    let digest = poseidon_words(&descriptor);

    Ok(PiCcsV1_1VerifierContext {
        package_identity,
        relation_words,
        application,
        nifs_key_words,
        commitment_key_words: commitment_key_words.to_vec(),
        descriptor_words: descriptor,
        digest,
    })
}

pub(super) fn stage1_verifier_binding(
    structural_identifier: [u64; 4],
    logical_columns: usize,
    relation_value_words: &[u64],
    application: &ApplicationIdentity,
) -> Result<Stage1VerifierBinding, PackageError> {
    let message_columns = u64::try_from(logical_columns.div_ceil(54))
        .map_err(|_| PackageError::Invalid("Stage 1 carrier block count"))?;
    let commitment_key_words = authority_words(PRODUCTION_VERIFIER_ROWS, message_columns, &PRODUCTION_SEED);
    validate_context_words(relation_value_words)?;
    validate_context_words(&commitment_key_words)?;

    let mut relation_words = relation_value_words.to_vec();
    relation_words.extend_from_slice(&structural_identifier);
    let commitment_digest = component_digest(4, &commitment_key_words)?;

    let mut nifs_key_words = bytes_as_words(NIFS_KEY_DOMAIN);
    append_framed(&mut nifs_key_words, &relation_words)?;
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut nifs_key_words, &commitment_digest)?;

    let relation = component_digest(1, &relation_words)?;
    let nifs_key = component_digest(3, &nifs_key_words)?;

    let mut descriptor_words = bytes_as_words(VERIFIER_CONTEXT_DOMAIN);
    append_framed(&mut descriptor_words, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut descriptor_words, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut descriptor_words, &relation)?;
    append_framed(&mut descriptor_words, &application.digest)?;
    append_framed(&mut descriptor_words, &nifs_key)?;
    append_framed(&mut descriptor_words, &commitment_digest)?;
    let digest = poseidon_words(&descriptor_words);

    let verifier_context = PiCcsV1_1VerifierContext {
        package_identity: structural_identifier,
        relation_words,
        application: application.clone(),
        nifs_key_words,
        commitment_key_words,
        descriptor_words: descriptor_words.clone(),
        digest,
    };

    let mut package_identity_words = bytes_as_words(PACKAGE_IDENTITY_DOMAIN);
    append_framed(&mut package_identity_words, &structural_identifier)?;
    append_framed(&mut package_identity_words, &descriptor_words)?;
    let package_identity = poseidon_words(&package_identity_words);

    let mut verification_key_words = bytes_as_words(VERIFICATION_KEY_DOMAIN);
    append_framed(&mut verification_key_words, &package_identity)?;
    append_framed(&mut verification_key_words, &descriptor_words)?;
    let verification_key_digest = poseidon_words(&verification_key_words);

    Ok(Stage1VerifierBinding {
        structural_identifier,
        package_identity,
        verifier_context,
        verification_key_words,
        verification_key_digest,
    })
}

fn component_digest(component: u64, words: &[u64]) -> Result<[u64; 4], PackageError> {
    let mut input = component_hasher(component, words.len())?;
    update_words(&mut input, words);
    Ok(input.finalize().map(|value| value.as_canonical_u64()))
}

fn component_hasher(component: u64, word_count: usize) -> Result<poseidon2::Poseidon2Hasher, PackageError> {
    let length = u64::try_from(word_count).map_err(|_| PackageError::Invalid("verifier-context word length"))?;
    let mut input = poseidon2::Poseidon2Hasher::default();
    for &byte in VERIFIER_CONTEXT_COMPONENT_DOMAIN {
        input.update(&[Goldilocks::from_u64(u64::from(byte))]);
    }
    update_words(&mut input, &[component, length]);
    Ok(input)
}

fn append_framed(target: &mut Vec<u64>, words: &[u64]) -> Result<(), PackageError> {
    let length = u64::try_from(words.len()).map_err(|_| PackageError::Invalid("verifier-context word length"))?;
    target.push(length);
    target.extend_from_slice(words);
    Ok(())
}

fn bytes_as_words(bytes: &[u8]) -> Vec<u64> {
    bytes.iter().map(|byte| u64::from(*byte)).collect()
}

fn validate_context_words(words: &[u64]) -> Result<(), PackageError> {
    if words.iter().any(|word| *word >= Goldilocks::ORDER_U64) {
        return Err(PackageError::Invalid("noncanonical verifier-context authority word"));
    }
    Ok(())
}

fn poseidon_words(words: &[u64]) -> [u64; 4] {
    let mut input = poseidon2::Poseidon2Hasher::default();
    update_words(&mut input, words);
    input.finalize().map(|value| value.as_canonical_u64())
}

fn update_words(input: &mut poseidon2::Poseidon2Hasher, words: &[u64]) {
    let mut fields = [Goldilocks::ZERO; poseidon2::RATE];
    for chunk in words.chunks(poseidon2::RATE) {
        for (field, &word) in fields.iter_mut().zip(chunk) {
            *field = Goldilocks::from_u64(word);
        }
        input.update(&fields[..chunk.len()]);
    }
}

pub(super) fn value_preimage_words(value: &Value) -> Result<Vec<u64>, PackageError> {
    let mut words = Vec::new();
    CanonicalSink {
        emit: &mut |chunk| {
            words.extend_from_slice(chunk);
            Ok(())
        },
    }
    .value(value)?;
    Ok(words)
}

struct CanonicalSink<'a> {
    emit: &'a mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
}

impl CanonicalSink<'_> {
    fn node(&mut self, tag: u64, value: u64) -> Result<(), PackageError> {
        (self.emit)(&[tag, value & 0xffff_ffff, value >> 32, 0])
    }

    fn number(&mut self, value: u64) -> Result<(), PackageError> {
        self.node(0, value)
    }

    fn index(&mut self, value: usize) -> Result<(), PackageError> {
        self.number(u64::try_from(value).map_err(|_| PackageError::Invalid("package index"))?)
    }

    fn array(&mut self, length: usize) -> Result<(), PackageError> {
        self.node(
            1,
            u64::try_from(length).map_err(|_| PackageError::Invalid("array length"))?,
        )
    }

    fn value(&mut self, value: &Value) -> Result<(), PackageError> {
        match value {
            Value::Number(number) => self.number(
                number
                    .as_u64()
                    .ok_or(PackageError::Invalid("non-natural package atom"))?,
            ),
            Value::Array(values) => {
                self.array(values.len())?;
                for child in values {
                    self.value(child)?;
                }
                Ok(())
            }
            _ => Err(PackageError::Invalid("nonnumeric package value")),
        }
    }
}

fn append_value_preimage(value: &Value, input: &mut poseidon2::Poseidon2Hasher) -> Result<(), PackageError> {
    CanonicalSink {
        emit: &mut |words| {
            update_words(input, words);
            Ok(())
        },
    }
    .value(value)
}

#[cfg(test)]
#[path = "../tests/unit/identity_stream.rs"]
mod tests;
