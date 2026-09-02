//! Poseidon2 identity binding for the canonical Lean-emitted package.

use neo_ajtai::nightstream_fprime_setup::production_authority_words;
use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use crate::package::{PackageError, PI_CCS_V1_1_ROUND_COUNT};
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
    8_237_867_231_673_714_158,
    9_137_925_141_728_451_729,
    15_386_715_550_800_926_991,
    8_441_232_538_509_246_241,
];
pub const POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY: [u64; 4] = [
    14_715_010_765_054_236_145,
    2_785_364_480_572_687_531,
    13_125_420_619_761_893_675,
    2_341_830_514_818_296_126,
];
pub const POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST: [u64; 4] = [
    1_860_265_443_911_764_719,
    6_962_543_029_970_685_374,
    879_560_073_388_521_708,
    7_535_128_577_962_597_164,
];

/// Verifier-owned context derived from one identity-checked package and the
/// canonical serialization of its commitment setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1VerifierContext {
    package_identity: [u64; 4],
    relation_words: Vec<u64>,
    application_words: Vec<u64>,
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

    pub fn application_words(&self) -> &[u64] {
        &self.application_words
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

/// Complete verifier-owned binding for one identity-checked Stage 1 package.
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
    let mut input = IDENTITY_DOMAIN.map(Goldilocks::from_u64).to_vec();
    append_value_preimage(package, &mut input)?;
    Ok(poseidon2::poseidon2_hash(&input).map(|value| value.as_canonical_u64()))
}

pub(super) fn pi_ccs_v1_1_verifier_context(
    package_identity: [u64; 4],
    commitment_key_words: &[u64],
) -> Result<PiCcsV1_1VerifierContext, PackageError> {
    validate_context_words(commitment_key_words)?;
    let relation_words = package_identity.to_vec();
    let application_words = package_identity.to_vec();
    let commitment_digest = component_digest(4, commitment_key_words)?;

    let mut nifs_key_words = bytes_as_words(NIFS_KEY_DOMAIN);
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut nifs_key_words, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut nifs_key_words, &relation_words)?;
    append_framed(&mut nifs_key_words, &commitment_digest)?;

    let relation = component_digest(1, &relation_words)?;
    let application = component_digest(2, &application_words)?;
    let nifs_key = component_digest(3, &nifs_key_words)?;

    let mut descriptor = bytes_as_words(VERIFIER_CONTEXT_DOMAIN);
    append_framed(&mut descriptor, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut descriptor, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut descriptor, &relation)?;
    append_framed(&mut descriptor, &application)?;
    append_framed(&mut descriptor, &nifs_key)?;
    append_framed(&mut descriptor, &commitment_digest)?;
    let digest = poseidon_words(&descriptor);

    Ok(PiCcsV1_1VerifierContext {
        package_identity,
        relation_words,
        application_words,
        nifs_key_words,
        commitment_key_words: commitment_key_words.to_vec(),
        descriptor_words: descriptor,
        digest,
    })
}

pub(super) fn stage1_verifier_binding(
    structural_identifier: [u64; 4],
    relation_value_words: &[u64],
    application_words: &[u64],
) -> Result<Stage1VerifierBinding, PackageError> {
    let commitment_key_words = production_authority_words();
    validate_context_words(relation_value_words)?;
    validate_context_words(application_words)?;
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
    let application = component_digest(2, application_words)?;
    let nifs_key = component_digest(3, &nifs_key_words)?;

    let mut descriptor_words = bytes_as_words(VERIFIER_CONTEXT_DOMAIN);
    append_framed(&mut descriptor_words, &VERIFIER_CONTEXT_PROFILE)?;
    append_framed(&mut descriptor_words, &VERIFIER_CONTEXT_SCHEDULE)?;
    append_framed(&mut descriptor_words, &relation)?;
    append_framed(&mut descriptor_words, &application)?;
    append_framed(&mut descriptor_words, &nifs_key)?;
    append_framed(&mut descriptor_words, &commitment_digest)?;
    let digest = poseidon_words(&descriptor_words);

    let verifier_context = PiCcsV1_1VerifierContext {
        package_identity: structural_identifier,
        relation_words,
        application_words: application_words.to_vec(),
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
    let mut preimage = bytes_as_words(VERIFIER_CONTEXT_COMPONENT_DOMAIN);
    preimage.push(component);
    append_framed(&mut preimage, words)?;
    Ok(poseidon_words(&preimage))
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
    let input = words
        .iter()
        .copied()
        .map(Goldilocks::from_u64)
        .collect::<Vec<_>>();
    poseidon2::poseidon2_hash(&input).map(|value| value.as_canonical_u64())
}

fn append_identity_node(input: &mut Vec<Goldilocks>, tag: u64, value: u64) {
    input.push(Goldilocks::from_u64(tag));
    input.push(Goldilocks::from_u64(value & 0xffff_ffff));
    input.push(Goldilocks::from_u64(value >> 32));
    input.push(Goldilocks::ZERO);
}

pub(super) fn value_preimage_words(value: &Value) -> Result<Vec<u64>, PackageError> {
    let mut words = Vec::new();
    append_value_preimage_words(value, &mut words)?;
    Ok(words)
}

fn append_value_preimage_words(value: &Value, words: &mut Vec<u64>) -> Result<(), PackageError> {
    match value {
        Value::Number(number) => {
            let value = number
                .as_u64()
                .ok_or(PackageError::Invalid("non-natural package atom"))?;
            words.extend([0, value & 0xffff_ffff, value >> 32, 0]);
        }
        Value::Array(values) => {
            let length = u64::try_from(values.len()).map_err(|_| PackageError::Invalid("array length"))?;
            words.extend([1, length & 0xffff_ffff, length >> 32, 0]);
            for child in values {
                append_value_preimage_words(child, words)?;
            }
        }
        _ => return Err(PackageError::Invalid("nonnumeric package value")),
    }
    Ok(())
}

fn append_value_preimage(value: &Value, input: &mut Vec<Goldilocks>) -> Result<(), PackageError> {
    match value {
        Value::Number(number) => {
            let value = number
                .as_u64()
                .ok_or(PackageError::Invalid("non-natural package atom"))?;
            append_identity_node(input, 0, value);
        }
        Value::Array(values) => {
            let length = u64::try_from(values.len()).map_err(|_| PackageError::Invalid("array length"))?;
            append_identity_node(input, 1, length);
            for child in values {
                append_value_preimage(child, input)?;
            }
        }
        _ => return Err(PackageError::Invalid("nonnumeric package value")),
    }
    Ok(())
}
