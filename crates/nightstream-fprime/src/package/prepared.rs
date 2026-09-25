//! Reusable native-package storage. Cached identities remain metadata; the
//! caller establishes verifier authority independently of this decoder.

use std::io::{Read, Write};
use std::sync::Arc;

use serde_json::Value;

use crate::application_records::{ApplicationRecords, PrivateSnapshot};
use crate::identity::{native_application_word_count, ApplicationIdentity};

use super::sealed::{decode_native_application_records, prepared_record_counts};
use super::{LoadedPerApplicationPackage, PackageError, GOLDILOCKS_MODULUS};

mod value;

const MAGIC: &[u8; 8] = b"NSFPREP1";

// Upper bound derived for the earlier 4,685,394-column key: the native
// reference had 32,037,533 array/u64 nodes and W+L <= 7,701. It still bounds
// the wide package at the approved 4,708,530-column capacity: the largest
// ordinary envelope (W+L = 2,851,939) has 26,540,836 nodes (assembly node test).
// Derivation: nightstream/tests/evidence/prepared-package-20260922/fixed-source-bound.md.
const NATIVE_REFERENCE_NODES: usize = 32_037_533;
const MAX_FIXED_SOURCE_NODES: usize = NATIVE_REFERENCE_NODES + 7_696;
// Compact numeric-array JSON needs at most 20 decimal digits and one separator
// per node; array delimiters fit this bound as well.
const MAX_FIXED_SOURCE_BYTES: u64 = 21 * MAX_FIXED_SOURCE_NODES as u64;

fn check_source_bytes(length: u64) -> Result<(), PackageError> {
    if length > MAX_FIXED_SOURCE_BYTES {
        return Err(PackageError::Invalid(
            "prepared fixed envelope exceeds compiler byte bound",
        ));
    }
    Ok(())
}

pub(super) fn snapshot(fixed: &Value) -> Result<PrivateSnapshot, PackageError> {
    value::validate(fixed, MAX_FIXED_SOURCE_NODES)?;
    let source = PrivateSnapshot::create(|output| {
        serde_json::to_writer(output, fixed)?;
        Ok(())
    })?;
    check_source_bytes(source.len())?;
    Ok(source)
}

fn write_word(output: &mut impl Write, value: u64) -> Result<(), PackageError> {
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_word(input: &mut impl Read) -> Result<u64, PackageError> {
    let mut word = [0; size_of::<u64>()];
    input.read_exact(&mut word)?;
    Ok(u64::from_le_bytes(word))
}

fn read_digest(input: &mut impl Read) -> Result<[u64; 4], PackageError> {
    let mut digest = [0; 4];
    for value in &mut digest {
        *value = read_word(input)?;
        if *value >= GOLDILOCKS_MODULUS {
            return Err(PackageError::NonCanonicalField {
                location: "prepared package identity",
                value: *value,
            });
        }
    }
    Ok(digest)
}

pub(super) fn write(
    mut output: impl Write,
    source: &PrivateSnapshot,
    records: &ApplicationRecords,
    structural_identifier: [u64; 4],
    application_identity: &ApplicationIdentity,
) -> Result<(), PackageError> {
    check_source_bytes(source.len())?;
    let (application_digest, word_count) = application_identity.cached_parts();
    let word_count =
        u64::try_from(word_count).map_err(|_| PackageError::Invalid("prepared application identity length"))?;
    output.write_all(MAGIC)?;
    write_word(&mut output, source.len())?;
    for word in structural_identifier.into_iter().chain(application_digest) {
        write_word(&mut output, word)?;
    }
    write_word(&mut output, word_count)?;
    source.copy_to(&mut output)?;
    records.write_to(&mut output)
}

/// Read reusable execution data without replaying whole-circuit identities.
/// This validates structure and record syntax. The cached identity components
/// do not authenticate the artifact or authorize it as a verifier's circuit.
pub fn load_compiled_application_package(mut input: impl Read) -> Result<LoadedPerApplicationPackage, PackageError> {
    let mut magic = [0; MAGIC.len()];
    input.read_exact(&mut magic)?;
    if magic != *MAGIC {
        return Err(PackageError::Invalid("prepared package format"));
    }
    let source_bytes = read_word(&mut input)?;
    check_source_bytes(source_bytes)?;
    let structural_identifier = read_digest(&mut input)?;
    let application_digest = read_digest(&mut input)?;
    let word_count = usize::try_from(read_word(&mut input)?)
        .map_err(|_| PackageError::Invalid("prepared application identity length"))?;
    let source = PrivateSnapshot::create(|output| {
        let copied = std::io::copy(&mut input.by_ref().take(source_bytes), output)?;
        if copied != source_bytes {
            return Err(PackageError::Invalid("truncated prepared fixed envelope"));
        }
        Ok(())
    })?;
    let fixed = source.with_reader(|input| value::decode(input, MAX_FIXED_SOURCE_NODES))?;
    let (rows, recipes) = prepared_record_counts(&fixed)?;
    let records = Arc::new(ApplicationRecords::read_from(&mut input, rows, recipes)?);
    let mut trailing = [0];
    if input.read(&mut trailing)? != 0 {
        return Err(PackageError::Invalid("prepared package has trailing bytes"));
    }
    let decoded = decode_native_application_records(&fixed, records)?;
    if word_count != native_application_word_count(decoded.application())? {
        return Err(PackageError::Invalid("prepared application identity word count"));
    }
    let application_identity = ApplicationIdentity::from_cached(application_digest, word_count)?;
    let package = decoded.bind(structural_identifier, application_identity, source);
    let _ = package.production_verifier_binding()?;
    Ok(package)
}

#[cfg(test)]
#[path = "../../tests/unit/prepared_package.rs"]
mod tests;
