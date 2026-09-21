//! Builds a circuit from the pinned verifier blueprint and a local Rust application.
//! The resulting binding is verifier configuration; it is never read from a proof.

mod application;
mod connect;
mod manifest;
mod source;
mod wire;

use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, load_prepared_application_value, LoadedPerApplicationPackage, PackageError,
    Stage1VerifierBinding,
};
use serde_json::Value;
use thiserror::Error;

use crate::application::ApplicationCircuit;
use manifest::{Counts, Manifest};

#[derive(Debug, Error)]
pub enum AssemblyError {
    #[error("invalid assembly data: {0}")]
    Invalid(&'static str),
    #[error("assembly dimension overflow")]
    Overflow,
    #[error("invalid assembly JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Package(#[from] PackageError),
}

fn word(value: &Value) -> Result<usize, AssemblyError> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(AssemblyError::Invalid("expected index"))
}

/// Prepare an application against the selected verifier, profile and setup.
/// The reference must pass the existing package and verification-key pins.
pub fn prepare(
    reference_bytes: &[u8],
    application: &ApplicationCircuit,
) -> Result<(LoadedPerApplicationPackage, Stage1VerifierBinding), AssemblyError> {
    std::thread::scope(|scope| {
        // Reference authorization and candidate identity have independent
        // Poseidon2 preimages. Neither result authorizes the other.
        let reference = scope.spawn(|| load_poseidon2_hash_chain_v1_package(reference_bytes).map(drop));
        let candidate = prepare_application(reference_bytes, application);
        reference
            .join()
            .map_err(|_| AssemblyError::Invalid("reference validation worker failed"))??;
        candidate
    })
}

fn prepare_application(
    reference_bytes: &[u8],
    application: &ApplicationCircuit,
) -> Result<(LoadedPerApplicationPackage, Stage1VerifierBinding), AssemblyError> {
    let manifest = Manifest::parse(include_bytes!("../../artifacts/shared-verifier-v1.json"))?;
    let reference: wire::Envelope = serde_json::from_slice(reference_bytes)?;
    let value = assemble(reference, &manifest, application)?;
    let package = load_prepared_application_value(value)?;
    let binding = package.production_verifier_binding()?;
    Ok((package, binding))
}

fn assemble(
    mut reference: wire::Envelope,
    manifest: &Manifest,
    application: &ApplicationCircuit,
) -> Result<Value, AssemblyError> {
    manifest.check_reference(&reference)?;
    let counts = Counts::of(application);
    manifest.check_dimensions(counts)?;
    let actual = application::plan(application, manifest)?;
    connect::matrix(&mut reference, manifest, counts)?;
    connect::assignment(&mut reference, manifest, counts)?;
    source::replace_application(&mut reference, actual, manifest, counts)?;
    Ok(serde_json::to_value(reference)?)
}

#[cfg(test)]
#[path = "../../tests/assembly_internal/encoding.rs"]
mod encoding_tests;
