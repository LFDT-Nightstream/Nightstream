//! The boundary to the Akita commitment scheme (ePrint 2026/1983): its
//! fields, its schedule catalog and its commitment bytes. `prover` holds the
//! calls that need Akita's prover crates, and `verifier` the calls that do not.

pub(crate) mod prover;
pub(crate) mod verifier;

use akita_config::{proof_optimized::fp64, TrustedScheduleCatalog};
use akita_params::ChunkedWitnessCfg;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::CommittedGroup;
use jolt_field::Ring;
use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::Error;

pub(crate) type Config = fp64::Dense;
/// Akita's `F_q`, with the modulus of `neo_math::F`.
pub(crate) type Base = fp64::Field;
/// Akita's `F_q[u]/(u² − 2)`.
pub(crate) type Extension = fp64::ExtensionField;

/// The catalog `artifacts/schedules/fp64_dense.aks` of the pinned Akita
/// revision. Akita checks each row against the policy of `Config` when it
/// decodes the bytes.
const SCHEDULES: &[u8] = include_bytes!("../artifacts/fp64_dense.aks");

/// `c` with `2c² = 7`. `K` is `F_q[u]/(u² − 7)`, so `a + b·u ↦ a + (b·c)·u`
/// is the isomorphism of `K` onto `Extension`.
const BRIDGE: u64 = 6_880_439_088_830_499_533;
const _: () = {
    let q = neo_math::field::Q as u128;
    assert!((BRIDGE as u128 * BRIDGE as u128 % q) * 2 % q == 7);
};

pub(crate) fn base(value: F) -> Base {
    Base::from_u64(value.as_canonical_u64())
}

pub(crate) fn extension(value: K) -> Extension {
    let [real, imaginary] = value.as_coeffs();
    Extension::new(base(real), base(imaginary * F::from_u64(BRIDGE)))
}

pub(crate) fn failed(error: akita_pcs::AkitaError) -> Error {
    Error::Akita(error.to_string())
}

fn catalog() -> Result<TrustedScheduleCatalog<Config>, Error> {
    TrustedScheduleCatalog::from_artifact_bytes(SCHEDULES, ChunkedWitnessCfg::default_non_chunked()).map_err(failed)
}

pub(crate) fn commitment_bytes(commitment: &CommittedGroup<Base>) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    commitment
        .serialize_compressed(&mut bytes)
        .map_err(|error| Error::Akita(error.to_string()))?;
    Ok(bytes)
}

/// The commitment that `bytes` encode. Bytes that are not the encoding of
/// their commitment fail, so a commitment has one encoding.
pub(crate) fn commitment(bytes: &[u8]) -> Result<CommittedGroup<Base>, Error> {
    let commitment = CommittedGroup::<Base>::deserialize_compressed(bytes, &())
        .map_err(|_| Error::Rejected("an Akita commitment does not decode"))?;
    if commitment_bytes(&commitment)? != bytes {
        return Err(Error::Rejected("an Akita commitment is not canonical"));
    }
    Ok(commitment)
}
