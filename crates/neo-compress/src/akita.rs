//! The boundary to the Akita commitment scheme (ePrint 2026/1983): its
//! fields, its schedule rows and its commitment bytes. `prover` holds each
//! call that needs Akita's prover crates. `verifier` and this module need none.
//!
//! A proof opens three polynomials in one Akita proof: the key polynomial,
//! `W` and `V`. A schedule row of Akita fixes the commitment profile of each
//! and the schedule of the proof. The rows are files of Akita's planner
//! (`scripts/akita_schedules.sh`), one file for each size of the key
//! polynomial.

pub(crate) mod prover;
pub(crate) mod verifier;

use akita_config::{proof_optimized::fp64, RecursiveCommitmentConfig, ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_params::{ChunkedWitnessCfg, GroupCommitPhaseParams};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::{sample_akita_setup_seed, CommittedGroup, SetupPrefixVerifierRegistry};
use jolt_field::Ring;
use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::key::ROW_VARS;
use crate::layout::LANE_VARS;
use crate::Error;

/// Akita's dense configuration over `F_q` with the schedules that offload
/// Akita's own setup (ePrint 2026/1983 section 9): the verifier reads a
/// short prefix of that setup and the commitment of a longer one.
pub(crate) type Config = RecursiveCommitmentConfig<fp64::Dense>;
/// Akita's `F_q`, with the modulus of `neo_math::F`.
pub(crate) type Base = fp64::Field;
/// Akita's `F_q[u]/(u² − 2)`.
pub(crate) type Extension = fp64::ExtensionField;

/// The schedule rows for one size of the key polynomial. Akita checks each
/// row against the policy of `Config` when it decodes the file.
pub(crate) struct Schedules {
    pub(crate) key_vars: usize,
    /// The variables of `W` and of `V` of each row, in increasing order.
    pub(crate) rows: &'static [(usize, usize)],
    bytes: &'static [u8],
    /// The bytes of the verification key of this size. The key is a
    /// function of public seeds, so a test computes it again.
    pub(crate) key: &'static [u8],
}

pub(crate) const SCHEDULES: [Schedules; 2] = [
    Schedules {
        key_vars: 20,
        rows: &[(16, 16)],
        bytes: include_bytes!("../artifacts/key20.aks"),
        key: include_bytes!("../artifacts/key20.vk"),
    },
    Schedules {
        key_vars: 30,
        rows: &[(26, 28), (27, 28)],
        bytes: include_bytes!("../artifacts/key30.aks"),
        key: include_bytes!("../artifacts/key30.vk"),
    },
];

/// The variables of the three polynomials of one proof: one schedule row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) key: usize,
    pub(crate) witness: usize,
    pub(crate) inverse: usize,
}

/// The polynomials of a proof, in the order of Akita's groups: the key
/// polynomial and `W` are committed before `V`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Polynomial {
    Key,
    Witness,
    Inverse,
}

impl Schedules {
    /// The rows of the key polynomial of `blocks` blocks of each key row.
    pub(crate) fn of_blocks(blocks: usize) -> Option<&'static Self> {
        SCHEDULES
            .iter()
            .find(|schedules| schedules.blocks() == blocks)
    }

    /// The key polynomial holds this many blocks of each key row.
    pub(crate) fn blocks(&self) -> usize {
        1 << (self.key_vars - LANE_VARS - ROW_VARS)
    }

    pub(crate) fn catalog(&self) -> Result<TrustedScheduleCatalog<Config>, Error> {
        TrustedScheduleCatalog::from_artifact_bytes(self.bytes, ChunkedWitnessCfg::default_non_chunked())
            .map_err(failed)
    }

    /// The first row: each row of one file has the same profile for the key
    /// polynomial, so the key is committed under this row.
    pub(crate) fn first(&self) -> Shape {
        let (witness, inverse) = self.rows[0];
        Shape {
            key: self.key_vars,
            witness,
            inverse,
        }
    }
}

/// The row of `catalog` that opens one polynomial of each size of `shape`.
pub(crate) fn row(catalog: &TrustedScheduleCatalog<Config>, shape: Shape) -> Result<&ResolvedScheduleRow, Error> {
    let single = |profile: &GroupCommitPhaseParams, vars: usize| {
        profile.group.num_vars() == vars && profile.group.num_polynomials() == 1
    };
    catalog
        .rows()
        .find(|row| {
            let profiles = row.profiles();
            single(&profiles.final_group, shape.inverse)
                && matches!(&profiles.precommitteds[..], [key, witness]
                    if single(key, shape.key) && single(witness, shape.witness))
        })
        .ok_or(Error::Layout("no Akita schedule row has these polynomials"))
}

/// The commitment profile that `row` fixes for `polynomial`.
pub(crate) fn profile(row: &ResolvedScheduleRow, polynomial: Polynomial) -> &GroupCommitPhaseParams {
    let profiles = row.profiles();
    match polynomial {
        Polynomial::Key => &profiles.precommitteds[0],
        Polynomial::Witness => &profiles.precommitteds[1],
        Polynomial::Inverse => &profiles.final_group,
    }
}

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

pub(crate) fn failed(error: impl std::fmt::Display) -> Error {
    Error::Akita(error.to_string())
}

fn bytes_of(value: &impl AkitaSerialize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    value.serialize_compressed(&mut bytes).map_err(failed)?;
    Ok(bytes)
}

pub(crate) fn commitment_bytes(commitment: &CommittedGroup<Base>) -> Result<Vec<u8>, Error> {
    bytes_of(commitment)
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

/// The commitments of prefixes of Akita's setup that `bytes` encode, with
/// one encoding each. They belong to Akita's public seed.
pub(crate) fn setup_prefixes(bytes: &[u8]) -> Result<SetupPrefixVerifierRegistry<Base>, Error> {
    let registry = SetupPrefixVerifierRegistry::<Base>::deserialize_compressed(bytes, &())
        .map_err(|_| Error::Rejected("the commitments of Akita's setup do not decode"))?;
    if bytes_of(&registry)? != bytes || *registry.setup_seed() != sample_akita_setup_seed() {
        return Err(Error::Rejected("the commitments of Akita's setup are not canonical"));
    }
    Ok(registry)
}

#[cfg(test)]
#[path = "../tests/unit/akita.rs"]
mod tests;
