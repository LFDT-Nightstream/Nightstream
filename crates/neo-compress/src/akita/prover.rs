//! Akita setup, commitment and opening for one size of the key polynomial.
//! The setup is a function of Akita's fixed public seed, so it carries no
//! secret and no caller input.

use std::sync::Arc;

use akita_params::BasisMode;
use akita_pcs::{
    AkitaCommitmentScheme, AkitaProverSetup, CommitOutput, CommitmentHandle, CpuBackend, DensePoly, GroupContext,
    OpeningClaims, PolynomialGroupClaims, SelectedProverOpeningData,
};
use akita_types::{CommittedGroup, SetupPrefixVerifierRegistry};
use neo_math::K;

use super::{bytes_of, commitment_bytes, extension, failed, profile, row, Base, Config, Extension, Polynomial};
use super::{Schedules, Shape};
use crate::Error;

pub(crate) struct Prover {
    scheme: AkitaCommitmentScheme<Config>,
    setup: AkitaProverSetup<Base>,
    backend: Arc<CpuBackend<Base, Extension>>,
}

/// One committed polynomial: its public commitment and the prover's handle,
/// which holds the polynomial.
#[derive(Clone)]
pub(crate) struct Committed {
    commitment: CommittedGroup<Base>,
    handle: CommitmentHandle<Base, Extension>,
    pub(crate) bytes: Vec<u8>,
}

impl Prover {
    pub(crate) fn new(schedules: &Schedules) -> Result<Self, Error> {
        let scheme = AkitaCommitmentScheme::new(schedules.catalog()?);
        let max_vars = schedules
            .rows
            .iter()
            .fold(schedules.key_vars, |max, &(witness, inverse)| {
                max.max(witness).max(inverse)
            });
        let setup = scheme.setup_prover(max_vars, 3).map_err(failed)?;
        let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).map_err(failed)?);
        Ok(Self { scheme, setup, backend })
    }

    /// Commit `polynomial` of a proof with `shape`: its evaluation table is
    /// `prefix`, then zeros. The row of `shape` fixes the profile.
    pub(crate) fn commit(&self, shape: Shape, polynomial: Polynomial, prefix: Vec<Base>) -> Result<Committed, Error> {
        let vars = match polynomial {
            Polynomial::Key => shape.key,
            Polynomial::Witness => shape.witness,
            Polynomial::Inverse => shape.inverse,
        };
        let schedules = self.scheme.schedules();
        let profile = profile(row(schedules, shape)?, polynomial);
        let source = self
            .backend
            .import_source(vec![DensePoly::from_field_prefix(vars, prefix).map_err(failed)?])
            .map_err(failed)?;
        let CommitOutput {
            committed_group,
            private_handle,
        } = self
            .backend
            .commit(schedules, &source, GroupContext::explicit(profile))
            .map_err(failed)?;
        Ok(Committed {
            bytes: commitment_bytes(&committed_group)?,
            commitment: committed_group,
            handle: private_handle,
        })
    }

    /// The bytes of the commitments of prefixes of Akita's setup that the
    /// rows of this prover read.
    pub(crate) fn setup_prefixes(&self) -> Result<Vec<u8>, Error> {
        let mut registry = SetupPrefixVerifierRegistry::new(self.setup.expanded.descriptor().setup_seed.clone());
        for slot in self.setup.prefix_slots.verifier_slots() {
            registry.insert(slot).map_err(failed)?;
        }
        bytes_of(&registry)
    }

    /// One proof that each committed polynomial has its value at its point,
    /// where `point[i]` binds bit `i` of the cell index. The polynomials are
    /// in the order of `Polynomial`. `session` binds the proof to the
    /// caller's transcript.
    pub(crate) fn open(&self, openings: [(Committed, &[K], K); 3], session: &[u8]) -> Result<Vec<u8>, Error> {
        let mut handles = Vec::with_capacity(openings.len());
        let mut groups = Vec::with_capacity(openings.len());
        for (committed, point, value) in openings {
            let point: Vec<_> = point.iter().copied().map(extension).collect();
            groups
                .push(PolynomialGroupClaims::new(point, vec![extension(value)], committed.commitment).map_err(failed)?);
            handles.push(committed.handle);
        }
        let claims = OpeningClaims::from_groups(groups).map_err(failed)?;
        let opening =
            SelectedProverOpeningData::from_committed_claims::<Config>(claims, handles, self.scheme.schedules())
                .map_err(failed)?;
        self.scheme
            .batched_prove(&self.setup, opening, &self.backend, session, BasisMode::Lagrange)
            .map_err(failed)
    }
}
