use std::sync::Arc;

use akita_pcs::{
    AkitaCommitmentScheme, AkitaError, AkitaProverSetup, AkitaSerialize, CommitOutput, CommitmentHandle, CpuBackend,
    DensePoly, FieldRunPoly, GroupContext, OpeningClaims, PolynomialGroupClaims, SelectedProverOpeningData,
    SourceHandle,
};
use jolt_field::Ring;
use neo_math::K;
use prover_akita_config as config;
use prover_akita_params::{self as params, BasisMode, GroupCommitPhaseParams};
use prover_akita_types::{CommittedGroup, SetupPrefixVerifierRegistry};
use rayon::prelude::*;

use super::{extension, failed, on_large_stacks, Base, Extension, Polynomial, Schedules, Shape};
#[allow(
    clippy::duplicate_mod,
    reason = "The prover and verifier use schedule types from different Akita revisions."
)]
#[path = "schedule.rs"]
mod schedule;
use crate::Error;
use schedule::{catalog, profile, row, Config};

pub(crate) struct Prover {
    scheme: AkitaCommitmentScheme<Config>,
    setup: AkitaProverSetup<Base>,
    backend: Arc<CpuBackend<Base, Extension>>,
}

#[derive(Clone)]
pub(crate) struct Committed {
    commitment: CommittedGroup<Base>,
    handle: CommitmentHandle<Base, Extension>,
    pub(crate) bytes: Vec<u8>,
}

impl Prover {
    pub(crate) fn new(schedules: &Schedules) -> Result<Self, Error> {
        on_large_stacks(|| {
            let scheme = AkitaCommitmentScheme::new(catalog(schedules)?);
            let max_vars = schedules
                .rows
                .iter()
                .fold(schedules.key_vars, |max, &(witness, inverse)| {
                    max.max(witness).max(inverse)
                });
            let setup = scheme.setup_prover(max_vars, 3).map_err(failed)?;
            let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).map_err(failed)?);
            Ok(Self { scheme, setup, backend })
        })
    }

    pub(crate) fn commit(&self, shape: Shape, polynomial: Polynomial, prefix: Vec<Base>) -> Result<Committed, Error> {
        let vars = match polynomial {
            Polynomial::Key => shape.key,
            Polynomial::Witness => shape.witness,
            Polynomial::Inverse => shape.inverse,
        };
        self.commit_source(&self.backend, shape, polynomial, |backend, profile| {
            let source = DensePoly::from_field_prefix(vars, prefix)?;
            let source = if polynomial == Polynomial::Key {
                prepare_key(source, profile)?
            } else {
                source
            };
            backend.import_source(vec![source])
        })
    }

    pub(crate) fn commit_key_words(
        &self,
        shape: Shape,
        words: Arc<Vec<u64>>,
    ) -> Result<(Committed, Option<DensePoly<Base>>), Error> {
        let Ok(source) = DensePoly::from_shared_canonical_u64_prefix(shape.key, Arc::clone(&words)) else {
            let prefix = words.par_iter().copied().map(Base::from_u64).collect();
            return Ok((self.commit(shape, Polynomial::Key, prefix)?, None));
        };
        let mut retained = None;
        let committed = self.commit_source(&self.backend, shape, Polynomial::Key, |backend, profile| {
            let source = prepare_key(source, profile)?;
            retained = Some(source.clone());
            backend.import_source(vec![source])
        })?;
        Ok((committed, retained))
    }

    pub(crate) fn commit_signed_witness(
        &self,
        shape: Shape,
        start: usize,
        (positive, negative): (&[u64], &[u64]),
        count: (usize, Base),
    ) -> Result<Committed, Error> {
        self.commit_source(&self.backend, shape, Polynomial::Witness, |backend, _| {
            backend.import_source(vec![FieldRunPoly::from_signed_span(
                shape.witness,
                start,
                positive.to_vec(),
                negative.to_vec(),
                count,
            )?])
        })
    }

    pub(crate) fn commit_inverse_runs(&self, shape: Shape, runs: Vec<(usize, Base)>) -> Result<Committed, Error> {
        self.commit_source(&self.backend, shape, Polynomial::Inverse, |backend, _| {
            backend.import_source(vec![FieldRunPoly::new(shape.inverse, runs)?])
        })
    }

    pub(crate) fn commit_lookup_columns(
        &self,
        shape: Shape,
        width: usize,
        codes: Arc<Vec<u16>>,
        tables: Vec<Vec<Base>>,
        metal: bool,
    ) -> Result<Committed, Error> {
        let device = metal.then(|| self.backend.with_metal_kernels());
        let backend = device.as_ref().unwrap_or(&self.backend);
        self.commit_source(backend, shape, Polynomial::Inverse, |backend, _| {
            backend.import_source(vec![FieldRunPoly::from_lookup_columns(
                shape.inverse,
                width,
                codes,
                tables,
            )?])
        })
    }

    fn commit_source(
        &self,
        backend: &CpuBackend<Base, Extension>,
        shape: Shape,
        polynomial: Polynomial,
        import: impl FnOnce(
                &CpuBackend<Base, Extension>,
                &GroupCommitPhaseParams,
            ) -> Result<SourceHandle<Base, Extension>, AkitaError>
            + Send,
    ) -> Result<Committed, Error> {
        let schedules = self.scheme.schedules();
        let profile = profile(row(schedules, shape)?, polynomial);
        on_large_stacks(|| {
            let source = import(backend, profile).map_err(failed)?;
            let CommitOutput {
                committed_group,
                private_handle,
            } = backend
                .commit(schedules, &source, GroupContext::explicit(profile))
                .map_err(failed)?;
            Ok(Committed {
                bytes: bytes_of(&committed_group)?,
                commitment: committed_group,
                handle: private_handle,
            })
        })
    }

    pub(crate) fn trim_commitment_transforms(&self) -> Result<(), Error> {
        self.backend.trim_caches().map(|_| ()).map_err(failed)
    }

    pub(crate) fn setup_prefixes(&self) -> Result<Vec<u8>, Error> {
        let mut registry = SetupPrefixVerifierRegistry::new(self.setup.expanded.descriptor().setup_seed.clone());
        for slot in self.setup.prefix_slots.verifier_slots() {
            registry.insert(slot).map_err(failed)?;
        }
        bytes_of(&registry)
    }

    /// `point[i]` binds bit `i` of the cell index.
    pub(crate) fn open(
        &self,
        openings: [(Committed, &[K], K); 3],
        session: &[u8],
        metal: bool,
    ) -> Result<Vec<u8>, Error> {
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
        let device = metal.then(|| self.backend.with_metal_kernels());
        let backend = device.as_ref().unwrap_or(&self.backend);
        on_large_stacks(|| {
            self.scheme
                .batched_prove(&self.setup, opening, backend, session, BasisMode::Lagrange)
                .map_err(failed)
        })
    }
}

fn prepare_key(source: DensePoly<Base>, profile: &GroupCommitPhaseParams) -> Result<DensePoly<Base>, AkitaError> {
    if profile.inner.matrix.ring_dimension() == 2048 && profile.inner_digits() == params::GadgetDigits::new(16, 4) {
        source.with_fixed_fold_ntt_2048(profile.blocks.positions_per_block)
    } else {
        Ok(source)
    }
}

fn bytes_of(value: &impl AkitaSerialize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    value.serialize_compressed(&mut bytes).map_err(failed)?;
    Ok(bytes)
}
