use super::config::{proof_optimized::fp64, RecursiveCommitmentConfig, ResolvedScheduleRow, TrustedScheduleCatalog};
use super::params::{ChunkedWitnessCfg, GroupCommitPhaseParams};
use super::{failed, Polynomial, Schedules, Shape};
use crate::Error;

pub(super) type Config = RecursiveCommitmentConfig<fp64::Dense>;

pub(crate) fn catalog(schedules: &Schedules) -> Result<TrustedScheduleCatalog<Config>, Error> {
    TrustedScheduleCatalog::from_artifact_bytes(schedules.bytes, ChunkedWitnessCfg::default_non_chunked())
        .map_err(failed)
}

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

pub(crate) fn profile(row: &ResolvedScheduleRow, polynomial: Polynomial) -> &GroupCommitPhaseParams {
    let profiles = row.profiles();
    match polynomial {
        Polynomial::Key => &profiles.precommitteds[0],
        Polynomial::Witness => &profiles.precommitteds[1],
        Polynomial::Inverse => &profiles.final_group,
    }
}
