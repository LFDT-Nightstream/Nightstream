//! Akita setup, commitment and opening. The setup is a function of Akita's
//! fixed public seed, so it carries no secret and no caller input.

use std::sync::Arc;

use akita_params::BasisMode;
use akita_pcs::{
    AkitaCommitmentScheme, AkitaProverSetup, CommitOutput, CommitmentHandle, CpuBackend, DensePoly, GroupContext,
    OpeningClaims, PolynomialGroupClaims, SelectedProverOpeningData,
};
use akita_types::{AkitaVerifierSetup, CommittedGroup};
use neo_math::K;

use super::{catalog, commitment_bytes, extension, failed, Base, Config, Extension};
use crate::Error;

pub(crate) struct Prover {
    scheme: AkitaCommitmentScheme<Config>,
    setup: AkitaProverSetup<Base>,
    backend: Arc<CpuBackend<Base, Extension>>,
}

/// One committed polynomial: its public commitment and the prover's handle.
pub(crate) struct Committed {
    commitment: CommittedGroup<Base>,
    handle: CommitmentHandle<Base, Extension>,
    pub(crate) bytes: Vec<u8>,
}

fn scheme_and_setup(max_vars: usize) -> Result<(AkitaCommitmentScheme<Config>, AkitaProverSetup<Base>), Error> {
    let scheme = AkitaCommitmentScheme::new(catalog()?);
    let setup = scheme.setup_prover(max_vars, 1).map_err(failed)?;
    Ok((scheme, setup))
}

/// The verifier's setup for polynomials of at most `max_vars` variables.
/// Akita derives it from the prover's setup, so this is the one place where
/// verification reads the prover crates.
pub(crate) fn verifier_setup(max_vars: usize) -> Result<AkitaVerifierSetup<Base>, Error> {
    let (scheme, setup) = scheme_and_setup(max_vars)?;
    scheme.setup_verifier(&setup).map_err(failed)
}

impl Prover {
    pub(crate) fn new(max_vars: usize) -> Result<Self, Error> {
        let (scheme, setup) = scheme_and_setup(max_vars)?;
        let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).map_err(failed)?);
        Ok(Self { scheme, setup, backend })
    }

    /// Commit the polynomial of `vars` variables whose evaluation table is
    /// `prefix`, then zeros.
    pub(crate) fn commit(&self, vars: usize, prefix: Vec<Base>) -> Result<Committed, Error> {
        let polynomial = DensePoly::from_field_prefix(vars, prefix).map_err(failed)?;
        let source = self
            .backend
            .import_source(vec![polynomial])
            .map_err(failed)?;
        let CommitOutput {
            committed_group,
            private_handle,
        } = self
            .backend
            .commit(
                self.scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .map_err(failed)?;
        Ok(Committed {
            bytes: commitment_bytes(&committed_group)?,
            commitment: committed_group,
            handle: private_handle,
        })
    }

    /// The proof that the committed polynomial is `value` at `point`, where
    /// `point[i]` binds bit `i` of the cell index. `session` binds the proof
    /// to the caller's transcript.
    pub(crate) fn open(&self, committed: Committed, point: &[K], value: K, session: &[u8]) -> Result<Vec<u8>, Error> {
        let point: Vec<_> = point.iter().copied().map(extension).collect();
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            vec![extension(value)],
            committed.commitment,
        )
        .map_err(failed)?])
        .map_err(failed)?;
        let opening = SelectedProverOpeningData::from_committed_claims::<Config>(
            claims,
            vec![committed.handle],
            self.scheme.schedules(),
        )
        .map_err(failed)?;
        self.scheme
            .batched_prove(&self.setup, opening, &self.backend, session, BasisMode::Lagrange)
            .map_err(failed)
    }
}
