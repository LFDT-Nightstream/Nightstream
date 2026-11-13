//! Akita opening verification. It reads the verifier setup and no prover
//! crate.

use akita_params::{BasisMode, CommittedGroupBatchProfile};
use akita_types::{AkitaVerifierSetup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::{AkitaVerifier, SetupPolicy};
use neo_math::K;

use super::{catalog, commitment, extension, failed, Base, Config};
use crate::Error;

/// The claim that the polynomial of `commitment` is `value` at `point`.
pub(crate) struct Opening<'a> {
    pub(crate) commitment: &'a [u8],
    pub(crate) point: &'a [K],
    pub(crate) value: K,
    pub(crate) proof: &'a [u8],
    pub(crate) session: &'a [u8],
}

pub(crate) struct Verifier {
    catalog: akita_config::TrustedScheduleCatalog<Config>,
    verifier: AkitaVerifier<Config>,
}

impl Verifier {
    pub(crate) fn new(setup: AkitaVerifierSetup<Base>) -> Result<Self, Error> {
        let catalog = catalog()?;
        let verifier = AkitaVerifier::new(setup, catalog.clone()).map_err(failed)?;
        Ok(Self { catalog, verifier })
    }

    pub(crate) fn verify(&self, opening: &Opening<'_>) -> Result<(), Error> {
        let commitment = commitment(opening.commitment)?;
        let declared = commitment.profile().group;
        if declared.num_vars() != opening.point.len() || declared.num_polynomials() != 1 {
            return Err(Error::Rejected("an Akita commitment has another shape"));
        }
        let profile = CommittedGroupBatchProfile::from_profiles(vec![*commitment.profile()]).map_err(failed)?;
        let selection = self
            .catalog
            .resolve_profiles(&profile, 0)
            .map_err(failed)?
            .selection();
        let point: Vec<_> = opening.point.iter().copied().map(extension).collect();
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            vec![extension(opening.value)],
            &commitment,
        )
        .map_err(failed)?])
        .map_err(failed)?;
        let statement = GroupBatchStatement::new(selection, claims).map_err(failed)?;
        self.verifier
            .batched_verify(
                opening.proof,
                opening.session,
                statement,
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
            )
            .map_err(|_| Error::Rejected("an Akita opening fails"))
    }
}
