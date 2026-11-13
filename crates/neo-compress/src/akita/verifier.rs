//! Akita opening verification. It reads no prover crate: the setup is a
//! prefix of Akita's public field stream, and the catalog fixes each
//! commitment profile.

use std::sync::Arc;

use akita_config::{ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_params::{
    verifier_setup_matrix_capacity_for_schedule, BasisMode, PolynomialGroupLayout, ScheduleLookupKey, SetupMode,
};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup,
    GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, SetupPrefixVerifierRegistry,
};
use akita_verifier::{AkitaVerifier, SetupPolicy};
use neo_math::K;

use super::{catalog, commitment, extension, failed, Config};
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
    catalog: TrustedScheduleCatalog<Config>,
    verifier: AkitaVerifier<Config>,
}

/// The catalog row of one polynomial of `vars` variables, with no earlier
/// commitment. It fixes the commitment profile and the schedule.
fn row(catalog: &TrustedScheduleCatalog<Config>, vars: usize) -> Result<&ResolvedScheduleRow, Error> {
    catalog
        .resolve_key(&ScheduleLookupKey {
            root_accumulated_setup_prefix_len: 0,
            final_group: PolynomialGroupLayout::singleton(vars),
            precommitteds: Vec::new(),
        })
        .map_err(failed)
}

impl Verifier {
    /// The verifier of openings of polynomials of each of `vars` variables.
    /// Its setup is the public matrix that their schedules read. `Config`
    /// plans no setup-prefix commitment, so the registry is empty.
    pub(crate) fn new(vars: &[usize]) -> Result<Self, Error> {
        let catalog = catalog()?;
        let mut elements = 0;
        for &vars in vars {
            let row = row(&catalog, vars)?;
            let layout = row.profiles().opening_layout().map_err(failed)?;
            let capacity = verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout, SetupMode::Check);
            elements = elements.max(capacity.map_err(failed)?.num_field_elements);
        }
        let seed = sample_akita_setup_seed();
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: vars.iter().copied().max().unwrap_or(0),
            max_num_batched_polys: 1,
            num_field_elements: elements,
            setup_seed: seed.clone(),
        };
        let expanded = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            descriptor,
            derive_public_matrix_prefix(elements, &seed),
        );
        let setup = AkitaVerifierSetup::from_parts(Arc::new(expanded), SetupPrefixVerifierRegistry::new(seed))
            .map_err(failed)?;
        let verifier = AkitaVerifier::new(setup, catalog.clone()).map_err(failed)?;
        Ok(Self { catalog, verifier })
    }

    /// Accept exactly an opening of a commitment with the profile that the
    /// catalog gives a polynomial of `point.len()` variables. The prover
    /// chooses no field of the profile.
    pub(crate) fn verify(&self, opening: &Opening<'_>) -> Result<(), Error> {
        let commitment = commitment(opening.commitment)?;
        let row = row(&self.catalog, opening.point.len())?;
        if *commitment.profile() != row.profiles().final_group {
            return Err(Error::Rejected("an Akita commitment has another profile"));
        }
        let point: Vec<_> = opening.point.iter().copied().map(extension).collect();
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            vec![extension(opening.value)],
            &commitment,
        )
        .map_err(failed)?])
        .map_err(failed)?;
        let statement = GroupBatchStatement::new(row.selection(), claims).map_err(failed)?;
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
