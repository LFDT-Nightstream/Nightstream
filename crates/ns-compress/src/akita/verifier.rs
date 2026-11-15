//! Akita opening verification. It reads no prover crate: the setup is a
//! prefix of Akita's public field stream and the commitments of longer
//! prefixes that the verification key carries, and the row fixes each
//! commitment profile.

use std::sync::Arc;

use akita_config::ResolvedScheduleRow;
use akita_params::{verifier_setup_matrix_capacity_for_schedule, BasisMode, SetupMode};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup,
    GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use akita_verifier::{AkitaVerifier, SetupPolicy};
use neo_math::K;

use super::{
    commitment, extension, failed, on_large_stacks, profile, row, setup_prefixes, Config, Polynomial, Shape, SCHEDULES,
};
use crate::Error;

/// The claim that the polynomial of `commitment` is `value` at `point`.
pub(crate) struct Opening<'a> {
    pub(crate) commitment: &'a [u8],
    pub(crate) point: &'a [K],
    pub(crate) value: K,
}

/// The verifier of the proofs of one schedule row.
pub(crate) struct Verifier {
    row: ResolvedScheduleRow,
    verifier: AkitaVerifier<Config>,
}

impl Verifier {
    /// The verifier of proofs with `shape`. Its setup is the prefix of
    /// Akita's public matrix that the row reads, and `prefixes` holds the
    /// commitments of the longer prefixes that the row offloads.
    pub(crate) fn new(shape: Shape, prefixes: &[u8]) -> Result<Self, Error> {
        let catalog = SCHEDULES
            .iter()
            .find(|schedules| schedules.key_vars == shape.key)
            .ok_or(Error::Layout("no Akita schedule row has a key of this size"))?
            .catalog()?;
        let row = row(&catalog, shape)?.clone();
        let layout = row.profiles().opening_layout().map_err(failed)?;
        let elements = verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout, SetupMode::Check)
            .map_err(failed)?
            .num_field_elements;
        let seed = sample_akita_setup_seed();
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: shape.key.max(shape.witness).max(shape.inverse),
            max_num_batched_polys: 3,
            num_field_elements: elements,
            setup_seed: seed.clone(),
        };
        let verifier = on_large_stacks(|| {
            let expanded = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                derive_public_matrix_prefix(elements, &seed),
            );
            let setup =
                AkitaVerifierSetup::from_parts(Arc::new(expanded), setup_prefixes(prefixes)?).map_err(failed)?;
            AkitaVerifier::for_selection(setup, catalog, row.selection(), None, akita_verifier::SetupMode::Check)
                .map_err(failed)
        })?;
        Ok(Self { row, verifier })
    }

    /// Accept exactly a proof that each commitment, with the profile of the
    /// row, opens to its value at its point. The openings are in the order
    /// of `Polynomial`. The prover chooses no field of a profile.
    pub(crate) fn verify(&self, openings: [Opening<'_>; 3], proof: &[u8], session: &[u8]) -> Result<(), Error> {
        let order = [Polynomial::Key, Polynomial::Witness, Polynomial::Inverse];
        let mut commitments = Vec::with_capacity(openings.len());
        for (opening, polynomial) in openings.iter().zip(order) {
            let commitment = commitment(opening.commitment)?;
            if commitment.profile() != profile(&self.row, polynomial) {
                return Err(Error::Rejected("an Akita commitment has another profile"));
            }
            commitments.push(commitment);
        }
        let groups = openings
            .iter()
            .zip(&commitments)
            .map(|(opening, commitment)| {
                let point: Vec<_> = opening.point.iter().copied().map(extension).collect();
                PolynomialGroupClaims::new(point, vec![extension(opening.value)], commitment).map_err(failed)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let claims = OpeningClaims::from_groups(groups).map_err(failed)?;
        let statement = GroupBatchStatement::new(self.row.selection(), claims).map_err(failed)?;
        on_large_stacks(|| {
            self.verifier
                .batched_verify(proof, session, statement, BasisMode::Lagrange, SetupPolicy::Check(None))
                .map_err(|_| Error::Rejected("the Akita opening fails"))
        })
    }
}
