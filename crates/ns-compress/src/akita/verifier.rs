use std::sync::Arc;

use akita_config::{self as config, ResolvedScheduleRow};
use akita_params::{self as params, verifier_setup_matrix_capacity_for_schedule, BasisMode, SetupMode};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup,
    CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, SetupPrefixVerifierRegistry,
};
use akita_verifier::{AkitaVerifier, SetupPolicy};
use neo_math::K;

use super::{extension, failed, on_large_stacks, Base, Polynomial, Schedules, Shape, SCHEDULES};
#[path = "schedule.rs"]
pub(crate) mod schedule;
use crate::Error;
use schedule::{catalog, profile, row, Config};

pub(crate) struct Opening<'a> {
    pub(crate) commitment: &'a [u8],
    pub(crate) point: &'a [K],
    pub(crate) value: K,
}

pub(crate) struct Verifier {
    row: ResolvedScheduleRow,
    verifier: AkitaVerifier<Config>,
}

impl Verifier {
    pub(crate) fn new(shape: Shape, prefixes: &[u8]) -> Result<Self, Error> {
        let schedules = SCHEDULES
            .iter()
            .find(|schedules| schedules.key_vars == shape.key)
            .ok_or(Error::Layout("no Akita schedule row has a key of this size"))?;
        let catalog = catalog(schedules)?;
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

pub(crate) fn commitment(bytes: &[u8]) -> Result<CommittedGroup<Base>, Error> {
    let commitment = CommittedGroup::<Base>::deserialize_compressed(bytes, &())
        .map_err(|_| Error::Rejected("an Akita commitment does not decode"))?;
    if bytes_of(&commitment)? != bytes {
        return Err(Error::Rejected("an Akita commitment is not canonical"));
    }
    Ok(commitment)
}

pub(crate) fn setup_prefixes(bytes: &[u8]) -> Result<SetupPrefixVerifierRegistry<Base>, Error> {
    let registry = SetupPrefixVerifierRegistry::<Base>::deserialize_compressed(bytes, &())
        .map_err(|_| Error::Rejected("the commitments of Akita's setup do not decode"))?;
    if bytes_of(&registry)? != bytes || *registry.setup_seed() != sample_akita_setup_seed() {
        return Err(Error::Rejected("the commitments of Akita's setup are not canonical"));
    }
    Ok(registry)
}

pub(crate) fn bytes_of(value: &impl AkitaSerialize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    value.serialize_compressed(&mut bytes).map_err(failed)?;
    Ok(bytes)
}
