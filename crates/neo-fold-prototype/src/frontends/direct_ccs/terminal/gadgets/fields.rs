//! Owns direct-CCS field/domain conversion helpers shared by terminal circuits.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::finalize::digest32_as_fields;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::claim::packed_bytes_field_values;

pub(super) fn push_constant_spartan_fields<I>(
    field_terms: &mut Vec<Vec<(bellpepper_core::Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: I,
) where
    I: IntoIterator<Item = SpartanF>,
{
    for value in values {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }
}

pub(crate) fn direct_domain_fields(domain: &[u8]) -> Vec<F> {
    direct_domain_spartan_fields(domain)
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect()
}

pub(super) fn direct_domain_spartan_fields(domain: &[u8]) -> Vec<SpartanF> {
    packed_bytes_field_values(domain)
}

pub(crate) fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(field_to_spartan)
}

pub(crate) fn u64_halves_as_spartan_fields(value: u64) -> [SpartanF; 2] {
    [
        SpartanF::from_canonical_u64(value & 0xffff_ffff),
        SpartanF::from_canonical_u64(value >> 32),
    ]
}

pub(crate) fn field_to_spartan(value: F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(crate) fn spartan_zero() -> SpartanF {
    SpartanF::from_canonical_u64(0)
}

pub(super) fn spartan_one() -> SpartanF {
    SpartanF::from_canonical_u64(1)
}
