//! Owns small shared terminal constraint helpers.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem};
use neo_math::F;
use p3_field::PrimeField64;

use crate::spartan_backend::SpartanF;

pub(crate) fn native_to_spartan(value: &F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(crate) fn enforce_boolean_allocated<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bit: &AllocatedNum<SpartanF>,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + bit.get_variable(),
        |lc| lc + bit.get_variable() - CS::one(),
        |lc| lc,
    );
}

pub(crate) fn enforce_allocated_num_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    expected: SpartanF,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
}
