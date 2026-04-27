//! Owns low-level circuit helpers for the terminal committed `F'` R2 check.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem};
use neo_math::F;
use p3_field::PrimeField64;

use crate::rv64im::ivc_snark::SpartanF;

pub(super) fn native_to_spartan(value: &F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(super) fn enforce_boolean_allocated<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    label: &str,
) {
    cs.enforce(
        || format!("{label}_boolean"),
        |lc| lc + value.get_variable(),
        |lc| lc + value.get_variable() - CS::one(),
        |lc| lc,
    );
}
