//! Owns in-circuit checks for the public Construction-2 `u_i` boundary.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::rv32im::main_relation_spartan::{
    construction2_commitment_digest_circuit, construction2_public_boundary_digest_circuit, enforce_digest_eq,
};
use crate::spartan_backend::SpartanF;

pub(super) struct Construction2PublicInputs<'a> {
    pub(super) fresh_instance_digest: &'a [AllocatedNum<SpartanF>; 4],
    pub(super) commitment_digest: &'a [AllocatedNum<SpartanF>; 4],
    pub(super) commitment_d: &'a AllocatedNum<SpartanF>,
    pub(super) commitment_kappa: &'a AllocatedNum<SpartanF>,
    pub(super) commitment_data: &'a [AllocatedNum<SpartanF>],
    pub(super) x_i: &'a [AllocatedNum<SpartanF>; 4],
}

pub(super) fn enforce_digest_eq_when_non_base<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &[AllocatedNum<SpartanF>; 4],
    rhs: &[AllocatedNum<SpartanF>; 4],
    chunk_count_in_halves: &[AllocatedNum<SpartanF>; 2],
    label: &str,
) {
    let two32 = SpartanF::from_canonical_u64(1u64 << 32);
    for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + lhs.get_variable() - rhs.get_variable(),
            |lc| lc + chunk_count_in_halves[0].get_variable() + (two32, chunk_count_in_halves[1].get_variable()),
            |lc| lc,
        );
    }
}

fn enforce_allocated_num_eq_constant<CS: ConstraintSystem<SpartanF>>(
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

pub(super) fn enforce_construction2_public_boundary<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public: Construction2PublicInputs<'_>,
    expected_commitment_d: u64,
    expected_commitment_kappa: u64,
    x_out_digest: &[AllocatedNum<SpartanF>; 4],
) -> Result<(), SynthesisError> {
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| "commitment_d_eq"),
        public.commitment_d,
        SpartanF::from_canonical_u64(expected_commitment_d),
        "commitment_d_eq",
    );
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| "commitment_kappa_eq"),
        public.commitment_kappa,
        SpartanF::from_canonical_u64(expected_commitment_kappa),
        "commitment_kappa_eq",
    );
    let expected_commitment_digest = construction2_commitment_digest_circuit(
        &mut cs.namespace(|| "expected_commitment_digest"),
        public.commitment_d,
        public.commitment_kappa,
        public.commitment_data,
        "expected_commitment_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "commitment_digest_eq"),
        public.commitment_digest,
        &expected_commitment_digest,
        "commitment_digest_eq",
    )?;
    let expected_fresh_instance_digest = construction2_public_boundary_digest_circuit(
        &mut cs.namespace(|| "expected_fresh_instance_digest"),
        public.commitment_digest,
        x_out_digest,
        "expected_fresh_instance_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "fresh_instance_digest_eq"),
        public.fresh_instance_digest,
        &expected_fresh_instance_digest,
        "fresh_instance_digest_eq",
    )?;
    enforce_digest_eq(&mut cs.namespace(|| "x_i_eq"), public.x_i, x_out_digest, "x_i_eq")
}
