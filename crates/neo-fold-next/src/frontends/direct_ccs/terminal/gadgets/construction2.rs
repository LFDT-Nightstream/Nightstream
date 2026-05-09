//! Construction-2 input-instance checks for the direct terminal circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_math::D;

use crate::construction2::{Construction2EncodedPublicInput, Construction2FreshInstance};
use crate::spartan_backend::SpartanF;

use super::super::public_io::enforce_digest_eq_constant;
use super::fields::{digest32_as_spartan_fields, field_to_spartan};

pub(crate) fn enforce_direct_construction2_input_u_i<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    input_u_i: &Construction2FreshInstance,
    expected_x_i: &Construction2EncodedPublicInput,
    chunk_count_in: u64,
    expected_kappa: usize,
) -> Result<(), SynthesisError> {
    let x_i = input_u_i.x_i().bytes();
    let x_i = digest32_as_spartan_fields(x_i)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("construction2_input_u_i_x_{idx}")), || {
                Ok(value)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_i: [AllocatedNum<SpartanF>; 4] = x_i.try_into().map_err(|_| SynthesisError::Unsatisfiable)?;
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "construction2_input_u_i_x_matches_x_in"),
        &x_i,
        expected_x_i.bytes(),
        "construction2_input_u_i_x_matches_x_in",
    )?;

    let commitment = input_u_i.commitment().commitment();
    if chunk_count_in == 0 && !input_u_i.is_canonical_zero_for(expected_kappa, input_u_i.x_i()) {
        return Err(SynthesisError::Unsatisfiable);
    }
    if chunk_count_in != 0
        && (commitment.d != D
            || commitment.kappa == 0
            || commitment
                .d
                .checked_mul(commitment.kappa)
                .map_or(true, |len| len != commitment.data.len()))
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let d = AllocatedNum::alloc(cs.namespace(|| "construction2_input_u_i_commitment_d"), || {
        Ok(SpartanF::from_canonical_u64(commitment.d as u64))
    })?;
    let kappa = AllocatedNum::alloc(cs.namespace(|| "construction2_input_u_i_commitment_kappa"), || {
        Ok(SpartanF::from_canonical_u64(commitment.kappa as u64))
    })?;
    cs.enforce(
        || "construction2_input_u_i_d",
        |lc| lc + d.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(D as u64), CS::one()),
    );
    let expected_kappa = commitment.data.len() / D;
    cs.enforce(
        || "construction2_input_u_i_kappa",
        |lc| lc + kappa.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(expected_kappa as u64), CS::one()),
    );
    for (idx, value) in commitment.data.iter().copied().enumerate() {
        let data = AllocatedNum::alloc(
            cs.namespace(|| format!("construction2_input_u_i_commitment_data_{idx}")),
            || Ok(field_to_spartan(value)),
        )?;
        if chunk_count_in == 0 {
            cs.enforce(
                || format!("construction2_x_only_u_i_data_{idx}"),
                |lc| lc + data.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            cs.enforce(
                || format!("construction2_carried_u_i_data_{idx}"),
                |lc| lc + data.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + data.get_variable(),
            );
        }
    }
    Ok(())
}
