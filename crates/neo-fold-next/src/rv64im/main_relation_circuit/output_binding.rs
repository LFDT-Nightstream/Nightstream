//! Owns circuit checks that bind Π_CCS outputs to authoritative fresh inputs and carried ME inputs.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{enforce_k_eq, KNumVar};

#[derive(Clone)]
pub struct FreshCcsClaimVar {
    pub c_data: Vec<AllocatedNum<SpartanF>>,
    pub x: Vec<AllocatedNum<SpartanF>>,
    pub m_in: usize,
}

pub fn alloc_fresh_ccs_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    fresh: &CcsClaim<Commitment, F>,
) -> Result<FreshCcsClaimVar, SynthesisError> {
    Ok(FreshCcsClaimVar {
        c_data: alloc_f_slice(cs, &fresh.c.data, "c_data")?,
        x: alloc_f_slice(cs, &embedded_fresh_x_values(fresh), "x")?,
        m_in: fresh.m_in,
    })
}

pub fn enforce_me_outputs_against_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    _params: &NeoParams,
    fresh_claims: &[FreshCcsClaimVar],
    me_inputs: &[CeClaimVar],
    me_outputs: &[CeClaimVar],
    r_prime: &[KNumVar],
    r_prime_values: &[K],
    s_col_prime: &[KNumVar],
    s_col_prime_values: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    let d_pad = D.next_power_of_two();
    if me_outputs.len() != fresh_claims.len() + me_inputs.len()
        || r_prime.len() != r_prime_values.len()
        || s_col_prime.len() != s_col_prime_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, output) in me_outputs.iter().enumerate() {
        if output.r_values != r_prime_values
            || output.s_col_values != s_col_prime_values
            || output.y_zcol.len() != d_pad
            || output.y_zcol_values.len() != d_pad
            || output.y_ring.len() < structure.t()
            || output.ct.len() < structure.t()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &output.r, r_prime, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &output.s_col, s_col_prime, &format!("{label}_s_col_{idx}"))?;
        for matrix_idx in 0..structure.t() {
            if output.y_ring_values[matrix_idx].len() < D {
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        if idx < fresh_claims.len() {
            enforce_fresh_output_binding(cs, &fresh_claims[idx], output, &format!("{label}_fresh_{idx}"))?;
        } else {
            let me_idx = idx - fresh_claims.len();
            enforce_me_input_output_binding(cs, &me_inputs[me_idx], output, &format!("{label}_me_input_{me_idx}"))?;
        }
    }

    Ok(())
}

fn enforce_fresh_output_binding<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    fresh: &FreshCcsClaimVar,
    output: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if output.m_in != fresh.m_in
        || output.x_rows != D
        || output.x_cols != fresh.m_in
        || output.c_data.len() != fresh.c_data.len()
        || output.x.len() != fresh.x.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, expected) in fresh.c_data.iter().enumerate() {
        cs.enforce(
            || format!("{label}_commitment_{idx}"),
            |lc| lc + output.c_data[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected.get_variable(),
        );
    }

    for idx in 0..output.x.len() {
        cs.enforce(
            || format!("{label}_x_{idx}"),
            |lc| lc + output.x[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + fresh.x[idx].get_variable(),
        );
    }

    Ok(())
}

fn enforce_me_input_output_binding<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    input: &CeClaimVar,
    output: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if output.c_data.len() != input.c_data.len()
        || output.m_in != input.m_in
        || output.x_rows != input.x_rows
        || output.x_cols != input.x_cols
        || output.x.len() != input.x.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for idx in 0..input.c_data.len() {
        cs.enforce(
            || format!("{label}_commitment_{idx}"),
            |lc| lc + output.c_data[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + input.c_data[idx].get_variable(),
        );
    }
    for idx in 0..input.x.len() {
        cs.enforce(
            || format!("{label}_x_{idx}"),
            |lc| lc + output.x[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + input.x[idx].get_variable(),
        );
    }
    Ok(())
}

fn enforce_equal_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left: &[KNumVar],
    right: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        enforce_k_eq(cs, lhs, rhs, &format!("{label}_{idx}"));
    }
    Ok(())
}

pub(crate) fn embedded_fresh_x_values(fresh: &CcsClaim<Commitment, F>) -> Vec<F> {
    let mut out = vec![F::ZERO; D * fresh.m_in];
    for col in 0..fresh.m_in {
        let lane = col % D;
        for row in 0..D {
            if row == lane {
                out[row * fresh.m_in + col] = fresh.x[col];
            }
        }
    }
    out
}

fn alloc_f_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[F],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect()
}
