//! Owns the current Construction-2 input allocation used by terminal F' R2.
//!
//! This module does not prove a commitment opening and does not publish a
//! witness-image digest. The terminal committed-step proof owns `u_i.C`.

use std::time::Instant;

use bellpepper_core::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::Field;
use p3_field::PrimeField64;

use super::private_digest_inputs;
use super::synthesize_support::emit_synthesize_trace;
use crate::rv32im::construction2::Rv32imMainRecursionConstruction2FreshInstance;
use crate::rv32im::f_prime::Rv32imMainRecursionFPrimeAdvice;
use crate::rv32im::main_relation_spartan::digest32_as_spartan_fields;
use crate::spartan_backend::SpartanF;

fn alloc_current_input_fresh_instance<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    current_input: &Rv32imMainRecursionConstruction2FreshInstance,
) -> Result<(Vec<AllocatedNum<SpartanF>>, [AllocatedNum<SpartanF>; 4], Vec<Boolean>), SynthesisError> {
    let commitment_data = current_input
        .commitment()
        .commitment()
        .data
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("current_input_commitment_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_i = private_digest_inputs(
        &mut cs.namespace(|| "current_input_x_i"),
        current_input.x_i().bytes(),
        "current_input_x_i",
    )?;
    let x_i_bit_values = current_input.x_i().bit_image();
    let x_i_bits = x_i_bit_values
        .iter()
        .enumerate()
        .map(|(idx, bit)| {
            AllocatedBit::alloc(cs.namespace(|| format!("current_input_x_i_bit_{idx}")), Some(*bit == 1))
                .map(Boolean::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (limb_idx, limb) in x_i.iter().enumerate() {
        let mut acc = LinearCombination::<SpartanF>::zero();
        for bit_idx in 0..64 {
            acc = add_boolean_lc_term(
                acc,
                &x_i_bits[limb_idx * 64 + bit_idx],
                SpartanF::from_canonical_u64(1u64 << bit_idx),
            );
        }
        cs.enforce(
            || format!("current_input_x_i_limb_{limb_idx}_packs_bits"),
            |_| acc,
            |lc| lc + CS::one(),
            |lc| lc + limb.get_variable(),
        );
    }
    Ok((commitment_data, x_i, x_i_bits))
}

fn add_boolean_lc_term(lc: LinearCombination<SpartanF>, bit: &Boolean, coeff: SpartanF) -> LinearCombination<SpartanF> {
    match bit {
        Boolean::Is(bit) => lc + (coeff, bit.get_variable()),
        Boolean::Not(bit) => {
            lc + (
                coeff,
                bellpepper_core::Variable::new_unchecked(bellpepper_core::Index::Input(0)),
            ) - (coeff, bit.get_variable())
        }
        Boolean::Constant(true) => {
            lc + (
                coeff,
                bellpepper_core::Variable::new_unchecked(bellpepper_core::Index::Input(0)),
            )
        }
        Boolean::Constant(false) => lc,
    }
}

fn chunk_count_is_zero<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    chunk_count_halves: &[AllocatedNum<SpartanF>; 2],
    chunk_count_value: u64,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let is_zero = AllocatedNum::alloc(cs.namespace(|| format!("{label}_is_zero")), || {
        Ok(if chunk_count_value == 0 {
            SpartanF::ONE
        } else {
            SpartanF::ZERO
        })
    })?;
    let inverse = AllocatedNum::alloc(cs.namespace(|| format!("{label}_inverse")), || {
        if chunk_count_value == 0 {
            Ok(SpartanF::ZERO)
        } else {
            let value = SpartanF::from_canonical_u64(chunk_count_value);
            value
                .invert()
                .into_option()
                .ok_or(SynthesisError::Unsatisfiable)
        }
    })?;
    let two32 = SpartanF::from_canonical_u64(1u64 << 32);
    let chunk_count_lc = || {
        LinearCombination::<SpartanF>::zero()
            + chunk_count_halves[0].get_variable()
            + (two32, chunk_count_halves[1].get_variable())
    };
    cs.enforce(
        || format!("{label}_is_zero_bool"),
        |lc| lc + is_zero.get_variable(),
        |lc| lc + is_zero.get_variable() - CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_zero_product"),
        |_| chunk_count_lc(),
        |lc| lc + is_zero.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_inverse_product"),
        |_| chunk_count_lc(),
        |lc| lc + inverse.get_variable(),
        |lc| lc + CS::one() - is_zero.get_variable(),
    );
    Ok(is_zero)
}

fn enforce_field_eq_constant_when_selected<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    expected: SpartanF,
    selector: &AllocatedNum<SpartanF>,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + value.get_variable() - (expected, CS::one()),
        |lc| lc + selector.get_variable(),
        |lc| lc,
    );
}

fn enforce_boolean_eq_constant_when_selected<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bit: &Boolean,
    expected: bool,
    selector: &AllocatedNum<SpartanF>,
    label: &str,
) {
    let bit_lc = add_boolean_lc_term(LinearCombination::<SpartanF>::zero(), bit, SpartanF::ONE);
    let expected = if expected { SpartanF::ONE } else { SpartanF::ZERO };
    cs.enforce(
        || label,
        |_| bit_lc - (expected, CS::one()),
        |lc| lc + selector.get_variable(),
        |lc| lc,
    );
}

fn enforce_current_input_u_perp_when_base<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv32imMainRecursionFPrimeAdvice,
    current_input: &Rv32imMainRecursionConstruction2FreshInstance,
    current_input_commitment_data: &[AllocatedNum<SpartanF>],
    current_input_x_i: &[AllocatedNum<SpartanF>; 4],
    current_input_x_i_bits: &[Boolean],
    chunk_count_in_halves: &[AllocatedNum<SpartanF>; 2],
) -> Result<(), SynthesisError> {
    let full_width =
        crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_canonical_full_width(
            witness.verifier_key_fs(),
            witness.phi_side(),
        )
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let expected = crate::rv32im::construction2::build_rv32im_main_recursion_construction2_default_fresh_instance(
        witness.verifier_key_fs(),
        full_width,
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;
    if witness.chunk_count_in() == 0
        && (current_input.commitment().commitment().d != expected.commitment().commitment().d
            || current_input.commitment().commitment().kappa != expected.commitment().commitment().kappa
            || current_input_commitment_data.len() != expected.commitment().commitment().data.len()
            || current_input_x_i_bits.len() != expected.x_i().bit_image().len())
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let is_base = chunk_count_is_zero(
        &mut cs.namespace(|| "base_current_input_selector"),
        chunk_count_in_halves,
        witness.chunk_count_in(),
        "base_current_input_selector",
    )?;
    for (idx, (value, expected)) in current_input_commitment_data
        .iter()
        .zip(expected.commitment().commitment().data.iter().copied())
        .enumerate()
    {
        enforce_field_eq_constant_when_selected(
            &mut cs.namespace(|| format!("base_current_input_commitment_{idx}")),
            value,
            SpartanF::from_canonical_u64(expected.as_canonical_u64()),
            &is_base,
            &format!("base_current_input_commitment_{idx}"),
        );
    }
    for (idx, (value, expected)) in current_input_x_i
        .iter()
        .zip(digest32_as_spartan_fields(expected.x_i().bytes()))
        .enumerate()
    {
        enforce_field_eq_constant_when_selected(
            &mut cs.namespace(|| format!("base_current_input_x_i_limb_{idx}")),
            value,
            expected,
            &is_base,
            &format!("base_current_input_x_i_limb_{idx}"),
        );
    }
    for (idx, (bit, expected)) in current_input_x_i_bits
        .iter()
        .zip(expected.x_i().bit_image().into_iter())
        .enumerate()
    {
        enforce_boolean_eq_constant_when_selected(
            &mut cs.namespace(|| format!("base_current_input_x_i_bit_{idx}")),
            bit,
            expected == 1,
            &is_base,
            &format!("base_current_input_x_i_bit_{idx}"),
        );
    }
    Ok(())
}

pub(super) fn construction2_current_input_x_from_live_step<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv32imMainRecursionFPrimeAdvice,
    chunk_count_in_halves: &[AllocatedNum<SpartanF>; 2],
    trace_prefix: Option<&str>,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let started = Instant::now();
    let current_input = witness
        .construction2_input_fresh_instance()
        .ok_or(SynthesisError::Unsatisfiable)?;
    let (current_input_commitment_data, current_input_x_i, current_input_x_i_bits) =
        alloc_current_input_fresh_instance(&mut cs.namespace(|| "current_input_fresh_instance"), current_input)?;
    enforce_current_input_u_perp_when_base(
        &mut cs.namespace(|| "base_current_input_u_perp"),
        witness,
        current_input,
        &current_input_commitment_data,
        &current_input_x_i,
        &current_input_x_i_bits,
        chunk_count_in_halves,
    )?;
    emit_synthesize_trace(trace_prefix, "construction2_current_input.alloc_current_input", started);
    Ok(current_input_x_i)
}
