use crate::spartan_backend::SpartanF;
use bellpepper_core::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::Field;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{ALPHABET_LEN, CANDIDATE_WORDS_PER_RHO, RHO_REJECTION_SLACK, U16_MOD5_WEIGHTS};

#[derive(Clone)]
pub(super) struct GoldilocksCoeffCandidateVar {
    pub(super) coeff: AllocatedNum<SpartanF>,
    pub(super) coeff_value: F,
    pub(super) reject_bit: AllocatedBit,
    pub(super) reject_value: bool,
}

pub(super) fn map_u16_bits_to_goldilocks_candidate<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    word_bits: &[Boolean],
    word_value: u16,
    label: &str,
) -> Result<(AllocatedNum<SpartanF>, F, AllocatedBit, bool), SynthesisError> {
    if word_bits.len() != 16 {
        return Err(SynthesisError::Unsatisfiable);
    }

    let popcount_value = word_value.count_ones() as u64;
    let reject_value = popcount_value == 16;
    let reject_bit = AllocatedBit::alloc(cs.namespace(|| format!("{label}_reject_bit")), Some(reject_value))?;
    let popcount_low_bits = alloc_small_bits(
        cs.namespace(|| format!("{label}_popcount_low_bits")),
        popcount_value & 0xF,
        4,
        &format!("{label}_popcount_low_bits"),
    )?;
    enforce_reject_bit_from_popcount(
        cs.namespace(|| format!("{label}_reject_check")),
        word_bits,
        &reject_bit,
        &popcount_low_bits,
        &format!("{label}_reject_check"),
    )?;

    let weighted_sum_value = U16_MOD5_WEIGHTS
        .iter()
        .enumerate()
        .fold(0u64, |acc, (bit_idx, weight)| {
            if ((word_value >> bit_idx) & 1) != 0 {
                acc + weight
            } else {
                acc
            }
        });
    let quotient_value = weighted_sum_value / ALPHABET_LEN as u64;
    let remainder_value = weighted_sum_value % ALPHABET_LEN as u64;

    let quotient_bits = alloc_small_bits(
        cs.namespace(|| format!("{label}_quotient_bits")),
        quotient_value,
        4,
        &format!("{label}_quotient_bits"),
    )?;
    let remainder_bits = alloc_small_bits(
        cs.namespace(|| format!("{label}_remainder_bits")),
        remainder_value,
        3,
        &format!("{label}_remainder_bits"),
    )?;

    // quotient_value lives in [0, 8]; if the high bit is set, all lower bits must be zero.
    for low_idx in 0..3 {
        enforce_bits_not_both_true(
            cs.namespace(|| format!("{label}_quotient_range_{low_idx}")),
            &quotient_bits[3],
            &quotient_bits[low_idx],
            &format!("{label}_quotient_range_{low_idx}"),
        )?;
    }
    // remainder_value lives in [0, 4]; if bit 2 is set, bits 0 and 1 must be zero.
    for low_idx in 0..2 {
        enforce_bits_not_both_true(
            cs.namespace(|| format!("{label}_remainder_range_{low_idx}")),
            &remainder_bits[2],
            &remainder_bits[low_idx],
            &format!("{label}_remainder_range_{low_idx}"),
        )?;
    }

    enforce_mod5_weighted_sum(
        cs.namespace(|| format!("{label}_mod5")),
        word_bits,
        &quotient_bits,
        &remainder_bits,
        &format!("{label}_mod5"),
    )?;

    let coeff_value = F::from_i64(remainder_value as i64 - 2);
    let coeff = AllocatedNum::alloc(cs.namespace(|| format!("{label}_coeff")), || {
        Ok(SpartanF::from_canonical_u64(coeff_value.as_canonical_u64()))
    })?;
    cs.enforce(
        || format!("{label}_coeff_relation"),
        |lc| lc + coeff.get_variable() + (SpartanF::from_canonical_u64(2), CS::one()),
        |lc| lc + CS::one(),
        |lc| {
            let mut acc = lc;
            let mut scale = SpartanF::ONE;
            for bit in &remainder_bits {
                acc = acc + (scale, bit.get_variable());
                scale += scale;
            }
            acc
        },
    );

    Ok((coeff, coeff_value, reject_bit, reject_value))
}

pub(super) fn compact_first_accepted_goldilocks_coeffs<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    candidates: &[GoldilocksCoeffCandidateVar],
    label: &str,
) -> Result<(Vec<AllocatedNum<SpartanF>>, Vec<F>), SynthesisError> {
    if candidates.len() != CANDIDATE_WORDS_PER_RHO {
        return Err(SynthesisError::Unsatisfiable);
    }

    let accepted_values = candidates
        .iter()
        .filter(|candidate| !candidate.reject_value)
        .map(|candidate| candidate.coeff_value)
        .take(D)
        .collect::<Vec<_>>();
    if accepted_values.len() != D {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut reject_prefix_counts = Vec::with_capacity(candidates.len() + 1);
    reject_prefix_counts.push(0usize);
    for candidate in candidates {
        let next = reject_prefix_counts
            .last()
            .copied()
            .expect("reject prefix counts should be seeded")
            + usize::from(candidate.reject_value);
        reject_prefix_counts.push(next);
    }

    let mut reject_prefix_vars = Vec::with_capacity(candidates.len() + 1);
    let prefix_zero = AllocatedNum::alloc(cs.namespace(|| format!("{label}_prefix_0")), || Ok(SpartanF::ZERO))?;
    cs.enforce(
        || format!("{label}_prefix_0_const"),
        |lc| lc + prefix_zero.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    reject_prefix_vars.push(prefix_zero);
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        let next_value = SpartanF::from_canonical_u64(reject_prefix_counts[candidate_idx + 1] as u64);
        let next = AllocatedNum::alloc(cs.namespace(|| format!("{label}_prefix_{}", candidate_idx + 1)), || {
            Ok(next_value)
        })?;
        cs.enforce(
            || format!("{label}_prefix_step_{candidate_idx}"),
            |lc| lc + reject_prefix_vars[candidate_idx].get_variable() + candidate.reject_bit.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + next.get_variable(),
        );
        reject_prefix_vars.push(next);
    }

    let mut coeffs = Vec::with_capacity(D);
    for slot_idx in 0..D {
        let mut selectors = Vec::with_capacity(RHO_REJECTION_SLACK + 1);
        let mut selector_values = Vec::with_capacity(RHO_REJECTION_SLACK + 1);
        for offset in 0..=RHO_REJECTION_SLACK {
            let candidate_idx = slot_idx + offset;
            let selector = AllocatedBit::alloc(
                cs.namespace(|| format!("{label}_slot_{slot_idx}_select_{offset}")),
                Some(reject_prefix_counts[candidate_idx] == offset && !candidates[candidate_idx].reject_value),
            )?;
            cs.enforce(
                || format!("{label}_slot_{slot_idx}_prefix_match_{offset}"),
                |lc| lc + selector.get_variable(),
                |lc| {
                    lc + reject_prefix_vars[candidate_idx].get_variable()
                        + (SpartanF::ZERO - SpartanF::from_canonical_u64(offset as u64), CS::one())
                },
                |lc| lc,
            );
            cs.enforce(
                || format!("{label}_slot_{slot_idx}_reject_gate_{offset}"),
                |lc| lc + selector.get_variable(),
                |lc| lc + candidates[candidate_idx].reject_bit.get_variable(),
                |lc| lc,
            );
            selectors.push(selector);
            selector_values
                .push(reject_prefix_counts[candidate_idx] == offset && !candidates[candidate_idx].reject_value);
        }

        cs.enforce(
            || format!("{label}_slot_{slot_idx}_one_hot"),
            |lc| lc + CS::one(),
            |lc| lc,
            |_| {
                selectors
                    .iter()
                    .fold(LinearCombination::zero(), |lc, selector| lc + selector.get_variable())
                    - (SpartanF::ONE, CS::one())
            },
        );

        let coeff_value = accepted_values[slot_idx];
        let coeff = AllocatedNum::alloc(cs.namespace(|| format!("{label}_slot_{slot_idx}_coeff_alloc")), || {
            Ok(SpartanF::from_canonical_u64(coeff_value.as_canonical_u64()))
        })?;
        let mut selected_terms = Vec::with_capacity(selectors.len());
        for (offset, selector) in selectors.iter().enumerate() {
            let candidate_idx = slot_idx + offset;
            let product_value = if selector_values[offset] {
                candidates[candidate_idx].coeff_value
            } else {
                F::ZERO
            };
            let product = AllocatedNum::alloc(
                cs.namespace(|| format!("{label}_slot_{slot_idx}_select_product_{offset}")),
                || Ok(SpartanF::from_canonical_u64(product_value.as_canonical_u64())),
            )?;
            cs.enforce(
                || format!("{label}_slot_{slot_idx}_select_product_eq_{offset}"),
                |lc| lc + selector.get_variable(),
                |lc| lc + candidates[candidate_idx].coeff.get_variable(),
                |lc| lc + product.get_variable(),
            );
            selected_terms.push(product);
        }
        cs.enforce(
            || format!("{label}_slot_{slot_idx}_sum"),
            |lc| {
                selected_terms
                    .iter()
                    .fold(lc, |acc, term| acc + term.get_variable())
            },
            |lc| lc + CS::one(),
            |lc| lc + coeff.get_variable(),
        );
        coeffs.push(coeff);
    }

    Ok((coeffs, accepted_values))
}

fn alloc_small_bits<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: u64,
    width: usize,
    label: &str,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    let mut out = Vec::with_capacity(width);
    for bit_idx in 0..width {
        let bit_value = ((value >> bit_idx) & 1) != 0;
        out.push(AllocatedBit::alloc(
            cs.namespace(|| format!("{label}_{bit_idx}")),
            Some(bit_value),
        )?);
    }
    Ok(out)
}

fn enforce_reject_bit_from_popcount<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    word_bits: &[Boolean],
    reject_bit: &AllocatedBit,
    popcount_low_bits: &[AllocatedBit],
    label: &str,
) -> Result<(), SynthesisError> {
    let mut relation = LinearCombination::zero();
    for bit in word_bits {
        relation = relation + &bit.lc(CS::one(), SpartanF::ONE);
    }
    let mut coeff = SpartanF::ONE;
    for bit in popcount_low_bits {
        relation = relation - (coeff, bit.get_variable());
        coeff += coeff;
    }
    relation = relation - (SpartanF::from_canonical_u64(16), reject_bit.get_variable());
    cs.enforce(
        || format!("{label}_relation"),
        |lc| lc + CS::one(),
        |lc| lc,
        |_| relation,
    );
    Ok(())
}

fn enforce_bits_not_both_true<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    a: &AllocatedBit,
    b: &AllocatedBit,
    label: &str,
) -> Result<(), SynthesisError> {
    cs.enforce(
        || format!("{label}_not_both_true"),
        |lc| lc + a.get_variable(),
        |lc| lc + b.get_variable(),
        |lc| lc,
    );
    Ok(())
}

fn enforce_mod5_weighted_sum<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    word_bits: &[Boolean],
    quotient_bits: &[AllocatedBit],
    remainder_bits: &[AllocatedBit],
    label: &str,
) -> Result<(), SynthesisError> {
    let mut relation = LinearCombination::zero();
    for (bit, weight) in word_bits.iter().zip(U16_MOD5_WEIGHTS.iter()) {
        relation = relation + &bit.lc(CS::one(), SpartanF::from_canonical_u64(*weight));
    }

    let mut q_coeff = SpartanF::from_canonical_u64(ALPHABET_LEN as u64);
    for bit in quotient_bits {
        relation = relation - (q_coeff, bit.get_variable());
        q_coeff += q_coeff;
    }

    let mut r_coeff = SpartanF::ONE;
    for bit in remainder_bits {
        relation = relation - (r_coeff, bit.get_variable());
        r_coeff += r_coeff;
    }

    cs.enforce(
        || format!("{label}_relation"),
        |lc| lc + CS::one(),
        |lc| lc,
        |_| relation,
    );
    Ok(())
}
