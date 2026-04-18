//! Owns fixed-round Π_RLC rho sampling gadgets for the RV64IM main relation circuit.
//!
//! This mirrors the repo's circuit-friendly `sample_rot_rhos_n` contract:
//! transcript-bound 16-bit words are mapped into the Goldilocks strong-set
//! alphabet `[-2, -1, 0, 1, 2]` with a fixed number of transcript squeezes.

use bellpepper_core::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::Field;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::transcript::Poseidon2TranscriptCircuit;

const ALPHABET_LEN: usize = 5;
const U16S_PER_DIGEST32: usize = 16;
const DIGESTS_PER_RHO: usize = D.div_ceil(U16S_PER_DIGEST32);
const CANDIDATE_WORDS_PER_RHO: usize = DIGESTS_PER_RHO * U16S_PER_DIGEST32;
const RHO_REJECTION_SLACK: usize = CANDIDATE_WORDS_PER_RHO - D;
const U16_MOD5_WEIGHTS: [u64; 16] = [1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3];

#[derive(Clone)]
struct GoldilocksCoeffCandidateVar {
    coeff: AllocatedNum<SpartanF>,
    coeff_value: F,
    reject_bit: AllocatedBit,
    reject_value: bool,
}

#[derive(Clone)]
pub struct RotRhoVar {
    pub coeffs: Vec<AllocatedNum<SpartanF>>,
    pub coeff_values: Vec<F>,
}

#[derive(Clone)]
pub struct RotRhoMatrixVar {
    rows: usize,
    cols: usize,
    entries: Vec<AllocatedNum<SpartanF>>,
    entry_values: Vec<F>,
}

impl RotRhoMatrixVar {
    pub fn entry(&self, row: usize, col: usize) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.entries[row * self.cols + col].clone())
    }

    pub fn entry_value(&self, row: usize, col: usize) -> Result<F, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.entry_values[row * self.cols + col])
    }
}

pub fn sample_goldilocks_rot_rhos<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    let mut out = Vec::with_capacity(count);
    for rho_idx in 0..count {
        tr.append_const_fields_raw(
            cs.namespace(|| format!("{label}_rho_index_{rho_idx}")),
            &[
                SpartanF::from_canonical_u64(0),
                SpartanF::from_canonical_u64(rho_idx as u64),
            ],
        )?;
        let mut candidates = Vec::with_capacity(CANDIDATE_WORDS_PER_RHO);
        for digest_idx in 0..DIGESTS_PER_RHO {
            tr.append_const_fields_raw(
                cs.namespace(|| format!("{label}_rho_chunk_msg_{rho_idx}_{digest_idx}")),
                &[
                    SpartanF::from_canonical_u64(1),
                    SpartanF::from_canonical_u64(rho_idx as u64 + digest_idx as u64),
                ],
            )?;
            let digest = tr.digest32(cs.namespace(|| format!("{label}_rho_digest_{rho_idx}_{digest_idx}")))?;
            let digest_values = core::array::from_fn(|idx| tr.state_values()[idx]);
            let words = digest_u16_words(
                cs.namespace(|| format!("{label}_rho_words_{rho_idx}_{digest_idx}")),
                &digest,
                &digest_values,
                &format!("{label}_rho_words_{rho_idx}_{digest_idx}"),
            )?;
            for (word_idx, (word_bits, word_value)) in words.into_iter().enumerate() {
                let (coeff, coeff_value, reject_bit, reject_value) = map_u16_bits_to_goldilocks_candidate(
                    cs.namespace(|| format!("{label}_rho_coeff_{rho_idx}_{digest_idx}_{word_idx}")),
                    &word_bits,
                    word_value,
                    &format!("{label}_rho_coeff_{rho_idx}_{digest_idx}_{word_idx}"),
                )?;
                candidates.push(GoldilocksCoeffCandidateVar {
                    coeff,
                    coeff_value,
                    reject_bit,
                    reject_value,
                });
            }
        }
        let (coeffs, coeff_values) = compact_first_accepted_goldilocks_coeffs(
            cs.namespace(|| format!("{label}_rho_accept_{rho_idx}")),
            &candidates,
            &format!("{label}_rho_accept_{rho_idx}"),
        )?;
        if coeffs.len() != D || coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        out.push(RotRhoVar { coeffs, coeff_values });
    }
    Ok(out)
}

fn alloc_rot_rhos_from_coeff_values<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    rhos: &[Vec<F>],
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    let mut out = Vec::with_capacity(rhos.len());
    for (rho_idx, coeff_values) in rhos.iter().enumerate() {
        if coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut coeffs = Vec::with_capacity(D);
        for (row, value) in coeff_values.iter().copied().enumerate() {
            let coeff = alloc_affine(
                cs.namespace(|| format!("{label}_rho_{rho_idx}_coeff_{row}")),
                &[],
                SpartanF::from_canonical_u64(value.as_canonical_u64()),
            )?;
            coeffs.push(coeff);
        }
        out.push(RotRhoVar {
            coeffs,
            coeff_values: coeff_values.clone(),
        });
    }
    Ok(out)
}

pub fn alloc_zero_rot_rhos<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    alloc_rot_rhos_from_coeff_values(
        cs.namespace(|| format!("{label}_zero")),
        &vec![vec![F::ZERO; D]; count],
        label,
    )
}

pub fn materialize_goldilocks_rot_matrices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let ring = neo_reductions::RotRing::goldilocks();
    let neg_phi = ring
        .phi_coeffs
        .iter()
        .map(|coeff| F::from_i64(-(*coeff as i64)))
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(rhos.len());
    for (rho_idx, rho) in rhos.iter().enumerate() {
        if rho.coeffs.len() != D || rho.coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut entries = vec![rho.coeffs[0].clone(); D * D];
        let mut entry_values = vec![F::ZERO; D * D];

        let mut prev_col = rho.coeffs.clone();
        let mut prev_values = rho.coeff_values.clone();
        for row in 0..D {
            entries[row * D] = prev_col[row].clone();
            entry_values[row * D] = prev_values[row];
        }

        for col in 1..D {
            let tail_var = prev_col[D - 1].clone();
            let tail_value = prev_values[D - 1];
            let mut next_col = Vec::with_capacity(D);
            let mut next_values = Vec::with_capacity(D);

            let top_value = neg_phi[0] * tail_value;
            let top = alloc_affine(
                cs.namespace(|| format!("{label}_rho_{rho_idx}_col_{col}_row_0")),
                &[(
                    tail_var.clone(),
                    SpartanF::from_canonical_u64(neg_phi[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(tail_value.as_canonical_u64()),
                )],
                SpartanF::ZERO,
            )?;
            next_col.push(top.clone());
            next_values.push(top_value);

            for row in 1..D {
                let value = prev_values[row - 1] + neg_phi[row] * tail_value;
                let entry = alloc_affine(
                    cs.namespace(|| format!("{label}_rho_{rho_idx}_col_{col}_row_{row}")),
                    &[
                        (
                            prev_col[row - 1].clone(),
                            SpartanF::ONE,
                            SpartanF::from_canonical_u64(prev_values[row - 1].as_canonical_u64()),
                        ),
                        (
                            tail_var.clone(),
                            SpartanF::from_canonical_u64(neg_phi[row].as_canonical_u64()),
                            SpartanF::from_canonical_u64(tail_value.as_canonical_u64()),
                        ),
                    ],
                    SpartanF::ZERO,
                )?;
                next_col.push(entry.clone());
                next_values.push(value);
            }

            for row in 0..D {
                entries[row * D + col] = next_col[row].clone();
                entry_values[row * D + col] = next_values[row];
            }
            prev_col = next_col;
            prev_values = next_values;
        }

        out.push(RotRhoMatrixVar {
            rows: D,
            cols: D,
            entries,
            entry_values,
        });
    }
    Ok(out)
}

pub fn alloc_rot_rho_matrices_from_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    mats: &[Mat<F>],
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let mut out = Vec::with_capacity(mats.len());
    for (mat_idx, mat) in mats.iter().enumerate() {
        if mat.rows() != D || mat.cols() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut entries = Vec::with_capacity(D * D);
        let mut entry_values = Vec::with_capacity(D * D);
        for row in 0..D {
            for col in 0..D {
                let value = mat[(row, col)];
                let entry = alloc_affine(
                    cs.namespace(|| format!("{label}_mat_{mat_idx}_{row}_{col}")),
                    &[],
                    SpartanF::from_canonical_u64(value.as_canonical_u64()),
                )?;
                entries.push(entry);
                entry_values.push(value);
            }
        }
        out.push(RotRhoMatrixVar {
            rows: D,
            cols: D,
            entries,
            entry_values,
        });
    }
    Ok(out)
}

pub fn alloc_zero_rot_rho_matrices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let zero = Mat::zero(D, D, F::ZERO);
    alloc_rot_rho_matrices_from_native(&mut cs.namespace(|| format!("{label}_zero")), &vec![zero; count], label)
}

fn digest_u16_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    digest: &[AllocatedNum<SpartanF>; 4],
    digest_values: &[SpartanF; 4],
    label: &str,
) -> Result<Vec<(Vec<Boolean>, u16)>, SynthesisError> {
    let mut out = Vec::with_capacity(U16S_PER_DIGEST32);
    for (limb_idx, limb) in digest.iter().enumerate() {
        let bits = limb.to_bits_le_strict(cs.namespace(|| format!("{label}_bits_{limb_idx}")))?;
        for word_idx in 0..4 {
            let start = word_idx * 16;
            let end = start + 16;
            let word_bits = bits[start..end].to_vec();
            let limb_value = digest_values[limb_idx].to_canonical_u64();
            let word_value = ((limb_value >> start) & 0xFFFF) as u16;
            out.push((word_bits, word_value));
        }
    }
    Ok(out)
}

fn map_u16_bits_to_goldilocks_candidate<CS: ConstraintSystem<SpartanF>>(
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

fn compact_first_accepted_goldilocks_coeffs<CS: ConstraintSystem<SpartanF>>(
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

fn alloc_affine<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    constant: SpartanF,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let mut value = constant;
    for (_, coeff, term_value) in terms {
        value += *coeff * *term_value;
    }
    let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc + (constant, CS::one());
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}
