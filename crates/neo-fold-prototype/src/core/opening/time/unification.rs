//! Sumcheck unification proof for reduced time-opening groups.

use neo_math::{from_complex, KExtensions, K};
use neo_reductions::error::PiCcsError;
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::opening::TimeOpeningUnificationProof;

use super::digest::{append_k_vec, append_point, opening_domain_tag, opening_source_tag};
use super::types::{OpeningReduction, OpeningReductionGroup};

#[inline]
fn ceil_log2_at_least_1(n: usize) -> usize {
    let need = n.max(1).next_power_of_two();
    (need.trailing_zeros() as usize).max(1)
}

fn group_value(group: &OpeningReductionGroup) -> K {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify_group_value");
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_group_value_meta",
        &[
            opening_domain_tag(group.domain),
            group.sources.len() as u64,
            group.point.len() as u64,
            group.claim_indices.len() as u64,
            group.coefficients.len() as u64,
        ],
    );
    let source_tags: Vec<u64> = group
        .sources
        .iter()
        .map(|&source| opening_source_tag(source))
        .collect();
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_group_value_sources",
        &source_tags,
    );
    append_point(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unify_group_value_point",
        &group.point,
    );
    let claim_indices_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_group_value_indices",
        &claim_indices_u64,
    );
    append_k_vec(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unify_group_value_coefficients",
        &group.coefficients,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_group_value_group_digest",
        &group.group_digest,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_group_value_reduced_digest",
        &group.reduced_digest,
    );
    let re = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_group_value/re");
    let im = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_group_value/im");
    from_complex(re, im)
}

fn bind_opening_unification_statement(
    tr: &mut Poseidon2Transcript,
    reduction: &OpeningReduction,
    ell_sel: usize,
    values: &[K],
) {
    tr.append_message(b"neo.fold.next/time_opening/reduction_unify_bind", &[]);
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_bind_reduction_digest",
        &reduction.digest,
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_bind_meta",
        &[
            reduction.groups.len() as u64,
            ell_sel as u64,
            reduction.can_unify as u64,
            opening_domain_tag(reduction.unified_domain),
            reduction.unified_point.len() as u64,
        ],
    );
    append_point(
        tr,
        b"neo.fold.next/time_opening/reduction_unify_bind_unified_point",
        &reduction.unified_point,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_bind_unified_digest",
        &reduction.unified_digest,
    );
    for (group_idx, (group, value)) in reduction.groups.iter().zip(values.iter()).enumerate() {
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_meta",
            &[
                group_idx as u64,
                opening_domain_tag(group.domain),
                group.sources.len() as u64,
                group.point.len() as u64,
                group.claim_indices.len() as u64,
                group.coefficients.len() as u64,
            ],
        );
        let source_tags: Vec<u64> = group
            .sources
            .iter()
            .map(|&source| opening_source_tag(source))
            .collect();
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_sources",
            &source_tags,
        );
        append_point(
            tr,
            b"neo.fold.next/time_opening/reduction_unify_bind_group_point",
            &group.point,
        );
        let claim_indices_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_indices",
            &claim_indices_u64,
        );
        append_k_vec(
            tr,
            b"neo.fold.next/time_opening/reduction_unify_bind_group_coefficients",
            &group.coefficients,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_digest",
            &group.group_digest,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_bind_reduced_digest",
            &group.reduced_digest,
        );
        tr.append_fields(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_value",
            &value.as_coeffs(),
        );
    }
}

struct GroupSelectorOracle {
    values: Vec<K>,
    ell_sel: usize,
    prefix: Vec<K>,
}

impl GroupSelectorOracle {
    fn new(values: Vec<K>, ell_sel: usize) -> Self {
        Self {
            values,
            ell_sel,
            prefix: Vec::with_capacity(ell_sel),
        }
    }

    #[inline]
    fn bit_at(index: usize, bit: usize) -> bool {
        ((index >> bit) & 1usize) == 1
    }

    #[inline]
    fn bit_weight(bit: bool, x: K) -> K {
        if bit {
            x
        } else {
            K::ONE - x
        }
    }

    fn eval_at_point(values: &[K], r: &[K]) -> K {
        let mut acc = K::ZERO;
        for (group_idx, value) in values.iter().enumerate() {
            let mut weight = K::ONE;
            for (bit_idx, &rv) in r.iter().enumerate() {
                weight *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), rv);
            }
            acc += weight * *value;
        }
        acc
    }
}

impl RoundOracle for GroupSelectorOracle {
    fn num_rounds(&self) -> usize {
        self.ell_sel.saturating_sub(self.prefix.len())
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.prefix.len() >= self.ell_sel {
            return vec![K::ZERO; points.len()];
        }
        let round_idx = self.prefix.len();
        let mut out = vec![K::ZERO; points.len()];
        for (group_idx, value) in self.values.iter().enumerate() {
            let mut prefix_weight = K::ONE;
            for (bit_idx, &bound) in self.prefix.iter().enumerate() {
                prefix_weight *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), bound);
            }
            for (slot, &x) in out.iter_mut().zip(points.iter()) {
                *slot += prefix_weight * Self::bit_weight(Self::bit_at(group_idx, round_idx), x) * *value;
            }
        }
        out
    }

    fn fold(&mut self, r: K) {
        self.prefix.push(r);
    }
}

pub(super) fn prove_opening_unification(
    reduction: &OpeningReduction,
) -> Result<TimeOpeningUnificationProof, PiCcsError> {
    if reduction.groups.is_empty() {
        return Ok(TimeOpeningUnificationProof::default());
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify");
    bind_opening_unification_statement(&mut tr, reduction, ell_sel, &values);
    let claimed_sum = values
        .iter()
        .copied()
        .fold(K::ZERO, |acc, value| acc + value);
    let mut oracle = GroupSelectorOracle::new(values, ell_sel);
    let (round_polys, r_unify) = run_sumcheck_prover(&mut tr, &mut oracle, claimed_sum)
        .map_err(|err| PiCcsError::ProtocolError(format!("time-opening unification prove failed: {err}")))?;
    Ok(TimeOpeningUnificationProof {
        claimed_sum,
        round_polys,
        r_unify,
    })
}

pub(super) fn verify_opening_unification(
    reduction: &OpeningReduction,
    proof: &TimeOpeningUnificationProof,
) -> Result<(), PiCcsError> {
    if reduction.groups.is_empty() {
        if proof.claimed_sum == K::ZERO && proof.round_polys.is_empty() && proof.r_unify.is_empty() {
            return Ok(());
        }
        return Err(PiCcsError::ProtocolError(
            "time-opening unification proof must be empty when there are no groups".into(),
        ));
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    if proof.round_polys.len() != ell_sel {
        return Err(PiCcsError::ProtocolError(format!(
            "time-opening unification round count {} != expected {ell_sel}",
            proof.round_polys.len()
        )));
    }
    let expected_sum = values
        .iter()
        .copied()
        .fold(K::ZERO, |acc, value| acc + value);
    if proof.claimed_sum != expected_sum {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification claimed_sum mismatch".into(),
        ));
    }
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify");
    bind_opening_unification_statement(&mut tr, reduction, ell_sel, &values);
    let (r_unify, final_value, ok) = verify_sumcheck_rounds(&mut tr, 1, proof.claimed_sum, &proof.round_polys);
    if !ok {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification sumcheck verification failed".into(),
        ));
    }
    if proof.r_unify != r_unify {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification selector point mismatch".into(),
        ));
    }
    let expected_final = GroupSelectorOracle::eval_at_point(&values, &proof.r_unify);
    if final_value != expected_final {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification final value mismatch".into(),
        ));
    }
    Ok(())
}
