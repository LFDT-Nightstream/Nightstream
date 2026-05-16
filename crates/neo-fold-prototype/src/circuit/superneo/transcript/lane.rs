//! Owns transcript-lane linear-combination algebra.

use core::cmp::Ordering;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, Index, SynthesisError, Variable};
use ff::Field;

use crate::spartan_backend::SpartanF;

#[derive(Clone)]
pub(super) struct TranscriptLane {
    pub(super) terms: Vec<(Variable, SpartanF)>,
    pub(super) constant: SpartanF,
    pub(super) value: SpartanF,
}

impl TranscriptLane {
    pub(super) fn from_allocated(value: AllocatedNum<SpartanF>, native: SpartanF) -> Self {
        Self {
            terms: vec![(value.get_variable(), SpartanF::ONE)],
            constant: SpartanF::ZERO,
            value: native,
        }
    }

    pub(super) fn from_variable(variable: Variable, native: SpartanF) -> Self {
        Self {
            terms: vec![(variable, SpartanF::ONE)],
            constant: SpartanF::ZERO,
            value: native,
        }
    }

    pub(super) fn from_terms(terms: Vec<(Variable, SpartanF)>, constant: SpartanF, native: SpartanF) -> Self {
        Self {
            terms: compact_terms(terms),
            constant,
            value: native,
        }
    }

    pub(super) fn from_constant(native: SpartanF) -> Self {
        Self {
            terms: vec![],
            constant: native,
            value: native,
        }
    }

    pub(super) fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    pub(super) fn add(&self, other: &Self) -> Self {
        let terms = if self.terms.is_empty() {
            other.terms.clone()
        } else if other.terms.is_empty() {
            self.terms.clone()
        } else {
            merge_compact_terms(&self.terms, &other.terms)
        };
        Self {
            terms,
            constant: self.constant + other.constant,
            value: self.value + other.value,
        }
    }

    pub(super) fn lc<CS: ConstraintSystem<SpartanF>>(&self) -> bellpepper_core::LinearCombination<SpartanF> {
        let mut res = bellpepper_core::LinearCombination::zero();
        res = res + (self.constant, CS::one());
        for (v, c) in &self.terms {
            res = res + (*c, *v);
        }
        res
    }

    pub(super) fn allocate_canonical<CS: ConstraintSystem<SpartanF>>(
        &self,
        mut cs: CS,
    ) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(self.value))?;
        cs.enforce(
            || "enforce_alloc",
            |lc| lc + out.get_variable(),
            |lc| lc + CS::one(),
            |_| self.lc::<CS>(),
        );
        Ok(out)
    }
}

fn compact_terms(mut terms: Vec<(Variable, SpartanF)>) -> Vec<(Variable, SpartanF)> {
    if terms.len() <= 1 {
        return terms
            .into_iter()
            .filter(|(_, coeff)| *coeff != SpartanF::ZERO)
            .collect();
    }

    terms.sort_unstable_by(|(left, _), (right, _)| compare_variables(*left, *right));

    let mut compacted = Vec::with_capacity(terms.len());
    for (variable, coeff) in terms {
        if coeff == SpartanF::ZERO {
            continue;
        }
        if let Some((last_variable, last_coeff)) = compacted.last_mut() {
            if *last_variable == variable {
                *last_coeff += coeff;
                if *last_coeff == SpartanF::ZERO {
                    compacted.pop();
                }
                continue;
            }
        }
        compacted.push((variable, coeff));
    }
    compacted
}

fn compare_variables(left: Variable, right: Variable) -> Ordering {
    match (left.get_unchecked(), right.get_unchecked()) {
        (Index::Input(left_idx), Index::Input(right_idx)) => left_idx.cmp(&right_idx),
        (Index::Aux(left_idx), Index::Aux(right_idx)) => left_idx.cmp(&right_idx),
        (Index::Input(_), Index::Aux(_)) => Ordering::Less,
        (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
    }
}

fn merge_compact_terms(left: &[(Variable, SpartanF)], right: &[(Variable, SpartanF)]) -> Vec<(Variable, SpartanF)> {
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;

    while left_idx < left.len() && right_idx < right.len() {
        let (left_var, left_coeff) = left[left_idx];
        let (right_var, right_coeff) = right[right_idx];
        match compare_variables(left_var, right_var) {
            Ordering::Less => {
                merged.push((left_var, left_coeff));
                left_idx += 1;
            }
            Ordering::Greater => {
                merged.push((right_var, right_coeff));
                right_idx += 1;
            }
            Ordering::Equal => {
                let coeff = left_coeff + right_coeff;
                if coeff != SpartanF::ZERO {
                    merged.push((left_var, coeff));
                }
                left_idx += 1;
                right_idx += 1;
            }
        }
    }

    merged.extend_from_slice(&left[left_idx..]);
    merged.extend_from_slice(&right[right_idx..]);
    merged
}

pub(super) fn combine_scaled_lanes(lanes: &[(&TranscriptLane, SpartanF)]) -> TranscriptLane {
    let mut terms_len = 0usize;
    let mut constant = SpartanF::ZERO;
    let mut value = SpartanF::ZERO;
    for (lane, scalar) in lanes {
        if *scalar == SpartanF::ZERO {
            continue;
        }
        terms_len += lane.terms.len();
        constant += lane.constant * *scalar;
        value += lane.value * *scalar;
    }
    if terms_len == 0 {
        return TranscriptLane::from_constant(value);
    }

    let mut terms = Vec::with_capacity(terms_len);
    for (lane, scalar) in lanes {
        if *scalar == SpartanF::ZERO || lane.terms.is_empty() {
            continue;
        }
        if *scalar == SpartanF::ONE {
            terms.extend(lane.terms.iter().copied());
        } else {
            terms.extend(
                lane.terms
                    .iter()
                    .map(|(variable, coeff)| (*variable, *coeff * *scalar)),
            );
        }
    }

    TranscriptLane {
        terms: compact_terms(terms),
        constant,
        value,
    }
}
