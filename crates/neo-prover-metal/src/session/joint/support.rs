//! Host-side encoders and field helpers for the one-joint oracle.

use neo_math::{KExtensions, Rq, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{EQUALITY_CHUNK_BITS, EQUALITY_CHUNK_VALUES};

pub(super) fn joint_term_metadata(
    structure: &neo_ccs::CcsStructure<F>,
) -> Result<(Vec<u64>, Vec<u64>), neo_reductions::PiCcsError> {
    let mut headers = Vec::with_capacity(structure.f.terms().len() * 3);
    let mut variables = Vec::new();
    for term in structure.f.terms() {
        let start = variables.len() / 2;
        for (matrix, &exponent) in term.exps.iter().enumerate() {
            if exponent != 0 {
                variables.extend_from_slice(&[matrix as u64, exponent as u64]);
            }
        }
        headers.extend_from_slice(&[
            term.coeff.as_canonical_u64(),
            start as u64,
            (variables.len() / 2 - start) as u64,
        ]);
    }
    Ok((headers, variables))
}

pub(super) fn equality_suffix_chunk_words(point: &[K], chunks_per_round: usize) -> Vec<u64> {
    let mut values = Vec::with_capacity(point.len() * chunks_per_round * EQUALITY_CHUNK_VALUES);
    for round in 0..point.len() {
        for chunk in 0..chunks_per_round {
            let start = round + 1 + chunk * EQUALITY_CHUNK_BITS;
            let end = (start + EQUALITY_CHUNK_BITS).min(point.len());
            for index in 0..EQUALITY_CHUNK_VALUES {
                let mut value = K::ONE;
                for (bit, &coordinate) in point[start.min(point.len())..end].iter().enumerate() {
                    value *= if index & (1 << bit) == 0 {
                        K::ONE - coordinate
                    } else {
                        coordinate
                    };
                }
                values.push(value);
            }
        }
    }
    k_words(&values)
}

pub(super) fn equality_round_affine(point: &[K], prefix: K, round: usize) -> (K, K) {
    let high = prefix * point[round];
    let low = prefix * (K::ONE - point[round]);
    (low, high - low)
}

pub(super) fn restrict_equality_prefix(prefix: K, point: K, challenge: K) -> K {
    prefix * ((K::ONE - point) * (K::ONE - challenge) + point * challenge)
}

pub(super) fn k_power(value: K, exponent: usize) -> K {
    (0..exponent).fold(K::ONE, |power, _| power * value)
}

pub(super) fn k_words(values: &[K]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|value| {
            let (real, imaginary) = value.to_limbs_u64();
            [real, imaginary]
        })
        .collect()
}

pub(super) fn ring_words(values: &[Rq; D]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|ring| ring.0.iter().map(PrimeField64::as_canonical_u64))
        .collect()
}

pub(super) fn nonempty(values: &[u64]) -> &[u64] {
    if values.is_empty() {
        &[0]
    } else {
        values
    }
}
