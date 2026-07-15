//! Host encodings at the Metal sumcheck boundary.

use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use neo_reductions::optimized_engine::oracle::{NcDigitTableView, RowTableSnapshot};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{KWords, MetalSumcheckTrace};

pub(super) fn k_to_words(value: K) -> KWords {
    KWords::new(value.real().as_canonical_u64(), value.imag().as_canonical_u64())
}

pub(super) fn flatten_tables(tables: &[Vec<K>]) -> Vec<u64> {
    tables
        .iter()
        .flat_map(|table| {
            table.iter().flat_map(|&value| {
                let words = k_to_words(value);
                [words.c0, words.c1]
            })
        })
        .collect()
}

pub(super) fn flatten_table(table: &[K]) -> Vec<u64> {
    table
        .iter()
        .flat_map(|&value| {
            let words = k_to_words(value);
            [words.c0, words.c1]
        })
        .collect()
}

pub(super) fn signed_unit_mask_words(witnesses: &[&Mat<F>], blocks: usize) -> Option<Vec<u64>> {
    let word_count = witnesses.len().checked_mul(blocks)?.checked_mul(2)?;
    let mut words = Vec::with_capacity(word_count);
    let negative_one = F::ZERO - F::ONE;
    for witness in witnesses {
        if witness.rows() != D || witness.cols() != blocks {
            return None;
        }
        if witness
            .virtual_constant_value()
            .is_some_and(|&value| value == F::ZERO)
        {
            words.resize(words.len() + 2 * blocks, 0);
            continue;
        }
        if let Some((positive, negative)) = witness.packed_signed_unit_column_masks() {
            if positive.len() != blocks || negative.len() != blocks {
                return None;
            }
            for (&positive, &negative) in positive.iter().zip(negative) {
                words.extend_from_slice(&[positive, negative]);
            }
            continue;
        }
        for column in 0..blocks {
            let mut positive = 0u64;
            let mut negative = 0u64;
            for row in 0..D {
                let value = witness[(row, column)];
                if value == F::ONE {
                    positive |= 1u64 << row;
                } else if value == negative_one {
                    negative |= 1u64 << row;
                } else if value != F::ZERO {
                    return None;
                }
            }
            words.extend_from_slice(&[positive, negative]);
        }
    }
    Some(words)
}

pub(super) fn dense_digit_rows(table: &NcDigitTableView<'_>, len: usize) -> Option<Vec<[K; D]>> {
    match table {
        NcDigitTableView::Zero { len: table_len } if *table_len == len => Some(vec![[K::ZERO; D]; len]),
        NcDigitTableView::Lane0(values) if values.len() == len => Some(
            values
                .iter()
                .map(|&value| {
                    let mut row = [K::ZERO; D];
                    row[0] = value;
                    row
                })
                .collect(),
        ),
        NcDigitTableView::Strided { width, values } if values.len() == len * *width => Some(
            values
                .chunks_exact(*width)
                .enumerate()
                .map(|(index, chunk)| {
                    let mut row = [K::ZERO; D];
                    for (offset, &value) in chunk.iter().enumerate() {
                        row[(index * *width + offset) % D] = value;
                    }
                    row
                })
                .collect(),
        ),
        NcDigitTableView::Dense(rows) if rows.len() == len => Some(rows.to_vec()),
        _ => None,
    }
}

pub(super) fn push_k_table(tables: &mut Vec<Vec<K>>, values: &[K], expected: usize) -> Option<usize> {
    if values.len() != expected {
        return None;
    }
    let index = tables.len();
    tables.push(values.to_vec());
    Some(index)
}

pub(super) fn push_row_table(tables: &mut Vec<Vec<K>>, table: &RowTableSnapshot<'_>, expected: usize) -> Option<usize> {
    if table.real.len() != expected || table.imag.is_some_and(|imag| imag.len() != expected) {
        return None;
    }
    let values = match table.imag {
        Some(imag) => table
            .real
            .iter()
            .zip(imag)
            .map(|(&real, &imag)| from_complex(real, imag))
            .collect(),
        None => table.real.iter().copied().map(K::from).collect(),
    };
    let index = tables.len();
    tables.push(values);
    Some(index)
}

pub(super) fn poly_mul_affine(poly: &mut [K], a: K, b: K, current_degree: usize) {
    let mut previous = K::ZERO;
    for coefficient in poly.iter_mut().take(current_degree + 2) {
        let old = *coefficient;
        *coefficient = a * old + b * previous;
        previous = old;
    }
}

pub(super) fn fold_host(table: &mut Vec<K>, challenge: K) {
    let half = table.len() / 2;
    for index in 0..half {
        let left = table[2 * index];
        table[index] = left + challenge * (table[2 * index + 1] - left);
    }
    table.truncate(half);
}

pub(super) fn decode_ajtai_y_eval(
    words: &[u64],
    witness_count: usize,
    matrix_count: usize,
) -> Option<Vec<Vec<[K; D]>>> {
    let form_rows = matrix_count.checked_mul(2)?;
    let expected = witness_count.checked_mul(form_rows)?.checked_mul(D)?;
    if words.len() != expected {
        return None;
    }
    Some(
        (0..witness_count)
            .map(|witness| {
                (0..matrix_count)
                    .map(|matrix| {
                        let real_base = (witness * form_rows + 2 * matrix) * D;
                        let imaginary_base = real_base + D;
                        std::array::from_fn(|coefficient| {
                            from_complex(
                                F::from_u64(words[real_base + coefficient]),
                                F::from_u64(words[imaginary_base + coefficient]),
                            )
                        })
                    })
                    .collect()
            })
            .collect(),
    )
}

pub(super) fn words_to_k(value: KWords) -> K {
    from_complex(F::from_u64(value.c0), F::from_u64(value.c1))
}

pub(super) fn decode_trace(trace: MetalSumcheckTrace) -> (Vec<Vec<K>>, Vec<K>, ([F; 8], usize)) {
    let coeffs = trace
        .coeffs
        .into_iter()
        .map(|round| round.into_iter().map(words_to_k).collect())
        .collect();
    let challenges = trace.challenges.into_iter().map(words_to_k).collect();
    let state = trace.transcript_state.map(F::from_u64);
    (coeffs, challenges, (state, trace.transcript_absorbed))
}
