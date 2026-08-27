//! Schema-8 expansion of Lean-owned compact-row invocation blocks.

use serde::Deserialize;
use serde_json::Value;

use super::{compact, source_map::source_to_spartan, PackageError, RawCompactRowInvocation};

const PHASE: u64 = 7;
const RING_DEGREE: usize = 54;
const POSITION_SLOTS: usize = 55;
const VALUE_SLOTS: usize = 54;
const COMBINATION_TEMPLATE_COUNT: usize = 108;
const PI_CCS_SOURCE_START: usize = 12_688_104;
const PI_CCS_TARGET_START: usize = 12_688_042;
const PILOT_PRIOR_PUBLIC_SOURCE_START: usize = 42_475;

#[derive(Debug, Deserialize)]
struct RawFirst54Block(u64, u64);

#[derive(Debug, Deserialize)]
struct RawCombinationFamily(u64, u64, u64, u64, u64, u64);

#[derive(Debug, Deserialize)]
struct RawCombinationBlock(
    u64,
    RawCombinationFamily,
    RawCombinationFamily,
    RawCombinationFamily,
    RawCombinationFamily,
);

#[derive(Clone, Copy)]
struct CombinationFamily {
    logical_start: usize,
    row_start: usize,
    fresh_start: usize,
    block_count: usize,
    cell_count: usize,
    value_stride: usize,
}

impl TryFrom<RawCombinationFamily> for CombinationFamily {
    type Error = PackageError;

    fn try_from(raw: RawCombinationFamily) -> Result<Self, Self::Error> {
        Ok(Self {
            logical_start: word(raw.0)?,
            row_start: word(raw.1)?,
            fresh_start: word(raw.2)?,
            block_count: word(raw.3)?,
            cell_count: word(raw.4)?,
            value_stride: word(raw.5)?,
        })
    }
}

pub(super) fn expand(blocks: Vec<Value>) -> Result<Vec<RawCompactRowInvocation>, PackageError> {
    if blocks.len() != 2 {
        return Err(PackageError::Invalid("compact plan block count"));
    }
    let first = tagged_payload(&blocks[0], 0)?;
    let combination = tagged_payload(&blocks[1], 1)?;
    let RawFirst54Block(source_count, round_count) = serde_json::from_value(first.clone())?;
    let RawCombinationBlock(source_count_combination, commitment, public_input, eval_k, eval_a) =
        serde_json::from_value(combination.clone())?;

    let mut invocations = expand_first54(word(source_count)?, word(round_count)?)?;
    invocations.extend(expand_combinations(
        word(source_count_combination)?,
        [
            commitment.try_into()?,
            public_input.try_into()?,
            eval_k.try_into()?,
            eval_a.try_into()?,
        ],
    )?);
    Ok(invocations)
}

fn tagged_payload(value: &Value, expected_tag: u64) -> Result<&Value, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("compact plan block"))?;
    match fields.as_slice() {
        [tag, payload] if tag.as_u64() == Some(expected_tag) => Ok(payload),
        _ => Err(PackageError::Invalid("compact plan block")),
    }
}

fn expand_first54(source_count: usize, round_count: usize) -> Result<Vec<RawCompactRowInvocation>, PackageError> {
    let count = product(&[source_count, round_count, POSITION_SLOTS + VALUE_SLOTS])?;
    let mut result = Vec::with_capacity(count);
    for source in 0..source_count {
        let selector_logical = linear(17_878_110, &[(source, 15_504)])?;
        let selector_row = linear(17_774_148, &[(source, 59_344)])?;
        let selector_fresh = linear(18_187_828, &[(source, 43_743)])?;
        for round in 0..round_count {
            let digest_round = round / 8;
            let lane = (round % 8) / 2;
            let part = round % 2;
            let decoder = linear(
                17_870_240,
                &[(source, 15_504), (digest_round, 992), (lane, 100), (part, 17)],
            )?;
            let reject = final_column(add(decoder, 16)?)?;
            let remainder = final_column(add(decoder, 1)?)?;
            let round_logical = linear(selector_logical, &[(round, POSITION_SLOTS + VALUE_SLOTS)])?;
            let round_row_prefix = if round == 0 {
                0
            } else {
                linear(325, &[(round - 1, 646)])?
            };
            let round_fresh_prefix = if round == 0 {
                0
            } else {
                linear(216, &[(round - 1, 537)])?
            };
            let prior_round_logical = round
                .checked_sub(1)
                .map(|prior| linear(selector_logical, &[(prior, POSITION_SLOTS + VALUE_SLOTS)]))
                .transpose()?;

            for slot in 0..POSITION_SLOTS {
                let template = COMBINATION_TEMPLATE_COUNT + if round == 0 { slot } else { POSITION_SLOTS + slot };
                let row_prefix = if round == 0 {
                    slot
                } else if slot == 0 {
                    0
                } else {
                    linear(1, &[(slot - 1, 7)])?
                };
                let fresh_prefix = if round == 0 || slot == 0 {
                    0
                } else {
                    product(&[6, slot - 1])?
                };
                let output = final_column(add(round_logical, slot)?)?;
                let ranges = if let Some(prior) = prior_round_logical {
                    vec![
                        [0, 1, reject, 1],
                        [1, POSITION_SLOTS, final_column(prior)?, 1],
                        [56, 1, output, 1],
                    ]
                } else {
                    vec![[0, 1, reject, 1], [1, 1, output, 1]]
                };
                result.push(raw(
                    template,
                    linear(selector_row, &[(1, round_row_prefix), (1, row_prefix)])?,
                    final_column(linear(selector_fresh, &[(1, round_fresh_prefix), (1, fresh_prefix)])?)?,
                    ranges,
                )?);
            }

            for slot in 0..VALUE_SLOTS {
                let template = COMBINATION_TEMPLATE_COUNT
                    + if round == 0 {
                        2 * POSITION_SLOTS + slot
                    } else {
                        2 * POSITION_SLOTS + VALUE_SLOTS + slot
                    };
                let position_fresh_count = if round == 0 { 0 } else { 321 };
                let position_row_count = if round == 0 { POSITION_SLOTS } else { 376 };
                let output = final_column(linear(round_logical, &[(1, POSITION_SLOTS), (1, slot)])?)?;
                let ranges = if let Some(prior) = prior_round_logical {
                    vec![
                        [0, 1, reject, 1],
                        [1, 1, remainder, 1],
                        [2, POSITION_SLOTS, final_column(prior)?, 1],
                        [57, VALUE_SLOTS, final_column(add(prior, POSITION_SLOTS)?)?, 1],
                        [111, 1, output, 1],
                    ]
                } else {
                    vec![[0, 1, reject, 1], [1, 1, remainder, 1], [2, 1, output, 1]]
                };
                result.push(raw(
                    template,
                    linear(
                        selector_row,
                        &[(1, round_row_prefix), (1, position_row_count), (slot, 5)],
                    )?,
                    final_column(linear(
                        selector_fresh,
                        &[(1, round_fresh_prefix), (1, position_fresh_count), (slot, 4)],
                    )?)?,
                    ranges,
                )?);
            }
        }
    }
    Ok(result)
}

fn expand_combinations(
    source_count: usize,
    families: [CombinationFamily; 4],
) -> Result<Vec<RawCompactRowInvocation>, PackageError> {
    let fresh_costs = (0..RING_DEGREE).map(lane_fresh_count).collect::<Vec<_>>();
    let row_costs = fresh_costs.iter().map(|cost| cost + 1).collect::<Vec<_>>();
    let fresh_sum = fresh_costs.iter().sum::<usize>();
    let row_sum = row_costs.iter().sum::<usize>();
    if fresh_sum != 8_100 || row_sum != 8_154 {
        return Err(PackageError::Invalid("combination lane costs"));
    }
    let fresh_prefixes = prefixes(&fresh_costs)?;
    let row_prefixes = prefixes(&row_costs)?;
    let capacity = families.iter().try_fold(0usize, |total, family| {
        add(
            total,
            product(&[source_count, family.block_count, RING_DEGREE, family.cell_count])?,
        )
    })?;
    let mut result = Vec::with_capacity(capacity);
    for (family_index, family) in families.into_iter().enumerate() {
        if family.cell_count == 0 {
            return Err(PackageError::Invalid("combination cell count"));
        }
        let step_size = product(&[family.block_count, RING_DEGREE, family.cell_count])?;
        let source_fresh_count = product(&[family.block_count, family.cell_count, fresh_sum])?;
        let source_row_count = product(&[family.block_count, family.cell_count, row_sum])?;
        for source in 0..source_count {
            for block in 0..family.block_count {
                for lane in 0..RING_DEGREE {
                    for cell in 0..family.cell_count {
                        let logical_index = linear(
                            0,
                            &[
                                (block, RING_DEGREE * family.cell_count),
                                (lane, family.cell_count),
                                (cell, 1),
                            ],
                        )?;
                        let output_source = linear(family.logical_start, &[(source, step_size), (logical_index, 1)])?;
                        let prior_column = if source == 0 {
                            0
                        } else {
                            final_column(linear(
                                family.logical_start,
                                &[(source - 1, step_size), (logical_index, 1)],
                            )?)?
                        };
                        let coordinate_fresh = linear(
                            0,
                            &[
                                (block, family.cell_count * fresh_sum),
                                (fresh_prefixes[lane], family.cell_count),
                                (cell, fresh_costs[lane]),
                            ],
                        )?;
                        let coordinate_row = linear(
                            0,
                            &[
                                (block, family.cell_count * row_sum),
                                (row_prefixes[lane], family.cell_count),
                                (cell, row_costs[lane]),
                            ],
                        )?;
                        result.push(raw(
                            if source == 0 { lane } else { RING_DEGREE + lane },
                            linear(family.row_start, &[(source, source_row_count), (coordinate_row, 1)])?,
                            final_column(linear(
                                family.fresh_start,
                                &[(source, source_fresh_count), (coordinate_fresh, 1)],
                            )?)?,
                            vec![
                                [0, RING_DEGREE, combination_challenge(source)?, 1],
                                [
                                    RING_DEGREE,
                                    RING_DEGREE,
                                    combination_value(family_index, source, block, cell)?,
                                    family.value_stride,
                                ],
                                [108, 1, prior_column, 1],
                                [109, 1, final_column(output_source)?, 1],
                            ],
                        )?);
                    }
                }
            }
        }
    }
    Ok(result)
}

fn combination_challenge(source: usize) -> Result<usize, PackageError> {
    linear(17_884_970, &[(source, 15_504)])
}

fn combination_value(family: usize, source: usize, block: usize, cell: usize) -> Result<usize, PackageError> {
    match family {
        0 if source == 0 => linear(84_950, &[(block, RING_DEGREE)]),
        0 => linear(91, &[(source - 1, 2_649), (block, RING_DEGREE)]),
        1 if source == 0 => source_to_spartan(PILOT_PRIOR_PUBLIC_SOURCE_START),
        1 => linear(1_064, &[(source - 1, 2_649)]),
        2 => linear(86_422, &[(source, 1_620), (cell, 1)]),
        3 => linear(86_530, &[(source, 1_620), (block, 108), (cell, 1)]),
        _ => Err(PackageError::Invalid("combination family")),
    }
}

fn lane_fresh_count(lane: usize) -> usize {
    let folded = if lane < 27 { lane + RING_DEGREE } else { lane + 27 };
    let twice = if lane + 81 <= 106 { term_count(lane + 81) } else { 0 };
    2 * (term_count(lane) + term_count(folded) + twice) + 2
}

fn term_count(degree: usize) -> usize {
    if degree < RING_DEGREE {
        degree + 1
    } else if degree <= 106 {
        107 - degree
    } else {
        0
    }
}

fn prefixes(costs: &[usize]) -> Result<Vec<usize>, PackageError> {
    let mut result = Vec::with_capacity(costs.len());
    let mut current = 0usize;
    for cost in costs {
        result.push(current);
        current = add(current, *cost)?;
    }
    Ok(result)
}

fn final_column(source: usize) -> Result<usize, PackageError> {
    if source < PI_CCS_SOURCE_START {
        return Err(PackageError::Invalid("plan local source column"));
    }
    add(PI_CCS_TARGET_START, source - PI_CCS_SOURCE_START)
}

fn raw(
    template: usize,
    row_start: usize,
    local_start: usize,
    ranges: Vec<[usize; 4]>,
) -> Result<RawCompactRowInvocation, PackageError> {
    Ok(compact::raw_invocation(
        PHASE,
        to_word(template)?,
        to_word(row_start)?,
        to_word(local_start)?,
        ranges
            .into_iter()
            .map(|range| {
                Ok([
                    to_word(range[0])?,
                    to_word(range[1])?,
                    to_word(range[2])?,
                    to_word(range[3])?,
                ])
            })
            .collect::<Result<Vec<_>, PackageError>>()?,
    ))
}

fn word(value: u64) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| PackageError::Invalid("plan word overflow"))
}

fn to_word(value: usize) -> Result<u64, PackageError> {
    u64::try_from(value).map_err(|_| PackageError::Invalid("plan word overflow"))
}

fn add(left: usize, right: usize) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid("plan arithmetic overflow"))
}

fn product(values: &[usize]) -> Result<usize, PackageError> {
    values.iter().try_fold(1usize, |result, value| {
        result
            .checked_mul(*value)
            .ok_or(PackageError::Invalid("plan arithmetic overflow"))
    })
}

fn linear(base: usize, terms: &[(usize, usize)]) -> Result<usize, PackageError> {
    terms.iter().try_fold(base, |result, (value, coefficient)| {
        add(result, product(&[*value, *coefficient])?)
    })
}
