//! Executes Lean-owned schema-8 witness blocks.
//!
//! Digest-lane blocks carry one logical start and source expression. The final
//! two explicit blocks carry the already-remapped PiDEC and running-transition
//! witness batches. This module accepts only that Lean-owned order and those
//! two exact tags.

use rayon::prelude::*;
use serde_json::Value;

use super::{source_map::source_to_spartan, PackageError, GOLDILOCKS_MODULUS};
use crate::witness::RawWitnessBatch;

const SOURCE_COUNT: usize = 17;
const ROUND_COUNT: usize = 8;
const LANE_COUNT: usize = 4;
const BLOCK_COUNT: usize = SOURCE_COUNT * ROUND_COUNT * LANE_COUNT;
const BATCHES_PER_BLOCK: usize = 9;
const PI_DEC_BATCH_COUNT: usize = 270;
const RUNNING_TRANSITION_BATCH_COUNT: usize = 1;

const CANONICAL_BIT_COUNT: usize = 64;
const CANONICAL_HALF_BIT_COUNT: usize = 32;
const CANONICAL_AUXILIARY_COUNT: usize = 66;
const CANONICAL_HIGH_MAX: u64 = 4_294_967_295;

const CANDIDATE_BIT_COUNT: usize = 16;
const QUOTIENT_BIT_COUNT: usize = 14;
const CANDIDATE_AUXILIARY_COUNT: usize = 17;
const NEGATIVE_ONE: u64 = GOLDILOCKS_MODULUS - 1;

pub(super) fn expand(mut blocks: Vec<Value>) -> Result<Vec<RawWitnessBatch>, PackageError> {
    if blocks.len() != BLOCK_COUNT + 2 {
        return Err(PackageError::Invalid("witness plan block count"));
    }
    let running_transition = blocks
        .pop()
        .ok_or(PackageError::Invalid("running-transition witness block"))?;
    let pi_dec = blocks
        .pop()
        .ok_or(PackageError::Invalid("PiDEC witness block"))?;
    let expanded = blocks
        .into_par_iter()
        .map(expand_digest_block)
        .collect::<Result<Vec<_>, _>>()?;
    let mut batches =
        Vec::with_capacity(BLOCK_COUNT * BATCHES_PER_BLOCK + PI_DEC_BATCH_COUNT + RUNNING_TRANSITION_BATCH_COUNT);
    for mut block in expanded {
        batches.append(&mut block);
    }
    batches.extend(expand_explicit_block(
        pi_dec,
        PI_DEC_BATCH_COUNT,
        "PiDEC witness batch count",
    )?);
    batches.extend(expand_explicit_block(
        running_transition,
        RUNNING_TRANSITION_BATCH_COUNT,
        "running-transition witness batch count",
    )?);
    Ok(batches)
}

fn expand_digest_block(value: Value) -> Result<Vec<RawWitnessBatch>, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("witness digest block"))?;
    let [tag, offset, source] = fields.as_slice() else {
        return Err(PackageError::Invalid("witness digest block"));
    };
    if tag.as_u64() != Some(0) {
        return Err(PackageError::Invalid("witness digest block tag"));
    }
    let offset = value_word(offset, "witness plan logical start")?;
    let source = remap_expr(source)?;
    let mut batches = Vec::with_capacity(BATCHES_PER_BLOCK);
    expand_canonical(offset, source, &mut batches)?;
    expand_candidate(offset, 0, &mut batches)?;
    expand_candidate(offset, 1, &mut batches)?;
    Ok(batches)
}

fn expand_explicit_block(
    value: Value,
    expected_count: usize,
    count_error: &'static str,
) -> Result<Vec<RawWitnessBatch>, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("witness explicit block"))?;
    let [tag, values] = fields.as_slice() else {
        return Err(PackageError::Invalid("witness explicit block"));
    };
    if tag.as_u64() != Some(1) {
        return Err(PackageError::Invalid("witness explicit block tag"));
    }
    let batches: Vec<RawWitnessBatch> = serde_json::from_value(values.clone())?;
    if batches.len() != expected_count {
        return Err(PackageError::Invalid(count_error));
    }
    Ok(batches)
}

fn expand_canonical(offset: usize, source: Value, batches: &mut Vec<RawWitnessBatch>) -> Result<(), PackageError> {
    let bit_hints = (0..CANONICAL_BIT_COUNT)
        .map(|index| bit_hint(source.clone(), index))
        .collect::<Result<Vec<_>, _>>()?;
    batches.push(raw_batch(offset, Vec::new(), bit_hints)?);

    let difference = canonical_high_difference(offset)?;
    batches.push(raw_batch(
        checked_add(offset, CANONICAL_BIT_COUNT, "canonical inverse start")?,
        Vec::new(),
        vec![hint(1, difference.clone())],
    )?);

    let inverse = logical_var(checked_add(offset, CANONICAL_BIT_COUNT, "canonical inverse column")?)?;
    let flag = sub(constant(1), multiply(difference, inverse));
    batches.push(raw_batch(
        checked_add(offset, CANONICAL_BIT_COUNT + 1, "canonical flag start")?,
        vec![flag],
        Vec::new(),
    )?);
    Ok(())
}

fn expand_candidate(word_offset: usize, part: usize, batches: &mut Vec<RawWitnessBatch>) -> Result<(), PackageError> {
    let offset = checked_add(
        checked_add(word_offset, CANONICAL_AUXILIARY_COUNT, "candidate logical start")?,
        part * CANDIDATE_AUXILIARY_COUNT,
        "candidate logical start",
    )?;
    let candidate = canonical_weighted_bits(word_offset, part * CANDIDATE_BIT_COUNT, CANDIDATE_BIT_COUNT)?;
    batches.push(raw_batch(
        offset,
        Vec::new(),
        vec![hint(2, candidate.clone()), hint(3, candidate)],
    )?);

    let quotient = logical_var(offset)?;
    let quotient_hints = (0..QUOTIENT_BIT_COUNT)
        .map(|index| bit_hint(quotient.clone(), index))
        .collect::<Result<Vec<_>, _>>()?;
    batches.push(raw_batch(
        checked_add(offset, 2, "candidate quotient-bit start")?,
        Vec::new(),
        quotient_hints,
    )?);

    let mut reject = constant(1);
    for index in 0..CANDIDATE_BIT_COUNT {
        let bit = logical_var(checked_add(
            word_offset,
            part * CANDIDATE_BIT_COUNT + index,
            "candidate source bit",
        )?)?;
        reject = multiply(reject, bit);
    }
    batches.push(raw_batch(
        checked_add(offset, 16, "candidate reject start")?,
        vec![reject],
        Vec::new(),
    )?);
    Ok(())
}

fn canonical_high_difference(offset: usize) -> Result<Value, PackageError> {
    Ok(sub(
        canonical_weighted_bits(offset, CANONICAL_HALF_BIT_COUNT, CANONICAL_HALF_BIT_COUNT)?,
        constant(CANONICAL_HIGH_MAX),
    ))
}

fn canonical_weighted_bits(offset: usize, bit_start: usize, count: usize) -> Result<Value, PackageError> {
    let mut expression = constant(0);
    for index in 0..count {
        let coefficient = 1u64
            .checked_shl(u32::try_from(index).map_err(|_| PackageError::Invalid("witness bit index"))?)
            .ok_or(PackageError::Invalid("witness bit coefficient"))?;
        let bit = logical_var(checked_add(offset, bit_start + index, "witness bit column")?)?;
        expression = add_expr(expression, multiply(constant(coefficient), bit));
    }
    Ok(expression)
}

fn remap_expr(value: &Value) -> Result<Value, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("witness plan expression"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Invalid("witness plan expression tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, column]) => logical_var(value_word(column, "witness plan expression column")?),
        (1, [_, value]) => {
            let value = value
                .as_u64()
                .ok_or(PackageError::Invalid("witness plan expression constant"))?;
            if value >= GOLDILOCKS_MODULUS {
                return Err(PackageError::NonCanonicalField {
                    location: "witness plan expression constant",
                    value,
                });
            }
            Ok(constant(value))
        }
        (2, [_, left, right]) => Ok(add_expr(remap_expr(left)?, remap_expr(right)?)),
        (3, [_, left, right]) => Ok(multiply(remap_expr(left)?, remap_expr(right)?)),
        _ => Err(PackageError::Invalid("witness plan expression")),
    }
}

fn raw_batch(start: usize, recipes: Vec<Value>, hints: Vec<Value>) -> Result<RawWitnessBatch, PackageError> {
    Ok(RawWitnessBatch(
        to_word(source_to_spartan(start)?, "witness batch start")?,
        recipes,
        hints,
    ))
}

fn logical_var(column: usize) -> Result<Value, PackageError> {
    Ok(Value::Array(vec![
        Value::from(0),
        Value::from(to_word(source_to_spartan(column)?, "witness expression column")?),
    ]))
}

fn constant(value: u64) -> Value {
    Value::Array(vec![Value::from(1), Value::from(value)])
}

fn add_expr(left: Value, right: Value) -> Value {
    Value::Array(vec![Value::from(2), left, right])
}

fn multiply(left: Value, right: Value) -> Value {
    Value::Array(vec![Value::from(3), left, right])
}

fn sub(left: Value, right: Value) -> Value {
    add_expr(left, multiply(constant(NEGATIVE_ONE), right))
}

fn bit_hint(source: Value, index: usize) -> Result<Value, PackageError> {
    Ok(Value::Array(vec![
        Value::from(0),
        source,
        Value::from(to_word(index, "witness bit index")?),
    ]))
}

fn hint(tag: u64, source: Value) -> Value {
    Value::Array(vec![Value::from(tag), source])
}

fn value_word(value: &Value, location: &'static str) -> Result<usize, PackageError> {
    value
        .as_u64()
        .ok_or(PackageError::Invalid(location))
        .and_then(|value| word(value, location))
}

fn word(value: u64, location: &'static str) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| PackageError::Invalid(location))
}

fn to_word(value: usize, location: &'static str) -> Result<u64, PackageError> {
    u64::try_from(value).map_err(|_| PackageError::Invalid(location))
}

fn checked_add(left: usize, right: usize, location: &'static str) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid(location))
}
