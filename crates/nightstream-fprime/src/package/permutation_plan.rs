//! Executes Lean-owned schema-8 Poseidon2 invocation blocks.
//!
//! The package selects every phase, start, initial state, absorb input, and
//! action. This module only implements the fixed Lean expansion rules.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use serde_json::Value;

use super::{
    source_map::source_to_spartan, PackageError, RawPermutationInvocation, RawSparseCombination, RawSparseTerm,
    GOLDILOCKS_MODULUS,
};

const STATE_WIDTH: usize = 8;
const ABSORB_RATE: usize = 4;
const PERMUTATION_SIZE: usize = 592;
const PERMUTATION_OUTPUT_OFFSET: usize = 584;

#[derive(Debug, Deserialize)]
struct RawActionBlock(u64, u64, u64, Vec<Value>, Vec<Value>);

#[derive(Debug, Deserialize)]
struct RawDirectBlock(u64, u64, u64, Vec<Value>);

struct AffineCombination {
    constant: Goldilocks,
    terms: Vec<(usize, Goldilocks)>,
}

pub(super) fn expand(blocks: Vec<Value>) -> Result<Vec<RawPermutationInvocation>, PackageError> {
    let mut invocations = Vec::new();
    for block in blocks {
        let fields = block
            .as_array()
            .ok_or(PackageError::Invalid("permutation plan block"))?;
        let tag = fields
            .first()
            .and_then(Value::as_u64)
            .ok_or(PackageError::Invalid("permutation plan block tag"))?;
        match (tag, fields.as_slice()) {
            (0, [_, payload]) => expand_action_block(payload, &mut invocations)?,
            (1, [_, payload]) => expand_direct_block(payload, &mut invocations)?,
            _ => return Err(PackageError::Invalid("permutation plan block")),
        }
    }
    Ok(invocations)
}

fn expand_action_block(value: &Value, invocations: &mut Vec<RawPermutationInvocation>) -> Result<(), PackageError> {
    let RawActionBlock(phase, row_start, witness_start, mut state, actions) = serde_json::from_value(value.clone())?;
    require_state(&state)?;
    let mut row_start = word(row_start, "permutation action row start")?;
    let mut witness_start = word(witness_start, "permutation action witness start")?;

    for action in actions {
        let fields = action
            .as_array()
            .ok_or(PackageError::Invalid("permutation action shape"))?;
        let tag = fields
            .first()
            .and_then(Value::as_u64)
            .ok_or(PackageError::Invalid("permutation action shape tag"))?;
        match (tag, fields.as_slice()) {
            (0, [_, input]) => {
                let input = input
                    .as_array()
                    .ok_or(PackageError::Invalid("permutation absorb input"))?;
                for chunk in input.chunks(ABSORB_RATE) {
                    let absorbed = state
                        .iter()
                        .enumerate()
                        .map(|(lane, current)| match chunk.get(lane) {
                            Some(value) => add_expression(current.clone(), value.clone()),
                            None => current.clone(),
                        })
                        .collect::<Vec<_>>();
                    push_invocation(phase, row_start, witness_start, &absorbed, invocations)?;
                    state = permutation_output(witness_start)?;
                    row_start = add(row_start, PERMUTATION_SIZE, "permutation row start")?;
                    witness_start = add(witness_start, PERMUTATION_SIZE, "permutation witness start")?;
                }
            }
            (1, [_]) => {
                push_invocation(phase, row_start, witness_start, &state, invocations)?;
                let second_state = permutation_output(witness_start)?;
                push_invocation(
                    phase,
                    add(row_start, PERMUTATION_SIZE, "permutation row start")?,
                    add(witness_start, PERMUTATION_SIZE, "permutation witness start")?,
                    &second_state,
                    invocations,
                )?;
                state = permutation_output(add(witness_start, PERMUTATION_SIZE, "permutation witness start")?)?;
                row_start = add(row_start, 2 * PERMUTATION_SIZE, "permutation row start")?;
                witness_start = add(witness_start, 2 * PERMUTATION_SIZE, "permutation witness start")?;
            }
            _ => return Err(PackageError::Invalid("permutation action shape")),
        }
    }
    Ok(())
}

fn expand_direct_block(value: &Value, invocations: &mut Vec<RawPermutationInvocation>) -> Result<(), PackageError> {
    let RawDirectBlock(phase, row_start, witness_start, state) = serde_json::from_value(value.clone())?;
    require_state(&state)?;
    push_invocation(
        phase,
        word(row_start, "direct permutation row start")?,
        word(witness_start, "direct permutation witness start")?,
        &state,
        invocations,
    )
}

fn push_invocation(
    phase: u64,
    row_start: usize,
    witness_start: usize,
    state: &[Value],
    invocations: &mut Vec<RawPermutationInvocation>,
) -> Result<(), PackageError> {
    require_state(state)?;
    let inputs = state
        .iter()
        .map(lower_affine)
        .map(|result| result.and_then(raw_combination))
        .collect::<Result<Vec<_>, _>>()?;
    invocations.push(RawPermutationInvocation(
        phase,
        to_word(row_start, "permutation row start")?,
        to_word(source_to_spartan(witness_start)?, "permutation witness start")?,
        inputs,
    ));
    Ok(())
}

fn permutation_output(witness_start: usize) -> Result<Vec<Value>, PackageError> {
    let output_start = add(witness_start, PERMUTATION_OUTPUT_OFFSET, "permutation output start")?;
    (0..STATE_WIDTH)
        .map(|lane| add(output_start, lane, "permutation output lane").and_then(variable_expression))
        .collect()
}

fn require_state(state: &[Value]) -> Result<(), PackageError> {
    if state.len() != STATE_WIDTH {
        return Err(PackageError::Invalid("permutation state width"));
    }
    Ok(())
}

fn lower_affine(value: &Value) -> Result<AffineCombination, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("permutation input expression"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Invalid("permutation input expression tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, column]) => {
            let column = value_word(column, "permutation input column")?;
            Ok(AffineCombination {
                constant: Goldilocks::ZERO,
                terms: vec![(source_to_spartan(column)?, Goldilocks::ONE)],
            })
        }
        (1, [_, constant]) => Ok(AffineCombination {
            constant: field_word(constant)?,
            terms: Vec::new(),
        }),
        (2, [_, left, right]) => {
            let mut left = lower_affine(left)?;
            let right = lower_affine(right)?;
            left.constant += right.constant;
            left.terms.extend(right.terms);
            Ok(left)
        }
        (3, [_, left, right]) => {
            if let Some(coefficient) = syntactic_constant(left)? {
                return Ok(scale(coefficient, lower_affine(right)?));
            }
            if let Some(coefficient) = syntactic_constant(right)? {
                return Ok(scale(coefficient, lower_affine(left)?));
            }
            Err(PackageError::Invalid("non-affine permutation input"))
        }
        _ => Err(PackageError::Invalid("permutation input expression")),
    }
}

fn syntactic_constant(value: &Value) -> Result<Option<Goldilocks>, PackageError> {
    let Some(fields) = value.as_array() else {
        return Err(PackageError::Invalid("permutation input expression"));
    };
    if fields.first().and_then(Value::as_u64) != Some(1) {
        return Ok(None);
    }
    match fields.as_slice() {
        [_, constant] => field_word(constant).map(Some),
        _ => Err(PackageError::Invalid("permutation input expression")),
    }
}

fn scale(coefficient: Goldilocks, mut combination: AffineCombination) -> AffineCombination {
    combination.constant *= coefficient;
    for (_, term_coefficient) in &mut combination.terms {
        *term_coefficient *= coefficient;
    }
    combination
}

fn raw_combination(combination: AffineCombination) -> Result<RawSparseCombination, PackageError> {
    Ok(RawSparseCombination(
        combination.constant.as_canonical_u64(),
        combination
            .terms
            .into_iter()
            .map(|(column, coefficient)| {
                Ok(RawSparseTerm(
                    to_word(column, "permutation input column")?,
                    coefficient.as_canonical_u64(),
                ))
            })
            .collect::<Result<Vec<_>, PackageError>>()?,
    ))
}

fn add_expression(left: Value, right: Value) -> Value {
    Value::Array(vec![Value::from(2), left, right])
}

fn variable_expression(column: usize) -> Result<Value, PackageError> {
    Ok(Value::Array(vec![
        Value::from(0),
        Value::from(to_word(column, "permutation output column")?),
    ]))
}

fn field_word(value: &Value) -> Result<Goldilocks, PackageError> {
    let value = value
        .as_u64()
        .ok_or(PackageError::Invalid("permutation input constant"))?;
    if value >= GOLDILOCKS_MODULUS {
        return Err(PackageError::NonCanonicalField {
            location: "permutation input constant",
            value,
        });
    }
    Ok(Goldilocks::from_u64(value))
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

fn add(left: usize, right: usize, location: &'static str) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid(location))
}
