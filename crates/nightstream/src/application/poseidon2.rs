//! Poseidon2 application hashes and the existing Poseidon2HashChainV1.
//! The round schedule follows Gadgets/Poseidon2/{Hash,Permutation,Layer}.lean.
//! It consumes no exported application constraints or witness recipes.

use std::collections::BTreeMap;

use neo_ccs::crypto::poseidon2_goldilocks::{poseidon2_hash, round_constants, Poseidon2RoundConstants, RATE, WIDTH};
use nightstream_fprime::{ApplicationWalk, Linear};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use super::{Affine, ApplicationBuilder, ApplicationCircuit, ApplicationError};

const BLOCK_LOCALS: usize = 1096;

const DOMAIN_TAG: &[u8; 40] = b"Nightstream/Stage1/Poseidon2HashChain/v1";

/// Native result of one Poseidon2HashChainV1 application step.
pub fn poseidon2_hash_chain_step(current: [Goldilocks; 4], message: [Goldilocks; 4]) -> [Goldilocks; 4] {
    let mut preimage: Vec<_> = DOMAIN_TAG
        .iter()
        .map(|byte| Goldilocks::from_u8(*byte))
        .collect();
    preimage.extend(current);
    preimage.extend(message);
    poseidon2_hash(&preimage)
}

/// Hashes the domain tag, four-word prior state, and four-word private message.
pub fn poseidon2_hash_chain_v1() -> Result<ApplicationCircuit, ApplicationError> {
    let mut builder = ApplicationBuilder::new(4)?;
    let mut preimage: Vec<Affine> = DOMAIN_TAG
        .iter()
        .map(|byte| scalar(u64::from(*byte)))
        .collect();
    preimage.extend(builder.input_state().map(Affine::from));
    preimage.extend(builder.private_inputs().iter().copied().map(Affine::from));
    let output = builder.poseidon2_hash(&preimage)?;
    builder.finish(output)
}

impl ApplicationBuilder {
    /// Hashes field expressions with the native Poseidon2 absorb/finalize rule.
    /// The caller must include a domain and lengths for variable-length inputs.
    /// Compilation validates each permutation before using its native block.
    pub fn poseidon2_hash(&mut self, preimage: &[Affine]) -> Result<[Affine; 4], ApplicationError> {
        for value in preimage {
            self.validate(value, true)?;
        }
        let constants = round_constants();
        let mut state = std::array::from_fn(|_| scalar(0));
        for block in preimage.chunks(RATE) {
            for (lane, value) in state.iter_mut().enumerate() {
                *value = value.clone() + block.get(lane).cloned().unwrap_or_else(|| scalar(0));
            }
            state = permutation(self, state, &constants)?;
        }
        state[0] = state[0].clone() + scalar(1);
        state = permutation(self, state, &constants)?;
        Ok(std::array::from_fn(|lane| state[lane].clone()))
    }
}

fn scalar(value: u64) -> Affine {
    Affine::constant(Goldilocks::from_u64(value))
}

fn sbox(builder: &mut ApplicationBuilder, value: Affine) -> Result<Affine, ApplicationError> {
    let square = builder.multiply(value.clone(), value.clone())?;
    let fourth = builder.multiply(square.into(), square.into())?;
    let sixth = builder.multiply(fourth.into(), square.into())?;
    Ok(builder.multiply(sixth.into(), value)?.into())
}

fn mat4(state: &[Affine; WIDTH], base: usize, lane: usize) -> Affine {
    let coefficients = match lane {
        0 => [2, 3, 1, 1],
        1 => [1, 2, 3, 1],
        2 => [1, 1, 2, 3],
        _ => [3, 1, 1, 2],
    };
    coefficients
        .iter()
        .enumerate()
        .map(|(offset, coefficient)| {
            if *coefficient == 1 {
                state[base + offset].clone()
            } else {
                state[base + offset].clone() * Goldilocks::from_u64(*coefficient)
            }
        })
        .reduce(|sum, value| sum + value)
        .expect("four matrix terms")
}

/// `M₄` on the lane's block plus the sum of all blocks at the same position.
/// Sums start from zero like Lean's `foldl (· + ·) 0`; recipe shape is identity input.
fn external(builder: &mut ApplicationBuilder, state: &[Affine; WIDTH]) -> Result<[Affine; WIDTH], ApplicationError> {
    let mut output = std::array::from_fn(|_| scalar(0));
    for (lane, value) in output.iter_mut().enumerate() {
        let column = (0..WIDTH / 4)
            .map(|block| mat4(state, 4 * block, lane % 4))
            .fold(scalar(0), |sum, term| sum + term);
        *value = builder
            .affine(mat4(state, 4 * (lane / 4), lane % 4) + column)?
            .into();
    }
    Ok(output)
}

fn full_round(
    builder: &mut ApplicationBuilder,
    state: &mut [Affine; WIDTH],
    constants: &[u64; WIDTH],
) -> Result<(), ApplicationError> {
    for (value, constant) in state.iter_mut().zip(constants) {
        *value = sbox(builder, value.clone() + scalar(*constant))?;
    }
    *state = external(builder, state)?;
    Ok(())
}

/// One Poseidon2 permutation of `state`: 1096 generated variables in the
/// order of `Gadgets/Poseidon2/Permutation.lean`. The step packages place it
/// as one block (`NativeShape::blocks`).
pub(crate) fn permutation(
    builder: &mut ApplicationBuilder,
    state: [Affine; WIDTH],
    constants: &Poseidon2RoundConstants,
) -> Result<[Affine; WIDTH], ApplicationError> {
    builder.begin_block(&state);
    let mut state = external(builder, &state)?;
    for round in &constants.initial {
        full_round(builder, &mut state, round)?;
    }
    for constant in &constants.internal {
        state[0] = sbox(builder, state[0].clone() + scalar(*constant))?;
        let sum = state
            .iter()
            .cloned()
            .fold(scalar(0), |sum, value| sum + value);
        let prior = state.clone();
        for lane in 0..WIDTH {
            state[lane] = builder
                .affine(prior[lane].clone() * Goldilocks::from_u64(constants.diag[lane]) + sum.clone())?
                .into();
        }
    }
    for round in &constants.terminal {
        full_round(builder, &mut state, round)?;
    }
    Ok(state)
}

/// A row's three sides: the constant and the terms, equal variables merged and
/// zero coefficients dropped.
type Sides = [(Goldilocks, BTreeMap<usize, Goldilocks>); 3];

/// The sides of the row that `recipe` generates.
fn recipe_sides(walk: &mut ApplicationWalk<'_>, recipe: usize) -> Result<Sides, ApplicationError> {
    let row = walk.recipe_row(recipe)?;
    let mut terms = Vec::new();
    let header = walk.row(row, &mut terms)?;
    let mut sides: Sides = header
        .constants
        .map(|constant| (Goldilocks::from_u64(constant), BTreeMap::new()));
    for term in terms {
        *sides[term.form.index()]
            .1
            .entry(term.variable)
            .or_insert(Goldilocks::ZERO) += Goldilocks::from_u64(term.coefficient);
    }
    for (_, terms) in &mut sides {
        terms.retain(|_, coefficient| *coefficient != Goldilocks::ZERO);
    }
    Ok(sides)
}

/// Rejects a block whose recipe rows are not `permutation`'s rows on its
/// lanes. The reference is one permutation of 16 witness words in a scratch
/// builder; each of its inputs becomes the block's lane and each of its locals
/// the block's local. The step packages drop these rows for the block
/// (`StepPackage::with_application`), so the rows and the block must agree.
pub(crate) fn check_blocks(application: &ApplicationCircuit) -> Result<(), ApplicationError> {
    let blocks = &application.shape().blocks;
    if blocks.is_empty() {
        return Ok(());
    }
    let mut scratch = ApplicationBuilder::new(WIDTH)?;
    let inputs = std::array::from_fn(|lane| Affine::from(scratch.private_inputs()[lane]));
    let state = permutation(&mut scratch, inputs, &round_constants())?;
    let reference = scratch.finish(std::array::from_fn(|lane| state[lane].clone()))?;
    let (inputs, locals) = (4..4 + WIDTH, reference.generated_range().start);
    let mut walk = reference.records().walk();
    let reference = (0..BLOCK_LOCALS)
        .map(|local| recipe_sides(&mut walk, local))
        .collect::<Result<Vec<_>, _>>()?;
    let mut walk = application.records().walk();
    let first = application.generated_range().start;
    for (index, block) in blocks.iter().enumerate() {
        let substitute = |variable: usize| -> Option<Linear> {
            if inputs.contains(&variable) {
                Some(block.lanes[variable - 4].clone())
            } else {
                Some(Linear {
                    constant: Goldilocks::ZERO,
                    terms: vec![(block.first_local + variable.checked_sub(locals)?, Goldilocks::ONE)],
                })
            }
        };
        let recipe = block
            .first_local
            .checked_sub(first)
            .ok_or(ApplicationError::Block(index))?;
        for (local, expected) in reference.iter().enumerate() {
            let actual = recipe_sides(&mut walk, recipe + local)?;
            let mut expected = expected.clone();
            for (constant, terms) in &mut expected {
                let mut mapped = BTreeMap::new();
                for (&variable, &coefficient) in terms.iter() {
                    let linear = substitute(variable).ok_or(ApplicationError::Block(index))?;
                    *constant += coefficient * linear.constant;
                    for (term, scale) in linear.terms {
                        *mapped.entry(term).or_insert(Goldilocks::ZERO) += coefficient * scale;
                    }
                }
                mapped.retain(|_, coefficient| *coefficient != Goldilocks::ZERO);
                *terms = mapped;
            }
            if actual != expected {
                return Err(ApplicationError::Block(index));
            }
        }
    }
    Ok(())
}
