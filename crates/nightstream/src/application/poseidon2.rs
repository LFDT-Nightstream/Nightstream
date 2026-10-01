//! Independent application builder for the existing Poseidon2HashChainV1.
//! The round schedule follows Gadgets/Poseidon2/{Hash,Permutation,Layer}.lean.
//! It consumes no exported application constraints or witness recipes.

use neo_ccs::crypto::poseidon2_goldilocks::{poseidon2_hash, round_constants, Poseidon2RoundConstants, RATE, WIDTH};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use super::{Affine, ApplicationBuilder, ApplicationCircuit, ApplicationError};

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
    let constants = round_constants();
    let mut preimage: Vec<Affine> = DOMAIN_TAG
        .iter()
        .map(|byte| scalar(u64::from(*byte)))
        .collect();
    preimage.extend(builder.input_state().map(Affine::from));
    preimage.extend(builder.private_inputs().iter().copied().map(Affine::from));
    let mut state = std::array::from_fn(|_| scalar(0));
    for block in preimage.chunks(RATE) {
        for (lane, value) in state.iter_mut().enumerate() {
            *value = value.clone() + block.get(lane).cloned().unwrap_or_else(|| scalar(0));
        }
        state = permutation(&mut builder, state, &constants)?;
    }
    state[0] = state[0].clone() + scalar(1);
    state = permutation(&mut builder, state, &constants)?;
    builder.finish(std::array::from_fn(|lane| state[lane].clone()))
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

fn permutation(
    builder: &mut ApplicationBuilder,
    state: [Affine; WIDTH],
    constants: &Poseidon2RoundConstants,
) -> Result<[Affine; WIDTH], ApplicationError> {
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

#[cfg(test)]
#[path = "../../tests/application_internal/hash_chain_fixture.rs"]
pub(crate) mod hash_chain_fixture;
