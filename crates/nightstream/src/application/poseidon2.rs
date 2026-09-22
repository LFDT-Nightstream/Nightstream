//! Independent application builder for the existing Poseidon2HashChainV1.
//! The round schedule follows Gadgets/Poseidon2/{Hash,Permutation,Layer}.lean.
//! It consumes no exported application constraints or witness recipes.

use neo_ccs::crypto::poseidon2_goldilocks::{poseidon2_hash, round_constants, Poseidon2RoundConstants};
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
    for block in preimage.chunks(4) {
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

fn mat4(state: &[Affine; 8], base: usize, lane: usize) -> Affine {
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

fn external(builder: &mut ApplicationBuilder, state: &[Affine; 8]) -> Result<[Affine; 8], ApplicationError> {
    let mut output = std::array::from_fn(|_| scalar(0));
    for (lane, value) in output.iter_mut().enumerate() {
        let block = mat4(state, if lane < 4 { 0 } else { 4 }, lane % 4);
        *value = builder
            .affine(block + mat4(state, 0, lane % 4) + mat4(state, 4, lane % 4))?
            .into();
    }
    Ok(output)
}

fn full_round(
    builder: &mut ApplicationBuilder,
    state: &mut [Affine; 8],
    constants: &[u64; 8],
) -> Result<(), ApplicationError> {
    for (value, constant) in state.iter_mut().zip(constants) {
        *value = sbox(builder, value.clone() + scalar(*constant))?;
    }
    *state = external(builder, state)?;
    Ok(())
}

fn permutation(
    builder: &mut ApplicationBuilder,
    state: [Affine; 8],
    constants: &Poseidon2RoundConstants,
) -> Result<[Affine; 8], ApplicationError> {
    let mut state = external(builder, &state)?;
    for round in &constants.initial {
        full_round(builder, &mut state, round)?;
    }
    for constant in &constants.internal {
        state[0] = sbox(builder, state[0].clone() + scalar(*constant))?;
        let sum = state
            .iter()
            .cloned()
            .reduce(|sum, value| sum + value)
            .expect("eight state lanes");
        let prior = state.clone();
        for lane in 0..8 {
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
