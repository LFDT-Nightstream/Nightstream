//! A larger application fixture, built with the existing Poseidon2 primitives.

use super::*;

pub(crate) fn two_link_hash_chain() -> Result<ApplicationCircuit, ApplicationError> {
    let mut builder = ApplicationBuilder::new(8)?;
    let constants = round_constants();
    let messages: Vec<Affine> = builder
        .private_inputs()
        .iter()
        .copied()
        .map(Affine::from)
        .collect();
    let mut current = builder.input_state().map(Affine::from);
    for message in messages.chunks(4) {
        let mut preimage: Vec<Affine> = DOMAIN_TAG
            .iter()
            .map(|byte| scalar(u64::from(*byte)))
            .collect();
        preimage.extend(current);
        preimage.extend(message.iter().cloned());
        let mut state = std::array::from_fn(|_| scalar(0));
        for block in preimage.chunks(4) {
            for (lane, value) in state.iter_mut().enumerate() {
                *value = value.clone() + block.get(lane).cloned().unwrap_or_else(|| scalar(0));
            }
            state = permutation(&mut builder, state, &constants)?;
        }
        state[0] = state[0].clone() + scalar(1);
        state = permutation(&mut builder, state, &constants)?;
        current = std::array::from_fn(|lane| state[lane].clone());
    }
    builder.finish(current)
}
