//! The light step circuit accepts step 0 of the Poseidon2 hash chain, built
//! the way Lean's `LightAssemblerSource.honestValues` builds it, with a zero
//! light fold of the default slot (the zero proof that today's base step
//! uses). Every row of the light relation holds on the witness that the
//! package's program fills.

use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::D;
use neo_params::NeoParams;
use neo_reductions::common::{sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    derive_light_transcript, encode_digest, load_option_b_packages, KWords, LightWords, OptionBState, Slot,
    OPTION_B_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as G;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn words(values: [u64; 4]) -> [G; 4] {
    values.map(G::from_u64)
}

/// `Poseidon2HashChainV1.step`: Poseidon2 over the domain tag, the state
/// and the message.
fn hash_chain_step(state: [G; 4], message: [G; 4]) -> [G; 4] {
    let mut input: Vec<G> = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|&byte| G::from_u64(u64::from(byte)))
        .collect();
    input.extend(state);
    input.extend(message);
    poseidon2_hash(&input)
}

fn k_words(value: [u64; 2]) -> KWords {
    value.map(G::from_u64)
}

#[test]
fn light_step_zero_holds_every_row() {
    let packages = load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        OPTION_B_PACKAGES_IDENTITY,
    )
    .expect("option-B packages");
    let light = packages.light();
    let keys = [words([1, 2, 3, 4]), words([5, 6, 7, 8])];
    let z0 = words([10, 11, 12, 13]);
    let message = words([20, 21, 22, 23]);
    let next = hash_chain_step(z0, message);
    let default = Slot::default_slot();
    let defaults = [default.clone(), default.clone()];
    let state = |iteration, current, pc| OptionBState {
        keys,
        iteration,
        z0,
        current,
        pc,
        slots: defaults.clone(),
    };
    let fresh_public_input = encode_digest(state(0, z0, 0).hash());
    let output = state(1, next, 1).hash();

    let canonical = |values: &[G]| {
        values
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>()
    };
    let rounds = vec![vec![[0u64; 2]; 9]; 28];
    let transcript = derive_light_transcript(&[0; 1188], &canonical(&fresh_public_input), &rounds, &vec![0; 2 * 540])
        .expect("light transcript");
    let mut sampler = Poseidon2Transcript::from_state_and_absorbed(transcript.outgoing_state().map(G::from_u64), 0);
    let rho = sample_rot_rhos_n_typed(
        &mut sampler,
        &NeoParams::nightstream_goldilocks_k16(),
        &RotRing::goldilocks(),
        1,
    )
    .expect("light challenge")
    .remove(0);
    let folded_public_input: Vec<G> = (0..270)
        .map(|column| {
            let (lane, block) = (column % D, column / D);
            (0..D).fold(G::ZERO, |sum, source| {
                sum + rho.as_mat()[(lane, source)] * fresh_public_input[block * D + source]
            })
        })
        .collect();
    let folded = Slot {
        point: transcript
            .round_point()
            .iter()
            .copied()
            .map(k_words)
            .collect(),
        public_input: folded_public_input,
        depth: 1,
        ..default.clone()
    };

    let step = LightWords {
        iteration: 0,
        prior_pc: 0,
        selected: 0,
        keys,
        states: [z0, z0, next],
        witness: message.to_vec(),
        slots: [
            default.clone(),
            default.clone(),
            default.clone(),
            folded,
            default.clone(),
            default.clone(),
            default.clone(),
        ],
        digests: [
            default.digest(),
            default.digest(),
            default.digest(),
            default.digest(),
            output,
        ],
        fresh_commitment: vec![G::ZERO; 1188],
        fresh_public_input,
        rounds: vec![[[G::ZERO; 2]; 9]; 28],
        outputs: [vec![[G::ZERO; 2]; 270], vec![[G::ZERO; 2]; 270]],
    };
    let shared = step.columns().expect("light step words");
    let rows = light.plan_rows().expect("plan rows");
    let columns = light.circuit_columns(&shared).expect("circuit columns");
    let columns = light
        .relation_columns(columns, &rows)
        .expect("relation columns");
    let failing: Vec<usize> = (0..rows.row_count())
        .filter(|&row| rows.residual(row, &columns) != G::ZERO)
        .collect();
    assert!(
        failing.is_empty(),
        "{} rows fail, first {:?}",
        failing.len(),
        &failing[..failing.len().min(8)]
    );
}
