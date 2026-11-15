//! The width-12 permutation and its grouped rounds against the known answers of
//! `scripts/poseidon2/vectors.json` (`decisions/poseidon2-p64-instance.md`).

use neo_application::poseidon2::{
    apply_full_round, apply_initial_linear, apply_partial_pair, permute, POSEIDON2_HALF_FULL_ROUNDS,
    POSEIDON2_PARTIAL_PAIRS, POSEIDON2_WIDTH,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;

#[derive(Deserialize)]
struct Vectors {
    q: String,
    vectors: Vec<Vector>,
}

#[derive(Deserialize)]
struct Vector {
    t: usize,
    name: String,
    input: Vec<String>,
    output: Vec<String>,
}

fn state(words: &[String]) -> [F; POSEIDON2_WIDTH] {
    assert_eq!(words.len(), POSEIDON2_WIDTH);
    core::array::from_fn(|lane| {
        let word: u64 = words[lane].parse().expect("a decimal word");
        assert!(word < F::ORDER_U64, "a vector word must be canonical");
        F::from_u64(word)
    })
}

fn grouped(mut state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    apply_initial_linear(&mut state);
    for round in 0..POSEIDON2_HALF_FULL_ROUNDS {
        apply_full_round(round, &mut state);
    }
    for pair in 0..POSEIDON2_PARTIAL_PAIRS {
        apply_partial_pair(pair, &mut state);
    }
    for round in POSEIDON2_HALF_FULL_ROUNDS..2 * POSEIDON2_HALF_FULL_ROUNDS {
        apply_full_round(round, &mut state);
    }
    state
}

#[test]
fn width_12_permutation_and_grouped_rounds_match_the_known_answers() {
    let vectors: Vectors =
        serde_json::from_str(include_str!("../../../scripts/poseidon2/vectors.json")).expect("vectors.json");
    assert_eq!(vectors.q, F::ORDER_U64.to_string());
    let cases: Vec<_> = vectors
        .vectors
        .iter()
        .filter(|vector| vector.t == POSEIDON2_WIDTH)
        .collect();
    assert_eq!(cases.len(), 5, "vectors.json has five width-12 vectors");
    for case in cases {
        let (input, output) = (state(&case.input), state(&case.output));
        assert_eq!(permute(input), output, "{}", case.name);
        assert_eq!(grouped(input), output, "grouped rounds, {}", case.name);
    }
}
