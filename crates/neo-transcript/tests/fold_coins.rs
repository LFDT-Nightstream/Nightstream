//! The v1.2 fold-transcript coin schedule: the `n`-th coin of a state is the
//! rate-lane pair `n`, a zero chunk follows the sixth coin, and every absorb
//! starts a new state at pair 0.

use neo_math::F;
use neo_transcript::{fold_domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::PrimeCharacteristicRing;

const RATE: usize = 12;

fn pair(state: &[F; 16], pair: usize) -> [F; 2] {
    [state[2 * pair], state[2 * pair + 1]]
}

#[test]
fn coins_walk_the_rate_pairs_and_refresh_after_the_sixth() {
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&fold_domain_chunk_v1_2());
    let first = transcript.state();

    for index in 0..RATE / 2 {
        let coin = transcript.read_coin_v1_2();
        assert_eq!(coin.value, pair(&first, index), "coin {index} reads pair {index}");
        assert_eq!(coin.refreshed, index == RATE / 2 - 1, "only the sixth coin refreshes");
    }

    let mut expected = Poseidon2Transcript::from_state_and_absorbed(first, 0);
    expected.absorb_v1_2(&[F::ZERO; RATE]);
    let second = transcript.state();
    assert_eq!(second, expected.state(), "the refresh absorbs one zero chunk");
    assert_ne!(second, first, "the seventh coin comes from a new state");

    let coin = transcript.read_coin_v1_2();
    assert_eq!(coin.value, pair(&second, 0));
    assert!(!coin.refreshed);
}

#[test]
fn an_absorb_restarts_the_coins_at_pair_zero() {
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&fold_domain_chunk_v1_2());
    transcript.read_coin_v1_2();
    transcript.read_coin_v1_2();

    transcript.absorb_v1_2(&[F::ONE]);
    let state = transcript.state();
    let coin = transcript.read_coin_v1_2();
    assert_eq!(
        coin.value,
        pair(&state, 0),
        "the first coin after an absorb reads pair 0"
    );
    assert!(!coin.refreshed);
}
