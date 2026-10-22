//! Production Poseidon2 Implementation over Goldilocks Field

use once_cell::sync::Lazy;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use rand_chacha_p3::rand_core::SeedableRng;
use rand_chacha_p3::ChaCha8Rng;

pub use neo_params::poseidon2_goldilocks::{CAPACITY, DIGEST_LEN, RATE, SEED, WIDTH};

pub static PERM: Lazy<Poseidon2Goldilocks<{ WIDTH }>> = Lazy::new(|| {
    let mut rng = ChaCha8Rng::from_seed(SEED);
    Poseidon2Goldilocks::<{ WIDTH }>::new_from_rng_128(&mut rng)
});

pub fn permutation() -> &'static Poseidon2Goldilocks<{ WIDTH }> {
    &PERM
}

#[inline]
pub fn permute_state(state: [Goldilocks; WIDTH]) -> [Goldilocks; WIDTH] {
    permutation().permute(state)
}

/// Regenerated from `SEED` with
/// the same draw order as `Poseidon2::new_from_rng_128` (external constants,
/// then internal).
pub struct Poseidon2RoundConstants {
    pub initial: Vec<[u64; WIDTH]>,
    /// Internal rounds, one constant (added to lane 0) per round.
    pub internal: Vec<u64>,
    pub terminal: Vec<[u64; WIDTH]>,
    /// Diagonal of the internal linear layer `1 + diag(v)`.
    pub diag: [u64; WIDTH],
}

pub fn round_constants() -> Poseidon2RoundConstants {
    use p3_field::PrimeField64;
    use p3_poseidon2::ExternalLayerConstants;
    use rand_p3::distr::StandardUniform;
    use rand_p3::RngExt as _;

    let mut rng = ChaCha8Rng::from_seed(SEED);
    let (rounds_f, rounds_p) = p3_poseidon2::poseidon2_round_numbers_128::<Goldilocks>(
        WIDTH,
        p3_goldilocks::poseidon1::GOLDILOCKS_S_BOX_DEGREE,
    )
    .expect("round numbers for Goldilocks width 16");
    let external = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(rounds_f, &mut rng);
    let internal: Vec<Goldilocks> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(rounds_p)
        .collect();

    let row = |r: &[Goldilocks; WIDTH]| core::array::from_fn(|i| r[i].as_canonical_u64());
    Poseidon2RoundConstants {
        initial: external.get_initial_constants().iter().map(row).collect(),
        internal: internal.iter().map(|c| c.as_canonical_u64()).collect(),
        terminal: external.get_terminal_constants().iter().map(row).collect(),
        diag: core::array::from_fn(|i| p3_goldilocks::MATRIX_DIAG_16_GOLDILOCKS[i].as_canonical_u64()),
    }
}

pub fn poseidon2_hash(input: &[Goldilocks]) -> [Goldilocks; DIGEST_LEN] {
    let perm = permutation();
    let mut state = [Goldilocks::ZERO; WIDTH];

    for chunk in input.chunks(RATE) {
        for (i, &x) in chunk.iter().enumerate() {
            state[i] += x;
        }
        state = perm.permute(state);
    }

    state[0] += Goldilocks::ONE;
    state = perm.permute(state);

    let mut out = [Goldilocks::ZERO; DIGEST_LEN];
    out.copy_from_slice(&state[..DIGEST_LEN]);
    out
}

#[derive(Default)]
pub struct Poseidon2Hasher {
    state: [Goldilocks; WIDTH],
    used: usize,
}

impl Poseidon2Hasher {
    pub fn update(&mut self, input: &[Goldilocks]) {
        for &value in input {
            self.state[self.used] += value;
            self.used += 1;
            if self.used == RATE {
                self.state = permutation().permute(self.state);
                self.used = 0;
            }
        }
    }

    pub fn finalize(mut self) -> [Goldilocks; DIGEST_LEN] {
        if self.used != 0 {
            self.state = permutation().permute(self.state);
        }
        self.state[0] += Goldilocks::ONE;
        self.state = permutation().permute(self.state);
        self.state[..DIGEST_LEN]
            .try_into()
            .expect("digest fits state")
    }
}

pub fn poseidon2_hash_single(x: Goldilocks) -> [Goldilocks; DIGEST_LEN] {
    poseidon2_hash(&[x])
}
