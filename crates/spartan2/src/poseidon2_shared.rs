//! Owns the shared Goldilocks Poseidon2 permutation cache for Spartan.

use neo_params::poseidon2_goldilocks::{SEED, WIDTH};
use once_cell::sync::Lazy;
use p3_goldilocks::Poseidon2Goldilocks;
use rand_chacha_p3::ChaCha8Rng;
use rand_chacha_p3::rand_core::SeedableRng;

static POSEIDON2_PERM: Lazy<Poseidon2Goldilocks<{ WIDTH }>> = Lazy::new(|| {
  let mut rng = ChaCha8Rng::from_seed(SEED);
  Poseidon2Goldilocks::<{ WIDTH }>::new_from_rng_128(&mut rng)
});

pub(crate) fn goldilocks_poseidon2_perm() -> &'static Poseidon2Goldilocks<{ WIDTH }> {
  &POSEIDON2_PERM
}
