//! Poseidon2 over `F_q`, `q = 2^64 − 59` (`decisions/poseidon2-p64-instance.md`).
//! The layers and the order of the constants are those of `p3-poseidon2`.

use neo_math::Fq;
use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{matmul_internal, mds_light_permutation, MDSMat4};

pub use super::poseidon2_constants::{INSTANCE_12, INSTANCE_16};
pub use neo_params::poseidon2::{CAPACITY, DIGEST_LEN, RATE, WIDTH};

pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 22;

/// The constants of one width. `scripts/poseidon2/emit_rust.py` writes the
/// two instances from `scripts/poseidon2/params.json`.
pub struct Instance<const T: usize> {
    pub initial: [[Fq; T]; HALF_FULL_ROUNDS],
    pub terminal: [[Fq; T]; HALF_FULL_ROUNDS],
    /// One constant for each partial round, added to lane 0.
    pub internal: [Fq; PARTIAL_ROUNDS],
    /// The internal linear layer is `J + diag(diagonal)`.
    pub diagonal: [Fq; T],
}

#[inline(always)]
fn sbox(x: Fq) -> Fq {
    InjectiveMonomial::<7>::injective_exp_n(&x)
}

impl<const T: usize> Instance<T> {
    pub fn permute(&self, mut state: [Fq; T]) -> [Fq; T] {
        let full_round = |state: &mut [Fq; T], constants: &[Fq; T]| {
            for (lane, &constant) in state.iter_mut().zip(constants) {
                *lane = sbox(*lane + constant);
            }
            mds_light_permutation(state, &MDSMat4);
        };
        mds_light_permutation(&mut state, &MDSMat4);
        for constants in &self.initial {
            full_round(&mut state, constants);
        }
        for &constant in &self.internal {
            state[0] = sbox(state[0] + constant);
            matmul_internal(&mut state, self.diagonal);
        }
        for constants in &self.terminal {
            full_round(&mut state, constants);
        }
        state
    }
}

/// The width-16 permutation of the transcript and of the sponge below.
#[inline]
pub fn permute_state(state: [Fq; WIDTH]) -> [Fq; WIDTH] {
    INSTANCE_16.permute(state)
}

pub fn poseidon2_hash(input: &[Fq]) -> [Fq; DIGEST_LEN] {
    let mut hasher = Poseidon2Hasher::default();
    hasher.update(input);
    hasher.finalize()
}

#[derive(Default)]
pub struct Poseidon2Hasher {
    state: [Fq; WIDTH],
    used: usize,
}

impl Poseidon2Hasher {
    pub fn update(&mut self, input: &[Fq]) {
        for &value in input {
            self.state[self.used] += value;
            self.used += 1;
            if self.used == RATE {
                self.state = permute_state(self.state);
                self.used = 0;
            }
        }
    }

    pub fn finalize(mut self) -> [Fq; DIGEST_LEN] {
        if self.used != 0 {
            self.state = permute_state(self.state);
        }
        self.state[0] += Fq::ONE;
        self.state = permute_state(self.state);
        self.state[..DIGEST_LEN]
            .try_into()
            .expect("digest fits state")
    }
}

pub fn poseidon2_hash_single(x: Fq) -> [Fq; DIGEST_LEN] {
    poseidon2_hash(&[x])
}
