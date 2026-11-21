//! Poseidon2 over `F_q`, `q = 2^64 − 59` (`decisions/poseidon2-p64-instance.md`).

use neo_math::Fq;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use p3_field::PrimeField64;
use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{matmul_internal, mds_light_permutation, MDSMat4};

pub use super::poseidon2_constants::{INSTANCE_12, INSTANCE_16};
pub use neo_params::poseidon2::{CAPACITY, DIGEST_LEN, RATE, WIDTH};

pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 22;
pub const PERMUTATION_LOCALS: usize = WIDTH + 2 * HALF_FULL_ROUNDS * 5 * WIDTH + PARTIAL_ROUNDS * (4 + WIDTH);
/// Lean `Spec.Poseidon2.sbox_eq_pow`.
pub const SBOX_DEGREE: u64 = 7;

pub struct Instance<const T: usize> {
    pub initial: [[Fq; T]; HALF_FULL_ROUNDS],
    pub terminal: [[Fq; T]; HALF_FULL_ROUNDS],
    pub internal: [Fq; PARTIAL_ROUNDS],
    /// The internal linear layer is `J + diag(diagonal)`.
    pub diagonal: [Fq; T],
}

#[inline(always)]
fn sbox(x: Fq) -> Fq {
    InjectiveMonomial::<7>::injective_exp_n(&x)
}

impl<const T: usize> Instance<T> {
    pub fn permute(&self, state: [Fq; T]) -> [Fq; T] {
        self.permute_rounds::<true, _>(state, |_| {})
    }

    pub fn permute_with_sboxes<C>(&self, state: [Fq; T], record: C) -> [Fq; T]
    where
        C: FnMut(&[Fq]),
    {
        self.permute_rounds::<false, _>(state, record)
    }

    fn permute_rounds<const WIDEN: bool, C>(&self, mut state: [Fq; T], mut record: C) -> [Fq; T]
    where
        C: FnMut(&[Fq]),
    {
        let full_round = |state: &mut [Fq; T], constants: &[Fq; T], record: &mut C| {
            for (lane, &constant) in state.iter_mut().zip(constants) {
                *lane = sbox(*lane + constant);
            }
            record(state);
            mds_light_permutation(state, &MDSMat4);
        };
        mds_light_permutation(&mut state, &MDSMat4);
        for constants in &self.initial {
            full_round(&mut state, constants, &mut record);
        }
        for &constant in &self.internal {
            state[0] = sbox(state[0] + constant);
            record(&state[..1]);
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if WIDEN && T == 16 {
                const {
                    let max = Fq::ORDER_U64 as u128 - 1;
                    assert!(max.checked_mul(max + 16).is_some());
                }
                let wide = |word: Fq| u128::from(word.as_canonical_u64());
                let sum = state.iter().copied().map(wide).sum::<u128>();
                for (word, &coefficient) in state.iter_mut().zip(&self.diagonal) {
                    *word = Fq::from_u128(sum + wide(*word) * wide(coefficient));
                }
                continue;
            }
            matmul_internal(&mut state, self.diagonal);
        }
        for constants in &self.terminal {
            full_round(&mut state, constants, &mut record);
        }
        state
    }
}

#[inline]
pub fn permute_state(state: [Fq; WIDTH]) -> [Fq; WIDTH] {
    INSTANCE_16.permute(state)
}

/// Lean `Gadgets/Poseidon2/Permutation.lean`.
pub fn permutation_locals(mut state: [Fq; WIDTH], locals: &mut [Fq; PERMUTATION_LOCALS]) {
    let mut locals = locals.as_mut_slice();
    mds_light_permutation(&mut state, &MDSMat4);
    write_locals(&state, &mut locals);
    for constants in &INSTANCE_16.initial {
        full_round_locals(&mut state, constants, &mut locals);
    }
    for &constant in &INSTANCE_16.internal {
        state[0] = sbox_locals(state[0] + constant, &mut locals);
        matmul_internal(&mut state, INSTANCE_16.diagonal);
        write_locals(&state, &mut locals);
    }
    for constants in &INSTANCE_16.terminal {
        full_round_locals(&mut state, constants, &mut locals);
    }
    debug_assert!(locals.is_empty());
}

fn full_round_locals(state: &mut [Fq; WIDTH], constants: &[Fq; WIDTH], locals: &mut &mut [Fq]) {
    for (lane, &constant) in state.iter_mut().zip(constants) {
        *lane = sbox_locals(*lane + constant, locals);
    }
    mds_light_permutation(state, &MDSMat4);
    write_locals(state, locals);
}

fn sbox_locals(value: Fq, locals: &mut &mut [Fq]) -> Fq {
    let square = value * value;
    let fourth = square * square;
    let sixth = fourth * square;
    let seventh = sixth * value;
    write_locals(&[square, fourth, sixth, seventh], locals);
    seventh
}

fn write_locals(values: &[Fq], locals: &mut &mut [Fq]) {
    let (words, rest) = std::mem::take(locals).split_at_mut(values.len());
    words.copy_from_slice(values);
    *locals = rest;
}

/// Lean `Spec.Poseidon2.hash`.
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
        self.state[self.used] += Fq::ONE;
        self.state = permute_state(self.state);
        self.state[..DIGEST_LEN]
            .try_into()
            .expect("digest fits state")
    }
}

pub fn poseidon2_hash_single(x: Fq) -> [Fq; DIGEST_LEN] {
    poseidon2_hash(&[x])
}
