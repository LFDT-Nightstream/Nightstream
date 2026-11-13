//! Poseidon2 for layer 1.
//!
//! Owns: the workspace Poseidon2 permutation rebuilt in Plonky3 0.8 types, the
//! Merkle leaf sponge and node compression on it, Plonky3's Merkle tree over
//! them (the verifier of every commitment), the duplex challenger, and the
//! handoff that seeds the challenger from the fold transcript. No other hash
//! family appears. Provers build trees in `merkle.rs`.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, SEED, WIDTH};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use p3_challenger_v08::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    ResamplingError,
};
use p3_field_v08::{Field, PrimeField64};
use p3_merkle_tree_v08::MerkleTreeMmcs;
use p3_symmetric_v08::{
    CryptographicHasher, PaddingFreeSponge, Permutation, PseudoCompressionFunction, TruncatedPermutation,
};
use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::prelude::*;

use crate::field::{gl, Gl};

/// Names the layer-1 protocol in the fold transcript before the handoff.
pub(crate) const DOMAIN: &[u8] = b"Nightstream/SuperNeo/compress/v1";

/// The workspace permutation: the same seed and draw order as
/// `neo_ccs::crypto::poseidon2_goldilocks::PERM`. A test pins equal outputs.
pub(crate) type Perm = p3_goldilocks_v08::Poseidon2Goldilocks<WIDTH>;

/// The candidates of one parallel pass of `Challenger::grind`.
const GRIND_PASS: u64 = 1 << 18;

/// Plonky3's duplex challenger with one change: `grind` returns the smallest
/// proof-of-work witness. Plonky3 returns the first witness that a parallel
/// search finds, so two runs of one prover gave different proof bytes. A
/// verifier accepts each witness that passes `check_witness`, and the
/// smallest is one of them.
#[derive(Clone)]
pub(crate) struct Challenger(DuplexChallenger<Gl, Perm, WIDTH, RATE>);

impl Challenger {
    pub(crate) fn new(permutation: Perm) -> Self {
        Self(DuplexChallenger::new(permutation))
    }
}

impl<T> CanObserve<T> for Challenger
where
    DuplexChallenger<Gl, Perm, WIDTH, RATE>: CanObserve<T>,
{
    fn observe(&mut self, value: T) {
        self.0.observe(value);
    }
}

impl<T> CanSample<T> for Challenger
where
    DuplexChallenger<Gl, Perm, WIDTH, RATE>: CanSample<T>,
{
    fn sample(&mut self) -> T {
        self.0.sample()
    }
}

impl CanSampleBits<usize> for Challenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.0.sample_bits(bits)
    }
}

impl CanSampleUniformBits<Gl> for Challenger {
    fn sample_uniform_bits<const RESAMPLE: bool>(&mut self, bits: usize) -> Result<usize, ResamplingError> {
        self.0.sample_uniform_bits::<RESAMPLE>(bits)
    }
}

impl FieldChallenger<Gl> for Challenger {}

impl GrindingChallenger for Challenger {
    type Witness = Gl;

    fn grind(&mut self, bits: usize) -> Gl {
        assert!(
            bits < 64 && (1u64 << bits) < Gl::ORDER_U64,
            "bit count must fit the field"
        );
        if bits == 0 {
            return Gl::new(0);
        }
        let passes = |candidate: u64| self.0.clone().check_witness(bits, Gl::new(candidate));
        let witness = (0..Gl::ORDER_U64.div_ceil(GRIND_PASS))
            .find_map(|pass| {
                let end = ((pass + 1) * GRIND_PASS).min(Gl::ORDER_U64);
                (pass * GRIND_PASS..end)
                    .into_par_iter()
                    .find_first(|&candidate| passes(candidate))
            })
            .expect("a proof-of-work witness exists");
        assert!(self.0.check_witness(bits, Gl::new(witness)));
        Gl::new(witness)
    }
}

pub(crate) type Packed = <Gl as Field>::Packing;
/// The Merkle leaf hash: an overwrite-mode sponge.
pub(crate) type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_LEN>;
/// The Merkle node compression: the permutation of both children, truncated.
pub(crate) type Compress = TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>;
/// Plonky3's binary Merkle tree over `Sponge` and `Compress`, with the root
/// as the commitment.
pub(crate) type Plain = MerkleTreeMmcs<Packed, Packed, Sponge, Compress, 2, DIGEST_LEN>;

pub(crate) fn plain() -> Plain {
    let perm = permutation().clone();
    Plain::new(Sponge::new(perm.clone()), Compress::new(perm), 0)
}

pub(crate) fn permutation() -> &'static Perm {
    static PERM: OnceLock<Perm> = OnceLock::new();
    PERM.get_or_init(|| Perm::new_from_rng_128(&mut ChaCha8Rng::from_seed(SEED)))
}

/// The Merkle leaf hash: an overwrite-mode sponge over `values`.
pub(crate) fn hash_leaf(values: &[Gl]) -> [Gl; DIGEST_LEN] {
    Sponge::new(permutation().clone()).hash_slice(values)
}

/// The Merkle node compression of a left and a right child.
pub(crate) fn compress(left: [Gl; DIGEST_LEN], right: [Gl; DIGEST_LEN]) -> [Gl; DIGEST_LEN] {
    Compress::new(permutation().clone()).compress([left, right])
}

/// One sponge block absorbed into `state` in place, as `hash_leaf` does:
/// overwrite `state[at..at + values.len()]`; permute when the rate is full.
/// Returns the new fill position. Used to hash many leaves in lockstep, one
/// leaf per lane when `T` is `Packed`.
pub(crate) fn absorb<T: Copy>(state: &mut [T; WIDTH], mut at: usize, values: &[T]) -> usize
where
    Perm: Permutation<[T; WIDTH]>,
{
    for &value in values {
        state[at] = value;
        at += 1;
        if at == RATE {
            permutation().permute_mut(state);
            at = 0;
        }
    }
    at
}

/// Close a lockstep sponge: permute a partial block, then squeeze.
pub(crate) fn squeeze<T: Copy>(state: &mut [T; WIDTH], at: usize) -> [T; DIGEST_LEN]
where
    Perm: Permutation<[T; WIDTH]>,
{
    if at != 0 {
        permutation().permute_mut(state);
    }
    std::array::from_fn(|lane| state[lane])
}

/// Take ownership of the fold transcript, name layer 1 in it, and return a
/// four-word digest of everything absorbed so far: the layer-1 seed.
pub(crate) fn seed(mut transcript: Poseidon2Transcript) -> [Gl; 4] {
    transcript.absorb_v1_2(&domain_chunk_v1_2(DOMAIN));
    transcript.squeeze_digest_v1_2().map(gl)
}

/// A layer-1 challenger that starts from `seed`.
pub(crate) fn challenger(seed: [Gl; 4]) -> Challenger {
    let mut challenger = Challenger::new(permutation().clone());
    challenger.observe_slice(&seed);
    challenger
}
