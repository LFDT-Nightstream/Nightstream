//! Poseidon2 for layer 1.
//!
//! Owns: the workspace Poseidon2 permutation rebuilt in Plonky3 0.8 types, the
//! Merkle tree and duplex challenger built on it, and the handoff that seeds
//! the challenger from the fold transcript. No other hash family appears.
//! Merkle leaves hash in batches, on the prover's device when it has one.

use std::any::TypeId;
use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, SEED, WIDTH};
use neo_transcript::{domain_chunk_v1_1, Poseidon2Transcript};
use p3_challenger_v08::{CanObserve, DuplexChallenger};
use p3_field_v08::{Field, PackedValue};
use p3_merkle_tree_v08::MerkleTreeMmcs;
use p3_symmetric_v08::{
    CryptographicHasher, CryptographicPermutation, PaddingFreeSponge, Permutation, PseudoCompressionFunction,
    TruncatedPermutation,
};
use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::prelude::*;

use crate::device::Accelerator;
use crate::field::{gl, Gl};

/// Names the layer-1 protocol in the fold transcript before the handoff.
pub(crate) const DOMAIN: &[u8] = b"Nightstream/SuperNeo/compress/v1";

/// The workspace permutation: the same seed and draw order as
/// `neo_ccs::crypto::poseidon2_goldilocks::PERM`. A test pins equal outputs.
pub(crate) type Perm = p3_goldilocks_v08::Poseidon2Goldilocks<WIDTH>;
pub(crate) type Challenger = DuplexChallenger<Gl, Perm, WIDTH, RATE>;

type Packed = <Gl as Field>::Packing;
type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_LEN>;
type Compress = TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>;
pub(crate) type Mmcs = MerkleTreeMmcs<Packed, Packed, LeafHash, Compress, 2, DIGEST_LEN>;

/// Rows per leaf batch. Plonky3 0.8's tree passes whole batches of this many
/// rows to `hash_many` once a hasher reports more than one lane.
const LEAF_BATCH: usize = 1 << 16;

/// The Merkle leaf hash: Plonky3's overwrite sponge, in batches. A batch runs
/// on the prover's device when it has one; otherwise its rows run in
/// parallel on the packed permutation, as in Plonky3's packed path. Both give
/// the sponge's digests.
#[derive(Clone)]
pub(crate) struct LeafHash {
    sponge: Sponge,
    accelerator: Accelerator,
}

impl<T> CryptographicHasher<T, [T; DIGEST_LEN]> for LeafHash
where
    T: Default + Copy + Send + Sync + 'static,
    Perm: CryptographicPermutation<[T; WIDTH]>,
{
    const LANES: usize = LEAF_BATCH;

    fn hash_iter<I>(&self, input: I) -> [T; DIGEST_LEN]
    where
        I: IntoIterator<Item = T>,
    {
        self.sponge.hash_iter(input)
    }

    /// Plonky3 batches only scalar rows, so only `T = Gl` reaches the device
    /// or the parallel path.
    fn hash_many(&self, input: &[T], out: &mut [[T; DIGEST_LEN]]) {
        if out.is_empty() {
            return;
        }
        assert!(input.len().is_multiple_of(out.len()), "a ragged leaf batch");
        let width = input.len() / out.len();
        let (Some(input), Some(out)) = (as_field(input), as_field_mut(out)) else {
            for (digest, row) in out.iter_mut().zip(input.chunks(width.max(1))) {
                *digest = self.sponge.hash_slice(if width == 0 { &[] } else { row });
            }
            return;
        };
        let on_device = self
            .accelerator
            .device()
            .filter(|_| width > 0)
            .and_then(|device| {
                // SAFETY: `Goldilocks` is `repr(transparent)` over `u64`.
                let words = unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u64>(), input.len()) };
                device.hash_rows(words, width)
            });
        match on_device {
            Some(digests) => {
                assert_eq!(digests.len(), out.len(), "the device hashed another row count");
                for (digest, words) in out.iter_mut().zip(digests) {
                    *digest = words.map(Gl::new);
                }
            }
            None => hash_packed(&self.sponge, input, width, out),
        }
    }
}

fn as_field<T: 'static>(values: &[T]) -> Option<&[Gl]> {
    // SAFETY: the types are equal.
    (TypeId::of::<T>() == TypeId::of::<Gl>())
        .then(|| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Gl>(), values.len()) })
}

fn as_field_mut<T: 'static>(values: &mut [[T; DIGEST_LEN]]) -> Option<&mut [[Gl; DIGEST_LEN]]> {
    // SAFETY: the types are equal.
    (TypeId::of::<T>() == TypeId::of::<Gl>()).then(|| unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<[Gl; DIGEST_LEN]>(), values.len())
    })
}

/// The rows of `input` hashed `Packed::WIDTH` at a time, in parallel.
fn hash_packed(sponge: &Sponge, input: &[Gl], width: usize, out: &mut [[Gl; DIGEST_LEN]]) {
    if width == 0 {
        out.fill(sponge.hash_slice(&[]));
        return;
    }
    let lanes = Packed::WIDTH;
    input
        .par_chunks(width * lanes)
        .zip(out.par_chunks_mut(lanes))
        .for_each(|(rows, digests)| {
            if digests.len() < lanes {
                for (digest, row) in digests.iter_mut().zip(rows.chunks_exact(width)) {
                    *digest = sponge.hash_slice(row);
                }
                return;
            }
            let packed: [Packed; DIGEST_LEN] =
                sponge.hash_iter((0..width).map(|i| Packed::from_fn(|lane| rows[lane * width + i])));
            for (lane, digest) in digests.iter_mut().enumerate() {
                *digest = std::array::from_fn(|k| packed[k].as_slice()[lane]);
            }
        });
}

pub(crate) fn permutation() -> &'static Perm {
    static PERM: OnceLock<Perm> = OnceLock::new();
    PERM.get_or_init(|| Perm::new_from_rng_128(&mut ChaCha8Rng::from_seed(SEED)))
}

/// The Merkle commitment scheme; its leaves hash on `accelerator`.
pub(crate) fn mmcs(accelerator: &Accelerator) -> Mmcs {
    let perm = permutation().clone();
    let leaves = LeafHash {
        sponge: Sponge::new(perm.clone()),
        accelerator: accelerator.clone(),
    };
    Mmcs::new(leaves, Compress::new(perm), 0)
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
/// Returns the new fill position. Used to hash many leaves in lockstep.
pub(crate) fn absorb(state: &mut [Gl; WIDTH], mut at: usize, values: &[Gl]) -> usize {
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
pub(crate) fn squeeze(state: &mut [Gl; WIDTH], at: usize) -> [Gl; DIGEST_LEN] {
    if at != 0 {
        permutation().permute_mut(state);
    }
    std::array::from_fn(|lane| state[lane])
}

/// Take ownership of the fold transcript, name layer 1 in it, and return a
/// four-word digest of everything absorbed so far: the layer-1 seed.
pub(crate) fn seed(mut transcript: Poseidon2Transcript) -> [Gl; 4] {
    transcript.absorb_v1_1(&domain_chunk_v1_1(DOMAIN));
    transcript.squeeze_digest_v1_1().map(gl)
}

/// A layer-1 challenger that starts from `seed`.
pub(crate) fn challenger(seed: [Gl; 4]) -> Challenger {
    let mut challenger = Challenger::new(permutation().clone());
    challenger.observe_slice(&seed);
    challenger
}
