//! Poseidon2 for layer 1.
//!
//! Owns: the workspace Poseidon2 permutation rebuilt in Plonky3 0.8 types, the
//! Merkle tree and duplex challenger built on it, and the handoff that seeds
//! the challenger from the fold transcript. No other hash family appears.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, SEED, WIDTH};
use neo_transcript::{domain_chunk_v1_1, Poseidon2Transcript};
use p3_challenger_v08::{CanObserve, DuplexChallenger};
use p3_field_v08::Field;
use p3_merkle_tree_v08::MerkleTreeMmcs;
use p3_symmetric_v08::{PaddingFreeSponge, TruncatedPermutation};
use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};

use crate::field::{gl, Gl};

/// Names the layer-1 protocol in the fold transcript before the handoff.
pub(crate) const DOMAIN: &[u8] = b"Nightstream/SuperNeo/compress/v1";

/// The workspace permutation: the same seed and draw order as
/// `neo_ccs::crypto::poseidon2_goldilocks::PERM`. A test pins equal outputs.
pub(crate) type Perm = p3_goldilocks_v08::Poseidon2Goldilocks<WIDTH>;
pub(crate) type Challenger = DuplexChallenger<Gl, Perm, WIDTH, RATE>;

type Packed = <Gl as Field>::Packing;
type LeafHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_LEN>;
type Compress = TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>;
pub(crate) type Mmcs = MerkleTreeMmcs<Packed, Packed, LeafHash, Compress, 2, DIGEST_LEN>;

pub(crate) fn permutation() -> &'static Perm {
    static PERM: OnceLock<Perm> = OnceLock::new();
    PERM.get_or_init(|| Perm::new_from_rng_128(&mut ChaCha8Rng::from_seed(SEED)))
}

pub(crate) fn mmcs() -> Mmcs {
    let perm = permutation().clone();
    Mmcs::new(LeafHash::new(perm.clone()), Compress::new(perm), 0)
}

/// Take ownership of the fold transcript, name layer 1 in it, and seed the
/// layer-1 challenger with a four-word digest of everything absorbed so far.
pub(crate) fn challenger(mut transcript: Poseidon2Transcript) -> Challenger {
    transcript.absorb_v1_1(&domain_chunk_v1_1(DOMAIN));
    let seed = transcript.squeeze_digest_v1_1();
    let mut challenger = Challenger::new(permutation().clone());
    challenger.observe_slice(&seed.map(gl));
    challenger
}
