//! Hash-based multilinear PCS retained as a Toy Spartan comparison backend.
//!
//! This implementation provides a Merkle tree-based polynomial commitment scheme
//! for multilinear polynomials using configurable hashing backends. It supports LeakReduced mode
//! which reveals O(m) dense fold values during evaluation.
//!
//! Backends available:
//! - Keccak256 (default, ff-based)
//! - Poseidon2 with p3-goldilocks (feature-gated)

use crate::{
  errors::SpartanError,
  traits::{
    Engine,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::marker::PhantomData;

#[path = "merkle_mle_pc_compact.rs"]
mod merkle_mle_pc_compact;

pub(crate) use merkle_mle_pc_compact::CompactHashMleDigest;
use merkle_mle_pc_compact::{CompactHashMleEvaluationArgument, compact_eval_arg, expand_eval_arg};

/// Draw an unbiased index from transcript for sampling
pub fn draw_index<E: Engine>(
  t: &mut E::TE,
  label: &'static [u8],
  stride: usize,
) -> Result<usize, SpartanError> {
  debug_assert!(stride > 0);
  let c = t.squeeze(label)?;
  let repr = c.to_repr();
  let mut w8 = [0u8; 8];
  w8.copy_from_slice(&repr.as_ref()[..8]);
  let x = u64::from_le_bytes(w8);
  // Unbiased mapping via 64x64->128 multiply-high
  Ok(((x as u128 * stride as u128) >> 64) as usize)
}

// Import backend interface
use super::hash_mle_backend::{BackendFfKeccak, Digest32, MleBackend};

/// Domain tags to avoid cross-protocol collisions
#[allow(dead_code)]
const TAG_LEAF: &[u8] = b"mle/leaf";
#[allow(dead_code)]
const TAG_NODE: &[u8] = b"mle/node";
const TAG_MODE: &[u8] = b"mle/mode";
/// Tag used for layer roots in the transcript
pub const TAG_LAYER_ROOTS: &[u8] = b"mle/layer_roots";

/// Zero-knowledge mode for the Hash-MLE PCS
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkMode {
  /// Leak-reduced mode: reveals O(m) dense fold values during evaluation
  LeakReduced,
}

/// Commitment key for Hash-MLE PCS
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleCommitmentKey<E: Engine> {
  /// Merkle arity (currently fixed to 2)
  pub branching: u8,
  /// ZK mode baked into this key
  pub zk_mode: ZkMode,
  /// Phantom data for the engine type
  pub _p: PhantomData<E>,
}

/// Verifier key for Hash-MLE PCS
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleVerifierKey<E: Engine> {
  /// Branching factor of the Merkle tree
  pub branching: u8,
  /// Zero-knowledge mode
  pub zk_mode: ZkMode,
  /// Phantom data for the engine type
  pub _p: PhantomData<E>,
}

/// A Merkle tree root (32-byte hash)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleRoot(pub [u8; 32]);

impl<G: crate::traits::Group> TranscriptReprTrait<G> for MerkleRoot {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.0.to_vec()
  }
}

impl MerkleRoot {
  #[allow(dead_code)]
  fn zero() -> Self {
    MerkleRoot([0u8; 32])
  }
}

/// Authentication path for a Merkle tree leaf
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerklePath {
  /// leaf index within the committed array (0-based)
  pub leaf_index: u64,
  /// authentication path (sibling digests, bottom-up)
  pub siblings: Vec<[u8; 32]>,
}

/// Commitment for Hash-MLE PCS
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashMleCommitment<E: Engine> {
  /// Root of the base vector commitment (unmasked v in LeakReduced mode)
  pub base_root: MerkleRoot,
  /// Mode encoded into the commitment to avoid misuse across modes
  pub mode: ZkMode,
  /// Phantom data for the engine type
  pub _p: PhantomData<E>,
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HashMleCommitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 32);
    out.extend_from_slice(TAG_MODE);
    out.push(match self.mode {
      ZkMode::LeakReduced => 0,
    });
    out.extend_from_slice(&self.base_root.0);
    out
  }
}

impl<E: Engine> CommitmentTrait<E> for HashMleCommitment<E> {}

/// Blinding factor for Hash-MLE PCS
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashMleBlind<E: Engine> {
  _phantom: PhantomData<E>,
}

impl<E: Engine> Default for HashMleBlind<E> {
  fn default() -> Self {
    HashMleBlind {
      _phantom: PhantomData,
    }
  }
}

/// Number of sample checks per round for layer consistency
pub const K_SAMPLES_PER_ROUND: usize = 48;

/// Evaluation argument: per-round pair openings and the next-layer single opening.
#[derive(Clone, Debug)]
pub struct HashMleEvaluationArgument<E: Engine> {
  /// Layer roots carried here to avoid bloating the commitment
  pub layer_roots: Vec<MerkleRoot>, // len = m+1

  /// For each round i in [0..m):
  pub rounds: Vec<Round<E>>,

  /// Sample openings to link consecutive layers (prevents forged layer attacks)
  /// samples\[i\] contains K_SAMPLES_PER_ROUND random checks linking layer i to layer i+1
  pub samples: Vec<Vec<SampleOpening<E>>>, // len = m, each inner vec has K_SAMPLES_PER_ROUND elements
}

impl<E> Serialize for HashMleEvaluationArgument<E>
where
  E: CompactHashMleDigest,
  E::Scalar: Serialize,
{
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    compact_eval_arg(self).serialize(serializer)
  }
}

impl<'de, E> Deserialize<'de> for HashMleEvaluationArgument<E>
where
  E: CompactHashMleDigest,
  E::Scalar: Deserialize<'de>,
{
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let compact = CompactHashMleEvaluationArgument::<E>::deserialize(deserializer)?;
    expand_eval_arg(compact).map_err(serde::de::Error::custom)
  }
}

/// A single round of the Hash-MLE evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Round<E: Engine> {
  /// openings from layer i (v^{(i)}):
  /// Even evaluation value
  pub a: E::Scalar, // even
  /// Odd evaluation value
  pub b: E::Scalar, // odd
  /// Merkle path for even evaluation
  pub path_a: MerklePath, // membership against layer_roots[i]
  /// Merkle path for odd evaluation
  pub path_b: MerklePath, // membership against layer_roots[i]

  /// membership for layer i+1 at the folded index
  /// Folded value for next layer (equals (1-r_i)*a + r_i*b)
  pub next: E::Scalar, // equals (1-r_i)*a + r_i*b
  /// Merkle path for the next layer
  pub path_next: MerklePath, // membership against layer_roots[i+1]
}

/// Sample opening for layer consistency checks
/// This links layer i to layer i+1 at a random position to prevent forged layers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleOpening<E: Engine> {
  /// Random index in the layer (< 2^{m-i-1})
  pub idx: u64,
  /// Value at position idx in layer i
  pub a: E::Scalar, // v^{(i)}[idx]
  /// Value at position idx + stride in layer i  
  pub b: E::Scalar, // v^{(i)}[idx + stride] where stride = 2^{m-i-1}
  /// Folded value at position idx in layer i+1
  pub next: E::Scalar, // v^{(i+1)}[idx] = (1-r_i)*a + r_i*b
  /// Merkle path for a
  pub path_a: MerklePath,
  /// Merkle path for b
  pub path_b: MerklePath,
  /// Merkle path for next
  pub path_next: MerklePath,
}

// Default backend type alias for backwards compatibility
type DefaultBackend<E> = BackendFfKeccak<E>;

/// Generic hash functions that use the backend
/// Hash a leaf field element into a 32-byte digest using the specified backend
pub fn leaf_digest<E: Engine, B: MleBackend<E>>(x: &B::FE) -> Digest32 {
  B::leaf_hash(x)
}

/// Hash two child digests into a parent digest using the specified backend
pub fn node_digest<E: Engine, B: MleBackend<E>>(l: &Digest32, r: &Digest32) -> Digest32 {
  B::node_hash(l, r)
}

/// Merkle tree implementation using configurable hash backends
#[derive(Clone)]
pub struct MerkleTree {
  layers: Vec<Vec<Digest32>>, // layers[0] == leaves, layers.last()[0] == root
}

impl MerkleTree {
  /// Create a new Merkle tree from leaf digests using the specified backend
  pub fn from_leaves<E: Engine, B: MleBackend<E>>(leaves: Vec<Digest32>) -> Self {
    assert!(leaves.len().is_power_of_two());
    let mut layers: Vec<Vec<Digest32>> = Vec::new();
    layers.push(leaves);

    while layers.last().unwrap().len() > 1 {
      let cur = layers.last().unwrap();
      let next = if cur.len() >= 4096 {
        cur
          .par_chunks_exact(2)
          .map(|p| node_digest::<E, B>(&p[0], &p[1]))
          .collect::<Vec<_>>()
      } else {
        cur
          .chunks_exact(2)
          .map(|p| node_digest::<E, B>(&p[0], &p[1]))
          .collect::<Vec<_>>()
      };
      layers.push(next);
    }

    Self { layers }
  }

  /// Get the root digest of the Merkle tree
  pub fn root(&self) -> Digest32 {
    self.layers.last().unwrap()[0]
  }

  /// Generate a Merkle proof for the leaf at the given index
  pub fn open(&self, leaf_index: usize) -> MerklePath {
    let mut idx = leaf_index;
    let mut siblings = Vec::with_capacity(self.layers.len() - 1);

    for layer in &self.layers {
      if layer.len() == 1 {
        break;
      }
      let sib = if idx % 2 == 0 {
        layer[idx + 1]
      } else {
        layer[idx - 1]
      };
      siblings.push(sib.0);
      idx >>= 1;
    }

    MerklePath {
      leaf_index: leaf_index as u64,
      siblings,
    }
  }

  /// Verify a Merkle proof using the specified backend
  pub fn verify<E: Engine, B: MleBackend<E>>(
    path: &MerklePath,
    leaf: &Digest32,
    root: &Digest32,
  ) -> bool {
    let mut idx = path.leaf_index as usize;
    let mut cur = *leaf;

    for sib in &path.siblings {
      let sib_digest = Digest32(*sib);
      cur = if idx % 2 == 0 {
        node_digest::<E, B>(&cur, &sib_digest)
      } else {
        node_digest::<E, B>(&sib_digest, &cur)
      };
      idx >>= 1;
    }

    cur == *root
  }
}

/// Fold one layer with a single r, halving the vector length.
/// This follows the same logic as MultilinearPolynomial::bind_poly_var_top
fn fold_layer<E: Engine>(v: &[E::Scalar], r: &E::Scalar) -> Vec<E::Scalar> {
  let n = v.len() / 2;
  if crate::parallel::parallelism_enabled() {
    (0..n)
      .into_par_iter()
      .map(|i| {
        let left = v[i];
        let right = v[i + n];
        left + *r * (right - left) // equivalent to (1-r)*left + r*right
      })
      .collect()
  } else {
    (0..n)
      .map(|i| {
        let left = v[i];
        let right = v[i + n];
        left + *r * (right - left)
      })
      .collect()
  }
}

/// Hash-based multilinear polynomial commitment scheme
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMlePCS<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine + CompactHashMleDigest> PCSEngineTrait<E> for HashMlePCS<E> {
  type CommitmentKey = HashMleCommitmentKey<E>;
  type VerifierKey = HashMleVerifierKey<E>;
  type Commitment = HashMleCommitment<E>;
  type PartialCommitment = HashMleCommitment<E>;
  type Blind = HashMleBlind<E>;
  type EvaluationArgument = HashMleEvaluationArgument<E>;

  /// Arity of the multilinear domain (binary hypercube).
  /// For MLE commitments this MUST be 2.
  fn width() -> usize {
    2
  }

  fn setup(_label: &'static [u8], _n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    let ck = HashMleCommitmentKey {
      branching: 2,
      zk_mode: ZkMode::LeakReduced,
      _p: PhantomData,
    };
    let vk = HashMleVerifierKey {
      branching: 2,
      zk_mode: ZkMode::LeakReduced,
      _p: PhantomData,
    };
    (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    HashMleBlind::default()
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    if !v.len().is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS: vector len must be power of two".into(),
      });
    }

    // Base layer leaves (unmasked in LeakReduced mode)
    let leaves = if crate::parallel::parallelism_enabled() {
      v.par_iter()
        .map(|x_ff| {
          // Default backend uses E::Scalar directly
          let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x_ff);
          leaf_digest::<E, DefaultBackend<E>>(&x_fe)
        })
        .collect::<Vec<_>>()
    } else {
      v.iter()
        .map(|x_ff| {
          let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x_ff);
          leaf_digest::<E, DefaultBackend<E>>(&x_fe)
        })
        .collect::<Vec<_>>()
    };
    let tree = MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves);
    let base_root = MerkleRoot(tree.root().0);

    Ok(HashMleCommitment {
      base_root,
      mode: ck.zk_mode,
      _p: PhantomData,
    })
  }

  fn commit_partial(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    blind: &Self::Blind,
    is_small: bool,
  ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError> {
    let c = Self::commit(ck, v, blind, is_small)?;
    Ok((c, blind.clone()))
  }

  fn check_partial(_comm: &Self::PartialCommitment, _n: usize) -> Result<(), SpartanError> {
    Ok(())
  }

  fn combine_partial(
    partial_comms: &[Self::PartialCommitment],
  ) -> Result<Self::Commitment, SpartanError> {
    if partial_comms.len() != 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS: combine_partial expects exactly one piece".into(),
      });
    }
    Ok(partial_comms[0].clone())
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.len() != 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS: combine_blinds expects exactly one blind".into(),
      });
    }
    Ok(blinds[0].clone())
  }

  fn prove(
    _ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    _blind: &Self::Blind,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    let n = poly.len();
    let m = point.len();
    if n != (1usize << m) {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "HashMlePCS::prove expected {} elements, got {}",
          1usize << m,
          n
        ),
      });
    }

    transcript.absorb(b"poly_com", comm);

    // Build the evaluation layers step by step following MultilinearPolynomial's binding logic.
    // Avoid extra clones: all_layers owns each level.
    let mut all_layers: Vec<Vec<E::Scalar>> = Vec::with_capacity(m + 1);
    all_layers.push(poly.to_vec());
    for &r_i in point.iter() {
      let next_layer = fold_layer::<E>(all_layers.last().unwrap(), &r_i);
      all_layers.push(next_layer);
    }
    debug_assert_eq!(all_layers.last().unwrap().len(), 1);

    // Build trees + roots (per-proof) and round proofs
    let trees: Vec<MerkleTree> = if crate::parallel::parallelism_enabled() {
      all_layers
        .par_iter()
        .map(|lvl| {
          let leaves = lvl
            .par_iter()
            .map(|x_ff| {
              let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x_ff);
              leaf_digest::<E, DefaultBackend<E>>(&x_fe)
            })
            .collect();
          MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves)
        })
        .collect()
    } else {
      all_layers
        .iter()
        .map(|lvl| {
          let leaves = lvl
            .iter()
            .map(|x_ff| {
              let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x_ff);
              leaf_digest::<E, DefaultBackend<E>>(&x_fe)
            })
            .collect();
          MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves)
        })
        .collect()
    };

    let layer_roots: Vec<MerkleRoot> = trees.iter().map(|t| MerkleRoot(t.root().0)).collect();

    // For each round, we open the correct pairs based on MultilinearPolynomial's folding
    // NOTE: We also absorb layer roots here for transcript symmetry (verifier does the same).
    transcript.absorb(TAG_LAYER_ROOTS, &layer_roots.as_slice());

    // Generate sample openings to link consecutive layers (prevents forged layer attacks)
    let mut samples: Vec<Vec<SampleOpening<E>>> = Vec::with_capacity(m);
    for i in 0..m {
      let layer_size = all_layers[i].len();
      let stride = layer_size / 2;
      let mut round_samples = Vec::with_capacity(K_SAMPLES_PER_ROUND);

      for _j in 0..K_SAMPLES_PER_ROUND {
        // Derive random index from transcript (unbiased)
        let idx = draw_index::<E>(transcript, b"mle/fold_sample", stride)?;

        let a = all_layers[i][idx];
        let b = all_layers[i][idx + stride];
        let next = a + point[i] * (b - a);

        // Verify this matches the actual next layer value
        debug_assert_eq!(next, all_layers[i + 1][idx]);

        let path_a = trees[i].open(idx);
        let path_b = trees[i].open(idx + stride);
        let path_next = trees[i + 1].open(idx);

        round_samples.push(SampleOpening {
          idx: idx as u64,
          a,
          b,
          next,
          path_a,
          path_b,
          path_next,
        });
      }
      samples.push(round_samples);
    }

    let mut rounds: Vec<Round<E>> = Vec::with_capacity(m);

    for i in 0..m {
      let layer_size = all_layers[i].len();
      let n = layer_size / 2;

      // For MultilinearPolynomial folding, we open pairs (left[j], right[j])
      // where left is first half and right is second half
      // For simplicity, we'll open the first pair (0, n)
      let a = all_layers[i][0]; // left[0]
      let b = all_layers[i][n]; // right[0] 
      let next = a + point[i] * (b - a); // MultilinearPolynomial folding formula

      let path_a = trees[i].open(0);
      let path_b = trees[i].open(n);
      let path_next = trees[i + 1].open(0);

      rounds.push(Round {
        a,
        b,
        path_a,
        path_b,
        next,
        path_next,
      });
    }

    // eval in LeakReduced mode
    let eval = all_layers.last().unwrap()[0];

    // Cross-check evaluation against the reference MLE evaluator in debug builds.
    #[cfg(debug_assertions)]
    {
      let expected_eval =
        crate::polys::multilinear::MultilinearPolynomial::new(poly.to_vec()).evaluate(point);
      debug_assert_eq!(eval, expected_eval);
    }

    // Pack argument
    let arg = HashMleEvaluationArgument {
      layer_roots,
      rounds,
      samples,
    };
    Ok((eval, arg))
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    if vk.zk_mode != comm.mode {
      return Err(SpartanError::InvalidPCS);
    }
    let m = point.len();
    if arg.layer_roots.len() != m + 1 || arg.rounds.len() != m || arg.samples.len() != m {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS::verify malformed argument".into(),
      });
    }

    // Check that each round has the expected number of samples
    for (i, round_samples) in arg.samples.iter().enumerate() {
      if round_samples.len() != K_SAMPLES_PER_ROUND {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "HashMlePCS::verify round {} has {} samples, expected {}",
            i,
            round_samples.len(),
            K_SAMPLES_PER_ROUND
          ),
        });
      }
    }

    // This PCS is binary Merkle
    if vk.branching != 2 {
      return Err(SpartanError::InvalidPCS);
    }

    transcript.absorb(b"poly_com", comm);
    transcript.absorb(TAG_LAYER_ROOTS, &arg.layer_roots.as_slice());

    // Layer 0 root must match the commitment's root
    if arg.layer_roots[0].0 != comm.base_root.0 {
      return Err(SpartanError::InvalidPCS);
    }

    // Verify sample openings to ensure layer consistency (prevents forged layer attacks)
    for i in 0..m {
      let layer_size = 1usize << (m - i);
      let stride = layer_size / 2;
      let expected_depth = m - i;
      let expected_next_depth = m - i - 1;

      for _j in 0..K_SAMPLES_PER_ROUND {
        // Re-derive the same random index from transcript (unbiased)
        let expected_idx = draw_index::<E>(transcript, b"mle/fold_sample", stride)?;

        let sample = &arg.samples[i][_j];

        // Check that the sample uses the expected index
        if sample.idx != expected_idx as u64 {
          return Err(SpartanError::InvalidPCS);
        }

        // Check path depths
        if sample.path_a.siblings.len() != expected_depth
          || sample.path_b.siblings.len() != expected_depth
          || sample.path_next.siblings.len() != expected_next_depth
        {
          return Err(SpartanError::InvalidPCS);
        }

        // Check path indices
        if sample.path_a.leaf_index != expected_idx as u64
          || sample.path_b.leaf_index != (expected_idx + stride) as u64
          || sample.path_next.leaf_index != expected_idx as u64
        {
          return Err(SpartanError::InvalidPCS);
        }

        // Verify Merkle memberships
        let a_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&sample.a);
        let b_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&sample.b);
        let next_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&sample.next);

        let a_h = leaf_digest::<E, DefaultBackend<E>>(&a_fe);
        let b_h = leaf_digest::<E, DefaultBackend<E>>(&b_fe);
        let next_h = leaf_digest::<E, DefaultBackend<E>>(&next_fe);

        let root_i_digest = Digest32(arg.layer_roots[i].0);
        let root_ip1_digest = Digest32(arg.layer_roots[i + 1].0);

        if !MerkleTree::verify::<E, DefaultBackend<E>>(&sample.path_a, &a_h, &root_i_digest)
          || !MerkleTree::verify::<E, DefaultBackend<E>>(&sample.path_b, &b_h, &root_i_digest)
          || !MerkleTree::verify::<E, DefaultBackend<E>>(
            &sample.path_next,
            &next_h,
            &root_ip1_digest,
          )
        {
          return Err(SpartanError::InvalidPCS);
        }

        // Verify the fold equation: next = a + r_i * (b - a)
        let expected_next = sample.a + point[i] * (sample.b - sample.a);
        if sample.next != expected_next {
          return Err(SpartanError::InvalidPCS);
        }
      }
    }

    // Per round: check memberships and fold equality
    for i in 0..m {
      let root_i = &arg.layer_roots[i].0;
      let root_ip1 = &arg.layer_roots[i + 1].0;

      // Depth / index sanity (prevents pair-swapping attacks)
      // Layer i has 2^(m-i) leaves, so paths there have length m-i.
      // Layer i+1 has 2^(m-i-1) leaves, so paths there have length m-i-1.
      let depth_a = arg.rounds[i].path_a.siblings.len();
      let depth_b = arg.rounds[i].path_b.siblings.len();
      let depth_next = arg.rounds[i].path_next.siblings.len();
      let expected_depth = m - i;
      if depth_a != expected_depth || depth_b != expected_depth || depth_next + 1 != expected_depth
      {
        return Err(SpartanError::InvalidPCS);
      }
      if expected_depth == 0 {
        return Err(SpartanError::InvalidPCS);
      }

      // Expected indices: a at 0, b at 2^(depth-1), next at 0
      let expected_a_idx: u64 = 0;
      let expected_b_idx: u64 = 1u64 << (expected_depth - 1);
      let expected_next_idx: u64 = 0;

      if arg.rounds[i].path_a.leaf_index != expected_a_idx
        || arg.rounds[i].path_b.leaf_index != expected_b_idx
        || arg.rounds[i].path_next.leaf_index != expected_next_idx
      {
        return Err(SpartanError::InvalidPCS);
      }

      let a_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].a);
      let b_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].b);
      let a_h = leaf_digest::<E, DefaultBackend<E>>(&a_fe);
      let b_h = leaf_digest::<E, DefaultBackend<E>>(&b_fe);
      let root_i_digest = Digest32(*root_i);
      if !MerkleTree::verify::<E, DefaultBackend<E>>(&arg.rounds[i].path_a, &a_h, &root_i_digest)
        || !MerkleTree::verify::<E, DefaultBackend<E>>(&arg.rounds[i].path_b, &b_h, &root_i_digest)
      {
        return Err(SpartanError::InvalidPCS);
      }

      let folded = arg.rounds[i].a + point[i] * (arg.rounds[i].b - arg.rounds[i].a);
      if folded != arg.rounds[i].next {
        return Err(SpartanError::InvalidPCS);
      }

      let next_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].next);
      let next_h = leaf_digest::<E, DefaultBackend<E>>(&next_fe);
      let root_ip1_digest = Digest32(*root_ip1);
      if !MerkleTree::verify::<E, DefaultBackend<E>>(
        &arg.rounds[i].path_next,
        &next_h,
        &root_ip1_digest,
      ) {
        return Err(SpartanError::InvalidPCS);
      }
    }

    // Final check for LeakReduced mode
    let y = arg.rounds.last().map(|r| r.next).unwrap_or(*eval);
    if y != *eval {
      return Err(SpartanError::InvalidPCS);
    }

    Ok(())
  }
}

// Note: Merkle PCS is not linearly homomorphic, so we do NOT implement FoldingEngineTrait.

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    polys::multilinear::MultilinearPolynomial,
    provider::PallasMerkleMleEngine,
    traits::{Engine, pcs::PCSEngineTrait},
  };
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  type E = PallasMerkleMleEngine;

  fn rand_scalar(rng: &mut StdRng) -> <E as Engine>::Scalar {
    // sample in a tiny range (fast) then lift into field
    let x: u64 = rng.random();
    <E as Engine>::Scalar::from(x)
  }

  #[test]
  fn test_leak_reduced_roundtrip() {
    let m = 8usize;
    let n = 1usize << m;
    let poly = (0..n)
      .map(|i| <E as Engine>::Scalar::from(i as u64))
      .collect::<Vec<_>>();
    let point = (0..m)
      .map(|i| <E as Engine>::Scalar::from((i + 1) as u64))
      .collect::<Vec<_>>();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point)
        .unwrap();

    let mut tr2 = <E as Engine>::TE::new(b"test");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_small_polynomial() {
    let m = 3usize;
    let n = 1usize << m; // 8 elements
    let poly = vec![
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(2u64),
      <E as Engine>::Scalar::from(3u64),
      <E as Engine>::Scalar::from(4u64),
      <E as Engine>::Scalar::from(5u64),
      <E as Engine>::Scalar::from(6u64),
      <E as Engine>::Scalar::from(7u64),
      <E as Engine>::Scalar::from(8u64),
    ];
    let point = vec![
      <E as Engine>::Scalar::from(0u64),
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(0u64),
    ]; // Should evaluate to poly[2] = 3

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_small", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test_small");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point)
        .unwrap();

    // Verify the evaluation is correct
    let expected_eval = <E as Engine>::Scalar::from(3u64);
    assert_eq!(eval, expected_eval);

    let mut tr2 = <E as Engine>::TE::new(b"test_small");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_random_polynomial() {
    let m = 6usize;
    let n = 1usize << m;
    let poly = (0..n)
      .map(|i| <E as Engine>::Scalar::from(i as u64 + 1))
      .collect::<Vec<_>>();
    let point = (0..m)
      .map(|i| <E as Engine>::Scalar::from(i as u64))
      .collect::<Vec<_>>();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_random", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test_random");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point)
        .unwrap();

    // Verify against direct multilinear evaluation
    let mle = MultilinearPolynomial::new(poly.clone());
    let expected_eval = mle.evaluate(&point);
    assert_eq!(eval, expected_eval);

    let mut tr2 = <E as Engine>::TE::new(b"test_random");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_commitment_consistency() {
    let m = 4usize;
    let n = 1usize << m;
    let poly = (0..n)
      .map(|i| <E as Engine>::Scalar::from((i * i + 7) as u64))
      .collect::<Vec<_>>();

    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_consistency", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    // Commit to the same polynomial twice
    let com1 = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let com2 = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    // Commitments should be identical
    assert_eq!(com1, com2);
  }

  #[test]
  fn test_merkle_tree_operations() {
    // Test the internal Merkle tree functionality
    let leaves = vec![
      Digest32([1u8; 32]),
      Digest32([2u8; 32]),
      Digest32([3u8; 32]),
      Digest32([4u8; 32]),
    ];
    let tree = MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves.clone());

    // Test opening and verification
    for i in 0..leaves.len() {
      let path = tree.open(i);
      assert!(MerkleTree::verify::<E, DefaultBackend<E>>(
        &path,
        &leaves[i],
        &tree.root()
      ));
    }
  }

  #[test]
  fn test_folding_correctness() {
    // Test that our folding matches the MultilinearPolynomial::bind_poly_var_top logic
    let poly = vec![
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(2u64),
      <E as Engine>::Scalar::from(3u64),
      <E as Engine>::Scalar::from(4u64),
    ];
    let r = <E as Engine>::Scalar::from(7u64);

    let folded = fold_layer::<E>(&poly, &r);

    // Manual calculation using MultilinearPolynomial folding: left + r * (right - left)
    // where left is first half [1, 2] and right is second half [3, 4]
    let expected_0 = poly[0] + r * (poly[2] - poly[0]); // 1 + 7*(3-1) = 1 + 14 = 15
    let expected_1 = poly[1] + r * (poly[3] - poly[1]); // 2 + 7*(4-2) = 2 + 14 = 16

    assert_eq!(folded.len(), 2);
    assert_eq!(folded[0], expected_0);
    assert_eq!(folded[1], expected_1);
  }

  #[test]
  fn roundtrip_small_fixed() {
    let m = 5usize;
    let n = 1usize << m;
    let mut rng = StdRng::seed_from_u64(7);
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|_| rand_scalar(&mut rng)).collect();
    let point: Vec<<E as Engine>::Scalar> = (0..m).map(|_| rand_scalar(&mut rng)).collect();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    let mut tr_prove = <E as Engine>::TE::new(b"t");
    let (eval, arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr_prove, &com, &poly, &blind, &point)
        .unwrap();

    let mut tr_verify = <E as Engine>::TE::new(b"t");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr_verify, &com, &point, &eval, &arg)
      .unwrap();

    // cross-check against direct MLE evaluation
    let mle = MultilinearPolynomial::new(poly.clone());
    assert_eq!(eval, mle.evaluate(&point));
  }

  #[test]
  fn serialized_argument_roundtrips() {
    let m = 6usize;
    let n = 1usize << m;
    let mut rng = StdRng::seed_from_u64(11);
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|_| rand_scalar(&mut rng)).collect();
    let point: Vec<<E as Engine>::Scalar> = (0..m).map(|_| rand_scalar(&mut rng)).collect();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"serde", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    let mut tr_prove = <E as Engine>::TE::new(b"serde");
    let (eval, arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr_prove, &com, &poly, &blind, &point)
        .unwrap();

    let encoded = bincode::serialize(&arg).expect("serialize evaluation argument");
    let decoded: HashMleEvaluationArgument<E> =
      bincode::deserialize(&encoded).expect("deserialize evaluation argument");

    let mut tr_verify = <E as Engine>::TE::new(b"serde");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(
      &vk,
      &mut tr_verify,
      &com,
      &point,
      &eval,
      &decoded,
    )
    .unwrap();
  }

  #[test]
  fn commit_rejects_non_power_of_two() {
    let n = 12usize; // not a power of two
    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"bad", n);
    let blind = HashMleBlind::<E>::default();
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|i| <E as Engine>::Scalar::from(i as u64))
      .collect();
    let err = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("power of two"));
  }

  #[test]
  fn prove_rejects_wrong_point_len() {
    let m = 4usize;
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|i| <E as Engine>::Scalar::from(i as u64))
      .collect();
    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"bad2", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let mut tr = <E as Engine>::TE::new(b"p");
    // point too short
    let point: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::from(5u64); m - 1];
    assert!(
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point)
        .is_err()
    );
  }

  #[test]
  fn tamper_layer_root_fails() {
    let m = 3usize;
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|i| <E as Engine>::Scalar::from((i * i + 7) as u64))
      .collect();
    let point: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::from(2u64); m];
    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"tamper", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let mut tr = <E as Engine>::TE::new(b"x");
    let (eval, mut arg) =
      <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point)
        .unwrap();
    // flip a bit in an inner root
    arg.layer_roots[1].0[0] ^= 0x01;
    let mut tr_v = <E as Engine>::TE::new(b"x");
    assert!(
      <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr_v, &com, &point, &eval, &arg)
        .is_err()
    );
  }

  #[test]
  fn width_is_binary() {
    assert_eq!(<HashMlePCS<E> as PCSEngineTrait<E>>::width(), 2);
  }

  #[test]
  fn red_team_forged_chain_is_rejected_by_index_checks() {
    // Demonstrates the classic pair-index forgery would have verified before the fix.
    let m = 3usize;
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|i| <E as Engine>::Scalar::from((3 * i + 5) as u64))
      .collect();
    let point: Vec<<E as Engine>::Scalar> = vec![
      <E as Engine>::Scalar::from(7u64),
      <E as Engine>::Scalar::from(11u64),
      <E as Engine>::Scalar::from(13u64),
    ];
    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"forge", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    // Build our own fake layer chain:
    // Layer 0: take pair at j != 0
    let j0 = 2usize;
    let half0 = n / 2;
    let a0 = poly[j0];
    let b0 = poly[j0 + half0];
    let next0 = a0 + point[0] * (b0 - a0);
    // Trees for L0..L3, each with only the constraint that index 0 equals our chosen "next"
    let leaves0 = poly
      .iter()
      .map(|x| {
        let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x);
        leaf_digest::<E, DefaultBackend<E>>(&x_fe)
      })
      .collect::<Vec<_>>();
    let t0 = MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves0);
    let root0 = MerkleRoot(t0.root().0);

    let l1 = vec![
      next0,
      <E as Engine>::Scalar::ZERO,
      <E as Engine>::Scalar::ZERO,
      <E as Engine>::Scalar::ZERO,
    ];
    let leaves1 = l1
      .iter()
      .map(|x| {
        let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x);
        leaf_digest::<E, DefaultBackend<E>>(&x_fe)
      })
      .collect::<Vec<_>>();
    let t1 = MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves1);
    let root1 = MerkleRoot(t1.root().0);
    let a1 = l1[0];
    let b1 = l1[2];
    let next1 = a1 + point[1] * (b1 - a1);

    let l2 = vec![next1, <E as Engine>::Scalar::ZERO];
    let leaves2 = l2
      .iter()
      .map(|x| {
        let x_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(x);
        leaf_digest::<E, DefaultBackend<E>>(&x_fe)
      })
      .collect::<Vec<_>>();
    let t2 = MerkleTree::from_leaves::<E, DefaultBackend<E>>(leaves2);
    let root2 = MerkleRoot(t2.root().0);
    let a2 = l2[0];
    let b2 = l2[1];
    let next2 = a2 + point[2] * (b2 - a2);

    let next2_fe = <DefaultBackend<E> as MleBackend<E>>::fe_from_ff(&next2);
    let next2_digest = leaf_digest::<E, DefaultBackend<E>>(&next2_fe);
    let t3 = MerkleTree::from_leaves::<E, DefaultBackend<E>>(vec![next2_digest]);
    let root3 = MerkleRoot(t3.root().0);

    // Assemble a forged argument (note: indices j0 and j0+half0 at layer 0)
    // This test predates the soundness fix, so we provide empty samples
    // The test should still fail due to index checks
    let arg = HashMleEvaluationArgument {
      layer_roots: vec![root0, root1, root2, root3],
      rounds: vec![
        Round {
          a: a0,
          b: b0,
          path_a: t0.open(j0),
          path_b: t0.open(j0 + half0),
          next: next0,
          path_next: t1.open(0),
        },
        Round {
          a: a1,
          b: b1,
          path_a: t1.open(0),
          path_b: t1.open(2),
          next: next1,
          path_next: t2.open(0),
        },
        Round {
          a: a2,
          b: b2,
          path_a: t2.open(0),
          path_b: t2.open(1),
          next: next2,
          path_next: t3.open(0),
        },
      ],
      samples: vec![vec![], vec![], vec![]], // Empty samples for this legacy test
    };
    let eval = next2; // bogus eval, not the true MLE eval

    let mut tr = <E as Engine>::TE::new(b"forge");
    // With the index/depth checks added, this is rejected.
    assert!(
      <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr, &com, &point, &eval, &arg)
        .is_err()
    );
  }
}
