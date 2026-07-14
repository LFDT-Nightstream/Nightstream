//! Hash-MLE PCS backed by p3_goldilocks::Goldilocks arithmetic and Poseidon2 hashing.
//! This requires that E::Scalar is also Goldilocks on the ff side (the usual "dummy group" engine).

use super::hash_mle_backend::{BackendP3Poseidon2Goldi as P3B, MleBackend};
use super::merkle_mle_pc::*; // reuse all public structs: HashMleCommitment{Key}, MerklePath, etc.
use crate::{
  errors::SpartanError,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Hash-MLE PCS implementation using p3/Goldilocks + Poseidon2 backend
pub struct HashMlePcsP3<E: Engine> {
  _p: core::marker::PhantomData<E>,
}

impl<E> PCSEngineTrait<E> for HashMlePcsP3<E>
where
  E: Engine<Scalar = crate::provider::goldi::F> + super::merkle_mle_pc::CompactHashMleDigest,
{
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
      _p: core::marker::PhantomData,
    };
    let vk = HashMleVerifierKey {
      branching: 2,
      zk_mode: ZkMode::LeakReduced,
      _p: core::marker::PhantomData,
    };
    (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    HashMleBlind::default()
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v_ff: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    if !v_ff.len().is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS_P3: vector len must be power of two".into(),
      });
    }
    // Convert leaves to p3 field and hash with Poseidon2
    let leaves = if crate::parallel::parallelism_enabled() {
      v_ff
        .par_iter()
        .map(|x| {
          let x_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(x);
          super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&x_fe)
        })
        .collect::<Vec<_>>()
    } else {
      v_ff
        .iter()
        .map(|x| {
          let x_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(x);
          super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&x_fe)
        })
        .collect::<Vec<_>>()
    };

    let tree = super::merkle_mle_pc::MerkleTree::from_leaves::<E, P3B<E>>(leaves);
    let base_root = MerkleRoot(tree.root().0);

    Ok(HashMleCommitment {
      base_root,
      mode: ck.zk_mode,
      _p: core::marker::PhantomData,
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

  fn combine_partial(pcs: &[Self::PartialCommitment]) -> Result<Self::Commitment, SpartanError> {
    if pcs.len() != 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS_P3: combine_partial expects exactly one piece".into(),
      });
    }
    Ok(pcs[0].clone())
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.len() != 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: "HashMlePCS_P3: combine_blinds expects exactly one".into(),
      });
    }
    Ok(blinds[0].clone())
  }

  fn prove(
    _ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly_ff: &[E::Scalar],
    _blind: &Self::Blind,
    point_ff: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    let n = poly_ff.len();
    let m = point_ff.len();
    if n != (1usize << m) {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "HashMlePCS_P3::prove expected {} elements, got {}",
          1usize << m,
          n
        ),
      });
    }

    transcript.absorb(b"poly_com", comm);

    // Convert to p3 field for arithmetic
    let poly = poly_ff.iter().map(P3B::<E>::fe_from_ff).collect::<Vec<_>>();
    let point = point_ff
      .iter()
      .map(P3B::<E>::fe_from_ff)
      .collect::<Vec<_>>();

    // Fold layers in p3 field
    let mut layers: Vec<Vec<<P3B<E> as MleBackend<E>>::FE>> = Vec::with_capacity(m + 1);
    layers.push(poly.clone());
    let mut cur = poly;
    for &r in &point {
      let n2 = cur.len() / 2;
      let mut next = Vec::with_capacity(n2);
      if crate::parallel::parallelism_enabled() {
        next.par_extend((0..n2).into_par_iter().map(|i| {
          let a = cur[i];
          let b = cur[i + n2];
          // (1-r)*a + r*b  ==  a + r*(b-a)
          P3B::<E>::add(a, P3B::<E>::mul(r, P3B::<E>::sub(b, a)))
        }));
      } else {
        next.extend((0..n2).map(|i| {
          let a = cur[i];
          let b = cur[i + n2];
          P3B::<E>::add(a, P3B::<E>::mul(r, P3B::<E>::sub(b, a)))
        }));
      }
      layers.push(next.clone());
      cur = next;
    }
    debug_assert_eq!(layers.last().unwrap().len(), 1);

    // Build Merkle trees per layer with Poseidon2 hashes
    use super::merkle_mle_pc::{HashMleEvaluationArgument, Round, TAG_LAYER_ROOTS};
    let trees: Vec<_> = layers
      .iter()
      .map(|lvl| {
        let leaves = if crate::parallel::parallelism_enabled() {
          lvl
            .par_iter()
            .map(|x| super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(x))
            .collect::<Vec<_>>()
        } else {
          lvl
            .iter()
            .map(|x| super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(x))
            .collect::<Vec<_>>()
        };
        super::merkle_mle_pc::MerkleTree::from_leaves::<E, P3B<E>>(leaves)
      })
      .collect();

    let layer_roots: Vec<_> = trees.iter().map(|t| MerkleRoot(t.root().0)).collect();
    transcript.absorb(TAG_LAYER_ROOTS, &layer_roots.as_slice());

    // Generate sample openings to link consecutive layers (prevents forged layer attacks)
    use super::merkle_mle_pc::{K_SAMPLES_PER_ROUND, SampleOpening};
    let mut samples: Vec<Vec<SampleOpening<E>>> = Vec::with_capacity(m);
    for i in 0..m {
      let layer_size = layers[i].len();
      let stride = layer_size / 2;
      let mut round_samples = Vec::with_capacity(K_SAMPLES_PER_ROUND);

      for _j in 0..K_SAMPLES_PER_ROUND {
        // Derive random index from transcript (unbiased)
        let idx = super::merkle_mle_pc::draw_index::<E>(transcript, b"mle/fold_sample", stride)?;

        let a = layers[i][idx];
        let b = layers[i][idx + stride];
        let next = P3B::<E>::add(a, P3B::<E>::mul(point[i], P3B::<E>::sub(b, a)));

        // Verify this matches the actual next layer value
        debug_assert_eq!(next, layers[i + 1][idx]);

        let path_a = trees[i].open(idx);
        let path_b = trees[i].open(idx + stride);
        let path_next = trees[i + 1].open(idx);

        round_samples.push(SampleOpening {
          idx: idx as u64,
          a: P3B::<E>::fe_to_ff(&a),
          b: P3B::<E>::fe_to_ff(&b),
          next: P3B::<E>::fe_to_ff(&next),
          path_a,
          path_b,
          path_next,
        });
      }
      samples.push(round_samples);
    }

    // Open the canonical pair per round (indices 0 and n), and the next @ 0
    let mut rounds = Vec::with_capacity(m);
    for i in 0..m {
      let layer = &layers[i];
      let n2 = layer.len() / 2;

      let a = layer[0];
      let b = layer[n2];
      let next = P3B::<E>::add(a, P3B::<E>::mul(point[i], P3B::<E>::sub(b, a)));

      let path_a = trees[i].open(0);
      let path_b = trees[i].open(n2);
      let path_next = trees[i + 1].open(0);

      // convert a, b, next back to ff for serialization in the existing Round<E>
      rounds.push(Round::<E> {
        a: P3B::<E>::fe_to_ff(&a),
        b: P3B::<E>::fe_to_ff(&b),
        next: P3B::<E>::fe_to_ff(&next),
        path_a,
        path_b,
        path_next,
      });
    }

    // Final eval is the single word at the top
    let eval = P3B::<E>::fe_to_ff(&layers.last().unwrap()[0]);

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
        reason: "HashMlePCS_P3::verify malformed argument".into(),
      });
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

    // Per round: check memberships and fold equality using p3 backend
    for i in 0..m {
      let root_i = &arg.layer_roots[i].0;
      let root_ip1 = &arg.layer_roots[i + 1].0;

      // Depth / index sanity (prevents pair-swapping attacks)
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

      // Convert to p3 field and hash with Poseidon2
      let a_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].a);
      let b_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].b);
      let a_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&a_fe);
      let b_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&b_fe);
      let root_i_digest = super::hash_mle_backend::Digest32(*root_i);
      if !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(
        &arg.rounds[i].path_a,
        &a_h,
        &root_i_digest,
      ) || !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(
        &arg.rounds[i].path_b,
        &b_h,
        &root_i_digest,
      ) {
        return Err(SpartanError::InvalidPCS);
      }

      let folded = arg.rounds[i].a + point[i] * (arg.rounds[i].b - arg.rounds[i].a);
      if folded != arg.rounds[i].next {
        return Err(SpartanError::InvalidPCS);
      }

      let next_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&arg.rounds[i].next);
      let next_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&next_fe);
      let root_ip1_digest = super::hash_mle_backend::Digest32(*root_ip1);
      if !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(
        &arg.rounds[i].path_next,
        &next_h,
        &root_ip1_digest,
      ) {
        return Err(SpartanError::InvalidPCS);
      }
    }

    // --- CRITICAL: verify random sample openings to link layers i -> i+1 ---
    use super::merkle_mle_pc::{K_SAMPLES_PER_ROUND, draw_index};
    for i in 0..m {
      let layer_size = 1usize << (m - i);
      let stride = layer_size / 2;
      let expected_depth = m - i;
      let expected_next_depth = m - i - 1;

      if arg.samples[i].len() != K_SAMPLES_PER_ROUND {
        return Err(SpartanError::InvalidPCS);
      }

      for _j in 0..K_SAMPLES_PER_ROUND {
        // re-derive index from transcript
        let expected_idx = draw_index::<E>(transcript, b"mle/fold_sample", stride)?;
        let s = &arg.samples[i][_j];

        // index / depth sanity
        if s.idx != expected_idx as u64
          || s.path_a.leaf_index != expected_idx as u64
          || s.path_b.leaf_index != (expected_idx + stride) as u64
          || s.path_next.leaf_index != expected_idx as u64
          || s.path_a.siblings.len() != expected_depth
          || s.path_b.siblings.len() != expected_depth
          || s.path_next.siblings.len() != expected_next_depth
        {
          return Err(SpartanError::InvalidPCS);
        }

        // Poseidon2 hashing in p3 field
        let a_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&s.a);
        let b_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&s.b);
        let next_fe = <P3B<E> as MleBackend<E>>::fe_from_ff(&s.next);
        let a_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&a_fe);
        let b_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&b_fe);
        let next_h = super::merkle_mle_pc::leaf_digest::<E, P3B<E>>(&next_fe);
        let root_i = super::hash_mle_backend::Digest32(arg.layer_roots[i].0);
        let root_ip1 = super::hash_mle_backend::Digest32(arg.layer_roots[i + 1].0);

        if !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(&s.path_a, &a_h, &root_i)
          || !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(&s.path_b, &b_h, &root_i)
          || !super::merkle_mle_pc::MerkleTree::verify::<E, P3B<E>>(
            &s.path_next,
            &next_h,
            &root_ip1,
          )
        {
          return Err(SpartanError::InvalidPCS);
        }

        // fold equation (in ff / E::Scalar)
        let expected_next = s.a + point[i] * (s.b - s.a);
        if s.next != expected_next {
          return Err(SpartanError::InvalidPCS);
        }
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::GoldilocksP3MerkleMleEngine as E;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  #[test]
  fn width_is_binary_p3() {
    assert_eq!(<HashMlePcsP3<E> as PCSEngineTrait<E>>::width(), 2);
  }

  #[test]
  fn serialized_argument_roundtrips_p3() {
    let m = 6usize;
    let n = 1usize << m;
    let mut rng = StdRng::seed_from_u64(19);
    let poly: Vec<<E as Engine>::Scalar> = (0..n)
      .map(|_| <E as Engine>::Scalar::from(rng.random::<u64>()))
      .collect();
    let point: Vec<<E as Engine>::Scalar> = (0..m)
      .map(|_| <E as Engine>::Scalar::from(rng.random::<u64>()))
      .collect();

    let (ck, vk) = <HashMlePcsP3<E> as PCSEngineTrait<E>>::setup(b"serde_p3", n);
    let blind = <HashMlePcsP3<E> as PCSEngineTrait<E>>::blind(&ck, n);
    let com = <HashMlePcsP3<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    let mut tr_prove = <E as Engine>::TE::new(b"serde_p3");
    let (eval, arg) = <HashMlePcsP3<E> as PCSEngineTrait<E>>::prove(
      &ck,
      &mut tr_prove,
      &com,
      &poly,
      &blind,
      &point,
    )
    .unwrap();

    let encoded = bincode::serialize(&arg).expect("serialize p3 evaluation argument");
    let decoded: HashMleEvaluationArgument<E> =
      bincode::deserialize(&encoded).expect("deserialize p3 evaluation argument");

    let mut tr_verify = <E as Engine>::TE::new(b"serde_p3");
    <HashMlePcsP3<E> as PCSEngineTrait<E>>::verify(
      &vk,
      &mut tr_verify,
      &com,
      &point,
      &eval,
      &decoded,
    )
    .unwrap();
  }
}
