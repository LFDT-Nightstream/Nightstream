use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
  DefaultBackend, Digest32, Engine, HashMleEvaluationArgument, K_SAMPLES_PER_ROUND, MerklePath,
  MerkleRoot, MleBackend, Round, SampleOpening, leaf_digest, node_digest,
};
use crate::provider::GoldilocksMerkleMleEngine;
#[cfg(feature = "p3_backend")]
use crate::provider::GoldilocksP3MerkleMleEngine;
#[cfg(feature = "p3_backend")]
use crate::provider::pcs::hash_mle_backend::BackendP3Poseidon2Goldi as P3Backend;
#[cfg(not(target_arch = "wasm32"))]
use crate::provider::{
  P256MerkleMleEngine, PallasMerkleMleEngine, T256MerkleMleEngine, VestaMerkleMleEngine,
};

pub(crate) trait CompactHashMleDigest: Engine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32;
  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32;
}

impl CompactHashMleDigest for GoldilocksMerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <DefaultBackend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, DefaultBackend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, DefaultBackend<Self>>(left, right)
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl CompactHashMleDigest for PallasMerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <DefaultBackend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, DefaultBackend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, DefaultBackend<Self>>(left, right)
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl CompactHashMleDigest for VestaMerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <DefaultBackend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, DefaultBackend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, DefaultBackend<Self>>(left, right)
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl CompactHashMleDigest for P256MerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <DefaultBackend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, DefaultBackend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, DefaultBackend<Self>>(left, right)
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl CompactHashMleDigest for T256MerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <DefaultBackend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, DefaultBackend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, DefaultBackend<Self>>(left, right)
  }
}

#[cfg(feature = "p3_backend")]
impl CompactHashMleDigest for GoldilocksP3MerkleMleEngine {
  fn compact_leaf_digest(value: &Self::Scalar) -> Digest32 {
    let fe = <P3Backend<Self> as MleBackend<Self>>::fe_from_ff(value);
    leaf_digest::<Self, P3Backend<Self>>(&fe)
  }

  fn compact_node_digest(left: &Digest32, right: &Digest32) -> Digest32 {
    node_digest::<Self, P3Backend<Self>>(left, right)
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct CompactHashMleEvaluationArgument<E: Engine> {
  pub layer_roots: Vec<MerkleRoot>,
  pub layers: Vec<CompactEvaluationLayer<E>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct CompactEvaluationLayer<E: Engine> {
  pub round_a: E::Scalar,
  pub round_b: E::Scalar,
  pub round_next: E::Scalar,
  pub sample_indices: Vec<u64>,
  pub sample_a: Vec<E::Scalar>,
  pub sample_b: Vec<E::Scalar>,
  pub sample_next: Vec<E::Scalar>,
  pub current_proof: CompactMerkleMultiProof,
  pub next_proof: CompactMerkleMultiProof,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CompactMerkleMultiProof {
  pub proof_digests: Vec<[u8; 32]>,
}

pub(crate) fn compact_eval_arg<E: CompactHashMleDigest>(
  arg: &HashMleEvaluationArgument<E>,
) -> CompactHashMleEvaluationArgument<E> {
  let layers = arg
    .rounds
    .iter()
    .zip(arg.samples.iter())
    .map(|(round, samples)| compact_layer(round, samples))
    .collect();
  CompactHashMleEvaluationArgument {
    layer_roots: arg.layer_roots.clone(),
    layers,
  }
}

pub(crate) fn expand_eval_arg<E: CompactHashMleDigest>(
  compact: CompactHashMleEvaluationArgument<E>,
) -> Result<HashMleEvaluationArgument<E>, String> {
  if compact.layer_roots.len() != compact.layers.len() + 1 {
    return Err("compact Hash-MLE argument has inconsistent layer count".into());
  }
  let layer_count = compact.layers.len();
  let mut rounds = Vec::with_capacity(layer_count);
  let mut samples = Vec::with_capacity(layer_count);
  for (layer_idx, layer) in compact.layers.into_iter().enumerate() {
    let current_depth = layer_count - layer_idx;
    let next_depth = current_depth.saturating_sub(1);
    let stride = 1u64 << next_depth;
    let current_positions = expected_current_positions(&layer.sample_indices, stride);
    let next_positions = expected_next_positions(&layer.sample_indices);
    let current_leaf_digests = current_layer_leaf_digests::<E>(&layer, stride)?;
    let next_leaf_digests = next_layer_leaf_digests::<E>(&layer)?;
    let current_paths = reconstruct_paths::<E>(
      &layer.current_proof,
      &current_positions,
      &current_leaf_digests,
      current_depth,
    )?;
    let next_paths = reconstruct_paths::<E>(
      &layer.next_proof,
      &next_positions,
      &next_leaf_digests,
      next_depth,
    )?;
    if layer.sample_indices.len() != K_SAMPLES_PER_ROUND
      || layer.sample_a.len() != K_SAMPLES_PER_ROUND
      || layer.sample_b.len() != K_SAMPLES_PER_ROUND
      || layer.sample_next.len() != K_SAMPLES_PER_ROUND
    {
      return Err(format!(
        "compact Hash-MLE layer {} does not carry the expected sample count",
        layer_idx
      ));
    }

    let round = Round {
      a: layer.round_a,
      b: layer.round_b,
      path_a: lookup_path(&current_paths, 0, "round.a")?,
      path_b: lookup_path(&current_paths, stride, "round.b")?,
      next: layer.round_next,
      path_next: lookup_path(&next_paths, 0, "round.next")?,
    };
    let round_samples = layer
      .sample_indices
      .into_iter()
      .zip(layer.sample_a.into_iter())
      .zip(layer.sample_b.into_iter())
      .zip(layer.sample_next.into_iter())
      .map(|(((idx, a), b), next)| {
        Ok(SampleOpening {
          idx,
          a,
          b,
          next,
          path_a: lookup_path(&current_paths, idx, "sample.a")?,
          path_b: lookup_path(&current_paths, idx + stride, "sample.b")?,
          path_next: lookup_path(&next_paths, idx, "sample.next")?,
        })
      })
      .collect::<Result<Vec<_>, String>>()?;
    rounds.push(round);
    samples.push(round_samples);
  }

  Ok(HashMleEvaluationArgument {
    layer_roots: compact.layer_roots,
    rounds,
    samples,
  })
}

fn compact_layer<E: Engine>(
  round: &Round<E>,
  samples: &[SampleOpening<E>],
) -> CompactEvaluationLayer<E> {
  let current_depth = round.path_a.siblings.len();
  let stride = 1u64 << current_depth.saturating_sub(1);
  let current_openings = std::iter::once((0u64, &round.path_a))
    .chain(std::iter::once((stride, &round.path_b)))
    .chain(samples.iter().flat_map(|sample| {
      [
        (sample.idx, &sample.path_a),
        (sample.idx + stride, &sample.path_b),
      ]
    }));
  let next_openings = std::iter::once((0u64, &round.path_next))
    .chain(samples.iter().map(|sample| (sample.idx, &sample.path_next)));
  CompactEvaluationLayer {
    round_a: round.a,
    round_b: round.b,
    round_next: round.next,
    sample_indices: samples.iter().map(|sample| sample.idx).collect(),
    sample_a: samples.iter().map(|sample| sample.a).collect(),
    sample_b: samples.iter().map(|sample| sample.b).collect(),
    sample_next: samples.iter().map(|sample| sample.next).collect(),
    current_proof: build_compact_proof(current_openings.collect()),
    next_proof: build_compact_proof(next_openings.collect()),
  }
}

fn build_compact_proof(openings: Vec<(u64, &MerklePath)>) -> CompactMerkleMultiProof {
  let unique_paths = dedup_paths(openings);
  if unique_paths.is_empty() {
    return CompactMerkleMultiProof {
      proof_digests: Vec::new(),
    };
  }
  let depth = unique_paths.values().next().unwrap().siblings.len();
  let positions = unique_paths.keys().copied().collect::<Vec<_>>();
  let mut proof_digests = Vec::new();
  collect_proof_digests(0, depth, &positions, &unique_paths, &mut proof_digests);
  CompactMerkleMultiProof { proof_digests }
}

fn dedup_paths<'a>(openings: Vec<(u64, &'a MerklePath)>) -> BTreeMap<u64, &'a MerklePath> {
  let mut unique = BTreeMap::new();
  for (position, path) in openings {
    match unique.get(&position) {
      Some(existing) => debug_assert_eq!(*existing, path),
      None => {
        unique.insert(position, path);
      }
    }
  }
  unique
}

fn collect_proof_digests(
  start: u64,
  height: usize,
  positions: &[u64],
  paths: &BTreeMap<u64, &MerklePath>,
  out: &mut Vec<[u8; 32]>,
) {
  if positions.is_empty() || height == 0 {
    return;
  }
  let split = start + (1u64 << (height - 1));
  let mid = positions.partition_point(|&position| position < split);
  let (left_positions, right_positions) = positions.split_at(mid);
  match (left_positions.is_empty(), right_positions.is_empty()) {
    (false, false) => {
      collect_proof_digests(start, height - 1, left_positions, paths, out);
      collect_proof_digests(split, height - 1, right_positions, paths, out);
    }
    (false, true) => {
      out.push(paths[&left_positions[0]].siblings[height - 1]);
      collect_proof_digests(start, height - 1, left_positions, paths, out);
    }
    (true, false) => {
      out.push(paths[&right_positions[0]].siblings[height - 1]);
      collect_proof_digests(split, height - 1, right_positions, paths, out);
    }
    (true, true) => {}
  }
}

fn current_layer_leaf_digests<E: CompactHashMleDigest>(
  layer: &CompactEvaluationLayer<E>,
  stride: u64,
) -> Result<BTreeMap<u64, Digest32>, String> {
  let mut digests = BTreeMap::new();
  insert_leaf_digest::<E>(&mut digests, 0, &layer.round_a, "round.a")?;
  insert_leaf_digest::<E>(&mut digests, stride, &layer.round_b, "round.b")?;
  for ((idx, a), b) in layer
    .sample_indices
    .iter()
    .copied()
    .zip(layer.sample_a.iter())
    .zip(layer.sample_b.iter())
  {
    insert_leaf_digest::<E>(&mut digests, idx, a, "sample.a")?;
    insert_leaf_digest::<E>(&mut digests, idx + stride, b, "sample.b")?;
  }
  Ok(digests)
}

fn next_layer_leaf_digests<E: CompactHashMleDigest>(
  layer: &CompactEvaluationLayer<E>,
) -> Result<BTreeMap<u64, Digest32>, String> {
  let mut digests = BTreeMap::new();
  insert_leaf_digest::<E>(&mut digests, 0, &layer.round_next, "round.next")?;
  for (idx, next) in layer
    .sample_indices
    .iter()
    .copied()
    .zip(layer.sample_next.iter())
  {
    insert_leaf_digest::<E>(&mut digests, idx, next, "sample.next")?;
  }
  Ok(digests)
}

fn insert_leaf_digest<E: CompactHashMleDigest>(
  digests: &mut BTreeMap<u64, Digest32>,
  position: u64,
  value: &E::Scalar,
  label: &str,
) -> Result<(), String> {
  let digest = E::compact_leaf_digest(value);
  match digests.get(&position) {
    Some(existing) if existing != &digest => Err(format!(
      "compact Hash-MLE multiproof carries inconsistent values for {label} at position {position}"
    )),
    Some(_) => Ok(()),
    None => {
      digests.insert(position, digest);
      Ok(())
    }
  }
}

fn reconstruct_paths<E: CompactHashMleDigest>(
  proof: &CompactMerkleMultiProof,
  positions: &[u64],
  leaf_digests: &BTreeMap<u64, Digest32>,
  depth: usize,
) -> Result<BTreeMap<u64, MerklePath>, String> {
  for position in positions {
    if !leaf_digests.contains_key(position) {
      return Err(format!(
        "compact Hash-MLE multiproof is missing a leaf digest for position {position}"
      ));
    }
  }
  let mut siblings_by_position = positions
    .iter()
    .copied()
    .map(|position| (position, vec![[0u8; 32]; depth]))
    .collect::<BTreeMap<_, _>>();
  let mut proof_iter = proof.proof_digests.iter();
  reconstruct_subtree::<E>(
    0,
    depth,
    positions,
    leaf_digests,
    &mut proof_iter,
    &mut siblings_by_position,
  )?;
  if proof_iter.next().is_some() {
    return Err("compact Hash-MLE multiproof carries extra proof digests".into());
  }
  Ok(
    siblings_by_position
      .into_iter()
      .map(|(position, siblings)| {
        (
          position,
          MerklePath {
            leaf_index: position,
            siblings,
          },
        )
      })
      .collect(),
  )
}

fn expected_current_positions(sample_indices: &[u64], stride: u64) -> Vec<u64> {
  let mut positions = BTreeSet::from([0, stride]);
  for idx in sample_indices {
    positions.insert(*idx);
    positions.insert(*idx + stride);
  }
  positions.into_iter().collect()
}

fn expected_next_positions(sample_indices: &[u64]) -> Vec<u64> {
  let mut positions = BTreeSet::from([0]);
  for idx in sample_indices {
    positions.insert(*idx);
  }
  positions.into_iter().collect()
}

fn reconstruct_subtree<E: CompactHashMleDigest>(
  start: u64,
  height: usize,
  positions: &[u64],
  leaf_digests: &BTreeMap<u64, Digest32>,
  proof_iter: &mut std::slice::Iter<'_, [u8; 32]>,
  siblings_by_position: &mut BTreeMap<u64, Vec<[u8; 32]>>,
) -> Result<Digest32, String> {
  if positions.is_empty() {
    return Err("compact Hash-MLE multiproof recursion reached an empty subtree".into());
  }
  if height == 0 {
    if positions.len() != 1 || positions[0] != start {
      return Err("compact Hash-MLE multiproof leaf layout mismatch".into());
    }
    return leaf_digests
      .get(&start)
      .copied()
      .ok_or_else(|| format!("missing leaf digest for position {start}"));
  }

  let split = start + (1u64 << (height - 1));
  let mid = positions.partition_point(|&position| position < split);
  let (left_positions, right_positions) = positions.split_at(mid);
  match (left_positions.is_empty(), right_positions.is_empty()) {
    (false, false) => {
      let left = reconstruct_subtree::<E>(
        start,
        height - 1,
        left_positions,
        leaf_digests,
        proof_iter,
        siblings_by_position,
      )?;
      let right = reconstruct_subtree::<E>(
        split,
        height - 1,
        right_positions,
        leaf_digests,
        proof_iter,
        siblings_by_position,
      )?;
      assign_sibling(left_positions, height - 1, right.0, siblings_by_position);
      assign_sibling(right_positions, height - 1, left.0, siblings_by_position);
      Ok(E::compact_node_digest(&left, &right))
    }
    (false, true) => {
      let right = Digest32(
        *proof_iter
          .next()
          .ok_or_else(|| "compact Hash-MLE multiproof ended early".to_string())?,
      );
      let left = reconstruct_subtree::<E>(
        start,
        height - 1,
        left_positions,
        leaf_digests,
        proof_iter,
        siblings_by_position,
      )?;
      assign_sibling(left_positions, height - 1, right.0, siblings_by_position);
      Ok(E::compact_node_digest(&left, &right))
    }
    (true, false) => {
      let left = Digest32(
        *proof_iter
          .next()
          .ok_or_else(|| "compact Hash-MLE multiproof ended early".to_string())?,
      );
      let right = reconstruct_subtree::<E>(
        split,
        height - 1,
        right_positions,
        leaf_digests,
        proof_iter,
        siblings_by_position,
      )?;
      assign_sibling(right_positions, height - 1, left.0, siblings_by_position);
      Ok(E::compact_node_digest(&left, &right))
    }
    (true, true) => Err("compact Hash-MLE multiproof recursion reached an empty split".into()),
  }
}

fn assign_sibling(
  positions: &[u64],
  level: usize,
  sibling: [u8; 32],
  siblings_by_position: &mut BTreeMap<u64, Vec<[u8; 32]>>,
) {
  for position in positions {
    siblings_by_position
      .get_mut(position)
      .expect("position is initialized before multiproof reconstruction")[level] = sibling;
  }
}

fn lookup_path(
  paths: &BTreeMap<u64, MerklePath>,
  position: u64,
  label: &str,
) -> Result<MerklePath, String> {
  paths
    .get(&position)
    .cloned()
    .ok_or_else(|| format!("compact Hash-MLE argument is missing {label} at position {position}"))
}
