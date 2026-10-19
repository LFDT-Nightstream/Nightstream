//! The Merkle commitment of the WHIR provers.
//!
//! Owns: the binary Merkle tree over committed rows, built on the prover's
//! device when it has one and otherwise in parallel on the CPU, and its
//! openings in Plonky3 0.8's formats. Verification is Plonky3's own tree
//! (`hash::Plain`). Commitments and openings equal Plonky3's (test
//! `tests/internal/merkle.rs`), so no proof depends on the device.
//!
//! A committed tree keeps its rows and digest layers in an unlinked scratch
//! file, not in memory: an opening reads only the rows and siblings it
//! names, so a commitment costs disk, not RAM, until its last opening.

use std::fs::File;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;
use p3_commit_v08::{BatchOpening, BatchOpeningRef, Mmcs as Scheme};
use p3_field_v08::{PackedValue, PrimeCharacteristicRing, PrimeField64};
use p3_matrix_v08::{Dimensions, Matrix};
use p3_merkle_tree_v08::{MerkleCap, PrunedMerklePaths};
use p3_symmetric_v08::{CryptographicHasher, PseudoCompressionFunction};
use rayon::prelude::*;

use crate::device::Accelerator;
use crate::field::Gl;
use crate::hash::{permutation, plain, Compress, Packed, Plain, Sponge};
use crate::{scratch, Error};

type Digest = [Gl; DIGEST_LEN];

/// The Merkle commitment scheme of the WHIR provers; trees build on the
/// accelerator's device when it has one.
#[derive(Clone)]
pub(crate) struct Mmcs {
    plain: Plain,
    accelerator: Accelerator,
    /// The first scratch-file failure, shared with every clone of one prover.
    failure: Arc<OnceLock<String>>,
}

impl Mmcs {
    pub(crate) fn new(accelerator: &Accelerator) -> Self {
        Self {
            plain: plain(),
            accelerator: accelerator.clone(),
            failure: Arc::default(),
        }
    }

    /// Fail if a tree of this prover lost its scratch file; its openings
    /// then read zeros.
    pub(crate) fn check(&self) -> Result<(), Error> {
        self.failure
            .get()
            .map_or(Ok(()), |reason| Err(Error::Device(reason.clone())))
    }
}

/// Committed rows of one power-of-two height and the root. The scratch file
/// holds the leaf words, row by row, then every digest layer below the root,
/// the leaves' first.
pub(crate) struct Tree<M> {
    rows: usize,
    widths: Vec<usize>,
    root: Digest,
    file: Option<File>,
    failure: Arc<OnceLock<String>>,
    matrices: PhantomData<M>,
}

impl<M: Matrix<Gl>> Tree<M> {
    fn build(matrices: Vec<M>, accelerator: &Accelerator, failure: Arc<OnceLock<String>>) -> Self {
        let rows = matrices[0].height();
        assert!(
            rows.is_power_of_two() && matrices.iter().all(|matrix| matrix.height() == rows),
            "WHIR commits matrices of one power-of-two height"
        );
        let width: usize = matrices.iter().map(|matrix| matrix.width()).sum();
        assert!(width > 0, "a Merkle leaf needs at least one word");
        let fill = |words: &mut [u64]| {
            words
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(r, row)| {
                    for (word, value) in row.iter_mut().zip(leaf(&matrices, r)) {
                        *word = value.as_canonical_u64();
                    }
                });
        };
        let layers = match accelerator
            .device()
            .and_then(|device| device.merkle_layers(rows, width, &fill))
        {
            Some(layers) => layers
                .into_iter()
                .map(|layer| {
                    layer
                        .into_iter()
                        .map(|digest| digest.map(Gl::new))
                        .collect()
                })
                .collect(),
            None => cpu_layers(&matrices, rows),
        };
        let root = layers.last().expect("a tree has a root")[0];
        let file = spill(&matrices, width, &layers[..layers.len() - 1])
            .map_err(|error| failure.set(format!("Merkle scratch file: {error}")))
            .ok();
        Self {
            rows,
            widths: matrices.iter().map(|matrix| matrix.width()).collect(),
            root,
            file,
            failure,
            matrices: PhantomData,
        }
    }

    /// `count` words at word `offset` of the scratch file; zeros, and a
    /// recorded failure, if the read fails.
    fn read(&self, offset: usize, count: usize) -> Vec<Gl> {
        let mut words = vec![0; count];
        if let Err(error) = self
            .file
            .as_ref()
            .map_or(Ok(()), |file| scratch::read(file, offset, &mut words))
        {
            let _ = self.failure.set(format!("Merkle scratch read: {error}"));
        }
        words.into_iter().map(Gl::new).collect()
    }

    fn rows(&self, index: usize) -> Vec<Vec<Gl>> {
        let width: usize = self.widths.iter().sum();
        let mut words = self.read(width * index, width).into_iter();
        self.widths
            .iter()
            .map(|&width| words.by_ref().take(width).collect())
            .collect()
    }

    /// How many layers hold siblings: every layer but the root's.
    fn depth(&self) -> usize {
        self.rows.trailing_zeros() as usize
    }

    /// Node `node` of layer `layer`, the leaves' digests being layer 0.
    fn digest(&self, layer: usize, node: usize) -> Digest {
        let width: usize = self.widths.iter().sum();
        let below: usize = (0..layer).map(|l| self.rows >> l).sum();
        let words = self.read(width * self.rows + DIGEST_LEN * (below + node), DIGEST_LEN);
        std::array::from_fn(|k| words[k])
    }

    /// The siblings on the path from leaf `index` to the root.
    fn path(&self, mut index: usize) -> Vec<Digest> {
        (0..self.depth())
            .map(|layer| {
                let sibling = self.digest(layer, index ^ 1);
                index /= 2;
                sibling
            })
            .collect()
    }

    /// Plonky3's pruned siblings of `indices`: level by level, in node
    /// order, the other node of each pair that holds exactly one known node.
    fn pruned(&self, indices: &[usize]) -> Vec<Digest> {
        let mut known = indices.to_vec();
        known.sort_unstable();
        known.dedup();
        let mut siblings = Vec::new();
        for layer in 0..self.depth() {
            let mut parents = Vec::with_capacity(known.len());
            let mut i = 0;
            while i < known.len() {
                let node = known[i];
                let both = node.is_multiple_of(2) && known.get(i + 1) == Some(&(node + 1));
                if !both {
                    siblings.push(self.digest(layer, node ^ 1));
                }
                parents.push(node / 2);
                i += if both { 2 } else { 1 };
            }
            known = parents;
        }
        siblings
    }
}

/// Write the leaf words, then `layers`, to a fresh scratch file.
fn spill<M: Matrix<Gl>>(matrices: &[M], width: usize, layers: &[Vec<Digest>]) -> std::io::Result<File> {
    let file = scratch::file()?;
    let rows = matrices[0].height();
    let block = (scratch::BLOCK_WORDS / width).max(1);
    (0..rows.div_ceil(block))
        .into_par_iter()
        .try_for_each(|chunk| {
            let first = chunk * block;
            let words: Vec<u64> = (first..rows.min(first + block))
                .flat_map(|r| leaf(matrices, r))
                .map(|value| value.as_canonical_u64())
                .collect();
            scratch::write(&file, width * first, &words)
        })?;
    let mut offset = width * rows;
    for layer in layers {
        let block = scratch::BLOCK_WORDS / DIGEST_LEN;
        layer
            .par_chunks(block)
            .enumerate()
            .try_for_each(|(chunk, digests)| {
                let words: Vec<u64> = digests
                    .iter()
                    .flatten()
                    .map(|value| value.as_canonical_u64())
                    .collect();
                scratch::write(&file, offset + DIGEST_LEN * block * chunk, &words)
            })?;
        offset += DIGEST_LEN * layer.len();
    }
    Ok(file)
}

/// Leaf `r`: row `r` of every matrix, in matrix order.
fn leaf<M: Matrix<Gl>>(matrices: &[M], r: usize) -> impl Iterator<Item = Gl> + '_ {
    matrices
        .iter()
        .flat_map(move |matrix| matrix.row(r).expect("the row is within the height"))
}

/// The digest layers on the CPU, `Packed::WIDTH` rows or nodes at a time,
/// in parallel.
fn cpu_layers<M: Matrix<Gl>>(matrices: &[M], rows: usize) -> Vec<Vec<Digest>> {
    let lanes = Packed::WIDTH;
    let sponge = Sponge::new(permutation().clone());
    let mut leaves = vec![[Gl::ZERO; DIGEST_LEN]; rows];
    leaves
        .par_chunks_mut(lanes)
        .enumerate()
        .for_each(|(chunk, digests)| {
            let first = chunk * lanes;
            if digests.len() < lanes {
                for (k, digest) in digests.iter_mut().enumerate() {
                    *digest = sponge.hash_iter(leaf(matrices, first + k));
                }
                return;
            }
            // Rows `first..first + lanes`, one per lane, read in place.
            let packed: [Packed; DIGEST_LEN] = sponge.hash_iter(
                matrices
                    .iter()
                    .flat_map(|matrix| matrix.vertically_packed_row::<Packed>(first)),
            );
            for (lane, digest) in digests.iter_mut().enumerate() {
                *digest = std::array::from_fn(|k| packed[k].as_slice()[lane]);
            }
        });
    let compress = Compress::new(permutation().clone());
    let mut layers = vec![leaves];
    while let Some(below) = layers.last().filter(|layer| layer.len() > 1) {
        let mut layer = vec![[Gl::ZERO; DIGEST_LEN]; below.len() / 2];
        layer
            .par_chunks_mut(lanes)
            .zip(below.par_chunks(2 * lanes))
            .for_each(|(digests, children)| {
                if digests.len() < lanes {
                    for (digest, pair) in digests.iter_mut().zip(children.chunks_exact(2)) {
                        *digest = compress.compress([pair[0], pair[1]]);
                    }
                    return;
                }
                let side = |s: usize| -> [Packed; DIGEST_LEN] {
                    std::array::from_fn(|k| Packed::from_fn(|lane| children[2 * lane + s][k]))
                };
                let packed = compress.compress([side(0), side(1)]);
                for (lane, digest) in digests.iter_mut().enumerate() {
                    *digest = std::array::from_fn(|k| packed[k].as_slice()[lane]);
                }
            });
        layers.push(layer);
    }
    layers
}

impl Scheme<Gl> for Mmcs {
    type ProverData<M> = Tree<M>;
    type Commitment = <Plain as Scheme<Gl>>::Commitment;
    type Proof = <Plain as Scheme<Gl>>::Proof;
    type MultiProof = <Plain as Scheme<Gl>>::MultiProof;
    type Error = <Plain as Scheme<Gl>>::Error;

    fn commit<M: Matrix<Gl>>(&self, inputs: Vec<M>) -> (Self::Commitment, Tree<M>) {
        let tree = Tree::build(inputs, &self.accelerator, self.failure.clone());
        (MerkleCap::new(vec![tree.root]), tree)
    }

    fn open_batch<M: Matrix<Gl>>(&self, index: usize, tree: &Tree<M>) -> BatchOpening<Gl, Self> {
        BatchOpening::new(tree.rows(index), tree.path(index))
    }

    fn get_matrices<'a, M: Matrix<Gl>>(&self, _tree: &'a Tree<M>) -> Vec<&'a M> {
        unreachable!("WHIR reads committed rows only through openings; the rows live in the scratch file")
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opening: BatchOpeningRef<'_, Gl, Self>,
    ) -> Result<(), Self::Error> {
        let opening = BatchOpeningRef::new(opening.opened_values, opening.opening_proof);
        self.plain.verify_batch(commit, dimensions, index, opening)
    }

    fn open_multi_batch<M: Matrix<Gl>>(
        &self,
        indices: &[usize],
        tree: &Tree<M>,
    ) -> (Vec<Vec<Vec<Gl>>>, Self::MultiProof) {
        let rows = indices.iter().map(|&index| tree.rows(index)).collect();
        let sibling_hashes = tree.pruned(indices);
        (rows, PrunedMerklePaths { sibling_hashes })
    }

    fn verify_multi_batch<R: AsRef<[Gl]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        self.plain
            .verify_multi_batch(commit, dimensions, indices, opened_values, proof)
    }
}
