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
//! Rows can arrive one coset of the row index at a time (rows `a`, `a + c`,
//! `a + 2c`, ...): the tree then holds one coset in memory, never all rows.

use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError};

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;
use p3_commit_v08::{BatchOpening, BatchOpeningRef, Mmcs as Scheme};
use p3_field_v08::{PackedValue, PrimeCharacteristicRing, PrimeField64};
use p3_matrix_v08::dense::RowMajorMatrixView;
use p3_matrix_v08::{Dimensions, Matrix};
use p3_merkle_tree_v08::{MerkleCap, PrunedMerklePaths};
use p3_symmetric_v08::{CryptographicHasher, PseudoCompressionFunction};
use rayon::prelude::*;

use crate::device::Accelerator;
use crate::field::Gl;
use crate::hash::{permutation, plain, Compress, Packed, Plain, Sponge};
use crate::{scratch, Error};

type Digest = [Gl; DIGEST_LEN];
type Commitment = <Plain as Scheme<Gl>>::Commitment;

/// The Merkle commitment scheme of the WHIR provers; trees build on the
/// accelerator's device when it has one.
#[derive(Clone)]
pub(crate) struct Mmcs {
    plain: Plain,
    accelerator: Accelerator,
    /// The first scratch-file failure, shared with every clone of one prover.
    failure: Arc<Mutex<Option<io::Error>>>,
}

impl Mmcs {
    pub(crate) fn new(accelerator: &Accelerator) -> Self {
        Self {
            plain: plain(),
            accelerator: accelerator.clone(),
            failure: Arc::default(),
        }
    }

    /// Fail if a commitment or an opening of this prover lost a scratch
    /// file; it then read zeros. The failure is returned once.
    pub(crate) fn check(&self) -> Result<(), Error> {
        let failure = self
            .failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        failure.map_or(Ok(()), |error| Err(Error::Io(error)))
    }

    /// Commit to `rows` rows of `width` words that arrive one coset at a
    /// time: `coset(a, values)` overwrites `values` with rows `a`,
    /// `a + cosets`, `a + 2·cosets`, ... back to back.
    pub(crate) fn commit_cosets<M>(
        &self,
        rows: usize,
        width: usize,
        cosets: usize,
        coset: impl FnMut(usize, &mut Vec<Gl>),
    ) -> (Commitment, Tree<M>) {
        let tree = Tree::build(rows, vec![width], cosets, &self.accelerator, &self.failure, coset);
        (MerkleCap::new(vec![tree.root]), tree)
    }
}

/// Committed rows of one power-of-two height and the root. The scratch file
/// holds the leaf words, row by row, then every digest layer below the root,
/// the leaves' first. Rows and leaf digests are stored coset after coset:
/// row `a + cosets·b` at position `a·(rows / cosets) + b`. The layers above
/// the leaves are in row order.
pub(crate) struct Tree<M> {
    rows: usize,
    widths: Vec<usize>,
    cosets: usize,
    root: Digest,
    file: Option<File>,
    matrices: PhantomData<M>,
}

impl<M> Tree<M> {
    /// The tree over `rows` rows of `widths` words that arrive one coset at
    /// a time, as in `Mmcs::commit_cosets`.
    fn build(
        rows: usize,
        widths: Vec<usize>,
        cosets: usize,
        accelerator: &Accelerator,
        failure: &Mutex<Option<io::Error>>,
        coset: impl FnMut(usize, &mut Vec<Gl>),
    ) -> Self {
        let width: usize = widths.iter().sum();
        assert!(
            rows.is_power_of_two(),
            "WHIR commits matrices of one power-of-two height"
        );
        assert!(width > 0, "a Merkle leaf needs at least one word");
        assert!(
            cosets.is_power_of_two() && cosets <= SUBTREE.min(rows),
            "every subtree holds rows of every coset"
        );
        let built = scratch::file().and_then(|file| {
            let root = tree(&file, rows, width, cosets, accelerator, coset)?;
            Ok((file, root))
        });
        let (file, root) = match built {
            Ok((file, root)) => (Some(file), root),
            Err(error) => {
                record(failure, error);
                (None, [Gl::ZERO; DIGEST_LEN])
            }
        };
        Self {
            rows,
            widths,
            cosets,
            root,
            file,
            matrices: PhantomData,
        }
    }

    /// Where row `index` and its leaf digest sit in their regions of the
    /// scratch file, counted in rows.
    fn position(&self, index: usize) -> usize {
        index % self.cosets * (self.rows / self.cosets) + index / self.cosets
    }

    /// `count` words at word `offset` of the scratch file; zeros, and a
    /// failure recorded in the opener's `failure`, if the read fails.
    fn read(&self, offset: usize, count: usize, failure: &Mutex<Option<io::Error>>) -> Vec<Gl> {
        let mut words = vec![0; count];
        if let Err(error) = self
            .file
            .as_ref()
            .map_or(Ok(()), |file| scratch::read(file, offset, &mut words))
        {
            record(failure, error);
        }
        words.into_iter().map(Gl::new).collect()
    }

    fn rows(&self, index: usize, failure: &Mutex<Option<io::Error>>) -> Vec<Vec<Gl>> {
        let width: usize = self.widths.iter().sum();
        let mut words = self
            .read(width * self.position(index), width, failure)
            .into_iter();
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
    fn digest(&self, layer: usize, node: usize, failure: &Mutex<Option<io::Error>>) -> Digest {
        let width: usize = self.widths.iter().sum();
        let slot = if layer == 0 { self.position(node) } else { node };
        let words = self.read(
            layer_offset(width, self.rows, layer) + DIGEST_LEN * slot,
            DIGEST_LEN,
            failure,
        );
        std::array::from_fn(|k| words[k])
    }

    /// The siblings on the path from leaf `index` to the root.
    fn path(&self, mut index: usize, failure: &Mutex<Option<io::Error>>) -> Vec<Digest> {
        (0..self.depth())
            .map(|layer| {
                let sibling = self.digest(layer, index ^ 1, failure);
                index /= 2;
                sibling
            })
            .collect()
    }

    /// Plonky3's pruned siblings of `indices`: level by level, in node
    /// order, the other node of each pair that holds exactly one known node.
    fn pruned(&self, indices: &[usize], failure: &Mutex<Option<io::Error>>) -> Vec<Digest> {
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
                    siblings.push(self.digest(layer, node ^ 1, failure));
                }
                parents.push(node / 2);
                i += if both { 2 } else { 1 };
            }
            known = parents;
        }
        siblings
    }
}

/// Keep `error` unless `failure` already holds an earlier one.
fn record(failure: &Mutex<Option<io::Error>>, error: io::Error) {
    failure
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert(error);
}

/// Rows of one subtree: its digest layers go to the scratch file before
/// the next is hashed; it bounds only the buffers.
const SUBTREE: usize = scratch::BLOCK_WORDS;

/// Where layer `layer` starts in the scratch file, in words: after the leaf
/// words and every layer below it.
fn layer_offset(width: usize, rows: usize, layer: usize) -> usize {
    width * rows + DIGEST_LEN * (0..layer).map(|below| rows >> below).sum::<usize>()
}

/// `values` as canonical words at word `offset`.
fn write_values(file: &File, offset: usize, values: &[Gl]) -> io::Result<()> {
    values
        .par_chunks(scratch::BLOCK_WORDS)
        .enumerate()
        .try_for_each(|(block, values)| {
            let words: Vec<u64> = values
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect();
            scratch::write(file, offset + block * scratch::BLOCK_WORDS, &words)
        })
}

fn write_digests(file: &File, offset: usize, digests: &[Digest]) -> io::Result<()> {
    let block = scratch::BLOCK_WORDS / DIGEST_LEN;
    digests
        .par_chunks(block)
        .enumerate()
        .try_for_each(|(chunk, digests)| {
            let words: Vec<u64> = digests
                .iter()
                .flatten()
                .map(|value| value.as_canonical_u64())
                .collect();
            scratch::write(file, offset + DIGEST_LEN * block * chunk, &words)
        })
}

/// Each coset's rows and leaf digests as the coset arrives, then the layers
/// above the leaves one subtree at a time, in row order. Every layer below
/// the root goes to `file`. Returns the root.
fn tree(
    file: &File,
    rows: usize,
    width: usize,
    cosets: usize,
    accelerator: &Accelerator,
    mut coset: impl FnMut(usize, &mut Vec<Gl>),
) -> io::Result<Digest> {
    let length = rows / cosets;
    let leaves_at = layer_offset(width, rows, 0);
    let mut values = vec![Gl::ZERO; length * width];
    for a in 0..cosets {
        coset(a, &mut values);
        assert_eq!(values.len(), length * width, "a coset keeps its shape");
        write_values(file, width * a * length, &values)?;
        write_digests(
            file,
            leaves_at + DIGEST_LEN * a * length,
            &leaves(&values, width, accelerator),
        )?;
    }
    drop(values);
    let size = SUBTREE.min(rows);
    let mut roots = Vec::with_capacity(rows / size);
    for start in (0..rows).step_by(size) {
        let layers = layers(read_leaves(file, rows, width, cosets, start, size)?, accelerator);
        for (layer, digests) in layers.iter().enumerate().take(layers.len() - 1).skip(1) {
            write_digests(
                file,
                layer_offset(width, rows, layer) + DIGEST_LEN * (start >> layer),
                digests,
            )?;
        }
        roots.push(layers.last().expect("a subtree has a root")[0]);
    }
    let mut layer = size.trailing_zeros() as usize;
    while roots.len() > 1 {
        write_digests(file, layer_offset(width, rows, layer), &roots)?;
        roots = compress_layer(&roots);
        layer += 1;
    }
    Ok(roots[0])
}

/// The leaf digests of rows `start..start + size` in row order, gathered
/// from every coset's stored leaves.
fn read_leaves(
    file: &File,
    rows: usize,
    width: usize,
    cosets: usize,
    start: usize,
    size: usize,
) -> io::Result<Vec<Digest>> {
    let (length, count) = (rows / cosets, size / cosets);
    let pieces = (0..cosets)
        .into_par_iter()
        .map(|a| {
            let mut words = vec![0; DIGEST_LEN * count];
            let offset = layer_offset(width, rows, 0) + DIGEST_LEN * (a * length + start / cosets);
            scratch::read(file, offset, &mut words).map(|()| words)
        })
        .collect::<io::Result<Vec<_>>>()?;
    Ok((0..size)
        .into_par_iter()
        .map(|j| std::array::from_fn(|k| Gl::new(pieces[j % cosets][DIGEST_LEN * (j / cosets) + k])))
        .collect())
}

/// Leaf `r`: row `r` of every matrix, in matrix order.
fn leaf<M: Matrix<Gl>>(matrices: &[M], r: usize) -> impl Iterator<Item = Gl> + '_ {
    matrices
        .iter()
        .flat_map(move |matrix| matrix.row(r).expect("the row is within the height"))
}

/// The leaf digests of the rows of `width` words in `values`, on the device
/// when it takes them.
fn leaves(values: &[Gl], width: usize, accelerator: &Accelerator) -> Vec<Digest> {
    // SAFETY: `Goldilocks` is `repr(transparent)` over `u64`.
    let words = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u64>(), values.len()) };
    match accelerator
        .device()
        .and_then(|device| device.merkle_leaves(words, width))
    {
        Some(digests) => digests
            .into_iter()
            .map(|digest| digest.map(Gl::new))
            .collect(),
        None => cpu_leaves(values, width),
    }
}

/// The leaf digests of the rows of `width` words in `values` on the CPU,
/// `Packed::WIDTH` rows at a time, in parallel.
fn cpu_leaves(values: &[Gl], width: usize) -> Vec<Digest> {
    let lanes = Packed::WIDTH;
    let matrix = RowMajorMatrixView::new(values, width);
    let sponge = Sponge::new(permutation().clone());
    let mut leaves = vec![[Gl::ZERO; DIGEST_LEN]; matrix.height()];
    leaves
        .par_chunks_mut(lanes)
        .enumerate()
        .for_each(|(chunk, digests)| {
            let first = chunk * lanes;
            if digests.len() < lanes {
                for (k, digest) in digests.iter_mut().enumerate() {
                    *digest = sponge.hash_slice(&values[(first + k) * width..][..width]);
                }
                return;
            }
            // Rows `first..first + lanes`, one per lane, read in place.
            let packed: [Packed; DIGEST_LEN] = sponge.hash_iter(matrix.vertically_packed_row::<Packed>(first));
            for (lane, digest) in digests.iter_mut().enumerate() {
                *digest = std::array::from_fn(|k| packed[k].as_slice()[lane]);
            }
        });
    leaves
}

/// Every digest layer of the subtree over `leaves`, leaves first and the
/// root last. The device builds the layers above the leaves when it takes
/// them: a node is its leaf hash of the two children's eight words.
fn layers(leaves: Vec<Digest>, accelerator: &Accelerator) -> Vec<Vec<Digest>> {
    let above = {
        let fill = |words: &mut [u64]| {
            words
                .par_chunks_mut(DIGEST_LEN)
                .zip(leaves.par_iter())
                .for_each(|(words, digest)| {
                    for (word, value) in words.iter_mut().zip(digest) {
                        *word = value.as_canonical_u64();
                    }
                });
        };
        accelerator
            .device()
            .filter(|_| leaves.len() > 1)
            .and_then(|device| device.merkle_layers(leaves.len() / 2, 2 * DIGEST_LEN, &fill))
    };
    let mut layers = vec![leaves];
    match above {
        Some(above) => layers.extend(above.into_iter().map(|layer| {
            layer
                .into_iter()
                .map(|digest| digest.map(Gl::new))
                .collect()
        })),
        None => {
            while let Some(below) = layers.last().filter(|layer| layer.len() > 1) {
                let layer = compress_layer(below);
                layers.push(layer);
            }
        }
    }
    layers
}

/// The layer above `below`, `Packed::WIDTH` nodes at a time, in parallel.
fn compress_layer(below: &[Digest]) -> Vec<Digest> {
    let lanes = Packed::WIDTH;
    let compress = Compress::new(permutation().clone());
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
    layer
}

#[cfg(test)]
impl<M> Tree<M> {
    /// Swap the scratch file for an empty one, so every later read fails.
    pub(crate) fn lose_scratch(&mut self) {
        self.file = scratch::file().ok();
    }
}

impl Scheme<Gl> for Mmcs {
    type ProverData<M> = Tree<M>;
    type Commitment = Commitment;
    type Proof = <Plain as Scheme<Gl>>::Proof;
    type MultiProof = <Plain as Scheme<Gl>>::MultiProof;
    type Error = <Plain as Scheme<Gl>>::Error;

    fn commit<M: Matrix<Gl>>(&self, inputs: Vec<M>) -> (Self::Commitment, Tree<M>) {
        let rows = inputs[0].height();
        assert!(
            inputs.iter().all(|matrix| matrix.height() == rows),
            "WHIR commits matrices of one power-of-two height"
        );
        let widths: Vec<usize> = inputs.iter().map(|matrix| matrix.width()).collect();
        let width: usize = widths.iter().sum();
        let tree = Tree::build(rows, widths, 1, &self.accelerator, &self.failure, |_, values| {
            values
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(r, row)| {
                    for (value, input) in row.iter_mut().zip(leaf(&inputs, r)) {
                        *value = input;
                    }
                });
        });
        (MerkleCap::new(vec![tree.root]), tree)
    }

    fn open_batch<M: Matrix<Gl>>(&self, index: usize, tree: &Tree<M>) -> BatchOpening<Gl, Self> {
        BatchOpening::new(tree.rows(index, &self.failure), tree.path(index, &self.failure))
    }

    fn get_matrices<'a, M: Matrix<Gl>>(&self, _tree: &'a Tree<M>) -> Vec<&'a M> {
        unreachable!("WHIR reads committed rows only through openings; the rows live in the scratch file")
    }

    fn get_matrix_heights<M: Matrix<Gl>>(&self, tree: &Tree<M>) -> Vec<usize> {
        vec![tree.rows; tree.widths.len()]
    }

    fn get_max_height<M: Matrix<Gl>>(&self, tree: &Tree<M>) -> usize {
        tree.rows
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
        let rows = indices
            .iter()
            .map(|&index| tree.rows(index, &self.failure))
            .collect();
        let sibling_hashes = tree.pruned(indices, &self.failure);
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
