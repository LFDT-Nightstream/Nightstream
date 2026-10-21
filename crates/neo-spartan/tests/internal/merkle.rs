//! The provers' Merkle tree against Plonky3's own: equal commitments, paths
//! and pruned multi-openings, on the CPU and through a device, for rows in
//! one piece and for rows that arrive one coset at a time.

use std::sync::{Arc, Mutex};

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;
use p3_commit_v08::Mmcs as _;
use p3_dft_v08::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_matrix_v08::extension::FlatMatrixView;

use super::word;
use crate::device::{Accelerator, Device};
use crate::field::{Ext, Gl};
use crate::hash::{compress, hash_leaf, plain};
use crate::merkle::{Mmcs, Tree};

/// A device that builds trees one hash at a time with the CPU sponge and
/// runs DFTs with Plonky3's, and records the shape of every request.
#[derive(Default)]
pub(super) struct Reference {
    /// `(rows, width)` of every tree request.
    pub(super) trees: Mutex<Vec<(usize, usize)>>,
    /// `(rows, width)` of every leaf request.
    pub(super) leaves: Mutex<Vec<(usize, usize)>>,
    /// Rows of every DFT request.
    pub(super) dfts: Mutex<Vec<usize>>,
}

impl Device for Reference {
    fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; DIGEST_LEN]>>> {
        self.trees.lock().unwrap().push((rows, width));
        let mut words = vec![0; rows * width];
        fill(&mut words);
        let leaves: Vec<[Gl; DIGEST_LEN]> = words
            .chunks_exact(width)
            .map(|row| hash_leaf(&row.iter().map(|&w| Gl::from_u64(w)).collect::<Vec<_>>()))
            .collect();
        let mut layers = vec![leaves];
        while let Some(below) = layers.last().filter(|layer| layer.len() > 1) {
            let layer = below
                .chunks_exact(2)
                .map(|pair| compress(pair[0], pair[1]))
                .collect();
            layers.push(layer);
        }
        let words = |digest: [Gl; DIGEST_LEN]| digest.map(|value| value.as_canonical_u64());
        Some(
            layers
                .into_iter()
                .map(|layer| layer.into_iter().map(words).collect())
                .collect(),
        )
    }

    fn merkle_leaves(&self, values: &[u64], width: usize) -> Option<Vec<[u64; DIGEST_LEN]>> {
        self.leaves
            .lock()
            .unwrap()
            .push((values.len() / width, width));
        Some(
            values
                .chunks_exact(width)
                .map(|row| {
                    hash_leaf(&row.iter().map(|&w| Gl::from_u64(w)).collect::<Vec<_>>())
                        .map(|value| value.as_canonical_u64())
                })
                .collect(),
        )
    }

    fn key_weights(&self, _powers: &[[u64; 3]], _blocks: usize) -> Option<Vec<[u64; 3]>> {
        None
    }

    fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<bool, String> {
        let rows = values.len() / width;
        assert_eq!(
            Gl::two_adic_generator(rows.trailing_zeros() as usize).as_canonical_u64(),
            generator,
            "the generator has the order of the row count"
        );
        self.dfts.lock().unwrap().push(rows);
        let matrix = RowMajorMatrix::new(values.iter().map(|&w| Gl::from_u64(w)).collect(), width);
        let evaluations = Radix2DFTSmallBatch::<Gl>::default().dft_batch(matrix);
        for (word, value) in values.iter_mut().zip(evaluations.values) {
            *word = value.as_canonical_u64();
        }
        Ok(true)
    }

    fn gkr_layer<'a>(
        &self,
        _trees: Vec<Vec<&'a mut [u64]>>,
        _eq: &'a mut [u64],
    ) -> Option<Box<dyn crate::device::GkrLayer + 'a>> {
        None
    }
}

fn matrix(seed: u64, height: usize, width: usize) -> RowMajorMatrix<Gl> {
    let values = (0..height * width)
        .map(|i| Gl::from_u64(word(seed, i as u64) % Gl::ORDER_U64))
        .collect();
    RowMajorMatrix::new(values, width)
}

#[test]
fn provers_tree_matches_plonky3() {
    let plain = plain();
    let reference = Arc::new(Reference::default());
    let schemes = [
        Mmcs::new(&Accelerator::cpu()),
        Mmcs::new(&Accelerator::new(reference.clone())),
    ];
    // Rows of one block, of four blocks and of a partial block; two
    // matrices side by side; a single row; rows of two CPU subtrees.
    let cases: [Vec<(usize, usize)>; 6] = [
        vec![(1 << 20, 2)],
        vec![(1 << 12, 16)],
        vec![(1 << 12, 48)],
        vec![(1 << 11, 5)],
        vec![(1 << 10, 7), (1 << 10, 9)],
        vec![(1, 3)],
    ];
    for (case, shapes) in cases.iter().enumerate() {
        let inputs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(m, &(height, width))| matrix(case as u64 * 10 + m as u64, height, width))
            .collect();
        let height = shapes[0].0;
        let (root, tree) = plain.commit(inputs.clone());
        // Unsorted, repeated, and sibling indices.
        let queries = [height - 1, 0, 1 % height, height / 2, 0, height / 3];
        for scheme in &schemes {
            let (ours, our_tree) = scheme.commit(inputs.clone());
            assert_eq!(ours, root, "case {case}");
            for &index in &queries {
                let (ours, theirs) = (scheme.open_batch(index, &our_tree), plain.open_batch(index, &tree));
                assert_eq!(ours.opened_values, theirs.opened_values, "case {case} index {index}");
                assert_eq!(ours.opening_proof, theirs.opening_proof, "case {case} index {index}");
            }
            let (ours, theirs) = (
                scheme.open_multi_batch(&queries, &our_tree),
                plain.open_multi_batch(&queries, &tree),
            );
            assert_eq!(ours.0, theirs.0, "case {case}");
            assert_eq!(ours.1.sibling_hashes, theirs.1.sibling_hashes, "case {case}");
        }
        // The device hashes the leaves of the rows, then the nodes above
        // one subtree's leaves at a time, never the whole tree.
        let leaves = std::mem::take(&mut *reference.leaves.lock().unwrap());
        let trees = std::mem::take(&mut *reference.trees.lock().unwrap());
        if case == 0 {
            assert_eq!(leaves, [(height, 2)]);
            assert_eq!(trees, [(height / 4, 8), (height / 4, 8)]);
        }
    }
    // WHIR's extension rounds commit flattened extension rows.
    let values: Vec<Ext> = (0..(1u64 << 10) * 16)
        .map(|i| Ext::from(Gl::from_u64(word(7, i) % Gl::ORDER_U64)) * Ext::from(Gl::from_u64(i + 3)))
        .collect();
    let flat = || FlatMatrixView::<Gl, Ext, _>::new(RowMajorMatrix::new(values.clone(), 16));
    let root = plain.commit(vec![flat()]).0;
    for scheme in &schemes {
        assert_eq!(scheme.commit(vec![flat()]).0, root);
    }
    // Control: one changed word changes the root.
    let mut changed = matrix(0, 1 << 12, 16);
    changed.values[123] += Gl::ONE;
    assert_ne!(
        schemes[0].commit(vec![changed]).0,
        plain.commit(vec![matrix(0, 1 << 12, 16)]).0
    );
}

/// Rows that arrive one coset at a time commit to Plonky3's tree over the
/// rows in order, with Plonky3's openings, and the device hashes one
/// coset's leaves at a time.
#[test]
fn cosets_commit_to_the_rows_in_order() {
    let plain = plain();
    let reference = Arc::new(Reference::default());
    let schemes = [
        Mmcs::new(&Accelerator::cpu()),
        Mmcs::new(&Accelerator::new(reference.clone())),
    ];
    // Two subtrees of sixteen cosets; one subtree of sixteen; four cosets;
    // one row per coset; two rows; one coset.
    let cases = [
        (1 << 20, 3, 16),
        (1 << 12, 48, 16),
        (1 << 10, 16, 4),
        (1 << 4, 5, 16),
        (2, 7, 2),
        (1 << 11, 2, 1),
    ];
    for (case, &(height, width, cosets)) in cases.iter().enumerate() {
        let rows = matrix(40 + case as u64, height, width);
        let (root, tree) = plain.commit(vec![rows.clone()]);
        let queries = [height - 1, 0, 1 % height, height / 2, 0, height / 3];
        for scheme in &schemes {
            let (ours, our_tree): (_, Tree<RowMajorMatrix<Gl>>) =
                scheme.commit_cosets(height, width, cosets, |a, values| {
                    for (b, row) in values.chunks_mut(width).enumerate() {
                        row.copy_from_slice(&rows.values[(a + cosets * b) * width..][..width]);
                    }
                });
            assert_eq!(ours, root, "case {case}");
            for &index in &queries {
                let (ours, theirs) = (scheme.open_batch(index, &our_tree), plain.open_batch(index, &tree));
                assert_eq!(ours.opened_values, theirs.opened_values, "case {case} index {index}");
                assert_eq!(ours.opening_proof, theirs.opening_proof, "case {case} index {index}");
            }
            let (ours, theirs) = (
                scheme.open_multi_batch(&queries, &our_tree),
                plain.open_multi_batch(&queries, &tree),
            );
            assert_eq!(ours.0, theirs.0, "case {case}");
            assert_eq!(ours.1.sibling_hashes, theirs.1.sibling_hashes, "case {case}");
            assert!(scheme.check().is_ok());
        }
        let leaves = std::mem::take(&mut *reference.leaves.lock().unwrap());
        assert_eq!(leaves, vec![(height / cosets, width); cosets], "case {case}");
        let trees = std::mem::take(&mut *reference.trees.lock().unwrap());
        assert!(
            !trees.is_empty()
                && trees
                    .iter()
                    .all(|&(rows, request_width)| request_width == 2 * DIGEST_LEN && rows <= (1 << 18)),
            "case {case}: {trees:?}"
        );
    }
    // Control: a coset in the wrong place changes the root.
    let rows = matrix(1, 1 << 10, 4);
    let (wrong, _): (_, Tree<RowMajorMatrix<Gl>>) = schemes[0].commit_cosets(1 << 10, 4, 4, |a, values| {
        for (b, row) in values.chunks_mut(4).enumerate() {
            row.copy_from_slice(&rows.values[((3 - a) + 4 * b) * 4..][..4]);
        }
    });
    assert_ne!(wrong, plain.commit(vec![rows]).0);
}

/// A scratch read that fails at an opening fails that opening's check, not
/// the committer's.
#[test]
fn a_failed_scratch_read_fails_the_opening() {
    let (committer, opener) = (Mmcs::new(&Accelerator::cpu()), Mmcs::new(&Accelerator::cpu()));
    let (_, mut tree) = committer.commit(vec![matrix(1, 1 << 10, 4)]);
    opener.open_multi_batch(&[0, 5], &tree);
    assert!(opener.check().is_ok(), "a readable tree opens");
    tree.lose_scratch();
    opener.open_multi_batch(&[0, 5], &tree);
    assert!(opener.check().is_err(), "the failed read fails the opener");
    assert!(committer.check().is_ok(), "the commitment itself held");
}
