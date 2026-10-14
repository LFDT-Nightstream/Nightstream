//! The provers' Merkle tree against Plonky3's own: equal commitments, paths
//! and pruned multi-openings, on the CPU and through a device.

use std::sync::Arc;

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;
use p3_commit_v08::Mmcs as _;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_matrix_v08::extension::FlatMatrixView;

use super::word;
use crate::device::{Accelerator, Device};
use crate::field::{Ext, Gl};
use crate::hash::{compress, hash_leaf, plain};
use crate::merkle::Mmcs;

/// A device that builds the tree one hash at a time with the CPU sponge.
struct Reference;

impl Device for Reference {
    fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; DIGEST_LEN]>>> {
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

    fn key_weights(&self, _powers: &[[u64; 3]], _blocks: usize) -> Option<Vec<[u64; 3]>> {
        None
    }

    fn dft_columns(&self, _values: &mut [u64], _width: usize, _generator: u64) -> Result<bool, String> {
        Ok(false)
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
    let schemes = [
        Mmcs::new(&Accelerator::cpu()),
        Mmcs::new(&Accelerator::new(Arc::new(Reference))),
    ];
    // Rows of one block, of four blocks and of a partial block; two
    // matrices side by side; a single row.
    let cases: [Vec<(usize, usize)>; 5] = [
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
