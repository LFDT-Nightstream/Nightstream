//! The batched Merkle leaf hasher against Plonky3's own tree: equal roots
//! and openings on the CPU and through a device.

use std::sync::Arc;

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, WIDTH};
use p3_commit_v08::Mmcs as _;
use p3_field_v08::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_merkle_tree_v08::MerkleTreeMmcs;
use p3_symmetric_v08::{PaddingFreeSponge, TruncatedPermutation};

use super::word;
use crate::device::{Accelerator, Device};
use crate::field::Gl;
use crate::hash::{hash_leaf, mmcs, permutation, Perm};

type Packed = <Gl as Field>::Packing;
type Plain = MerkleTreeMmcs<
    Packed,
    Packed,
    PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_LEN>,
    TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>,
    2,
    DIGEST_LEN,
>;

/// A device that hashes one row at a time with the CPU sponge.
struct Reference;

impl Device for Reference {
    fn hash_rows(&self, words: &[u64], width: usize) -> Option<Vec<[u64; DIGEST_LEN]>> {
        let digests = words
            .chunks_exact(width)
            .map(|row| {
                let row: Vec<Gl> = row.iter().map(|&w| Gl::from_u64(w)).collect();
                hash_leaf(&row).map(|value| value.as_canonical_u64())
            })
            .collect();
        Some(digests)
    }
}

fn matrix(seed: u64, height: usize, width: usize) -> RowMajorMatrix<Gl> {
    let values = (0..height * width)
        .map(|i| Gl::from_u64(word(seed, i as u64) % Gl::ORDER_U64))
        .collect();
    RowMajorMatrix::new(values, width)
}

#[test]
fn batched_leaves_match_plonky3() {
    let perm = permutation().clone();
    let plain = Plain::new(PaddingFreeSponge::new(perm.clone()), TruncatedPermutation::new(perm), 0);
    let accelerators = [Accelerator::cpu(), Accelerator::new(Arc::new(Reference))];
    // Two leaf batches, rows of one block, of four blocks, and a partial
    // block; then a shorter matrix injected higher up.
    let cases: [Vec<(usize, usize)>; 4] = [
        vec![(1 << 17, 16)],
        vec![(1 << 17, 48)],
        vec![(1 << 16, 5)],
        vec![(1 << 17, 7), (1 << 15, 9)],
    ];
    for (case, shapes) in cases.iter().enumerate() {
        let inputs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(m, &(height, width))| matrix(case as u64 * 10 + m as u64, height, width))
            .collect();
        let (root, tree) = plain.commit(inputs.clone());
        for accelerator in &accelerators {
            let (ours, our_tree) = mmcs(accelerator).commit(inputs.clone());
            assert_eq!(ours, root, "case {case}");
            for index in [0, 1, 4097, shapes[0].0 - 1] {
                let (ours, theirs) = (
                    mmcs(accelerator).open_batch(index, &our_tree),
                    plain.open_batch(index, &tree),
                );
                assert_eq!(ours.opened_values, theirs.opened_values, "case {case} index {index}");
                assert_eq!(ours.opening_proof, theirs.opening_proof, "case {case} index {index}");
            }
        }
    }
    // Control: one changed word changes the root.
    let mut changed = matrix(0, 1 << 17, 16);
    changed.values[123] += Gl::ONE;
    assert_ne!(
        mmcs(&Accelerator::cpu()).commit(vec![changed]).0,
        plain.commit(vec![matrix(0, 1 << 17, 16)]).0
    );
}
