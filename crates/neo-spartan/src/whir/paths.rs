//! Full authentication paths from a Plonky3 0.8 pruned multiproof of a binary
//! tree that holds one matrix.
//!
//! Input parsing only, with no authority: the verifier checks every returned
//! path against the root. Plonky3 sends, level by level and group by group in
//! ascending order, the one sibling of each group that no queried leaf below
//! it fixes; the others are nodes computed from queried leaves.

use std::collections::BTreeMap;

use crate::field::Gl;
use crate::hash::compress;

pub(crate) type Digest = [Gl; 4];

/// The full path (bottom first) of every index in `indices`, given each
/// index's leaf digest and the pruned `siblings`. `None` if the proof does not
/// have exactly the siblings the walk needs.
pub(crate) fn expand(
    indices: &[usize],
    leaves: &[Digest],
    siblings: &[Digest],
    depth: usize,
) -> Option<Vec<Vec<Digest>>> {
    assert_eq!(indices.len(), leaves.len());
    let mut known: BTreeMap<usize, Digest> = indices
        .iter()
        .copied()
        .zip(leaves.iter().copied())
        .collect();
    let mut levels = Vec::with_capacity(depth);
    let mut supplied = siblings.iter();
    for _ in 0..depth {
        let mut fetched: BTreeMap<usize, Digest> = BTreeMap::new();
        let mut parents = BTreeMap::new();
        let nodes: Vec<(usize, Digest)> = known.iter().map(|(&i, &d)| (i, d)).collect();
        let mut k = 0;
        while k < nodes.len() {
            let (index, digest) = nodes[k];
            let (left, right) = if index % 2 == 0 {
                match nodes.get(k + 1) {
                    Some(&(next, sibling)) if next == index + 1 => {
                        k += 1;
                        (digest, sibling)
                    }
                    _ => {
                        let sibling = *supplied.next()?;
                        fetched.insert(index + 1, sibling);
                        (digest, sibling)
                    }
                }
            } else {
                let sibling = *supplied.next()?;
                fetched.insert(index - 1, sibling);
                (sibling, digest)
            };
            parents.insert(index / 2, compress(left, right));
            k += 1;
        }
        fetched.extend(known);
        levels.push(fetched);
        known = parents;
    }
    if supplied.next().is_some() {
        return None;
    }
    Some(
        indices
            .iter()
            .map(|&index| {
                (0..depth)
                    .map(|level| levels[level][&((index >> level) ^ 1)])
                    .collect()
            })
            .collect(),
    )
}
