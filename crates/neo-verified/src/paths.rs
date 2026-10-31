//! Full authentication paths from a Plonky3 0.8 pruned multiproof of a binary tree that holds
//! one matrix, for `Hint::Paths`.
//!
//! Input parsing only, with no authority: the verifier checks every returned path against the
//! root. Plonky3 sends, level by level and group by group in ascending order, the one sibling
//! of each group that no queried leaf below it fixes; the others are nodes computed from
//! queried leaves.

use crate::circuit::{DIGEST_LEN, WIDTH};
use crate::field::{gl_word, Gl, GL_ZERO};
use crate::poseidon2::permute;

pub type Node = [Gl; DIGEST_LEN];

/// A pruned multiproof: the leaf digest of each query in query order, the siblings that it
/// sends, and the depth of the tree.
#[derive(Clone)]
pub struct Paths {
    pub leaves: Vec<Node>,
    pub siblings: Vec<Node>,
    pub depth: usize,
}

/// p3 `TruncatedPermutation` on two nodes: `[left, right, 0…]`, permute, first four lanes.
pub fn compress_nodes(left: Node, right: Node) -> Node {
    let mut state = [GL_ZERO; WIDTH];
    let mut lane = 0;
    while lane < DIGEST_LEN {
        state[lane] = left[lane];
        state[DIGEST_LEN.wrapping_add(lane)] = right[lane];
        lane = lane.wrapping_add(1);
    }
    let state = permute(state);
    [state[0], state[1], state[2], state[3]]
}

/// `nodes` with `(index, node)` in index order; a node already at `index` is replaced.
fn insert(nodes: &mut Vec<(usize, Node)>, index: usize, node: Node) {
    let mut sorted = Vec::new();
    let mut placed = false;
    let mut k = 0;
    while k < nodes.len() {
        let (at, existing) = nodes[k];
        if !placed && index <= at {
            sorted.push((index, node));
            placed = true;
        }
        if at != index {
            sorted.push((at, existing));
        }
        k = k.wrapping_add(1);
    }
    if !placed {
        sorted.push((index, node));
    }
    *nodes = sorted;
}

/// The node at `index` among `nodes`.
fn find(nodes: &[(usize, Node)], index: usize) -> Option<Node> {
    let mut found = None;
    let mut k = 0;
    while k < nodes.len() {
        if nodes[k].0 == index {
            found = Some(nodes[k].1);
        }
        k = k.wrapping_add(1);
    }
    found
}

/// The nodes that one level of the walk reads, and the known nodes one level up.
struct Level {
    /// The siblings fetched at this level and the known nodes, by index.
    fetched: Vec<(usize, Node)>,
    known: Vec<(usize, Node)>,
    parents: Vec<(usize, Node)>,
}

/// One level of the walk over `known`, taking siblings from `next`. `None` when `siblings`
/// runs out.
fn walk(known: Vec<(usize, Node)>, siblings: &[Node], next: &mut usize) -> Option<Level> {
    let mut fetched = Vec::new();
    let mut parents = Vec::new();
    let mut complete = true;
    let mut k = 0;
    while k < known.len() && complete {
        let (index, digest) = known[k];
        let paired = index.is_multiple_of(2)
            && k.wrapping_add(1) < known.len()
            && known[k.wrapping_add(1)].0 == index.wrapping_add(1);
        if paired {
            parents.push((
                index.wrapping_shr(1),
                compress_nodes(digest, known[k.wrapping_add(1)].1),
            ));
            k = k.wrapping_add(2);
        } else if *next < siblings.len() {
            let sibling = siblings[*next];
            *next = next.wrapping_add(1);
            if index.is_multiple_of(2) {
                fetched.push((index.wrapping_add(1), sibling));
                parents.push((index.wrapping_shr(1), compress_nodes(digest, sibling)));
            } else {
                fetched.push((index.wrapping_sub(1), sibling));
                parents.push((index.wrapping_shr(1), compress_nodes(sibling, digest)));
            }
            k = k.wrapping_add(1);
        } else {
            complete = false;
        }
    }
    if complete {
        Some(Level {
            fetched,
            known,
            parents,
        })
    } else {
        None
    }
}

/// The words of `Hint::Paths` for the leaf indices `indices`: the full path (bottom first) of
/// every index, as canonical words by query, level and lane. None when the proof does not have
/// exactly the siblings that the walk needs.
pub fn path_words(proof: &Paths, indices: &[u64]) -> Vec<u64> {
    let mut known = Vec::new();
    let mut q = 0;
    while q < indices.len() && q < proof.leaves.len() {
        insert(&mut known, indices[q] as usize, proof.leaves[q]);
        q = q.wrapping_add(1);
    }
    let mut levels = Vec::new();
    let mut next = 0;
    let mut complete = indices.len() == proof.leaves.len();
    let mut level = 0;
    while level < proof.depth && complete {
        match walk(known, &proof.siblings, &mut next) {
            Some(walked) => {
                known = walked.parents;
                levels.push((walked.fetched, walked.known));
            }
            None => {
                known = Vec::new();
                complete = false;
            }
        }
        level = level.wrapping_add(1);
    }
    complete = complete && next == proof.siblings.len();
    let mut words = Vec::new();
    let mut q = 0;
    while q < indices.len() && complete {
        let mut level = 0;
        while level < levels.len() && complete {
            let (fetched, known) = &levels[level];
            let sibling = (indices[q] as usize).wrapping_shr(level as u32) ^ 1;
            let node = match find(fetched, sibling) {
                Some(node) => Some(node),
                None => find(known, sibling),
            };
            match node {
                Some(node) => {
                    let mut lane = 0;
                    while lane < DIGEST_LEN {
                        words.push(gl_word(node[lane]));
                        lane = lane.wrapping_add(1);
                    }
                }
                None => complete = false,
            }
            level = level.wrapping_add(1);
        }
        q = q.wrapping_add(1);
    }
    if complete {
        words
    } else {
        Vec::new()
    }
}
