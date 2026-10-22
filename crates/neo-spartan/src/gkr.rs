//! Fraction sums `Σ_x p(x) / q(x)` over several trees, proven by one batched
//! GKR (logUp-GKR, Papini and Haböck, ePrint 2023/1284).
//!
//! Owns: the layered fraction trees, the batched layer sum-checks, and the
//! leaf claims. Does not own what the leaves mean: the caller checks each
//! root and each leaf claim against its own columns.
//!
//! Layer `k` of a tree has `2^k` nodes indexed by the low `k` bits of a leaf
//! index; a node of layer `k - 1` combines the two layer-`k` nodes that differ
//! in bit `k - 1`. A leaf is `(Π_j f_j(x), d(x))` with zero, one or two
//! numerator factors `f_j` and one denominator `d`, each a multilinear table.
//! All trees start at the root together; at each layer the trees that reach
//! it share one sum-check, so trees of equal depth share the leaf point.
//! The leaf claim of a tree is every factor's and the denominator's value at
//! its leaf point.
//!
//! The prover never stores leaves: it computes them from the caller's tables
//! each time it reads them, and recomputes the layer above them the same way.
//! It builds the other levels bottom up, one tree at a time, and keeps each
//! in a scratch file until its layer's sum-check (top down) reads it back.
//! A layer that reads computed entries streams its first round, so the prover
//! holds half of that layer.

use p3_challenger_v08::FieldChallenger;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::circuit::algebra;
use crate::circuit::hash::Duplex;
use crate::circuit::Backend;
use crate::device::Accelerator;
use crate::field::{eq_table, ext_words, fold_low, Ext, Gl};
use crate::hash::Challenger;
use crate::{scratch, sumcheck, Error};

/// One tree layer, or stored leaves: numerator factors (none means all ones)
/// and the denominator, each of length `2^depth`.
#[derive(Default)]
pub(crate) struct Tree {
    pub(crate) factors: Vec<Vec<Ext>>,
    pub(crate) denominator: Vec<Ext>,
}

/// What the verifier knows of a tree before the proof.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeShape {
    pub(crate) depth: usize,
    pub(crate) factors: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct GkrProof {
    /// `(P, Q)` at each tree's root, in tree order.
    pub(crate) roots: Vec<[Ext; 2]>,
    /// One entry per layer `k = 1..=` the largest depth.
    pub(crate) layers: Vec<Layer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Layer {
    /// `k - 1` round messages of the batched sum-check.
    pub(crate) rounds: Vec<Vec<Ext>>,
    /// For each tree that reaches layer `k`, in tree order: each part's values
    /// at the two children. The parts are `P, Q` inside the tree and the
    /// factors, then the denominator, at its leaf layer.
    pub(crate) children: Vec<Vec<Ext>>,
}

/// One tree's root fraction and its leaf claim (`E` is `Ext` for the prover,
/// a backend value for the verifier).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TreeClaim<E> {
    pub(crate) root: [E; 2],
    pub(crate) point: Vec<E>,
    /// The factors' values, then the denominator's, at `point`.
    pub(crate) values: Vec<E>,
}

impl Tree {
    fn depth(&self) -> usize {
        let depth = self.denominator.len().trailing_zeros() as usize;
        assert!(depth >= 1 && self.denominator.len() == 1 << depth);
        assert!(self.factors.len() <= 2 && self.factors.iter().all(|f| f.len() == 1 << depth));
        depth
    }

    fn parts(&self) -> impl Iterator<Item = &Vec<Ext>> {
        self.factors
            .iter()
            .chain(std::iter::once(&self.denominator))
    }

    fn parts_mut(&mut self) -> impl Iterator<Item = &mut Vec<Ext>> {
        self.factors
            .iter_mut()
            .chain(std::iter::once(&mut self.denominator))
    }
}

/// The batched round degree: 3, or `factors + 2` at a product leaf layer.
fn degree(factors: impl Iterator<Item = usize>) -> usize {
    factors.map(|f| f + 2).max().unwrap_or(0).max(3)
}

/// One part of a tree's leaves: `fill(from, out)` writes leaves
/// `from..from + out.len()`.
pub(crate) type Part<'a> = Box<dyn Fn(usize, &mut [Ext]) + Sync + 'a>;

/// A tree's `2^depth` leaves, computed on demand: the prover never stores
/// them.
pub(crate) struct Leaves<'a> {
    depth: usize,
    /// At most two numerator factors, then the denominator.
    parts: Vec<Part<'a>>,
}

impl<'a> Leaves<'a> {
    /// `count` leaves (a power of two, at least two) with numerator
    /// `factors` (none means all ones) and `denominator`.
    pub(crate) fn new(count: usize, mut factors: Vec<Part<'a>>, denominator: Part<'a>) -> Self {
        assert!(count >= 2 && count.is_power_of_two() && factors.len() <= 2);
        factors.push(denominator);
        Self {
            depth: count.trailing_zeros() as usize,
            parts: factors,
        }
    }
}

/// The part whose leaf `x` is `value(x)`.
pub(crate) fn part<'a>(value: impl Fn(usize) -> Ext + Sync + 'a) -> Part<'a> {
    Box::new(move |from, out: &mut [Ext]| {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, entry)| *entry = value(from + i))
    })
}

impl Tree {
    /// These stored leaves as `Leaves`.
    pub(crate) fn leaves(&self) -> Leaves<'_> {
        Leaves::new(
            1 << self.depth(),
            self.factors.iter().map(|factor| stored(factor)).collect(),
            stored(&self.denominator),
        )
    }
}

fn stored(values: &[Ext]) -> Part<'_> {
    Box::new(move |from, out: &mut [Ext]| out.copy_from_slice(&values[from..from + out.len()]))
}

/// One tree layer in the scratch file: its parts, one after another.
struct Spilled {
    offset: usize,
    parts: usize,
    nodes: usize,
}

/// Append `tree`'s parts at word `end`.
fn spill(file: &std::fs::File, end: &mut usize, tree: &Tree) -> Result<Spilled, Error> {
    let spilled = Spilled {
        offset: *end,
        parts: tree.factors.len() + 1,
        nodes: tree.denominator.len(),
    };
    for part in tree.parts() {
        scratch::write(file, *end, table_words(part))?;
        *end += 3 * part.len();
    }
    Ok(spilled)
}

fn restore(file: &std::fs::File, spilled: &Spilled) -> Result<Tree, Error> {
    let mut parts = (0..spilled.parts)
        .map(|part| {
            let mut values = vec![Ext::ZERO; spilled.nodes];
            let offset = spilled.offset + 3 * spilled.nodes * part;
            scratch::read(file, offset, ext_words_mut(&mut values))?;
            Ok(values)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let denominator = parts.pop().expect("a tree has a denominator");
    Ok(Tree {
        factors: parts,
        denominator,
    })
}

/// Each node's `P = p_0·q_1 + p_1·q_0` and `Q = q_0·q_1` from both children's
/// entries of every part (factors, then the denominator; no factors means
/// `p = 1`).
fn parents(children: &[[&[Ext]; 2]], p: &mut [Ext], q: &mut [Ext]) {
    let factors = children.len() - 1;
    p.par_iter_mut().zip(q).enumerate().for_each(|(i, (p, q))| {
        let (q0, q1) = (children[factors][0][i], children[factors][1][i]);
        *p = match &children[..factors] {
            [] => q0 + q1,
            [a] => a[0][i] * q1 + a[1][i] * q0,
            [a, c] => a[0][i] * c[0][i] * q1 + a[1][i] * c[1][i] * q0,
            _ => unreachable!("a tree has at most two factors"),
        };
        *q = q0 * q1;
    });
}

/// The layer above `level`, from its two halves.
fn parent(level: &Tree) -> Tree {
    let half = level.denominator.len() / 2;
    let children: Vec<[&[Ext]; 2]> = level
        .parts()
        .map(|part| {
            let (low, high) = part.split_at(half);
            [low, high]
        })
        .collect();
    let (mut p, mut q) = (vec![Ext::ZERO; half], vec![Ext::ZERO; half]);
    parents(&children, &mut p, &mut q);
    Tree {
        factors: vec![p],
        denominator: q,
    }
}

/// One tree's layer as its sum-check reads it.
enum Source<'a> {
    /// The tree's leaves.
    Leaves(&'a Leaves<'a>),
    /// The layer above the leaves, recomputed from them.
    AboveLeaves(&'a Leaves<'a>),
    /// A layer read back from the scratch file.
    Stored(Tree),
}

impl Source<'_> {
    fn parts(&self) -> usize {
        match self {
            Source::Leaves(leaves) => leaves.parts.len(),
            Source::AboveLeaves(_) => 2,
            Source::Stored(tree) => tree.factors.len() + 1,
        }
    }

    fn nodes(&self) -> usize {
        match self {
            Source::Leaves(leaves) => 1 << leaves.depth,
            Source::AboveLeaves(leaves) => 1 << (leaves.depth - 1),
            Source::Stored(tree) => tree.denominator.len(),
        }
    }

    /// Fill `out`, one slice per part, with each part's entries from `from`.
    /// The layer above the leaves computes its leaves in `scratch`.
    fn read(&self, from: usize, out: &mut [&mut [Ext]], scratch: &mut Vec<Ext>) {
        match self {
            Source::Leaves(leaves) => out
                .par_iter_mut()
                .zip(&leaves.parts)
                .for_each(|(out, fill)| fill(from, out)),
            Source::AboveLeaves(leaves) => {
                let (half, count) = (1 << (leaves.depth - 1), out[0].len());
                // Each leaf part's entries under both children, side by side.
                scratch.resize(2 * count * leaves.parts.len(), Ext::ZERO);
                scratch
                    .par_chunks_mut(count)
                    .enumerate()
                    .for_each(|(j, values)| leaves.parts[j / 2](j % 2 * half + from, values));
                let children: Vec<[&[Ext]; 2]> = scratch
                    .chunks(2 * count)
                    .map(|pair| {
                        let (low, high) = pair.split_at(count);
                        [low, high]
                    })
                    .collect();
                let [p, q] = out else {
                    unreachable!("the layer above the leaves has two parts")
                };
                parents(&children, p, q);
            }
            Source::Stored(tree) => {
                for (out, values) in out.iter_mut().zip(tree.parts()) {
                    out.copy_from_slice(&values[from..from + out.len()]);
                }
            }
        }
    }

    /// The whole layer in memory, read `CHUNK` entries at a time.
    fn into_tree(self) -> Tree {
        if let Source::Stored(tree) = self {
            return tree;
        }
        let nodes = self.nodes();
        let mut parts = vec![vec![Ext::ZERO; nodes]; self.parts()];
        let mut scratch = Vec::new();
        for from in (0..nodes).step_by(CHUNK) {
            let mut out: Vec<&mut [Ext]> = parts
                .iter_mut()
                .map(|part| &mut part[from..(from + CHUNK).min(nodes)])
                .collect();
            self.read(from, &mut out, &mut scratch);
        }
        let denominator = parts.pop().expect("a tree has a denominator");
        Tree {
            factors: parts,
            denominator,
        }
    }
}

/// Pairs (or entries) read at once by a streamed pass; it bounds only the
/// buffers.
const CHUNK: usize = scratch::BLOCK_WORDS / 4;

/// Fill `buffers` (one per tree and part, both children's entries side by
/// side) with pairs `m0..m1` of each source, every tree in parallel, each
/// with its own scratch.
fn read_pairs(sources: &[Source], m0: usize, m1: usize, buffers: &mut [Tree], scratch: &mut [Vec<Ext>]) {
    let count = m1 - m0;
    buffers
        .par_iter_mut()
        .zip(scratch)
        .zip(sources)
        .for_each(|((tree, scratch), source)| {
            let half = source.nodes() / 2;
            let (mut low, mut high): (Vec<&mut [Ext]>, Vec<&mut [Ext]>) = tree
                .parts_mut()
                .map(|values| {
                    values.resize(4 * count, Ext::ZERO);
                    values.split_at_mut(2 * count)
                })
                .unzip();
            source.read(2 * m0, &mut low, scratch);
            source.read(half + 2 * m0, &mut high, scratch);
        });
}

/// Round 1 of a layer read from `sources`: the message from one pass, then
/// the tables folded at the round's challenge from a second, so the layer is
/// in memory only at half its size.
fn first_round(
    sources: &[Source],
    eq: &[Ext],
    weights: &[Ext],
    mu: Ext,
    degree: usize,
    challenger: &mut Challenger,
) -> (Vec<Ext>, Ext, Vec<Tree>) {
    let half = sources[0].nodes() / 2;
    let quarter = half / 2;
    let mut buffers: Vec<Tree> = sources
        .iter()
        .map(|source| Tree {
            factors: vec![Vec::new(); source.parts() - 1],
            denominator: Vec::new(),
        })
        .collect();
    let mut scratch = vec![Vec::new(); sources.len()];
    let mut message = vec![Ext::ZERO; degree];
    for m0 in (0..quarter).step_by(CHUNK) {
        let m1 = (m0 + CHUNK).min(quarter);
        read_pairs(sources, m0, m1, &mut buffers, &mut scratch);
        let part = host_message(&buffers, &eq[2 * m0..2 * m1], weights, mu, degree);
        for (sum, value) in message.iter_mut().zip(part) {
            *sum += value;
        }
    }
    let r = sumcheck::send(&message, challenger);
    let mut folded: Vec<Tree> = sources
        .iter()
        .map(|source| Tree {
            factors: vec![vec![Ext::ZERO; half]; source.parts() - 1],
            denominator: vec![Ext::ZERO; half],
        })
        .collect();
    for m0 in (0..quarter).step_by(CHUNK) {
        let m1 = (m0 + CHUNK).min(quarter);
        read_pairs(sources, m0, m1, &mut buffers, &mut scratch);
        for (tree, out) in buffers.iter().zip(&mut folded) {
            for (values, target) in tree.parts().zip(out.parts_mut()) {
                // `fold_low` of each child's entries, into place.
                let (low, high) = values.split_at(2 * (m1 - m0));
                for (from, at) in [(low, m0), (high, quarter + m0)] {
                    target[at..at + m1 - m0]
                        .par_iter_mut()
                        .zip(from.par_chunks_exact(2))
                        .for_each(|(value, pair)| *value = pair[0] + r * (pair[1] - pair[0]));
                }
            }
        }
    }
    (message, r, folded)
}

/// Prove the fraction sums of `trees`; return the proof and one claim per
/// tree. A layer's rounds run on the accelerator's device when it takes them
/// and every tree's layer is stored; a layer that reads leaves streams its
/// first round on the host.
pub(crate) fn prove(
    trees: &[Leaves<'_>],
    challenger: &mut Challenger,
    accelerator: &Accelerator,
) -> Result<(GkrProof, Vec<TreeClaim<Ext>>), Error> {
    let file = scratch::file()?;
    let mut end = 0;
    let depths: Vec<usize> = trees.iter().map(|leaves| leaves.depth).collect();
    let mut roots: Vec<[Ext; 2]> = Vec::new();
    // stored[i][k - 1] is tree i at layer k, below the layer above its leaves.
    let mut stored: Vec<Vec<Spilled>> = Vec::new();
    for leaves in trees {
        let mut spilled = Vec::new();
        let mut level = Source::AboveLeaves(leaves).into_tree();
        if level.denominator.len() > 1 {
            level = parent(&level);
        }
        while level.denominator.len() > 1 {
            let next = parent(&level);
            spilled.push(spill(&file, &mut end, &level)?);
            level = next;
        }
        spilled.reverse();
        roots.push([level.factors[0][0], level.denominator[0]]);
        stored.push(spilled);
    }
    for root in &roots {
        challenger.observe_algebra_slice(root);
    }

    let mut claims: Vec<Option<TreeClaim<Ext>>> = vec![None; depths.len()];
    let mut point = Vec::new();
    let mut layers = Vec::new();
    for k in 1..=depths.iter().copied().max().unwrap_or(0) {
        let active: Vec<usize> = (0..depths.len()).filter(|&i| depths[i] >= k).collect();
        let sources = active
            .iter()
            .map(|&i| {
                Ok(match depths[i] - k {
                    0 => Source::Leaves(&trees[i]),
                    1 => Source::AboveLeaves(&trees[i]),
                    _ => Source::Stored(restore(&file, &stored[i][k - 1])?),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mu: Ext = challenger.sample_algebra_element();
        let nu: Ext = challenger.sample_algebra_element();
        let degree = degree(sources.iter().map(|source| source.parts() - 1));
        let weights: Vec<Ext> = std::iter::successors(Some(Ext::ONE), |&w| Some(w * nu))
            .take(sources.len())
            .collect();
        let mut eq = eq_table(&point);
        let (mut rounds, mut next) = (Vec::new(), Vec::new());
        let streamed = eq.len() > 1
            && sources
                .iter()
                .any(|source| !matches!(source, Source::Stored(_)));
        let mut tables: Vec<Tree> = if streamed {
            let (message, r, tables) = first_round(&sources, &eq, &weights, mu, degree, challenger);
            drop(sources);
            fold_low(&mut eq, r);
            rounds.push(message);
            next.push(r);
            tables
        } else {
            sources.into_iter().map(Source::into_tree).collect()
        };
        let (rest, rest_point, stride) = prove_layer(&mut tables, eq, &weights, mu, degree, challenger, accelerator)?;
        rounds.extend(rest);
        next.extend(rest_point);
        let children: Vec<Vec<Ext>> = tables
            .iter()
            .map(|tree| {
                tree.parts()
                    .flat_map(|part| [part[0], part[stride]])
                    .collect()
            })
            .collect();
        for values in &children {
            challenger.observe_algebra_slice(values);
        }
        let tau: Ext = challenger.sample_algebra_element();
        next.push(tau);
        for (&i, values) in active.iter().zip(&children) {
            if depths[i] == k {
                claims[i] = Some(TreeClaim {
                    root: roots[i],
                    point: next.clone(),
                    values: values
                        .chunks(2)
                        .map(|pair| pair[0] + tau * (pair[1] - pair[0]))
                        .collect(),
                });
            }
        }
        point = next;
        layers.push(Layer { rounds, children });
    }
    let claims = claims
        .into_iter()
        .map(|claim| claim.expect("every tree ends"))
        .collect();
    Ok((GkrProof { roots, layers }, claims))
}

/// The rounds of one layer: the messages, the challenges, and where the
/// folded tables keep their entries (logical entry `j` at element
/// `j·stride`). A device folds in place at a doubling stride; the CPU
/// compacts.
fn prove_layer(
    tables: &mut [Tree],
    mut eq: Vec<Ext>,
    weights: &[Ext],
    mu: Ext,
    degree: usize,
    challenger: &mut Challenger,
    accelerator: &Accelerator,
) -> Result<(Vec<Vec<Ext>>, Vec<Ext>, usize), Error> {
    // The nodes are 0, 2, ..., degree: `degree` of them.
    assert!(degree <= NODES);
    let count = eq.len().trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(count);
    let mut point = Vec::with_capacity(count + 1);
    if count > 0 {
        let words: Vec<Vec<&mut [u64]>> = tables
            .iter_mut()
            .map(|tree| tree.parts_mut().map(|part| ext_words_mut(part)).collect())
            .collect();
        let layer = accelerator
            .device()
            .and_then(|device| device.gkr_layer(words, ext_words_mut(&mut eq)));
        if let Some(mut layer) = layer {
            let weights: Vec<[u64; 3]> = weights.iter().map(|&w| words_of_ext(w)).collect();
            for _ in 0..count {
                let message: Vec<Ext> = layer
                    .message(&weights, words_of_ext(mu), degree)
                    .map_err(Error::Device)?
                    .into_iter()
                    .map(ext_of_words)
                    .collect();
                let r = sumcheck::send(&message, challenger);
                layer.fold(words_of_ext(r)).map_err(Error::Device)?;
                rounds.push(message);
                point.push(r);
            }
            return Ok((rounds, point, 1 << count));
        }
    }
    while eq.len() > 1 {
        let message = host_message(tables, &eq, weights, mu, degree);
        let r = sumcheck::send(&message, challenger);
        fold_low(&mut eq, r);
        for tree in tables.iter_mut() {
            for table in tree.parts_mut() {
                fold_low(table, r);
            }
        }
        rounds.push(message);
        point.push(r);
    }
    Ok((rounds, point, 1))
}

fn words_of_ext(value: Ext) -> [u64; 3] {
    ext_words(value).map(|word| word.as_canonical_u64())
}

fn ext_of_words(words: [u64; 3]) -> Ext {
    Ext::from_basis_coefficients_fn(|c| Gl::new(words[c]))
}

/// The base words of `values`, three per element.
fn table_words(values: &[Ext]) -> &[u64] {
    // SAFETY: `Ext` is `repr(transparent)` over `[Goldilocks; 3]` and
    // `Goldilocks` over `u64`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u64>(), 3 * values.len()) }
}

/// The base words of `values`, three per element.
fn ext_words_mut(values: &mut [Ext]) -> &mut [u64] {
    // SAFETY: `Ext` is `repr(transparent)` over `[Goldilocks; 3]` and
    // `Goldilocks` over `u64`; a device writes canonical words.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u64>(), 3 * values.len()) }
}

/// A `GkrProof` as backend words, or zeros of its shape.
pub(crate) struct GkrView<B: Backend> {
    roots: Vec<[B::E; 2]>,
    /// Per layer: the round messages and each active tree's children.
    layers: Vec<(Vec<Vec<B::E>>, Vec<Vec<B::E>>)>,
}

/// Factors of each active tree at layer `k`: one inside a tree, the tree's
/// own count at its leaf layer.
fn layer_factors(shapes: &[TreeShape], k: usize) -> Vec<usize> {
    shapes
        .iter()
        .filter(|shape| shape.depth >= k)
        .map(|shape| if shape.depth == k { shape.factors } else { 1 })
        .collect()
}

impl<B: Backend> GkrView<B> {
    pub(crate) fn read(b: &mut B, proof: Option<&GkrProof>, shapes: &[TreeShape]) -> Result<Self, Error> {
        let layer_count = shapes.iter().map(|shape| shape.depth).max().unwrap_or(0);
        if shapes
            .iter()
            .any(|shape| shape.depth == 0 || shape.factors > 2)
            || proof.is_some_and(|p| p.roots.len() != shapes.len() || p.layers.len() != layer_count)
        {
            return Err(Error::Rejected("GKR shape"));
        }
        let word = |b: &mut B, value: Option<Ext>| algebra::private_ext(b, value.unwrap_or(Ext::ZERO));
        let roots = (0..shapes.len())
            .map(|i| std::array::from_fn(|side| word(b, proof.map(|p| p.roots[i][side]))))
            .collect();
        let mut layers = Vec::with_capacity(layer_count);
        for k in 1..=layer_count {
            let factors = layer_factors(shapes, k);
            let degree = degree(factors.iter().copied());
            let layer = proof.map(|p| &p.layers[k - 1]);
            if layer.is_some_and(|l| {
                l.rounds.len() != k - 1
                    || l.rounds.iter().any(|m| m.len() != degree)
                    || l.children.len() != factors.len()
                    || l.children
                        .iter()
                        .zip(&factors)
                        .any(|(c, f)| c.len() != 2 * (f + 1))
            }) {
                return Err(Error::Rejected("GKR layer shape"));
            }
            let rounds = (0..k - 1)
                .map(|r| {
                    (0..degree)
                        .map(|j| word(b, layer.map(|l| l.rounds[r][j])))
                        .collect()
                })
                .collect();
            let children = factors
                .iter()
                .enumerate()
                .map(|(t, f)| {
                    (0..2 * (f + 1))
                        .map(|j| word(b, layer.map(|l| l.children[t][j])))
                        .collect()
                })
                .collect();
            layers.push((rounds, children));
        }
        Ok(Self { roots, layers })
    }
}

/// `p0·q1 + p1·q0 + μ·q0·q1` from one tree's child values: each part's pair.
fn combine<B: Backend>(b: &mut B, children: &[B::E], mu: B::E) -> B::E {
    let factors = children.len() / 2 - 1;
    let p = |b: &mut B, side: usize| -> B::E {
        let mut total = algebra::ext_one(b);
        for j in 0..factors {
            total = b.ext_mul(total, children[2 * j + side]);
        }
        total
    };
    let (p0, p1) = (p(b, 0), p(b, 1));
    let (q0, q1) = (children[2 * factors], children[2 * factors + 1]);
    let a = b.ext_mul(p0, q1);
    let c = b.ext_mul(p1, q0);
    let qq = b.ext_mul(q0, q1);
    let d = b.ext_mul(mu, qq);
    let sum = b.ext_add(a, c);
    b.ext_add(sum, d)
}

/// Verify `view` for trees of `shapes`; return one claim per tree. The
/// caller checks the roots and the leaf values.
pub(crate) fn verify<B: Backend>(
    b: &mut B,
    view: &GkrView<B>,
    shapes: &[TreeShape],
    duplex: &mut Duplex<B>,
) -> Result<Vec<TreeClaim<B::E>>, Error> {
    for root in &view.roots {
        duplex.observe_ext(b, root[0]);
        duplex.observe_ext(b, root[1]);
    }
    let mut current: Vec<[B::E; 2]> = view.roots.clone();
    let mut claims: Vec<Option<TreeClaim<B::E>>> = (0..shapes.len()).map(|_| None).collect();
    let mut point: Vec<B::E> = Vec::new();
    for (index, (rounds, children)) in view.layers.iter().enumerate() {
        let k = index + 1;
        let active: Vec<usize> = (0..shapes.len())
            .filter(|&i| shapes[i].depth >= k)
            .collect();
        let mu = duplex.sample_ext(b);
        let nu = duplex.sample_ext(b);
        let mut claim = algebra::ext_zero(b);
        let mut expected = algebra::ext_zero(b);
        let mut weight = algebra::ext_one(b);
        for (&i, values) in active.iter().zip(children) {
            let [p, q] = current[i];
            let mq = b.ext_mul(mu, q);
            let pq = b.ext_add(p, mq);
            let term = b.ext_mul(weight, pq);
            claim = b.ext_add(claim, term);
            let combined = combine(b, values, mu);
            let term = b.ext_mul(weight, combined);
            expected = b.ext_add(expected, term);
            weight = b.ext_mul(weight, nu);
        }
        let (mut next, last) = sumcheck::replay(b, duplex, rounds, claim);
        let eq = algebra::eq_eval(b, &point, &next);
        let target = b.ext_mul(eq, expected);
        b.assert_ext_equal(last, target, "GKR layer")?;
        for values in children {
            for &value in values {
                duplex.observe_ext(b, value);
            }
        }
        let tau = duplex.sample_ext(b);
        next.push(tau);
        for (&i, values) in active.iter().zip(children) {
            let reduced: Vec<B::E> = values
                .chunks(2)
                .map(|pair| {
                    let difference = b.ext_sub(pair[1], pair[0]);
                    let step = b.ext_mul(tau, difference);
                    b.ext_add(pair[0], step)
                })
                .collect();
            if shapes[i].depth == k {
                claims[i] = Some(TreeClaim {
                    root: view.roots[i],
                    point: next.clone(),
                    values: reduced,
                });
            } else {
                current[i] = [reduced[0], reduced[1]];
            }
        }
        point = next;
    }
    Ok(claims
        .into_iter()
        .map(|claim| claim.expect("every tree ends"))
        .collect())
}

/// The batched sum-check of `Σ_y eq(point, y)·Σ_t ν^t·(P0·Q1 + P1·Q0 + μ·Q0·Q1)`
/// over the trees' layer tables, whose halves are the two children. Returns
/// the rounds and the point; the tables end with length 2.
/// The most round nodes: `0, 2, 3, 4` at degree 4, two factors at a leaf.
const NODES: usize = 4;
/// The most parts of a tree: two factors and the denominator.
const PARTS: usize = 3;

/// Move the lines in `value` from node `j - 1` to node `j` of `0, 2, 3, ...`.
fn step(value: &mut [[Ext; 2]], slope: &[[Ext; 2]], j: usize) {
    let times = match j {
        0 => return,
        1 => 2,
        _ => 1,
    };
    for (value, slope) in value.iter_mut().zip(slope) {
        for b in 0..2 {
            for _ in 0..times {
                value[b] += slope[b];
            }
        }
    }
}

/// The round message at nodes `0, 2, ..., degree`, on the CPU.
fn host_message(tables: &[Tree], eq: &[Ext], weights: &[Ext], mu: Ext, degree: usize) -> Vec<Ext> {
    let sums = (0..eq.len() / 2)
        .into_par_iter()
        .fold(
            || [Ext::ZERO; NODES],
            |mut sums, m| {
                let mut totals = [Ext::ZERO; NODES];
                for (tree, &weight) in tables.iter().zip(weights) {
                    let half = tree.denominator.len() / 2;
                    let f = tree.factors.len();
                    // Each part's line through its two children, walked
                    // over the nodes 0, 2, 3, ... by adding its slope.
                    let mut value = [[Ext::ZERO; 2]; PARTS];
                    let mut slope = [[Ext::ZERO; 2]; PARTS];
                    for (k, table) in tree.parts().enumerate() {
                        for b in 0..2 {
                            let i = b * half + 2 * m;
                            value[k][b] = table[i];
                            slope[k][b] = table[i + 1] - table[i];
                        }
                    }
                    for (j, total) in totals.iter_mut().enumerate().take(degree) {
                        step(&mut value[..=f], &slope[..=f], j);
                        let [q0, q1] = value[f];
                        // p_b · q_{1-b} summed; a tree without factors has p = 1.
                        let cross = match &value[..f] {
                            [] => q0 + q1,
                            [a] => a[0] * q1 + a[1] * q0,
                            [a, c] => a[0] * c[0] * q1 + a[1] * c[1] * q0,
                            _ => unreachable!("a tree has at most two factors"),
                        };
                        *total += weight * (cross + mu * q0 * q1);
                    }
                }
                let mut e = [[eq[2 * m]; 2]];
                let de = [[eq[2 * m + 1] - eq[2 * m]; 2]];
                for (j, (sum, total)) in sums.iter_mut().zip(totals).enumerate().take(degree) {
                    step(&mut e, &de, j);
                    *sum += e[0][0] * total;
                }
                sums
            },
        )
        .reduce(|| [Ext::ZERO; NODES], |a, b| std::array::from_fn(|j| a[j] + b[j]));
    sums[..degree].to_vec()
}
