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
//! The prover takes the trees one at a time and keeps each level in a scratch
//! file from the moment it is built (bottom up) until its layer's sum-check
//! (top down). A layer's first round streams from the file, so the prover
//! holds one tree while building and half a layer while proving.

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

/// The leaves of one tree: numerator factors (none means all ones) and the
/// denominator, each of length `2^depth`.
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

    fn numerator(&self, x: usize) -> Ext {
        self.factors.iter().map(|factor| factor[x]).product()
    }

    /// The layer above: one factor `P` and the denominator `Q`.
    fn parent(&self) -> Tree {
        let half = self.denominator.len() / 2;
        let (p, q) = (0..half)
            .into_par_iter()
            .map(|y| {
                let (p0, p1) = (self.numerator(y), self.numerator(y + half));
                let (q0, q1) = (self.denominator[y], self.denominator[y + half]);
                (p0 * q1 + p1 * q0, q0 * q1)
            })
            .unzip();
        Tree {
            factors: vec![p],
            denominator: q,
        }
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

/// One tree layer in the scratch file: its parts, one after another.
struct Spilled {
    offset: usize,
    parts: usize,
    nodes: usize,
}

fn scratch_error(error: std::io::Error) -> Error {
    Error::Device(format!("GKR scratch file: {error}"))
}

/// Append `tree`'s parts at word `end`.
fn spill(file: &std::fs::File, end: &mut usize, tree: &Tree) -> Result<Spilled, Error> {
    let spilled = Spilled {
        offset: *end,
        parts: tree.factors.len() + 1,
        nodes: tree.denominator.len(),
    };
    for part in tree.parts() {
        scratch::write(file, *end, table_words(part)).map_err(scratch_error)?;
        *end += 3 * part.len();
    }
    Ok(spilled)
}

fn restore(file: &std::fs::File, spilled: &Spilled) -> Result<Tree, Error> {
    let mut parts = (0..spilled.parts)
        .map(|part| {
            let mut values = vec![Ext::ZERO; spilled.nodes];
            let offset = spilled.offset + 3 * spilled.nodes * part;
            scratch::read(file, offset, ext_words_mut(&mut values)).map_err(scratch_error)?;
            Ok(values)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let denominator = parts.pop().expect("a tree has a denominator");
    Ok(Tree {
        factors: parts,
        denominator,
    })
}

/// Pairs of a streamed first round read at once; it bounds only the buffers.
const CHUNK: usize = scratch::BLOCK_WORDS / 4;

/// Round 1 of a layer whose tables are in the scratch file: the message from
/// one pass, then the tables folded at the round's challenge from a second,
/// so the layer is in memory only at half its size.
fn first_round(
    file: &std::fs::File,
    spilled: &[&Spilled],
    eq: &[Ext],
    weights: &[Ext],
    mu: Ext,
    degree: usize,
    challenger: &mut Challenger,
) -> Result<(Vec<Ext>, Ext, Vec<Tree>), Error> {
    let nodes = spilled[0].nodes;
    let (half, quarter) = (nodes / 2, nodes / 4);
    // Pairs `m0..m1` of every tree: both children's entries of every part.
    let chunk = |m0: usize, m1: usize| -> Result<Vec<Tree>, Error> {
        spilled
            .iter()
            .map(|level| {
                let mut parts = (0..level.parts)
                    .map(|part| {
                        let mut values = vec![Ext::ZERO; 4 * (m1 - m0)];
                        let (low, high) = values.split_at_mut(2 * (m1 - m0));
                        for (child, out) in [low, high].into_iter().enumerate() {
                            let offset = level.offset + 3 * (level.nodes * part + child * half + 2 * m0);
                            scratch::read(file, offset, ext_words_mut(out)).map_err(scratch_error)?;
                        }
                        Ok(values)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let denominator = parts.pop().expect("a tree has a denominator");
                Ok(Tree {
                    factors: parts,
                    denominator,
                })
            })
            .collect()
    };
    let mut message = vec![Ext::ZERO; degree];
    for m0 in (0..quarter).step_by(CHUNK) {
        let m1 = (m0 + CHUNK).min(quarter);
        let part = host_message(&chunk(m0, m1)?, &eq[2 * m0..2 * m1], weights, mu, degree);
        for (sum, value) in message.iter_mut().zip(part) {
            *sum += value;
        }
    }
    let r = sumcheck::send(&message, challenger);
    let mut folded: Vec<Tree> = spilled
        .iter()
        .map(|level| Tree {
            factors: vec![vec![Ext::ZERO; half]; level.parts - 1],
            denominator: vec![Ext::ZERO; half],
        })
        .collect();
    for m0 in (0..quarter).step_by(CHUNK) {
        let m1 = (m0 + CHUNK).min(quarter);
        for (tree, out) in chunk(m0, m1)?.iter_mut().zip(&mut folded) {
            for (table, target) in tree.parts_mut().zip(out.parts_mut()) {
                fold_low(table, r);
                let count = m1 - m0;
                target[m0..m1].copy_from_slice(&table[..count]);
                target[quarter + m0..quarter + m1].copy_from_slice(&table[count..]);
            }
        }
    }
    Ok((message, r, folded))
}

/// Prove the fraction sums of `trees`; return the proof and one claim per
/// tree. A layer's rounds after the first run on the accelerator's device
/// when it takes them.
pub(crate) fn prove(
    trees: impl IntoIterator<Item = Tree>,
    challenger: &mut Challenger,
    accelerator: &Accelerator,
) -> Result<(GkrProof, Vec<TreeClaim<Ext>>), Error> {
    let file = scratch::file().map_err(scratch_error)?;
    let mut end = 0;
    let mut depths = Vec::new();
    let mut roots: Vec<[Ext; 2]> = Vec::new();
    // levels[i][k - 1] is tree i at layer k; its last entry is its leaves.
    let mut levels: Vec<Vec<Spilled>> = Vec::new();
    for leaves in trees {
        depths.push(leaves.depth());
        let mut spilled = Vec::new();
        let mut level = leaves;
        while level.denominator.len() > 1 {
            let parent = level.parent();
            spilled.push(spill(&file, &mut end, &level)?);
            level = parent;
        }
        spilled.reverse();
        roots.push([level.factors[0][0], level.denominator[0]]);
        levels.push(spilled);
    }
    for root in &roots {
        challenger.observe_algebra_slice(root);
    }

    let mut claims: Vec<Option<TreeClaim<Ext>>> = vec![None; depths.len()];
    let mut point = Vec::new();
    let mut layers = Vec::new();
    for k in 1..=depths.iter().copied().max().unwrap_or(0) {
        let active: Vec<usize> = (0..depths.len()).filter(|&i| depths[i] >= k).collect();
        let spilled: Vec<&Spilled> = active.iter().map(|&i| &levels[i][k - 1]).collect();
        let mu: Ext = challenger.sample_algebra_element();
        let nu: Ext = challenger.sample_algebra_element();
        let degree = degree(spilled.iter().map(|level| level.parts - 1));
        let weights: Vec<Ext> = std::iter::successors(Some(Ext::ONE), |&w| Some(w * nu))
            .take(spilled.len())
            .collect();
        let mut eq = eq_table(&point);
        let (mut rounds, mut next) = (Vec::new(), Vec::new());
        let mut tables: Vec<Tree> = if eq.len() > 1 {
            let (message, r, tables) = first_round(&file, &spilled, &eq, &weights, mu, degree, challenger)?;
            fold_low(&mut eq, r);
            rounds.push(message);
            next.push(r);
            tables
        } else {
            spilled
                .iter()
                .map(|level| restore(&file, level))
                .collect::<Result<_, _>>()?
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
