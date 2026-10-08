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

use p3_challenger_v08::FieldChallenger;
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::circuit::algebra;
use crate::circuit::hash::Duplex;
use crate::circuit::Backend;
use crate::field::{eq_table, fold_low, Ext, Gl};
use crate::hash::Challenger;
use crate::sumcheck;
use crate::Error;

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

/// Prove the fraction sums of `trees`; return the proof and one claim per tree.
pub(crate) fn prove(trees: Vec<Tree>, challenger: &mut Challenger) -> (GkrProof, Vec<TreeClaim<Ext>>) {
    let depths: Vec<usize> = trees.iter().map(Tree::depth).collect();
    // levels[i][k] is tree i at layer k; levels[i][depth] is its leaves.
    let mut levels: Vec<Vec<Tree>> = trees
        .into_iter()
        .map(|leaves| {
            let mut levels = vec![leaves];
            while levels.last().expect("nonempty").denominator.len() > 1 {
                let parent = levels.last().expect("nonempty").parent();
                levels.push(parent);
            }
            levels.reverse();
            levels
        })
        .collect();
    let roots: Vec<[Ext; 2]> = levels
        .iter()
        .map(|levels| [levels[0].factors[0][0], levels[0].denominator[0]])
        .collect();
    for root in &roots {
        challenger.observe_algebra_slice(root);
    }

    let mut claims: Vec<Option<TreeClaim<Ext>>> = vec![None; depths.len()];
    let mut point = Vec::new();
    let mut layers = Vec::new();
    for k in 1..=depths.iter().copied().max().unwrap_or(0) {
        let active: Vec<usize> = (0..depths.len()).filter(|&i| depths[i] >= k).collect();
        let mut tables: Vec<Tree> = active
            .iter()
            .map(|&i| std::mem::take(&mut levels[i][k]))
            .collect();
        let mu: Ext = challenger.sample_algebra_element();
        let nu: Ext = challenger.sample_algebra_element();
        let degree = degree(tables.iter().map(|tree| tree.factors.len()));
        let (rounds, mut next) = prove_layer(eq_table(&point), &mut tables, mu, nu, degree, challenger);
        let children: Vec<Vec<Ext>> = tables
            .iter()
            .map(|tree| tree.parts().flat_map(|part| [part[0], part[1]]).collect())
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
    (GkrProof { roots, layers }, claims)
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
fn prove_layer(
    mut eq: Vec<Ext>,
    tables: &mut [Tree],
    mu: Ext,
    nu: Ext,
    degree: usize,
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>) {
    let weights: Vec<Ext> = std::iter::successors(Some(Ext::ONE), |&w| Some(w * nu))
        .take(tables.len())
        .collect();
    let nodes: Vec<Ext> = std::iter::once(0)
        .chain(2..=degree)
        .map(|x| Ext::from(Gl::from_usize(x)))
        .collect();
    let mut rounds = Vec::new();
    let mut point = Vec::new();
    while eq.len() > 1 {
        let tables_ref = &*tables;
        let message = (0..eq.len() / 2)
            .into_par_iter()
            .map(|m| {
                let at = |table: &[Ext], base: usize, x: Ext| table[base] + x * (table[base + 1] - table[base]);
                nodes
                    .iter()
                    .map(|&x| {
                        let mut total = Ext::ZERO;
                        for (tree, &weight) in tables_ref.iter().zip(&weights) {
                            let half = tree.denominator.len() / 2;
                            let child = |table: &[Ext], b: usize| at(table, b * half + 2 * m, x);
                            let p = |b: usize| tree.factors.iter().map(|f| child(f, b)).product::<Ext>();
                            let (q0, q1) = (child(&tree.denominator, 0), child(&tree.denominator, 1));
                            total += weight * (p(0) * q1 + p(1) * q0 + mu * q0 * q1);
                        }
                        at(&eq, 2 * m, x) * total
                    })
                    .collect::<Vec<Ext>>()
            })
            .reduce(
                || vec![Ext::ZERO; nodes.len()],
                |a, b| a.iter().zip(&b).map(|(&x, &y)| x + y).collect(),
            );
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
    (rounds, point)
}
