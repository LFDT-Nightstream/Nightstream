//! The norm conjunct's fraction sum `Σ_x 1 / (β - z(x))`, proven by GKR
//! (logUp-GKR, Papini and Haböck, ePrint 2023/1284).
//!
//! Owns: the layered fraction tree over the committed cube, its prover and
//! its verifier. Layer `k` has `2^k` nodes indexed by the low `k` bits of a
//! leaf index; a node of layer `k - 1` combines the two layer-`k` nodes that
//! differ in bit `k - 1`. Leaves are `(p, q) = (1, β - z(x))`, so the leaf
//! claim is a claim on `z` at one point.

use p3_challenger_v08::FieldChallenger;
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::{eq_eval, eq_table, fold_low, Ext, Gl};
use crate::hash::Challenger;
use crate::sumcheck;
use crate::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct GkrProof {
    /// `(P_0, Q_0)`: the root fraction.
    pub(crate) root: [Ext; 2],
    /// One entry per layer `k = 1..=variables`.
    pub(crate) layers: Vec<Layer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Layer {
    /// `k - 1` degree-3 round messages (none for `k = 1`).
    pub(crate) rounds: Vec<Vec<Ext>>,
    /// `P_k(ρ', 0), P_k(ρ', 1), Q_k(ρ', 0), Q_k(ρ', 1)`; the leaf layer sends
    /// only the two `Q` values because every leaf `P` is one.
    pub(crate) children: Vec<Ext>,
}

/// The leaf claim: `q~(point) = β - z~(point)`.
pub(crate) struct LeafClaim {
    pub(crate) point: Vec<Ext>,
    pub(crate) q: Ext,
}

/// Prove the fraction sum of the leaves `(1, β - z(x))`; return the proof and
/// the leaf point.
pub(crate) fn prove(z: &[Gl], beta: Ext, challenger: &mut Challenger) -> (GkrProof, Vec<Ext>) {
    let variables = z.len().trailing_zeros() as usize;
    assert_eq!(z.len(), 1 << variables);
    let leaf = |x: usize| beta - z[x];

    // tree[k] = (P_k, Q_k) for k < variables; the leaves are computed from z.
    let mut tree: Vec<(Vec<Ext>, Vec<Ext>)> = Vec::with_capacity(variables);
    let half = z.len() / 2;
    let (p, q): (Vec<Ext>, Vec<Ext>) = (0..half)
        .into_par_iter()
        .map(|y| {
            let (q0, q1) = (leaf(y), leaf(y + half));
            (q0 + q1, q0 * q1)
        })
        .unzip();
    tree.push((p, q));
    while tree.last().expect("nonempty").0.len() > 1 {
        let (p, q) = tree.last().expect("nonempty");
        let half = p.len() / 2;
        let next = (0..half)
            .into_par_iter()
            .map(|y| {
                let (p0, p1, q0, q1) = (p[y], p[y + half], q[y], q[y + half]);
                (p0 * q1 + p1 * q0, q0 * q1)
            })
            .unzip();
        tree.push(next);
    }
    tree.reverse();

    let root = [tree[0].0[0], tree[0].1[0]];
    challenger.observe_algebra_slice(&root);
    let mut point = Vec::new();
    let mut layers = Vec::with_capacity(variables);
    for k in 1..=variables {
        let half = 1 << (k - 1);
        let (p0, p1, q0, q1) = if k == variables {
            let q0 = (0..half).into_par_iter().map(leaf).collect();
            let q1 = (0..half).into_par_iter().map(|y| leaf(y + half)).collect();
            (None, None, q0, q1)
        } else {
            let (p, q) = &tree[k];
            (
                Some(p[..half].to_vec()),
                Some(p[half..].to_vec()),
                q[..half].to_vec(),
                q[half..].to_vec(),
            )
        };
        let (rounds, next, children) = if k == 1 {
            let children = layer_children(&p0, &p1, &q0, &q1);
            (Vec::new(), Vec::new(), children)
        } else {
            let mu: Ext = challenger.sample_algebra_element();
            prove_layer(eq_table(&point), p0, p1, q0, q1, mu, challenger)
        };
        challenger.observe_algebra_slice(&children);
        let tau: Ext = challenger.sample_algebra_element();
        point = next;
        point.push(tau);
        layers.push(Layer { rounds, children });
    }
    (GkrProof { root, layers }, point)
}

/// Verify the tree whose root must equal `table_sum`; return the leaf claim.
pub(crate) fn verify(
    proof: &GkrProof,
    variables: usize,
    table_sum: Ext,
    challenger: &mut Challenger,
) -> Result<LeafClaim, Error> {
    if proof.layers.len() != variables {
        return Err(Error::Rejected("GKR layer count"));
    }
    let [mut p, mut q] = proof.root;
    challenger.observe_algebra_slice(&proof.root);
    if q == Ext::ZERO || p != q * table_sum {
        return Err(Error::Rejected("GKR root against the histogram"));
    }
    let mut point = Vec::new();
    for (index, layer) in proof.layers.iter().enumerate() {
        let k = index + 1;
        let leaf = k == variables;
        let [p0, p1, q0, q1] = match (leaf, layer.children.as_slice()) {
            (true, &[q0, q1]) => [Ext::ONE, Ext::ONE, q0, q1],
            (false, &[p0, p1, q0, q1]) => [p0, p1, q0, q1],
            _ => return Err(Error::Rejected("GKR children")),
        };
        let next = if k == 1 {
            if !layer.rounds.is_empty() || p != p0 * q1 + p1 * q0 || q != q0 * q1 {
                return Err(Error::Rejected("GKR first layer"));
            }
            Vec::new()
        } else {
            let mu: Ext = challenger.sample_algebra_element();
            if layer.rounds.len() != k - 1 {
                return Err(Error::Rejected("GKR round count"));
            }
            let (next, last) = sumcheck::verify(&layer.rounds, 3, p + mu * q, challenger)?;
            if last != eq_eval(&point, &next) * (p0 * q1 + p1 * q0 + mu * q0 * q1) {
                return Err(Error::Rejected("GKR layer"));
            }
            next
        };
        challenger.observe_algebra_slice(&layer.children);
        let tau: Ext = challenger.sample_algebra_element();
        p = p0 + tau * (p1 - p0);
        q = q0 + tau * (q1 - q0);
        point = next;
        point.push(tau);
    }
    Ok(LeafClaim { point, q })
}

fn layer_children(p0: &Option<Vec<Ext>>, p1: &Option<Vec<Ext>>, q0: &[Ext], q1: &[Ext]) -> Vec<Ext> {
    match (p0, p1) {
        (Some(p0), Some(p1)) => vec![p0[0], p1[0], q0[0], q1[0]],
        _ => vec![q0[0], q1[0]],
    }
}

/// Sum-check of `Σ_y eq(ρ, y)·(P0·Q1 + P1·Q0 + μ·Q0·Q1)`; absent `P` tables
/// are the all-ones leaf numerators.
fn prove_layer(
    mut eq: Vec<Ext>,
    mut p0: Option<Vec<Ext>>,
    mut p1: Option<Vec<Ext>>,
    mut q0: Vec<Ext>,
    mut q1: Vec<Ext>,
    mu: Ext,
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>, Vec<Ext>) {
    let mut rounds = Vec::new();
    let mut point = Vec::new();
    while eq.len() > 1 {
        let at = |table: &[Ext], m: usize, x: Ext| table[2 * m] + x * (table[2 * m + 1] - table[2 * m]);
        let nodes = [Ext::ZERO, Ext::TWO, Ext::from(Gl::from_u8(3))];
        let message = (0..eq.len() / 2)
            .into_par_iter()
            .map(|m| {
                nodes.map(|x| {
                    let (a0, a1) = match (&p0, &p1) {
                        (Some(p0), Some(p1)) => (at(p0, m, x), at(p1, m, x)),
                        _ => (Ext::ONE, Ext::ONE),
                    };
                    let (b0, b1) = (at(&q0, m, x), at(&q1, m, x));
                    at(&eq, m, x) * (a0 * b1 + a1 * b0 + mu * b0 * b1)
                })
            })
            .reduce(|| [Ext::ZERO; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
            .to_vec();
        let r = sumcheck::send(&message, challenger);
        fold_low(&mut eq, r);
        for table in [&mut p0, &mut p1].into_iter().flatten() {
            fold_low(table, r);
        }
        fold_low(&mut q0, r);
        fold_low(&mut q1, r);
        rounds.push(message);
        point.push(r);
    }
    let children = layer_children(&p0, &p1, &q0, &q1);
    (rounds, point, children)
}
