//! The batched fraction-sum GKR and the histogram side of logUp.

use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing};

use super::{fresh, gkr_verify, word};
use crate::circuit::Fast;
use crate::field::{signed, Ext, Gl};
use crate::gkr::{prove, GkrProof, Leaves, Tree, TreeClaim, TreeShape};
use crate::norm::{histogram, words};
use crate::sumcheck::evaluate;

const BOUND: u32 = 5;

fn witness(variables: usize) -> Vec<Gl> {
    (0..1u64 << variables)
        .map(|i| signed((word(11, i) % (2 * BOUND as u64 + 1)) as i64 - BOUND as i64))
        .collect()
}

fn ext(seed: u64, index: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, 4 * index + i as u64)))
}

fn beta() -> Ext {
    ext(12, 0)
}

fn norm_tree(z: &[Gl], beta: Ext) -> Tree {
    Tree {
        factors: Vec::new(),
        denominator: z.iter().map(|&value| beta - value).collect(),
    }
}

fn run_verify(proof: &GkrProof, shapes: &[TreeShape]) -> Option<Vec<TreeClaim<Ext>>> {
    gkr_verify(proof, shapes).ok()
}

fn table_sum(counts: &[u32], beta: Ext) -> Ext {
    crate::norm::table_sum(&mut Fast, &words(counts), BOUND, beta).unwrap()
}

#[test]
fn leaf_claim_matches_the_witness_and_the_root_matches_the_histogram() {
    for variables in [1, 2, 7] {
        let z = witness(variables);
        let beta = beta();
        let (proof, prover) = prove(
            &[norm_tree(&z, beta).leaves()],
            &mut fresh(),
            &crate::device::Accelerator::cpu(),
        )
        .unwrap();
        let total = table_sum(&histogram(&z, BOUND).unwrap(), beta);
        let direct: Ext = z.iter().map(|&value| (beta - value).inverse()).sum();
        assert_eq!(total, direct);

        let shape = TreeShape {
            depth: variables,
            factors: 0,
        };
        let claims = run_verify(&proof, &[shape]).unwrap();
        assert_eq!(claims, prover);
        let [p, q] = claims[0].root;
        assert_eq!(p, q * total);
        let z_ext: Vec<Ext> = z.iter().map(|&value| Ext::from(value)).collect();
        assert_eq!(claims[0].values, vec![beta - evaluate(&z_ext, &claims[0].point)]);
    }
}

#[test]
fn forged_histograms_reject() {
    let z = witness(7);
    let beta = beta();
    let (proof, _) = prove(
        &[norm_tree(&z, beta).leaves()],
        &mut fresh(),
        &crate::device::Accelerator::cpu(),
    )
    .unwrap();
    let claims = run_verify(&proof, &[TreeShape { depth: 7, factors: 0 }]).unwrap();
    let [p, q] = claims[0].root;
    let counts = histogram(&z, BOUND).unwrap();
    assert_eq!(p, q * table_sum(&counts, beta));
    // One witness value claimed as another table value.
    let mut forged = counts.clone();
    let from = forged.iter().position(|&count| count > 0).unwrap();
    forged[from] -= 1;
    let next = (from + 1) % forged.len();
    forged[next] += 1;
    assert_ne!(p, q * table_sum(&forged, beta));
}

/// Depths 8, 8, 6, 5 with 0, 2, 1, 2 numerator factors.
fn batch() -> (Vec<TreeShape>, Vec<Vec<Vec<Ext>>>) {
    let shapes = vec![
        TreeShape { depth: 8, factors: 0 },
        TreeShape { depth: 8, factors: 2 },
        TreeShape { depth: 6, factors: 1 },
        TreeShape { depth: 5, factors: 2 },
    ];
    let parts = parts_of(&shapes);
    (shapes, parts)
}

/// Every part of every tree: pseudorandom extension values.
fn parts_of(shapes: &[TreeShape]) -> Vec<Vec<Vec<Ext>>> {
    shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            (0..=shape.factors)
                .map(|j| {
                    (0..1u64 << shape.depth)
                        .map(|x| ext(100 + 10 * i as u64 + j as u64, x))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn trees(parts: &[Vec<Vec<Ext>>]) -> Vec<Tree> {
    parts
        .iter()
        .map(|parts| {
            let (denominator, factors) = parts.split_last().unwrap();
            Tree {
                factors: factors.to_vec(),
                denominator: denominator.clone(),
            }
        })
        .collect()
}

fn leaves(trees: &[Tree]) -> Vec<Leaves<'_>> {
    trees.iter().map(Tree::leaves).collect()
}

/// The verified claims are the prover's, each root is its tree's direct
/// fraction sum, and each leaf claim is the parts' values at its point.
fn assert_direct(shapes: &[TreeShape], parts: &[Vec<Vec<Ext>>]) -> Vec<TreeClaim<Ext>> {
    let (proof, prover) = prove(&leaves(&trees(parts)), &mut fresh(), &crate::device::Accelerator::cpu()).unwrap();
    let claims = run_verify(&proof, shapes).unwrap();
    assert_eq!(claims, prover);
    for (claim, parts) in claims.iter().zip(parts) {
        let (denominator, factors) = parts.split_last().unwrap();
        let [p, q] = claim.root;
        let direct: Ext = (0..denominator.len())
            .map(|x| factors.iter().map(|f| f[x]).product::<Ext>() * denominator[x].inverse())
            .sum();
        assert_eq!(p, q * direct);
        let values: Vec<Ext> = parts
            .iter()
            .map(|part| evaluate(part, &claim.point))
            .collect();
        assert_eq!(claim.values, values);
    }
    claims
}

#[test]
fn batched_trees_match_direct_sums_and_leaf_values() {
    let (shapes, parts) = batch();
    let claims = assert_direct(&shapes, &parts);
    let proof = prove(
        &leaves(&trees(&parts)),
        &mut fresh(),
        &crate::device::Accelerator::cpu(),
    )
    .unwrap()
    .0;
    assert_eq!(claims[0].point, claims[1].point);
    assert_ne!(claims[2].point[..5], claims[3].point[..]);
    // Control: the same proof read with another factor count fails.
    let mut wrong = shapes.clone();
    wrong[3].factors = 1;
    assert!(run_verify(&proof, &wrong).is_none());
}

/// Layers wider than one streamed chunk: the first rounds read leaves, the
/// layers above them, and stored layers beside them in several chunks.
#[test]
fn trees_wider_than_a_chunk_match_direct_sums_and_leaf_values() {
    let shapes = vec![
        TreeShape { depth: 22, factors: 1 },
        TreeShape { depth: 21, factors: 0 },
        TreeShape { depth: 20, factors: 2 },
    ];
    assert_direct(&shapes, &parts_of(&shapes));
}

#[test]
fn every_tampered_message_rejects_or_moves_a_claim() {
    let (shapes, parts) = batch();
    let (proof, honest) = prove(
        &leaves(&trees(&parts)),
        &mut fresh(),
        &crate::device::Accelerator::cpu(),
    )
    .unwrap();
    // A tampered message must fail, or change a root or leaf claim the caller checks.
    let caught = |proof: &GkrProof| run_verify(proof, &shapes).is_none_or(|claims| claims != honest);
    for tree in 0..shapes.len() {
        for side in 0..2 {
            let mut tampered = proof.clone();
            tampered.roots[tree][side] += Ext::ONE;
            assert!(caught(&tampered), "root {tree} side {side}");
        }
    }
    for (k, layer) in proof.layers.iter().enumerate() {
        for (round, message) in layer.rounds.iter().enumerate() {
            for node in 0..message.len() {
                let mut tampered = proof.clone();
                tampered.layers[k].rounds[round][node] += Ext::ONE;
                assert!(
                    run_verify(&tampered, &shapes).is_none(),
                    "layer {k} round {round} node {node}"
                );
            }
        }
        for (tree, values) in layer.children.iter().enumerate() {
            for value in 0..values.len() {
                let mut tampered = proof.clone();
                tampered.layers[k].children[tree][value] += Ext::ONE;
                assert!(
                    run_verify(&tampered, &shapes).is_none(),
                    "layer {k} tree {tree} child {value}"
                );
            }
        }
    }
    let mut short = proof.clone();
    short.layers.pop();
    assert!(run_verify(&short, &shapes).is_none());
}

#[test]
fn the_prover_refuses_a_witness_beyond_the_bound() {
    let mut z = witness(4);
    z[3] = signed(BOUND as i64 + 1);
    assert!(histogram(&z, BOUND).is_err());
}

/// A device that runs GKR layers on the CPU under the device contract: the
/// caller's tables in place, folded at a doubling stride.
struct StridedLayers;

struct StridedLayer<'a> {
    trees: Vec<Vec<&'a mut [u64]>>,
    eq: &'a mut [u64],
    stride: usize,
}

fn at(table: &[u64], index: usize) -> Ext {
    Ext::from_basis_coefficients_fn(|c| Gl::new(table[3 * index + c]))
}

fn put(table: &mut [u64], index: usize, value: Ext) {
    for (c, word) in BasedVectorSpace::<Gl>::as_basis_coefficients_slice(&value)
        .iter()
        .enumerate()
    {
        table[3 * index + c] = p3_field_v08::PrimeField64::as_canonical_u64(word);
    }
}

fn words_ext(words: [u64; 3]) -> Ext {
    Ext::from_basis_coefficients_fn(|c| Gl::new(words[c]))
}

impl crate::device::Device for StridedLayers {
    fn merkle_layers(
        &self,
        _rows: usize,
        _width: usize,
        _fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; 4]>>> {
        None
    }

    fn merkle_leaves(&self, _values: &[u64], _width: usize) -> Option<Vec<[u64; 4]>> {
        None
    }

    fn key_weights(&self, _powers: &[[u64; 3]], _blocks: usize) -> Option<Vec<[u64; 3]>> {
        None
    }

    fn dft_columns(&self, _values: &mut [u64], _width: usize, _generator: u64) -> Result<bool, String> {
        Ok(false)
    }

    fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Option<Box<dyn crate::device::GkrLayer + 'a>> {
        Some(Box::new(StridedLayer { trees, eq, stride: 1 }))
    }
}

impl crate::device::GkrLayer for StridedLayer<'_> {
    fn message(&mut self, weights: &[[u64; 3]], mu: [u64; 3], degree: usize) -> Result<Vec<[u64; 3]>, String> {
        let s = self.stride;
        let pairs = self.eq.len() / 3 / s / 2;
        let mu = words_ext(mu);
        let nodes: Vec<u64> = std::iter::once(0).chain(2..=degree as u64).collect();
        let line = |table: &[u64], i: usize, x: u64| {
            let low = at(table, i * s);
            low + Ext::from(Gl::from_u64(x)) * (at(table, (i + 1) * s) - low)
        };
        Ok(nodes
            .iter()
            .map(|&x| {
                let mut sum = Ext::ZERO;
                for m in 0..pairs {
                    let mut total = Ext::ZERO;
                    for (parts, &weight) in self.trees.iter().zip(weights) {
                        let half = 2 * pairs;
                        let f = parts.len() - 1;
                        let child = |part: usize, b: usize| line(parts[part], b * half + 2 * m, x);
                        let p = |b: usize| (0..f).map(|part| child(part, b)).product::<Ext>();
                        let (q0, q1) = (child(f, 0), child(f, 1));
                        total += words_ext(weight) * (p(0) * q1 + p(1) * q0 + mu * q0 * q1);
                    }
                    sum += line(self.eq, 2 * m, x) * total;
                }
                let words = BasedVectorSpace::<Gl>::as_basis_coefficients_slice(&sum);
                std::array::from_fn(|c| p3_field_v08::PrimeField64::as_canonical_u64(&words[c]))
            })
            .collect())
    }

    fn fold(&mut self, r: [u64; 3]) -> Result<(), String> {
        let (s, r) = (self.stride, words_ext(r));
        let fold = |table: &mut [u64]| {
            for k in 0..table.len() / 3 / s / 2 {
                let (a, b) = (at(table, 2 * k * s), at(table, (2 * k + 1) * s));
                put(table, 2 * k * s, a + r * (b - a));
            }
        };
        for parts in &mut self.trees {
            parts.iter_mut().for_each(|part| fold(part));
        }
        fold(self.eq);
        self.stride *= 2;
        Ok(())
    }
}

/// Layers run under the device contract (in place, at a doubling stride)
/// give the CPU prover's proof and claims.
#[test]
fn device_layers_give_the_cpu_proof() {
    let (_, parts) = batch();
    let device = crate::device::Accelerator::new(std::sync::Arc::new(StridedLayers));
    let (proof, claims) = prove(&leaves(&trees(&parts)), &mut fresh(), &device).unwrap();
    assert_eq!(
        (proof, claims),
        prove(
            &leaves(&trees(&parts)),
            &mut fresh(),
            &crate::device::Accelerator::cpu()
        )
        .unwrap()
    );
}
