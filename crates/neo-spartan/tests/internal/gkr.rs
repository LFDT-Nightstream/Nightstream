//! The batched fraction-sum GKR and the histogram side of logUp.

use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing};

use super::{fresh, gkr_verify, word};
use crate::circuit::Native;
use crate::field::{signed, Ext, Gl};
use crate::gkr::{prove, GkrProof, Tree, TreeClaim, TreeShape};
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
    crate::norm::table_sum(&mut Native, &words(counts), BOUND, beta).unwrap()
}

#[test]
fn leaf_claim_matches_the_witness_and_the_root_matches_the_histogram() {
    for variables in [1, 2, 7] {
        let z = witness(variables);
        let beta = beta();
        let (proof, prover) = prove(vec![norm_tree(&z, beta)], &mut fresh());
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
    let (proof, _) = prove(vec![norm_tree(&z, beta)], &mut fresh());
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
    let parts = shapes
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
        .collect();
    (shapes, parts)
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

#[test]
fn batched_trees_match_direct_sums_and_leaf_values() {
    let (shapes, parts) = batch();
    let (proof, prover) = prove(trees(&parts), &mut fresh());
    let claims = run_verify(&proof, &shapes).unwrap();
    assert_eq!(claims, prover);
    for (claim, parts) in claims.iter().zip(&parts) {
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
    assert_eq!(claims[0].point, claims[1].point);
    assert_ne!(claims[2].point[..5], claims[3].point[..]);
    // Control: the same proof read with another factor count fails.
    let mut wrong = shapes.clone();
    wrong[3].factors = 1;
    assert!(run_verify(&proof, &wrong).is_none());
}

#[test]
fn every_tampered_message_rejects_or_moves_a_claim() {
    let (shapes, parts) = batch();
    let (proof, honest) = prove(trees(&parts), &mut fresh());
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
