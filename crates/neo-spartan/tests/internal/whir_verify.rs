//! Our WHIR verifier against Plonky3's `verify_at`: the same values and the
//! same challenger state on honest openings, the same rejections on mutated
//! ones, and satisfied rows that do not depend on the proof.

use p3_challenger_v08::CanSample;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_sumcheck_v08::OpeningBatch;

use super::word;
use crate::circuit::hash::Duplex;
use crate::circuit::record::{Recorder, Trace};
use crate::circuit::{poseidon2, Backend, Native};
use crate::field::{Ext, Gl};
use crate::hash::{permutation, Challenger};
use crate::pcs::{Opening, Pcs, TablePlan};
use crate::whir::{observe_root, verify, OpeningView};

fn ext(seed: u64, index: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, 4 * index + i as u64)))
}

/// A committed and opened toy plan with its points and Plonky3's verdict.
struct Case {
    pcs: Pcs,
    root: [Gl; 4],
    opening: Opening,
    points: Vec<Vec<Ext>>,
    commitment: crate::pcs::Commitment,
}

fn case(plans: Vec<TablePlan>, seed: u64) -> Case {
    let points: Vec<Vec<Ext>> = plans
        .iter()
        .flat_map(|plan| plan.points.iter().map(move |_| plan.variables))
        .enumerate()
        .map(|(batch, variables)| {
            (0..variables)
                .map(|c| ext(seed + batch as u64, c as u64))
                .collect()
        })
        .collect();
    let tables: Vec<Vec<Gl>> = plans
        .iter()
        .enumerate()
        .map(|(t, plan)| {
            (0..plan.width << plan.variables)
                .map(|i| Gl::from_u64(word(seed * 31 + t as u64, i as u64)))
                .collect()
        })
        .collect();
    let pcs = Pcs::new(plans, 100.0, &|_| Vec::new()).unwrap();
    let mut prover = Challenger::new(permutation().clone());
    let (commitment, data) = pcs.commit(tables, &mut prover);
    let opening = pcs.open(data, &points, &mut prover);
    let root = commitment.roots()[0];
    Case {
        pcs,
        root,
        opening,
        points,
        commitment,
    }
}

/// Plonky3's verdict and its next sample.
fn theirs(case: &Case, opening: &Opening) -> Option<(Vec<Vec<Ext>>, Gl)> {
    let mut challenger = Challenger::new(permutation().clone());
    case.pcs.observe(&case.commitment, &mut challenger);
    let values = case
        .pcs
        .verify(&case.commitment, opening, &case.points, &mut challenger)
        .ok()?;
    Some((values, CanSample::<Gl>::sample(&mut challenger)))
}

/// Our verdict on `Native` and its next sample.
fn ours(case: &Case, opening: &Opening, root: [Gl; 4]) -> Option<(Vec<Vec<Ext>>, Gl)> {
    let b = &mut Native;
    let mut duplex = Duplex::new(b);
    observe_root(b, &mut duplex, root);
    let view = OpeningView::read(b, &case.pcs, Some(opening)).ok()?;
    let values = verify(b, &case.pcs, root, &view, &case.points, &mut duplex).ok()?;
    Some((values, duplex.sample(b)))
}

fn plans() -> Vec<Vec<TablePlan>> {
    vec![
        vec![TablePlan {
            variables: 12,
            width: 1,
            points: vec![vec![0]; 2],
        }],
        vec![
            TablePlan {
                variables: 11,
                width: 1,
                points: vec![vec![0]; 2],
            },
            TablePlan {
                variables: 9,
                width: 2,
                points: vec![vec![0, 1]],
            },
            TablePlan {
                variables: 7,
                width: 9,
                points: vec![(0..8).collect(), vec![8], (0..9).collect()],
            },
            TablePlan {
                variables: 3,
                width: 1,
                points: vec![vec![0]],
            },
        ],
    ]
}

#[test]
fn honest_openings_match_p3() {
    for (seed, plan) in plans().into_iter().enumerate() {
        let case = case(plan, 100 + seed as u64);
        let expected = theirs(&case, &case.opening).expect("Plonky3 accepts");
        assert_eq!(ours(&case, &case.opening, case.root), Some(expected), "plan {seed}");
    }
}

#[test]
fn mutated_openings_reject_in_both() {
    let case = case(plans().remove(1), 7);
    let mut cases: Vec<(&str, Opening, [Gl; 4])> = Vec::new();
    let mut push = |name, change: &dyn Fn(&mut Opening)| {
        let mut opening = case.opening.clone();
        change(&mut opening);
        cases.push((name, opening, case.root));
    };
    push("eval", &|o| {
        let mut current = o.evals[2].current().to_vec();
        current[0] += Ext::ONE;
        o.evals[2] = OpeningBatch::new(current, Vec::new());
    });
    push("initial OOD", &|o| o.whir.initial_ood_answers[0] += Ext::ONE);
    push("initial sum-check", &|o| {
        o.whir.initial_sumcheck.polynomial_evaluations[1][0] += Ext::ONE
    });
    push("round OOD", &|o| o.whir.rounds[0].ood_answers[0] += Ext::ONE);
    push("round sum-check", &|o| {
        o.whir.rounds[0].sumcheck.polynomial_evaluations[0][1] += Ext::ONE
    });
    push("final polynomial", &|o| {
        o.whir.final_poly.as_mut().unwrap().as_mut_slice()[0] += Ext::ONE
    });
    push("query row", &|o| match &mut o.whir.rounds[0].openings {
        p3_whir::pcs::proof::QueryOpenings::Base(opening) => opening.rows[0][3] += Gl::ONE,
        p3_whir::pcs::proof::QueryOpenings::Extension(opening) => opening.rows[0][3] += Ext::ONE,
    });
    push("sibling", &|o| match &mut o.whir.final_openings {
        p3_whir::pcs::proof::QueryOpenings::Base(opening) => opening.proof.sibling_hashes[0][1] += Gl::ONE,
        p3_whir::pcs::proof::QueryOpenings::Extension(opening) => opening.proof.sibling_hashes[0][1] += Gl::ONE,
    });
    for (name, opening, root) in &cases {
        assert!(theirs(&case, opening).is_none(), "Plonky3 accepts {name}");
        assert!(ours(&case, opening, *root).is_none(), "we accept {name}");
    }
    let mut root = case.root;
    root[2] += Gl::ONE;
    assert!(ours(&case, &case.opening, root).is_none(), "root");
}

fn record(case: &Case, opening: Option<&Opening>) -> (Trace, Option<&'static str>) {
    let mut recorder = Recorder::new(Trace::default());
    let b = &mut recorder;
    let root = case.root.map(|word| b.private(word));
    let points: Vec<Vec<<Recorder<Trace> as Backend>::E>> = case
        .points
        .iter()
        .map(|point| point.iter().map(|&x| b.ext_constant(x)).collect())
        .collect();
    let mut duplex = Duplex::new(b);
    observe_root(b, &mut duplex, root);
    let view = OpeningView::read(b, &case.pcs, opening).unwrap();
    verify(b, &case.pcs, root, &view, &points, &mut duplex).unwrap();
    recorder.finish()
}

#[test]
fn recorded_verification_holds_and_does_not_depend_on_the_proof() {
    let case = case(plans().remove(1), 9);
    let (honest, failure) = record(&case, Some(&case.opening));
    assert_eq!(failure, None);
    for r in 0..honest.rows.len() {
        assert_eq!(honest.row_value(r), Gl::ZERO, "row {r}");
    }
    for block in 0..honest.blocks.len() / poseidon2::CELLS {
        assert_eq!(honest.block_failure(block), None, "block {block}");
    }
    let (shape, _) = record(&case, None);
    assert_eq!(honest.rows, shape.rows);
    assert_eq!(honest.entries, shape.entries);
    assert_eq!(honest.glue.len(), shape.glue.len());
    assert_eq!(honest.blocks.len(), shape.blocks.len());
}
