//! Our WHIR prover against Plonky3's `WhirProver` built from Plonky3's own
//! DFT and Merkle tree: the same commitment and opening bytes, the same
//! challenger after them, on the CPU and through a device; Plonky3's
//! verifier accepts ours.

use std::sync::Arc;

use p3_challenger_v08::CanSample;
use p3_commit_v08::MultilinearPcs;
use p3_dft_v08::Radix2DFTSmallBatch;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_sumcheck_v08::layout::SuffixProver;
use p3_sumcheck_v08::PrescribedPointPcs;
use p3_whir::pcs::prover::WhirProver;

use super::merkle::Reference;
use super::word;
use crate::device::Accelerator;
use crate::field::{Ext, Gl};
use crate::hash::{challenger, plain, Challenger, Plain};
use crate::pcs::{Pcs, Profile, TablePlan, LAYER1, SHRINK};

/// Plonky3's prover with Plonky3's DFT and Merkle tree.
type Theirs = WhirProver<Ext, Gl, Radix2DFTSmallBatch<Gl>, Plain, Challenger, SuffixProver<Gl, Ext>>;

fn plan(variables: usize, width: usize, points: Vec<Vec<usize>>) -> TablePlan {
    TablePlan {
        variables,
        width,
        points,
    }
}

/// Commit to and open `plans` with our prover and with Plonky3's, from one
/// challenger: the bytes and the challengers after must be equal, and
/// Plonky3's verifier must accept ours. Returns the round count.
fn check(profile: Profile, plans: Vec<TablePlan>, seed: u64, accelerator: &Accelerator) -> usize {
    let tables: Vec<Vec<Gl>> = plans
        .iter()
        .enumerate()
        .map(|(t, plan)| {
            (0..plan.width << plan.variables)
                .map(|i| Gl::from_u64(word(seed * 31 + t as u64, i as u64)))
                .collect()
        })
        .collect();
    let points: Vec<Vec<Ext>> = plans
        .iter()
        .flat_map(|plan| plan.points.iter().map(move |_| plan.variables))
        .enumerate()
        .map(|(batch, variables)| {
            (0..variables)
                .map(|c| {
                    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed + batch as u64, (4 * c + i) as u64)))
                })
                .collect()
        })
        .collect();
    let pcs = Pcs::new(profile, plans, 100.0, &|_| Vec::new()).unwrap();
    let start = challenger([Gl::from_u64(seed); 4]);

    let mut ours = start.clone();
    let (commitment, data) = pcs.commit(tables.clone(), &mut ours, accelerator).unwrap();
    let opening = pcs.open(data, &points, &mut ours, accelerator).unwrap();

    let reference = Theirs::new(pcs.config().clone(), Radix2DFTSmallBatch::default(), plain());
    let mut theirs = start.clone();
    let (their_commitment, their_data) = reference.commit(pcs.witness(tables), &mut theirs).unwrap();
    let their_opening = reference
        .open_at(their_data, pcs.protocol(), &pcs.p3_points(&points), &mut theirs)
        .unwrap();

    assert_eq!(
        bincode::serialize(&commitment).unwrap(),
        bincode::serialize(&their_commitment).unwrap(),
        "commitment"
    );
    assert_eq!(
        bincode::serialize(&opening).unwrap(),
        bincode::serialize(&their_opening).unwrap(),
        "opening"
    );
    let mut verifier = start;
    pcs.observe(&commitment, &mut verifier);
    pcs.verify(&commitment, &opening, &points, &mut verifier)
        .expect("Plonky3 accepts our opening");
    let next = CanSample::<Gl>::sample(&mut ours);
    assert_eq!(
        next,
        CanSample::<Gl>::sample(&mut theirs),
        "challenger after the opening"
    );
    assert_eq!(next, CanSample::<Gl>::sample(&mut verifier), "verifier's challenger");
    pcs.config().n_rounds()
}

/// One table; four tables of different heights and widths, one below the
/// first folding round (the shape of P0); a taller pair (more rounds).
fn layer1_cases() -> Vec<Vec<TablePlan>> {
    vec![
        vec![plan(12, 1, vec![vec![0]; 2])],
        vec![
            plan(11, 1, vec![vec![0]; 2]),
            plan(9, 2, vec![vec![0, 1]]),
            plan(7, 9, vec![(0..8).collect(), vec![8], (0..9).collect()]),
            plan(3, 1, vec![vec![0]]),
        ],
        vec![
            plan(16, 2, vec![vec![0, 1], vec![1]]),
            plan(14, 6, vec![(0..6).collect()]),
        ],
    ]
}

#[test]
fn layer1_openings_equal_plonky3() {
    let rounds: Vec<usize> = layer1_cases()
        .into_iter()
        .enumerate()
        .map(|(seed, plans)| check(LAYER1, plans, 11 + seed as u64, &Accelerator::cpu()))
        .collect();
    assert!(
        rounds.iter().any(|&rounds| rounds >= 2),
        "some case folds through two rounds: {rounds:?}"
    );
}

#[test]
fn device_openings_equal_plonky3() {
    let device = Arc::new(Reference::default());
    for (seed, plans) in layer1_cases().into_iter().enumerate() {
        check(LAYER1, plans, 21 + seed as u64, &Accelerator::new(device.clone()));
    }
    assert!(!device.dfts.lock().unwrap().is_empty(), "the device ran the DFTs");
    assert!(!device.trees.lock().unwrap().is_empty(), "the device hashed the trees");
}

/// The shrink profile grinds. Plonky3 grinds in parallel and takes any
/// witness it finds, so both provers run on one thread: the first witness.
#[test]
fn shrink_openings_equal_plonky3() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        check(SHRINK, vec![plan(13, 1, vec![vec![0]])], 31, &Accelerator::cpu());
        check(
            SHRINK,
            vec![plan(12, 2, vec![vec![0, 1]]), plan(6, 3, vec![vec![2]])],
            32,
            &Accelerator::cpu(),
        );
    });
}
