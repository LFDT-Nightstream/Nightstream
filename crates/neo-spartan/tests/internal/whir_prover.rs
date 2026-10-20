//! Our WHIR prover against Plonky3's `WhirProver` built from Plonky3's own
//! DFT and Merkle tree: the same commitment and opening bytes, the same
//! challenger after them, on the CPU and through a device; Plonky3's
//! verifier accepts ours. A measurement reports the time and heap of our
//! prover on a P0-like opening.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use p3_challenger_v08::CanSample;
use p3_commit_v08::MultilinearPcs;
use p3_dft_v08::Radix2DFTSmallBatch;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_sumcheck_v08::layout::SuffixProver;
use p3_sumcheck_v08::PrescribedPointPcs;
use p3_whir::pcs::prover::WhirProver;
use rayon::prelude::*;

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

/// Pseudo-random tables for `plans`, each its columns one after another.
fn tables(plans: &[TablePlan], seed: u64) -> Vec<Vec<Gl>> {
    plans
        .iter()
        .enumerate()
        .map(|(t, plan)| {
            (0..plan.width << plan.variables)
                .into_par_iter()
                .map(|i| Gl::from_u64(word(seed * 31 + t as u64, i as u64)))
                .collect()
        })
        .collect()
}

/// One pseudo-random point per planned batch.
fn points(plans: &[TablePlan], seed: u64) -> Vec<Vec<Ext>> {
    plans
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
        .collect()
}

/// Commit to and open `plans` with our prover and with Plonky3's, from one
/// challenger: the bytes and the challengers after must be equal, and
/// Plonky3's verifier must accept ours. Returns the round count.
fn check(profile: Profile, plans: Vec<TablePlan>, seed: u64, accelerator: &Accelerator) -> usize {
    let (tables, points) = (tables(&plans, seed), points(&plans, seed));
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
    assert!(
        !device.leaves.lock().unwrap().is_empty(),
        "the device hashed the leaves"
    );
    assert!(!device.trees.lock().unwrap().is_empty(), "the device hashed the nodes");
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

/// Counts live heap bytes while `COUNTING` is set. A grown block counts
/// once, as if it grew in place.
struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn count(bytes: isize) {
    if COUNTING.load(Ordering::Relaxed) {
        let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size() as isize);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size() as isize);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size as isize - layout.size() as isize);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `phase`, then report its time, the counted heap at its start and its
/// heap peak.
fn measure<T>(name: &str, phase: impl FnOnce() -> T) -> T {
    let start = LIVE.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let started = Instant::now();
    let value = phase();
    let gigabytes = |bytes: isize| bytes as f64 / 1e9;
    eprintln!(
        "{name}: {:.2?}, heap {:.2} GB at the start, peak {:.2} GB",
        started.elapsed(),
        gigabytes(start),
        gigabytes(PEAK.load(Ordering::Relaxed))
    );
    value
}

/// P0's four tables (production at e402a6d99) with half their rows:
/// `2^27` stacked cells instead of `2^28`.
fn p0_like() -> Vec<TablePlan> {
    vec![
        plan(25, 1, vec![vec![0]; 2]),
        plan(25, 2, vec![vec![0, 1]]),
        plan(20, 9, vec![(0..8).collect(), vec![8], (0..9).collect()]),
        plan(19, 1, vec![vec![0]]),
    ]
}

/// Our commitment and opening of a P0-like table set on the CPU: the time
/// and heap of each, counted from the tables on. Plonky3's own prover then
/// gives the same bytes, unmeasured.
#[test]
#[ignore = "Measurement; run alone: `-- --ignored --exact internal_tests::whir_prover::p0_like_heap_and_time`."]
fn p0_like_heap_and_time() {
    let plans = p0_like();
    let pcs = Pcs::new(LAYER1, plans.clone(), 100.0, &|_| Vec::new()).unwrap();
    let points = points(&plans, 5);
    let cpu = Accelerator::cpu();
    COUNTING.store(true, Ordering::Relaxed);
    let inputs = tables(&plans, 5);
    let mut ours = challenger([Gl::ONE; 4]);
    let (commitment, data) = measure("commit", || pcs.commit(inputs, &mut ours, &cpu).unwrap());
    let opening = measure("open", || pcs.open(data, &points, &mut ours, &cpu).unwrap());
    COUNTING.store(false, Ordering::Relaxed);

    let reference = Theirs::new(pcs.config().clone(), Radix2DFTSmallBatch::default(), plain());
    let mut theirs = challenger([Gl::ONE; 4]);
    let (their_commitment, data) = reference
        .commit(pcs.witness(tables(&plans, 5)), &mut theirs)
        .unwrap();
    let their_opening = reference
        .open_at(data, pcs.protocol(), &pcs.p3_points(&points), &mut theirs)
        .unwrap();
    assert_eq!(
        bincode::serialize(&commitment).unwrap(),
        bincode::serialize(&their_commitment).unwrap()
    );
    assert_eq!(
        bincode::serialize(&opening).unwrap(),
        bincode::serialize(&their_opening).unwrap()
    );
}
