//! Step words of the Poseidon2 hash chain for the bounded-depth tests, built the
//! way Lean's `LightAssemblerSource.honestValues` and
//! `RefreshAssemblerSource.honestValues` build them, and the check that every
//! row of a circuit's relation holds on them.

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::D;
use neo_params::NeoParams;
use neo_reductions::common::{sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    encode_digest, load_bounded_depth_packages, BoundedDepthPackages, BoundedDepthState, FoldWords, KWords,
    PiCcsV1_2Transcript, Slot, StepPackage, StepWords, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as G;

pub fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

pub fn packages() -> BoundedDepthPackages {
    load_bounded_depth_packages(
        &artifact("nightstream-fprime-bounded-depth-light-v1.json"),
        &artifact("nightstream-fprime-bounded-depth-refresh-v1.json"),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .expect("bounded-depth packages")
}

pub fn words(values: [u64; 4]) -> [G; 4] {
    values.map(G::from_u64)
}

/// `Poseidon2HashChainV1.step`: Poseidon2 over the domain tag, the state
/// and the message.
pub fn hash_chain_step(state: [G; 4], message: [G; 4]) -> [G; 4] {
    let mut input: Vec<G> = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|&byte| G::from_u64(u64::from(byte)))
        .collect();
    input.extend(state);
    input.extend(message);
    poseidon2_hash(&input)
}

pub fn canonical(values: &[G]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

/// Step 0 of the hash chain for circuit `circuit` (0 light, 1 refresh): the
/// words without the fold, and the fresh public input.
pub struct StepZero {
    pub keys: [[G; 4]; 2],
    pub z0: [G; 4],
    pub message: [G; 4],
    pub next: [G; 4],
    pub output: [G; 4],
    pub fresh_public_input: Vec<G>,
}

impl StepZero {
    pub fn new(circuit: u64) -> Self {
        let keys = [words([1, 2, 3, 4]), words([5, 6, 7, 8])];
        let z0 = words([10, 11, 12, 13]);
        let message = words([20, 21, 22, 23]);
        let next = hash_chain_step(z0, message);
        let state = |iteration, current, pc| BoundedDepthState {
            keys,
            iteration,
            z0,
            current,
            pc,
            slots: [Slot::default_slot(), Slot::default_slot()],
        };
        Self {
            fresh_public_input: encode_digest(state(0, z0, 0).hash()),
            output: state(1, next, circuit + 1).hash(),
            keys,
            z0,
            message,
            next,
        }
    }

    /// The fold of a zero proof: the commitments and evaluations stay 0, the
    /// public input is the first challenge times the fresh public input, and
    /// the point is the round point.
    pub fn folded(&self, transcript: &PiCcsV1_2Transcript, challenges: usize, depth: u64) -> Slot {
        let mut sampler = Poseidon2Transcript::from_state_and_absorbed(transcript.outgoing_state().map(G::from_u64), 0);
        let rho = sample_rot_rhos_n_typed(
            &mut sampler,
            &NeoParams::nightstream_goldilocks_k16(),
            &RotRing::goldilocks(),
            challenges,
        )
        .expect("challenges")
        .remove(0);
        let public_input = (0..270)
            .map(|column| {
                let (lane, block) = (column % D, column / D);
                (0..D).fold(G::ZERO, |sum, source| {
                    sum + rho.as_mat()[(lane, source)] * self.fresh_public_input[block * D + source]
                })
            })
            .collect();
        Slot {
            point: transcript
                .round_point()
                .iter()
                .map(|value| value.map(G::from_u64) as KWords)
                .collect(),
            public_input,
            depth,
            ..Slot::default_slot()
        }
    }

    pub fn words(&self, folded: Slot, fold: FoldWords) -> StepWords {
        let default = Slot::default_slot();
        StepWords {
            iteration: 0,
            prior_pc: 0,
            selected: 0,
            keys: self.keys,
            states: [self.z0, self.z0, self.next],
            witness: self.message.to_vec(),
            slots: [
                default.clone(),
                default.clone(),
                default.clone(),
                folded,
                default.clone(),
                default.clone(),
                default.clone(),
            ],
            digests: [
                default.digest(),
                default.digest(),
                default.digest(),
                default.digest(),
                self.output,
            ],
            fresh_commitment: vec![G::ZERO; 1188],
            fresh_public_input: self.fresh_public_input.clone(),
            rounds: vec![[[G::ZERO; 2]; 9]; 28],
            fold,
        }
    }
}

/// Checks every row on the step's witness and returns its logical assignment.
pub fn assert_every_row_holds(package: &StepPackage, step: &StepWords) -> Vec<G> {
    let shared = step.columns().expect("step words");
    let expanded = package.clone().expand().expect("expanded package");
    let (rows, layout) = (expanded.rows(), expanded.layout());
    let columns = package.circuit_columns(&shared).expect("circuit columns");
    let columns = package
        .relation_columns(columns, rows)
        .expect("relation columns");
    let failing: Vec<usize> = (0..rows.row_count())
        .filter(|&row| rows.residual(row, &columns) != G::ZERO)
        .collect();
    assert!(
        failing.is_empty(),
        "circuit {}: {} rows fail, first {:?}",
        package.circuit(),
        failing.len(),
        &failing[..failing.len().min(8)]
    );
    let logical = layout.assignment(&columns).expect("logical assignment");
    let port = |row, port| {
        layout
            .evaluate(rows, row, port, &logical)
            .expect("logical row")
    };
    let failing: Vec<usize> = (0..rows.row_count())
        .filter(|&row| port(row, 0) * port(row, 1) - port(row, 2) + port(row, 3).exp_u64(7) != G::ZERO)
        .collect();
    assert!(
        failing.is_empty(),
        "circuit {}: {} logical rows fail, first {:?}",
        package.circuit(),
        failing.len(),
        &failing[..failing.len().min(8)]
    );
    logical
}
