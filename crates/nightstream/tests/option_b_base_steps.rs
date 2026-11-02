//! Both step circuits accept step 0 of the Poseidon2 hash chain, built the
//! way Lean's `LightAssemblerSource.honestValues` and
//! `RefreshAssemblerSource.honestValues` build it, with a zero fold of the
//! default slot (the zero proof that today's base step uses). Every row of
//! the circuit's relation holds on the witness that the package's program
//! fills.

use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::D;
use neo_params::NeoParams;
use neo_reductions::common::{sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    derive_light_transcript, derive_refresh_transcript, encode_digest, load_option_b_packages, Child, FoldWords,
    KWords, OptionBPackages, OptionBState, PiCcsV1_2Transcript, Slot, StepPackage, StepWords,
    OPTION_B_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as G;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn packages() -> OptionBPackages {
    load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        OPTION_B_PACKAGES_IDENTITY,
    )
    .expect("option-B packages")
}

fn words(values: [u64; 4]) -> [G; 4] {
    values.map(G::from_u64)
}

/// `Poseidon2HashChainV1.step`: Poseidon2 over the domain tag, the state
/// and the message.
fn hash_chain_step(state: [G; 4], message: [G; 4]) -> [G; 4] {
    let mut input: Vec<G> = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|&byte| G::from_u64(u64::from(byte)))
        .collect();
    input.extend(state);
    input.extend(message);
    poseidon2_hash(&input)
}

fn canonical(values: &[G]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

/// Step 0 of the hash chain for circuit `circuit` (0 light, 1 refresh): the
/// words without the fold, and the fresh public input.
struct StepZero {
    keys: [[G; 4]; 2],
    z0: [G; 4],
    message: [G; 4],
    next: [G; 4],
    output: [G; 4],
    fresh_public_input: Vec<G>,
}

impl StepZero {
    fn new(circuit: u64) -> Self {
        let keys = [words([1, 2, 3, 4]), words([5, 6, 7, 8])];
        let z0 = words([10, 11, 12, 13]);
        let message = words([20, 21, 22, 23]);
        let next = hash_chain_step(z0, message);
        let state = |iteration, current, pc| OptionBState {
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
    fn folded(&self, transcript: &PiCcsV1_2Transcript, challenges: usize, depth: u64) -> Slot {
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

    fn words(&self, folded: Slot, fold: FoldWords) -> StepWords {
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

fn assert_every_row_holds(package: &StepPackage, step: &StepWords) {
    let shared = step.columns().expect("step words");
    let rows = package.plan_rows().expect("plan rows");
    let columns = package.circuit_columns(&shared).expect("circuit columns");
    let columns = package
        .relation_columns(columns, &rows)
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
}

#[test]
fn light_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(0);
    let transcript = derive_light_transcript(
        &[0; 1188],
        &canonical(&step.fresh_public_input),
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 2 * 540],
    )
    .expect("light transcript");
    let folded = step.folded(&transcript, 1, 1);
    let fold = FoldWords::Light {
        outputs: [vec![[G::ZERO; 2]; 270], vec![[G::ZERO; 2]; 270]],
    };
    assert_every_row_holds(packages.light(), &step.words(folded, fold));
}

#[test]
fn refresh_step_zero_holds_every_row() {
    let packages = packages();
    let step = StepZero::new(1);
    let transcript = derive_refresh_transcript(
        &[0; 1188],
        &canonical(&step.fresh_public_input),
        &vec![0; 16 * (1188 + 540)],
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 17 * 540],
    )
    .expect("refresh transcript");
    let folded = step.folded(&transcript, 17, 0);
    let child = Child {
        commitment: vec![G::ZERO; 1188],
        evaluations: vec![[G::ZERO; 2]; 270],
        digits: vec![G::ZERO; 270],
    };
    let fold = FoldWords::Refresh {
        children: vec![child; 16],
        outputs: vec![vec![[G::ZERO; 2]; 270]; 17],
    };
    assert_every_row_holds(packages.refresh(), &step.words(folded, fold));
}
