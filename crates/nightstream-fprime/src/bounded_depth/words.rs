//! The light step's shared columns (Lean `LightAssembler.interface`,
//! `LightAssemblerSource.env`) and the two-slot hashes they carry (Lean
//! `Lifecycle/BoundedDepth/State`). Columns 0 to 269 are the public input `x`, the
//! encoding of the output digest (`encHash`); the step's words follow from
//! column 270 in the assembler's groups.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;

use crate::package::PackageError;
use neo_transcript::domain_chunk_v1_2;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;

type G = Goldilocks;

/// A K value as its two base-field coefficients.
pub type KWords = [G; 2];

pub const PUBLIC_WORDS: usize = 270;
pub const COMMITMENT_WORDS: usize = 1188;
const EVALUATIONS: usize = 270;
const POINT_COORDINATES: usize = 28;
const ROUND_COEFFICIENTS: usize = 9;
const SOURCE_OUTPUTS: usize = 270;
const CHILDREN: usize = 16;

/// A slot: its claim and its depth (`State.SlotData`). The evaluations are
/// `Eval_K` (54) then `Eval_A` (four matrices of 54).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub commitment: Vec<G>,
    pub evaluations: Vec<KWords>,
    pub point: Vec<KWords>,
    pub public_input: Vec<G>,
    pub depth: u64,
}

impl Slot {
    /// HyperNova Definition 12's default running instance at depth 0
    /// (`Step.defaultSlot`): every word is 0.
    pub fn default_slot() -> Self {
        Self {
            commitment: vec![G::ZERO; COMMITMENT_WORDS],
            evaluations: vec![[G::ZERO; 2]; EVALUATIONS],
            point: vec![[G::ZERO; 2]; POINT_COORDINATES],
            public_input: vec![G::ZERO; PUBLIC_WORDS],
            depth: 0,
        }
    }

    /// `State.serializeSlot`, in the order of `SlotExpr.words`.
    pub fn words(&self) -> Vec<G> {
        let mut words = self.commitment.clone();
        words.extend(self.evaluations.iter().flatten());
        words.extend(self.point.iter().flatten());
        words.extend(&self.public_input);
        words.push(G::from_u64(self.depth));
        words
    }

    /// `State.slotDigest`.
    pub fn digest(&self) -> [G; 4] {
        let mut input = domain_chunk_v1_2(b"HyperNova/NIVC/slot/v3").to_vec();
        input.extend(self.words());
        poseidon2_hash(&input)
    }
}

/// What the state hash binds (`State.State`).
#[derive(Clone, Debug)]
pub struct State {
    pub keys: [[G; 4]; 2],
    pub iteration: u64,
    pub z0: [G; 4],
    pub current: [G; 4],
    pub pc: u64,
    pub slots: [Slot; 2],
}

impl State {
    /// `State.stateHash`.
    pub fn hash(&self) -> [G; 4] {
        let mut input = domain_chunk_v1_2(b"HyperNova/NIVC/state/v3").to_vec();
        input.extend(self.keys.iter().flatten());
        input.push(G::from_u64(self.iteration));
        input.extend(self.z0);
        input.extend(self.current);
        input.push(G::from_u64(self.pc));
        input.extend(self.slots[0].digest());
        input.extend(self.slots[1].digest());
        poseidon2_hash(&input)
    }
}

/// `encHash`: the marker 1, the digest's 256 little-endian bits, then zeros.
pub fn encode_digest(digest: [G; 4]) -> Vec<G> {
    let mut words = vec![G::ZERO; PUBLIC_WORDS];
    words[0] = G::ONE;
    for (word, value) in digest.iter().enumerate() {
        let value = p3_field::PrimeField64::as_canonical_u64(value);
        for bit in 0..64 {
            words[1 + 64 * word + bit] = G::from_u64((value >> bit) & 1);
        }
    }
    words
}

/// A PiDEC child of the refresh step: its commitment, its evaluation family
/// and its digit words (`Refresh.Split`, `publicInputSplit`).
#[derive(Clone, Debug)]
pub struct Child {
    pub commitment: Vec<G>,
    pub evaluations: Vec<KWords>,
    pub digits: Vec<G>,
}

/// The words of the step's fold after the fresh instance: the light fold's
/// two-source PiCCS output, or the refresh fold's 16 children and 17-source
/// output. The round messages lie between them in both.
#[derive(Clone, Debug)]
pub enum FoldWords {
    Light {
        outputs: [Vec<KWords>; 2],
    },
    Refresh {
        children: Vec<Child>,
        outputs: Vec<Vec<KWords>>,
    },
}

/// A step's shared words (`LightAssemblerSource.Values`,
/// `RefreshAssemblerSource.Values`). The slots are the two prior slots, the
/// active slot, the two inner slots and the two next slots; the digests are
/// the two prior slot digests, the two next slot digests and the output
/// digest. The digest select's words (`digestSelect`) follow the fold: the new
/// slot's digest, the digest of each inner slot and, for a light step, the
/// active slot's digest (Lean `LightAssembler.interface`).
#[derive(Clone, Debug)]
pub struct StepWords {
    pub iteration: u64,
    pub prior_pc: u64,
    pub selected: u64,
    pub keys: [[G; 4]; 2],
    pub states: [[G; 4]; 3],
    pub witness: Vec<G>,
    pub slots: [Slot; 7],
    pub digests: [[G; 4]; 5],
    pub fresh_commitment: Vec<G>,
    pub fresh_public_input: Vec<G>,
    pub rounds: Vec<[KWords; ROUND_COEFFICIENTS]>,
    pub fold: FoldWords,
}

impl StepWords {
    /// The shared columns below the step's `sharedEnd`.
    pub fn columns(&self) -> Result<Vec<G>, PackageError> {
        let slots_fit = self.slots.iter().all(|slot| {
            slot.commitment.len() == COMMITMENT_WORDS
                && slot.evaluations.len() == EVALUATIONS
                && slot.point.len() == POINT_COORDINATES
                && slot.public_input.len() == PUBLIC_WORDS
        });
        let outputs_fit = |outputs: &[Vec<KWords>], count| {
            outputs.len() == count && outputs.iter().all(|output| output.len() == SOURCE_OUTPUTS)
        };
        let fold_fits = match &self.fold {
            FoldWords::Light { outputs } => outputs_fit(outputs, 2),
            FoldWords::Refresh { children, outputs } => {
                outputs_fit(outputs, 1 + CHILDREN)
                    && children.len() == CHILDREN
                    && children.iter().all(|child| {
                        child.commitment.len() == COMMITMENT_WORDS
                            && child.evaluations.len() == EVALUATIONS
                            && child.digits.len() == PUBLIC_WORDS
                    })
            }
        };
        if !slots_fit
            || !fold_fits
            || self.selected > 1
            || self.fresh_commitment.len() != COMMITMENT_WORDS
            || self.fresh_public_input.len() != PUBLIC_WORDS
            || self.rounds.len() != POINT_COORDINATES
        {
            return Err(PackageError::Input("step word shapes"));
        }
        let iteration = G::from_u64(self.iteration);
        let mut columns = encode_digest(self.digests[4]);
        columns.extend([
            iteration,
            iteration.try_inverse().unwrap_or(G::ZERO),
            G::from_bool(self.iteration != 0),
            G::from_u64(self.prior_pc),
            G::from_u64(self.selected),
        ]);
        columns.extend(self.keys.iter().flatten());
        columns.extend(self.states.iter().flatten());
        columns.extend(&self.witness);
        for slot in &self.slots {
            columns.extend(slot.words());
        }
        columns.extend(self.digests.iter().flatten());
        columns.extend(&self.fresh_commitment);
        columns.extend(&self.fresh_public_input);
        let outputs = match &self.fold {
            FoldWords::Light { outputs } => outputs.as_slice(),
            FoldWords::Refresh { children, outputs } => {
                columns.extend(children.iter().flat_map(|child| child.commitment.iter()));
                columns.extend(
                    children
                        .iter()
                        .flat_map(|child| child.evaluations.iter().flatten()),
                );
                columns.extend(children.iter().flat_map(|child| child.digits.iter()));
                outputs.as_slice()
            }
        };
        columns.extend(self.rounds.iter().flatten().flatten());
        for output in outputs {
            columns.extend(output.iter().flatten());
        }
        columns.extend(self.slots[3 + self.selected as usize].digest());
        columns.extend(self.slots[3].digest());
        columns.extend(self.slots[4].digest());
        if let FoldWords::Light { .. } = self.fold {
            columns.extend(self.slots[2].digest());
        }
        Ok(columns)
    }
}
