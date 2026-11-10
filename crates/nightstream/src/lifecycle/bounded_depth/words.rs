//! The step words that the bounded-depth circuits read (`LightAssemblerSource.Values`,
//! `RefreshAssemblerSource.Values`), built from the two-slot state and a fold.
use crate::folding::bounded_depth::{Kind, Proof};
use crate::folding::CeClaim;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::common::{sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    derive_light_transcript, encode_digest, BoundedDepthState, Child, FoldWords, KWords, PackageError, Slot, StepWords,
    PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const COMMITMENT_WORDS: usize = 1188;
const ROUNDS: usize = 28;
const ROUND_COEFFICIENTS: usize = 9;
const SOURCE_WORDS: usize = 2 * 270;

fn k_words(value: K) -> KWords {
    value.as_coeffs()
}

/// A claim's 270 `K` values: `Eval_K`, then the four `Eval_A` families.
fn evaluations(claim: &CeClaim) -> Vec<KWords> {
    claim.eval_k[..D]
        .iter()
        .chain(claim.eval_a.iter().flat_map(|family| family[..D].iter()))
        .copied()
        .map(k_words)
        .collect()
}

fn public_input(claim: &CeClaim) -> Vec<F> {
    (0..PUBLIC_WORDS)
        .map(|column| claim.X[(column % D, column / D)])
        .collect()
}

/// The slot words of an accumulator claim at `depth` (`State.SlotData`).
pub(crate) fn slot(claim: &CeClaim, depth: u64) -> Slot {
    Slot {
        commitment: claim.c.data().to_vec(),
        evaluations: evaluations(claim),
        point: claim.r.iter().copied().map(k_words).collect(),
        public_input: public_input(claim),
        depth,
    }
}

/// The program counter of the circuit that runs a fold kind; the circuit's
/// instances fold into slot `pc − 1`.
pub(crate) fn pc(kind: Kind) -> u64 {
    match kind {
        Kind::Light => 1,
        Kind::Refresh => 2,
    }
}

/// `Schedule.nextDepth`.
pub(crate) fn next_depth(kind: Kind, depth: u64) -> u64 {
    match kind {
        Kind::Light => depth + 1,
        Kind::Refresh => 0,
    }
}

/// What a step reads of its fold: the folded slot, the fresh commitment, the
/// PiCCS rounds and the fold's words.
struct Fold {
    folded: Slot,
    commitment: Vec<F>,
    rounds: Vec<[KWords; ROUND_COEFFICIENTS]>,
    words: FoldWords,
}

/// The words of a step from the prior state, the next current state, the
/// application witness and the fold. Returns the words and the output state.
fn words(
    prior: &BoundedDepthState,
    next: [F; 4],
    witness: &[F],
    kind: Kind,
    fold: Fold,
) -> (StepWords, BoundedDepthState) {
    let selected = prior.pc.saturating_sub(1) as usize;
    let mut inner = prior.slots.clone();
    inner[selected] = fold.folded;
    let next_slots = if prior.iteration == 0 {
        prior.slots.clone()
    } else {
        inner.clone()
    };
    let output = BoundedDepthState {
        iteration: prior.iteration + 1,
        current: next,
        pc: pc(kind),
        slots: next_slots.clone(),
        ..prior.clone()
    };
    let [prior_0, prior_1] = prior.slots.clone();
    let [inner_0, inner_1] = inner;
    let [next_0, next_1] = next_slots;
    let words = StepWords {
        iteration: prior.iteration,
        prior_pc: prior.pc,
        selected: selected as u64,
        keys: prior.keys,
        states: [prior.z0, prior.current, next],
        witness: witness.to_vec(),
        digests: [
            prior_0.digest(),
            prior_1.digest(),
            next_0.digest(),
            next_1.digest(),
            output.hash(),
        ],
        slots: [
            prior_0,
            prior_1,
            prior.slots[selected].clone(),
            inner_0,
            inner_1,
            next_0,
            next_1,
        ],
        fresh_commitment: fold.commitment,
        fresh_public_input: encode_digest(prior.hash()),
        rounds: fold.rounds,
        fold: fold.words,
    };
    (words, output)
}

/// The words of a fold step (iteration > 0): `proof` folded the previous
/// step's instance, committed as `fresh_commitment`, into slot `prior.pc − 1`,
/// giving `folded` at the next depth.
pub(crate) fn fold_step(
    prior: &BoundedDepthState,
    next: [F; 4],
    witness: &[F],
    kind: Kind,
    folded: &CeClaim,
    fresh_commitment: &[F],
    proof: &Proof,
) -> (StepWords, BoundedDepthState) {
    let depth = next_depth(kind, prior.slots[prior.pc as usize - 1].depth);
    let outputs: Vec<_> = proof.pi_ccs.outputs.iter().map(evaluations).collect();
    let words_of_fold = match kind {
        Kind::Light => FoldWords::Light {
            outputs: outputs.try_into().expect("a light fold has two outputs"),
        },
        Kind::Refresh => FoldWords::Refresh {
            children: proof
                .children
                .iter()
                .map(|child| Child {
                    commitment: child.c.data().to_vec(),
                    evaluations: evaluations(child),
                    digits: public_input(child),
                })
                .collect(),
            outputs,
        },
    };
    let rounds = proof
        .pi_ccs
        .sumcheck
        .sumcheck_rounds
        .iter()
        .map(|round| std::array::from_fn(|coefficient| k_words(round.get(coefficient).copied().unwrap_or(K::ZERO))))
        .collect();
    let fold = Fold {
        folded: slot(folded, depth),
        commitment: fresh_commitment.to_vec(),
        rounds,
        words: words_of_fold,
    };
    words(prior, next, witness, kind, fold)
}

/// The words of the base step (iteration 0) of the light circuit: the zero
/// proof of HyperNova's base case folds the default fresh instance, whose
/// public input is the hash of `prior`, into the default slot. Its rounds and
/// outputs are 0, so the folded slot keeps zero commitments and evaluations;
/// its point is the round point and its public input is the light challenge
/// times the fresh public input (`Light.fold`).
pub(crate) fn base_step(
    prior: &BoundedDepthState,
    next: [F; 4],
    witness: &[F],
) -> Result<(StepWords, BoundedDepthState), PackageError> {
    let fresh_public_input = encode_digest(prior.hash());
    let canonical: Vec<u64> = fresh_public_input
        .iter()
        .map(PrimeField64::as_canonical_u64)
        .collect();
    let transcript = derive_light_transcript(
        &[0; COMMITMENT_WORDS],
        &canonical,
        &vec![vec![[0u64; 2]; ROUND_COEFFICIENTS]; ROUNDS],
        &vec![0; 2 * SOURCE_WORDS],
    )?;
    let mut sampler = Poseidon2Transcript::from_state_and_absorbed(transcript.outgoing_state().map(F::from_u64), 0);
    let rho = sample_rot_rhos_n_typed(
        &mut sampler,
        &neo_params::NeoParams::nightstream_goldilocks_k16(),
        &RotRing::goldilocks(),
        1,
    )
    .map_err(|_| PackageError::Input("light challenge"))?
    .remove(0);
    let public_input = (0..PUBLIC_WORDS)
        .map(|column| {
            let (lane, block) = (column % D, column / D);
            (0..D).fold(F::ZERO, |sum, source| {
                sum + rho.as_mat()[(lane, source)] * fresh_public_input[block * D + source]
            })
        })
        .collect();
    let folded = Slot {
        point: transcript
            .round_point()
            .iter()
            .map(|value| value.map(F::from_u64))
            .collect(),
        public_input,
        depth: next_depth(Kind::Light, 0),
        ..Slot::default_slot()
    };
    let fold = Fold {
        folded,
        commitment: vec![F::ZERO; COMMITMENT_WORDS],
        rounds: vec![[[F::ZERO; 2]; ROUND_COEFFICIENTS]; ROUNDS],
        words: FoldWords::Light {
            outputs: [vec![[F::ZERO; 2]; 270], vec![[F::ZERO; 2]; 270]],
        },
    };
    Ok(words(prior, next, witness, Kind::Light, fold))
}
