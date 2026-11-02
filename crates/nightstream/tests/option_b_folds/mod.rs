//! Real option-B folds checked by the step circuits (Lean
//! `Spec/Folding/Light`, `Spec/Folding/Refresh`). Light steps 1 and 2 fold
//! the previous step's instance into slot 0. Refresh step 3 splits slot 0's
//! accumulator with PiDEC and folds step 2's instance with the 16 children.
//! Every row of each step's relation holds, on the source columns and
//! through the slot layout.

#[path = "../option_b_support/mod.rs"]
mod support;

use super::Relation;
use crate::folding::option_b::{prove, Kind, Proof};
use crate::folding::{pi_rlc, CeClaim, RunningInstance};
use neo_math::{KExtensions, D, F, K};
use nightstream_fprime::{
    derive_light_transcript, encode_digest, Child, FoldWords, KWords, OptionBState, Slot, StepWords, PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use support::{assert_every_row_holds, canonical, hash_chain_step, packages, words, StepZero};

const MINIMUM_SECURITY_BITS: u32 = 114;

fn k_words(value: K) -> KWords {
    value.as_coeffs()
}

/// A claim as the 270 `K` values a slot or a PiCCS output carries: `Eval_K`,
/// then the four `Eval_A` families.
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

fn slot(claim: &CeClaim, depth: u64) -> Slot {
    Slot {
        commitment: claim.c.data().to_vec(),
        evaluations: evaluations(claim),
        point: claim.r.iter().copied().map(k_words).collect(),
        public_input: public_input(claim),
        depth,
    }
}

/// HyperNova Definition 12's default running instance (`Step.defaultSlot`).
fn zero_accumulator(relation: &Relation) -> pi_rlc::Output {
    let mut zero = RunningInstance::canonical_zero(relation.params(), relation.structure(), PUBLIC_WORDS)
        .expect("zero accumulator");
    pi_rlc::Output {
        claim: zero.claims.swap_remove(0),
        witness: zero.witnesses.swap_remove(0),
    }
}

/// The program counter of the circuit that runs a fold kind.
fn pc(kind: Kind) -> u64 {
    match kind {
        Kind::Light => 1,
        Kind::Refresh => 2,
    }
}

/// Step `iteration` of the circuit that runs `kind`. The previous step ran
/// the light circuit, so the fold goes into slot 0, which carries `prior`;
/// slot 1 stays the default slot. Returns the words and the next slot 0.
#[allow(clippy::too_many_arguments)]
fn step(
    kind: Kind,
    zero: &StepZero,
    iteration: u64,
    states: [[F; 4]; 3],
    message: [F; 4],
    prior: &Slot,
    fresh_commitment: Vec<F>,
    next: &CeClaim,
    proof: &Proof,
) -> (StepWords, Slot) {
    let default = Slot::default_slot();
    let state = |iteration, current, pc, slot: &Slot| {
        OptionBState {
            keys: zero.keys,
            iteration,
            z0: zero.z0,
            current,
            pc,
            slots: [slot.clone(), default.clone()],
        }
        .hash()
    };
    let depth = match kind {
        Kind::Light => prior.depth + 1,
        Kind::Refresh => 0,
    };
    let folded = slot(next, depth);
    let output = state(iteration + 1, states[2], pc(kind), &folded);
    let outputs: Vec<_> = proof.pi_ccs.outputs.iter().map(evaluations).collect();
    let fold = match kind {
        Kind::Light => FoldWords::Light {
            outputs: outputs.try_into().expect("two light outputs"),
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
    let words = StepWords {
        iteration,
        prior_pc: pc(Kind::Light),
        selected: 0,
        keys: zero.keys,
        states,
        witness: message.to_vec(),
        slots: [
            prior.clone(),
            default.clone(),
            prior.clone(),
            folded.clone(),
            default.clone(),
            folded.clone(),
            default.clone(),
        ],
        digests: [
            prior.digest(),
            default.digest(),
            folded.digest(),
            default.digest(),
            output,
        ],
        fresh_commitment,
        fresh_public_input: encode_digest(state(iteration, states[1], pc(Kind::Light), prior)),
        rounds: proof
            .pi_ccs
            .sumcheck
            .sumcheck_rounds
            .iter()
            .map(|round| std::array::from_fn(|coefficient| k_words(round.get(coefficient).copied().unwrap_or(K::ZERO))))
            .collect(),
        fold,
    };
    (words, folded)
}

/// Step 1 folds step 0's instance into the default slot. Step 2 folds step
/// 1's instance into the accumulator that step 1 made, whose witness lies
/// outside {-1, 0, 1}, so the light rule that skips the running source's
/// norm is what lets it fold. Step 3 refreshes that accumulator.
#[test]
fn light_and_refresh_folds_hold_every_row() {
    let packages = packages();
    let light = Relation::new(packages.light(), MINIMUM_SECURITY_BITS).expect("light relation");
    let zero = StepZero::new(0);
    let transcript = derive_light_transcript(
        &[0; 1188],
        &canonical(&zero.fresh_public_input),
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 2 * 540],
    )
    .expect("light transcript");
    let fold = FoldWords::Light {
        outputs: [vec![[F::ZERO; 2]; 270], vec![[F::ZERO; 2]; 270]],
    };
    let mut logical = assert_every_row_holds(packages.light(), &zero.words(zero.folded(&transcript, 1, 1), fold));
    let mut accumulator = zero_accumulator(&light);
    let mut prior = Slot::default_slot();
    let mut current = zero.next;
    for (iteration, kind, message) in [
        (1, Kind::Light, [30, 31, 32, 33]),
        (2, Kind::Light, [40, 41, 42, 43]),
        (3, Kind::Refresh, [50, 51, 52, 53]),
    ] {
        if iteration == 2 {
            assert!(
                accumulator
                    .witness
                    .as_slice()
                    .iter()
                    .any(|value| !matches!(value.as_canonical_u64(), 0 | 1) && *value != -F::ONE),
                "the accumulator's witness leaves the digit range"
            );
        }
        let fresh = light.instance(&logical).expect("fresh instance");
        let fresh_commitment = fresh.claim.c.data().to_vec();
        let fresh_public_input = fresh.claim.x.clone();
        let (next, proof) = prove(kind, light.context(), fresh, accumulator).expect("fold");
        let message = words(message);
        let next_state = hash_chain_step(current, message);
        let (step_words, slot) = step(
            kind,
            &zero,
            iteration,
            [zero.z0, current, next_state],
            message,
            &prior,
            fresh_commitment,
            &next.claim,
            &proof,
        );
        assert_eq!(fresh_public_input, step_words.fresh_public_input);
        let package = match kind {
            Kind::Light => packages.light(),
            Kind::Refresh => packages.refresh(),
        };
        logical = assert_every_row_holds(package, &step_words);
        (accumulator, prior, current) = (next, slot, next_state);
    }
}
