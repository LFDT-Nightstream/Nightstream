//! A `FinalProof` as backend words: what the final program reads.
//!
//! Input parsing only, with no authority. Fields that the program derives
//! (the fresh public input, the PiCCS outputs' instances and digests, the
//! PiRLC parent) are not read. With no proof every word is zero: a shape run.

use neo_math::{KExtensions, D, F, K};
use neo_spartan::{Backend, Error};
use nightstream_fprime::{
    PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT,
    PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::PrimeField64;

use super::k::Kw;
use crate::folding::{CcsClaim, CeClaim};
use crate::lifecycle::verify::{commitment_has_selected_shape, evaluation_has_selected_shape};
use crate::lifecycle::FinalProof;

/// The commitment rows of one claim.
pub(super) const ROWS: usize = neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS as usize;
/// Ring columns of a public input.
pub(super) const COLUMNS: usize = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D;

/// `Eval_K` and `Eval_A` of one claim.
pub(super) struct Evaluations<B: Backend> {
    pub(super) eval_k: [Kw<B>; D],
    pub(super) eval_a: Vec<[Kw<B>; D]>,
}

pub(super) struct FinalWords<B: Backend> {
    /// The running claims' commitment rows.
    pub(super) commitments: Vec<Vec<[B::F; D]>>,
    /// The running claims' public inputs, Lean column order (`j` is lane
    /// `j mod 54` of column `j / 54`).
    pub(super) digits: Vec<Vec<B::F>>,
    pub(super) evaluations: Vec<Evaluations<B>>,
    /// The running claims' shared point.
    pub(super) point: Vec<Kw<B>>,
    pub(super) fresh: Vec<[B::F; D]>,
    pub(super) rounds: Vec<Vec<Kw<B>>>,
    /// PiCCS outputs: the fresh source, then the running ones.
    pub(super) outputs: Vec<Evaluations<B>>,
}

fn shape(ok: bool, what: &'static str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error::Shape(what))
    }
}

fn evaluations_shape(claim: &CeClaim) -> bool {
    evaluation_has_selected_shape(&claim.eval_k)
        && claim.eval_a.len() == PI_CCS_V1_1_MATRIX_COUNT
        && claim
            .eval_a
            .iter()
            .all(|values| evaluation_has_selected_shape(values))
        && claim.adv.is_none()
}

fn check(proof: &FinalProof) -> Result<(), Error> {
    let running = &proof.running;
    shape(running.len() == PI_DEC_V1_1_CHILD_COUNT, "running claim count")?;
    let point = &running[0].r;
    shape(point.len() == PI_CCS_V1_1_ROUND_COUNT, "running point length")?;
    for claim in running {
        shape(
            commitment_has_selected_shape(&claim.c)
                && claim.m_in == PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
                && claim.X.rows() == D
                && claim.X.cols() == COLUMNS
                && claim.r == *point
                && evaluations_shape(claim),
            "running claim shape or shared point",
        )?;
    }
    let fresh: &CcsClaim = &proof.fresh;
    shape(
        commitment_has_selected_shape(&fresh.c)
            && fresh.m_in == PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
            && fresh.x.len() == PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
            && fresh.adv.is_none(),
        "fresh claim shape",
    )?;
    let rounds = &proof.pi_ccs.sumcheck.sumcheck_rounds;
    shape(
        rounds.len() == PI_CCS_V1_1_ROUND_COUNT
            && rounds
                .iter()
                .all(|round| round.len() == PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT),
        "PiCCS round shape",
    )?;
    let outputs = &proof.pi_ccs.outputs;
    shape(
        outputs.len() == PI_CCS_V1_1_SOURCE_COUNT && outputs.iter().all(evaluations_shape),
        "PiCCS output shape",
    )
}

impl<B: Backend> FinalWords<B> {
    pub(super) fn read(b: &mut B, proof: Option<&FinalProof>) -> Result<Self, Error> {
        if let Some(proof) = proof {
            check(proof)?;
        }
        let word = |b: &mut B, value: Option<F>| b.private(value.map_or(0, |value| value.as_canonical_u64()));
        let k = |b: &mut B, value: Option<K>| -> Kw<B> {
            let [re, im] = value.map_or([F::default(); 2], |value| value.as_coeffs());
            [word(b, Some(re)), word(b, Some(im))]
        };
        let rings = |b: &mut B, commitment: Option<&neo_ajtai::Commitment>| -> Vec<[B::F; D]> {
            (0..ROWS)
                .map(|row| std::array::from_fn(|lane| word(b, commitment.map(|c| c.col(row)[lane]))))
                .collect()
        };
        let evaluations = |b: &mut B, claim: Option<&CeClaim>| Evaluations {
            eval_k: std::array::from_fn(|lane| k(b, claim.map(|c| c.eval_k[lane]))),
            eval_a: (0..PI_CCS_V1_1_MATRIX_COUNT)
                .map(|matrix| std::array::from_fn(|lane| k(b, claim.map(|c| c.eval_a[matrix][lane]))))
                .collect(),
        };
        let running = |i: usize| proof.map(|proof| &proof.running[i]);
        let commitments = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|i| rings(b, running(i).map(|claim| &claim.c)))
            .collect();
        let digits = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|i| {
                (0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS)
                    .map(|j| word(b, running(i).map(|claim| claim.X[(j % D, j / D)])))
                    .collect()
            })
            .collect();
        let running_evaluations = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|i| evaluations(b, running(i)))
            .collect();
        let point = (0..PI_CCS_V1_1_ROUND_COUNT)
            .map(|t| k(b, running(0).map(|claim| claim.r[t])))
            .collect();
        let fresh = rings(b, proof.map(|proof| &proof.fresh.c));
        let rounds = (0..PI_CCS_V1_1_ROUND_COUNT)
            .map(|round| {
                (0..PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
                    .map(|i| k(b, proof.map(|proof| proof.pi_ccs.sumcheck.sumcheck_rounds[round][i])))
                    .collect()
            })
            .collect();
        let outputs = (0..PI_CCS_V1_1_SOURCE_COUNT)
            .map(|source| evaluations(b, proof.map(|proof| &proof.pi_ccs.outputs[source])))
            .collect();
        Ok(Self {
            commitments,
            digits,
            evaluations: running_evaluations,
            point,
            fresh,
            rounds,
            outputs,
        })
    }
}
