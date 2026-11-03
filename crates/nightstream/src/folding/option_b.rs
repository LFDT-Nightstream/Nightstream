//! Option B's folds of the previous step's instance into one slot's
//! accumulator (Lean `Spec/Folding/Light`, `Spec/Folding/Refresh`). The light
//! fold runs PiCCS over the fresh instance and the accumulator, with norm
//! terms on the fresh source only, then PiRLC with weights ρ and 1. The
//! refresh fold splits the accumulator with PiDEC, binds the 16 children
//! after the statement, then runs PiCCS over the fresh instance and the
//! children and PiRLC with 17 sampled challenges. Neither fold ends in PiDEC.
use super::{
    ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsInstance, CeClaim, Error, FoldContext,
    RunningInstance,
};
use neo_math::{KExtensions, D, F};
use neo_reductions::{FoldProfile, NormSources};
use neo_transcript::domain_chunk_v1_2;
use nightstream_fprime::{LIGHT_DOMAIN, REFRESH_DOMAIN};

/// The fold a step circuit runs on the previous step's instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Light,
    Refresh,
}

impl Kind {
    /// The fold's PiCCS profile, which binds `children` after the statement,
    /// and its PiRLC weights.
    fn profile(self, children: &[F]) -> (FoldProfile<'_>, pi_rlc::Weights) {
        match self {
            Kind::Light => (
                FoldProfile {
                    domain: domain_chunk_v1_2(LIGHT_DOMAIN),
                    children: &[],
                    norm: NormSources::Fresh,
                },
                pi_rlc::Weights::Light,
            ),
            Kind::Refresh => (
                FoldProfile {
                    domain: domain_chunk_v1_2(REFRESH_DOMAIN),
                    children,
                    norm: NormSources::All,
                },
                pi_rlc::Weights::Sampled,
            ),
        }
    }
}

/// What the next step circuit reads of a fold besides the accumulators: the
/// PiCCS proof and, for a refresh fold, the PiDEC children.
#[derive(Clone)]
pub(crate) struct Proof {
    pub children: Vec<CeClaim>,
    pub pi_ccs: pi_ccs::Proof,
}

pub(crate) fn prove(
    kind: Kind,
    context: FoldContext<'_>,
    fresh: CcsInstance,
    accumulator: pi_rlc::Output,
) -> Result<(pi_rlc::Output, Proof), Error> {
    let (running, children) = match kind {
        Kind::Light => (
            RunningInstance::new(vec![accumulator.claim], vec![accumulator.witness]),
            Vec::new(),
        ),
        Kind::Refresh => {
            let (split, proof) = pi_dec::prove_with_production_key(context, &accumulator.claim, accumulator.witness)?;
            (RunningInstance::new(split.claims, split.witnesses), proof.children)
        }
    };
    let words = child_words(&children);
    let (profile, weights) = kind.profile(&words);
    let mut tr = Transcript::session();
    let CcsInstance { claim, witness } = fresh;
    let pi_ccs = pi_ccs::prove_from_parts_with_rows(&mut tr, context, &profile, &claim, &witness, &running)?;
    let sources: Vec<_> = std::iter::once(&witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (next, _) = pi_rlc::prove_refs(
        &mut tr,
        context.params,
        context.structure,
        ajtai_rlc_mixer,
        weights,
        &pi_ccs.outputs,
        &sources,
    )?;
    Ok((next, Proof { children, pi_ccs }))
}

/// Each child's commitment, then its `Eval_K` and `Eval_A` families as
/// coefficient pairs (Lean `OptionB.childWords`).
fn child_words(children: &[CeClaim]) -> Vec<F> {
    children
        .iter()
        .flat_map(|child| {
            let evaluations = std::iter::once(&child.eval_k)
                .chain(&child.eval_a)
                .flat_map(|family| family[..D].iter().flat_map(|value| value.as_coeffs()));
            child.c.data().iter().copied().chain(evaluations)
        })
        .collect()
}
