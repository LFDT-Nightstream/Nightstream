//! Lean `Spec/Folding/Light`, `Spec/Folding/Refresh`.
use super::{
    ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsInstance, CeClaim, Error, FoldContext,
    RunningInstance,
};
use crate::engine::Backend;
use neo_math::{KExtensions, D, F};
use neo_reductions::{FoldProfile, NormSources};
use neo_transcript::domain_chunk_v1_2;
use nightstream_fprime::{LIGHT_DOMAIN, REFRESH_DOMAIN};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Light,
    Refresh,
}

impl Kind {
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

#[derive(Clone)]
pub(crate) struct Proof {
    pub children: Vec<CeClaim>,
    pub pi_ccs: pi_ccs::Proof,
}

pub(crate) fn prove(
    kind: Kind,
    backend: &Backend,
    context: FoldContext<'_>,
    fresh: &CcsInstance,
    accumulator: pi_rlc::Output,
) -> Result<(pi_rlc::Output, Proof), Error> {
    let (running, children) = match kind {
        Kind::Light => (
            RunningInstance::new(vec![accumulator.claim], vec![accumulator.witness]),
            Vec::new(),
        ),
        Kind::Refresh => {
            let (split, proof) =
                pi_dec::prove_with_production_key(backend, context, &accumulator.claim, accumulator.witness)?;
            (RunningInstance::new(split.claims, split.witnesses), proof.children)
        }
    };
    let words = child_words(&children);
    let (profile, weights) = kind.profile(&words);
    let mut tr = Transcript::session();
    let CcsInstance { claim, witness } = fresh;
    let pi_ccs = pi_ccs::prove_from_parts_with_rows(&mut tr, backend, context, &profile, claim, witness, &running)?;
    let sources: Vec<_> = std::iter::once(&witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (next, _) = pi_rlc::prove_refs(
        &mut tr,
        backend,
        context.params,
        context.structure,
        ajtai_rlc_mixer,
        weights,
        &pi_ccs.outputs,
        &sources,
    )?;
    Ok((next, Proof { children, pi_ccs }))
}

/// Lean `BoundedDepth.childWords`.
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
