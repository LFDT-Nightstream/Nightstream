//! Fixed Construction-2 interface `CE^k x CCS -> CE^k`.
//!
//! Owns: the validated fixed-size accumulator wrapper and one-fresh-instance
//! prove/verify entrypoints.
//!
//! Does not own: Pi_CCS, Pi_RLC, or Pi_DEC internals, transcript primitives, or
//! backend execution.
//!
//! Emits constraints: no.
//!
//! Authority boundary: constructors validate accumulator shape and checked
//! parent recomposition; only [`verify_fixed`] authorizes a verifier-side output.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Fixed accumulator | [`FixedNifsAccumulator`] | no | Shape and parent validation |
//! | Prover fold | [`prove_fixed`] | no | Validated input accumulator and fresh instance |
//! | Verifier fold | [`verify_fixed`] | no | Checked NIFS proof |

use neo_ajtai::AjtaiSModule;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, DecMixer, RlcMixer, Structure};
use crate::paper::{nifs, pi_dec};

/// Validated fixed-shape `R1 = CE(b,L)^k` accumulator.
///
/// The formal instance is [`Self::claims`]. The wrapped
/// [`RunningInstance::parent_authority`] is a verifier-checked, deterministic
/// Π_DEC recomposition cache used by the optimized transcript, not an extra
/// formal accumulator coordinate.
#[derive(Clone, Debug)]
pub struct FixedNifsAccumulator {
    running: RunningInstance,
}

impl FixedNifsAccumulator {
    pub fn canonical_zero(
        pp: &Params,
        structure: &Structure,
        combine_b_pows: DecMixer,
        m_in: usize,
    ) -> Result<Self, Error> {
        let running = RunningInstance::canonical_zero(pp, structure, m_in)?;
        Self::from_prover_running(pp, structure, combine_b_pows, running)
    }

    pub fn from_prover_running(
        pp: &Params,
        structure: &Structure,
        combine_b_pows: DecMixer,
        running: RunningInstance,
    ) -> Result<Self, Error> {
        validate_fixed_shape(pp, &running, WitnessMode::Prover)?;
        validate_parent(pp, structure, combine_b_pows, &running)?;
        Ok(Self { running })
    }

    pub fn from_verifier_running(
        pp: &Params,
        structure: &Structure,
        combine_b_pows: DecMixer,
        running: RunningInstance,
    ) -> Result<Self, Error> {
        validate_fixed_shape(pp, &running, WitnessMode::Verifier)?;
        validate_parent(pp, structure, combine_b_pows, &running)?;
        Ok(Self { running })
    }

    pub fn claims(&self) -> &[crate::paper::relations::CeClaim] {
        self.running.formal_claims()
    }

    pub fn running(&self) -> &RunningInstance {
        &self.running
    }

    pub fn into_running(self) -> RunningInstance {
        self.running
    }
}

/// Canonical Construction-2 prover call: one fresh instance, one fixed-k
/// accumulator, one fixed-k output accumulator.
#[allow(clippy::too_many_arguments)]
pub fn prove_fixed(
    tr: &mut Transcript,
    pp: &Params,
    structure: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh: CcsInstance,
    running: &FixedNifsAccumulator,
) -> Result<(FixedNifsAccumulator, NifsProof), Error> {
    let (next, proof) = nifs::prove(
        tr,
        pp,
        structure,
        cache,
        log,
        None,
        mix_rhos_commits,
        combine_b_pows,
        vec![fresh],
        running.running(),
    )?;
    Ok((
        FixedNifsAccumulator::from_prover_running(pp, structure, combine_b_pows, next)?,
        proof,
    ))
}

/// Canonical Construction-2 verifier call. The returned accumulator contains
/// public claims only; witnesses never cross the verifier boundary.
#[allow(clippy::too_many_arguments)]
pub fn verify_fixed(
    tr: &mut Transcript,
    pp: &Params,
    structure: &Structure,
    cache: &OptimizedStructureCache,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh: &CcsClaim,
    running: &FixedNifsAccumulator,
    proof: &NifsProof,
) -> Result<FixedNifsAccumulator, Error> {
    let next = nifs::verify(
        tr,
        pp,
        structure,
        cache,
        mix_rhos_commits,
        combine_b_pows,
        std::slice::from_ref(fresh),
        running.running(),
        proof,
    )?;
    FixedNifsAccumulator::from_verifier_running(pp, structure, combine_b_pows, next)
}

#[derive(Clone, Copy)]
enum WitnessMode {
    Prover,
    Verifier,
}

fn validate_fixed_shape(pp: &Params, running: &RunningInstance, mode: WitnessMode) -> Result<(), Error> {
    let expected = pp.k_rho() as usize;
    if running.claims.len() != expected {
        return Err(Error::FixedShape {
            what: "running CE claim count",
            expected,
            got: running.claims.len(),
        });
    }
    let expected_witnesses = match mode {
        WitnessMode::Prover => expected,
        WitnessMode::Verifier => 0,
    };
    if running.witnesses.len() != expected_witnesses {
        return Err(Error::FixedShape {
            what: "running CE witness count",
            expected: expected_witnesses,
            got: running.witnesses.len(),
        });
    }
    if running.parent_authority.is_none() {
        return Err(Error::FixedShape {
            what: "derived decomposition parent count",
            expected: 1,
            got: 0,
        });
    }
    Ok(())
}

fn validate_parent(
    pp: &Params,
    structure: &Structure,
    combine_b_pows: DecMixer,
    running: &RunningInstance,
) -> Result<(), Error> {
    let parent = running
        .decomposition_parent()
        .expect("fixed shape validated parent presence");
    pi_dec::verify(
        pp,
        structure,
        combine_b_pows,
        parent,
        &pi_dec::Proof {
            children: running.claims.clone(),
        },
    )?;
    Ok(())
}
