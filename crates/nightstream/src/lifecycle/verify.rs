//! Selected terminal verification against an external state statement.
//! The exact running and fresh openings use the package relation and fixed
//! key. Parent caches, carried frame digests and redundant `w` storage are
//! non-authoritative; every witness check uses the complete matrix `Z`.

use neo_ajtai::{nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS, Commitment};
use neo_math::{D, F, K};
use neo_reductions::{
    common::{project_x_from_witness_mat, validate_fresh_witness_tail_zero},
    superneo_eval::{evaluate_terminal_rows, SuperneoCachedRelationError, SuperneoZBlocks},
    PiCcsError,
};
use nightstream_fprime::{PackageError, PI_CCS_V1_1_ROUND_COUNT, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    PiCcsV1_1PackageBridgeError, PreparedLifecycle, Stage1Envelope, Stage1State,
};

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("selected terminal commitment: {0}")]
    Commitment(#[from] crate::engine::EngineError),
    #[error("selected terminal device evaluation: {0}")]
    Device(#[source] PiCcsError),
    #[error("selected terminal statement: {0}")]
    Statement(&'static str),
    #[error("selected terminal running child {index}: {reason}")]
    Running { index: usize, reason: &'static str },
    #[error("selected terminal fresh opening: {0}")]
    Fresh(&'static str),
    #[error("selected terminal state hash: {0}")]
    StateHash(#[from] PiCcsV1_1PackageBridgeError),
    #[error("selected terminal package: {0}")]
    Package(#[from] PackageError),
    #[error("selected terminal running opening computation: {0}")]
    RunningOpenings(#[source] PiCcsError),
    #[error("selected terminal fresh CCS relation: {0}")]
    FreshRelation(#[source] SuperneoCachedRelationError),
}

impl PreparedLifecycle {
    /// Check the selected `Terminal.HoldsFor` boundary. The external state
    /// fixes the advertised endpoint. Active envelopes perform full CE and
    /// CCS opening checks, with no extra fold or caller-supplied evaluator.
    pub fn verify(&self, expected_state: &Stage1State, envelope: &Stage1Envelope) -> Result<(), VerifyError> {
        if envelope.state() != expected_state {
            return Err(VerifyError::Statement("envelope differs from the external state"));
        }
        if expected_state.iteration() >= F::ORDER_U64 {
            return Err(VerifyError::Statement(
                "iteration is not a canonical Goldilocks counter",
            ));
        }
        if envelope.is_initial() {
            if expected_state.iteration() != 0 || expected_state.current() != expected_state.z0() {
                return Err(VerifyError::Statement(
                    "bottom requires zero iterations and equal endpoints",
                ));
            }
            return Ok(());
        }
        if expected_state.iteration() == 0 {
            return Err(VerifyError::Statement("an active proof requires a positive iteration"));
        }
        let running = envelope
            .running()
            .ok_or(VerifyError::Statement("missing running payload"))?;
        let fresh = envelope
            .fresh()
            .ok_or(VerifyError::Statement("missing fresh payload"))?;
        if running.claims.len() != PI_DEC_V1_1_CHILD_COUNT || running.witnesses.len() != PI_DEC_V1_1_CHILD_COUNT {
            return Err(VerifyError::Statement(
                "running claim or witness count differs from the selected profile",
            ));
        }
        let public_width = self.package.logical_public_input_count();
        let point = &running.claims[0].r;
        for (index, (claim, witness)) in running.claims.iter().zip(&running.witnesses).enumerate() {
            if claim.adv.is_some() {
                return Err(VerifyError::Running {
                    index,
                    reason: "plain claims cannot carry auxiliary commitments",
                });
            }
            if !commitment_has_selected_shape(&claim.c) || claim.m_in != public_width {
                return Err(VerifyError::Running {
                    index,
                    reason: "commitment or public-input shape",
                });
            }
            if claim.r.len() != PI_CCS_V1_1_ROUND_COUNT || claim.r.as_slice() != point.as_slice() {
                return Err(VerifyError::Running {
                    index,
                    reason: "running claims must share the selected evaluation point",
                });
            }
            if !evaluation_has_selected_shape(&claim.eval_k)
                || claim.eval_a.len() != self.structure.t()
                || claim
                    .eval_a
                    .iter()
                    .any(|values| !evaluation_has_selected_shape(values))
            {
                return Err(VerifyError::Running {
                    index,
                    reason: "evaluation shape or nonzero surplus coefficients",
                });
            }
            let projected = project_x_from_witness_mat(witness, self.structure.m, claim.m_in).map_err(|_| {
                VerifyError::Running {
                    index,
                    reason: "complete witness shape",
                }
            })?;
            if projected != claim.X {
                return Err(VerifyError::Running {
                    index,
                    reason: "witness public projection differs from X",
                });
            }
        }
        if fresh.claim.adv.is_some() {
            return Err(VerifyError::Fresh("plain claims cannot carry auxiliary commitments"));
        }
        if !commitment_has_selected_shape(&fresh.claim.c)
            || fresh.claim.m_in != public_width
            || fresh.claim.x.len() != public_width
        {
            return Err(VerifyError::Fresh("commitment or public-input shape"));
        }

        // The formal terminal preimage contains the semantic running claims,
        // not parent_authority or fold_digest. The single selected pc is one.
        let preimage = serialize_pi_ccs_v1_1_state_preimage(
            self.binding.verifier_context().digest().map(F::from_u64),
            expected_state.iteration(),
            expected_state.z0(),
            expected_state.current(),
            &running.claims,
            1,
        )?;
        let public = encode_pi_ccs_v1_1_public_input(pi_ccs_v1_1_state_hash(&preimage)?)?;
        if fresh
            .claim
            .x
            .iter()
            .zip(&public)
            .any(|(actual, expected)| actual.as_canonical_u64() != *expected)
            || fresh.claim.x.len() != public.len()
        {
            return Err(VerifyError::Fresh(
                "public input differs from the recomputed terminal state hash",
            ));
        }
        validate_fresh_witness_tail_zero(&fresh.witness.Z, self.structure.m, "selected terminal")
            .map_err(|_| VerifyError::Fresh("complete witness shape or nonzero fresh completion tail"))?;
        for (column, value) in fresh.claim.x.iter().enumerate() {
            if fresh.witness.Z[(column % D, column / D)] != *value {
                return Err(VerifyError::Fresh("witness public projection differs from x"));
            }
        }
        let commitments = self.backend.commit(&running.witnesses)?;
        for (index, (claim, commitment)) in running.claims.iter().zip(commitments).enumerate() {
            if commitment != claim.c {
                return Err(VerifyError::Running {
                    index,
                    reason: "fixed-key commitment differs from the witness",
                });
            }
        }
        let commitment = self
            .backend
            .commit(std::slice::from_ref(&fresh.witness.Z))?
            .remove(0);
        if commitment != fresh.claim.c {
            return Err(VerifyError::Fresh("fixed-key commitment differs from the witness"));
        }

        let rows = self.matrix_rows();
        let workspace_bytes = self.matrix_workspace_bytes()?;
        let evaluated = match &self.backend {
            #[cfg(feature = "metal")]
            crate::engine::Backend::Metal(device) => device
                .lock()
                .map_err(|_| device_lock_error())?
                .evaluate_terminal_rows(
                    &rows,
                    workspace_bytes,
                    &self.structure,
                    &running.witnesses,
                    point,
                    &fresh.witness.Z,
                )
                .map_err(VerifyError::Device)?,
            _ => {
                let blocks = running
                    .witnesses
                    .iter()
                    .enumerate()
                    .map(|(index, witness)| {
                        SuperneoZBlocks::from_witness_mat(witness, self.structure.m).map_err(|_| VerifyError::Running {
                            index,
                            reason: "complete witness block conversion",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let fresh_blocks = SuperneoZBlocks::from_witness_mat(&fresh.witness.Z, self.structure.m)
                    .map_err(|_| VerifyError::Fresh("complete witness block conversion"))?;
                evaluate_terminal_rows(&rows, point, &blocks, &self.structure.f, &fresh_blocks, workspace_bytes)
                    .map_err(VerifyError::RunningOpenings)?
            }
        };
        if let Some(row) = evaluated.first_unsatisfied_row {
            return Err(VerifyError::FreshRelation(
                SuperneoCachedRelationError::UnsatisfiedRow { row },
            ));
        }
        let openings = evaluated.openings;
        for (index, claim) in running.claims.iter().enumerate() {
            let actual = openings.get(index).ok_or(VerifyError::Running {
                index,
                reason: "missing complete witness opening",
            })?;
            if !evaluation_matches(&claim.eval_k, &actual.eval_k) {
                return Err(VerifyError::Running {
                    index,
                    reason: "Eval_K differs from the complete witness opening",
                });
            }
            if actual.eval_a.len() != self.structure.t()
                || claim
                    .eval_a
                    .iter()
                    .zip(&actual.eval_a)
                    .any(|(recorded, expected)| !evaluation_matches(recorded, expected))
            {
                return Err(VerifyError::Running {
                    index,
                    reason: "Eval_A differs from the complete witness openings",
                });
            }
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
fn device_lock_error() -> VerifyError {
    VerifyError::Device(PiCcsError::BackendFailure {
        backend: "metal",
        reason: "terminal device session lock was poisoned".into(),
    })
}

fn commitment_has_selected_shape(commitment: &Commitment) -> bool {
    let rows = PRODUCTION_VERIFIER_ROWS as usize;
    commitment.d == D && commitment.kappa == rows && commitment.data.len() == D * rows
}

fn evaluation_has_selected_shape(values: &[K]) -> bool {
    values.len() >= D && values[D..].iter().all(|value| *value == K::ZERO)
}

fn evaluation_matches(recorded: &[K], expected: &[K]) -> bool {
    evaluation_has_selected_shape(recorded) && expected.len() == D && recorded[..D] == *expected
}
