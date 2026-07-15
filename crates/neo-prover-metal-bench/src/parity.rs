//! Canonical CPU/Metal audit-authority comparison for benchmark gates.

use neo_fold_clean::lifecycle::UncompressedAudit;
use neo_fold_clean::paper::construction2::{FinalFoldProof, ProofState, RunningInstance, StepProof};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance};

pub(crate) fn audit_authority_eq(left: &UncompressedAudit, right: &UncompressedAudit) -> bool {
    state_eq(&left.proof.state, &right.proof.state)
        && optional_final_fold_eq(left.proof.final_fold.as_ref(), right.proof.final_fold.as_ref())
        && left.steps.len() == right.steps.len()
        && left
            .steps
            .iter()
            .zip(&right.steps)
            .all(|(left, right)| step_eq(left, right))
        && public_batches_eq(&left.public_batches, &right.public_batches)
}

fn state_eq(
    left: &neo_fold_clean::paper::construction2::State,
    right: &neo_fold_clean::paper::construction2::State,
) -> bool {
    left.chunk_count == right.chunk_count
        && left.step_count == right.step_count
        && left.z_0 == right.z_0
        && left.z_i == right.z_i
        && left.initial_semantic_state_digest == right.initial_semantic_state_digest
        && left.semantic_state_digest == right.semantic_state_digest
        && left.pc == right.pc
        && left.acc_digest == right.acc_digest
        && left.public_trace == right.public_trace
        && left.nebula == right.nebula
        && proof_state_eq(&left.proof, &right.proof)
}

fn proof_state_eq(left: &ProofState, right: &ProofState) -> bool {
    match (left, right) {
        (ProofState::Initial, ProofState::Initial) => true,
        (
            ProofState::Active {
                running: left_running,
                latest: left_latest,
            },
            ProofState::Active {
                running: right_running,
                latest: right_latest,
            },
        ) => {
            let Ok(left_running) = left_running.materialize() else {
                return false;
            };
            let Ok(right_running) = right_running.materialize() else {
                return false;
            };
            running_eq(&left_running, &right_running) && instances_eq(&left_latest.instances, &right_latest.instances)
        }
        _ => false,
    }
}

fn step_eq(left: &StepProof, right: &StepProof) -> bool {
    if left.nebula_open != right.nebula_open
        || left.semantic_state_digest != right.semantic_state_digest
        || left.x_out != right.x_out
    {
        return false;
    }
    match (left.fold.materialized_recursive(), right.fold.materialized_recursive()) {
        (Ok(None), Ok(None)) => true,
        (Ok(Some(left)), Ok(Some(right))) => nifs_eq(&left, &right),
        _ => false,
    }
}

fn optional_final_fold_eq(left: Option<&FinalFoldProof>, right: Option<&FinalFoldProof>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.x_out == right.x_out
                && nifs_eq(&left.nifs, &right.nifs)
                && running_eq(
                    &left.terminal_inputs.pre_final_running,
                    &right.terminal_inputs.pre_final_running,
                )
                && instances_eq(
                    &left.terminal_inputs.latest.instances,
                    &right.terminal_inputs.latest.instances,
                )
                && left.terminal_inputs.pre_nebula == right.terminal_inputs.pre_nebula
        }
        _ => false,
    }
}

fn nifs_eq(left: &NifsProof, right: &NifsProof) -> bool {
    left.pi_ccs.outputs == right.pi_ccs.outputs
        && left.pi_ccs.outputs_digest == right.pi_ccs.outputs_digest
        && sumcheck_eq(&left.pi_ccs.sumcheck, &right.pi_ccs.sumcheck)
        && left.pi_rlc.combined == right.pi_rlc.combined
        && left.pi_dec.children == right.pi_dec.children
}

fn sumcheck_eq(
    left: &neo_fold_clean::paper::pi_ccs::SumcheckProof,
    right: &neo_fold_clean::paper::pi_ccs::SumcheckProof,
) -> bool {
    match (serde_json::to_vec(left), serde_json::to_vec(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn running_eq(left: &RunningInstance, right: &RunningInstance) -> bool {
    left.claims == right.claims && left.witnesses == right.witnesses && left.parent_authority == right.parent_authority
}

fn instances_eq(left: &[CcsInstance], right: &[CcsInstance]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            claim_eq(&left.claim, &right.claim)
                && left.witness.w == right.witness.w
                && left.witness.Z == right.witness.Z
        })
}

fn public_batches_eq(left: &[Vec<CcsClaim>], right: &[Vec<CcsClaim>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| claim_eq(left, right))
        })
}

fn claim_eq(left: &CcsClaim, right: &CcsClaim) -> bool {
    left.c == right.c && left.x == right.x && left.m_in == right.m_in && left.adv == right.adv
}
