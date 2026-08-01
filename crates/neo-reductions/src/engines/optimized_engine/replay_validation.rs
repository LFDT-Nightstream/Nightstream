//! Terminal replay checks for the optimized Π_CCS prover.
//!
//! Owns only host-side consistency validation after a replay/terminal-state
//! pass. The prover flow and transcript scheduling stay in `prove.rs`.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;

use super::legacy_types::PiCcsReplayTerminalState;
use crate::engines::utils;
use crate::error::PiCcsError;

pub(super) fn validate_replay_terminal_state(
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    replay: &PiCcsReplayTerminalState,
) -> Result<(), PiCcsError> {
    utils::validate_me_outputs_against_inputs(
        s,
        params,
        fresh_claims,
        me_inputs,
        &replay.me_outputs,
        &replay.row_chals,
        &replay.s_col,
    )?;
    let r_inputs = utils::shared_me_input_r(me_inputs, replay.row_chals.len())?;
    let rhs_fe = super::rhs_terminal_identity_fe_with_k_mcs(
        s,
        params,
        &replay.challenges_public,
        &replay.row_chals,
        &replay.alpha_prime,
        &replay.me_outputs,
        fresh_claims.len(),
        r_inputs,
    );
    if replay.sumcheck_final != rhs_fe {
        return Err(PiCcsError::ProtocolError(
            "optimized replay FE terminal state does not match relation identity".into(),
        ));
    }

    let rhs_nc = super::rhs_terminal_identity_nc(
        params,
        &replay.challenges_public,
        &replay.s_col,
        &replay.alpha_prime_nc,
        &replay.me_outputs,
    );
    if replay.sumcheck_final_nc != rhs_nc {
        return Err(PiCcsError::ProtocolError(
            "optimized replay NC terminal state does not match relation identity".into(),
        ));
    }

    Ok(())
}
