//! Small helpers shared by the direct CCS IVC state and terminal circuit.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::{get_global_pp_for_dims, Commitment};
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;

use super::circuit_util::field_to_spartan;
use super::ivc::DirectCcsFPrimeSnarkError;
use crate::ivc::{SuperNeoIvcState, SuperNeoIvcTranscriptSnapshot};
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::claim::alloc_ce_claim;
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::superneo_nifs_circuit::SuperNeoClaimBundle;

pub(crate) fn validate_direct_ajtai_context(
    params: &NeoParams,
    structure: &CcsStructure<F>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let witness_cols = structure.m.div_ceil(D);
    let pp = get_global_pp_for_dims(D, witness_cols).map_err(|err| {
        DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS program requires a registered Ajtai PP for (d,m)=({D},{witness_cols}): {err}"
        ))
    })?;
    if pp.kappa != params.kappa as usize {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS Ajtai PP kappa mismatch for (d,m)=({D},{witness_cols}): registered {}, params {}",
            pp.kappa, params.kappa
        )));
    }
    Ok(())
}

pub(crate) fn alloc_initial_transcript<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    snapshot: Option<&SuperNeoIvcTranscriptSnapshot>,
) -> Result<Poseidon2TranscriptCircuit, SynthesisError> {
    match snapshot {
        Some(snapshot) => {
            let _ = cs;
            Poseidon2TranscriptCircuit::from_constant_state(snapshot.state.map(field_to_spartan), snapshot.absorbed)
        }
        None => Poseidon2TranscriptCircuit::new(cs.namespace(|| "session_transcript"), b"neo.fold.next/session"),
    }
}

pub(crate) fn alloc_initial_claim_bundle<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<SuperNeoClaimBundle, SynthesisError> {
    claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| {
            alloc_ce_claim(
                &mut cs.namespace(|| format!("initial_carry_claim_{idx}")),
                claim,
                &format!("initial_carry_claim_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map(SuperNeoClaimBundle::from_effective_claims)
}

pub(crate) fn superneo_ivc_states_match(left: &SuperNeoIvcState, right: &SuperNeoIvcState) -> bool {
    left.chunk_count == right.chunk_count
        && left.step_count == right.step_count
        && left.transcript == right.transcript
        && left.carry.claims == right.carry.claims
        && left.carry.witnesses == right.carry.witnesses
}
