//! Allocates the terminal circuit's incoming SuperNeo carry state.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};

use super::gadgets::field_to_spartan;
use crate::ivc::SuperNeoIvcTranscriptSnapshot;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::claim::alloc_ce_claim;
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::superneo_nifs_circuit::SuperNeoClaimBundle;

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
