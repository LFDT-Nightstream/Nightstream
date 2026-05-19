//! Final CE consistency checks for the direct terminal circuit.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ccs::{CcsStructure, CcsWitness};
use neo_math::F;
use neo_params::NeoParams;

use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::ce_consistency::enforce_paper_ce_claim_consistency;
use crate::superneo_circuit::claim::CircuitCeClaim;
use crate::superneo_circuit::witness::alloc_packed_witness;

pub(crate) fn enforce_direct_terminal_final_ce_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CircuitCeClaim],
    witnesses: &[CcsWitness<F>],
) -> Result<(), SynthesisError> {
    if claims.len() != witnesses.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (claim, witness)) in claims.iter().zip(witnesses.iter()).enumerate() {
        let witness = alloc_packed_witness(
            &mut cs.namespace(|| format!("final_claim_{idx}_witness")),
            witness,
            &format!("final_claim_{idx}_witness"),
        )?;
        enforce_paper_ce_claim_consistency(
            &mut cs.namespace(|| format!("final_claim_{idx}_ce_consistency")),
            params,
            structure,
            structure,
            &witness,
            claim,
            SpartanF::from_canonical_u64(7),
            &format!("final_claim_{idx}_ce_consistency"),
        )?;
    }
    Ok(())
}
