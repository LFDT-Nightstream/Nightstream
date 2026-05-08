//! Owns direct-CCS terminal CE witness normalization and diagnostics.
//!
//! The terminal `F'` circuit keeps final post-DEC CE projections private and
//! proves their CE relation inline. This module owns the relation measurement
//! and witness-shape checks for that private terminal accumulator.

use bellpepper_core::ConstraintSystem;
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;

use super::ivc::DirectCcsFPrimeSnarkError;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanF};
use crate::superneo_circuit::ce_consistency::{
    debug_enforce_paper_ce_claim_consistency_with_breakdown, PaperCeRelationConstraintBreakdown,
};
use crate::superneo_circuit::claim::alloc_ce_claim;
use crate::superneo_circuit::witness::alloc_packed_witness;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DirectFinalCeRelationBreakdown {
    pub total_relation_constraints: usize,
    pub relation_breakdown: PaperCeRelationConstraintBreakdown,
}

pub(crate) fn final_carry_witnesses(zs: &[Mat<F>]) -> Result<Vec<CcsWitness<F>>, DirectCcsFPrimeSnarkError> {
    zs.iter()
        .enumerate()
        .map(|(idx, z)| {
            if z.rows() != D {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "final CE witness {idx} has {} rows, expected {D}",
                    z.rows()
                )));
            }
            Ok(CcsWitness {
                w: Vec::new(),
                Z: z.clone(),
            })
        })
        .collect()
}

pub(crate) fn measure_direct_final_ce_relation_breakdown(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
) -> Result<DirectFinalCeRelationBreakdown, DirectCcsFPrimeSnarkError> {
    if claims.len() != witnesses.len() {
        return Err(DirectCcsFPrimeSnarkError::Synthesis(
            "direct terminal final CE measurement requires one witness per claim".into(),
        ));
    }

    let mut cs = ShapeCS::<NeoFoldDeciderEngine>::new();
    let mut out = DirectFinalCeRelationBreakdown::default();
    for (claim_index, (claim, witness)) in claims.iter().zip(witnesses.iter()).enumerate() {
        let claim = alloc_ce_claim(
            &mut cs.namespace(|| format!("final_claim_{claim_index}")),
            claim,
            &format!("final_claim_{claim_index}"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let witness = alloc_packed_witness(
            &mut cs.namespace(|| format!("final_claim_{claim_index}_witness")),
            witness,
            &format!("final_claim_{claim_index}_witness"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let breakdown = debug_enforce_paper_ce_claim_consistency_with_breakdown(
            &mut cs,
            params,
            structure,
            structure,
            &witness,
            &claim,
            SpartanF::from_canonical_u64(7),
            &format!("final_claim_{claim_index}_ce_consistency"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        out.relation_breakdown.add_assign(breakdown);
    }
    out.total_relation_constraints = cs.num_constraints();
    Ok(out)
}
