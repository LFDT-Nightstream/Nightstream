//! Owns core CE-consistency gadgets for SuperNeo recursive circuits.
//!
//! This module mirrors the native `neo_ccs::check_ce_consistency` boundary:
//! `c = L(Z)`, `X = L_x(Z)`, optional `y_zcol = Z_digits·chi(s_col)`,
//! `y_ring = \widehat{\bar{M}_j z}(r)` in SuperNeo ring-coefficient form,
//! `ct[j] = y_ring[j][0]`, and balanced digit representability for each
//! packed witness coefficient.

mod commitment;
mod evaluation;

pub use commitment::{enforce_ajtai_commitment_consistency, enforce_ajtai_commitment_linear_consistency};
pub(crate) use evaluation::debug_paper_dec_child_y_ring_formula_mismatch;
use evaluation::{chi_table_var, enforce_claim_y_ring_from_point_var, enforce_claim_y_zcol_from_digits_var};

use crate::spartan_backend::{Rv32imDeciderEngine, ShapeCS, SpartanF};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ajtai::{get_global_pp_for_dims, precompute_rot_columns};
use neo_ccs::{tensor_point, CcsMatrix, CcsStructure, CcsWitness, CeClaim};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{validate_superneo_witness_mat, witness_mat_get_f};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::claim::CircuitCeClaim;
use super::k_field::{alloc_constant_k, enforce_k_eq, k_add, k_base_mul_var, k_mul, k_scalar_mul, KNum, KNumVar};
use super::witness::{
    alloc_balanced_digit_witness, enforce_balanced_digit_alphabet, enforce_x_projection, BalancedDigitWitnessVar,
    PackedWitnessVar,
};

pub fn enforce_ce_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs, params, structure, structure, witness, claim, delta, true, true, label,
    )
}

pub fn enforce_ce_consistency_without_x<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs, params, structure, structure, witness, claim, delta, true, false, label,
    )
}

/// Opens a final carried CE claim against the SuperNeo paper relation.
///
/// This is the `R1` check used by the compressed IVC verifier: `c = L(Z)`,
/// `x = L_in(Z)`, `||Z|| < b`, and `y_j = \widehat{M_j Z}(r)`. It deliberately
/// does not authorize backend transport fields such as `s_col`, `y_zcol`, or
/// `ct`; those are verifier-replay cargo, not part of the paper CE relation.
pub fn enforce_paper_ce_claim_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment_consistency(
        &mut cs.namespace(|| format!("{label}_commitment")),
        witness,
        claim,
        &format!("{label}_commitment"),
    )?;
    enforce_x_projection(
        &mut cs.namespace(|| format!("{label}_x_projection")),
        witness,
        claim,
        base_structure.m,
        &format!("{label}_x"),
    )?;
    enforce_balanced_digit_alphabet(
        &mut cs.namespace(|| format!("{label}_digits")),
        witness,
        base_structure.m,
        params,
        &format!("{label}_digits"),
    )?;

    let (chi_r, chi_r_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_r")),
        &claim.openings.r,
        &claim.openings.r_values,
        delta,
        &format!("{label}_chi_r"),
    )?;
    for (matrix_idx, matrix) in ring_structure.matrices.iter().enumerate() {
        enforce_claim_y_ring_from_point_var(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            ring_structure.n,
            matrix,
            &chi_r,
            &chi_r_values,
            D,
            &claim.openings.y_ring[matrix_idx],
            delta,
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaperCeRelationConstraintBreakdown {
    pub commitment: usize,
    pub x_projection: usize,
    pub norm: usize,
    pub y_eval: usize,
}

impl PaperCeRelationConstraintBreakdown {
    pub fn add_assign(&mut self, other: Self) {
        self.commitment += other.commitment;
        self.x_projection += other.x_projection;
        self.norm += other.norm;
        self.y_eval += other.y_eval;
    }

    pub fn total(&self) -> usize {
        self.commitment + self.x_projection + self.norm + self.y_eval
    }
}

pub fn debug_enforce_paper_ce_claim_consistency_with_breakdown(
    cs: &mut ShapeCS<Rv32imDeciderEngine>,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<PaperCeRelationConstraintBreakdown, SynthesisError> {
    let mut breakdown = PaperCeRelationConstraintBreakdown::default();

    let before = cs.num_constraints();
    enforce_ajtai_commitment_consistency(
        &mut cs.namespace(|| format!("{label}_commitment")),
        witness,
        claim,
        &format!("{label}_commitment"),
    )?;
    breakdown.commitment = cs.num_constraints() - before;

    let before = cs.num_constraints();
    enforce_x_projection(
        &mut cs.namespace(|| format!("{label}_x_projection")),
        witness,
        claim,
        base_structure.m,
        &format!("{label}_x"),
    )?;
    breakdown.x_projection = cs.num_constraints() - before;

    let before = cs.num_constraints();
    enforce_balanced_digit_alphabet(
        &mut cs.namespace(|| format!("{label}_digits")),
        witness,
        base_structure.m,
        params,
        &format!("{label}_digits"),
    )?;
    breakdown.norm = cs.num_constraints() - before;

    let before = cs.num_constraints();
    let (chi_r, chi_r_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_r")),
        &claim.openings.r,
        &claim.openings.r_values,
        delta,
        &format!("{label}_chi_r"),
    )?;
    for (matrix_idx, matrix) in ring_structure.matrices.iter().enumerate() {
        enforce_claim_y_ring_from_point_var(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            ring_structure.n,
            matrix,
            &chi_r,
            &chi_r_values,
            D,
            &claim.openings.y_ring[matrix_idx],
            delta,
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
    }
    breakdown.y_eval = cs.num_constraints() - before;
    Ok(breakdown)
}

pub fn enforce_backend_claim_consistency_with_x<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs,
        params,
        base_structure,
        ring_structure,
        witness,
        claim,
        delta,
        true,
        true,
        label,
    )
}

pub fn enforce_backend_claim_consistency_without_x<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs,
        params,
        base_structure,
        ring_structure,
        witness,
        claim,
        delta,
        true,
        false,
        label,
    )
}

pub fn enforce_output_claim_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs,
        params,
        base_structure,
        ring_structure,
        witness,
        claim,
        delta,
        false,
        false,
        label,
    )
}

/// Opens a DEC child only against the paper `CE(b, L)` projection.
///
/// The backend `y_zcol` and `ct` channels remain replay-owned, while `s_col`
/// is now treated as non-authoritative transport shell outside the paper CE
/// surface. The explicit witness-opening layer only proves the child has
/// paper-level commitment, norm, and `y_ring` semantics over the true ring
/// degree `D`. Any padded backend tail beyond `D` remains replay-owned
/// convenience structure.
pub fn enforce_paper_dec_child_claim_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment_consistency(
        &mut cs.namespace(|| format!("{label}_commitment")),
        witness,
        claim,
        &format!("{label}_commitment"),
    )?;
    enforce_balanced_digit_alphabet(
        &mut cs.namespace(|| format!("{label}_digits")),
        witness,
        base_structure.m,
        params,
        &format!("{label}_digits"),
    )?;

    let (chi_r, chi_r_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_r")),
        &claim.openings.r,
        &claim.openings.r_values,
        delta,
        &format!("{label}_chi_r"),
    )?;
    for (matrix_idx, matrix) in ring_structure.matrices.iter().enumerate() {
        enforce_claim_y_ring_from_point_var(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            ring_structure.n,
            matrix,
            &chi_r,
            &chi_r_values,
            D,
            &claim.openings.y_ring[matrix_idx],
            delta,
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
    }
    Ok(())
}

fn enforce_backend_claim_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    delta: SpartanF,
    check_commitment: bool,
    check_x: bool,
    label: &str,
) -> Result<(), SynthesisError> {
    if check_commitment {
        enforce_ajtai_commitment_consistency(
            &mut cs.namespace(|| format!("{label}_commitment")),
            witness,
            claim,
            &format!("{label}_commitment"),
        )?;
    }

    if check_x {
        enforce_x_projection(
            &mut cs.namespace(|| format!("{label}_x_projection")),
            witness,
            claim,
            base_structure.m,
            &format!("{label}_x"),
        )?;
    }

    if !(claim.norm_check.s_col.is_empty() && claim.norm_check.y_zcol.is_empty()) {
        let (chi_s, chi_s_values) = chi_table_var(
            &mut cs.namespace(|| format!("{label}_chi_s")),
            &claim.norm_check.s_col,
            &claim.norm_check.s_col_values,
            delta,
            &format!("{label}_chi_s"),
        )?;
        let digit_witness = alloc_balanced_digit_witness(
            &mut cs.namespace(|| format!("{label}_digits")),
            witness,
            base_structure.m,
            params,
            delta,
            &format!("{label}_digits"),
        )?;
        enforce_claim_y_zcol_from_digits_var(
            &mut cs.namespace(|| format!("{label}_y_zcol")),
            &digit_witness,
            base_structure.m,
            &chi_s,
            &chi_s_values,
            &claim.norm_check.y_zcol,
            delta,
            &format!("{label}_y_zcol"),
        )?;
    }

    let (chi_r, chi_r_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_r")),
        &claim.openings.r,
        &claim.openings.r_values,
        delta,
        &format!("{label}_chi_r"),
    )?;
    for (matrix_idx, matrix) in ring_structure.matrices.iter().enumerate() {
        enforce_claim_y_ring_from_point_var(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            ring_structure.n,
            matrix,
            &chi_r,
            &chi_r_values,
            D,
            &claim.openings.y_ring[matrix_idx],
            delta,
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
        enforce_k_eq(
            &mut cs.namespace(|| format!("{label}_ct_{matrix_idx}")),
            claim
                .openings
                .ct
                .get(matrix_idx)
                .ok_or(SynthesisError::Unsatisfiable)?,
            claim.openings.y_ring[matrix_idx]
                .first()
                .ok_or(SynthesisError::Unsatisfiable)?,
            &format!("{label}_ct_{matrix_idx}"),
        );
    }
    Ok(())
}
