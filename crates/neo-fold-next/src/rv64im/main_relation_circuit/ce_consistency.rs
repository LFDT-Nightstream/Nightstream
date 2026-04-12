//! Owns core CE-consistency gadgets for the RV64IM main relation circuit.
//!
//! This module mirrors the native `neo_ccs::check_ce_consistency` boundary:
//! `c = L(Z)`, `X = L_x(Z)`, `y_zcol = Z_digits·chi(s_col)`, `y_ring = Z M_j^T chi(r)`,
//! `ct[j] = y_ring[j][0]`, and balanced digit representability for each packed
//! witness coefficient.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::{get_global_pp_for_dims, precompute_rot_columns};
use neo_ccs::{build_superneo_ring_forms, CcsStructure};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{enforce_k_eq, enforce_k_eq_weighted_base_linear_combination, KNumVar};
use super::witness::{
    alloc_balanced_digit_witness, enforce_balanced_digit_alphabet, enforce_x_projection, BalancedDigitWitnessVar,
    PackedWitnessVar,
};

pub fn enforce_ce_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
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
    claim: &CeClaimVar,
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_backend_claim_consistency(
        cs, params, structure, structure, witness, claim, delta, true, false, label,
    )
}

pub fn enforce_backend_claim_consistency_with_x<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
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
    claim: &CeClaimVar,
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
    claim: &CeClaimVar,
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

pub fn enforce_ajtai_commitment_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment(cs, witness, claim, label)
}

/// Opens a DEC child only against the paper `CE(b, L)` projection.
///
/// The backend `s_col`, `y_zcol`, and `ct` channels remain replay-owned:
/// `Π_DEC` public arithmetic still binds them structurally, but the explicit
/// witness-opening layer only proves the child has paper-level commitment,
/// norm, and `y_ring` semantics over the true ring degree `D`. Any padded
/// backend tail beyond `D` remains replay-owned convenience structure.
pub fn enforce_paper_dec_child_claim_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment(
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

    let ring_forms =
        build_superneo_ring_forms(ring_structure, &claim.r_values).map_err(|_| SynthesisError::Unsatisfiable)?;
    for (matrix_idx, forms) in ring_forms.iter().enumerate() {
        enforce_claim_y_ring_from_forms(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            forms,
            D,
            &claim.y_ring[matrix_idx],
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
    claim: &CeClaimVar,
    delta: SpartanF,
    check_commitment: bool,
    check_x: bool,
    label: &str,
) -> Result<(), SynthesisError> {
    if check_commitment {
        enforce_ajtai_commitment(
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

    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let chi_s = neo_ccs::tensor_point::<K>(&claim.s_col_values);
        let digit_witness = alloc_balanced_digit_witness(
            &mut cs.namespace(|| format!("{label}_digits")),
            witness,
            base_structure.m,
            params,
            delta,
            &format!("{label}_digits"),
        )?;
        enforce_claim_y_zcol_from_digits(
            &mut cs.namespace(|| format!("{label}_y_zcol")),
            &digit_witness,
            base_structure.m,
            &chi_s,
            &claim.y_zcol,
            &format!("{label}_y_zcol"),
        )?;
    }

    let ring_forms =
        build_superneo_ring_forms(ring_structure, &claim.r_values).map_err(|_| SynthesisError::Unsatisfiable)?;
    for (matrix_idx, forms) in ring_forms.iter().enumerate() {
        enforce_claim_y_ring_from_forms(
            &mut cs.namespace(|| format!("{label}_y_ring_{matrix_idx}")),
            witness,
            ring_structure.m,
            forms,
            D,
            &claim.y_ring[matrix_idx],
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
        enforce_k_eq(
            &mut cs.namespace(|| format!("{label}_ct_{matrix_idx}")),
            claim
                .ct
                .get(matrix_idx)
                .ok_or(SynthesisError::Unsatisfiable)?,
            claim.y_ring[matrix_idx]
                .first()
                .ok_or(SynthesisError::Unsatisfiable)?,
            &format!("{label}_ct_{matrix_idx}"),
        );
    }
    Ok(())
}

fn enforce_ajtai_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    let rows = ajtai_commitment_rows(witness.rows(), witness.cols())?;
    if rows.len() != claim.c_data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (coord_idx, (coeffs, actual)) in rows.iter().zip(claim.c_data.iter()).enumerate() {
        if coeffs.len() != witness.row_major_values().len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        cs.enforce(
            || format!("{label}_{coord_idx}"),
            |lc| {
                let mut acc = lc;
                for (coeff, value) in coeffs.iter().zip(witness.row_major_values().iter()) {
                    if *coeff != F::ZERO {
                        acc = acc
                            + (
                                SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                                value.get_variable(),
                            );
                    }
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + actual.get_variable(),
        );
    }
    Ok(())
}

fn ajtai_commitment_rows(rows: usize, cols: usize) -> Result<Vec<Vec<F>>, SynthesisError> {
    let pp = get_global_pp_for_dims(rows, cols).map_err(|_| SynthesisError::Unsatisfiable)?;
    let coord_count = rows
        .checked_mul(pp.kappa)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let witness_len = rows
        .checked_mul(cols)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let mut out = vec![vec![F::ZERO; witness_len]; coord_count];

    for (commit_col, pp_row) in pp.m_rows.iter().enumerate() {
        for (witness_col, ring_el) in pp_row.iter().copied().enumerate() {
            let mut rots = [[F::ZERO; D]; D];
            precompute_rot_columns(ring_el, &mut rots);
            for witness_row in 0..rows {
                let base = witness_row
                    .checked_mul(cols)
                    .and_then(|start| start.checked_add(witness_col))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                for coord_row in 0..rows {
                    let coord = commit_col
                        .checked_mul(rows)
                        .and_then(|start| start.checked_add(coord_row))
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    out[coord][base] = rots[witness_row][coord_row];
                }
            }
        }
    }

    Ok(out)
}

fn enforce_claim_y_ring_from_forms<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    forms: &[[K; D]],
    active_rho_count: usize,
    target: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if witness.rows() != D || forms.len() < expected_m || active_rho_count > D || target.len() < active_rho_count {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (rho, target) in target.iter().take(active_rho_count).enumerate() {
        let mut terms = Vec::new();
        for logical_col in 0..expected_m {
            let weight = forms[logical_col][rho];
            if weight == K::ZERO {
                continue;
            }
            let z_var = witness.logical_entry(expected_m, logical_col)?;
            let coeffs = weight.as_coeffs();
            terms.push((
                SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                z_var.get_variable(),
            ));
        }
        enforce_k_eq_weighted_base_linear_combination(cs, target, &terms, &format!("{label}_{rho}"));
    }
    Ok(())
}

fn enforce_claim_y_zcol_from_digits<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digits: &BalancedDigitWitnessVar,
    expected_m: usize,
    chi_s: &[K],
    target: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if digits.logical_cols() != expected_m || target.len() < D {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (rho, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        if rho < D {
            for logical_col in 0..expected_m {
                let weight = chi_s.get(logical_col).copied().unwrap_or(K::ZERO);
                if weight == K::ZERO {
                    continue;
                }
                let digit = digits
                    .digit_vars(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                if let Some(digit) = digit {
                    let coeffs = weight.as_coeffs();
                    terms.push((
                        SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                        SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                        digit.get_variable(),
                    ));
                }
            }
        }
        enforce_k_eq_weighted_base_linear_combination(cs, target, &terms, &format!("{label}_{rho}"));
    }
    Ok(())
}
