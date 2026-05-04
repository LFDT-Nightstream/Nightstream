//! Owns core CE-consistency gadgets for SuperNeo recursive circuits.
//!
//! This module mirrors the native `neo_ccs::check_ce_consistency` boundary:
//! `c = L(Z)`, `X = L_x(Z)`, optional `y_zcol = Z_digits·chi(s_col)`,
//! `y_ring = \widehat{\bar{M}_j z}(r)` in SuperNeo ring-coefficient form,
//! `ct[j] = y_ring[j][0]`, and balanced digit representability for each
//! packed witness coefficient.

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

use super::claim::CeClaimVar;
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
    claim: &CeClaimVar,
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
        &claim.r,
        &claim.r_values,
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
            &claim.y_ring[matrix_idx],
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
    claim: &CeClaimVar,
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
        &claim.r,
        &claim.r_values,
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
            &claim.y_ring[matrix_idx],
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
    enforce_ajtai_commitment_data_consistency(cs, witness, &claim.c_data, label)
}

pub fn enforce_ajtai_commitment_data_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment(cs, witness, c_data, label)
}

pub fn enforce_ajtai_commitment_linear_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rows: usize,
    cols: usize,
    witness_entries: &[LinearCombination<SpartanF>],
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment_linear(cs, rows, cols, witness_entries, c_data, label)
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
    claim: &CeClaimVar,
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
        &claim.r,
        &claim.r_values,
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
            &claim.y_ring[matrix_idx],
            delta,
            &format!("{label}_y_ring_{matrix_idx}"),
        )?;
    }
    Ok(())
}

pub(crate) fn debug_paper_dec_child_y_ring_formula_mismatch(
    structure: &CcsStructure<F>,
    witness: &CcsWitness<F>,
    claim: &CeClaim<Commitment, F, K>,
) -> Result<Option<String>, String> {
    validate_superneo_witness_mat(&witness.Z, structure.m).map_err(|err| err.to_string())?;
    let chi_r = tensor_point::<K>(&claim.r);
    for (matrix_idx, matrix) in structure.matrices.iter().enumerate() {
        let row_cap = core::cmp::min(core::cmp::min(matrix.rows(), structure.n), chi_r.len());
        for rho in 0..D {
            let mut acc = K::ZERO;
            for (row, weight) in chi_r.iter().copied().enumerate().take(row_cap) {
                let row_terms = row_ring_projection_terms(matrix, row, witness.Z.cols().saturating_mul(D), rho)
                    .map_err(|err| err.to_string())?;
                let mut row_component = F::ZERO;
                for (logical_col, coeff) in row_terms {
                    let z_coeff = if logical_col < structure.m {
                        witness_mat_get_f(&witness.Z, structure.m, logical_col % D, logical_col)
                    } else if witness.Z.cols() == structure.m.div_ceil(D) {
                        witness.Z[(logical_col % D, logical_col / D)]
                    } else {
                        F::ZERO
                    };
                    row_component += coeff * z_coeff;
                }
                acc += weight.scale_base(row_component);
            }
            let target = claim
                .y_ring
                .get(matrix_idx)
                .and_then(|row| row.get(rho))
                .copied()
                .unwrap_or(K::ZERO);
            if acc != target {
                return Ok(Some(format!(
                    "circuit CE y_ring formula mismatch: matrix={matrix_idx}, rho={rho}, claim={}, formula={}",
                    format_k(target),
                    format_k(acc),
                )));
            }
        }
    }
    Ok(None)
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

    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let (chi_s, chi_s_values) = chi_table_var(
            &mut cs.namespace(|| format!("{label}_chi_s")),
            &claim.s_col,
            &claim.s_col_values,
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
            &claim.y_zcol,
            delta,
            &format!("{label}_y_zcol"),
        )?;
    }

    let (chi_r, chi_r_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_r")),
        &claim.r,
        &claim.r_values,
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
            &claim.y_ring[matrix_idx],
            delta,
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
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    let witness_entries = witness
        .row_major_values()
        .iter()
        .map(|value| LinearCombination::<SpartanF>::zero() + value.get_variable())
        .collect::<Vec<_>>();
    enforce_ajtai_commitment_linear(cs, witness.rows(), witness.cols(), &witness_entries, c_data, label)
}

fn enforce_ajtai_commitment_linear<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness_rows: usize,
    witness_cols: usize,
    witness_entries: &[LinearCombination<SpartanF>],
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    let rows = ajtai_commitment_rows(witness_rows, witness_cols)?;
    if rows.len() != c_data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (coord_idx, (coeffs, actual)) in rows.iter().zip(c_data.iter()).enumerate() {
        if coeffs.len() != witness_entries.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        cs.enforce(
            || format!("{label}_{coord_idx}"),
            |lc| {
                let mut acc = lc;
                for (coeff, entry) in coeffs.iter().zip(witness_entries.iter()) {
                    if *coeff != F::ZERO {
                        acc = acc + (SpartanF::from_canonical_u64(coeff.as_canonical_u64()), entry);
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

fn enforce_claim_y_ring_from_point_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    active_row_count: usize,
    matrix: &CcsMatrix<F>,
    chi_r: &[KNumVar],
    chi_r_values: &[K],
    active_rho_count: usize,
    target: &[KNumVar],
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    if witness.rows() != D || active_rho_count > D || target.len() < active_rho_count {
        return Err(SynthesisError::Unsatisfiable);
    }
    let row_cap = core::cmp::min(core::cmp::min(matrix.rows(), active_row_count), chi_r.len());
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    for (rho, target) in target.iter().take(active_rho_count).enumerate() {
        let mut acc = zero.clone();
        let mut acc_value = K::ZERO;
        for row in 0..row_cap {
            let row_terms = row_ring_projection_terms(matrix, row, witness_y_ring_width(witness, expected_m)?, rho)?;
            let affine_terms = row_terms
                .iter()
                .map(|(logical_col, coeff)| {
                    let z_var = witness.logical_entry(expected_m, *logical_col)?;
                    let z_value = witness.logical_value(expected_m, *logical_col)?;
                    Ok((
                        z_var,
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        SpartanF::from_canonical_u64(z_value.as_canonical_u64()),
                    ))
                })
                .collect::<Result<Vec<_>, SynthesisError>>()?;
            let row_component = alloc_affine_base(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_component")),
                &affine_terms,
                SpartanF::ZERO,
            )?;
            let row_component_value = row_terms
                .iter()
                .try_fold(F::ZERO, |acc, (logical_col, coeff)| {
                    Ok::<_, SynthesisError>(acc + *coeff * witness.logical_value(expected_m, *logical_col)?)
                })?;
            let term_value = chi_r_values[row].scale_base(row_component_value);
            let term = k_base_mul_var(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_term")),
                &chi_r[row],
                row_component.get_variable(),
                KNum::from_neo_k(chi_r_values[row]),
                SpartanF::from_canonical_u64(row_component_value.as_canonical_u64()),
                KNum::from_neo_k(term_value),
                &format!("{label}_row_{row}_rho_{rho}_term"),
            )?;
            acc_value += term_value;
            acc = k_add(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_acc")),
                &acc,
                &term,
                Some(KNum::from_neo_k(acc_value)),
                &format!("{label}_row_{row}_rho_{rho}_acc"),
            )?;
        }
        let _ = delta;
        enforce_k_eq(cs, target, &acc, &format!("{label}_{rho}"));
    }
    Ok(())
}

fn row_ring_projection_terms(
    matrix: &CcsMatrix<F>,
    row: usize,
    effective_m: usize,
    rho: usize,
) -> Result<Vec<(usize, F)>, SynthesisError> {
    if rho >= D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let block_count = effective_m.div_ceil(D);
    let mut terms = Vec::new();
    for blk in 0..block_count {
        let base = blk * D;
        let mut a = [F::ZERO; D];
        for (off, coeff) in a.iter_mut().enumerate() {
            *coeff = matrix_entry_base_f(matrix, row, base + off);
        }
        if a.iter().all(|value| *value == F::ZERO) {
            continue;
        }
        let a_bar = Rq(superneo_bar_block(a));
        for off in 0..D {
            let logical_col = base + off;
            if logical_col >= effective_m {
                break;
            }
            let mut basis = [F::ZERO; D];
            basis[off] = F::ONE;
            let coeff = a_bar.mul(&Rq(basis)).0[rho];
            if coeff != F::ZERO {
                terms.push((logical_col, coeff));
            }
        }
    }
    Ok(terms)
}

fn witness_y_ring_width(witness: &PackedWitnessVar, expected_m: usize) -> Result<usize, SynthesisError> {
    if expected_m == 0 || witness.rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    if witness.cols() == expected_m.div_ceil(D) {
        return witness
            .cols()
            .checked_mul(D)
            .ok_or(SynthesisError::Unsatisfiable);
    }
    Err(SynthesisError::Unsatisfiable)
}

fn matrix_entry_base_f(matrix: &CcsMatrix<F>, row: usize, col: usize) -> F {
    if row >= matrix.rows() || col >= matrix.cols() {
        return F::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == col {
                F::ONE
            } else {
                F::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let start = csc.col_ptr[col];
            let end = csc.col_ptr[col + 1];
            let mut acc = F::ZERO;
            for idx in start..end {
                if csc.row_idx[idx] == row {
                    acc += csc.vals[idx];
                }
            }
            acc
        }
    }
}

fn format_k(value: K) -> String {
    let [re, im] = value.as_coeffs();
    format!("({}, {})", re.as_canonical_u64(), im.as_canonical_u64())
}

fn alloc_affine_base<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(bellpepper_core::num::AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    constant: SpartanF,
) -> Result<bellpepper_core::num::AllocatedNum<SpartanF>, SynthesisError> {
    let mut value = constant;
    for (_, coeff, term_value) in terms {
        value += *coeff * *term_value;
    }
    let out = bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc + (constant, CS::one());
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}

fn enforce_claim_y_zcol_from_digits_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digits: &BalancedDigitWitnessVar,
    expected_m: usize,
    chi_s: &[KNumVar],
    chi_s_values: &[K],
    target: &[KNumVar],
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    if digits.logical_cols() != expected_m
        || target.len() < D
        || chi_s.len() < expected_m
        || chi_s_values.len() < expected_m
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    for (rho, target) in target.iter().enumerate() {
        let mut acc = zero.clone();
        let mut acc_value = K::ZERO;
        if rho < D {
            for logical_col in 0..expected_m {
                let weight = chi_s
                    .get(logical_col)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let weight_value = *chi_s_values
                    .get(logical_col)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let digit = digits
                    .digit_vars(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let digit_value = *digits
                    .digit_values(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let term_value = weight_value * K::from(digit_value);
                let term = k_base_mul_var(
                    &mut cs.namespace(|| format!("{label}_term_{logical_col}_{rho}")),
                    weight,
                    digit.get_variable(),
                    KNum::from_neo_k(weight_value),
                    SpartanF::from_canonical_u64(digit_value.as_canonical_u64()),
                    KNum::from_neo_k(term_value),
                    &format!("{label}_term_{logical_col}_{rho}"),
                )?;
                acc_value += term_value;
                acc = k_add(
                    &mut cs.namespace(|| format!("{label}_acc_{logical_col}_{rho}")),
                    &acc,
                    &term,
                    Some(KNum::from_neo_k(acc_value)),
                    &format!("{label}_acc_{logical_col}_{rho}"),
                )?;
            }
        }
        let _ = delta;
        enforce_k_eq(cs, target, &acc, &format!("{label}_{rho}"));
    }
    Ok(())
}

fn chi_table_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    point_vars: &[KNumVar],
    point_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(Vec<KNumVar>, Vec<K>), SynthesisError> {
    if point_vars.len() != point_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let one = alloc_constant_k(cs, KNum::from_neo_k(K::ONE), &format!("{label}_one"))?;
    let mut out_vars = vec![one.clone()];
    let mut out_values = vec![K::ONE];

    for (bit, (bit_var, bit_value)) in point_vars.iter().zip(point_values.iter()).enumerate() {
        let neg = k_scalar_mul(
            &mut cs.namespace(|| format!("{label}_neg_{bit}")),
            -SpartanF::ONE,
            bit_var,
            Some(KNum::from_neo_k(-*bit_value)),
            &format!("{label}_neg_{bit}"),
        )?;
        let one_minus_value = K::ONE - *bit_value;
        let one_minus = k_add(
            &mut cs.namespace(|| format!("{label}_one_minus_{bit}")),
            &one,
            &neg,
            Some(KNum::from_neo_k(one_minus_value)),
            &format!("{label}_one_minus_{bit}"),
        )?;

        let prior_len = out_vars.len();
        let mut next_vars = Vec::with_capacity(prior_len * 2);
        let mut next_values = Vec::with_capacity(prior_len * 2);

        for idx in 0..prior_len {
            let next_value = out_values[idx] * one_minus_value;
            let next_var = k_mul(
                &mut cs.namespace(|| format!("{label}_zero_branch_{bit}_{idx}")),
                &out_vars[idx],
                &one_minus,
                KNum::from_neo_k(out_values[idx]),
                KNum::from_neo_k(one_minus_value),
                KNum::from_neo_k(next_value),
                delta,
                &format!("{label}_zero_branch_{bit}_{idx}"),
            )?;
            next_vars.push(next_var);
            next_values.push(next_value);
        }
        for idx in 0..prior_len {
            let next_value = out_values[idx] * *bit_value;
            let next_var = k_mul(
                &mut cs.namespace(|| format!("{label}_one_branch_{bit}_{idx}")),
                &out_vars[idx],
                bit_var,
                KNum::from_neo_k(out_values[idx]),
                KNum::from_neo_k(*bit_value),
                KNum::from_neo_k(next_value),
                delta,
                &format!("{label}_one_branch_{bit}_{idx}"),
            )?;
            next_vars.push(next_var);
            next_values.push(next_value);
        }

        out_vars = next_vars;
        out_values = next_values;
    }

    Ok((out_vars, out_values))
}
