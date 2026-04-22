//! Owns the recursive ring-action gadgets for dense `Π_RLC` commitment columns.
//!
//! This file owns the field-valued `c_data` and K-valued `y_ring` ring products. It
//! does not own `x` or transcript/rho sampling policy.

use crate::rv64im::ivc_snark::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_math::{KExtensions, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use std::sync::OnceLock;

use super::super::k_field::KNumVar;
use super::super::rho_sampling::RotRhoVar;

#[path = "ring_action_inner.rs"]
mod ring_action_inner;

use self::ring_action_inner::{
    mul_recursive_toom3_chunk_affine_exprs, mul_recursive_toom3_chunk_k_affine_k_exprs, FieldAffineExpr, KAffineExpr,
};

const KARATSUBA_SPLIT: usize = D / 3;
const KARATSUBA_CHUNK_OUT: usize = 2 * KARATSUBA_SPLIT - 1;
const INNER_TOOM_SPLIT: usize = KARATSUBA_SPLIT / 3;
const INNER_TOOM_OUT: usize = 2 * INNER_TOOM_SPLIT - 1;
const REDUCTION_CHUNK_LEN: usize = KARATSUBA_CHUNK_OUT;
const REDUCTION_FAMILY_COUNT: usize = 5;

pub(super) fn enforce_rho_coeff_left_action_on_dense_commitment_columns_toom3_with_vars<
    CS: ConstraintSystem<SpartanF>,
>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    cols: usize,
    children: &[Vec<AllocatedNum<SpartanF>>],
    child_native_values: &[Vec<F>],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if parent.len() != D * cols
        || children.is_empty()
        || children.len() != child_native_values.len()
        || children.len() != rhos.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for ((child, native_child), rho) in children
        .iter()
        .zip(child_native_values.iter())
        .zip(rhos.iter())
    {
        if child.len() != D * cols
            || native_child.len() != D * cols
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let rho_evals = rhos
        .iter()
        .enumerate()
        .map(|(_, rho)| build_field_karatsuba_affine_evals(&rho.coeffs, &rho.coeff_values))
        .collect::<Result<Vec<_>, _>>()?;

    let mut parent_terms = vec![Vec::<(SpartanF, AllocatedNum<SpartanF>)>::new(); D * cols];
    for (child_idx, ((child, native_child), rho_eval)) in children
        .iter()
        .zip(child_native_values.iter())
        .zip(rho_evals.iter())
        .enumerate()
    {
        for col in 0..cols {
            let child_col_vars = extract_column(child, col);
            let child_col_values = extract_column_native(native_child, col);
            let child_evals = build_field_karatsuba_affine_evals(&child_col_vars, &child_col_values)?;
            let product = mul_rho_child_column_toom3_rhs_affine(
                cs,
                rho_eval,
                &child_evals,
                &format!("{label}_product_{child_idx}_{col}"),
            )?;
            let reduced_terms = reduce_product_terms_mod_phi_81(product);
            for row in 0..D {
                parent_terms[col * D + row].extend(reduced_terms[row].iter().cloned());
            }
        }
    }

    for (idx, target) in parent.iter().enumerate() {
        enforce_field_linear_sum_eq(cs, target, &parent_terms[idx], &format!("{label}_eq_{idx}"));
    }
    Ok(())
}

pub(super) fn enforce_rho_coeff_left_action_on_y_row_toom3_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    child_rows: &[Vec<KNumVar>],
    child_row_values: &[Vec<K>],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() < D
        || child_rows.is_empty()
        || child_rows.len() != child_row_values.len()
        || child_rows.len() != rhos.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for ((child_row, native_row), rho) in child_rows
        .iter()
        .zip(child_row_values.iter())
        .zip(rhos.iter())
    {
        if child_row.len() < D || native_row.len() < D || rho.coeffs.len() != D || rho.coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let rho_evals = rhos
        .iter()
        .enumerate()
        .map(|(_, rho)| build_field_karatsuba_affine_evals(&rho.coeffs, &rho.coeff_values))
        .collect::<Result<Vec<_>, _>>()?;

    let mut target_terms = vec![Vec::<(SpartanF, KNumVar)>::new(); D];
    for (child_idx, ((child_row, native_row), rho_eval)) in child_rows
        .iter()
        .zip(child_row_values.iter())
        .zip(rho_evals.iter())
        .enumerate()
    {
        let child_evals = build_k_row_karatsuba_affine_evals(child_row, native_row)?;
        let product = mul_rho_child_k_row_toom3(cs, rho_eval, &child_evals, &format!("{label}_product_{child_idx}"))?;
        let reduced_terms = reduce_k_product_terms_mod_phi_81(product);
        for row in 0..D {
            target_terms[row].extend(reduced_terms[row].iter().cloned());
        }
    }

    for row in 0..D {
        enforce_k_linear_sum_eq(cs, &target[row], &target_terms[row], &format!("{label}_{row}"));
    }
    for row in D..target.len() {
        enforce_k_linear_sum_eq(cs, &target[row], &[], &format!("{label}_{row}"));
    }
    Ok(())
}

#[derive(Clone)]
struct KaratsubaProduct {
    c0: Vec<AllocatedNum<SpartanF>>,
    p1: Vec<AllocatedNum<SpartanF>>,
    pm1: Vec<AllocatedNum<SpartanF>>,
    p2: Vec<AllocatedNum<SpartanF>>,
    c4: Vec<AllocatedNum<SpartanF>>,
}

#[derive(Clone)]
struct KaratsubaAffineEvalSet {
    p0: Vec<FieldAffineExpr>,
    p1: Vec<FieldAffineExpr>,
    pm1: Vec<FieldAffineExpr>,
    p2: Vec<FieldAffineExpr>,
    p4: Vec<FieldAffineExpr>,
}

#[derive(Clone)]
struct KaratsubaKAffineEvalSet {
    p0: Vec<KAffineExpr>,
    p1: Vec<KAffineExpr>,
    pm1: Vec<KAffineExpr>,
    p2: Vec<KAffineExpr>,
    p4: Vec<KAffineExpr>,
}

#[derive(Clone)]
struct KaratsubaKProduct {
    c0: Vec<KNumVar>,
    p1: Vec<KNumVar>,
    pm1: Vec<KNumVar>,
    p2: Vec<KNumVar>,
    c4: Vec<KNumVar>,
}

fn build_field_karatsuba_affine_evals(
    vars: &[AllocatedNum<SpartanF>],
    values: &[F],
) -> Result<KaratsubaAffineEvalSet, SynthesisError> {
    if vars.len() != D || values.len() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let build_expr = |coeff_terms: &[(usize, F)]| -> Vec<FieldAffineExpr> {
        (0..KARATSUBA_SPLIT)
            .map(|idx| {
                let mut terms = Vec::new();
                let mut value = F::ZERO;
                for (block, scale) in coeff_terms {
                    let offset = block * KARATSUBA_SPLIT + idx;
                    terms.push((vars[offset].clone(), *scale));
                    value += values[offset] * *scale;
                }
                FieldAffineExpr { terms, value }
            })
            .collect()
    };
    Ok(KaratsubaAffineEvalSet {
        p0: build_expr(&[(0, F::ONE)]),
        p1: build_expr(&[(0, F::ONE), (1, F::ONE), (2, F::ONE)]),
        pm1: build_expr(&[(0, F::ONE), (1, -F::ONE), (2, F::ONE)]),
        p2: build_expr(&[(0, F::ONE), (1, F::from_u64(2)), (2, F::from_u64(4))]),
        p4: build_expr(&[(2, F::ONE)]),
    })
}

fn build_k_row_karatsuba_affine_evals(
    child_row_vars: &[KNumVar],
    child_row_values: &[K],
) -> Result<KaratsubaKAffineEvalSet, SynthesisError> {
    if child_row_vars.len() < D || child_row_values.len() < D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let build_expr = |coeff_terms: &[(usize, F)]| -> Vec<KAffineExpr> {
        (0..KARATSUBA_SPLIT)
            .map(|idx| {
                let mut terms = Vec::new();
                let mut value = K::ZERO;
                for (block, scale) in coeff_terms {
                    let offset = block * KARATSUBA_SPLIT + idx;
                    terms.push((child_row_vars[offset].clone(), *scale));
                    value += child_row_values[offset] * K::from(*scale);
                }
                KAffineExpr { terms, value }
            })
            .collect()
    };
    Ok(KaratsubaKAffineEvalSet {
        p0: build_expr(&[(0, F::ONE)]),
        p1: build_expr(&[(0, F::ONE), (1, F::ONE), (2, F::ONE)]),
        pm1: build_expr(&[(0, F::ONE), (1, -F::ONE), (2, F::ONE)]),
        p2: build_expr(&[(0, F::ONE), (1, F::from_u64(2)), (2, F::from_u64(4))]),
        p4: build_expr(&[(2, F::ONE)]),
    })
}

fn mul_rho_child_column_toom3_rhs_affine<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &KaratsubaAffineEvalSet,
    rhs: &KaratsubaAffineEvalSet,
    label: &str,
) -> Result<KaratsubaProduct, SynthesisError> {
    let c0 = mul_recursive_toom3_chunk_affine_exprs(cs, &lhs.p0, &rhs.p0, &format!("{label}_p0"))?;
    let p1 = mul_recursive_toom3_chunk_affine_exprs(cs, &lhs.p1, &rhs.p1, &format!("{label}_p1"))?;
    let pm1 = mul_recursive_toom3_chunk_affine_exprs(cs, &lhs.pm1, &rhs.pm1, &format!("{label}_pm1"))?;
    let p2 = mul_recursive_toom3_chunk_affine_exprs(cs, &lhs.p2, &rhs.p2, &format!("{label}_p2"))?;
    let c4 = mul_recursive_toom3_chunk_affine_exprs(cs, &lhs.p4, &rhs.p4, &format!("{label}_p4"))?;
    Ok(KaratsubaProduct { c0, p1, pm1, p2, c4 })
}

fn mul_rho_child_k_row_toom3<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &KaratsubaAffineEvalSet,
    rhs: &KaratsubaKAffineEvalSet,
    label: &str,
) -> Result<KaratsubaKProduct, SynthesisError> {
    let c0 = mul_recursive_toom3_chunk_k_affine_k_exprs(cs, &lhs.p0, &rhs.p0, &format!("{label}_p0"))?;
    let p1 = mul_recursive_toom3_chunk_k_affine_k_exprs(cs, &lhs.p1, &rhs.p1, &format!("{label}_p1"))?;
    let pm1 = mul_recursive_toom3_chunk_k_affine_k_exprs(cs, &lhs.pm1, &rhs.pm1, &format!("{label}_pm1"))?;
    let p2 = mul_recursive_toom3_chunk_k_affine_k_exprs(cs, &lhs.p2, &rhs.p2, &format!("{label}_p2"))?;
    let c4 = mul_recursive_toom3_chunk_k_affine_k_exprs(cs, &lhs.p4, &rhs.p4, &format!("{label}_p4"))?;
    Ok(KaratsubaKProduct { c0, p1, pm1, p2, c4 })
}

fn reduce_product_terms_mod_phi_81(product: KaratsubaProduct) -> Vec<Vec<(SpartanF, AllocatedNum<SpartanF>)>> {
    let recipes = reduced_product_recipes();
    recipes
        .iter()
        .map(|row_recipe| {
            row_recipe
                .iter()
                .map(|(family, idx, scale)| {
                    let term = match family {
                        0 => product.c0[*idx].clone(),
                        1 => product.p1[*idx].clone(),
                        2 => product.pm1[*idx].clone(),
                        3 => product.p2[*idx].clone(),
                        4 => product.c4[*idx].clone(),
                        _ => unreachable!("invalid reduction family"),
                    };
                    (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term)
                })
                .collect()
        })
        .collect()
}

fn reduce_k_product_terms_mod_phi_81(product: KaratsubaKProduct) -> Vec<Vec<(SpartanF, KNumVar)>> {
    let recipes = reduced_product_recipes();
    recipes
        .iter()
        .map(|row_recipe| {
            row_recipe
                .iter()
                .map(|(family, idx, scale)| {
                    let term = match family {
                        0 => product.c0[*idx].clone(),
                        1 => product.p1[*idx].clone(),
                        2 => product.pm1[*idx].clone(),
                        3 => product.p2[*idx].clone(),
                        4 => product.c4[*idx].clone(),
                        _ => unreachable!("invalid reduction family"),
                    };
                    (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term)
                })
                .collect()
        })
        .collect()
}

fn reduced_product_recipes() -> &'static [Vec<(u8, usize, F)>] {
    static RECIPES: OnceLock<Vec<Vec<(u8, usize, F)>>> = OnceLock::new();
    RECIPES.get_or_init(build_reduced_product_recipes)
}

fn build_reduced_product_recipes() -> Vec<Vec<(u8, usize, F)>> {
    let half = inv_two();
    let third = inv_three();
    let sixth = inv_six();
    let mut coeff_terms = vec![Vec::<(u8, usize, F)>::new(); 2 * D - 1];
    add_scaled_recipe_terms(&mut coeff_terms, 0, 0, F::ONE);
    add_scaled_recipe_terms(&mut coeff_terms, KARATSUBA_SPLIT, 0, -half);
    add_scaled_recipe_terms(&mut coeff_terms, KARATSUBA_SPLIT, 1, F::ONE);
    add_scaled_recipe_terms(&mut coeff_terms, KARATSUBA_SPLIT, 2, -third);
    add_scaled_recipe_terms(&mut coeff_terms, KARATSUBA_SPLIT, 3, -sixth);
    add_scaled_recipe_terms(&mut coeff_terms, KARATSUBA_SPLIT, 4, F::from_u64(2));
    add_scaled_recipe_terms(&mut coeff_terms, 2 * KARATSUBA_SPLIT, 0, -F::ONE);
    add_scaled_recipe_terms(&mut coeff_terms, 2 * KARATSUBA_SPLIT, 1, half);
    add_scaled_recipe_terms(&mut coeff_terms, 2 * KARATSUBA_SPLIT, 2, half);
    add_scaled_recipe_terms(&mut coeff_terms, 2 * KARATSUBA_SPLIT, 4, -F::ONE);
    add_scaled_recipe_terms(&mut coeff_terms, 3 * KARATSUBA_SPLIT, 0, half);
    add_scaled_recipe_terms(&mut coeff_terms, 3 * KARATSUBA_SPLIT, 1, -half);
    add_scaled_recipe_terms(&mut coeff_terms, 3 * KARATSUBA_SPLIT, 2, -sixth);
    add_scaled_recipe_terms(&mut coeff_terms, 3 * KARATSUBA_SPLIT, 3, sixth);
    add_scaled_recipe_terms(&mut coeff_terms, 3 * KARATSUBA_SPLIT, 4, -F::from_u64(2));
    add_scaled_recipe_terms(&mut coeff_terms, 4 * KARATSUBA_SPLIT, 4, F::ONE);

    for i in (D..(2 * D - 1)).rev() {
        let moved = std::mem::take(&mut coeff_terms[i]);
        for (family, idx, scale) in moved {
            push_recipe_term(&mut coeff_terms[i - D], family, idx, -scale);
            let idx_27 = i - 27;
            if idx_27 < D {
                push_recipe_term(&mut coeff_terms[idx_27], family, idx, -scale);
            } else {
                push_recipe_term(&mut coeff_terms[idx_27 - D], family, idx, scale);
                if idx_27 - 27 < D {
                    push_recipe_term(&mut coeff_terms[idx_27 - 27], family, idx, scale);
                }
            }
        }
    }
    coeff_terms.truncate(D);
    for row_terms in &mut coeff_terms {
        row_terms.retain(|(family, _, scale)| *family != u8::MAX && *scale != F::ZERO);
    }
    coeff_terms
}

fn add_scaled_recipe_terms(coeff_terms: &mut [Vec<(u8, usize, F)>], offset: usize, family: u8, scale: F) {
    if scale == F::ZERO {
        return;
    }
    debug_assert!((family as usize) < REDUCTION_FAMILY_COUNT);
    for idx in 0..REDUCTION_CHUNK_LEN {
        push_recipe_term(&mut coeff_terms[offset + idx], family, idx, scale);
    }
}

fn push_recipe_term(row_terms: &mut Vec<(u8, usize, F)>, family: u8, idx: usize, scale: F) {
    if scale == F::ZERO {
        return;
    }
    for existing in row_terms.iter_mut() {
        if existing.0 == family && existing.1 == idx {
            let updated = existing.2 + scale;
            if updated == F::ZERO {
                *existing = (u8::MAX, 0, F::ZERO);
            } else {
                existing.2 = updated;
            }
            return;
        }
    }
    row_terms.push((family, idx, scale));
}

fn alloc_affine_field_terms<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(AllocatedNum<SpartanF>, F, F)],
    value: F,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || {
        Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
    })?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc;
            for (term, coeff, _) in terms {
                rhs = rhs
                    + (
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        term.get_variable(),
                    );
            }
            rhs
        },
    );
    Ok(out)
}

fn alloc_k_affine_terms<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    terms: &[(KNumVar, F)],
    value: K,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let coeffs = value.as_coeffs();
    let out_c0 = cs.alloc(
        || format!("{label}_c0_alloc"),
        || Ok(SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64())),
    )?;
    let out_c1 = cs.alloc(
        || format!("{label}_c1_alloc"),
        || Ok(SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64())),
    )?;
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| {
            let mut acc = lc;
            for (term, scale) in terms {
                acc = acc + (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term.c0);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out_c0,
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| {
            let mut acc = lc;
            for (term, scale) in terms {
                acc = acc + (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term.c1);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out_c1,
    );
    Ok(KNumVar { c0: out_c0, c1: out_c1 })
}

fn enforce_field_linear_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &AllocatedNum<SpartanF>,
    terms: &[(SpartanF, AllocatedNum<SpartanF>)],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_sum"),
        |lc| {
            let mut acc = lc;
            for (scale, term) in terms {
                acc = acc + (*scale, term.get_variable());
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.get_variable(),
    );
}

fn enforce_k_linear_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    terms: &[(SpartanF, KNumVar)],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0_sum"),
        |lc| {
            let mut acc = lc;
            for (scale, term) in terms {
                acc = acc + (*scale, term.c0);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c0,
    );
    cs.enforce(
        || format!("{label}_c1_sum"),
        |lc| {
            let mut acc = lc;
            for (scale, term) in terms {
                acc = acc + (*scale, term.c1);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c1,
    );
}

fn extract_column(values: &[AllocatedNum<SpartanF>], col: usize) -> Vec<AllocatedNum<SpartanF>> {
    values[col * D..(col + 1) * D].to_vec()
}

fn extract_column_native(values: &[F], col: usize) -> Vec<F> {
    values[col * D..(col + 1) * D].to_vec()
}

fn inv_two() -> F {
    static INV_TWO: OnceLock<F> = OnceLock::new();
    *INV_TWO.get_or_init(|| F::from_u64(2).inverse())
}

fn inv_three() -> F {
    static INV_THREE: OnceLock<F> = OnceLock::new();
    *INV_THREE.get_or_init(|| F::from_u64(3).inverse())
}

fn inv_six() -> F {
    static INV_SIX: OnceLock<F> = OnceLock::new();
    *INV_SIX.get_or_init(|| F::from_u64(6).inverse())
}
