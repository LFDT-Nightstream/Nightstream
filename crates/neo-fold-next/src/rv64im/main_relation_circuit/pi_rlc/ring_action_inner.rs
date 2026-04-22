use super::*;
use crate::rv64im::main_relation_circuit::k_field::KNum;

#[derive(Clone)]
pub(super) struct FieldAffineExpr {
    pub(super) terms: Vec<(AllocatedNum<SpartanF>, F)>,
    pub(super) value: F,
}

#[derive(Clone)]
pub(super) struct KAffineExpr {
    pub(super) terms: Vec<(KNumVar, F)>,
    pub(super) value: K,
}

#[derive(Clone)]
struct InnerToomAffineEvalSet {
    p0: Vec<FieldAffineExpr>,
    p1: Vec<FieldAffineExpr>,
    pm1: Vec<FieldAffineExpr>,
    p2: Vec<FieldAffineExpr>,
    p4: Vec<FieldAffineExpr>,
}

#[derive(Clone)]
struct InnerToomProduct {
    c0_terms: Vec<Vec<AllocatedNum<SpartanF>>>,
    c0_values: [F; INNER_TOOM_OUT],
    p1_terms: Vec<Vec<AllocatedNum<SpartanF>>>,
    p1_values: [F; INNER_TOOM_OUT],
    pm1_terms: Vec<Vec<AllocatedNum<SpartanF>>>,
    pm1_values: [F; INNER_TOOM_OUT],
    p2_terms: Vec<Vec<AllocatedNum<SpartanF>>>,
    p2_values: [F; INNER_TOOM_OUT],
    c4_terms: Vec<Vec<AllocatedNum<SpartanF>>>,
    c4_values: [F; INNER_TOOM_OUT],
}

#[derive(Clone)]
struct InnerToomKAffineEvalSet {
    p0: Vec<KAffineExpr>,
    p1: Vec<KAffineExpr>,
    pm1: Vec<KAffineExpr>,
    p2: Vec<KAffineExpr>,
    p4: Vec<KAffineExpr>,
}

#[derive(Clone)]
struct InnerToomKProduct {
    c0_terms: Vec<Vec<KNumVar>>,
    c0_values: [K; INNER_TOOM_OUT],
    p1_terms: Vec<Vec<KNumVar>>,
    p1_values: [K; INNER_TOOM_OUT],
    pm1_terms: Vec<Vec<KNumVar>>,
    pm1_values: [K; INNER_TOOM_OUT],
    p2_terms: Vec<Vec<KNumVar>>,
    p2_values: [K; INNER_TOOM_OUT],
    c4_terms: Vec<Vec<KNumVar>>,
    c4_values: [K; INNER_TOOM_OUT],
}

pub(super) fn mul_recursive_toom3_chunk_affine_exprs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &[FieldAffineExpr],
    rhs: &[FieldAffineExpr],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    let lhs_evals = build_inner_toom_affine_evals_from_exprs(lhs)?;
    let rhs_evals = build_inner_toom_affine_evals_from_exprs(rhs)?;
    let product = mul_inner_toom_affine_product(cs, &lhs_evals, &rhs_evals, label)?;
    finalize_inner_toom_field_outputs(cs, product, label)
}

fn finalize_inner_toom_field_outputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    product: InnerToomProduct,
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    let recipes = inner_toom_product_recipes();
    Ok(recipes
        .iter()
        .enumerate()
        .map(|(row_idx, row_recipe)| {
            let mut value = F::ZERO;
            let mut terms = Vec::with_capacity(row_recipe.len());
            for (family, idx, scale) in row_recipe {
                let family_terms = match family {
                    0 => &product.c0_terms[*idx],
                    1 => &product.p1_terms[*idx],
                    2 => &product.pm1_terms[*idx],
                    3 => &product.p2_terms[*idx],
                    4 => &product.c4_terms[*idx],
                    _ => unreachable!("invalid inner reduction family"),
                };
                let native_term = match family {
                    0 => product.c0_values[*idx],
                    1 => product.p1_values[*idx],
                    2 => product.pm1_values[*idx],
                    3 => product.p2_values[*idx],
                    4 => product.c4_values[*idx],
                    _ => unreachable!("invalid inner reduction family"),
                };
                value += native_term * *scale;
                for term in family_terms {
                    terms.push((term.clone(), *scale, F::ZERO));
                }
            }
            alloc_affine_field_terms(cs.namespace(|| format!("{label}_out_{row_idx}")), &terms, value)
        })
        .collect::<Result<Vec<_>, _>>()?)
}

pub(super) fn mul_recursive_toom3_chunk_k_affine_k_exprs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &[FieldAffineExpr],
    rhs: &[KAffineExpr],
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    let lhs_evals = build_inner_toom_affine_evals_from_exprs(lhs)?;
    let rhs_evals = build_inner_toom_k_affine_evals_from_exprs(rhs)?;
    let product = mul_inner_toom_k_product(cs, &lhs_evals, &rhs_evals, label)?;
    let recipes = inner_toom_product_recipes();
    Ok(recipes
        .iter()
        .enumerate()
        .map(|(row_idx, row_recipe)| {
            let mut value = K::ZERO;
            let mut terms = Vec::with_capacity(row_recipe.len());
            for (family, idx, scale) in row_recipe {
                let family_terms = match family {
                    0 => &product.c0_terms[*idx],
                    1 => &product.p1_terms[*idx],
                    2 => &product.pm1_terms[*idx],
                    3 => &product.p2_terms[*idx],
                    4 => &product.c4_terms[*idx],
                    _ => unreachable!("invalid inner reduction family"),
                };
                let native_term = match family {
                    0 => product.c0_values[*idx],
                    1 => product.p1_values[*idx],
                    2 => product.pm1_values[*idx],
                    3 => product.p2_values[*idx],
                    4 => product.c4_values[*idx],
                    _ => unreachable!("invalid inner reduction family"),
                };
                value += native_term * K::from(*scale);
                for term in family_terms {
                    terms.push((term.clone(), *scale));
                }
            }
            alloc_k_affine_terms(
                &mut cs.namespace(|| format!("{label}_out_{row_idx}")),
                &terms,
                value,
                &format!("{label}_out_{row_idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?)
}

fn build_inner_toom_affine_evals_from_exprs(
    exprs: &[FieldAffineExpr],
) -> Result<InnerToomAffineEvalSet, SynthesisError> {
    if exprs.len() != KARATSUBA_SPLIT {
        return Err(SynthesisError::Unsatisfiable);
    }
    let build_expr = |coeff_terms: &[(usize, F)]| -> Vec<FieldAffineExpr> {
        (0..INNER_TOOM_SPLIT)
            .map(|idx| {
                let mut terms = Vec::new();
                let mut value = F::ZERO;
                for (block, scale) in coeff_terms {
                    let offset = block * INNER_TOOM_SPLIT + idx;
                    for (term, term_scale) in &exprs[offset].terms {
                        terms.push((term.clone(), *term_scale * *scale));
                    }
                    value += exprs[offset].value * *scale;
                }
                FieldAffineExpr { terms, value }
            })
            .collect()
    };
    Ok(InnerToomAffineEvalSet {
        p0: build_expr(&[(0, F::ONE)]),
        p1: build_expr(&[(0, F::ONE), (1, F::ONE), (2, F::ONE)]),
        pm1: build_expr(&[(0, F::ONE), (1, -F::ONE), (2, F::ONE)]),
        p2: build_expr(&[(0, F::ONE), (1, F::from_u64(2)), (2, F::from_u64(4))]),
        p4: build_expr(&[(2, F::ONE)]),
    })
}

fn build_inner_toom_k_affine_evals_from_exprs(
    exprs: &[KAffineExpr],
) -> Result<InnerToomKAffineEvalSet, SynthesisError> {
    if exprs.len() != KARATSUBA_SPLIT {
        return Err(SynthesisError::Unsatisfiable);
    }
    let build_expr = |coeff_terms: &[(usize, F)]| -> Vec<KAffineExpr> {
        (0..INNER_TOOM_SPLIT)
            .map(|idx| {
                let mut terms = Vec::new();
                let mut value = K::ZERO;
                for (block, scale) in coeff_terms {
                    let offset = block * INNER_TOOM_SPLIT + idx;
                    for (term, term_scale) in &exprs[offset].terms {
                        terms.push((term.clone(), *term_scale * *scale));
                    }
                    value += exprs[offset].value * K::from(*scale);
                }
                KAffineExpr { terms, value }
            })
            .collect()
    };
    Ok(InnerToomKAffineEvalSet {
        p0: build_expr(&[(0, F::ONE)]),
        p1: build_expr(&[(0, F::ONE), (1, F::ONE), (2, F::ONE)]),
        pm1: build_expr(&[(0, F::ONE), (1, -F::ONE), (2, F::ONE)]),
        p2: build_expr(&[(0, F::ONE), (1, F::from_u64(2)), (2, F::from_u64(4))]),
        p4: build_expr(&[(2, F::ONE)]),
    })
}

fn mul_inner_toom_affine_product<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &InnerToomAffineEvalSet,
    rhs: &InnerToomAffineEvalSet,
    label: &str,
) -> Result<InnerToomProduct, SynthesisError> {
    let c0 = mul_schoolbook_inner_chunk_affine(cs, &lhs.p0, &rhs.p0, &format!("{label}_p0"))?;
    let p1 = mul_schoolbook_inner_chunk_affine(cs, &lhs.p1, &rhs.p1, &format!("{label}_p1"))?;
    let pm1 = mul_schoolbook_inner_chunk_affine(cs, &lhs.pm1, &rhs.pm1, &format!("{label}_pm1"))?;
    let p2 = mul_schoolbook_inner_chunk_affine(cs, &lhs.p2, &rhs.p2, &format!("{label}_p2"))?;
    let c4 = mul_schoolbook_inner_chunk_affine(cs, &lhs.p4, &rhs.p4, &format!("{label}_p4"))?;
    Ok(InnerToomProduct {
        c0_terms: c0.0,
        c0_values: c0.1,
        p1_terms: p1.0,
        p1_values: p1.1,
        pm1_terms: pm1.0,
        pm1_values: pm1.1,
        p2_terms: p2.0,
        p2_values: p2.1,
        c4_terms: c4.0,
        c4_values: c4.1,
    })
}

fn mul_inner_toom_k_product<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &InnerToomAffineEvalSet,
    rhs: &InnerToomKAffineEvalSet,
    label: &str,
) -> Result<InnerToomKProduct, SynthesisError> {
    let c0 = mul_schoolbook_inner_chunk_k_affine(cs, &lhs.p0, &rhs.p0, &format!("{label}_p0"))?;
    let p1 = mul_schoolbook_inner_chunk_k_affine(cs, &lhs.p1, &rhs.p1, &format!("{label}_p1"))?;
    let pm1 = mul_schoolbook_inner_chunk_k_affine(cs, &lhs.pm1, &rhs.pm1, &format!("{label}_pm1"))?;
    let p2 = mul_schoolbook_inner_chunk_k_affine(cs, &lhs.p2, &rhs.p2, &format!("{label}_p2"))?;
    let c4 = mul_schoolbook_inner_chunk_k_affine(cs, &lhs.p4, &rhs.p4, &format!("{label}_p4"))?;
    Ok(InnerToomKProduct {
        c0_terms: c0.0,
        c0_values: c0.1,
        p1_terms: p1.0,
        p1_values: p1.1,
        pm1_terms: pm1.0,
        pm1_values: pm1.1,
        p2_terms: p2.0,
        p2_values: p2.1,
        c4_terms: c4.0,
        c4_values: c4.1,
    })
}

fn mul_schoolbook_inner_chunk_affine<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &[FieldAffineExpr],
    rhs: &[FieldAffineExpr],
    label: &str,
) -> Result<(Vec<Vec<AllocatedNum<SpartanF>>>, [F; INNER_TOOM_OUT]), SynthesisError> {
    if lhs.len() != INNER_TOOM_SPLIT || rhs.len() != INNER_TOOM_SPLIT {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut sum_terms = vec![Vec::<AllocatedNum<SpartanF>>::new(); INNER_TOOM_OUT];
    let mut out_values = [F::ZERO; INNER_TOOM_OUT];
    for i in 0..INNER_TOOM_SPLIT {
        for j in 0..INNER_TOOM_SPLIT {
            let product = alloc_affine_field_product(
                &mut cs.namespace(|| format!("{label}_mul_{i}_{j}")),
                &lhs[i],
                &rhs[j],
                &format!("{label}_mul_{i}_{j}"),
            )?;
            sum_terms[i + j].push(product);
            out_values[i + j] += lhs[i].value * rhs[j].value;
        }
    }
    Ok((sum_terms, out_values))
}

fn mul_schoolbook_inner_chunk_k_affine<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &[FieldAffineExpr],
    rhs: &[KAffineExpr],
    label: &str,
) -> Result<(Vec<Vec<KNumVar>>, [K; INNER_TOOM_OUT]), SynthesisError> {
    if lhs.len() != INNER_TOOM_SPLIT || rhs.len() != INNER_TOOM_SPLIT {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut sum_terms = vec![Vec::<KNumVar>::new(); INNER_TOOM_OUT];
    let mut out_values = [K::ZERO; INNER_TOOM_OUT];
    for i in 0..INNER_TOOM_SPLIT {
        for j in 0..INNER_TOOM_SPLIT {
            let term = alloc_affine_k_product(
                &mut cs.namespace(|| format!("{label}_mul_{i}_{j}")),
                &lhs[i],
                &rhs[j],
                &format!("{label}_mul_{i}_{j}"),
            )?;
            sum_terms[i + j].push(term);
            out_values[i + j] += K::from(lhs[i].value) * rhs[j].value;
        }
    }
    Ok((sum_terms, out_values))
}

fn alloc_affine_field_product<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &FieldAffineExpr,
    rhs: &FieldAffineExpr,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_alloc")), || {
        Ok(SpartanF::from_canonical_u64((lhs.value * rhs.value).as_canonical_u64()))
    })?;
    cs.enforce(
        || format!("{label}_mul"),
        |lc| {
            let mut acc = lc;
            for (term, scale) in &lhs.terms {
                acc = acc
                    + (
                        SpartanF::from_canonical_u64(scale.as_canonical_u64()),
                        term.get_variable(),
                    );
            }
            acc
        },
        |lc| {
            let mut acc = lc;
            for (term, scale) in &rhs.terms {
                acc = acc
                    + (
                        SpartanF::from_canonical_u64(scale.as_canonical_u64()),
                        term.get_variable(),
                    );
            }
            acc
        },
        |lc| lc + out.get_variable(),
    );
    Ok(out)
}

fn alloc_affine_k_product<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    lhs: &FieldAffineExpr,
    rhs: &KAffineExpr,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let product_value = KNum::from_neo_k(K::from(lhs.value) * rhs.value);
    let out_c0 = cs.alloc(|| format!("{label}_c0"), || Ok(product_value.c0))?;
    let out_c1 = cs.alloc(|| format!("{label}_c1"), || Ok(product_value.c1))?;
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| {
            let mut acc = lc;
            for (term, scale) in &lhs.terms {
                acc = acc
                    + (
                        SpartanF::from_canonical_u64(scale.as_canonical_u64()),
                        term.get_variable(),
                    );
            }
            acc
        },
        |lc| {
            let mut acc = lc;
            for (term, scale) in &rhs.terms {
                acc = acc + (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term.c0);
            }
            acc
        },
        |lc| lc + out_c0,
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| {
            let mut acc = lc;
            for (term, scale) in &lhs.terms {
                acc = acc
                    + (
                        SpartanF::from_canonical_u64(scale.as_canonical_u64()),
                        term.get_variable(),
                    );
            }
            acc
        },
        |lc| {
            let mut acc = lc;
            for (term, scale) in &rhs.terms {
                acc = acc + (SpartanF::from_canonical_u64(scale.as_canonical_u64()), term.c1);
            }
            acc
        },
        |lc| lc + out_c1,
    );
    Ok(KNumVar { c0: out_c0, c1: out_c1 })
}

fn inner_toom_product_recipes() -> &'static [Vec<(u8, usize, F)>] {
    static RECIPES: OnceLock<Vec<Vec<(u8, usize, F)>>> = OnceLock::new();
    RECIPES.get_or_init(build_inner_toom_product_recipes)
}

fn build_inner_toom_product_recipes() -> Vec<Vec<(u8, usize, F)>> {
    let half = inv_two();
    let third = inv_three();
    let sixth = inv_six();
    let mut coeff_terms = vec![Vec::<(u8, usize, F)>::new(); 2 * KARATSUBA_SPLIT - 1];
    add_scaled_inner_recipe_terms(&mut coeff_terms, 0, 0, F::ONE);
    add_scaled_inner_recipe_terms(&mut coeff_terms, INNER_TOOM_SPLIT, 0, -half);
    add_scaled_inner_recipe_terms(&mut coeff_terms, INNER_TOOM_SPLIT, 1, F::ONE);
    add_scaled_inner_recipe_terms(&mut coeff_terms, INNER_TOOM_SPLIT, 2, -third);
    add_scaled_inner_recipe_terms(&mut coeff_terms, INNER_TOOM_SPLIT, 3, -sixth);
    add_scaled_inner_recipe_terms(&mut coeff_terms, INNER_TOOM_SPLIT, 4, F::from_u64(2));
    add_scaled_inner_recipe_terms(&mut coeff_terms, 2 * INNER_TOOM_SPLIT, 0, -F::ONE);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 2 * INNER_TOOM_SPLIT, 1, half);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 2 * INNER_TOOM_SPLIT, 2, half);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 2 * INNER_TOOM_SPLIT, 4, -F::ONE);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 3 * INNER_TOOM_SPLIT, 0, half);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 3 * INNER_TOOM_SPLIT, 1, -half);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 3 * INNER_TOOM_SPLIT, 2, -sixth);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 3 * INNER_TOOM_SPLIT, 3, sixth);
    add_scaled_inner_recipe_terms(&mut coeff_terms, 3 * INNER_TOOM_SPLIT, 4, -F::from_u64(2));
    add_scaled_inner_recipe_terms(&mut coeff_terms, 4 * INNER_TOOM_SPLIT, 4, F::ONE);
    coeff_terms
}

fn add_scaled_inner_recipe_terms(coeff_terms: &mut [Vec<(u8, usize, F)>], offset: usize, family: u8, scale: F) {
    if scale == F::ZERO {
        return;
    }
    debug_assert!((family as usize) < REDUCTION_FAMILY_COUNT);
    for idx in 0..INNER_TOOM_OUT {
        push_recipe_term(&mut coeff_terms[offset + idx], family, idx, scale);
    }
}
