use super::*;

pub(super) fn enforce_equal_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left: &[KNumVar],
    right: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        enforce_k_eq(cs, l, r, &format!("{label}_{idx}"));
    }
    Ok(())
}

pub(super) fn enforce_k_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &KNumVar,
    expected: K,
    label: &str,
) {
    let coeffs = expected.as_coeffs();
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + actual.c0,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()), CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + actual.c1,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()), CS::one()),
    );
}

pub(super) fn enforce_field_affine_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &AllocatedNum<SpartanF>,
    linear_terms: &[(SpartanF, AllocatedNum<SpartanF>)],
    product_terms: &[AllocatedNum<SpartanF>],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_sum"),
        |lc| {
            let mut acc = lc;
            for (scale, term) in linear_terms {
                acc = acc + (*scale, term.get_variable());
            }
            for term in product_terms {
                acc = acc + term.get_variable();
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.get_variable(),
    );
}

pub(super) fn enforce_k_affine_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    linear_terms: &[(SpartanF, SpartanF, bellpepper_core::Variable)],
    product_terms: &[KNumVar],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0_sum"),
        |lc| {
            let mut acc = lc;
            for (coeff_c0, _, variable) in linear_terms {
                acc = acc + (*coeff_c0, *variable);
            }
            for term in product_terms {
                acc = acc + term.c0;
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
            for (_, coeff_c1, variable) in linear_terms {
                acc = acc + (*coeff_c1, *variable);
            }
            for term in product_terms {
                acc = acc + term.c1;
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c1,
    );
}

pub(super) fn scale_k_by_f_var<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    scalar: &AllocatedNum<SpartanF>,
    scalar_value: F,
    value: &KNumVar,
    value_native: K,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let term_value = K::from(scalar_value) * value_native;
    k_base_mul_var(
        &mut cs,
        value,
        scalar.get_variable(),
        KNum::from_neo_k(value_native),
        SpartanF::from_canonical_u64(scalar_value.as_canonical_u64()),
        KNum::from_neo_k(term_value),
        label,
    )
}

pub(super) fn ensure_zero_commit_suffix(
    children: &[CircuitCeClaim],
    zero_commit_suffix_len: usize,
) -> Result<(), SynthesisError> {
    if zero_commit_suffix_len == 0 {
        return Ok(());
    }
    let zero_commit_suffix_start = children.len().saturating_sub(zero_commit_suffix_len);
    for child in &children[zero_commit_suffix_start..] {
        if child
            .commitment
            .data_values
            .iter()
            .any(|value| *value != F::ZERO)
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        if child
            .openings
            .y_ring_values
            .iter()
            .any(|row| row.iter().any(|value| *value != K::ZERO))
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    Ok(())
}

pub(super) fn alloc_rot_rho_entry_from_coeffs<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    rho: &RotRhoVar,
    row: usize,
    col: usize,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    if row >= D || col >= D || rho.coeffs.len() != D || rho.coeff_values.len() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    if col == 0 {
        return Ok(rho.coeffs[row].clone());
    }
    let mut value = F::ZERO;
    let mut terms = Vec::new();
    for coeff_idx in 0..D {
        let basis_coeff = GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, col)];
        if basis_coeff == F::ZERO {
            continue;
        }
        value += basis_coeff * rho.coeff_values[coeff_idx];
        terms.push((
            rho.coeffs[coeff_idx].clone(),
            SpartanF::from_canonical_u64(basis_coeff.as_canonical_u64()),
            SpartanF::from_canonical_u64(rho.coeff_values[coeff_idx].as_canonical_u64()),
        ));
    }
    alloc_affine_field_terms(
        cs.namespace(|| format!("{label}_affine")),
        &terms,
        SpartanF::from_canonical_u64(value.as_canonical_u64()),
    )
}

pub(super) fn alloc_affine_field_terms<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    value: SpartanF,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc;
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}
