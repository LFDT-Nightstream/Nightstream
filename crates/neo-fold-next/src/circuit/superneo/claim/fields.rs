use super::*;

pub fn packed_bytes_field_values(bytes: &[u8]) -> Vec<SpartanF> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_LIMB));
    out.push(SpartanF::from_canonical_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(SpartanF::from_canonical_u64(u64::from_le_bytes(limb)));
    }
    out
}

pub(super) fn alloc_f_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[F],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect()
}

pub(super) fn alloc_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[K],
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_k(cs, Some(KNum::from_neo_k(*value)), &format!("{label}_{idx}")))
        .collect()
}

pub(super) fn extend_packed_bytes_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    bytes: &[u8],
    label: &str,
) -> Result<(), SynthesisError> {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(AllocatedNum::alloc(cs.namespace(|| format!("{label}_len")), || {
        Ok(SpartanF::from_canonical_u64(bytes.len() as u64))
    })?);
    for (idx, chunk) in bytes.chunks(BYTES_PER_LIMB).enumerate() {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_limb_{idx}")),
            || Ok(SpartanF::from_canonical_u64(u64::from_le_bytes(limb))),
        )?);
    }
    Ok(())
}

pub(super) fn alloc_packed_bytes_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bytes: &[u8],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    let mut out = Vec::new();
    extend_packed_bytes_as_fields(cs, &mut out, bytes, label)?;
    Ok(out)
}

pub(super) fn enforce_packed_bytes_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>],
    expected_bytes: &[u8],
    label: &str,
) -> Result<(), SynthesisError> {
    let expected = packed_bytes_field_values(expected_bytes);
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}_eq"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (*expected, CS::one()),
        );
    }
    Ok(())
}

pub(super) fn extend_f_slice_values(dst: &mut Vec<SpartanF>, values: &[F]) {
    dst.push(SpartanF::from_canonical_u64(values.len() as u64));
    dst.extend(
        values
            .iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    );
}

pub(super) fn enforce_f_slice_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>],
    expected: &[F],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}_eq"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (SpartanF::from_canonical_u64(expected.as_canonical_u64()), CS::one()),
        );
    }
    Ok(())
}

pub(super) fn enforce_f_slice_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>],
    expected: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}_eq"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected.get_variable(),
        );
    }
    Ok(())
}

pub(super) fn extend_k_slice_values(dst: &mut Vec<SpartanF>, values: &[K]) {
    dst.push(SpartanF::from_canonical_u64(values.len() as u64));
    let coeff_len = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    dst.push(SpartanF::from_canonical_u64(coeff_len as u64));
    for value in values {
        let coeffs = value.as_coeffs();
        dst.push(SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()));
        dst.push(SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()));
    }
}

pub(super) fn enforce_k_slice_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[KNumVar],
    expected: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let expected = KNum::from_neo_k(*expected);
        cs.enforce(
            || format!("{label}_{idx}_c0_eq"),
            |lc| lc + actual.c0,
            |lc| lc + CS::one(),
            |lc| lc + (expected.c0, CS::one()),
        );
        cs.enforce(
            || format!("{label}_{idx}_c1_eq"),
            |lc| lc + actual.c1,
            |lc| lc + CS::one(),
            |lc| lc + (expected.c1, CS::one()),
        );
    }
    Ok(())
}

pub(super) fn enforce_k_slice_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[KNumVar],
    expected: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        enforce_k_eq(
            &mut cs.namespace(|| format!("{label}_{idx}")),
            actual,
            expected,
            &format!("{label}_{idx}"),
        );
    }
    Ok(())
}
