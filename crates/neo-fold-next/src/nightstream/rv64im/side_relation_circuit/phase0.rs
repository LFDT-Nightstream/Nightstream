//! Owns exact Phase 0 point-derivation and payload-evaluation gadgets for the side relation.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::{
    get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, precompute_rot_columns,
    set_global_pp_seeded, Commitment,
};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use crate::rv64im::kernel::phase0_full_width_for_schema;
use crate::rv64im::kernel::rv64im_exact_stage_pp_seed;
use crate::rv64im::kernel::{CommitmentContextId, FamilyEvalSchemaId, PackedColumnOracleRef};
use crate::rv64im::main_relation_circuit::k_field::{
    alloc_constant_k, alloc_k, enforce_k_eq, k_add, k_base_mul_var, k_mul, k_scalar_mul, KNum, KNumVar,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;

pub fn derive_phase0_point<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    opened_object_digest_vars: &[AllocatedNum<SpartanF>; 4],
    opened_object_digest: [u8; 32],
    commitment_context: &CommitmentContextId,
    schema: FamilyEvalSchemaId,
    slot: u32,
    binding_digest_vars: &[AllocatedNum<SpartanF>; 4],
    binding_digest: [u8; 32],
    row_domain_log_size: usize,
    label: &str,
) -> Result<(Vec<KNumVar>, Vec<K>), SynthesisError> {
    let opened_object = crate::rv64im::kernel::OpenedAjtaiObjectId {
        family: schema.family_kind(),
        commitment_root_digest: [0; 32],
        layout_version: 0,
        row_domain_log_size: row_domain_log_size as u32,
        digest: opened_object_digest,
    };
    let point =
        crate::rv64im::kernel::derive_phase0_point(&opened_object, commitment_context, schema, slot, binding_digest);

    let mut point_tr = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_point_tr")),
        b"neo.fold.next/rv64im/opening_convergence/phase0/point",
    )?;
    point_tr.append_field_vars_raw(
        cs.namespace(|| format!("{label}_point_opened_object")),
        &opened_object_digest_vars
            .iter()
            .map(|value| value.get_variable())
            .collect::<Vec<_>>(),
        &digest32_as_spartan_fields(opened_object_digest),
    )?;
    point_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_point_pp_seed")),
        &digest32_as_spartan_fields(commitment_context.pp_seed_digest),
    )?;
    point_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_point_module_shape")),
        &digest32_as_spartan_fields(commitment_context.module_shape_digest),
    )?;
    point_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_point_meta")),
        &[
            SpartanF::from_canonical_u64(schema.tag()),
            SpartanF::from_canonical_u64(slot as u64),
        ],
    )?;
    point_tr.append_field_vars_raw(
        cs.namespace(|| format!("{label}_point_binding")),
        &binding_digest_vars
            .iter()
            .map(|value| value.get_variable())
            .collect::<Vec<_>>(),
        &digest32_as_spartan_fields(binding_digest),
    )?;

    let mut point_vars = Vec::with_capacity(row_domain_log_size);
    for coord_index in 0..row_domain_log_size {
        point_tr.append_const_fields_raw(
            cs.namespace(|| format!("{label}_coord_index_{coord_index}")),
            &[SpartanF::from_canonical_u64(coord_index as u64)],
        )?;
        let coord = point_tr.challenge_fields_raw(cs.namespace(|| format!("{label}_coord_{coord_index}")), 2)?;
        point_vars.push(KNumVar {
            c0: coord[0].get_variable(),
            c1: coord[1].get_variable(),
        });
    }
    Ok((point_vars, point))
}

pub fn evaluate_payload_from_packed_rows<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_columns: &[PackedColumnOracleRef],
    packed_column_matrix_entries: &[Vec<AllocatedNum<SpartanF>>],
    point_vars: &[KNumVar],
    point_values: &[K],
    label: &str,
) -> Result<(Vec<Vec<KNumVar>>, Vec<Vec<K>>), SynthesisError> {
    if packed_columns.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let row_len = packed_columns[0].rows.len();
    if packed_columns
        .iter()
        .any(|column| column.rows.len() != row_len)
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    if packed_column_matrix_entries.len() != packed_columns.len()
        || packed_column_matrix_entries
            .iter()
            .any(|entries| entries.len() != row_len.saturating_mul(D))
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let delta = SpartanF::from_canonical_u64(7);
    let (weights_vars, weights_values) = chi_table(
        &mut cs.namespace(|| format!("{label}_chi")),
        point_vars,
        point_values,
        delta,
        &format!("{label}_chi"),
    )?;
    if weights_vars.len() != row_len {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut column_vars = Vec::with_capacity(packed_columns.len());
    let mut column_values = Vec::with_capacity(packed_columns.len());
    for (column_index, column) in packed_columns.iter().enumerate() {
        let mut coeff_vars = Vec::with_capacity(D);
        let mut coeff_values = Vec::with_capacity(D);
        let matrix_entries = &packed_column_matrix_entries[column_index];
        for rho in 0..D {
            let coeff_value = weights_values
                .iter()
                .enumerate()
                .fold(K::ZERO, |acc, (row_index, weight_value)| {
                    let coeff = column.rows[row_index][rho];
                    if coeff == neo_math::F::ZERO {
                        acc
                    } else {
                        acc + (*weight_value * K::from(coeff))
                    }
                });
            let coeff_var = alloc_k(
                &mut cs.namespace(|| format!("{label}_column_{column_index}_rho_{rho}")),
                Some(KNum::from_neo_k(coeff_value)),
                &format!("{label}_column_{column_index}_rho_{rho}"),
            )?;
            let term_vars = weights_vars
                .iter()
                .enumerate()
                .map(|(row_index, weight_var)| {
                    let coeff = SpartanF::from_canonical_u64(column.rows[row_index][rho].as_canonical_u64());
                    let entry_var = matrix_entries[rho * row_len + row_index].get_variable();
                    k_base_mul_var(
                        &mut cs.namespace(|| format!("{label}_column_{column_index}_rho_{rho}_row_{row_index}")),
                        weight_var,
                        entry_var,
                        KNum::from_neo_k(weights_values[row_index]),
                        coeff,
                        KNum::from_neo_k(weights_values[row_index] * K::from(column.rows[row_index][rho])),
                        &format!("{label}_column_{column_index}_rho_{rho}_row_{row_index}"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            cs.enforce(
                || format!("{label}_column_{column_index}_rho_{rho}_sum_c0"),
                |lc| term_vars.iter().fold(lc, |acc, term| acc + term.c0),
                |lc| lc + CS::one(),
                |lc| lc + coeff_var.c0,
            );
            cs.enforce(
                || format!("{label}_column_{column_index}_rho_{rho}_sum_c1"),
                |lc| term_vars.iter().fold(lc, |acc, term| acc + term.c1),
                |lc| lc + CS::one(),
                |lc| lc + coeff_var.c1,
            );
            coeff_vars.push(coeff_var);
            coeff_values.push(coeff_value);
        }
        column_vars.push(coeff_vars);
        column_values.push(coeff_values);
    }
    Ok((column_vars, column_values))
}

pub fn enforce_payload_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[Vec<KNumVar>],
    expected: &[Vec<K>],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (column_index, (actual_column, expected_column)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual_column.len() != expected_column.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for (rho, (actual_coeff, expected_coeff)) in actual_column.iter().zip(expected_column.iter()).enumerate() {
            let expected_var = alloc_k(
                &mut cs.namespace(|| format!("{label}_column_{column_index}_rho_{rho}_expected")),
                Some(KNum::from_neo_k(*expected_coeff)),
                &format!("{label}_column_{column_index}_rho_{rho}_expected"),
            )?;
            enforce_k_eq(
                &mut cs.namespace(|| format!("{label}_column_{column_index}_rho_{rho}_eq")),
                actual_coeff,
                &expected_var,
                &format!("{label}_column_{column_index}_rho_{rho}_eq"),
            );
        }
    }
    Ok(())
}

pub fn enforce_point_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[KNumVar],
    expected: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (coord_index, (actual_coord, expected_coord)) in actual.iter().zip(expected.iter()).enumerate() {
        let expected_var = alloc_k(
            &mut cs.namespace(|| format!("{label}_coord_{coord_index}_expected")),
            Some(KNum::from_neo_k(*expected_coord)),
            &format!("{label}_coord_{coord_index}_expected"),
        )?;
        enforce_k_eq(
            &mut cs.namespace(|| format!("{label}_coord_{coord_index}_eq")),
            actual_coord,
            &expected_var,
            &format!("{label}_coord_{coord_index}_eq"),
        );
    }
    Ok(())
}

pub fn enforce_commitment_root_and_opened_object_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    schema: FamilyEvalSchemaId,
    commitment_context: &CommitmentContextId,
    packed_columns: &[PackedColumnOracleRef],
    commitment_vector: &[Commitment],
    opened_object: &crate::rv64im::kernel::OpenedAjtaiObjectId,
    label: &str,
) -> Result<([AllocatedNum<SpartanF>; 4], Vec<Vec<AllocatedNum<SpartanF>>>), SynthesisError> {
    if packed_columns.is_empty()
        || packed_columns.len() != commitment_vector.len()
        || packed_columns
            .iter()
            .any(|column| column.rows.len() != packed_columns[0].rows.len())
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let row_len = packed_columns[0].rows.len();
    ensure_phase0_exact_pp(schema, row_len)?;

    let mut commitment_tr = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_commitment_root_tr")),
        b"neo.fold.next/rv64im/opening_convergence/phase0/commitment_root",
    )?;
    commitment_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_commitment_len")),
        &[SpartanF::from_canonical_u64(commitment_vector.len() as u64)],
    )?;

    let mut packed_column_matrix_entries = Vec::with_capacity(packed_columns.len());
    for (column_index, (column, commitment)) in packed_columns
        .iter()
        .zip(commitment_vector.iter())
        .enumerate()
    {
        let matrix_entries = alloc_packed_column_matrix_entries(
            &mut cs.namespace(|| format!("{label}_column_{column_index}_matrix")),
            column,
            &format!("{label}_column_{column_index}_matrix"),
        )?;
        let coord_terms = commitment_coord_linear_combinations(
            &mut cs.namespace(|| format!("{label}_column_{column_index}_commitment")),
            &matrix_entries,
            commitment,
            &format!("{label}_column_{column_index}_commitment"),
        )?;
        let coord_values = commitment
            .data
            .iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
            .collect::<Vec<_>>();
        commitment_tr.append_field_linear_combinations_raw(
            cs.namespace(|| format!("{label}_column_{column_index}_commit_coords")),
            &coord_terms,
            &vec![SpartanF::ZERO; coord_terms.len()],
            &coord_values,
        )?;
        packed_column_matrix_entries.push(matrix_entries);
    }

    let commitment_root_vars = commitment_tr.digest32(cs.namespace(|| format!("{label}_commitment_root_digest")))?;
    let commitment_root_digest_vars = alloc_digest32_witness(
        &mut cs.namespace(|| format!("{label}_commitment_root_expected")),
        opened_object.commitment_root_digest,
        &format!("{label}_commitment_root_expected"),
    )?;
    enforce_digest_var_eq(
        &mut cs.namespace(|| format!("{label}_commitment_root_eq")),
        &commitment_root_vars,
        &commitment_root_digest_vars,
        &format!("{label}_commitment_root_eq"),
    )?;

    let mut opened_object_tr = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_opened_object_tr")),
        b"neo.fold.next/rv64im/opening_convergence/phase0/opened_object",
    )?;
    opened_object_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_opened_object_meta")),
        &[
            SpartanF::from_canonical_u64(opened_object.family.tag()),
            SpartanF::from_canonical_u64(opened_object.layout_version),
            SpartanF::from_canonical_u64(opened_object.row_domain_log_size as u64),
        ],
    )?;
    opened_object_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_opened_object_pp_seed")),
        &crate::finalize::digest32_as_fields(commitment_context.pp_seed_digest)
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )?;
    opened_object_tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_opened_object_module_shape")),
        &crate::finalize::digest32_as_fields(commitment_context.module_shape_digest)
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )?;
    opened_object_tr.append_field_vars_raw(
        cs.namespace(|| format!("{label}_opened_object_root")),
        &commitment_root_digest_vars
            .iter()
            .map(|value| value.get_variable())
            .collect::<Vec<_>>(),
        &digest32_as_spartan_fields(opened_object.commitment_root_digest),
    )?;
    let opened_object_digest_vars =
        opened_object_tr.digest32(cs.namespace(|| format!("{label}_opened_object_digest")))?;
    let opened_object_digest_expected_vars = alloc_digest32_witness(
        &mut cs.namespace(|| format!("{label}_opened_object_digest_expected")),
        opened_object.digest,
        &format!("{label}_opened_object_digest_expected"),
    )?;
    enforce_digest_var_eq(
        &mut cs.namespace(|| format!("{label}_opened_object_digest_eq")),
        &opened_object_digest_vars,
        &opened_object_digest_expected_vars,
        &format!("{label}_opened_object_digest_eq"),
    )?;
    Ok((opened_object_digest_vars, packed_column_matrix_entries))
}

fn chi_table<CS: ConstraintSystem<SpartanF>>(
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
            cs,
            -SpartanF::ONE,
            bit_var,
            Some(KNum::from_neo_k(-*bit_value)),
            &format!("{label}_neg_{bit}"),
        )?;
        let one_minus_value = K::ONE - *bit_value;
        let one_minus = k_add(
            cs,
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
                cs,
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
                cs,
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

fn ensure_phase0_exact_pp(schema: FamilyEvalSchemaId, cols: usize) -> Result<(), SynthesisError> {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(phase0_full_width_for_schema(schema))
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let want_kappa = params.kappa as usize;
    if has_global_pp_for_dims(D, cols) {
        if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, cols) {
            if kappa != want_kappa || seed != rv64im_exact_stage_pp_seed() {
                return Err(SynthesisError::Unsatisfiable);
            }
        } else {
            let pp = get_global_pp_for_dims(D, cols).map_err(|_| SynthesisError::Unsatisfiable)?;
            if pp.kappa != want_kappa {
                return Err(SynthesisError::Unsatisfiable);
            }
        }
    } else {
        set_global_pp_seeded(D, want_kappa, cols, rv64im_exact_stage_pp_seed())
            .map_err(|_| SynthesisError::Unsatisfiable)?;
    }
    Ok(())
}

fn alloc_digest32_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let values = digest32_as_spartan_fields(digest);
    Ok(core::array::from_fn(|idx| {
        AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(values[idx]))
            .expect("digest witness allocation must succeed")
    }))
}

fn enforce_digest_var_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
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

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    crate::finalize::digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}

fn alloc_packed_column_matrix_entries<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    column: &PackedColumnOracleRef,
    label: &str,
) -> Result<Vec<bellpepper_core::num::AllocatedNum<SpartanF>>, SynthesisError> {
    let cols = column.rows.len();
    let mut entries = Vec::with_capacity(D * cols);
    for rho in 0..D {
        for col in 0..cols {
            let value = SpartanF::from_canonical_u64(column.rows[col][rho].as_canonical_u64());
            let out =
                bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| format!("{label}_{rho}_{col}")), || {
                    Ok(value)
                })?;
            entries.push(out);
        }
    }
    Ok(entries)
}

fn commitment_coord_linear_combinations<CS: ConstraintSystem<SpartanF>>(
    _cs: &mut CS,
    matrix_entries: &[bellpepper_core::num::AllocatedNum<SpartanF>],
    commitment: &Commitment,
    _label: &str,
) -> Result<Vec<Vec<(bellpepper_core::Variable, SpartanF)>>, SynthesisError> {
    let rows = ajtai_commitment_rows(D, matrix_entries.len() / D)?;
    if rows.len() != commitment.data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    rows.iter()
        .map(|coeffs| {
            let terms = coeffs
                .iter()
                .zip(matrix_entries.iter())
                .filter_map(|(coeff, entry)| {
                    if *coeff == neo_math::F::ZERO {
                        None
                    } else {
                        Some((
                            entry.get_variable(),
                            SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        ))
                    }
                })
                .collect::<Vec<_>>();
            if terms.is_empty() {
                return Err(SynthesisError::Unsatisfiable);
            }
            Ok(terms)
        })
        .collect()
}

fn ajtai_commitment_rows(rows: usize, cols: usize) -> Result<Vec<Vec<neo_math::F>>, SynthesisError> {
    let pp = get_global_pp_for_dims(rows, cols).map_err(|_| SynthesisError::Unsatisfiable)?;
    let coord_count = rows
        .checked_mul(pp.kappa)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let witness_len = rows
        .checked_mul(cols)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let mut out = vec![vec![neo_math::F::ZERO; witness_len]; coord_count];
    for (commit_col, pp_row) in pp.m_rows.iter().enumerate() {
        for (witness_col, ring_el) in pp_row.iter().copied().enumerate() {
            let mut rots = [[neo_math::F::ZERO; D]; D];
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
