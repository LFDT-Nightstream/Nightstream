//! Owns exact vector-package public-step reconstruction for side relation gadgets.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::{
    get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, precompute_rot_columns,
    set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use crate::rv64im::kernel::rv64im_exact_stage_pp_seed;
use crate::rv64im::main_relation_circuit::claim::packed_bytes_field_values;
use crate::rv64im::main_relation_circuit::public_chunk::{public_step_digest, CcsClaimVar, PublicStepVar};
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};

use super::word::{alloc_u64, U64Var};

const EXACT_VECTOR_PACKAGE_K_RHO: u32 = 24;
const EXACT_VECTOR_PACKAGE_B: u64 = 1 << EXACT_VECTOR_PACKAGE_K_RHO;

pub fn exact_vector_packaged_step_digest_from_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    base_label: &str,
    words: &[U64Var],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let step = exact_vector_packaged_step_var_from_words_with_step_label(
        &mut cs.namespace(|| format!("{base_label}_step")),
        base_label,
        base_label,
        words,
    )?;
    public_step_digest(&mut cs.namespace(|| format!("{base_label}_digest")), &step, base_label)
}

pub fn exact_vector_packaged_step_var_from_words_with_step_label<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    namespace_label: &str,
    step_label: &str,
    words: &[U64Var],
) -> Result<PublicStepVar, SynthesisError> {
    let logical_vars = words
        .iter()
        .flat_map(U64Var::limb16_vars)
        .collect::<Vec<_>>();
    let logical_values = words
        .iter()
        .flat_map(|word| {
            word.limb16_values()
                .into_iter()
                .map(|value| F::from_u64(value.to_canonical_u64()))
        })
        .collect::<Vec<_>>();
    build_exact_vector_step(cs, namespace_label, step_label, &logical_vars, &logical_values)
}

pub fn exact_vector_packaged_step_digest_from_native_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    base_label: &str,
    words: &[u64],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let step = exact_vector_packaged_step_var_from_native_words_with_step_label(
        &mut cs.namespace(|| format!("{base_label}_step")),
        base_label,
        base_label,
        words,
    )?;
    public_step_digest(&mut cs.namespace(|| format!("{base_label}_digest")), &step, base_label)
}

pub fn exact_vector_packaged_step_var_from_native_words_with_step_label<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    namespace_label: &str,
    step_label: &str,
    words: &[u64],
) -> Result<PublicStepVar, SynthesisError> {
    let mut cs = cs;
    let word_vars = words
        .iter()
        .enumerate()
        .map(|(idx, word)| {
            alloc_u64(
                cs.namespace(|| format!("{namespace_label}_word_{idx}")),
                *word,
                &format!("{namespace_label}_word_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    exact_vector_packaged_step_var_from_words_with_step_label(&mut cs, namespace_label, step_label, &word_vars)
}

fn build_exact_vector_step<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    namespace_label: &str,
    step_label: &str,
    logical_vars: &[AllocatedNum<SpartanF>],
    logical_values: &[F],
) -> Result<PublicStepVar, SynthesisError> {
    if logical_vars.len() != logical_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let full_width = logical_vars
        .len()
        .checked_add(1)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let params = exact_vector_package_params(full_width)?;
    let cols = commit_cols_for_full_width(full_width);
    let log = ensure_exact_stage_pp(&params, cols)?;
    let packed_values = build_exact_vector_native_packed(&params, logical_values)?;
    let commitment = log.commit(&packed_values);

    let one = alloc_constant(cs.namespace(|| format!("{namespace_label}_one")), SpartanF::ONE, "one")?;
    let zero = alloc_constant(
        cs.namespace(|| format!("{namespace_label}_zero")),
        SpartanF::ZERO,
        "zero",
    )?;
    let matrix_entries = build_packed_matrix_vars(cols, &one, &zero, logical_vars);
    let c_data = alloc_commitment(
        cs.namespace(|| format!("{namespace_label}_commitment")),
        &matrix_entries,
        &commitment,
        &format!("{namespace_label}_commitment"),
    )?;
    let label = format!("{step_label}/selected_claim_package");
    Ok(PublicStepVar {
        claim: CcsClaimVar {
            c_data,
            x: vec![one],
            m_in: 1,
            commitment_d: D,
            commitment_kappa: commitment.kappa,
        },
        label_encoding: alloc_const_packed_bytes(
            cs.namespace(|| format!("{namespace_label}_label")),
            label.as_bytes(),
            &format!("{namespace_label}_label"),
        )?,
    })
}

fn exact_vector_package_params(full_width: usize) -> Result<NeoParams, SynthesisError> {
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|_| SynthesisError::Unsatisfiable)?;
    params.k_rho = EXACT_VECTOR_PACKAGE_K_RHO;
    params.B = EXACT_VECTOR_PACKAGE_B;
    Ok(params)
}

fn ensure_exact_stage_pp(params: &NeoParams, cols: usize) -> Result<AjtaiSModule, SynthesisError> {
    let want_kappa = params.kappa as usize;
    if has_global_pp_for_dims(D, cols) {
        if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, cols) {
            if kappa != want_kappa || seed != rv64im_exact_stage_pp_seed() {
                return Err(SynthesisError::Unsatisfiable);
            }
        }
    } else {
        set_global_pp_seeded(D, want_kappa, cols, rv64im_exact_stage_pp_seed())
            .map_err(|_| SynthesisError::Unsatisfiable)?;
    }
    AjtaiSModule::from_global_for_dims(D, cols).map_err(|_| SynthesisError::Unsatisfiable)
}

fn build_exact_vector_native_packed(params: &NeoParams, values: &[F]) -> Result<Mat<F>, SynthesisError> {
    let mut full_vector = Vec::with_capacity(values.len() + 1);
    full_vector.push(F::ONE);
    full_vector.extend_from_slice(values);
    encode_vector_for_full_width(params, full_vector.len(), &full_vector).map_err(|_| SynthesisError::Unsatisfiable)
}

fn build_packed_matrix_vars(
    cols: usize,
    one: &AllocatedNum<SpartanF>,
    zero: &AllocatedNum<SpartanF>,
    logical_vars: &[AllocatedNum<SpartanF>],
) -> Vec<AllocatedNum<SpartanF>> {
    let mut entries = vec![zero.clone(); D * cols];
    entries[0] = one.clone();
    for (index, value) in logical_vars.iter().enumerate() {
        let logical_index = index + 1;
        let block = logical_index / D;
        let rho = logical_index % D;
        entries[rho * cols + block] = value.clone();
    }
    entries
}

fn alloc_commitment<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    matrix_entries: &[AllocatedNum<SpartanF>],
    commitment: &Commitment,
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    let rows = ajtai_commitment_rows(D, matrix_entries.len() / D)?;
    if rows.len() != commitment.data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    rows.iter()
        .zip(commitment.data.iter())
        .enumerate()
        .map(|(coord_idx, (coeffs, value))| {
            let actual = AllocatedNum::alloc(cs.namespace(|| format!("{label}_{coord_idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })?;
            cs.enforce(
                || format!("{label}_eq_{coord_idx}"),
                |lc| {
                    coeffs
                        .iter()
                        .zip(matrix_entries.iter())
                        .fold(lc, |acc, (coeff, entry)| {
                            if *coeff == F::ZERO {
                                acc
                            } else {
                                acc + (
                                    SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                                    entry.get_variable(),
                                )
                            }
                        })
                },
                |lc| lc + CS::one(),
                |lc| lc + actual.get_variable(),
            );
            Ok(actual)
        })
        .collect()
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

fn alloc_const_packed_bytes<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    bytes: &[u8],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    packed_bytes_field_values(bytes)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant(cs.namespace(|| format!("{label}_{idx}")), value, label))
        .collect()
}

fn alloc_constant<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}
