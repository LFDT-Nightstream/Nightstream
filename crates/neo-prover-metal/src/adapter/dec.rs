//! Pi_DEC projection, digit splitting, and resident child-claim construction.

use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{CeClaim, Structure};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{commitment_from_words, FreshLaneCommitmentPlan};
use crate::{
    MetalAjtaiLowNormPlan, MetalAjtaiRingForms, MetalDecFormPlan, MetalDecPublicProjection, MetalError,
    MetalResidentWitness, MetalResidentWitnessSnapshot, MetalSession,
};

/// Host-visible Pi_DEC material plus ownership handles for resident children.
pub(super) struct MetalDecOutput {
    pub(super) witnesses: Vec<Mat<F>>,
    pub(super) digit_nonzero: Vec<bool>,
    pub(super) commitments: Vec<Commitment>,
    pub(super) child_adv: Option<Vec<LaneCommitments<Commitment>>>,
    pub(super) y_ring: Vec<Vec<[K; D]>>,
    pub(super) resident_claims: Option<Vec<CeClaim>>,
    pub(super) witness_snapshot: Option<MetalResidentWitnessSnapshot>,
    pub(super) resident_output_id: Option<u64>,
    pub(super) form_build: Duration,
    pub(super) projection: Duration,
    pub(super) lane_commit_gpu: Duration,
    pub(super) y_zcol_gpu: Duration,
    pub(super) host_materialization: Duration,
    pub(super) forms_on_metal: bool,
}

/// Selects the available form source, performs the complete base-2 split and
/// projections, and materializes witnesses only when residency is not retained.
pub(super) fn split_dec_on_metal(
    session: &MetalSession,
    params: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    claim: &CeClaim,
    parent_cols: usize,
    resident_parent: &mut MetalResidentWitness,
    child_count: usize,
    retain_resident: bool,
    commitment_plan: &MetalAjtaiLowNormPlan,
    lane_plan: Option<&FreshLaneCommitmentPlan>,
    form_plan: Option<&MetalDecFormPlan>,
    prebuilt_forms: Option<&MetalAjtaiRingForms>,
) -> Result<MetalDecOutput, MetalError> {
    if params.b() != 2 || parent_cols == 0 {
        return Err(MetalError::Shape(
            "Metal Pi_DEC supports the production base-2 packed witness",
        ));
    }
    let kappa = params.kappa() as usize;
    let entries = D * parent_cols;
    let form_started = Instant::now();
    let row_domain = 1usize
        .checked_shl(claim.r.len() as u32)
        .ok_or(MetalError::Shape("Pi_DEC row challenge dimensions overflow"))?;
    let n_eff = s.n.min(row_domain);
    let chi_r = prebuilt_forms
        .is_none()
        .then(|| neo_ccs::utils::tensor_point_parallel::<K>(&claim.r));
    let dense_forms = if form_plan.is_none() {
        let forms = cache
            .superneo()
            .build_ring_linear_forms(chi_r.as_deref().expect("chi built without prebuilt forms"), n_eff);
        if forms.len() != s.t() {
            return Err(MetalError::Shape("Pi_DEC ring-form count does not match the CCS"));
        }
        let mut words = Vec::with_capacity(2 * s.t() * entries);
        for form in forms {
            let (real, imaginary) = form.to_dense_block_coeffs();
            if real.len() != entries || imaginary.len() != entries {
                return Err(MetalError::Shape("Pi_DEC ring-form width does not match the witness"));
            }
            words.extend(real.into_iter().map(|value| value.as_canonical_u64()));
            words.extend(imaginary.into_iter().map(|value| value.as_canonical_u64()));
        }
        Some(words)
    } else {
        None
    };
    let form_rows = 2 * s.t();
    let form_build = form_started.elapsed();
    let projection_started = Instant::now();
    let public_projection = if retain_resident && !claim.s_col.is_empty() {
        Some(MetalDecPublicProjection {
            active_rows: s.m,
            s_col: &claim.s_col,
        })
    } else {
        if retain_resident && !claim.y_zcol.is_empty() {
            return Err(MetalError::Shape("Metal Pi_DEC parent has an incomplete NC channel"));
        }
        None
    };
    let material = match (form_plan, prebuilt_forms) {
        (Some(plan), Some(forms)) => session.split_dec_base2_with_prebuilt_ring_forms(
            resident_parent,
            child_count,
            plan,
            forms,
            &claim.r,
            n_eff,
            public_projection,
            commitment_plan,
        )?,
        (Some(plan), None) => session.split_dec_base2_with_ring_form_plan(
            resident_parent,
            child_count,
            plan,
            cache.superneo(),
            chi_r.as_deref().expect("chi built without prebuilt forms"),
            n_eff,
            public_projection,
            commitment_plan,
        )?,
        (None, _) => session.split_dec_base2_with_ring_forms(
            resident_parent,
            child_count,
            form_rows,
            dense_forms.as_deref().expect("dense forms built above"),
            public_projection,
            commitment_plan,
        )?,
    };
    let projection = projection_started.elapsed();
    let (child_adv, lane_commit_gpu) = if let Some(lane_plan) = lane_plan {
        let masks = session.resident_child_masks(&material.resident_children, s.m)?;
        let (words, gpu) = session.ajtai_lane_commitments_from_masks(
            &lane_plan.ops,
            &lane_plan.mem,
            &masks,
            child_count,
            parent_cols,
            &lane_plan.ranges,
        )?;
        let words_per_commitment = lane_plan.kappa * D;
        if words.len() != 3 * child_count * words_per_commitment {
            return Err(MetalError::Shape(
                "Metal Pi_DEC lane commitments have inconsistent dimensions",
            ));
        }
        let commitment = |lane: usize, child: usize| {
            let start = (lane * child_count + child) * words_per_commitment;
            commitment_from_words(&words[start..start + words_per_commitment], lane_plan.kappa)
        };
        let child_adv = (0..child_count)
            .map(|child| LaneCommitments {
                ops: commitment(0, child),
                is: commitment(1, child),
                fs: commitment(2, child),
            })
            .collect();
        (Some(child_adv), gpu)
    } else {
        (None, Duration::ZERO)
    };
    let (y_zcol_words, y_zcol_gpu) = (material.y_zcol_words, material.y_zcol_gpu);
    let host_started = Instant::now();
    let public_cols = claim.m_in.div_ceil(D);
    let public_masks = retain_resident
        .then(|| session.resident_child_mask_prefix(&material.resident_children, public_cols))
        .transpose()?;
    let children = if retain_resident {
        // These shape-only placeholders keep the ordinary RunningInstance
        // lightweight. The deferred carrier replaces them from its immutable
        // mask snapshot whenever execution crosses back to the CPU.
        (0..child_count)
            .map(|_| Mat::virtual_constant(D, parent_cols, F::ZERO))
            .collect::<Vec<_>>()
    } else {
        let child_mask_words = session.materialize_resident_child_masks(&material.resident_children);
        let expected_mask_words = child_count
            .checked_mul(parent_cols)
            .and_then(|words| words.checked_mul(2))
            .ok_or(MetalError::Shape("Metal Pi_DEC mask dimensions overflow"))?;
        if child_mask_words.len() != expected_mask_words {
            return Err(MetalError::Shape("Metal Pi_DEC output dimensions are inconsistent"));
        }
        let mut children = Vec::with_capacity(child_count);
        for masks in child_mask_words.chunks_exact(parent_cols * 2) {
            let mut positive = Vec::with_capacity(parent_cols);
            let mut negative = Vec::with_capacity(parent_cols);
            for pair in masks.chunks_exact(2) {
                positive.push(pair[0]);
                negative.push(pair[1]);
            }
            children.push(
                Mat::compact_signed_unit_from_column_masks(D, parent_cols, &positive, &negative)
                    .map_err(MetalError::Shape)?,
            );
        }
        children
    };
    let expected_y_words = child_count * form_rows * D;
    if material.y_words.len() != expected_y_words {
        return Err(MetalError::Shape("Metal Pi_DEC y output dimensions are inconsistent"));
    }
    let y_ring: Vec<Vec<[K; D]>> = (0..child_count)
        .map(|child| {
            (0..s.t())
                .map(|matrix| {
                    let real_base = (child * form_rows + 2 * matrix) * D;
                    let imaginary_base = real_base + D;
                    std::array::from_fn(|coefficient| {
                        K::from_coeffs([
                            F::from_u64(material.y_words[real_base + coefficient]),
                            F::from_u64(material.y_words[imaginary_base + coefficient]),
                        ])
                    })
                })
                .collect()
        })
        .collect();
    let nonzero = if retain_resident {
        material.child_nonzero.clone()
    } else {
        let nonzero = children
            .iter()
            .map(|child| {
                child
                    .packed_signed_unit_nonzero_count()
                    .is_some_and(|count| count != 0)
            })
            .collect::<Vec<_>>();
        if material.child_nonzero != nonzero {
            return Err(MetalError::Shape(
                "Metal Pi_DEC nonzero mask does not match the materialized children",
            ));
        }
        nonzero
    };
    let words_per_commitment = kappa * D;
    if material.commitment_words.len() != child_count * words_per_commitment {
        return Err(MetalError::Shape("Metal Pi_DEC commitment dimensions are inconsistent"));
    }
    let commitments: Vec<Commitment> = material
        .commitment_words
        .chunks_exact(words_per_commitment)
        .map(|words| Commitment {
            d: D,
            kappa,
            data: words.iter().copied().map(F::from_u64).collect(),
        })
        .collect();
    // Claims are ordinary protocol values even when their projections came
    // from Metal; only witness storage remains deferred behind the carrier.
    let resident_claims = if let Some(public_masks) = public_masks {
        let child_x = child_public_x_from_masks(&public_masks, child_count, claim.m_in)?;
        let child_y_zcol = child_y_zcol_from_words(&y_zcol_words, child_count, !claim.s_col.is_empty())?;
        Some(build_resident_dec_claims(
            params,
            s,
            claim,
            &commitments,
            child_adv.as_deref(),
            &y_ring,
            child_x,
            child_y_zcol,
        )?)
    } else {
        None
    };
    let witness_snapshot = retain_resident.then(|| material.resident_children.snapshot());
    let resident_id = if retain_resident {
        Some(session.retain_running_children(material.resident_children))
    } else {
        drop(material.resident_children);
        None
    };
    Ok(MetalDecOutput {
        witnesses: children,
        digit_nonzero: nonzero,
        commitments,
        child_adv,
        y_ring,
        resident_claims,
        witness_snapshot,
        resident_output_id: resident_id,
        form_build,
        projection,
        lane_commit_gpu,
        y_zcol_gpu,
        host_materialization: host_started.elapsed(),
        forms_on_metal: form_plan.is_some(),
    })
}

fn child_public_x_from_masks(words: &[u64], child_count: usize, m_in: usize) -> Result<Vec<Mat<F>>, MetalError> {
    let public_cols = m_in.div_ceil(D);
    if words.len() != 2 * child_count * public_cols {
        return Err(MetalError::Shape(
            "Metal Pi_DEC public-mask dimensions are inconsistent",
        ));
    }
    let valid_rows = (1u64 << D) - 1;
    let negative_one = F::ZERO - F::ONE;
    let mut children = Vec::with_capacity(child_count);
    for child in 0..child_count {
        let mut x = Mat::zero(D, m_in, F::ZERO);
        for column in 0..public_cols {
            let base = 2 * (child * public_cols + column);
            let positive = words[base];
            let negative = words[base + 1];
            if positive & negative != 0 || (positive | negative) & !valid_rows != 0 {
                return Err(MetalError::Shape("Metal Pi_DEC public masks are not signed-unit"));
            }
            for row in 0..D {
                let bit = 1u64 << row;
                x[(row, column)] = if positive & bit != 0 {
                    F::ONE
                } else if negative & bit != 0 {
                    negative_one
                } else {
                    F::ZERO
                };
            }
        }
        children.push(x);
    }
    Ok(children)
}

fn child_y_zcol_from_words(words: &[u64], child_count: usize, enabled: bool) -> Result<Vec<Vec<K>>, MetalError> {
    if !enabled {
        return words
            .is_empty()
            .then(|| vec![Vec::new(); child_count])
            .ok_or(MetalError::Shape("Metal Pi_DEC emitted an unexpected NC channel"));
    }
    if words.len() != 2 * child_count * D {
        return Err(MetalError::Shape("Metal Pi_DEC NC-channel dimensions are inconsistent"));
    }
    Ok((0..child_count)
        .map(|child| {
            let mut values = vec![K::ZERO; D.next_power_of_two()];
            for (row, value) in values[..D].iter_mut().enumerate() {
                let base = 2 * (child * D + row);
                *value = K::from_coeffs([F::from_u64(words[base]), F::from_u64(words[base + 1])]);
            }
            values
        })
        .collect())
}

/// Reconstructs canonical child claims from the projections returned by Metal;
/// no resident buffer or generation id becomes part of these protocol values.
#[allow(clippy::too_many_arguments)]
fn build_resident_dec_claims(
    params: &Params,
    s: &Structure,
    parent: &CeClaim,
    commitments: &[Commitment],
    child_adv: Option<&[LaneCommitments<Commitment>]>,
    y_ring: &[Vec<[K; D]>],
    child_x: Vec<Mat<F>>,
    child_y_zcol: Vec<Vec<K>>,
) -> Result<Vec<CeClaim>, MetalError> {
    let child_count = commitments.len();
    if y_ring.len() != child_count
        || child_x.len() != child_count
        || child_y_zcol.len() != child_count
        || child_adv.is_some_and(|values| values.len() != child_count)
    {
        return Err(MetalError::Shape(
            "Metal Pi_DEC resident claim dimensions are inconsistent",
        ));
    }
    Ok((0..child_count)
        .map(|child| {
            let y_ring = y_ring[child]
                .iter()
                .map(|coefficients| {
                    let mut row = coefficients.to_vec();
                    row.resize(D.next_power_of_two(), K::ZERO);
                    row
                })
                .collect::<Vec<_>>();
            let ct = neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_ring, params.inner(), s.m);
            CeClaim {
                adv: child_adv.map(|values| values[child].clone()),
                c_step_coords: Vec::new(),
                u_offset: 0,
                u_len: 0,
                c: commitments[child].clone(),
                X: child_x[child].clone(),
                r: parent.r.clone(),
                s_col: parent.s_col.clone(),
                y_ring,
                ct,
                // The additive split carries the parent's auxiliary term once;
                // copying it to every child would change recomposition.
                aux_openings: if child == 0 {
                    parent.aux_openings.clone()
                } else {
                    vec![K::ZERO; parent.aux_openings.len()]
                },
                y_zcol: child_y_zcol[child].clone(),
                m_in: parent.m_in,
                fold_digest: parent.fold_digest,
            }
        })
        .collect())
}
