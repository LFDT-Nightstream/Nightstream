//! NIFS prover orchestration over Metal-backed sumcheck state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::f_prime::compiler::{nifs_ce_shape_from_claim, FPrimeFoldPostSummary};
use neo_fold_clean::paper::digest::AccumulatorHandle;
use neo_fold_clean::paper::nifs::{
    Error, NifsFreshInstancesRequest, NifsPostFoldSummary, NifsProverAdapter, NifsProverOutput, NifsProverRequest,
    NifsRunningCarrier,
};
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim, Structure};
use neo_fold_clean::paper::{pi_ccs, pi_dec, pi_rlc};
use neo_fold_clean::{CcsInstance, CcsWitness, RunningInstance};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::{BackendTranscriptMode, OptimizedStructureCache};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::sumcheck::{MetalFeBackend, MetalNcBackend};
use crate::{
    fold_output::{metal_proof_carrier, metal_running_carrier, MetalFoldOutput, MetalRunningCarrier},
    MetalActivity, MetalAjtaiLowNormPlan, MetalDecFormPlan, MetalError, MetalResidentWitness, MetalSession,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalNifsProfile {
    pub total: Duration,
    pub pi_ccs: Duration,
    pub pi_rlc: Duration,
    pub pi_dec: Duration,
    pub dec_form_build: Duration,
    pub dec_projection: Duration,
    pub dec_host_materialization: Duration,
    pub fe_rounds: usize,
    pub fe_on_metal: bool,
    pub nc_rounds: usize,
    pub nc_on_metal: bool,
    pub folded_tables: usize,
    pub rlc_witness_on_metal: bool,
    pub rlc_witness_resident_only: bool,
    pub rlc_rho_small_coefficients: bool,
    pub dec_split_on_metal: bool,
    pub dec_recomposition_on_metal: bool,
    pub dec_forms_on_metal: bool,
    pub dec_y_on_metal: bool,
    pub dec_commit_on_metal: bool,
    pub resident_running_input: bool,
    pub resident_running_output: bool,
    pub proof_deferred: bool,
    pub running_deferred: bool,
    pub recursive_compile_reverify_required: bool,
    pub activity: MetalActivity,
}

pub struct MetalNifsProver {
    session: MetalSession,
    fresh_commitment_plan: Option<FreshCommitmentPlan>,
    dec_form_plan: Option<MetalDecFormPlan>,
    last_profile: Option<MetalNifsProfile>,
}

struct FreshCommitmentPlan {
    d: usize,
    cols: usize,
    kappa: usize,
    plan: MetalAjtaiLowNormPlan,
}

struct MetalDecOutput {
    witnesses: Vec<Mat<F>>,
    digit_nonzero: Vec<bool>,
    commitments: Vec<Commitment>,
    y_ring: Vec<Vec<[K; D]>>,
    resident_output_id: Option<u64>,
    form_build: Duration,
    projection: Duration,
    host_materialization: Duration,
    forms_on_metal: bool,
}

impl MetalNifsProver {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
            fresh_commitment_plan: None,
            dec_form_plan: None,
            last_profile: None,
        })
    }

    pub fn session(&self) -> &MetalSession {
        &self.session
    }

    pub fn last_profile(&self) -> Option<MetalNifsProfile> {
        self.last_profile
    }

    pub fn take_last_profile(&mut self) -> Option<MetalNifsProfile> {
        self.last_profile.take()
    }

    fn ensure_ajtai_plan(&mut self, log: &neo_ajtai::AjtaiSModule, cols: usize) -> Result<(), Error> {
        let pp = log
            .verification_pp()
            .map_err(|_| backend_unavailable("materialize Ajtai parameters for Metal commitments"))?;
        if pp.d != D || pp.m != cols {
            return Err(backend_unavailable(
                "Ajtai parameter dimensions do not match the Metal witness",
            ));
        }
        let rebuild = self
            .fresh_commitment_plan
            .as_ref()
            .is_none_or(|cached| cached.d != pp.d || cached.cols != pp.m || cached.kappa != pp.kappa);
        if rebuild {
            let matrix = pp
                .m_rows
                .iter()
                .flat_map(|row| {
                    row.iter()
                        .flat_map(|value| value.0.iter().map(PrimeField64::as_canonical_u64))
                })
                .collect::<Vec<_>>();
            let plan = self
                .session
                .prepare_ajtai_low_norm(&matrix, pp.kappa, pp.m)
                .map_err(|_| backend_unavailable("upload Ajtai parameters for Metal commitments"))?;
            self.fresh_commitment_plan = Some(FreshCommitmentPlan {
                d: pp.d,
                cols: pp.m,
                kappa: pp.kappa,
                plan,
            });
        }
        Ok(())
    }

    fn commit_low_norm(&self, message: &[i8]) -> Result<Commitment, Error> {
        let cached = self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan must be installed before use");
        let words = self
            .session
            .ajtai_low_norm_with_plan(&cached.plan, message)
            .map_err(|_| backend_unavailable("compute Metal Ajtai commitment"))?;
        Ok(Commitment {
            d: D,
            kappa: cached.kappa,
            data: words.into_iter().map(F::from_u64).collect(),
        })
    }

    fn ensure_dec_form_plan(&mut self, s: &Structure, cache: &OptimizedStructureCache) -> Result<bool, MetalError> {
        let compact = s
            .matrices
            .iter()
            .any(|matrix| !matrix.seeded_phi81_blocks().is_empty() || !matrix.geometric_runs().is_empty());
        if compact {
            self.dec_form_plan = None;
            return Ok(false);
        }
        if self
            .dec_form_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(cache.superneo()))
        {
            self.dec_form_plan = Some(self.session.prepare_dec_ring_forms(cache.superneo())?);
        }
        Ok(true)
    }

    fn post_fold_summary(&self, running: &RunningInstance) -> Result<NifsPostFoldSummary, Error> {
        let parent = running
            .parent_authority
            .as_ref()
            .ok_or_else(|| backend_unavailable("post-fold running accumulator is missing its Pi_RLC parent"))?;
        let handle = AccumulatorHandle::from_running_parts(&running.claims, Some(parent));
        let f_prime = FPrimeFoldPostSummary {
            parent_shape: nifs_ce_shape_from_claim(parent, 0),
            child_count: running.claims.len() as u64,
            acc_digest: handle.digest_fields(),
        };
        Ok(NifsPostFoldSummary::new(Some(handle.digest()), Some(f_prime)))
    }
}

impl NifsProverAdapter for MetalNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        if request.pp.b() != 2 {
            return Err(backend_unavailable(
                "Metal NIFS currently requires the production b=2 profile",
            ));
        }

        let total_started = Instant::now();
        let activity_before = self.session.activity();
        let mut fresh_claims = Vec::with_capacity(request.fresh.len());
        let mut fresh_witnesses = Vec::with_capacity(request.fresh.len());
        for instance in request.fresh {
            fresh_claims.push(instance.claim);
            fresh_witnesses.push(instance.witness);
        }

        let witness_refs = fresh_witnesses
            .iter()
            .map(|witness| &witness.Z)
            .chain(request.running.witnesses.iter())
            .collect::<Vec<_>>();
        let resident_running_id = metal_running_carrier(request.running_carrier)
            .and_then(MetalRunningCarrier::resident_id)
            .filter(|&id| self.session.resident_running_shape(id).is_some());
        if let Some((child_count, cols)) = resident_running_id.and_then(|id| self.session.resident_running_shape(id)) {
            if child_count != request.running.witnesses.len()
                || request
                    .running
                    .witnesses
                    .iter()
                    .any(|witness| witness.rows() != D || witness.cols() != cols)
            {
                return Err(backend_unavailable(
                    "Metal running carrier does not match the materialized running witness shape",
                ));
            }
        }

        // Keep the experimental sumcheck candidates compiled without paying
        // to prepare production-sized tables when neither backend is selected.
        let fe_backend = MetalFeBackend::new(&self.session);
        let nc_backend = MetalNcBackend::new(&self.session, &[], 0);
        let ccs_started = Instant::now();
        let pi_ccs_proof = pi_ccs::prove_from_parts_with_backends_and_transcript_mode(
            request.tr,
            request.pp,
            request.s,
            request.cache,
            request.log,
            &fresh_claims,
            &fresh_witnesses,
            request.running,
            None,
            None,
            BackendTranscriptMode::Replay,
            None,
            None,
        )?;
        let pi_ccs_elapsed = ccs_started.elapsed();
        let fe_profile = fe_backend.profile();
        let nc_profile = nc_backend.profile();
        if fe_profile.metal_failed || nc_profile.metal_failed {
            return Err(backend_unavailable("Metal sumcheck table folding failed"));
        }

        let rlc_started = Instant::now();
        let (rlc_output, pi_rlc_proof) = pi_rlc::prove_refs_with_resident_witness(
            request.tr,
            request.pp,
            request.s,
            request.mix_rhos_commits,
            &pi_ccs_proof.outputs,
            &witness_refs,
            |rhos, witnesses| {
                mix_witnesses_on_metal(
                    &self.session,
                    rhos,
                    witnesses,
                    fresh_witnesses.len(),
                    resident_running_id,
                )
            },
        )?;
        let pi_rlc_elapsed = rlc_started.elapsed();
        let resident_mix = rlc_output
            .witness
            .map_err(|_| backend_unavailable("Metal Pi_RLC witness mixing failed"))?;

        let dec_started = Instant::now();
        self.ensure_ajtai_plan(request.log, resident_mix.cols())?;
        let forms_on_metal = self
            .ensure_dec_form_plan(request.s, request.cache)
            .map_err(|_| backend_unavailable("prepare Metal Pi_DEC ring forms"))?;
        let commitment_plan = &self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above")
            .plan;
        let dec_material = split_dec_on_metal(
            &self.session,
            request.s,
            request.cache,
            &rlc_output.claim,
            resident_mix.cols(),
            &resident_mix,
            request.pp.k_rho() as usize,
            request.pp.b(),
            request.pp.kappa() as usize,
            request.cache_output_for_next_step,
            commitment_plan,
            forms_on_metal.then_some(
                self.dec_form_plan
                    .as_ref()
                    .expect("Pi_DEC form plan installed above"),
            ),
        )
        .map_err(|_| backend_unavailable("Metal Pi_DEC witness split failed"))?;
        let (dec_output, pi_dec_proof) = pi_dec::prove_from_split_material(
            request.pp,
            request.s,
            request.cache,
            request.lanes,
            request.combine_b_pows,
            &rlc_output.claim,
            dec_material.witnesses,
            dec_material.digit_nonzero,
            dec_material.commitments,
            dec_material.y_ring,
        )?;
        let pi_dec_elapsed = dec_started.elapsed();

        let next_running = RunningInstance {
            claims: dec_output.claims,
            witnesses: dec_output.witnesses,
            parent_authority: Some(rlc_output.claim),
        };
        let post_fold_summary = self.post_fold_summary(&next_running)?;
        let activity = activity_delta(activity_before, self.session.activity());
        let resident_running_output = request.cache_output_for_next_step;
        self.last_profile = Some(MetalNifsProfile {
            total: total_started.elapsed(),
            pi_ccs: pi_ccs_elapsed,
            pi_rlc: pi_rlc_elapsed,
            pi_dec: pi_dec_elapsed,
            dec_form_build: dec_material.form_build,
            dec_projection: dec_material.projection,
            dec_host_materialization: dec_material.host_materialization,
            fe_rounds: 0,
            fe_on_metal: false,
            nc_rounds: 0,
            nc_on_metal: false,
            folded_tables: fe_profile.folded_tables,
            rlc_witness_on_metal: true,
            rlc_witness_resident_only: true,
            rlc_rho_small_coefficients: true,
            dec_split_on_metal: true,
            dec_recomposition_on_metal: true,
            dec_forms_on_metal: dec_material.forms_on_metal,
            dec_y_on_metal: true,
            dec_commit_on_metal: true,
            resident_running_input: resident_running_id.is_some(),
            resident_running_output,
            proof_deferred: true,
            running_deferred: resident_running_output,
            recursive_compile_reverify_required: false,
            activity,
        });
        let resident_output_id = if resident_running_output {
            Some(
                dec_material
                    .resident_output_id
                    .ok_or_else(|| backend_unavailable("Metal Pi_DEC resident output was not retained"))?,
            )
        } else {
            None
        };
        let output = Arc::new(MetalFoldOutput::new(next_running, resident_output_id));
        let proof = metal_proof_carrier(pi_ccs_proof, pi_rlc_proof, pi_dec_proof, Arc::clone(&output))?;
        if resident_running_output {
            Ok(NifsProverOutput::deferred(
                NifsRunningCarrier::deferred(Arc::new(MetalRunningCarrier::new(output))),
                proof,
            )
            .with_post_fold_summary(post_fold_summary))
        } else {
            Ok(
                NifsProverOutput::deferred(NifsRunningCarrier::materialized(output.running().clone()), proof)
                    .with_post_fold_summary(post_fold_summary),
            )
        }
    }

    fn requires_recursive_compile_reverify(&self) -> bool {
        false
    }

    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        let valid = request.assignments.iter().all(|assignment| {
            assignment.len() == request.s.m
                && request.m_in <= assignment.len()
                && assignment
                    .iter()
                    .all(|&value| neo_math::balanced::within_nc_bound(value, request.pp.b()))
        });
        if !valid {
            return Ok(None);
        }

        let cols = request.s.m.div_ceil(D);
        if request.log.dims() != (D, cols) {
            return Ok(None);
        }
        self.ensure_ajtai_plan(request.log, cols)?;

        let mut instances = Vec::with_capacity(request.assignments.len());
        for assignment in request.assignments {
            let mut message = assignment
                .iter()
                .map(|&value| neo_math::balanced::to_balanced_i128(value) as i8)
                .collect::<Vec<_>>();
            message.resize(cols * D, 0);
            let commitment = self.commit_low_norm(&message)?;
            let mut z = Mat::zero(D, cols, F::ZERO);
            for (index, &value) in assignment.iter().enumerate() {
                z.set(index % D, index / D, value);
            }
            instances.push(CcsInstance {
                claim: CcsClaim {
                    adv: None,
                    c: commitment,
                    x: assignment[..request.m_in].to_vec(),
                    m_in: request.m_in,
                },
                witness: CcsWitness { w: Vec::new(), Z: z },
            });
        }
        Ok(Some(instances))
    }
}

fn mix_witnesses_on_metal(
    session: &MetalSession,
    rhos: &[Mat<F>],
    witnesses: &[&Mat<F>],
    fresh_count: usize,
    resident_id: Option<u64>,
) -> Result<MetalResidentWitness, MetalError> {
    let Some(cols) = witnesses.first().map(|witness| witness.cols()) else {
        return Err(MetalError::Shape("Pi_RLC witness input is empty"));
    };
    if rhos.len() != witnesses.len()
        || rhos.iter().any(|rho| rho.rows() != D || rho.cols() != D)
        || witnesses
            .iter()
            .any(|witness| witness.rows() != D || witness.cols() != cols)
    {
        return Err(MetalError::Shape("Pi_RLC witness inputs have inconsistent dimensions"));
    }
    let mut rho_coefficients = Vec::with_capacity(rhos.len() * D * D);
    for rho in rhos {
        for row in 0..D {
            for column in 0..D {
                let coefficient = neo_math::balanced::to_balanced_i128(rho[(row, column)]);
                if !(-6..=6).contains(&coefficient) {
                    return Err(MetalError::Shape(
                        "Pi_RLC rho entry exceeds the fixed Phi81 challenge bound",
                    ));
                }
                rho_coefficients.push(coefficient as i8);
            }
        }
    }
    let uploaded_witnesses = if resident_id.is_some() {
        &witnesses[..fresh_count]
    } else {
        witnesses
    };
    let witness_words = uploaded_witnesses
        .iter()
        .flat_map(|witness| {
            (0..D).flat_map(move |row| (0..cols).map(move |column| witness[(row, column)].as_canonical_u64()))
        })
        .collect::<Vec<_>>();
    if let Some(resident_id) = resident_id {
        session.mix_rlc_witnesses_with_resident_id(
            &rho_coefficients,
            &witness_words,
            fresh_count,
            witnesses.len(),
            cols,
            resident_id,
        )
    } else {
        session.mix_rlc_witnesses_resident(&rho_coefficients, &witness_words, witnesses.len(), cols)
    }
}

fn split_dec_on_metal(
    session: &MetalSession,
    s: &Structure,
    cache: &OptimizedStructureCache,
    claim: &CeClaim,
    parent_cols: usize,
    resident_parent: &MetalResidentWitness,
    child_count: usize,
    base: u32,
    kappa: usize,
    retain_resident: bool,
    commitment_plan: &MetalAjtaiLowNormPlan,
    form_plan: Option<&MetalDecFormPlan>,
) -> Result<MetalDecOutput, MetalError> {
    if base != 2 || parent_cols == 0 {
        return Err(MetalError::Shape(
            "Metal Pi_DEC supports the production base-2 packed witness",
        ));
    }
    let entries = D * parent_cols;
    let form_started = Instant::now();
    let row_domain = 1usize
        .checked_shl(claim.r.len() as u32)
        .ok_or(MetalError::Shape("Pi_DEC row challenge dimensions overflow"))?;
    let chi_r = neo_ccs::utils::tensor_point_parallel::<K>(&claim.r);
    let n_eff = s.n.min(row_domain);
    let chi_words = chi_r
        .iter()
        .flat_map(|value| {
            let (real, imaginary) = value.to_limbs_u64();
            [real, imaginary]
        })
        .collect::<Vec<_>>();
    let dense_forms = if form_plan.is_none() {
        let forms = cache.superneo().build_ring_linear_forms(&chi_r, n_eff);
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
    let material = match form_plan {
        Some(plan) => session.split_dec_base2_with_ring_form_plan(
            resident_parent,
            child_count,
            plan,
            &chi_words,
            n_eff,
            commitment_plan,
        )?,
        None => session.split_dec_base2_with_ring_forms(
            resident_parent,
            child_count,
            form_rows,
            dense_forms.as_deref().expect("dense forms built above"),
            commitment_plan,
        )?,
    };
    let projection = projection_started.elapsed();
    let host_started = Instant::now();
    let expected_mask_words = child_count
        .checked_mul(parent_cols)
        .and_then(|words| words.checked_mul(2))
        .ok_or(MetalError::Shape("Metal Pi_DEC mask dimensions overflow"))?;
    if material.child_mask_words.len() != expected_mask_words {
        return Err(MetalError::Shape("Metal Pi_DEC output dimensions are inconsistent"));
    }
    let mut children = Vec::with_capacity(child_count);
    for masks in material.child_mask_words.chunks_exact(parent_cols * 2) {
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
    let expected_y_words = child_count * form_rows * D;
    if material.y_words.len() != expected_y_words {
        return Err(MetalError::Shape("Metal Pi_DEC y output dimensions are inconsistent"));
    }
    let y_ring = (0..child_count)
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
    let words_per_commitment = kappa * D;
    if material.commitment_words.len() != child_count * words_per_commitment {
        return Err(MetalError::Shape("Metal Pi_DEC commitment dimensions are inconsistent"));
    }
    let commitments = material
        .commitment_words
        .chunks_exact(words_per_commitment)
        .map(|words| Commitment {
            d: D,
            kappa,
            data: words.iter().copied().map(F::from_u64).collect(),
        })
        .collect();
    let resident_id = retain_resident.then(|| session.retain_running_children(material.resident_children));
    Ok(MetalDecOutput {
        witnesses: children,
        digit_nonzero: nonzero,
        commitments,
        y_ring,
        resident_output_id: resident_id,
        form_build,
        projection,
        host_materialization: host_started.elapsed(),
        forms_on_metal: form_plan.is_some(),
    })
}

fn backend_unavailable(reason: &'static str) -> Error {
    Error::BackendUnavailable {
        backend: "metal",
        reason,
    }
}

fn activity_delta(before: MetalActivity, after: MetalActivity) -> MetalActivity {
    MetalActivity {
        command_buffers: after.command_buffers.saturating_sub(before.command_buffers),
        dispatches: after.dispatches.saturating_sub(before.dispatches),
        host_waits: after.host_waits.saturating_sub(before.host_waits),
        allocated_bytes: after.allocated_bytes.saturating_sub(before.allocated_bytes),
        uploaded_bytes: after.uploaded_bytes.saturating_sub(before.uploaded_bytes),
        downloaded_bytes: after
            .downloaded_bytes
            .saturating_sub(before.downloaded_bytes),
        current_allocated_bytes: after.current_allocated_bytes,
    }
}
