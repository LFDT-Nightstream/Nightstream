//! NIFS prover orchestration over Metal-backed sumcheck state.

use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::frontends::f_prime::compiler::{nifs_ce_shape_from_claim, FPrimeFoldPostSummary};
use neo_fold_clean::paper::digest::{self, AccumulatorHandle};
use neo_fold_clean::paper::nifs::{
    Error, NifsFreshInstancesRequest, NifsPostFoldSummary, NifsProverAdapter, NifsProverOutput, NifsProverRequest,
    NifsRunningCarrier,
};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    CE_CLAIM_SIS_CONFIG, PI_CCS_OUTPUTS_SIS_CONFIG, PI_RLC_PROJECTION_SIS_CONFIG,
};
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim, LaneRanges, LaneScheme, Structure};
use neo_fold_clean::paper::{pi_ccs, pi_dec, pi_rlc};
use neo_fold_clean::{CcsInstance, CcsWitness, RunningInstance};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::{BackendTranscriptMode, OptimizedStructureCache};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::sumcheck::{MetalFeBackend, MetalNcBackend};
use crate::{
    fold_output::{metal_proof_carrier, metal_running_carrier, MetalFoldOutput, MetalRunningCarrier},
    MetalActivity, MetalAjtaiLowNormPlan, MetalAjtaiRingForms, MetalDecFormPlan, MetalDecPublicProjection, MetalError,
    MetalFeOraclePlan, MetalResidentWitness, MetalResidentWitnessSnapshot, MetalSession, MetalWitnessMasks,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalNifsProfile {
    pub total: Duration,
    pub fresh_commit_total: Duration,
    pub fresh_commit_gpu: Duration,
    pub fresh_commit_count: usize,
    pub fresh_masks_reused: bool,
    pub fresh_lane_commit_gpu: Duration,
    pub fresh_lane_commit_count: usize,
    pub fresh_lanes_from_resident_masks: bool,
    pub pi_ccs: Duration,
    pub fe_sumcheck: Duration,
    pub nc_sumcheck: Duration,
    pub pi_ccs_activity: MetalActivity,
    pub ajtai_y_eval: Duration,
    pub ajtai_seeded_build: Duration,
    pub ajtai_device_eval: Duration,
    pub ajtai_tensor_gpu: Duration,
    pub ajtai_form_gpu: Duration,
    pub ajtai_tail_gpu: Duration,
    pub ajtai_seeded_patch_entries: usize,
    pub ajtai_seeded_patch_bytes: usize,
    pub ajtai_form_blocks: usize,
    pub ajtai_form_bytes: usize,
    pub ajtai_explicit_coefficients: usize,
    pub ajtai_signed_unit_coefficients: usize,
    /// Explicit-list bins: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, and 64+ entries.
    pub ajtai_explicit_form_list_histogram: [usize; 8],
    pub ajtai_max_explicit_form_list_entries: usize,
    pub ajtai_parallel_form_lists: usize,
    pub ajtai_parallel_form_entries: usize,
    pub pi_rlc: Duration,
    pub pi_rlc_activity: MetalActivity,
    pub pi_dec: Duration,
    pub pi_dec_activity: MetalActivity,
    pub dec_form_build: Duration,
    pub dec_projection: Duration,
    pub dec_lane_commit_gpu: Duration,
    pub dec_y_zcol_gpu: Duration,
    pub dec_host_materialization: Duration,
    pub fe_rounds: usize,
    pub fe_mcs_tables: usize,
    pub fe_mcs_table_bytes: usize,
    pub fe_seeded_build: Duration,
    pub fe_seeded_patch_entries: usize,
    pub fe_seeded_patch_bytes: usize,
    pub fe_explicit_coefficients: usize,
    /// Explicit row-list bins: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, and 64+ entries.
    pub fe_explicit_row_list_histogram: [usize; 8],
    pub fe_max_explicit_row_entries: usize,
    pub fe_carried_eval_on_metal: bool,
    pub fe_on_metal: bool,
    pub ajtai_y_eval_on_metal: bool,
    pub nc_rounds: usize,
    pub nc_input_witnesses: usize,
    pub nc_active_witnesses: usize,
    pub nc_on_metal: bool,
    pub nc_mask_native_on_metal: bool,
    pub witness_masks_shared: bool,
    pub folded_tables: usize,
    pub rlc_witness_on_metal: bool,
    pub rlc_witness_resident_only: bool,
    pub rlc_witness_masks_reused: bool,
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
    fresh_lane_commitment_plan: Option<FreshLaneCommitmentPlan>,
    dec_form_plan: Option<MetalDecFormPlan>,
    fe_oracle_plan: Option<MetalFeOraclePlan>,
    cached_fresh_masks: Option<CachedFreshMasks>,
    last_profile: Option<MetalNifsProfile>,
}

struct FreshCommitmentPlan {
    d: usize,
    cols: usize,
    kappa: usize,
    plan: MetalAjtaiLowNormPlan,
}

struct FreshLaneCommitmentPlan {
    kappa: usize,
    ops_seed: [u8; 32],
    mem_seed: [u8; 32],
    ranges: LaneRanges,
    ops: MetalAjtaiLowNormPlan,
    mem: MetalAjtaiLowNormPlan,
}

struct CachedFreshMasks {
    host_commitments: Vec<Commitment>,
    masks: MetalWitnessMasks,
    commit_total: Duration,
    commit_gpu: Duration,
    lane_commit_gpu: Duration,
    lane_commit_count: usize,
}

struct MetalDecOutput {
    witnesses: Vec<Mat<F>>,
    digit_nonzero: Vec<bool>,
    commitments: Vec<Commitment>,
    child_adv: Option<Vec<LaneCommitments<Commitment>>>,
    y_ring: Vec<Vec<[K; D]>>,
    resident_claims: Option<Vec<CeClaim>>,
    witness_snapshot: Option<MetalResidentWitnessSnapshot>,
    resident_output_id: Option<u64>,
    form_build: Duration,
    projection: Duration,
    lane_commit_gpu: Duration,
    y_zcol_gpu: Duration,
    host_materialization: Duration,
    forms_on_metal: bool,
}

struct MetalPiCcsOutputsDigest<'a> {
    session: &'a MetalSession,
}

impl pi_ccs::PiCcsOutputsDigestBackend for MetalPiCcsOutputsDigest<'_> {
    fn digest_outputs(&mut self, outputs: &[CeClaim]) -> Result<[F; 4], pi_ccs::Error> {
        self.session
            .sis_accumulator_digest_resident(
                PI_CCS_OUTPUTS_SIS_CONFIG,
                &digest::pi_ccs_outputs_digest_preimage(outputs),
            )
            .map_err(|_| pi_ccs::Error::OutputDigestBackend)
    }
}

impl MetalNifsProver {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
            fresh_commitment_plan: None,
            fresh_lane_commitment_plan: None,
            dec_form_plan: None,
            fe_oracle_plan: None,
            cached_fresh_masks: None,
            last_profile: None,
        })
    }

    pub fn session(&self) -> &MetalSession {
        &self.session
    }

    /// Prepare verifier-owned Metal state before the online fold loop.
    ///
    /// The static matrix views and seeded commitment plans remain resident
    /// and are reused by every subsequent proof for the same preprocessing
    /// context. Skipping this call preserves correctness and prepares them
    /// lazily during the first fold.
    pub fn prepare_static(
        &mut self,
        log: &neo_ajtai::AjtaiSModule,
        s: &Structure,
        cache: &OptimizedStructureCache,
        lanes: Option<&LaneScheme>,
    ) -> Result<(), Error> {
        let cols = s.m.div_ceil(D);
        self.ensure_ajtai_plan(log, cols)?;
        if let Some(lanes) = lanes {
            self.ensure_lane_ajtai_plan(lanes, cols)?;
        }
        self.ensure_dec_form_plan(s, cache)
            .map_err(|_| backend_unavailable("prepare static Metal matrix state"))?;
        Ok(())
    }

    pub fn last_profile(&self) -> Option<MetalNifsProfile> {
        self.last_profile
    }

    pub fn take_last_profile(&mut self) -> Option<MetalNifsProfile> {
        self.last_profile.take()
    }

    fn ensure_ajtai_plan(&mut self, log: &neo_ajtai::AjtaiSModule, cols: usize) -> Result<(), Error> {
        let (d, m) = log.dims();
        let kappa = log.kappa();
        if d != D || m != cols {
            return Err(backend_unavailable(
                "Ajtai parameter dimensions do not match the Metal witness",
            ));
        }
        let rebuild = self
            .fresh_commitment_plan
            .as_ref()
            .is_none_or(|cached| cached.d != d || cached.cols != m || cached.kappa != kappa);
        if rebuild {
            let plan = if let Some((seeded_kappa, seed)) = log.seeded_params() {
                if seeded_kappa != kappa {
                    return Err(backend_unavailable("seeded Ajtai parameters have inconsistent kappa"));
                }
                self.session.prepare_ajtai_low_norm_seeded(seed, kappa, m)
            } else {
                let pp = log
                    .verification_pp()
                    .map_err(|_| backend_unavailable("materialize Ajtai parameters for Metal commitments"))?;
                let matrix = pp
                    .m_rows
                    .iter()
                    .flat_map(|row| {
                        row.iter()
                            .flat_map(|value| value.0.iter().map(PrimeField64::as_canonical_u64))
                    })
                    .collect::<Vec<_>>();
                self.session.prepare_ajtai_low_norm(&matrix, kappa, m)
            }
            .map_err(|_| backend_unavailable("prepare Ajtai parameters for Metal commitments"))?;
            self.fresh_commitment_plan = Some(FreshCommitmentPlan {
                d,
                cols: m,
                kappa,
                plan,
            });
        }
        Ok(())
    }

    fn ensure_lane_ajtai_plan(&mut self, scheme: &LaneScheme, full_cols: usize) -> Result<(), Error> {
        let ranges = scheme.lane_ranges();
        let (kappa, ops_seed, mem_seed) = scheme.seeded_setup();
        if ranges.ops.end > full_cols
            || ranges.is.end > full_cols
            || ranges.fs.end > full_cols
            || self
                .fresh_commitment_plan
                .as_ref()
                .is_none_or(|plan| plan.kappa != kappa)
        {
            return Err(backend_unavailable(
                "Nebula lane parameters do not match the Metal witness commitment",
            ));
        }
        let rebuild = self
            .fresh_lane_commitment_plan
            .as_ref()
            .is_none_or(|cached| {
                cached.kappa != kappa
                    || cached.ops_seed != ops_seed
                    || cached.mem_seed != mem_seed
                    || cached.ranges != *ranges
            });
        if rebuild {
            let ops = self
                .session
                .prepare_ajtai_low_norm_seeded(ops_seed, kappa, ranges.ops.len())
                .map_err(|_| backend_unavailable("prepare seeded Metal Nebula ops commitment"))?;
            let mem = self
                .session
                .prepare_ajtai_low_norm_seeded(mem_seed, kappa, ranges.is.len())
                .map_err(|_| backend_unavailable("prepare seeded Metal Nebula memory commitment"))?;
            self.fresh_lane_commitment_plan = Some(FreshLaneCommitmentPlan {
                kappa,
                ops_seed,
                mem_seed,
                ranges: ranges.clone(),
                ops,
                mem,
            });
        }
        Ok(())
    }

    fn ensure_dec_form_plan(&mut self, s: &Structure, cache: &OptimizedStructureCache) -> Result<bool, MetalError> {
        let _ = s;
        if self
            .fe_oracle_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(cache.superneo()))
        {
            self.fe_oracle_plan = Some(self.session.prepare_fe_oracle(cache.superneo())?);
        }
        if self
            .dec_form_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(cache.superneo()))
        {
            self.dec_form_plan = Some(
                self.session.prepare_dec_ring_forms(
                    cache.superneo(),
                    self.fe_oracle_plan
                        .as_ref()
                        .expect("FE oracle plan installed above"),
                )?,
            );
        }
        Ok(true)
    }

    fn post_fold_summary(
        &self,
        running: &RunningInstance,
        parent_digest: [F; 4],
    ) -> Result<NifsPostFoldSummary, Error> {
        let parent = running
            .parent_authority
            .as_ref()
            .ok_or_else(|| backend_unavailable("post-fold running accumulator is missing its Pi_RLC parent"))?;
        let handle = AccumulatorHandle::from_parent_digest(running.claims.len(), Some(parent_digest));
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
        let cached_fresh_masks = self.cached_fresh_masks.take().filter(|cached| {
            cached.host_commitments.len() == fresh_claims.len()
                && cached
                    .masks
                    .matches(fresh_claims.len(), request.s.m.div_ceil(D))
                && cached
                    .host_commitments
                    .iter()
                    .zip(&fresh_claims)
                    .all(|(cached, claim)| cached == &claim.c)
        });
        let fresh_commit_total = cached_fresh_masks
            .as_ref()
            .map_or(Duration::ZERO, |cached| cached.commit_total);
        let fresh_commit_gpu = cached_fresh_masks
            .as_ref()
            .map_or(Duration::ZERO, |cached| cached.commit_gpu);
        let fresh_commit_count = cached_fresh_masks
            .as_ref()
            .map_or(0, |cached| cached.host_commitments.len());
        let fresh_lane_commit_gpu = cached_fresh_masks
            .as_ref()
            .map_or(Duration::ZERO, |cached| cached.lane_commit_gpu);
        let fresh_lane_commit_count = cached_fresh_masks
            .as_ref()
            .map_or(0, |cached| cached.lane_commit_count);
        let fresh_lanes_from_resident_masks = fresh_lane_commit_count != 0;
        let fresh_masks_reused = cached_fresh_masks.is_some();

        let witness_refs = fresh_witnesses
            .iter()
            .map(|witness| &witness.Z)
            .chain(request.running.witnesses.iter())
            .collect::<Vec<_>>();
        let running_carrier = metal_running_carrier(request.running_carrier);
        let running_parent_digest = running_carrier.map(MetalRunningCarrier::parent_digest);
        let running_accumulator_handle = running_parent_digest.map(|parent_digest| {
            AccumulatorHandle::from_parent_digest(request.running.claims.len(), Some(parent_digest)).digest_fields()
        });
        let resident_running_id = running_carrier
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

        let forms_on_metal = self
            .ensure_dec_form_plan(request.s, request.cache)
            .map_err(|_| backend_unavailable("prepare Metal Ajtai ring forms"))?;
        let pi_ccs_activity_before = self.session.activity();
        self.session.reset_sumcheck_durations();
        let mut nc_backend = MetalNcBackend::new(
            &self.session,
            &witness_refs,
            request.s.m,
            fresh_witnesses.len(),
            cached_fresh_masks.as_ref().map(|cached| &cached.masks),
            resident_running_id,
        )
        .map_err(|_| backend_unavailable("prepare shared Metal witness masks"))?;
        let shared_witness_masks = nc_backend.shared_masks().cloned();
        let mut fe_backend = MetalFeBackend::new(&self.session)
            .y_eval_only(forms_on_metal.then(|| {
                self.dec_form_plan
                    .as_ref()
                    .expect("Ajtai form plan installed above")
            }))
            .witness_masks(shared_witness_masks.as_ref());
        if let Some(plan) = self.fe_oracle_plan.as_ref() {
            fe_backend = fe_backend.oracle_plan(plan, resident_running_id);
        }
        let mut outputs_digest_backend = MetalPiCcsOutputsDigest { session: &self.session };
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
            Some(&mut fe_backend),
            Some(&mut nc_backend),
            BackendTranscriptMode::DeviceSnapshot,
            running_parent_digest,
            running_accumulator_handle,
            Some(&mut outputs_digest_backend),
        )?;
        let pi_ccs_elapsed = ccs_started.elapsed();
        let (fe_sumcheck, nc_sumcheck) = self.session.sumcheck_durations();
        let fe_profile = fe_backend.profile();
        let ajtai_forms = fe_backend.take_ajtai_forms();
        drop(fe_backend);
        let nc_profile = nc_backend.profile();
        if fe_profile.metal_failed || nc_profile.metal_failed {
            return Err(backend_unavailable("Metal Pi_CCS accelerator failed"));
        }
        let pi_ccs_activity = activity_delta(pi_ccs_activity_before, self.session.activity());

        let rlc_started = Instant::now();
        let pi_rlc_activity_before = self.session.activity();
        let rlc_witness_masks_reused = Cell::new(false);
        let rlc_result = pi_rlc::prove_refs_with_resident_witness_and_projection_digest(
            request.tr,
            request.pp,
            request.s,
            request.mix_rhos_commits,
            &pi_ccs_proof.outputs,
            &witness_refs,
            pi_ccs_proof.outputs_digest,
            |rhos, witnesses| {
                mix_witnesses_on_metal(
                    &self.session,
                    &nc_backend,
                    rhos,
                    witnesses,
                    fresh_witnesses.len(),
                    resident_running_id,
                )
                .map(|(witness, masks_reused)| {
                    rlc_witness_masks_reused.set(masks_reused);
                    witness
                })
            },
            |preimage| {
                self.session
                    .sis_accumulator_digest_resident(PI_RLC_PROJECTION_SIS_CONFIG, preimage)
                    .map_err(|_| pi_rlc::Error::BackendProjectionDigest)
            },
        );
        nc_backend.recycle();
        let (rlc_output, pi_rlc_proof) = rlc_result?;
        let mut resident_mix = rlc_output
            .witness
            .map_err(|_| backend_unavailable("Metal Pi_RLC witness mixing failed"))?;
        let parent_digest = self
            .session
            .sis_accumulator_digest_resident(
                CE_CLAIM_SIS_CONFIG,
                &digest::ce_claim_digest_preimage(&rlc_output.claim),
            )
            .map_err(|_| backend_unavailable("compute Metal Pi_RLC parent digest"))?;
        let pi_rlc_elapsed = rlc_started.elapsed();
        let pi_rlc_activity = activity_delta(pi_rlc_activity_before, self.session.activity());

        let dec_started = Instant::now();
        let pi_dec_activity_before = self.session.activity();
        self.ensure_ajtai_plan(request.log, resident_mix.cols())?;
        let commitment_plan = &self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above")
            .plan;
        let mut dec_material = split_dec_on_metal(
            &self.session,
            request.pp,
            request.s,
            request.cache,
            &rlc_output.claim,
            resident_mix.cols(),
            &mut resident_mix,
            request.pp.k_rho() as usize,
            true,
            commitment_plan,
            request.lanes.map(|_| {
                self.fresh_lane_commitment_plan
                    .as_ref()
                    .expect("Nebula lane commitment plans installed during fresh construction")
            }),
            forms_on_metal.then(|| {
                self.dec_form_plan
                    .as_ref()
                    .expect("Pi_DEC form plan installed above")
            }),
            ajtai_forms.as_ref(),
        )
        .map_err(|_| backend_unavailable("Metal Pi_DEC witness split failed"))?;
        if let Some(forms) = ajtai_forms {
            self.session.recycle_ajtai_ring_forms(forms);
        }
        let resident_output_id = dec_material.resident_output_id;
        let witness_snapshot = dec_material.witness_snapshot.take();
        let dec_form_build = dec_material.form_build;
        let dec_projection = dec_material.projection;
        let dec_lane_commit_gpu = dec_material.lane_commit_gpu;
        let dec_y_zcol_gpu = dec_material.y_zcol_gpu;
        let dec_host_materialization = dec_material.host_materialization;
        let dec_forms_on_metal = dec_material.forms_on_metal;
        let (dec_output, pi_dec_proof) = if let Some(claims) = dec_material.resident_claims.take() {
            pi_dec::prove_from_accelerator_claims(
                request.pp,
                request.s,
                request.combine_b_pows,
                &rlc_output.claim,
                claims,
                dec_material.witnesses,
            )?
        } else {
            pi_dec::prove_from_split_material(
                request.pp,
                request.s,
                request.cache,
                request.lanes,
                dec_material.child_adv,
                request.combine_b_pows,
                &rlc_output.claim,
                dec_material.witnesses,
                dec_material.digit_nonzero,
                dec_material.commitments,
                dec_material.y_ring,
            )?
        };
        let pi_dec_elapsed = dec_started.elapsed();
        let pi_dec_activity = activity_delta(pi_dec_activity_before, self.session.activity());

        let next_running = RunningInstance {
            claims: dec_output.claims,
            witnesses: dec_output.witnesses,
            parent_authority: Some(rlc_output.claim),
        };
        let post_fold_summary = self.post_fold_summary(&next_running, parent_digest)?;
        let activity = activity_delta(activity_before, self.session.activity());
        let resident_running_output = resident_output_id.is_some();
        self.last_profile = Some(MetalNifsProfile {
            total: total_started.elapsed(),
            fresh_commit_total,
            fresh_commit_gpu,
            fresh_commit_count,
            fresh_masks_reused,
            fresh_lane_commit_gpu,
            fresh_lane_commit_count,
            fresh_lanes_from_resident_masks,
            pi_ccs: pi_ccs_elapsed,
            fe_sumcheck,
            nc_sumcheck,
            pi_ccs_activity,
            ajtai_y_eval: fe_profile.ajtai_y_eval,
            ajtai_seeded_build: fe_profile.ajtai_seeded_build,
            ajtai_device_eval: fe_profile.ajtai_device_eval,
            ajtai_tensor_gpu: fe_profile.ajtai_tensor_gpu,
            ajtai_form_gpu: fe_profile.ajtai_form_gpu,
            ajtai_tail_gpu: fe_profile.ajtai_tail_gpu,
            ajtai_seeded_patch_entries: fe_profile.ajtai_seeded_patch_entries,
            ajtai_seeded_patch_bytes: fe_profile.ajtai_seeded_patch_bytes,
            ajtai_form_blocks: fe_profile.ajtai_form_blocks,
            ajtai_form_bytes: fe_profile.ajtai_form_bytes,
            ajtai_explicit_coefficients: fe_profile.ajtai_explicit_coefficients,
            ajtai_signed_unit_coefficients: fe_profile.ajtai_signed_unit_coefficients,
            ajtai_explicit_form_list_histogram: fe_profile.ajtai_explicit_form_list_histogram,
            ajtai_max_explicit_form_list_entries: fe_profile.ajtai_max_explicit_form_list_entries,
            ajtai_parallel_form_lists: fe_profile.ajtai_parallel_form_lists,
            ajtai_parallel_form_entries: fe_profile.ajtai_parallel_form_entries,
            pi_rlc: pi_rlc_elapsed,
            pi_rlc_activity,
            pi_dec: pi_dec_elapsed,
            pi_dec_activity,
            dec_form_build,
            dec_projection,
            dec_lane_commit_gpu,
            dec_y_zcol_gpu,
            dec_host_materialization,
            fe_rounds: fe_profile.fe_rounds,
            fe_mcs_tables: fe_profile.fe_mcs_tables,
            fe_mcs_table_bytes: fe_profile.fe_mcs_table_bytes,
            fe_seeded_build: fe_profile.fe_seeded_build,
            fe_seeded_patch_entries: fe_profile.fe_seeded_patch_entries,
            fe_seeded_patch_bytes: fe_profile.fe_seeded_patch_bytes,
            fe_explicit_coefficients: fe_profile.fe_explicit_coefficients,
            fe_explicit_row_list_histogram: fe_profile.fe_explicit_row_list_histogram,
            fe_max_explicit_row_entries: fe_profile.fe_max_explicit_row_entries,
            fe_carried_eval_on_metal: fe_profile.fe_carried_eval_on_metal,
            fe_on_metal: fe_profile.fe_rounds > 0,
            ajtai_y_eval_on_metal: fe_profile.ajtai_y_eval_on_metal,
            nc_rounds: nc_profile.nc_rounds,
            nc_input_witnesses: nc_profile.nc_input_witnesses,
            nc_active_witnesses: nc_profile.nc_active_witnesses,
            nc_on_metal: nc_profile.nc_rounds > 0,
            nc_mask_native_on_metal: nc_profile.nc_mask_native_on_metal,
            witness_masks_shared: shared_witness_masks.is_some()
                && fe_profile.ajtai_y_eval_on_metal
                && nc_profile.nc_mask_native_on_metal
                && rlc_witness_masks_reused.get(),
            folded_tables: fe_profile.folded_tables + nc_profile.folded_tables,
            rlc_witness_on_metal: true,
            rlc_witness_resident_only: true,
            rlc_witness_masks_reused: rlc_witness_masks_reused.get(),
            rlc_rho_small_coefficients: true,
            dec_split_on_metal: true,
            dec_recomposition_on_metal: true,
            dec_forms_on_metal,
            dec_y_on_metal: true,
            dec_commit_on_metal: true,
            resident_running_input: resident_running_id.is_some(),
            resident_running_output,
            proof_deferred: true,
            running_deferred: resident_running_output,
            recursive_compile_reverify_required: false,
            activity,
        });
        let resident_output_id = Some(
            resident_output_id.ok_or_else(|| backend_unavailable("Metal Pi_DEC resident output was not retained"))?,
        );
        let output = Arc::new(MetalFoldOutput::new(
            next_running,
            resident_output_id,
            witness_snapshot,
            parent_digest,
        ));
        let proof = metal_proof_carrier(pi_ccs_proof, pi_rlc_proof, pi_dec_proof, Arc::clone(&output))?;
        Ok(NifsProverOutput::deferred(
            NifsRunningCarrier::deferred(Arc::new(MetalRunningCarrier::new(output))),
            proof,
        )
        .with_post_fold_summary(post_fold_summary))
    }

    fn requires_recursive_compile_reverify(&self) -> bool {
        false
    }

    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        self.cached_fresh_masks = None;
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
        if request.assignments.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if let Some(scheme) = request.lane_scheme {
            self.ensure_lane_ajtai_plan(scheme, cols)?;
        }

        let commit_started = Instant::now();
        let mut mask_words = Vec::with_capacity(request.assignments.len() * cols * 2);
        let mut witnesses = Vec::with_capacity(request.assignments.len());
        for assignment in request.assignments {
            let mut positive = vec![0u64; cols];
            let mut negative = vec![0u64; cols];
            for (index, &value) in assignment.iter().enumerate() {
                let digit = neo_math::balanced::to_balanced_i128(value) as i8;
                if digit == 1 {
                    positive[index / D] |= 1u64 << (index % D);
                } else if digit == -1 {
                    negative[index / D] |= 1u64 << (index % D);
                }
            }
            for (&positive, &negative) in positive.iter().zip(&negative) {
                mask_words.extend_from_slice(&[positive, negative]);
            }
            witnesses.push(
                Mat::compact_signed_unit_from_column_masks(D, cols, &positive, &negative)
                    .map_err(|_| backend_unavailable("pack the fresh Metal witness"))?,
            );
        }
        let masks = self
            .session
            .prepare_witness_masks(&mask_words, request.assignments.len(), cols, request.s.m)
            .map_err(|_| backend_unavailable("upload the fresh Metal witness batch"))?;
        let cached_plan = self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above");
        let plan = &cached_plan.plan;
        let kappa = cached_plan.kappa;
        let (commitment_words, commit_gpu) = self
            .session
            .ajtai_low_norm_many_from_masks(plan, &masks, request.assignments.len())
            .map_err(|_| backend_unavailable("compute batched Metal Ajtai commitments"))?;
        let words_per_commitment = kappa * D;
        if commitment_words.len() != request.assignments.len() * words_per_commitment {
            return Err(backend_unavailable(
                "batched Metal Ajtai commitments have the wrong shape",
            ));
        }
        let commitments = commitment_words
            .chunks_exact(words_per_commitment)
            .map(|words| commitment_from_words(words, kappa))
            .collect::<Vec<_>>();
        let (lane_commitments, lane_commit_gpu) = if request.lane_scheme.is_some() {
            let lane_plan = self
                .fresh_lane_commitment_plan
                .as_ref()
                .expect("Nebula lane commitment plans installed above");
            let (words, gpu) = self
                .session
                .ajtai_lane_commitments_from_masks(
                    &lane_plan.ops,
                    &lane_plan.mem,
                    &masks,
                    request.assignments.len(),
                    cols,
                    &lane_plan.ranges,
                )
                .map_err(|_| backend_unavailable("compute Metal Nebula lane commitments from resident masks"))?;
            let expected_words = 3 * request.assignments.len() * words_per_commitment;
            if words.len() != expected_words {
                return Err(backend_unavailable(
                    "batched Metal Nebula lane commitments have the wrong shape",
                ));
            }
            let commitment = |lane: usize, witness: usize| {
                let start = (lane * request.assignments.len() + witness) * words_per_commitment;
                commitment_from_words(&words[start..start + words_per_commitment], kappa)
            };
            let lanes = (0..request.assignments.len())
                .map(|witness| LaneCommitments {
                    ops: commitment(0, witness),
                    is: commitment(1, witness),
                    fs: commitment(2, witness),
                })
                .collect::<Vec<_>>();
            (Some(lanes), gpu)
        } else {
            (None, Duration::ZERO)
        };
        let mut instances = Vec::with_capacity(request.assignments.len());
        for (index, ((assignment, z), commitment)) in request
            .assignments
            .iter()
            .zip(witnesses)
            .zip(commitments.iter().cloned())
            .enumerate()
        {
            instances.push(CcsInstance {
                claim: CcsClaim {
                    adv: lane_commitments.as_ref().map(|lanes| lanes[index].clone()),
                    c: commitment,
                    x: assignment[..request.m_in].to_vec(),
                    m_in: request.m_in,
                },
                witness: CcsWitness { w: Vec::new(), Z: z },
            });
        }
        self.cached_fresh_masks = Some(CachedFreshMasks {
            host_commitments: commitments,
            masks,
            commit_total: commit_started.elapsed(),
            commit_gpu,
            lane_commit_gpu,
            lane_commit_count: lane_commitments.as_ref().map_or(0, Vec::len),
        });
        Ok(Some(instances))
    }
}

fn commitment_from_words(words: &[u64], kappa: usize) -> Commitment {
    Commitment {
        d: D,
        kappa,
        data: words.iter().copied().map(F::from_u64).collect(),
    }
}

fn mix_witnesses_on_metal(
    session: &MetalSession,
    nc_backend: &MetalNcBackend<'_>,
    rhos: &[Mat<F>],
    witnesses: &[&Mat<F>],
    fresh_count: usize,
    resident_id: Option<u64>,
) -> Result<(MetalResidentWitness, bool), MetalError> {
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
    if let Some(witness) = nc_backend.enqueue_rlc_witness_mix_from_resident_masks(
        &rho_coefficients,
        fresh_count,
        witnesses.len(),
        cols,
        resident_id,
    )? {
        return Ok((witness, true));
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
        session
            .enqueue_rlc_witness_mix_with_resident_id(
                &rho_coefficients,
                &witness_words,
                fresh_count,
                witnesses.len(),
                cols,
                resident_id,
            )
            .map(|witness| (witness, false))
    } else {
        session
            .enqueue_rlc_witness_mix(&rho_coefficients, &witness_words, witnesses.len(), cols)
            .map(|witness| (witness, false))
    }
}

fn split_dec_on_metal(
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
        session.recycle_dec_children(material.resident_children);
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
