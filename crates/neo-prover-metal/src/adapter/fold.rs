//! Online Metal NIFS fold ordering and phase handoffs.
//!
//! This module owns orchestration only. Sumcheck, witness residency, and
//! digit-splitting details stay behind their respective backend modules.

use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use neo_ccs::Mat;
use neo_fold_clean::paper::digest::{self, AccumulatorHandle};
use neo_fold_clean::paper::nifs::{Error, NifsProverOutput, NifsProverRequest, NifsRunningCarrier};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::PI_RLC_PROJECTION_SIS_CONFIG;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_fold_clean::paper::{pi_ccs, pi_dec, pi_rlc};
use neo_fold_clean::{CcsWitness, RunningInstance};
use neo_math::{D, F};
use neo_reductions::optimized_engine::BackendTranscriptMode;

use super::{
    activity_delta, backend_failure, backend_unavailable, mix_witnesses_on_metal, split_dec_on_metal, CachedFreshMasks,
    MetalFreshProfile, MetalNifsProfile, MetalNifsProver, MetalPiCcsOutputsDigest, MetalPiCcsProfile,
    MetalPiDecProfile, MetalPiRlcProfile, MetalResidencyProfile,
};
use crate::fold_output::{metal_proof_carrier, metal_running_carrier, MetalFoldOutput, MetalRunningCarrier};
use crate::sumcheck::{MetalFeBackend, MetalNcBackend};
use crate::{MetalAjtaiRingForms, MetalResidentWitness, MetalResidentWitnessSnapshot, MetalWitnessMasks};

struct RunningInput {
    parent_accumulator_digest: Option<[F; 4]>,
    accumulator_handle: Option<[F; 4]>,
    resident_id: Option<u64>,
}

struct PiCcsPhase<'session> {
    proof: pi_ccs::Proof,
    nc_backend: MetalNcBackend<'session>,
    ajtai_forms: Option<MetalAjtaiRingForms>,
    profile: MetalPiCcsProfile,
    shared_witness_masks: bool,
}

struct PiRlcPhase {
    claim: CeClaim,
    witness: MetalResidentWitness,
    proof: pi_rlc::Proof,
    parent_accumulator_digest: [F; 4],
    profile: MetalPiRlcProfile,
    witness_masks_reused: bool,
}

struct PiDecPhase {
    running: RunningInstance,
    proof: pi_dec::Proof,
    resident_id: Option<u64>,
    witness_snapshot: Option<MetalResidentWitnessSnapshot>,
    profile: MetalPiDecProfile,
}

impl MetalNifsProver {
    /// Runs Pi_CCS, Pi_RLC, and Pi_DEC while handing resident artifacts directly
    /// from each phase to the next and retaining the resulting child generation.
    pub(super) fn prove_fold(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        if request.pp.b() != 2 {
            return Err(backend_unavailable(
                "Metal NIFS currently requires the production b=2 profile",
            ));
        }

        let mut request = request;
        let total_started = Instant::now();
        let activity_before = self.session.activity();

        let (fresh_claims, fresh_witnesses): (Vec<_>, Vec<_>) = std::mem::take(&mut request.fresh)
            .into_iter()
            .map(|instance| (instance.claim, instance.witness))
            .unzip();
        let (cached_fresh_masks, fresh_profile) = self.take_fresh_masks(&fresh_claims, request.s.m);
        let witness_refs = fresh_witnesses
            .iter()
            .map(|witness| &witness.Z)
            .chain(request.running.witnesses.iter())
            .collect::<Vec<_>>();
        let running = self.running_input(request.running_carrier, request.running)?;

        let forms_on_metal = self
            .ensure_dec_form_plan(request.s, request.cache)
            .map_err(|error| backend_failure("prepare Ajtai ring forms", error))?;
        let PiCcsPhase {
            proof: pi_ccs_proof,
            mut nc_backend,
            ajtai_forms,
            profile: mut pi_ccs_profile,
            shared_witness_masks,
        } = self.prove_pi_ccs(
            &mut request,
            &fresh_claims,
            &fresh_witnesses,
            &witness_refs,
            cached_fresh_masks.as_ref().map(|cached| &cached.masks),
            &running,
            forms_on_metal,
        )?;

        // Pi_CCS deliberately returns its NC backend: Pi_RLC can consume the
        // same signed-mask buffer before the plan is recycled.
        let pi_rlc = self.prove_pi_rlc(
            &mut request,
            &pi_ccs_proof,
            &witness_refs,
            &mut nc_backend,
            fresh_witnesses.len(),
            running.resident_id,
        )?;
        drop(nc_backend);
        pi_ccs_profile.witness_masks_shared = shared_witness_masks
            && pi_ccs_profile.ajtai.y_eval_on_metal
            && pi_ccs_profile.nc.mask_native_on_metal
            && pi_rlc.witness_masks_reused;

        let pi_dec = self.prove_pi_dec(&request, pi_rlc.claim, pi_rlc.witness, forms_on_metal, ajtai_forms)?;
        let post_fold_summary = self.post_fold_summary(&pi_dec.running)?;
        let activity = activity_delta(activity_before, self.session.activity());
        let resident_running_output = pi_dec.resident_id.is_some();
        self.last_profile = Some(MetalNifsProfile {
            total: total_started.elapsed(),
            fresh: fresh_profile,
            pi_ccs: pi_ccs_profile,
            pi_rlc: pi_rlc.profile,
            pi_dec: pi_dec.profile,
            residency: MetalResidencyProfile {
                running_input: running.resident_id.is_some(),
                running_output: resident_running_output,
                proof_deferred: true,
                running_deferred: resident_running_output,
                recursive_compile_reverify_required: false,
            },
            activity,
        });

        let resident_id = Some(
            pi_dec
                .resident_id
                .ok_or_else(|| backend_unavailable("Metal Pi_DEC resident output was not retained"))?,
        );
        let output = Arc::new(MetalFoldOutput::new(
            pi_dec.running,
            self.session.ownership_id(),
            resident_id,
            pi_dec.witness_snapshot,
            pi_rlc.parent_accumulator_digest,
        ));
        let proof = metal_proof_carrier(pi_ccs_proof, pi_rlc.proof, pi_dec.proof, Arc::clone(&output))?;
        Ok(NifsProverOutput::deferred(
            NifsRunningCarrier::deferred(Arc::new(MetalRunningCarrier::new(output))),
            proof,
        )
        .with_post_fold_summary(post_fold_summary))
    }

    /// Consumes the one-shot fresh-mask cache only when both witness shape and
    /// canonical claim commitments still identify the batch that produced it.
    fn take_fresh_masks(
        &mut self,
        fresh_claims: &[CcsClaim],
        assignment_len: usize,
    ) -> (Option<CachedFreshMasks>, MetalFreshProfile) {
        let cached = self.cached_fresh_masks.take().filter(|cached| {
            cached.host_commitments.len() == fresh_claims.len()
                && cached
                    .masks
                    .matches(fresh_claims.len(), assignment_len.div_ceil(D))
                && cached
                    .host_commitments
                    .iter()
                    .zip(fresh_claims)
                    .all(|(cached, claim)| cached == &claim.c)
        });
        let profile = cached
            .as_ref()
            .map_or_else(MetalFreshProfile::default, |cached| MetalFreshProfile {
                commit_total: cached.commit_total,
                commit_gpu: cached.commit_gpu,
                commit_count: cached.host_commitments.len(),
                masks_reused: true,
                lane_commit_gpu: cached.lane_commit_gpu,
                lane_commit_count: cached.lane_commit_count,
                lanes_from_resident_masks: cached.lane_commit_count != 0,
            });
        (cached, profile)
    }

    /// Resolves a deferred running carrier into protocol compression values and,
    /// independently, a session-local generation that is safe to reuse.
    fn running_input(
        &self,
        carrier: Option<&NifsRunningCarrier>,
        running: &RunningInstance,
    ) -> Result<RunningInput, Error> {
        let carrier = metal_running_carrier(carrier);
        if carrier.is_some_and(|carrier| carrier.session_ownership_id() != self.session.ownership_id()) {
            return Err(backend_unavailable(
                "Metal running carrier belongs to a different session",
            ));
        }
        let parent_accumulator_digest = carrier.map(MetalRunningCarrier::parent_accumulator_digest);
        let accumulator_handle = carrier.map(|_| AccumulatorHandle::from_claims(&running.claims).digest_fields());
        // A generation id is a session-local capability, not protocol data.
        // A carrier that supplies one must match the generation currently
        // retained by this session; invalid capabilities fail closed.
        let resident_id = carrier.and_then(MetalRunningCarrier::resident_id);

        if let Some(id) = resident_id {
            let (child_count, cols) = self
                .session
                .resident_running_shape(id)
                .ok_or_else(|| backend_unavailable("Metal running carrier generation is stale"))?;
            if child_count != running.witnesses.len()
                || running
                    .witnesses
                    .iter()
                    .any(|witness| witness.rows() != D || witness.cols() != cols)
            {
                return Err(backend_unavailable(
                    "Metal running carrier does not match the materialized running witness shape",
                ));
            }
        }
        Ok(RunningInput {
            parent_accumulator_digest,
            accumulator_handle,
            resident_id,
        })
    }

    /// Builds shared FE/NC device inputs, then lets the canonical prover own
    /// transcript order while Metal computes the sumcheck traces.
    #[allow(clippy::too_many_arguments)]
    fn prove_pi_ccs<'session>(
        &'session self,
        request: &mut NifsProverRequest<'_>,
        fresh_claims: &[CcsClaim],
        fresh_witnesses: &[CcsWitness],
        witness_refs: &[&Mat<F>],
        fresh_masks: Option<&MetalWitnessMasks>,
        running: &RunningInput,
        forms_on_metal: bool,
    ) -> Result<PiCcsPhase<'session>, Error> {
        let activity_before = self.session.activity();
        self.session.reset_sumcheck_durations();
        let mut nc_backend = MetalNcBackend::new(
            &self.session,
            witness_refs,
            request.s.m,
            fresh_witnesses.len(),
            fresh_masks,
            running.resident_id,
        )
        .map_err(|error| backend_failure("prepare shared witness masks", error))?;
        let shared_witness_masks = nc_backend.shared_masks().cloned();
        let mut fe_backend = MetalFeBackend::new(&self.session)
            .y_eval_only(forms_on_metal.then(|| {
                self.dec_form_plan
                    .as_ref()
                    .expect("Ajtai form plan installed above")
            }))
            .witness_masks(shared_witness_masks.as_ref());
        if let Some(plan) = self.fe_oracle_plan.as_ref() {
            fe_backend = fe_backend.oracle_plan(plan, running.resident_id);
        }

        let mut outputs_digest_backend = MetalPiCcsOutputsDigest { session: &self.session };
        let started = Instant::now();
        // The canonical prover owns absorb order. Metal receives its snapshot,
        // produces the round trace, and returns the final state for continuation.
        let proof = pi_ccs::prove_from_parts_with_backends_and_transcript_mode(
            request.tr,
            request.pp,
            request.s,
            request.cache,
            request.log,
            fresh_claims,
            fresh_witnesses,
            request.running,
            Some(&mut fe_backend),
            Some(&mut nc_backend),
            BackendTranscriptMode::DeviceSnapshot,
            running.parent_accumulator_digest,
            running.accumulator_handle,
            Some(&mut outputs_digest_backend),
        )?;
        let elapsed = started.elapsed();
        let (fe_sumcheck, nc_sumcheck) = self.session.sumcheck_durations();
        let fe_profile = fe_backend.profile();
        let ajtai_forms = fe_backend.take_ajtai_forms();
        drop(fe_backend);
        let nc_profile = nc_backend.profile();
        if let Some(reason) = fe_profile.failure.as_deref() {
            return Err(backend_failure("Pi_CCS FE", reason));
        }
        if let Some(reason) = nc_profile.failure.as_deref() {
            return Err(backend_failure("Pi_CCS NC", reason));
        }

        Ok(PiCcsPhase {
            proof,
            nc_backend,
            ajtai_forms,
            profile: MetalPiCcsProfile::from_sumchecks(
                elapsed,
                fe_sumcheck,
                nc_sumcheck,
                activity_delta(activity_before, self.session.activity()),
                fe_profile,
                nc_profile,
                false,
            ),
            shared_witness_masks: shared_witness_masks.is_some(),
        })
    }

    /// Consumes Pi_CCS's signed masks for witness mixing before recycling the NC
    /// plan, leaving the mixed witness resident for Pi_DEC.
    #[allow(clippy::too_many_arguments)]
    fn prove_pi_rlc(
        &self,
        request: &mut NifsProverRequest<'_>,
        pi_ccs_proof: &pi_ccs::Proof,
        witness_refs: &[&Mat<F>],
        nc_backend: &mut MetalNcBackend<'_>,
        fresh_count: usize,
        resident_id: Option<u64>,
    ) -> Result<PiRlcPhase, Error> {
        let started = Instant::now();
        let activity_before = self.session.activity();
        let witness_masks_reused = Cell::new(false);
        let result = pi_rlc::prove_refs_with_resident_witness_and_projection_digest(
            request.tr,
            request.pp,
            request.s,
            request.mix_rhos_commits,
            &pi_ccs_proof.outputs,
            witness_refs,
            pi_ccs_proof.outputs_digest,
            |rhos, witnesses| {
                mix_witnesses_on_metal(&self.session, nc_backend, rhos, witnesses, fresh_count, resident_id).map(
                    |(witness, masks_reused)| {
                        witness_masks_reused.set(masks_reused);
                        witness
                    },
                )
            },
            |preimage| {
                self.session
                    .sis_accumulator_digest_resident(PI_RLC_PROJECTION_SIS_CONFIG, preimage)
                    .map_err(|_| pi_rlc::Error::BackendProjectionDigest)
            },
        );
        nc_backend.recycle();
        let (output, proof) = result?;
        let witness = output
            .witness
            .map_err(|error| backend_failure("Pi_RLC witness mixing", error))?;
        let parent_accumulator_digest = digest::accumulator_ce_claim_digest(&output.claim);
        let masks_reused = witness_masks_reused.get();
        Ok(PiRlcPhase {
            claim: output.claim,
            witness,
            proof,
            parent_accumulator_digest,
            profile: MetalPiRlcProfile {
                elapsed: started.elapsed(),
                activity: activity_delta(activity_before, self.session.activity()),
                witness_on_metal: true,
                witness_resident_only: true,
                witness_masks_reused: masks_reused,
                rho_small_coefficients: true,
            },
            witness_masks_reused: masks_reused,
        })
    }

    /// Splits and projects the resident Pi_RLC witness, constructs ordinary
    /// protocol claims, and installs the child buffers as the next generation.
    fn prove_pi_dec(
        &mut self,
        request: &NifsProverRequest<'_>,
        parent_claim: CeClaim,
        mut resident_parent: MetalResidentWitness,
        forms_on_metal: bool,
        ajtai_forms: Option<MetalAjtaiRingForms>,
    ) -> Result<PiDecPhase, Error> {
        let started = Instant::now();
        let activity_before = self.session.activity();
        self.ensure_ajtai_plan(request.log, resident_parent.cols())?;
        if let Some(lanes) = request.lanes {
            self.ensure_lane_ajtai_plan(lanes, resident_parent.cols())?;
        }
        let commitment_plan = &self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above")
            .plan;
        let mut material = split_dec_on_metal(
            &self.session,
            request.pp,
            request.s,
            request.cache,
            &parent_claim,
            resident_parent.cols(),
            &mut resident_parent,
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
        .map_err(|error| backend_failure("Pi_DEC witness split", error))?;
        if let Some(forms) = ajtai_forms {
            self.session.recycle_ajtai_ring_forms(forms);
        }

        // Preserve the compact child snapshot for explicit materialization,
        // while the opaque generation id drives the next device-resident fold.
        let resident_id = material.resident_output_id;
        let witness_snapshot = material.witness_snapshot.take();
        let form_build = material.form_build;
        let projection = material.projection;
        let lane_commit_gpu = material.lane_commit_gpu;
        let y_zcol_gpu = material.y_zcol_gpu;
        let host_materialization = material.host_materialization;
        let forms_on_metal = material.forms_on_metal;
        let (children, proof) = if let Some(claims) = material.resident_claims.take() {
            pi_dec::prove_from_accelerator_claims(
                request.pp,
                request.s,
                request.combine_b_pows,
                &parent_claim,
                claims,
                material.witnesses,
            )?
        } else {
            pi_dec::prove_from_split_material(
                request.pp,
                request.s,
                request.cache,
                request.lanes,
                material.child_adv,
                request.combine_b_pows,
                &parent_claim,
                material.witnesses,
                material.digit_nonzero,
                material.commitments,
                material.y_ring,
            )?
        };
        let profile = MetalPiDecProfile {
            elapsed: started.elapsed(),
            activity: activity_delta(activity_before, self.session.activity()),
            form_build,
            projection,
            lane_commit_gpu,
            y_zcol_gpu,
            host_materialization,
            split_on_metal: true,
            recomposition_on_metal: true,
            forms_on_metal,
            y_on_metal: true,
            commit_on_metal: true,
        };
        Ok(PiDecPhase {
            running: RunningInstance::new(children.claims, children.witnesses, Some(parent_claim), None),
            proof,
            resident_id,
            witness_snapshot,
            profile,
        })
    }
}
