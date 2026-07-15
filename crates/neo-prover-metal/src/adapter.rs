//! NIFS prover orchestration over Metal-backed sumcheck state.

mod dec;
mod fold;
mod profile;

pub use profile::{
    MetalAjtaiProfile, MetalFeProfile, MetalFreshProfile, MetalNcProfile, MetalNifsProfile, MetalPiCcsProfile,
    MetalPiDecProfile, MetalPiRlcProfile, MetalResidencyProfile,
};

use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::frontends::f_prime::compiler::{nifs_ce_shape_from_claim, FPrimeFoldPostSummary};
use neo_fold_clean::paper::digest::{self, AccumulatorHandle};
use neo_fold_clean::paper::nifs::{
    Error, NifsFreshInstancesRequest, NifsPostFoldSummary, NifsProverAdapter, NifsProverOutput, NifsProverRequest,
};
use neo_fold_clean::paper::pi_ccs;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::PI_CCS_OUTPUTS_SIS_CONFIG;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim, LaneRanges, LaneScheme, Structure};
use neo_fold_clean::{CcsInstance, CcsWitness, RunningInstance};
use neo_math::{D, F};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::sumcheck::MetalNcBackend;
use crate::{
    MetalActivity, MetalAjtaiLowNormPlan, MetalDecFormPlan, MetalError, MetalFeOraclePlan, MetalResidentWitness,
    MetalSession, MetalWitnessMasks,
};
use dec::split_dec_on_metal;

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
            .map_err(|error| backend_failure("prepare static matrix state", error))?;
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
                    .map_err(|error| backend_failure("materialize Ajtai commitment parameters", error))?;
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
            .map_err(|error| backend_failure("prepare Ajtai commitment parameters", error))?;
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
                .map_err(|error| backend_failure("prepare seeded Nebula ops commitment", error))?;
            let mem = self
                .session
                .prepare_ajtai_low_norm_seeded(mem_seed, kappa, ranges.is.len())
                .map_err(|error| backend_failure("prepare seeded Nebula memory commitment", error))?;
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
        self.prove_fold(request)
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
        if let Some(scheme) = request.lane_scheme {
            self.ensure_lane_ajtai_plan(scheme, cols)?;
        }
        if request.assignments.is_empty() {
            return Ok(Some(Vec::new()));
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
                    .map_err(|error| backend_failure("pack fresh witness", error))?,
            );
        }
        let masks = self
            .session
            .prepare_witness_masks(&mask_words, request.assignments.len(), cols, request.s.m)
            .map_err(|error| backend_failure("upload fresh witness batch", error))?;
        let cached_plan = self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above");
        let plan = &cached_plan.plan;
        let kappa = cached_plan.kappa;
        let (commitment_words, commit_gpu) = self
            .session
            .ajtai_low_norm_many_from_masks(plan, &masks, request.assignments.len())
            .map_err(|error| backend_failure("compute batched Ajtai commitments", error))?;
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
                .map_err(|error| backend_failure("compute Nebula lane commitments", error))?;
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

fn backend_unavailable(reason: &'static str) -> Error {
    Error::BackendUnavailable {
        backend: "metal",
        reason,
    }
}

fn backend_failure(phase: &'static str, error: impl std::fmt::Display) -> Error {
    Error::BackendFailure {
        backend: "metal",
        phase,
        reason: error.to_string(),
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
