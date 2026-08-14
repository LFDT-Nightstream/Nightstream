//! NIFS prover construction and static-plan ownership.
//!
//! On supported Apple builds, the selected one-joint proof uses the Metal
//! oracle and reusable device plans. Unsupported builds fail explicitly.

use std::sync::{Arc, Weak};

use neo_ajtai::{Commitment, PP};
use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::paper::nifs::{
    AcceleratorCrosscheckNifsProver, Error, NifsFreshInstancesRequest, NifsFreshSignedUnitAssignment,
    NifsFreshSignedUnitInstancesRequest, NifsProof, NifsProverAdapter, NifsProverRequest, OptimizedNifsProverAdapter,
};
use neo_fold_clean::paper::relations::{CcsClaim, LaneRanges, LaneScheme, Structure};
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use neo_fold_clean::FinalWitnessOpeningBackend;
use neo_fold_clean::RunningInstance;
use neo_fold_clean::{CcsInstance, CcsWitness};
use neo_math::ring::Rq;
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use neo_math::K;
use neo_math::{D, F};
use neo_reductions::optimized_engine::OptimizedStructureCache;
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use neo_reductions::optimized_engine::{PaperJointOracleBackend, PaperJointOracleInput, PaperJointRoundOracle};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use crate::session::{MetalJointMatrixPlan, MetalPaperJointOracle};
use crate::{MetalAjtaiLowNormPlan, MetalError, MetalSession};

/// Stateful Metal implementation of the canonical NIFS prover adapter.
///
/// One instance retains structure-static plans and the most recent running
/// witness generation so repeated folds avoid rebuilding or re-uploading them.
pub struct MetalNifsProver {
    session: MetalSession,
    fresh_commitment_plan: Option<FreshCommitmentPlan>,
    fresh_lane_commitment_plan: Option<FreshLaneCommitmentPlan>,
    #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
    joint_matrix_plan: Option<MetalJointMatrixPlan>,
}

struct FreshCommitmentPlan {
    d: usize,
    cols: usize,
    kappa: usize,
    identity: FreshCommitmentPlanIdentity,
    plan: MetalAjtaiLowNormPlan,
}

enum FreshCommitmentPlanIdentity {
    Seeded([u8; 32]),
    Materialized(Weak<PP<Rq>>),
}

impl FreshCommitmentPlanIdentity {
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Seeded(lhs), Self::Seeded(rhs)) => lhs == rhs,
            (Self::Materialized(lhs), Self::Materialized(rhs)) => Weak::ptr_eq(lhs, rhs),
            _ => false,
        }
    }
}

struct FreshLaneCommitmentPlan {
    kappa: usize,
    ops_seed: [u8; 32],
    mem_seed: [u8; 32],
    ranges: LaneRanges,
    ops: MetalAjtaiLowNormPlan,
    mem: MetalAjtaiLowNormPlan,
}

impl MetalNifsProver {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
            fresh_commitment_plan: None,
            fresh_lane_commitment_plan: None,
            #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
            joint_matrix_plan: None,
        })
    }

    pub fn session(&self) -> &MetalSession {
        &self.session
    }

    /// Wrap this complete Metal selection in an optimized-CPU NIFS crosscheck.
    pub fn crosschecked(self) -> AcceleratorCrosscheckNifsProver<Self> {
        AcceleratorCrosscheckNifsProver::new(self)
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
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        self.ensure_joint_matrix_plan(cache)?;
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        let _ = cache;
        Ok(())
    }

    #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
    fn ensure_joint_matrix_plan(&mut self, cache: &OptimizedStructureCache) -> Result<(), Error> {
        let superneo = cache.superneo_arc();
        if self
            .joint_matrix_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(superneo.as_ref()))
        {
            self.joint_matrix_plan = Some(
                self.session
                    .prepare_joint_matrix_plan(superneo)
                    .map_err(|error| backend_failure("prepare one-joint matrix plan", error))?,
            );
        }
        Ok(())
    }

    fn ensure_ajtai_plan(&mut self, log: &neo_ajtai::AjtaiSModule, cols: usize) -> Result<(), Error> {
        let (d, m) = log.dims();
        let kappa = log.kappa();
        if d != D || m != cols {
            return Err(backend_unavailable(
                "Ajtai parameter dimensions do not match the Metal witness",
            ));
        }
        let (identity, materialized_pp) = if let Some((seeded_kappa, seed)) = log.seeded_params() {
            if seeded_kappa != kappa {
                return Err(backend_unavailable("seeded Ajtai parameters have inconsistent kappa"));
            }
            (FreshCommitmentPlanIdentity::Seeded(seed), None)
        } else {
            let pp = log
                .verification_pp()
                .map_err(|error| backend_failure("materialize Ajtai commitment parameters", error))?;
            let identity = FreshCommitmentPlanIdentity::Materialized(Arc::downgrade(&pp));
            (identity, Some(pp))
        };
        let rebuild = self.fresh_commitment_plan.as_ref().is_none_or(|cached| {
            cached.d != d || cached.cols != m || cached.kappa != kappa || !cached.identity.matches(&identity)
        });
        if rebuild {
            let plan = match (&identity, materialized_pp) {
                (FreshCommitmentPlanIdentity::Seeded(seed), None) => {
                    self.session.prepare_ajtai_low_norm_seeded(*seed, kappa, m)
                }
                (FreshCommitmentPlanIdentity::Materialized(_), Some(pp)) => {
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
                _ => unreachable!("Ajtai plan identity and parameters are constructed together"),
            }
            .map_err(|error| backend_failure("prepare Ajtai commitment parameters", error))?;
            self.fresh_commitment_plan = Some(FreshCommitmentPlan {
                d,
                cols: m,
                kappa,
                identity,
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

    fn build_fresh_signed_unit_instances_inner(
        &mut self,
        log: &neo_ajtai::AjtaiSModule,
        s: &Structure,
        m_in: usize,
        assignments: &[NifsFreshSignedUnitAssignment],
        lane_scheme: Option<&LaneScheme>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        if assignments
            .iter()
            .any(|assignment| assignment.len() != s.m || m_in > assignment.len())
        {
            return Ok(None);
        }

        let cols = s.m.div_ceil(D);
        if log.dims() != (D, cols) {
            return Ok(None);
        }
        self.ensure_ajtai_plan(log, cols)?;
        if let Some(scheme) = lane_scheme {
            self.ensure_lane_ajtai_plan(scheme, cols)?;
        }
        if assignments.is_empty() {
            return Ok(Some(Vec::new()));
        }

        // Layout is witness-major, then ring block, with one positive and one
        // negative bitset per block. The same upload feeds the full and
        // optional Nebula lane commitments.
        let mut mask_words = Vec::with_capacity(assignments.len() * cols * 2);
        let mut witnesses = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            let positive = assignment.positive_masks();
            let negative = assignment.negative_masks();
            for (&positive, &negative) in positive.iter().zip(negative) {
                mask_words.extend_from_slice(&[positive, negative]);
            }
            witnesses.push(
                Mat::compact_signed_unit_from_column_masks(D, cols, positive, negative)
                    .map_err(|error| backend_failure("pack fresh witness", error))?,
            );
        }
        let masks = self
            .session
            .prepare_witness_masks(&mask_words, assignments.len(), cols, s.m)
            .map_err(|error| backend_failure("upload fresh witness batch", error))?;
        let cached_plan = self
            .fresh_commitment_plan
            .as_ref()
            .expect("Ajtai commitment plan installed above");
        let plan = &cached_plan.plan;
        let kappa = cached_plan.kappa;
        let (commitment_words, _commit_gpu) = self
            .session
            .ajtai_low_norm_many_from_masks(plan, &masks, assignments.len())
            .map_err(|error| backend_failure("compute batched Ajtai commitments", error))?;
        let words_per_commitment = kappa * D;
        if commitment_words.len() != assignments.len() * words_per_commitment {
            return Err(backend_unavailable(
                "batched Metal Ajtai commitments have the wrong shape",
            ));
        }
        let commitments = commitment_words
            .chunks_exact(words_per_commitment)
            .map(|words| commitment_from_words(words, kappa))
            .collect::<Vec<_>>();
        let (lane_commitments, _lane_commit_gpu) = if lane_scheme.is_some() {
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
                    assignments.len(),
                    cols,
                    &lane_plan.ranges,
                )
                .map_err(|error| backend_failure("compute Nebula lane commitments", error))?;
            let expected_words = 3 * assignments.len() * words_per_commitment;
            if words.len() != expected_words {
                return Err(backend_unavailable(
                    "batched Metal Nebula lane commitments have the wrong shape",
                ));
            }
            let commitment = |lane: usize, witness: usize| {
                let start = (lane * assignments.len() + witness) * words_per_commitment;
                commitment_from_words(&words[start..start + words_per_commitment], kappa)
            };
            let lanes = (0..assignments.len())
                .map(|witness| LaneCommitments {
                    ops: commitment(0, witness),
                    is: commitment(1, witness),
                    fs: commitment(2, witness),
                })
                .collect::<Vec<_>>();
            (Some(lanes), gpu)
        } else {
            (None, std::time::Duration::ZERO)
        };
        let mut instances = Vec::with_capacity(assignments.len());
        for (index, ((assignment, z), commitment)) in assignments
            .iter()
            .zip(witnesses)
            .zip(commitments.iter().cloned())
            .enumerate()
        {
            instances.push(CcsInstance {
                claim: CcsClaim {
                    adv: lane_commitments.as_ref().map(|lanes| lanes[index].clone()),
                    c: commitment,
                    x: assignment
                        .public_input(m_in)
                        .expect("fresh assignment shape checked above"),
                    m_in,
                },
                witness: CcsWitness { w: Vec::new(), Z: z },
            });
        }
        Ok(Some(instances))
    }
}

impl NifsProverAdapter for MetalNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), Error> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            return neo_fold_clean::paper::nifs::prove_with_joint_oracle_backend(request, self);
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = request;
            Err(backend_unavailable(
                "the Metal backend requires an Apple target and the production shader library",
            ))
        }
    }

    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        if request.pp.b() < 2 {
            return Ok(None);
        }
        let Some(assignments) = request
            .assignments
            .iter()
            .map(|assignment| NifsFreshSignedUnitAssignment::from_dense(assignment))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        self.build_fresh_signed_unit_instances_inner(
            request.log,
            request.s,
            request.m_in,
            &assignments,
            request.lane_scheme,
        )
    }

    fn build_fresh_signed_unit_instances(
        &mut self,
        request: NifsFreshSignedUnitInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        if request.pp.b() < 2 {
            return Ok(None);
        }
        self.build_fresh_signed_unit_instances_inner(
            request.log,
            request.s,
            request.m_in,
            request.assignments,
            request.lane_scheme,
        )
    }
}

impl OptimizedNifsProverAdapter for MetalNifsProver {}

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
impl PaperJointOracleBackend for MetalNifsProver {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, neo_reductions::PiCcsError> {
        self.ensure_joint_matrix_plan(input.cache)
            .map_err(|error| neo_reductions::PiCcsError::ProtocolError(error.to_string()))?;
        let plan = self
            .joint_matrix_plan
            .as_ref()
            .expect("one-joint matrix plan installed above");
        Ok(Box::new(MetalPaperJointOracle::new(&self.session, plan, input)?))
    }

    fn dec_openings(
        &mut self,
        cache: &OptimizedStructureCache,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Option<Vec<Vec<[K; D]>>>, neo_reductions::PiCcsError> {
        self.ensure_joint_matrix_plan(cache)
            .map_err(|error| neo_reductions::PiCcsError::ProtocolError(error.to_string()))?;
        let plan = self
            .joint_matrix_plan
            .as_ref()
            .expect("one-joint matrix plan installed above");
        self.session
            .eval_joint_dec_openings(plan, witnesses, point, assignment_width)
            .map_err(|error| {
                neo_reductions::PiCcsError::ProtocolError(format!("Metal one-joint PiDEC openings: {error}"))
            })
    }
}

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
impl FinalWitnessOpeningBackend for MetalNifsProver {
    fn final_witness_openings(
        &mut self,
        cache: &OptimizedStructureCache,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Option<Vec<Vec<[K; D]>>>, String> {
        self.ensure_joint_matrix_plan(cache)
            .map_err(|error| error.to_string())?;
        let plan = self
            .joint_matrix_plan
            .as_ref()
            .expect("one-joint matrix plan installed above");
        self.session
            .eval_joint_dec_openings(plan, witnesses, point, assignment_width)
            .map_err(|error| format!("Metal final witness openings: {error}"))
    }
}

fn commitment_from_words(words: &[u64], kappa: usize) -> Commitment {
    Commitment {
        d: D,
        kappa,
        data: words.iter().copied().map(F::from_u64).collect(),
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
