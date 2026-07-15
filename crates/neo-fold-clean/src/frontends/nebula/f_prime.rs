//! Composition of one current `S_mem` execution with authoritative F'.
//!
//! The current application suffix is produced here; the core F' relation
//! consumes the previous claim's suffix through NIFS.V. Keeping those two
//! directions in one wrapper makes Nebula's one-step delay explicit.

mod chain;
mod shape;

pub use chain::{NebulaFPrimeChainBuilder, NebulaFPrimeChainError, NebulaFPrimePreprocessing};

use neo_math::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::application::{enforce_memory_ports, ApplicationError, NebulaApplication};
use crate::frontends::nebula::circuit::{SMemCircuit, SMemR1csError};
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix,
    audit_multi_branch_selective_low_norm_width_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, FieldR1csLoweringError, LowNormR1csError,
    MultiBranchLowNormR1cs, SelectiveLowNormShape, SelectiveLowNormWidthAudit, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::NebulaConfig;
use crate::paper::digest;
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, Error as FPrimeError, FPrimeBaseInputs,
    FPrimeRecursiveInputs, FPrimeStepConfig, FPrimeStepOutput,
};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcVerifierRelation;
use crate::paper::relations::{CcsInstance, LaneRanges, LaneSchemeError, RelationError, Structure};

#[derive(Debug, Error)]
pub enum NebulaFPrimeError {
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    App(#[from] crate::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    R1csIvc(#[from] crate::frontends::r1cs_f_prime::ivc::R1csIvcError),
    #[error("composed Nebula F': FPrimeStepConfig has no Nebula configuration")]
    MissingNebulaConfig,
    #[error("composed Nebula F': S_mem public step width {actual} != configured width {expected}")]
    StepPublicWidth { actual: usize, expected: usize },
    #[error("composed Nebula F': configured suffix width {actual} != delayed Nebula width {expected}")]
    SuffixWidth { actual: usize, expected: usize },
    #[error(transparent)]
    SMem(#[from] SMemR1csError),
    #[error(transparent)]
    FPrime(#[from] FPrimeError),
}

#[derive(Debug, Error)]
pub enum NebulaFPrimeRelationError {
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    R1csIvc(#[from] crate::frontends::r1cs_f_prime::ivc::R1csIvcError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error(transparent)]
    Lanes(#[from] LaneSchemeError),
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
    #[error(transparent)]
    Composition(#[from] NebulaFPrimeError),
    #[error("fixed Nebula F': {0}")]
    Geometry(String),
    #[error("fixed Nebula F': encoded branch does not satisfy the authoritative relation at row {row}")]
    Unsatisfied { row: usize },
    #[error("fixed Nebula F': preprocessing was built for a different relation")]
    PreprocessingMismatch,
    #[error(
        "fixed Nebula F': direct low-norm relation needs at least {minimum_bits} committed bits, exceeding the {budget_bits}-bit Road A budget; shared-prefix candidates: {candidate_widths:?}"
    )]
    CompileBudgetExceeded {
        minimum_bits: usize,
        budget_bits: usize,
        candidate_widths: Vec<(usize, usize)>,
    },
    #[error(
        "fixed Nebula F': relation shape did not stabilize after {rounds} rounds \
         (last verifier relation {input_rows}x{input_cols}, next {output_rows}x{output_cols})"
    )]
    NoFixedPoint {
        rounds: usize,
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    },
}

/// Road A whole-step budget. The selective width census enforces this exact
/// committed-coordinate ceiling before allocating the rectangular relation.
pub const ROAD_A_COMMITTED_BIT_BUDGET: usize = 16_000_000;

/// Branches of the single folded relation. Bootstrap-recursive is distinct
/// because its NIFS input accumulator is empty; steady recursive carries
/// exactly `k_rho` running claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeBranch {
    Base,
    BootstrapRecursive,
    Recursive,
}

/// Field-native dimensions of one authoritative F' arm, before low-norm
/// bit lowering. `columns` includes the implicit constant-one column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeFieldArmShape {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub poseidon2_permutations: usize,
}

/// Shape-only audit of all three Road A arms against one verifier relation.
/// This deliberately stops before low-norm compilation, whose output can be
/// much larger than the field-native matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeFieldShapeAudit {
    pub verifier_rows: usize,
    pub verifier_columns: usize,
    pub base: NebulaFPrimeFieldArmShape,
    pub bootstrap_recursive: NebulaFPrimeFieldArmShape,
    pub recursive: NebulaFPrimeFieldArmShape,
}

impl NebulaFPrimeBranch {
    const fn index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::BootstrapRecursive => 1,
            Self::Recursive => 2,
        }
    }
}

/// One foldable low-norm relation for all three Nebula F' branches, plus
/// the lane scheme remapped onto their shared `S_mem` region.
pub struct NebulaFPrimeRelation {
    relation: MultiBranchLowNormR1cs,
    config: NebulaConfig,
    application: Option<NebulaApplication>,
    arm_shapes: [NebulaFPrimeFieldArmShape; 3],
    width_audit: SelectiveLowNormWidthAudit,
    preprocessing_digest: Option<[F; 4]>,
}

impl NebulaFPrimeRelation {
    /// Compile the single three-arm relation to a verifier-shape fixed point.
    ///
    /// Recursive-arm matrices are synthesized from shape-correct placeholder
    /// messages. Their witness values need not satisfy the rows: R1CS shape and
    /// coefficients must be deterministic functions of `(params, folded
    /// relation shape)`. The active R4 encoder test supplies honest assignments
    /// to all three compiled arms, including an interior segment step, and
    /// therefore fails if live synthesis drifts from this fixed relation.
    pub fn compile_fixed_point(params: &Params, plan: &NebulaPlan) -> Result<Self, NebulaFPrimeRelationError> {
        Self::compile_fixed_point_inner(params, plan, None, Some(ROAD_A_COMMITTED_BIT_BUDGET))
    }

    pub fn compile_application_fixed_point(
        params: &Params,
        plan: &NebulaPlan,
        application: NebulaApplication,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        application.validate_for(plan)?;
        Self::compile_fixed_point_inner(params, plan, Some(application), Some(ROAD_A_COMMITTED_BIT_BUDGET))
    }

    /// Compile an over-budget application relation solely for explicit
    /// production profiling. Normal constructors retain the 16M gate.
    #[cfg(feature = "perf-timers")]
    #[doc(hidden)]
    pub fn compile_application_fixed_point_unbounded_for_profile(
        params: &Params,
        plan: &NebulaPlan,
        application: NebulaApplication,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        application.validate_for(plan)?;
        Self::compile_fixed_point_inner(params, plan, Some(application), None)
    }

    fn compile_fixed_point_inner(
        params: &Params,
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
        committed_bit_budget: Option<usize>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        const MAX_ROUNDS: usize = 8;

        let mut verifier_relation = SplitNcVerifierRelation::from_structure(plan.circuit().structure());
        let mut last_output = (verifier_relation.n(), verifier_relation.m());
        for round in 0..MAX_ROUNDS {
            #[cfg(feature = "perf-timers")]
            let round_started = std::time::Instant::now();
            let input_signature = verifier_relation_signature(&verifier_relation);
            #[cfg(feature = "perf-timers")]
            let synthesis_started = std::time::Instant::now();
            let arms = shape::synthesize_arm_shapes(params, &verifier_relation, plan, application.as_ref())?;
            #[cfg(feature = "perf-timers")]
            let synthesis_elapsed = synthesis_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let arm_shapes = [
                (arms.base.n, arms.base.m),
                (arms.bootstrap_recursive.n, arms.bootstrap_recursive.m),
                (arms.recursive.n, arms.recursive.m),
            ];
            let shared_private_candidates = application.as_ref().map_or_else(
                || vec![plan.circuit().cols() - plan.circuit().m_in()],
                |_| {
                    let mut candidates = arms.shared_private_candidates.clone();
                    candidates.push(arms.shared_private_fields);
                    candidates
                },
            );
            #[cfg(feature = "perf-timers")]
            let lowering_started = std::time::Instant::now();
            let arm_relations = [arms.base, arms.bootstrap_recursive, arms.recursive];
            let (shared_private_fields, next_shape, candidate_widths) =
                select_low_norm_shape(&arm_relations, plan, shared_private_candidates, committed_bit_budget)?;
            let output_signature = shape_signature(&next_shape);
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[fprime-fixed-point] round={round} input={}x{} t={} u={} arms=base:{}x{},bootstrap:{}x{},recursive:{}x{} output={}x{} t={} u={} synth={:.3}s lower={:.3}s total={:.3}s",
                input_signature.0,
                input_signature.1,
                input_signature.2,
                input_signature.3,
                arm_shapes[0].0,
                arm_shapes[0].1,
                arm_shapes[1].0,
                arm_shapes[1].1,
                arm_shapes[2].0,
                arm_shapes[2].1,
                next_shape.rows,
                next_shape.columns,
                next_shape.polynomial.arity(),
                next_shape.polynomial.max_degree(),
                synthesis_elapsed.as_secs_f64(),
                lowering_started.elapsed().as_secs_f64(),
                round_started.elapsed().as_secs_f64(),
            );
            last_output = (next_shape.rows, next_shape.columns);
            if round > 0 && input_signature == output_signature {
                return Self::compile_owned_selected(
                    arm_relations,
                    plan,
                    application.clone(),
                    shared_private_fields,
                    next_shape,
                    candidate_widths,
                    committed_bit_budget,
                );
            }
            verifier_relation =
                SplitNcVerifierRelation::from_parts(next_shape.rows, next_shape.columns, next_shape.polynomial);
        }
        Err(NebulaFPrimeRelationError::NoFixedPoint {
            rounds: MAX_ROUNDS,
            input_rows: verifier_relation.n(),
            input_cols: verifier_relation.m(),
            output_rows: last_output.0,
            output_cols: last_output.1,
        })
    }

    /// Measure the three field-native arms without constructing their
    /// low-norm union. This is the safe entry point for Road A cost audits.
    pub fn audit_field_shapes(
        params: &Params,
        verifier_structure: &Structure,
        plan: &NebulaPlan,
    ) -> Result<NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError> {
        shape::audit_arm_shapes(params, verifier_structure, plan)
    }

    /// Attribute the exact low-norm assignment width without allocating the
    /// compiled CCS matrices.
    pub fn audit_low_norm_width(
        params: &Params,
        verifier_structure: &Structure,
        plan: &NebulaPlan,
    ) -> Result<SelectiveLowNormWidthAudit, NebulaFPrimeRelationError> {
        let verifier_relation = SplitNcVerifierRelation::from_structure(verifier_structure);
        let arms = shape::synthesize_arm_shapes(params, &verifier_relation, plan, None)?;
        let circuit = plan.circuit();
        let shared_private_fields = circuit.cols() - circuit.m_in();
        Ok(audit_multi_branch_selective_low_norm_width_with_alignment(
            &[arms.base, arms.bootstrap_recursive, arms.recursive],
            shared_private_fields,
            D,
            circuit.m_in() % D,
        )?)
    }

    /// Compile already-synthesized base and recursive arms. All arms must
    /// come from this module's composition functions, which allocate the
    /// same current `S_mem` assignment before branch-specific F' advice.
    pub fn compile(
        base: &SparseR1cs,
        bootstrap_recursive: &SparseR1cs,
        recursive: &SparseR1cs,
        plan: &NebulaPlan,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let arms = [base.clone(), bootstrap_recursive.clone(), recursive.clone()];
        let shared_private_fields = plan.circuit().cols() - plan.circuit().m_in();
        Self::compile_owned(arms, plan, None, vec![shared_private_fields])
    }

    fn compile_owned(
        arms: [SparseR1cs; 3],
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
        shared_private_candidates: Vec<usize>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let (shared_private_fields, shape, candidate_widths) = select_low_norm_shape(
            &arms,
            plan,
            shared_private_candidates,
            Some(ROAD_A_COMMITTED_BIT_BUDGET),
        )?;
        Self::compile_owned_selected(
            arms,
            plan,
            application,
            shared_private_fields,
            shape,
            candidate_widths,
            Some(ROAD_A_COMMITTED_BIT_BUDGET),
        )
    }

    fn compile_owned_selected(
        arms: [SparseR1cs; 3],
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
        shared_private_fields: usize,
        shape: SelectiveLowNormShape,
        candidate_widths: Vec<(usize, usize)>,
        committed_bit_budget: Option<usize>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let circuit = plan.circuit();
        let shared_private_bit_fields = circuit.cols() - circuit.m_in();
        let arm_shapes: [NebulaFPrimeFieldArmShape; 3] = std::array::from_fn(|index| NebulaFPrimeFieldArmShape {
            rows: arms[index].n,
            columns: arms[index].m,
            public_columns: arms[index].m_in,
            poseidon2_permutations: arms[index].poseidon2_permutations(),
        });
        let width_audit = shape.audit.clone();
        let relation = build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            &arms,
            shared_private_fields,
            shared_private_bit_fields,
            D,
            circuit.m_in() % D,
        )?;
        if relation_signature(relation.structure()) != shape_signature(&shape) {
            return Err(NebulaFPrimeRelationError::Geometry(
                "shape-only selective audit differs from emitted relation".into(),
            ));
        }
        if let Some(budget_bits) = committed_bit_budget {
            if relation.structure().m > budget_bits {
                return Err(NebulaFPrimeRelationError::CompileBudgetExceeded {
                    minimum_bits: relation.structure().m,
                    budget_bits,
                    candidate_widths,
                });
            }
        }
        let remapped_ranges = remap_lane_ranges(&relation, &arms, circuit)?;
        let mut config = plan.config();
        config.scheme = config.scheme.remap_ranges(remapped_ranges)?;
        Ok(Self {
            relation,
            config,
            application,
            arm_shapes,
            width_audit,
            preprocessing_digest: None,
        })
    }

    pub fn structure(&self) -> &Structure {
        self.relation.structure()
    }

    pub(crate) fn structure_arc(&self) -> std::sync::Arc<Structure> {
        self.relation.structure_arc()
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len()
    }

    pub fn nebula_config(&self) -> &NebulaConfig {
        &self.config
    }

    pub fn application(&self) -> Option<&NebulaApplication> {
        self.application.as_ref()
    }

    #[doc(hidden)]
    pub fn field_arm_shapes(&self) -> [NebulaFPrimeFieldArmShape; 3] {
        self.arm_shapes
    }

    #[doc(hidden)]
    pub fn low_norm_width_audit(&self) -> &SelectiveLowNormWidthAudit {
        &self.width_audit
    }

    fn arm_shape(&self, branch: NebulaFPrimeBranch) -> NebulaFPrimeFieldArmShape {
        self.arm_shapes[branch.index()]
    }

    pub(super) fn bind_preprocessing(&mut self, prep: &Preprocessing) -> Result<(), NebulaFPrimeRelationError> {
        let structure = self.structure();
        let prep_structure = prep.structure();
        if (structure.n, structure.m, structure.t(), structure.max_degree())
            != (
                prep_structure.n,
                prep_structure.m,
                prep_structure.t(),
                prep_structure.max_degree(),
            )
            || prep.public_input_len != Some(self.public_input_len())
        {
            return Err(NebulaFPrimeRelationError::PreprocessingMismatch);
        }
        self.preprocessing_digest = Some(*prep.structure_digest());
        Ok(())
    }

    pub fn encode(
        &self,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<Vec<F>, NebulaFPrimeRelationError> {
        #[cfg(feature = "perf-timers")]
        let encode_started = std::time::Instant::now();
        let assignment = self.encode_for_deferred_nifs(branch, field_assignment)?;
        #[cfg(feature = "perf-timers")]
        let encode_elapsed = encode_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let validate_started = std::time::Instant::now();
        if let Some(row) = self.relation.first_unsatisfied_row(&assignment) {
            return Err(NebulaFPrimeRelationError::Unsatisfied { row });
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[fprime-encode] branch={branch:?} lower={:.3}s validate={:.3}s committed={}",
            encode_elapsed.as_secs_f64(),
            validate_started.elapsed().as_secs_f64(),
            assignment.len(),
        );
        Ok(assignment)
    }

    /// Encode a synthesized arm while deferring the full relation scan to
    /// the NIFS proof that immediately consumes this instance. Encoding still
    /// enforces field width, aliases, selectors, and derived-value geometry;
    /// only the redundant prover-side sparse matrix evaluation is omitted.
    pub(super) fn encode_for_deferred_nifs(
        &self,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<Vec<F>, NebulaFPrimeRelationError> {
        self.relation
            .encode(branch.index(), field_assignment)
            .map_err(Into::into)
    }

    /// Encode, commit, and attach the product-commitment sidecar used by the
    /// delayed Nebula transition. The full witness commitment and the three
    /// lane commitments are disjoint maps over the same fixed assignment.
    pub fn build_instance(
        &self,
        prep: &Preprocessing,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<CcsInstance, NebulaFPrimeRelationError> {
        #[cfg(feature = "perf-timers")]
        let total_started = std::time::Instant::now();
        let structure_matches = self.preprocessing_digest.map_or_else(
            || digest::structure_digest(self.structure()) == *prep.structure_digest(),
            |bound| bound == *prep.structure_digest(),
        );
        if !structure_matches || prep.public_input_len != Some(self.public_input_len()) {
            return Err(NebulaFPrimeRelationError::PreprocessingMismatch);
        }
        #[cfg(feature = "perf-timers")]
        let encode_started = std::time::Instant::now();
        let assignment = self.encode(branch, field_assignment)?;
        #[cfg(feature = "perf-timers")]
        let encode_elapsed = encode_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let instance_started = std::time::Instant::now();
        let mut instance = CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            &assignment,
            self.public_input_len(),
        )?;
        #[cfg(feature = "perf-timers")]
        let instance_elapsed = instance_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let adv_started = std::time::Instant::now();
        self.attach_lane_commitment(&mut instance)?;
        #[cfg(feature = "perf-timers")]
        let adv_elapsed = adv_started.elapsed();
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[fprime-instance] branch={branch:?} encode={:.3}s ccs+commit={:.3}s adv={:.3}s total={:.3}s field_cols={} committed={} packed={}x{}",
            encode_elapsed.as_secs_f64(),
            instance_elapsed.as_secs_f64(),
            adv_elapsed.as_secs_f64(),
            total_started.elapsed().as_secs_f64(),
            field_assignment.len(),
            assignment.len(),
            instance.witness.Z.rows(),
            instance.witness.Z.cols(),
        );
        Ok(instance)
    }

    pub(super) fn attach_lane_commitment(&self, instance: &mut CcsInstance) -> Result<(), NebulaFPrimeRelationError> {
        let adv = self.config.scheme.commit(&instance.witness.Z)?;
        if adv.ops.kappa != instance.claim.c.kappa {
            return Err(NebulaFPrimeRelationError::Geometry(
                "lane and full-witness commitments use different kappa".into(),
            ));
        }
        instance.claim.adv = Some(adv);
        Ok(())
    }
}

fn relation_signature(structure: &Structure) -> (usize, usize, usize, u32) {
    (structure.n, structure.m, structure.t(), structure.max_degree())
}

fn verifier_relation_signature(relation: &SplitNcVerifierRelation) -> (usize, usize, usize, u32) {
    (relation.n(), relation.m(), relation.t(), relation.max_degree())
}

fn shape_signature(shape: &SelectiveLowNormShape) -> (usize, usize, usize, u32) {
    (
        shape.rows,
        shape.columns,
        shape.polynomial.arity(),
        shape.polynomial.max_degree(),
    )
}

fn select_low_norm_shape(
    arms: &[SparseR1cs; 3],
    plan: &NebulaPlan,
    mut shared_private_candidates: Vec<usize>,
    committed_bit_budget: Option<usize>,
) -> Result<(usize, SelectiveLowNormShape, Vec<(usize, usize)>), NebulaFPrimeRelationError> {
    let circuit = plan.circuit();
    let shared_private_bit_fields = circuit.cols() - circuit.m_in();
    shared_private_candidates.push(shared_private_bit_fields);
    shared_private_candidates.sort_unstable();
    shared_private_candidates.dedup();

    let mut best = None;
    let mut candidate_widths = Vec::new();
    for shared_private_fields in shared_private_candidates {
        if shared_private_fields < shared_private_bit_fields {
            continue;
        }
        let shape = audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(
            arms,
            shared_private_fields,
            shared_private_bit_fields,
            D,
            circuit.m_in() % D,
        )?;
        candidate_widths.push((shared_private_fields, shape.audit.total_coordinates));
        if best
            .as_ref()
            .is_none_or(|(_, best_shape): &(usize, SelectiveLowNormShape)| {
                shape.audit.total_coordinates < best_shape.audit.total_coordinates
            })
        {
            best = Some((shared_private_fields, shape));
        }
    }
    let (shared_private_fields, shape) =
        best.ok_or_else(|| NebulaFPrimeRelationError::Geometry("no valid shared-private prefix candidate".into()))?;
    if let Some(budget_bits) = committed_bit_budget {
        if shape.audit.total_coordinates > budget_bits || shape.columns > budget_bits {
            return Err(NebulaFPrimeRelationError::CompileBudgetExceeded {
                minimum_bits: shape.audit.total_coordinates.max(shape.columns),
                budget_bits,
                candidate_widths,
            });
        }
    }
    Ok((shared_private_fields, shape, candidate_widths))
}

fn remap_lane_ranges(
    relation: &MultiBranchLowNormR1cs,
    arms: &[SparseR1cs; 3],
    circuit: &SMemCircuit,
) -> Result<LaneRanges, NebulaFPrimeRelationError> {
    let source = circuit.lane_ranges();
    Ok(LaneRanges {
        ops: remap_lane_range(relation, arms, circuit, source.ops)?,
        is: remap_lane_range(relation, arms, circuit, source.is)?,
        fs: remap_lane_range(relation, arms, circuit, source.fs)?,
    })
}

fn remap_lane_range(
    relation: &MultiBranchLowNormR1cs,
    arms: &[SparseR1cs; 3],
    circuit: &SMemCircuit,
    source_ring_columns: core::ops::Range<usize>,
) -> Result<core::ops::Range<usize>, NebulaFPrimeRelationError> {
    let source_start = source_ring_columns.start * D;
    let source_end = source_ring_columns.end * D;
    if source_start < circuit.m_in() || source_end > circuit.cols() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "S_mem lane lies outside its private assignment prefix".into(),
        ));
    }

    let mut fixed_start = None;
    let mut expected = 0usize;
    for source_col in source_start..source_end {
        let private_offset = source_col - circuit.m_in();
        let slots: Vec<(usize, usize)> = arms
            .iter()
            .enumerate()
            .map(|(arm, shape)| {
                relation
                    .field_slot(arm, shape.m_in + private_offset)
                    .ok_or_else(|| NebulaFPrimeRelationError::Geometry("missing S_mem lane slot".into()))
            })
            .collect::<Result<_, _>>()?;
        if slots.iter().any(|slot| *slot != slots[0]) || slots[0].1 != 1 {
            return Err(NebulaFPrimeRelationError::Geometry(
                "S_mem lane is not a shared one-bit slot".into(),
            ));
        }
        let base_slot = slots[0];
        match fixed_start {
            None => {
                fixed_start = Some(base_slot.0);
                expected = base_slot.0;
            }
            Some(_) if base_slot.0 != expected => {
                return Err(NebulaFPrimeRelationError::Geometry(
                    "S_mem lane slots are not contiguous in the fixed assignment".into(),
                ))
            }
            Some(_) => {}
        }
        expected += 1;
    }
    let fixed_start = fixed_start.ok_or_else(|| NebulaFPrimeRelationError::Geometry("empty S_mem lane".into()))?;
    if fixed_start % D != 0 || expected % D != 0 {
        return Err(NebulaFPrimeRelationError::Geometry(
            "S_mem lane is not aligned to whole ring columns after fixed-shape lowering".into(),
        ));
    }
    Ok(fixed_start / D..expected / D)
}

/// Wires of one composed application/F' execution.
pub struct NebulaFPrimeStepOutput {
    pub f_prime: FPrimeStepOutput,
    /// Exact `S_mem` assignment wires, including its constant-one column.
    pub s_mem: Vec<Var>,
    /// Application assignment wires. Empty for the memory-only relation.
    pub application: Vec<Var>,
    /// `[step_x_bits || open || bits(D_pre)]` produced for the next step.
    pub current_public_suffix: Vec<Var>,
    /// Contiguous normalized private prefix allocated before branch-specific
    /// F' verifier advice.
    pub shared_private_fields: usize,
    /// Natural current-application allocation boundaries that may be shared
    /// across all lifecycle arms without changing column order.
    pub shared_private_candidates: Vec<usize>,
}

impl NebulaFPrimeStepOutput {
    /// Public field columns passed to `lower_field_r1cs`. The lowering adds
    /// the one implicit constant column in front of this sequence.
    pub fn public_outputs(&self) -> Vec<Var> {
        let mut out = self.f_prime.x_out_bits.clone();
        out.extend_from_slice(&self.current_public_suffix);
        out
    }
}

pub fn enforce_nebula_f_prime_base_step(
    builder: &mut R1csBuilder,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    let current = enforce_current_application(builder, s_mem, s_mem_assignment, None, current_d_pre, cfg)?;
    let shared_private_fields = builder.witness().len() - 1 - current.public_suffix.len();
    let shared_private_candidates = shared_private_candidates(builder, &current.public_suffix);
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
        shared_private_fields,
        shared_private_candidates,
    })
}

pub fn enforce_nebula_f_prime_recursive_step(
    builder: &mut R1csBuilder,
    pp: &Params,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    let current = enforce_current_application(builder, s_mem, s_mem_assignment, None, current_d_pre, cfg)?;
    let shared_private_fields = builder.witness().len() - 1 - current.public_suffix.len();
    let shared_private_candidates = shared_private_candidates(builder, &current.public_suffix);
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_recursive_step_circuit(builder, pp, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
        shared_private_fields,
        shared_private_candidates,
    })
}

pub fn enforce_nebula_application_f_prime_base_step(
    builder: &mut R1csBuilder,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    application: &NebulaApplication,
    application_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    let current = enforce_current_application(
        builder,
        s_mem,
        s_mem_assignment,
        Some((application, application_assignment)),
        current_d_pre,
        cfg,
    )?;
    let shared_private_fields = builder.witness().len() - 1 - current.public_suffix.len();
    let shared_private_candidates = shared_private_candidates(builder, &current.public_suffix);
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    if let Some(semantic) = current.semantic {
        crate::frontends::r1cs_f_prime::ivc::shape::bind_semantic_state(
            builder,
            application.recursive_plan(),
            &f_prime,
            semantic,
            true,
        );
    }
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
        shared_private_fields,
        shared_private_candidates,
    })
}

pub fn enforce_nebula_application_f_prime_recursive_step(
    builder: &mut R1csBuilder,
    pp: &Params,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    application: &NebulaApplication,
    application_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    let current = enforce_current_application(
        builder,
        s_mem,
        s_mem_assignment,
        Some((application, application_assignment)),
        current_d_pre,
        cfg,
    )?;
    let shared_private_fields = builder.witness().len() - 1 - current.public_suffix.len();
    let shared_private_candidates = shared_private_candidates(builder, &current.public_suffix);
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_recursive_step_circuit(builder, pp, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    if let Some(semantic) = current.semantic {
        crate::frontends::r1cs_f_prime::ivc::shape::bind_semantic_state(
            builder,
            application.recursive_plan(),
            &f_prime,
            semantic,
            false,
        );
    }
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
        shared_private_fields,
        shared_private_candidates,
    })
}

struct CurrentApplication {
    s_mem: Vec<Var>,
    application: Vec<Var>,
    public_suffix: Vec<Var>,
    semantic: Option<crate::frontends::r1cs_f_prime::ivc::shape::SemanticWires>,
}

fn shared_private_candidates(builder: &R1csBuilder, public_suffix: &[Var]) -> Vec<usize> {
    let mut boundaries = builder
        .column_family_ranges()
        .iter()
        .map(|range| range.column_end)
        .collect::<Vec<_>>();
    boundaries.push(builder.witness().len());
    let mut candidates = boundaries
        .into_iter()
        .filter(|&end| end > 1)
        .map(|end| {
            let public_before = public_suffix.iter().filter(|var| var.col() < end).count();
            end - 1 - public_before
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn enforce_current_application(
    builder: &mut R1csBuilder,
    circuit: &SMemCircuit,
    assignment: &[F],
    application: Option<(&NebulaApplication, &[F])>,
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
) -> Result<CurrentApplication, NebulaFPrimeError> {
    let nebula = cfg.nebula.ok_or(NebulaFPrimeError::MissingNebulaConfig)?;
    let expected_step_width = nebula.stacks.x_bits();
    let actual_step_width = circuit.m_in() - 1;
    if actual_step_width != expected_step_width {
        return Err(NebulaFPrimeError::StepPublicWidth {
            actual: actual_step_width,
            expected: expected_step_width,
        });
    }
    let expected_suffix = delayed_nebula_public_suffix_len(nebula.stacks);
    if cfg.public_input_layout.suffix_len() != expected_suffix {
        return Err(NebulaFPrimeError::SuffixWidth {
            actual: cfg.public_input_layout.suffix_len(),
            expected: expected_suffix,
        });
    }

    let s_mem_start = builder.rows();
    let s_mem_column_start = builder.witness().len();
    let s_mem = if application.is_some() {
        let vars = circuit.allocate_r1cs_assignment(builder, assignment)?;
        builder.record_row_family("nebula.application.s_mem_assignment", s_mem_start);
        builder.record_column_family("nebula.application.s_mem_assignment", s_mem_column_start);
        vars
    } else {
        let vars = circuit.enforce_in_r1cs(builder, assignment)?;
        builder.record_row_family("nebula.application.s_mem", s_mem_start);
        builder.record_column_family("nebula.application.s_mem", s_mem_column_start);
        vars
    };
    let (application_vars, semantic) = if let Some((application, application_assignment)) = application {
        let relation_start = builder.rows();
        let relation_column_start = builder.witness().len();
        let vars = application.shape().enforce_in_f_prime(
            builder,
            application_assignment,
            crate::frontends::r1cs_f_prime::ivc::shape::pin_app_constant(application.recursive_plan()),
        )?;
        builder.record_row_family("nebula.application.relation", relation_start);
        builder.record_column_family("nebula.application.relation", relation_column_start);
        let s_mem_constraints_start = builder.rows();
        let s_mem_constraints_column_start = builder.witness().len();
        circuit.enforce_allocated_r1cs(builder, &s_mem)?;
        builder.record_row_family("nebula.application.s_mem_constraints", s_mem_constraints_start);
        builder.record_column_family("nebula.application.s_mem_constraints", s_mem_constraints_column_start);
        let memory_start = builder.rows();
        let memory_column_start = builder.witness().len();
        enforce_memory_ports(
            builder,
            circuit,
            &s_mem,
            application_assignment,
            &vars,
            application.memory(),
        )?;
        builder.record_row_family("nebula.application.memory_ports", memory_start);
        builder.record_column_family("nebula.application.memory_ports", memory_column_start);
        let semantic_start = builder.rows();
        let semantic_column_start = builder.witness().len();
        let semantic = crate::frontends::r1cs_f_prime::ivc::shape::enforce_semantic_digests(
            builder,
            application.recursive_plan(),
            application_assignment,
            &vars,
        )?;
        builder.record_row_family("nebula.application.semantic", semantic_start);
        builder.record_column_family("nebula.application.semantic", semantic_column_start);
        (vars, Some(semantic))
    } else {
        (Vec::new(), None)
    };
    let suffix_column_start = builder.witness().len();
    let mut suffix = s_mem[1..circuit.m_in()].to_vec();
    let open = builder.alloc(if current_d_pre.is_some() { F::ONE } else { F::ZERO });
    enforce_bit(builder, open);
    suffix.push(open);

    let mut not_open = Lc::from_const(F::ONE);
    not_open.add_term(open, -F::ONE);
    for digest in current_d_pre.unwrap_or([[F::ZERO; 4]; 3]) {
        for lane in digest {
            let value = lane.as_canonical_u64();
            for bit in 0..64 {
                let wire = builder.alloc(F::from_u64((value >> bit) & 1));
                enforce_bit(builder, wire);
                builder.enforce(&not_open, &Lc::from_var(wire), &Lc::zero());
                suffix.push(wire);
            }
        }
    }
    builder.record_column_family("nebula.application.public_suffix", suffix_column_start);
    debug_assert_eq!(suffix.len(), expected_suffix);
    Ok(CurrentApplication {
        s_mem,
        application: application_vars,
        public_suffix: suffix,
        semantic,
    })
}
