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
use crate::frontends::nebula::circuit::{SMemCircuit, SMemR1csError};
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_width_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, FieldR1csLoweringError, LowNormR1csError,
    MultiBranchLowNormR1cs, SelectiveLowNormWidthAudit, SparseR1cs,
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
use crate::paper::relations::{CcsInstance, LaneRanges, LaneSchemeError, RelationError, Structure};

#[derive(Debug, Error)]
pub enum NebulaFPrimeError {
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
        "fixed Nebula F': direct low-norm relation needs at least {minimum_bits} committed bits, exceeding the {budget_bits}-bit Road A budget"
    )]
    CompileBudgetExceeded {
        minimum_bits: usize,
        budget_bits: usize,
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
/// committed-coordinate ceiling before allocating the square output matrices.
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
    arm_shapes: [NebulaFPrimeFieldArmShape; 3],
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
        const MAX_ROUNDS: usize = 8;

        let mut verifier_structure = plan.circuit().structure().clone();
        let mut last_output = (verifier_structure.n, verifier_structure.m);
        for round in 0..MAX_ROUNDS {
            let input_signature = relation_signature(&verifier_structure);
            let arms = shape::synthesize_arm_shapes(params, &verifier_structure, plan)?;
            drop(verifier_structure);
            let next = Self::compile_owned([arms.base, arms.bootstrap_recursive, arms.recursive], plan)?;
            last_output = (next.structure().n, next.structure().m);
            if round > 0 && input_signature == relation_signature(next.structure()) {
                return Ok(next);
            }
            verifier_structure = next.relation.into_structure();
        }
        Err(NebulaFPrimeRelationError::NoFixedPoint {
            rounds: MAX_ROUNDS,
            input_rows: verifier_structure.n,
            input_cols: verifier_structure.m,
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
        let arms = shape::synthesize_arm_shapes(params, verifier_structure, plan)?;
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
        Self::compile_owned(arms, plan)
    }

    fn compile_owned(arms: [SparseR1cs; 3], plan: &NebulaPlan) -> Result<Self, NebulaFPrimeRelationError> {
        let circuit = plan.circuit();
        let shared_private_fields = circuit.cols() - circuit.m_in();
        let arm_shapes: [NebulaFPrimeFieldArmShape; 3] = std::array::from_fn(|index| NebulaFPrimeFieldArmShape {
            rows: arms[index].n,
            columns: arms[index].m,
            public_columns: arms[index].m_in,
            poseidon2_permutations: arms[index].poseidon2_permutations(),
        });
        let width_audit = audit_multi_branch_selective_low_norm_width_with_alignment(
            &arms,
            shared_private_fields,
            D,
            circuit.m_in() % D,
        )?;
        if width_audit.total_coordinates > ROAD_A_COMMITTED_BIT_BUDGET {
            return Err(NebulaFPrimeRelationError::CompileBudgetExceeded {
                minimum_bits: width_audit.total_coordinates,
                budget_bits: ROAD_A_COMMITTED_BIT_BUDGET,
            });
        }
        let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(
            &arms,
            shared_private_fields,
            D,
            circuit.m_in() % D,
        )?;
        if relation.structure().m > ROAD_A_COMMITTED_BIT_BUDGET {
            return Err(NebulaFPrimeRelationError::CompileBudgetExceeded {
                minimum_bits: relation.structure().m,
                budget_bits: ROAD_A_COMMITTED_BIT_BUDGET,
            });
        }
        let remapped_ranges = remap_lane_ranges(&relation, &arms, circuit)?;
        let mut config = plan.config();
        config.scheme = config.scheme.remap_ranges(remapped_ranges)?;
        Ok(Self {
            relation,
            config,
            arm_shapes,
            preprocessing_digest: None,
        })
    }

    pub fn structure(&self) -> &Structure {
        self.relation.structure()
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len()
    }

    pub fn nebula_config(&self) -> &NebulaConfig {
        &self.config
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
        let assignment = self.relation.encode(branch.index(), field_assignment)?;
        if let Some(row) = self.relation.first_unsatisfied_row(&assignment) {
            return Err(NebulaFPrimeRelationError::Unsatisfied { row });
        }
        Ok(assignment)
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
        let structure_matches = self.preprocessing_digest.map_or_else(
            || digest::structure_digest(self.structure()) == *prep.structure_digest(),
            |bound| bound == *prep.structure_digest(),
        );
        if !structure_matches || prep.public_input_len != Some(self.public_input_len()) {
            return Err(NebulaFPrimeRelationError::PreprocessingMismatch);
        }
        let assignment = self.encode(branch, field_assignment)?;
        let mut instance = CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            &assignment,
            self.public_input_len(),
        )?;
        let adv = self.config.scheme.commit(&instance.witness.Z)?;
        if adv.ops.kappa != instance.claim.c.kappa {
            return Err(NebulaFPrimeRelationError::Geometry(
                "lane and full-witness commitments use different kappa".into(),
            ));
        }
        instance.claim.adv = Some(adv);
        Ok(instance)
    }
}

fn relation_signature(structure: &Structure) -> (usize, usize, usize, u32) {
    (structure.n, structure.m, structure.t(), structure.max_degree())
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
    /// `[step_x_bits || open || bits(D_pre)]` produced for the next step.
    pub current_public_suffix: Vec<Var>,
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
    let (s_mem, current_public_suffix) =
        enforce_current_application(builder, s_mem, s_mem_assignment, current_d_pre, cfg)?;
    let f_prime = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem,
        current_public_suffix,
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
    let (s_mem, current_public_suffix) =
        enforce_current_application(builder, s_mem, s_mem_assignment, current_d_pre, cfg)?;
    let f_prime = enforce_f_prime_recursive_step_circuit(builder, pp, cfg, inputs)?;
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem,
        current_public_suffix,
    })
}

fn enforce_current_application(
    builder: &mut R1csBuilder,
    circuit: &SMemCircuit,
    assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
) -> Result<(Vec<Var>, Vec<Var>), NebulaFPrimeError> {
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

    let s_mem = circuit.enforce_in_r1cs(builder, assignment)?;
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
    debug_assert_eq!(suffix.len(), expected_suffix);
    Ok((s_mem, suffix))
}
