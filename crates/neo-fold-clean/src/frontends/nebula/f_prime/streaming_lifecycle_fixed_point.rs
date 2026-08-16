//! Verifier-shape closure for the monolithic streaming lifecycle source.
//!
//! Owns the deterministic shape iteration from one folded verifier relation
//! to the selectively lowered full base/recursive source. This is a reference
//! upper bound. It is not the final 400-phase relation. It does not own source
//! artifact identities, satisfying assignments, preprocessing, or terminal
//! acceptance.

use neo_math::D;

use super::streaming_lifecycle_relation::synthesize_streaming_lifecycle_source_arm_shapes;
use super::NebulaFPrimeRelationError;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, selective_polynomial,
    SelectiveLowNormWidthAudit,
};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_circuit::PiCcsVerifierRelation;

pub const STREAMING_LIFECYCLE_FULL_SOURCE_JOINT_DOMAIN_BITS: u32 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
    rows: usize,
    columns: usize,
    public_columns: usize,
    matrix_count: usize,
    max_degree: u32,
}

impl NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn public_columns(self) -> usize {
        self.public_columns
    }

    pub const fn matrix_count(self) -> usize {
        self.matrix_count
    }

    pub const fn max_degree(self) -> u32 {
        self.max_degree
    }

    const fn signature(self) -> (usize, usize, usize, u32) {
        (self.rows, self.columns, self.matrix_count, self.max_degree)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFullSourceShape {
    rows: usize,
    columns: usize,
    public_columns: usize,
}

impl NebulaFPrimeStreamingLifecycleFullSourceShape {
    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn public_columns(self) -> usize {
        self.public_columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound {
    input: NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape,
    effective_lambda: u32,
    base_source: NebulaFPrimeStreamingLifecycleFullSourceShape,
    recursive_source: NebulaFPrimeStreamingLifecycleFullSourceShape,
    output: NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape,
}

impl NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound {
    pub const fn input(self) -> NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
        self.input
    }

    pub const fn effective_lambda(self) -> u32 {
        self.effective_lambda
    }

    pub const fn base_source(self) -> NebulaFPrimeStreamingLifecycleFullSourceShape {
        self.base_source
    }

    pub const fn recursive_source(self) -> NebulaFPrimeStreamingLifecycleFullSourceShape {
        self.recursive_source
    }

    pub const fn output(self) -> NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
        self.output
    }

    pub fn is_closed(self) -> bool {
        self.input.signature() == self.output.signature()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFullSourceFixedPointAudit {
    rounds: Vec<NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound>,
    effective_lambda: u32,
    joint_domain_bits: u32,
    width: SelectiveLowNormWidthAudit,
}

impl NebulaFPrimeStreamingLifecycleFullSourceFixedPointAudit {
    pub fn rounds(&self) -> &[NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound] {
        &self.rounds
    }

    pub const fn effective_lambda(&self) -> u32 {
        self.effective_lambda
    }

    pub const fn joint_domain_bits(&self) -> u32 {
        self.joint_domain_bits
    }

    pub const fn fits_joint_domain(&self) -> bool {
        self.joint_domain_bits <= STREAMING_LIFECYCLE_FULL_SOURCE_JOINT_DOMAIN_BITS
    }

    pub fn width(&self) -> &SelectiveLowNormWidthAudit {
        &self.width
    }

    pub fn fixed_point(&self) -> &NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound {
        self.rounds
            .last()
            .expect("a successful fixed-point audit has a terminal round")
    }
}

/// Discover the exact Appendix B.2 monolithic full-source verifier-shape
/// fixed point. The result is a reference upper bound, not the final phased
/// relation. The final round uses the strongest effective security level
/// accepted by that exact shape.
pub fn production_streaming_lifecycle_full_source_fixed_point_audit(
    plan: &NebulaPlan,
) -> Result<NebulaFPrimeStreamingLifecycleFullSourceFixedPointAudit, NebulaFPrimeRelationError> {
    let reference_params = Params::production();
    let verifier_rows = usize::try_from(reference_params.m())
        .map_err(|_| NebulaFPrimeRelationError::Geometry("streaming fixed-point row domain exceeds usize".into()))?;
    let verifier_columns = verifier_rows / D * D;
    let mut verifier = PiCcsVerifierRelation::from_parts(verifier_rows, verifier_columns, selective_polynomial());
    let mut seen = Vec::new();
    let mut rounds = Vec::new();

    loop {
        let input = verifier_shape(&verifier, 648);
        seen.push(input.signature());
        let round_params = Params::for_ccs_shape(input.rows, input.columns, input.matrix_count, input.max_degree)
            .map_err(|error| {
                NebulaFPrimeRelationError::Geometry(format!(
                    "streaming fixed-point input Appendix B.2 profile: {error}"
                ))
            })?;
        if !round_params.has_production_core() {
            return Err(NebulaFPrimeRelationError::Geometry(
                "streaming fixed-point input parameters lost the Appendix B.2 core".into(),
            ));
        }
        let arms = synthesize_streaming_lifecycle_source_arm_shapes(&round_params, verifier.clone(), plan)?;
        let base_source = source_shape(&arms[0]);
        let recursive_source = source_shape(&arms[1]);
        let prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            Vec::from(arms),
            0,
            0,
            D,
            0,
            round_params.b(),
        )?;
        let output_summary = prepared.shape_summary();
        let output = NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
            rows: output_summary.rows,
            columns: output_summary.columns,
            public_columns: output_summary.public_input_len,
            matrix_count: output_summary.polynomial.arity(),
            max_degree: output_summary.polynomial.max_degree(),
        };
        rounds.push(NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound {
            input,
            effective_lambda: round_params.lambda(),
            base_source,
            recursive_source,
            output,
        });

        if input.signature() == output.signature() {
            let effective = Params::for_ccs_shape(output.rows, output.columns, output.matrix_count, output.max_degree)
                .map_err(|error| {
                    NebulaFPrimeRelationError::Geometry(format!("streaming fixed-point Appendix B.2 profile: {error}"))
                })?;
            if !effective.has_production_core() {
                return Err(NebulaFPrimeRelationError::Geometry(
                    "streaming fixed-point parameters lost the Appendix B.2 core".into(),
                ));
            }

            let exact_arms = synthesize_streaming_lifecycle_source_arm_shapes(&effective, verifier, plan)?;
            if source_shape(&exact_arms[0]) != base_source || source_shape(&exact_arms[1]) != recursive_source {
                return Err(NebulaFPrimeRelationError::Geometry(
                    "shape-specific security parameters changed streaming source geometry".into(),
                ));
            }
            let exact_prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
                Vec::from(exact_arms),
                0,
                0,
                D,
                0,
                effective.b(),
            )?;
            let exact = exact_prepared.shape_summary();
            let exact_output = NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
                rows: exact.rows,
                columns: exact.columns,
                public_columns: exact.public_input_len,
                matrix_count: exact.polynomial.arity(),
                max_degree: exact.polynomial.max_degree(),
            };
            if exact_output != output {
                return Err(NebulaFPrimeRelationError::Geometry(
                    "shape-specific security parameters changed the streaming fixed point".into(),
                ));
            }
            let (_, exact_compiler) = exact_prepared.into_source_audit_parts()?;

            return Ok(NebulaFPrimeStreamingLifecycleFullSourceFixedPointAudit {
                rounds,
                effective_lambda: effective.lambda(),
                joint_domain_bits: joint_domain_bits(output.rows, output.columns)?,
                width: exact_compiler.width().clone(),
            });
        }

        if seen.contains(&output.signature()) {
            return Err(NebulaFPrimeRelationError::NoFixedPoint {
                rounds: rounds.len(),
                input_rows: input.rows,
                input_cols: input.columns,
                output_rows: output.rows,
                output_cols: output.columns,
            });
        }
        verifier = PiCcsVerifierRelation::from_parts(output.rows, output.columns, output_summary.polynomial);
    }
}

fn verifier_shape(
    relation: &PiCcsVerifierRelation,
    public_columns: usize,
) -> NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
    NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape {
        rows: relation.n(),
        columns: relation.m(),
        public_columns,
        matrix_count: relation.t(),
        max_degree: relation.max_degree(),
    }
}

fn source_shape(source: &crate::frontends::r1cs_f_prime::SparseR1cs) -> NebulaFPrimeStreamingLifecycleFullSourceShape {
    NebulaFPrimeStreamingLifecycleFullSourceShape {
        rows: source.n,
        columns: source.m,
        public_columns: source.m_in,
    }
}

fn joint_domain_bits(rows: usize, columns: usize) -> Result<u32, NebulaFPrimeRelationError> {
    rows.max(columns)
        .max(2)
        .checked_next_power_of_two()
        .map(usize::trailing_zeros)
        .ok_or_else(|| NebulaFPrimeRelationError::Geometry("streaming fixed-point joint domain overflow".into()))
}
