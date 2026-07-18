//! Exact product-sum plan for the fixed production Pi_RLC projection manifest.
//!
//! Owns: the 31-role/15-pair activation gate and conversion of each validated
//! projection identity into 34 retained-output bindings plus two terminal
//! limb identities.
//!
//! Does not own: source-row replay, product-sum rank checking, emitted carry
//! gates, transcript authority, or the exact-or-bad-root security reduction.
//!
//! Emits constraints: no. [`super::product_sum`] validates and emits the plan.
//!
//! Authority boundary: source R1CS rows remain the local implementation
//! arithmetic reference. Any non-standalone production role activates an exact
//! manifest check; drift rejects instead of selecting a nearby lowering. The
//! separate Π_RLC paper refinement decides semantic sufficiency.
//!
//! | Child path | Mathematical obligation | Retained ordinary fields | Synthetic canonical fields | Product-sum rows | Lowered columns | Lowered rows | Lean theorem |
//! |---|---|---:|---:|---:|---:|---:|---|
//! | `evaluations.inputs` | 15 polynomial evaluations, two limbs each | 30 | 60 | 90 | `30*41 + 60*95 = 6,930` | `615 + 2,850 + 960 + 90 = 4,515` | `certificate_evaluation53_chunk_lengths` |
//! | `evaluations.output` | parent polynomial evaluation, two limbs | 2 | 4 | 6 | `2*41 + 4*95 = 462` | `41 + 190 + 64 + 6 = 301` | `certificate_evaluation53_chunk_lengths` |
//! | `evaluations.quotient` | quotient polynomial evaluation, two limbs | 2 | 4 | 6 | `2*41 + 4*95 = 462` | `41 + 190 + 64 + 6 = 301` | `certificate_evaluation52_chunk_lengths` |
//! | `final_limb_checks` | 15 rho/input products minus q/Phi and output | 0 | 2 | 4 | `2*95 = 190` | `95 + 32 + 4 = 131` | `certificate_terminal_chunk_lengths` |
//! | complete identity | exact sum of the four leaves | 34 | 70 | 106 | `8,044` | `5,248` | `certificateProjectionIdentity_iff_emitted` |
//!
//! The cited Lean results prove the abstract arithmetic schedules. Concrete
//! generated source/emitted assignments and their column decoders are not yet
//! theorem inputs; Rust's fail-closed trace replay is the present production
//! conformance layer.

use std::collections::BTreeMap;

use neo_math::ring::{D, PHI_MID_DEGREE};
use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{ProductFactorTrace, ProductSumBatchTrace, ProductSumIdentityTrace};
use crate::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use crate::engine::r1cs_circuit::{
    Lc, PolynomialEvaluationTraceEntry, ProjectionIdentityRole, R1csEncodingTrace, R1csSnapshot, Var,
};

use super::product_sum::ProductSumBatchPlan;
use super::GadgetNativeError;

const IDENTITY_COUNT: usize = 31;
const PAIR_COUNT: usize = 15;
const EVALUATION_COUNT: usize = PAIR_COUNT + 2;
const RETAINED_FIELDS: usize = 2 * EVALUATION_COUNT;
const SOURCE_ROWS_PER_IDENTITY: usize = 1_916;
const SOURCE_COLUMNS_PER_IDENTITY: usize = 1_914;

/// Diagnostic operation owned by one compact evaluation identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionEvaluationKind {
    Input { pair: usize },
    Output,
    Quotient,
}

/// Exact treatment of coefficient zero in an evaluation limb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionCoefficientZero {
    SubtractFromResult,
    Absent,
}

/// One exact evaluation operation in the production compact plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionEvaluationCompactionAudit {
    pub kind: ProjectionEvaluationKind,
    pub source_row_offset: usize,
    pub source_row_count: usize,
    pub coefficient_count: usize,
    pub product_coefficient_indices: Vec<usize>,
    pub power_indices_by_limb: [Vec<usize>; 2],
    pub retained_ordinals: [usize; 2],
    pub retained_column_offsets: [usize; 2],
    pub coefficient_zero: [ProjectionCoefficientZero; 2],
    pub product_counts: [usize; 2],
    pub chunk_sizes: [Vec<usize>; 2],
}

/// Semantic operand named by one terminal direct-product factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionFinalOperand {
    RhoEvaluation { pair: usize, limb: usize },
    InputEvaluation { pair: usize, limb: usize },
    QuotientEvaluation { limb: usize },
    Phi { limb: usize },
}

/// Exact coefficient used by one terminal direct-product factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionFinalCoefficient {
    One,
    NegOne,
    W,
    NegW,
}

/// One ordered terminal direct-product factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionFinalFactorAudit {
    pub left: ProjectionFinalOperand,
    pub right: ProjectionFinalOperand,
    pub coefficient: ProjectionFinalCoefficient,
}

/// One exact final-limb operation in the production compact plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionFinalLimbCompactionAudit {
    pub limb: usize,
    pub source_row_offset: usize,
    pub result_retained_ordinal: usize,
    pub chunk_sizes: Vec<usize>,
    pub factors: Vec<ProjectionFinalFactorAudit>,
}

/// One nonzero entry of the retained-output binding matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionRetainedBindingAudit {
    pub identity: usize,
    pub retained_ordinal: usize,
    pub coefficient: ProjectionFinalCoefficient,
}

/// Normalized schema extracted from the exact product-sum plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionIdentityCompactionSchema {
    pub source_rows: usize,
    pub source_columns: usize,
    pub retained_column_offsets: Vec<usize>,
    pub evaluations: Vec<ProjectionEvaluationCompactionAudit>,
    pub retained_bindings: Vec<ProjectionRetainedBindingAudit>,
    pub final_limbs: Vec<ProjectionFinalLimbCompactionAudit>,
}

/// Read-only production audit used by the generated Lean certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionIdentityCompactionAudit {
    /// Roles label cost ownership only; source-row validation supplies semantic authority.
    pub roles: Vec<ProjectionIdentityRole>,
    pub schema: ProjectionIdentityCompactionSchema,
}

/// Validate and expose the exact normalized production compaction plan.
#[doc(hidden)]
pub fn audit_projection_identity_compaction(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<ProjectionIdentityCompactionAudit, GadgetNativeError> {
    let plans = exact_product_sum_batches(source, trace)?;
    if plans.len() != IDENTITY_COUNT {
        return manifest("compaction plan census");
    }
    let identities = trace.projection_identities();
    let representative = normalized_compaction_schema(trace, &identities[0], &plans[0])?;
    for (identity, plan) in identities.iter().zip(&plans).skip(1) {
        if normalized_compaction_schema(trace, identity, plan)? != representative {
            return manifest("normalized compaction schema");
        }
    }
    Ok(ProjectionIdentityCompactionAudit {
        roles: identities.iter().map(|identity| identity.role).collect(),
        schema: representative,
    })
}

pub(super) fn exact_product_sum_batches(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<Vec<ProductSumBatchPlan>, GadgetNativeError> {
    let identities = trace.projection_identities();
    if identities.is_empty()
        || identities
            .iter()
            .all(|identity| identity.role == ProjectionIdentityRole::Standalone)
    {
        return Ok(Vec::new());
    }

    let validated = validate_projection_identity_traces(source, trace)?;
    let expected_roles = expected_roles();
    if validated.census.identities != IDENTITY_COUNT
        || validated.census.pairs != IDENTITY_COUNT * PAIR_COUNT
        || validated.census.polynomial_evaluations != IDENTITY_COUNT * EVALUATION_COUNT
        || validated.census.k_products != IDENTITY_COUNT * (PAIR_COUNT + 1)
        || validated.census.source_rows != IDENTITY_COUNT * SOURCE_ROWS_PER_IDENTITY
        || validated.census.source_columns != IDENTITY_COUNT * SOURCE_COLUMNS_PER_IDENTITY
        || validated.roles != expected_roles
    {
        return manifest("global role or geometry census");
    }

    identities
        .iter()
        .enumerate()
        .map(|(index, identity)| {
            if identity.input_columns.len() != PAIR_COUNT
                || identity.source_rows.len() != SOURCE_ROWS_PER_IDENTITY
                || identity.allocated_columns.len() != SOURCE_COLUMNS_PER_IDENTITY
                || identity.final_limb_rows.start + 2 != identity.source_rows.end
                || identity.final_limb_rows.end != identity.source_rows.end
            {
                return manifest("per-identity fixed geometry");
            }
            if identity.role != expected_roles[index] {
                return manifest("ordered identity role");
            }
            build_batch(trace, identity)
        })
        .collect()
}

fn build_batch(
    trace: &R1csEncodingTrace,
    identity: &crate::engine::r1cs_circuit::ProjectionIdentityTraceEntry,
) -> Result<ProductSumBatchPlan, GadgetNativeError> {
    let evaluations = trace.polynomial_evaluations();
    let products = trace.k_muls();
    let mut identities = Vec::with_capacity(RETAINED_FIELDS + 2);
    let mut retained_columns = Vec::with_capacity(RETAINED_FIELDS);
    let mut stage_rows = Vec::with_capacity(RETAINED_FIELDS + 2);

    for evaluation_index in identity.input_evaluations.clone() {
        append_evaluation(
            evaluations
                .get(evaluation_index)
                .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                    detail: "input evaluation index",
                })?,
            &mut identities,
            &mut retained_columns,
            &mut stage_rows,
        );
    }
    append_evaluation(
        evaluations
            .get(identity.output_evaluation)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "output evaluation index",
            })?,
        &mut identities,
        &mut retained_columns,
        &mut stage_rows,
    );
    append_evaluation(
        evaluations
            .get(identity.quotient_evaluation)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "quotient evaluation index",
            })?,
        &mut identities,
        &mut retained_columns,
        &mut stage_rows,
    );

    if identities.len() != RETAINED_FIELDS || retained_columns.len() != RETAINED_FIELDS {
        return manifest("retained evaluation boundary");
    }
    for limb in 0..2 {
        let mut factors = Vec::with_capacity(2 * (PAIR_COUNT + 1));
        for product_index in identity.pair_products.clone() {
            append_k_limb_factors(&products[product_index], limb, F::ONE, &mut factors);
        }
        append_k_limb_factors(&products[identity.quotient_phi_product], limb, -F::ONE, &mut factors);
        factors.shrink_to_fit();
        identities.push(ProductSumIdentityTrace {
            factors,
            result: Lc::from_var(column_var(evaluations[identity.output_evaluation].output_cols[limb])),
        });
        stage_rows.push(identity.final_limb_rows.start + limb);
    }

    Ok(ProductSumBatchPlan::traced_terminal(
        ProductSumBatchTrace {
            row_start: identity.source_rows.start,
            row_end: identity.source_rows.end,
            allocated_columns: identity.allocated_columns.clone().collect(),
            retained_columns,
            identities,
        },
        stage_rows,
        identity.final_limb_rows.clone().collect(),
    ))
}

fn append_evaluation(
    evaluation: &PolynomialEvaluationTraceEntry,
    identities: &mut Vec<ProductSumIdentityTrace>,
    retained_columns: &mut Vec<usize>,
    stage_rows: &mut Vec<usize>,
) {
    for limb in 0..2 {
        let factors = (1..evaluation.coefficient_cols.len())
            .map(|coefficient| ProductFactorTrace {
                left: Lc::from_var(column_var(evaluation.coefficient_cols[coefficient])),
                right: Lc::from_var(column_var(evaluation.power_cols[coefficient][limb])),
                coefficient: F::ONE,
            })
            .collect();
        let mut result = Lc::from_var(column_var(evaluation.output_cols[limb]));
        if limb == 0 {
            result.add_term(column_var(evaluation.coefficient_cols[0]), -F::ONE);
        }
        identities.push(ProductSumIdentityTrace { factors, result });
        retained_columns.push(evaluation.output_cols[limb]);
        stage_rows.push(evaluation.row_start);
    }
}

fn append_k_limb_factors(
    product: &crate::engine::r1cs_circuit::KMulTraceEntry,
    limb: usize,
    sign: F,
    factors: &mut Vec<ProductFactorTrace>,
) {
    if limb == 0 {
        factors.push(ProductFactorTrace {
            left: product.a[0].clone(),
            right: product.b[0].clone(),
            coefficient: sign,
        });
        factors.push(ProductFactorTrace {
            left: product.a[1].clone(),
            right: product.b[1].clone(),
            coefficient: sign * <Fq as BinomiallyExtendable<2>>::W,
        });
    } else {
        factors.push(ProductFactorTrace {
            left: product.a[0].clone(),
            right: product.b[1].clone(),
            coefficient: sign,
        });
        factors.push(ProductFactorTrace {
            left: product.a[1].clone(),
            right: product.b[0].clone(),
            coefficient: sign,
        });
    }
}

fn normalized_compaction_schema(
    trace: &R1csEncodingTrace,
    identity: &crate::engine::r1cs_circuit::ProjectionIdentityTraceEntry,
    plan: &ProductSumBatchPlan,
) -> Result<ProjectionIdentityCompactionSchema, GadgetNativeError> {
    let batch = plan.trace();
    let evaluations = trace.polynomial_evaluations();
    let retained_ordinals = batch
        .retained_columns
        .iter()
        .enumerate()
        .map(|(ordinal, &column)| (column, ordinal))
        .collect::<BTreeMap<_, _>>();
    let evaluation_indices = identity
        .input_evaluations
        .clone()
        .map(|index| {
            (
                ProjectionEvaluationKind::Input {
                    pair: index - identity.input_evaluations.start,
                },
                index,
            )
        })
        .chain([
            (ProjectionEvaluationKind::Output, identity.output_evaluation),
            (ProjectionEvaluationKind::Quotient, identity.quotient_evaluation),
        ])
        .collect::<Vec<_>>();
    if evaluation_indices.len() != EVALUATION_COUNT || batch.identities.len() != RETAINED_FIELDS + 2 {
        return manifest("compact identity census");
    }

    let mut evaluation_audits = Vec::with_capacity(EVALUATION_COUNT);
    for (evaluation_ordinal, (kind, evaluation_index)) in evaluation_indices.into_iter().enumerate() {
        let evaluation = evaluations
            .get(evaluation_index)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "audit evaluation index",
            })?;
        let mut retained = [0; 2];
        let mut retained_offsets = [0; 2];
        let mut product_counts = [0; 2];
        let mut chunks = [Vec::new(), Vec::new()];
        for limb in 0..2 {
            let product_identity = &batch.identities[2 * evaluation_ordinal + limb];
            retained[limb] = *retained_ordinals.get(&evaluation.output_cols[limb]).ok_or(
                GadgetNativeError::ProjectionIdentityManifest {
                    detail: "evaluation retained ordinal",
                },
            )?;
            retained_offsets[limb] = evaluation.output_cols[limb] - identity.allocated_columns.start;
            product_counts[limb] = product_identity.factors.len();
            chunks[limb] = product_identity
                .factors
                .chunks(super::MAX_PRODUCT_TERMS)
                .map(<[_]>::len)
                .collect();
            if product_identity.factors.len() + 1 != evaluation.coefficient_cols.len() {
                return manifest("evaluation product count");
            }
            for (coefficient, factor) in product_identity.factors.iter().enumerate() {
                let coefficient = coefficient + 1;
                if factor.coefficient != F::ONE
                    || !lc_eq(
                        &factor.left,
                        &Lc::from_var(column_var(evaluation.coefficient_cols[coefficient])),
                    )
                    || !lc_eq(
                        &factor.right,
                        &Lc::from_var(column_var(evaluation.power_cols[coefficient][limb])),
                    )
                {
                    return manifest("evaluation factor schedule");
                }
            }
            let mut expected_result = Lc::from_var(column_var(evaluation.output_cols[limb]));
            if limb == 0 {
                expected_result.add_term(column_var(evaluation.coefficient_cols[0]), -F::ONE);
            }
            if !lc_eq(&product_identity.result, &expected_result) {
                return manifest("evaluation coefficient-zero schedule");
            }
        }
        evaluation_audits.push(ProjectionEvaluationCompactionAudit {
            kind,
            source_row_offset: evaluation.row_start - identity.source_rows.start,
            source_row_count: evaluation.row_end - evaluation.row_start,
            coefficient_count: evaluation.coefficient_cols.len(),
            product_coefficient_indices: (1..evaluation.coefficient_cols.len()).collect(),
            power_indices_by_limb: [
                (1..evaluation.coefficient_cols.len()).collect(),
                (1..evaluation.coefficient_cols.len()).collect(),
            ],
            retained_ordinals: retained,
            retained_column_offsets: retained_offsets,
            coefficient_zero: [
                ProjectionCoefficientZero::SubtractFromResult,
                ProjectionCoefficientZero::Absent,
            ],
            product_counts,
            chunk_sizes: chunks,
        });
    }

    let retained_bindings = retained_binding_audit(batch, &retained_ordinals)?;
    let expected_bindings = (0..RETAINED_FIELDS)
        .map(|identity| ProjectionRetainedBindingAudit {
            identity,
            retained_ordinal: identity,
            coefficient: ProjectionFinalCoefficient::One,
        })
        .collect::<Vec<_>>();
    if retained_bindings != expected_bindings {
        return manifest("retained binding matrix");
    }

    let mut final_limbs = Vec::with_capacity(2);
    for limb in 0..2 {
        let product_identity = &batch.identities[RETAINED_FIELDS + limb];
        let factor_audits = expected_final_factors(limb);
        if product_identity.factors.len() != factor_audits.len() {
            return manifest("terminal factor count");
        }
        for (factor, audit) in product_identity.factors.iter().zip(&factor_audits) {
            if factor.coefficient != audit.coefficient.field()
                || !lc_eq(&factor.left, &final_operand_lc(trace, identity, audit.left)?)
                || !lc_eq(&factor.right, &final_operand_lc(trace, identity, audit.right)?)
            {
                return manifest("terminal factor schedule");
            }
        }
        let result_column = evaluations[identity.output_evaluation].output_cols[limb];
        if !lc_eq(&product_identity.result, &Lc::from_var(column_var(result_column))) {
            return manifest("terminal result schedule");
        }
        final_limbs.push(ProjectionFinalLimbCompactionAudit {
            limb,
            source_row_offset: identity.final_limb_rows.start + limb - identity.source_rows.start,
            result_retained_ordinal: *retained_ordinals.get(&result_column).ok_or(
                GadgetNativeError::ProjectionIdentityManifest {
                    detail: "terminal result retained ordinal",
                },
            )?,
            chunk_sizes: product_identity
                .factors
                .chunks(super::MAX_PRODUCT_TERMS)
                .map(<[_]>::len)
                .collect(),
            factors: factor_audits,
        });
    }

    Ok(ProjectionIdentityCompactionSchema {
        source_rows: identity.source_rows.len(),
        source_columns: identity.allocated_columns.len(),
        retained_column_offsets: batch
            .retained_columns
            .iter()
            .map(|column| column - identity.allocated_columns.start)
            .collect(),
        evaluations: evaluation_audits,
        retained_bindings,
        final_limbs,
    })
}

fn retained_binding_audit(
    batch: &ProductSumBatchTrace,
    retained_ordinals: &BTreeMap<usize, usize>,
) -> Result<Vec<ProjectionRetainedBindingAudit>, GadgetNativeError> {
    let mut bindings = Vec::new();
    for (identity, product_identity) in batch.identities.iter().take(RETAINED_FIELDS).enumerate() {
        for (column, coefficient) in normalized_terms(&product_identity.result) {
            let Some(&retained_ordinal) = retained_ordinals.get(&column) else {
                continue;
            };
            bindings.push(ProjectionRetainedBindingAudit {
                identity,
                retained_ordinal,
                coefficient: ProjectionFinalCoefficient::from_field(coefficient).ok_or(
                    GadgetNativeError::ProjectionIdentityManifest {
                        detail: "retained binding coefficient",
                    },
                )?,
            });
        }
    }
    Ok(bindings)
}

fn expected_final_factors(limb: usize) -> Vec<ProjectionFinalFactorAudit> {
    let mut factors = Vec::with_capacity(2 * (PAIR_COUNT + 1));
    for pair in 0..PAIR_COUNT {
        if limb == 0 {
            factors.push(ProjectionFinalFactorAudit {
                left: ProjectionFinalOperand::RhoEvaluation { pair, limb: 0 },
                right: ProjectionFinalOperand::InputEvaluation { pair, limb: 0 },
                coefficient: ProjectionFinalCoefficient::One,
            });
            factors.push(ProjectionFinalFactorAudit {
                left: ProjectionFinalOperand::RhoEvaluation { pair, limb: 1 },
                right: ProjectionFinalOperand::InputEvaluation { pair, limb: 1 },
                coefficient: ProjectionFinalCoefficient::W,
            });
        } else {
            factors.push(ProjectionFinalFactorAudit {
                left: ProjectionFinalOperand::RhoEvaluation { pair, limb: 0 },
                right: ProjectionFinalOperand::InputEvaluation { pair, limb: 1 },
                coefficient: ProjectionFinalCoefficient::One,
            });
            factors.push(ProjectionFinalFactorAudit {
                left: ProjectionFinalOperand::RhoEvaluation { pair, limb: 1 },
                right: ProjectionFinalOperand::InputEvaluation { pair, limb: 0 },
                coefficient: ProjectionFinalCoefficient::One,
            });
        }
    }
    if limb == 0 {
        factors.push(ProjectionFinalFactorAudit {
            left: ProjectionFinalOperand::QuotientEvaluation { limb: 0 },
            right: ProjectionFinalOperand::Phi { limb: 0 },
            coefficient: ProjectionFinalCoefficient::NegOne,
        });
        factors.push(ProjectionFinalFactorAudit {
            left: ProjectionFinalOperand::QuotientEvaluation { limb: 1 },
            right: ProjectionFinalOperand::Phi { limb: 1 },
            coefficient: ProjectionFinalCoefficient::NegW,
        });
    } else {
        factors.push(ProjectionFinalFactorAudit {
            left: ProjectionFinalOperand::QuotientEvaluation { limb: 0 },
            right: ProjectionFinalOperand::Phi { limb: 1 },
            coefficient: ProjectionFinalCoefficient::NegOne,
        });
        factors.push(ProjectionFinalFactorAudit {
            left: ProjectionFinalOperand::QuotientEvaluation { limb: 1 },
            right: ProjectionFinalOperand::Phi { limb: 0 },
            coefficient: ProjectionFinalCoefficient::NegOne,
        });
    }
    factors
}

fn final_operand_lc(
    trace: &R1csEncodingTrace,
    identity: &crate::engine::r1cs_circuit::ProjectionIdentityTraceEntry,
    operand: ProjectionFinalOperand,
) -> Result<Lc, GadgetNativeError> {
    let evaluations = trace.polynomial_evaluations();
    let column = |column| Lc::from_var(column_var(column));
    match operand {
        ProjectionFinalOperand::RhoEvaluation { pair, limb } => identity
            .rho_evaluation_outputs
            .get(pair)
            .and_then(|output| output.get(limb))
            .copied()
            .map(column)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "rho evaluation operand",
            }),
        ProjectionFinalOperand::InputEvaluation { pair, limb } => evaluations
            .get(identity.input_evaluations.start + pair)
            .and_then(|evaluation| evaluation.output_cols.get(limb))
            .copied()
            .map(column)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "input evaluation operand",
            }),
        ProjectionFinalOperand::QuotientEvaluation { limb } => evaluations
            .get(identity.quotient_evaluation)
            .and_then(|evaluation| evaluation.output_cols.get(limb))
            .copied()
            .map(column)
            .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                detail: "quotient evaluation operand",
            }),
        ProjectionFinalOperand::Phi { limb } => {
            let mut value = identity
                .power_columns
                .get(D)
                .and_then(|power| power.get(limb))
                .copied()
                .map(column)
                .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                    detail: "Phi high operand",
                })?;
            let middle = identity
                .power_columns
                .get(PHI_MID_DEGREE)
                .and_then(|power| power.get(limb))
                .copied()
                .ok_or(GadgetNativeError::ProjectionIdentityManifest {
                    detail: "Phi middle operand",
                })?;
            value.add_term(column_var(middle), F::ONE);
            if limb == 0 {
                value.add_constant(F::ONE);
            }
            Ok(value)
        }
    }
}

impl ProjectionFinalCoefficient {
    fn field(self) -> F {
        let w = <Fq as BinomiallyExtendable<2>>::W;
        match self {
            Self::One => F::ONE,
            Self::NegOne => -F::ONE,
            Self::W => w,
            Self::NegW => -w,
        }
    }

    fn from_field(value: F) -> Option<Self> {
        [Self::One, Self::NegOne, Self::W, Self::NegW]
            .into_iter()
            .find(|coefficient| coefficient.field() == value)
    }
}

fn normalized_terms(value: &Lc) -> BTreeMap<usize, F> {
    let mut terms = BTreeMap::new();
    for &(column, coefficient) in &value.terms {
        *terms.entry(column).or_insert(F::ZERO) += coefficient;
    }
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    terms
}

fn lc_eq(left: &Lc, right: &Lc) -> bool {
    left.constant == right.constant && normalized_terms(left) == normalized_terms(right)
}

fn expected_roles() -> Vec<ProjectionIdentityRole> {
    let mut roles = Vec::with_capacity(IDENTITY_COUNT);
    roles.extend((0..18).map(|lane| ProjectionIdentityRole::CommitmentLane { lane }));
    roles.extend((0..5).map(|column| ProjectionIdentityRole::ActiveXColumn { column }));
    for row in 0..3 {
        roles.extend((0..2).map(|limb| ProjectionIdentityRole::YRingLimb { row, limb }));
    }
    roles.extend((0..2).map(|limb| ProjectionIdentityRole::YZColLimb { limb }));
    roles
}

fn column_var(column: usize) -> Var {
    Var::from_column_for_trace(column)
}

fn manifest<T>(detail: &'static str) -> Result<T, GadgetNativeError> {
    Err(GadgetNativeError::ProjectionIdentityManifest { detail })
}
