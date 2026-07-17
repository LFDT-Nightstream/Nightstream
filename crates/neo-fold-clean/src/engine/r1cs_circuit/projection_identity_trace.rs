//! Exact validation of production Π_RLC projection-identity provenance.
//!
//! Owns: full source-row reconstruction, row/column geometry, topological
//! dependency checks, non-escape, identity ownership, and exact trace census.
//!
//! Does not own: transcript authority, the exact-or-bad-root reduction, or a
//! compact lowering. A validated trace only proves that metadata names the
//! authoritative R1CS program exactly.
//!
//! Emits constraints: no.
//!
//! Authority boundary: source R1CS rows are authoritative. Trace metadata is
//! rejected unless every named row, column, dependency, and final equation
//! replays exactly.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `evaluations.inputs` | Ordered degree-53 evaluations for all 15 inputs | no | `ring_action::enforce_eval_at_beta` | compact evaluation refinement open |
//! | `evaluations.output` | Ordered degree-53 parent evaluation | no | `ring_action::enforce_eval_at_beta` | compact evaluation refinement open |
//! | `evaluations.quotient` | Ordered degree-52 quotient evaluation | no | `ring_action::enforce_eval_at_beta` | compact evaluation refinement open |
//! | `k_products.rho_times_input` | Fifteen exact Karatsuba products | no | `field_ext::enforce_k_mul` | final-limb batching refinement open |
//! | `k_products.quotient_times_phi` | Exact quotient/Phi Karatsuba product | no | `field_ext::enforce_k_mul` | final-limb batching refinement open |
//! | `final_limb_checks` | Both extension-field limbs agree | no | `ring_action` | `Semantics/ProjectionBoundary.lean` |

use std::collections::BTreeMap;

use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::encoding_trace::{
    KMulTraceEntry, PolynomialEvaluationTraceEntry, ProjectionIdentityRole, ProjectionIdentityTraceEntry,
    R1csEncodingTrace,
};
use super::field_ext::KLc;
use super::ring_action::PROJECTION_QUOTIENT_LEN;
use super::{Lc, R1csSnapshot, Var};
use neo_math::ring::{D, PHI_MID_DEGREE};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectionIdentityTraceCensus {
    pub identities: usize,
    pub pairs: usize,
    pub polynomial_evaluations: usize,
    pub k_products: usize,
    pub source_rows: usize,
    pub source_columns: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedProjectionIdentityTrace {
    pub census: ProjectionIdentityTraceCensus,
    pub roles: Vec<ProjectionIdentityRole>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProjectionIdentityTraceError {
    #[error("projection identity {identity} has invalid {phase} geometry")]
    Geometry {
        identity: usize,
        phase: &'static str,
    },
    #[error("projection identity {identity} {phase} references an out-of-range trace index")]
    Index {
        identity: usize,
        phase: &'static str,
    },
    #[error("projection identity {identity} {phase} does not match source row {row}")]
    RowMismatch {
        identity: usize,
        phase: &'static str,
        row: usize,
    },
    #[error("projection identity {identity} {phase} is not topological at column {column}")]
    NonTopological {
        identity: usize,
        phase: &'static str,
        column: usize,
    },
    #[error("projection identity {identity} overlaps identity {previous} at {kind} {index}")]
    Overlap {
        identity: usize,
        previous: usize,
        kind: &'static str,
        index: usize,
    },
    #[error("projection identity {owner} temporary column {column} escapes at row {row}")]
    Escape {
        owner: usize,
        column: usize,
        row: usize,
    },
}

pub fn validate_projection_identity_traces(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<ValidatedProjectionIdentityTrace, ProjectionIdentityTraceError> {
    let identities = trace.projection_identities();
    let evaluations = trace.polynomial_evaluations();
    let k_products = trace.k_muls();
    let mut row_owner = vec![None; source.rows()];
    let mut column_owner = vec![None; source.cols()];
    let mut evaluation_owner = vec![None; evaluations.len()];
    let mut product_owner = vec![None; k_products.len()];
    let mut census = ProjectionIdentityTraceCensus::default();
    let mut roles = Vec::with_capacity(identities.len());

    for (identity_index, identity) in identities.iter().enumerate() {
        validate_identity_bounds(source, identity, identity_index)?;
        claim_range(&mut row_owner, identity.source_rows.clone(), identity_index, "row")?;
        claim_range(
            &mut column_owner,
            identity.allocated_columns.clone(),
            identity_index,
            "column",
        )?;
        claim_range(
            &mut evaluation_owner,
            identity.input_evaluations.clone(),
            identity_index,
            "evaluation",
        )?;
        claim_index(
            &mut evaluation_owner,
            identity.output_evaluation,
            identity_index,
            "output evaluation",
        )?;
        claim_index(
            &mut evaluation_owner,
            identity.quotient_evaluation,
            identity_index,
            "quotient evaluation",
        )?;
        claim_range(
            &mut product_owner,
            identity.pair_products.clone(),
            identity_index,
            "K product",
        )?;
        claim_index(
            &mut product_owner,
            identity.quotient_phi_product,
            identity_index,
            "quotient/Phi product",
        )?;
        validate_identity(source, trace, identity, identity_index)?;
        census.identities += 1;
        census.pairs += identity.input_columns.len();
        census.polynomial_evaluations += identity.input_evaluations.len() + 2;
        census.k_products += identity.pair_products.len() + 1;
        census.source_rows += identity.source_rows.len();
        census.source_columns += identity.allocated_columns.len();
        roles.push(identity.role);
    }

    validate_no_escape(source, identities, &column_owner)?;
    Ok(ValidatedProjectionIdentityTrace { census, roles })
}

fn validate_identity_bounds(
    source: &R1csSnapshot,
    identity: &ProjectionIdentityTraceEntry,
    identity_index: usize,
) -> Result<(), ProjectionIdentityTraceError> {
    let pair_count = identity.input_columns.len();
    let geometry = identity.source_rows.is_empty()
        || identity.source_rows.end > source.rows()
        || identity.allocated_columns.is_empty()
        || identity.allocated_columns.end > source.cols()
        || pair_count == 0
        || identity.rho_columns.len() != pair_count
        || identity.rho_evaluation_outputs.len() != pair_count
        || identity.input_evaluations.len() != pair_count
        || identity.pair_products.len() != pair_count
        || identity.power_columns.len() <= D
        || identity.output_columns.len() != D
        || identity.quotient_columns.len() != PROJECTION_QUOTIENT_LEN
        || identity.final_limb_rows.len() != 2
        || identity
            .rho_columns
            .iter()
            .chain(&identity.input_columns)
            .any(|columns| columns.len() != D);
    if geometry {
        return Err(ProjectionIdentityTraceError::Geometry {
            identity: identity_index,
            phase: "identity",
        });
    }
    Ok(())
}

fn validate_identity(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    identity: &ProjectionIdentityTraceEntry,
    identity_index: usize,
) -> Result<(), ProjectionIdentityTraceError> {
    let evaluations = trace.polynomial_evaluations();
    let products = trace.k_muls();
    let mut row = identity.source_rows.start;
    let mut column = identity.allocated_columns.start;

    for pair in 0..identity.input_columns.len() {
        let evaluation_index = identity.input_evaluations.start + pair;
        let evaluation = evaluations
            .get(evaluation_index)
            .ok_or(ProjectionIdentityTraceError::Index {
                identity: identity_index,
                phase: "input evaluation",
            })?;
        (row, column) = validate_evaluation(
            source,
            evaluation,
            &identity.input_columns[pair],
            &identity.power_columns,
            row,
            column,
            identity_index,
            "input evaluation",
        )?;
        let product_index = identity.pair_products.start + pair;
        let product = products
            .get(product_index)
            .ok_or(ProjectionIdentityTraceError::Index {
                identity: identity_index,
                phase: "rho/input product",
            })?;
        let expected_left = k_from_columns(identity.rho_evaluation_outputs[pair]);
        let expected_right = k_from_columns(evaluation.output_cols);
        (row, column) = validate_k_product(
            source,
            product,
            &expected_left,
            &expected_right,
            row,
            column,
            identity_index,
            "rho/input product",
        )?;
    }

    let output_evaluation = evaluations
        .get(identity.output_evaluation)
        .ok_or(ProjectionIdentityTraceError::Index {
            identity: identity_index,
            phase: "output evaluation",
        })?;
    (row, column) = validate_evaluation(
        source,
        output_evaluation,
        &identity.output_columns,
        &identity.power_columns,
        row,
        column,
        identity_index,
        "output evaluation",
    )?;

    let quotient_evaluation =
        evaluations
            .get(identity.quotient_evaluation)
            .ok_or(ProjectionIdentityTraceError::Index {
                identity: identity_index,
                phase: "quotient evaluation",
            })?;
    (row, column) = validate_evaluation(
        source,
        quotient_evaluation,
        &identity.quotient_columns,
        &identity.power_columns,
        row,
        column,
        identity_index,
        "quotient evaluation",
    )?;

    let quotient_phi = products
        .get(identity.quotient_phi_product)
        .ok_or(ProjectionIdentityTraceError::Index {
            identity: identity_index,
            phase: "quotient/Phi product",
        })?;
    let quotient_value = k_from_columns(quotient_evaluation.output_cols);
    let mut phi_c0 = Lc::from_var(column_var(identity.power_columns[D][0])).add_scaled(
        &Lc::from_var(column_var(identity.power_columns[PHI_MID_DEGREE][0])),
        F::ONE,
    );
    phi_c0.add_constant(F::ONE);
    let phi = KLc {
        c0: phi_c0,
        c1: Lc::from_var(column_var(identity.power_columns[D][1])).add_scaled(
            &Lc::from_var(column_var(identity.power_columns[PHI_MID_DEGREE][1])),
            F::ONE,
        ),
    };
    (row, column) = validate_k_product(
        source,
        quotient_phi,
        &quotient_value,
        &phi,
        row,
        column,
        identity_index,
        "quotient/Phi product",
    )?;

    if identity.final_limb_rows != (row..row + 2) {
        return Err(ProjectionIdentityTraceError::Geometry {
            identity: identity_index,
            phase: "final limb rows",
        });
    }
    let pair_outputs = identity
        .pair_products
        .clone()
        .map(|index| &products[index])
        .collect::<Vec<_>>();
    for limb in 0..2 {
        let mut difference = Lc::zero();
        for product in &pair_outputs {
            difference.add_term(product.output[limb], F::ONE);
        }
        difference.add_term(quotient_phi.output[limb], -F::ONE);
        difference.add_term(column_var(output_evaluation.output_cols[limb]), -F::ONE);
        validate_row(
            source,
            row + limb,
            &difference,
            &Lc::from_var(Var::ONE),
            &Lc::zero(),
            identity_index,
            "final limb check",
        )?;
    }
    row += 2;
    if row != identity.source_rows.end || column != identity.allocated_columns.end {
        return Err(ProjectionIdentityTraceError::Geometry {
            identity: identity_index,
            phase: "complete identity",
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_evaluation(
    source: &R1csSnapshot,
    evaluation: &PolynomialEvaluationTraceEntry,
    coefficients: &[usize],
    powers: &[[usize; 2]],
    row: usize,
    column: usize,
    identity: usize,
    phase: &'static str,
) -> Result<(usize, usize), ProjectionIdentityTraceError> {
    let products = 2 * coefficients.len().saturating_sub(1);
    let columns = products + 2;
    if coefficients.is_empty()
        || evaluation.row_start != row
        || evaluation.row_end != row + columns
        || evaluation.allocated_columns != (column..column + columns).collect::<Vec<_>>()
        || evaluation.coefficient_cols != coefficients
        || evaluation.power_cols != powers[..coefficients.len()]
        || evaluation.output_cols != [column + products, column + products + 1]
    {
        return Err(ProjectionIdentityTraceError::Geometry { identity, phase });
    }
    for coefficient in 1..coefficients.len() {
        for limb in 0..2 {
            let offset = 2 * (coefficient - 1) + limb;
            let target = column + offset;
            validate_topological(
                target,
                [coefficients[coefficient], powers[coefficient][limb]],
                identity,
                phase,
            )?;
            validate_row(
                source,
                row + offset,
                &Lc::from_var(column_var(coefficients[coefficient])),
                &Lc::from_var(column_var(powers[coefficient][limb])),
                &Lc::from_var(column_var(target)),
                identity,
                phase,
            )?;
        }
    }
    let output_c0 = evaluation.output_cols[0];
    let output_c1 = evaluation.output_cols[1];
    let mut sum_c0 = Lc::from_var(column_var(coefficients[0]));
    let mut sum_c1 = Lc::zero();
    for offset in 0..products / 2 {
        sum_c0.add_term(column_var(column + 2 * offset), F::ONE);
        sum_c1.add_term(column_var(column + 2 * offset + 1), F::ONE);
    }
    validate_topological(output_c0, sum_c0.terms.iter().map(|(input, _)| *input), identity, phase)?;
    validate_row(
        source,
        evaluation.row_end - 2,
        &Lc::from_var(column_var(output_c0)).add_scaled(&sum_c0, -F::ONE),
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
        identity,
        phase,
    )?;
    validate_topological(output_c1, sum_c1.terms.iter().map(|(input, _)| *input), identity, phase)?;
    validate_row(
        source,
        evaluation.row_end - 1,
        &Lc::from_var(column_var(output_c1)).add_scaled(&sum_c1, -F::ONE),
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
        identity,
        phase,
    )?;
    Ok((evaluation.row_end, column + columns))
}

#[allow(clippy::too_many_arguments)]
fn validate_k_product(
    source: &R1csSnapshot,
    product: &KMulTraceEntry,
    expected_a: &KLc,
    expected_b: &KLc,
    row: usize,
    column: usize,
    identity: usize,
    phase: &'static str,
) -> Result<(usize, usize), ProjectionIdentityTraceError> {
    if product.source_rows != (row..row + 5)
        || product
            .a
            .iter()
            .zip([&expected_a.c0, &expected_a.c1])
            .any(|(actual, expected)| normalize_lc(actual) != normalize_lc(expected))
        || product
            .b
            .iter()
            .zip([&expected_b.c0, &expected_b.c1])
            .any(|(actual, expected)| normalize_lc(actual) != normalize_lc(expected))
        || product.intermediates.map(Var::col) != [column, column + 1, column + 2]
        || product.output.map(Var::col) != [column + 3, column + 4]
    {
        return Err(ProjectionIdentityTraceError::Geometry { identity, phase });
    }
    let [p, q, r] = product.intermediates;
    let sum_a = product.a[0].clone().add_scaled(&product.a[1], F::ONE);
    let sum_b = product.b[0].clone().add_scaled(&product.b[1], F::ONE);
    let w = <Fq as BinomiallyExtendable<2>>::W;
    let output_c0 = Lc::from_var(product.output[0])
        .add_scaled(&Lc::from_var(p), -F::ONE)
        .add_scaled(&Lc::from_var(q), -w);
    let output_c1 = Lc::from_var(product.output[1])
        .add_scaled(&Lc::from_var(r), -F::ONE)
        .add_scaled(&Lc::from_var(p), F::ONE)
        .add_scaled(&Lc::from_var(q), F::ONE);
    let rows = [
        (product.a[0].clone(), product.b[0].clone(), Lc::from_var(p)),
        (product.a[1].clone(), product.b[1].clone(), Lc::from_var(q)),
        (sum_a, sum_b, Lc::from_var(r)),
        (output_c0, Lc::from_var(Var::ONE), Lc::zero()),
        (output_c1, Lc::from_var(Var::ONE), Lc::zero()),
    ];
    for (offset, (a, b, c)) in rows.iter().enumerate() {
        let target = column + offset;
        // The last two rows are linear definitions written as
        // `target - expression = 0`, so their A-side necessarily mentions the
        // target itself.  Topology concerns only the other dependencies.
        let inputs = a
            .terms
            .iter()
            .chain(&b.terms)
            .map(|(input, _)| *input)
            .filter(|input| *input != target);
        validate_topological(target, inputs, identity, phase)?;
        validate_row(source, row + offset, a, b, c, identity, phase)?;
    }
    Ok((row + 5, column + 5))
}

fn validate_topological(
    target: usize,
    inputs: impl IntoIterator<Item = usize>,
    identity: usize,
    phase: &'static str,
) -> Result<(), ProjectionIdentityTraceError> {
    if inputs
        .into_iter()
        .any(|input| input != 0 && input >= target)
    {
        return Err(ProjectionIdentityTraceError::NonTopological {
            identity,
            phase,
            column: target,
        });
    }
    Ok(())
}

fn validate_row(
    source: &R1csSnapshot,
    row: usize,
    a: &Lc,
    b: &Lc,
    c: &Lc,
    identity: usize,
    phase: &'static str,
) -> Result<(), ProjectionIdentityTraceError> {
    if row >= source.rows()
        || source.a_row(row) != normalize_lc(a)
        || source.b_row(row) != normalize_lc(b)
        || source.c_row(row) != normalize_lc(c)
    {
        return Err(ProjectionIdentityTraceError::RowMismatch { identity, phase, row });
    }
    Ok(())
}

fn normalize_lc(lc: &Lc) -> Vec<(usize, F)> {
    let mut terms = BTreeMap::<usize, F>::new();
    for &(column, coefficient) in &lc.terms {
        *terms.entry(column).or_insert(F::ZERO) += coefficient;
    }
    *terms.entry(0).or_insert(F::ZERO) += lc.constant;
    terms
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn claim_range(
    owners: &mut [Option<usize>],
    range: std::ops::Range<usize>,
    identity: usize,
    kind: &'static str,
) -> Result<(), ProjectionIdentityTraceError> {
    if range.is_empty() || range.end > owners.len() {
        return Err(ProjectionIdentityTraceError::Index { identity, phase: kind });
    }
    for index in range {
        claim_index(owners, index, identity, kind)?;
    }
    Ok(())
}

fn claim_index(
    owners: &mut [Option<usize>],
    index: usize,
    identity: usize,
    kind: &'static str,
) -> Result<(), ProjectionIdentityTraceError> {
    let slot = owners
        .get_mut(index)
        .ok_or(ProjectionIdentityTraceError::Index { identity, phase: kind })?;
    if let Some(previous) = slot.replace(identity) {
        return Err(ProjectionIdentityTraceError::Overlap {
            identity,
            previous,
            kind,
            index,
        });
    }
    Ok(())
}

fn validate_no_escape(
    source: &R1csSnapshot,
    identities: &[ProjectionIdentityTraceEntry],
    column_owner: &[Option<usize>],
) -> Result<(), ProjectionIdentityTraceError> {
    for row in 0..source.rows() {
        for &(column, _) in source
            .a_row(row)
            .iter()
            .chain(source.b_row(row))
            .chain(source.c_row(row))
        {
            let Some(owner) = column_owner.get(column).and_then(|owner| *owner) else {
                continue;
            };
            if !identities[owner].source_rows.contains(&row) {
                return Err(ProjectionIdentityTraceError::Escape { owner, column, row });
            }
        }
    }
    Ok(())
}

fn k_from_columns(columns: [usize; 2]) -> KLc {
    KLc {
        c0: Lc::from_var(column_var(columns[0])),
        c1: Lc::from_var(column_var(columns[1])),
    }
}

fn column_var(column: usize) -> Var {
    Var::from_column_for_trace(column)
}
