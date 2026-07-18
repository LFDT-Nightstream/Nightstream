//! Exact active PiRLC `y_zcol` projection-identity ownership.
//!
//! Owns: the complete source-R1CS tree for the shared beta/rho work and both
//! delayed-NC coefficient-limb identities. Every stage-owned interval is
//! matched to one retained arithmetic trace and replayed against the exact
//! sparse A/B/C rows.
//!
//! Does not own: PiCCS source truth, Fiat-Shamir authority, the bad-root
//! probability bound, SIS/Poseidon2 binding, PiDEC child authority, encoded
//! lowering, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the semantic boundary supplied by the NIFS emitter
//! names the actual parent, quotient, and beta columns. This module proves
//! that those columns participate in the exact one-point identity; it does
//! not prove that beta was sampled after authoritative commitments.
//!
//! | Protocol → phase → family | Equation | Fixed multiplicity | Source rows |
//! |---|---|---:|---:|
//! | `nifs.pi_rlc.verify.projection_shared.beta_ladder` | globally shared: `p[0]=1`; `p[j]=p[j-1]·beta` | 54 K products | 272 |
//! | `nifs.pi_rlc.verify.projection_shared.rho_evaluations` | globally shared: `rho_s(beta)=Σ_j rho_s[j] beta^j` | 15 evaluations | 1,620 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb0` | low-limb `input_s(beta)` | 15 | 1,620 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.k_products.rho_times_input.limb0` | low-limb `rho_s(beta)·input_s(beta)` | 15 | 75 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0` | low-limb `parent(beta)` | one | 108 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.quotient.limb0` | low-limb `q(beta)` | one | 106 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.k_products.quotient_times_phi.limb0` | low-limb `q(beta)·(beta^54+beta^27+1)` | one | 5 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.final_limb_checks.limb0` | low-limb final identity | two base-field rows | 2 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb1` | high-limb `input_s(beta)` | 15 | 1,620 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.k_products.rho_times_input.limb1` | high-limb `rho_s(beta)·input_s(beta)` | 15 | 75 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1` | high-limb `parent(beta)` | one | 108 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.quotient.limb1` | high-limb `q(beta)` | one | 106 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.k_products.quotient_times_phi.limb1` | high-limb `q(beta)·(beta^54+beta^27+1)` | one | 5 |
//! | `nifs.pi_rlc.verify.identities.y_zcol.final_limb_checks.limb1` | high-limb final identity | two base-field rows | 2 |

use std::collections::HashSet;
use std::ops::Range;

use neo_math::ring::{D, PHI_MID_DEGREE};
use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{PolynomialEvaluationTrace, ProductSumBatchTrace};
use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::{Lc, PiRlcYZcolBoundaryAudit};
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::reductions::pi_ccs_output_message::Profile;
use crate::paper::reductions::pi_rlc_circuit::stage;

use super::super::invalid;
use super::certificate::{self, PiRlcYZcolKMulAudit, PiRlcYZcolPolynomialEvaluationAudit};
use super::rows::{same_lc, PiRlcYZcolProjectionRowAudit, SelectedR1csRows};
use super::PiCcsOutputYZcolProjectionInputAudit;

const LIMBS: usize = 2;
const K_MUL_ROWS: usize = 5;
const BETA_LADDER_ROWS: usize = 2 + D * K_MUL_ROWS;
const FULL_EVALUATION_ROWS: usize = 2 * D;
const QUOTIENT_EVALUATION_ROWS: usize = 2 * PROJECTION_QUOTIENT_LEN;
const FINAL_ROWS: usize = 2;

/// Shared one-point work consumed by both coefficient-limb identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionSharedAudit {
    beta_ladder_rows: Range<usize>,
    rho_evaluation_rows: Range<usize>,
    beta_columns: [usize; 2],
    power_columns: Vec<[usize; 2]>,
    rho_columns: Vec<Vec<usize>>,
    rho_evaluation_outputs: Vec<[usize; 2]>,
    beta_products: Vec<PiRlcYZcolKMulAudit>,
    rho_evaluations: Vec<PiRlcYZcolPolynomialEvaluationAudit>,
    allocated_column_count: usize,
}

impl PiRlcYZcolProjectionSharedAudit {
    pub fn beta_ladder_rows(&self) -> Range<usize> {
        self.beta_ladder_rows.clone()
    }

    pub fn rho_evaluation_rows(&self) -> Range<usize> {
        self.rho_evaluation_rows.clone()
    }

    pub fn beta_columns(&self) -> [usize; 2] {
        self.beta_columns
    }

    pub fn power_columns(&self) -> &[[usize; 2]] {
        &self.power_columns
    }

    pub fn rho_columns(&self) -> &[Vec<usize>] {
        &self.rho_columns
    }

    pub fn rho_evaluation_outputs(&self) -> &[[usize; 2]] {
        &self.rho_evaluation_outputs
    }

    pub fn beta_products(&self) -> &[PiRlcYZcolKMulAudit] {
        &self.beta_products
    }

    pub fn rho_evaluations(&self) -> &[PiRlcYZcolPolynomialEvaluationAudit] {
        &self.rho_evaluations
    }

    pub fn row_count(&self) -> usize {
        self.beta_ladder_rows.len() + self.rho_evaluation_rows.len()
    }

    pub fn allocated_column_count(&self) -> usize {
        self.allocated_column_count
    }
}

/// One complete base-field coefficient-limb identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionLimbAudit {
    rows: Range<usize>,
    input_evaluation_rows: Vec<Range<usize>>,
    rho_product_rows: Vec<Range<usize>>,
    output_evaluation_rows: Range<usize>,
    quotient_evaluation_rows: Range<usize>,
    quotient_phi_rows: Range<usize>,
    final_rows: Range<usize>,
    parent_columns: Vec<usize>,
    quotient_columns: Vec<usize>,
    input_evaluations: Vec<PiRlcYZcolPolynomialEvaluationAudit>,
    rho_products: Vec<PiRlcYZcolKMulAudit>,
    parent_evaluation: PiRlcYZcolPolynomialEvaluationAudit,
    quotient_evaluation: PiRlcYZcolPolynomialEvaluationAudit,
    quotient_phi_product: PiRlcYZcolKMulAudit,
    allocated_column_count: usize,
}

impl PiRlcYZcolProjectionLimbAudit {
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn input_evaluation_rows(&self) -> &[Range<usize>] {
        &self.input_evaluation_rows
    }

    pub fn rho_product_rows(&self) -> &[Range<usize>] {
        &self.rho_product_rows
    }

    pub fn output_evaluation_rows(&self) -> Range<usize> {
        self.output_evaluation_rows.clone()
    }

    pub fn quotient_evaluation_rows(&self) -> Range<usize> {
        self.quotient_evaluation_rows.clone()
    }

    pub fn quotient_phi_rows(&self) -> Range<usize> {
        self.quotient_phi_rows.clone()
    }

    pub fn final_rows(&self) -> Range<usize> {
        self.final_rows.clone()
    }

    pub fn parent_columns(&self) -> &[usize] {
        &self.parent_columns
    }

    pub fn quotient_columns(&self) -> &[usize] {
        &self.quotient_columns
    }

    pub fn input_evaluations(&self) -> &[PiRlcYZcolPolynomialEvaluationAudit] {
        &self.input_evaluations
    }

    pub fn rho_products(&self) -> &[PiRlcYZcolKMulAudit] {
        &self.rho_products
    }

    pub fn parent_evaluation(&self) -> &PiRlcYZcolPolynomialEvaluationAudit {
        &self.parent_evaluation
    }

    pub fn quotient_evaluation(&self) -> &PiRlcYZcolPolynomialEvaluationAudit {
        &self.quotient_evaluation
    }

    pub fn quotient_phi_product(&self) -> &PiRlcYZcolKMulAudit {
        &self.quotient_phi_product
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn allocated_column_count(&self) -> usize {
        self.allocated_column_count
    }
}

/// Protocol → shared phase → two coefficient-limb identity tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionIdentityAudit {
    shared: PiRlcYZcolProjectionSharedAudit,
    limbs: [PiRlcYZcolProjectionLimbAudit; LIMBS],
    source_rows: Vec<PiRlcYZcolProjectionRowAudit>,
}

impl PiRlcYZcolProjectionIdentityAudit {
    pub fn shared(&self) -> &PiRlcYZcolProjectionSharedAudit {
        &self.shared
    }

    pub fn limb(&self, limb: usize) -> &PiRlcYZcolProjectionLimbAudit {
        &self.limbs[limb]
    }

    pub fn source_rows(&self) -> &[PiRlcYZcolProjectionRowAudit] {
        &self.source_rows
    }

    pub fn row_count(&self) -> usize {
        self.shared.row_count()
            + self
                .limbs
                .iter()
                .map(PiRlcYZcolProjectionLimbAudit::row_count)
                .sum::<usize>()
    }

    pub fn allocated_column_count(&self) -> usize {
        self.shared.allocated_column_count()
            + self
                .limbs
                .iter()
                .map(PiRlcYZcolProjectionLimbAudit::allocated_column_count)
                .sum::<usize>()
    }
}

struct LimbRanges {
    input_evaluations: Vec<Range<usize>>,
    rho_products: Vec<Range<usize>>,
    output_evaluation: Range<usize>,
    quotient_evaluation: Range<usize>,
    quotient_phi: Range<usize>,
    final_checks: Range<usize>,
}

struct KMulView {
    a: [Lc; 2],
    b: [Lc; 2],
    output: [usize; 2],
}

pub(super) fn recover(
    arm: &SparseR1cs,
    profile: Profile,
    inputs: &[Vec<PiCcsOutputYZcolProjectionInputAudit>; LIMBS],
    boundary: &PiRlcYZcolBoundaryAudit,
) -> Result<PiRlcYZcolProjectionIdentityAudit, R1csIvcError> {
    let beta_ladder_rows = unique_stage(arm, stage::PROJECTION_SHARED_BETA_LADDER, BETA_LADDER_ROWS)?;
    let rho_evaluation_rows = unique_stage(
        arm,
        stage::PROJECTION_SHARED_RHO_EVALUATIONS,
        profile.source_count() * FULL_EVALUATION_ROWS,
    )?;
    if beta_ladder_rows.end != rho_evaluation_rows.start {
        return Err(invalid(format!(
            "PiRLC shared beta/rho stages are not adjacent: {beta_ladder_rows:?}, {rho_evaluation_rows:?}"
        )));
    }
    let limb_ranges = [
        recover_limb_ranges(arm, profile, 0)?,
        recover_limb_ranges(arm, profile, 1)?,
    ];

    let mut selected_ranges = vec![beta_ladder_rows.clone(), rho_evaluation_rows.clone()];
    for ranges in &limb_ranges {
        selected_ranges.extend(ranges.input_evaluations.iter().cloned());
        selected_ranges.extend(ranges.rho_products.iter().cloned());
        selected_ranges.extend([
            ranges.output_evaluation.clone(),
            ranges.quotient_evaluation.clone(),
            ranges.quotient_phi.clone(),
            ranges.final_checks.clone(),
        ]);
    }
    validate_trace_ownership(
        arm,
        &selected_ranges,
        &beta_ladder_rows,
        &rho_evaluation_rows,
        &limb_ranges,
        profile,
    )?;
    let rows = SelectedR1csRows::recover(arm, &selected_ranges)?;

    let (power_columns, beta_columns, mut allocated_columns, beta_products) =
        recover_beta_ladder(arm, &rows, beta_ladder_rows.clone(), boundary)?;
    let (rho_columns, rho_evaluation_outputs, rho_allocated_columns, rho_evaluations) = recover_rho_evaluations(
        arm,
        &rows,
        rho_evaluation_rows.clone(),
        profile.source_count(),
        &power_columns,
    )?;
    allocated_columns.extend(rho_allocated_columns);
    let shared = PiRlcYZcolProjectionSharedAudit {
        beta_ladder_rows,
        rho_evaluation_rows,
        beta_columns,
        power_columns,
        rho_columns,
        rho_evaluation_outputs,
        beta_products,
        rho_evaluations,
        allocated_column_count: allocated_columns.len(),
    };
    let (limb0, limb0_columns) = recover_limb(arm, &rows, profile, 0, &limb_ranges[0], inputs, boundary, &shared)?;
    let (limb1, limb1_columns) = recover_limb(arm, &rows, profile, 1, &limb_ranges[1], inputs, boundary, &shared)?;
    allocated_columns.extend(limb0_columns);
    allocated_columns.extend(limb1_columns);
    validate_allocated_columns(arm, &allocated_columns)?;
    let limbs = [limb0, limb1];
    let source_rows = rows.into_indexed_rows();
    let audit = PiRlcYZcolProjectionIdentityAudit {
        shared,
        limbs,
        source_rows,
    };
    let expected = BETA_LADDER_ROWS
        + profile.source_count() * FULL_EVALUATION_ROWS
        + LIMBS
            * (profile.source_count() * (FULL_EVALUATION_ROWS + K_MUL_ROWS)
                + FULL_EVALUATION_ROWS
                + QUOTIENT_EVALUATION_ROWS
                + K_MUL_ROWS
                + FINAL_ROWS);
    if audit.row_count() != expected {
        return Err(invalid(format!(
            "PiRLC y_zcol projection tree owns {} rows, expected {expected}",
            audit.row_count()
        )));
    }
    if audit.source_rows.len() != expected {
        return Err(invalid(format!(
            "PiRLC y_zcol projection certificate retains {} source rows, expected {expected}",
            audit.source_rows.len()
        )));
    }
    let expected_columns = expected - LIMBS * FINAL_ROWS;
    if audit.allocated_column_count() != expected_columns {
        return Err(invalid(format!(
            "PiRLC y_zcol projection tree owns {} allocated columns, expected {expected_columns}",
            audit.allocated_column_count()
        )));
    }
    Ok(audit)
}

fn validate_trace_ownership(
    arm: &SparseR1cs,
    selected_rows: &[Range<usize>],
    beta_ladder: &Range<usize>,
    rho_evaluations: &Range<usize>,
    limbs: &[LimbRanges; LIMBS],
    profile: Profile,
) -> Result<(), R1csIvcError> {
    let mut expected_polynomials = HashSet::<(usize, usize)>::new();
    for source in 0..profile.source_count() {
        let start = rho_evaluations.start + source * FULL_EVALUATION_ROWS;
        expected_polynomials.insert((start, start + FULL_EVALUATION_ROWS));
    }
    for limb in limbs {
        expected_polynomials.extend(
            limb.input_evaluations
                .iter()
                .map(|rows| (rows.start, rows.end)),
        );
        expected_polynomials.insert((limb.output_evaluation.start, limb.output_evaluation.end));
        expected_polynomials.insert((limb.quotient_evaluation.start, limb.quotient_evaluation.end));
    }

    let mut expected_products = HashSet::<(usize, usize)>::new();
    for exponent in 0..D {
        let start = beta_ladder.start + 2 + exponent * K_MUL_ROWS;
        expected_products.insert((start, start + K_MUL_ROWS));
    }
    for limb in limbs {
        expected_products.extend(limb.rho_products.iter().map(|rows| (rows.start, rows.end)));
        expected_products.insert((limb.quotient_phi.start, limb.quotient_phi.end));
    }

    validate_trace_ranges(
        "polynomial evaluation",
        arm.polynomial_evaluation_traces()
            .iter()
            .map(|trace| trace.row_start..trace.row_end),
        selected_rows,
        expected_polynomials,
    )?;
    validate_trace_ranges(
        "product sum",
        arm.product_sum_batch_traces()
            .iter()
            .map(|trace| trace.row_start..trace.row_end),
        selected_rows,
        expected_products,
    )
}

fn validate_trace_ranges(
    family: &str,
    traces: impl Iterator<Item = Range<usize>>,
    selected_rows: &[Range<usize>],
    mut expected: HashSet<(usize, usize)>,
) -> Result<(), R1csIvcError> {
    let mut prior_start = None;
    for trace in traces.filter(|trace| selected_rows.iter().any(|rows| ranges_overlap(trace, rows))) {
        if prior_start.is_some_and(|prior| trace.start <= prior) {
            return Err(invalid(format!(
                "PiRLC y_zcol {family} traces are duplicated or out of source-row order"
            )));
        }
        prior_start = Some(trace.start);
        if !expected.remove(&(trace.start, trace.end)) {
            return Err(invalid(format!(
                "PiRLC y_zcol has an unexpected or crossing {family} trace at {trace:?}"
            )));
        }
    }
    if !expected.is_empty() {
        return Err(invalid(format!(
            "PiRLC y_zcol omits {} expected {family} traces",
            expected.len()
        )));
    }
    Ok(())
}

fn validate_allocated_columns(arm: &SparseR1cs, columns: &[usize]) -> Result<(), R1csIvcError> {
    if columns.iter().any(|&column| column == 0 || column >= arm.m) {
        return Err(invalid(
            "PiRLC y_zcol projection allocation includes the constant wire or an out-of-bounds column",
        ));
    }
    if columns.iter().copied().collect::<HashSet<_>>().len() != columns.len() {
        return Err(invalid(
            "PiRLC y_zcol projection leaves do not own disjoint fresh columns",
        ));
    }
    Ok(())
}

fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn recover_limb_ranges(arm: &SparseR1cs, profile: Profile, limb: usize) -> Result<LimbRanges, R1csIvcError> {
    let paths = limb_paths(limb);
    let input_evaluations = repeated_stages(
        arm,
        paths.input_evaluations,
        profile.source_count(),
        FULL_EVALUATION_ROWS,
    )?;
    let rho_products = repeated_stages(arm, paths.rho_products, profile.source_count(), K_MUL_ROWS)?;
    let output_evaluation = unique_stage(arm, paths.output_evaluation, FULL_EVALUATION_ROWS)?;
    let quotient_evaluation = unique_stage(arm, paths.quotient_evaluation, QUOTIENT_EVALUATION_ROWS)?;
    let quotient_phi = unique_stage(arm, paths.quotient_phi, K_MUL_ROWS)?;
    let final_checks = unique_stage(arm, paths.final_checks, FINAL_ROWS)?;

    let mut ordered = Vec::with_capacity(2 * profile.source_count() + 4);
    for source in 0..profile.source_count() {
        ordered.push(input_evaluations[source].clone());
        ordered.push(rho_products[source].clone());
    }
    ordered.extend([
        output_evaluation.clone(),
        quotient_evaluation.clone(),
        quotient_phi.clone(),
        final_checks.clone(),
    ]);
    for pair in ordered.windows(2) {
        if pair[0].end != pair[1].start {
            return Err(invalid(format!(
                "PiRLC y_zcol limb {limb} has a row gap between {:?} and {:?}",
                pair[0], pair[1]
            )));
        }
    }
    Ok(LimbRanges {
        input_evaluations,
        rho_products,
        output_evaluation,
        quotient_evaluation,
        quotient_phi,
        final_checks,
    })
}

fn recover_beta_ladder(
    arm: &SparseR1cs,
    rows: &SelectedR1csRows,
    stage_rows: Range<usize>,
    boundary: &PiRlcYZcolBoundaryAudit,
) -> Result<(Vec<[usize; 2]>, [usize; 2], Vec<usize>, Vec<PiRlcYZcolKMulAudit>), R1csIvcError> {
    let beta = boundary.beta_columns();
    let mut powers = Vec::with_capacity(D + 1);
    let mut allocated_columns = Vec::with_capacity(BETA_LADDER_ROWS);
    let mut products = Vec::with_capacity(D);
    for exponent in 1..=D {
        let start = stage_rows.start + 2 + (exponent - 1) * K_MUL_ROWS;
        let product_rows = start..start + K_MUL_ROWS;
        let trace = unique_product_trace(arm, &product_rows)?;
        let view = validate_k_mul(rows, trace, &format!("PiRLC beta ladder exponent {exponent}"))?;
        let previous = if exponent == 1 {
            [
                single_var(&view.a[0], "beta^0 c0")?,
                single_var(&view.a[1], "beta^0 c1")?,
            ]
        } else {
            powers[exponent - 1]
        };
        expect_pair(
            &view.a,
            previous,
            &format!("PiRLC beta ladder exponent {exponent} input"),
        )?;
        expect_pair(&view.b, beta, &format!("PiRLC beta ladder exponent {exponent} beta"))?;
        if exponent == 1 {
            powers.push(previous);
            allocated_columns.extend(previous);
            let mut pin_one = variable(previous[0]);
            pin_one.add_constant(-F::ONE);
            rows.expect(stage_rows.start, &pin_one, &one(), &zero(), "PiRLC beta^0 c0 pin")?;
            rows.expect(
                stage_rows.start + 1,
                &variable(previous[1]),
                &one(),
                &zero(),
                "PiRLC beta^0 c1 pin",
            )?;
        }
        allocated_columns.extend(trace.allocated_columns.iter().copied());
        products.push(certificate::k_mul(trace));
        powers.push(view.output);
    }
    if powers.len() != D + 1 || stage_rows.end != stage_rows.start + BETA_LADDER_ROWS {
        return Err(invalid("PiRLC beta ladder does not own exactly powers 0 through 54"));
    }
    Ok((powers, beta, allocated_columns, products))
}

fn recover_rho_evaluations(
    arm: &SparseR1cs,
    rows: &SelectedR1csRows,
    stage_rows: Range<usize>,
    source_count: usize,
    powers: &[[usize; 2]],
) -> Result<
    (
        Vec<Vec<usize>>,
        Vec<[usize; 2]>,
        Vec<usize>,
        Vec<PiRlcYZcolPolynomialEvaluationAudit>,
    ),
    R1csIvcError,
> {
    let mut rho_columns = Vec::with_capacity(source_count);
    let mut outputs = Vec::with_capacity(source_count);
    let mut allocated_columns = Vec::with_capacity(source_count * FULL_EVALUATION_ROWS);
    let mut evaluations = Vec::with_capacity(source_count);
    for source in 0..source_count {
        let start = stage_rows.start + source * FULL_EVALUATION_ROWS;
        let evaluation_rows = start..start + FULL_EVALUATION_ROWS;
        let trace = unique_polynomial_trace(arm, &evaluation_rows)?;
        validate_polynomial(
            rows,
            trace,
            None,
            powers,
            &format!("PiRLC rho evaluation source {source}"),
        )?;
        rho_columns.push(trace.coefficient_cols.clone());
        outputs.push(trace.output_cols);
        allocated_columns.extend(trace.allocated_columns.iter().copied());
        evaluations.push(certificate::polynomial(trace));
    }
    Ok((rho_columns, outputs, allocated_columns, evaluations))
}

#[allow(clippy::too_many_arguments)]
fn recover_limb(
    arm: &SparseR1cs,
    rows: &SelectedR1csRows,
    profile: Profile,
    limb: usize,
    ranges: &LimbRanges,
    inputs: &[Vec<PiCcsOutputYZcolProjectionInputAudit>; LIMBS],
    boundary: &PiRlcYZcolBoundaryAudit,
    shared: &PiRlcYZcolProjectionSharedAudit,
) -> Result<(PiRlcYZcolProjectionLimbAudit, Vec<usize>), R1csIvcError> {
    if inputs[limb].len() != profile.source_count() {
        return Err(invalid(format!(
            "PiRLC y_zcol limb {limb} has {} typed inputs, expected {}",
            inputs[limb].len(),
            profile.source_count()
        )));
    }
    let mut product_outputs = Vec::with_capacity(profile.source_count());
    let mut input_evaluations = Vec::with_capacity(profile.source_count());
    let mut rho_products = Vec::with_capacity(profile.source_count());
    let mut allocated_columns = Vec::with_capacity(
        profile.source_count() * (FULL_EVALUATION_ROWS + K_MUL_ROWS)
            + FULL_EVALUATION_ROWS
            + QUOTIENT_EVALUATION_ROWS
            + K_MUL_ROWS,
    );
    for source in 0..profile.source_count() {
        let evaluation = unique_polynomial_trace(arm, &ranges.input_evaluations[source])?;
        validate_polynomial(
            rows,
            evaluation,
            Some(inputs[limb][source].coefficient_columns()),
            &shared.power_columns,
            &format!("PiRLC y_zcol limb {limb} input {source}"),
        )?;
        allocated_columns.extend(evaluation.allocated_columns.iter().copied());
        input_evaluations.push(certificate::polynomial(evaluation));
        let product = unique_product_trace(arm, &ranges.rho_products[source])?;
        let view = validate_k_mul(
            rows,
            product,
            &format!("PiRLC y_zcol limb {limb} rho/input product {source}"),
        )?;
        expect_pair(
            &view.a,
            shared.rho_evaluation_outputs[source],
            &format!("PiRLC y_zcol limb {limb} rho evaluation {source}"),
        )?;
        expect_pair(
            &view.b,
            evaluation.output_cols,
            &format!("PiRLC y_zcol limb {limb} input evaluation {source}"),
        )?;
        allocated_columns.extend(product.allocated_columns.iter().copied());
        rho_products.push(certificate::k_mul(product));
        product_outputs.push(view.output);
    }

    let parent_columns = boundary.parent_columns(limb);
    let output = unique_polynomial_trace(arm, &ranges.output_evaluation)?;
    validate_polynomial(
        rows,
        output,
        Some(parent_columns),
        &shared.power_columns,
        &format!("PiRLC y_zcol limb {limb} parent evaluation"),
    )?;
    allocated_columns.extend(output.allocated_columns.iter().copied());
    let quotient_columns = boundary.quotient_columns(limb);
    let quotient = unique_polynomial_trace(arm, &ranges.quotient_evaluation)?;
    validate_polynomial(
        rows,
        quotient,
        Some(quotient_columns),
        &shared.power_columns,
        &format!("PiRLC y_zcol limb {limb} quotient evaluation"),
    )?;
    allocated_columns.extend(quotient.allocated_columns.iter().copied());

    let quotient_phi = unique_product_trace(arm, &ranges.quotient_phi)?;
    let quotient_phi_view = validate_k_mul(
        rows,
        quotient_phi,
        &format!("PiRLC y_zcol limb {limb} quotient/Phi product"),
    )?;
    expect_pair(
        &quotient_phi_view.a,
        quotient.output_cols,
        &format!("PiRLC y_zcol limb {limb} quotient evaluation product input"),
    )?;
    let phi = phi_lcs(&shared.power_columns);
    if !same_lc(&quotient_phi_view.b[0], &phi[0]) || !same_lc(&quotient_phi_view.b[1], &phi[1]) {
        return Err(invalid(format!(
            "PiRLC y_zcol limb {limb} does not multiply by beta^54 + beta^27 + 1"
        )));
    }
    allocated_columns.extend(quotient_phi.allocated_columns.iter().copied());

    for coefficient_limb in 0..2 {
        let mut equation = zero();
        for product in &product_outputs {
            equation.add_term_column(product[coefficient_limb], F::ONE);
        }
        equation.add_term_column(quotient_phi_view.output[coefficient_limb], -F::ONE);
        equation.add_term_column(output.output_cols[coefficient_limb], -F::ONE);
        rows.expect(
            ranges.final_checks.start + coefficient_limb,
            &equation,
            &one(),
            &zero(),
            &format!("PiRLC y_zcol limb {limb} final coefficient {coefficient_limb}"),
        )?;
    }

    let identity_rows = ranges.input_evaluations[0].start..ranges.final_checks.end;
    Ok((
        PiRlcYZcolProjectionLimbAudit {
            rows: identity_rows,
            input_evaluation_rows: ranges.input_evaluations.clone(),
            rho_product_rows: ranges.rho_products.clone(),
            output_evaluation_rows: ranges.output_evaluation.clone(),
            quotient_evaluation_rows: ranges.quotient_evaluation.clone(),
            quotient_phi_rows: ranges.quotient_phi.clone(),
            final_rows: ranges.final_checks.clone(),
            parent_columns: parent_columns.to_vec(),
            quotient_columns: quotient_columns.to_vec(),
            input_evaluations,
            rho_products,
            parent_evaluation: certificate::polynomial(output),
            quotient_evaluation: certificate::polynomial(quotient),
            quotient_phi_product: certificate::k_mul(quotient_phi),
            allocated_column_count: allocated_columns.len(),
        },
        allocated_columns,
    ))
}

fn validate_polynomial(
    rows: &SelectedR1csRows,
    trace: &PolynomialEvaluationTrace,
    expected_coefficients: Option<&[usize]>,
    powers: &[[usize; 2]],
    owner: &str,
) -> Result<(), R1csIvcError> {
    let coefficient_count = trace.coefficient_cols.len();
    if coefficient_count == 0
        || trace.power_cols.len() != coefficient_count
        || trace.allocated_columns.len() != 2 * coefficient_count
        || trace.row_end - trace.row_start != 2 * coefficient_count
        || powers.len() < coefficient_count
        || trace.power_cols != powers[..coefficient_count]
    {
        return Err(invalid(format!("{owner} has invalid polynomial-evaluation geometry")));
    }
    if trace
        .allocated_columns
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != trace.allocated_columns.len()
    {
        return Err(invalid(format!("{owner} reuses an allocated polynomial column")));
    }
    if expected_coefficients.is_some_and(|expected| trace.coefficient_cols != expected) {
        return Err(invalid(format!("{owner} consumes different coefficient columns")));
    }
    let output_start = 2 * (coefficient_count - 1);
    if trace.output_cols
        != [
            trace.allocated_columns[output_start],
            trace.allocated_columns[output_start + 1],
        ]
    {
        return Err(invalid(format!(
            "{owner} output columns do not terminate its allocation"
        )));
    }
    for coefficient in 1..coefficient_count {
        for limb in 0..2 {
            let offset = 2 * (coefficient - 1) + limb;
            rows.expect(
                trace.row_start + offset,
                &variable(trace.coefficient_cols[coefficient]),
                &variable(trace.power_cols[coefficient][limb]),
                &variable(trace.allocated_columns[offset]),
                &format!("{owner} product coefficient {coefficient} limb {limb}"),
            )?;
        }
    }
    let mut c0 = variable(trace.output_cols[0]);
    c0.add_term_column(trace.coefficient_cols[0], -F::ONE);
    let mut c1 = variable(trace.output_cols[1]);
    for coefficient in 1..coefficient_count {
        c0.add_term_column(trace.allocated_columns[2 * (coefficient - 1)], -F::ONE);
        c1.add_term_column(trace.allocated_columns[2 * (coefficient - 1) + 1], -F::ONE);
    }
    rows.expect(trace.row_end - 2, &c0, &one(), &zero(), &format!("{owner} c0 sum"))?;
    rows.expect(trace.row_end - 1, &c1, &one(), &zero(), &format!("{owner} c1 sum"))?;
    Ok(())
}

fn validate_k_mul(
    rows: &SelectedR1csRows,
    trace: &ProductSumBatchTrace,
    owner: &str,
) -> Result<KMulView, R1csIvcError> {
    if trace.row_end - trace.row_start != K_MUL_ROWS
        || trace.allocated_columns.len() != K_MUL_ROWS
        || trace.identities.len() != 2
        || trace.identities[0].factors.len() != 2
        || trace.identities[1].factors.len() != 2
    {
        return Err(invalid(format!("{owner} is not one exact five-row K multiplication")));
    }
    if trace
        .allocated_columns
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != K_MUL_ROWS
    {
        return Err(invalid(format!("{owner} reuses an allocated K-multiplication column")));
    }
    let first = &trace.identities[0];
    let second = &trace.identities[1];
    let w = <Fq as BinomiallyExtendable<2>>::W;
    if first.factors[0].coefficient != F::ONE
        || first.factors[1].coefficient != w
        || second
            .factors
            .iter()
            .any(|factor| factor.coefficient != F::ONE)
    {
        return Err(invalid(format!("{owner} has the wrong extension-field coefficients")));
    }
    let a = [first.factors[0].left.clone(), first.factors[1].left.clone()];
    let b = [first.factors[0].right.clone(), first.factors[1].right.clone()];
    if !same_lc(&second.factors[0].left, &a[0])
        || !same_lc(&second.factors[0].right, &b[1])
        || !same_lc(&second.factors[1].left, &a[1])
        || !same_lc(&second.factors[1].right, &b[0])
    {
        return Err(invalid(format!(
            "{owner} product-sum trace does not describe one K product"
        )));
    }
    let output = [single_var(&first.result, owner)?, single_var(&second.result, owner)?];
    if output != [trace.allocated_columns[3], trace.allocated_columns[4]] || trace.retained_columns != output {
        return Err(invalid(format!(
            "{owner} retained outputs do not match its exact allocation"
        )));
    }
    let [p, q, r, out0, out1] = trace.allocated_columns.as_slice() else {
        unreachable!("five-column K multiplication checked above")
    };
    rows.expect(trace.row_start, &a[0], &b[0], &variable(*p), &format!("{owner} p"))?;
    rows.expect(trace.row_start + 1, &a[1], &b[1], &variable(*q), &format!("{owner} q"))?;
    let sum_a = a[0].clone().add_scaled(&a[1], F::ONE);
    let sum_b = b[0].clone().add_scaled(&b[1], F::ONE);
    rows.expect(
        trace.row_start + 2,
        &sum_a,
        &sum_b,
        &variable(*r),
        &format!("{owner} r"),
    )?;
    let mut c0 = variable(*out0);
    c0.add_term_column(*p, -F::ONE);
    c0.add_term_column(*q, -w);
    rows.expect(trace.row_start + 3, &c0, &one(), &zero(), &format!("{owner} c0"))?;
    let mut c1 = variable(*out1);
    c1.add_term_column(*r, -F::ONE);
    c1.add_term_column(*p, F::ONE);
    c1.add_term_column(*q, F::ONE);
    rows.expect(trace.row_start + 4, &c1, &one(), &zero(), &format!("{owner} c1"))?;
    Ok(KMulView { a, b, output })
}

fn unique_polynomial_trace<'a>(
    arm: &'a SparseR1cs,
    rows: &Range<usize>,
) -> Result<&'a PolynomialEvaluationTrace, R1csIvcError> {
    let matching = arm
        .polynomial_evaluation_traces()
        .iter()
        .filter(|trace| trace.row_start == rows.start && trace.row_end == rows.end)
        .collect::<Vec<_>>();
    let [trace] = matching.as_slice() else {
        return Err(invalid(format!(
            "PiRLC y_zcol rows {rows:?} match {} polynomial-evaluation traces, expected one",
            matching.len()
        )));
    };
    Ok(*trace)
}

fn unique_product_trace<'a>(
    arm: &'a SparseR1cs,
    rows: &Range<usize>,
) -> Result<&'a ProductSumBatchTrace, R1csIvcError> {
    let matching = arm
        .product_sum_batch_traces()
        .iter()
        .filter(|trace| trace.row_start == rows.start && trace.row_end == rows.end)
        .collect::<Vec<_>>();
    let [trace] = matching.as_slice() else {
        return Err(invalid(format!(
            "PiRLC y_zcol rows {rows:?} match {} product-sum traces, expected one",
            matching.len()
        )));
    };
    Ok(*trace)
}

fn unique_stage(arm: &SparseR1cs, path: &'static str, expected_rows: usize) -> Result<Range<usize>, R1csIvcError> {
    let ranges = nonempty_stage_ranges(arm, path);
    validate_stage_occurrences(arm, path, &ranges)?;
    let [rows] = ranges.as_slice() else {
        return Err(invalid(format!(
            "PiRLC y_zcol stage `{path}` has {} nonempty ranges, expected one",
            ranges.len()
        )));
    };
    if rows.len() != expected_rows {
        return Err(invalid(format!(
            "PiRLC y_zcol stage `{path}` owns {} rows, expected {expected_rows}",
            rows.len()
        )));
    }
    Ok(rows.clone())
}

fn repeated_stages(
    arm: &SparseR1cs,
    path: &'static str,
    expected_count: usize,
    expected_rows: usize,
) -> Result<Vec<Range<usize>>, R1csIvcError> {
    let ranges = nonempty_stage_ranges(arm, path);
    validate_stage_occurrences(arm, path, &ranges)?;
    if ranges.len() != expected_count || ranges.iter().any(|range| range.len() != expected_rows) {
        return Err(invalid(format!(
            "PiRLC y_zcol stage `{path}` has {} ranges with lengths {:?}, expected {expected_count} ranges of {expected_rows}",
            ranges.len(),
            ranges.iter().map(Range::len).collect::<Vec<_>>()
        )));
    }
    Ok(ranges)
}

fn nonempty_stage_ranges(arm: &SparseR1cs, path: &'static str) -> Vec<Range<usize>> {
    arm.physical_stage_ranges()
        .iter()
        .filter(|range| range.path() == path && range.row_start() < range.row_end())
        .map(|range| range.rows())
        .collect()
}

fn validate_stage_occurrences(
    arm: &SparseR1cs,
    path: &'static str,
    nonempty: &[Range<usize>],
) -> Result<(), R1csIvcError> {
    let matching = arm
        .physical_stage_ranges()
        .iter()
        .filter(|range| range.path() == path)
        .map(|range| range.rows())
        .collect::<Vec<_>>();
    let empty = matching
        .iter()
        .filter(|range| range.is_empty())
        .collect::<Vec<_>>();
    let expected_empty = usize::from(stage::IDENTITY_PHASE_NODES.contains(&path));
    if matching.len() != nonempty.len() + expected_empty || empty.len() != expected_empty {
        return Err(invalid(format!(
            "PiRLC y_zcol stage `{path}` has {} empty organizational occurrences, expected {expected_empty}",
            empty.len()
        )));
    }
    if let ([organizational], Some(first)) = (empty.as_slice(), nonempty.first()) {
        if organizational.start > first.start {
            return Err(invalid(format!(
                "PiRLC y_zcol stage `{path}` places its zero-cost organizational node after emitted leaves"
            )));
        }
    }
    Ok(())
}

struct LimbPaths {
    input_evaluations: &'static str,
    rho_products: &'static str,
    output_evaluation: &'static str,
    quotient_evaluation: &'static str,
    quotient_phi: &'static str,
    final_checks: &'static str,
}

fn limb_paths(limb: usize) -> LimbPaths {
    match limb {
        0 => LimbPaths {
            input_evaluations: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB0,
            rho_products: stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB0,
            output_evaluation: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB0,
            quotient_evaluation: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB0,
            quotient_phi: stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB0,
            final_checks: stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB0,
        },
        1 => LimbPaths {
            input_evaluations: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB1,
            rho_products: stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB1,
            output_evaluation: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB1,
            quotient_evaluation: stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB1,
            quotient_phi: stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB1,
            final_checks: stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB1,
        },
        _ => unreachable!("two coefficient limbs"),
    }
}

fn phi_lcs(powers: &[[usize; 2]]) -> [Lc; 2] {
    let mut c0 = zero();
    c0.add_term_column(powers[D][0], F::ONE);
    c0.add_term_column(powers[PHI_MID_DEGREE][0], F::ONE);
    c0.add_constant(F::ONE);
    let mut c1 = zero();
    c1.add_term_column(powers[D][1], F::ONE);
    c1.add_term_column(powers[PHI_MID_DEGREE][1], F::ONE);
    [c0, c1]
}

fn expect_pair(actual: &[Lc; 2], expected: [usize; 2], owner: &str) -> Result<(), R1csIvcError> {
    if !same_lc(&actual[0], &variable(expected[0])) || !same_lc(&actual[1], &variable(expected[1])) {
        return Err(invalid(format!("{owner} consumes different extension-field columns")));
    }
    Ok(())
}

fn single_var(lc: &Lc, owner: &str) -> Result<usize, R1csIvcError> {
    if lc.constant == F::ZERO && lc.terms.len() == 1 && lc.terms[0].1 == F::ONE {
        return Ok(lc.terms[0].0);
    }
    Err(invalid(format!("{owner} is not one exact witness column")))
}

fn variable(column: usize) -> Lc {
    Lc {
        terms: vec![(column, F::ONE)],
        constant: F::ZERO,
    }
}

fn one() -> Lc {
    variable(0)
}

fn zero() -> Lc {
    Lc::zero()
}

trait LcColumnExt {
    fn add_term_column(&mut self, column: usize, coefficient: F);
}

impl LcColumnExt for Lc {
    fn add_term_column(&mut self, column: usize, coefficient: F) {
        if coefficient != F::ZERO {
            self.terms.push((column, coefficient));
        }
    }
}
