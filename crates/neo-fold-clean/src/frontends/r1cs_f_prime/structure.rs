//! CCS structure for one `enc(F')` step that hosts an R1CS app circuit.
//!
//! Reuses every row the shared F' shell structure
//! ([`crate::frontends::f_prime::structure::build_f_prime_structure`])
//! emits (semantic Boolean rows, ring-action shell, state-out / public-x_out
//! digest bindings, selector, Poseidon transitions). On top of the
//! shell we append exactly `r1cs.n()` product rows — one per R1CS
//! constraint — that enforce
//! `(A_i · z_app) * (B_i · z_app) = (C_i · z_app)`, where each variable
//! `z_app[j]` is recomposed from its verifier-owned app-private slot in the
//! `app_private` region.

use neo_ccs::{sparse_r1cs_to_ccs, CcsMatrix};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::engine::r1cs_circuit::builder::{
    BalancedTernaryDecomposition, CanonicalU64Decomposition, CenteredUnitTrace, ColumnFamilyRange, Lc,
    PiDecStrictAudit, PolynomialEvaluationTrace, Poseidon2HashAudit, Poseidon2PermutationTrace, ProductSumBatchTrace,
    R1csBuilder, RowFamilyRange, ShiftedTernaryCanonicalTrace, SumcheckRoundAudit, Var,
};
use crate::engine::r1cs_circuit::PhysicalStageRange;
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::direct_ccs::R1cs;
use crate::frontends::f_prime::image::{FPrimeImageLayout, PoseidonPreimageLaneSource};
use crate::frontends::f_prime::structure::{
    app_variable_terms, emit_shell_rows, f_prime_lane_slots, AppVariableSlot, FPrimeStructure, MixedGateBuilder,
};
use crate::paper::relations::Structure;

/// Sparse R1CS shape for large app circuits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseR1cs {
    pub a: CcsMatrix<F>,
    pub b: CcsMatrix<F>,
    pub c: CcsMatrix<F>,
    pub n: usize,
    pub m: usize,
    pub m_in: usize,
    canonical_u64_decompositions: Vec<CanonicalU64Decomposition>,
    balanced_ternary_decompositions: Vec<BalancedTernaryDecomposition>,
    boolean_columns: Vec<usize>,
    boolean_constraint_rows: Vec<(usize, usize)>,
    centered_unit_columns: Vec<usize>,
    centered_unit_traces: Vec<CenteredUnitTrace>,
    shifted_ternary_canonical_traces: Vec<ShiftedTernaryCanonicalTrace>,
    equality_pairs: Vec<(usize, usize, usize)>,
    poseidon2_traces: Vec<Poseidon2PermutationTrace>,
    poseidon2_hash_audits: Vec<Poseidon2HashAudit>,
    polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
    product_sum_batch_traces: Vec<ProductSumBatchTrace>,
    row_family_ranges: Vec<RowFamilyRange>,
    sumcheck_round_audits: Vec<SumcheckRoundAudit>,
    pi_dec_strict_audits: Vec<PiDecStrictAudit>,
    column_family_ranges: Vec<ColumnFamilyRange>,
    physical_stage_ranges: Vec<PhysicalStageRange>,
}

impl SparseR1cs {
    pub fn new(
        a: CcsMatrix<F>,
        b: CcsMatrix<F>,
        c: CcsMatrix<F>,
        n: usize,
        m: usize,
        m_in: usize,
    ) -> Result<Self, FrontendError> {
        Self::new_with_canonical_u64_decompositions(
            a,
            b,
            c,
            n,
            m,
            m_in,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    pub(crate) fn new_with_canonical_u64_decompositions(
        a: CcsMatrix<F>,
        b: CcsMatrix<F>,
        c: CcsMatrix<F>,
        n: usize,
        m: usize,
        m_in: usize,
        canonical_u64_decompositions: Vec<CanonicalU64Decomposition>,
        balanced_ternary_decompositions: Vec<BalancedTernaryDecomposition>,
        boolean_columns: Vec<usize>,
        boolean_constraint_rows: Vec<(usize, usize)>,
        centered_unit_columns: Vec<usize>,
        centered_unit_traces: Vec<CenteredUnitTrace>,
        shifted_ternary_canonical_traces: Vec<ShiftedTernaryCanonicalTrace>,
        equality_pairs: Vec<(usize, usize, usize)>,
        poseidon2_traces: Vec<Poseidon2PermutationTrace>,
        poseidon2_hash_audits: Vec<Poseidon2HashAudit>,
        polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
        product_sum_batch_traces: Vec<ProductSumBatchTrace>,
        row_family_ranges: Vec<RowFamilyRange>,
        sumcheck_round_audits: Vec<SumcheckRoundAudit>,
        pi_dec_strict_audits: Vec<PiDecStrictAudit>,
        column_family_ranges: Vec<ColumnFamilyRange>,
        physical_stage_ranges: Vec<PhysicalStageRange>,
    ) -> Result<Self, FrontendError> {
        let out = Self {
            a,
            b,
            c,
            n,
            m,
            m_in,
            canonical_u64_decompositions,
            balanced_ternary_decompositions,
            boolean_columns,
            boolean_constraint_rows,
            centered_unit_columns,
            centered_unit_traces,
            shifted_ternary_canonical_traces,
            equality_pairs,
            poseidon2_traces,
            poseidon2_hash_audits,
            polynomial_evaluation_traces,
            product_sum_batch_traces,
            row_family_ranges,
            sumcheck_round_audits,
            pi_dec_strict_audits,
            column_family_ranges,
            physical_stage_ranges,
        };
        out.validate_shape()?;
        Ok(out)
    }

    pub(crate) fn canonical_u64_decompositions(&self) -> &[CanonicalU64Decomposition] {
        &self.canonical_u64_decompositions
    }

    pub(crate) fn balanced_ternary_decompositions(&self) -> &[BalancedTernaryDecomposition] {
        &self.balanced_ternary_decompositions
    }

    pub(crate) fn boolean_columns(&self) -> &[usize] {
        &self.boolean_columns
    }

    pub(crate) fn boolean_constraint_rows(&self) -> &[(usize, usize)] {
        &self.boolean_constraint_rows
    }

    pub(crate) fn centered_unit_columns(&self) -> &[usize] {
        &self.centered_unit_columns
    }

    pub(crate) fn centered_unit_traces(&self) -> &[CenteredUnitTrace] {
        &self.centered_unit_traces
    }

    pub(crate) fn shifted_ternary_canonical_traces(&self) -> &[ShiftedTernaryCanonicalTrace] {
        &self.shifted_ternary_canonical_traces
    }

    pub(crate) fn equality_pairs(&self) -> &[(usize, usize, usize)] {
        &self.equality_pairs
    }

    pub(crate) fn poseidon2_permutations(&self) -> usize {
        self.poseidon2_traces.len()
    }

    pub(crate) fn poseidon2_traces(&self) -> &[Poseidon2PermutationTrace] {
        &self.poseidon2_traces
    }

    pub(crate) fn poseidon2_hash_audits(&self) -> &[Poseidon2HashAudit] {
        &self.poseidon2_hash_audits
    }

    pub(crate) fn polynomial_evaluation_traces(&self) -> &[PolynomialEvaluationTrace] {
        &self.polynomial_evaluation_traces
    }

    pub(crate) fn product_sum_batch_traces(&self) -> &[ProductSumBatchTrace] {
        &self.product_sum_batch_traces
    }

    /// Assurance-only ownership ranges preserved from the field-R1CS builder.
    pub fn row_family_ranges(&self) -> &[RowFamilyRange] {
        &self.row_family_ranges
    }

    /// Exact generated claimed-chain SumCheck wire schedules.
    ///
    /// This assurance-only metadata names columns and row intervals; callers
    /// must compare the referenced sparse rows before assigning semantics.
    #[doc(hidden)]
    pub fn sumcheck_round_audits(&self) -> &[SumcheckRoundAudit] {
        &self.sumcheck_round_audits
    }

    /// Exact strict-PiDEC input schedules after public-column normalization.
    #[doc(hidden)]
    pub fn pi_dec_strict_audits(&self) -> &[PiDecStrictAudit] {
        &self.pi_dec_strict_audits
    }

    /// Exact sequential row intervals and normalized private-allocation spans
    /// preserved across column lowering.
    ///
    /// Paths are caller-supplied diagnostic labels. A consumer claiming an
    /// exact semantic ledger must reject an empty slice and separately check
    /// its expected root and complete path universe.
    pub fn physical_stage_ranges(&self) -> &[PhysicalStageRange] {
        &self.physical_stage_ranges
    }

    /// Nested allocation-family spans after public-prefix normalization.
    /// These are diagnostic boundaries, not semantic ownership or value
    /// authority; consumers must validate the expected family vocabulary.
    pub fn column_family_ranges(&self) -> &[ColumnFamilyRange] {
        &self.column_family_ranges
    }

    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        let (ar, ac) = (self.a.rows(), self.a.cols());
        let (br, bc) = (self.b.rows(), self.b.cols());
        let (cr, cc) = (self.c.rows(), self.c.cols());
        if ar != self.n || br != self.n || cr != self.n || ac != self.m || bc != self.m || cc != self.m {
            return Err(FrontendError::ShapeMismatch {
                a_rows: ar,
                a_cols: ac,
                b_rows: br,
                b_cols: bc,
                c_rows: cr,
                c_cols: cc,
            });
        }
        if self.m_in > self.m {
            return Err(FrontendError::PublicInputTooLarge {
                m_in: self.m_in,
                m: self.m,
            });
        }
        Ok(())
    }

    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        if z.len() != self.m {
            return Err(FrontendError::AssignmentLength {
                got: z.len(),
                expected: self.m,
            });
        }
        let mut az = vec![F::ZERO; self.n];
        let mut bz = vec![F::ZERO; self.n];
        let mut cz = vec![F::ZERO; self.n];
        self.a.add_mul_into(z, &mut az, self.n);
        self.b.add_mul_into(z, &mut bz, self.n);
        self.c.add_mul_into(z, &mut cz, self.n);
        for row in 0..self.n {
            if az[row] * bz[row] != cz[row] {
                return Err(FrontendError::Unsatisfied { row });
            }
        }
        Ok(())
    }

    pub fn to_structure(&self) -> Structure {
        sparse_r1cs_to_ccs(self.a.clone(), self.b.clone(), self.c.clone()).expect("valid sparse R1CS structure")
    }

    pub(crate) fn conservative_var_widths(&self) -> Vec<usize> {
        conservative_var_widths(r1cs_coeff_rows_sparse(self), self.n, self.m)
    }
}

/// R1CS representation accepted by the R1CS-F' compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum R1csShape {
    Dense(R1cs),
    Sparse(SparseR1cs),
}

impl From<R1cs> for R1csShape {
    fn from(value: R1cs) -> Self {
        Self::Dense(value)
    }
}

impl From<&R1cs> for R1csShape {
    fn from(value: &R1cs) -> Self {
        Self::Dense(value.clone())
    }
}

impl From<SparseR1cs> for R1csShape {
    fn from(value: SparseR1cs) -> Self {
        Self::Sparse(value)
    }
}

impl From<&SparseR1cs> for R1csShape {
    fn from(value: &SparseR1cs) -> Self {
        Self::Sparse(value.clone())
    }
}

impl From<&R1csShape> for R1csShape {
    fn from(value: &R1csShape) -> Self {
        value.clone()
    }
}

impl R1csShape {
    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        match self {
            Self::Dense(r1cs) => r1cs.validate_shape(),
            Self::Sparse(r1cs) => r1cs.validate_shape(),
        }
    }

    pub fn m(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.m(),
            Self::Sparse(r1cs) => r1cs.m,
        }
    }

    pub fn n(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.n(),
            Self::Sparse(r1cs) => r1cs.n,
        }
    }

    pub fn m_in(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.m_in,
            Self::Sparse(r1cs) => r1cs.m_in,
        }
    }

    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        match self {
            Self::Dense(r1cs) => r1cs.is_satisfied_by(z),
            Self::Sparse(r1cs) => r1cs.is_satisfied_by(z),
        }
    }

    /// Embed this application relation into an authoritative F' builder.
    ///
    /// Application columns are private to F'; even the application's own
    /// public inputs become witness data whose digest is surfaced through
    /// the F' state. When the verifier-owned plan uses the conventional
    /// constant lane, column zero is tied to one. Every original R1CS row is
    /// emitted unchanged.
    pub(crate) fn enforce_in_f_prime(
        &self,
        builder: &mut R1csBuilder,
        assignment: &[F],
        pin_constant_one: bool,
    ) -> Result<Vec<Var>, FrontendError> {
        if assignment.len() != self.m() {
            return Err(FrontendError::AssignmentLength {
                got: assignment.len(),
                expected: self.m(),
            });
        }

        let vars = builder.alloc_vec(assignment);
        if pin_constant_one {
            if let Some(&one) = vars.first() {
                builder.enforce_eq(&Lc::from_var(one), &Lc::from_var(Var::ONE));
            }
        }
        for (index, is_boolean) in self.boolean_constrained_variables().into_iter().enumerate() {
            if is_boolean {
                builder.record_boolean(vars[index]);
            }
        }

        let (a, b, c) = match self {
            Self::Dense(r1cs) => (
                dense_matrix_row_lcs(&r1cs.a, &vars),
                dense_matrix_row_lcs(&r1cs.b, &vars),
                dense_matrix_row_lcs(&r1cs.c, &vars),
            ),
            Self::Sparse(r1cs) => (
                sparse_matrix_row_lcs(&r1cs.a, &vars, r1cs.n),
                sparse_matrix_row_lcs(&r1cs.b, &vars, r1cs.n),
                sparse_matrix_row_lcs(&r1cs.c, &vars, r1cs.n),
            ),
        };
        for row in 0..self.n() {
            builder.enforce(&a[row], &b[row], &c[row]);
        }
        Ok(vars)
    }

    pub fn to_structure(&self) -> Structure {
        match self {
            Self::Dense(r1cs) => r1cs.to_structure(),
            Self::Sparse(r1cs) => r1cs.to_structure(),
        }
    }

    /// Conservative syntactic Boolean-variable detector.
    ///
    /// Seeds from explicit Boolean rows and exact fixed-zero rows, then
    /// propagates through exact copy rows and exact products of already-Boolean values. This is
    /// deliberately narrow evidence for one-bit app-private slots, not
    /// a theorem prover for arbitrary Boolean implications.
    pub fn boolean_constrained_variables(&self) -> Vec<bool> {
        let rows = r1cs_coeff_rows(self);
        boolean_constrained_variables_from_rows(&rows, self.n(), self.m())
    }

    /// Conservative syntactic range detector for app-private slots.
    ///
    /// Variables proven Boolean get width 1. Variables proven to be a
    /// non-negative affine combination of already bounded variables get the
    /// minimum bit width that covers that range. Everything else remains a
    /// full canonical Goldilocks lane.
    ///
    /// Completeness invariant (oracle-tested in
    /// `tests/system/width_inference_oracle.rs`): for every satisfying
    /// assignment whose values fit canonical 64-bit lanes, the returned
    /// width of each variable covers that variable's value. Every rule may
    /// only narrow when that is derivable; on any doubt the variable keeps
    /// the full lane.
    pub fn conservative_app_private_var_widths(&self) -> Vec<usize> {
        let rows = r1cs_coeff_rows(self);
        conservative_var_widths(rows, self.n(), self.m())
    }
}

fn dense_matrix_row_lcs(matrix: &neo_ccs::Mat<F>, vars: &[Var]) -> Vec<Lc> {
    let mut out = vec![Lc::zero(); matrix.rows()];
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            let coefficient = matrix[(row, col)];
            if coefficient != F::ZERO {
                out[row].add_term(vars[col], coefficient);
            }
        }
    }
    out
}

fn sparse_matrix_row_lcs(matrix: &CcsMatrix<F>, vars: &[Var], rows: usize) -> Vec<Lc> {
    let mut out = vec![Lc::zero(); rows];
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows).min(vars.len()) {
                out[row].add_term(vars[row], F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols.min(vars.len()) {
                for index in csc.column_range(col) {
                    let row = csc.row_index(index);
                    if row < rows {
                        out[row].add_term(vars[col], csc.vals[index]);
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            for col in 0..csc.ncols.min(vars.len()) {
                for index in csc.column_range(col) {
                    let row = csc.row_index(index);
                    if row < rows {
                        out[row].add_term(vars[col], csc.vals[index]);
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows && col < vars.len() {
                        out[row].add_term(vars[col], coefficient);
                    }
                });
            }
            for run in geometric_runs {
                run.for_each_term(|row, col, coefficient| {
                    if row < rows && col < vars.len() {
                        out[row].add_term(vars[col], coefficient);
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("R1CS matrix row expansion requires materialized content")
        }
    }
    out
}

fn conservative_var_widths(rows: R1csCoeffRows, row_count: usize, var_count: usize) -> Vec<usize> {
    let booleans = boolean_constrained_variables_from_rows(&rows, row_count, var_count);
    let mut bounds = booleans
        .iter()
        .map(|&is_boolean| is_boolean.then_some(1u128))
        .collect::<Vec<_>>();
    if !bounds.is_empty() {
        bounds[0] = Some(1);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for row in 0..row_count {
            if let Some((target, bound)) = bounded_affine_output_var(&rows.a[row], &rows.b[row], &rows.c[row], &bounds)
            {
                if bounds[target].is_none_or(|old| bound < old) {
                    bounds[target] = Some(bound);
                    changed = true;
                }
            }
        }
    }

    bounds
        .into_iter()
        .map(|bound| bound.map_or(POSEIDON2_GOLDILOCKS_BITS, bit_width_for_max))
        .collect()
}

fn boolean_constrained_variables_from_rows(rows: &R1csCoeffRows, row_count: usize, var_count: usize) -> Vec<bool> {
    let mut out = vec![false; var_count];
    for row in 0..row_count {
        if !rows.c[row].is_empty() {
            continue;
        }
        if let Some(var) = fixed_zero_row_var(&rows.a[row], &rows.b[row]) {
            out[var] = true;
        } else if let Some(var) = boolean_row_var(&rows.a[row], &rows.b[row]) {
            out[var] = true;
        } else if let Some(var) = boolean_row_var(&rows.b[row], &rows.a[row]) {
            out[var] = true;
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for row in 0..row_count {
            if let Some((a, b)) = copy_row_vars(&rows.a[row], &rows.b[row], &rows.c[row]) {
                if out[a] && !out[b] {
                    out[b] = true;
                    changed = true;
                } else if out[b] && !out[a] {
                    out[a] = true;
                    changed = true;
                }
            }
            if let Some(target) = product_boolean_output_var(&rows.a[row], &rows.b[row], &rows.c[row], &out) {
                if !out[target] {
                    out[target] = true;
                    changed = true;
                }
            }
            if let Some(target) = affine_boolean_output_var(&rows.a[row], &rows.b[row], &rows.c[row], &out) {
                if !out[target] {
                    out[target] = true;
                    changed = true;
                }
            }
            if let Some(target) = binary_boolean_output_var(&rows.a[row], &rows.b[row], &rows.c[row], &out) {
                if !out[target] {
                    out[target] = true;
                    changed = true;
                }
            }
        }
    }
    out
}

fn bounded_affine_output_var(
    a: &[(usize, F)],
    b: &[(usize, F)],
    c: &[(usize, F)],
    bounds: &[Option<u128>],
) -> Option<(usize, u128)> {
    if is_one_lc(a) {
        return bounded_linear_equality_output(b, c, bounds);
    }
    if is_one_lc(b) {
        return bounded_linear_equality_output(a, c, bounds);
    }
    None
}

fn bounded_linear_equality_output(
    left: &[(usize, F)],
    right: &[(usize, F)],
    bounds: &[Option<u128>],
) -> Option<(usize, u128)> {
    if let Some(target) = single_unit_output_var(right) {
        let bound = nonnegative_affine_bound(left, bounds)?;
        return Some((target, bound));
    }
    if let Some(target) = single_unit_output_var(left) {
        let bound = nonnegative_affine_bound(right, bounds)?;
        return Some((target, bound));
    }
    None
}

fn nonnegative_affine_bound(lc: &[(usize, F)], bounds: &[Option<u128>]) -> Option<u128> {
    let mut max = 0u128;
    for &(var, coeff) in lc {
        if coeff == F::ZERO {
            continue;
        }
        let coeff = small_nonnegative_coeff(coeff)?;
        let var_bound = bounds.get(var).copied().flatten()?;
        max = max.checked_add(coeff.checked_mul(var_bound)?)?;
        if max >= (1u128 << POSEIDON2_GOLDILOCKS_BITS) {
            return None;
        }
    }
    Some(max)
}

fn small_nonnegative_coeff(coeff: F) -> Option<u128> {
    let value = coeff.as_canonical_u64();
    // Coefficients above 2^63 may be negative representatives or modular
    // wraparound. Reject them instead of pretending they prove a small range.
    if value >= (1u64 << 63) {
        None
    } else {
        Some(value as u128)
    }
}

fn bit_width_for_max(max: u128) -> usize {
    let width = if max == 0 {
        1
    } else {
        (u128::BITS - max.leading_zeros()) as usize
    };
    width.clamp(1, POSEIDON2_GOLDILOCKS_BITS)
}

pub(crate) struct R1csCoeffRows {
    pub(crate) a: Vec<Vec<(usize, F)>>,
    pub(crate) b: Vec<Vec<(usize, F)>>,
    pub(crate) c: Vec<Vec<(usize, F)>>,
}

pub(crate) fn r1cs_coeff_rows(r1cs: &R1csShape) -> R1csCoeffRows {
    match r1cs {
        R1csShape::Dense(r1cs) => R1csCoeffRows {
            a: dense_coeff_rows(&r1cs.a),
            b: dense_coeff_rows(&r1cs.b),
            c: dense_coeff_rows(&r1cs.c),
        },
        R1csShape::Sparse(r1cs) => R1csCoeffRows {
            a: sparse_coeff_rows(&r1cs.a, r1cs.n),
            b: sparse_coeff_rows(&r1cs.b, r1cs.n),
            c: sparse_coeff_rows(&r1cs.c, r1cs.n),
        },
    }
}

fn r1cs_coeff_rows_sparse(r1cs: &SparseR1cs) -> R1csCoeffRows {
    R1csCoeffRows {
        a: sparse_coeff_rows(&r1cs.a, r1cs.n),
        b: sparse_coeff_rows(&r1cs.b, r1cs.n),
        c: sparse_coeff_rows(&r1cs.c, r1cs.n),
    }
}

fn dense_coeff_rows(m: &neo_ccs::Mat<F>) -> Vec<Vec<(usize, F)>> {
    let mut rows = vec![Vec::new(); m.rows()];
    for row in 0..m.rows() {
        for col in 0..m.cols() {
            let coeff = m[(row, col)];
            if coeff != F::ZERO {
                rows[row].push((col, coeff));
            }
        }
    }
    rows
}

fn sparse_coeff_rows(m: &CcsMatrix<F>, rows: usize) -> Vec<Vec<(usize, F)>> {
    let mut out = vec![Vec::new(); rows];
    match m {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows) {
                out[row].push((row, F::ONE));
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        out[row].push((col, csc.vals[idx]));
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            for col in 0..csc.ncols {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        out[row].push((col, csc.vals[idx]));
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows {
                        out[row].push((col, coefficient));
                    }
                });
            }
            for run in geometric_runs {
                run.for_each_term(|row, col, coefficient| {
                    if row < rows {
                        out[row].push((col, coefficient));
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("R1CS coefficient extraction requires materialized content")
        }
    }
    out
}

fn boolean_row_var(linear_var: &[(usize, F)], one_minus_var: &[(usize, F)]) -> Option<usize> {
    let var = single_scaled_var(linear_var)?;
    if affine_root_is_bool_complement(one_minus_var, var) {
        Some(var)
    } else {
        None
    }
}

fn fixed_zero_row_var(a: &[(usize, F)], b: &[(usize, F)]) -> Option<usize> {
    if is_one_lc(a) {
        single_scaled_var(b)
    } else if is_one_lc(b) {
        single_scaled_var(a)
    } else {
        None
    }
}

fn copy_row_vars(a: &[(usize, F)], b: &[(usize, F)], c: &[(usize, F)]) -> Option<(usize, usize)> {
    if is_one_lc(b) {
        return equal_single_var_pair(a, c);
    }
    if is_one_lc(a) {
        return equal_single_var_pair(b, c);
    }
    None
}

fn product_boolean_output_var(
    a: &[(usize, F)],
    b: &[(usize, F)],
    c: &[(usize, F)],
    known_boolean: &[bool],
) -> Option<usize> {
    let (target, coeff) = single_var_with_coeff(c)?;
    if coeff != F::ONE || target == 0 {
        return None;
    }
    if boolean_affine_lc(a, known_boolean) && boolean_affine_lc(b, known_boolean) {
        Some(target)
    } else {
        None
    }
}

fn affine_boolean_output_var(
    a: &[(usize, F)],
    b: &[(usize, F)],
    c: &[(usize, F)],
    known_boolean: &[bool],
) -> Option<usize> {
    if is_one_lc(a) && boolean_affine_lc(b, known_boolean) {
        return single_unit_output_var(c);
    }
    if is_one_lc(b) && boolean_affine_lc(a, known_boolean) {
        return single_unit_output_var(c);
    }
    None
}

fn binary_boolean_output_var(
    a: &[(usize, F)],
    b: &[(usize, F)],
    c: &[(usize, F)],
    known_boolean: &[bool],
) -> Option<usize> {
    let (left, left_coeff) = single_known_boolean_var_with_coeff(a, known_boolean)?;
    let (right, right_coeff) = single_known_boolean_var_with_coeff(b, known_boolean)?;
    if left == right {
        return None;
    }

    let product_coeff = left_coeff * right_coeff;
    if product_coeff != F::ONE && product_coeff != F::from_u64(2) {
        return None;
    }
    affine_sum_minus_target(c, left, right)
}

fn single_known_boolean_var_with_coeff(lc: &[(usize, F)], known_boolean: &[bool]) -> Option<(usize, F)> {
    let (var, coeff) = single_var_with_coeff(lc)?;
    if coeff != F::ZERO && known_boolean.get(var).copied().unwrap_or(false) {
        Some((var, coeff))
    } else {
        None
    }
}

fn affine_sum_minus_target(lc: &[(usize, F)], left: usize, right: usize) -> Option<usize> {
    if lc.len() != 3 {
        return None;
    }
    let mut saw_left = false;
    let mut saw_right = false;
    let mut target = None;
    for &(var, coeff) in lc {
        if var == left && coeff == F::ONE {
            saw_left = true;
        } else if var == right && coeff == F::ONE {
            saw_right = true;
        } else if var != 0 && var != left && var != right && coeff == F::ZERO - F::ONE {
            target = Some(var);
        } else {
            return None;
        }
    }
    if saw_left && saw_right {
        target
    } else {
        None
    }
}

fn single_unit_output_var(lc: &[(usize, F)]) -> Option<usize> {
    let (var, coeff) = single_var_with_coeff(lc)?;
    if coeff == F::ONE {
        Some(var)
    } else {
        None
    }
}

fn equal_single_var_pair(left: &[(usize, F)], right: &[(usize, F)]) -> Option<(usize, usize)> {
    let (left_var, left_coeff) = single_var_with_coeff(left)?;
    let (right_var, right_coeff) = single_var_with_coeff(right)?;
    if left_var != right_var && left_coeff == right_coeff {
        Some((left_var, right_var))
    } else {
        None
    }
}

fn single_var_with_coeff(lc: &[(usize, F)]) -> Option<(usize, F)> {
    if lc.len() == 1 && lc[0].0 != 0 && lc[0].1 != F::ZERO {
        Some(lc[0])
    } else {
        None
    }
}

fn single_scaled_var(lc: &[(usize, F)]) -> Option<usize> {
    single_var_with_coeff(lc).map(|(var, _)| var)
}

fn boolean_affine_lc(lc: &[(usize, F)], known_boolean: &[bool]) -> bool {
    if lc.is_empty() {
        return true;
    }
    if lc.len() == 1 && lc[0].0 == 0 {
        return lc[0].1 == F::ZERO || lc[0].1 == F::ONE;
    }
    if let Some((var, coeff)) = single_var_with_coeff(lc) {
        return coeff == F::ONE && known_boolean.get(var).copied().unwrap_or(false);
    }
    if lc.len() != 2 {
        return false;
    }
    let mut constant = F::ZERO;
    let mut var = None;
    let mut var_coeff = F::ZERO;
    for &(col, coeff) in lc {
        if col == 0 {
            constant += coeff;
        } else if var.is_none() || var == Some(col) {
            var = Some(col);
            var_coeff += coeff;
        } else {
            return false;
        }
    }
    let Some(var) = var else {
        return constant == F::ZERO || constant == F::ONE;
    };
    known_boolean.get(var).copied().unwrap_or(false) && constant == F::ONE && var_coeff == F::ZERO - F::ONE
}

fn affine_root_is_bool_complement(lc: &[(usize, F)], var: usize) -> bool {
    if lc.len() != 2 {
        return false;
    }
    let mut constant = F::ZERO;
    let mut var_coeff = F::ZERO;
    for &(col, coeff) in lc {
        if col == 0 {
            constant += coeff;
        } else if col == var {
            var_coeff += coeff;
        } else {
            return false;
        }
    }
    constant != F::ZERO && var_coeff != F::ZERO && constant + var_coeff == F::ZERO
}

fn is_one_lc(lc: &[(usize, F)]) -> bool {
    lc.len() == 1 && lc[0].0 == 0 && lc[0].1 == F::ONE
}

/// Layout anchors returned alongside the [`FPrimeStructure`] when the
/// latter was produced by [`build_r1cs_f_prime_structure`]. Tests use
/// the row-start / row-count fields to confirm each R1CS constraint
/// became its own structure row; the encoder reads `app_var_slots` to
/// fill `app_private` in the right order.
#[derive(Clone, Debug)]
pub struct R1csRowAnchors {
    /// Variable assignment slots: `app_var_slots[j]` is the committed
    /// slot for R1CS variable `z[j]` (one bit for proven-Boolean
    /// variables, 64 bits for field variables).
    pub app_var_slots: Vec<AppVariableSlot>,
    /// First row index of the appended R1CS product block.
    pub r1cs_row_start: usize,
    /// Number of R1CS product rows appended (`= r1cs.n()`).
    pub r1cs_row_count: usize,
    /// Whether the structure emits a row pinning conventional R1CS
    /// variable `z[0]` to the CCS constant-one column.
    pub constant_lane_pinned: bool,
}

/// Build the CCS structure for an R1CS app step.
///
/// The layout must reserve either legacy `r1cs.m() * 64` bits inside
/// its `app_private` region or one typed slot per variable via
/// `layout.config.app_private_var_widths`.
pub fn build_r1cs_f_prime_structure<R>(layout: FPrimeImageLayout, r1cs: R) -> (FPrimeStructure, R1csRowAnchors)
where
    R: Into<R1csShape>,
{
    let r1cs = r1cs.into();
    let image_end = layout.end;
    assert!(
        image_end >= 2,
        "FPrimeImageLayout::end = {image_end} too small; need constant slot + ≥1 bit column"
    );
    if layout.config.app_private_var_widths.is_empty() {
        assert_eq!(
            layout.app_private.bits,
            r1cs.m() * POSEIDON2_GOLDILOCKS_BITS,
            "layout.app_private must reserve r1cs.m() * 64 bits (set plan.limbs = r1cs.m() * 64 + 1)"
        );
    } else {
        assert_eq!(
            layout.config.app_private_var_widths.len(),
            r1cs.m(),
            "typed app-private layout must provide one width per R1CS variable"
        );
    }

    let lane_slots = f_prime_lane_slots(&layout);
    let app_var_slots = lane_slots.app_assignment_lanes.clone();
    assert_eq!(
        app_var_slots.len(),
        r1cs.m(),
        "app-private slot count must match R1CS variable count"
    );

    let mut builder = MixedGateBuilder::with_estimated_rows(image_end);
    emit_shell_rows(&layout, &lane_slots, &mut builder);
    let constant_lane_pinned = requires_r1cs_constant_lane_pin(&layout);
    if constant_lane_pinned {
        pin_r1cs_constant_lane(&app_var_slots, &mut builder);
    }

    let r1cs_row_start = builder.rows();
    append_r1cs_rows(&app_var_slots, &r1cs, &mut builder);
    let r1cs_row_count = builder.rows() - r1cs_row_start;
    debug_assert_eq!(r1cs_row_count, r1cs.n());

    let ccs = builder.finish(image_end);
    let structure = FPrimeStructure {
        layout,
        ccs,
        lane_slots,
    };
    let anchors = R1csRowAnchors {
        app_var_slots,
        r1cs_row_start,
        r1cs_row_count,
        constant_lane_pinned,
    };
    (structure, anchors)
}

/// Pin the app R1CS conventional constant lane `z[0]` to the F' constant
/// column. We emit this whenever the typed layout narrows any app
/// variable below a full canonical lane, because the width prover treats
/// `z[0]` as the conventional constant-one lane. An anchored transition
/// can use `z[0]` as an ordinary value on one semantic side. If it carries
/// `z[0]` on both sides, the lane is the invariant constant and is pinned
/// directly instead of relying on a digest to make it authoritative.
fn requires_r1cs_constant_lane_pin(layout: &FPrimeImageLayout) -> bool {
    let app_var_zero_semantic_roles = layout
        .config
        .poseidon_transition_enforcements
        .iter()
        .filter(|enforcement| {
            enforcement
                .preimage_lanes
                .iter()
                .any(|source| match source {
                    PoseidonPreimageLaneSource::AppAssignmentLane(0) => true,
                    PoseidonPreimageLaneSource::AppAssignmentBitPack(indices) => indices.contains(&0),
                    _ => false,
                })
        })
        .count();

    layout
        .config
        .app_private_var_widths
        .iter()
        .any(|&width| width < POSEIDON2_GOLDILOCKS_BITS)
        || (layout.config.initial_semantic_state_digest_anchor.is_some() && app_var_zero_semantic_roles != 1)
        || layout
            .config
            .poseidon_transition_enforcements
            .iter()
            .any(|enforcement| {
                enforcement
                    .preimage_lanes
                    .iter()
                    .any(|source| matches!(source, PoseidonPreimageLaneSource::AppAssignmentBitPack(_)))
            })
}

fn pin_r1cs_constant_lane(app_var_slots: &[AppVariableSlot], builder: &mut MixedGateBuilder) {
    let constant_slot = app_var_slots
        .first()
        .expect("R1CS-F' structure requires at least the z[0] app variable");
    builder.linear(app_variable_terms(*constant_slot), [(0, F::ONE)]);
}

/// Append one product row per R1CS constraint. For row `i`:
///
/// ```text
/// (Σ_j A[i,j] · lane_terms(z_j)) ·
/// (Σ_j B[i,j] · lane_terms(z_j))
///   = (Σ_j C[i,j] · lane_terms(z_j))
/// ```
///
/// Each variable's 64 bits are recomposed inline via `lane_terms`; no
/// fresh witness columns are minted.
fn append_r1cs_rows(app_var_slots: &[AppVariableSlot], r1cs: &R1csShape, builder: &mut MixedGateBuilder) {
    match r1cs {
        R1csShape::Dense(r1cs) => {
            for row in 0..r1cs.n() {
                let left = dense_matrix_row_terms(&r1cs.a, row, app_var_slots);
                let right = dense_matrix_row_terms(&r1cs.b, row, app_var_slots);
                let out = dense_matrix_row_terms(&r1cs.c, row, app_var_slots);
                builder.product(left, right, out);
            }
        }
        R1csShape::Sparse(r1cs) => {
            let left = sparse_matrix_row_terms(&r1cs.a, app_var_slots, r1cs.n);
            let right = sparse_matrix_row_terms(&r1cs.b, app_var_slots, r1cs.n);
            let out = sparse_matrix_row_terms(&r1cs.c, app_var_slots, r1cs.n);
            for row in 0..r1cs.n {
                builder.product(
                    left[row].iter().copied(),
                    right[row].iter().copied(),
                    out[row].iter().copied(),
                );
            }
        }
    }
}

/// Expand one matrix row `M[row, ·]` into `(col, coeff)` terms over the
/// F' bit-frame: each nonzero `M[row, j]` contributes a scaled lane
/// sum `M[row, j] · Σ_i 2^i · z[bit_start_j + i]`.
fn dense_matrix_row_terms(m: &neo_ccs::Mat<F>, row: usize, app_var_slots: &[AppVariableSlot]) -> Vec<(usize, F)> {
    let mut out: Vec<(usize, F)> = Vec::new();
    for (j, slot) in app_var_slots.iter().enumerate() {
        let coeff = m[(row, j)];
        if coeff != F::ZERO {
            out.extend(
                app_variable_terms(*slot)
                    .into_iter()
                    .map(|(col, c)| (col, c * coeff)),
            );
        }
    }
    out
}

fn sparse_matrix_row_terms(m: &CcsMatrix<F>, app_var_slots: &[AppVariableSlot], rows: usize) -> Vec<Vec<(usize, F)>> {
    let mut out = vec![Vec::new(); rows];
    match m {
        CcsMatrix::Identity { n } => {
            for (row, slot) in app_var_slots.iter().take((*n).min(rows)).enumerate() {
                out[row].extend(app_variable_terms(*slot));
            }
        }
        CcsMatrix::Csc(csc) => {
            for (col, slot) in app_var_slots.iter().enumerate().take(csc.ncols) {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        let coeff = csc.vals[idx];
                        out[row].extend(
                            app_variable_terms(*slot)
                                .into_iter()
                                .map(|(lane_col, lane_coeff)| (lane_col, lane_coeff * coeff)),
                        );
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            for (col, slot) in app_var_slots.iter().enumerate().take(csc.ncols) {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        let coeff = csc.vals[idx];
                        out[row].extend(
                            app_variable_terms(*slot)
                                .into_iter()
                                .map(|(lane_col, lane_coeff)| (lane_col, lane_coeff * coeff)),
                        );
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows && col < app_var_slots.len() {
                        out[row].extend(
                            app_variable_terms(app_var_slots[col])
                                .into_iter()
                                .map(|(lane_col, lane_coeff)| (lane_col, lane_coeff * coefficient)),
                        );
                    }
                });
            }
            for run in geometric_runs {
                run.for_each_term(|row, col, coefficient| {
                    if row < rows && col < app_var_slots.len() {
                        out[row].extend(
                            app_variable_terms(app_var_slots[col])
                                .into_iter()
                                .map(|(lane_col, lane_coeff)| (lane_col, lane_coeff * coefficient)),
                        );
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("application matrix lowering requires materialized content")
        }
    }
    out
}
