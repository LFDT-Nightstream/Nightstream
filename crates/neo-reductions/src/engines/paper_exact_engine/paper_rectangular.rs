//! Direct reference evaluator for corrected rectangular-paper `Pi_CCS`.
//!
//! This module owns literal row-domain FE and column-domain NC evaluation.
//! It does not import an optimized oracle, evaluator cache, transformed
//! matrix, sparse cache, or production digit table.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{superneo_bar_block, Fq, KExtensions, Rq, D, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::engines::pi_ccs_protocol::Challenges;
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

fn chi_table(point: &[K]) -> Vec<K> {
    let size = 1usize << point.len();
    let mut table = Vec::with_capacity(size);
    for index in 0..size {
        let mut weight = K::ONE;
        for (bit, &challenge) in point.iter().enumerate() {
            weight *= if (index >> bit) & 1 == 1 {
                challenge
            } else {
                K::ONE - challenge
            };
        }
        table.push(weight);
    }
    table
}

#[inline]
fn matrix_entry<Ff>(matrix: &CcsMatrix<Ff>, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if row >= matrix.rows() || column >= matrix.cols() {
        return Ff::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == column {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let mut value = Ff::ZERO;
            for entry in csc.column_range(column) {
                if csc.row_index(entry) == row {
                    value += csc.vals[entry];
                }
            }
            value
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut value = Ff::ZERO;
            for entry in csc.column_range(column) {
                if csc.row_index(entry) == row {
                    value += csc.vals[entry];
                }
            }
            for block in blocks {
                value += block.entry::<Ff>(row, column);
            }
            for run in geometric_runs {
                value += run.entry(row, column);
            }
            value
        }
    }
}

fn validate_packed_witness<Ff>(witness: &Mat<Ff>, width: usize) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if witness.rows() != D || witness.cols() != width.div_ceil(D) {
        return Err(PiCcsError::InvalidInput(format!(
            "paper rectangular witness shape is {}x{}, expected {}x{}",
            witness.rows(),
            witness.cols(),
            D,
            width.div_ceil(D)
        )));
    }
    Ok(())
}

fn packed_coefficients<Ff>(witness: &Mat<Ff>, width: usize) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let padded_width = width.div_ceil(D) * D;
    (0..padded_width)
        .map(|column| K::from(witness[(column % D, column / D)]))
        .collect()
}

#[inline]
fn assignment_value<Ff>(witness: &Mat<Ff>, column: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    K::from(witness[(column % D, column / D)])
}

fn direct_ring_row<Ff>(matrix: &CcsMatrix<Ff>, row: usize, assignment: &[K]) -> [K; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut output = [K::ZERO; D];
    for block in 0..matrix.cols().div_ceil(D) {
        let base = block * D;
        let mut matrix_block = [Fq::ZERO; D];
        let mut assignment_real = [Fq::ZERO; D];
        let mut assignment_imag = [Fq::ZERO; D];
        for lane in 0..D {
            let column = base + lane;
            if column < matrix.cols() {
                matrix_block[lane] = K::from(matrix_entry(matrix, row, column)).real();
            }
            if column < assignment.len() {
                let [real, imag] = assignment[column].as_coeffs();
                assignment_real[lane] = real;
                assignment_imag[lane] = imag;
            }
        }
        let lifted_matrix = Rq(superneo_bar_block(matrix_block));
        let real_product = lifted_matrix.mul(&Rq(assignment_real));
        let imag_product = lifted_matrix.mul(&Rq(assignment_imag));
        for coefficient in 0..D {
            output[coefficient] += K::from_coeffs([real_product.0[coefficient], imag_product.0[coefficient]]);
        }
    }
    output
}

/// Evaluate every coefficient of one matrix image at a row-domain point.
pub fn direct_ring_mle<Ff>(matrix: &CcsMatrix<Ff>, assignment: &[K], row_weights: &[K]) -> [K; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut output = [K::ZERO; D];
    for (row, &weight) in row_weights.iter().take(matrix.rows()).enumerate() {
        let row_value = direct_ring_row(matrix, row, assignment);
        for coefficient in 0..D {
            output[coefficient] += weight * row_value[coefficient];
        }
    }
    output
}

#[inline]
fn range_product<Ff>(value: K, base: u32) -> K
where
    Ff: Field + PrimeCharacteristicRing,
    K: From<Ff>,
{
    let mut product = K::ONE;
    for integer in -((base as i64) - 1)..=((base as i64) - 1) {
        product *= value - K::from(Ff::from_i64(integer));
    }
    product
}

#[inline]
fn paper_gamma_power(gamma: K, exponent: usize) -> K {
    let mut power = K::ONE;
    for _ in 0..exponent {
        power *= gamma;
    }
    power
}

/// Literal zero-based form of the paper exponent `2K+k+I(i,j,l)`.
pub fn paper_carried_gamma_exponent(
    fresh_count: usize,
    running_count: usize,
    matrix_count: usize,
    running: usize,
    matrix: usize,
    coefficient: usize,
) -> usize {
    let local = running + running_count * matrix + running_count * matrix_count * coefficient;
    2 * fresh_count + running_count + local
}

#[inline]
fn paper_equality(point: &[K], target: &[K]) -> K {
    assert_eq!(point.len(), target.len(), "paper equality point length mismatch");
    let mut product = K::ONE;
    for (&left, &right) in point.iter().zip(target) {
        product *= (K::ONE - left) * (K::ONE - right) + left * right;
    }
    product
}

/// Literal corrected paper target with the outer carried-block shift.
pub(super) fn paper_fe_initial<Ff>(
    structure: &CcsStructure<Ff>,
    challenges: &Challenges,
    fresh_count: usize,
    running: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field,
{
    let running_count = running.len();
    let mut target = K::ZERO;
    for (running_index, claim) in running.iter().enumerate() {
        if claim.y_ring.len() != structure.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "paper running claim {running_index} has {} matrix images, expected {}",
                claim.y_ring.len(),
                structure.t()
            )));
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(format!(
                    "paper running claim {running_index} matrix {matrix} has {} coefficients, expected at least {D}",
                    coefficients.len()
                )));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                let exponent = paper_carried_gamma_exponent(
                    fresh_count,
                    running_count,
                    structure.t(),
                    running_index,
                    matrix,
                    coefficient,
                );
                target += paper_gamma_power(challenges.gamma, exponent) * value;
            }
        }
    }
    Ok(target)
}

/// Literal row-terminal evaluation from paper protocol step 4.
pub(super) fn paper_fe_terminal<Ff>(
    structure: &CcsStructure<Ff>,
    challenges: &Challenges,
    fresh_count: usize,
    prior_point: Option<&[K]>,
    row_point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput("too few paper FE outputs".into()));
    }
    let mut fresh_ccs = K::ZERO;
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        if output.ct.len() < structure.t() {
            return Err(PiCcsError::InvalidInput("paper FE output ct is too short".into()));
        }
        fresh_ccs +=
            paper_gamma_power(challenges.gamma, source) * structure.f.eval_in_ext::<K>(&output.ct[..structure.t()]);
    }

    let running_count = outputs.len() - fresh_count;
    let mut carried = K::ZERO;
    for (running, output) in outputs.iter().skip(fresh_count).enumerate() {
        if output.y_ring.len() != structure.t() {
            return Err(PiCcsError::InvalidInput("paper FE output matrix count mismatch".into()));
        }
        for (matrix, coefficients) in output.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(
                    "paper FE output coefficient row is too short".into(),
                ));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                let exponent = paper_carried_gamma_exponent(
                    fresh_count,
                    running_count,
                    structure.t(),
                    running,
                    matrix,
                    coefficient,
                );
                carried += paper_gamma_power(challenges.gamma, exponent) * value;
            }
        }
    }
    let carried_gate = prior_point.map_or(K::ZERO, |prior| paper_equality(row_point, prior));
    Ok(paper_equality(row_point, &challenges.beta_r) * fresh_ccs + carried_gate * carried)
}

/// Literal column-terminal evaluation of the shifted paper norm block.
pub(super) fn paper_nc_terminal<Ff>(
    params: &neo_params::NeoParams,
    challenges: &Challenges,
    fresh_count: usize,
    column_point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut norm = K::ZERO;
    for (source, output) in outputs.iter().enumerate() {
        if output.y_zcol.len() < D {
            return Err(PiCcsError::InvalidInput("paper NC output opening is too short".into()));
        }
        let value = output.y_zcol.iter().take(D).copied().sum();
        norm += paper_gamma_power(challenges.gamma, fresh_count + source) * range_product::<Ff>(value, params.b);
    }
    Ok(paper_equality(column_point, &challenges.beta_m) * norm)
}

fn all_witnesses<'a, Ff>(fresh: &'a [CcsWitness<Ff>], running: &'a [Mat<Ff>]) -> impl Iterator<Item = &'a Mat<Ff>> {
    fresh.iter().map(|witness| &witness.Z).chain(running)
}

/// Direct row-domain FE oracle.
pub struct PaperRectangularFeOracle<'a, Ff> {
    structure: &'a CcsStructure<Ff>,
    fresh: &'a [CcsWitness<Ff>],
    running: &'a [Mat<Ff>],
    challenges: Challenges,
    prior_point: Option<Vec<K>>,
    variables: usize,
    degree: usize,
    round: usize,
    fixed: Vec<K>,
}

impl<'a, Ff> PaperRectangularFeOracle<'a, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<Ff>,
        fresh: &'a [CcsWitness<Ff>],
        running: &'a [Mat<Ff>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        variables: usize,
        degree: usize,
    ) -> Result<Self, PiCcsError> {
        if challenges.beta_r.len() != variables || !challenges.is_paper_rectangular() {
            return Err(PiCcsError::InvalidInput(
                "paper rectangular FE challenge shape mismatch".into(),
            ));
        }
        if running.is_empty() != prior_point.is_none() {
            return Err(PiCcsError::InvalidInput(
                "paper rectangular FE requires one prior row point exactly when running sources exist".into(),
            ));
        }
        for witness in all_witnesses(fresh, running) {
            validate_packed_witness(witness, structure.m)?;
        }
        Ok(Self {
            structure,
            fresh,
            running,
            challenges,
            prior_point: prior_point.map(<[K]>::to_vec),
            variables,
            degree,
            round: 0,
            fixed: Vec::with_capacity(variables),
        })
    }

    pub fn evaluate(&self, point: &[K]) -> K {
        let row_weights = chi_table(point);
        let mut fresh_ccs = K::ZERO;
        for (source, witness) in self.fresh.iter().enumerate() {
            let assignment = packed_coefficients(&witness.Z, self.structure.m);
            let matrix_values: Vec<K> = self
                .structure
                .matrices
                .iter()
                .map(|matrix| direct_ring_mle(matrix, &assignment, &row_weights)[0])
                .collect();
            fresh_ccs +=
                paper_gamma_power(self.challenges.gamma, source) * self.structure.f.eval_in_ext::<K>(&matrix_values);
        }

        let mut carried = K::ZERO;
        let running_count = self.running.len();
        for (running, witness) in self.running.iter().enumerate() {
            let assignment = packed_coefficients(witness, self.structure.m);
            for (matrix, source_matrix) in self.structure.matrices.iter().enumerate() {
                let coefficients = direct_ring_mle(source_matrix, &assignment, &row_weights);
                for (coefficient, value) in coefficients.into_iter().enumerate() {
                    let exponent = paper_carried_gamma_exponent(
                        self.fresh.len(),
                        running_count,
                        self.structure.t(),
                        running,
                        matrix,
                        coefficient,
                    );
                    carried += paper_gamma_power(self.challenges.gamma, exponent) * value;
                }
            }
        }
        let carried_gate = self
            .prior_point
            .as_deref()
            .map_or(K::ZERO, |prior| paper_equality(point, prior));
        paper_equality(point, &self.challenges.beta_r) * fresh_ccs + carried_gate * carried
    }

    fn round_evaluations(&self, points: &[K]) -> Vec<K> {
        let remaining = self.variables - self.round - 1;
        let tails = 1usize << remaining;
        points
            .iter()
            .map(|&value| {
                let mut sum = K::ZERO;
                for tail in 0..tails {
                    let mut point = self.fixed.clone();
                    point.push(value);
                    for bit in 0..remaining {
                        point.push(if (tail >> bit) & 1 == 1 { K::ONE } else { K::ZERO });
                    }
                    sum += self.evaluate(&point);
                }
                sum
            })
            .collect()
    }
}

impl<Ff> RoundOracle for PaperRectangularFeOracle<'_, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.round_evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.variables
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn fold(&mut self, challenge: K) {
        self.fixed.push(challenge);
        self.round += 1;
    }
}

/// Direct column-domain NC oracle.
pub struct PaperRectangularNcOracle<'a, Ff> {
    structure: &'a CcsStructure<Ff>,
    params: &'a neo_params::NeoParams,
    fresh: &'a [CcsWitness<Ff>],
    running: &'a [Mat<Ff>],
    challenges: Challenges,
    variables: usize,
    degree: usize,
    round: usize,
    fixed: Vec<K>,
}

impl<'a, Ff> PaperRectangularNcOracle<'a, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<Ff>,
        params: &'a neo_params::NeoParams,
        fresh: &'a [CcsWitness<Ff>],
        running: &'a [Mat<Ff>],
        challenges: Challenges,
        variables: usize,
        degree: usize,
    ) -> Result<Self, PiCcsError> {
        if challenges.beta_m.len() != variables || !challenges.is_paper_rectangular() {
            return Err(PiCcsError::InvalidInput(
                "paper rectangular NC challenge shape mismatch".into(),
            ));
        }
        for witness in all_witnesses(fresh, running) {
            validate_packed_witness(witness, structure.m)?;
        }
        Ok(Self {
            structure,
            params,
            fresh,
            running,
            challenges,
            variables,
            degree,
            round: 0,
            fixed: Vec::with_capacity(variables),
        })
    }

    pub fn evaluate(&self, point: &[K]) -> K {
        let column_weights = chi_table(point);
        let mut norm = K::ZERO;
        for (source, witness) in all_witnesses(self.fresh, self.running).enumerate() {
            let mut value = K::ZERO;
            for (column, &weight) in column_weights.iter().take(self.structure.m).enumerate() {
                value += weight * assignment_value(witness, column);
            }
            let exponent = self.fresh.len() + source;
            norm += paper_gamma_power(self.challenges.gamma, exponent) * range_product::<Ff>(value, self.params.b);
        }
        paper_equality(point, &self.challenges.beta_m) * norm
    }

    fn round_evaluations(&self, points: &[K]) -> Vec<K> {
        let remaining = self.variables - self.round - 1;
        let tails = 1usize << remaining;
        points
            .iter()
            .map(|&value| {
                let mut sum = K::ZERO;
                for tail in 0..tails {
                    let mut point = self.fixed.clone();
                    point.push(value);
                    for bit in 0..remaining {
                        point.push(if (tail >> bit) & 1 == 1 { K::ONE } else { K::ZERO });
                    }
                    sum += self.evaluate(&point);
                }
                sum
            })
            .collect()
    }
}

impl<Ff> RoundOracle for PaperRectangularNcOracle<'_, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.round_evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.variables
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn fold(&mut self, challenge: K) {
        self.fixed.push(challenge);
        self.round += 1;
    }
}

/// Direct one-polynomial square-paper oracle. This is the executable baseline
/// from which the rectangular FE/NC split is checked.
pub struct PaperJointSquareOracle<'a, Ff> {
    structure: &'a CcsStructure<Ff>,
    params: &'a neo_params::NeoParams,
    fresh: &'a [CcsWitness<Ff>],
    running: &'a [Mat<Ff>],
    challenges: Challenges,
    prior_point: Option<Vec<K>>,
    variables: usize,
    degree: usize,
    round: usize,
    fixed: Vec<K>,
}

impl<'a, Ff> PaperJointSquareOracle<'a, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<Ff>,
        params: &'a neo_params::NeoParams,
        fresh: &'a [CcsWitness<Ff>],
        running: &'a [Mat<Ff>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        variables: usize,
        degree: usize,
    ) -> Result<Self, PiCcsError> {
        if structure.n != structure.m {
            return Err(PiCcsError::InvalidInput(
                "paper joint oracle requires a square CCS matrix domain".into(),
            ));
        }
        if challenges.beta_r != challenges.beta_m
            || challenges.beta_r.len() != variables
            || !challenges.is_paper_rectangular()
        {
            return Err(PiCcsError::InvalidInput(
                "paper joint oracle requires one shared equality point".into(),
            ));
        }
        if running.is_empty() != prior_point.is_none() {
            return Err(PiCcsError::InvalidInput(
                "paper joint oracle prior-point presence mismatch".into(),
            ));
        }
        for witness in all_witnesses(fresh, running) {
            validate_packed_witness(witness, structure.m)?;
        }
        Ok(Self {
            structure,
            params,
            fresh,
            running,
            challenges,
            prior_point: prior_point.map(<[K]>::to_vec),
            variables,
            degree,
            round: 0,
            fixed: Vec::with_capacity(variables),
        })
    }

    pub fn evaluate(&self, point: &[K]) -> K {
        let weights = chi_table(point);
        let mut fresh_ccs = K::ZERO;
        for (source, witness) in self.fresh.iter().enumerate() {
            let assignment = packed_coefficients(&witness.Z, self.structure.m);
            let matrix_values: Vec<K> = self
                .structure
                .matrices
                .iter()
                .map(|matrix| direct_ring_mle(matrix, &assignment, &weights)[0])
                .collect();
            fresh_ccs +=
                paper_gamma_power(self.challenges.gamma, source) * self.structure.f.eval_in_ext::<K>(&matrix_values);
        }

        let mut norm = K::ZERO;
        for (source, witness) in all_witnesses(self.fresh, self.running).enumerate() {
            let mut value = K::ZERO;
            for (column, &weight) in weights.iter().take(self.structure.m).enumerate() {
                value += weight * assignment_value(witness, column);
            }
            norm += paper_gamma_power(self.challenges.gamma, self.fresh.len() + source)
                * range_product::<Ff>(value, self.params.b);
        }

        let running_count = self.running.len();
        let mut carried = K::ZERO;
        for (running, witness) in self.running.iter().enumerate() {
            let assignment = packed_coefficients(witness, self.structure.m);
            for (matrix, source_matrix) in self.structure.matrices.iter().enumerate() {
                for (coefficient, value) in direct_ring_mle(source_matrix, &assignment, &weights)
                    .into_iter()
                    .enumerate()
                {
                    let exponent = paper_carried_gamma_exponent(
                        self.fresh.len(),
                        running_count,
                        self.structure.t(),
                        running,
                        matrix,
                        coefficient,
                    );
                    carried += paper_gamma_power(self.challenges.gamma, exponent) * value;
                }
            }
        }

        let paper_gate = paper_equality(point, &self.challenges.beta_r);
        let carried_gate = self
            .prior_point
            .as_deref()
            .map_or(K::ZERO, |prior| paper_equality(point, prior));
        paper_gate * (fresh_ccs + norm) + carried_gate * carried
    }

    fn round_evaluations(&self, points: &[K]) -> Vec<K> {
        let remaining = self.variables - self.round - 1;
        let tails = 1usize << remaining;
        points
            .iter()
            .map(|&value| {
                let mut sum = K::ZERO;
                for tail in 0..tails {
                    let mut point = self.fixed.clone();
                    point.push(value);
                    for bit in 0..remaining {
                        point.push(if (tail >> bit) & 1 == 1 { K::ONE } else { K::ZERO });
                    }
                    sum += self.evaluate(&point);
                }
                sum
            })
            .collect()
    }
}

impl<Ff> RoundOracle for PaperJointSquareOracle<'_, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.round_evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.variables
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn fold(&mut self, challenge: K) {
        self.fixed.push(challenge);
        self.round += 1;
    }
}

/// Build all output openings with direct matrix and witness loops.
#[allow(clippy::too_many_arguments)]
pub fn build_outputs<Ff, L>(
    structure: &CcsStructure<Ff>,
    fresh_claims: &[CcsClaim<Cmt, Ff>],
    fresh_witnesses: &[CcsWitness<Ff>],
    running_claims: &[CeClaim<Cmt, Ff, K>],
    running_witnesses: &[Mat<Ff>],
    row_point: &[K],
    column_point: &[K],
    fold_digest: [u8; 32],
    _commitment: &L,
) -> Result<Vec<CeClaim<Cmt, Ff, K>>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    L: neo_ccs::traits::SModuleHomomorphism<Ff, Cmt>,
{
    let row_weights = chi_table(row_point);
    let column_weights = chi_table(column_point);
    let d_pad = D.next_power_of_two();

    let openings = |witness: &Mat<Ff>| -> Result<(Vec<Vec<K>>, Vec<K>, Vec<K>), PiCcsError> {
        validate_packed_witness(witness, structure.m)?;
        let assignment = packed_coefficients(witness, structure.m);
        let mut y_ring = Vec::with_capacity(structure.t());
        for matrix in &structure.matrices {
            let coefficients = direct_ring_mle(matrix, &assignment, &row_weights);
            let mut padded = coefficients.to_vec();
            padded.resize(d_pad, K::ZERO);
            y_ring.push(padded);
        }
        let ct = y_ring.iter().map(|coefficients| coefficients[0]).collect();
        let mut y_zcol = vec![K::ZERO; d_pad];
        for (column, &weight) in column_weights.iter().take(structure.m).enumerate() {
            y_zcol[column % D] += weight * assignment_value(witness, column);
        }
        Ok((y_ring, ct, y_zcol))
    };

    let mut outputs = Vec::with_capacity(fresh_claims.len() + running_claims.len());
    for (claim, witness) in fresh_claims.iter().zip(fresh_witnesses) {
        let (y_ring, ct, y_zcol) = openings(&witness.Z)?;
        let required_columns = claim.m_in.div_ceil(D);
        let mut X = Mat::zero(D, claim.m_in, Ff::ZERO);
        for column in 0..required_columns {
            for row in 0..D {
                X[(row, column)] = witness.Z[(row, column)];
            }
        }
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X,
            r: row_point.to_vec(),
            s_col: column_point.to_vec(),
            y_ring,
            ct,
            aux_openings: Vec::new(),
            y_zcol,
            m_in: claim.m_in,
            fold_digest,
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
            adv: None,
        });
    }
    for (claim, witness) in running_claims.iter().zip(running_witnesses) {
        let (y_ring, ct, y_zcol) = openings(witness)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: claim.X.clone(),
            r: row_point.to_vec(),
            s_col: column_point.to_vec(),
            y_ring,
            ct,
            aux_openings: Vec::new(),
            y_zcol,
            m_in: claim.m_in,
            fold_digest,
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
            adv: None,
        });
    }
    Ok(outputs)
}
