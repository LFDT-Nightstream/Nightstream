//! Direct one-joint PiCCS evaluator from SuperNeo Section 7.3.
//!
//! The sole rectangular specialization is the paper's padded identity
//! `M_1 = [I; 0]`. This file uses explicit loops and owns its formula copy.
//! It does not import optimized evaluators, caches, or protocol flow.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{Fq, D, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::engines::pi_ccs_joint::JointDims;
use crate::engines::pi_ccs_protocol::Challenges;
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

use super::paper_matrix::matrix_entry;
use super::paper_ring::PaperRing;

pub(super) fn dimensions<Ff>(
    params: &neo_params::NeoParams,
    structure: &CcsStructure<Ff>,
    fresh_count: usize,
    running_count: usize,
) -> Result<JointDims, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if structure.n == 0 || structure.m == 0 || fresh_count == 0 {
        return Err(PiCcsError::InvalidInput(
            "PaperExact requires nonzero dimensions and at least one fresh source".into(),
        ));
    }
    if fresh_count > neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize {
        return Err(PiCcsError::InvalidInput(
            "PaperExact fresh source count exceeds the paper profile".into(),
        ));
    }
    if running_count > params.k_rho as usize {
        return Err(PiCcsError::InvalidInput(
            "PaperExact running source count exceeds k_rho".into(),
        ));
    }
    if structure
        .matrices
        .iter()
        .flat_map(|matrix| matrix.seeded_phi81_blocks())
        .any(|block| block.has_superneo_transformed_columns())
    {
        return Err(PiCcsError::InvalidInput(
            "PaperExact requires original, untransformed CCS matrices".into(),
        ));
    }
    if structure.f.eval(&vec![Ff::ZERO; structure.t()]) != Ff::ZERO {
        return Err(PiCcsError::InvalidInput(
            "PaperExact zero-row padding requires f(0,...,0)=0".into(),
        ));
    }

    let assignment_width = structure.m.div_ceil(D) * D;
    let row_count = structure.n.max(assignment_width).next_power_of_two().max(2);
    let variables = row_count.trailing_zeros() as usize;
    let matrix_count = structure
        .t()
        .checked_add(1)
        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact matrix count overflow".into()))?;
    let degree = (structure.max_degree() as usize + 1)
        .max(2 * params.b as usize)
        .max(2);
    let source_count = fresh_count
        .checked_add(running_count)
        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact source count overflow".into()))?;
    let norm_degree = fresh_count
        .checked_add(source_count.saturating_sub(1))
        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact norm degree overflow".into()))?;
    let carried_count = running_count
        .checked_mul(matrix_count)
        .and_then(|value| value.checked_mul(D))
        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact carried count overflow".into()))?;
    let carried_degree = if carried_count == 0 {
        0
    } else {
        2usize
            .checked_mul(fresh_count)
            .and_then(|value| value.checked_add(running_count))
            .and_then(|value| value.checked_add(carried_count - 1))
            .ok_or_else(|| PiCcsError::InvalidInput("PaperExact carried degree overflow".into()))?
    };
    let soundness_factor = norm_degree
        .max(carried_degree)
        .checked_add(
            variables
                .checked_mul(degree)
                .ok_or_else(|| PiCcsError::InvalidInput("PaperExact SumCheck factor overflow".into()))?,
        )
        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact soundness factor overflow".into()))?;
    params
        .extension_check_factor(soundness_factor as u128)
        .map_err(|error| PiCcsError::ExtensionPolicyFailed(error.to_string()))?;

    Ok(JointDims {
        assignment_width,
        row_count,
        variables,
        matrix_count,
        degree,
    })
}

pub(super) fn validate_public_instances<Ff>(
    structure: &CcsStructure<Ff>,
    fresh: &[CcsClaim<Cmt, Ff>],
    running: &[CeClaim<Cmt, Ff, K>],
) -> Result<(), PiCcsError>
where
    Ff: Field + Copy,
{
    for (index, claim) in fresh.iter().enumerate() {
        if claim.m_in > structure.m || claim.x.len() != claim.m_in || claim.m_in % D != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "PaperExact fresh claim {index} is not a complete whole-ring public input"
            )));
        }
    }
    let matrix_count = structure.t() + 1;
    for (index, claim) in running.iter().enumerate() {
        if claim.m_in > structure.m
            || claim.m_in % D != 0
            || claim.X.rows() != D
            || claim.X.cols() != claim.m_in
            || claim.y_ring.len() != matrix_count
            || claim.ct.len() != matrix_count
        {
            return Err(PiCcsError::InvalidInput(format!(
                "PaperExact running claim {index} does not have the paper CE shape"
            )));
        }
        for column in claim.m_in / D..claim.X.cols() {
            for row in 0..D {
                if claim.X[(row, column)] != Ff::ZERO {
                    return Err(PiCcsError::InvalidInput(format!(
                        "PaperExact running claim {index} has a nonzero inactive public-input slot"
                    )));
                }
            }
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() != D.next_power_of_two()
                || coefficients[0] != claim.ct[matrix]
                || coefficients.iter().skip(D).any(|&value| value != K::ZERO)
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "PaperExact running claim {index} matrix image {matrix} is not canonical"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn paper_prior_point<'a, Ff>(
    running: &'a [CeClaim<Cmt, Ff, K>],
    variables: usize,
) -> Result<Option<&'a [K]>, PiCcsError> {
    let Some(first) = running.first() else {
        return Ok(None);
    };
    if first.r.len() != variables || running.iter().any(|claim| claim.r != first.r) {
        return Err(PiCcsError::InvalidInput(
            "PaperExact running claims must share the complete prior point".into(),
        ));
    }
    Ok(Some(&first.r))
}

fn boolean_weight(point: &[K], index: usize) -> K {
    let mut weight = K::ONE;
    for (bit, &challenge) in point.iter().enumerate() {
        weight *= if (index >> bit) & 1 == 1 {
            challenge
        } else {
            K::ONE - challenge
        };
    }
    weight
}

fn equality(point: &[K], target: &[K]) -> K {
    assert_eq!(point.len(), target.len(), "paper equality point length mismatch");
    let mut product = K::ONE;
    for (&left, &right) in point.iter().zip(target) {
        product *= (K::ONE - left) * (K::ONE - right) + left * right;
    }
    product
}

fn gamma_power(gamma: K, exponent: usize) -> K {
    let mut power = K::ONE;
    for _ in 0..exponent {
        power *= gamma;
    }
    power
}

fn carried_exponent(
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

fn packed_assignment<Ff>(witness: &Mat<Ff>, dims: JointDims) -> Result<Vec<K>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if witness.rows() != D || witness.cols() * D != dims.assignment_width {
        return Err(PiCcsError::InvalidInput(format!(
            "PaperExact witness shape is {}x{}, expected {D}x{}",
            witness.rows(),
            witness.cols(),
            dims.assignment_width / D
        )));
    }
    Ok((0..dims.assignment_width)
        .map(|column| K::from(witness[(column % D, column / D)]))
        .collect())
}

pub(super) fn validate_fresh_assignment<Ff>(
    witness: &Mat<Ff>,
    logical_width: usize,
    dims: JointDims,
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let assignment = packed_assignment(witness, dims)?;
    if assignment
        .iter()
        .skip(logical_width)
        .any(|&value| value != K::ZERO)
    {
        return Err(PiCcsError::InvalidInput(
            "PaperExact fresh assignment has a nonzero completed-carrier tail".into(),
        ));
    }
    Ok(())
}

pub(super) fn direct_public_input<Ff>(
    witness: &Mat<Ff>,
    logical_width: usize,
    public_width: usize,
) -> Result<Mat<Ff>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if public_width % D != 0 {
        return Err(PiCcsError::InvalidInput(
            "PaperExact requires the public input to contain whole ring elements".into(),
        ));
    }
    if witness.rows() != D || witness.cols() != logical_width.div_ceil(D) || public_width > logical_width {
        return Err(PiCcsError::InvalidInput(
            "PaperExact public-input projection shape mismatch".into(),
        ));
    }
    let active_columns = public_width / D;
    let mut output = Mat::zero(D, public_width, Ff::ZERO);
    for column in 0..active_columns {
        for row in 0..D {
            output[(row, column)] = witness[(row, column)];
        }
    }
    Ok(output)
}

fn ring_product(ring: &PaperRing, matrix_block: [Fq; D], assignment: &[K], block: usize) -> [K; D] {
    let mut assignment_block = [K::ZERO; D];
    for lane in 0..D {
        if let Some(value) = assignment.get(block * D + lane) {
            assignment_block[lane] = *value;
        }
    }
    ring.transformed_product(matrix_block, assignment_block)
}

fn direct_ring_row<Ff>(ring: &PaperRing, matrix: &CcsMatrix<Ff>, row: usize, assignment: &[K]) -> [K; D]
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
    K: From<Ff>,
{
    let mut output = [K::ZERO; D];
    for block in 0..assignment.len().div_ceil(D) {
        let mut matrix_block = [Fq::ZERO; D];
        for (lane, slot) in matrix_block.iter_mut().enumerate() {
            *slot = Fq::from_u64(matrix_entry(matrix, row, block * D + lane, ring).as_canonical_u64());
        }
        let product = ring_product(ring, matrix_block, assignment, block);
        for coefficient in 0..D {
            output[coefficient] += product[coefficient];
        }
    }
    output
}

fn identity_ring_row(ring: &PaperRing, row: usize, assignment: &[K]) -> [K; D] {
    if row >= assignment.len() {
        return [K::ZERO; D];
    }
    let block = row / D;
    let mut basis = [Fq::ZERO; D];
    basis[row % D] = Fq::ONE;
    ring_product(ring, basis, assignment, block)
}

pub(super) fn direct_ring_mle<Ff>(ring: &PaperRing, matrix: &CcsMatrix<Ff>, assignment: &[K], point: &[K]) -> [K; D]
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
    K: From<Ff>,
{
    let mut output = [K::ZERO; D];
    for row in 0..matrix.rows() {
        let weight = boolean_weight(point, row);
        let value = direct_ring_row(ring, matrix, row, assignment);
        for coefficient in 0..D {
            output[coefficient] += weight * value[coefficient];
        }
    }
    output
}

pub(super) fn direct_identity_ring_mle(ring: &PaperRing, assignment: &[K], point: &[K]) -> [K; D] {
    let mut output = [K::ZERO; D];
    for row in 0..assignment.len() {
        let weight = boolean_weight(point, row);
        let value = identity_ring_row(ring, row, assignment);
        for coefficient in 0..D {
            output[coefficient] += weight * value[coefficient];
        }
    }
    output
}

pub(super) fn initial_claim<Ff>(
    structure: &CcsStructure<Ff>,
    challenges: &Challenges,
    fresh_count: usize,
    running: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field,
{
    let matrix_count = structure.t() + 1;
    let running_count = running.len();
    let mut target = K::ZERO;
    for (running_index, claim) in running.iter().enumerate() {
        if claim.y_ring.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(format!(
                "PaperExact running claim {running_index} has {} matrix images, expected {matrix_count}",
                claim.y_ring.len()
            )));
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(
                    "PaperExact running matrix image is too short".into(),
                ));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                target += gamma_power(
                    challenges.gamma,
                    carried_exponent(
                        fresh_count,
                        running_count,
                        matrix_count,
                        running_index,
                        matrix,
                        coefficient,
                    ),
                ) * value;
            }
        }
    }
    Ok(target)
}

pub(super) fn terminal<Ff>(
    structure: &CcsStructure<Ff>,
    params: &neo_params::NeoParams,
    challenges: &Challenges,
    fresh_count: usize,
    prior_point: Option<&[K]>,
    point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput(
            "PaperExact output source count is too small".into(),
        ));
    }
    let matrix_count = structure.t() + 1;
    let mut fresh_residual = K::ZERO;
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        if output.ct.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "PaperExact fresh output matrix count mismatch".into(),
            ));
        }
        fresh_residual += gamma_power(challenges.gamma, source) * structure.f.eval_in_ext::<K>(&output.ct[1..]);
    }
    let mut norm = K::ZERO;
    for (source, output) in outputs.iter().enumerate() {
        let assignment_value = *output
            .ct
            .first()
            .ok_or_else(|| PiCcsError::InvalidInput("PaperExact identity output is missing".into()))?;
        norm += gamma_power(challenges.gamma, fresh_count + source) * range_product::<Ff>(assignment_value, params.b);
    }
    let running_count = outputs.len() - fresh_count;
    let mut carried = K::ZERO;
    for (running, output) in outputs.iter().skip(fresh_count).enumerate() {
        if output.y_ring.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "PaperExact carried output matrix count mismatch".into(),
            ));
        }
        for (matrix, coefficients) in output.y_ring.iter().enumerate() {
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                carried += gamma_power(
                    challenges.gamma,
                    carried_exponent(fresh_count, running_count, matrix_count, running, matrix, coefficient),
                ) * value;
            }
        }
    }
    let paper_part = equality(point, &challenges.alpha) * (fresh_residual + norm);
    let carried_part = prior_point.map_or(K::ZERO, |prior| equality(point, prior) * carried);
    Ok(paper_part + carried_part)
}

pub struct PaperJointOracle<'a, Ff> {
    structure: &'a CcsStructure<Ff>,
    params: &'a neo_params::NeoParams,
    fresh: &'a [CcsWitness<Ff>],
    running: &'a [Mat<Ff>],
    challenges: Challenges,
    prior_point: Option<Vec<K>>,
    dims: JointDims,
    round: usize,
    fixed: Vec<K>,
    ring: PaperRing,
}

impl<'a, Ff> PaperJointOracle<'a, Ff>
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
        dims: JointDims,
    ) -> Result<Self, PiCcsError> {
        if !challenges.has_expected_dimension(dims.variables) {
            return Err(PiCcsError::InvalidInput(
                "PaperExact joint challenge shape mismatch".into(),
            ));
        }
        if running.is_empty() != prior_point.is_none() {
            return Err(PiCcsError::InvalidInput(
                "PaperExact prior-point presence mismatch".into(),
            ));
        }
        for witness in fresh.iter().map(|value| &value.Z).chain(running) {
            let _ = packed_assignment(witness, dims)?;
        }
        Ok(Self {
            structure,
            params,
            fresh,
            running,
            challenges,
            prior_point: prior_point.map(<[K]>::to_vec),
            dims,
            round: 0,
            fixed: Vec::with_capacity(dims.variables),
            ring: PaperRing::new(),
        })
    }

    pub fn evaluate(&self, point: &[K]) -> K {
        let mut fresh_residual = K::ZERO;
        for (source, witness) in self.fresh.iter().enumerate() {
            let assignment = packed_assignment(&witness.Z, self.dims).expect("validated PaperExact witness");
            let application_values: Vec<K> = self
                .structure
                .matrices
                .iter()
                .map(|matrix| direct_ring_mle(&self.ring, matrix, &assignment, point)[0])
                .collect();
            fresh_residual +=
                gamma_power(self.challenges.gamma, source) * self.structure.f.eval_in_ext::<K>(&application_values);
        }

        let mut norm = K::ZERO;
        for (source, witness) in self
            .fresh
            .iter()
            .map(|value| &value.Z)
            .chain(self.running)
            .enumerate()
        {
            let assignment = packed_assignment(witness, self.dims).expect("validated PaperExact witness");
            let value = assignment
                .iter()
                .enumerate()
                .fold(K::ZERO, |sum, (row, &entry)| sum + boolean_weight(point, row) * entry);
            norm += gamma_power(self.challenges.gamma, self.fresh.len() + source)
                * range_product::<Ff>(value, self.params.b);
        }

        let matrix_count = self.structure.t() + 1;
        let running_count = self.running.len();
        let mut carried = K::ZERO;
        for (running, witness) in self.running.iter().enumerate() {
            let assignment = packed_assignment(witness, self.dims).expect("validated PaperExact witness");
            let identity = direct_identity_ring_mle(&self.ring, &assignment, point);
            for (coefficient, value) in identity.into_iter().enumerate() {
                carried += gamma_power(
                    self.challenges.gamma,
                    carried_exponent(self.fresh.len(), running_count, matrix_count, running, 0, coefficient),
                ) * value;
            }
            for (application, matrix) in self.structure.matrices.iter().enumerate() {
                for (coefficient, value) in direct_ring_mle(&self.ring, matrix, &assignment, point)
                    .into_iter()
                    .enumerate()
                {
                    carried += gamma_power(
                        self.challenges.gamma,
                        carried_exponent(
                            self.fresh.len(),
                            running_count,
                            matrix_count,
                            running,
                            application + 1,
                            coefficient,
                        ),
                    ) * value;
                }
            }
        }

        equality(point, &self.challenges.alpha) * (fresh_residual + norm)
            + self
                .prior_point
                .as_deref()
                .map_or(K::ZERO, |prior| equality(point, prior) * carried)
    }

    fn round_evaluations(&self, values: &[K]) -> Vec<K> {
        let remaining = self.dims.variables - self.round - 1;
        let tail_count = 1usize << remaining;
        values
            .iter()
            .map(|&value| {
                let mut sum = K::ZERO;
                for tail in 0..tail_count {
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

impl<Ff> RoundOracle for PaperJointOracle<'_, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.round_evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.dims.variables
    }

    fn degree_bound(&self) -> usize {
        self.dims.degree
    }

    fn fold(&mut self, challenge: K) {
        self.fixed.push(challenge);
        self.round += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_outputs<Ff, L>(
    structure: &CcsStructure<Ff>,
    fresh_claims: &[CcsClaim<Cmt, Ff>],
    fresh_witnesses: &[CcsWitness<Ff>],
    running_claims: &[CeClaim<Cmt, Ff, K>],
    running_witnesses: &[Mat<Ff>],
    point: &[K],
    dims: JointDims,
    commitment: &L,
) -> Result<Vec<CeClaim<Cmt, Ff, K>>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    L: neo_ccs::traits::SModuleHomomorphism<Ff, Cmt>,
{
    let _ = commitment;
    let ring = PaperRing::new();
    let d_pad = D.next_power_of_two();
    let openings = |witness: &Mat<Ff>| -> Result<(Vec<Vec<K>>, Vec<K>), PiCcsError> {
        let assignment = packed_assignment(witness, dims)?;
        let mut y_ring = Vec::with_capacity(dims.matrix_count);
        let mut identity = direct_identity_ring_mle(&ring, &assignment, point).to_vec();
        identity.resize(d_pad, K::ZERO);
        y_ring.push(identity);
        for matrix in &structure.matrices {
            let mut coefficients = direct_ring_mle(&ring, matrix, &assignment, point).to_vec();
            coefficients.resize(d_pad, K::ZERO);
            y_ring.push(coefficients);
        }
        let ct = y_ring.iter().map(|coefficients| coefficients[0]).collect();
        Ok((y_ring, ct))
    };

    let mut outputs = Vec::with_capacity(fresh_claims.len() + running_claims.len());
    for (claim, witness) in fresh_claims.iter().zip(fresh_witnesses) {
        let (y_ring, ct) = openings(&witness.Z)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: direct_public_input(&witness.Z, structure.m, claim.m_in)?,
            r: point.to_vec(),
            y_ring,
            ct,
            m_in: claim.m_in,
            fold_digest: [0u8; 32],
            adv: claim.adv.clone(),
        });
    }
    for (claim, witness) in running_claims.iter().zip(running_witnesses) {
        let (y_ring, ct) = openings(witness)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: claim.X.clone(),
            r: point.to_vec(),
            y_ring,
            ct,
            m_in: claim.m_in,
            fold_digest: [0u8; 32],
            adv: claim.adv.clone(),
        });
    }
    Ok(outputs)
}
