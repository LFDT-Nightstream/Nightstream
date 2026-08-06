//! Optimized one-joint PiCCS evaluator for the padded-row paper protocol.
//!
//! Application matrices use the production cache. The virtual padded
//! identity and the joint polynomial remain explicit protocol data.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{superneo_bar_block, Fq, KExtensions, Rq, D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engines::pi_ccs_joint::{
    carried_gamma_exponent, equality, gamma_power, range_product, JointDims, ProtocolTrace,
};
use crate::engines::pi_ccs_joint_protocol::{self, TranscriptBinding};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;
use crate::superneo_eval::SuperneoZBlocks;

use super::OptimizedStructureCache;

fn fold_table(table: &mut Vec<K>, challenge: K) {
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[2 * index];
        table[index] = low + (table[2 * index + 1] - low) * challenge;
    }
    table.truncate(half);
}

fn at(table: &[K], pair: usize, point: K) -> K {
    let low = table[2 * pair];
    low + (table[2 * pair + 1] - low) * point
}

fn identity_ring_row(row: usize, assignment: &[K]) -> [K; D] {
    if row >= assignment.len() {
        return [K::ZERO; D];
    }
    let block = row / D;
    let mut basis = [Fq::ZERO; D];
    basis[row % D] = Fq::ONE;
    let transformed = Rq(superneo_bar_block(basis));
    let mut real = [Fq::ZERO; D];
    let mut imaginary = [Fq::ZERO; D];
    for lane in 0..D {
        let [low, high] = assignment[block * D + lane].as_coeffs();
        real[lane] = low;
        imaginary[lane] = high;
    }
    let real_product = transformed.mul(&Rq(real));
    let imaginary_product = transformed.mul(&Rq(imaginary));
    std::array::from_fn(|coefficient| K::from_coeffs([real_product.0[coefficient], imaginary_product.0[coefficient]]))
}

pub(crate) fn identity_ring_mle(assignment: &[K], weights: &[K]) -> [K; D] {
    let mut output = [K::ZERO; D];
    for (row, &weight) in weights.iter().take(assignment.len()).enumerate() {
        let value = identity_ring_row(row, assignment);
        for coefficient in 0..D {
            output[coefficient] += weight * value[coefficient];
        }
    }
    output
}

pub(crate) fn initial_claim<Ff>(
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
                "optimized running claim {running_index} has {} matrix images, expected {matrix_count}",
                claim.y_ring.len()
            )));
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(
                    "optimized running matrix image is too short".into(),
                ));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                target += gamma_power(
                    challenges.gamma,
                    carried_gamma_exponent(
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

pub(crate) fn terminal<Ff>(
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
            "optimized output source count is too small".into(),
        ));
    }
    let matrix_count = structure.t() + 1;
    let mut fresh_residual = K::ZERO;
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        if output.ct.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized fresh output matrix count mismatch".into(),
            ));
        }
        fresh_residual += gamma_power(challenges.gamma, source) * structure.f.eval_in_ext::<K>(&output.ct[1..]);
    }
    let mut norm = K::ZERO;
    for (source, output) in outputs.iter().enumerate() {
        let assignment = *output
            .ct
            .first()
            .ok_or_else(|| PiCcsError::InvalidInput("optimized identity output is missing".into()))?;
        norm += gamma_power(challenges.gamma, fresh_count + source) * range_product::<Ff>(assignment, params.b);
    }
    let running_count = outputs.len() - fresh_count;
    let mut carried = K::ZERO;
    for (running, output) in outputs.iter().skip(fresh_count).enumerate() {
        if output.y_ring.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized carried output matrix count mismatch".into(),
            ));
        }
        for (matrix, coefficients) in output.y_ring.iter().enumerate() {
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                carried += gamma_power(
                    challenges.gamma,
                    carried_gamma_exponent(fresh_count, running_count, matrix_count, running, matrix, coefficient),
                ) * value;
            }
        }
    }
    Ok(equality(point, &challenges.alpha) * (fresh_residual + norm)
        + prior_point.map_or(K::ZERO, |prior| equality(point, prior) * carried))
}

pub struct OptimizedPaperJointOracle<'a> {
    structure: &'a CcsStructure<F>,
    base: u32,
    challenges: Challenges,
    dims: JointDims,
    round: usize,
    equality_alpha: Vec<K>,
    equality_prior: Option<Vec<K>>,
    fresh_application_tables: Vec<Vec<Vec<K>>>,
    assignment_tables: Vec<Vec<K>>,
    norm_weights: Vec<K>,
    carried_table: Vec<K>,
}

impl<'a> OptimizedPaperJointOracle<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<F>,
        params: &neo_params::NeoParams,
        fresh: &[CcsWitness<F>],
        running: &[Mat<F>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        dims: JointDims,
        cache: &OptimizedStructureCache,
    ) -> Result<Self, PiCcsError> {
        if !challenges.has_expected_dimension(dims.variables) {
            return Err(PiCcsError::InvalidInput(
                "optimized joint challenge shape mismatch".into(),
            ));
        }
        if running.is_empty() != prior_point.is_none() {
            return Err(PiCcsError::InvalidInput(
                "optimized prior-point presence mismatch".into(),
            ));
        }
        let matrices = cache.superneo().matrix_caches();
        if matrices.len() != structure.t() {
            return Err(PiCcsError::ProtocolError(
                "optimized matrix-cache count mismatch".into(),
            ));
        }

        let decode = |witness: &Mat<F>| crate::common::decode_superneo_coeffs_from_witness_mat(witness, structure.m);
        let mut fresh_application_tables = Vec::with_capacity(fresh.len());
        for witness in fresh {
            let assignment = decode(&witness.Z)?;
            let blocks = SuperneoZBlocks::from_z(&assignment);
            let mut per_matrix = Vec::with_capacity(structure.t());
            for matrix in matrices {
                let mut table = vec![K::ZERO; dims.row_count];
                for (row, value) in table.iter_mut().take(structure.n).enumerate() {
                    *value = matrix.row_dot_with_blocks(row, &blocks);
                }
                per_matrix.push(table);
            }
            fresh_application_tables.push(per_matrix);
        }

        let mut assignments = Vec::with_capacity(fresh.len() + running.len());
        for witness in fresh.iter().map(|value| &value.Z).chain(running) {
            let assignment = decode(witness)?;
            let mut table = vec![K::ZERO; dims.row_count];
            table[..dims.assignment_width].copy_from_slice(&assignment);
            assignments.push(table);
        }
        let norm_weights = (0..assignments.len())
            .map(|source| gamma_power(challenges.gamma, fresh.len() + source))
            .collect();

        let matrix_count = structure.t() + 1;
        let mut carried_table = vec![K::ZERO; dims.row_count];
        for (running_index, witness) in running.iter().enumerate() {
            let assignment = decode(witness)?;
            let blocks = SuperneoZBlocks::from_z(&assignment);
            for row in 0..dims.assignment_width {
                for (coefficient, value) in identity_ring_row(row, &assignment).into_iter().enumerate() {
                    carried_table[row] += gamma_power(
                        challenges.gamma,
                        carried_gamma_exponent(fresh.len(), running.len(), matrix_count, running_index, 0, coefficient),
                    ) * value;
                }
            }
            for (application, matrix) in matrices.iter().enumerate() {
                for (row, slot) in carried_table.iter_mut().take(structure.n).enumerate() {
                    for (coefficient, value) in matrix
                        .row_dot_ring_with_blocks(row, &blocks)
                        .into_iter()
                        .enumerate()
                    {
                        *slot += gamma_power(
                            challenges.gamma,
                            carried_gamma_exponent(
                                fresh.len(),
                                running.len(),
                                matrix_count,
                                running_index,
                                application + 1,
                                coefficient,
                            ),
                        ) * value;
                    }
                }
            }
        }

        Ok(Self {
            structure,
            base: params.b,
            challenges: challenges.clone(),
            dims,
            round: 0,
            equality_alpha: neo_ccs::utils::tensor_point::<K>(&challenges.alpha),
            equality_prior: prior_point.map(neo_ccs::utils::tensor_point::<K>),
            fresh_application_tables,
            assignment_tables: assignments,
            norm_weights,
            carried_table,
        })
    }

    fn evaluations(&self, points: &[K]) -> Vec<K> {
        let pairs = self.equality_alpha.len() / 2;
        let mut output = vec![K::ZERO; points.len()];
        let mut application_values = vec![K::ZERO; self.structure.t()];
        for pair in 0..pairs {
            for (point_index, &point) in points.iter().enumerate() {
                let mut fresh_residual = K::ZERO;
                for (source, tables) in self.fresh_application_tables.iter().enumerate() {
                    for (matrix, table) in tables.iter().enumerate() {
                        application_values[matrix] = at(table, pair, point);
                    }
                    fresh_residual += gamma_power(self.challenges.gamma, source)
                        * self.structure.f.eval_in_ext::<K>(&application_values);
                }
                let mut norm = K::ZERO;
                for (table, &weight) in self.assignment_tables.iter().zip(&self.norm_weights) {
                    norm += weight * range_product::<F>(at(table, pair, point), self.base);
                }
                let carried_gate = self
                    .equality_prior
                    .as_ref()
                    .map_or(K::ZERO, |table| at(table, pair, point));
                output[point_index] += at(&self.equality_alpha, pair, point) * (fresh_residual + norm)
                    + carried_gate * at(&self.carried_table, pair, point);
            }
        }
        output
    }
}

impl RoundOracle for OptimizedPaperJointOracle<'_> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.dims.variables
    }

    fn degree_bound(&self) -> usize {
        self.dims.degree
    }

    fn fold(&mut self, challenge: K) {
        fold_table(&mut self.equality_alpha, challenge);
        if let Some(table) = &mut self.equality_prior {
            fold_table(table, challenge);
        }
        for source in &mut self.fresh_application_tables {
            for table in source {
                fold_table(table, challenge);
            }
        }
        for table in &mut self.assignment_tables {
            fold_table(table, challenge);
        }
        fold_table(&mut self.carried_table, challenge);
        self.round += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_outputs<L>(
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    point: &[K],
    dims: JointDims,
    _commitment: &L,
    cache: &OptimizedStructureCache,
) -> Result<Vec<CeClaim<Cmt, F, K>>, PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    let weights = neo_ccs::utils::tensor_point::<K>(point);
    let d_pad = D.next_power_of_two();
    let openings = |witness: &Mat<F>| -> Result<(Vec<Vec<K>>, Vec<K>), PiCcsError> {
        let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, structure.m)?;
        let mut y_ring = Vec::with_capacity(dims.matrix_count);
        let mut identity = identity_ring_mle(&assignment, &weights).to_vec();
        identity.resize(d_pad, K::ZERO);
        y_ring.push(identity);
        y_ring.extend(
            crate::superneo_eval::eval_all_mats_ring_cached(cache.superneo(), &assignment, &weights, structure.n)
                .into_iter()
                .map(|coefficients| {
                    let mut row = coefficients.to_vec();
                    row.resize(d_pad, K::ZERO);
                    row
                }),
        );
        let ct = y_ring.iter().map(|row| row[0]).collect();
        Ok((y_ring, ct))
    };

    let mut outputs = Vec::with_capacity(fresh_claims.len() + running_claims.len());
    for (claim, witness) in fresh_claims.iter().zip(fresh_witnesses) {
        let (y_ring, ct) = openings(&witness.Z)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: crate::common::project_x_from_witness_mat(&witness.Z, structure.m, claim.m_in)?,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_with_trace<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    cache: &OptimizedStructureCache,
    binding: TranscriptBinding,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        super::PiCcsProvePerf,
        ProtocolTrace,
    ),
    PiCcsError,
> {
    let started = std::time::Instant::now();
    cache.validate_shape(structure)?;
    if fresh_claims.len() != fresh_witnesses.len() || running_claims.len() != running_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "optimized claim/witness count mismatch".into(),
        ));
    }
    for (index, witness) in fresh_witnesses.iter().enumerate() {
        crate::common::validate_fresh_witness_tail_zero(
            &witness.Z,
            structure.m,
            &format!("optimized fresh witness {index}"),
        )?;
    }
    for witness in running_witnesses {
        crate::common::validate_superneo_witness_mat(witness, structure.m)?;
    }
    let bind_started = std::time::Instant::now();
    let mut trace = ProtocolTrace::default();
    let (dims, challenges) = pi_ccs_joint_protocol::bind_and_sample_with_trace(
        transcript,
        &mut trace,
        params,
        structure,
        fresh_claims,
        running_claims,
        binding,
        Some(cache.pi_ccs_matrix_digest()),
    )?;
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.variables)?;
    let initial = initial_claim(structure, &challenges, fresh_claims.len(), running_claims)?;
    let sumcheck_started = std::time::Instant::now();
    let mut oracle = OptimizedPaperJointOracle::new(
        structure,
        params,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        prior_point,
        dims,
        cache,
    )?;
    let (rounds, round_challenges, final_claim) =
        pi_ccs_joint_protocol::prove_phase(transcript, &mut trace, initial, &mut oracle)?;
    let sumcheck_ms = sumcheck_started.elapsed().as_secs_f64() * 1_000.0;
    let output_started = std::time::Instant::now();
    let mut outputs = build_outputs(
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &round_challenges,
        dims,
        commitment,
        cache,
    )?;
    let expected = terminal::<F>(
        structure,
        params,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &round_challenges,
        &outputs,
    )?;
    if final_claim != expected {
        return Err(PiCcsError::SumcheckError(
            "optimized terminal value does not match the paper output message".into(),
        ));
    }
    let digest = pi_ccs_joint_protocol::absorb_outputs(transcript, &mut trace, &outputs, fresh_claims.len(), dims)?;
    for output in &mut outputs {
        output.fold_digest = digest;
    }
    let proof = pi_ccs_joint_protocol::assemble_proof(rounds);
    let output_materialize_ms = output_started.elapsed().as_secs_f64() * 1_000.0;
    let perf = super::PiCcsProvePerf {
        bind_ms,
        sample_challenges_ms: 0.0,
        sumcheck_ms,
        output_materialize_ms,
        total_ms: started.elapsed().as_secs_f64() * 1_000.0,
    };
    Ok((outputs, proof, perf, trace))
}

#[allow(clippy::too_many_arguments)]
pub fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, super::PiCcsProvePerf), PiCcsError> {
    let (outputs, proof, perf, _) = prove_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
        TranscriptBinding::claims(),
    )?;
    Ok((outputs, proof, perf))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_with_binding<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    cache: &OptimizedStructureCache,
    binding: TranscriptBinding,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, super::PiCcsProvePerf), PiCcsError> {
    let (outputs, proof, perf, _) = prove_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
        binding,
    )?;
    Ok((outputs, proof, perf))
}
