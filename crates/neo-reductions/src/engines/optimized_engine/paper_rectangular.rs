//! Optimized evaluators for canonical rectangular-paper `Pi_CCS`.
//!
//! These oracles precompute Boolean row or column tables and fold them in
//! place. Their algebra and coefficient order match the independent direct
//! reference, but their matrix evaluation uses the production SuperNeo cache.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_protocol::{
    carried_gamma_exponent, fe_initial_claim, fe_terminal, gamma_power, nc_terminal, Challenges, PiCcsProof,
};
use crate::engines::pi_ccs_rectangular;
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;
use crate::superneo_eval::SuperneoZBlocks;

use super::OptimizedStructureCache;

#[inline]
fn fold_table(table: &mut Vec<K>, challenge: K) {
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[2 * index];
        table[index] = low + (table[2 * index + 1] - low) * challenge;
    }
    table.truncate(half);
}

#[inline]
fn at(table: &[K], pair: usize, point: K) -> K {
    let low = table[2 * pair];
    low + (table[2 * pair + 1] - low) * point
}

#[inline]
fn range_product(value: K, base: u32) -> K {
    let mut product = K::ONE;
    for integer in -((base as i64) - 1)..=((base as i64) - 1) {
        product *= value - K::from(F::from_i64(integer));
    }
    product
}

fn packed_blocks(witness: &Mat<F>, width: usize) -> Result<SuperneoZBlocks, PiCcsError> {
    let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, width)?;
    Ok(SuperneoZBlocks::from_z(&assignment))
}

/// Streaming row-domain oracle for the exact FE polynomial.
pub struct OptimizedPaperRectangularFeOracle<'a> {
    structure: &'a CcsStructure<F>,
    challenges: Challenges,
    degree: usize,
    rounds: usize,
    round: usize,
    equality_row: Vec<K>,
    equality_prior: Option<Vec<K>>,
    fresh_matrix_tables: Vec<Vec<Vec<K>>>,
    carried_table: Vec<K>,
}

impl<'a> OptimizedPaperRectangularFeOracle<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<F>,
        fresh: &[CcsWitness<F>],
        running: &[Mat<F>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        variables: usize,
        degree: usize,
        cache: &OptimizedStructureCache,
    ) -> Result<Self, PiCcsError> {
        if challenges.beta_r.len() != variables || !challenges.is_paper_rectangular() {
            return Err(PiCcsError::InvalidInput(
                "optimized rectangular FE challenge shape mismatch".into(),
            ));
        }
        if running.is_empty() != prior_point.is_none() {
            return Err(PiCcsError::InvalidInput(
                "optimized rectangular FE prior-point presence mismatch".into(),
            ));
        }
        let row_count = 1usize << variables;
        let equality_row = neo_ccs::utils::tensor_point::<K>(&challenges.beta_r);
        let equality_prior = prior_point.map(neo_ccs::utils::tensor_point::<K>);
        let matrix_caches = cache.superneo().matrix_caches();
        if matrix_caches.len() != structure.t() {
            return Err(PiCcsError::ProtocolError(
                "optimized rectangular FE matrix-cache count mismatch".into(),
            ));
        }

        let mut fresh_matrix_tables = Vec::with_capacity(fresh.len());
        for witness in fresh {
            let blocks = packed_blocks(&witness.Z, structure.m)?;
            let mut per_matrix = Vec::with_capacity(structure.t());
            for matrix in matrix_caches {
                let mut table = vec![K::ZERO; row_count];
                for (row, slot) in table.iter_mut().take(structure.n).enumerate() {
                    *slot = matrix.row_dot_with_blocks(row, &blocks);
                }
                per_matrix.push(table);
            }
            fresh_matrix_tables.push(per_matrix);
        }

        let mut carried_table = vec![K::ZERO; row_count];
        let running_count = running.len();
        for (running_index, witness) in running.iter().enumerate() {
            let blocks = packed_blocks(witness, structure.m)?;
            for (matrix_index, matrix) in matrix_caches.iter().enumerate() {
                for (row, slot) in carried_table.iter_mut().take(structure.n).enumerate() {
                    let coefficients = matrix.row_dot_ring_with_blocks(row, &blocks);
                    for (coefficient, value) in coefficients.into_iter().enumerate() {
                        let exponent = carried_gamma_exponent(
                            fresh.len(),
                            running_count,
                            structure.t(),
                            running_index,
                            matrix_index,
                            coefficient,
                        );
                        *slot += gamma_power(challenges.gamma, exponent) * value;
                    }
                }
            }
        }

        Ok(Self {
            structure,
            challenges,
            degree,
            rounds: variables,
            round: 0,
            equality_row,
            equality_prior,
            fresh_matrix_tables,
            carried_table,
        })
    }

    fn evaluations(&self, points: &[K]) -> Vec<K> {
        let pairs = self.equality_row.len() / 2;
        let mut output = vec![K::ZERO; points.len()];
        let mut matrix_values = vec![K::ZERO; self.structure.t()];
        for pair in 0..pairs {
            for (point_index, &point) in points.iter().enumerate() {
                let mut fresh_ccs = K::ZERO;
                for (source, tables) in self.fresh_matrix_tables.iter().enumerate() {
                    for (matrix, table) in tables.iter().enumerate() {
                        matrix_values[matrix] = at(table, pair, point);
                    }
                    fresh_ccs +=
                        gamma_power(self.challenges.gamma, source) * self.structure.f.eval_in_ext::<K>(&matrix_values);
                }
                let carried_gate = self
                    .equality_prior
                    .as_ref()
                    .map_or(K::ZERO, |table| at(table, pair, point));
                output[point_index] += at(&self.equality_row, pair, point) * fresh_ccs
                    + carried_gate * at(&self.carried_table, pair, point);
            }
        }
        output
    }
}

impl RoundOracle for OptimizedPaperRectangularFeOracle<'_> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn fold(&mut self, challenge: K) {
        fold_table(&mut self.equality_row, challenge);
        if let Some(table) = self.equality_prior.as_mut() {
            fold_table(table, challenge);
        }
        for source in &mut self.fresh_matrix_tables {
            for table in source {
                fold_table(table, challenge);
            }
        }
        fold_table(&mut self.carried_table, challenge);
        self.round += 1;
    }
}

/// Streaming column-domain oracle for the exact NC polynomial.
pub struct OptimizedPaperRectangularNcOracle {
    base: u32,
    degree: usize,
    rounds: usize,
    round: usize,
    equality_column: Vec<K>,
    assignments: Vec<Vec<K>>,
    source_weights: Vec<K>,
}

impl OptimizedPaperRectangularNcOracle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &CcsStructure<F>,
        params: &neo_params::NeoParams,
        fresh: &[CcsWitness<F>],
        running: &[Mat<F>],
        challenges: Challenges,
        variables: usize,
        degree: usize,
    ) -> Result<Self, PiCcsError> {
        if challenges.beta_m.len() != variables || !challenges.is_paper_rectangular() {
            return Err(PiCcsError::InvalidInput(
                "optimized rectangular NC challenge shape mismatch".into(),
            ));
        }
        let column_count = 1usize << variables;
        let mut assignments = Vec::with_capacity(fresh.len() + running.len());
        for witness in fresh.iter().map(|witness| &witness.Z).chain(running) {
            crate::common::validate_superneo_witness_mat(witness, structure.m)?;
            let mut table = vec![K::ZERO; column_count];
            for (column, value) in table.iter_mut().take(structure.m).enumerate() {
                *value = crate::common::witness_mat_get_k(witness, structure.m, column % D, column);
            }
            assignments.push(table);
        }
        let source_weights = (0..assignments.len())
            .map(|source| gamma_power(challenges.gamma, fresh.len() + source))
            .collect();
        Ok(Self {
            base: params.b,
            degree,
            rounds: variables,
            round: 0,
            equality_column: neo_ccs::utils::tensor_point::<K>(&challenges.beta_m),
            assignments,
            source_weights,
        })
    }

    fn evaluations(&self, points: &[K]) -> Vec<K> {
        let pairs = self.equality_column.len() / 2;
        let mut output = vec![K::ZERO; points.len()];
        for pair in 0..pairs {
            for (point_index, &point) in points.iter().enumerate() {
                let mut norm = K::ZERO;
                for (table, &weight) in self.assignments.iter().zip(&self.source_weights) {
                    norm += weight * range_product(at(table, pair, point), self.base);
                }
                output[point_index] += at(&self.equality_column, pair, point) * norm;
            }
        }
        output
    }
}

impl RoundOracle for OptimizedPaperRectangularNcOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.evaluations(points)
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn fold(&mut self, challenge: K) {
        fold_table(&mut self.equality_column, challenge);
        for table in &mut self.assignments {
            fold_table(table, challenge);
        }
        self.round += 1;
    }
}

/// Build output openings through the optimized evaluator cache.
#[allow(clippy::too_many_arguments)]
pub fn build_outputs<L>(
    structure: &CcsStructure<F>,
    params: &neo_params::NeoParams,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    row_point: &[K],
    column_point: &[K],
    fold_digest: [u8; 32],
    _commitment: &L,
    cache: &OptimizedStructureCache,
) -> Result<Vec<CeClaim<Cmt, F, K>>, PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    let row_weights = neo_ccs::utils::tensor_point::<K>(row_point);
    let column_weights = neo_ccs::utils::tensor_point::<K>(column_point);
    let d_pad = D.next_power_of_two();
    let openings = |witness: &Mat<F>| -> Result<(Vec<Vec<K>>, Vec<K>, Vec<K>), PiCcsError> {
        let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, structure.m)?;
        let y_ring: Vec<Vec<K>> =
            crate::superneo_eval::eval_all_mats_ring_cached(cache.superneo(), &assignment, &row_weights, structure.n)
                .into_iter()
                .map(|coefficients| {
                    let mut coefficients = coefficients.to_vec();
                    coefficients.resize(d_pad, K::ZERO);
                    coefficients
                })
                .collect();
        let ct = y_ring.iter().map(|coefficients| coefficients[0]).collect();
        let y_zcol = crate::common::compute_y_zcol_from_witness(params, witness, structure.m, &column_weights, d_pad)?;
        Ok((y_ring, ct, y_zcol))
    };

    let mut outputs = Vec::with_capacity(fresh_claims.len() + running_claims.len());
    for (claim, witness) in fresh_claims.iter().zip(fresh_witnesses) {
        let (y_ring, ct, y_zcol) = openings(&witness.Z)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: crate::common::project_x_from_witness_mat(&witness.Z, structure.m, claim.m_in)?,
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

/// Prove the canonical rectangular-paper algebra with optimized Boolean
/// tables. The transcript and proof assembly are shared with PaperExact.
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
    prove_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
        pi_ccs_rectangular::TranscriptBinding::claims(),
    )
}

/// Prove under a caller-recomputed public transcript binding.
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
    binding: pi_ccs_rectangular::TranscriptBinding,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, super::PiCcsProvePerf), PiCcsError> {
    let total_started = std::time::Instant::now();
    if fresh_claims.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "optimized rectangular prove: empty fresh claim list".into(),
        ));
    }
    if fresh_claims.len() != fresh_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "optimized rectangular prove: fresh claim/witness count mismatch".into(),
        ));
    }
    if running_claims.len() != running_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "optimized rectangular prove: running claim/witness count mismatch".into(),
        ));
    }

    let bind_started = std::time::Instant::now();
    let (dims, challenges) = pi_ccs_rectangular::bind_and_sample_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        binding,
    )?;
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.ell_n)?;
    let initial_fe = fe_initial_claim(structure, &challenges, fresh_claims.len(), running_claims)?;

    let fe_started = std::time::Instant::now();
    let mut fe_oracle = OptimizedPaperRectangularFeOracle::new(
        structure,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        prior_point,
        dims.ell_n,
        dims.d_sc,
        cache,
    )?;
    let fe = pi_ccs_rectangular::prove_phase(
        transcript,
        crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
        initial_fe,
        &mut fe_oracle,
    )?;
    let fe_sumcheck_ms = fe_started.elapsed().as_secs_f64() * 1_000.0;

    let nc_started = std::time::Instant::now();
    let mut nc_oracle = OptimizedPaperRectangularNcOracle::new(
        structure,
        params,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        dims.ell_m,
        dims.d_sc,
    )?;
    let nc = pi_ccs_rectangular::prove_phase(
        transcript,
        crate::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
        K::ZERO,
        &mut nc_oracle,
    )?;
    let nc_sumcheck_ms = nc_started.elapsed().as_secs_f64() * 1_000.0;

    let output_started = std::time::Instant::now();
    let fold_digest = transcript.digest32();
    let outputs = build_outputs(
        structure,
        params,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &fe.challenges,
        &nc.challenges,
        fold_digest,
        commitment,
        cache,
    )?;
    let expected_fe = fe_terminal(
        structure,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &fe.challenges,
        &outputs,
    )?;
    let expected_nc = nc_terminal::<F>(params, &challenges, fresh_claims.len(), &nc.challenges, &outputs)?;
    if fe.final_claim != expected_fe || nc.final_claim != expected_nc {
        return Err(PiCcsError::SumcheckError(
            "optimized rectangular terminal evaluation does not match output openings".into(),
        ));
    }
    let proof = pi_ccs_rectangular::assemble_proof(challenges, initial_fe, fe, nc, fold_digest);
    let output_materialize_ms = output_started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
        outputs,
        proof,
        super::PiCcsProvePerf {
            bind_ms,
            sample_challenges_ms: 0.0,
            fe_sumcheck_ms,
            nc_sumcheck_ms,
            output_materialize_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}
