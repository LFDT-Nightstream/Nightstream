//! Optimized SuperNeo v1.1 PiCCS evaluator.
//!
//! Application matrices use the production cache. Pad remains the separate
//! `Eval_K` family; genuine matrices remain the `Eval_A` family.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{superneo_bar_block, Fq, KExtensions, Rq, D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};

pub use super::cpu_oracle::OptimizedPaperJointOracle;
use crate::engines::pi_ccs_joint::{
    equality, eval_a_gamma_exponent, eval_k_gamma_exponent, gamma_power, range_product, JointDims, ProtocolTrace,
    TerminalComponents,
};
use crate::engines::pi_ccs_joint_protocol::{self, PaperJointRoundOracle, TranscriptBinding, V1_1OutputOpening};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;
use crate::superneo_eval::SuperneoEvalCache;

use super::OptimizedStructureCache;

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
    let matrix_count = structure.t();
    let running_count = running.len();
    let mut eval_k = K::ZERO;
    let mut eval_a = K::ZERO;
    for (running_index, claim) in running.iter().enumerate() {
        if claim.eval_k.len() < D || claim.eval_a.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized running claim {running_index} does not have separate Eval_K and Eval_A"
            )));
        }
        for (coefficient, &value) in claim.eval_k.iter().take(D).enumerate() {
            eval_k += gamma_power(
                challenges.gamma,
                eval_k_gamma_exponent(running_count, running_index, coefficient),
            ) * value;
        }
        for (matrix, coefficients) in claim.eval_a.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(
                    "optimized running Eval_A image is too short".into(),
                ));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                eval_a += gamma_power(
                    challenges.gamma,
                    eval_a_gamma_exponent(running_count, matrix_count, running_index, matrix, coefficient),
                ) * value;
            }
        }
    }
    let _ = fresh_count;
    Ok(eval_k + gamma_power(challenges.gamma, running_count * D) * eval_a)
}

pub(crate) fn terminal_components<Ff>(
    structure: &CcsStructure<Ff>,
    params: &neo_params::NeoParams,
    challenges: &Challenges,
    fresh_count: usize,
    prior_point: Option<&[K]>,
    point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<TerminalComponents, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput(
            "optimized output source count is too small".into(),
        ));
    }
    let matrix_count = structure.t();
    let mut fresh_residual = K::ZERO;
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        if output.eval_a.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized fresh output matrix count mismatch".into(),
            ));
        }
        let matrix_evaluations = output
            .eval_a
            .iter()
            .map(|evaluation| evaluation.first().copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| PiCcsError::InvalidInput("optimized Eval_A output is empty".into()))?;
        fresh_residual += gamma_power(challenges.gamma, source) * structure.f.eval_in_ext::<K>(&matrix_evaluations);
    }
    let mut norm = K::ZERO;
    for (source, output) in outputs.iter().enumerate() {
        let assignment = *output
            .eval_k
            .first()
            .ok_or_else(|| PiCcsError::InvalidInput("optimized Eval_K output is missing".into()))?;
        norm += gamma_power(challenges.gamma, source) * range_product::<Ff>(assignment, params.b);
    }
    let running_count = outputs.len() - fresh_count;
    let mut eval_k = K::ZERO;
    let mut eval_a = K::ZERO;
    for (running, output) in outputs.iter().skip(fresh_count).enumerate() {
        if output.eval_k.len() < D || output.eval_a.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized carried output Eval_K/Eval_A shape mismatch".into(),
            ));
        }
        for (coefficient, &value) in output.eval_k.iter().take(D).enumerate() {
            eval_k += gamma_power(
                challenges.gamma,
                eval_k_gamma_exponent(running_count, running, coefficient),
            ) * value;
        }
        for (matrix, coefficients) in output.eval_a.iter().enumerate() {
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                eval_a += gamma_power(
                    challenges.gamma,
                    eval_a_gamma_exponent(running_count, matrix_count, running, matrix, coefficient),
                ) * value;
            }
        }
    }
    let prior_equality = prior_point.map_or(K::ZERO, |prior| equality(point, prior));
    let eval_a_shift = gamma_power(challenges.gamma, running_count * D);
    let constraint_shift = gamma_power(challenges.gamma, running_count * D * (matrix_count + 1));
    let eval_k = prior_equality * eval_k;
    let eval_a = prior_equality * eval_a;
    let terminal = eval_k
        + eval_a_shift * eval_a
        + constraint_shift
            * equality(point, &challenges.alpha)
            * (fresh_residual + gamma_power(challenges.gamma, fresh_count) * norm);
    Ok(TerminalComponents {
        eval_k,
        eval_a,
        ccs: fresh_residual,
        norm,
        terminal,
    })
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
    Ok(terminal_components(structure, params, challenges, fresh_count, prior_point, point, outputs)?.terminal)
}

/// Canonical inputs available to an implementation of the one-joint oracle.
///
/// These values are prover inputs. The canonical driver remains responsible
/// for the transcript and for all verifier-visible proof messages.
pub struct PaperJointOracleInput<'a> {
    pub structure: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    pub fresh_witnesses: &'a [CcsWitness<F>],
    pub running_witnesses: &'a [Mat<F>],
    pub challenges: Challenges,
    pub prior_point: Option<&'a [K]>,
    pub dims: JointDims,
    pub cache: &'a OptimizedStructureCache,
}

/// Factory for a protocol-neutral one-joint evaluator.
pub trait PaperJointOracleBackend {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError>;

    /// Evaluate separate v1_1 PiDEC child openings with the same static
    /// matrix plan. The canonical PiDEC prover checks their radix
    /// recomposition before it returns a proof.
    fn dec_openings(
        &mut self,
        _cache: &OptimizedStructureCache,
        _witnesses: &[Mat<F>],
        _point: &[K],
        _assignment_width: usize,
    ) -> Result<Option<Vec<V1_1OutputOpening>>, PiCcsError> {
        Ok(None)
    }
}

enum OracleSource<'a> {
    Cached {
        cache: &'a OptimizedStructureCache,
        backend: Option<&'a mut dyn PaperJointOracleBackend>,
    },
    Rows {
        cache: &'a SuperneoEvalCache,
    },
    Complete {
        oracle: &'a mut dyn PaperJointRoundOracle,
        challenges: &'a Challenges,
    },
}

#[allow(clippy::too_many_arguments)]
fn build_outputs(
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    point: &[K],
    dims: JointDims,
    cache: Option<&OptimizedStructureCache>,
    precomputed_openings: Option<&[V1_1OutputOpening]>,
) -> Result<Vec<CeClaim<Cmt, F, K>>, PiCcsError> {
    if precomputed_openings.is_none() && cache.is_none() {
        return Err(PiCcsError::ProtocolError(
            "complete optimized oracle did not return output openings".into(),
        ));
    }
    let weights = precomputed_openings
        .is_none()
        .then(|| neo_ccs::utils::tensor_point::<K>(point));
    let d_pad = D.next_power_of_two();
    let source_count = fresh_claims.len() + running_claims.len();
    if precomputed_openings.is_some_and(|openings| openings.len() != source_count) {
        return Err(PiCcsError::ProtocolError(
            "optimized opening backend returned the wrong source count".into(),
        ));
    }
    let openings = |source: usize, witness: &Mat<F>| -> Result<V1_1OutputOpening, PiCcsError> {
        if let Some(precomputed) = precomputed_openings {
            let opening = precomputed
                .get(source)
                .ok_or_else(|| PiCcsError::ProtocolError("optimized opening source is missing".into()))?;
            if opening.eval_k.len() != D
                || opening.eval_a.len() != dims.matrix_count
                || opening.eval_a.iter().any(|row| row.len() != D)
            {
                return Err(PiCcsError::ProtocolError(
                    "optimized opening backend returned a non-canonical v1_1 evaluation".into(),
                ));
            }
            let mut eval_k = opening.eval_k.clone();
            eval_k.resize(d_pad, K::ZERO);
            let eval_a = opening
                .eval_a
                .iter()
                .map(|row| {
                    let mut padded = row.clone();
                    padded.resize(d_pad, K::ZERO);
                    padded
                })
                .collect::<Vec<_>>();
            return Ok(V1_1OutputOpening { eval_k, eval_a });
        }
        let weights = weights
            .as_deref()
            .expect("host openings require the equality tensor");
        let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, structure.m)?;
        let mut eval_k = identity_ring_mle(&assignment, &weights).to_vec();
        eval_k.resize(d_pad, K::ZERO);
        let cache = cache.expect("host openings require the validated structure cache");
        let eval_a =
            crate::superneo_eval::eval_all_mats_ring_cached(cache.superneo(), &assignment, &weights, structure.n)
                .into_iter()
                .map(|coefficients| {
                    let mut row = coefficients.to_vec();
                    row.resize(d_pad, K::ZERO);
                    row
                })
                .collect();
        Ok(V1_1OutputOpening { eval_k, eval_a })
    };

    let mut outputs = Vec::with_capacity(fresh_claims.len() + running_claims.len());
    for (source, (claim, witness)) in fresh_claims.iter().zip(fresh_witnesses).enumerate() {
        let opening = openings(source, &witness.Z)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: crate::common::project_x_from_witness_mat(&witness.Z, structure.m, claim.m_in)?,
            r: point.to_vec(),
            eval_k: opening.eval_k,
            eval_a: opening.eval_a,
            m_in: claim.m_in,
            fold_digest: [0u8; 32],
            adv: claim.adv.clone(),
        });
    }
    for (running, (claim, witness)) in running_claims.iter().zip(running_witnesses).enumerate() {
        let opening = openings(fresh_claims.len() + running, witness)?;
        outputs.push(CeClaim {
            c: claim.c.clone(),
            X: claim.X.clone(),
            r: point.to_vec(),
            eval_k: opening.eval_k,
            eval_a: opening.eval_a,
            m_in: claim.m_in,
            fold_digest: [0u8; 32],
            adv: claim.adv.clone(),
        });
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn prove_with_trace_inner(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    binding: TranscriptBinding,
    source: OracleSource<'_>,
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
    let cache = match &source {
        OracleSource::Cached { cache, .. } => {
            cache.validate_structure(structure)?;
            Some(*cache)
        }
        OracleSource::Rows { cache } => {
            if cache.relation_shape()
                != Some((
                    structure.n,
                    crate::common::superneo_carrier_width(structure.m),
                    structure.t(),
                ))
            {
                return Err(PiCcsError::InvalidInput(
                    "prover row-cache shape does not match the header".into(),
                ));
            }
            None
        }
        OracleSource::Complete { .. } => None,
    };
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
        cache.map(OptimizedStructureCache::matrix_digest),
    )?;
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.variables)?;
    let initial = initial_claim(structure, &challenges, fresh_claims.len(), running_claims)?;
    let sumcheck_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let oracle_started = std::time::Instant::now();
    let mut built_oracle = None;
    let oracle: &mut dyn PaperJointRoundOracle = match source {
        OracleSource::Cached { cache, backend } => {
            let input = PaperJointOracleInput {
                structure,
                params,
                fresh_witnesses,
                running_witnesses,
                challenges: challenges.clone(),
                prior_point,
                dims,
                cache,
            };
            built_oracle = Some(match backend {
                Some(backend) => backend.create(input)?,
                None => Box::new(OptimizedPaperJointOracle::new(
                    input.structure,
                    input.params,
                    input.fresh_witnesses,
                    input.running_witnesses,
                    input.challenges,
                    input.prior_point,
                    input.dims,
                    input.cache,
                )?),
            });
            built_oracle.as_mut().expect("constructed oracle").as_mut()
        }
        OracleSource::Rows { cache } => {
            built_oracle = Some(Box::new(OptimizedPaperJointOracle::from_rows(
                structure,
                params,
                fresh_witnesses,
                running_witnesses,
                challenges.clone(),
                prior_point,
                dims,
                cache,
            )?));
            built_oracle
                .as_mut()
                .expect("constructed row-cache oracle")
                .as_mut()
        }
        OracleSource::Complete {
            oracle,
            challenges: prepared,
        } => {
            if prepared != &challenges {
                return Err(PiCcsError::InvalidInput(
                    "complete optimized oracle challenges do not match the bound statement".into(),
                ));
            }
            if oracle.num_rounds() != dims.variables || oracle.degree_bound() != dims.degree {
                return Err(PiCcsError::InvalidInput(
                    "complete optimized oracle does not have the selected round shape".into(),
                ));
            }
            oracle
        }
    };
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-ccs/phase] oracle-build={:.3}s",
        oracle_started.elapsed().as_secs_f64()
    );
    #[cfg(feature = "perf-timers")]
    let rounds_started = std::time::Instant::now();
    let (rounds, round_challenges, final_claim) =
        pi_ccs_joint_protocol::prove_phase(transcript, &mut trace, initial, oracle)?;
    #[cfg(feature = "perf-timers")]
    eprintln!("[pi-ccs/phase] rounds={:.3}s", rounds_started.elapsed().as_secs_f64());
    let sumcheck_ms = sumcheck_started.elapsed().as_secs_f64() * 1_000.0;
    let output_started = std::time::Instant::now();
    let precomputed_openings = oracle.output_openings(&round_challenges)?;
    drop(built_oracle);
    let mut outputs = build_outputs(
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &round_challenges,
        dims,
        cache,
        precomputed_openings.as_deref(),
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
    #[cfg(feature = "perf-timers")]
    eprintln!("[pi-ccs/phase] outputs={:.3}s", output_started.elapsed().as_secs_f64());
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

pub(crate) fn prove_with_trace<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    _commitment: &L,
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
    prove_with_trace_inner(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        binding,
        OracleSource::Cached { cache, backend: None },
    )
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
        TranscriptBinding::digest_only(),
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

#[allow(clippy::too_many_arguments)]
pub fn prove_with_binding_and_backend<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    _commitment: &L,
    cache: &OptimizedStructureCache,
    binding: TranscriptBinding,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, super::PiCcsProvePerf), PiCcsError> {
    let (outputs, proof, perf, _) = prove_with_trace_inner(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        binding,
        OracleSource::Cached {
            cache,
            backend: Some(backend),
        },
    )?;
    Ok((outputs, proof, perf))
}

/// Run the canonical optimized prover with an already prepared evaluator.
/// The evaluator must use these exact ordered sources and prepared challenges,
/// and return all ring openings. The driver rebinds the statement and rejects
/// missing openings; this route never constructs a host cache or full tensor.
#[allow(clippy::too_many_arguments)]
pub fn prove_with_complete_oracle(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    prepared: &Challenges,
    oracle: &mut dyn PaperJointRoundOracle,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        super::PiCcsProvePerf,
        ProtocolTrace,
    ),
    PiCcsError,
> {
    prove_with_trace_inner(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        TranscriptBinding::digest_only(),
        OracleSource::Complete {
            oracle,
            challenges: prepared,
        },
    )
}

/// Prove with row-cache arithmetic supplied by the owning relation adapter.
/// This cache is prover data. This function does not create or validate a
/// verifier-artifact authority; the owner must supply its matching header.
#[allow(clippy::too_many_arguments)]
pub fn prove_with_row_cache(
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    cache: &SuperneoEvalCache,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        super::PiCcsProvePerf,
        ProtocolTrace,
    ),
    PiCcsError,
> {
    prove_with_trace_inner(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        TranscriptBinding::digest_only(),
        OracleSource::Rows { cache },
    )
}
