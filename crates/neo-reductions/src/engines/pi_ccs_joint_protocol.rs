//! Optimized-side protocol flow for one-joint padded-row PiCCS.
//!
//! This file independently implements the Lean-owned transcript schedule. It
//! does not call the PaperExact transcript, SumCheck driver, or proof assembly.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_joint::{
    build_joint_dims, JointDims, ProtocolTrace, TraceEvent, ALPHA_TAG, COMPACT_BINDING_TAG, GAMMA_TAG,
    PROTOCOL_VERSION, PUBLIC_INPUT_TAG, ROUND_CHALLENGE_TAG, ROUND_TAG, STATEMENT_TAG,
};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;

/// Fallible evaluator boundary for the selected one-joint SumCheck.
///
/// The canonical driver below owns every transcript action and checks each
/// returned round message. A device backend can only evaluate and fold its
/// private multilinear tables.
pub trait PaperJointRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError>;
    fn num_rounds(&self) -> usize;
    fn degree_bound(&self) -> usize;
    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError>;

    /// Return canonical ring openings at the completed SumCheck point when
    /// the evaluator can produce them without rebuilding its private state.
    /// The outer prover still validates the terminal claim and owns every
    /// transcript action. `None` selects the canonical host computation.
    fn output_openings(&mut self, _point: &[K]) -> Result<Option<Vec<Vec<Vec<K>>>>, PiCcsError> {
        Ok(None)
    }
}

/// Fiat--Shamir binding profile for the interactive public statement.
///
/// `Claims` is the independent-reference profile. `Digests` is the recursive
/// profile: the circuit recomputes both values from authoritative claim wires
/// before it absorbs them. The digest form changes transport cost only; the
/// joint polynomial and all prover messages stay unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptBinding {
    Claims,
    Digests {
        public_instance_digest: [F; 4],
        running_accumulator_handle: Option<[F; 4]>,
    },
}

impl Default for TranscriptBinding {
    fn default() -> Self {
        Self::Claims
    }
}

impl TranscriptBinding {
    pub const fn claims() -> Self {
        Self::Claims
    }

    pub const fn digest(public_instance_digest: [F; 4]) -> Self {
        Self::Digests {
            public_instance_digest,
            running_accumulator_handle: None,
        }
    }

    pub const fn digest_and_handle(public_instance_digest: [F; 4], running_accumulator_handle: [F; 4]) -> Self {
        Self::Digests {
            public_instance_digest,
            running_accumulator_handle: Some(running_accumulator_handle),
        }
    }
}

fn k_fields(output: &mut Vec<F>, value: K) {
    output.extend_from_slice(&value.as_coeffs());
}

fn append(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, fields: Vec<F>) {
    transcript.append_fields_unframed(&fields);
    trace.events.push(TraceEvent::Absorb(fields));
}

fn squeeze(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, label: u64, index: Option<usize>) -> K {
    let fields = match index {
        Some(index) => vec![F::from_u64(label), F::from_u64(index as u64)],
        None => vec![F::from_u64(label)],
    };
    append(transcript, trace, fields);
    let sampled = transcript.challenge_fields_raw(2);
    let value = neo_math::from_complex(sampled[0], sampled[1]);
    trace
        .events
        .push(TraceEvent::Challenge { label, index, value });
    value
}

fn append_commitment(fields: &mut Vec<F>, commitment: &Cmt) {
    fields.push(F::from_u64(commitment.d as u64));
    fields.push(F::from_u64(commitment.kappa as u64));
    fields.extend_from_slice(&commitment.data);
}

fn append_running_claim(fields: &mut Vec<F>, claim: &CeClaim<Cmt, F, K>, matrix_count: usize) {
    append_commitment(fields, &claim.c);
    fields.push(F::from_u64(claim.m_in as u64));
    fields.push(F::from_u64(claim.X.rows() as u64));
    fields.push(F::from_u64(claim.X.cols() as u64));
    for row in 0..claim.X.rows() {
        for column in 0..claim.X.cols() {
            fields.push(claim.X[(row, column)]);
        }
    }
    for &value in &claim.r {
        k_fields(fields, value);
    }
    for matrix in 0..matrix_count {
        for &value in claim.y_ring.get(matrix).into_iter().flatten().take(D) {
            k_fields(fields, value);
        }
    }
}

fn validate_selected_inputs(
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    dims: JointDims,
) -> Result<(), PiCcsError> {
    for (index, claim) in fresh.iter().enumerate() {
        if claim.m_in > structure.m || claim.x.len() != claim.m_in || claim.m_in % D != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized fresh claim {index} is not a complete whole-ring public input"
            )));
        }
    }
    for (index, claim) in running.iter().enumerate() {
        if claim.m_in > structure.m
            || claim.m_in % D != 0
            || claim.X.rows() != D
            || claim.X.cols() != neo_ccs::superneo_public_x_cols(claim.m_in)
            || claim.y_ring.len() != dims.matrix_count
            || claim.ct.len() != dims.matrix_count
        {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized running claim {index} does not have the selected paper shape"
            )));
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() != D.next_power_of_two()
                || coefficients[0] != claim.ct[matrix]
                || coefficients.iter().skip(D).any(|&value| value != K::ZERO)
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "optimized running claim {index} matrix image {matrix} is not canonical"
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_and_sample_with_trace(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    binding: TranscriptBinding,
    expected_matrix_digest: Option<&[F; 4]>,
) -> Result<(JointDims, Challenges), PiCcsError> {
    let dims = build_joint_dims(params, structure, fresh.len(), running.len())?;
    validate_selected_inputs(structure, fresh, running, dims)?;
    let matrix_digest: [F; 4] = match expected_matrix_digest {
        Some(matrix_digest) => *matrix_digest,
        None => crate::engines::utils::digest_ccs_matrices(structure)
            .try_into()
            .map_err(|digest: Vec<F>| {
                PiCcsError::ProtocolError(format!(
                    "Pi_CCS expected four matrix-digest fields, got {}",
                    digest.len()
                ))
            })?,
    };
    let mut public = vec![
        F::from_u64(PUBLIC_INPUT_TAG),
        F::from_u64(PROTOCOL_VERSION),
        F::from_u64(dims.variables as u64),
        F::from_u64(fresh.len() as u64),
        F::from_u64(running.len() as u64),
        F::from_u64(dims.matrix_count as u64),
        F::from_u64(D as u64),
        F::from_u64(dims.assignment_width as u64),
        F::from_u64(dims.row_count as u64),
        F::from_u64(dims.degree as u64),
        F::from_u64(structure.n as u64),
        F::from_u64(structure.m as u64),
    ];
    public.extend(matrix_digest);
    match binding {
        TranscriptBinding::Claims => {
            public.push(F::ZERO);
            for claim in running {
                append_running_claim(&mut public, claim, dims.matrix_count);
            }
            for claim in fresh {
                append_commitment(&mut public, &claim.c);
                public.push(F::from_u64(claim.m_in as u64));
                public.push(F::from_u64(claim.x.len() as u64));
                public.extend_from_slice(&claim.x);
            }
        }
        TranscriptBinding::Digests {
            public_instance_digest,
            running_accumulator_handle,
        } => {
            public.push(F::from_u64(COMPACT_BINDING_TAG));
            public.extend_from_slice(&public_instance_digest);
            public.push(F::from_u64(running.len() as u64));
            match running_accumulator_handle {
                Some(handle) => {
                    public.push(F::ONE);
                    public.extend_from_slice(&handle);
                }
                None => public.push(F::ZERO),
            }
        }
    }
    append(transcript, trace, public);

    let prior_point = running.first().map(|claim| claim.r.as_slice());
    let mut statement = vec![
        F::from_u64(STATEMENT_TAG),
        F::from_u64(dims.variables as u64),
        F::from_u64(fresh.len() as u64),
        F::from_u64(running.len() as u64),
        F::from_u64(dims.matrix_count as u64),
        F::from_u64(D as u64),
        F::from_u64(structure.max_degree() as u64),
        F::from_u64(structure.f.terms().len() as u64),
    ];
    for term in structure.f.terms() {
        statement.push(term.coeff);
        statement.push(F::ZERO);
        statement.push(F::ZERO);
        statement.extend(term.exps.iter().map(|&value| F::from_u64(value as u64)));
    }
    match binding {
        TranscriptBinding::Claims => {
            statement.push(F::ZERO);
            statement.push(F::from_u64(dims.variables as u64));
            for coordinate in 0..dims.variables {
                k_fields(&mut statement, prior_point.map_or(K::ZERO, |point| point[coordinate]));
            }
            statement.push(F::from_u64((running.len() * dims.matrix_count * D) as u64));
            for coefficient in 0..D {
                for matrix in 0..dims.matrix_count {
                    for claim in running {
                        let value = claim
                            .y_ring
                            .get(matrix)
                            .and_then(|row| row.get(coefficient))
                            .copied()
                            .ok_or_else(|| {
                                PiCcsError::InvalidInput("optimized carried statement is incomplete".into())
                            })?;
                        k_fields(&mut statement, value);
                    }
                }
            }
        }
        TranscriptBinding::Digests { .. } => statement.push(F::from_u64(COMPACT_BINDING_TAG)),
    }
    append(transcript, trace, statement);

    let alpha = (0..dims.variables)
        .map(|index| squeeze(transcript, trace, ALPHA_TAG, Some(index)))
        .collect();
    let gamma = squeeze(transcript, trace, GAMMA_TAG, None);
    Ok((dims, Challenges::new(alpha, gamma)))
}

pub fn prove_phase<O: PaperJointRoundOracle + ?Sized>(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    initial_claim: K,
    oracle: &mut O,
) -> Result<(Vec<Vec<K>>, Vec<K>, K), PiCcsError> {
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut rounds = Vec::with_capacity(oracle.num_rounds());
    let mut challenges = Vec::with_capacity(oracle.num_rounds());
    for round in 0..oracle.num_rounds() {
        let points: Vec<K> = (0..=oracle.degree_bound())
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        let evaluations = oracle.evals_at(&points)?;
        if evaluations.len() != points.len() || evaluations[0] + evaluations[1] != claim {
            let actual =
                evaluations.first().copied().unwrap_or(K::ZERO) + evaluations.get(1).copied().unwrap_or(K::ZERO);
            return Err(PiCcsError::SumcheckError(format!(
                "optimized joint SumCheck invariant failed at round {round}: expected {claim:?}, got {actual:?}"
            )));
        }
        let coefficients = crate::sumcheck::interpolate_from_evals(&points, &evaluations);
        let mut fields = vec![
            F::from_u64(ROUND_TAG),
            F::from_u64(round as u64),
            F::from_u64(coefficients.len() as u64),
        ];
        for &coefficient in &coefficients {
            k_fields(&mut fields, coefficient);
        }
        append(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round));
        claim = crate::sumcheck::poly_eval_k(&coefficients, challenge);
        oracle.fold(challenge)?;
        rounds.push(coefficients);
        challenges.push(challenge);
    }
    trace.rounds = rounds.clone();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((rounds, challenges, claim))
}

fn verify_phase(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    dims: JointDims,
    initial_claim: K,
    rounds: &[Vec<K>],
) -> Result<(Vec<K>, K), PiCcsError> {
    if rounds.len() != dims.variables || rounds.iter().any(|round| round.len() != dims.degree + 1) {
        return Err(PiCcsError::InvalidInput(
            "optimized joint SumCheck message shape mismatch".into(),
        ));
    }
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(dims.variables);
    for (round_index, coefficients) in rounds.iter().enumerate() {
        if coefficients[0] + crate::sumcheck::poly_eval_k(coefficients, K::ONE) != claim {
            return Err(PiCcsError::SumcheckError(format!(
                "optimized verifier rejected SumCheck round {round_index}"
            )));
        }
        let mut fields = vec![
            F::from_u64(ROUND_TAG),
            F::from_u64(round_index as u64),
            F::from_u64(coefficients.len() as u64),
        ];
        for &coefficient in coefficients {
            k_fields(&mut fields, coefficient);
        }
        append(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round_index));
        claim = crate::sumcheck::poly_eval_k(coefficients, challenge);
        challenges.push(challenge);
    }
    trace.rounds = rounds.to_vec();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((challenges, claim))
}

/// Encode the complete Section 7.3 output message in
/// source/matrix/coefficient order. Each quadratic-extension value is low
/// limb first.
pub fn output_message_fields(outputs: &[CeClaim<Cmt, F, K>], dims: JointDims) -> Result<Vec<F>, PiCcsError> {
    let mut fields = vec![F::from_u64(COMPACT_BINDING_TAG)];
    for output in outputs {
        if output.y_ring.len() != dims.matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized output matrix family is incomplete".into(),
            ));
        }
        for matrix in 0..dims.matrix_count {
            for coefficient in 0..D {
                k_fields(
                    &mut fields,
                    *output.y_ring[matrix]
                        .get(coefficient)
                        .ok_or_else(|| PiCcsError::InvalidInput("optimized ring output is incomplete".into()))?,
                );
            }
        }
    }
    Ok(fields)
}

pub fn absorb_outputs(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    outputs: &[CeClaim<Cmt, F, K>],
    fresh_count: usize,
    dims: JointDims,
) -> Result<[u8; 32], PiCcsError> {
    let _ = fresh_count;
    // The output message is the input to Pi_RLC. Pi_RLC recomputes and
    // absorbs its canonical output digest before it samples rho. Pi_CCS must
    // therefore finish here, after it checks the output values but before the
    // next reduction binds them. This is the paper's interactive order.
    let _ = output_message_fields(outputs, dims)?;
    let digest = transcript.digest32();
    trace.final_digest = digest;
    Ok(digest)
}

pub fn assemble_proof(rounds: Vec<Vec<K>>) -> PiCcsProof {
    let mut proof = PiCcsProof::new(rounds);
    proof.canonicalize();
    proof
}

fn validate_outputs(
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    point: &[K],
    dims: JointDims,
) -> Result<(), PiCcsError> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(PiCcsError::InvalidInput(
            "optimized output source count mismatch".into(),
        ));
    }
    for output in outputs {
        if output.r != point
            || output.X.rows() != D
            || output.X.cols() != neo_ccs::superneo_public_x_cols(output.m_in)
            || output.y_ring.len() != dims.matrix_count
            || output.ct.len() != dims.matrix_count
        {
            return Err(PiCcsError::InvalidInput(
                "optimized output does not have the one-joint shape".into(),
            ));
        }
        for (matrix, row) in output.y_ring.iter().enumerate() {
            if row.len() != D.next_power_of_two()
                || row[0] != output.ct[matrix]
                || row.iter().skip(D).any(|&value| value != K::ZERO)
            {
                return Err(PiCcsError::InvalidInput("optimized output is not canonical".into()));
            }
        }
    }
    for (claim, output) in fresh.iter().zip(outputs) {
        if claim.c != output.c || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "fresh output changed its public instance".into(),
            ));
        }
        if claim.m_in % D != 0 || claim.x.len() != claim.m_in {
            return Err(PiCcsError::InvalidInput(
                "fresh public input is not whole-ring aligned".into(),
            ));
        }
        for (coordinate, &value) in claim.x.iter().enumerate() {
            if output.X[(coordinate % D, coordinate / D)] != value {
                return Err(PiCcsError::ProtocolError(
                    "fresh output changed a public input coordinate".into(),
                ));
            }
        }
    }
    for (claim, output) in running.iter().zip(outputs.iter().skip(fresh.len())) {
        if claim.c != output.c || claim.X != output.X || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "carried output changed its public instance".into(),
            ));
        }
    }
    let _ = structure;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_with_trace(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: TranscriptBinding,
    expected_matrix_digest: Option<&[F; 4]>,
) -> Result<(bool, ProtocolTrace), PiCcsError> {
    let mut trace = ProtocolTrace::default();
    let (dims, challenges) = bind_and_sample_with_trace(
        transcript,
        &mut trace,
        params,
        structure,
        fresh,
        running,
        binding,
        expected_matrix_digest,
    )?;
    let prior_point = crate::engines::utils::shared_me_input_r(running, dims.variables)?;
    let initial =
        crate::engines::optimized_engine::paper_joint::initial_claim(structure, &challenges, fresh.len(), running)?;
    let (point, final_claim) = verify_phase(transcript, &mut trace, dims, initial, &proof.sumcheck_rounds)?;
    validate_outputs(structure, fresh, running, outputs, &point, dims)?;
    let expected = crate::engines::optimized_engine::paper_joint::terminal::<F>(
        structure,
        params,
        &challenges,
        fresh.len(),
        prior_point,
        &point,
        outputs,
    )?;
    let digest = absorb_outputs(transcript, &mut trace, outputs, fresh.len(), dims)?;
    if outputs.iter().any(|output| output.fold_digest != digest) {
        return Err(PiCcsError::ProtocolError(
            "optimized output digest does not match transcript replay".into(),
        ));
    }
    Ok((final_claim == expected, trace))
}

#[allow(clippy::too_many_arguments)]
pub fn verify_with_binding(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: TranscriptBinding,
) -> Result<bool, PiCcsError> {
    Ok(verify_with_trace(
        transcript, params, structure, fresh, running, outputs, proof, binding, None,
    )?
    .0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_with_binding_and_matrix_digest(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: TranscriptBinding,
    expected_matrix_digest: &[F; 4],
) -> Result<bool, PiCcsError> {
    Ok(verify_with_trace(
        transcript,
        params,
        structure,
        fresh,
        running,
        outputs,
        proof,
        binding,
        Some(expected_matrix_digest),
    )?
    .0)
}

pub fn verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    verify_with_binding(
        transcript,
        params,
        structure,
        fresh,
        running,
        outputs,
        proof,
        TranscriptBinding::claims(),
    )
}
