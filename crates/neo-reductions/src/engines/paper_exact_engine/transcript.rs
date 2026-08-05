//! Independent PaperExact Fiat--Shamir and SumCheck schedule.
//!
//! This is a direct Rust transcription of the Lean-owned tags 40--47. It
//! does not call the optimized transcript or proof-assembly implementation.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CeClaim};
use neo_math::{KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

use crate::engines::pi_ccs_joint::{JointDims, ProtocolTrace, TraceEvent};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

// These constants intentionally duplicate the selected codec profile. A
// protocol-tag change on only one side must fail the crosscheck.
const PUBLIC_INPUT_TAG: u64 = 40;
const PROTOCOL_VERSION: u64 = 2;
const STATEMENT_TAG: u64 = 41;
const COMPACT_BINDING_TAG: u64 = 47;
const ALPHA_TAG: u64 = 42;
const GAMMA_TAG: u64 = 43;
const ROUND_TAG: u64 = 45;
const ROUND_CHALLENGE_TAG: u64 = 46;

/// Public-statement transport used by the independent reference transcript.
///
/// This type is deliberately separate from the optimized transcript type.
/// The compact values are caller-supplied compression values; the recursive
/// verifier must recompute them from authoritative claim wires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PaperTranscriptBinding {
    Claims,
    Digests {
        public_instance_digest: [F; 4],
        running_accumulator_handle: Option<[F; 4]>,
    },
}

impl PaperTranscriptBinding {
    pub(crate) const fn claims() -> Self {
        Self::Claims
    }

    pub(crate) const fn digests(public_instance_digest: [F; 4], running_accumulator_handle: Option<[F; 4]>) -> Self {
        Self::Digests {
            public_instance_digest,
            running_accumulator_handle,
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
    for value in &claim.r {
        k_fields(fields, *value);
    }
    for matrix in 0..matrix_count {
        for &value in claim
            .y_ring
            .get(matrix)
            .into_iter()
            .flatten()
            .take(neo_math::D)
        {
            k_fields(fields, value);
        }
    }
}

fn paper_matrix_digest(structure: &CcsStructure<F>) -> Vec<F> {
    use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};

    const SEED: u64 = 0x434353445F4D4154;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;

    for &byte in b"neo/ccs/matrices/v1" {
        if absorbed >= 15 {
            poseidon2.permute_mut(&mut state);
            absorbed = 0;
        }
        state[absorbed] = Goldilocks::from_u32(byte as u32);
        absorbed += 1;
    }
    if absorbed + 3 >= 16 {
        poseidon2.permute_mut(&mut state);
        absorbed = 0;
    }
    state[absorbed] = Goldilocks::from_u64(structure.n as u64);
    state[absorbed + 1] = Goldilocks::from_u64(structure.m as u64);
    state[absorbed + 2] = Goldilocks::from_u64(structure.t() as u64);
    poseidon2.permute_mut(&mut state);

    for (matrix_index, matrix) in structure.matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(matrix_index as u64);
        absorbed += 1;

        let mut emit = |row: usize, column: usize, value: u64| {
            if absorbed + 3 > 15 {
                poseidon2.permute_mut(&mut state);
                absorbed = 0;
            }
            state[absorbed] = Goldilocks::from_u64(row as u64);
            state[absorbed + 1] = Goldilocks::from_u64(column as u64);
            state[absorbed + 2] = Goldilocks::from_u64(value);
            absorbed += 3;
        };

        match matrix {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, structure.n);
                debug_assert_eq!(*n, structure.m);
                for row in 0..structure.n {
                    emit(row, row, F::ONE.as_canonical_u64());
                }
            }
            CcsMatrix::Csc(csc) => {
                let mut entries = Vec::with_capacity(csc.vals.len());
                for column in 0..csc.ncols {
                    for index in csc.column_range(column) {
                        entries.push((csc.row_index(index), column, csc.vals[index].as_canonical_u64()));
                    }
                }
                entries.sort_unstable_by_key(|&(row, column, _)| (row, column));
                for (row, column, value) in entries {
                    emit(row, column, value);
                }
            }
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                let mut entries = Vec::with_capacity(csc.vals.len());
                for column in 0..csc.ncols {
                    for index in csc.column_range(column) {
                        entries.push((csc.row_index(index), column, csc.vals[index].as_canonical_u64()));
                    }
                }
                entries.sort_unstable_by_key(|&(row, column, _)| (row, column));
                for (row, column, value) in entries {
                    emit(row, column, value);
                }

                for (block_index, block) in blocks.iter().enumerate() {
                    emit(usize::MAX, block_index, 0x5048_4938_3153_4545);
                    emit(usize::MAX - 1, block.row_start(), block.kappa() as u64);
                    emit(usize::MAX - 2, block.message_cols(), block.chunk_size() as u64);
                    emit(
                        usize::MAX - 3,
                        block.word_starts().len(),
                        u64::from(block.has_superneo_transformed_columns()),
                    );
                    emit(usize::MAX - 4, usize::MAX, block.word_width() as u64);
                    for (word, &start) in block.word_starts().iter().enumerate() {
                        emit(usize::MAX - 4, word, start as u64);
                    }
                    for (seed_row, seeds) in block.chunk_seeds_by_row().iter().enumerate() {
                        for (chunk, seed) in seeds.iter().enumerate() {
                            for limb in 0..4 {
                                let value = u64::from_le_bytes(
                                    seed[limb * 8..(limb + 1) * 8]
                                        .try_into()
                                        .expect("PaperExact matrix seed limb"),
                                );
                                emit(usize::MAX - 5 - seed_row, chunk * 4 + limb, value);
                            }
                        }
                    }
                }
                for (run_index, run) in geometric_runs.iter().enumerate() {
                    let sentinel = usize::MAX / 2;
                    emit(sentinel, run_index, 0x4745_4f4d_5255_4e31);
                    emit(sentinel - 1, run.row(), run.column_start() as u64);
                    emit(sentinel - 2, run.len(), run.initial().as_canonical_u64());
                    emit(sentinel - 3, run_index, run.ratio().as_canonical_u64());
                }
            }
        }
        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
}

fn paper_poly_eval(coefficients: &[K], point: K) -> K {
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(K::ZERO, |value, coefficient| value * point + coefficient)
}

fn paper_interpolate(points: &[K], evaluations: &[K]) -> Vec<K> {
    assert_eq!(
        points.len(),
        evaluations.len(),
        "PaperExact interpolation shape mismatch"
    );
    let count = points.len();
    let mut coefficients = vec![K::ZERO; count];
    for index in 0..count {
        let mut numerator = vec![K::ZERO; count];
        numerator[0] = K::ONE;
        let mut degree = 0;
        for other in 0..count {
            if index == other {
                continue;
            }
            let mut next = vec![K::ZERO; count];
            for coefficient in 0..=degree {
                next[coefficient] -= points[other] * numerator[coefficient];
                next[coefficient + 1] += numerator[coefficient];
            }
            numerator = next;
            degree += 1;
        }
        let mut denominator = K::ONE;
        for other in 0..count {
            if index != other {
                denominator *= points[index] - points[other];
            }
        }
        let scale = evaluations[index] * denominator.inv();
        for coefficient in 0..=degree {
            coefficients[coefficient] += scale * numerator[coefficient];
        }
    }
    coefficients
}

pub(super) fn bind_and_sample(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    dims: JointDims,
    binding: PaperTranscriptBinding,
) -> Result<Challenges, PiCcsError> {
    let matrix_digest = paper_matrix_digest(structure);
    let mut public = vec![
        F::from_u64(PUBLIC_INPUT_TAG),
        F::from_u64(PROTOCOL_VERSION),
        F::from_u64(dims.variables as u64),
        F::from_u64(fresh.len() as u64),
        F::from_u64(running.len() as u64),
        F::from_u64(dims.matrix_count as u64),
        F::from_u64(neo_math::D as u64),
        F::from_u64(dims.assignment_width as u64),
        F::from_u64(dims.row_count as u64),
        F::from_u64(dims.degree as u64),
        F::from_u64(structure.n as u64),
        F::from_u64(structure.m as u64),
    ];
    public.extend(matrix_digest);
    match binding {
        PaperTranscriptBinding::Claims => {
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
        PaperTranscriptBinding::Digests {
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
        F::from_u64(neo_math::D as u64),
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
        PaperTranscriptBinding::Claims => {
            statement.push(F::ZERO);
            statement.push(F::from_u64(dims.variables as u64));
            for coordinate in 0..dims.variables {
                k_fields(&mut statement, prior_point.map_or(K::ZERO, |point| point[coordinate]));
            }
            let carried_count = running.len() * dims.matrix_count * neo_math::D;
            statement.push(F::from_u64(carried_count as u64));
            for coefficient in 0..neo_math::D {
                for matrix in 0..dims.matrix_count {
                    for claim in running {
                        let value = claim
                            .y_ring
                            .get(matrix)
                            .and_then(|row| row.get(coefficient))
                            .copied()
                            .ok_or_else(|| {
                                PiCcsError::InvalidInput("PaperExact carried statement is incomplete".into())
                            })?;
                        k_fields(&mut statement, value);
                    }
                }
            }
        }
        PaperTranscriptBinding::Digests { .. } => statement.push(F::from_u64(COMPACT_BINDING_TAG)),
    }
    append(transcript, trace, statement);

    let alpha = (0..dims.variables)
        .map(|index| squeeze(transcript, trace, ALPHA_TAG, Some(index)))
        .collect();
    let gamma = squeeze(transcript, trace, GAMMA_TAG, None);
    Ok(Challenges::new(alpha, gamma))
}

pub(super) fn prove_sumcheck<O: RoundOracle>(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    initial_claim: K,
    oracle: &mut O,
) -> Result<(Vec<Vec<K>>, Vec<K>, K), PiCcsError> {
    trace.initial_claim = initial_claim;
    let mut running_claim = initial_claim;
    let mut rounds = Vec::with_capacity(oracle.num_rounds());
    let mut challenges = Vec::with_capacity(oracle.num_rounds());
    for round in 0..oracle.num_rounds() {
        let degree = oracle.degree_bound();
        let points: Vec<K> = (0..=degree)
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        let evaluations = oracle.evals_at(&points);
        if evaluations.len() != degree + 1 || evaluations[0] + evaluations[1] != running_claim {
            return Err(PiCcsError::SumcheckError(format!(
                "PaperExact joint SumCheck invariant failed at round {round}"
            )));
        }
        let coefficients = paper_interpolate(&points, &evaluations);
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
        running_claim = paper_poly_eval(&coefficients, challenge);
        oracle.fold(challenge);
        rounds.push(coefficients);
        challenges.push(challenge);
    }
    trace.rounds = rounds.clone();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = running_claim;
    Ok((rounds, challenges, running_claim))
}

pub(super) fn verify_sumcheck(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    dims: JointDims,
    initial_claim: K,
    rounds: &[Vec<K>],
) -> Result<(Vec<K>, K), PiCcsError> {
    if rounds.len() != dims.variables || rounds.iter().any(|round| round.len() != dims.degree + 1) {
        return Err(PiCcsError::InvalidInput(
            "PaperExact joint SumCheck message shape mismatch".into(),
        ));
    }
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(dims.variables);
    for (round_index, coefficients) in rounds.iter().enumerate() {
        if coefficients[0] + paper_poly_eval(coefficients, K::ONE) != claim {
            return Err(PiCcsError::SumcheckError(format!(
                "PaperExact verifier rejected SumCheck round {round_index}"
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
        claim = paper_poly_eval(coefficients, challenge);
        challenges.push(challenge);
    }
    trace.rounds = rounds.to_vec();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((challenges, claim))
}

pub(super) fn absorb_outputs(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    outputs: &[CeClaim<Cmt, F, K>],
    fresh_count: usize,
    dims: JointDims,
) -> Result<[u8; 32], PiCcsError> {
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput(
            "PaperExact output source count mismatch".into(),
        ));
    }
    let _ = fresh_count;
    let mut fields = Vec::new();
    // SuperNeo Section 7.3 sends y' after the SumCheck. Validate the complete
    // message shape here. PiRLC binds its canonical digest before sampling
    // rho; PiCCS adds no extra non-paper output equation.
    for output in outputs {
        if output.y_ring.len() != dims.matrix_count {
            return Err(PiCcsError::InvalidInput(
                "PaperExact output matrix family is incomplete".into(),
            ));
        }
        for matrix in 0..dims.matrix_count {
            for coefficient in 0..neo_math::D {
                k_fields(
                    &mut fields,
                    *output.y_ring[matrix]
                        .get(coefficient)
                        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact ring output is incomplete".into()))?,
                );
            }
        }
    }
    let _ = fields;
    let digest = transcript.digest32();
    trace.final_digest = digest;
    Ok(digest)
}

pub(super) fn assemble_proof(rounds: Vec<Vec<K>>) -> PiCcsProof {
    PiCcsProof::new(rounds)
}

/// Encode the one-joint proof without using the production codec.
///
/// The duplicate constants and field order are intentional. PaperExact must
/// detect a production codec change that is not made in the reference codec.
#[doc(hidden)]
pub fn encode_proof(proof: &PiCcsProof) -> Result<Vec<u8>, PiCcsError> {
    const PROOF_TAG: u64 = 1102;
    const CODEC_VERSION: u64 = 1;

    let coefficient_count = proof.sumcheck_rounds.first().map_or(0, Vec::len);
    if proof
        .sumcheck_rounds
        .iter()
        .any(|round| round.len() != coefficient_count)
    {
        return Err(PiCcsError::InvalidInput(
            "PaperExact codec requires one fixed round degree".into(),
        ));
    }

    fn push_u64(output: &mut Vec<u8>, value: u64) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    let mut output = Vec::with_capacity(4 * 8 + proof.sumcheck_rounds.len() * coefficient_count * 16);
    push_u64(&mut output, PROOF_TAG);
    push_u64(&mut output, CODEC_VERSION);
    push_u64(&mut output, proof.sumcheck_rounds.len() as u64);
    push_u64(&mut output, coefficient_count as u64);
    for round in &proof.sumcheck_rounds {
        for &coefficient in round {
            let (low, high) = coefficient.to_limbs_u64();
            push_u64(&mut output, low);
            push_u64(&mut output, high);
        }
    }
    Ok(output)
}
