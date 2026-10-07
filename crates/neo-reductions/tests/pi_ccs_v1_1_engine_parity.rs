#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{dec_children_with_commit, rlc_with_commit, FoldingMode};
use neo_reductions::engines::crosscheck_engine::{crosscheck_prove, crosscheck_verify};
use neo_reductions::engines::paper_exact_engine::paper_joint::PaperJointOracle;
use neo_reductions::engines::pi_ccs_joint::{
    build_joint_dims, eval_a_gamma_exponent, eval_k_gamma_exponent, gamma_power, TraceEvent,
};
use neo_reductions::engines::pi_ccs_protocol::Challenges;
use neo_reductions::engines::{CrossCheckEngine, OptimizedEngine, PaperExactEngine, PiCcsEngine};
use neo_reductions::optimized_engine::canonical_audit::OptimizedPaperJointOracle;
use neo_reductions::optimized_engine::{optimized_verify_with_trace, OptimizedStructureCache, PaperJointRoundOracle};
use neo_reductions::sumcheck::RoundOracle;
use neo_reductions::superneo_eval::{CachedMatrixRows, MatrixWindow};
use neo_reductions::{split_b_matrix_k, verify_and_export_pi_ccs_receipt, PiCcsError, PiCcsProof};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

mod zero_running;

type Claim = CcsClaim<neo_ajtai::Commitment, F>;
type Output = CeClaim<neo_ajtai::Commitment, F, K>;

fn rectangular_ccs(rows: usize, columns: usize) -> CcsStructure<F> {
    let mut first = Mat::zero(rows, columns, F::ZERO);
    for column in 0..columns {
        first[(column % rows, column)] = F::from_u64((column % 5 + 1) as u64);
    }
    CcsStructure::new(
        vec![first.clone(), first],
        SparsePoly::new(
            2,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![1, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0, 1],
                },
            ],
        ),
    )
    .expect("valid rectangular CCS")
}

/// The selected Nightstream profile, with its statistical target set to the
/// census of this shape, as the lifecycle does. PaperExact requires its rank.
fn selected_parameters(structure: &CcsStructure<F>) -> NeoParams {
    let mut params = NeoParams::nightstream_goldilocks_k16();
    let summary = params
        .padded_row_security_summary_for_shape(
            structure.domain_rows(),
            structure.m,
            structure.t(),
            structure.max_degree(),
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .expect("shape census");
    params.lambda = params.lambda.min(summary.security_bits);
    params
}

fn committer(params: &NeoParams, columns: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5041_4444_4544_524f);
    let public_parameters = ajtai_setup(&mut rng, D, params.kappa as usize, columns.div_ceil(D)).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(public_parameters))
}

fn combine_commitments_b_pows(commitments: &[neo_ajtai::Commitment], base: u32) -> neo_ajtai::Commitment {
    let mut output = neo_ajtai::Commitment::zeros(commitments[0].d, commitments[0].kappa);
    let mut power = F::ONE;
    let base = F::from_u64(base as u64);
    for commitment in commitments {
        let mut term = commitment.clone();
        for value in &mut term.data {
            *value *= power;
        }
        output.add_inplace(&term);
        power *= base;
    }
    output
}

fn source(log: &AjtaiSModule, columns: usize, seed: usize) -> (Claim, CcsWitness<F>) {
    let mut values: Vec<F> = (0..columns)
        .map(|column| match (seed + 5 * column) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let m_in = if columns >= D { D } else { 0 };
    if m_in == D {
        values[..D].fill(F::ZERO);
        values[0] = F::ONE;
        for lane in 0..4 {
            values[lane + 1] = F::from_u64(((seed + lane) & 1) as u64);
        }
    }
    let mut Z = Mat::zero(D, columns.div_ceil(D), F::ZERO);
    for (column, &value) in values.iter().enumerate() {
        Z[(column % D, column / D)] = value;
    }
    // The digest-only transcript decodes the prior digest from the marker-offset
    // bit cells of the first fresh public input; this ring sets four of them.
    // Smaller oracle-only fixtures keep an empty public prefix.
    (
        CcsClaim {
            adv: None,
            c: log.commit(&Z),
            x: values[..m_in].to_vec(),
            m_in,
        },
        CcsWitness {
            w: values[m_in..].to_vec(),
            Z,
        },
    )
}

/// PiCCS through the public engine that serves `mode`.
#[allow(clippy::too_many_arguments)]
fn mode_prove(
    mode: FoldingMode,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    witnesses: &[CcsWitness<F>],
    running: &[Output],
    running_witnesses: &[Mat<F>],
    log: &AjtaiSModule,
) -> Result<(Vec<Output>, PiCcsProof), PiCcsError> {
    match mode {
        FoldingMode::Optimized => OptimizedEngine.prove(
            transcript,
            params,
            structure,
            claims,
            witnesses,
            running,
            running_witnesses,
            log,
        ),
        FoldingMode::PaperExact => PaperExactEngine.prove(
            transcript,
            params,
            structure,
            claims,
            witnesses,
            running,
            running_witnesses,
            log,
        ),
        FoldingMode::OptimizedWithCrosscheck => crosscheck().prove(
            transcript,
            params,
            structure,
            claims,
            witnesses,
            running,
            running_witnesses,
            log,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn mode_verify(
    mode: FoldingMode,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    running: &[Output],
    outputs: &[Output],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    match mode {
        FoldingMode::Optimized => {
            OptimizedEngine.verify(transcript, params, structure, claims, running, outputs, proof)
        }
        FoldingMode::PaperExact => {
            PaperExactEngine.verify(transcript, params, structure, claims, running, outputs, proof)
        }
        FoldingMode::OptimizedWithCrosscheck => {
            crosscheck().verify(transcript, params, structure, claims, running, outputs, proof)
        }
    }
}

fn crosscheck() -> CrossCheckEngine<OptimizedEngine, PaperExactEngine> {
    CrossCheckEngine {
        inner: OptimizedEngine,
        ref_oracle: PaperExactEngine,
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_mode(
    mode: FoldingMode,
    label: &'static [u8],
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    witnesses: &[CcsWitness<F>],
    running: &[Output],
    running_witnesses: &[Mat<F>],
    log: &AjtaiSModule,
) -> Result<(Vec<Output>, PiCcsProof), PiCcsError> {
    prove_mode_and_state(
        mode,
        label,
        params,
        structure,
        claims,
        witnesses,
        running,
        running_witnesses,
        log,
    )
    .map(|(outputs, proof, _)| (outputs, proof))
}

#[allow(clippy::too_many_arguments)]
fn prove_mode_and_state(
    mode: FoldingMode,
    label: &'static [u8],
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    witnesses: &[CcsWitness<F>],
    running: &[Output],
    running_witnesses: &[Mat<F>],
    log: &AjtaiSModule,
) -> Result<
    (
        Vec<Output>,
        PiCcsProof,
        [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
    ),
    PiCcsError,
> {
    let mut transcript = Poseidon2Transcript::new(label);
    let (outputs, proof) = mode_prove(
        mode,
        &mut transcript,
        params,
        structure,
        claims,
        witnesses,
        running,
        running_witnesses,
        log,
    )?;
    Ok((outputs, proof, transcript.state()))
}

fn seed_running(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &Claim,
    witness: &CcsWitness<F>,
    log: &AjtaiSModule,
) -> Output {
    let (zero, zero_witnesses) = zero_running::zero_running(params, structure, 1, claim.m_in);
    prove_mode(
        FoldingMode::PaperExact,
        b"padded-row/seed",
        params,
        structure,
        std::slice::from_ref(claim),
        std::slice::from_ref(witness),
        &zero,
        &zero_witnesses,
        log,
    )
    .expect("seed proof")
    .0
    .remove(0)
}

fn assert_parity(rows: usize, columns: usize) {
    let structure = rectangular_ccs(rows, columns);
    let params = selected_parameters(&structure);
    let log = committer(&params, columns);
    let (prior_claim, prior_witness) = source(&log, columns, 1);
    let running = vec![seed_running(&params, &structure, &prior_claim, &prior_witness, &log)];
    let running_witnesses = vec![prior_witness.Z];
    let (first_claim, first_witness) = source(&log, columns, 2);
    let (second_claim, second_witness) = source(&log, columns, 7);
    let claims = vec![first_claim, second_claim];
    let witnesses = vec![first_witness, second_witness];
    let label = b"padded-row/parity";

    let (paper_outputs, paper_proof, paper_state) = prove_mode_and_state(
        FoldingMode::PaperExact,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    )
    .expect("PaperExact proof");
    let (optimized_outputs, optimized_proof, optimized_state) = prove_mode_and_state(
        FoldingMode::Optimized,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    )
    .expect("optimized proof");

    assert_eq!(paper_proof, optimized_proof);
    assert_eq!(paper_outputs, optimized_outputs);
    assert_eq!(paper_state, optimized_state);
    assert!(paper_outputs
        .iter()
        .all(|output| { output.eval_k.len() == D.next_power_of_two() && output.eval_a.len() == structure.t() }));
    assert_eq!(
        paper_proof.canonical_bytes().unwrap(),
        optimized_proof.canonical_bytes().unwrap()
    );

    for mode in [FoldingMode::PaperExact, FoldingMode::Optimized] {
        assert!(mode_verify(
            mode,
            &mut Poseidon2Transcript::new(label),
            &params,
            &structure,
            &claims,
            &running,
            &optimized_outputs,
            &optimized_proof,
        )
        .expect("verification"));
    }
}

#[test]
fn one_joint_engines_are_byte_exact_for_both_rectangular_directions() {
    assert_parity(D, D);
    assert_parity(D / 2, D);
    assert_parity(2 * D, D);
    assert_parity(D / 2, D + 1);
    assert_parity(2 * D, D + 1);
}

#[test]
fn every_joint_round_polynomial_and_fold_matches() {
    for (rows, columns) in [(4, 8), (16, 8)] {
        let structure = rectangular_ccs(rows, columns);
        let params = selected_parameters(&structure);
        let log = committer(&params, columns);
        let (_, first) = source(&log, columns, 3);
        let (_, second) = source(&log, columns, 8);
        let (_, running_source) = source(&log, columns, 11);
        let fresh = vec![first, second];
        let running = vec![running_source.Z];
        let dims = build_joint_dims(&params, &structure, fresh.len(), running.len()).expect("joint dims");
        let alpha = (0..dims.variables)
            .map(|index| K::from(F::from_u64((3 + index) as u64)))
            .collect();
        let prior: Vec<K> = (0..dims.variables)
            .map(|index| K::from(F::from_u64((47 + index) as u64)))
            .collect();
        let challenges = Challenges::new(alpha, K::from(F::from_u64(13)));
        let cache = OptimizedStructureCache::build(&structure).expect("cache");
        let source = CachedMatrixRows::new(cache.superneo()).expect("matrix rows");
        let payload = fresh.len() * structure.t() * size_of::<K>() + size_of::<F>() + size_of::<K>();
        let workspace_bytes = MatrixWindow::required_workspace(&source, 0..structure.n, payload)
            .expect("fixture row workspace")
            + (dims.degree + 1 + structure.t()) * size_of::<K>();
        let mut paper = PaperJointOracle::new(
            &structure,
            &params,
            &fresh,
            &running,
            challenges.clone(),
            Some(&prior),
            dims,
        )
        .expect("paper oracle");
        let mut optimized = OptimizedPaperJointOracle::new(
            &structure,
            &params,
            &fresh,
            &running,
            challenges,
            Some(&prior),
            dims,
            &source,
            workspace_bytes,
        )
        .expect("optimized oracle");
        let points: Vec<K> = (0..=dims.degree)
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        for round in 0..dims.variables {
            assert_eq!(
                paper.evals_at(&points),
                optimized
                    .evals_at(&points)
                    .expect("optimized round evaluations"),
                "round {round}"
            );
            let challenge = K::from(F::from_u64((19 + 3 * round) as u64));
            paper.fold(challenge);
            optimized.fold(challenge).expect("optimized oracle fold");
        }
    }
}

#[test]
fn public_crosscheck_compares_the_complete_execution() {
    let structure = rectangular_ccs(D / 2, D + 1);
    let params = selected_parameters(&structure);
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 5);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let label = b"padded-row/crosscheck";
    let (outputs, proof) = prove_mode(
        mode.clone(),
        label,
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    )
    .expect("crosscheck proof");
    assert!(mode_verify(
        mode,
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &outputs,
        &proof,
    )
    .expect("crosscheck verify"));
}

#[test]
fn v1_1_transcript_matches_the_independent_reference() {
    let structure = rectangular_ccs(D / 2, D + 1);
    let params = selected_parameters(&structure);
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 6);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    let label = b"pi-ccs/v1_1/crosscheck";
    let (outputs, proof) = crosscheck_prove(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    )
    .expect("v1_1 transcript crosscheck proof");
    assert!(crosscheck_verify(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &outputs,
        &proof,
    )
    .expect("v1_1 transcript crosscheck verify"));
}

/// Both engines absorb the Lean `decodeHash` of the first fresh public input
/// as the prior digest. The input is `encHash(d)`, built by the encoder for four
/// distinct words that use all 64 bit positions. The crosscheck proves that the
/// optimized and PaperExact traces agree. Running frames are not read.
#[test]
fn prior_digest_comes_from_the_fresh_public_input() {
    let public = 5 * D;
    let columns = public + D;
    let structure = rectangular_ccs(D / 2, columns);
    let params = selected_parameters(&structure);
    let log = committer(&params, columns);
    let digest: [u64; 4] = [1 << 63 | 5, 0x1234_5678_9abc_def0, 3, 0x7fff_ffff_0000_0001];
    let mut values = vec![F::ZERO; columns];
    values[0] = F::ONE;
    for (word, value) in digest.iter().enumerate() {
        for bit in 0..64 {
            values[1 + 64 * word + bit] = F::from_u64((value >> bit) & 1);
        }
    }
    for (column, value) in values.iter_mut().enumerate().skip(public) {
        *value = if column % 2 == 0 { F::ONE } else { -F::ONE };
    }
    let mut Z = Mat::zero(D, columns / D, F::ZERO);
    for (column, &value) in values.iter().enumerate() {
        Z[(column % D, column / D)] = value;
    }
    let claim = Claim {
        adv: None,
        c: log.commit(&Z),
        x: values[..public].to_vec(),
        m_in: public,
    };
    let witness = CcsWitness {
        w: values[public..].to_vec(),
        Z,
    };
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, public);
    let label = b"pi-ccs/v1_1/prior-digest";
    let (outputs, proof) = crosscheck_prove(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    )
    .expect("prior-digest crosscheck proof");

    let mut reframed = running.clone();
    reframed[0].fold_digest = [7; 32];
    assert!(crosscheck_verify(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &reframed,
        &outputs,
        &proof,
    )
    .expect("a running frame is not transcript input"));

    let (accepted, trace) = optimized_verify_with_trace(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &outputs,
        &proof,
    )
    .expect("prior-digest trace");
    assert!(accepted);
    let mut block = vec![F::from_u64(4)];
    block.extend(digest.map(F::from_u64));
    assert_eq!(trace.events[1], TraceEvent::Absorb(block));
}

#[test]
fn accepting_v1_1_path_exports_receipt_and_rejects_mutations() {
    let columns = D + 1;
    let structure = rectangular_ccs(D / 2, columns);
    let params = selected_parameters(&structure);
    let log = committer(&params, columns);
    let cache = OptimizedStructureCache::build(&structure).expect("structure cache");
    let (claim, witness) = source(&log, columns, 12);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    let label = b"padded-row/execution-receipt";
    let (outputs, proof, _, _) = neo_reductions::optimized_engine::optimized_prove_with_cache_and_precompute_and_perf(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
        &cache,
    )
    .expect("v1_1 production proof");

    let receipt = verify_and_export_pi_ccs_receipt(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &outputs,
        &proof,
        &cache,
    )
    .expect("accepted v1_1 execution receipt");
    assert_eq!(receipt.proof.proof_bytes, proof.canonical_bytes().unwrap());
    assert_eq!(receipt.proof.output_eval_k.len(), outputs.len() * D);
    assert_eq!(receipt.proof.output_eval_a.len(), outputs.len() * structure.t() * D);
    assert_eq!(receipt.statement.relation_id.len(), 4);
    assert_eq!(
        receipt.statement.transcript_absorptions[0],
        vec![
            78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 83, 117, 112, 101, 114, 78, 101, 111, 47, 80, 105,
            67, 67, 83, 47, 100, 105, 103, 101, 115, 116, 45, 111, 110, 108, 121, 47, 118, 49, 95, 49,
        ]
    );

    let mut changed_proof = proof.clone();
    changed_proof.sumcheck_rounds[0][0] += K::ONE;
    assert!(verify_and_export_pi_ccs_receipt(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &outputs,
        &changed_proof,
        &cache,
    )
    .is_err());

    let mut changed_output = outputs.clone();
    changed_output[0].eval_k[0] += K::ONE;
    assert!(verify_and_export_pi_ccs_receipt(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &running,
        &changed_output,
        &proof,
        &cache,
    )
    .is_err());

    let mut changed_claim = claim.clone();
    changed_claim.c.data[0] += F::ONE;
    assert!(verify_and_export_pi_ccs_receipt(
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&changed_claim),
        &running,
        &outputs,
        &proof,
        &cache,
    )
    .is_err());
}

#[test]
fn public_crosscheck_covers_v1_1_rlc_and_dec() {
    let columns = D + 1;
    let structure = rectangular_ccs(D / 2, columns);
    let params = selected_parameters(&structure);
    let log = committer(&params, columns);
    let (claim, witness) = source(&log, columns, 4);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let (outputs, _) = prove_mode(
        mode.clone(),
        b"padded-row/rlc-dec",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    )
    .expect("PiCCS crosscheck");

    let rho = Mat::identity(D);
    let typed_rhos =
        neo_reductions::api::rot_rhos_from_mats(&params, std::slice::from_ref(&rho), "padded-row RLC identity")
            .expect("typed identity rho");
    let (parent, mixed_witness) = rlc_with_commit(
        mode.clone(),
        &structure,
        &params,
        &typed_rhos,
        &outputs[..1],
        std::slice::from_ref(&witness.Z),
        D.next_power_of_two().trailing_zeros() as usize,
        |_, commitments| commitments[0].clone(),
    )
    .expect("PiRLC crosscheck");
    assert_eq!(parent.eval_k.len(), D.next_power_of_two());
    assert_eq!(parent.eval_a.len(), structure.t());

    let split_witnesses =
        split_b_matrix_k(&mixed_witness, params.k_rho as usize, params.b).expect("canonical PiDEC split");
    let child_commitments: Vec<_> = split_witnesses
        .iter()
        .map(|child| log.commit(child))
        .collect();
    let (children, y_valid, x_valid, commitment_valid) = dec_children_with_commit(
        mode,
        &structure,
        &params,
        &parent,
        &split_witnesses,
        D.next_power_of_two().trailing_zeros() as usize,
        &child_commitments,
        combine_commitments_b_pows,
    );
    assert!(y_valid && x_valid && commitment_valid);
    assert_eq!(children[0].eval_k.len(), D.next_power_of_two());
    assert_eq!(children[0].eval_a.len(), structure.t());
}

#[test]
fn verifier_matches_the_paper_mutation_boundary() {
    let structure = rectangular_ccs(D / 2, D);
    let params = selected_parameters(&structure);
    let log = committer(&params, D);
    let (first_claim, first_witness) = source(&log, D, 1);
    let (second_claim, second_witness) = source(&log, D, 2);
    let claims = vec![first_claim, second_claim];
    let witnesses = vec![first_witness, second_witness];
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, claims.len(), claims[0].m_in);
    let label = b"padded-row/mutation";
    let (outputs, proof) = prove_mode(
        FoldingMode::Optimized,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    )
    .expect("proof");
    let rejects = |claims: &[Claim], outputs: &[Output], proof: &PiCcsProof| {
        !matches!(
            mode_verify(
                FoldingMode::Optimized,
                &mut Poseidon2Transcript::new(label),
                &params,
                &structure,
                claims,
                &running,
                outputs,
                proof,
            ),
            Ok(true)
        )
    };

    let mut changed_round = proof.clone();
    changed_round.sumcheck_rounds[0][0] += K::ONE;
    assert!(rejects(&claims, &outputs, &changed_round));

    let mut changed_output = outputs.clone();
    changed_output[0].eval_k[0] += K::ONE;
    assert!(rejects(&claims, &changed_output, &proof));

    // Section 7.3 does not use a fresh source's nonconstant output
    // coefficients in the PiCCS terminal equation. The production transcript
    // still absorbs the complete output message before PiRLC samples rho.
    // This mutation keeps the old transcript digest, so verification must
    // reject the stale binding without adding a terminal equation.
    let mut changed_fresh_ring_coefficient = outputs.clone();
    changed_fresh_ring_coefficient[0].eval_k[1] += K::ONE;
    assert!(rejects(&claims, &changed_fresh_ring_coefficient, &proof));

    let mut changed_order = claims.clone();
    changed_order.reverse();
    assert!(rejects(&changed_order, &outputs, &proof));

    let mut extra_round = proof.clone();
    extra_round
        .sumcheck_rounds
        .push(vec![K::ZERO; proof.sumcheck_rounds[0].len()]);
    assert!(rejects(&claims, &outputs, &extra_round));

    let mut changed_digest = outputs.clone();
    changed_digest[0].fold_digest[0] ^= 1;
    assert!(rejects(&claims, &changed_digest, &proof));
}

#[test]
fn crosscheck_rejects_a_noncanonical_dec_split_that_recomposes() {
    let columns = D;
    let structure = rectangular_ccs(D / 2, columns);
    let params = selected_parameters(&structure);
    assert!(params.k_rho >= 2);
    let log = committer(&params, columns);
    let (claim, witness) = source(&log, columns, 4);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let (outputs, _) = prove_mode(
        mode.clone(),
        b"padded-row/noncanonical-dec",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    )
    .expect("PiCCS crosscheck");
    let rho = Mat::identity(D);
    let typed_rhos = neo_reductions::api::rot_rhos_from_mats(&params, std::slice::from_ref(&rho), "noncanonical DEC")
        .expect("typed identity rho");
    let (parent, mixed_witness) = rlc_with_commit(
        mode.clone(),
        &structure,
        &params,
        &typed_rhos,
        &outputs[..1],
        std::slice::from_ref(&witness.Z),
        D.next_power_of_two().trailing_zeros() as usize,
        |_, commitments| commitments[0].clone(),
    )
    .expect("PiRLC crosscheck");
    let mut split = split_b_matrix_k(&mixed_witness, params.k_rho as usize, params.b).expect("canonical split");
    let coordinate = (0..split[0].rows())
        .flat_map(|row| (0..split[0].cols()).map(move |column| (row, column)))
        .find(|&(row, column)| split[0][(row, column)] != F::ZERO && split[1][(row, column)] == F::ZERO)
        .expect("fixture has a one-digit nonzero value");
    let first = split[0][coordinate];
    split[0][coordinate] = -first;
    split[1][coordinate] = first;
    let child_commitments: Vec<_> = split.iter().map(|child| log.commit(child)).collect();

    let (_, y_valid, x_valid, commitment_valid) = dec_children_with_commit(
        mode,
        &structure,
        &params,
        &parent,
        &split,
        D.next_power_of_two().trailing_zeros() as usize,
        &child_commitments,
        combine_commitments_b_pows,
    );
    assert!(!y_valid && !x_valid && !commitment_valid);
}

#[test]
fn full_carrier_tail_is_part_of_the_padded_identity_relation() {
    let columns = 257;
    let structure = rectangular_ccs(8, columns);
    let params = selected_parameters(&structure);
    let log = committer(&params, columns);
    let (claim, mut witness) = source(&log, columns, 1);
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, claim.m_in);
    witness.Z[(257 % D, 257 / D)] = F::from_u64(2);
    let result = prove_mode(
        FoldingMode::OptimizedWithCrosscheck,
        b"padded-row/full-carrier",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &log,
    );
    let error = result
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(error.contains("fresh carrier_col=257 must be zero"), "{error}");
}

#[test]
fn v1_1_eval_k_and_eval_a_gamma_slots_are_separate() {
    assert_eq!(eval_k_gamma_exponent(2, 0, 0), 0);
    assert_eq!(eval_k_gamma_exponent(2, 1, 0), 1);
    assert_eq!(eval_k_gamma_exponent(2, 0, 1), 2);

    assert_eq!(eval_a_gamma_exponent(2, 3, 0, 0, 0), 0);
    assert_eq!(eval_a_gamma_exponent(2, 3, 1, 0, 0), 1);
    assert_eq!(eval_a_gamma_exponent(2, 3, 0, 1, 0), 2);
    assert_eq!(eval_a_gamma_exponent(2, 3, 0, 0, 1), 6);
}

#[test]
fn carried_gamma_power_handles_protocol_scale_exponents() {
    let gamma = K::from_coeffs([F::from_u64(3), F::from_u64(5)]);
    let squarings = ((usize::BITS - 2) as usize).min(40);
    let exponent = 1usize << squarings;
    let expected = (0..squarings).fold(gamma, |power, _| power * power);

    assert_eq!(gamma_power(gamma, exponent), expected);
    for exponent in 0..128 {
        let linear = (0..exponent).fold(K::ONE, |power, _| power * gamma);
        assert_eq!(gamma_power(gamma, exponent), linear);
    }
}

#[test]
fn canonical_codec_is_versioned_and_not_bincode() {
    let proof = PiCcsProof::new(vec![vec![K::ZERO; 2]]);
    let bytes = proof.canonical_bytes().unwrap();
    assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1102);
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
}

#[test]
fn paper_exact_sources_do_not_import_optimized_computation() -> Result<(), PiCcsError> {
    let sources = [
        include_str!("../src/engines/paper_exact_engine/mod.rs"),
        include_str!("../src/engines/paper_exact_engine/prove.rs"),
        include_str!("../src/engines/paper_exact_engine/verify.rs"),
        include_str!("../src/engines/paper_exact_engine/paper_joint.rs"),
        include_str!("../src/engines/paper_exact_engine/paper_matrix.rs"),
        include_str!("../src/engines/paper_exact_engine/paper_ring.rs"),
        include_str!("../src/engines/paper_exact_engine/transcript.rs"),
        include_str!("../src/engines/paper_exact_engine/rlc_dec.rs"),
    ];
    for forbidden in [
        "crate::optimized_engine",
        "engines::optimized_engine",
        "SuperneoEvalCache",
        "eval_all_mats_cached",
        "eval_all_mats_ring_cached",
        "build_joint_dims",
        "shared_me_input_r",
        "interpolate_from_evals",
        "poly_eval_k",
        "project_x_from_witness_mat",
        ".canonicalize()",
        "superneo_bar_block",
        "Rq::",
        "block.entry",
        "run.entry",
        "chi_table",
        "validate_rhos_are_rotation_matrices",
    ] {
        if sources.iter().any(|source| source.contains(forbidden)) {
            return Err(PiCcsError::ProtocolError(format!(
                "PaperExact contains forbidden dependency: {forbidden}"
            )));
        }
    }
    Ok(())
}
