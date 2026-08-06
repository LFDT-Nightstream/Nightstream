#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, CscMat, Mat, SeededPhi81LinearBlock, SparsePoly, Term,
};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{dec_children_with_commit, prove, rlc_with_commit, verify, FoldingMode};
use neo_reductions::engines::crosscheck_engine::{crosscheck_prove_with_binding, crosscheck_verify_with_binding};
use neo_reductions::engines::paper_exact_engine::paper_joint::PaperJointOracle;
use neo_reductions::engines::pi_ccs_joint::{build_joint_dims, carried_gamma_exponent};
use neo_reductions::engines::pi_ccs_joint_protocol::TranscriptBinding;
use neo_reductions::engines::pi_ccs_protocol::Challenges;
use neo_reductions::optimized_engine::canonical_audit::OptimizedPaperJointOracle;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_reductions::sumcheck::RoundOracle;
use neo_reductions::{split_b_matrix_k, PiCcsError, PiCcsProof};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

fn seeded_phi81_ccs() -> CcsStructure<F> {
    let seed = [0x6D; 32];
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, 1, 1);
    let block = SeededPhi81LinearBlock::new_with_word_width(0, vec![1], 41, 1, 1, chunk_size, chunk_seeds)
        .expect("seeded Phi81 block");
    let matrix = CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), D, D + 1), vec![block])
        .expect("seeded Phi81 matrix");
    CcsStructure::new_sparse(vec![matrix], SparsePoly::new(1, Vec::new())).expect("seeded Phi81 CCS")
}

fn transformed_seeded_phi81_ccs() -> CcsStructure<F> {
    let seed = [0x6D; 32];
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, 1, 1);
    let block = SeededPhi81LinearBlock::new_with_word_width(0, vec![1], 41, 1, 1, chunk_size, chunk_seeds)
        .expect("seeded Phi81 block")
        .with_superneo_transformed_columns();
    let matrix = CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), D, D + 1), vec![block])
        .expect("seeded Phi81 matrix");
    CcsStructure::new_sparse(vec![matrix], SparsePoly::new(1, Vec::new())).expect("seeded Phi81 CCS")
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
    let values: Vec<F> = (0..columns)
        .map(|column| match (seed + 5 * column) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let mut Z = Mat::zero(D, columns.div_ceil(D), F::ZERO);
    for (column, &value) in values.iter().enumerate() {
        Z[(column % D, column / D)] = value;
    }
    // The paper public-input projection is a ring-module projection. This
    // small fixture has no complete public ring, so it uses an empty public
    // prefix instead of exposing a partial ring.
    let m_in = 0;
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
    prove(
        mode,
        &mut Poseidon2Transcript::new(label),
        params,
        structure,
        claims,
        witnesses,
        running,
        running_witnesses,
        log,
    )
}

fn seed_running(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &Claim,
    witness: &CcsWitness<F>,
    log: &AjtaiSModule,
) -> Output {
    prove_mode(
        FoldingMode::PaperExact,
        b"padded-row/seed",
        params,
        structure,
        std::slice::from_ref(claim),
        std::slice::from_ref(witness),
        &[],
        &[],
        log,
    )
    .expect("seed proof")
    .0
    .remove(0)
}

fn assert_parity(rows: usize, columns: usize) {
    let structure = rectangular_ccs(rows, columns);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(rows.max(columns)).expect("parameters");
    let log = committer(&params, columns);
    let (prior_claim, prior_witness) = source(&log, columns, 1);
    let running = vec![seed_running(&params, &structure, &prior_claim, &prior_witness, &log)];
    let running_witnesses = vec![prior_witness.Z];
    let (first_claim, first_witness) = source(&log, columns, 2);
    let (second_claim, second_witness) = source(&log, columns, 7);
    let claims = vec![first_claim, second_claim];
    let witnesses = vec![first_witness, second_witness];
    let label = b"padded-row/parity";

    let (paper_outputs, paper_proof) = prove_mode(
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
    let (optimized_outputs, optimized_proof) = prove_mode(
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
    assert!(paper_outputs
        .iter()
        .all(|output| output.y_ring.len() == structure.t() + 1));
    assert_eq!(paper_proof.canonical_bytes(), optimized_proof.canonical_bytes());

    for mode in [FoldingMode::PaperExact, FoldingMode::Optimized] {
        assert!(verify(
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
        let params = NeoParams::goldilocks_auto_r1cs_ccs(rows.max(columns)).expect("parameters");
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
            &cache,
        )
        .expect("optimized oracle");
        let points: Vec<K> = (0..=dims.degree)
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        for round in 0..dims.variables {
            assert_eq!(paper.evals_at(&points), optimized.evals_at(&points), "round {round}");
            let challenge = K::from(F::from_u64((19 + 3 * round) as u64));
            paper.fold(challenge);
            optimized.fold(challenge);
        }
    }
}

#[test]
fn public_crosscheck_compares_the_complete_execution() {
    let structure = rectangular_ccs(D / 2, D + 1);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D + 1).expect("parameters");
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 5);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let label = b"padded-row/crosscheck";
    let (outputs, proof) = prove_mode(
        mode.clone(),
        label,
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
        &log,
    )
    .expect("crosscheck proof");
    assert!(verify(
        mode,
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &[],
        &outputs,
        &proof,
    )
    .expect("crosscheck verify"));
}

#[test]
fn compact_recursive_transcript_matches_the_independent_reference() {
    let structure = rectangular_ccs(D / 2, D + 1);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D + 1).expect("parameters");
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 6);
    let label = b"padded-row/compact-crosscheck";
    let public_instance_digest = [F::from_u64(11), F::from_u64(12), F::from_u64(13), F::from_u64(14)];
    let running_handle = [F::from_u64(21), F::from_u64(22), F::from_u64(23), F::from_u64(24)];
    let binding = TranscriptBinding::digest_and_handle(public_instance_digest, running_handle);
    let (outputs, proof) = crosscheck_prove_with_binding(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
        &log,
        binding,
    )
    .expect("compact transcript crosscheck proof");
    assert!(crosscheck_verify_with_binding(
        &(),
        &(),
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &[],
        &outputs,
        &proof,
        binding,
    )
    .expect("compact transcript crosscheck verify"));
}

#[test]
fn public_crosscheck_covers_seeded_phi81_matrix_descriptors() {
    let structure = seeded_phi81_ccs();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D + 1).expect("parameters");
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 9);
    let label = b"padded-row/seeded-phi81";
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let (outputs, proof) = prove_mode(
        mode.clone(),
        label,
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
        &log,
    )
    .expect("seeded Phi81 crosscheck proof");
    assert!(verify(
        mode,
        &mut Poseidon2Transcript::new(label),
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &[],
        &outputs,
        &proof,
    )
    .expect("seeded Phi81 crosscheck verify"));
}

#[test]
fn selected_engines_reject_pretransformed_seeded_matrix_descriptors() {
    let structure = transformed_seeded_phi81_ccs();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D + 1).expect("parameters");
    let log = committer(&params, D + 1);
    let (claim, witness) = source(&log, D + 1, 9);

    for mode in [FoldingMode::PaperExact, FoldingMode::Optimized] {
        let error = prove_mode(
            mode,
            b"padded-row/pretransformed-seeded-phi81",
            &params,
            &structure,
            std::slice::from_ref(&claim),
            std::slice::from_ref(&witness),
            &[],
            &[],
            &log,
        )
        .expect_err("the selected protocol must apply the paper transform exactly once");
        assert!(
            error.to_string().contains("untransformed CCS matrices"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn public_crosscheck_covers_identity_first_rlc_and_dec() {
    let columns = D + 1;
    let structure = rectangular_ccs(D / 2, columns);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(columns).expect("parameters");
    let log = committer(&params, columns);
    let (claim, witness) = source(&log, columns, 4);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let (outputs, _) = prove_mode(
        mode.clone(),
        b"padded-row/rlc-dec",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
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
        &outputs,
        std::slice::from_ref(&witness.Z),
        D.next_power_of_two().trailing_zeros() as usize,
        |_, commitments| commitments[0].clone(),
    )
    .expect("PiRLC crosscheck");
    assert_eq!(parent.y_ring.len(), structure.t() + 1);

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
    assert_eq!(children[0].y_ring.len(), structure.t() + 1);
}

#[test]
fn verifier_matches_the_paper_mutation_boundary() {
    let structure = rectangular_ccs(D / 2, D);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("parameters");
    let log = committer(&params, D);
    let (first_claim, first_witness) = source(&log, D, 1);
    let (second_claim, second_witness) = source(&log, D, 2);
    let claims = vec![first_claim, second_claim];
    let witnesses = vec![first_witness, second_witness];
    let label = b"padded-row/mutation";
    let (outputs, proof) = prove_mode(
        FoldingMode::Optimized,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &[],
        &[],
        &log,
    )
    .expect("proof");
    let rejects = |claims: &[Claim], outputs: &[Output], proof: &PiCcsProof| {
        !matches!(
            verify(
                FoldingMode::Optimized,
                &mut Poseidon2Transcript::new(label),
                &params,
                &structure,
                claims,
                &[],
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
    changed_output[0].y_ring[0][0] += K::ONE;
    changed_output[0].ct[0] += K::ONE;
    assert!(rejects(&claims, &changed_output, &proof));

    // Section 7.3 does not use a fresh source's nonconstant output
    // coefficients in the PiCCS terminal equation. PiCCS therefore accepts
    // this mutation. PiRLC binds the complete output message before it samples
    // rho, and the next PiCCS invocation checks the coefficient as carried
    // evaluation data. Rejecting it here would add a non-paper equation.
    let mut changed_fresh_ring_coefficient = outputs.clone();
    changed_fresh_ring_coefficient[0].y_ring[0][1] += K::ONE;
    assert!(!rejects(&claims, &changed_fresh_ring_coefficient, &proof));

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
    let params = NeoParams::goldilocks_auto_r1cs_ccs(columns).expect("parameters");
    assert!(params.k_rho >= 2);
    let log = committer(&params, columns);
    let (claim, witness) = source(&log, columns, 4);
    let mode = FoldingMode::OptimizedWithCrosscheck;
    let (outputs, _) = prove_mode(
        mode.clone(),
        b"padded-row/noncanonical-dec",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
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
        &outputs,
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
    let params = NeoParams::goldilocks_auto_r1cs_ccs(512).expect("parameters");
    let log = committer(&params, columns);
    let (claim, mut witness) = source(&log, columns, 1);
    witness.Z[(257 % D, 257 / D)] = F::from_u64(2);
    let result = prove_mode(
        FoldingMode::OptimizedWithCrosscheck,
        b"padded-row/full-carrier",
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
        &log,
    );
    assert!(result.is_err());
}

#[test]
fn carried_gamma_slots_include_the_identity_matrix() {
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 0, 0), 6);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 1, 0, 0), 7);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 1, 0), 8);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 0, 1), 12);
}

#[test]
fn canonical_codec_is_versioned_and_not_bincode() {
    let proof = PiCcsProof::new(vec![vec![K::ZERO; 2]]);
    let bytes = proof.canonical_bytes();
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
