#[path = "../support/mod.rs"]
mod support;

use std::{fs, path::PathBuf};

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage, PiCcsV1_1ProofInputs,
};
use neo_fold_clean::paper::construction2::{LaneCommitmentMode, RunningInstance};
use neo_fold_clean::paper::nifs::{
    self, AcceleratorCrosscheckNifsProver, CrosscheckNifsProver, NifsProof, NifsProverAdapter, NifsProverRequest,
    OptimizedCpuNifsProver, OptimizedNifsProverAdapter, PaperExactNifsProver,
};
use neo_fold_clean::paper::relations::{CeClaim, LaneRanges, LaneScheme};
use neo_math::{KExtensions, D, F, K};
use nightstream_fprime::{
    load, PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT,
    PI_CCS_V1_1_STATE_PREIMAGE_WORDS,
};
use p3_field::PrimeCharacteristicRing;

// Validated phase-local Pilot + PiCCS + PiRLC + PiDEC package identity.
// The final Stage 1 package must rerun every gate before replacing it.
const PACKAGE_IDENTITY: [u64; 4] = [
    12_756_407_480_944_487_176,
    17_097_603_764_386_178_571,
    11_791_428_871_054_057_896,
    14_346_937_702_828_624_285,
];

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn parity_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn canonical_running(prep: &neo_fold_clean::Preprocessing) -> RunningInstance {
    RunningInstance::canonical_zero(&prep.params, prep.structure(), D, LaneCommitmentMode::Plain)
        .expect("canonical nonempty SuperNeo accumulator")
}

fn rectangular_relation(rows: usize, columns: usize) -> R1cs {
    let mut a = Mat::zero(rows, columns, F::ZERO);
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;
    let mut b = Mat::zero(rows, columns, F::ZERO);
    b[(0, 0)] = F::ONE;
    let mut c = Mat::zero(rows, columns, F::ZERO);
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: D }
}

fn rectangular_assignment(columns: usize, lhs: F, rhs: F) -> Vec<F> {
    let mut assignment = vec![F::ZERO; columns];
    assignment[0] = F::ONE;
    assignment[1] = lhs;
    assignment[2] = rhs;
    assignment[3] = lhs + rhs;
    assignment
}

fn crosscheck_rectangular_case(rows: usize, columns: usize, seed: u64) {
    let r1cs = rectangular_relation(rows, columns);
    let prep = direct_ccs::preprocess_seeded(&r1cs, seed).expect("rectangular preprocess");
    let fresh = [(F::ONE, F::ZERO), (F::ZERO, F::ONE), (-F::ONE, F::ONE)]
        .into_iter()
        .map(|(lhs, rhs)| {
            direct_ccs::build_instance(&prep, &r1cs, &rectangular_assignment(columns, lhs, rhs))
                .expect("rectangular fresh instance")
        })
        .collect();
    let mut prover = CrosscheckNifsProver;
    let mut transcript = Transcript::session();
    nifs::prove_with_adapter(
        &mut prover,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &canonical_running(&prep),
    )
    .expect("rectangular PaperExact NIFS matches optimized CPU");
}

#[test]
fn paper_exact_and_optimized_cpu_nifs_are_byte_exact() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 101), support::toy_instance(&prep, 103)];
    let running = canonical_running(&prep);

    let mut optimized = OptimizedCpuNifsProver;
    let mut optimized_transcript = Transcript::session();
    let (optimized_running, optimized_proof) = nifs::prove_with_adapter(
        &mut optimized,
        &mut optimized_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh.clone(),
        &running,
    )
    .expect("optimized NIFS");

    let mut paper_exact = PaperExactNifsProver;
    let mut reference_transcript = Transcript::session();
    let (reference_running, reference_proof) = nifs::prove_with_adapter(
        &mut paper_exact,
        &mut reference_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("PaperExact NIFS");

    nifs::require_nifs_execution_match(
        optimized_transcript.snapshot(),
        &optimized_running,
        &optimized_proof,
        reference_transcript.snapshot(),
        &reference_running,
        &reference_proof,
    )
    .expect("complete NIFS executions match");
    assert!(optimized_proof
        .canonical_bytes()
        .starts_with(b"NS-NIFS-PROOF"));
}

#[test]
fn nonzero_pi_ccs_messages_map_to_the_lean_package_without_offsets() {
    let prep = support::toy_preprocessing();
    let fresh_claim = support::toy_instance(&prep, 127).claim;
    let extension =
        |first: usize, second: usize| neo_math::from_complex(F::from_u64(first as u64), F::from_u64(second as u64));
    let rounds: Vec<Vec<K>> = (0..PI_CCS_V1_1_ROUND_COUNT)
        .map(|round| {
            (0..PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
                .map(|coefficient| extension(1 + round * 17 + coefficient, 10_001 + round * 19 + coefficient))
                .collect()
        })
        .collect();
    let outputs = (0..PI_CCS_V1_1_SOURCE_COUNT)
        .map(|source| {
            let mut eval_k = vec![K::ZERO; D.next_power_of_two()];
            for (coefficient, value) in eval_k[..PI_CCS_V1_1_COEFFICIENT_COUNT]
                .iter_mut()
                .enumerate()
            {
                *value = extension(20_000 + source * 100 + coefficient, 30_000 + source * 100 + coefficient);
            }
            let mut eval_a = vec![vec![K::ZERO; D.next_power_of_two()]; PI_CCS_V1_1_MATRIX_COUNT];
            for (matrix, family) in eval_a.iter_mut().enumerate() {
                for (coefficient, value) in family[..PI_CCS_V1_1_COEFFICIENT_COUNT]
                    .iter_mut()
                    .enumerate()
                {
                    let ordinal = source * PI_CCS_V1_1_MATRIX_COUNT * PI_CCS_V1_1_COEFFICIENT_COUNT
                        + matrix * PI_CCS_V1_1_COEFFICIENT_COUNT
                        + coefficient;
                    *value = extension(40_000 + ordinal, 60_000 + ordinal);
                }
            }
            CeClaim {
                c: fresh_claim.c.clone(),
                X: Mat::zero(D, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D, F::ZERO),
                r: vec![K::ZERO; PI_CCS_V1_1_ROUND_COUNT],
                eval_k,
                eval_a,
                m_in: PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
                fold_digest: [source as u8; 32],
                adv: None,
            }
        })
        .collect();
    let proof = neo_fold_clean::paper::pi_ccs::Proof {
        sumcheck: neo_fold_clean::paper::pi_ccs::SumcheckProof::new(rounds.clone()),
        outputs,
    };

    let bridge = PiCcsV1_1ProofInputs::from_proof(std::slice::from_ref(&fresh_claim), &proof)
        .expect("exact v1_1 PiCCS proof-message bridge");
    assert_eq!(bridge.fresh_commitment().len(), fresh_claim.c.data.len());
    for (actual, expected) in bridge.round_messages().iter().zip(&rounds) {
        for (actual, expected) in actual.iter().zip(expected) {
            let (low, high) = expected.to_limbs_u64();
            assert_eq!(*actual, [low, high]);
        }
    }

    let expected_outputs = bridge.output_evaluations().clone();
    let running = (0..16)
        .map(|_| CeClaim {
            c: neo_ajtai::Commitment::zeros(D, 18),
            X: Mat::zero(D, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D, F::ZERO),
            r: vec![K::ZERO; PI_CCS_V1_1_ROUND_COUNT],
            eval_k: vec![K::ZERO; D.next_power_of_two()],
            eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; PI_CCS_V1_1_MATRIX_COUNT],
            m_in: PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
            fold_digest: [0; 32],
            adv: None,
        })
        .collect::<Vec<_>>();
    let parity: serde_json::Value =
        serde_json::from_slice(&fs::read(parity_path()).expect("Lean parity bytes")).expect("Lean parity JSON");
    let parity = parity.as_array().expect("Lean parity tuple");
    assert_eq!(parity[0].as_u64(), Some(7));
    let parity_input = parity[1].as_array().expect("Lean parity input tuple");
    let authority: Vec<Vec<u64>> =
        serde_json::from_value(parity_input[11].clone()).expect("Lean verifier-context authority");
    let package = load(&fs::read(package_path()).expect("Lean package bytes"), PACKAGE_IDENTITY)
        .expect("verifier-owned Lean package");
    let verifier_context = package
        .derive_pi_ccs_v1_1_verifier_context(&authority[3])
        .expect("package-bound verifier context");
    assert_eq!(verifier_context.relation_words(), authority[0]);
    assert_eq!(verifier_context.application_words(), authority[1]);
    assert_eq!(verifier_context.nifs_key_words(), authority[2]);
    assert_eq!(verifier_context.commitment_key_words(), authority[3]);
    let verifier_key_digest = verifier_context.digest().map(F::from_u64);
    let z0 = [F::from_u64(201), F::from_u64(202), F::from_u64(203), F::from_u64(204)];
    let current = [F::from_u64(301), F::from_u64(302), F::from_u64(303), F::from_u64(304)];
    let prior_preimage = serialize_pi_ccs_v1_1_state_preimage(verifier_key_digest, 7, z0, current, &running, 1)
        .expect("canonical Lean prior-state preimage");
    assert_eq!(prior_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(
        &prior_preimage[..23],
        [72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67, 47, 115, 116, 97, 116, 101, 47, 118, 49]
    );
    assert_eq!(prior_preimage[23], 4);
    assert_eq!(prior_preimage[28], 7);
    assert_eq!(prior_preimage[29], 4);
    assert_eq!(prior_preimage[34], 4);
    let running_point_words = 2 * PI_CCS_V1_1_ROUND_COUNT;
    assert_eq!(prior_preimage[39], running_point_words as u64);
    assert_eq!(prior_preimage[40 + running_point_words], 972);
    assert_eq!(*prior_preimage.last().expect("program counter"), 1);
    let digest = pi_ccs_v1_1_state_hash(&prior_preimage).expect("Lean stateHash replay");
    let prior_public_input = encode_pi_ccs_v1_1_public_input(digest).expect("Lean encHash replay");
    assert_eq!(prior_public_input[0], 1);
    for (word, value) in digest.iter().copied().enumerate() {
        for bit in 0..64 {
            assert_eq!(prior_public_input[1 + word * 64 + bit], (value >> bit) & 1);
        }
    }
    assert!(prior_public_input[257..].iter().all(|word| *word == 0));

    let lean_preimage: Vec<u64> = serde_json::from_value(parity_input[0].clone()).expect("Lean state preimage");
    let lean_digest: [u64; 4] = serde_json::from_value(parity_input[3].clone()).expect("Lean state digest");
    let lean_context: [u64; 4] = serde_json::from_value(parity_input[4].clone()).expect("Lean verifier context");
    let lean_public_input: Vec<u64> = serde_json::from_value(parity_input[2].clone()).expect("Lean state public input");
    assert_eq!(
        pi_ccs_v1_1_state_hash(&lean_preimage).expect("Lean preimage replay"),
        lean_digest
    );
    assert_eq!(
        encode_pi_ccs_v1_1_public_input(lean_digest).expect("Lean public-input replay"),
        lean_public_input,
    );
    assert_eq!(&lean_preimage[24..28], lean_context);

    let inputs = bridge
        .into_package_inputs(
            prior_preimage.clone(),
            prior_preimage,
            prior_public_input,
            digest,
            verifier_context,
        )
        .expect("complete package input value");
    let encoded = package
        .encode_pi_ccs_v1_1_inputs(&inputs)
        .expect("package-owned physical encoding");
    assert!(encoded.private_values().iter().any(|value| *value != 0));
    let decoded = package
        .pi_ccs_v1_1_output_evaluations(encoded.private_values())
        .expect("package output decoder");
    assert_eq!(decoded, expected_outputs);
}

#[test]
fn paper_exact_and_optimized_cpu_match_both_rectangular_directions() {
    crosscheck_rectangular_case(2, 2 * D, 0x4e49_4653_524c_5431);
    crosscheck_rectangular_case(2 * D, D, 0x4e49_4653_5247_5431);
}

#[test]
fn crosscheck_nifs_covers_a_carried_accumulator() {
    let prep = support::toy_preprocessing();
    let mut crosscheck = CrosscheckNifsProver;

    let mut first_transcript = Transcript::session();
    let (running, _) = nifs::prove_with_adapter(
        &mut crosscheck,
        &mut first_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![support::toy_instance(&prep, 107)],
        &canonical_running(&prep),
    )
    .expect("first crosschecked NIFS fold");

    let fresh = vec![support::toy_instance(&prep, 109)];
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let mut second_transcript = Transcript::session();
    let (next_running, proof) = nifs::prove_with_adapter(
        &mut crosscheck,
        &mut second_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("second crosschecked NIFS fold");

    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("verify crosschecked NIFS proof");
    assert_eq!(verified.claims, next_running.claims);
    assert_eq!(verified.parent_authority, next_running.parent_authority);
}

#[test]
fn crosscheck_nifs_covers_carried_auxiliary_commitments() {
    let columns = 3 * D;
    let r1cs = rectangular_relation(2, columns);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4e49_4653_4144_5631).expect("adv preprocess");
    let lanes = LaneScheme::from_seeds(
        prep.params.kappa() as usize,
        LaneRanges {
            ops: 0..1,
            is: 1..2,
            fs: 2..3,
        },
        [0xA5; 32],
        [0x5A; 32],
    )
    .expect("adv lane scheme");
    let instance = |lhs, rhs| {
        let mut instance = direct_ccs::build_instance(&prep, &r1cs, &rectangular_assignment(columns, lhs, rhs))
            .expect("adv fresh instance");
        instance.claim.adv = Some(lanes.commit(&instance.witness.Z).expect("adv commitment"));
        instance
    };
    let initial_running =
        RunningInstance::canonical_zero(&prep.params, prep.structure(), D, LaneCommitmentMode::Nebula)
            .expect("canonical Nebula accumulator");

    let mut prover = CrosscheckNifsProver;
    let mut first_transcript = Transcript::session();
    let (running, _) = nifs::prove_with_adapter(
        &mut prover,
        &mut first_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        Some(&lanes),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![instance(F::ONE, F::ZERO)],
        &initial_running,
    )
    .expect("first adv crosscheck");

    let mut second_transcript = Transcript::session();
    nifs::prove_with_adapter(
        &mut prover,
        &mut second_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        Some(&lanes),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![instance(F::ZERO, F::ONE)],
        &running,
    )
    .expect("carried adv crosscheck");
}

#[test]
fn paper_exact_verifier_rejects_pi_rlc_and_pi_dec_value_mutations() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 111)];
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let running = canonical_running(&prep);
    let mut prover_transcript = Transcript::session();
    let (_, proof) = nifs::prove_with_adapter(
        &mut OptimizedCpuNifsProver,
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("optimized NIFS");

    let mut rlc_mutation = proof.clone();
    rlc_mutation.pi_rlc.combined.eval_k[0] += K::ONE;
    let mut rlc_transcript = Transcript::session();
    assert!(nifs::verify_paper_exact(
        &mut rlc_transcript,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &rlc_mutation,
    )
    .is_err());

    let mut dec_mutation = proof;
    dec_mutation.pi_dec.children[0].eval_k[0] += K::ONE;
    let mut dec_transcript = Transcript::session();
    assert!(nifs::verify_paper_exact(
        &mut dec_transcript,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &dec_mutation,
    )
    .is_err());
}

#[test]
fn complete_nifs_comparator_rejects_a_round_mutation() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 113)];
    let mut optimized = OptimizedCpuNifsProver;
    let mut transcript = Transcript::session();
    let (running, proof) = nifs::prove_with_adapter(
        &mut optimized,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &canonical_running(&prep),
    )
    .expect("optimized NIFS");

    let mut changed_transcript = Transcript::session();
    changed_transcript.restore_snapshot(transcript.snapshot());
    changed_transcript.append_fields(b"test/nifs_crosscheck/extra", &[F::ONE]);
    assert!(nifs::require_nifs_execution_match(
        transcript.snapshot(),
        &running,
        &proof,
        changed_transcript.snapshot(),
        &running,
        &proof,
    )
    .is_err());

    let mut mutated = proof.clone();
    mutated.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    assert!(nifs::require_nifs_execution_match(
        transcript.snapshot(),
        &running,
        &proof,
        transcript.snapshot(),
        &running,
        &mutated,
    )
    .is_err());
}

struct PassThroughOptimized;

impl NifsProverAdapter for PassThroughOptimized {
    fn prove(
        &mut self,
        request: NifsProverRequest<'_>,
    ) -> Result<(RunningInstance, NifsProof), neo_fold_clean::paper::nifs::Error> {
        OptimizedCpuNifsProver.prove(request)
    }
}

impl OptimizedNifsProverAdapter for PassThroughOptimized {}

struct MutatingOptimized;

impl NifsProverAdapter for MutatingOptimized {
    fn prove(
        &mut self,
        request: NifsProverRequest<'_>,
    ) -> Result<(RunningInstance, NifsProof), neo_fold_clean::paper::nifs::Error> {
        let (running, mut proof) = OptimizedCpuNifsProver.prove(request)?;
        proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
        Ok((running, proof))
    }
}

impl OptimizedNifsProverAdapter for MutatingOptimized {}

struct MutatingWitnessOptimized;

impl NifsProverAdapter for MutatingWitnessOptimized {
    fn prove(
        &mut self,
        request: NifsProverRequest<'_>,
    ) -> Result<(RunningInstance, NifsProof), neo_fold_clean::paper::nifs::Error> {
        let (mut running, proof) = OptimizedCpuNifsProver.prove(request)?;
        running.witnesses[0][(0, 0)] += F::ONE;
        Ok((running, proof))
    }
}

impl OptimizedNifsProverAdapter for MutatingWitnessOptimized {}

#[test]
fn accelerator_crosscheck_accepts_an_exact_optimized_backend() {
    let prep = support::toy_preprocessing();
    let mut crosscheck = AcceleratorCrosscheckNifsProver::new(PassThroughOptimized);
    let mut transcript = Transcript::session();
    nifs::prove_with_adapter(
        &mut crosscheck,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![support::toy_instance(&prep, 127)],
        &canonical_running(&prep),
    )
    .expect("accelerator crosscheck accepts exact optimized output");
}

#[test]
fn accelerator_crosscheck_rejects_a_backend_round_mutation() {
    let prep = support::toy_preprocessing();
    let mut crosscheck = AcceleratorCrosscheckNifsProver::new(MutatingOptimized);
    let mut transcript = Transcript::session();
    assert!(nifs::prove_with_adapter(
        &mut crosscheck,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![support::toy_instance(&prep, 131)],
        &canonical_running(&prep),
    )
    .is_err());
}

#[test]
fn accelerator_crosscheck_rejects_a_backend_witness_mutation() {
    let prep = support::toy_preprocessing();
    let mut crosscheck = AcceleratorCrosscheckNifsProver::new(MutatingWitnessOptimized);
    let mut transcript = Transcript::session();
    assert!(nifs::prove_with_adapter(
        &mut crosscheck,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![support::toy_instance(&prep, 137)],
        &canonical_running(&prep),
    )
    .is_err());
}
