#[path = "../support/mod.rs"]
mod support;

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs::{
    self, AcceleratorCrosscheckNifsProver, CrosscheckNifsProver, NifsProverAdapter, NifsProverOutput,
    NifsProverRequest, OptimizedCpuNifsProver, OptimizedNifsProverAdapter, PaperExactNifsProver,
};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

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
        &RunningInstance::default(),
    )
    .expect("rectangular PaperExact NIFS matches optimized CPU");
}

#[test]
fn paper_exact_and_optimized_cpu_nifs_are_byte_exact() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 101), support::toy_instance(&prep, 103)];
    let running = RunningInstance::default();

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
        &RunningInstance::default(),
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
        &RunningInstance::default(),
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
    let running = RunningInstance::default();
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
    rlc_mutation.pi_rlc.combined.y_ring[0][0] += K::ONE;
    rlc_mutation.pi_rlc.combined.ct[0] += K::ONE;
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
    dec_mutation.pi_dec.children[0].y_ring[0][0] += K::ONE;
    dec_mutation.pi_dec.children[0].ct[0] += K::ONE;
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
        &RunningInstance::default(),
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
    ) -> Result<NifsProverOutput, neo_fold_clean::paper::nifs::Error> {
        OptimizedCpuNifsProver.prove(request)
    }
}

impl OptimizedNifsProverAdapter for PassThroughOptimized {}

struct MutatingOptimized;

impl NifsProverAdapter for MutatingOptimized {
    fn prove(
        &mut self,
        request: NifsProverRequest<'_>,
    ) -> Result<NifsProverOutput, neo_fold_clean::paper::nifs::Error> {
        let (running, mut proof) = OptimizedCpuNifsProver
            .prove(request)?
            .into_materialized_parts()?;
        proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
        Ok(NifsProverOutput::materialized(running, proof))
    }
}

impl OptimizedNifsProverAdapter for MutatingOptimized {}

struct MutatingWitnessOptimized;

impl NifsProverAdapter for MutatingWitnessOptimized {
    fn prove(
        &mut self,
        request: NifsProverRequest<'_>,
    ) -> Result<NifsProverOutput, neo_fold_clean::paper::nifs::Error> {
        let (mut running, proof) = OptimizedCpuNifsProver
            .prove(request)?
            .into_materialized_parts()?;
        running.witnesses[0][(0, 0)] += F::ONE;
        Ok(NifsProverOutput::materialized(running, proof))
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
        &RunningInstance::default(),
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
        &RunningInstance::default(),
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
        &RunningInstance::default(),
    )
    .is_err());
}
