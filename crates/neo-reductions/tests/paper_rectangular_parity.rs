#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{prove, verify, FoldingMode};
use neo_reductions::engines::paper_exact_engine::paper_rectangular::{
    paper_carried_gamma_exponent, PaperJointSquareOracle, PaperRectangularFeOracle, PaperRectangularNcOracle,
};
use neo_reductions::engines::pi_ccs_protocol::{carried_gamma_exponent, Challenges};
use neo_reductions::engines::{CrossCheckEngine, CrosscheckCfg, PiCcsEngine};
use neo_reductions::optimized_engine::canonical_audit::{
    OptimizedPaperRectangularFeOracle, OptimizedPaperRectangularNcOracle,
};
use neo_reductions::optimized_engine::PiCcsProofVariant;
use neo_reductions::optimized_engine::{
    optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf,
    optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf, OptimizedStructureCache,
};
use neo_reductions::sumcheck::RoundOracle;
use neo_reductions::{PiCcsError, PiCcsProof};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Claim = CcsClaim<neo_ajtai::Commitment, F>;
type Output = CeClaim<neo_ajtai::Commitment, F, K>;

#[derive(Clone)]
struct ParallelStart {
    state: Arc<(Mutex<usize>, Condvar)>,
}

impl ParallelStart {
    fn new() -> Self {
        Self {
            state: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }

    fn meet(&self) -> Result<(), PiCcsError> {
        let (started, ready) = &*self.state;
        let mut started = started
            .lock()
            .map_err(|_| PiCcsError::ProtocolError("cross-check start probe lock was poisoned".into()))?;
        *started += 1;
        ready.notify_all();
        let (started, timeout) = ready
            .wait_timeout_while(started, Duration::from_secs(2), |started| *started < 2)
            .map_err(|_| PiCcsError::ProtocolError("cross-check start probe wait was poisoned".into()))?;
        if timeout.timed_out() && *started < 2 {
            return Err(PiCcsError::ProtocolError(
                "the second cross-check engine did not start concurrently".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ParallelProbeEngine {
    start: ParallelStart,
}

impl PiCcsEngine for ParallelProbeEngine {
    fn prove<L: SModuleHomomorphism<F, neo_ajtai::Commitment> + Sync>(
        &self,
        _transcript: &mut Poseidon2Transcript,
        _params: &NeoParams,
        _structure: &CcsStructure<F>,
        _fresh_claims: &[Claim],
        _fresh_witnesses: &[CcsWitness<F>],
        _running_claims: &[Output],
        _running_witnesses: &[Mat<F>],
        _commitment: &L,
    ) -> Result<(Vec<Output>, PiCcsProof), PiCcsError> {
        self.start.meet()?;
        Ok((Vec::new(), PiCcsProof::new(Vec::new(), None)))
    }

    fn verify(
        &self,
        _transcript: &mut Poseidon2Transcript,
        _params: &NeoParams,
        _structure: &CcsStructure<F>,
        _fresh_claims: &[Claim],
        _running_claims: &[Output],
        _outputs: &[Output],
        _proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError> {
        self.start.meet()?;
        Ok(true)
    }
}

fn rectangular_ccs(rows: usize, columns: usize) -> CcsStructure<F> {
    let mut matrix = Mat::zero(rows, columns, F::ZERO);
    for index in 0..rows.min(columns) {
        matrix[(index, index)] = F::ONE;
    }
    CcsStructure::new(
        vec![matrix.clone(), matrix],
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

fn nontrivial_rectangular_ccs(rows: usize, columns: usize) -> CcsStructure<F> {
    let mut first = Mat::zero(rows, columns, F::ZERO);
    let mut second = Mat::zero(rows, columns, F::ZERO);
    for column in 0..columns {
        first[(column % rows, column)] = F::from_u64((column % 5 + 1) as u64);
        second[((3 * column + 1) % rows, column)] = F::from_u64((column % 7 + 2) as u64);
        second[((5 * column + 2) % rows, column)] -= F::ONE;
    }
    CcsStructure::new(
        vec![first, second],
        SparsePoly::new(
            2,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![1, 1],
                },
                Term {
                    coeff: F::from_u64(2),
                    exps: vec![1, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0, 1],
                },
            ],
        ),
    )
    .expect("valid nontrivial rectangular CCS")
}

fn audit_point(variables: usize, offset: u64) -> Vec<K> {
    (0..variables)
        .map(|index| K::from(F::from_u64(offset + index as u64)))
        .collect()
}

fn assert_round_oracles_equal<P: RoundOracle, O: RoundOracle>(
    paper: &mut P,
    optimized: &mut O,
    rounds: usize,
    degree: usize,
) {
    assert_eq!(paper.num_rounds(), rounds);
    assert_eq!(optimized.num_rounds(), rounds);
    assert_eq!(paper.degree_bound(), degree);
    assert_eq!(optimized.degree_bound(), degree);
    let points: Vec<K> = (0..=degree)
        .map(|value| K::from(F::from_u64(value as u64)))
        .collect();
    for round in 0..rounds {
        assert_eq!(
            paper.evals_at(&points),
            optimized.evals_at(&points),
            "round polynomial differs at round {round}"
        );
        let challenge = K::from(F::from_u64(19 + 3 * round as u64));
        paper.fold(challenge);
        optimized.fold(challenge);
    }
}

fn committer(params: &NeoParams, columns: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5041_5045_5252_4543);
    let public_parameters = ajtai_setup(&mut rng, D, params.kappa as usize, columns.div_ceil(D)).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(public_parameters))
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
    let m_in = 2.min(columns);
    let claim = CcsClaim {
        adv: None,
        c: log.commit(&Z),
        x: values[..m_in].to_vec(),
        m_in,
    };
    let witness = CcsWitness {
        w: values[m_in..].to_vec(),
        Z,
    };
    (claim, witness)
}

#[allow(clippy::too_many_arguments)]
fn prove_with_mode(
    mode: FoldingMode,
    label: &'static [u8],
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    witnesses: &[CcsWitness<F>],
    running: &[Output],
    running_witnesses: &[Mat<F>],
    log: &AjtaiSModule,
) -> (Vec<Output>, PiCcsProof) {
    let mut transcript = Poseidon2Transcript::new(label);
    prove(
        mode,
        &mut transcript,
        params,
        structure,
        claims,
        witnesses,
        running,
        running_witnesses,
        log,
    )
    .expect("canonical rectangular proof")
}

fn seed_running_claim(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &Claim,
    witness: &CcsWitness<F>,
    log: &AjtaiSModule,
) -> Output {
    prove_with_mode(
        FoldingMode::PaperExact,
        b"paper-rectangular/seed",
        params,
        structure,
        std::slice::from_ref(claim),
        std::slice::from_ref(witness),
        &[],
        &[],
        log,
    )
    .0
    .remove(0)
}

fn assert_parity(rows: usize, columns: usize) {
    let structure = rectangular_ccs(rows, columns);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(rows.max(columns)).expect("parameters");
    let log = committer(&params, columns);
    let (claim_0, witness_0) = source(&log, columns, 1);
    let (claim_1, witness_1) = source(&log, columns, 2);
    let (claim_2, witness_2) = source(&log, columns, 7);
    let running = vec![seed_running_claim(&params, &structure, &claim_0, &witness_0, &log)];
    let running_witnesses = vec![witness_0.Z.clone()];
    let claims = vec![claim_1, claim_2];
    let witnesses = vec![witness_1, witness_2];
    let label = b"paper-rectangular/parity";

    let (paper_outputs, paper_proof) = prove_with_mode(
        FoldingMode::PaperExact,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    );
    let (optimized_outputs, optimized_proof) = prove_with_mode(
        FoldingMode::Optimized,
        label,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    );

    assert_eq!(paper_proof.variant, PiCcsProofVariant::PaperRectangularV1);
    assert_eq!(optimized_proof.variant, PiCcsProofVariant::PaperRectangularV1);
    assert_eq!(paper_outputs, optimized_outputs, "output claims differ");
    assert_eq!(
        paper_proof.canonical_bytes().expect("paper proof bytes"),
        optimized_proof
            .canonical_bytes()
            .expect("optimized proof bytes"),
        "proof bytes differ"
    );

    for mode in [FoldingMode::PaperExact, FoldingMode::Optimized] {
        let mut transcript = Poseidon2Transcript::new(label);
        assert!(
            verify(
                mode,
                &mut transcript,
                &params,
                &structure,
                &claims,
                &running,
                &optimized_outputs,
                &optimized_proof,
            )
            .expect("canonical verifier"),
            "cross-engine verification failed"
        );
    }
}

#[test]
fn paper_exact_and_optimized_are_byte_exact_for_both_rectangular_directions() {
    assert_parity(D / 2, D);
    assert_parity(2 * D, D);
}

#[test]
fn every_round_polynomial_and_fold_matches_on_nontrivial_invalid_witnesses() {
    for (rows, columns) in [(4, 8), (16, 8)] {
        let structure = nontrivial_rectangular_ccs(rows, columns);
        let params = NeoParams::goldilocks_auto_r1cs_ccs(rows.max(columns)).expect("parameters");
        let log = committer(&params, columns);
        let (_, fresh_0) = source(&log, columns, 3);
        let (_, fresh_1) = source(&log, columns, 8);
        let (_, running_0) = source(&log, columns, 11);
        let fresh = vec![fresh_0, fresh_1];
        let running = vec![running_0.Z];
        let dims =
            neo_reductions::engines::utils::build_dims_and_policy(&params, &structure).expect("rectangular dimensions");
        let beta_r = audit_point(dims.ell_n, 3);
        let beta_m = audit_point(dims.ell_m, 29);
        let prior = audit_point(dims.ell_n, 47);
        let challenges = Challenges::paper_rectangular(beta_r, beta_m, K::from(F::from_u64(13)));
        let cache = OptimizedStructureCache::build(&structure).expect("optimized cache");

        let mut paper_fe = PaperRectangularFeOracle::new(
            &structure,
            &fresh,
            &running,
            challenges.clone(),
            Some(&prior),
            dims.ell_n,
            dims.d_sc,
        )
        .expect("paper FE oracle");
        let mut optimized_fe = OptimizedPaperRectangularFeOracle::new(
            &structure,
            &fresh,
            &running,
            challenges.clone(),
            Some(&prior),
            dims.ell_n,
            dims.d_sc,
            &cache,
        )
        .expect("optimized FE oracle");
        assert_round_oracles_equal(&mut paper_fe, &mut optimized_fe, dims.ell_n, dims.d_sc);

        let mut paper_nc = PaperRectangularNcOracle::new(
            &structure,
            &params,
            &fresh,
            &running,
            challenges.clone(),
            dims.ell_m,
            dims.d_sc,
        )
        .expect("paper NC oracle");
        let mut optimized_nc = OptimizedPaperRectangularNcOracle::new(
            &structure, &params, &fresh, &running, challenges, dims.ell_m, dims.d_sc,
        )
        .expect("optimized NC oracle");
        assert_round_oracles_equal(&mut paper_nc, &mut optimized_nc, dims.ell_m, dims.d_sc);
    }
}

fn assert_public_crosscheck(rows: usize, columns: usize) {
    let structure = rectangular_ccs(rows, columns);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(rows.max(columns)).expect("parameters");
    let log = committer(&params, columns);
    let (running_source, running_witness) = source(&log, columns, 3);
    let running = vec![seed_running_claim(
        &params,
        &structure,
        &running_source,
        &running_witness,
        &log,
    )];
    let running_witnesses = vec![running_witness.Z];
    let (claim_0, witness_0) = source(&log, columns, 5);
    let (claim_1, witness_1) = source(&log, columns, 9);
    let claims = vec![claim_0, claim_1];
    let witnesses = vec![witness_0, witness_1];
    let mode = FoldingMode::OptimizedWithCrosscheck(CrosscheckCfg::default());
    let label = b"paper-rectangular/public-crosscheck";
    let mut prover_transcript = Poseidon2Transcript::new(label);
    let (outputs, proof) = prove(
        mode.clone(),
        &mut prover_transcript,
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    )
    .expect("public exact cross-check proof");
    assert_eq!(proof.variant, PiCcsProofVariant::PaperRectangularV1);

    let mut verifier_transcript = Poseidon2Transcript::new(label);
    assert!(verify(
        mode,
        &mut verifier_transcript,
        &params,
        &structure,
        &claims,
        &running,
        &outputs,
        &proof,
    )
    .expect("public exact cross-check verification"));
}

#[test]
fn public_crosscheck_mode_enforces_exact_reference_parity() {
    assert_public_crosscheck(D / 2, D);
    assert_public_crosscheck(2 * D, D);
}

#[test]
fn crosscheck_starts_both_engines_concurrently() {
    let structure = rectangular_ccs(2, D);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("parameters");
    let log = committer(&params, D);

    let prove_start = ParallelStart::new();
    let prove_engine = CrossCheckEngine {
        inner: ParallelProbeEngine {
            start: prove_start.clone(),
        },
        ref_oracle: ParallelProbeEngine { start: prove_start },
        cfg: CrosscheckCfg::default(),
    };
    let mut prover_transcript = Poseidon2Transcript::new(b"paper-rectangular/parallel-probe");
    let (_, proof) = prove_engine
        .prove(&mut prover_transcript, &params, &structure, &[], &[], &[], &[], &log)
        .expect("cross-check prover engines must overlap");

    let verify_start = ParallelStart::new();
    let verify_engine = CrossCheckEngine {
        inner: ParallelProbeEngine {
            start: verify_start.clone(),
        },
        ref_oracle: ParallelProbeEngine { start: verify_start },
        cfg: CrosscheckCfg::default(),
    };
    let mut verifier_transcript = Poseidon2Transcript::new(b"paper-rectangular/parallel-probe");
    assert!(verify_engine
        .verify(&mut verifier_transcript, &params, &structure, &[], &[], &[], &proof)
        .expect("cross-check verifier engines must overlap"));
}

fn proof_is_rejected(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[Claim],
    running: &[Output],
    outputs: &[Output],
    proof: &PiCcsProof,
) -> bool {
    let mut transcript = Poseidon2Transcript::new(b"paper-rectangular/parity");
    !matches!(
        verify(
            FoldingMode::Optimized,
            &mut transcript,
            params,
            structure,
            claims,
            running,
            outputs,
            proof,
        ),
        Ok(true)
    )
}

#[test]
fn canonical_verifier_rejects_independent_protocol_mutations() {
    let rows = D / 2;
    let columns = D;
    let structure = rectangular_ccs(rows, columns);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(columns).expect("parameters");
    let log = committer(&params, columns);
    let (claim_0, witness_0) = source(&log, columns, 1);
    let (claim_1, witness_1) = source(&log, columns, 2);
    let (claim_2, witness_2) = source(&log, columns, 4);
    let running = vec![seed_running_claim(&params, &structure, &claim_0, &witness_0, &log)];
    let running_witnesses = vec![witness_0.Z];
    let claims = vec![claim_1, claim_2];
    let witnesses = vec![witness_1, witness_2];
    let (outputs, proof) = prove_with_mode(
        FoldingMode::Optimized,
        b"paper-rectangular/parity",
        &params,
        &structure,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
        &log,
    );

    let mut changed_fe = proof.clone();
    changed_fe.sumcheck_rounds[0][0] += K::ONE;
    assert!(proof_is_rejected(
        &params,
        &structure,
        &claims,
        &running,
        &outputs,
        &changed_fe
    ));

    let mut changed_nc = proof.clone();
    changed_nc.sumcheck_rounds_nc[0][0] += K::ONE;
    assert!(proof_is_rejected(
        &params,
        &structure,
        &claims,
        &running,
        &outputs,
        &changed_nc
    ));

    let mut changed_gamma = proof.clone();
    changed_gamma.challenges_public.gamma += K::ONE;
    assert!(proof_is_rejected(
        &params,
        &structure,
        &claims,
        &running,
        &outputs,
        &changed_gamma,
    ));

    let mut changed_output = outputs.clone();
    changed_output[0].y_zcol[0] += K::ONE;
    assert!(proof_is_rejected(
        &params,
        &structure,
        &claims,
        &running,
        &changed_output,
        &proof,
    ));

    let mut changed_source_order = claims.clone();
    changed_source_order.reverse();
    assert!(proof_is_rejected(
        &params,
        &structure,
        &changed_source_order,
        &running,
        &outputs,
        &proof,
    ));
}

#[test]
fn square_joint_oracle_is_exactly_the_fe_nc_decomposition() {
    let structure = rectangular_ccs(D, D);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("parameters");
    let log = committer(&params, D);
    let (claim_0, witness_0) = source(&log, D, 1);
    let (claim_1, witness_1) = source(&log, D, 2);
    let running = vec![seed_running_claim(&params, &structure, &claim_0, &witness_0, &log)];
    let running_witnesses = vec![witness_0.Z.clone()];
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &structure).expect("dimensions");
    let beta: Vec<K> = (0..dims.ell_n)
        .map(|index| K::from(F::from_u64((index + 2) as u64)))
        .collect();
    let challenges = Challenges::paper_rectangular(beta.clone(), beta, K::from(F::from_u64(7)));

    let mut invalid = witness_1.clone();
    invalid.Z[(0, 0)] = F::from_u64(3);
    let invalid_fresh = vec![invalid];
    let prior = running[0].r.as_slice();
    let fe = PaperRectangularFeOracle::new(
        &structure,
        &invalid_fresh,
        &running_witnesses,
        challenges.clone(),
        Some(prior),
        dims.ell_n,
        dims.d_sc,
    )
    .expect("FE oracle");
    let nc = PaperRectangularNcOracle::new(
        &structure,
        &params,
        &invalid_fresh,
        &running_witnesses,
        challenges.clone(),
        dims.ell_m,
        dims.d_sc,
    )
    .expect("NC oracle");
    let joint = PaperJointSquareOracle::new(
        &structure,
        &params,
        &invalid_fresh,
        &running_witnesses,
        challenges.clone(),
        Some(prior),
        dims.ell_n,
        dims.d_sc,
    )
    .expect("joint oracle");
    let arbitrary_point: Vec<K> = (0..dims.ell_n)
        .map(|index| K::from(F::from_u64((11 + index) as u64)))
        .collect();
    assert_eq!(
        joint.evaluate(&arbitrary_point),
        fe.evaluate(&arbitrary_point) + nc.evaluate(&arbitrary_point)
    );
    for point_index in 0..(1usize << dims.ell_n) {
        let point: Vec<K> = (0..dims.ell_n)
            .map(|bit| if (point_index >> bit) & 1 == 1 { K::ONE } else { K::ZERO })
            .collect();
        assert_eq!(joint.evaluate(&point), fe.evaluate(&point) + nc.evaluate(&point));
    }

    let mut transcript = Poseidon2Transcript::new(b"paper-joint-square/executable");
    let (initial, phase) = neo_reductions::paper_exact_engine::paper_joint_square_prove_phase(
        &mut transcript,
        &params,
        &structure,
        std::slice::from_ref(&claim_1),
        std::slice::from_ref(&witness_1),
        &running,
        &running_witnesses,
        challenges,
    )
    .expect("executable one-joint paper SumCheck");
    assert_eq!(phase.rounds.len(), dims.ell_n);
    assert_eq!(
        neo_reductions::sumcheck::poly_eval_k(&phase.rounds[0], K::ZERO)
            + neo_reductions::sumcheck::poly_eval_k(&phase.rounds[0], K::ONE),
        initial
    );
}

#[test]
fn carried_gamma_slots_follow_the_absolute_paper_layout() {
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 0, 0), 6);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 1, 0, 0), 7);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 1, 0), 8);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 0, 0, 1), 12);
    assert_eq!(carried_gamma_exponent(2, 2, 3, 1, 2, D - 1), 11 + 6 * (D - 1));

    for coefficient in 0..D {
        for matrix in 0..3 {
            for running in 0..2 {
                assert_eq!(
                    carried_gamma_exponent(2, 2, 3, running, matrix, coefficient),
                    paper_carried_gamma_exponent(2, 2, 3, running, matrix, coefficient),
                    "optimized and literal-paper gamma layouts differ"
                );
            }
        }
    }
    assert_ne!(
        paper_carried_gamma_exponent(2, 2, 3, 0, 0, 0) + 1,
        carried_gamma_exponent(2, 2, 3, 0, 0, 0),
        "the gamma-offset mutation must be observable"
    );
}

#[test]
fn digest_and_running_handle_entrypoint_uses_canonical_rectangular_variant() {
    let structure = rectangular_ccs(D / 2, D);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("parameters");
    let log = committer(&params, D);
    let cache = OptimizedStructureCache::build(&structure).expect("cache");
    let (claim, witness) = source(&log, D, 2);
    let public_digest = [F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    let running_handle = [F::from_u64(5), F::from_u64(6), F::from_u64(7), F::from_u64(8)];
    let mut prover_transcript = Poseidon2Transcript::new(b"paper-rectangular/bound");
    let (outputs, proof, _, precompute) = optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf(
        &mut prover_transcript,
        &params,
        &structure,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&witness),
        &[],
        &[],
        public_digest,
        running_handle,
        &log,
        &cache,
    )
    .expect("bound canonical proof");
    assert_eq!(proof.variant, PiCcsProofVariant::PaperRectangularV1);
    assert_eq!(precompute.row_chals, proof.sumcheck_challenges);

    let mut verifier_transcript = Poseidon2Transcript::new(b"paper-rectangular/bound");
    let (valid, _) = optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf(
        &mut verifier_transcript,
        &params,
        &structure,
        std::slice::from_ref(&claim),
        &[],
        &outputs,
        &proof,
        &cache,
        public_digest,
        running_handle,
    )
    .expect("bound canonical verification");
    assert!(valid);
}

#[test]
fn paper_exact_active_sources_have_no_optimized_dependency() -> Result<(), PiCcsError> {
    let sources = [
        include_str!("../src/engines/paper_exact_engine/mod.rs"),
        include_str!("../src/engines/paper_exact_engine/prove.rs"),
        include_str!("../src/engines/paper_exact_engine/verify.rs"),
        include_str!("../src/engines/paper_exact_engine/paper_rectangular.rs"),
        include_str!("../src/engines/paper_exact_engine/rlc_dec.rs"),
    ];
    for forbidden in [
        "crate::optimized_engine",
        "engines::optimized_engine",
        "SuperneoEvalCache",
        "eval_all_mats_cached",
        "eval_all_mats_ring_cached",
        "binary_search",
        "use crate::engines::pi_ccs_protocol::{carried_gamma_exponent",
        "use crate::engines::pi_ccs_protocol::{gamma_power",
        "use crate::engines::pi_ccs_protocol::{fe_initial_claim",
        "use crate::engines::pi_ccs_protocol::fe_initial_claim",
        "use crate::engines::pi_ccs_protocol::fe_terminal",
        "use crate::engines::pi_ccs_protocol::nc_terminal",
    ] {
        if sources.iter().any(|source| source.contains(forbidden)) {
            return Err(PiCcsError::ProtocolError(format!(
                "PaperExact source contains forbidden dependency: {forbidden}"
            )));
        }
    }
    Ok(())
}

#[test]
fn canonical_optimized_surface_has_no_legacy_split_dependency() -> Result<(), PiCcsError> {
    let module = include_str!("../src/engines/optimized_engine/mod.rs");
    let canonical_surface = module
        .split("pub mod legacy_split_nc")
        .next()
        .ok_or_else(|| PiCcsError::ProtocolError("missing optimized module surface".into()))?;
    for forbidden in [
        "pub mod oracle",
        "pub mod prove",
        "pub mod verify",
        "pub use prove::optimized_prove_with_device_backends",
        "pub use replay_entrypoints::",
        "pub use oracle::OptimizedOracle",
        "pub use block_lane_entrypoints::",
    ] {
        if canonical_surface.contains(forbidden) {
            return Err(PiCcsError::ProtocolError(format!(
                "canonical optimized surface exposes legacy SplitNC item: {forbidden}"
            )));
        }
    }

    let canonical_evaluator = include_str!("../src/engines/optimized_engine/paper_rectangular.rs");
    for forbidden in ["BlockLaneNc", "OptimizedOracle", "oracle::NcOracle::new"] {
        if canonical_evaluator.contains(forbidden) {
            return Err(PiCcsError::ProtocolError(format!(
                "canonical optimized evaluator depends on legacy SplitNC: {forbidden}"
            )));
        }
    }

    let canonical_sources = [
        canonical_evaluator,
        include_str!("../src/engines/pi_ccs_rectangular.rs"),
        include_str!("../src/engines/pi_ccs_protocol.rs"),
    ];
    for forbidden in [
        "legacy_split_nc",
        "OptimizedOracle",
        "oracle::NcOracle::new",
        "q_at_point_paper_exact",
        "claimed_initial_sum_from_inputs_with_k_mcs",
        "rhs_terminal_identity_nc",
    ] {
        if canonical_sources
            .iter()
            .any(|source| source.contains(forbidden))
        {
            return Err(PiCcsError::ProtocolError(format!(
                "canonical optimized implementation depends on legacy SplitNC: {forbidden}"
            )));
        }
    }
    Ok(())
}
