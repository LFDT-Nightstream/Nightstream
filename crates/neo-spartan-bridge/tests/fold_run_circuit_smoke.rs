#![allow(non_snake_case)]
#![allow(unused_imports)]

use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::Commitment as Cmt;
use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::r1cs_to_ccs;
use neo_ccs::{Mat, MeInstance};
use neo_fold::folding::{FoldRun, FoldStep};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{me_from_z_balanced, Accumulator, FoldingSession, ProveInput};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils as reductions_utils;
use neo_reductions::PiCcsProof;
use neo_spartan_bridge::circuit::fold_circuit::CircuitPolyTerm;
use neo_spartan_bridge::circuit::witness::PiCcsChallenges;
use neo_spartan_bridge::circuit::{FoldRunCircuit, FoldRunInstance, FoldRunWitness};
use neo_spartan_bridge::CircuitF;
use neo_spartan_bridge::{prove_fold_run, verify_fold_run};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn dummy_me_instance() -> MeInstance<Cmt, F, K> {
    // Dummy Ajtai commitment with zeros
    let c = Cmt::zeros(D, 1);

    // X ∈ F^{D×m_in} with m_in = 1, all zeros
    let X = Mat::zero(D, 1, F::ZERO);

    // r ∈ K^{log n} – use empty vector so no constraints depend on it
    let r: Vec<K> = Vec::new();

    // y_j ∈ K^d for j=0..t-1. Take t = 1, each row length D, all zeros.
    let y_row = vec![K::ZERO; D];
    let y = vec![y_row];

    // y_scalars: one per j; unused by the circuit, but populate with zero.
    let y_scalars = vec![K::ZERO];

    MeInstance::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        y,
        y_scalars,
        m_in: 1,
        fold_digest: [0u8; 32],
    }
}

fn dummy_pi_ccs_proof_zero() -> PiCcsProof {
    PiCcsProof {
        sumcheck_rounds: Vec::new(),
        sc_initial_sum: Some(K::ZERO),
        sumcheck_challenges: Vec::new(),
        challenges_public: neo_reductions::Challenges {
            alpha: Vec::new(),
            beta_a: Vec::new(),
            beta_r: Vec::new(),
            gamma: K::ZERO,
        },
        sumcheck_final: K::ZERO,
        header_digest: Vec::new(),
        _extra: None,
    }
}

fn dummy_pi_ccs_challenges_zero() -> PiCcsChallenges {
    PiCcsChallenges {
        alpha: Vec::new(),
        beta_a: Vec::new(),
        beta_r: Vec::new(),
        gamma: K::ZERO,
        r_prime: Vec::new(),
        alpha_prime: Vec::new(),
        sumcheck_challenges: Vec::new(),
    }
}

fn build_trivial_fold_run_and_instance() -> (FoldRunInstance, FoldRunWitness) {
    // Single ME output used throughout the step.
    let me_out = dummy_me_instance();

    // RLC: trivial identity mix – parent == child, ρ = I_D.
    let rlc_rhos_step = vec![Mat::identity(D)];
    let rlc_parent = me_out.clone();

    // DEC: trivial k=1 decomposition – parent == only child.
    let dec_children = vec![rlc_parent.clone()];

    // Π-CCS proof: all zeros / empty sumcheck.
    let proof = dummy_pi_ccs_proof_zero();

    // Single fold step.
    let step = FoldStep {
        ccs_out: vec![me_out.clone()],
        ccs_proof: proof.clone(),
        rlc_rhos: rlc_rhos_step.clone(),
        rlc_parent: rlc_parent.clone(),
        dec_children: dec_children.clone(),
    };

    // FoldRun with one step and final outputs equal to DEC children.
    let run = FoldRun {
        steps: vec![step],
        final_outputs: dec_children.clone(),
    };

    // Public instance: empty initial accumulator, final accumulator = final_outputs,
    // and zero-valued Π-CCS challenges for the single step.
    let instance = FoldRunInstance {
        params_digest: [0u8; 32],
        ccs_digest: [0u8; 32],
        mcs_digest: [0u8; 32],
        initial_accumulator: Vec::new(),
        final_accumulator: run.final_outputs.clone(),
        pi_ccs_challenges: vec![dummy_pi_ccs_challenges_zero()],
    };

    // Witness: includes the FoldRun, the Π-CCS proof, and RLC ρ matrices.
    // Z-witnesses and DEC children Z are unused by the circuit today, so we keep them empty.
    let witness = FoldRunWitness {
        fold_run: run,
        pi_ccs_proofs: vec![proof],
        witnesses: vec![Vec::new()],
        rlc_rhos: vec![rlc_rhos_step],
        dec_children_z: vec![Vec::new()],
    };

    (instance, witness)
}

#[test]
fn fold_run_circuit_trivial_satisfied() {
    let (instance, witness) = build_trivial_fold_run_and_instance();

    // Delta for K multiplication (Goldilocks K uses δ = 7) and base parameter b=3.
    let delta = CircuitF::from(7u64);
    // Use b=1 in this trivial test so range products degenerate to a single factor.
    let base_b = 1u32;
    let poly_f = Vec::new(); // No CCS polynomial constraints in this trivial test.

    let circuit = FoldRunCircuit::new(instance, Some(witness), delta, base_b, poly_f);

    let mut cs = TestConstraintSystem::<CircuitF>::new();
    circuit
        .synthesize(&mut cs)
        .expect("circuit synthesis should succeed");
    assert!(
        cs.is_satisfied(),
        "trivial FoldRunCircuit instance should have a satisfied constraint system"
    );
}

#[test]
fn fold_run_circuit_initial_sum_mismatch_unsatisfied() {
    let (instance, mut witness) = build_trivial_fold_run_and_instance();

    // Tamper with sc_initial_sum to violate the T-binding constraint.
    if let Some(first_proof) = witness.pi_ccs_proofs.get_mut(0) {
        first_proof.sc_initial_sum = Some(K::from(F::from_u64(5)));
        first_proof.sumcheck_final = K::from(F::from_u64(5));
    }

    let delta = CircuitF::from(7u64);
    let base_b = 1u32;
    let poly_f = Vec::new();

    let circuit = FoldRunCircuit::new(instance, Some(witness), delta, base_b, poly_f);

    let mut cs = TestConstraintSystem::<CircuitF>::new();
    circuit
        .synthesize(&mut cs)
        .expect("circuit synthesis should succeed");
    assert!(
        !cs.is_satisfied(),
        "tampering sc_initial_sum should break the initial-sum binding constraint"
    );
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

/// Non-trivial regression test: build the three-step k=3 optimized
/// FoldRun from `session_multifold_r1cs_paper_exact_nontrivial` and
/// prove/verify it end-to-end with Spartan2. This exercises:
/// - non-empty sumcheck rounds
/// - non-trivial Π-CCS challenges
/// - multi-step RLC/DEC chaining
#[test]
fn fold_run_circuit_optimized_nontrivial_satisfied() {
    // --- Build the R1CS and CCS as in the neo-fold test ---
    let n_constraints = 5usize;
    let n_vars = 5usize;

    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);

    // (1) (x0 + x1) * (x2) = w0
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    B[(0, 2)] = F::ONE;
    C[(0, 3)] = F::ONE;

    // (2) (w0) * (x1) = w1
    A[(1, 3)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 4)] = F::ONE;

    // (3) x0 * x0 = x0
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;

    // (4) x1 * x1 = x1
    A[(3, 1)] = F::ONE;
    B[(3, 1)] = F::ONE;
    C[(3, 1)] = F::ONE;

    // (5) x2 * x2 = x2
    A[(4, 2)] = F::ONE;
    B[(4, 2)] = F::ONE;
    C[(4, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(A, B, C);

    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(n_constraints).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    let dims = reductions_utils::build_dims_and_policy(&params, &ccs.ensure_identity_first().expect("identity-first"))
        .expect("dims");
    let ell_n = dims.ell_n;

    let r: Vec<K> = vec![K::from(F::from_u64(5)); ell_n];
    let m_in = 3;

    // Seed 1: x=[1,1,1], w=[2,2]
    let z_seed_1: Vec<F> = vec![F::ONE, F::ONE, F::ONE, F::from_u64(2), F::from_u64(2)];
    // Seed 2: x=[1,0,1], w=[1,0]
    let z_seed_2: Vec<F> = vec![F::ONE, F::ZERO, F::ONE, F::ONE, F::ZERO];

    let (me1, Z1) = me_from_z_balanced(&params, &ccs, &l, &z_seed_1, &r, m_in).expect("seed1 ME ok");
    let (me2, Z2) = me_from_z_balanced(&params, &ccs, &l, &z_seed_2, &r, m_in).expect("seed2 ME ok");

    let acc = Accumulator {
        me: vec![me1, me2],
        witnesses: vec![Z1, Z2],
    };
    let initial_accumulator = acc.me.clone();

    let mut session = FoldingSession::new(FoldingMode::PaperExact, params, l.clone())
        .with_initial_accumulator(acc, &ccs)
        .expect("with_initial_accumulator");

    // Step 1: x = [1,1,1], w = [2,2]
    {
        let x: Vec<F> = vec![F::ONE, F::ONE, F::ONE];
        let w: Vec<F> = vec![F::from_u64(2), F::from_u64(2)];
        let input = ProveInput {
            ccs: &ccs,
            public_input: &x,
            witness: &w,
            output_claims: &[],
        };
        session.add_step_from_io(&input).expect("add_step 1");
    }

    // Step 2: x = [1,0,1], w = [1,0]
    {
        let x: Vec<F> = vec![F::ONE, F::ZERO, F::ONE];
        let w: Vec<F> = vec![F::ONE, F::ZERO];
        let input = ProveInput {
            ccs: &ccs,
            public_input: &x,
            witness: &w,
            output_claims: &[],
        };
        session.add_step_from_io(&input).expect("add_step 2");
    }

    // Step 3: x = [0,1,1], w = [1,1]
    {
        let x: Vec<F> = vec![F::ZERO, F::ONE, F::ONE];
        let w: Vec<F> = vec![F::ONE, F::ONE];
        let input = ProveInput {
            ccs: &ccs,
            public_input: &x,
            witness: &w,
            output_claims: &[],
        };
        session.add_step_from_io(&input).expect("add_step 3");
    }

    let run = session
        .fold_and_prove(&ccs)
        .expect("fold_and_prove should produce a FoldRun");

    // Sanity: the native paper-exact verifier should accept this run.
    let mcss_public = session.mcss_public();
    let ok = session
        .verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "paper-exact verification should pass");

    // --- Build FoldRunWitness for the Spartan bridge (Π-CCS proofs + RLC data) ---
    let steps_len = run.steps.len();
    let pi_ccs_proofs: Vec<PiCcsProof> = run.steps.iter().map(|s| s.ccs_proof.clone()).collect();
    let rlc_rhos: Vec<Vec<Mat<F>>> = run.steps.iter().map(|s| s.rlc_rhos.clone()).collect();
    let witnesses: Vec<Vec<Mat<F>>> = vec![Vec::new(); steps_len];
    let dec_children_z: Vec<Vec<Mat<F>>> = vec![Vec::new(); steps_len];

    let witness = FoldRunWitness::from_fold_run(run.clone(), pi_ccs_proofs, witnesses, rlc_rhos, dec_children_z);

    // Optional: directly synthesize the FoldRun circuit into a TestConstraintSystem
    // to inspect which constraint (if any) is unsatisfied before wrapping it in Spartan2.
    #[cfg(feature = "debug-logs")]
    {
        use neo_spartan_bridge::circuit::witness::PiCcsChallenges;

        // Rebuild Π-CCS challenges exactly as the bridge API does.
        let s_norm = ccs.ensure_identity_first().expect("identity-first");
        let dims = reductions_utils::build_dims_and_policy(&params, &s_norm).expect("dims");
        let ell_n = dims.ell_n;
        let ell = dims.ell;

        let mut pi_ccs_challenges = Vec::with_capacity(run.steps.len());
        for (step_idx, step) in run.steps.iter().enumerate() {
            let proof = &step.ccs_proof;

            assert_eq!(
                proof.sumcheck_rounds.len(),
                ell,
                "FoldRun step {}: expected {} sumcheck rounds, got {}",
                step_idx,
                ell,
                proof.sumcheck_rounds.len()
            );
            assert_eq!(
                proof.sumcheck_challenges.len(),
                ell,
                "FoldRun step {}: expected {} sumcheck challenges, got {}",
                step_idx,
                ell,
                proof.sumcheck_challenges.len()
            );

            let alpha = proof.challenges_public.alpha.clone();
            let beta_a = proof.challenges_public.beta_a.clone();
            let beta_r = proof.challenges_public.beta_r.clone();
            let gamma = proof.challenges_public.gamma;

            let sumcheck_challenges = proof.sumcheck_challenges.clone();
            let (r_prime_slice, alpha_prime_slice) = sumcheck_challenges.split_at(ell_n);

            pi_ccs_challenges.push(PiCcsChallenges {
                alpha,
                beta_a,
                beta_r,
                gamma,
                r_prime: r_prime_slice.to_vec(),
                alpha_prime: alpha_prime_slice.to_vec(),
                sumcheck_challenges,
            });
        }

        let instance = FoldRunInstance {
            params_digest: [0u8; 32],
            ccs_digest: [0u8; 32],
            mcs_digest: [0u8; 32],
            initial_accumulator: initial_accumulator.clone(),
            final_accumulator: run.final_outputs.clone(),
            pi_ccs_challenges,
        };

        // Rebuild CCS polynomial terms as in the bridge API.
        let poly_f: Vec<CircuitPolyTerm> = ccs
            .f
            .terms()
            .iter()
            .map(|term| {
                use p3_field::PrimeField64;
                let coeff_circ = CircuitF::from(term.coeff.as_canonical_u64());
                CircuitPolyTerm {
                    coeff: coeff_circ,
                    coeff_native: term.coeff,
                    exps: term.exps.iter().map(|e| *e as u32).collect(),
                }
            })
            .collect();

        let delta = CircuitF::from(7u64);
        let circuit = FoldRunCircuit::new(instance.clone(), Some(witness.clone()), delta, params.b, poly_f);

        let mut cs = TestConstraintSystem::<CircuitF>::new();
        circuit
            .synthesize(&mut cs)
            .expect("FoldRunCircuit synthesis for nontrivial run should succeed");
        if !cs.is_satisfied() {
            println!(
                "R1CS for optimized nontrivial FoldRun is unsatisfied; first failing constraint: {:?}",
                cs.which_is_unsatisfied()
            );
        }
    }

    // --- End-to-end Spartan2 proof and verification for this FoldRun ---
    let spartan_proof = prove_fold_run(&params, &ccs, &initial_accumulator, &run, witness)
        .expect("Spartan prove_fold_run should succeed");

    let ok_spartan = verify_fold_run(&params, &ccs, &spartan_proof).expect("Spartan verify_fold_run should run");
    assert!(ok_spartan, "Spartan2 verification should accept the optimized FoldRun");
}
