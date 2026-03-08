#![allow(non_snake_case)]
#![allow(unused_imports)]

use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::Commitment as Cmt;
use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::r1cs_to_ccs;
use neo_ccs::{CeClaim, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{me_from_z_balanced, Accumulator, FoldingSession, ProveInput};
use neo_fold::shard::StepLinkingConfig;
use neo_fold::shard::{BatchedTimeProof, MemSidecarProof, StepProof};
use neo_fold::shard::{FoldStep, ShardProof as FoldRun};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils as reductions_utils;
use neo_reductions::PiCcsProof;
use neo_spartan_bridge::circuit::fold_circuit::CircuitPolyTerm;
use neo_spartan_bridge::circuit::witness::PiCcsChallenges;
use neo_spartan_bridge::circuit::{FoldRunCircuit, FoldRunInstance, FoldRunWitness};
use neo_spartan_bridge::CircuitF;
use neo_spartan_bridge::{prove_fold_run, setup_fold_run, verify_fold_run};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn dummy_me_instance() -> CeClaim<Cmt, F, K> {
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

    CeClaim::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        s_col: vec![],
        y_ring: y,
        ct: y_scalars,
        aux_openings: vec![],
        y_zcol: vec![K::ZERO; D],
        m_in: 1,
        fold_digest: [0u8; 32],
    }
}

fn dummy_pi_ccs_proof_zero() -> PiCcsProof {
    PiCcsProof {
        variant: neo_reductions::optimized_engine::PiCcsProofVariant::SplitNcV1,
        sumcheck_rounds: Vec::new(),
        sc_initial_sum: Some(K::ZERO),
        sumcheck_challenges: Vec::new(),
        sumcheck_rounds_nc: Vec::new(),
        sc_initial_sum_nc: Some(K::ZERO),
        sumcheck_challenges_nc: Vec::new(),
        challenges_public: neo_reductions::Challenges {
            alpha: Vec::new(),
            beta_a: Vec::new(),
            beta_r: Vec::new(),
            beta_m: Vec::new(),
            gamma: K::ZERO,
        },
        sumcheck_final: K::ZERO,
        sumcheck_final_nc: K::ZERO,
        header_digest: Vec::new(),
        _extra: None,
    }
}

fn dummy_pi_ccs_challenges_zero() -> PiCcsChallenges {
    PiCcsChallenges {
        alpha: Vec::new(),
        beta_a: Vec::new(),
        beta_r: Vec::new(),
        beta_m: Vec::new(),
        gamma: K::ZERO,
        r_prime: Vec::new(),
        alpha_prime: Vec::new(),
        sumcheck_challenges: Vec::new(),
        s_col_prime: Vec::new(),
        alpha_prime_nc: Vec::new(),
        sumcheck_challenges_nc: Vec::new(),
    }
}

fn build_trivial_fold_run_and_instance() -> (FoldRunInstance, FoldRunWitness) {
    // Single ME output used throughout the step.
    let me_out = dummy_me_instance();

    // RLC: trivial identity mix – parent == child, ρ = I_D.
    let params = NeoParams::goldilocks_127();
    let rlc_rhos_step =
        neo_fold::pi_ccs::rot_rhos_from_mats(&params, &[Mat::identity(D)], "fold_run_circuit_smoke:rhos")
            .expect("typed identity rho");
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
        cpu_sumcheck: Default::default(),
        shift_sumcheck: Default::default(),
        time_cpu_commitments: Vec::new(),
        time_mem_commitments: Vec::new(),
        time_t: 0,
        time_declared_len: 0,
        time_col_ids: Vec::new(),
        memory_time_proofs: Vec::new(),
        openings: Vec::new(),
        opening_proofs: Vec::new(),
        opening_manifest: Default::default(),
        opening_reduction: Default::default(),
        opening_unification: Default::default(),
        joint_opening_lane: Default::default(),
        folding_lanes: Default::default(),
    };

    // FoldRun with one step (final outputs computed from steps).
    let run = FoldRun {
        steps: vec![StepProof {
            fold: step,
            mem: MemSidecarProof {
                val_me_claims: Vec::new(),
                wb_me_claims: Vec::new(),
                wp_me_claims: Vec::new(),
                poseidon_cycle_me_claims: Vec::new(),
                poseidon_local_me_claims: Vec::new(),
                shout_addr_pre: Default::default(),
                proofs: Vec::new(),
            },
            batched_time: BatchedTimeProof {
                claimed_sums: Vec::new(),
                degree_bounds: Vec::new(),
                labels: Vec::new(),
                round_polys: Vec::new(),
            },
            poseidon_local_time: None,
            poseidon_cycle_fold: Vec::new(),
            poseidon_local_fold: Vec::new(),
            val_fold: Vec::new(),
            wb_fold: Vec::new(),
            wp_fold: Vec::new(),
            compressed_substeps: None,
            stage8_fold: Vec::new(),
        }],
        output_proof: None,
        riscv_profile: None,
        riscv_memory_layout: None,
        segment_meta: None,
    };

    // Public instance: empty initial accumulator, final accumulator = DEC children,
    // and zero-valued Π-CCS challenges for the single step.
    let initial_accumulator: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let instance = FoldRunInstance {
        params_digest: [0u8; 32],
        ccs_digest: [0u8; 32],
        initial_accumulator: initial_accumulator.clone(),
        final_accumulator: run.compute_final_main_children(&initial_accumulator),
        pi_ccs_challenges: vec![dummy_pi_ccs_challenges_zero()],
    };

    // Witness: includes the FoldRun, the Π-CCS proof, and RLC ρ matrices.
    // Z-witnesses and DEC children Z are unused by the circuit today, so we keep them empty.
    let witness = FoldRunWitness {
        fold_run: run,
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
    if let Some(first_step) = witness.fold_run.steps.get_mut(0) {
        first_step.fold.ccs_proof.sc_initial_sum = Some(K::from(F::from_u64(5)));
        first_step.fold.ccs_proof.sumcheck_final = K::from(F::from_u64(5));
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

fn setup_ajtai_for_commit_cols(m_cols: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m_cols).expect("Ajtai setup should succeed");
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

    let commit_cols = n_vars.div_ceil(D);
    setup_ajtai_for_commit_cols(commit_cols);
    let l = AjtaiSModule::from_global_for_dims(D, commit_cols).expect("AjtaiSModule init");

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

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone())
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
    // Link x[2] across steps: it's constant (=1) in this fixture.
    session.set_step_linking(StepLinkingConfig::new(vec![(2, 2)]));
    let ok = session
        .verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "paper-exact verification should pass");

    // --- Build FoldRunWitness for the Spartan bridge (Π-CCS proofs + RLC data) ---
    let steps_len = run.steps.len();
    let rlc_rhos = run.steps.iter().map(|s| s.fold.rlc_rhos.clone()).collect();
    let witnesses: Vec<Vec<Mat<F>>> = vec![Vec::new(); steps_len];
    let dec_children_z: Vec<Vec<Mat<F>>> = vec![Vec::new(); steps_len];

    let witness = FoldRunWitness::from_fold_run(run.clone(), witnesses, rlc_rhos, dec_children_z);

    // Optional: directly synthesize the FoldRun circuit into a TestConstraintSystem
    // to inspect which constraint (if any) is unsatisfied before wrapping it in Spartan2.
    #[cfg(feature = "debug-logs")]
    {
        use neo_spartan_bridge::circuit::witness::PiCcsChallenges;

        // Rebuild Π-CCS challenges exactly as the bridge API does.
        let s_norm = ccs.ensure_identity_first().expect("identity-first");
        let dims = reductions_utils::build_dims_and_policy(&params, &s_norm).expect("dims");
        let ell_n = dims.ell_n;
        let ell_m = dims.ell_m;
        let ell = dims.ell;
        let ell_nc = dims.ell_nc;

        let mut pi_ccs_challenges = Vec::with_capacity(run.steps.len());
        for (step_idx, step) in run.steps.iter().enumerate() {
            let proof = &step.fold.ccs_proof;

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

            assert_eq!(
                proof.sumcheck_rounds_nc.len(),
                ell_nc,
                "FoldRun step {}: expected {} NC sumcheck rounds, got {}",
                step_idx,
                ell_nc,
                proof.sumcheck_rounds_nc.len()
            );
            assert_eq!(
                proof.sumcheck_challenges_nc.len(),
                ell_nc,
                "FoldRun step {}: expected {} NC sumcheck challenges, got {}",
                step_idx,
                ell_nc,
                proof.sumcheck_challenges_nc.len()
            );

            let alpha = proof.challenges_public.alpha.clone();
            let beta_a = proof.challenges_public.beta_a.clone();
            let beta_r = proof.challenges_public.beta_r.clone();
            let beta_m = proof.challenges_public.beta_m.clone();
            let gamma = proof.challenges_public.gamma;

            let sumcheck_challenges = proof.sumcheck_challenges.clone();
            let (r_prime_slice, alpha_prime_slice) = sumcheck_challenges.split_at(ell_n);

            let sumcheck_challenges_nc = proof.sumcheck_challenges_nc.clone();
            let (s_col_prime_slice, alpha_prime_nc_slice) = sumcheck_challenges_nc.split_at(ell_m);

            pi_ccs_challenges.push(PiCcsChallenges {
                alpha,
                beta_a,
                beta_r,
                beta_m,
                gamma,
                r_prime: r_prime_slice.to_vec(),
                alpha_prime: alpha_prime_slice.to_vec(),
                sumcheck_challenges,
                s_col_prime: s_col_prime_slice.to_vec(),
                alpha_prime_nc: alpha_prime_nc_slice.to_vec(),
                sumcheck_challenges_nc,
            });
        }

        let instance = FoldRunInstance {
            params_digest: [0u8; 32],
            ccs_digest: [0u8; 32],
            initial_accumulator: initial_accumulator.clone(),
            final_accumulator: run.compute_final_main_children(&initial_accumulator),
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
    let keypair = setup_fold_run(&params, &ccs, &initial_accumulator, &run, witness.clone())
        .expect("Spartan setup_fold_run should succeed");

    let spartan_proof = prove_fold_run(&keypair.pk, &params, &ccs, &initial_accumulator, &run, witness)
        .expect("Spartan prove_fold_run should succeed");

    let ok_spartan =
        verify_fold_run(&keypair.vk, &params, &ccs, &spartan_proof).expect("Spartan verify_fold_run should run");
    assert!(ok_spartan, "Spartan2 verification should accept the optimized FoldRun");
}
