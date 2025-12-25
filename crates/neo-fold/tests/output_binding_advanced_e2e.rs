//! Advanced end-to-end tests for output binding integration.
//!
//! These tests exercise the full output binding path through:
//! - `fold_shard_prove_with_output_binding`
//! - `fold_shard_verify_with_output_binding`
//!
//! Focus areas:
//! - Multi-step memory traces
//! - Tampered shard proofs with output binding
//! - Multi-address patterns at e2e level
//! - Stress tests with larger address spaces
//! - Mismatched configuration scenarios

#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::output_binding::{OutputBindingConfig, OB_INC_TOTAL_LABEL};
use neo_fold::shard::{fold_shard_prove_with_output_binding, fold_shard_verify_with_output_binding, CommitMixers};
use neo_fold::PiCcsError;
use neo_math::{D, F, K};
use neo_memory::encode::encode_mem_for_twist;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Test Infrastructure
// ============================================================================

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

/// Build a test fixture with the given memory trace configuration.
fn build_test_fixture(
    num_bits: usize,
    steps: usize,
    memory_ops: Vec<(bool, u64, u64)>, // (is_write, addr, value)
) -> (
    NeoParams,
    CcsStructure<F>,
    Vec<StepWitnessBundle<Cmt, F, K>>,
    Vec<F>,
) {
    let k = 1usize << num_bits;
    let n = 4usize; // Minimal CCS size
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;
    let l = DummyCommit::default();

    // Build memory layout and trace
    let mem_layout = PlainMemLayout { k, d: 1, n_side: k };
    let mem_init = MemInit::Zero;

    let actual_steps = memory_ops.len();
    assert!(actual_steps <= steps);

    let mut has_read = vec![F::ZERO; actual_steps];
    let mut has_write = vec![F::ZERO; actual_steps];
    let mut read_addr = vec![0u64; actual_steps];
    let mut write_addr = vec![0u64; actual_steps];
    let mut read_val = vec![F::ZERO; actual_steps];
    let mut write_val = vec![F::ZERO; actual_steps];
    let mut inc_at_write_addr = vec![F::ZERO; actual_steps];

    // Track state for increments
    let mut current_state = vec![F::ZERO; k];

    for (i, &(is_write, addr, value)) in memory_ops.iter().enumerate() {
        if is_write {
            has_write[i] = F::ONE;
            write_addr[i] = addr;
            write_val[i] = F::from_u64(value);
            let old_val = current_state.get(addr as usize).copied().unwrap_or(F::ZERO);
            inc_at_write_addr[i] = F::from_u64(value) - old_val;
            if (addr as usize) < k {
                current_state[addr as usize] = F::from_u64(value);
            }
        } else {
            has_read[i] = F::ONE;
            read_addr[i] = addr;
            read_val[i] = current_state.get(addr as usize).copied().unwrap_or(F::ZERO);
        }
    }

    let plain_mem = PlainMemTrace {
        steps: actual_steps,
        has_read,
        has_write,
        read_addr,
        write_addr,
        read_val,
        write_val,
        inc_at_write_addr,
    };

    // CPU witness (trivial)
    let z: Vec<F> = vec![F::ZERO; ccs.m];
    let Z = neo_memory::encode::ajtai_encode_vector(&params, &z);
    let c = l.commit(&Z);
    let mcs_inst = McsInstance { c, x: vec![], m_in: 0 };
    let mcs_wit = McsWitness { w: z, Z };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &plain_mem,
        &commit_fn,
        Some(ccs.m),
        mcs_inst.m_in,
    );

    let steps_witness: Vec<StepWitnessBundle<Cmt, F, K>> = vec![StepWitnessBundle {
        mcs: (mcs_inst, mcs_wit),
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData,
    }];

    (params, ccs, steps_witness, current_state)
}

// ============================================================================
// Multi-Step Memory Trace Tests
// ============================================================================

#[test]
fn test_multi_step_sequential_writes() -> Result<(), PiCcsError> {
    // Multiple writes to different addresses
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 0, 10),
        (true, 1, 20),
        (true, 2, 30),
        (true, 3, 40),
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 4, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Claim all outputs
    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new()
            .with_output(0, F::from_u64(10))
            .with_output(1, F::from_u64(20))
            .with_output(2, F::from_u64(30))
            .with_output(3, F::from_u64(40)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"multi_step_seq");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"multi_step_seq");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

#[test]
fn test_multi_step_same_address_rewrites() -> Result<(), PiCcsError> {
    // Multiple writes to the same address (cumulative)
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 1, 10),  // addr 1 = 10
        (true, 1, 25),  // addr 1 = 25
        (true, 1, 50),  // addr 1 = 50
        (true, 1, 100), // addr 1 = 100
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 4, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Claim final value only
    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(1, F::from_u64(100)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"same_addr_rewrite");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"same_addr_rewrite");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

// ============================================================================
// Tampered Shard Proof Tests
// ============================================================================

#[test]
fn test_tampered_output_proof_in_shard_fails() -> Result<(), PiCcsError> {
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 42)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 1, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(2, F::from_u64(42)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"tamper_shard_output");
    let mut proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    // Tamper with the output proof
    if let Some(ref mut output_proof) = proof.output_proof {
        if let Some(first_round) = output_proof.output_sc.round_polys.get_mut(0) {
            if let Some(coeff) = first_round.get_mut(0) {
                *coeff += K::ONE;
            }
        }
    }

    let mut tr_verify = Poseidon2Transcript::new(b"tamper_shard_output");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    );

    assert!(res.is_err(), "tampered output proof in shard must fail");
    Ok(())
}

#[test]
fn test_fixed_challenge_style_inc_total_forgery_is_rejected() -> Result<(), PiCcsError> {
    use neo_math::KExtensions;

    fn eval_poly(coeffs: &[K], x: K) -> K {
        coeffs.iter().rev().fold(K::ZERO, |acc, &c| acc * x + c)
    }

    fn forge_round_poly(current_claim: K, r: K, next_claim: K, degree_bound: usize) -> Vec<K> {
        let mut coeffs = vec![K::ZERO; degree_bound + 1];
        let two = K::from_u64(2);
        let denom = K::ONE - two * r;

        if denom != K::ZERO {
            // Degree-1: g(x) = a x + b
            // Constraints:
            //   g(0)+g(1) = a + 2b = current_claim
            //   g(r) = a r + b = next_claim
            let a = (current_claim - two * next_claim) * denom.inv();
            let b = next_claim - a * r;
            coeffs[0] = b;
            coeffs[1] = a;
            return coeffs;
        }

        // Fallback for r == 1/2: choose degree-2 with b=0.
        // g(x) = u x^2 + a x, with:
        //   g(0)+g(1) = u + a = current_claim
        //   g(r) = u r^2 + a r = next_claim
        // => u = (next_claim - current_claim*r) / (r^2 - r), a = current_claim - u
        let denom2 = r * r - r;
        assert_ne!(denom2, K::ZERO, "unexpected r where r^2-r=0");
        let u = (next_claim - current_claim * r) * denom2.inv();
        let a = current_claim - u;
        coeffs[1] = a;
        coeffs[2] = u;
        coeffs
    }

    // Use empty I/O so the output sumcheck final check is insensitive to `inc_total_claim`.
    // This isolates the inc_total sumcheck binding (the historical vulnerability).
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 10), (true, 3, 25)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 2, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(2, ProgramIO::new());

    let mut tr_prove = Poseidon2Transcript::new(b"fixed_challenge_style_forgery");
    let mut proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    // Sanity: honest proof verifies.
    let mut tr_verify_ok = Poseidon2Transcript::new(b"fixed_challenge_style_forgery");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify_ok,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    // Forge ONLY the inc_total claim/rounds as if challenges were fixed and independent.
    // Under the old (unsound) fixed-challenge design, this kind of forgery can keep the same
    // terminal value while changing the claimed sum arbitrarily.
    let last_step = proof.steps.last_mut().expect("non-empty");
    let inc_idx = last_step
        .batched_time
        .labels
        .len()
        .checked_sub(1)
        .expect("has claims");
    assert_eq!(
        last_step.batched_time.labels[inc_idx],
        OB_INC_TOTAL_LABEL,
        "inc_total claim must be last when output binding is enabled"
    );

    let ell_n = last_step.batched_time.round_polys[0].len();
    let r_time = last_step
        .fold
        .ccs_proof
        .sumcheck_challenges
        .get(..ell_n)
        .expect("r_time present");

    let original_claim = last_step.batched_time.claimed_sums[inc_idx];
    let mut desired_final = original_claim;
    for (coeffs, &r) in last_step.batched_time.round_polys[inc_idx].iter().zip(r_time.iter()) {
        desired_final = eval_poly(coeffs, r);
    }

    let forged_claim = original_claim + K::ONE;
    assert_ne!(forged_claim, original_claim);

    let degree_bound = last_step.batched_time.degree_bounds[inc_idx];
    let mut forged_rounds = Vec::with_capacity(ell_n);
    let mut current = forged_claim;
    for &r in r_time {
        let next = desired_final;
        let coeffs = forge_round_poly(current, r, next, degree_bound);
        forged_rounds.push(coeffs);
        current = next;
    }

    // Check the forged rounds do map forged_claim -> desired_final under the fixed challenges.
    let mut check_final = forged_claim;
    for (coeffs, &r) in forged_rounds.iter().zip(r_time.iter()) {
        check_final = eval_poly(coeffs, r);
    }
    assert_eq!(check_final, desired_final);

    last_step.batched_time.claimed_sums[inc_idx] = forged_claim;
    last_step.batched_time.round_polys[inc_idx] = forged_rounds;

    // In the new (sound) design, the inc_total claim participates in the batched-time transcript,
    // so this "fixed-challenge style" forgery must be rejected.
    let mut tr_verify_bad = Poseidon2Transcript::new(b"fixed_challenge_style_forgery");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify_bad,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    );
    assert!(res.is_err(), "forged inc_total claimed sum must fail verification");

    Ok(())
}

#[test]
fn test_wrong_output_claim_in_shard_fails() -> Result<(), PiCcsError> {
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 42)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 1, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg_prove = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(2, F::from_u64(42)),
    );
    let ob_cfg_verify = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(2, F::from_u64(99)), // WRONG!
    );

    let mut tr_prove = Poseidon2Transcript::new(b"wrong_claim_shard");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg_prove,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"wrong_claim_shard");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg_verify,
    );

    assert!(res.is_err(), "wrong output claim in verifier must fail");
    Ok(())
}

#[test]
fn test_wrong_address_claim_fails() -> Result<(), PiCcsError> {
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 42)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 1, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg_prove = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(2, F::from_u64(42)),
    );
    let ob_cfg_verify = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(1, F::from_u64(42)), // WRONG ADDRESS!
    );

    let mut tr_prove = Poseidon2Transcript::new(b"wrong_addr_shard");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg_prove,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"wrong_addr_shard");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg_verify,
    );

    assert!(res.is_err(), "wrong address claim in verifier must fail");
    Ok(())
}

// ============================================================================
// Multi-Address Pattern Tests
// ============================================================================

#[test]
fn test_sparse_multi_address_e2e() -> Result<(), PiCcsError> {
    // Sparse writes to addresses 0, 3 (leaving 1, 2 untouched)
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 0, 111),
        (true, 3, 222),
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 2, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new()
            .with_output(0, F::from_u64(111))
            .with_output(3, F::from_u64(222)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"sparse_e2e");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"sparse_e2e");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

// ============================================================================
// Missing Output Proof Tests
// ============================================================================

#[test]
fn test_missing_output_proof_fails() -> Result<(), PiCcsError> {
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 42)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 1, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new().with_output(2, F::from_u64(42)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"missing_output_proof");
    let mut proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    // Remove the output proof entirely
    proof.output_proof = None;

    let mut tr_verify = Poseidon2Transcript::new(b"missing_output_proof");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    );

    assert!(res.is_err(), "missing output proof must fail verification");
    Ok(())
}

// ============================================================================
// Partial Claims Tests (claim subset of written addresses)
// ============================================================================

/// EXPECTED BEHAVIOR: Partial claims succeed verification.
///
/// Output binding constrains only the claimed addresses in `ProgramIO`. It does not
/// require that all written addresses are claimed.
#[test]
fn test_claim_subset_of_writes_succeeds() -> Result<(), PiCcsError> {
    // Write to all 4 addresses but only claim 2
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 0, 10),
        (true, 1, 20),
        (true, 2, 30),
        (true, 3, 40),
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 4, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Only claim addresses 1 and 2 (not 0 and 3)
    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new()
            .with_output(1, F::from_u64(20))
            .with_output(2, F::from_u64(30)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"subset_claims");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"subset_claims");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

// ============================================================================
// Boundary Address Tests
// ============================================================================

#[test]
fn test_min_and_max_addresses() -> Result<(), PiCcsError> {
    // Write to min (0) and max (3) addresses for 2-bit space
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 0, 100),
        (true, 3, 200),
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 2, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(
        2,
        ProgramIO::new()
            .with_output(0, F::from_u64(100))
            .with_output(3, F::from_u64(200)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"boundary_addrs");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"boundary_addrs");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

// ============================================================================
// Larger Address Space Test
// ============================================================================

#[test]
fn test_larger_address_space_3_bits() -> Result<(), PiCcsError> {
    // 3-bit address space (8 addresses)
    let memory_ops: Vec<(bool, u64, u64)> = vec![
        (true, 0, 10),
        (true, 4, 20),
        (true, 7, 30),
    ];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(3, 3, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let ob_cfg = OutputBindingConfig::new(
        3,
        ProgramIO::new()
            .with_output(0, F::from_u64(10))
            .with_output(4, F::from_u64(20))
            .with_output(7, F::from_u64(30)),
    );

    let mut tr_prove = Poseidon2Transcript::new(b"larger_addr_3bit");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"larger_addr_3bit");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}

// ============================================================================
// Empty Claims Test
// ============================================================================

#[test]
fn test_empty_claims_e2e() -> Result<(), PiCcsError> {
    // Write to memory but claim nothing
    let memory_ops: Vec<(bool, u64, u64)> = vec![(true, 2, 42)];
    let (params, ccs, steps_witness, final_memory_state) = build_test_fixture(2, 1, memory_ops);
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Empty claims
    let ob_cfg = OutputBindingConfig::new(2, ProgramIO::new());

    let mut tr_prove = Poseidon2Transcript::new(b"empty_claims_e2e");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )?;

    let mut tr_verify = Poseidon2Transcript::new(b"empty_claims_e2e");
    let _ = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg,
    )?;

    Ok(())
}
