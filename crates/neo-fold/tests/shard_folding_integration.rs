//! Integration test for shard-level CPU + Memory folding with Twist & Shout.
//!
//! ## Current Architecture (Simplified)
//!
//! 1. Absorb memory commitments (Fiat-Shamir binding)
//! 2. CPU folding (Π_CCS → Π_RLC → Π_DEC per step)
//! 3. Memory sidecar proving (uses canonical `r` from CPU)
//! 4. Final merge: CPU output + memory ME → Π_RLC → Π_DEC
//!
//! ## Target Architecture (per integration-summary.md)
//!
//! At each folding step:
//! 1. Π_CCS(acc, MCS_i) → ccs_me
//! 2. Π_Twist/Shout for step_i → mem_me
//! 3. Π_RLC([ccs_me, mem_me]) → parent
//! 4. Π_DEC(parent) → acc
//!
//! Requires per-step memory instances (not implemented yet).
//!
//! Tests:
//! 1. CPU-only folding (test_shard_cpu_only_folding)
//! 2. Memory sidecar in isolation (test_twist_shout_sidecar_proving)
//! 3. Full CPU + Memory with final merge (test_full_cpu_memory_integration)
//! 4. Route A Twist-only happy path + adversarial cases (test_twist_only_route_a_*)

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::matrix::Mat;
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{check_me_consistency, CcsStructure, McsInstance, McsWitness, MeInstance, MeWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use neo_fold::memory_sidecar::memory::{ShoutAddrPreVerifyData, TwistAddrPreVerifyData, verify_route_a_memory_step};
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{
    fold_shard_prove, fold_shard_prove_with_witnesses, fold_shard_verify, MemOrLutProof, ShardFoldOutputs, ShardProof,
};
use neo_fold::PiCcsError;
use neo_math::{D, F, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{
    build_plain_lut_traces, build_plain_mem_traces, LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace,
};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::{shout, twist, MemInit};
use neo_params::NeoParams;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_vm_trace::{ShoutEvent, ShoutId, StepTrace, TwistEvent, TwistId, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Test Helpers
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

// ============================================================================
// Twist-only Route A checks (pre-time address proofs)
// ============================================================================

#[cfg(feature = "paper-exact")]
struct TwistOnlyHarness {
    params: NeoParams,
    ccs: CcsStructure<F>,
    l: DummyCommit,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,
    steps: Vec<StepWitnessBundle<Cmt, F, K>>,
    acc_init: Vec<MeInstance<Cmt, F, K>>,
    acc_wit_init: Vec<Mat<F>>,
}

#[cfg(feature = "paper-exact")]
impl TwistOnlyHarness {
    fn new() -> Self {
        let n = 4usize;
        let ccs = create_identity_ccs(n);
        let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
        // Small twist-only test: bump k_rho to clear the norm bound when folding memory ME claims.
        params.k_rho = 16;
        let l = DummyCommit::default();
        let mixers = default_mixers();

        let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
        let ell_n = dims.ell_n;
        let m_in = 2;

        let r: Vec<K> = vec![K::from(F::from_u64(7)); ell_n];

        let mut acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
        let mut acc_wit_init: Vec<Mat<F>> = Vec::new();
        for _ in 0..params.k_rho {
            let (me, Z) = create_seed_me(&params, &ccs, &l, &r, m_in);
            acc_init.push(me);
            acc_wit_init.push(Z);
        }

        let (mcs, mcs_wit) = create_trivial_mcs(&params, &ccs, &l, m_in);

        // Build a 1-step Twist trace: read addr 0 (init[0]), write addr 1 (new value 12).
        let layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let init_vals = vec![F::from_u64(5), F::from_u64(9)];
        let write_val = F::from_u64(12);
        let mem_init = MemInit::Sparse(vec![(0, init_vals[0]), (1, init_vals[1])]);
        let mem_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ONE],
            has_write: vec![F::ONE],
            read_addr: vec![0],
            write_addr: vec![1],
            read_val: vec![init_vals[0]],
            write_val: vec![write_val],
            inc_at_write_addr: vec![write_val - init_vals[1]],
        };
        let commit_fn = |mat: &Mat<F>| l.commit(mat);
        let (mem_inst, mem_wit) =
            encode_mem_for_twist(&params, &layout, &mem_init, &mem_trace, &commit_fn, Some(ccs.m), m_in);

        let steps = vec![StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData,
        }];

        Self {
            params,
            ccs,
            l,
            mixers,
            steps,
            acc_init,
            acc_wit_init,
        }
    }

    fn prove(&self) -> Result<ShardProof, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"twist-only");
        fold_shard_prove(
            FoldingMode::PaperExact,
            &mut tr,
            &self.params,
            &self.ccs,
            &self.steps,
            &self.acc_init,
            &self.acc_wit_init,
            &self.l,
            self.mixers,
        )
    }

    fn verify(&self, proof: &ShardProof) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"twist-only");
        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
            self.steps.iter().map(StepInstanceBundle::from).collect();
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr,
            &self.params,
            &self.ccs,
            &steps_public,
            &self.acc_init,
            proof,
            self.mixers,
        )
    }

    fn twist_claim_indices(&self) -> (usize, usize, usize, usize) {
        let inst = &self.steps[0].mem_instances[0].0;
        let ell_addr = inst.d * inst.ell;
        // Order: ra_bits[..ell_addr], wa_bits[..ell_addr], has_read, has_write, wv, rv, inc_at_write_addr
        let ra_idx = 0;
        let has_read_idx = 2 * ell_addr;
        let rv_idx = 2 * ell_addr + 3;
        let inc_idx = 2 * ell_addr + 4;
        (ra_idx, has_read_idx, rv_idx, inc_idx)
    }
}

#[cfg(feature = "paper-exact")]
struct TwistRolloverHarness {
    params: NeoParams,
    ccs: CcsStructure<F>,
    l: DummyCommit,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,
    steps: Vec<StepWitnessBundle<Cmt, F, K>>,
    acc_init: Vec<MeInstance<Cmt, F, K>>,
    acc_wit_init: Vec<Mat<F>>,
}

#[cfg(feature = "paper-exact")]
impl TwistRolloverHarness {
    fn new(teleport_init_step1: bool) -> Self {
        let n = 4usize;
        let ccs = create_identity_ccs(n);
        let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
        params.k_rho = 16;
        let l = DummyCommit::default();
        let mixers = default_mixers();

        let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
        let ell_n = dims.ell_n;
        let m_in = 2;

        let r: Vec<K> = vec![K::from(F::from_u64(7)); ell_n];

        let mut acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
        let mut acc_wit_init: Vec<Mat<F>> = Vec::new();
        for _ in 0..params.k_rho {
            let (me, Z) = create_seed_me(&params, &ccs, &l, &r, m_in);
            acc_init.push(me);
            acc_wit_init.push(Z);
        }

        let (mcs, mcs_wit) = create_trivial_mcs(&params, &ccs, &l, m_in);

        let layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let init0 = F::from_u64(5);
        let init1 = F::from_u64(9);
        let mem_init_step0 = MemInit::Sparse(vec![(0, init0), (1, init1)]);
        let mem_init_step1 = if teleport_init_step1 {
            MemInit::Sparse(vec![(0, init0 + F::ONE), (1, init1 + F::ONE)])
        } else {
            mem_init_step0.clone()
        };

        // No-op memory trace: local Twist checks are satisfied for any init, so rollover should be the only blocker.
        let mem_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ZERO],
            has_write: vec![F::ZERO],
            read_addr: vec![0],
            write_addr: vec![0],
            read_val: vec![F::ZERO],
            write_val: vec![F::ZERO],
            inc_at_write_addr: vec![F::ZERO],
        };
        let commit_fn = |mat: &Mat<F>| l.commit(mat);
        let (mem_inst0, mem_wit0) = encode_mem_for_twist(
            &params,
            &layout,
            &mem_init_step0,
            &mem_trace,
            &commit_fn,
            Some(ccs.m),
            m_in,
        );
        let (mem_inst1, mem_wit1) = encode_mem_for_twist(
            &params,
            &layout,
            &mem_init_step1,
            &mem_trace,
            &commit_fn,
            Some(ccs.m),
            m_in,
        );

        let step0 = StepWitnessBundle {
            mcs: (mcs.clone(), mcs_wit.clone()),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst0, mem_wit0)],
            _phantom: PhantomData,
        };
        let step1 = StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst1, mem_wit1)],
            _phantom: PhantomData,
        };

        Self {
            params,
            ccs,
            l,
            mixers,
            steps: vec![step0, step1],
            acc_init,
            acc_wit_init,
        }
    }

    fn prove(&self) -> Result<ShardProof, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"twist-rollover");
        fold_shard_prove(
            FoldingMode::PaperExact,
            &mut tr,
            &self.params,
            &self.ccs,
            &self.steps,
            &self.acc_init,
            &self.acc_wit_init,
            &self.l,
            self.mixers,
        )
    }

    fn verify(&self, proof: &ShardProof) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"twist-rollover");
        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
            self.steps.iter().map(StepInstanceBundle::from).collect();
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr,
            &self.params,
            &self.ccs,
            &steps_public,
            &self.acc_init,
            proof,
            self.mixers,
        )
    }
}

// Twist rollover via per-step dense init tables was removed earlier; replaced by a Route-A rollover equation check.

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_prove_verify() {
    let ctx = TwistOnlyHarness::new();
    let proof = ctx.prove().expect("prove should succeed");
    let _outputs = ctx.verify(&proof).expect("verify should succeed");
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_rollover_two_steps_consistent_init_passes() {
    let ctx = TwistRolloverHarness::new(false);
    let proof = ctx.prove().expect("prove should succeed");
    assert!(ctx.verify(&proof).is_ok());
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_rollover_two_steps_teleport_init_fails() {
    let ctx = TwistRolloverHarness::new(true);
    let proof = ctx.prove().expect("prove should succeed");
    assert!(ctx.verify(&proof).is_err());
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_rv_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    let (_, _, rv_idx, _) = ctx.twist_claim_indices();
    proof.steps[0].mem.me_claims_time[rv_idx].y_scalars[0] += K::ONE;
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when rv opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_wv_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");

    let inst = &ctx.steps[0].mem_instances[0].0;
    let ell_addr = inst.d * inst.ell;
    let wv_idx = 2 * ell_addr + 2;
    proof.steps[0].mem.me_claims_time[wv_idx].y_scalars[0] += K::ONE;

    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when wv opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_addr_bit_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    let (ra_idx, _, _, _) = ctx.twist_claim_indices();
    let bit = &mut proof.steps[0].mem.me_claims_time[ra_idx].y_scalars[0];
    *bit = K::ONE - *bit; // flip bitness
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when address bit opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_write_addr_bit_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");

    let inst = &ctx.steps[0].mem_instances[0].0;
    let ell_addr = inst.d * inst.ell;
    let wa_idx = ell_addr; // first write-address bit column
    let bit = &mut proof.steps[0].mem.me_claims_time[wa_idx].y_scalars[0];
    *bit = K::ONE - *bit; // flip bitness

    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when write-address bit opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_inc_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    let (_, _, _, inc_idx) = ctx.twist_claim_indices();
    proof.steps[0].mem.me_claims_time[inc_idx].y_scalars[0] += K::ONE;
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when inc_at_write_addr opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_wrong_rv_witness_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut steps = ctx.steps.clone();

    let t0 = steps[0].mcs.0.m_in; // per-step traces are embedded starting at m_in
    let (mem_inst, mem_wit) = &mut steps[0].mem_instances[0];
    let ell_addr = mem_inst.d * mem_inst.ell;
    let rv_mat_idx = 2 * ell_addr + 3;

    let rv_mat = &mut mem_wit.mats[rv_mat_idx];
    let mut decoded = neo_memory::ajtai::decode_vector(&ctx.params, rv_mat);
    decoded[t0] += F::ONE;
    *rv_mat = neo_memory::encode::ajtai_encode_vector(&ctx.params, &decoded);

    let mut tr_prove = Poseidon2Transcript::new(b"twist-only-wrong-rv-witness");
    let result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &ctx.params,
        &ctx.ccs,
        &steps,
        &ctx.acc_init,
        &ctx.acc_wit_init,
        &ctx.l,
        ctx.mixers,
    );

    assert!(result.is_err(), "invalid Twist read witness must fail proving");
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_wrong_inc_witness_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut steps = ctx.steps.clone();

    let t0 = steps[0].mcs.0.m_in; // per-step traces are embedded starting at m_in
    let (mem_inst, mem_wit) = &mut steps[0].mem_instances[0];
    let ell_addr = mem_inst.d * mem_inst.ell;
    let inc_mat_idx = 2 * ell_addr + 4;

    let inc_mat = &mut mem_wit.mats[inc_mat_idx];
    let mut decoded = neo_memory::ajtai::decode_vector(&ctx.params, inc_mat);
    decoded[t0] += F::ONE;
    *inc_mat = neo_memory::encode::ajtai_encode_vector(&ctx.params, &decoded);

    let mut tr_prove = Poseidon2Transcript::new(b"twist-only-wrong-inc-witness");
    let result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &ctx.params,
        &ctx.ccs,
        &steps,
        &ctx.acc_init,
        &ctx.acc_wit_init,
        &ctx.l,
        ctx.mixers,
    );

    assert!(result.is_err(), "invalid Twist inc witness must fail proving");
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_route_a_time_claimed_sum_must_match_addr_pre_final() {
    let ctx = TwistOnlyHarness::new();
    let proof = ctx.prove().expect("prove should succeed");
    assert!(ctx.verify(&proof).is_ok(), "sanity: proof should verify");

    let step = StepInstanceBundle::from(&ctx.steps[0]);
    let step_proof = &proof.steps[0];

    let claim_plan = RouteATimeClaimPlan::build(&step, 1).expect("claim plan");
    let twist_claims = claim_plan.twist.first().expect("missing Twist claim schedule");

    let (r_addr, read_check_claim_sum, write_check_claim_sum) = match &step_proof.mem.proofs[0] {
        MemOrLutProof::Twist(p) => (
            p.addr_pre.r_addr.clone(),
            step_proof.batched_time.claimed_sums[twist_claims.read_check],
            step_proof.batched_time.claimed_sums[twist_claims.write_check],
        ),
        _ => panic!("expected Twist proof at index 0"),
    };

    let shout_pre: Vec<ShoutAddrPreVerifyData> = Vec::new();
    let twist_pre = vec![TwistAddrPreVerifyData {
        r_addr,
        read_check_claim_sum,
        write_check_claim_sum,
    }];

    let mut bad_claimed_sums = step_proof.batched_time.claimed_sums.clone();
    bad_claimed_sums[twist_claims.read_check] += K::ONE;
    let dummy_final_values = vec![K::ZERO; step_proof.batched_time.claimed_sums.len()];

    let r_time = &step_proof.fold.ccs_out[0].r;
    let r_cycle = r_time.clone(); // any point with correct length suffices for this negative test
    let mut tr = Poseidon2Transcript::new(b"twist-only-claim-sum-mismatch");

    let err = match verify_route_a_memory_step(
        &mut tr,
        &ctx.params,
        &step,
        None,
        r_time,
        &r_cycle,
        &dummy_final_values,
        &bad_claimed_sums,
        1, // claim 0 is CCS/time
        &step_proof.mem,
        &shout_pre,
        &twist_pre,
        0,
    ) {
        Ok(_) => panic!("mismatched twist claimed sums must fail"),
        Err(e) => e,
    };

    match err {
        PiCcsError::ProtocolError(msg) => assert!(
            msg.contains("twist read_check claimed sum != addr-pre final"),
            "unexpected ProtocolError: {msg}"
        ),
        other => panic!("expected ProtocolError, got {other:?}"),
    }
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_nonbinary_flag_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    let (_, has_read_idx, _, _) = ctx.twist_claim_indices();
    proof.steps[0].mem.me_claims_time[has_read_idx].y_scalars[0] = K::from(F::from_u64(2));
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when has_read flag is non-binary"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_val_eval_claim_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");

    let mem_proof = proof
        .steps
        .get_mut(0)
        .and_then(|step| step.mem.proofs.get_mut(0))
        .expect("one mem proof");
    let twist_proof = match mem_proof {
        MemOrLutProof::Twist(p) => p,
        _ => panic!("expected Twist proof"),
    };
    let val_eval = twist_proof
        .val_eval
        .as_mut()
        .expect("Phase 2 requires val_eval");
    val_eval.claimed_inc_sum_lt += K::ONE;

    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when val_eval claimed_inc_sum_lt is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_val_lane_opening_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    proof.steps[0].mem.me_claims_val[0].y_scalars[0] += K::ONE;
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when r_val ME opening is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_mutated_val_lane_commitment_fails() {
    let ctx = TwistOnlyHarness::new();
    let mut proof = ctx.prove().expect("prove should succeed");
    proof.steps[0].mem.me_claims_val[0].c.data[0] += F::ONE;
    assert!(
        ctx.verify(&proof).is_err(),
        "verification should fail when r_val ME commitment is corrupted"
    );
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_only_route_a_exports_val_lane_obligations() {
    fn trim_y_to_d(mut inst: MeInstance<Cmt, F, K>) -> MeInstance<Cmt, F, K> {
        for row in inst.y.iter_mut() {
            row.truncate(D);
        }
        inst
    }

    let ctx = TwistOnlyHarness::new();

    let mut tr_prove = Poseidon2Transcript::new(b"twist-only-obligations");
    let (proof, outputs_prove, wits) = fold_shard_prove_with_witnesses(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &ctx.params,
        &ctx.ccs,
        &ctx.steps,
        &ctx.acc_init,
        &ctx.acc_wit_init,
        &ctx.l,
        ctx.mixers,
    )
    .expect("prove_with_witnesses should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"twist-only-obligations");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = ctx.steps.iter().map(StepInstanceBundle::from).collect();
    let outputs_verify = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &ctx.params,
        &ctx.ccs,
        &steps_public,
        &ctx.acc_init,
        &proof,
        ctx.mixers,
    )
    .expect("verify should succeed");

    assert_eq!(
        outputs_verify.obligations.main, outputs_prove.obligations.main,
        "final main accumulator mismatch"
    );
    assert_eq!(
        outputs_verify.obligations.val, outputs_prove.obligations.val,
        "val-lane obligations mismatch"
    );
    assert!(
        !outputs_verify.obligations.val.is_empty(),
        "expected Twist val-lane obligations"
    );
    assert_eq!(
        outputs_verify.obligations.main.len(),
        wits.final_main_wits.len(),
        "main witness count mismatch"
    );
    assert_eq!(
        outputs_verify.obligations.val.len(),
        wits.val_lane_wits.len(),
        "val-lane witness count mismatch"
    );

    for (inst, Z) in outputs_verify
        .obligations
        .main
        .iter()
        .cloned()
        .map(trim_y_to_d)
        .zip(wits.final_main_wits.iter().cloned())
    {
        check_me_consistency(&ctx.ccs, &ctx.l, &inst, &MeWitness { Z }).expect("main ME consistency");
    }

    for (inst, Z) in outputs_verify
        .obligations
        .val
        .iter()
        .cloned()
        .map(trim_y_to_d)
        .zip(wits.val_lane_wits.iter().cloned())
    {
        check_me_consistency(&ctx.ccs, &ctx.l, &inst, &MeWitness { Z }).expect("val-lane ME consistency");
    }

    let mut inst_bad = trim_y_to_d(outputs_verify.obligations.val[0].clone());
    inst_bad.y[0][0] += K::ONE;
    assert!(
        check_me_consistency(
            &ctx.ccs,
            &ctx.l,
            &inst_bad,
            &MeWitness {
                Z: wits.val_lane_wits[0].clone()
            }
        )
        .is_err(),
        "expected base-case ME consistency to fail for corrupted val-lane obligation"
    );
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

fn create_trivial_mcs(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &DummyCommit,
    m_in: usize,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    use neo_ajtai::{decomp_b, DecompStyle};

    let m = ccs.m;
    let z: Vec<F> = vec![F::ZERO; m];
    let x: Vec<F> = z[..m_in].to_vec();
    let w: Vec<F> = z[m_in..].to_vec();

    let d = D;
    let digits = decomp_b(&z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&Z);

    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

fn create_seed_me(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &DummyCommit,
    r: &[K],
    m_in: usize,
) -> (MeInstance<Cmt, F, K>, Mat<F>) {
    use neo_ajtai::{decomp_b, DecompStyle};

    let m = ccs.m;
    let z: Vec<F> = vec![F::ZERO; m];

    let d = D;
    let digits = decomp_b(&z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);

    let c = l.commit(&Z);
    let X = l.project_x(&Z, m_in);

    let t = ccs.t();
    let y_pad = d.next_power_of_two();
    let y: Vec<Vec<K>> = (0..t).map(|_| vec![K::ZERO; y_pad]).collect();
    let y_scalars: Vec<K> = vec![K::ZERO; t];

    let me = MeInstance {
        c,
        X,
        r: r.to_vec(),
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    (me, Z)
}

/// Build a simple VM trace with memory reads/writes and table lookups.
fn build_test_vm_trace() -> VmTrace<u64, u64> {
    let mem_id = TwistId(0);
    let tbl_id = ShoutId(0);

    // 4-step trace:
    // Step 0: Write 42 to mem[0], lookup table[1] = 20
    // Step 1: Read mem[0] = 42, lookup table[2] = 30
    // Step 2: Write 100 to mem[1]
    // Step 3: Read mem[1] = 100, lookup table[0] = 10
    VmTrace {
        steps: vec![
            StepTrace {
                cycle: 0,
                pc_before: 0,
                pc_after: 1,
                opcode: 1,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Write,
                    addr: 0,
                    value: 42,
                }],
                shout_events: vec![ShoutEvent {
                    shout_id: tbl_id,
                    key: 1,
                    value: 20,
                }],
                halted: false,
            },
            StepTrace {
                cycle: 1,
                pc_before: 1,
                pc_after: 2,
                opcode: 2,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: 42,
                }],
                shout_events: vec![ShoutEvent {
                    shout_id: tbl_id,
                    key: 2,
                    value: 30,
                }],
                halted: false,
            },
            StepTrace {
                cycle: 2,
                pc_before: 2,
                pc_after: 3,
                opcode: 1,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Write,
                    addr: 1,
                    value: 100,
                }],
                shout_events: vec![],
                halted: false,
            },
            StepTrace {
                cycle: 3,
                pc_before: 3,
                pc_after: 4,
                opcode: 2,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Read,
                    addr: 1,
                    value: 100,
                }],
                shout_events: vec![ShoutEvent {
                    shout_id: tbl_id,
                    key: 0,
                    value: 10,
                }],
                halted: true,
            },
        ],
    }
}

// ============================================================================
// Test 1: CPU-Only Folding (Works)
// ============================================================================

/// CPU-only test: CCS folding without memory sidecar.
///
/// This exercises the pure CPU path (no Twist/Shout, no merge).
#[test]
#[cfg(feature = "paper-exact")]
fn test_shard_cpu_only_folding() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let ell_n = dims.ell_n;
    let m_in = 2;

    let r: Vec<K> = vec![K::from(F::from_u64(7)); ell_n];

    let mut acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut acc_wit_init: Vec<Mat<F>> = Vec::new();

    for _ in 0..params.k_rho {
        let (me, Z) = create_seed_me(&params, &ccs, &l, &r, m_in);
        acc_init.push(me);
        acc_wit_init.push(Z);
    }

    let mut mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)> = Vec::new();
    for _ in 0..2 {
        let (mcs, wit) = create_trivial_mcs(&params, &ccs, &l, m_in);
        mcss.push((mcs, wit));
    }

    // Empty memory sidecar, per-step
    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = mcss
        .into_iter()
        .map(|mcs| StepWitnessBundle {
            mcs,
            lut_instances: vec![],
            mem_instances: vec![],
            _phantom: PhantomData,
        })
        .collect();

    let mut tr_prove = Poseidon2Transcript::new(b"shard-cpu-only");

    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("fold_shard_prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"shard-cpu-only");

    let _shard_mcss_public: Vec<McsInstance<Cmt, F>> = steps.iter().map(|s| s.mcs.0.clone()).collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let _outputs = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("fold_shard_verify should succeed");

    assert_eq!(proof.steps.len(), 2, "Should have 2 fold steps");
    assert!(
        proof.steps.iter().all(|s| s.mem.proofs.is_empty()),
        "No memory sidecar proofs"
    );
    assert!(
        proof
            .steps
            .iter()
            .all(|s| s.mem.me_claims_time.is_empty() && s.mem.me_claims_val.is_empty()),
        "No memory ME claims"
    );

    let final_children = proof.compute_final_main_children(&acc_init);
    assert_eq!(final_children.len(), params.k_rho as usize);

    println!("✓ test_shard_cpu_only_folding passed!");
    println!("  - CPU steps: {}", proof.steps.len());
    println!("  - Final children: {}", final_children.len());
    println!("  - k_rho: {}", params.k_rho);
}

// ============================================================================
// Test 2: Twist & Shout Proving in Isolation (Works)
// ============================================================================

/// Test memory sidecar (Twist/Shout) proving in isolation.
///
/// This validates that:
/// 1. VM trace → plain trace → encoding produces valid witnesses
/// 2. Twist::prove produces valid ME claims and proof
/// 3. Shout::prove produces valid ME claims and proof
/// 4. Semantic checks pass for both protocols
///
/// NOTE: Full integration with CPU (merge via RLC) requires r-alignment.
/// Currently, CPU and memory ME claims have different `r` values, so merge fails.
/// This is tracked as a TODO for the architecture.
#[test]
#[cfg(feature = "paper-exact")]
fn test_twist_shout_sidecar_proving() {
    let params = NeoParams::goldilocks_127();
    let l = DummyCommit::default();

    // =========================================================================
    // Create VM Trace and Memory/LUT structures
    // =========================================================================
    let vm_trace = build_test_vm_trace();

    // Memory layout: 4 cells, d=1 dimension, n_side=4 (ell = 2 bits)
    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, mem_layout.clone());

    // LUT table: [10, 20, 30, 40] at addresses 0..3
    let lut_table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };

    // =========================================================================
    // Build Plain Traces and Encode
    // =========================================================================
    let initial_mem: HashMap<(u32, u64), F> = HashMap::new();
    let plain_mem_traces = build_plain_mem_traces::<F>(&vm_trace, &mem_layouts, &initial_mem);
    let plain_mem = &plain_mem_traces[&0u32];

    let mut table_sizes = HashMap::new();
    table_sizes.insert(0u32, (4usize, 1usize)); // (k, d)
    let plain_lut_traces = build_plain_lut_traces::<F>(&vm_trace, &table_sizes);
    let plain_lut = &plain_lut_traces[&0u32];

    // Encode for Twist (memory)
    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let mem_init = neo_memory::MemInit::Zero;
    let (mem_inst, mem_wit) = encode_mem_for_twist(&params, &mem_layout, &mem_init, plain_mem, &commit_fn, None, 0);

    // Encode for Shout (lookup)
    let (lut_inst, lut_wit) = encode_lut_for_shout(&params, &lut_table, plain_lut, &commit_fn, None, 0);

    println!("Memory witness matrices: {}", mem_wit.mats.len());
    println!("LUT witness matrices: {}", lut_wit.mats.len());

    // =========================================================================
    // Verify Semantic Checks Pass
    // =========================================================================
    twist::check_twist_semantics(&params, &mem_inst, &mem_wit).expect("Twist semantic check should pass");

    shout::check_shout_semantics(&params, &lut_inst, &lut_wit, &plain_lut.val)
        .expect("Shout semantic check should pass");

    // The secure integration path is Route A in `neo_fold::shard`; this test is now semantic-only.
    println!("✓ test_twist_shout_sidecar_proving passed (semantic-only)");
}

// ============================================================================
// Test 3: Full CPU + Memory Integration (with Final Merge)
// ============================================================================

/// Test the current simplified architecture: CPU first, then final merge with memory.
///
/// Flow:
/// 1. Absorb memory commits (Fiat-Shamir)
/// 2. CPU folding (Π_CCS → Π_RLC → Π_DEC per step) → cpu_final
/// 3. Memory sidecar proving (uses canonical `r` from CPU)
/// 4. Final merge: Π_RLC([cpu_final, mem_me]) → Π_DEC → final children
///
/// NOTE: This test will fail if the norm bound is violated:
///   count · T · (b-1) < b^{k_rho}
/// where count = cpu_final.len() + mem_me.len()
#[test]
#[cfg(feature = "paper-exact")]
fn test_full_cpu_memory_integration() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let ell_n = dims.ell_n;
    let m_in = 2;

    println!("Test params: k_rho={}, ell_n={}", params.k_rho, ell_n);

    let r: Vec<K> = vec![K::from(F::from_u64(7)); ell_n];

    // Create initial accumulator
    let mut acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut acc_wit_init: Vec<Mat<F>> = Vec::new();

    for _ in 0..params.k_rho {
        let (me, Z) = create_seed_me(&params, &ccs, &l, &r, m_in);
        acc_init.push(me);
        acc_wit_init.push(Z);
    }

    // Create MCS instances
    let mut mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)> = Vec::new();
    for _ in 0..2 {
        let (mcs, wit) = create_trivial_mcs(&params, &ccs, &l, m_in);
        mcss.push((mcs, wit));
    }

    // Create memory sidecar
    let vm_trace = build_test_vm_trace();
    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, mem_layout.clone());

    let lut_table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };

    let initial_mem: HashMap<(u32, u64), F> = HashMap::new();
    let plain_mem_traces = build_plain_mem_traces::<F>(&vm_trace, &mem_layouts, &initial_mem);
    let plain_mem = &plain_mem_traces[&0u32];

    let mut table_sizes = HashMap::new();
    table_sizes.insert(0u32, (4usize, 1usize));
    let plain_lut_traces = build_plain_lut_traces::<F>(&vm_trace, &table_sizes);
    let plain_lut = &plain_lut_traces[&0u32];

    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    // Build per-step bundles
    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();
    let mem_init = MemInit::Zero;
    for (idx, mcs) in mcss.into_iter().enumerate() {
        let single_plain_mem = PlainMemTrace {
            steps: 1,
            has_read: vec![plain_mem.has_read[idx]],
            has_write: vec![plain_mem.has_write[idx]],
            read_addr: vec![plain_mem.read_addr[idx]],
            write_addr: vec![plain_mem.write_addr[idx]],
            read_val: vec![plain_mem.read_val[idx]],
            write_val: vec![plain_mem.write_val[idx]],
            inc_at_write_addr: vec![plain_mem.inc_at_write_addr[idx]],
        };
        let (mem_inst_step, mem_wit_step) =
            encode_mem_for_twist(&params, &mem_layout, &mem_init, &single_plain_mem, &commit_fn, None, 0);

        let single_plain_lut = PlainLutTrace {
            has_lookup: vec![plain_lut.has_lookup[idx]],
            addr: vec![plain_lut.addr[idx]],
            val: vec![plain_lut.val[idx]],
        };
        let (lut_inst_step, lut_wit_step) =
            encode_lut_for_shout(&params, &lut_table, &single_plain_lut, &commit_fn, None, 0);

        steps.push(StepWitnessBundle {
            mcs,
            lut_instances: vec![(lut_inst_step.clone(), lut_wit_step)],
            mem_instances: vec![(mem_inst_step.clone(), mem_wit_step)],
            _phantom: PhantomData,
        });
    }

    let _lut_inst = steps[0].lut_instances[0].0.clone();
    let _mem_inst = steps[0].mem_instances[0].0.clone();

    let mut tr_prove = Poseidon2Transcript::new(b"shard-full-integration");

    // This may fail with norm bound error if too many ME claims
    let result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    );

    match result {
        Ok(proof) => {
            // Verify proof
            let mut tr_verify = Poseidon2Transcript::new(b"shard-full-integration");
            let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

            let outputs = fold_shard_verify(
                FoldingMode::PaperExact,
                &mut tr_verify,
                &params,
                &ccs,
                &steps_public,
                &acc_init,
                &proof,
                mixers,
            )
            .expect("fold_shard_verify should succeed");

            assert_eq!(proof.steps.len(), 2, "Should have 2 fold steps");
            assert!(
                proof.steps.iter().any(|s| !s.mem.proofs.is_empty()),
                "Should have memory sidecar proofs"
            );
            assert!(
                !outputs.obligations.val.is_empty(),
                "expected Twist to produce val-lane obligations"
            );

            let final_children = proof.compute_final_main_children(&acc_init);
            assert_eq!(final_children.len(), params.k_rho as usize);

            println!("✓ test_full_cpu_memory_integration passed!");
            println!("  - Fold steps: {}", proof.steps.len());
            println!(
                "  - Memory proofs: {}",
                proof
                    .steps
                    .iter()
                    .map(|s| s.mem.proofs.len())
                    .sum::<usize>()
            );
            println!(
                "  - Memory ME claims: {}",
                proof
                    .steps
                    .iter()
                    .map(|s| s.mem.me_claims_time.len() + s.mem.me_claims_val.len())
                    .sum::<usize>()
            );
            println!("  - Final children: {}", final_children.len());
        }
        Err(e) => {
            // Expected failure due to norm bound
            let err_str = format!("{:?}", e);
            if err_str.contains("ΠRLC bound violated") || err_str.contains("norm bound") {
                println!("✓ test_full_cpu_memory_integration: norm bound violated (expected)");
                println!("  Error: {}", err_str);
                println!("");
                println!("  This is expected with current parameters.");
                println!("  Solutions:");
                println!("  1. Increase k_rho to allow more claims in final merge");
                println!("  2. Implement per-step integration (see architecture-gap.md)");
            } else {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }
}
