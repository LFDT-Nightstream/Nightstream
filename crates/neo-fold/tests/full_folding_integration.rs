#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    matrix::Mat,
    poly::SparsePoly,
    relations::{CcsStructure, McsInstance, McsWitness, MeInstance},
};
use neo_fold::finalize::{FinalizeReport, ObligationFinalizer};
use neo_fold::session::FoldingSession;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, fold_shard_verify_and_finalize, ShardObligations};
use neo_fold::shard::CommitMixers;
use neo_fold::PiCcsError;
use neo_math::{D, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

/// Dummy commit that produces zero commitments.
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

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = neo_ajtai::decomp_b(z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn build_add_ccs_mcs(
    params: &NeoParams,
    l: &DummyCommit,
    const_one: F,
    lhs0: F,
    lhs1: F,
    out: F,
) -> (CcsStructure<F>, McsInstance<Cmt, F>, McsWitness<F>) {
    // CCS with n=m=4, constraint: const_one + x1 + x2 - x3 = 0 (in row 0).
    // Rows 1..3 are dummy constraints (all zeros) to keep a small but square instance.
    //
    // We require square CCS so `ensure_identity_first()` can insert M₀ = I_n, which is
    // assumed by the Ajtai/NC pipeline and by Route A's witness-free terminal checks.
    // Columns: [const_one (public input), lhs0, lhs1, out]
    let mut m0 = Mat::zero(4, 4, F::ZERO);
    m0[(0, 0)] = F::ONE; // picks const_one (public input)
    let mut m1 = Mat::zero(4, 4, F::ZERO);
    m1[(0, 1)] = F::ONE; // picks lhs0
    let mut m2 = Mat::zero(4, 4, F::ZERO);
    m2[(0, 2)] = F::ONE; // picks lhs1
    let mut m3 = Mat::zero(4, 4, F::ZERO);
    m3[(0, 3)] = F::ONE; // picks out

    let term_const = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![1, 0, 0, 0],
    };
    let term_x1 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 1, 0, 0],
    };
    let term_x2 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 0, 1, 0],
    };
    let term_neg_out = neo_ccs::poly::Term {
        coeff: F::ZERO - F::ONE,
        exps: vec![0, 0, 0, 1],
    };
    let f = SparsePoly::new(4, vec![term_const, term_x1, term_x2, term_neg_out]);

    let s = CcsStructure::new(vec![m0, m1, m2, m3], f).expect("CCS");

    // Public input x = [const_one]; witness w = [lhs0, lhs1, out]
    let z = vec![const_one, lhs0, lhs1, out];
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);
    let w = z.clone(); // all witness, m_in=0

    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit = McsWitness { w, Z };
    (s, inst, wit)
}

fn default_mixers() -> Mixers {
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

fn build_single_chunk_inputs() -> (
    NeoParams,
    CcsStructure<F>,
    StepWitnessBundle<Cmt, F, K>,
    Vec<MeInstance<Cmt, F, K>>,
    Vec<Mat<F>>,
    DummyCommit,
    F,
) {
    let m = 4usize; // const_one (public), lhs0, lhs1, out
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16, // bump k_rho to satisfy Π_RLC norm bound comfortably for this test
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");
    let l = DummyCommit::default();

    // Program values: lookup_val + write_val = out
    let const_one = F::ONE;
    let write_val = F::from_u64(1);
    let lookup_val = F::from_u64(1);
    let out_val = const_one + write_val + lookup_val;

    // Build CCS (single chunk) enforcing out = write_val + lookup_val
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, lookup_val, write_val, out_val);
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Plain memory trace for one step
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let plain_mem = PlainMemTrace {
        steps: 1,
        // One write to addr 0 with value 1 (no reads).
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(1)],
        inc_at_write_addr: vec![F::from_u64(1)],
    };
    let mem_init = MemInit::Zero;

    // Plain lookup trace for one step
    let plain_lut = PlainLutTrace {
        // One lookup: key 0 -> value 1 (must match table content).
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(1)],
    };
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(1), F::from_u64(2)],
    };

    // Encode memory/lookup for this chunk
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
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &lut_table, &plain_lut, &commit_fn, Some(ccs.m), mcs_inst.m_in);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs_inst.clone(), mcs_wit.clone()),
        lut_instances: vec![(lut_inst.clone(), lut_wit)],
        mem_instances: vec![(mem_inst.clone(), mem_wit)],
        _phantom: PhantomData::<K>,
    };

    (params, ccs, step_bundle, acc_init, acc_wit_init, l, out_val)
}

// ============================================================================
// Tests using FoldingSession
// ============================================================================

#[test]
fn full_folding_integration_single_chunk() {
    let (params, ccs, step_bundle, _acc_init, _acc_wit_init, l, out_val) = build_single_chunk_inputs();

    // Use FoldingSession
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle.clone());

    // Prove
    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");

    // Verify using the session's collected steps
    session.verify_collected(&ccs, &proof).expect("verify should succeed");

    // Print a short summary so it's clear what was enforced.
    let step0 = &proof.steps[0];
    let mem_me_time = step0.mem.me_claims_time.len();
    let mem_me_val = step0.mem.me_claims_val.len();
    let ccs_me = step0.fold.ccs_out.len();
    let total_me = mem_me_time + ccs_me;
    let children = step0.fold.dec_children.len();
    println!("Full folding step:");
    println!("  CCS ME count: {}", ccs_me);
    println!("  Twist+Shout ME count (r_time lane): {}", mem_me_time);
    println!("  Twist ME count (r_val lane): {}", mem_me_val);
    println!("  Total ME into RLC (r_time lane): {}", total_me);
    println!("  Children after DEC: {}", children);
    println!("  Lookup enforced: key 0 -> val 1 from table [1, 2]");
    println!("  Memory enforced: write addr 0 := 1 (inc +1)");

    // Program output comes from CCS: out = const_one + lookup + write = 3.
    println!("  Program output (CCS out) = {}", out_val.as_canonical_u64());

    // Show a small, deterministic slice of the folded output.
    let final_children = proof.compute_final_main_children(&[]);
    if let Some(first) = final_children.first() {
        let r_len = first.r.len();
        let y0_prefix: Vec<K> = first
            .y
            .get(0)
            .map(|row| row.iter().take(2).cloned().collect())
            .unwrap_or_default();
        let y_scalars_prefix: Vec<K> = first.y_scalars.iter().take(2).cloned().collect();
        println!(
            "  First child: r_len={}, y[0][..2]={:?}, y_scalars[..2]={:?}",
            r_len, y0_prefix, y_scalars_prefix
        );
    }
}

#[test]
fn full_folding_integration_multi_step_chunk() {
    let (params, ccs, step_bundle_1, _acc_init, _acc_wit_init, l, _out_val) = build_single_chunk_inputs();

    let (mcs_inst, mcs_wit) = step_bundle_1.mcs.clone();

    // 4-step RW memory trace (k=2) with alternating write/read.
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    let plain_mem = PlainMemTrace {
        steps: 4,
        has_read: vec![F::ZERO, F::ONE, F::ZERO, F::ONE],
        has_write: vec![F::ONE, F::ZERO, F::ONE, F::ZERO],
        read_addr: vec![0, 0, 0, 1],
        write_addr: vec![0, 0, 1, 0],
        read_val: vec![F::ZERO, F::ONE, F::ZERO, F::from_u64(2)],
        write_val: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
        inc_at_write_addr: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
    };
    let mem_init = MemInit::Zero;

    // 4-step RO lookup trace (k=2) with lookups at steps 0 and 2.
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::ONE, F::from_u64(2)],
    };
    let plain_lut = PlainLutTrace {
        has_lookup: vec![F::ONE, F::ZERO, F::ONE, F::ZERO],
        addr: vec![0, 0, 1, 0],
        val: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
    };

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
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(&params, &lut_table, &plain_lut, &commit_fn, Some(ccs.m), mcs_inst.m_in);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs_inst, mcs_wit),
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData,
    };

    // Use FoldingSession
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle.clone());

    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");
}

// ============================================================================
// Tamper tests (still use low-level API for precise control)
// ============================================================================

#[test]
fn tamper_batched_claimed_sum_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-claim");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    // Claim 0 is ccs/time; claim 1 is the first Shout time claim in this fixture.
    proof.steps[0].batched_time.claimed_sums[1] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-claim");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "tampered claimed sum must fail verification");
}

#[test]
fn tamper_me_opening_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-me");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    // Mutate a memory ME opening used in terminal checks.
    proof.steps[0].mem.me_claims_time[0].y_scalars[0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-me");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "tampered ME opening must fail verification");
}

#[test]
fn tamper_shout_addr_pre_round_poly_fails() {
    use neo_fold::shard::MemOrLutProof;

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-shout-addr-pre");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mem0 = proof.steps.get_mut(0).expect("one step");
    let shout0 = mem0.mem.proofs.get_mut(0).expect("one Shout proof");
    let shout_proof = match shout0 {
        MemOrLutProof::Shout(p) => p,
        _ => panic!("expected Shout proof"),
    };

    shout_proof.addr_pre.round_polys[0][0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-shout-addr-pre");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(
        result.is_err(),
        "tampered Shout addr-pre round poly must fail verification"
    );
}

#[test]
fn tamper_twist_val_eval_round_poly_fails() {
    use neo_fold::shard::MemOrLutProof;

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-twist-val-eval-rounds");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mem0 = proof.steps.get_mut(0).expect("one step");
    let twist0 = mem0.mem.proofs.get_mut(1).expect("one Twist proof");
    let twist_proof = match twist0 {
        MemOrLutProof::Twist(p) => p,
        _ => panic!("expected Twist proof"),
    };
    let val_eval = twist_proof.val_eval.as_mut().expect("val_eval present");

    val_eval.rounds_lt[0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-twist-val-eval-rounds");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(
        result.is_err(),
        "tampered Twist val-eval round poly must fail verification"
    );
}

#[test]
fn missing_val_fold_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-missing-val-fold");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    assert!(
        proof.steps[0].val_fold.is_some(),
        "fixture should produce val_fold when Twist is present"
    );
    proof.steps[0].val_fold = None;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-missing-val-fold");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "missing val_fold must fail verification");
}

#[test]
fn verify_and_finalize_receives_val_lane() {
    struct RequireValLane;

    impl ObligationFinalizer<Cmt, F, K> for RequireValLane {
        type Error = PiCcsError;

        fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error> {
            if obligations.val.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "expected non-empty val-lane obligations for Twist".into(),
                ));
            }
            Ok(FinalizeReport {
                did_finalize_main: !obligations.main.is_empty(),
                did_finalize_val: !obligations.val.is_empty(),
            })
        }
    }

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-finalizer");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-finalizer");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let mut fin = RequireValLane;
    fold_shard_verify_and_finalize(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &mut fin,
    )
    .expect("verify_and_finalize should succeed");
}

#[test]
fn main_only_finalizer_is_rejected_when_val_lane_present() {
    struct MainOnly;

    impl ObligationFinalizer<Cmt, F, K> for MainOnly {
        type Error = PiCcsError;

        fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error> {
            Ok(FinalizeReport {
                did_finalize_main: !obligations.main.is_empty(),
                did_finalize_val: false,
            })
        }
    }

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-finalizer-main-only");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-finalizer-main-only");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let mut fin = MainOnly;
    let res = fold_shard_verify_and_finalize(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &mut fin,
    );

    assert!(
        res.is_err(),
        "finalizer that does not finalize val-lane obligations must be rejected"
    );
}

#[test]
fn wrong_shout_lookup_value_witness_fails() {
    let (params, ccs, mut step_bundle, acc_init, acc_wit_init, l, _out_val) = build_single_chunk_inputs();
    let mixers = default_mixers();

    // Corrupt the Shout `val` column at the active lookup step.
    let lut_inst = &step_bundle.lut_instances[0].0;
    let ell_addr = lut_inst.d * lut_inst.ell;
    let val_mat_idx = ell_addr + 1;
    let t0 = 0usize; // m_in=0 in this fixture

    let val_mat = &mut step_bundle.lut_instances[0].1.mats[val_mat_idx];
    let mut decoded = neo_memory::ajtai::decode_vector(&params, val_mat);
    decoded[t0] += F::ONE;
    *val_mat = neo_memory::encode::ajtai_encode_vector(&params, &decoded);

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-wrong-shout-witness");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed (even with invalid lookup witness)");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-wrong-shout-witness");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "invalid Shout lookup witness must fail verification");
}
