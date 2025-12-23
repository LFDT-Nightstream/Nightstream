#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    matrix::Mat,
    poly::SparsePoly,
    relations::{CcsStructure, McsInstance, McsWitness, MeInstance},
};
use neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

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
    let mut m0 = Mat::zero(4, 4, F::ZERO);
    m0[(0, 0)] = F::ONE;
    let mut m1 = Mat::zero(4, 4, F::ZERO);
    m1[(0, 1)] = F::ONE;
    let mut m2 = Mat::zero(4, 4, F::ZERO);
    m2[(0, 2)] = F::ONE;
    let mut m3 = Mat::zero(4, 4, F::ZERO);
    m3[(0, 3)] = F::ONE;

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

    let z = vec![const_one, lhs0, lhs1, out];
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);
    let w = z.clone();

    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit = McsWitness { w, Z };
    (s, inst, wit)
}

fn build_single_chunk_inputs() -> (
    NeoParams,
    CcsStructure<F>,
    StepWitnessBundle<Cmt, F, K>,
    Vec<MeInstance<Cmt, F, K>>,
    Vec<Mat<F>>,
    DummyCommit,
    Mixers,
) {
    let m = 4usize;
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16,
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");
    let l = DummyCommit::default();
    let mixers = default_mixers();

    let const_one = F::ONE;
    let write_val = F::from_u64(1);
    let lookup_val = F::from_u64(1);
    let out_val = const_one + write_val + lookup_val;

    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, lookup_val, write_val, out_val);
    let _ = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let plain_mem = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ONE],
        inc_at_write_addr: vec![F::ONE],
    };
    let mem_init = MemInit::Zero;

    let plain_lut = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::ONE],
    };
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::ONE, F::from_u64(2)],
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
        mcs: (mcs_inst.clone(), mcs_wit.clone()),
        lut_instances: vec![(lut_inst.clone(), lut_wit)],
        mem_instances: vec![(mem_inst.clone(), mem_wit)],
        _phantom: PhantomData::<K>,
    };

    (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers)
}

#[test]
fn tamper_batched_time_round_poly_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-round");
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

    proof.steps[0].batched_time.round_polys[0][0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-round");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time round poly must fail verification"
    );
}

#[test]
fn tamper_ccs_header_digest_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-ccs-header-digest");
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

    proof.steps[0].fold.ccs_proof.header_digest[0] ^= 1;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-ccs-header-digest");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered CCS header digest must fail verification"
    );
}

#[test]
fn tamper_ccs_challenges_public_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-ccs-challenges-public");
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

    proof.steps[0].fold.ccs_proof.challenges_public.gamma += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-ccs-challenges-public");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered CCS challenges_public must fail verification"
    );
}

#[test]
fn tamper_rlc_inputs_changes_rho_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let ccs_only_step: StepWitnessBundle<Cmt, F, K> = StepWitnessBundle::from(step_bundle.mcs.clone());

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-inputs-changes-rho");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[ccs_only_step.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].fold.ccs_out[0].c.data[0] += F::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-inputs-changes-rho");
    let steps_public = [StepInstanceBundle::from(&ccs_only_step)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC inputs must fail rho transcript binding"
    );
}

#[test]
fn tamper_rlc_rho_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-rho");
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

    proof.steps[0].fold.rlc_rhos[0][(0, 0)] += F::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-rho");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC rho must fail verification"
    );
}

#[test]
fn tamper_rlc_parent_y_scalars_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-parent-y-scalars");
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
        !proof.steps[0].fold.rlc_parent.y_scalars.is_empty(),
        "fixture should produce a non-empty y_scalars table"
    );
    proof.steps[0].fold.rlc_parent.y_scalars[0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-parent-y-scalars");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC parent y_scalars must fail verification"
    );
}

#[test]
fn tamper_dec_child_y_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-dec-child-y");
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

    let child0 = proof
        .steps
        .get_mut(0)
        .and_then(|s| s.fold.dec_children.get_mut(0))
        .expect("fixture should produce at least one DEC child");
    child0.y[0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-dec-child-y");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered DEC child y must fail verification"
    );
}

#[test]
fn tamper_batched_time_label_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-label");
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

    proof.steps[0].batched_time.labels[0] = b"bad/label".as_slice();

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-label");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time label must fail verification"
    );
}

#[test]
fn tamper_batched_time_degree_bound_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-degree");
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

    proof.steps[0].batched_time.degree_bounds[0] += 1;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-degree");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time degree bound must fail verification"
    );
}

#[test]
fn tamper_batched_time_static_claim_sum_nonzero_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-static-sum");
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

    let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let step_inst = StepInstanceBundle::from(&step_bundle);
    let metas = RouteATimeClaimPlan::time_claim_metas_for_step(&step_inst, dims.d_sc);
    let static_idx = metas
        .iter()
        .enumerate()
        .find_map(|(i, m)| (!m.is_dynamic).then_some(i))
        .expect("fixture should include at least one static claim");

    proof.steps[0].batched_time.claimed_sums[static_idx] = K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-static-sum");
    let steps_public = [step_inst];
    assert!(
        fold_shard_verify(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "non-zero claimed_sum for static batched claim must fail verification"
    );
}
