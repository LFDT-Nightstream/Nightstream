#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove, fold_shard_prove_ccs_only_batched, fold_shard_verify, fold_shard_verify_ccs_only_batched,
    CommitMixers,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::ajtai::{commit_cols_for_ccs_m, encode_vector_for_ccs_m};
use neo_memory::witness::{MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for c in cs.iter().skip(1) {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, c);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let m_commit = commit_cols_for_ccs_m(m);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m_commit).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

fn identity_ccs(n: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS")
}

fn build_step(params: &NeoParams, l: &AjtaiSModule, m: usize, m_in: usize, seed: u64) -> StepWitnessBundle<Cmt, F, K> {
    // SuperNeo packed NC checks require bounded coefficients; keep synthetic witnesses in {-1,0,1}.
    let z: Vec<F> = (0..m)
        .map(|i| match (seed.wrapping_add(i as u64)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = encode_vector_for_ccs_m(params, m, &z).expect("encode witness for CCS width");
    let c = l.commit(&Z);
    StepWitnessBundle::from((CcsClaim { c, x, m_in }, CcsWitness { w, Z }))
}

#[test]
fn ccs_only_mcs_batched_k2_prove_verify() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = (0..5usize)
        .map(|i| build_step(&params, &l, ccs.m, 2, 100 + (i as u64) * 100))
        .collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    // Reduction-level K>1 smoke: Π_CCS prove/verify must accept two MCS slots.
    {
        let mcs_list: Vec<CcsClaim<Cmt, F>> = steps[..2].iter().map(|s| s.mcs.0.clone()).collect();
        let mcs_wits: Vec<CcsWitness<F>> = steps[..2].iter().map(|s| s.mcs.1.clone()).collect();
        let mut tr0 = Poseidon2Transcript::new(b"neo.fold/session");
        let (out, pi) = neo_fold::pi_ccs::prove(
            FoldingMode::Optimized,
            &mut tr0,
            &params,
            &ccs,
            &mcs_list,
            &mcs_wits,
            &[],
            &[],
            &l,
        )
        .expect("pi_ccs prove k_mcs=2");
        let mut tr1 = Poseidon2Transcript::new(b"neo.fold/session");
        let ok = neo_fold::pi_ccs::verify(
            FoldingMode::Optimized,
            &mut tr1,
            &params,
            &ccs,
            &mcs_list,
            &[],
            &out,
            &pi,
        )
        .expect("pi_ccs verify result");
        assert!(ok, "pi_ccs verify should pass for k_mcs=2");
    }

    let mut tr_p = Poseidon2Transcript::new(b"neo.fold/session");
    let proof = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
        2,
    )
    .expect("prove");
    assert_eq!(proof.steps.len(), 3, "5 steps batched by 2 should yield 3 fold steps");

    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/session");
    let outputs = fold_shard_verify_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &steps_public,
        &[],
        &proof,
        mixers,
        2,
    )
    .expect("verify");

    assert!(
        outputs.obligations.val.is_empty(),
        "ccs-only batched path must not emit val obligations"
    );
    assert_eq!(outputs.obligations.main.len(), params.k_rho as usize);
}

#[test]
fn ccs_only_default_shard_api_auto_batches() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = (0..5usize)
        .map(|i| build_step(&params, &l, ccs.m, 2, 1000 + (i as u64) * 100))
        .collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"neo.fold/session");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove");

    assert!(
        proof.steps.len() < steps.len(),
        "default shard API should auto-batch ccs-only steps when safe"
    );

    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/session");
    let outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &steps_public,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");

    assert!(
        outputs.obligations.val.is_empty(),
        "ccs-only path must not emit val obligations"
    );
    assert_eq!(outputs.obligations.main.len(), params.k_rho as usize);
}

#[test]
fn ccs_only_mcs_batched_rejects_sidecars() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let mut step = build_step(&params, &l, ccs.m, 2, 1234);
    step.mem_instances.push((
        MemInstance::<Cmt, F> {
            mem_id: 0,
            comms: Vec::new(),
            k: 2,
            d: 1,
            n_side: 2,
            steps: 1,
            lanes: 1,
            ell: 1,
            init: MemInit::Zero,
            init_digest: None,
            guest_addr_remap: None,
        },
        MemWitness { mats: Vec::new() },
    ));

    let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
    let err = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr,
        &params,
        &ccs,
        &[step],
        &[],
        &[],
        &l,
        mixers,
        2,
    )
    .expect_err("sidecar steps should be rejected");

    assert!(
        err.to_string()
            .contains("ccs-only batching does not support mem/lut sidecars"),
        "unexpected error: {err}"
    );
}
