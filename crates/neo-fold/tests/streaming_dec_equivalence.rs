#![allow(non_snake_case)]

use neo_ajtai::{
    set_global_pp, set_global_pp_seeded, setup_par, try_get_loaded_global_pp_for_dims, AjtaiSModule,
    Commitment as Cmt,
};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_prove_with_witnesses, fold_shard_verify, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(_rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert_eq!(cs.len(), 1, "test mixers expect k=1");
        cs[0].clone()
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let b_f = F::from_u64(b as u64);
        let mut pow = b_f;
        for i in 1..cs.len() {
            for (a, &x) in acc.data.iter_mut().zip(cs[i].data.iter()) {
                *a += x * pow;
            }
            pow *= b_f;
        }
        acc
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn build_single_step_bundle(params: &NeoParams, l: &AjtaiSModule, m: usize) -> StepWitnessBundle<Cmt, F, K> {
    let m_in = 0usize;
    let z: Vec<F> = (0..m)
        .map(|i| F::from_u64((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03))
        .collect();
    let Z = encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);
    let mcs_inst = McsInstance {
        c,
        x: vec![],
        m_in,
    };
    let mcs_wit = McsWitness { w: z, Z };
    StepWitnessBundle::from((mcs_inst, mcs_wit))
}

fn assert_step_fold_eq(a: &neo_fold::shard::FoldStep, b: &neo_fold::shard::FoldStep) {
    assert_eq!(a.ccs_out, b.ccs_out, "ccs_out mismatch");
    assert_eq!(a.rlc_rhos, b.rlc_rhos, "rlc_rhos mismatch");
    assert_eq!(a.rlc_parent, b.rlc_parent, "rlc_parent mismatch");
    assert_eq!(a.dec_children, b.dec_children, "dec_children mismatch");
}

#[test]
fn streaming_dec_matches_materialized_dec_with_loaded_pp() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = setup_par(&mut rng, D, params.kappa as usize, ccs.m).expect("setup_par");
    set_global_pp(pp).expect("set_global_pp");
    assert!(
        try_get_loaded_global_pp_for_dims(D, ccs.m).is_some(),
        "expected loaded PP"
    );
    let l = AjtaiSModule::from_global_for_dims(D, ccs.m).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> = steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), 1);
    assert_eq!(proof_mat.steps.len(), 1);
    assert_step_fold_eq(&proof_stream.steps[0].fold, &proof_mat.steps[0].fold);

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}

#[test]
fn streaming_dec_matches_materialized_dec_with_seeded_pp() {
    let n = 17usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let seed = [7u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, ccs.m, seed).expect("set_global_pp_seeded");
    assert!(
        try_get_loaded_global_pp_for_dims(D, ccs.m).is_none(),
        "expected PP to remain unloaded for seeded entry"
    );
    let l = AjtaiSModule::from_global_for_dims(D, ccs.m).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> = steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), 1);
    assert_eq!(proof_mat.steps.len(), 1);
    assert_step_fold_eq(&proof_stream.steps[0].fold, &proof_mat.steps[0].fold);

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}
