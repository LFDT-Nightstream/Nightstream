#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::CcsStructure;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::finalize::{ObligationFinalizer, ReferenceFinalizer};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove_with_witnesses, fold_shard_verify, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

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
    let mcs_inst = neo_ccs::relations::McsInstance { c, x: vec![], m_in };
    let mcs_wit = neo_ccs::relations::McsWitness { w: z, Z };
    StepWitnessBundle::from((mcs_inst, mcs_wit))
}

#[test]
fn reference_finalizer_accepts_and_rejects_tampering() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let seed = [7u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, ccs.m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, ccs.m).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> = steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_prove = Poseidon2Transcript::new(b"ref-finalizer/smoke");
    let (proof, outputs, wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove_with_witnesses");

    let mut tr_verify = Poseidon2Transcript::new(b"ref-finalizer/smoke");
    let _ = fold_shard_verify(
        mode,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");

    // Finalization succeeds with correct witnesses.
    let mut fin_ok = ReferenceFinalizer::new(
        params.clone(),
        Arc::new(ccs.clone()),
        l.clone(),
        wits.final_main_wits.clone(),
        wits.val_lane_wits.clone(),
    )
    .expect("ReferenceFinalizer::new");
    fin_ok.finalize(&outputs.obligations).expect("finalize ok");

    // Tamper a witness matrix: finalization must reject.
    let mut bad_main = wits.final_main_wits.clone();
    if let Some(first) = bad_main.first_mut() {
        first.as_mut_slice()[0] = first.as_slice()[0] + F::ONE;
    }
    let mut fin_bad = ReferenceFinalizer::new(
        params,
        Arc::new(ccs),
        l,
        bad_main,
        wits.val_lane_wits,
    )
    .expect("ReferenceFinalizer::new");
    assert!(
        fin_bad.finalize(&outputs.obligations).is_err(),
        "tampered witness must be rejected"
    );
}
