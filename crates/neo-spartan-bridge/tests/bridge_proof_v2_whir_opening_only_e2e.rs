#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::CcsStructure;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_closure_proof::prove_whir_p3_ajtai_opening_only_v1;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove_with_witnesses, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_spartan_bridge::bridge_proof_v2::compute_closure_statement_v1;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{prove_fold_run, setup_fold_run, verify_bridge_proof_v2, BridgeProofV2};
use neo_transcript::{Poseidon2Transcript, Transcript};
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
    let z_mat = encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&z_mat);
    let mcs_inst = neo_ccs::relations::McsInstance { c, x: vec![], m_in };
    let mcs_wit = neo_ccs::relations::McsWitness { w: z, Z: z_mat };
    StepWitnessBundle::from((mcs_inst, mcs_wit))
}

#[test]
fn bridge_proof_v2_whir_opening_only_roundtrip_and_tamper() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let seed = [13u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, ccs.m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, ccs.m).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    // IMPORTANT: the Spartan bridge circuit replays the native session transcript which is
    // instantiated with this fixed label.
    let mut tr_prove = Poseidon2Transcript::new(b"neo.fold/session");
    let (fold_run, outputs, wits) = fold_shard_prove_with_witnesses(
        mode,
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

    let vm_digest = [0u8; 32];
    let witness = FoldRunWitness::new(fold_run, steps_instance.clone(), vec![], vm_digest, None);
    let (pk, vk) = setup_fold_run(&params, &ccs, &witness).expect("setup_fold_run");
    let spartan = prove_fold_run(&pk, &params, &ccs, witness).expect("prove_fold_run");

    let closure_stmt = compute_closure_statement_v1(&spartan.statement);
    let closure = prove_whir_p3_ajtai_opening_only_v1(
        &closure_stmt,
        &params,
        &ccs,
        &outputs.obligations,
        &wits.final_main_wits,
        &wits.val_lane_wits,
    )
    .expect("prove whir opening-only closure");

    let proof = BridgeProofV2 {
        spartan: spartan.clone(),
        closure_stmt,
        closure,
    };

    let ok = verify_bridge_proof_v2(
        &vk,
        &params,
        &ccs,
        &vm_digest,
        &steps_instance,
        None,
        &[],
        &proof,
    )
    .expect("verify");
    assert!(ok, "BridgeProofV2 must verify");

    // Tamper witness for closure proof: Phase-1 proof still verifies, closure must fail.
    let mut bad_main = wits.final_main_wits.clone();
    if let Some(first) = bad_main.first_mut() {
        first.as_mut_slice()[0] = first.as_slice()[0] + F::ONE;
    }
    let bad_closure = prove_whir_p3_ajtai_opening_only_v1(
        &proof.expected_closure_statement(),
        &params,
        &ccs,
        &outputs.obligations,
        &bad_main,
        &wits.val_lane_wits,
    )
    .expect("prove whir opening-only closure (bad witness)");
    let bad_proof = BridgeProofV2 {
        spartan,
        closure_stmt: proof.expected_closure_statement(),
        closure: bad_closure,
    };

    assert!(
        verify_bridge_proof_v2(
            &vk,
            &params,
            &ccs,
            &vm_digest,
            &steps_instance,
            None,
            &[],
            &bad_proof,
        )
        .is_err(),
        "tampered closure witness must be rejected"
    );
}
