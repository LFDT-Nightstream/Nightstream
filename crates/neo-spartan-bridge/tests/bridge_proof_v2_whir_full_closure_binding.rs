use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::CcsStructure;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove_with_witnesses, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{
    prove_bridge_proof_v2_whir_p3_full_closure, setup_fold_run, verify_bridge_proof_v2,
    verify_bridge_proof_v2_statement_only,
};
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
fn bridge_proof_v2_whir_full_closure_is_bound_to_spartan_statement() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let seed = [14u8; 32];
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
    let (fold_run, _outputs, wits) =
        fold_shard_prove_with_witnesses(mode, &mut tr_prove, &params, &ccs, &steps_witness, &[], &[], &l, mixers)
            .expect("prove_with_witnesses");

    let vm_digest_a = [0u8; 32];
    let vm_digest_b = [1u8; 32];

    let witness_a = FoldRunWitness::new(fold_run.clone(), steps_instance.clone(), vec![], vm_digest_a, None);
    let (pk, vk) = setup_fold_run(&params, &ccs, &witness_a).expect("setup_fold_run");

    let proof_a = prove_bridge_proof_v2_whir_p3_full_closure(
        &pk,
        &params,
        &ccs,
        witness_a,
        &wits.final_main_wits,
        &wits.val_lane_wits,
    )
    .expect("prove A");

    let witness_b = FoldRunWitness::new(fold_run, steps_instance.clone(), vec![], vm_digest_b, None);
    let proof_b = prove_bridge_proof_v2_whir_p3_full_closure(
        &pk,
        &params,
        &ccs,
        witness_b,
        &wits.final_main_wits,
        &wits.val_lane_wits,
    )
    .expect("prove B");

    assert!(
        verify_bridge_proof_v2(&vk, &params, &ccs, &vm_digest_a, &steps_instance, None, &[], &proof_a)
            .expect("verify A"),
        "proof A must verify"
    );
    assert!(
        verify_bridge_proof_v2(&vk, &params, &ccs, &vm_digest_b, &steps_instance, None, &[], &proof_b)
            .expect("verify B"),
        "proof B must verify"
    );

    assert!(
        verify_bridge_proof_v2_statement_only(
            &vk,
            &proof_a.spartan.statement,
            &proof_a,
            Some(&params),
            Some(&ccs),
            None
        )
        .expect("verify A statement-only"),
        "A statement-only verification must succeed"
    );

    // Mix `spartan` from A with `closure` from B. Spartan verification should still pass, but
    // closure verification must fail because the closure proof is bound to a different statement.
    let mut mixed = proof_a.clone();
    mixed.closure = proof_b.closure.clone();

    assert!(
        verify_bridge_proof_v2(&vk, &params, &ccs, &vm_digest_a, &steps_instance, None, &[], &mixed).is_err(),
        "mixed proof must be rejected (closure bound to different statement)"
    );
    assert!(
        verify_bridge_proof_v2_statement_only(
            &vk,
            &proof_a.spartan.statement,
            &mixed,
            Some(&params),
            Some(&ccs),
            None,
        )
        .is_err(),
        "mixed proof must be rejected in statement-only verification as well"
    );

    // Corrupt closure proof bytes: must fail verification.
    let mut tampered = proof_a.clone();
    let neo_closure_proof::ClosureProofV1::OpaqueBytes { proof_bytes } = &mut tampered.closure;
    let last = proof_bytes.last_mut().expect("non-empty proof_bytes");
    *last ^= 1;

    assert!(
        verify_bridge_proof_v2(&vk, &params, &ccs, &vm_digest_a, &steps_instance, None, &[], &tampered).is_err(),
        "tampered closure bytes must be rejected"
    );
}
