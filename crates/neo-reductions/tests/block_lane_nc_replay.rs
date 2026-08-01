use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::legacy_split_nc::{
    optimized_prove_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf,
    optimized_verify_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf,
};
use neo_reductions::optimized_engine::{OptimizedStructureCache, PiCcsProofVariant};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn fixture() -> (
    NeoParams,
    CcsStructure<F>,
    CcsClaim<neo_ajtai::Commitment, F>,
    CcsWitness<F>,
    AjtaiSModule,
    OptimizedStructureCache,
) {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("base-two parameters");
    let structure =
        CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).expect("zero-polynomial CCS");
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xB10C_1A9E);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, 1).expect("small Ajtai key");
    let log = AjtaiSModule::new(Arc::new(pp));
    let z = Mat::zero(D, 1, F::ZERO);
    let claim = CcsClaim {
        adv: None,
        c: log.commit(&z),
        x: Vec::new(),
        m_in: 0,
    };
    let witness = CcsWitness {
        w: vec![F::ZERO; D],
        Z: z,
    };
    let cache = OptimizedStructureCache::build(&structure).expect("optimized cache");
    (params, structure, claim, witness, log, cache)
}

fn prove_fixture() -> (
    NeoParams,
    CcsStructure<F>,
    CcsClaim<neo_ajtai::Commitment, F>,
    AjtaiSModule,
    OptimizedStructureCache,
    Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>>,
    neo_reductions::optimized_engine::PiCcsProof,
) {
    let (params, structure, claim, witness, log, cache) = fixture();
    let digest = [F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    let handle = [F::from_u64(5), F::from_u64(6), F::from_u64(7), F::from_u64(8)];
    let mut transcript = Poseidon2Transcript::new(b"neo.reductions/block-lane-replay/v1");
    let (outputs, proof, _, _) =
        optimized_prove_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf(
            &mut transcript,
            &params,
            &structure,
            core::slice::from_ref(&claim),
            core::slice::from_ref(&witness),
            &[],
            &[],
            digest,
            handle,
            None,
            &log,
            &cache,
        )
        .expect("block-lane prove");
    (params, structure, claim, log, cache, outputs, proof)
}

#[test]
fn block_lane_prove_verify_replays_fixed_19_plus_6_rounds() {
    let (params, structure, claim, _log, cache, outputs, proof) = prove_fixture();
    assert_eq!(proof.variant, PiCcsProofVariant::BlockLaneNcDelayedV1);
    assert_eq!(proof.sumcheck_rounds_nc.len(), 25);
    assert_eq!(outputs[0].s_col.len(), 19);
    assert!(outputs[0].y_zcol[D..].iter().all(|value| *value == K::ZERO));

    let digest = [F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    let handle = [F::from_u64(5), F::from_u64(6), F::from_u64(7), F::from_u64(8)];
    let mut transcript = Poseidon2Transcript::new(b"neo.reductions/block-lane-replay/v1");
    let (accepted, _) =
        optimized_verify_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf(
            &mut transcript,
            &params,
            &structure,
            core::slice::from_ref(&claim),
            &[],
            &outputs,
            &proof,
            &cache,
            digest,
            handle,
            None,
        )
        .expect("block-lane verify");
    assert!(accepted);
}

#[test]
fn block_lane_verify_rejects_nonzero_padding() {
    let (params, structure, claim, _log, cache, mut outputs, proof) = prove_fixture();
    outputs[0].y_zcol[D] = K::ONE;
    let mut transcript = Poseidon2Transcript::new(b"neo.reductions/block-lane-replay/v1");
    let result = optimized_verify_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf(
        &mut transcript,
        &params,
        &structure,
        core::slice::from_ref(&claim),
        &[],
        &outputs,
        &proof,
        &cache,
        [F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)],
        [F::from_u64(5), F::from_u64(6), F::from_u64(7), F::from_u64(8)],
        None,
    );
    assert!(result.is_err());
}
