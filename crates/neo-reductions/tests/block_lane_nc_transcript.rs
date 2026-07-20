use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    bind_me_inputs_accumulator_handle, build_dims_and_policy, digest_ccs_matrices,
    pi_ccs_header_bundle_digest_fields_for_variant, sample_beta_block, sample_delayed_projection_challenges,
    PiCcsTranscriptVariant, PI_CCS_BLOCK_NC_BATCH_WEIGHT_RAW_TAG, PI_CCS_BLOCK_NC_BETA_RAW_TAG,
    PI_CCS_BLOCK_NC_PRODUCER_BETA_RAW_TAG,
};
use neo_reductions::optimized_engine::oracle::BLOCK_LANE_NC_BLOCK_VARIABLES;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn identity_left(rows: usize, columns: usize) -> Mat<F> {
    let mut matrix = Mat::zero(rows, columns, F::ZERO);
    for index in 0..rows.min(columns) {
        matrix.set(index, index, F::ONE);
    }
    matrix
}

fn sample_after_handle(handle: [F; 4]) -> (Vec<K>, K, K) {
    let mut transcript = Poseidon2Transcript::new(b"block-lane-nc-transcript-test/v1");
    transcript.append_fields_raw(&[F::from_u64(99)]);
    bind_me_inputs_accumulator_handle(&mut transcript, 14, &handle).expect("bind pending-family handle");
    let beta_block = sample_beta_block(&mut transcript, BLOCK_LANE_NC_BLOCK_VARIABLES).expect("sample block challenge");
    let (producer_beta, batch_weight) =
        sample_delayed_projection_challenges(&mut transcript).expect("sample delayed challenges");
    (beta_block, producer_beta, batch_weight)
}

#[test]
fn delayed_challenges_are_domain_separated_and_post_binding() {
    assert_ne!(PI_CCS_BLOCK_NC_BETA_RAW_TAG, PI_CCS_BLOCK_NC_PRODUCER_BETA_RAW_TAG);
    assert_ne!(PI_CCS_BLOCK_NC_BETA_RAW_TAG, PI_CCS_BLOCK_NC_BATCH_WEIGHT_RAW_TAG);
    assert_ne!(
        PI_CCS_BLOCK_NC_PRODUCER_BETA_RAW_TAG,
        PI_CCS_BLOCK_NC_BATCH_WEIGHT_RAW_TAG
    );

    let baseline = sample_after_handle([F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)]);
    let rebound = sample_after_handle([F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    assert_ne!(
        baseline, rebound,
        "the complete handle must be fixed before every NC challenge"
    );

    let mut early = Poseidon2Transcript::new(b"block-lane-nc-transcript-test/v1");
    early.append_fields_raw(&[F::from_u64(99)]);
    let early_beta = sample_beta_block(&mut early, BLOCK_LANE_NC_BLOCK_VARIABLES).expect("early block challenge");
    let early_delayed = sample_delayed_projection_challenges(&mut early).expect("early delayed challenges");
    bind_me_inputs_accumulator_handle(
        &mut early,
        14,
        &[F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)],
    )
    .expect("late handle binding remains well-formed but is the wrong schedule");

    assert_ne!(baseline.0, early_beta);
    assert_ne!((baseline.1, baseline.2), early_delayed);
}

#[test]
fn block_lane_header_is_versioned_and_binds_block_geometry() {
    let logical_width = 2 * D;
    let structure = CcsStructure::new(vec![identity_left(D, logical_width)], SparsePoly::new(1, Vec::new()))
        .expect("small CCS structure");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("small parameters");
    let dims = build_dims_and_policy(&params, &structure).expect("valid dimensions");
    assert_eq!(dims.ell_block, 1);
    assert_eq!(dims.ell_block_nc, dims.ell_block + dims.ell_d);

    let matrix_digest = digest_ccs_matrices(&structure);
    let legacy = pi_ccs_header_bundle_digest_fields_for_variant(
        &params,
        &structure,
        dims,
        &matrix_digest,
        PiCcsTranscriptVariant::SplitNcV1,
    )
    .expect("legacy header");
    let block = pi_ccs_header_bundle_digest_fields_for_variant(
        &params,
        &structure,
        dims,
        &matrix_digest,
        PiCcsTranscriptVariant::BlockLaneNcDelayedV1,
    )
    .expect("block-lane header");
    assert_ne!(legacy, block, "proof variants must not share a transcript header");
}
