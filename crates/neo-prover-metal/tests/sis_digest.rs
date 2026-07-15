#![cfg(all(feature = "metal", target_vendor = "apple", neo_metal_shaders))]

use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{accumulator_digest, PI_RLC_PROJECTION_SIS_CONFIG};
use neo_math::F;
use neo_prover_metal::MetalSession;
use p3_field::PrimeCharacteristicRing;

#[test]
fn metal_sis_digest_matches_the_canonical_projection_digest() {
    let fields = (0..257)
        .map(|index| F::from_u64((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
        .collect::<Vec<_>>();
    let expected = accumulator_digest(PI_RLC_PROJECTION_SIS_CONFIG, &fields).expect("canonical SIS digest");
    let session = MetalSession::new().expect("Metal session");
    let actual = session
        .sis_accumulator_digest(PI_RLC_PROJECTION_SIS_CONFIG, &fields)
        .expect("Metal SIS digest");
    let cached = session
        .sis_accumulator_digest(PI_RLC_PROJECTION_SIS_CONFIG, &fields)
        .expect("cached Metal SIS digest");
    assert_eq!(actual, expected);
    assert_eq!(cached, expected);
}
