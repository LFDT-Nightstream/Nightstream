use neo_fold::pi_ccs::nc_core::{range_product, nc_interpolated};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

#[test]
fn nc_interpolation_respects_endpoints() {
    let b = 3u32;
    let y0 = K::from(F::from_u64(7));
    let y1 = K::from(F::from_u64(9));
    assert_eq!(nc_interpolated::<F>(y0, y1, K::ZERO, b), range_product::<F>(y0, b));
    assert_eq!(nc_interpolated::<F>(y0, y1, K::ONE,  b), range_product::<F>(y1, b));
}
