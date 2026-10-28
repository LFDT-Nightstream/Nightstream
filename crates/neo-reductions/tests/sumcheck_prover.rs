use neo_math::{Fq, K};
use neo_reductions::sumcheck::{interpolate_from_evals, poly_eval_k};
use p3_field::PrimeCharacteristicRing;

#[test]
fn polynomial_helpers_match_interpolation() {
    let coeffs = [
        K::from(Fq::from_u64(3)),
        K::from(Fq::from_u64(2)),
        K::from(Fq::from_u64(5)),
    ];
    let x = Fq::from_u64(7);
    assert_eq!(poly_eval_k(&coeffs, K::from(x)), K::from(Fq::from_u64(262)));
    assert_eq!(poly_eval_k(&[], K::from(x)), K::ZERO);

    let xs = [K::ZERO, K::ONE, K::from(Fq::from_u64(2))];
    let ys = [
        K::from(Fq::from_u64(1)),
        K::from(Fq::from_u64(6)),
        K::from(Fq::from_u64(17)),
    ];
    let interpolated = interpolate_from_evals(&xs, &ys);
    for (&point, &expected) in xs.iter().zip(&ys) {
        assert_eq!(poly_eval_k(&interpolated, point), expected);
    }
}
