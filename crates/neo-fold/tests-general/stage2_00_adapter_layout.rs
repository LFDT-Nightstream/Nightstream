//! Stage 2: Adapter-only tests — witness layout and padding

use neo_ccs::Mat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn legacy_witness_is_col_major_and_power_of_two() {
    // Modern witness with a tiny Z (2×3) row-major data
    let d = 2usize; let m = 3usize;
    // Use balanced digits within range ±(b-1) for b=2 ⇒ {-1, 0, 1}
    let data_row_major = vec![
        F::from_i64(-1), F::from_i64( 0), F::from_i64( 1), // row 0
        F::from_i64( 1), F::from_i64(-1), F::from_i64( 0), // row 1
    ];
    let modern_wit = neo_ccs::MeWitness { Z: Mat::from_row_major(d, m, data_row_major.clone()) };

    // Base b = 2 (range ∈ {-1,0,1}) — we won't hit range checks with our small values
    let params = neo_params::NeoParams::goldilocks_small_circuits();

    let legacy = neo_fold::bridge_adapter::modern_to_legacy_witness(&modern_wit, &params)
        .expect("adapter");

    // Expect column-major flattening: idx = c*d + r
    // Columns are [-1,1], [0,-1], [1,0]
    let expected_col_major = vec![-1i64, 1, 0, -1, 1, 0];
    // Padded to next power-of-two: 6 -> 8
    let mut expected = expected_col_major.clone();
    expected.resize(8, 0);

    #[allow(deprecated)]
    {
        assert_eq!(legacy.z_digits, expected, "z_digits col-major + padding mismatch");
    }
}
