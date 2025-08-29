use p3_goldilocks::Goldilocks as Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Returns true iff all coordinates are in {0,1}.
/// Note: Currently unused but kept for potential future optimizations.
#[allow(dead_code)]
#[inline]
pub fn is_binary_vec(v: &[Fq]) -> bool {
    v.iter().all(|&x| x == Fq::ZERO || x == Fq::ONE)
}

/// Adds column `col` into `acc`, optionally scaled by small integer digit in [-(b-1)..b-1].
#[allow(dead_code)]
#[inline]
pub fn add_scaled_col(acc: &mut [Fq], col: &[Fq], small_digit: i32) {
    debug_assert_eq!(acc.len(), col.len());
    match small_digit {
        0 => {}
        1 => for (a,c) in acc.iter_mut().zip(col) { *a += *c; }
        -1 => for (a,c) in acc.iter_mut().zip(col) { *a -= *c; }
        k => {
            let kf = if k >= 0 { Fq::from_u64(k as u64) } else { Fq::ZERO - Fq::from_u64((-k) as u64) };
            for (a,c) in acc.iter_mut().zip(col) { *a += *c * kf; }
        }
    }
}

/// Convert a Goldilocks element to a balanced i128 in [-(q-1)/2, (q-1)/2].
pub fn to_balanced_i128(x: Fq) -> i128 {
    let q: u128 = (1u128 << 64) - (1u128 << 32) + 1; // Goldilocks modulus
    let u: u128 = x.as_canonical_u64() as u128;
    // map to symmetric interval
    let half = (q - 1) / 2;
    if u <= half { u as i128 } else { (u as i128) - (q as i128) }
}
