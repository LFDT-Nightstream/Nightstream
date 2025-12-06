use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as Fq;

/// Convert a Goldilocks element to a balanced i128 in [-(q-1)/2, (q-1)/2].
pub fn to_balanced_i128(x: Fq) -> i128 {
    let q: u128 = (1u128 << 64) - (1u128 << 32) + 1; // Goldilocks modulus
    let u: u128 = x.as_canonical_u64() as u128;
    // map to symmetric interval
    let half = (q - 1) / 2;
    if u <= half {
        u as i128
    } else {
        (u as i128) - (q as i128)
    }
}
