use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as Fq;

/// Convert a Goldilocks element to a balanced i64 in [-(q-1)/2, (q-1)/2].
///
/// This is a faster variant of `neo_math::balanced::to_balanced_i128` that avoids i128 division/mod in hot paths.
pub fn to_balanced_i64(x: Fq) -> i64 {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64;
    const HALF: u64 = (Q - 1) / 2;
    let u = x.as_canonical_u64();
    if u <= HALF {
        u as i64
    } else {
        -((Q - u) as i64)
    }
}
