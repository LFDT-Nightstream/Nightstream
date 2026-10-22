use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[inline]
pub fn to_balanced_i128<F: PrimeField64 + PrimeCharacteristicRing>(v: F) -> i128 {
    let q = F::ORDER_U64 as u128;
    let u = v.as_canonical_u64() as u128;
    let half = (q - 1) / 2;
    if u <= half {
        u as i128
    } else {
        (u as i128) - (q as i128)
    }
}

#[inline]
pub fn within_nc_bound<F: PrimeField64 + PrimeCharacteristicRing>(v: F, b: u32) -> bool {
    if b < 2 {
        return false;
    }
    let x = to_balanced_i128(v);
    let bound = (b as i128) - 1;
    (-bound..=bound).contains(&x)
}
