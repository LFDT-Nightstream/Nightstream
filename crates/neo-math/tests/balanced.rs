//! Centered field representatives and strict norm-bound behavior.

use neo_math::balanced::{to_balanced_i128, within_nc_bound};
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::{rand_core::RngCore, rand_core::SeedableRng, ChaCha8Rng};

#[test]
fn balanced_representatives_use_the_centered_interval() {
    assert_eq!(to_balanced_i128(Fq::ZERO), 0);
    assert_eq!(to_balanced_i128(Fq::ONE), 1);
    assert_eq!(to_balanced_i128(-Fq::ONE), -1);

    let half = (Fq::ORDER_U64 as i128 - 1) / 2;
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
    for _ in 0..1_000 {
        let value = to_balanced_i128(Fq::from_u64(rng.next_u64()));
        assert!((-half..=half).contains(&value), "out of range: {value}");
    }
}

#[test]
fn strict_norm_bound_has_exact_centered_boundaries() {
    assert!(!within_nc_bound(Fq::ZERO, 0));
    assert!(!within_nc_bound(Fq::ZERO, 1));

    for value in [Fq::ZERO, Fq::ONE, -Fq::ONE] {
        assert!(within_nc_bound(value, 2));
    }
    for value in [Fq::from_u64(2), -Fq::from_u64(2)] {
        assert!(!within_nc_bound(value, 2));
    }

    for value in [Fq::from_u64(9), -Fq::from_u64(9)] {
        assert!(within_nc_bound(value, 10));
    }
    for value in [Fq::from_u64(10), -Fq::from_u64(10)] {
        assert!(!within_nc_bound(value, 10));
    }
}
