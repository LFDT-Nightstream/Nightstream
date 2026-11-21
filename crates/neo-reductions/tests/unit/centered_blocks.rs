use super::DenseBlocks;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn centered_rows_preserve_transposed_signed_endpoints_and_wide_fallback() {
    let blocks = 2048;
    let mut values = vec![F::ZERO; blocks * D];
    for lane in 0..D {
        for block in 1..blocks {
            values[lane * blocks + block] = F::from_i16(((lane * 13 + block) % 256) as i16 - 128);
        }
    }
    let (storage, nonzero) = DenseBlocks::try_centered(&values, blocks).unwrap();
    assert!(!nonzero[0] && nonzero[1..].iter().all(|&flag| flag));
    for block in 0..blocks {
        let expanded = storage.ring(block);
        for lane in 0..D {
            assert_eq!(storage.coefficient(block, lane), values[lane * blocks + block]);
            assert_eq!(expanded.0[lane], values[lane * blocks + block]);
        }
    }
    for wide in [-129, 128, 4096] {
        *values.last_mut().unwrap() = F::from_i16(wide);
        assert!(DenseBlocks::try_centered(&values, blocks).is_none());
    }
}
