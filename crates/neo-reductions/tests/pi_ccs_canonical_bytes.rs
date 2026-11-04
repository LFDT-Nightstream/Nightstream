//! Canonical PiCCS proof bytes exist only for rectangular round messages.

use neo_math::{F, K};
use neo_reductions::PiCcsProof;
use p3_field::PrimeCharacteristicRing;

#[test]
fn canonical_bytes_reject_ragged_rounds() {
    let value = |word| K::from(F::from_u64(word));
    let left = PiCcsProof::new(vec![vec![value(1)], vec![value(2)], vec![value(3), value(4)]]);
    let right = PiCcsProof::new(vec![vec![value(1)], vec![value(2), value(3)], vec![value(4)]]);
    let rectangular = PiCcsProof::new(vec![vec![value(1), value(2)], vec![value(3), value(4)]]);

    assert_ne!(left, right, "the proof objects must be distinct");
    assert!(left.canonical_bytes().is_err() && right.canonical_bytes().is_err());
    assert!(rectangular.canonical_bytes().is_ok());
}
