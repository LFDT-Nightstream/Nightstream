use neo_ccs::{CcsWitness, Mat};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn packed_witness_reconstructs_private_suffix_without_flat_copy() {
    let assignment = (0..11).map(Goldilocks::from_u64).collect::<Vec<_>>();
    let mut packed = Mat::zero(3, 4, Goldilocks::ZERO);
    for (column, &value) in assignment.iter().enumerate() {
        packed[(column % 3, column / 3)] = value;
    }
    let witness = CcsWitness {
        w: Vec::new(),
        Z: packed,
    };

    assert_eq!(witness.private_len(4, assignment.len()), Some(7));
    assert_eq!(
        witness.private_values(4, assignment.len()).as_deref(),
        Some(&assignment[4..])
    );
    assert_eq!(
        witness.private_len(4, 13),
        None,
        "packed padding is not authoritative data"
    );
}

#[test]
fn explicit_private_witness_remains_borrowed() {
    let private = vec![Goldilocks::ONE, Goldilocks::ZERO];
    let witness = CcsWitness {
        w: private.clone(),
        Z: Mat::zero(0, 0, Goldilocks::ZERO),
    };
    let values = witness.private_values(2, 4).expect("explicit witness");
    assert!(matches!(values, std::borrow::Cow::Borrowed(_)));
    assert_eq!(values.as_ref(), private);
}
