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

#[test]
fn signed_unit_column_masks_reconstruct_exact_row_major_values() {
    let positive = [1 << 0 | 1 << 53, 1 << 7, 0, 1 << 31];
    let negative = [1 << 11, 1 << 1 | 1 << 42, 1 << 52, 0];
    let packed = Mat::<Goldilocks>::compact_signed_unit_from_column_masks(54, 4, &positive, &negative)
        .expect("valid column masks");
    let mut dense = Mat::zero(54, 4, Goldilocks::ZERO);
    for column in 0..4 {
        for row in 0..54 {
            let bit = 1u64 << row;
            if positive[column] & bit != 0 {
                dense[(row, column)] = Goldilocks::ONE;
            } else if negative[column] & bit != 0 {
                dense[(row, column)] = Goldilocks::ZERO - Goldilocks::ONE;
            }
        }
    }

    assert!(packed.is_packed_signed_unit());
    assert_eq!(packed.packed_signed_unit_nonzero_count(), Some(8));
    assert_eq!(
        packed.packed_signed_unit_column_masks(),
        Some((&positive[..], &negative[..]))
    );
    assert_eq!(packed, dense);
}

#[test]
fn signed_unit_column_masks_reject_noncanonical_shapes() {
    assert!(Mat::<Goldilocks>::compact_signed_unit_from_column_masks(65, 1, &[0], &[0]).is_err());
    assert!(Mat::<Goldilocks>::compact_signed_unit_from_column_masks(4, 2, &[0], &[0, 0]).is_err());
    assert!(Mat::<Goldilocks>::compact_signed_unit_from_column_masks(4, 1, &[1 << 4], &[0]).is_err());
    assert!(Mat::<Goldilocks>::compact_signed_unit_from_column_masks(4, 1, &[1 << 2], &[1 << 2]).is_err());

    let high_row = Mat::<Goldilocks>::compact_signed_unit_from_column_masks(64, 1, &[1 << 63], &[0])
        .expect("row 63 is canonical for a 64-row matrix");
    assert_eq!(high_row[(63, 0)], Goldilocks::ONE);
}
