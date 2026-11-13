//! The Akita boundary on one polynomial: the order of its variables, and the
//! commitment profile that the verifier accepts for a point.

use akita_types::CommittedGroup;
use neo_math::{from_complex, F, K};
use p3_field::PrimeCharacteristicRing;

use super::prover::Prover;
use super::verifier::{Opening, Verifier};
use super::{base, catalog, commitment, commitment_bytes};
use crate::Error;

const VARS: usize = 16;
const LARGER: usize = 20;
const SESSION: &[u8] = b"neo-compress/tests/akita";

/// Cells 0 to 4 of the polynomial. The other cells are zero, so no two
/// variables have the same role.
fn cells() -> Vec<F> {
    (1..=5).map(F::from_u64).collect()
}

fn point() -> Vec<K> {
    (0..VARS as u64)
        .map(|variable| from_complex(F::from_u64(3 + variable), F::from_u64(100 + 7 * variable)))
        .collect()
}

/// The multilinear polynomial with the evaluation table `cells()` at
/// `point`, where coordinate `variable` binds bit `bit(variable)` of the
/// cell index.
fn value(point: &[K], bit: impl Fn(usize) -> usize) -> K {
    cells()
        .into_iter()
        .enumerate()
        .map(|(index, cell)| {
            let weight: K = point
                .iter()
                .enumerate()
                .map(|(variable, &coordinate)| {
                    if index >> bit(variable) & 1 == 1 {
                        coordinate
                    } else {
                        K::ONE - coordinate
                    }
                })
                .product();
            weight * cell
        })
        .sum()
}

/// The commitment of `cells()` as a polynomial of `vars` variables, and its
/// opening to `value` at `point`.
fn opened(vars: usize, point: &[K], value: K) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let prover = Prover::new(vars)?;
    let committed = prover.commit(vars, cells().into_iter().map(base).collect())?;
    let bytes = committed.bytes.clone();
    Ok((bytes, prover.open(committed, point, value, SESSION)?))
}

fn verify(vars: &[usize], commitment: &[u8], point: &[K], value: K, proof: &[u8]) -> Result<(), Error> {
    Verifier::new(vars)?.verify(&Opening {
        commitment,
        point,
        value,
        proof,
        session: SESSION,
    })
}

#[test]
fn coordinate_b_of_a_point_binds_bit_b_of_the_cell_index() {
    let point = point();
    let low_first = value(&point, |variable| variable);
    let high_first = value(&point, |variable| VARS - 1 - variable);
    assert_ne!(low_first, high_first);
    let (commitment, proof) = opened(VARS, &point, low_first).expect("opening");
    verify(&[VARS], &commitment, &point, low_first, &proof).expect("the value with bit 0 first verifies");
    assert!(verify(&[VARS], &commitment, &point, high_first, &proof).is_err());
    let other_order = opened(VARS, &point, high_first)
        .and_then(|(_, proof)| verify(&[VARS], &commitment, &point, high_first, &proof));
    assert!(other_order.is_err(), "the value with bit 0 last has no opening");
}

/// A commitment of 20 variables with a correct opening at the point after
/// four zeros has the same value as the polynomial of 16 variables.
#[test]
fn a_commitment_of_another_size_does_not_verify() {
    let point = point();
    let value = value(&point, |variable| variable);
    let mut padded = point.clone();
    padded.resize(LARGER, K::ZERO);
    let (commitment, proof) = opened(LARGER, &padded, value).expect("opening");
    verify(&[VARS, LARGER], &commitment, &padded, value, &proof).expect("the opening of 20 variables verifies");
    assert!(matches!(
        verify(&[VARS, LARGER], &commitment, &point, value, &proof),
        Err(Error::Rejected(_))
    ));
}

/// The rows of a commitment of 16 variables decode under the profile of 20
/// variables. Akita rejects a change of one profile field when it decodes.
#[test]
fn a_commitment_under_another_profile_does_not_verify() {
    let point = point();
    let value = value(&point, |variable| variable);
    let (honest, proof) = opened(VARS, &point, value).expect("opening");
    verify(&[VARS], &honest, &point, value, &proof).expect("the honest opening verifies");

    let catalog = catalog().expect("catalog");
    let other = catalog
        .rows()
        .map(|row| row.profiles())
        .find(|profiles| profiles.precommitteds.is_empty() && profiles.final_group.group.num_vars() == LARGER)
        .expect("the row of 20 variables")
        .final_group;
    let rows = commitment(&honest)
        .expect("commitment")
        .commitment()
        .clone();
    let relabeled = CommittedGroup::new(other, rows).expect("the rows fit the other profile");
    let relabeled = commitment_bytes(&relabeled).expect("bytes");
    for vars in [&[VARS][..], &[VARS, LARGER]] {
        assert!(matches!(
            verify(vars, &relabeled, &point, value, &proof),
            Err(Error::Rejected(_))
        ));
    }
}
