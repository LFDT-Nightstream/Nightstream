use neo_ccs::{build_superneo_ring_forms, poly::SparsePoly, CcsStructure, Mat};
use neo_math::{superneo_bar_block, Rq, D, K};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

pub fn superneo_y_ring(s: &CcsStructure<Fq>, z: &Mat<Fq>, r: &[K]) -> Vec<Vec<K>> {
    let forms = build_superneo_ring_forms(s, r).expect("SuperNeo ring forms");
    forms
        .into_iter()
        .map(|matrix_forms| {
            let mut row = vec![K::ZERO; D];
            for c in 0..s.m {
                let z_c = K::from(z[(c % D, c / D)]);
                for rho in 0..D {
                    row[rho] += matrix_forms[c][rho] * z_c;
                }
            }
            row
        })
        .collect()
}

#[test]
fn superneo_ring_forms_start_with_the_padded_identity() {
    let structure =
        CcsStructure::new(vec![Mat::zero(1, D, Fq::ZERO)], SparsePoly::new(1, vec![])).expect("valid one-row CCS");
    let point = vec![K::ZERO; D.next_power_of_two().trailing_zeros() as usize];
    let forms = build_superneo_ring_forms(&structure, &point).expect("identity-first forms");

    assert_eq!(forms.len(), structure.t() + 1);

    let witness_lane = 7;
    let mut identity_row = [Fq::ZERO; D];
    identity_row[0] = Fq::ONE;
    let mut witness_basis = [Fq::ZERO; D];
    witness_basis[witness_lane] = Fq::ONE;
    let expected = Rq(superneo_bar_block(identity_row)).mul(&Rq(witness_basis));
    for (actual, expected) in forms[0][witness_lane].iter().zip(expected.0) {
        assert_eq!(*actual, K::from(expected));
    }
    assert!(forms[1]
        .iter()
        .flatten()
        .all(|coefficient| *coefficient == K::ZERO));
}
