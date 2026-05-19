use neo_ccs::{build_superneo_ring_forms, CcsStructure, Mat};
use neo_math::{D, K};
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
