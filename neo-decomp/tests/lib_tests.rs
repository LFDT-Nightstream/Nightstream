use neo_decomp::*;
use neo_fields::F;
use neo_modint::{Coeff, ModInt};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use quickcheck_macros::quickcheck;

#[test]
fn roundtrip() {
    let z = vec![F::from_u64(5), F::from_u64(9)];
    let mat = decomp_b(&z, 2, 4);
    for (col, &orig) in z.iter().enumerate() {
        let mut acc = F::ZERO;
        for row in (0..4).rev() {
            acc = acc * F::from_u64(2) + mat.get(row, col).unwrap();
        }
        assert_eq!(acc, orig);
    }
}

#[test]
fn test_signed_decomp_roundtrip() {
    let mut rng = rand::rng();
    let n = 8;
    let b = 3u64;
    let k = 45;
    let z: Vec<ModInt> = (0..n).map(|_| ModInt::random(&mut rng)).collect();
    let (matrix, g) = signed_decomp_b(&z, b, k);
    let recon = reconstruct_decomp(&matrix, &g);
    assert_eq!(recon, z);
    let r = (b as i128 - 1) / 2;
    for row in 0..k {
        for col in 0..n {
            let v: i128 = matrix.get(row, col).unwrap().into();
            assert!(v.abs() <= r);
        }
    }
}

#[quickcheck]
fn prop_decomp_reconstruct(z: Vec<ModInt>) -> bool {
    let b = 3u64;
    let k = 45;
    let (matrix, g) = signed_decomp_b(&z, b, k);
    reconstruct_decomp(&matrix, &g) == z
}
