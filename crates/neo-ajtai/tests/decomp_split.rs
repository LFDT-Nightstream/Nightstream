#![allow(non_snake_case)]
use neo_ajtai::*;
use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;

#[test]
fn decomp_and_split_inverse() {
    let d = 54usize; let m = 16usize; let b = 2u32; let k = 12usize;

    // Random small vector z in range [-b^d .. b^d)
    let z: Vec<Fq> = (0..m).map(|i| Fq::from_u64((i%2) as u64)).collect();
    let Z = decomp_b(&z, b, d, DecompStyle::NonNegative);
    assert_range_b(&Z, b);

    // Recompose to z and check
    let mut z_back = vec![Fq::ZERO; m];
    for j in 0..m {
        let mut pow = Fq::ONE;
        for i in 0..d {
            let dij = Z[j*d + i];
            z_back[j] += dij * pow;
            pow = pow + pow; // b=2
        }
    }
    assert_eq!(z, z_back, "decomp_b does not invert");

    // Split then recombine
    let Zs = split_b(&Z, b, d, m, k, DecompStyle::NonNegative);
    for Zi in &Zs { assert_range_b(Zi, b); }

    let mut Z_back = vec![Fq::ZERO; d*m];
    let mut pow = Fq::ONE;
    for Zi in &Zs {
        for (a,&x) in Z_back.iter_mut().zip(Zi) { *a += x * pow; }
        pow = pow + pow; // b=2
    }
    assert_eq!(Z, Z_back, "split_b recomposition failed");
}
