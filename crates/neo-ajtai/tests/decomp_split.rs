#![allow(non_snake_case)]
use neo_ajtai::*;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

#[test]
fn decomp_and_split_inverse() {
    let d = 54usize;
    let m = 16usize;
    let b = 2u32;
    let k = 12usize;

    // Random small vector z in range [-b^d .. b^d)
    let z: Vec<Fq> = (0..m).map(|i| Fq::from_u64((i % 2) as u64)).collect();
    let Z = decomp_b(&z, b, d, DecompStyle::NonNegative);
    assert_range_b(&Z, b).expect("Range check should pass");

    // Recompose to z and check
    let mut z_back = vec![Fq::ZERO; m];
    for j in 0..m {
        let mut pow = Fq::ONE;
        for i in 0..d {
            let dij = Z[j * d + i];
            z_back[j] += dij * pow;
            pow = pow + pow; // b=2
        }
    }
    assert_eq!(z, z_back, "decomp_b does not invert");

    // Split then recombine
    let Zs = split_b(&Z, b, d, m, k, DecompStyle::NonNegative);
    for Zi in &Zs {
        assert_range_b(Zi, b).expect("Range check should pass");
    }

    let mut Z_back = vec![Fq::ZERO; d * m];
    let mut pow = Fq::ONE;
    for Zi in &Zs {
        for (a, &x) in Z_back.iter_mut().zip(Zi) {
            *a += x * pow;
        }
        pow = pow + pow; // b=2
    }
    assert_eq!(Z, Z_back, "split_b recomposition failed");
}

#[test]
fn nonnegative_decomposition_round_trips_small_values() {
    let values = [0, 1, 2, 3, 17, 255, 65_535, 99_999]
        .into_iter()
        .map(Fq::from_u64)
        .collect::<Vec<_>>();
    let digit_count = 64;
    let digits = decomp_b(&values, 2, digit_count, DecompStyle::NonNegative);

    let mut recomposed = vec![Fq::ZERO; values.len()];
    for column in 0..values.len() {
        let mut power = Fq::ONE;
        for row in 0..digit_count {
            recomposed[column] += digits[column * digit_count + row] * power;
            power += power;
        }
    }
    assert_eq!(recomposed, values);
}

#[test]
fn split_b_recomposes_nontrivial_digits_and_rejects_range_violations() {
    let rows = 4;
    let columns = 3;
    let digit_count = 8;
    let input = (0..rows * columns)
        .map(|index| Fq::from_u64((index * 7 + 1) as u64))
        .collect::<Vec<_>>();
    let digits = split_b(&input, 2, rows, columns, digit_count, DecompStyle::Balanced);

    let mut recomposed = vec![Fq::ZERO; input.len()];
    let mut power = Fq::ONE;
    for digit in &digits {
        assert_range_b(digit, 2).expect("split digit must satisfy the strict range");
        for (result, &value) in recomposed.iter_mut().zip(digit) {
            *result += value * power;
        }
        power += power;
    }
    assert_eq!(recomposed, input);

    assert!(assert_range_b(&[Fq::from_u64(2)], 2).is_err());
    assert!(assert_range_b(&[Fq::ZERO, Fq::ONE, -Fq::ONE], 2).is_ok());
}
