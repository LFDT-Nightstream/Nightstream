use p3_field::PrimeField64;

use super::*;

fn form(column: usize) -> Linear {
    Linear {
        constant: F::ZERO,
        terms: vec![(column, F::ONE)],
    }
}

fn product(left: usize, right: usize, output: usize) -> Row {
    Row {
        a: form(left),
        b: form(right),
        c: form(output),
    }
}

fn at(row: &Row, values: &[F]) -> F {
    let eval = |linear: &Linear| {
        linear
            .terms
            .iter()
            .fold(linear.constant, |value, &(column, coefficient)| {
                value + values[column] * coefficient
            })
    };
    eval(&row.a) * eval(&row.b) - eval(&row.c)
}

#[test]
fn half_adder_and_companion_force_the_original_product() {
    let gate = Product {
        row: 0,
        left: 1,
        right: 2,
        output: 3,
    };
    let row = rows(&[gate], 4).next().unwrap();
    for a in 0..=1 {
        for b in 0..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    for e in -1..=1 {
                        let values = [1, a, b, c, d, e].map(F::from_i64);
                        let holds = at(&row, &values) == F::ZERO && d + e == 1;
                        assert_eq!(holds, c == a * b && d == (a ^ b) && e == 1 - d);
                    }
                }
            }
        }
    }
}

#[test]
fn lowering_follows_proved_inputs_without_circular_product_rows() {
    let raw = [product(1, 1, 1), product(2, 2, 2), product(1, 2, 3), product(3, 2, 4)];
    assert_eq!(Product::checked(raw.iter(), &[true; 5]).len(), 2);
    let cyclic = [product(1, 2, 3), product(3, 2, 1), product(2, 2, 2)];
    assert!(Product::checked(cyclic.iter(), &[true; 4]).is_empty());
    let reversed = [product(1, 1, 1), product(2, 2, 2), product(3, 2, 4), product(1, 2, 3)];
    assert_eq!(Product::checked(reversed.iter(), &[true; 5]).len(), 1);
    let mut excluded = [true; 5];
    excluded[3] = false;
    assert!(Product::checked(raw.iter(), &excluded).is_empty());
}

#[test]
fn twenty_four_residuals_cannot_wrap_but_twenty_five_can() {
    let modulus = u128::from(F::ORDER_U64);
    assert!(6u128.pow(24) - 1 < modulus);
    assert!(6u128.pow(25) - 1 > modulus);
    let gates = (0..25)
        .map(|i| Product {
            row: i,
            left: 1 + 3 * i,
            right: 2 + 3 * i,
            output: 3 + 3 * i,
        })
        .collect::<Vec<_>>();
    assert_eq!(rows(&gates, 100).count(), 2);
    let mut value = -(modulus as i128);
    let mut digits = Vec::new();
    for _ in 0..25 {
        let digit = value.rem_euclid(6);
        let digit = if digit == 5 { -1 } else { digit };
        digits.push(digit);
        value = (value - digit) / 6;
    }
    assert_eq!(value, -1);
    digits[24] -= 6;
    let mut assignment = vec![F::ZERO; 150];
    let mut unbounded = F::ZERO;
    let mut coefficient = F::ONE;
    for (index, digit) in digits.into_iter().enumerate() {
        let mut found = false;
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    for d in 0..=1 {
                        if a + b - 2 * c - d == digit {
                            let gate = &gates[index];
                            assignment[gate.left] = F::from_i64(a as i64);
                            assignment[gate.right] = F::from_i64(b as i64);
                            assignment[gate.output] = F::from_i64(c as i64);
                            assignment[100 + 2 * index] = F::from_i64(d as i64);
                            found = true;
                        }
                    }
                }
            }
        }
        assert!(found);
        unbounded += coefficient * F::from_i64(digit as i64);
        coefficient *= F::from_u64(6);
    }
    assert_eq!(unbounded, F::ZERO);
    assert!(rows(&gates, 100).any(|row| at(&row, &assignment) != F::ZERO));
}
