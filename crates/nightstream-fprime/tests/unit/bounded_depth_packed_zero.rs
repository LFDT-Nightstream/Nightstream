use super::*;

fn linear(constant: i64, terms: &[(usize, i64)]) -> Linear {
    Linear {
        constant: F::from_i64(constant),
        terms: terms.iter().map(|&(c, k)| (c, F::from_i64(k))).collect(),
    }
}

fn forcing(column: usize) -> Row {
    let value = linear(0, &[(column, 1)]);
    Row {
        a: value.clone(),
        b: value.clone(),
        c: value,
    }
}

#[test]
fn zero_product_requires_forced_inputs_and_handles_complements() {
    let raw = vec![
        forcing(1),
        forcing(2),
        Row {
            a: linear(0, &[(1, 1)]),
            b: linear(1, &[(2, -1)]),
            c: Linear::default(),
        },
    ];
    let zeros = ZeroProduct::checked(raw.iter(), |_| true);
    assert_eq!(zeros.len(), 1);
    assert!(ZeroProduct::checked(raw[2..].iter(), |_| true).is_empty());
    assert!(ZeroProduct::checked(raw.iter(), |column| column != 1).is_empty());
    let row = rows(&zeros, 3).next().unwrap();
    for a in 0..=1 {
        for b in 0..=1 {
            for u in -1..=1 {
                let values = [1, a, b, u].map(F::from_i64);
                let residual = row
                    .a
                    .terms
                    .iter()
                    .fold(row.a.constant, |v, &(column, c)| v + values[column] * c);
                assert_eq!(residual == F::ZERO, a * (1 - b) == 0 && u == a + 1 - b);
            }
        }
    }
}

#[test]
fn original_boolean_inputs_are_needed_for_completeness() {
    let source = ZeroProduct {
        row: 0,
        left: Input {
            column: 1,
            complement: false,
        },
        right: Input {
            column: 2,
            complement: false,
        },
    };
    assert_eq!(source.left.value(&[F::ONE, F::NEG_ONE, F::ONE]), F::NEG_ONE);
    for a in 0..=1 {
        for b in 0..=1 {
            assert_eq!((-1..=1).any(|u| u == a + b), a * b == 0);
        }
    }
    let zeros = vec![source; 32];
    assert_eq!(rows(&zeros, 3).count(), 2);
}

#[test]
fn a_thirty_second_zero_residual_would_wrap_the_modulus() {
    let zeros = (0..32)
        .map(|i| ZeroProduct {
            row: i,
            left: Input {
                column: 1 + 2 * i,
                complement: false,
            },
            right: Input {
                column: 2 + 2 * i,
                complement: false,
            },
        })
        .collect::<Vec<_>>();
    let mut values = vec![F::ZERO; 97];
    values[0] = F::ONE;
    let mut unbounded = F::ZERO;
    let mut coefficient = F::ONE;
    for (index, zero) in zeros.iter().enumerate() {
        if index == 0 || index >= 16 {
            values[65 + index] = F::NEG_ONE;
        }
        if index >= 16 {
            values[zero.left.column] = F::ONE;
            values[zero.right.column] = F::ONE;
        }
        unbounded += coefficient * (values[65 + index] - zero.left.value(&values) - zero.right.value(&values));
        coefficient *= F::from_u64(4);
    }
    assert_eq!(unbounded, F::ZERO);
    assert!(rows(&zeros, 65).any(|row| row
        .a
        .terms
        .iter()
        .fold(row.a.constant, |v, &(column, c)| v + values[column] * c)
        != F::ZERO));
}
