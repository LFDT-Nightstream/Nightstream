use super::*;

fn form(constant: F, terms: &[(usize, F)]) -> Linear {
    Linear {
        constant,
        terms: terms.to_vec(),
    }
}

fn bit(column: usize) -> Row {
    let value = form(F::ZERO, &[(column, F::ONE)]);
    Row {
        a: value.clone(),
        b: value.clone(),
        c: value,
    }
}

fn residual(row: &Row, values: &[F]) -> F {
    let at = |side: &Linear| {
        side.terms
            .iter()
            .fold(side.constant, |sum, &(column, coefficient)| {
                sum + values[column] * coefficient
            })
    };
    at(&row.a) * at(&row.b) - at(&row.c)
}

#[test]
fn packing_counts_unique_sources_and_restores_source_rows() {
    for count in [0, 1, 30, 31, 32, 62, 63] {
        let mut original = (1..=count).map(bit).collect::<Vec<_>>();
        if count > 0 {
            original.push(bit(1));
        }
        original.push(bit(200));
        let mut rows = original.clone();
        let packed = PackedBooleans::lower(&mut rows, 300, 400, |column| column < 200);
        assert_eq!(packed.sources, (1..=count).collect::<Vec<_>>());
        assert_eq!(packed.row_count(), count.div_ceil(31));
        assert_eq!(rows.len(), 1 + count.div_ceil(31));
        assert_eq!(
            packed.original_rows(&rows).unwrap(),
            original.iter().collect::<Vec<_>>()
        );
        packed.validate(&rows, 300, 400).unwrap();
        let mut values = vec![F::ZERO; 400];
        for (index, &source) in packed.sources.iter().enumerate() {
            values[source] = F::from_usize(index % 2);
            values[packed.start + index] = F::ONE - values[source];
        }
        assert!(rows.iter().all(|row| residual(row, &values) == F::ZERO));
    }
}

#[test]
fn signed_unit_pairs_hold_exactly_at_complementary_bits() {
    let mut rows = vec![bit(1), bit(2)];
    let packed = PackedBooleans::lower(&mut rows, 3, 5, |_| true);
    assert_eq!(packed.row_count(), 1);
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let values = [F::ONE, F::from_i64(a), F::from_i64(b), F::from_i64(c), F::from_i64(d)];
                    let holds = residual(&rows[0], &values) == F::ZERO;
                    assert_eq!(
                        holds,
                        (a == 0 || a == 1) && a + c == 1 && (b == 0 || b == 1) && b + d == 1
                    );
                }
            }
        }
    }
}

#[test]
fn the_thirty_second_pair_would_allow_modulus_wrap() {
    let mut rows = (1..=32).map(bit).collect::<Vec<_>>();
    let packed = PackedBooleans::lower(&mut rows, 33, 65, |_| true);
    assert_eq!(rows.len(), 2);
    let mut values = vec![F::ZERO; 65];
    let mut coefficient = F::ONE;
    let mut invalid_single_row = F::ZERO;
    for index in 0..32 {
        let (source, companion) = if index < 2 {
            (F::ZERO, F::ZERO)
        } else if index >= 3 {
            (F::NEG_ONE, F::NEG_ONE)
        } else {
            (F::ZERO, F::ONE)
        };
        values[index + 1] = source;
        values[packed.start + index] = companion;
        invalid_single_row += coefficient * (source + companion - F::ONE);
        coefficient *= F::from_u64(4);
    }
    assert_eq!(invalid_single_row, F::ZERO);
    assert!(rows.iter().any(|row| residual(row, &values) != F::ZERO));
}

#[test]
fn packing_requires_the_signed_unit_bound() {
    let mut rows = vec![bit(1), bit(2)];
    PackedBooleans::lower(&mut rows, 3, 5, |_| true);
    let values = [F::ONE, F::from_i64(-2), F::ONE, F::NEG_ONE, F::ONE];
    assert_eq!(residual(&rows[0], &values), F::ZERO);
    assert_ne!(residual(&bit(1), &values), F::ZERO);
    assert!(![F::NEG_ONE, F::ZERO, F::ONE].contains(&values[1]));
}

#[test]
fn malformed_companion_maps_and_changed_rows_are_rejected() {
    let mut rows = vec![bit(1), bit(2)];
    let packed = PackedBooleans::lower(&mut rows, 3, 5, |_| true);
    let mut wrong = packed.clone();
    wrong.sources.reverse();
    assert!(wrong.validate(&rows, 3, 5).is_err());
    let mut wrong = packed.clone();
    wrong.sources[1] = 1;
    assert!(wrong.validate(&rows, 3, 5).is_err());
    let mut wrong = packed.clone();
    wrong.removed[1].0 = 0;
    assert!(wrong.validate(&rows, 3, 5).is_err());
    assert!(packed.validate(&rows, 3, 4).is_err());
    assert!(packed.validate(&rows, 4, 6).is_err());
    rows[0].a.constant += F::ONE;
    assert!(packed.validate(&rows, 3, 5).is_err());
}

#[test]
fn product_maps_restore_raw_rows_and_reject_circular_input_proofs() {
    let product = |a, b, c| Row {
        a: form(F::ZERO, &[(a, F::ONE)]),
        b: form(F::ZERO, &[(b, F::ONE)]),
        c: form(F::ZERO, &[(c, F::ONE)]),
    };
    let original = vec![bit(1), bit(2), product(1, 2, 3), product(3, 2, 4)];
    let mut rows = original.clone();
    let packed = PackedBooleans::lower(&mut rows, 5, 11, |_| true);
    assert_eq!(packed.products.len(), 2);
    assert_eq!(packed.end(), 11);
    assert_eq!(packed.row_count(), 2);
    assert_eq!(
        packed.original_rows(&rows).unwrap(),
        original.iter().collect::<Vec<_>>()
    );
    packed.validate(&rows, 5, 11).unwrap();
    for a in 0..=1 {
        for b in 0..=1 {
            let mut values = vec![F::ZERO; 11];
            values[1] = F::from_i64(a);
            values[2] = F::from_i64(b);
            values[3] = F::from_i64(a * b);
            values[4] = F::from_i64(a * b);
            values[5] = F::ONE - values[1];
            values[6] = F::ONE - values[2];
            for (index, gate) in packed.products.iter().enumerate() {
                let xor = packed.product_start() + 2 * index;
                values[xor] = values[gate.left] + values[gate.right] - F::from_u64(2) * values[gate.output];
                values[xor + 1] = F::ONE - values[xor];
            }
            assert!(rows.iter().all(|row| residual(row, &values) == F::ZERO));
        }
    }
    let mut wrong = packed.clone();
    wrong.products[0].left = 4;
    wrong.removed[2].1 = product(4, 2, 3);
    let mut forged = wrong.rows().collect::<Vec<_>>();
    assert!(wrong.validate(&forged, 5, 11).is_err());
    forged[0].a.constant += F::ONE;
    assert!(packed.validate(&forged, 5, 11).is_err());
}

/// A package has a fixed width. When the companion columns of every bit do
/// not fit, the packing takes the bits, then the zero products, then the
/// products that fit, and the other rows stay as they are.
#[test]
fn packing_stops_at_the_package_width() {
    let product = |a, b, c| Row {
        a: form(F::ZERO, &[(a, F::ONE)]),
        b: form(F::ZERO, &[(b, F::ONE)]),
        c: form(F::ZERO, &[(c, F::ONE)]),
    };
    let zero = |a, b| Row {
        a: form(F::ZERO, &[(a, F::ONE)]),
        b: form(F::ZERO, &[(b, F::ONE)]),
        c: Linear::default(),
    };
    let mut original = (1..=10).map(bit).collect::<Vec<_>>();
    original.extend([product(1, 2, 11), zero(3, 4), zero(5, 6)]);
    let start = 12;
    // Everything fits in 10 + 2 + 2 columns.
    for (room, sources, zeros, products) in [
        (14, 10, 2, 1),
        (13, 10, 2, 0),
        (11, 10, 1, 0),
        (10, 10, 0, 0),
        (4, 4, 0, 0),
        (0, 0, 0, 0),
    ] {
        let width = start + room;
        let mut rows = original.clone();
        let packed = PackedBooleans::lower(&mut rows, start, width, |_| true);
        assert_eq!(packed.sources, (1..=sources).collect::<Vec<_>>(), "room {room}");
        assert_eq!(packed.zeros.len(), zeros, "room {room}");
        assert_eq!(packed.products.len(), products, "room {room}");
        assert!(packed.end() <= width, "room {room}");
        packed.validate(&rows, start, width).unwrap();
        assert_eq!(
            packed.original_rows(&rows).unwrap(),
            original.iter().collect::<Vec<_>>(),
            "room {room}"
        );
        // An honest assignment satisfies the kept rows and the packed rows.
        let mut values = vec![F::ZERO; width.max(start)];
        values[0] = F::ONE;
        for column in [1, 2, 3, 5, 7, 9] {
            values[column] = F::ONE;
        }
        values[11] = F::ONE;
        for (index, &source) in packed.sources.iter().enumerate() {
            values[packed.start + index] = F::ONE - values[source];
        }
        for (index, gate) in packed.products.iter().enumerate() {
            let xor = packed.product_start() + 2 * index;
            values[xor] = values[gate.left] + values[gate.right] - F::from_u64(2) * values[gate.output];
            values[xor + 1] = F::ONE - values[xor];
        }
        for (index, gate) in packed.zeros.iter().enumerate() {
            values[packed.zero_start() + index] = gate.left.value(&values) + gate.right.value(&values);
        }
        assert!(rows.iter().all(|row| residual(row, &values) == F::ZERO), "room {room}");
        // A wrong bit, product or zero product fails in each case.
        for (column, wrong) in [(10, F::from_u64(2)), (11, F::ZERO), (4, F::ONE)] {
            let mut forged = values.clone();
            forged[column] = wrong;
            assert!(
                rows.iter().any(|row| residual(row, &forged) != F::ZERO),
                "room {room} column {column}"
            );
        }
    }
}
