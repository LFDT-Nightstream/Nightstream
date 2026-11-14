use std::{path::PathBuf, sync::Arc};

use super::*;
use crate::{
    load_bounded_depth_packages, ApplicationForm, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
    BoundedDepthPackages, NativeShape, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};

fn variable(column: usize) -> Linear {
    Linear {
        constant: F::ZERO,
        terms: vec![(column, F::ONE)],
    }
}

fn constant(value: i64) -> Linear {
    Linear {
        constant: F::from_i64(value),
        terms: Vec::new(),
    }
}

fn guarded(guard: Linear, residual: Linear) -> Row {
    Row {
        a: guard,
        b: residual,
        c: Linear::default(),
    }
}

fn boolean(column: usize) -> Row {
    guarded(variable(column), variable(column).plus(constant(-1)))
}

fn packages(rows: &[Row]) -> BoundedDepthPackages {
    packages_with_words(rows, 12)
}

fn packages_with_words(rows: &[Row], words: usize) -> BoundedDepthPackages {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    let base = load_bounded_depth_packages(
        &std::fs::read(root.join("nightstream-fprime-bounded-depth-light-v1.json")).unwrap(),
        &std::fs::read(root.join("nightstream-fprime-bounded-depth-refresh-v1.json")).unwrap(),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .unwrap();
    let mut records = ApplicationRecordsWriter::new().unwrap();
    for row in rows {
        let sides = [&row.a, &row.b, &row.c];
        records
            .append_row(
                ApplicationRowHeader {
                    constants: sides.map(|side| side.constant.as_canonical_u64()),
                    term_counts: sides.map(|side| side.terms.len()),
                },
                [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C]
                    .into_iter()
                    .zip(sides)
                    .flat_map(|(form, side)| {
                        side.terms.iter().map(move |&(variable, coefficient)| {
                            Ok(ApplicationTerm {
                                form,
                                variable,
                                coefficient: coefficient.as_canonical_u64(),
                            })
                        })
                    }),
            )
            .unwrap();
    }
    base.with_application(Arc::new(records.finish().unwrap()), words, &NativeShape::default())
        .unwrap()
}

fn fixture() -> Vec<Row> {
    let mut rows = [4, 5, 6, 7].map(boolean).to_vec();
    rows.extend([
        guarded(
            variable(8).scale(F::from_u64(7)),
            variable(4).plus(variable(5).scale(-F::ONE)),
        ),
        guarded(
            variable(8).scale(F::from_u64(11)),
            variable(6).plus(variable(7).scale(-F::ONE)),
        ),
        guarded(
            variable(8),
            variable(4)
                .plus(variable(5).scale(-F::ONE))
                .plus(variable(6).scale(F::TWO))
                .plus(variable(7).scale(-F::TWO)),
        ),
    ]);
    rows
}

fn value(form: &Linear, columns: &[F]) -> F {
    form.terms
        .iter()
        .fold(form.constant, |sum, &(column, coefficient)| {
            sum + coefficient * columns[column]
        })
}

#[test]
fn word_packing_preserves_all_ports_and_source_assignments() {
    let rows = fixture();
    let packages = packages(&rows);
    for package in [packages.light(), packages.refresh()] {
        let packing = &package.word_packing;
        assert_eq!(packing.groups.len(), 1);
        assert_eq!(packing.groups[0].sources.len(), 3);
        assert_eq!(packing.groups[0].pairs.len(), 2);
        assert_eq!(packing.groups[0].certificates.len(), 1);
        let plan = package.plan_rows().unwrap();
        let source = package.source_rows().unwrap();
        let layout = package.layout().unwrap();
        assert_eq!(source.piece_ranges()[0].len(), rows.len());
        assert_eq!(plan.piece_ranges()[0].len(), 2);
        for assignment in 0..16 {
            for guard in [0, 1, 2, F::ORDER_U64 - 1] {
                let mut columns = vec![F::ZERO; package.source_end()];
                columns[0] = F::ONE;
                let first = super::super::application::WITNESS_START;
                for bit in 0..4 {
                    columns[first + bit] = F::from_u64((assignment >> bit) & 1);
                }
                columns[first + 4] = F::from_u64(guard);
                for (offset, &column) in package.packed_booleans.sources.iter().enumerate() {
                    columns[package.packed_booleans.start + offset] = F::ONE - columns[column];
                }
                let original = source.piece_ranges()[0]
                    .clone()
                    .all(|row| source.residual(row, &columns) == F::ZERO);
                assert_eq!(
                    original,
                    plan.piece_ranges()[0]
                        .clone()
                        .all(|row| plan.residual(row, &columns) == F::ZERO)
                );
                let logical = layout.assignment(&columns).unwrap();
                let restored = package.source_assignment(&logical).unwrap();
                assert_eq!(
                    original,
                    source.piece_ranges()[0]
                        .clone()
                        .all(|row| source.residual(row, &restored) == F::ZERO)
                );
                for row in plan.piece_ranges()[0].clone() {
                    for port in 0..4 {
                        assert_eq!(
                            plan.evaluate(row, port, &columns),
                            layout.evaluate(&plan, row, port, &logical).unwrap()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sixty_four_pairs_are_split_before_goldilocks_wraparound() {
    let group = Group {
        sources: vec![0, 1, 2],
        certificates: Vec::new(),
        guard: constant(1),
        pairs: (1..=64)
            .map(|column| (Bit::Zero, Bit::Column(column)))
            .collect(),
    };
    let mut columns = vec![F::ONE; 65];
    for bit in 0..64 {
        columns[bit + 1] = F::from_u64((F::ORDER_U64 >> bit) & 1);
    }
    let unbounded = group
        .pairs
        .iter()
        .enumerate()
        .fold(F::ZERO, |sum, (bit, _)| {
            sum - F::from_u64(1u64 << bit) * columns[bit + 1]
        });
    assert_eq!(unbounded, F::ZERO);
    let rows = group.rows();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| value(&row.b, &columns) != F::ZERO));
}

#[test]
fn matcher_rejects_missing_bits_collisions_and_wide_words() {
    let known = BTreeMap::from([(1, 10), (2, 11), (3, 12)]);
    assert!(binary_pairs(&variable(4), &known).is_none());
    assert!(binary_pairs(&variable(1).plus(variable(2)), &known).is_none());
    assert!(binary_pairs(&variable(1).plus(constant(1)), &known).is_none());
    assert!(binary_pairs(&variable(1).scale(F::from_u64(3)), &known).is_none());
    assert!(binary_pairs(&variable(1).scale(F::from_u64(1u64 << 32)), &known).is_none());
    assert!(binary_pairs(&variable(1).scale(F::from_u64(1u64 << 31)), &known).is_some());
    assert!(binary_pairs(&constant(1).scale(F::from_u64(u32::MAX as u64)), &known).is_some());
}

#[test]
fn constants_aliases_and_guard_scaling_preserve_equality() {
    let known = BTreeMap::from([(1, 10), (2, 11)]);
    let residual = normalized(
        variable(1)
            .plus(variable(1).scale(F::TWO))
            .plus(constant(-3)),
    );
    assert!(binary_pairs(&residual, &known).is_none());
    let residual = normalized(
        variable(1)
            .plus(variable(1).scale(-F::ONE))
            .plus(variable(2))
            .plus(constant(-1)),
    );
    assert_eq!(binary_pairs(&residual, &known), Some(vec![(Bit::One, Bit::Column(2))]));
    let guard = variable(8).plus(constant(9));
    assert_eq!(guard_key(&guard), guard_key(&guard.clone().scale(F::from_u64(19))));
    assert_eq!(guard_key(&constant(0)), None);
    let group = Group {
        sources: vec![1, 2],
        certificates: vec![10],
        guard,
        pairs: vec![(Bit::One, Bit::Column(1)), (Bit::Zero, Bit::Column(2))],
    };
    for left in 0..2 {
        for right in 0..2 {
            let mut columns = vec![F::ZERO; 9];
            columns[1] = F::from_u64(left);
            columns[2] = F::from_u64(right);
            assert_eq!(value(&group.rows()[0].b, &columns) == F::ZERO, left == 1 && right == 0);
        }
    }
}

#[test]
fn certificate_maps_and_packing_identity_are_checked() {
    let baseline = packages(&fixture());
    for mutation in 0..4 {
        let mut package = baseline.light().clone();
        match mutation {
            0 => package.word_packing.groups[0].certificates.clear(),
            1 => package.word_packing.groups[0].sources[0] += 1,
            2 => package.word_packing.groups[0].pairs[0] = (Bit::Zero, Bit::One),
            _ => package.word_packing.groups[0].guard.constant += F::ONE,
        }
        assert!(package.plan_rows().is_err());
        assert!(package.source_rows().is_err());
    }
    let mut changed = fixture();
    changed[4].a.constant += F::ONE;
    assert_ne!(baseline.identity(), packages(&changed).identity());
}

#[test]
fn width_one_and_circular_certificates_do_not_supply_bits() {
    let rows = vec![guarded(variable(8), variable(4)), guarded(variable(8), variable(5))];
    let compiled = packages(&rows);
    let mut package = compiled.light().clone();
    let first = super::super::application::WITNESS_START;
    package.slots = package
        .slots
        .iter()
        .flat_map(|&(start, count, kind)| {
            (start..start + count)
                .map(move |column| (column, 1, if [first, first + 1].contains(&column) { 0 } else { kind }))
        })
        .collect();
    assert!(WordPacking::checked(&package).unwrap().groups.is_empty());
    let circular = packages_for_circular();
    assert!(circular.light().word_packing.groups.is_empty());
    let mut missing = fixture();
    missing.remove(0);
    let missing = packages(&missing);
    assert!(missing.light().word_packing.groups.is_empty());
}

fn packages_for_circular() -> BoundedDepthPackages {
    packages(&[
        guarded(variable(4), variable(5).plus(constant(-1))),
        guarded(variable(5), variable(4).plus(constant(-1))),
        guarded(variable(8), variable(4)),
        guarded(variable(8), variable(5)),
    ])
}

fn reject_changed_certificate_columns(original: &StepPackage, columns: &[usize]) {
    assert_eq!(original.word_packing.groups[0].pairs.len(), 2);
    for &changed in columns {
        for kind in [None, Some(1), Some(2)] {
            let mut package = original.clone();
            package.slots = package
                .slots
                .iter()
                .flat_map(|&(start, count, old)| {
                    (start..start + count).filter_map(move |column| {
                        if column == changed {
                            kind.map(|kind| (column, 1, kind))
                        } else {
                            Some((column, 1, old))
                        }
                    })
                })
                .collect();
            assert!(WordPacking::checked(&package).unwrap().groups.is_empty());
            assert!(package.plan_rows().is_err());
            assert!(package.source_rows().is_err());
        }
        let mut package = original.clone();
        package
            .field_lowering
            .affine
            .insert(changed, (0, constant(0)));
        assert!(WordPacking::checked(&package).unwrap().groups.is_empty());
        assert!(package.plan_rows().is_err());
        assert!(package.source_rows().is_err());
    }
}

#[test]
fn every_pair_of_a_used_certificate_needs_unchanged_bit_slots() {
    let mut rows = (4..35).map(boolean).collect::<Vec<_>>();
    rows.extend([
        guarded(variable(35), variable(4)),
        guarded(variable(35).scale(F::from_u64(9)), variable(5)),
    ]);
    let compiled = packages_with_words(&rows, 32);
    for package in [compiled.light(), compiled.refresh()] {
        assert_eq!(package.packed_booleans.sources.len(), 31);
        let unused = super::super::application::WITNESS_START + 30;
        reject_changed_certificate_columns(package, &[unused, package.packed_booleans.start + 30]);
    }
}

#[test]
fn trailing_product_pairs_share_the_full_certificate_premise() {
    let mut rows = (4..7).map(boolean).collect::<Vec<_>>();
    rows.extend([
        Row {
            a: variable(4),
            b: variable(5),
            c: variable(6),
        },
        guarded(variable(7), variable(4)),
        guarded(variable(7).scale(F::from_u64(9)), variable(5)),
    ]);
    let compiled = packages_with_words(&rows, 4);
    for package in [compiled.light(), compiled.refresh()] {
        assert_eq!(package.packed_booleans.products.len(), 1);
        let product = package.packed_booleans.product_start();
        assert_eq!(package.packed_booleans.pairs().last(), Some((product, product + 1)));
        reject_changed_certificate_columns(package, &[product, product + 1]);
    }
}
